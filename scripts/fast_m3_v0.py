"""Fast M3 v0 gate: 4 runs x 120s on Kitching conditions.

Runs:
1. nominal_noise — tuned PI, LO noise, no pilot. Allan plausibility + slope.
2. open_pilot — no controller, deterministic 1 Hz / 10 Hz pilot, low noise.
   Open-loop pilot SNR + open y pilot amplitude.
3. closed_pilot_nominal — tuned PI, same pilot, low noise. Suppression + phase.
4. closed_pilot_weak — kp=1000 ki=0, pilot, low noise. Controller sensitivity.

Gate criteria:
- Allan ratios at tau in {1, 10} vs Kitching in [0.2, 5.0]
- Log-log slope over {1, 3, 10} in [-0.8, -0.2]
- Open-loop pilot SNR > 15 dB
- closed/open y pilot amplitude ratio < 0.3
- Cancellation phase 180 +/- 45 deg
- Nominal suppression at least 6 dB better than weak
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_m3_m4_gates import make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.evaluation.pilot_probe import (  # noqa: E402
    cancellation_phase_deg,
    overlapping_allan_slope,
    pilot_amplitude,
    pilot_snr_db,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

PILOT_FREQ_HZ = 1.0
PILOT_AMP_HZ = 10.0
DURATION_S = 120.0
ANALYSIS_S = 100.0  # discard first 20s
DECIMATION_RATE_HZ = 1000.0
KITCHING = {1.0: 4.0e-11, 10.0: 1.3e-11, 100.0: 4.0e-12}


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    sys.stdout.flush()


def trim_to_analysis(arr: np.ndarray) -> np.ndarray:
    n_skip = int((DURATION_S - ANALYSIS_S) * DECIMATION_RATE_HZ)
    return arr[n_skip : n_skip + int(ANALYSIS_S * DECIMATION_RATE_HZ)]


DISC_NOISE_AMP_CI = 7.0e-4  # from configs/v1_recipe.yaml (calibrated)


def run_one(
    label: str,
    controller,
    pilot_freq_Hz: float,
    pilot_amp_Hz: float,
    lo_noise_scale: float,
    seed: int,
    disc_noise_amp_ci: float = DISC_NOISE_AMP_CI,
):
    twin = make_calibrated_twin(cell_temperature_K=360.0, buffer_pressure_torr=100.0)
    dist = Disturbance.from_recipe("clean")
    trace = dist.generate(duration_s=DURATION_S, sample_rate_Hz=10_000.0, seed=seed)

    log(
        f"Run [{label}]: starting (controller={'PI' if controller else 'open'}, "
        f"pilot={pilot_freq_Hz}Hz x {pilot_amp_Hz}Hz, "
        f"lo_scale={lo_noise_scale}, disc_amp={disc_noise_amp_ci:.0e})"
    )
    res = run_fast_loop(
        twin, controller, trace, DURATION_S,
        pilot_freq_Hz=pilot_freq_Hz,
        pilot_amp_Hz=pilot_amp_Hz,
        lo_noise_scale=lo_noise_scale,
        disc_noise_amp_ci=disc_noise_amp_ci,
        rng_seed=seed,
        n_warmup_steps=5_000,
    )
    log(f"  wall {res['wall_s']:.1f} s")
    return res


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    pi_nominal = PIController.from_recipe()
    pi_weak = PIController(kp=1000.0, ki=0.0, control_dt_s=0.001, rf_limit_Hz=1000.0)

    log("=== Fast M3 v0 ===")
    log(
        f"PI nominal: kp={pi_nominal.kp}, ki={pi_nominal.ki}, "
        f"dt={pi_nominal.control_dt_s}, rf_limit={pi_nominal.rf_limit_Hz}"
    )
    log(f"disc_noise_amp_ci = {DISC_NOISE_AMP_CI:.0e}")

    # nominal_noise: full disc noise + full LO noise, no pilot. Allan + slope.
    res_nom_noise = run_one(
        "nominal_noise", pi_nominal, 0.0, 0.0, 1.0, 42,
        disc_noise_amp_ci=DISC_NOISE_AMP_CI,
    )
    # open_pilot: no controller, pilot, low LO + zero disc (pilot must dominate).
    res_open_pilot = run_one(
        "open_pilot", None, PILOT_FREQ_HZ, PILOT_AMP_HZ, 0.05, 43,
        disc_noise_amp_ci=0.0,
    )
    # closed_pilot_nominal: PI, pilot, low LO + zero disc.
    res_closed_nom = run_one(
        "closed_pilot_nominal", pi_nominal, PILOT_FREQ_HZ, PILOT_AMP_HZ, 0.05, 44,
        disc_noise_amp_ci=0.0,
    )
    # closed_pilot_weak: weak PI, pilot, low LO + zero disc.
    res_closed_weak = run_one(
        "closed_pilot_weak", pi_weak, PILOT_FREQ_HZ, PILOT_AMP_HZ, 0.05, 45,
        disc_noise_amp_ci=0.0,
    )

    # ---------------------------------------------------------------------
    # Allan + slope on nominal_noise (full 120s — demean implicit)
    # ---------------------------------------------------------------------
    y_nom = res_nom_noise["y"] - np.mean(res_nom_noise["y"])
    sig_taus = [1.0, 3.0, 10.0]
    allan = overlapping_allan(y_nom, DECIMATION_RATE_HZ, sig_taus)
    sigma_1 = allan[1.0]
    sigma_10 = allan[10.0]
    slope_135 = overlapping_allan_slope(allan, sig_taus)
    ratio_1 = sigma_1 / KITCHING[1.0]
    ratio_10 = sigma_10 / KITCHING[10.0]
    log(f"Allan: sigma_y(1s)={sigma_1:.3e} (ratio {ratio_1:.2f}), "
        f"sigma_y(10s)={sigma_10:.3e} (ratio {ratio_10:.2f}), slope={slope_135:.2f}")

    # ---------------------------------------------------------------------
    # Pilot analysis on last 100s of each pilot run.
    # ---------------------------------------------------------------------
    y_open = trim_to_analysis(res_open_pilot["y"])
    y_closed = trim_to_analysis(res_closed_nom["y"])
    y_weak = trim_to_analysis(res_closed_weak["y"])
    err_open = trim_to_analysis(res_open_pilot["error_signal"])
    rf_cmd_closed = trim_to_analysis(res_closed_nom["rf_cmd"])
    pilot_closed = trim_to_analysis(res_closed_nom["pilot"])

    open_err_snr = pilot_snr_db(err_open, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)
    A_y_open = pilot_amplitude(y_open, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)
    A_y_closed = pilot_amplitude(y_closed, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)
    A_y_weak = pilot_amplitude(y_weak, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)
    suppression = A_y_closed / A_y_open if A_y_open > 0 else float("nan")
    suppression_weak = A_y_weak / A_y_open if A_y_open > 0 else float("nan")
    phase_deg = cancellation_phase_deg(
        rf_cmd_closed, pilot_closed, DECIMATION_RATE_HZ, PILOT_FREQ_HZ
    )
    # Sensitivity: nominal in dB - weak in dB; want > 6 dB
    if suppression > 0 and suppression_weak > 0:
        sens_db = 20.0 * (np.log10(suppression_weak) - np.log10(suppression))
    else:
        sens_db = float("nan")

    log(f"Open-loop pilot SNR (error): {open_err_snr:.1f} dB")
    log(
        f"Pilot amp open / closed-nominal / closed-weak: "
        f"{A_y_open:.3e} / {A_y_closed:.3e} / {A_y_weak:.3e}"
    )
    log(
        f"Suppression (closed/open) nominal: {suppression:.3e}, "
        f"weak: {suppression_weak:.3e}, sens: {sens_db:.1f} dB"
    )
    log(f"Cancellation phase: {phase_deg:.1f} deg (target 180+/-45)")

    # ---------------------------------------------------------------------
    # Gates
    # ---------------------------------------------------------------------
    allan_pass = (0.2 <= ratio_1 <= 5.0) and (0.2 <= ratio_10 <= 5.0)
    slope_pass = -0.8 <= slope_135 <= -0.2
    snr_pass = open_err_snr > 15.0
    sup_pass = (suppression < 0.3) if np.isfinite(suppression) else False
    phase_pass = (
        np.isfinite(phase_deg)
        and (135.0 <= phase_deg <= 225.0 or -225.0 <= phase_deg <= -135.0)
    )
    sens_pass = sens_db > 6.0 if np.isfinite(sens_db) else False

    gate = {
        "milestone": "M3_v0_fast",
        "duration_s": DURATION_S,
        "analysis_s": ANALYSIS_S,
        "pilot_freq_Hz": PILOT_FREQ_HZ,
        "pilot_amp_Hz": PILOT_AMP_HZ,
        "noise_injection_point": "rf_actual_pre_step",
        "pi_nominal_gains": {
            "kp": pi_nominal.kp,
            "ki": pi_nominal.ki,
            "control_dt_s": pi_nominal.control_dt_s,
            "rf_limit_Hz": pi_nominal.rf_limit_Hz,
        },
        "metrics": {
            "sigma_y_1s": sigma_1,
            "sigma_y_10s": sigma_10,
            "kitching_1s": KITCHING[1.0],
            "kitching_10s": KITCHING[10.0],
            "ratio_1s": ratio_1,
            "ratio_10s": ratio_10,
            "slope_loglog_1_3_10": slope_135,
            "open_err_snr_db": open_err_snr,
            "A_y_pilot_open": A_y_open,
            "A_y_pilot_closed_nominal": A_y_closed,
            "A_y_pilot_closed_weak": A_y_weak,
            "suppression_nominal": suppression,
            "suppression_weak": suppression_weak,
            "controller_sensitivity_db": sens_db,
            "cancellation_phase_deg": phase_deg,
        },
        "gates": {
            "allan_pass": bool(allan_pass),
            "slope_pass": bool(slope_pass),
            "snr_pass": bool(snr_pass),
            "suppression_pass": bool(sup_pass),
            "phase_pass": bool(phase_pass),
            "controller_sensitivity_pass": bool(sens_pass),
        },
        "all_gates_pass": bool(
            allan_pass and slope_pass and snr_pass and sup_pass and phase_pass and sens_pass
        ),
    }

    log("\n=== Gate verdicts ===")
    for name, v in gate["gates"].items():
        log(f"  {name}: {'PASS' if v else 'FAIL'}")
    log(f"  ALL: {'PASS' if gate['all_gates_pass'] else 'FAIL'}")

    out_path = data_dir / "gate_M3_v0.json"
    out_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
