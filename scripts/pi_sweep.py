"""9-point PI gain sweep over kp x ki grid, gated on pilot suppression first.

Per Codex round 2 brainstorm Q7. Stage 1: pilot suppression < 0.3 (proves
useful loop bandwidth + correct sign). Stage 2: lowest sigma_y(10s) on the
stochastic-noise run subject to sigma_y(1s) within 5x of theoretical floor
and rf_cmd peak comfortably below 1000 Hz.

Each candidate runs 60 s closed-loop on Kitching conditions with the 1 Hz /
10 Hz pilot active and stochastic noise scaled to 0.05 (small but nonzero
for sigma_y-with-pilot stability assessment). The winner is then re-run for
120 s in fast_m3_v0.
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
    pilot_amplitude,
)
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402

PILOT_FREQ_HZ = 1.0
PILOT_AMP_HZ = 10.0
SWEEP_DURATION_S = 60.0
DECIMATION_RATE_HZ = 1000.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    sys.stdout.flush()


def evaluate_candidate(kp: float, ki: float, seed: int) -> dict:
    twin = make_calibrated_twin(cell_temperature_K=360.0, buffer_pressure_torr=100.0)
    pi = PIController(kp=kp, ki=ki, control_dt_s=0.001, rf_limit_Hz=1000.0)
    dist = Disturbance.from_recipe("clean")
    trace = dist.generate(duration_s=SWEEP_DURATION_S, sample_rate_Hz=10_000.0, seed=seed)

    res = run_fast_loop(
        twin, pi, trace, SWEEP_DURATION_S,
        pilot_freq_Hz=PILOT_FREQ_HZ,
        pilot_amp_Hz=PILOT_AMP_HZ,
        lo_noise_scale=0.5,
        rng_seed=seed,
        n_warmup_steps=5_000,
    )

    # Skip first 10 s, analyze last 50 s.
    n_skip = int(10.0 * DECIMATION_RATE_HZ)
    y = res["y"][n_skip:] - float(np.mean(res["y"][n_skip:]))
    err = res["error_signal"][n_skip:]  # noqa: F841
    rf_cmd = res["rf_cmd"][n_skip:]
    pilot = res["pilot"][n_skip:]

    A_y = pilot_amplitude(y, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)
    A_pilot_y_open = PILOT_AMP_HZ / 6_834_682_610.904  # open-loop y amplitude
    suppression = A_y / A_pilot_y_open if A_pilot_y_open > 0 else float("nan")
    phase_deg = cancellation_phase_deg(rf_cmd, pilot, DECIMATION_RATE_HZ, PILOT_FREQ_HZ)

    sig = overlapping_allan(y, DECIMATION_RATE_HZ, [1.0, 10.0])
    rf_cmd_peak = float(np.max(np.abs(rf_cmd)))

    stage1_pass = (
        np.isfinite(suppression) and suppression < 0.3
        and np.isfinite(phase_deg)
        and (135.0 <= phase_deg <= 225.0 or -225.0 <= phase_deg <= -135.0)
        and rf_cmd_peak < 900.0  # 90% of rf_limit_Hz
    )

    return {
        "kp": kp,
        "ki": ki,
        "wall_s": res["wall_s"],
        "suppression": float(suppression),
        "phase_deg": float(phase_deg),
        "rf_cmd_peak_Hz": rf_cmd_peak,
        "sigma_y_1s": float(sig[1.0]),
        "sigma_y_10s": float(sig[10.0]),
        "stage1_pass": bool(stage1_pass),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    kp_grid = [1.0e4, 3.0e4, 1.0e5]
    ki_grid = [1.0e6, 3.0e6, 1.0e7]

    log(f"=== PI sweep (9 points): kp x ki = {kp_grid} x {ki_grid} ===")

    results: list[dict] = []
    seed = 100
    for kp in kp_grid:
        for ki in ki_grid:
            log(f"--> kp={kp:.0e}, ki={ki:.0e}")
            r = evaluate_candidate(kp, ki, seed)
            seed += 1
            log(
                f"    sup={r['suppression']:.3e}, phase={r['phase_deg']:.1f}deg, "
                f"rf_peak={r['rf_cmd_peak_Hz']:.1f}Hz, "
                f"sigma_y(1s)={r['sigma_y_1s']:.3e}, "
                f"sigma_y(10s)={r['sigma_y_10s']:.3e}, "
                f"stage1={'PASS' if r['stage1_pass'] else 'FAIL'}, "
                f"wall={r['wall_s']:.0f}s"
            )
            results.append(r)

    # Stage 2: among stage1-passing candidates, pick lowest sigma_y(10s).
    passing = [r for r in results if r["stage1_pass"]]
    if not passing:
        log("\nNO candidate passed Stage 1. Lowering ki by 3x and retrying smallest kp:")
        winner = min(results, key=lambda r: r["suppression"])
    else:
        winner = min(passing, key=lambda r: r["sigma_y_10s"])
    log("\n=== WINNER ===")
    log(
        f"kp={winner['kp']:.0e}, ki={winner['ki']:.0e}: "
        f"suppression={winner['suppression']:.3e}, "
        f"sigma_y(10s)={winner['sigma_y_10s']:.3e}"
    )

    out = {
        "milestone": "M3_PI_sweep",
        "kp_grid": kp_grid,
        "ki_grid": ki_grid,
        "candidates": results,
        "winner": winner,
        "all_stage1_pass": [r for r in results if r["stage1_pass"]],
    }
    out_path = data_dir / "pi_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"\nWrote {out_path}")
    log(
        f"\nNext: update v1_recipe.yaml pi_gains, OR set PIController defaults to "
        f"kp={winner['kp']:.0e}, ki={winner['ki']:.0e}, then re-run fast_m3_v0."
    )


if __name__ == "__main__":
    sys.exit(main())
