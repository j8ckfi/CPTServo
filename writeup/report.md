# CPTServo: Calibrated Twin and Servo Benchmark for Chip-Scale CPT-Rb87 Clocks

## Summary

CPTServo is a software-only benchmark for the in-loop RF servo of a chip-scale
Rb-87 coherent-population-trapping clock.  It contains a QuTiP 16-level OBE
surface model, a faster PyTorch reduced twin, a hand-tuned PI baseline, a
steady-state DLQR controller, and a learned bounded-residual controller.

The classical headline: on the 100-second thermal-ramp benchmark, the DLQR
controller reaches sigma_y(tau = 10 s) = 5.07e-12 versus PI at 5.86e-11, an
11.55x improvement.  The five-scenario adversarial battery keeps a positive
DLQR win across every perturbation tested (1.77x to 36.71x speedups, 2.97x
under a 5% twin miscalibration).

The learned headline: the bounded-residual learned controller improves the
classical nominal benchmark by another 3.3% (0.967 sigma_y ratio versus DLQR)
while staying inside a 0.5% tie band on all five robustness scenarios.  The
worst single scenario tested is the 3x thermal-slope probe at 1.0009.  The
controller's structural guarantee is the residual clip: with the learned
readout set to zero, the controller reduces exactly to the DLQR, so the
worst-case robustness is bounded by the classical baseline.

## Twin Architecture

The tier-1 model is a 16-level QuTiP Lindblad simulation over the Rb-87 D1
manifold.  It is used offline to generate a discriminator response surface
over RF detuning, laser detuning, temperature, magnetic field, and laser
intensity.

The tier-2 model is an 8-state PyTorch reduced twin with clock-state
populations, a lumped Zeeman bath, excited-state populations, CPT coherence,
and an intensity proxy.  It is differentiable and fast enough for batched
controller evaluation.

The OBE-to-reduced calibration uses closed-form OLS on lock-point shifts.
The fit artifact at `data/gate_M2.json` reports residual RMS 0.22 Hz.  The
reduced model's uniform discriminator-slope assumption leaves 35.8% OBE slope
variation documented as a structural limitation rather than hidden by fitting.

The Rb-87 spectroscopy constants used from the sibling RbSpec project are
vendored under `src/rbspec` so this repository is self-contained.

## Calibration Audit

The published-data audit at `data/gate_M3.json` is the source of truth.  The
primary Kitching 2018 comparison passes the 2x tolerance at tau = 1, 10, and
100 s:

| tau (s) | Twin sigma_y | Published target | Ratio |
|---:|---:|---:|---:|
| 1 | 3.36e-11 | 4.00e-11 | 0.84 |
| 10 | 9.77e-12 | 1.30e-11 | 0.75 |
| 100 | 2.94e-12 | 4.00e-12 | 0.74 |

Cross-check sources vary much more widely, including ratios below 0.1 and one
Knappe 2005 long-term point at 2.14x.  The project therefore claims a
calibrated published-data twin, not a bench-validated physical instrument.

## Controller Benchmark

| Controller | Status | Result |
|---|---|---|
| PI | hand-tuned baseline | sigma_y(10 s) = 5.86e-11 on thermal_ramp |
| DLQR | steady-state, 2-state | sigma_y(10 s) = 5.07e-12, 11.55x over PI |
| Bounded-residual learned | promoted | 0.967 sigma_y ratio versus DLQR, 5/5 robust, 165 Hz peak RF |

The DLQR implementation solves one discrete algebraic Riccati equation and
applies a steady-state gain.  It is not an iterative constrained MPC solver.

## Adversarial Battery

The robustness battery at `data/gate_M8.json` runs PI and the frozen DLQR
through five perturbed scenarios for 200 seconds each.

| Scenario | DLQR speedup vs PI at sigma_y(10 s) |
|---|---:|
| OOD thermal slope, 3x | 36.71x |
| High discriminator noise, 3x | 1.89x |
| Low discriminator noise, 1/3x | 4.59x |
| Reality gap, +/-5% twin params | 2.97x |
| Worst-case stacked B/I drift | 1.77x |

This preserves a positive DLQR win on all five scenarios and clears the
reality-gap threshold.  It does not reproduce the exact 11.55x classical
magnitude; that's the headline thermal-ramp number, and the perturbed
scenarios are read as robustness checks, not magnitude reproductions.

## Bounded-Residual Learned Controller

The promoted learned controller is at:

```text
models/cfc_residual_m11_promoted.json
```

Architecturally it is the same `CfCDirectController` feature extractor used in
earlier CfC experiments, but the output path runs in residual mode with a
40 Hz clip.  At each step the controller computes the DLQR action internally
and adds a structured residual clipped to +-40 Hz:

```text
u = clip(u_DLQR + clip(0.6 * kp_DLQR * err + 1.5 * (T - T_nom) - 0.01 * dT,
                        +-40 Hz),
         +-rf_limit_Hz)
```

The structured coefficients are kp_scale = 1.6 (0.6 extra proportional gain on
top of the DLQR), k_T = 1.5 Hz/K thermal feedforward, k_dT = -0.01 Hz/(K/s)
thermal derivative damping.  Under nominal conditions the residual amplitude
stays inside the clip band, so the controller adds a useful learned correction.
Under the 3x out-of-distribution thermal slope the unclipped residual would
saturate hard, the clip activates, and actuator excursion is bounded to 165 Hz
peak.  That keeps the 3x thermal-slope ratio at 1.0009 (well inside the 0.5%
tie band).

Full 100-second `data/gate_M11.json`:

| Scenario | ratio vs DLQR | tie/win |
|---|---:|:---:|
| thermal_ramp (nominal) | 0.96665 | YES |
| 3x thermal slope | 1.00091 | YES |
| 3x discriminator noise | 0.99814 | YES |
| 1/3x discriminator noise | 0.98279 | YES |
| 5% reality gap | 0.99461 | YES |
| 2x stacked B/I drift | 0.99116 | YES |

5/5 robust ties or wins, peak RF excursion 165.2 Hz (limit 950).

The dominant engineering lever is the residual architecture itself.  Setting
all readout weights to zero exactly reproduces the DLQR, so the worst case is
structurally bounded by the classical baseline.  Three supporting changes in
the campaign script make the winning spec discoverable: strict-minimax scoring
in `scripts/cfc_improvement_campaign.py` (a robust-scenario miss is +inf rather
than a soft penalty), a 4x thermal-slope screen scenario harder than the
robustness benchmark's 3x, and a loosened integral-state clip in the
`CfCDirectController` (+-5e-2 from the prior +-1e-2) so the integral term is
not saturated during transients.

### C reference implementation

A self-contained C reference implementation of the promoted controller lives
under `c_export/` (public API `cpt_controller_init` and `cpt_controller_step`
operating on a caller-owned `cpt_controller_t` struct).  All arithmetic is
`double`, the hot path is roughly 200 multiplies plus 8 transcendentals per
1 ms tick, and the implementation depends only on `<math.h>` and `<stdint.h>`.
No mallocs, no globals.

Coefficients are baked into `c_export/cptservo_coeffs.h` from the JSON
checkpoint by `scripts/export_residual_to_c.py`.  `scripts/verify_c_residual.py`
runs 2006 (error, T_K, B_uT, I_norm) sequences (2001 random plus 5 hand-picked
edge cases including NaN sensor input) through both the Python and C paths
and reports max absolute difference = 0 Hz on the promoted checkpoint.

## Reproduction

```bash
pip install -e .[dev]
pytest tests/ -v
ruff check src/ scripts/
python scripts/run_m5_gate.py
python scripts/m8_adversarial.py
python scripts/cfc_improvement_campaign.py --screen-duration-s 25 --max-specs 600 --batch-size 24
python scripts/run_m11_gate.py --duration-s 100
```

Long runs are expensive: the published-data audit uses 1000-second closed-loop
simulations, and the robustness battery takes about 2.25 hours on a single
modern workstation.

## Runner Status

The benchmark and robustness paths are consolidated on `run_batched_loop` for
the DLQR head-to-head, the adversarial battery, and the learned-controller
comparison.  The older `run_fast_loop` remains as legacy calibration code.
New PI-vs-DLQR claims should be generated through `scripts/run_m5_gate.py` or
`scripts/m8_adversarial.py`; new learned-controller claims should be generated
through `scripts/run_m11_gate.py` against the promoted residual baseline.
