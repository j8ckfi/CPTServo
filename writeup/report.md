# CPTServo: Calibrated Twin and Servo Benchmark for Chip-Scale CPT-Rb87 Clocks

## Pitch

CPTServo is a software-only benchmark for the in-loop RF servo of a chip-scale
Rb-87 coherent-population-trapping clock.  It contains a QuTiP 16-level OBE
surface model, a faster PyTorch reduced twin, a hand-tuned PI baseline, a
steady-state DLQR controller historically labeled RH-LQR, and a learned
bounded-residual controller promoted by the M11 gate.

The classical headline: in the 100-second `thermal_ramp` benchmark, RH-LQR/DLQR
reaches sigma_y(tau=10 s) = 5.07e-12 versus PI at 5.86e-11, an 11.55x
improvement.  The M8 adversarial battery preserves a positive LQR win across
five perturbation probes (1.77x to 36.71x speedups, 2.97x reality-gap).

The learned headline: the M11 promotion gate promotes a bounded learned
residual on top of RH-LQR at a 0.9667 nominal M5 ratio with 5/5 robust ties
or wins.  The promoted controller adds at most 40 Hz of learned RF action on
top of the LQR baseline, so the worst case structurally degrades back to
RH-LQR.

## Twin Architecture

Tier 1 is a 16-level QuTiP Lindblad model over the Rb-87 D1 manifold.  It is
used offline to generate a discriminator response surface over RF detuning,
laser detuning, temperature, magnetic field, and laser intensity.

Tier 2 is an 8-state PyTorch reduced twin with clock-state populations, a
lumped Zeeman bath, excited-state populations, CPT coherence, and intensity
proxy.  It is differentiable and fast enough for batched controller evaluation.

The OBE-to-reduced calibration uses closed-form OLS on lock-point shifts.
`data/gate_M2.json` reports residual RMS 0.22 Hz.  The reduced model's uniform
discriminator-slope assumption leaves 35.8% OBE slope variation documented as
a structural limitation rather than hidden by fitting.

The Rb-87 spectroscopy constants used from the sibling RbSpec project are
vendored under `src/rbspec` so this repository is self-contained.

## Calibration Audit

`data/gate_M3.json` is the source of truth.  The primary Kitching 2018 audit
passes the 2x tolerance at tau={1,10,100}s:

| tau (s) | Twin sigma_y | Published target | Ratio |
|---:|---:|---:|---:|
| 1 | 3.36e-11 | 4.00e-11 | 0.84 |
| 10 | 9.77e-12 | 1.30e-11 | 0.75 |
| 100 | 2.94e-12 | 4.00e-12 | 0.74 |

Cross-check sources vary much more widely, including ratios below 0.1 and one
Knappe 2005 long-term point at 2.14x.  The project therefore claims a
calibrated published-data twin, not a bench-validated physical instrument.

## Controller Benchmark

| Controller | State | Key result |
|---|---|---|
| PI | implemented baseline | M5 thermal-ramp sigma_y(10s) = 5.86e-11 |
| RH-LQR/DLQR | steady-state 2-state DLQR | M5 sigma_y(10s) = 5.07e-12, 11.55x over PI |
| Direct CfC (residual mode) | promoted M11 learned controller | nominal M5 ratio 0.9667 vs RH-LQR with 5/5 robust ties/wins; worst probe 1.0009 inside 1.005 band; rf peak 165 Hz |

The name "RH-LQR" is retained because it is used throughout the artifacts, but
the implementation solves one discrete algebraic Riccati equation and applies
a steady-state gain.  It is not an iterative constrained MPC solver.

## Adversarial Battery

M8 runs PI and the frozen RH-LQR/DLQR controller under five perturbed probes
for 200 seconds each.  `data/gate_M8.json` and `tests/adversarial/REPORT.md`
are the source of truth.

| Probe | LQR speedup vs PI at tau=10s |
|---|---:|
| OOD thermal slope, 3x | 36.71x |
| High discriminator noise, 3x | 1.89x |
| Low discriminator noise, 1/3x | 4.59x |
| Reality gap, +/-5% twin params | 2.97x |
| Worst-case stacked B/I drift | 1.77x |

This preserves a positive LQR win on 5/5 probes and clears the reality-gap
threshold.  It does not preserve or reproduce the exact 11.55x M5 magnitude.

## Promoted Learned Controller (M11)

The M11 promotion gate promotes a bounded residual on RH-LQR.  The promoted
checkpoint is:

```text
models/cfc_residual_m11_promoted.json
```

The controller uses the same `CfCDirectController` feature extraction as the
direct-CfC family, but with `residual_mode=True` and `residual_limit_Hz=40`.
At each step it computes `u_LQR` internally and adds a structured residual
clipped to +-40 Hz:

```text
u = clip(u_LQR + clip(0.6*kp_lqr*err + 1.5*(T - T_nom) - 0.01*dT, +-40 Hz), +-rf_limit)
```

The structured coefficients are kp_scale=1.6 (0.6 extra proportional gain on
top of LQR), k_T=1.5 Hz/K thermal feedforward, k_dT=-0.01 Hz/(K/s) thermal
derivative damping.  Under nominal conditions the residual amplitude stays
inside the clip band, so the controller adds a useful learned correction.
Under the 3x OOD thermal slope the unclipped residual would saturate hard, the
clip activates, and actuator excursion is bounded to 165 Hz peak.  That keeps
`ood_3x_thermal_slope` at 1.0009 (well inside the 0.5% tie band).

Full 100-second `data/gate_M11.json`:

| Probe | ratio vs RH-LQR | tie/win |
|---|---:|:---:|
| m5_thermal_ramp | 0.96665 | YES |
| ood_3x_thermal_slope | 1.00091 | YES |
| high_disc_noise_3x | 0.99814 | YES |
| low_disc_noise_third | 0.98279 | YES |
| reality_gap_5pct | 0.99461 | YES |
| worst_case_2x_stacked | 0.99116 | YES |

`gate_pass = true`, 5/5 robust ties/wins, `rf_abs_max_Hz = 165.2` (well under
the 950 Hz actuator headroom).

The dominant engineering lever is the residual architecture itself: setting
all readout weights to zero exactly reproduces RH-LQR, so the worst case is
structurally bounded by the LQR baseline.  Three supporting levers in the
campaign make the winning spec discoverable: strict-minimax scoring in
`scripts/cfc_improvement_campaign.py` (any robust probe miss is +inf rather
than a soft penalty), a 4x slope screen probe harder than the gate's 3x, and
a loosened integral-state clip in `CfCDirectController` (+-5e-2 from the
prior +-1e-2) so the integral term is not saturated during transients.

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
edge cases including NaN sensors) through both the Python and C
implementations and reports max |diff| = 0 Hz on the promoted checkpoint.

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

Longer gates are expensive: M3 uses 1000-second closed-loop simulations, and
M8 takes about 2.25 hours on the saved run.

## Runner Status

The positive LQR benchmark path is consolidated on `run_batched_loop` for M5,
M8, and M11.  The older `run_fast_loop` remains as legacy M3 calibration code.
New PI-vs-LQR claims should be generated through `scripts/run_m5_gate.py` or
`scripts/m8_adversarial.py`; new learned-controller claims should be generated
through `scripts/run_m11_gate.py` against the promoted residual baseline.
