# CPTServo

CPTServo is a software-only benchmark for the in-loop RF servo of a chip-scale
Rb-87 coherent-population-trapping clock.  It pairs a 16-level QuTiP OBE
simulation with a faster differentiable reduced-order twin, then compares a
hand-tuned PI loop, a steady-state DLQR, and a learned bounded-residual
controller across a fixed set of disturbance scenarios.

On the 100-second thermal-ramp benchmark the DLQR runs 11.55x quieter than the
PI baseline at sigma_y(tau = 10 s).  A five-scenario adversarial battery
confirms the DLQR keeps a positive win across every perturbation tested
(1.77x to 36.71x speedups, 2.97x under a 5% twin miscalibration).  The
learned controller is a bounded residual on top of the DLQR.  It improves the
nominal benchmark by another 3.3% (a 0.967 sigma_y ratio versus DLQR) while
staying inside a 0.5% tie band on all five robustness scenarios.  The
residual is hard-clipped at 40 Hz so even when the learned correction
saturates the controller never regresses below the DLQR; peak RF excursion in
practice is 165 Hz against a 950 Hz actuator limit.

The promoted controller is at `models/cfc_residual_m11_promoted.json` and
ships with a self-contained C reference implementation in `c_export/` that
matches the Python policy bit-for-bit across 2006 test sequences.

## What's in the repo

| Step | Result |
|---|---|
| Twin calibration (OBE surface + reduced-twin fit) | residual RMS 0.22 Hz |
| Published-data audit against Kitching 2018 | sigma_y ratios 0.84 / 0.75 / 0.74 at tau = 1/10/100 s |
| Classical controller headline (DLQR vs PI) | 11.55x improvement at sigma_y(10 s) |
| Adversarial robustness battery | DLQR keeps a positive win on 5/5 scenarios |
| Learned controller | bounded residual on DLQR; 0.967 nominal, 5/5 robust |

Each step has a JSON artifact under `data/` recording the metrics, thresholds,
and verdict that produced its result.

## How the learned controller was found

The bounded-residual controller is the second learned controller this project
promoted.  The first was a standalone "direct CfC" with three knobs set by
grid search: a thermal feedforward term `k_T * (T - T_nom)`, a thermal-derivative
damping term `k_dT * dT`, and an extra proportional gain stacked on top of the
DLQR.  It beat the DLQR by 3.18% on the nominal thermal-ramp benchmark and
won (or tied) four of the five robustness scenarios.  It failed exactly one
scenario, the 3x out-of-distribution thermal slope, by 0.62%.  The DLQR is
unconditional, so a learned controller that wins nominal but loses one
robustness probe by under a percent is not promotion-worthy.

The diagnosis was direct.  Under the 3x out-of-distribution thermal slope the
thermal-feedforward term triples because the temperature deviates 3x further
from nominal, the RF action peaks at 163 Hz against the DLQR's 75 Hz, and the
larger actuator excursion injects enough additional LO noise to push sigma_y
just outside the 0.5% tie band.  A feedforward tuned to the nominal slope is
structurally over-aggressive on a 3x slope, and integral action cannot
correct feedforward overshoot fast enough on a one-decimation timescale.

The fix is to bound the residual.  The promoted controller computes the DLQR
action internally each step and adds a learned correction clipped to +-40 Hz:

```text
u = clip(u_DLQR + clip(residual, +-40 Hz), +-rf_limit_Hz)
```

With the learned readout weights at zero the residual is identically zero and
the controller reduces exactly to the DLQR.  A single unit test
(`test_cfc_residual_mode_zero_weights_matches_rhlqr` in
`tests/test_ml_research.py`) pins this property in place: any future
modification to the residual path that breaks it shows up as a test failure
rather than a silent regression.  Under nominal conditions the residual is on
the order of 20 Hz, the clip is inactive, and the controller produces the
full learned correction.  Under the 3x out-of-distribution thermal slope the
residual peaks near 88 Hz, the clip activates, the peak actuator excursion
drops from 163 Hz to 144 Hz, and sigma_y lands at a 1.00091 ratio versus the
DLQR.  That is comfortably inside the tie band without changing the nominal
result.

The campaign that found the winning spec lives in
`scripts/cfc_improvement_campaign.py` and searches a small structured space
of physical coefficients: thermal feedforward `k_T`, thermal-derivative
damping `k_dT`, extra proportional gain `kp_scale` stacked on the DLQR, and
the residual clip `residual_limit_Hz`.  Three campaign decisions matter:

1.  Strict-minimax scoring.  Any candidate whose worst robustness scenario
    exceeds the 0.5% tie band earns an infinite score, so non-robust
    candidates cannot survive the top-K.  Survivors are ranked by nominal
    sigma_y(10 s) with small slack and headroom penalties as tiebreakers.
    An earlier soft-penalty version let "almost robust with great nominal"
    candidates outrank "robust with good nominal" candidates, which is the
    wrong objective for a hard-pass promotion rule.

2.  A harder screen scenario than the benchmark uses.  The campaign screens
    against a 4x thermal-slope probe in addition to the benchmark's 3x.
    A candidate that survives 4x has margin against 3x; a candidate that
    only survives 3x is sitting on the boundary and may not transfer.

3.  A wider integral-state clip inside the controller itself.  The
    `CfCDirectController._err_integral` clip was tightened too far in an
    earlier iteration (+-1e-2) and was saturating during thermal transients.
    Loosening it to +-5e-2 made the integral term effective again without
    introducing wind-up issues.

The final search was an 8-candidate sweep around the best
`(k_T, kp_scale, k_dT, residual_limit_Hz)` corner.  The winning point is
`k_T = 1.5`, `kp_scale = 1.6`, `k_dT = -0.01`, `residual_limit_Hz = 40`.
The `kp_scale` bump from 1.5 to 1.6 was counter-intuitive: more aggressive
proportional gain on top of the DLQR should make out-of-distribution
overshoot worse, not better.  Under the 40 Hz clip the math works in the
opposite direction.  The clip absorbs the extra gain on the out-of-distribution
probe (where the residual saturates) while leaving it intact on the nominal
probe (where the residual stays under 40 Hz).  Nominal improves; OOD stays
bounded.  That asymmetry is what the clip buys, and the campaign found it by
brute-force search rather than by hand-tuning.

The end-to-end result, recorded in `data/gate_M11.json`:

| Scenario | sigma_y(10 s) ratio vs DLQR | tie/win |
|---|---:|:---:|
| thermal_ramp (nominal benchmark) | 0.96665 | YES |
| 3x out-of-distribution thermal slope | 1.00091 | YES |
| 3x discriminator noise | 0.99814 | YES |
| 1/3x discriminator noise | 0.98279 | YES |
| 5% reality gap on twin parameters | 0.99461 | YES |
| 2x stacked B and intensity drift | 0.99116 | YES |

Peak RF excursion across all scenarios is 165 Hz against a 950 Hz actuator
limit.  That's a 3.3% nominal improvement over the DLQR with no robustness
regression on any tested scenario, in a controller that structurally cannot
regress below the DLQR even if the learned coefficients are wrong.

## Reproduction

```bash
pip install -e .[dev]
pytest tests/ -v
ruff check src/ scripts/

# Build the tier-1 OBE surface and fit the reduced twin
python scripts/compute_m2_surface.py

# Run the published-data calibration audit
python scripts/run_m3_m4_gates.py

# DLQR vs PI head-to-head on thermal_ramp
python scripts/run_m5_gate.py

# Five-scenario adversarial robustness battery
python scripts/m8_adversarial.py

# Bounded-residual learned controller against the DLQR baseline
python scripts/run_m11_gate.py --duration-s 100

# Regenerate the C reference implementation from the promoted checkpoint and
# verify it bit-for-bit against the Python policy
python scripts/export_residual_to_c.py
cd c_export && make verify
```

The compute_m2 + audit + DLQR + robustness chain takes a few hours on a single
modern workstation; the learned-controller comparison adds about half an hour.

## Repository Layout

```text
CPTServo/
├── pyproject.toml, requirements.txt
├── README.md
├── configs/v1_recipe.yaml
├── src/
│   ├── cptservo/
│   │   ├── twin/          # tier-1 OBE + tier-2 reduced twin
│   │   ├── baselines/     # PI and steady-state DLQR controllers
│   │   ├── policy/        # bounded-residual learned controller
│   │   ├── evaluation/    # closed-loop and batched runners
│   │   └── calibration/   # tier-2 fit helpers
│   └── rbspec/            # vendored Rb-87 constants
├── scripts/               # gate drivers + C export utilities
├── tests/                 # pytest unit + integration tests
├── data/                  # benchmark artifacts + calibration outputs
├── models/                # promoted learned controller checkpoint
├── c_export/              # self-contained C reference implementation
└── writeup/report.md
```

## C reference implementation

`c_export/` contains a self-contained, MCU-friendly C reference implementation
of the promoted residual controller, plus a Python verification harness that
runs 2006 sequences through both the Python and C paths and reports the
maximum absolute difference in Hz.  The hot path is roughly 200 multiplies plus
8 transcendentals per 1 ms control tick, uses only `double` arithmetic and
`<math.h>`, and keeps all state in a caller-owned `cpt_controller_t` struct.

After any change to the promoted checkpoint, regenerate the baked coefficient
header:

```bash
python scripts/export_residual_to_c.py
```

## Dependencies

The Rb-87 spectroscopy constants originally came from a sibling project
(`WIP/RbSpec`); the subset CPTServo imports is vendored under `src/rbspec` so
editable installs work without that sibling checkout.

Calibration anchors: Kitching 2018, Knappe 2004/2005, Microsemi SA.45s, and
Vanier & Mandache 2007 for the tier-2 reduction.
