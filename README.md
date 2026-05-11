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
