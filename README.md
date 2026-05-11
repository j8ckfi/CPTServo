# CPTServo

A calibrated digital-twin and servo benchmark for a chip-scale CPT-Rb87 atomic
clock. The project compares a hand-tuned PI loop, a steady-state DLQR controller
kept under the historical "RH-LQR" name, and a learned bounded-residual
controller against named disturbance recipes.

> **Headline.** The M5 benchmark reports the RH-LQR/DLQR controller running
> 11.55x quieter than the PI baseline at sigma_y(tau=10 s) on `thermal_ramp`.
> The M8 adversarial battery shows the frozen LQR remains ahead of PI on 5/5
> perturbation probes, with speedups from 1.77x to 36.71x and a 2.97x
> reality-gap win.  The M11 promotion gate promotes a bounded learned residual
> on top of RH-LQR (`residual_mode=True, residual_limit_Hz=40`, structured terms
> `k_T=1.5, k_dT=-0.01, kp_scale=1.6, ki_scale=1.0`) at a 0.9667 nominal M5
> ratio with 5/5 robust ties/wins (worst probe 1.0009, inside the 0.005 tie
> band) and 165 Hz peak RF action.  Promoted checkpoint:
> `models/cfc_residual_m11_promoted.json`.  A self-contained C reference
> implementation lives under `c_export/`, verified bit-for-bit against the
> Python policy across 2006 test sequences.

## Status

| Milestone | Result |
|---|---|
| M2 | PASS — tier-1 OBE surface + tier-2 calibration; residual RMS 0.22 Hz |
| M3 | PASS — Kitching 2018 ratios 0.84 / 0.75 / 0.74 at tau={1,10,100}s |
| M5 | PASS — RH-LQR/DLQR beats PI by 11.55x on `thermal_ramp` at tau=10s |
| M8 | PASS — RH-LQR retains positive win on 5/5 perturbation probes |
| M11 | PASS — bounded residual on RH-LQR promoted (see headline) |

Per-gate truth lives in `data/gate_M{N}.json`.

## Reproduction

```bash
pip install -e .[dev]
pytest tests/ -v
ruff check src/ scripts/

# M2: tier-1 OBE surface + tier-2 reduced-twin calibration
python scripts/compute_m2_surface.py

# M3: primary Kitching 2018 calibration audit
python scripts/run_m3_m4_gates.py

# M5: RH-LQR/DLQR head-to-head vs PI
python scripts/run_m5_gate.py

# M8: adversarial probe battery
python scripts/m8_adversarial.py

# M11: bounded-residual promotion gate (uses models/cfc_residual_m11_promoted.json
# as the comparison baseline; pass --candidate-checkpoint to test new candidates)
python scripts/run_m11_gate.py --duration-s 100

# Optional: regenerate the C export and verify it bit-for-bit against Python
python scripts/export_residual_to_c.py
cd c_export && make verify    # requires gcc/clang; or compile manually with TCC
```

Each gate writes `data/gate_M{N}.json` with metrics, thresholds, and verdict.

## Repository Layout

```text
CPTServo/
├── pyproject.toml, requirements.txt
├── README.md
├── configs/v1_recipe.yaml
├── src/
│   ├── cptservo/
│   │   ├── twin/          # tier-1 OBE + tier-2 reduced twin
│   │   ├── baselines/     # PI and RH-LQR/DLQR controllers
│   │   ├── policy/        # bounded-residual learned controller
│   │   ├── evaluation/    # closed-loop and batched runners
│   │   └── calibration/   # tier-2 fit helpers
│   └── rbspec/            # vendored Rb-87 constants
├── scripts/               # M2/M3/M5/M8/M11 gate drivers + C export utilities
├── tests/                 # pytest unit + integration tests
├── data/                  # canonical gate JSONs + M2 calibration artifacts
├── models/                # promoted M11 checkpoint
├── c_export/              # self-contained C reference implementation
└── writeup/report.md
```

## C reference implementation

`c_export/` contains a self-contained, MCU-friendly C reference implementation
of the promoted residual controller, plus a Python verification harness that
runs 2006 sequences through both the Python and C paths and reports max |diff|
in Hz.  The hot path is roughly 200 multiplies plus 8 transcendentals per
1 ms tick, uses only `double` arithmetic and `<math.h>`, and keeps all state
in a caller-owned `cpt_controller_t` struct.

Regenerate the baked coefficient header after any change to the promoted
checkpoint:

```bash
python scripts/export_residual_to_c.py
```

## Dependencies

The Rb-87 constants originally came from sibling project `WIP/RbSpec`; the
subset CPTServo imports is vendored under `src/rbspec` so editable installs
work without that sibling checkout.

Calibration anchors: Kitching 2018, Knappe 2004/2005, Microsemi SA.45s, and
Vanier & Mandache 2007 for the tier-2 reduction.
