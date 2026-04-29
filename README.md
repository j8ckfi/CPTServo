# CPTServo

A calibrated digital twin and servo benchmark for a chip-scale CPT-Rb87 atomic clock. Three controllers (hand-tuned PI, receding-horizon LQR, learned PPO) are run head-to-head against five named disturbance recipes, with a five-probe adversarial battery validating the headline win.

> **Headline.** A first-principles 8-level CPT twin, calibrated against Kitching 2018 + Knappe 2004/2005 + Microsemi SA.45s published σ_y curves to within 2× across τ ∈ {1, 10, 100, 1000} s. **Receding-horizon LQR runs 5.97× quieter than the PI baseline at σ_y(τ=10 s) on `thermal_ramp`**, and a five-probe adversarial battery — including a ±5% reality-gap perturbation of the calibrated parameters — preserves that win on every probe (5/5).

See `writeup/report.md` for the full writeup.

## Status

All milestones (M1-M10) closed. Milestone status lives in the project PRD at `Research/.omc/prd.json` and a free-form log in `progress.txt`. Per-gate numerics in `data/gate_M{N}.json`.

| Milestone | Result |
|---|---|
| M1 | reduced twin v0 + literature scan (CLEAR) |
| M2 | tier-1 OBE surface + tier-2 calibration (peak-slope <5%, lock-shift <2%) |
| M3 | public-data calibration audit (Kitching ratios 0.84 / 0.75 / 0.74 — passes 2×) |
| M4 | subsumed by M3 anti-fudge architecture |
| M5 | RH-LQR baseline beats PI by 5.97× on thermal_ramp τ=10s |
| M6 | APG architecture verified; noise-floor-limited at short horizons (PARTIAL) |
| M7 | PPO infrastructure verified; underfits PI by 40× (DOCUMENTED FAIL — finding, not kill) |
| M8 | adversarial battery 5/5 — FULL HEADLINE PRESERVED |
| M9 | Jetson deployment dropped (RTX 4070 deemed sufficient for the writeup) |
| M10 | writeup + README + reproduction recipe (this document) |

## Reproduction

```bash
cd WIP/CPTServo
pip install -e .[dev]
pytest tests/ -v
ruff check src/ scripts/

# M2 OBE surface + tier-2 calibration (~10 min)
python scripts/compute_m2_surface.py

# M3 + M4: calibration audit + PI baseline (~5 min)
python scripts/run_m3_m4_gates.py

# M5: RH-LQR head-to-head vs PI (~5 min)
python scripts/run_m5_gate.py

# M6: APG training (Modal, ~30 min wall on 8-CPU + 64 GB RAM)
modal run modal_apg_train.py::train
python scripts/m6_apg_gate.py

# M7: PPO curriculum + gate (~25 min train + ~60 min gate)
python scripts/m7_ppo_train.py
python scripts/m7_ppo_gate.py

# M8: adversarial battery (~2 hours wall)
python scripts/m8_adversarial.py
```

Each gate writes `data/gate_M{N}.json` with the numeric metrics, threshold, and pass/fail verdict.

## Repository layout

```
WIP/CPTServo/
├── pyproject.toml
├── requirements.txt
├── README.md
├── progress.txt                              # free-form milestone log
├── configs/v1_recipe.yaml                    # Layer-0 frozen artifact
├── src/cptservo/
│   ├── twin/                                 # tier-1 OBE + tier-2 reduced
│   ├── baselines/                            # pi.py, rh_lqr.py
│   ├── policy/                               # apg.py, cpt_env.py, training.py
│   ├── evaluation/                           # batched_runner.py, allan.py
│   └── calibration/                          # fit_reduced.py
├── scripts/                                  # m2..m8 gate drivers
├── tests/                                    # pytest unit + integration
│   └── adversarial/REPORT.md                 # M8 aggregator
├── data/                                     # gate JSONs, OBE surface (h5)
├── models/                                   # ppo_best.zip, apg_*.pt
├── logs/                                     # m{2..8}_*.log
└── writeup/report.md                         # 2-3pp PNTGuard-style writeup
```

## Acknowledgements

Physical constants imported from sibling project `RbSpec` (`WIP/RbSpec/src/rbspec/solver.py`): HF_GROUND, Doppler width helpers, Voigt profile, Rb-87 D1/D2 wavelengths and oscillator strengths.

Calibration anchored to: Kitching, J. *Applied Physics Reviews* 5, 031302 (2018); Knappe, S. et al. *Optics Letters* 29(7), 695 (2004); Knappe, S. et al. *Applied Physics Letters* 86, 154102 (2005); Microsemi SA.45s CSAC datasheet.

Tier-2 reduction theory: Vanier, J. & Mandache, C. *Applied Physics B* 87, 565 (2007).
