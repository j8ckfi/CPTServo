"""Skeleton for notebooks/01_calibration_audit.ipynb (M3 hard kill gate).

This is a Python-script form of the notebook content; jupytext-style. M2/M3
will convert this to an executed .ipynb.

The M3 gate compares twin sigma_y(tau in {1, 10, 100, 1000} s) against published
values from Kitching 2018 (primary) plus Knappe 2004, Knappe 2005, Microsemi
SA.45s datasheet, Liu 2025 (cross-checks). Tolerance: factor of 2x.
"""

# %% [markdown]
# # M3: Public-Data Calibration Audit
#
# **Hard kill gate**: twin must reproduce published Allan deviation values
# within a factor of 2x at tau in {1, 10, 100, 1000} s for the primary
# Kitching 2018 reference. Failure halts ralph and pivots to twin+baselines-only.

# %%
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cptservo.twin.allan import overlapping_allan
from cptservo.twin.disturbance import Disturbance
from cptservo.twin.reduced import ReducedTwin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGS_DIR = PROJECT_ROOT / "notebooks" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load published Allan deviations

# %%
with open(DATA_DIR / "published_allan.json", encoding="utf-8") as fh:
    published = json.load(fh)

primary_id = published["calibration_strategy"]["primary_target_id"]
primary = next(s for s in published["sources"] if s["id"] == primary_id)
print(f"Primary calibration target: {primary['citation']}")

# %% [markdown]
# ## Run twin under matched-disturbance recipes
#
# For Kitching 2018 the matched recipe is `clean` at the operating conditions
# in the source's `operating_conditions` block. We integrate at 10 kHz physics,
# decimate to 1 kHz, and compute overlapping Allan at the four target taus.

# %%
EVAL_TAUS = published["calibration_strategy"]["evaluation_taus_s"]
DECIMATION_Hz = 1000.0
PHYSICS_Hz = 10_000.0


def run_twin_for_source(source: dict, duration_s: float = 10_000.0) -> dict[float, float]:
    """Run the twin in the source's matched-disturbance recipe and return
    sigma_y at each evaluation tau."""
    cond = source["operating_conditions"]
    recipe_name = published["calibration_strategy"]["matched_disturbance_recipes"].get(
        source["id"], "clean"
    )
    dist = Disturbance.from_recipe(recipe_name)
    trace = dist.generate(duration_s=duration_s, sample_rate_Hz=PHYSICS_Hz, seed=42)

    twin = ReducedTwin(
        cell_temperature_K=float(cond.get("cell_temperature_K_nominal", 333.15)),
        buffer_pressure_torr=float(cond.get("buffer_pressure_torr_nominal", 25.0)),
        device="cpu",
    )
    state = twin.initial_state(batch_size=1)
    controls = torch.zeros(1, 2, dtype=torch.float64)
    decimation = int(PHYSICS_Hz / DECIMATION_Hz)
    n_dec = int(duration_s * DECIMATION_Hz)
    y = np.empty(n_dec, dtype=np.float64)

    dist_arr = np.stack(
        [trace.T_K, trace.B_uT, trace.laser_intensity_norm], axis=1
    ).astype(np.float64)
    dist_tensor = torch.from_numpy(dist_arr)

    with torch.no_grad():
        for k in range(n_dec):
            for j in range(decimation):
                idx = k * decimation + j
                state = twin.step(
                    state, controls, dist_tensor[idx : idx + 1], 1.0 / PHYSICS_Hz
                )
            y[k] = twin.fractional_frequency_error(state, controls).item()

    return overlapping_allan(y, DECIMATION_Hz, EVAL_TAUS)


# %% [markdown]
# ## Comparison table

# %%
rows = []
twin_results = {}
for source in published["sources"]:
    print(f"Running twin for source: {source['id']}")
    twin_results[source["id"]] = run_twin_for_source(source, duration_s=10_000.0)
    for tau_s in EVAL_TAUS:
        twin_sigma = twin_results[source["id"]].get(tau_s, np.nan)
        source_pt = next(
            (p for p in source["allan_points"] if abs(p["tau_s"] - tau_s) < 0.5),
            None,
        )
        source_sigma = source_pt["sigma_y"] if source_pt else np.nan
        ratio = (twin_sigma / source_sigma) if (source_pt and source_sigma > 0) else np.nan
        rows.append(
            {
                "source": source["id"],
                "tau_s": tau_s,
                "twin_sigma_y": float(twin_sigma) if np.isfinite(twin_sigma) else None,
                "published_sigma_y": float(source_sigma) if np.isfinite(source_sigma) else None,
                "ratio": float(ratio) if np.isfinite(ratio) else None,
                "is_primary": source.get("primary", False),
            }
        )

# %% [markdown]
# ## Gate evaluation

# %%
primary_rows = [r for r in rows if r["is_primary"] and r["ratio"] is not None]
ratios = [r["ratio"] for r in primary_rows]
all_within_2x = all(0.5 < r < 2.0 for r in ratios)
max_ratio = max(ratios) if ratios else None
min_ratio = min(ratios) if ratios else None

gate = {
    "sources": [s["id"] for s in published["sources"]],
    "rows": rows,
    "primary_target": primary_id,
    "primary_ratios": ratios,
    "max_ratio": max_ratio,
    "min_ratio": min_ratio,
    "all_within_2x": all_within_2x,
    "gate_pass": all_within_2x,
}
out_path = DATA_DIR / "gate_M3.json"
out_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
print(f"Wrote {out_path}")
print(json.dumps(gate, indent=2))

# %% [markdown]
# ## Allan curve overlay plot

# %%
fig, ax = plt.subplots(figsize=(8, 6))
for source in published["sources"]:
    pts = sorted(source["allan_points"], key=lambda p: p["tau_s"])
    ax.loglog(
        [p["tau_s"] for p in pts],
        [p["sigma_y"] for p in pts],
        marker="o",
        label=f"{source['id']} (published)",
        alpha=0.6,
    )
    twin_sigma = twin_results[source["id"]]
    twin_taus = sorted(twin_sigma.keys())
    ax.loglog(
        twin_taus,
        [twin_sigma[t] for t in twin_taus],
        marker="s",
        linestyle="--",
        label=f"{source['id']} (twin)",
    )
ax.set_xlabel(r"$\tau$ (s)")
ax.set_ylabel(r"$\sigma_y(\tau)$")
ax.set_title("Twin vs. published Allan deviation — calibration audit")
ax.legend(fontsize=8)
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "allan_calibration.png", dpi=150)
print("Saved plot to", FIGS_DIR / "allan_calibration.png")
