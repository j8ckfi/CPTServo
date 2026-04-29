"""Convert 01_calibration_audit_skeleton.py to 01_calibration_audit.ipynb.

Parses jupytext-style %% cell markers and appends the M2 discriminator
overlay cell.  Run once from the notebooks/ directory.
"""

from __future__ import annotations

from pathlib import Path

import nbformat

# ---------------------------------------------------------------------------
# Parse skeleton
# ---------------------------------------------------------------------------
skeleton = Path("01_calibration_audit_skeleton.py").read_text(encoding="utf-8-sig")
lines = skeleton.splitlines()

cells: list[nbformat.NotebookNode] = []
current_type: str | None = None
current_lines: list[str] = []


def _flush(cell_type: str | None, cell_lines: list[str]) -> None:
    """Flush accumulated lines as a notebook cell."""
    if cell_type is None:
        return
    while cell_lines and not cell_lines[0].strip():
        cell_lines.pop(0)
    while cell_lines and not cell_lines[-1].strip():
        cell_lines.pop()
    if not cell_lines:
        return
    if cell_type == "markdown":
        md_lines: list[str] = []
        for ln in cell_lines:
            if ln.startswith("# "):
                md_lines.append(ln[2:])
            elif ln.rstrip() == "#":
                md_lines.append("")
            else:
                md_lines.append(ln)
        src = "\n".join(md_lines) + "\n"
        cells.append(nbformat.v4.new_markdown_cell(src))
    else:
        src = "\n".join(cell_lines) + "\n"
        cells.append(nbformat.v4.new_code_cell(src))


for line in lines:
    stripped = line.strip()
    if stripped == "# %% [markdown]":
        _flush(current_type, current_lines)
        current_type = "markdown"
        current_lines = []
    elif stripped == "# %%":
        _flush(current_type, current_lines)
        current_type = "code"
        current_lines = []
    else:
        if current_type is not None:
            current_lines.append(line)

_flush(current_type, current_lines)

# ---------------------------------------------------------------------------
# M2 overlay cell
# ---------------------------------------------------------------------------
OVERLAY_MD = """\
## M2: Tier-2 vs Tier-1 discriminator overlay

Sweep `rf_detuning` at a fixed slice (T=343 K, B=50 uT, I=1.0) and overlay
the tier-1 (full OBE) and tier-2 (ReducedTwin) discriminator curves.
Figure saved to `notebooks/figs/discriminator_t1_t2_overlay.png`.
"""

OVERLAY_CODE = """\
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from cptservo.twin.full_obe import _find_lock_point_and_slope, full_obe_discriminator
from cptservo.twin.reduced import ReducedTwin

# --- Parameters for overlay slice ---
RF_GRID = np.linspace(-200.0, 200.0, 41)  # Hz
T_SLICE = 343.0   # K
B_SLICE = 50.0    # uT
I_SLICE = 1.0     # normalised

# --- Tier-1: full OBE ---
disc_t1 = np.array([
    full_obe_discriminator(rf, 0.0, T_SLICE, B_SLICE, I_SLICE)["discriminator"]
    for rf in RF_GRID
])
lp_t1, sl_t1 = _find_lock_point_and_slope(RF_GRID, disc_t1)
print(f"Tier-1 lock point: {lp_t1:.4f} Hz, slope: {sl_t1:.4e}")

# --- Tier-2: ReducedTwin (with calibrated params if available, else defaults) ---
cal_path = pathlib.Path("../data/reduced_calibration.json")
if cal_path.exists():
    with open(cal_path, encoding="utf-8-sig") as fh:
        cal = json.load(fh)
    ls = cal["light_shift_coeff"]
    buf = cal["buffer_gas_shift_coeff"]
    zee = cal["lumped_zeeman_coeff"]
else:
    ls, buf, zee = 0.0, -7.4e6, 7.0

twin = ReducedTwin(
    buffer_pressure_torr=25.0,
    light_shift_coeff=ls,
    buffer_gas_shift_coeff=buf,
    lumped_zeeman_coeff=zee,
    device="cpu",
    dtype=torch.float64,
)

N_SETTLE = 5000
N_AVG = 200
DT = 1e-4

disc_t2: list[float] = []
dist = torch.tensor([[T_SLICE, B_SLICE, I_SLICE]], dtype=torch.float64)
for rf_det in RF_GRID:
    ctrl = torch.tensor([[0.0, float(rf_det)]], dtype=torch.float64)
    state = twin.initial_state(batch_size=1)
    for _ in range(N_SETTLE):
        state = twin.step(state, ctrl, dist, DT)
    acc = 0.0
    for _ in range(N_AVG):
        state = twin.step(state, ctrl, dist, DT)
        acc += state[0, 6].item()
    disc_t2.append(acc / N_AVG)

disc_t2_arr = np.array(disc_t2)
# Normalise tier-2 to tier-1 peak for visual comparison
peak_t1 = max(np.max(np.abs(disc_t1)), 1e-30)
peak_t2 = max(np.max(np.abs(disc_t2_arr)), 1e-30)
scale = peak_t1 / peak_t2
disc_t2_scaled = disc_t2_arr * scale
lp_t2_raw, sl_t2_raw = _find_lock_point_and_slope(RF_GRID, disc_t2_arr)
print(f"Tier-2 lock point: {lp_t2_raw:.4f} Hz, slope: {sl_t2_raw:.4e} (unscaled)")

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(RF_GRID, disc_t1, "b-o", markersize=4, label="Tier-1 (full OBE, 16-level)")
ax.plot(RF_GRID, disc_t2_scaled, "r--s", markersize=4,
        label=f"Tier-2 (ReducedTwin, scaled x{scale:.2f})")
ax.axvline(lp_t1, color="blue", linestyle=":", alpha=0.7,
           label=f"T1 lock pt {lp_t1:.2f} Hz")
ax.axvline(lp_t2_raw, color="red", linestyle=":", alpha=0.7,
           label=f"T2 lock pt {lp_t2_raw:.2f} Hz")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("RF detuning (Hz)")
ax.set_ylabel("Discriminator signal (arb.)")
ax.set_title(
    f"CPT discriminator: tier-1 vs tier-2\\n"
    f"(T={T_SLICE} K, B={B_SLICE} uT, I={I_SLICE})"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()

figs_dir = pathlib.Path("../notebooks/figs")
figs_dir.mkdir(parents=True, exist_ok=True)
out_fig = figs_dir / "discriminator_t1_t2_overlay.png"
fig.savefig(out_fig, dpi=150)
print(f"Saved: {out_fig}")
plt.show()
"""

cells.append(nbformat.v4.new_markdown_cell(OVERLAY_MD))
cells.append(nbformat.v4.new_code_cell(OVERLAY_CODE))

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.11",
}

out_path = Path("01_calibration_audit.ipynb")
with open(out_path, "w", encoding="utf-8") as fh:
    nbformat.write(nb, fh)

print(f"Written: {out_path} ({len(cells)} cells)")
for idx, c in enumerate(cells):
    preview = c.source[:70].replace("\n", " ")
    print(f"  [{idx}] {c.cell_type}: {preview}")
