"""Verify the C residual controller against the Python CfCDirectController."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cptservo.policy.ml_research import CfCDirectController  # noqa: E402


def _binary_path() -> Path:
    candidates = [
        ROOT / "c_export" / "test_residual",
        ROOT / "c_export" / "test_residual.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("test_residual binary not found; run make all in c_export")


def _format_input(seq: list[tuple[float, float, float, float]]) -> str:
    lines = []
    for row in seq:
        parts = []
        for value in row:
            if math.isnan(value):
                parts.append("nan")
            else:
                parts.append(f"{value:.17g}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _make_sequence(n_random: int = 2000) -> list[tuple[float, float, float, float]]:
    rng = np.random.default_rng(1729)
    seq: list[tuple[float, float, float, float]] = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 400.0, 50.0, 1.0),
        (2.5e-4, math.nan, 50.25, 1.01),
        (0.2, 333.15, 50.0, 1.0),
        (8.0e-4, 333.5, 49.5, 0.98),
        (-8.0e-4, 332.8, 50.5, 1.02),
    ]
    errors = rng.uniform(-1.0e-3, 1.0e-3, size=n_random)
    temps = rng.uniform(332.0, 335.0, size=n_random)
    fields = rng.uniform(49.0, 51.0, size=n_random)
    intensities = rng.uniform(0.95, 1.05, size=n_random)
    seq.extend(
        (float(e), float(t), float(b), float(i))
        for e, t, b, i in zip(errors, temps, fields, intensities, strict=True)
    )
    return seq


def main() -> int:
    checkpoint = ROOT / "models" / "cfc_residual_m11_promoted.json"
    controller = CfCDirectController.load(checkpoint)
    seq = _make_sequence()

    py_outputs = []
    for error, T_K, B_uT, I_norm in seq:
        _, rf = controller.step(error, {"T_K": T_K, "B_uT": B_uT, "I_norm": I_norm})
        py_outputs.append(float(rf))
    py_arr = np.asarray(py_outputs, dtype=np.float64)

    proc = subprocess.run(
        [str(_binary_path())],
        input=_format_input(seq),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(f"FAIL: C binary exited with code {proc.returncode}")
        return proc.returncode

    c_arr = np.asarray([float(line) for line in proc.stdout.splitlines()], dtype=np.float64)
    if c_arr.shape != py_arr.shape:
        print(f"FAIL: output length mismatch python={py_arr.size} c={c_arr.size}")
        return 1

    diffs = np.abs(py_arr - c_arr)
    worst_idx = int(np.argmax(diffs))
    worst = float(diffs[worst_idx])
    tolerance = 1.0e-9
    if worst < tolerance:
        print(f"PASS: compared {py_arr.size} samples; max |diff| = {worst:.3e} Hz")
        print(f"worst index={worst_idx} python={py_arr[worst_idx]:.17g} c={c_arr[worst_idx]:.17g}")
        return 0

    print(f"FAIL: compared {py_arr.size} samples; max |diff| = {worst:.3e} Hz")
    print(f"worst index={worst_idx} python={py_arr[worst_idx]:.17g} c={c_arr[worst_idx]:.17g}")
    print(f"input={seq[worst_idx]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
