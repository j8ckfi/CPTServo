"""Quick smoke test: 500 s closed-loop on Kitching conditions, sigma_y at multiple tau.

Verifies the noise budget calibration before committing to a full 3-hour M3 audit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_m3_m4_gates import make_calibrated_twin, run_fast_loop  # noqa: E402

from cptservo.baselines.pi import PIController  # noqa: E402
from cptservo.twin.allan import overlapping_allan  # noqa: E402
from cptservo.twin.disturbance import Disturbance  # noqa: E402


def main() -> None:
    print(f"[{time.strftime('%H:%M:%S')}] Kitching-conditions smoke test (500 s clean)...")
    twin = make_calibrated_twin(cell_temperature_K=360.0, buffer_pressure_torr=100.0)
    pi = PIController.from_recipe()
    dist = Disturbance.from_recipe("clean")
    trace = dist.generate(duration_s=500.0, sample_rate_Hz=10_000.0, seed=42)

    t0 = time.perf_counter()
    res = run_fast_loop(twin, pi, trace, 500.0)
    wall = time.perf_counter() - t0
    print(f"[{time.strftime('%H:%M:%S')}] sim wall {wall:.0f} s")

    y = res["y"] - np.mean(res["y"])
    taus = [1.0, 10.0, 100.0]
    sig = overlapping_allan(y, 1000.0, taus)

    targets = {1.0: 4e-11, 10.0: 1.3e-11, 100.0: 4e-12}  # Kitching 2018
    print("\ntau_s     twin_sigma_y    kitching       ratio")
    for tau in taus:
        s = sig[tau]
        t = targets[tau]
        r = s / t
        verdict = "PASS" if 0.5 < r < 2.0 else "FAIL"
        print(f"{tau:7.1f}   {s:.3e}     {t:.3e}     {r:.3f}   {verdict}")


if __name__ == "__main__":
    main()
