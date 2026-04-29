"""Quick benchmark: realtime factor across batch sizes, devices, compiled or eager."""

from __future__ import annotations

import time

import torch

from cptservo.twin.reduced import ReducedTwin


def bench(
    device: str, batch_size: int, n_steps: int = 10_000, compile_step: bool = False
) -> tuple[float, float]:
    twin = ReducedTwin(device=device)
    step_fn = twin.step
    if compile_step:
        try:
            step_fn = torch.compile(twin.step, mode="reduce-overhead")
        except Exception as e:
            print(f"  compile failed: {e}")
            return 0.0, 0.0

    state = twin.initial_state(batch_size=batch_size)
    controls = torch.zeros(batch_size, 2, dtype=torch.float64, device=device)
    disturbance = torch.tensor(
        [333.15, 50.0, 1.0], dtype=torch.float64, device=device
    ).expand(batch_size, 3)

    # Warmup
    for _ in range(100):
        state = step_fn(state, controls, disturbance, 1.0e-4)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            state = step_fn(state, controls, disturbance, 1.0e-4)
    if device == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    sim_time_s = n_steps * 1.0e-4
    rt_factor = sim_time_s / wall
    sample_seconds_per_wall_second = batch_size * sim_time_s / wall
    return rt_factor, sample_seconds_per_wall_second


def main() -> None:
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
    print()
    print(f"{'device':<6} {'compiled':<9} {'B':>6} {'rt_factor':>12} {'sample_s/wall_s':>18}")
    print("-" * 60)
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for compiled in [False, True]:
        for device in devices:
            for B in [1, 64, 1024]:
                rt, sps = bench(device, B, compile_step=compiled)
                tag = "yes" if compiled else "no"
                print(f"{device:<6} {tag:<9} {B:>6} {rt:>12.2f} {sps:>18.1f}")


if __name__ == "__main__":
    main()
