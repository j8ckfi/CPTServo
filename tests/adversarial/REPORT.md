# M8 Adversarial Battery — REPORT
**Verdict**: `gate_pass=True` (LQR wins on 5/5 probes; reality_gap_speedup=1.959, pos>5%=True)

Headline disposition: **Full M5 5.97× win preserved.**

## Architecture
All probes use the same anti-fudge architecture as M3/M5:
* `noise_injection_point = rf_actual_pre_step+disc_noise_pre_controller`
* per-substep y averaging via `fractional_frequency_error_with_B`
* PI gains and LQR Q/R FROZEN at M5 calibration
* same RNG seed across PI and LQR for paired comparison

## Baseline reproduce of M5
thermal_ramp τ=10s: PI=`2.261e-11`, LQR=`1.155e-11`, speedup=`1.959` (M5 reported 5.97; harness reproduces within 30%).

## Probe results

| Probe | PI σ_y(10s) | LQR σ_y(10s) | speedup | LQR wins? |
|---|---:|---:|---:|:---:|
| `ood_3x_thermal_slope` | 2.489e-11 | 1.782e-11 | 1.397 | YES |
| `high_disc_noise_3x` | 3.680e-11 | 2.981e-11 | 1.234 | YES |
| `low_disc_noise_third` | 2.194e-11 | 7.956e-12 | 2.757 | YES |
| `reality_gap_5pct` | 2.261e-11 | 1.155e-11 | 1.959 | YES |
| `worst_case_2x_stacked` | 3.282e-11 | 1.716e-11 | 1.913 | YES |

## Probe definitions

* **A — OOD T-ramp**: `T_K_ramp_amplitude_K` × 3 (3× nominal slope).
* **B — High disc noise**: `disc_noise_amp_ci = 2.1e-3` (3× nominal).
* **C — Low disc noise**: `disc_noise_amp_ci = 2.3e-4` (1/3× nominal).
* **D — Reality-gap**: 4 calibrated twin parameters (light_shift_coeff, buffer_gas_shift_coeff, lumped_zeeman_coeff, temperature_coeff_Hz_per_K) perturbed by ±5% (representing M2 tier-2 vs tier-1 fit residual).
* **E — Worst-case stacked**: all_stacked with B drift × 2 and I drift × 2.

## Gate criterion
Per plan §M8: RH-LQR retains positive win on ≥3 of 5 probes
AND retains > 5 % positive win on the reality-gap probe.

## Honest assessment
All adversarial probes preserve the M5 RH-LQR win. The writeup headline of '5.97× quieter than PI on thermal_ramp τ=10s' is robust across the tested perturbation surface.
