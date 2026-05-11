# M8 Adversarial Battery — REPORT
**Verdict**: `gate_pass=True` (LQR wins on 5/5 probes; reality_gap_speedup=2.975, pos>5%=True)

Headline disposition: **Robust positive LQR win preserved.**

## Architecture
All probes use the same anti-fudge architecture as M3/M5:
* `noise_injection_point = rf_actual_pre_step+disc_noise_pre_controller`
* per-substep y averaging via `fractional_frequency_error_with_B`
* PI gains and LQR Q/R FROZEN at M5 calibration
* same RNG seed across PI and LQR for paired comparison

## Baseline reproduce of M5
thermal_ramp τ=10s: PI=`2.981e-11`, LQR=`1.002e-11`, speedup=`2.975`. This 200-second adversarial harness is not used to reproduce the M5 11.55x magnitude; it tests whether the frozen LQR retains a positive win under perturbations.

## Probe results

| Probe | PI σ_y(10s) | LQR σ_y(10s) | speedup | LQR wins? |
|---|---:|---:|---:|:---:|
| `ood_3x_thermal_slope` | 6.062e-10 | 1.651e-11 | 36.710 | YES |
| `high_disc_noise_3x` | 5.657e-11 | 3.000e-11 | 1.886 | YES |
| `low_disc_noise_third` | 1.825e-11 | 3.973e-12 | 4.593 | YES |
| `reality_gap_5pct` | 2.981e-11 | 1.002e-11 | 2.975 | YES |
| `worst_case_2x_stacked` | 2.903e-11 | 1.639e-11 | 1.771 | YES |

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
All adversarial probes preserve a positive RH-LQR win. The robust conclusion is not that the exact 11.55x M5 magnitude reappears here; it is that the frozen LQR remains ahead of PI across the tested perturbation surface.
