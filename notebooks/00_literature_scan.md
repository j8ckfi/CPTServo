# M1 Literature Scan -- CPTServo Project
**Date:** 2026-04-27
**Scope:** 2024-2026 primary; 2023 included where highly relevant
**Searcher:** Document-Specialist subagent (claude-sonnet-4-6)
**Purpose:** Hard-kill gate. Determine whether a 2024-2026 paper claims a reinforcement-learning, learned, or differentiable-physics servo for a chip-scale CPT/vapor-cell atomic clock with sigma_y improvement.

---

## Databases and Search Terms Covered

**Databases searched:** arXiv (physics.atom-ph, quant-ph, physics.ins-det, cs.LG), Google Scholar, IEEE Xplore (via web search), OSA/Optica (Optics Express, Optics Letters, CLEO proceedings), AIP (Applied Physics Letters, Review of Scientific Instruments), APS (Phys Rev Applied, Phys Rev A, Phys Rev Research), Nature (Scientific Reports, Nature Communications, Nature Photonics), Metrologia (via Google Scholar), Sensors (MDPI), IEEE Trans IM, IEEE Trans UFFC, IFCS-EFTF 2024/2025 proceedings, Justia/Google Patents.

**Search term combinations executed (all with 2023-2026 date filters):**
- "reinforcement learning" + "atomic clock"
- "reinforcement learning" + "CPT" OR "vapor cell" OR "frequency stabilization"
- "deep learning" / "neural network" + "servo" + "atomic" OR "clock" OR "CPT"
- "neural network" + "CSAC" OR "chip-scale atomic clock"
- "machine learning" + "frequency lock"
- "differentiable simulation" + "atomic" OR "clock" OR "spectroscopy"
- "analytic policy gradient" + "physics" OR "servo"
- "learned controller" + "atomic clock" OR "frequency reference"
- "physics-informed neural network" / "PINN" + "atomic clock" OR "frequency standard"
- "Allan deviation" + "reinforcement learning" OR "neural network" + clock
- "frequency stabilization" + "PPO" OR "SAC" OR "policy gradient" + laser/optical
- "CPT" + "machine learning" OR "learned controller" + frequency stability
- Specific author sweeps: Kitching, Knappe, Newman, Maurice, Pottie, Esnault, Boudot, Hummon, Bandi, Shah, Givon, Gallego, Krzyzewski
- "Mesa Quantum" (technical terms, patents, publications)
- "Vescent" + atomic/optical clock
- IFCS-EFTF 2024 and 2025 program sweeps
- arXiv physics.atom-ph monthly listings (2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2025-02, 2025-03, 2025-10)

**Total papers reviewed or confirmed absent:** 34

---

## Section 1 -- Killer Prior Art (if any)

**No killer prior art was found.** The subsections below document all findings and assessments.

---

### 1.1 Near-Miss Papers

#### [NM-1] Atomic clock locking with Bayesian quantum parameter estimation

**Citation:** Han, C. et al. "Atomic clock locking with Bayesian quantum parameter estimation: scheme and experiment." arXiv:2306.06608, submitted June 2023, updated October 2024. Published Phys. Rev. Applied 22, 044058 (2024). DOI: 10.1103/PhysRevApplied.22.044058

**Summary:** Implements an adaptive Bayesian quantum frequency estimation protocol on a cold-atom CPT clock (not chip-scale; cold atoms in a vacuum system). Claims 5.1(4) dB improvement in fractional frequency stability vs. conventional PID locking. The scheme uses Bayesian inference to optimally update frequency estimates from quantum measurements -- this is classical statistical estimation, not a trained neural network or RL policy.

**Verdict: NEAR-MISS.** The system is cold-atom (not chip-scale hot-vapor CPT). The control algorithm is Bayesian parameter estimation, not a learned/trained servo policy. The sigma_y improvement claim is real and notable, so the framing of CPTServo must distinguish: (a) hot-vapor chip-scale vs. cold-atom, (b) trained RL/APG policy vs. Bayesian estimator. Cite this paper explicitly as the closest statistical-control prior art and explain why APG through a differentiable twin is a distinct approach with different SWaP constraints and deployment target.

---

#### [NM-2] Artificial neural networks for laser frequency stabilization (CLEO 2024)

**Citation:** Winkler, L. and Noelleke, C. (TOPTICA Photonics AG). "Artificial neural networks for laser frequency stabilization." CLEO 2024, paper SW3H.4. https://opg.optica.org/abstract.cfm?uri=CLEO_SI-2024-SW3H.4

**Summary:** A neural-network approach for automatic laser frequency lock acquisition. The NN identifies the target spectral line under varying operating conditions and achieves the lock autonomously. No atomic clock, no CPT physics package, no vapor cell, no Allan deviation claim. This is laser-side acquisition automation for general spectroscopy.

**Verdict: NEAR-MISS (framing only).** Not a chip-scale CPT clock servo. Technique is lock-acquisition automation, not a trained policy that improves long-term sigma_y. Cite as evidence that NN-based laser control is being explored commercially (TOPTICA), and clearly distinguish: CPTServo targets the microwave servo loop and system-level sigma_y, not laser lock acquisition.

---

#### [NM-3] Neural Network Assisted Laser Frequency Locking System (JLT 2023)

**Citation:** Ding, S. et al. "Neural Network Assisted Laser Frequency Locking System." Journal of Lightwave Technology, vol. 41, no. 15, pp. 4915-4921, 2023. https://opg.optica.org/jlt/abstract.cfm?URI=jlt-41-15-4915

**Summary:** A feedforward neural network replaces part of the PID algorithm to reduce nonlinearity in the error signal of a laser frequency locking system. Claims 2x reduction in peak-to-peak deviation and RMS error vs. standard PID. Application domain: optical communications and CV-QKD, not atomic clocks or vapor cells.

**Verdict: NEAR-MISS (framing only).** No CPT physics package, no vapor cell, no Allan deviation metric. The NN is a static nonlinearity compensator, not a trained RL policy through a differentiable twin. Cite as NN-augmented PID prior art for laser locking; distinguish from system-level atomic clock servo improvement.

---

#### [NM-4] Linien: FPGA-based tool with ML spectroscopy optimization (RSI 2022)

**Citation:** Wiegand, B., Leykauf, B., Jorders, R., Krutzik, M. "Linien: A versatile, user-friendly, open-source FPGA-based tool for frequency stabilization and spectroscopy parameter optimization." Review of Scientific Instruments 93, 063001 (2022). arXiv:2203.02947. DOI: 10.1063/5.0090384

**Summary:** Open-source FPGA lock tool for laser spectroscopy with ML-based automatic spectroscopy parameter optimization (lock-point selection). The ML component is heuristic search/optimization on spectroscopy scan parameters, not a trained RL/APG policy improving Allan deviation. Not applied to atomic clocks or CPT vapor cells specifically.

**Verdict: NEAR-MISS (framing only).** Generic laser spectroscopy tooling. The "machine learning" is parameter optimization, not differentiable-physics policy gradient. No sigma_y claim. Cite as tooling context and distinguish.

---

#### [NM-5] Fast surrogate modelling of EIT in atomic quantum systems using LSTM (arXiv:2510.02603)

**Citation:** Burdon Hita, I.S. et al. "Fast surrogate modelling of EIT in atomic quantum systems using LSTM neural networks." arXiv:2510.02603, October 2025.

**Summary:** Uses LSTM networks to build a fast surrogate model of EIT (electromagnetically induced transparency) spectra in atomic systems. EIT is the two-laser analog of CPT (single laser implies CPT; bichromatic field implies EIT -- same dark-state physics). This is a physics surrogate/emulator, not a control policy or servo. No clock performance claims.

**Verdict: NEAR-MISS (conceptually adjacent).** A differentiable surrogate of EIT/CPT-like atomic physics is exactly the "digital twin" building block CPTServo proposes. This paper motivates our approach but does not implement a servo or claim sigma_y improvement. Cite as motivation for differentiable atomic-physics surrogates; our work extends to closed-loop servo training via APG.

---

### 1.2 Adjacent Papers (cite and move on)

| ID | Citation | Verdict note |
|----|----------|-------------|
| ADJ-1 | Rivera-Aguilar et al. 2025, arXiv:2503.01681, Sci Reports. Microcell CPT 10^-12 at 1 day via SABR-CPT + FPGA. | Physics-package SOTA; classical control only. Sets performance target. |
| ADJ-2 | Vishnyakov et al. arXiv:2502.11587 (Feb 2025). PDH-style B-field shift reduction in CPT clock. | Classical modulation technique; no ML. |
| ADJ-3 | Chen et al. APL 126, 224002 (June 2025). CPT CSAC with chip-ECDL replacing VCSEL. | Hardware improvement; no ML servo. |
| ADJ-4 | Anton et al. arXiv:2312.13397 (2023/2024). ML benchmarking for cold-atom MOT optimization. | ML for atom number, not clock stability or servo. |
| ADJ-5 | "Model-free optical processors using in situ RL with PPO." Light: Sci & Appl 2025, arXiv:2507.05583. | PPO for diffractive optical processor; not atomic clocks. |
| ADJ-6 | Multistability RL in mode-locked fiber laser. Sci Rep 2024 (PMC11501165). | RL for laser cavity state; not clocks. Already known to team. |
| ADJ-7 | Kazda et al. 2019. Classical optimal control on atomic clocks (Pontryagin). | Closest non-PID classical prior art. Already known. |
| ADJ-8 | Wei et al. 2020. ANN for clock ensemble timescale. MAPAN. | ANN at ensemble level, not single-clock servo. Already known. |
| ADJ-9 | "Disciplining a Rb Atomic Clock Based on Adaptive Kalman Filter." Sensors 24, 4495 (2024). | Adaptive Kalman filter for Rb clock disciplining; not RL, not CPT chip-scale. |
| ADJ-10 | "Realizing a deep RL agent for real-time quantum feedback." Nature Comms 2023. DOI: 10.1038/s41467-023-42901-3. | RL for qubit state init on FPGA; not a frequency servo. |
| ADJ-11 | "Robust quantum control using RL from demonstration." npj Quantum Inf 2025. | Quantum gate control via RL; not atomic clock frequency servo. |
| ADJ-12 | "Semiparametric + ML for satellite clock bias prediction." Sci Reports 2025. | ML for GNSS clock prediction (ensemble timescale); not physics-package servo. |

---

### 1.3 Confirmed Negative Results (searched, nothing found)

The following combinations were explicitly searched and returned zero papers matching the kill criterion (RL/differentiable-physics/learned servo on chip-scale CPT or hot-vapor atomic clock with sigma_y improvement):

- arXiv query "CPT clock machine learning" -- zero results returned by arXiv search engine.
- arXiv query "neural network atomic clock servo" -- zero results.
- arXiv query "atomic clock reinforcement learning" -- zero physics papers; only robotics/CS hits.
- IEEE Xplore (via web): "chip-scale atomic clock machine learning servo 2023-2025" -- no matching papers.
- Phys Rev Applied 2024-2025: "atomic clock machine learning servo" -- no matching papers.
- Metrologia: "atomic clock machine learning reinforcement" -- no matching papers.
- Sensors MDPI: "atomic clock machine learning servo frequency stability 2024-2025" -- no matching papers.
- PINN + "atomic clock" OR "frequency standard" 2024-2025 -- no matching papers.
- "differentiable simulation" + "atomic" + "spectroscopy" + "clock" -- no matching papers.
- "analytic policy gradient" + "physics" OR "atomic" -- no papers on atomic clock applications.
- "learned controller" + "frequency reference" OR "atomic oscillator" -- no matching papers.
- "Allan deviation" + "reinforcement learning" OR "neural network" + clock/oscillator 2023-2025 -- no matching papers on atomic physics frequency standards.
- IFCS-EFTF 2025 program (Queretaro, May 2025): no indexed sessions or papers combining ML + CPT + servo.
- EFTF 2024 (Neuchatel): web-search sweep of EFTF 2024 + ML + CPT + servo returned zero.
- Author sweep (Kitching, Knappe, Newman, Maurice, Pottie, Esnault, Boudot, Hummon, Bandi, Shah, Givon, Gallego, Krzyzewski) 2024-2026: zero papers combining ML with CPT clock servo control.
- Mesa Quantum: no technical publications found. One abandoned patent (US20250013204A1) uses conventional PID/PLL only.
- Vescent: optical clock product line, no ML servo publications.
- "Jetson" + "atomic clock" + "machine learning servo" -- zero papers.

---

## Section 2 -- Mandatory Cite List for the Writeup

| # | Citation | What they did / Why we cite |
|---|----------|-----------------------------|
| 1 | Kazda et al. 2019, classical optimal control on atomic clocks via Pontryagin maximum principle. | Closest non-PID classical control prior art. Establishes the gap our learned servo fills. |
| 2 | Wei et al. 2020, ANN for clock ensemble timescale algorithm. MAPAN. | ANN applied to clocks at ensemble/prediction level, not single-clock physics-package servo. |
| 3 | Rivera-Aguilar et al. 2025, arXiv:2503.01681, Sci Reports. Microcell CPT 10^-12 at 1 day via SABR-CPT. | Physics-package SOTA; sets the performance target. Classical control only; we ask whether ML servo can reach or surpass it under disturbance. |
| 4 | Chen et al. APL 126, 224002 (2025). CPT CSAC with chip-ECDL. | Hardware lineage; shows CPT CSAC is an active area. Our work adds a software-defined ML servo to this family. |
| 5 | Han et al. Phys. Rev. Applied 22, 044058 (2024). arXiv:2306.06608. Bayesian clock locking, 5.1 dB sigma_y improvement. | Closest statistical-learning prior art for sigma_y improvement via non-PID control. Distinguish: cold-atom vs hot-vapor chip-scale; Bayesian estimator vs trained APG policy. |
| 6 | Winkler & Noelleke, CLEO 2024 SW3H.4. ANN for laser frequency lock acquisition (TOPTICA). | Commercial NN-based locking. Distinguishes laser-side acquisition from system-level microwave servo. |
| 7 | Ding et al. JLT 41:15, 2023. NN-assisted laser frequency locking. | NN nonlinearity compensator for laser lock. Distinguishes static NN compensator from trained RL policy through differentiable twin. |
| 8 | Wiegand et al. RSI 93, 063001 (2022), arXiv:2203.02947. Linien FPGA tool + ML spectroscopy optimization. | Open-source context; ML used for lock-point selection heuristics, not sigma_y optimization. |
| 9 | Burdon Hita et al. arXiv:2510.02603 (Oct 2025). LSTM surrogate for EIT in atomic systems. | Motivates differentiable atomic-physics surrogates; our work extends to closed-loop servo training via APG. |
| 10 | Kitching et al. Appl. Phys. Rev. 5, 031302 (2018). Chip-scale atomic devices. | Physics-package lineage; establishes the CSAC design space CPTServo targets. |
| 11 | Knappe, S. and Krzyzewski, S. US20250013204A1 (U. Colorado, filed 2024). LED-based CPT CSAC. | Mesa Quantum IP lineage. Core technology uses PID/PLL only; our servo is a direct upgrade path. |
| 12 | "Multistability RL in mode-locked fiber laser." Sci Rep 2024, PMC11501165. | RL on a laser system (different domain). Establishes RL feasibility for optical-physics control; distinguishes from clock servo. |
| 13 | Anton et al. arXiv:2312.13397 (2024). ML benchmarking for cold-atom optimization. | ML-for-AMO tooling context; not clock stability. |
| 14 | "Disciplining a Rb Atomic Clock Based on Adaptive Kalman Filter." Sensors 24, 4495 (2024). | Adaptive filter for Rb clock disciplining; not RL, not CPT chip-scale. Establishes that adaptive control is explored at the disciplining level. |

---

## Section 3 -- Specific Mesa-Relevant Context

**Company overview:** Mesa Quantum (formerly Mesa Quantum Systems Inc.), incorporated December 2024, Albuquerque NM / Boulder CO. CEO/co-founder: Sristy Agrawal. CTO/co-founder: Wale Lawal. Technology transfer from Knappe lab at University of Colorado.

**Funding:** $3.7M seed round; $1.9M SpaceWERX Phase II SBIR contract for miniaturized atomic clocks; $500K Colorado OEDIT grant (April 2025; first recipient of New Mexico state quantum grant). Associated with Harvard Innovation Labs.

**Technology:** Chip-scale vapor-cell quantum sensors. Product roadmap: atomic clocks (GPS-denied sync, defense, autonomous systems), gyroscopes, magnetometers, Rydberg RF sensors, gravimeters, accelerometers. First atomic clock prototype expected 2026. Currently using Vescent MIRIDIAN1 optical atomic clock at Elevate Quantum shared commercialization lab in Albuquerque for reference/characterization (confirmed December 2025).

**Knappe/Krzyzewski patent:** US20250013204A1. Filed April 15, 2024. Published January 9, 2025. Status: Abandoned. Inventors: Svenja Knappe (Boulder CO), Sean Krzyzewski (Albuquerque NM). Assignee: University of Colorado Colorado Springs. Describes a chip-scale CPT atomic clock using LED or quantum-dot laser source (replacing expensive VCSEL) with narrowband optical filter and Rb/Cs vapor cell. Control architecture stated in all 20 claims: conventional PID loop and PLL. Zero ML content. Priority date July 2, 2020. The abandoned status likely reflects re-filing or assignment to Mesa Quantum directly rather than through CU -- worth confirming in Mesa outreach but not a concern for CPTServo novelty.

**Related earlier patent:** US20230384737A1 (Google Patents) appears to be a related Knappe/CU filing on similar LED-source CPT clock technology.

**What Mesa has NOT published:** No technical papers, conference abstracts (IFCS, EFTF, CLEO, APS March Meeting, etc.), or arXiv preprints attributed to Mesa Quantum or its founders (Lawal, Agrawal) were found in any searched database as of April 2026. The company is pre-publication and pre-prototype. This is a significant commercial gap: they have physics-package IP and SBIR funding but no public control-software methodology. CPTServo directly addresses this gap.

**Vescent collaboration context:** Vescent (Golden CO) sells the MIRIDIAN1 optical atomic clock and recently partnered with Danish National Metrology Institute (DFM) for a user-configurable optical clock achieving 200 fs timing instability at 1 s, demonstrated at PTTI January 2024. Vescent technology is Sr/Yb optical lattice class -- several orders of magnitude better stability than CPT-Rb87 chip-scale but also orders of magnitude larger, heavier, and more power-hungry. Mesa uses Vescent as a lab reference instrument, not as their target product form factor.

**Strategic framing for Mesa pitch:** Mesa has the physics package (vapor cell, VCSEL/LED laser, Rb87 vapor chemistry, microfabrication). They do not have a software-defined servo that can be updated post-deployment, adapts to component aging or temperature drift, or has been trained against a characterized disturbance model. CPTServo is a direct plug-in upgrade to their control stack. The Jetson Orin Nano deployment target matches their stated application space (autonomous systems, defense -- both Jetson-class embedded compute). The digital-twin training methodology also provides Mesa with a fast simulation tool for physics-package design space exploration.

---

## Section 4 -- Verdict

**CLEAR.**

After searching 34 papers and conference-proceedings entries across arXiv (physics.atom-ph, quant-ph, cs.LG, physics.ins-det), IEEE Xplore, AIP Publishing, APS journals, Optica/OSA, Nature/Scientific Reports, Metrologia, Sensors (MDPI), IFCS-EFTF 2024 and 2025, and patent databases -- using all required search-term combinations and targeted author sweeps -- no paper was found that implements a reinforcement-learning, differentiable-physics, or learned servo on a chip-scale CPT or hot-vapor-cell atomic clock with a claimed Allan deviation improvement. The five near-miss papers are each disqualified from killer status: NM-1 uses cold atoms and Bayesian estimation (not RL/APG on chip-scale hardware); NM-2 and NM-3 address laser lock acquisition in non-clock systems with no sigma_y metric; NM-4 is a generic spectroscopy lock tool with heuristic parameter search; NM-5 is a physics surrogate with no servo or sigma_y claim. The specific combination of (1) differentiable digital twin of a chip-scale CPT-Rb87 physics package, (2) analytic policy gradient training through that twin, and (3) Jetson Orin Nano deployment with sigma_y comparison against PI and LQR baselines has no direct prior art in the 2023-2026 literature. The project may proceed to implementation.
