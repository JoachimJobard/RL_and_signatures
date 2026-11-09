## Decision material: which dynamical systems carry the AAAI-2027 experiments

This document verifies facts and lays out options. The selection, and any scientific rationale for it, are the owner's. Where an option would require a rationale the owner has not stated, the rationale is left blank rather than supplied. Every number below is either measured off disk or explicitly marked NOT MEASURED.

### 1. The paper's scope (established, not relitigated)

The paper is the theorem plus H1. H2 is out of scope. The three learners — analytic vertical derivative / bootstrapped actor (value gradient), Doya Equation 20 actor-critic, and no-bootstrap policy gradient — appear as "three mechanisms, one law, no claim about strength". The signature is one instantiation of the theorem among three, never a performance claim.

### 2. Where H1 is measured (the 140-run evidence base)

The five canonical `main_unified` studies are pure value-gradient: 25 + 25 + 25 + 30 + 35 = 140 runs, five seeds each. Actor-critic runs on these five plants number 0 (11 actor-critic runs exist elsewhere: 9 in `oracle_ladder_high_gap`, 2 in debug directories). Policy-gradient runs in `main_unified` number 0. **No plant on disk carries a three-learner statement.**

H1 verdict per plant (value gradient, five seeds). The metric is **not commensurable across plants**: the three linear plants are scored as $\rho$ (suboptimality against an exact delayed-LQR oracle, `metric_is_suboptimality=true`); the two Mackey–Glass plants as raw cost $J$ (`metric_is_suboptimality=false`).

| Plant | Metric | Markovian (success) | Best history variant (success) | H1 |
| --- | --- | --- | --- | --- |
| delayed_oscillator_high_gap | $\rho$ | 0.3801 (5/5) | signature depth 3, 0.1575 (5/5) | Holds |
| delayed_velocity_v2 | $\rho$ | 0.5723 (0/5) | signature depth 3, 0.3295 (5/5) | Holds |
| linear_cell | $\rho$ | 0.1267 (5/5) | signature depth 2, 0.2836 (5/5) | **Fails** |
| mackey_glass_chaotic | $J$ | 43.78 (0/5) | signature depth 2, 0.0880 (5/5) | Holds |
| mackey_glass_limit_cycle | $J$ | 30.30 (0/5) | signature depth 4, 0.0494 (5/5) | Holds |

All values MEASURED from each study's `aggregation_data.json`. H1 holds on four of five and fails on `linear_cell`. Raw history is the unreliable family (raw_history capacity 1 succeeds 0/5 on delayed_oscillator, linear_cell and mackey_glass_chaotic; raw_history capacity 3 on mackey_glass_limit_cycle is NaN at 0/5, dim 4959, three diverged). The env-identity checks confirm `linear_cell` is a genuine delay plant ($\tau=1.0>0$, $\lVert A_1\rVert_F=0.245\neq0$), so its H1 failure is a substantive negative result, not a construction artefact.

### 3. The campaign cells versus the evidence base

Campaign cells at HEAD (c5697ff): `markovian` (double_integrator), `linear_dde` (linear_dde_scalar), `hopfield_linear`, `hopfield_nonlinear`, `mg_chaotic`.

**Overlap with the five H1-measured plants is exactly one** (MEASURED, by config comparison):

| Campaign cell | Same plant in `main_unified`? | Notes |
| --- | --- | --- |
| markovian (double_integrator) | No | $A_1=0$, delay $=0$ exactly — Markovian; no `main_unified` plant is Markovian |
| linear_dde (linear_dde_scalar) | No | scalar (1-D); nearest by delay only is `linear_cell` (2-D), differing in $A, A_1, R$, step size, resolution |
| hopfield_linear | No | class `DelayedHopfieldNetwork`, absent from `main_unified` |
| hopfield_nonlinear | No | class `DelayedHopfieldNetwork`, absent from `main_unified` |
| mg_chaotic (MG_1D_chaotic) | **Yes (exact)** | bit-identical to `mackey_glass_chaotic_study` on every dynamics parameter |

Consequently, running the campaign as configured would place a fresh value-gradient study on four of five cells (no prior value-gradient baseline) and reproduce/extend an existing value-gradient study on only the fifth (`mg_chaotic`). Even on that fifth, the prior data is value-gradient only, so a three-learner statement there still requires the actor-critic and policy-gradient arms.

### 4. The double_integrator ($\tau=0$) cell

$A_1$ and the delay vector instantiate to exactly zero (MEASURED), so history is irrelevant and H1 must fail by construction — a sound falsification control. The stronger reading, that the theorem predicts the three feature maps produce the **same** control at $\tau=0$, is MEASURED FALSE at finite sample: the window does not collapse ($11$ taps), the three feature maps remain distinct, and on the on-distribution CARE-trajectory windows the two history representations give control error $0.107$ (raw_history) and $0.106$ (signature) against $\text{mean}\lvert u^\star\rvert=0.111$ while fitting $V$ to residual $\sim 10^{-8}$ — the $\ker G$ unidentified-gradient pathology the paper already documents. Agreement holds only in the exact-fit / population limit (control error $\sim 10^{-7}$ on 4000 richly-spanning random windows). Retaining double_integrator is therefore compatible with measurement **as a falsification control**, not as a three-representation consistency prediction.

### 5. The Hopfield pair

`hopfield_linear` ($\varepsilon=0$) and `hopfield_nonlinear` ($\varepsilon=1$) have bit-identical linearisation $A=-I$, $A_1=W$, $B=I$ (MEASURED, max element-wise difference $0.0$) because $\varphi_\varepsilon'(0)=1$; $\varepsilon$ moves the finite-amplitude nonlinearity — the H2 lever — with H1 held fixed. Both are genuine delay plants ($\tau=0.5>0$, $A_1=W$ with spectral radius $2$). With H2 out of scope, the second arm reduces to a matched-linearisation replication of the first rather than an independent H1 test; it is not "nothing" (a nonlinear-in-delay H1-positive plant is a distinct finite-amplitude regime) but the H1 driver $(W,\tau)$ is shared by construction. The pair consumes **40%** of the campaign (2 of 5 cells; 90 of 225 tasks). No Hopfield run exists anywhere in `main_unified`.

### 6. Position verdict

The position's three claims, against measured evidence only:

1. **"Cell list designed for H2, now out of scope."** Partially supported; the blanket form fails. The Hopfield pair (2 of 5 cells) is a measured H2 construct. The other three are H1-relevant: markovian is an H1 falsification control, linear_dde a linear-in-delay H1 plant, mg_chaotic a stress/robustness cell ("NOT the primary H2 evidence" per its config).
2. **"Cells do not match where H1 is measured; no three-learner statement on any plant."** Supported in substance, one over-statement. The overlap is one of five (mg_chaotic matches exactly), not zero. The no-three-learner-statement sub-claim holds fully (140 value-gradient, 0 actor-critic, 0 policy-gradient on the five H1 plants).
3. **"Select H1-measured cells, keep double_integrator for a $\tau=0$ consistency check, move Hopfield to Paper 2."** Partially survives. double_integrator is sound as a falsification control only; the $\tau=0$ three-representation agreement reading is measured false. Moving Hopfield out is defensible (out-of-scope H2 differentiator, 40% of tasks) but not forced (each arm is a valid H1 plant); "evaporates" overstates.

Net: a sound basis for a decision provided (i) "designed for H2" is narrowed to the Hopfield pair, (ii) the overlap is stated as one of five, and (iii) double_integrator's $\tau=0$ role is a falsification control, never a consistency prediction.

### 7. Options

Task unit: each cell contributes $3\ \text{learners}\times 3\ \text{representations}\times 5\ \text{seeds}=45$ tasks (MEASURED from the HEAD launcher; a two-learner variant dropping policy gradient is $30$/cell). The `signatures` actor-critic is wired for all three representations at HEAD (`signatures_jax.py:188`); the "could only ever run signature" constraint was historical (pre-2ae3b73).

| Option | Cell list | Tasks |
| --- | --- | --- |
| 1. Current campaign, unchanged | markovian, linear_dde, hopfield_linear, hopfield_nonlinear, mg_chaotic | 225 |
| 2. H1-measured + double_integrator, Hopfield to Paper 2 | double_integrator, delayed_oscillator_high_gap, delayed_velocity_oscillator, linear_cell, mg_chaotic, mg_limit_cycle | 270 |
| 3. H1-measured plants only | delayed_oscillator_high_gap, delayed_velocity_oscillator, linear_cell, mg_chaotic, mg_limit_cycle | 225 |
| 4. Campaign minus Hopfield | markovian, linear_dde, mg_chaotic | 135 |
| 5. Single overlap only | mg_chaotic | 45 |

Per-option buys / costs / foreclosures are given in the structured `options` field. Common facts: the SLURM ceiling is 4 h/task (cpu_p1, qos_cpu-t3, MEASURED launcher setting); the only measured runtime is a file-timestamp proxy of median 396 s (range 385–445 s) over the 25 `linear_cell` value-gradient runs — a light 2-D low-window cell — and it does **not** bound the heavy cells (mg_chaotic window 71, raw_history dim 2627) or the actor-critic / policy-gradient arms, whose runtimes are NOT MEASURED; total elapsed wall-clock is NOT MEASURED (array concurrency is scheduler-dependent). Options 2 and 3 require a launcher code change (owner's): `delayed_oscillator_high_gap`, `delayed_velocity_oscillator` and `linear_cell` (= `delay_jax.yaml`) have complete env configs but no `case` entry in `rl_array.slurm`, and their previous effective windows are not recorded in the launcher, so bit-for-bit reproduction of the prior value-gradient numbers is NOT guaranteed.

### 8. Cross-cutting caveats (independent of the cell list)

These bear on any three-learner comparison on any cell list and are the audit's FIX-FIRST items (not re-measured here): `++agent.training.tau_polyak=1.0` is now set at HEAD (`rl_array.slurm:129`); the policy gradient takes about 16× fewer optimiser steps per episode than the actor-critic; policy-gradient `cost_reduction_pct` is unmeasured on every cell (only a 40-episode, seed-0 stability probe exists); actor-critic output validity at the 1000-episode budget on markovian and raw_history is NOT MEASURED, and smoke runs recorded divergence to NaN on mg_chaotic / raw_history.

### 9. Not measured

Per-task logged wall-clock (no elapsed field in `run_context.yaml`); heavy-cell and non-value-gradient runtimes; total elapsed wall-clock per option; the launcher's bit-comparability claim for the pre-existing cells; the previous effective windows of the three not-yet-dispatchable H1 plants; any H1, actor-critic or policy-gradient result on the two Hopfield plants, on double_integrator, or on linear_dde_scalar.