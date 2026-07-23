# H1 representation comparison — selection and final protocol (pre-registration)

Date fixed: 2026-07-23. This document is written **before** the runs it governs are launched, and
is not to be edited on the basis of their outcome. Any deviation is recorded as an explicit,
dated amendment at the end, stating what changed and why.

## 1. Question

H1: for a continuous-time control task with state delay, does the **history/signature
representation** of the observation improve control quality, and does the improvement grow with the
delay? The comparison is a $3 \times 3$ factorial — three learners
$\{\text{value\_gradient},\ \text{actor\_critic},\ \text{policy\_gradient}\}$ against three
representations $\{\text{markovian},\ \text{raw\_history},\ \text{signature}\}$ — evaluated across a
delay ladder of three environments.

## 2. Environments (delay ladder, one representative per regime)

| Cell | Dynamics | Delay | Role |
|---|---|---|---|
| `harmonic_oscillator` | ODE (no delay) | $0$ | H1 must **fail** here (markovian state is sufficient) |
| `linear_dde_scalar` | linear delay differential equation | fixed | clean linear-DDE test of H1 |
| `MG_1D_limit_cycle` | Mackey–Glass (nonlinear DDE), limit cycle | $6$ | nonlinear-DDE regime |

`MG_1D_chaotic` (a second, harder nonlinear DDE) is **deferred** as a harder regime; its exclusion
is a stated scope decision, not an outcome-conditioned drop. It is recorded that, on the earlier
biased campaign, the signature representation was *also* in difficulty there (value_gradient
signature 52 % / actor_critic signature near 0 %), so its exclusion does not conceal a signature
success.

## 3. Motivation — three confounds, each measured, each corrected here

1. **Aggregation bias.** Diverged seeds were previously dropped from the denominator, and %opt was
   averaged over positive seeds only, giving optimistic point estimates. *Correction*: report two
   axes (convergence rate; %opt conditional on convergence), median over converged seeds.
2. **Single-seed selection.** Learning rates selected on seed 42 alone gave 77–102 % on seed 42 but
   collapsed to $-147$ … $74$ % on seeds 0–4 — the selection overfits the seed. *Correction*:
   select on **three** seeds by the median, and report on **disjoint** test seeds.
3. **Insufficient / non-uniform budget.** At a fixed 1000-episode budget the value_gradient +
   signature combination had **not** plateaued (late-gain ratio $0.36$, $0/5$ plateaued on
   `harmonic_oscillator`), so it was measured mid-climb and understated. *Correction*: **adaptive
   budget** — train each configuration to its own plateau (patience-based early stopping) under a
   common cap; a configuration that does not plateau within the cap is flagged, not silently
   under-measured.

## 4. Knobs — fixed a priori vs selected

**Fixed a priori** (identical across all cells and combinations, no per-combination tuning):
Ornstein–Uhlenbeck exploration correlation time $\tau_n = 1.0$, discount enabled with rate $\gamma$
per cell (`harmonic` $0.5$, `linear_dde` $0.5$, `MG_limit` $0.10$),
`actor_optimizer`/`critic_optimizer` $\to$ Adam, one rollout per update, evaluation interval $50$
episodes, divergence-abort threshold $\lVert x \rVert > 10^4$, master seed shared across every
learner and representation within a seed.

**Selected** by the procedure of Section 5, at two distinct granularities:

- **Per cell, shared across all nine combinations of the cell**: the exploration-noise standard
  deviation $\sigma \in \{0.3, 0.5\}$. This is a shared exploration level, **never** tuned per
  combination — a per-combination $\sigma$ would advantage whichever representation the noise level
  happened to suit, the confound rejected in the earlier campaign. Holding $\sigma$ common across
  the nine combinations keeps the representation comparison at matched exploration.
- **Per combination**: the learning rate — `critic_lr` for value_gradient, `actor_lr` for
  policy_gradient, the pair $(\text{actor\_lr}, \text{critic\_lr})$ for actor_critic.

The learning-rate grid is **identical across representations** within a learner, so no
representation is advantaged by a finer grid; it is run **at each** $\sigma$:

| Learner | Grid | Points |
|---|---|---|
| value_gradient | `critic_lr` $\in \{0.03, 0.1, 0.3, 1\}$ | 4 |
| policy_gradient | `actor_lr` $\in \{0.01, 0.1, 1, 10\}$ | 4 |
| actor_critic | `actor_lr` $\in \{0.3, 3, 30\}$, `critic_lr` $\in \{0.01, 1\}$ | 6 |

## 5. Selection procedure

- **Seeds**: $\{40, 41, 42\}$ (three), disjoint from the test seeds.
- **Budget**: cap `n_episodes = 2000`, `patience = 10` evaluations (a 500-episode no-improvement
  window at `eval_interval = 50`), best-checkpoint restore on stop, `monitor_snr = true`.
- **Stability gate** — a $(\text{combination}, \text{learning rate})$ is *admissible* only if, on
  **at least two of the three** selection seeds, all of:
  1. no `[DIVERGED]` marker ($\lVert x \rVert$ never exceeded $10^4$);
  2. **plateau reached** — early stopping fired before the cap, or, if the cap was reached, the
     late-gain ratio (absolute performance gain over the last 20 % of training divided by the total
     range) is below $0.05$;
  3. **signal present** — median actor-gradient signal-to-noise ratio over the second half of
     training is at least $1$.
- **Grid extent**: the full learning-rate grid $\times$ nine combinations $\times$ three seeds is
  run at **each** $\sigma \in \{0.3, 0.5\}$ (a doubling of the selection cost, on billed `cpu_p1`).
- **Per-cell $\sigma$ selection (representation-neutral)**: a combination is *admissible at $\sigma$*
  if at least one of its learning rates passes the stability gate at that $\sigma$. Per cell, select
  the $\sigma$ that maximises the **number of admissible combinations** — a criterion that rewards
  broad stability rather than any one representation — tie-broken by the median %opt across
  admissible combinations. The selected $\sigma$ is then held common across all nine combinations of
  the cell.
- **Per-combination learning-rate ranking** (at the cell's selected $\sigma$): among the admissible
  learning rates, select the one with the greatest **median %opt** over the three selection seeds.
  If **no** learning rate is admissible at the selected $\sigma$, the combination is a genuine
  failure: record it as such and carry the least-diverging learning rate into the final runs,
  flagged, so its failure is reported rather than hidden.

## 6. Final procedure (reported result)

- **Learning rate**: the value selected in Section 5, per combination.
- **Seeds**: $\{0, 1, 2, 3, 4\}$ (five), disjoint from the selection seeds.
- **Budget**: identical to selection (cap 2000, patience 10, SNR recorded) — same convergence stage
  for every combination.
- **Report — two axes, never collapsed into one number**:
  1. **Convergence rate** = fraction of the five seeds with no `[DIVERGED]` marker (learnability /
     optimisation difficulty);
  2. **%opt $\mid$ converged** = median over converged seeds of
     $\%\mathrm{opt} = 100 \times (J_0 - J_{\text{agent}})/(J_0 - J_{\text{oracle}})$, where $J_0$ is
     the no-control cost and $J_{\text{oracle}}$ the delayed-LQR (linearised-delayed) optimum
     (representational quality when optimisation succeeds).
  The worst converged seed is reported alongside the median.

## 7. Provenance

Every run records its full command line, resolved seeds, hyperparameters, and library versions
(shared run-context helper). The selection manifests, the frozen learning-rate table produced by
Section 5, and the final manifests are committed. SLURM logs are written to a `slurm/` subfolder of
each run group. Figures regenerate from saved metrics without recomputation.

## Amendments

(none)
