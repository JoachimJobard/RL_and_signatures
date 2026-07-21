# Pre-registration — H1 (does history help?) across three learners

**Written 2026-07-17, before the run.** This document fixes the hypothesis, the design, the
metric, the hyperparameters, and the falsification conditions *ahead of* the campaign, so that the
result cannot be steered by the data (the anti-HARKing commitment of `documents/paper/aaai_paper_plan.md`).
The outcome is to be reported whatever it is. Commit this file before submitting any job.

**Status: DRAFT for the project owner's confirmation.** Three decisions are the owner's, flagged
inline as **[OWNER]**; the hypothesis and its wording are the owner's and are transcribed, not
supplied. Nothing below is to be treated as decided until the owner confirms the flagged points.

**Scope.** This is the RL half of the AAAI paper: the theorem and **H1 only**. H2 (the signature
versus a degree-two history polynomial) is **out of scope** — not refuted-and-withheld, simply not
this paper's subject. No oracle half, no off-sheet axis, no conditioning statistics. This document
does not amend the protected oracle/H1-H2 documents (`aaai_paper_plan.md`,
`2026-07-02_preregistration_offsheet_h2_replication.md`, the 2026-06-30 benchmark report).

---

## 1. Hypothesis (H1)

> **H1.** On a genuinely non-Markovian (delayed) plant, a **history** representation of the state
> yields lower closed-loop cost than the **Markovian** representation of the current state; on a
> Markovian plant it does not.

Operationally, with $I(a,c,r,s)$ the closed-loop cost of learner $a$ on cell $c$, representation
$r \in \{\text{markovian}, \text{raw\_history}, \text{signature}\}$, seed $s$, and $\tilde I(a,c,r)$
the **median over the five seeds** with 95% bootstrap confidence interval $\mathrm{CI}(a,c,r)$:

$$\Delta_1(a,c) = \tilde I(a,c,\text{raw\_history}) - \tilde I(a,c,\text{markovian}).$$

H1 **holds** on $(a,c)$ when $\Delta_1(a,c) < 0$ with **disjoint** 95% confidence intervals; it is
**undecided** when the intervals overlap; it **fails** when $\Delta_1(a,c) > 0$ with disjoint
intervals. The verdict is reported **per (learner, cell)**.

The signature arm is measured and reported, but this paper makes **no claim that the signature is a
richer representation** (that is H2, refuted for the oracle half and out of scope here). The three
learners are presented as **three mechanisms instantiating one control law, with no claim of an
epistemic hierarchy between them** — a design frame, not a hypothesis with a strength claim.

---

## 2. The hyperparameter protocol (the load-bearing section)

H1 compares representations. The single largest threat to it is that the representation contrast
could be confounded by hyperparameter choices — if `raw_history` and `markovian` were run with
different learning rates, exploration, or budgets, a difference in cost could not be attributed to
the representation. Two commitments remove that threat, and one limitation is declared honestly.

### 2.1 Within a learner, only the representation changes — so H1 is immune to tuning

**Verified from the launcher (`rl_array.slurm:86-91`): between the three representations the array
changes exactly `agent.signature.kind` and the capacity flag (`degree=2` for the polynomial arms,
`depth=2` for the signature). Every other hyperparameter is identical.** Concretely, `markovian`
and `raw_history` are run with the **same** learning rate, exploration noise, window, divergence
rule, discount, target-network setting, and — under the shared-seed policy — the **same** derived
per-role seeds, hence the same initial path and the same sampler trajectory. Therefore a decided
$\Delta_1(a,c)$ is attributable to the representation and to nothing else. **This is what makes H1
clean regardless of whether the fixed hyperparameter values are individually optimal.**

The fixed values, shared across all three representations and (where applicable) matched across the
three learners, resolved under the array's exact invocation:

| parameter | value_gradient | signatures (AC) | policy_gradient (PG) | shared? |
|---|---|---|---|---|
| `n_episodes` (array override) | 1000 | 1000 | 1000 | same |
| `critic_lr` | 1e-3 | 1e-3 | 1e-3 | same |
| `scale` | 1.0 | 1.0 | 1.0 | same |
| `clip_action` | null | null | null | same (no clipping) |
| `clip_gradient` | null | null | null | same (no clipping) |
| `divergence_threshold` | null | null | null | same (**no trimming**) |
| `tau_polyak` (array override) | 1.0 | 1.0 | 1.0 | same (**no target network**) |
| `noise.sigma` | 0.1 | 0.1 | 0.1 | same |
| `noise.schedule` | constant | constant | constant | same |
| `noise.length_scale` | 0.002 | 0.002 | 0.002 | same |
| `discount.discounted` | false | false | false | same (undiscounted) |
| depth / degree | 2 | 2 | 2 | same (H2 axis frozen) |
| precision | float64 | float64 | float64 | same (weights included, commit `2f4e507`) |

### 2.2 Across learners, some hyperparameters necessarily differ — and this is declared, not hidden

The three learners are different algorithms, so a subset of hyperparameters has no shared meaning:

| parameter | VG | AC | PG | why it differs |
|---|---|---|---|---|
| `actor_lr` | — | 1e-3 | 1e-3 | the value gradient has no actor |
| `algorithm.semi_gradient` | — | true | true | the value-gradient critic update is not a semi-gradient TD step |
| `algorithm.actor_target` | — | td | monte_carlo | the defining difference between AC and PG |
| `algorithm.actor_update_frequency` | — | **10** | **1** | **[OWNER] — see below** |
| `algorithm.lstd` | false | — | — | value-gradient-specific critic solver (off) |

**[OWNER] decision 1 — the actor-cadence asymmetry.** The actor-critic updates its actor every 10
steps; the policy gradient updates once per episode. Measured consequence: over an episode the AC
takes roughly **16 times more optimiser steps** on its actor than the PG. This does **not** confound
H1 (it is constant across the three representations within each learner) but it does make the
**AC-versus-PG** comparison unequal in optimiser budget. Options: (a) declare it and report the
cross-learner comparison as "fair but not perfect" (the owner's stated frame), stating the cadence
difference in the paper; (b) match the per-episode optimiser-step count across AC and PG and re-fix
this document. The draft assumes **(a)** unless the owner chooses (b).

### 2.3 The declared limitation — a single fixed hyperparameter set, not per-representation tuning

A **single** hyperparameter set is used for all representations and all cells; nothing is tuned
per-representation or per-cell. This is the **conservative, anti-HARKing** choice: per-representation
tuning would introduce researcher degrees of freedom that could be steered toward "history helps".
The cost of the conservative choice, stated plainly: where the fixed set disadvantages a
representation on a particular cell — for instance a learning rate ill-suited to `raw_history`'s
~3320-dimensional feature vector on `platoon`, or an exploration scale mismatched to a cell's
timescale — a decided $\Delta_1$ reflects **that representation under this fixed shared budget**,
not the representation's best attainable cost. H1 as tested is therefore *"does history help under a
fixed, shared, untuned budget"*, which is a legitimate and confound-free question; it is not *"can
history be tuned to help"*. This limitation is pre-registered, not discovered after the fact.

**[OWNER] decision 2 — provenance of the fixed values.** The shared values above are inherited
defaults and this-session matches (`critic_lr`, `noise.length_scale`, `tau_polyak`, `divergence_threshold`
were set or matched on 2026-07-16/17), **not** the product of a tuning sweep on the campaign cells.
If the owner wants any value justified by a pre-campaign sweep on **held-out** cells or seeds
(never the test cells), that sweep is specified and run *before* this document is committed;
otherwise the values stand as the pre-registered fixed set with the limitation of 2.3.

---

## 3. Design

**Learners (3).** `value_gradient` (analytic control from the critic's vertical derivative),
`signatures` (continuous actor-critic, Doya 2000 Equation 20), `policy_gradient` (REINFORCE, no
baseline). All three consume the same `make_representation` factory and the same three
representations.

**Cells (5).** Selected on methodological grounds — spanning the measured non-Markovianity gate
(history-kernel ratio, `run/study/hopfield_delay_impact.py`) and the state-dimension axis, **not**
by data availability (commit `ea7add7`):

| cell | gate | dim | in-delay | role |
|---|---|---|---|---|
| `markovian` (double_integrator) | 0.000 | 2 | — | **falsification control**: $A_1=0$, $\tau=0$ exactly, so H1 **must fail** by construction. Exact CARE oracle. |
| `mg_chaotic` | 0.115 | 1 | nonlinear | nonlinear scalar |
| `hopfield_nonlinear` | 0.300 | 2 | nonlinear | nonlinear multichannel |
| `linear_dde` | 0.346 | 1 | linear | linear scalar, fully delayed ($A=0$) |
| `platoon` | 0.398 | 10 | linear | high-dimensional; stresses `raw_history` conditioning |

**Representations (3).** `markovian`, `raw_history` (degree 2), `signature` (depth 2).

**Seeds (5).** Master seeds $0,1,2,3,4$; shared-seed policy, so each (learner, cell, representation)
at a given seed derives the same per-role seeds.

**Grid.** $3 \text{ learners} \times 5 \text{ cells} \times 3 \text{ representations} \times 5
\text{ seeds} = \mathbf{225}$ tasks, submitted as three per-learner arrays (75 each).

---

## 4. Metric and analysis

- **Per task:** closed-loop cost $I$ at the pre-registered evaluation initial condition and horizon,
  written to a per-task `eval.pkl` (no shared-file race). Where an exact oracle exists (the three
  linear cells, via the delayed-LQR oracle), the normalised suboptimality $\rho = (I - I^\star)/|I^\star|$
  is the reported quantity; on the nonlinear cells the raw cost $I$ is reported. **The two are not
  commensurable across cells**, so H1 is decided **per cell**, never by pooling across cells.
- **Aggregate per (learner, cell, representation):** the **median** over the five seeds and its
  **95% bootstrap confidence interval on the median** (the cost is heavy-tailed; the median is the
  pre-registered central statistic). **[OWNER] decision 3 — the CI multiplier.** The existing
  `summary.yaml` files record a $1.96$ (normal) interval; at $n=5$ the correct Student multiplier is
  $t(4,0.975)=2.776$, so those intervals are ~42% too narrow. This campaign reports the bootstrap CI
  on the median; if a parametric interval is reported anywhere it uses $t(4,0.975)$, never $1.96$.
- **Divergent / non-finite seeds** are reported as a **separate count** per (learner, cell,
  representation), never dropped and never averaged into a NaN. With trajectory-trimming off and the
  finiteness guards on (commit `8d0f4b0`), a divergence surfaces loudly and is counted; on
  `platoon`/`raw_history` a divergence is itself an H1 finding (history representation broken by
  conditioning), not a run to be discarded.
- **Verdict per (learner, cell):** the §1 rule (disjoint CIs). No success-rate threshold decides any
  verdict; a success rate, if reported, is descriptive only.
- **Reproducibility:** every figure regenerable from the saved artefacts without re-training; any
  number not readable off a concrete run is reported as **not measured**.

---

## 5. Falsification conditions

- **H1 is falsified on a (learner, cell)** when $\Delta_1(a,c) > 0$ with disjoint 95% CIs on a
  non-Markovian cell (history representation strictly worse than Markovian). This is a real
  possible outcome and is to be reported: it already occurs for the value gradient on `linear_cell`
  in the existing oracle-half evidence (a weakly-delayed plant where the Markovian arm wins).
- **The falsification control:** on `markovian` (double_integrator) H1 **must fail** ($A_1=0$,
  $\tau=0$). A *decided H1 win* there indicates a **harness defect**, not a finding, and triggers
  §6 (repair and re-run), not a report.

---

## 6. What is NOT changed after seeing results

The hypothesis, the learners, the cells, the representations, the seeds, the fixed hyperparameter
set (§2), the metric, the per-(learner, cell) verdict rule, and the falsification conditions are
fixed by this document. Only the reported numbers are filled in afterwards. If the pipeline reveals
a defect — in particular a decided H1 win on the `markovian` control — the defect is documented, the
fix is committed, and the campaign is **re-run**; a verdict is not read off a known-defective run.
Such a re-run is a documented correction, not a post-hoc adjustment of the analysis.

---

## 7. Scope limit carried from the oracle half

This campaign may report **measured RL performance** ("the history critic attains lower closed-loop
cost than the Markovian critic on cells X, Y, under all three learners"). It may **not** assert that
the signature is a **richer representation** of history: that claim (H2) is refuted by the oracle
half's pre-registered off-sheet control and is the subject of the second paper.
