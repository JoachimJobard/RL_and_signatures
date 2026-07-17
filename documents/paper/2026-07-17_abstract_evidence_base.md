# The AAAI abstract's evidence base — measured, 2026-07-17

*Every number here was read off the repository on 2026-07-17, not inferred and not quoted from an
agent's summary. Companion to `2026-07-16_aaai_rl_paper_audit.md` (scope and theory),
`2026-07-17_aaai_submission_logistics.md` (mechanics) and `2026-07-17_aaai_abstract_candidates.md`
(drafts — **its "5 studies x 5 seeds x 3 representations" figure is corrected by Section 1 below**).*

*The draft `.tex` is not touched by this file.*

---

## 1. What `data/main_unified/` actually contains — 140 runs, not "5 x 5 x 3"

Three representation **families**, each instantiated at several degrees or depths, with a variant
count that **differs per plant**:

| Study | Variants | Runs |
|---|---|---|
| `delayed_oscillator_high_gap_study` | markovian_deg2, raw_deg1, raw_deg2, sig_depth2, sig_depth3 | 25 |
| `delayed_velocity_study_v2` | the same five | 25 |
| `linear_cell_study` | the same five | 25 |
| `mackey_glass_chaotic_study` | the above plus sig_depth4 | 30 |
| `mackey_glass_limit_cycle_study` | the above plus raw_deg3, sig_depth4 | 35 |

Total **140 runs**: five plants, five seeds (0–4), five to seven feature maps. `sig_depth3` is a
distinct variant carrying its own success rate — hence the earlier observation that "the signature"
scored 1/5 somewhere. There is no single object called "the signature" in this table.

## 2. The metric is NOT commensurable across the five plants

`run/study/aggregate_representation_study.py:261` sets `metric_is_suboptimality = use_oracle`:

| Plant class | Metric | Threshold |
|---|---|---|
| Linear (delayed_oscillator, delayed_velocity, linear_cell) | **rho**, normalised suboptimality against the **exact delayed-LQR oracle** | `DEFAULT_RHO_MAX = 0.5` |
| Nonlinear (both Mackey–Glass) | **raw cost J** — no oracle exists | `DEFAULT_J_MAX = 1.0` |

**Consequence, binding on the abstract:** no sentence may aggregate all five into one number.
Markovian scores 0.380 (rho) on one plant and 43.78 (J) on another; these are different quantities.
*"Four of five plants"* is a legitimate count. *"Mean improvement across five plants"* is not.

## 3. The H1 scorecard (value gradient, 5 seeds, ci95 recorded)

| Plant | Metric | Markovian | Best raw-history | Best signature | H1 |
|---|---|---|---|---|---|
| delayed_oscillator | rho | **0.380** (5/5) | 0.413 (5/5) | 0.157 (5/5) | signature only |
| delayed_velocity | rho | 0.572 (0/5) | 0.500 (2/5) | 0.330 (5/5) | holds |
| linear_cell | rho | **0.127** (5/5) | 0.322 (3/5) | 0.284 (5/5) | **FAILS — Markovian wins** |
| mackey_glass_chaotic | J | 43.78 (0/5) | 0.297 (5/5) | 0.088 (5/5) | holds |
| mackey_glass_limit_cycle | J | 30.30 (0/5) | 0.281 (5/5) | 0.049 (5/5) | holds |

**H1 holds on four of five plants and fails on `linear_cell`**, where Markovian features reach
rho = 0.127 at 5/5 success against a best history variant of 0.284. This is a loss, not a tie, and
it is quantified against an *exact* oracle rather than asserted.

**Raw history is the unreliable family.** It is beaten by Markovian features on `delayed_oscillator`
and `linear_cell`; `raw_deg3` on mackey_glass_limit_cycle is **NaN at 0/5 success** (feature
dimension 4959); and `raw_deg1` posts 0/5 success on three separate plants *with very small
confidence intervals* — it converges reliably to a controller that does not achieve the task.

## 4. A reporting defect: `delay: 0.0` in every `summary.yaml`

`aggregate_representation_study.py:154` reads
`delay=float(env_params.get("delay", 0.0) or 0.0)`, which cannot parse the structured
`_target_: numpy.array` node and falls back silently to the default. The plants are genuinely
delayed — `delayed_oscillator_high_gap`'s own config gives `delay: numpy.array([0.2, 0.2])` with
`A1 = [[0, 0], [-6, -6]]` non-zero, i.e. **tau = 0.2**.

**It does not corrupt rho.** `compute_oracle_costs` instantiates the real environment
(`hydra.utils.instantiate(env_params)`, line 172) and builds the oracle from that object
(`delayed_lqr_for_env(env)`, line 179); the misparsed scalar never reaches it. The defect is
cosmetic, but a reader of any `summary.yaml` would wrongly conclude the plants carry no delay.
Worth repairing before the artefacts are cited.

## 5. What the other two learners have measured — nothing about H1

| Learner | H1 measured? | Evidence |
|---|---|---|
| `value_gradient` | **yes** | the 140 runs of Sections 1–3 |
| `signatures` (actor-critic) | **no — and it was structurally impossible** | `2ae3b73` is the only commit adding `make_representation` to `signatures_jax.py`; at its parent that count is 0 and `conf/agent/signatures.yaml` has no `kind:`. The actor-critic could only ever run `signature`; H1 requires markovian. Its only prior data (`oracle_ladder_high_gap`: 9 runs, 1 cell, 3 seeds, signature-only) is an oracle-ladder study — a different question |
| `policy_gradient` | **no** | `cost_reduction_pct` is not measured on any cell. Only stability: 40 episodes, seed 0, four cell/rep pairs, 0 divergence cuts, peaks 0.36–2.96 |

The 150-task campaign (5 cells x 3 representations x 5 seeds x 2 agents) has **never been
launched**.

## 6. Consequences for the abstract

**Admissible (each true today, immune to every campaign outcome):**
1. The representation-agnostic verification theorem. Proven; no run can refute it.
2. Its three feature-map instantiations: Markovian recovers Doya (2000) exactly; a discretised
   window returns the gradient with respect to its most recent node; the signature returns
   `DPhi(x_t) v = S(x_tilde_t) tensor iota(v)`.
3. **The implementation already exhibits the agnosticism**: `value_gradient_jax.py:203` takes
   `grad_path[-1]` by generic `jax.grad`; nothing in it knows it is differentiating a signature.
4. Three learners instantiate the same law by mechanisms sharing nothing — analytic vertical
   derivative, a bootstrapped actor (Doya Equation 20), and REINFORCE with no bootstrap. **Stated
   as a design, with no claim of epistemic hierarchy between them** (owner's decision, 2026-07-17).
5. H1 measured under the value gradient at five seeds, **on four of five plants, failing on
   `linear_cell`**.
6. A falsification control of delay exactly zero (`double_integrator`: `A1: numpy.zeros`,
   `delay: numpy.zeros`) on which H1 must fail by construction.

**Inadmissible:**
1. *"H1 holds for the value gradient and the actor-critic"* — never measured, and impossible before
   `2ae3b73`.
2. Any policy-gradient performance number — none exists, and its own audit returns
   `estimator_is_correct = NO` (it reinforces the diverging action, measured advantage +23.1) and
   `tests_are_adequate = NO` (a sign-inverted estimator passes all 13 tests).
3. Learner-invariance / H4, or that three-learner agreement is "strictly stronger" evidence about
   the representation. H4 was agent-invented and reverted on the owner's instruction (`49db500`);
   the "strictly stronger" argument in `5b410d6` is the same class of agent-authored reasoning and
   has never been endorsed.
4. Any representational advantage for the signature. **H2 is out of scope for this paper**
   (owner, 2026-07-17: *"we dont care about h2 in this paper"*) — Paper 2's subject.
5. The functional Hamilton–Jacobi–Bellman equation as a contribution — it is cited
   (Vinter & Kwong 1981).
6. *"Continuous actor-critic methods are well known to be unstable"* — uncited and contradicted by
   Doya (2000), Figure 5, which measured the actor-critic **slower** (about 70 trials against 15),
   not unstable.
