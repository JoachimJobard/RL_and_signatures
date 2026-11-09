## 1. Read this before writing a single sentence

Two claims in circulation are false, and both would enter a submitted abstract if transcribed.

**The first is the project owner's own sentence.** "The results obtained so far show that the three algorithms seem to work on most of the environment, with the H1 hypothesis true (in the VG and AC cases)." Neither conjunct survives contact with the artefacts.

- **H1 is not measured for the actor-critic.** Not weakly, not ambiguously, not as a null. H1 is the contrast `I[raw_history] < I[markovian]`; it requires two arms at a common cell. The actor-critic has never been run with a `markovian` representation, anywhere, at any seed, at any budget. The count of (cell, seed) pairs at which actor-critic H1 is computable is **0**. The quantity was never computed because one of its two arms was never run.
- **The third algorithm has never been run at all.** `grep -rl "monte_carlo" data/ --include=*.yaml` returns 0 files. Worse: the campaign launcher **rejects it outright** — executed, `AGENT=policy_gradient` yields `Error: AGENT must be 'value_gradient' or 'signatures' (got 'policy_gradient')` (`rl_launch.sh:41-44`). The learner was added today (commit `5b410d6`); the allow-list was never extended. The 150-task design covers **two** learners, not three.

**The second is in the document you are most likely writing from.** `documents/paper/2026-07-17_aaai_abstract_candidates.md:16` states:

> | H1 holds on delay plants | **measured, 5 studies × 5 seeds × 3 representations, value gradient only** | `data/main_unified/` |

That row is **refuted by the artefacts it cites**. Recomputed below: H1 is decided-and-holds on **one** study of five, undecided on four, and the point estimate runs **against** H1 on two. The document is right about the actor-critic (its row 19 correctly says "NOT measured") and wrong about the value gradient. Do not inherit row 16.

---

## 2. The actor-critic's record, precisely

`grep -rl "CTACSignatureJAX" data/ --include=config.yaml` → 20 files → **11 distinct runs**: 9 under `data/main_unified/oracle_ladder_high_gap/` (mirrored under `data/h1h2_report_runs/`, hence the doubling) and 2 `_debug_guard_check` probes of today at 30 episodes.

The 9 carry **no `signature.kind` key at all** — they predate the `make_representation` factory whose `kind` selector defines the three arms. Their axis, read off every config, is `actor_oracle`/`critic_oracle`: three `false/false`, three `false/true`, three `true/false`, at seeds 0–2. **That is an oracle ladder at a fixed representation** — and the oracle axis is precisely what the owner's scope statement excludes ("the AAAI paper is the RL half only: no oracle"). The single axis these runs vary is the axis the paper does not have.

Two corrections to the record, in opposite directions:

- **Against the task brief:** the 9 runs are *not* PNG-only. `find data -ipath '*oracle_ladder*' -name '*.npz'` returns 9 `training_snapshot.npz` under the `h1h2_report_runs` mirror (the `main_unified` copy *is* PNG+YAML only, which is why the omission was inferred). They load and are regenerable. **This changes nothing** — a regenerable measurement of one representation across an oracle ladder is still not an H1 contrast.
- **For the task brief:** one inventory "corrected" the brief by claiming the runs are on cell `oracle_ladder_high_gap`, not `delayed_oscillator_high_gap`. **That correction is wrong; disregard it.** `oracle_ladder_high_gap` is the `experiment_group` field. The env block sets A=[[0,1],[-1,-0.3]], A1=[[0,0],[-6,-6]], delay=[0.2,0.2], Q=eye(2), R=[[0.1]], step_size=0.05, resolution=4 — matching `conf/env/delayed_oscillator_high_gap.yaml` parameter for parameter. The brief was right.

Finally, the 9 runs are **stale against the agent as currently defined**. Every parameter checked differs from `conf/agent/signatures.yaml` at HEAD: n_episodes 800→2000, critic_lr 0.01→1e-3, clip_action 10.0→null, divergence_threshold 50.0→null, window_size 5→40, preheat true→false, burning_steps 5→0. The configuration that produced them no longer exists.

---

## 3. What H1 actually does, for the value gradient

Campaign-matched capacity (`rl_array.slurm:87-89` fixes markovian degree 2, raw_history degree 2, signature depth 2). Mean ± the recorded z-based ci95, with Welch's two-sample t-test computed on the persisted per-seed values (n=5 per arm). Read from `data/main_unified/*/summary.yaml`.

| Study | raw_history (deg 2) | markovian (deg 2) | Verdict | Welch p |
|---|---|---|---|---|
| `mackey_glass_chaotic_study` | 0.2970 ± 0.1241 | 43.7833 ± 12.0134 | **DECIDED — holds** | 0.0021 |
| `delayed_velocity_study_v2` | 0.4996 ± 0.0576 | 0.5723 ± 0.0398 | undecided, point favours | 0.0805 |
| `mackey_glass_limit_cycle_study` | 15.3139 ± 17.9541 | 30.3046 ± 13.4763 | undecided, point favours | 0.2297 |
| `linear_cell_study` | 0.3221 ± 0.1895 | 0.1267 ± 0.0165 | undecided, **point AGAINST** | 0.1133 |
| `delayed_oscillator_high_gap_study` | 0.4127 ± 0.0364 | 0.3801 ± 0.0242 | undecided, **point AGAINST** | 0.1870 |

**H1: decided-and-holds 1 / 5. Undecided 4 / 5. Point estimate against 2 / 5.** Welch agrees with the disjoint-CI rule on every cell. Verdicts are unchanged under the correct Student-t multiplier.

Three qualifications that matter more than the table:

1. **The one hold is a baseline failure, not a margin.** On `mackey_glass_chaotic` the markovian arm records `success_rate: 0.0` — it does not stabilise the plant at all — at cost 43.78 against 0.297. H1 is "decided" there because the Markovian baseline collapses.
2. **The recorded intervals are anti-conservative.** ci95/sem = exactly **1.9600** (a normal quantile; `linear_cell_study`, markovian cell: ci95 0.016463554519638095 / sem 0.00839977271410107). At n=5 the correct multiplier is t(4,0.975)=**2.776**, so every recorded interval is ~42 % too narrow. Widen anything you quote.
3. **The metrics are not on one scale.** `metric_is_suboptimality: true` for the three JAXDDEEnv studies (analytic oracle available), `false` for the two Mackey–Glass studies (raw cost J). Both are lower-is-better; they are not comparable.

### H2, for completeness — because you must know it exists in order not to lean on it

At the same capacity: `delayed_oscillator_high_gap` sig2 0.2940 vs raw2 0.4127 (**holds**, p=0.0020); `delayed_velocity_v2` 0.3317 vs 0.4996 (**holds**, p=0.0040); `mackey_glass_chaotic` 0.0880 vs 0.2970 (holds under z, **undecided under t**, p=0.0283); `linear_cell` 0.2836 vs 0.3221 (undecided, p=0.7254); `mackey_glass_limit_cycle` 0.0681 vs 15.3139 (undecided, p=0.1714).

**H2 is the better-supported hypothesis in this data — and H2 is the one the paper is forbidden from claiming.** Note also that capacity is not monotone: on `linear_cell_study` the signature at depth 3 records 3.8635 ± 1.9199 at `success_rate: 0.2`, decisively *worse* than raw_history at degree 2.

---

## 4. The five studies are not the five cells

The candidates say "verified on five delay plants". Two disjoint sets are being conflated.

- **Campaign cells** (`rl_launch.sh:77`): markovian, linear_dde, hopfield_linear, hopfield_nonlinear, mg_chaotic — 1000 episodes, seeds 0–4, 75 tasks per agent.
- **Studies with data**: delayed_oscillator_high_gap (800 ep, window 5), delayed_velocity_v2 (2000, window 6), linear_cell (1000, window 20), mackey_glass_chaotic (800, window 70), mackey_glass_limit_cycle (2000, window 28).

Only **mg_chaotic** is in both, and at window 70 against the campaign's WIN=71 and 800 episodes against 1000. Five different budgets, five different windows: these are five heterogeneous legacy studies, not a designed suite. `grep -rh "_target_: src.envs" data/main_unified --include=config.yaml | sort | uniq -c` returns exactly two environments — JAXDDEEnv (101) and MackeyGlass1DEnv (69). **Neither Hopfield cell has any RL data**, for either learner: 40 % of the campaign, including the only controlled pair in the design.

An abstract written on 21 July describing "five delay plants" (the legacy studies) may not match the paper's 28 July table (the campaign cells).

---

## 5. The formula in all four candidates must not be printed

Every candidate prints `u* = -½ R⁻¹ g(x)ᵀ DΦ(x)ᵀ ∇Ψ(Φ(x))`. It has two independent defects.

**Sign.** `env_rk_jax.py:88` sets `reward = -(xᵀQx + uᵀRu)` with R positive definite (`conf/env/*.yaml`: `R: object: [[0.1]]`). Maximising the Hamiltonian gives `u = +½ R⁻¹ Bᵀ ∇V` — exactly `value_gradient_jax.py:204`. The printed −½ is correct only under the opposite convention. Pick the code's convention and restate the theorem in it.

**Projection — the more serious.** `DΦᵀ∇Ψ`, read naturally, is the gradient with respect to the **whole path**. The code does not use it: line 203 takes `grad_path[-1]`, the **tip alone**. Measured (probe replicating lines 180–207, window 6, n_state 2):

| Representation | Retained tip norm | Discarded older-node gradient |
|---|---|---|
| markovian | 5.225154e+00 | **0.000000e+00 — identically zero** |
| raw_history | 5.703707e+00 | **1.049227e+01 — exceeds the tip** |
| signature | 5.436909e+00 | 4.069289e+00 |

The discarded part is nonzero for exactly the two representations the paper cares about, and for `raw_history` it is *larger* than the part retained. The code is correct — the tip component is the vertical derivative, which is what HJB stationarity requires — but the formula as printed is not what the code computes. **If a symbol appears in the abstract at all, it must name the vertical/tip derivative explicitly.** (Magnitudes are probe-dependent — random path, random head; the structural fact, zero versus nonzero, is what is robust. Do not quote the magnitudes.)

That same table is the **measured witness** for "Markovian features recover Doya exactly": the markovian map depends on the tip node alone, so the tip gradient *is* the whole gradient and the law degenerates to Doya's with no special-casing.

---

## 6. Claims that survive any campaign outcome

Write the abstract from these. All are proven or are design facts; none depends on an experiment that has not run.

1. The verification theorem and its factorisation into a representation Jacobian and a head gradient. Realised in code, not merely in prose: `value_gradient_jax.py:186` is a generic `jax.grad` call with no branch on representation kind; all three kinds traverse the identical path (feature_dim 5 / 90 / 30, all returning shape (6,2)).
2. Representation-agnosticism: three feature maps ordinarily presented as competitors are instantiations of one statement.
3. Markovian features recover Doya (2000) exactly — now with a measured witness.
4. A nonlinear head requires no new theorem: one autodiff call, no code change. (A statement about structure. No run with a nonlinear head exists.)
5. The functional HJB is **cited, not claimed**. Mandatory.
6. The falsifiable design: cells selected by a measured non-Markovianity gate (`rl_launch.sh:53-72`: markovian 0.000/+0.00 %, linear_dde 0.346/+117.34 %, hopfield_linear 0.300/+54.86 %, hopfield_nonlinear 0.300/+54.86 %, mg_chaotic 0.115/+89.09 %; dadebo_cstr disqualified at 0.009/+1.03 %).

---

## 7. Contingent claims — the exact wording that survives

**H1.**
- Survives: *"The prediction is examined on delay plants selected by a measured non-Markovianity gate, and on a Markovian control cell of delay exactly zero on which, by construction, it must fail."*
- Survives: *"…examined at five seeds and three representations under a value-gradient learner."* ("Examined" commits to the experiment, not the outcome.)
- **Does not survive:** *"H1 holds on delay plants."* Decided on 1 of 5.
- **Does not survive:** *"The construction is verified on five delay plants."* "Verified" will be read as "confirmed"; it is confirmed on one.
- **Does not survive:** *"History representations help on delayed plants."* Contradicted by the value gradient's own record on two of five.

**The control cell.**
- Survives: *"…a Markovian cell of delay exactly zero on which the history representations can afford no advantage, by construction rather than by measurement."* ("By construction" is load-bearing; keep it.)
- **Does not survive:** *"…and fails there, as measured."* No H1 number exists for that cell. **Note a precision the candidates document gets wrong while reaching the right conclusion:** it says the control cell was "never run outside `_debug_`". It *was* run — searching by parameter value (Hydra inlines env params, so the string never appears in a config) finds 15 non-debug value-gradient runs on a zero-A1, zero-delay plant under `data/main_unified/trick_ablation_convergence/`. But all 15 are `kind: markovian` — **one arm only** — their axis is a training-trick ablation, and there is no `summary.yaml`. The H1 contrast on the control cell fails for the same structural reason as the actor-critic's. The conclusion stands; the stated reason does not.

**The three learners.**
- Survives: *"Three learners instantiate the same law by different mechanisms, so that agreement across them **would** isolate the law from the optimiser."* Conditional mood only.
- **Does not survive:** any indicative — "the learners agree", "agreement isolates", "all three succeed".
- If the third learner is named, describe it from the code: **REINFORCE with no baseline, A_t = R_t** (`signatures_jax.py:818`). **Never** from `conf/agent/policy_gradient.yaml` or a run log — `signatures_jax.py:96` *prints* "REINFORCE with a value baseline (advantage A_t = R_t − V(s_t))" into every log, and it is false at HEAD. Commit `8d0f4b0` removed the baseline without updating four sites of prose.

**The signature.**
- **Does not survive, under any phrasing:** any representational-advantage claim. Note the specific trap: the theorem is your headline, and the most natural celebratory sentence — *"one theorem covering three representations, of which the signature is the richest"* — violates the constraint. Keep the framing **unifying, never ranking**.

---

## 8. Two further traps in the inherited record

- **The `delay` field in every `summary.yaml` reads 0.0** — for every cell of every study, *including* the three JAXDDEEnv studies whose env config sets `delay: [0.2, 0.2]`. The field is vestigial. One inventory cited it as corroboration that the control cell has zero delay; that corroboration is worthless. Use `conf/env/double_integrator.yaml` (`A1: numpy.zeros`, `delay: numpy.zeros`) instead.
- **The draft side is unverified.** No `.tex` and no `.bib` exist in this repository; the draft is on Overleaf. Every draft-side defect — the sign error, the prior-art assertions (Vinter & Kwong 1981; Kalsi, Lyons & Perez Arribas 2020; Drobac et al.) — rests on the audit's own reading of Overleaf commit `fe5da43` and was **not** re-checked. An abstract claiming novelty against unchecked prior art is the highest single rejection risk here.
- **The augmentation residual (1.14 against 6.2e-5)** exists as prose in one Markdown file and nowhere else; no artefact reproduces it. Its correct use is negative and needs no number: **promise no sliding-window or Chen property**; prove the theorem from the tip derivative alone.

---

## 9. Arithmetic the abstract is a commitment against

The abstract is due **21 July**; the paper **28 July**. As of HEAD `8d0f4b0`:

- The campaign is **not launched** (`find data -maxdepth 3 -type d -name 'h1h2_rl_*'` → nothing; `data/h1h2_report_runs/rl/` is empty).
- It is **75 tasks per agent** at **1000 episodes** (`AGENT=signatures` probe output).
- The launcher **cannot dispatch the policy gradient** at all — an unfixed plumbing defect, not a scheduling matter.
- Two of five cells (both Hopfield) have **no data for either learner**.

No actor-critic and no policy-gradient number can exist by 21 July. Every sentence must therefore be written to survive the case where the campaign returns nothing, and — equally — the case where it returns a *negative* result. Given that H1's point estimate already runs against the hypothesis on two of five legacy studies, a negative campaign outcome is not a remote contingency.

**The safe abstract states the contribution first (theorem, factorisation, unification, Doya recovered exactly), cites the HJB, describes the design as falsifiable with a gate and a zero-delay control, and introduces the learners as a design whose agreement *would* isolate the law from the optimiser. It attaches no H1 verdict, no number, and no formula.**