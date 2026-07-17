# Maths-versus-plumbing audit — findings and launch decision

*Written 2026-07-17 by the refutation workflow `wf_23fe6377-3da`: 35 canonical defects merged from
94 raw auditor items, each attacked by three independent skeptics (does it exist / is it armed in
the array's exact invocation / is it agent-asymmetric), 107 agents, zero errors. The body below is
that workflow's own synthesis, reproduced verbatim.*

> **This file replaces the register committed as `0959f4b`, which was wrong in three ways. They are
> recorded here rather than quietly dropped, because two of them were argued to the project owner
> as findings.**
>
> 1. **"Nine blocking verdicts" was a counting error.** The workflow journal accumulates rows across
>    crash/resume cycles, so a check that ran twice appears twice. Deduplicated by cache key the
>    count is **4** (last run) to **6** (any run), across **three** distinct defects, of which the
>    synthesis below confirms exactly **one** meets the blocking criterion.
> 2. **The "method note" was fabricated.** `0959f4b` claimed the blocking count "rose monotonically
>    with coverage — 5 at 13%, 7 at 40%, 9 at 66%" and concluded that an audit scoped to the
>    critical items "would have reported a launch-safe campaign". That climb was duplicate rows
>    accumulating, not defects being found. Deduplicated, the count is ~5 at 13% and 4–6 at 98% —
>    **flat**. The evidence does not support the claim, and the claim was used to argue against a
>    proposal the owner had already rejected on sounder grounds (an "inert" classification cannot be
>    verified by trusting the classification). Its prediction that "the expected count at full
>    coverage exceeds nine" is also refuted: it plateaued.
> 3. **The `mg_chaotic` evaluation was overstated**, and the synthesis below corrects it in place:
>    the eval series at HEAD is `['nan', −5.70865, −5.76243]`, not `[nan, nan, nan]`, and
>    `_best_critic_params is None` measures **False** — the checkpoint restore *does* fire. The
>    catastrophic version did not reproduce.
>
> `0959f4b` also framed the target network as a trilemma in which every direction cost something.
> The synthesis found a fourth option it had missed: scope the change to the campaign with an array
> flag, which touches no source file and no config default.

---

## Scope and provenance

This register concerns the plumbing of the 150-task actor-critic campaign — the clamps, defaults, discretisations and configuration paths that sit between the mathematics and the reported number. The decision it serves is whether to launch now.

**The audit brief is stale.** It names HEAD `0e457c2`; the repository is at `0959f4b`, seven commits later. Three of the brief's stated ground truths are false at HEAD, including its nominated "prime suspect". All findings below are against `0959f4b`. The working tree was not modified: `git status --porcelain --untracked-files=no` is empty, no job was submitted, and all probe artefacts were removed.

**A prior register already exists.** Commit `0959f4b` ("Register the campaign's launch blockers…") records the owner's own conclusion, *"THE CAMPAIGN IS NOT TO BE LAUNCHED"*, from 70 of 105 lens-checks. This report agrees on the diagnosis of the one blocking defect and disagrees on the remedy: a fourth direction exists that the register did not consider, and it is one line.

## Launch decision: FIX-FIRST — one line, then launch

Exactly one defect satisfies the stated blocking criterion (reproduced **and** armed **and** agent-asymmetric): the **critic target network**. It closes with a single flag in the array. This is not a reason to delay a campaign.

---

## Blocking

### The value gradient bootstraps from a lagged target network; the actor-critic does not

At fixed representation the two learners regress different critic objectives. Quoted at HEAD:

| | line | `V_next` |
|---|---|---|
| value gradient | `value_gradient_jax.py:262` | `critic.apply(jax.lax.stop_gradient(target_params), sig_next)` |
| actor-critic | `signatures_jax.py:619` | `critic.apply(jax.lax.stop_gradient(critic_params), sig_next)` |

`grep -c target_params src/agents/signatures_jax.py` returns **0**: the actor-critic has no target network whatsoever. It therefore implements Doya's semi-gradient exactly — a stop-gradient on the *same* function — and its lag term is identically zero by construction. The value gradient bootstraps from a *second, lagged* function, so its coded residual is

> r + (V_target(s') − V_online(s))/dt,

a finite difference of two **different** functions, which is not the time derivative of any single function and is therefore not Doya's residual.

**Armed.** `agent.training.tau_polyak` resolves to 0.01 in all 15 value-gradient configurations and is ABSENT from all 15 actor-critic configurations (measured by Hydra composition of the array's verbatim overrides).

**Magnitude (measured).** hopfield_nonlinear, seed 42, 20 episodes, campaign x0 = [0.5, 0.0], dt = 0.1, `tau_polyak` = 0.01. The lag term is (V_target(s') − V_online(s'))/dt; the derivative it perturbs is (V_online(s') − V_online(s))/dt.

| Representation | mean \|dV/dt\| | mean \|lag\| | ratio |
|---|---|---|---|
| signature | 3.2327e-02 | 1.1375e-01 | **3.5187** |
| raw_history | 1.1212e-02 | 3.4733e-02 | 3.0977 |
| markovian | — | — | not measured (faithfully) |

These reproduce an independent skeptic's figures to four significant figures. The contamination is **representation-dependent and largest on the signature arm** — the representation under test — so it does not cancel in the actor-critic-versus-value-gradient contrast.

**Aggravating, separately measured.** `value_gradient_jax.py:108-110` splits `key_critic, key_target, self.key` and initialises the target from an **independent** key. Measured max|critic − target| at initialisation = **3.397e-02**, against `network.std_init` = 0.01: the target does not begin as a copy of the critic, so early training subtracts an unrelated random function rather than a stale estimate.

**Claim strength.** The existence and the asymmetry are *proven by quoted source*. The magnitudes are *measured* on one cell, one seed, 20 episodes. The effect on the campaign's final reported cost is **not measured** — that would require the counterfactual run.

### The fix: one line

```
++agent.training.tau_polyak=1.0 \
```
inserted into the python invocation at `rl_array.slurm:125-133`.

**Mechanism, verified by execution.** `optax.incremental_update(new_critic, target, step_size=1.0)` returns `new_critic` exactly (measured on a deliberately far-apart pair). Hence `target_params == critic_params` at every step after the first, and `value_gradient_jax.py:262` collapses onto `signatures_jax.py:619` — Doya's semi-gradient, no target network, on both arms.

**Measured end to end** (same cell/seed/budget as above):

| Representation | `tau_polyak` | mean \|lag\| | ratio |
|---|---|---|---|
| signature | 0.01 | 1.1375e-01 | 3.5187 |
| signature | **1.0** | 1.0697e-04 | **0.0032** |
| raw_history | 0.01 | 3.4733e-02 | 3.0977 |
| raw_history | **1.0** | 8.7068e-05 | **0.0063** |

**The override form is load-bearing and must not be simplified.** The array issues one invocation for both agents:

| form | value_gradient | signatures |
|---|---|---|
| `agent.training.tau_polyak=1.0` | OK | **REJECTED** (ConfigCompositionException) |
| `+agent.training.tau_polyak=1.0` | **REJECTED** | OK |
| `++agent.training.tau_polyak=1.0` | OK (1.0) | OK (1.0) |

A bare override would kill all 75 actor-critic tasks at composition. Only `++` composes for both. It is inert for the actor-critic (`grep -c tau_polyak src/agents/signatures_jax.py` returns 0). Both agents were run end to end through `main_unified.py` with the flag and reached `EXPERIMENT COMPLETE`.

**Residual, measured and not eliminated.** At `tau_polyak` = 1.0, max|lag| remains 3.2090e-01 while the mean is 1.0697e-04; 3.2090e-01 / 3000 = 1.07e-04 equals the mean exactly, so the lag is non-zero on precisely **one step per run** — the first, before the first `incremental_update` overwrites the independently-initialised target. Closing that last step requires initialising `target_params` from `key_critic`, which is a code change and is bit-changing. Not recommended.

### Would the fix invalidate the existing five-seed value-gradient results? **No.**

This is where this report departs from the existing register, which framed the choice as three directions each carrying a real cost: fix the value gradient's code (invalidates the results), give the actor-critic a target network (preserves them but propagates a non-Doya device to both arms), or document it (leaves the comparison confounded).

There is a **fourth**: scope the change to the *campaign* via an array flag. A flag alters no source file and no config default — `conf/agent/value_gradient.yaml:18` remains `tau_polyak: 0.01` — so the existing five-seed results stay bit-reproducible by re-running at the unchanged default, and the float32 cast in `_fill_buffer_initial` is untouched.

**The genuine cost, stated plainly.** The new campaign's value-gradient runs would not be bit-comparable with the *old* five-seed history, which used 0.01. The array's own comments (`rl_array.slurm:54-66`) show that cross-campaign comparability was deliberately purchased when the per-cell WIN values were chosen. This is a real trade and it belongs to the project owner. Weighing against it: the campaign is a fresh 5-cell × 3-representation × 5-seed design whose two Hopfield cells have no prior value-gradient data at all; and — the uncomfortable fact the register states and this report confirms — **Doya (2000) has no target network**, so `tau_polyak = 1.0` makes the value gradient Doya-*faithful* rather than merely matched. The deviation does not move the fixed point; it changes the estimator.

**If the owner declines the flag**, the fallback is LAUNCH-WITH-CAVEAT: the within-agent representation contrast (H1/H2) remains sound, since no REPARG variant touches the bootstrap, but the actor-critic-versus-value-gradient contrast must not be reported as a controlled comparison.

---

## Changes the answer (armed, not blocking)

### Actor-critic non-finite evaluation on mg_chaotic — partially reproduced, register overstated

Measured at HEAD through the real `train()` path (MG_1D_chaotic / raw_history / seed 0, 41 episodes, `Eval x0: [0.8]` confirmed in the log):

| agent | eval 0 | eval 1 | eval 2 | best params `None`? |
|---|---|---|---|---|
| signatures | peak ‖x‖ = **nan**, returned **nan** | −5.70865 | −5.76243 | **False** |
| value_gradient | peak 56.3444, −25297.5 | peak 4.05268, −260.659 | peak 1.06121, −151.842 | False |

**Reproduced**: the actor-critic's evaluation does go non-finite where the value gradient's, on the identical cell/representation/seed, does not. **Corrected**: commit `0959f4b` records the series as `[nan, nan, nan]` and states the checkpoint selection "NEVER FIRES". At HEAD the series is `['nan', −5.70865, −5.76243]` and `_best_critic_params is None` measures **False** — the restore *does* fire. The catastrophic version did not reproduce.

What is confirmed: `nan > -inf` is False (measured), so a NaN evaluation is **silently swallowed** — it neither updates nor is penalised against the incumbent, and enters `eval_reward` as `nan`.

**Consequence bounded.** The campaign's reported metric is `eval/total_cost_agent`, read by `aggregate_representation_study.py:152` from `collect_evaluation_data` (`main_unified.py:167`), **not** from `_evaluate_noiseless`, which reaches the reported number only by selecting which parameters the evaluator scores. So the NaN corrupts a plotted diagnostic, and — in the corner where *every* evaluation is non-finite — would silently ship the final diverged parameters.

**Not measured at the campaign's 1000 episodes.** 41 episodes is not evidence about 1000; see the horizon caveat below.

**Minimal fix, bit-safe**: return `-inf` rather than `nan` on a non-finite rollout. The selection outcome is identical by construction (both lose every comparison), hence bit-inert for both agents; only the recorded series changes.

*An audit correction worth recording:* the first probe run here was **not faithful** — it called `build_agent` then `agent.train()` directly, bypassing `train.py:206` which sets `agent.x0`, so the evaluation ran from **zeros** (peak ‖x‖ = 0) and returned finite. That result was discarded, not reported. The finite/NaN distinction turns entirely on the initial condition being wired.

### Divergence-threshold truncation — self-announcing, but unequal gradient budgets

The cut fires on mg_chaotic for both agents (skeptic-measured, not re-measured here: actor-critic at ‖x‖ = 108.15, t = 42.25 of 60; value gradient at 100.22, t = 45.50). The "silent" half of the canonical claim is **refuted**: commit `0fcf731` made both agents report the activation with the raw pre-cut value. The cut is disarmed in evaluation (`time_only=True` at `value_gradient_jax.py:708` and `signatures_jax.py:1164`), and `src/training/evaluate.py` has no cut, so `eval/total_cost_agent` is a full-horizon integral for both arms.

The residue is real: a cut episode withholds its remaining temporal-difference updates, so an arm cut more often trains on fewer transitions. That is an asymmetry in **effect**, not configuration (100.0 is passed explicitly to both). Register; do not change the threshold, which would alter the value gradient wherever it fires.

### The horizon dependence of every "not armed" verdict

Quoted from commit `0959f4b`, not re-measured here: two probes of action clipping at different budgets **disagree** on the same (cell, representation) pair — 20 episodes reports **0/3020 = 0.000%** binding on double_integrator/signature; 200 episodes reports **4535/25218 = 17.98%** with a pre-clip |u| of 151. The actor's output grows during training, so a short probe reports absence of evidence that reads as evidence of absence.

**The campaign runs N_EPISODES = 1000** (`rl_launch.sh:77`). A large fraction of the verdicts in this register rest on 2-, 3-, 20- or 41-episode probes and conclude "measured not to fire". Those conclusions are unsound for any quantity that grows with training — which includes the divergence cut, the actor's control magnitude, and the non-finite evaluation. This report's own mg_chaotic measurement (41 episodes) carries the identical caveat. No 1000-episode probe was run: **not measured**.

---

## Wrong object (armed, agent-symmetric — corrupts no contrast)

- **Exploration is white noise, not smooth.** Both agents resolve `smooth: true`, `length_scale: 0.002`. At that length scale the squared-exponential kernel on the control grid is **exactly** the identity in float64 for all five cells — the adjacent-step correlation exp(−dt²/2ℓ²) = exp(−1250) at dt = 0.1 underflows to exactly zero — so the Cholesky factor is √(1+1e-6)·I and the delivered path is white. This is a fidelity gap against Doya (2000), whose exploration is low-pass-filtered Ornstein–Uhlenbeck (τ_n = 1.0); the stated purpose of smooth exploration on a delay plant, exciting the history window coherently, measurably does not occur. Identical on both arms and all representations, so it biases no contrast. **Do not "fix" it**: restoring correlation would change the value gradient bit-for-bit *and* break the actor-critic, whose score function (`signatures_jax.py:646`, `noise/(sigma**2 + 1e-8)`) is the correct score only for per-step independent noise. The naive repair would manufacture an asymmetry where none exists.

- **The initial condition is sampled as a point, never as a path.** `_sample_initial_state` draws a point and `env_rk_jax.py:77` tiles it; `history_function` is unreachable from every agent call site. Never executed here (`fix_initial_state` resolves true), so no number is corrupted. The durable content is a **scope limitation**: the delay testbed has never been exercised from a non-constant initial path, and the H1/H2 contrast is measured on constant-φ initial conditions only. The paper must not imply otherwise.

- **The actor-critic has no NaN guard in `_is_episode_done`.** `value_gradient_jax.py:630-641` carries an explicit `jnp.any(jnp.isnan(x))` termination; `signatures_jax.py:475-506` has none, and `nan > 100.0` is False. The two agents apply different training-termination rules on a non-finite state. The brief's premise that training terminates on a non-finite state is true for the value gradient and **false** for the actor-critic. The fix touches `signatures_jax.py` only (no value-gradient bit risk) but changes actor-critic behaviour, so it must land *before* the campaign, not between halves of it. Not measured to fire in training.

---

## Necessary discretisations (legitimate; arithmetic checked)

- **`runge_kutta4` is globally second-order on every delayed cell.** The k2/k3 stages interpolate the delayed state linearly (`solver_buffer_jax.py:62`, local error O(h²)), capping the global order at min(p, q+1) = 2 regardless of the tableau. Measured order **exactly 2.000** at every one of six halvings; the isolating control (same tableau, `has_delay=False`) returns **4.000**, confirming the interpolant as the cap and the tableau as correct. Effect on the reported cost: 1.5e-03 relative on linear_dde at the campaign step — common-mode across both agents and all representations, hence cancelling in every contrast, and each agent is trained *and* evaluated on the same discrete plant, so the reported cost is self-consistent. **The defect is documentation**: the method's name and the env comments (`resolution: 4  # RK4 sub-steps`) assert an accuracy the scheme does not deliver. Fixing the interpolant would change both arms bit-for-bit and invalidate the five-seed history. Rename or comment.

- **The temporal-difference quotient is first-order, with a right-endpoint reward.** Consistent and first-order (measured: endpoint term falls with ratio 1.983, 1.992, 1.996 under dt-halving, i.e. exactly O(dt)). Character-identical at `value_gradient_jax.py:263` and `signatures_jax.py:620`, dt from the same env object, so it cannot separate the arms. One limb of the canonical claim is refuted: the "left-endpoint discount" never executes, `discount.discounted` resolving False in all 30 configurations. Document; do not change.

- **The window over-spans [−τ, 0] by three taps** on linear_dde (1.60 against τ = 1.0) and mg_chaotic (17.75 against 17.0), and by **zero** on both Hopfield cells (0.50 = τ exactly). Identical for both agents. A window strictly containing x_t is a *superset* of the delay state, from which the true value functional remains exactly representable — over-parametrisation, not a wrong object. Inherited by transcription from the dead "+3" formula. Shortening it would invalidate the five-seed history; record why two cells carry three redundant taps and two carry none.

- **Front-padding of φ** fires on three of five cells (pads 10 / 3 / 3; zero on both Hopfield cells) and is numerically exact: `env_rk_jax.py:77` tiles x0 whenever `history_function` is None, so φ is constant on all five cells (measured max|φ − φ[0]| = 0.000e+00) and the constant extension *is* φ. The canonical "3 versus 1" asymmetry is an artefact of the default window path the campaign does not take. Same scope condition as the point-sampler.

- **The signature Gram is severely ill-conditioned** (cond 3.749e23, rank 20 of 30 on the markovian cell's signature arm; raw_history worse at 72 of 275) — **this is the measurand, not a defect.** Density at infinite depth is not realisability at depth 2 on finite on-policy data, which is what the campaign exists to measure. The canonical mechanism is refuted: no Gram solve executes, the critic being fitted by Adam. Regularising it away would remove the phenomenon under study.

---

## Inert but undocumented

| Item | Status | VG bit risk of fixing |
|---|---|---|
| float32 narrowing of φ (`value_gradient_jax.py:164/167/172`) | The one surviving numerical agent asymmetry: max\|VG − AC\| = **1.192093e-08** on linear_dde and mg_chaotic (x0 = [0.8] not binary32-representable), **exactly 0** on the other three. Propagates to the control at 7.19e-09–3.3e-08 relative. Retained by explicit decision; documented at `signatures_jax.py:281-297`. | **YES** on two of five cells |
| Cholesky jitter `1e-6 * I` | Inert by **degeneracy**, not design: since K = I exactly, the jitter is a uniform √(1+1e-6) = **1 + 5.000e-07** rescale of σ. The comment "for Cholesky stability" is unsupported (factorisation succeeds without it) and the SVD fallback is dead. Foot-gun: at ℓ = 0.2 (the `configs.py:66` default) the jitter exceeds min eig(K) = 2.780e-08 by ~36× and becomes a live unlogged regulariser. | **YES** — must stay |
| `dt <= 1e-9 → dt = 1e-4` | Unreachable: `step_size` ∈ {0.1, 0.2, 0.25}, margin 1e8; assigned once, never mutated. Presented as a division guard, it would substitute a magic constant rather than fail — it should raise. | NO (branch never executes) |
| Entire LSTD block | Dead: `lstd` resolves False in 15/15; a line trace recorded 300 semi-gradient entries and **zero** LSTD entries. Contains two latent defects (ridge units inconsistent across branches; discount applied unconditionally). Note: lowering `discount.tau` 100 → 1 (`15fe75c`) is inert now but *multiplies* the dormant LSTD discrepancy if ever armed. | NO |
| Adaptive noise clip + 1e-6 ε | Dead (`schedule: constant`). The ε is genuinely inert even if armed (1.000e-07 relative); the **clip is not** — measured saturating at the floor on 100.00% of evaluations when forced live. They should not be documented as one. | NO |
| `compute_reward` (`env_rk_jax.py:91-100`) | No call site anywhere. Its kernel predicate merely flips the reward's sign and its scan never advances the state — wrong twice over, if ever wired up. | NO — free to delete |
| Resolved window not persisted | Agents mutate a detached config copy (`train.py:100-103`), so `config.yaml` records the nominal window. Under `force_signature_window=true`, persisted = used, so the campaign is unaffected; historical default-path runs recorded overridden windows. Fixing persistence is bit-safe; making cfg authoritative is **not**. | NO (persistence only) |

---

## False alarms (refuted)

- **`clip_action` agent asymmetry** — the brief's nominated prime suspect. **Already closed at HEAD** by `905e7d1`: `conf/agent/signatures.yaml:32` reads `clip_action: null`, matched to `value_gradient.yaml:14`. All 30 configurations resolve `clip_action = None` for both agents, so `do_clip` is False and no clip is traced into either compiled law. The cached 6.0% figure is moot. *The sting*: the 20-episode probe that suggested "inert" reported 0.000% while a 200-episode probe on the same pair reported 17.98%. The defect was real; the reasoning that nearly dismissed it was not.
- **"+3 versus +1" window divergence** — refuted as armed. Both formulas sit behind `if not force_signature_window:`; a call-counting spy recorded **zero** calls across ten campaign-shaped runs. Windows identical for both agents on all five cells (10/8/5/5/71). The single flag is the only thing holding this closed — **that line must never be removed**.
- **`force_signature_window` inverted semantics** — the *code* is correct; the *comment* (at `configs.py:88` and five YAMLs) is inverted. No arithmetic reads a comment. Repairing code-to-comment would discard every per-cell WIN and re-arm the +3/+1 asymmetry across all 150 tasks.
- **LSTD ridge / LSTD discount** — unreachable (`lstd` False; zero traced entries). The supporting conditioning figures cannot have come from a campaign configuration.
- **`lstsq(rcond=None)`, H1/H2 cost cap, harness float32 window, metric ε, window-sweep script** — all **off the campaign path**; `main_unified.py`'s import closure contains no `run/study` or `src/solvers` module. On the merits, `rcond=None` is additionally correct: it is the minimum-norm least-squares solution, the well-posed choice on a rank-deficient design; the alternative returns ‖θ‖ 3.8e4 times larger fitting no better (R² agreeing to eight decimals).
- **`training.scale` chain-rule factor** — latent; `scale` = 1.0 in all 30 configurations. The named arming path is refuted: `normalize_entries` exists only in the actor-critic's config, and the actor-critic has no chain rule through the state. Positive control passes: the implemented law reproduces the CARE control to **8.04e-08** relative at scale = 1.
- **`origin_augmentation` basepoint mechanism** — refuted by control. With the ramp **off**, tap-0 concentration persists and on mg_chaotic *increases* (40.70 → 67.46): it is intrinsic to the depth-2 signature, whose first level is x_end − x_0. The feature map is bit-identical between agents on all 15 cell × representation pairs.
- **Subsample anchoring, delay float32 off-by-one, fallback window 10, `state_augmentation`, dead config params** — unreachable or bit-identical under the campaign's flags. Two notes: the stride's remainder condition *fails* on double_integrator ((capacity−1) % resolution = 1), contrary to the canonical claim, but is inert because the reset history is constant; and `state_augmentation` aborts loudly with a shape error rather than corrupting a number.
- **"Evaluation bypasses the divergence guard"** — misdiagnosed. A fixed-horizon evaluation is the *correct* protocol: the declared objective is the finite-horizon integral, and cutting at divergence would **flatter** a diverging agent by halting cost accumulation exactly where cost explodes. The real defect at that site is the non-finite return.

---

## Recommendation

1. **Add one line** to `rl_array.slurm`: `++agent.training.tau_polyak=1.0 \`. Verified to compose for both agents, inert for the actor-critic, and to collapse the lag ratio from 3.5187 to 0.0032 on the signature arm. Changes no code; the existing five-seed results stay bit-reproducible.
2. **Rule on the trade** it implies: the new value-gradient runs lose bit-comparability with the old five-seed history. This is the owner's call, not the audit's. Doya-faithfulness argues for taking it.
3. **Consider, before launch** (actor-critic only, no value-gradient bit risk): return `-inf` rather than `nan` from a non-finite evaluation, and add the missing NaN guard to `signatures_jax._is_episode_done`. Both must land before the campaign rather than between halves of it.
4. **Do not** touch the float32 cast, the Cholesky jitter, the exploration length scale, the divergence threshold, the window sizes, or the RK4 interpolant. Each would invalidate the five-seed history, and several would manufacture asymmetries that do not currently exist.
5. **Record in the write-up**: exploration is white despite `smooth: true`; the plant integrator is second-order on delayed cells despite its name; the testbed has only ever been exercised from constant initial paths; and, if the flag is declined, that the actor-critic-versus-value-gradient contrast is not a controlled comparison.