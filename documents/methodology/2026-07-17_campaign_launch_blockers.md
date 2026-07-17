# Campaign launch blockers — the maths-versus-plumbing audit, interim register

*Status: interim. Written 2026-07-17 from 70 of 105 lens-checks (66%) of the refutation workflow
`wf_23fe6377-3da`, whose synthesis agent did not run. Every number below is quoted from a verdict
that reproduced its finding by execution, or is marked* **not measured**. *The evidence is
persisted at `~/claude-workspace-rho/ac_campaign_2026-07-16/`.*

**The actor-critic campaign (5 cells × 3 representations × 5 seeds × 2 learners = 150 tasks) is
NOT to be launched while blockers 2 and 3 stand.** Nine blocking verdicts were returned; they
collapse into the three defects below.

---

## Method note: why the coverage matters

The blocking count rose **monotonically with coverage** — 5 verdicts at 13%, 7 at 40%, 9 at 66%.
Every increment of coverage armed another defect. An audit truncated at "the critical items" would
therefore have reported a launch-safe campaign; the register below exists because it was not
truncated. Two of the three defects were found only after the point at which scoping-down had been
proposed.

---

## Blocker 1 — Action clipping. **CLOSED** (commit `905e7d1`)

The array never overrode `agent.training.clip_action`, so the value gradient ran at `null` (no
clipping, guarded at `value_gradient_jax.py:197`) and the actor-critic at `10.0`, applied
**unguarded** at three sites. The two agents also clipped **different objects**: the value gradient
clips the greedy control $\mu$ *before* adding exploration noise (so its applied action is
unbounded), the actor-critic clips the total action $\mu + n$ *after*.

Measured binding, by two independent probes at different horizons:

| probe | cell / representation | binding | max pre-clip $\lvert u \rvert$ |
|---|---|---|---|
| 20 episodes | `double_integrator` / `raw_history` | 152/2973 = 5.113% | 2014.44 |
| 20 episodes | `double_integrator` / `signature` | **0/3020 = 0.000%** | 0.349 |
| **200 episodes** | `double_integrator` / **`signature`** | **4535/25218 = 17.98%** | 151 |
| 200 episodes | `mg_chaotic` / `raw_history` | ARMED (reported; not reproduced here) | not measured |

**The 20-episode and 200-episode probes disagree on the same (cell, representation) pair.** The
actor's output grows during training, so a short probe reports absence of evidence and reads as
evidence of absence. The campaign runs **1000** episodes. Any clamp probe must state its horizon.

$H_1$ is the `raw_history`-against-`markovian` contrast, so the clamp fired on **one arm of it** on
the falsification-control cell. Closed actor-critic-side (`clip_action: null`, guard added at the
three sites), so the value gradient is bit-unchanged; verified by effect, the two arms that never
bound being bit-identical before and after (max $\lvert u \rvert$ 0.352286 and 0.349321).

---

## Blocker 2 — The `mg_chaotic` evaluation is non-finite for the actor-critic. **OPEN**

Both agents evaluate with `_is_episode_done(x_t, time_only=True)` (`value_gradient_jax.py:708`,
`signatures_jax.py:1164`), and the `time_only` branch short-circuits **every state guard** — both
the NaN test and the divergence cut. The evaluation therefore always integrates the full horizon.

That is **correct in principle**: the objective is the finite-horizon integral $\int_0^T r\,dt$, and
a truncated evaluation would report a different quantity. It is **broken in practice**, measured on
`MG_1D_chaotic` / `raw_history` / seed 0 under the array's exact invocation:

| | eval peaks $\lVert x \rVert$ (episodes 0/20/40) | eval series | best checkpoint |
|---|---|---|---|
| `AGENT=signatures` | $3.37\times10^{28}$, $1.44\times10^{22}$, $1.26\times10^{27}$ — **all non-finite** | `[nan, nan, nan]` | **never selected** (`_best_critic_params is None`) |
| `AGENT=value_gradient` | 52.64, 2.80, 1.19 — all finite | `[-18622.02, -263.9, …]` | selected |

The actor-critic's evaluation diverges to $10^{28}$, goes non-finite, and **because every
evaluation is NaN the best-checkpoint selection never fires at all**, so that arm's reported number
is not a measurement of anything. The code is structurally identical for both agents: the asymmetry
is in **what fires**, not in what is written.

**Consequence for the design.** `mg_chaotic` cannot carry an actor-critic $H_1$ verdict in its
current state. Either the cell is excluded from the actor-critic half and that exclusion recorded,
or the evaluation is made to represent divergence honestly (report the run as **diverged**, with
its peak norm, rather than as a cost). Adding the divergence cut to the evaluation rollout is
**not** bit-safe in general: it would alter the value gradient's eval series wherever a value-
gradient eval exceeds 100, changing the argmax of checkpoint selection and hence the headline
number. Measured on this cell/seed the value gradient peaks at 52.64 (< 100) so the cut would not
fire — but the peak eval norm is **not measured** for seeds 1–4, the other four cells, or the other
two representations, so that variant must not be applied until it is.

---

## Blocker 3 — The critic target network. **OPEN. A decision for the project owner, not a fix.**

Six of the nine blocking verdicts concern this. `conf/agent/value_gradient.yaml:18` sets
`tau_polyak: 0.01` and the array never overrides it, so the value gradient bootstraps from a
**Polyak-averaged target network**:

```
value_gradient_jax.py:261   V_next = critic.apply(jax.lax.stop_gradient(target_params), sig_next)
value_gradient_jax.py:275   optax.incremental_update(new_params_critic, target_params, step_size=tau_polyak)
```

whereas the actor-critic bootstraps from the **live critic** (`signatures_jax.py:619`). The two
losses "differ in exactly one token". **At fixed representation the two learners therefore optimise
different critic objectives** — a fourth agent-asymmetry confound, structural rather than
parametric, requiring no measurement to establish its existence.

Three directions, each with a real cost. None is free, and the choice is the author's:

| direction | value gradient | cost |
|---|---|---|
| **1. Remove the target network from the value gradient** (match the actor-critic, and match Doya) | **invalidated** — changes the critic gradient on the first step, hence every trajectory | Blocked by the standing constraint that the existing five-seed $H_1$/$H_2$ results stay reproducible. |
| **2. Add a Polyak target network to the actor-critic** | bit-unchanged | Propagates the device to **both** arms: the comparison becomes symmetric-and-wrong rather than correct. Defensible **only** if the campaign's claim is explicitly restricted to a representation contrast at a fixed, non-Doya critic rule — and stated as such in the write-up, since **neither arm would then implement the greedy control of Doya (2000)**. |
| **3. Do not fix; document** | bit-unchanged | The comparison stays confounded; every cross-learner statement must carry the asymmetry. |

**The fact that makes this uncomfortable, and which the paper must not elide: Doya (2000) has no
target network.** It is a later engineering device. Doya's continuous-time residual is
$\delta = r - V/\tau + \dot V$ with $\dot V$ taken from the *same* critic. So the **actor-critic is
the Doya-faithful arm and the value gradient is not** — and the existing five-seed value-gradient
results, which the paper's empirical claims rest on, were produced by a critic that deviates from
the algorithm the paper cites. The deviation does not move the fixed point (a Polyak target with
$\tau_{\text{polyak}} = 0.01$ converges to the same solution); it changes the estimator and the
transient. That is a claim-strength question, not a correctness question, and it belongs in the
write-up rather than in a silent fix.

---

## Confounds closed, for the record

| # | asymmetry | resolution |
|---|---|---|
| 1 | Window (VG $\lceil\tau/\Delta t\rceil + 3$ against AC $+1$) | Closed: `force_signature_window=true` with explicit per-cell windows. Refuted as a campaign confound by execution — "with `force_signature_window=true`, value_gradient and signatures agree exactly on every cell × representation". |
| 2 | Divergence threshold (50.0 against 100.0) | Closed: the array passes 100.0 to both. |
| 3 | Initial condition (zero-control preheat against the DDE's initial path $\varphi$) | Closed, commit `67020c4`. |
| 4 | Action clipping | Closed, commit `905e7d1`. See Blocker 1. |
| 5 | `critic_lr` (1e-3 against 1e-2), `noise.length_scale` (0.002 against 0.02) | Closed actor-critic-side, commit `0fcf731`. Neither was ever tuned: `git log -S critic_lr` returns one commit per config, `a02cc92`. |
| 6 | `discount.tau` (100.0 against 1.0) | Closed, commit `15fe75c`. **Inert and verified so**: `discounted: false` guards it on every live path and `algorithm.lstd` resolves to `False`, so the one unconditional use is unreachable. The value gradient is bit-identical across all three representations. |
| 7 | Initial-path dtype (`float32` cast in a `float64` pipeline) | **Retained deliberately.** The two learners start from initial paths differing by $\approx 1.19\times10^{-8}$ on three cells. Declared, not fixed, so it cannot later be discovered and reported as a finding. |

---

## Also established

- **`training.scale`**: the missing $1/\text{scale}$ chain-rule factor in the control law is
  **latent, not armed** — every agent config sets `scale: 1.0` and
  `conf/agent/signatures.yaml:72` sets `normalize_entries: false`, and `signatures_jax.py:93` arms
  the factor only when that flag is true. A foot-gun a future config would arm.
- **The value-gradient control law is correct**: the decisive test against the exact CARE solution
  on the double integrator **passes**. The sign error is in the paper, not the code.
- **The exploration noise is numerically white in both agents** despite `smooth: true`: the
  adjacent-step correlation $\exp(-\Delta t^2/2\ell^2)$ is $0.000$ at $\ell = 0.002$ and
  $3.7\times10^{-6}$ at $\ell = 0.02$ ($\Delta t = 0.1$). Both covariance matrices are numerically
  the identity. `src/configs.py:66` defaults $\ell$ to 0.2, which **would** give correlation 0.883;
  both shipped configs override that default. This is a fidelity gap against Doya, whose
  exploration is low-pass filtered (Ornstein–Uhlenbeck, $\tau_n = 1.0$) and therefore correlated.
  It is symmetric across agents and so is not a confound.
- **The divergence cut does not touch any reported number**: `src/training/evaluate.py`, which
  computes `cost_reduction_pct`, contains no divergence test and always simulates the full horizon.
  The cut is a **training** device (it bounds compute and prevents a non-finite target destroying
  the critic) — categorically unlike the clip, which overwrote the control law itself. It is now
  logged (commit `0fcf731`). **But it is not innocent**: a cut episode is shorter, and the campaign
  trains a fixed number of *episodes*, so an arm that is cut receives systematically fewer gradient
  updates than one that is not (actor-critic on `markovian`: `raw_history` cut 5/20, `signature`
  0/20). Its count must be reported per (learner, cell, representation).

---

## Viability of the five cells (measured 2026-07-17, `value_gradient` / `signature` / seed 0)

Every cell produces a working controller, so no cell's 30 tasks are spent on an uncontrollable
plant. `cost_reduction_pct` is **self-relative** — the closed-loop cost against the *same plant left
uncontrolled* — so it states that the controller helps, never that it is near-optimal.

| cell | cost reduction | episodes |
|---|---|---|
| `markovian` | **+29.70%** | 200 |
| `linear_dde` | +91.30% | 200 |
| `hopfield_linear` | +98.60% | 200 |
| `hopfield_nonlinear` | +32.70% | 2 (its 200-episode run was destroyed by a concurrent cleanup) |
| `mg_chaotic` | +98.30% | 200 |

`markovian` is conspicuously the weakest, on the one cell whose exact CARE optimum is known in
closed form. That is worth reading against the oracle rather than against zero.

---

## Open, and not to be mistaken for closed

1. **Blocker 2** (`mg_chaotic` non-finite evaluation) and **Blocker 3** (target network) stand.
2. **35 of 105 lens-checks never ran**, and the synthesis agent never ran. The register above is
   two-thirds of an audit. Given that the blocking count rose monotonically with coverage, the
   expected number of blockers at full coverage is **higher than nine**.
3. **The Monte-Carlo policy gradient (commit `5b410d6`) has its own open register** — 28 findings,
   21 reproduced, including a sign flip invisible to all 13 of its tests and an
   optimistically-biased return on a truncated episode. It must not enter the campaign until that
   register is closed.
