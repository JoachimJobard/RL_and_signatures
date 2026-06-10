# Correctness code review — signature-based continuous-time RL

**Repository:** `RL_and_signatures` (Joachim Jobard internship handoff)
**Reviewed commit:** `a1a18f1` (tagged `internship-handoff-2026-06-10`)
**Date:** 2026-06-10
**Reviewer:** code-review pass against the master thesis and Doya (2000).

## 1. Scope and method

This review assesses the *correctness* of the source under `src/` (agents, environments,
numerical integrators, signature representation, training/evaluation) and the experiment
drivers under `run/`. It does **not** yet change any code; it produces a prioritised list of
findings to be addressed in later phases (fixes, unit tests, plot utilities, cluster port).

**Authoritative sources for "intended behaviour":**

- J. Jobard, *Continuous Reinforcement Learning for Delayed Dynamical Systems: A Path
  Signature-Based Representation Approach* (2026) — `documents/references/`. Cited as
  *(thesis, p.N / Eq.K)*.
- K. Doya, *Reinforcement Learning in Continuous Time and Space* (2000) — `documents/references/`.

Every finding carries a **verdict** against the thesis:

- **CONTRADICTS** — the code implements something the thesis specifies differently (a true defect).
- **MATCHES** — verified faithful to the thesis (recorded to prevent re-flagging).
- **OUT-OF-SCOPE** — code with no counterpart in the thesis (e.g. the soft actor-critic); not
  validated against any specification, treated as unsupported until a spec is supplied.
- **DECISION** — correctness depends on a modelling choice the thesis leaves open; needs the
  user's call.

Severity: **critical** (silently wrong results on a path the experiments exercise) ·
**major** (wrong or inert behaviour on a configurable axis) · **minor** (hygiene, dead code,
latent edge case).

---

## 2. Executive summary

The numerical core is in better shape than a first pass suggested: the delay-DDE integrator is
a **correct** method-of-steps RK4 with linear history interpolation, faithful to the thesis
(§7.1 below), and the value-gradient control law's factor ½ is **correct** given the running
cost `uᵀRu`. The substantive defects are:

1. **Mackey–Glass running cost does not match the thesis** — the code penalises the control
   *rate* `Δu` (plus a vanishing effort term), the thesis penalises control *effort*
   `0.1·u²`. *(F-D1, major, CONTRADICTS.)*
2. **`critic_oracle` value sign is inconsistent** between agents and wrong relative to the
   thesis's reward-value convention `V* = −xᵀPx`. *(F-A1, critical, CONTRADICTS.)*
3. **Signature path augmentation diverges from the thesis** — an extra plain-time channel is
   kept (the thesis removed it) and the origin channel has the opposite sign. *(F-C1, major,
   CONTRADICTS.)*
4. **Configured `window_size` is silently overridden**, and the window-size sweep experiment
   therefore varies nothing. *(F-C2 / F-H1, major.)*
5. **`clip_gradient` is ignored** (hardcoded ±10, per-element) in the signature and
   value-gradient agents. *(F-B3, major, CONTRADICTS config.)*
6. **The signature representation is computed in float32** under a float64 pipeline. *(F-E1,
   major.)*
7. A cluster of **inert flags and dead code paths** (`semi_gradient`, `integral_td`,
   `time_origin`, two divergent loss implementations) that make the configuration surface
   misleading. *(Section 8.)*
8. The **soft actor-critic (`CSAC_jax.py`) is outside the thesis scope** and carries at least
   one latent crash; it should be quarantined or removed unless a specification is provided.
   *(F-G1.)*

---

## 3. Findings — value function and sign conventions

**Thesis convention (p.21, Eq.24–26; confirmed p.45):** the value is the *supremum of the
discounted reward integral*, with reward `r = −(xᵀQx + uᵀRu)`. Hence `V` is **negative** for a
stabilising controller and the LQR oracle value is `V*(x) = −xᵀP x`. The thesis explicitly
flags a *positive* learned signature-value as "incoherent" (p.45).

### F-A1 — `critic_oracle` value sign inconsistent (critical, CONTRADICTS)
`src/agents/signatures_jax.py:417-418` evaluates the oracle as `V_t = xᵀ P x` (**positive**),
while `src/agents/base_jax.py:124-126` and the `get_value` helpers
(`signatures_jax.py:649`, `base_jax.py:432`) use the correct **`−xᵀP x`**. The two oracle code
paths thus disagree in sign; the positive form contradicts the thesis convention and corrupts
the TD error wherever the `critic_oracle` baseline is used.
**Action:** change `signatures_jax.py:417-418` to `−xᵀP x`.

### F-A2 — value-gradient control law factor ½ is correct (MATCHES)
`src/agents/value_gradient_jax.py:164` uses `u = ½ R⁻¹ Bᵀ ∂V/∂x`. With running cost `uᵀRu`
(no ½) and the negative-reward value (`∂V/∂x = −2P x`), this yields `u = −R⁻¹BᵀP x`, the
correct LQR gain (thesis Eq.43/52, p.30/32). **No change.** This is correct *iff* the learned
critic actually converges to the negative-reward value — see F-A1 and the open decision D-1.

---

## 4. Findings — continuous-time TD error and actor update

**Thesis convention (Alg.1 p.23, Alg.2 p.33):** semi-gradient differential TD,
`δ = r − V/τ + dV/dt`, with `dV/dt` a finite difference; only the value-gradient agent uses a
target network (Polyak `τ_polyak = 0.01`); the three CTAC variants do not. *(The thesis's
printed Alg.1 writes the difference as `(V_t − V_{t+Δt})/Δt`, i.e. `−dV/dt`, and omits the
`δ` factor on the actor update; both are typesetting errors in the thesis — Doya's correct
forms are `+dV/dt` and `δ·(noise/σ²)·∂μ/∂w`. The code uses the correct forms, see below.)*

### F-B1 — TD error sign and target-network usage match the thesis (MATCHES)
All agents compute `δ = r + (V_next − V_t)/dt (− V_t/τ if discounted)`
(`signatures_jax.py:470-476`, `base_jax.py:155-157`, `value_gradient_jax.py:219-225`), i.e.
`+dV/dt` — Doya-correct, not the thesis's mis-signed print. Target network present only in the
value-gradient agent (`value_gradient_jax.py:237-238`) and absent in the CTAC agents — matches
the thesis tables (p.79–81). **No change.**

### F-B2 — actor update uses the correct score `noise/σ²` with the `δ` factor (MATCHES)
The live actor loss `−δ·⟨noise/σ², μ⟩` (`signatures_jax.py:499-502`, `base_jax.py:182-183`)
reproduces Doya's noise-correlation update including the advantage/`δ` factor (the thesis's
printed `n/σ` without `δ` is a typo). **No change**, but see F-F2 (a dead, mis-scaled duplicate).

### F-B3 — `clip_gradient` ignored; hardcoded ±10, per-element (major, CONTRADICTS config)
`signatures_jax.py:484,509` and `value_gradient_jax.py:232` clip gradients to a literal
`[−10, 10]` **per element**, ignoring `training.clip_gradient` (set to `5.0` in
`signatures.yaml`). `base_jax.py:164` (CTACJAX) and `CSAC_jax.py` honour the config, and CSAC
uses `clip_by_global_norm` — so three different clipping semantics coexist.
**Action:** read `self.training.clip_gradient`; standardise on global-norm clipping across agents.

### F-B4 — actor optimizer `b1=0.1` (major, DECISION)
`signatures_jax.py:153` builds the actor Adam with `b1=0.1` (near-memoryless first moment),
asymmetric with the critic (default `b1=0.9`) and undocumented. Almost certainly an unintended
leftover. **Action:** confirm intent; default to `b1=0.9` unless the thesis/notes justify it.

---

## 5. Findings — signature representation

**Thesis convention (§4.1.1 p.25; Alg.4 p.37; §5.3.1 p.38):** the path is the single augmented
channel set `x̃_t = (x_t, −x(0)/h · t)` on `[−h, 0]` — normalised time, basepoint via `x(0)`,
**plain time channel deliberately removed**; truncation depth ∈ {2,3,4}; window length `h = τ`
by default, rolling buffer sampled at the control step `Δt`.

### F-C1 — augmentation channels diverge from the thesis (major, CONTRADICTS)
`src/utils/dynamic_signature.py:162-168`, with the live config
`time_augmentation: true, origin_augmentation: true` (`conf/agent/signatures.yaml`):
- The code keeps a **plain time channel** (`time_aug` branch, line 167-168) that the thesis
  explicitly removed for being non-vanishing at the window edge.
- The origin channel is `x(0)·linspace(−1,0,n)` (line 163-164), which runs `−x(0) → 0` over the
  window; the thesis channel `−x(0)/h·t` runs `+x(0) → 0` — a **sign flip** (and hence a sign
  flip of every iterated integral involving that channel).
**Action:** to match the thesis, set `time_augmentation: false`, keep `origin_augmentation:
true`, and correct the origin channel sign to `−data[0]·relative_times` (or equivalently
`data[0]·linspace(time_origin,0,n)`). If both channels are intentionally retained as an
extension beyond the thesis, document it at the definition site.

### F-C2 — configured `window_size` silently overridden (major)
`src/agents/signatures_jax.py:129-130`: unless `force_signature_window` is true, the configured
`window_size` (40 in `signatures.yaml`, 20 elsewhere) is overwritten by
`ceil(max_delay/step_size)+1`. This *coincides* with the thesis default `h = τ`, so the default
runs are fine — **but** (a) the log message at `:132` is inverted (it announces the override in
the branch that keeps the configured value), and (b) the window-size sweep cannot work without
`force_signature_window` (see F-H1). **Action:** fix the inverted log; make the window-sweep
driver set `force_signature_window: true`.

### F-C3 — `time_origin` is dead config surface (minor)
`SignatureConfig.time_origin` (default `1.0`, `src/configs.py:77`) is read at
`signatures_jax.py:137` but absent from every signature YAML, so it is untunable without editing
code. With `time_origin = 1.0` the time channel is normalised to `[−1,0]` regardless of the
physical window length `h` — consistent with the thesis's normalised-time convention (§4.1.1),
so the value is fine; only the config exposure is missing.

### F-C4 — `signature_size` closed form divides by `d−1` (minor)
`src/utils/dynamic_signature.py:35` (`SlidingSignature`) uses `(d**(depth+1)-1)//(d-1) - 1`,
which raises `ZeroDivisionError` for `d == 1` (a 1-state system with no augmentation). The
production class `SlidingSignatureJAX:124` uses the safe summation form. Latent only for the
un-augmented scalar case. **Action:** replace with the summation form for uniformity.

---

## 6. Findings — environments, dynamics, costs

### F-D1 — Mackey–Glass cost penalises `Δu`, thesis penalises effort (major, CONTRADICTS)
`src/envs/mackey_glass_1D.py:85-91` computes `(x−x_target)ᵀQ(x−x_target) + ΔuᵀRΔu + ε·uᵀRu`
with `Δu = u − last_u`, `ε = 1e-3`. The thesis cost (Eq.62, p.45) is `x² + 0.1·u²` — control
**effort**, target 0, no rate term. The implemented objective (heavy `Δu` rate penalty, near-
zero effort penalty) is materially different and will produce a different optimal controller.
The `else` branch (lines 89-90, effort-only) is moreover **unreachable**: the guard
`if self.x_target is not None or ...` is always true since `x_target` is always assigned in
`__init__`. **Action:** implement the thesis cost `xᵀQx + 0.1·uᵀRu`; remove the tautological
guard; promote the hardcoded `ε` and the cost form to config so the `Δu` variant remains
selectable as an explicit extension.

### F-D2 — delay-DDE integrator is a correct method-of-steps RK4 (MATCHES)
`src/envs/env_rk_jax.py:45-59` + `src/utils/solver_buffer_jax.py:36-62`. For an RK stage at
time `t + c·h` (`c ∈ {0, ½, 1}`), `adjusted_delay_steps = τ/h − c` retrieves `x(t + c·h − τ)`
by backward **linear** interpolation between the two buffered samples straddling it — exactly
the thesis's method-of-steps RK4 with a linear history interpolant (Alg.3 p.35), with the
acknowledged loss of the full 4th-order accuracy. The buffer is correctly frozen within a
sub-step and appended after. **Verified correct — not a bug.** (An earlier audit flagged this
as critical on the basis that `x(t−τ+h/2)` differs from `x(t+h/2−τ)`; those are identical, so
the concern is void.) *Edge case (minor):* if `τ < h` the adjusted delay can go negative and
index a future/stale slot; all current configs have `τ ≫ h`, so add a guard/assertion rather
than a fix.

### F-D3 — chemical CSTR `B(x)` is self-consistent (MATCHES, with a thesis typo noted)
`src/envs/chemical_process.py` hardcodes the control as `−u₀(x₂+0.25)` / `−u₁(x₄+0.25)` in the
dynamics and `get_B` returns the matching `−(x₂+0.25)`, `−(x₄+0.25)`; the value-gradient agent
uses this same `get_B`. Internally consistent. (The thesis Eq.60, p.42 prints `B = ∂f/∂u` with
*positive* entries, which contradicts its own Eq.57 dynamics — a thesis typo; the code follows
the mathematically correct `∂f/∂u`.) **No change**, but note the base linear `dynamics` uses
`self.B`, not `get_B`, so any future state-dependent linear env must override both consistently.

### F-D4 — reward sampled at the final sub-step (minor, DECISION)
`env_rk_jax.py:88` returns the instantaneous reward at the last RK sub-state, not the integral
`∫ r dt` over the control interval. The thesis algorithm uses a pointwise `r_t`, so this is
defensible, but the evaluation metric is a cumulative integral (thesis Eq.61). **Decision:**
confirm whether the learning reward should be the sub-step integral; if so, accumulate it in the
`lax.scan`.

### F-D5 — first-step `Δu` penalty depends on reset internals (minor)
`reset` sets `last_u = 0` (`env_rk_jax.py:79`); with the `Δu` cost (F-D1) the first step is
penalised as a jump from zero. Becomes moot once F-D1 switches to an effort cost.

### F-D6 — three env configs have a broken `_target_` (major, CONTRADICTS)
`conf/env/delay_jax_unstable.yaml:5`, `unstable_oscillator_delay.yaml:5`,
`harmonic_oscillator_rk_jax.yaml:5` reference `src.env_rk_jax.JAXDDEEnv`; the module is
`src.envs.env_rk_jax` (cf. the correct `delay_jax.yaml`). Instantiating any of these raises
`ModuleNotFoundError`. **Action:** fix the module path (a unit test instantiating every env
config would catch this class of error — Phase 3).

---

## 7. Findings — numerical precision and RNG

### F-E1 — signature features computed in float32 under a float64 pipeline (major)
`main_unified.py:40` enables `jax_enable_x64`, but the signature buffer hard-casts to
float32 (`dynamic_signature.py:24,75,81,100`; append at `signatures_jax.py:202`). Depth-3
signatures of a `2d+1`-channel path are sums of `d³` products of small windowed increments;
float32 loses precision there, and the downstream float64 networks then consume float32-rounded
features. **Action:** carry the buffer/signature in float64 (respect the global x64 flag).

### F-E2 — RNG threaded through mutable `self.key` (minor, DECISION/hygiene)
Action selection and episode setup reassign `self.key` via `jax.random.split` inside Python
methods rather than threading keys functionally. It is correct under the single-threaded eager
loop (no key reuse observed), but it is the implicit-global-state pattern the project's RNG
policy warns against. **Action (Phase 2):** thread explicit per-role keys derived from one
master seed; this also enables the shared-seed ablation policy.

---

## 8. Inert flags, dead code, and duplicated logic

These do not change current results but make the configuration surface untrustworthy and invite
future edits to the wrong code path. Resolve during the Phase-2 reorganisation.

- **`semi_gradient` flag inert for the signature agent** — `signatures_jax.py:472` hardwires
  `stop_gradient` on `V_next`, so the agent is *always* semi-gradient. This **matches** the
  thesis (semi-gradient throughout), so the *behaviour* is correct; only the flag is a lie.
  `base_jax.py:151-154` does honour it. **Action:** either wire the flag or delete it and
  document that semi-gradient is the only supported mode. *(MATCHES behaviourally.)*
- **`integral_td` flag inert** — implemented only in the dead `_compute_td_error`
  (`signatures_jax.py:306-318`); the live critic losses never branch on it. The thesis uses
  differential TD only, so this is fine to delete. *(MATCHES behaviourally.)*
- **Two divergent loss implementations** — the dead methods `critic_loss_fn`/`actor_loss_fn`
  (`signatures_jax.py:325-350`) use different math than the live closures (the dead actor loss
  multiplies by raw `noise`, missing the `1/σ²`; the dead path concatenates the *unscaled*
  state). Delete the dead methods. *(F-F2.)*
- **Duplicated `CriticFlax`** in `networks/value_gradient_nets.py` and
  `networks/LQR_actor_critics.py` (identical bodies, different importers) — consolidate.
- **`integral_volterra.py` duplicates `chemical_process.py`** (same class, misleading name) —
  remove one.
- **`signature_jax.yaml` uses a flat schema** the `_make_config` loader does not consume; it is
  stale/legacy versus the nested `signatures.yaml`. Confirm which configs are live and delete
  the dead ones.

---

## 9. Out-of-scope code

### F-G1 — soft actor-critic (`CSAC_jax.py`) has no thesis counterpart (major, OUT-OF-SCOPE)
The thesis defines four agents (signature-CTAC, vanilla CTAC, full-trajectory CTAC,
value-gradient) and **no** soft/entropy-regularised variant (§4.2, p.26; future work only). The
`CSAC` agent is therefore unvalidated against any specification. It also carries concrete latent
defects: `get_eval_action` references `self.signature_conf` though the attribute is
`self.signature` (`CSAC_jax.py:524` → `AttributeError` on the non-oracle eval path), and the
critic-loss scaling `½·δ_M²/dt/(1+dt/τ)` diverges as `dt → 0` for an `O(1)` martingale
increment. **Action:** quarantine CSAC out of the default experiment set, or obtain its source
specification (the paper it follows) before relying on it. Do not include it in correctness
claims tied to the thesis.

---

## 10. Findings — experiment drivers

### F-H1 — `run_comparison_window_size.py` sweeps nothing (major)
`run/MG_transient/run_comparison_window_size.py` embeds `${agent.signature.window_size}` in the
run name but never overrides it with a list, and (per F-C2) the value is overridden anyway. The
"window-size" experiment degenerates to a single window. **Action:** sweep
`agent.signature.window_size` *and* set `agent.signature.force_signature_window=true`.

### F-H2 — driver inconsistencies (minor)
MG drivers launch with bare `python`, the chemical driver with `uv run python`
(`run_comparison_chemical.py:164`) — only one matches a given cluster environment; the
`delays`/`window_size` parameters in `build_cmd` are partly unused. These will be normalised
when the drivers are rewritten as cluster job-array launchers (Phase 5) and reorganised to the
`script_data_dir` convention (Phase 2).

---

## 11. Prioritised action list

**Correctness fixes (do first, each with a regression test in Phase 3):**
1. F-A1 — `critic_oracle` value sign `+xᵀPx → −xᵀPx` (`signatures_jax.py:417-418`).
2. F-D1 — Mackey–Glass cost → thesis effort cost `xᵀQx + 0.1·uᵀRu`.
3. F-C1 — signature augmentation → single origin channel, correct sign; `time_augmentation:false`.
4. F-B3 — honour `clip_gradient`; unify clipping semantics.
5. F-D6 — fix the three broken env-config `_target_` paths.
6. F-E1 — signature/buffer in float64.

**Configuration integrity:**
7. F-C2 / F-H1 — window override log + window-size sweep.
8. Section 8 — remove inert flags and dead/duplicated code, or wire them.

**Out-of-scope / decisions:**
9. F-G1 — quarantine CSAC.
10. Decisions D-1…D-4 below.

## 12. Open decisions for the user

- **D-1 (sign convention):** confirm all agents should use the negative-reward value
  `V* = −xᵀPx` (thesis convention). This makes F-A1 a straightforward fix and validates F-A2.
- **D-2 (MG cost):** confirm the thesis effort cost `x² + 0.1u²` is the target and the `Δu`
  rate-penalty variant should be demoted to an explicit, config-selectable extension.
- **D-3 (signature augmentation):** confirm we should match the thesis exactly (origin channel
  only) rather than keep the current two-channel-with-sign-flip variant.
- **D-4 (CSAC):** keep `CSAC_jax.py` as an unsupported extension (quarantined) or remove it.
