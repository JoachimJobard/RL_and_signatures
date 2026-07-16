# Pre-registration — learner-invariance of the representation ordering (actor-critic versus value gradient)

**Written 2026-07-16, before the run.** This document fixes the hypothesis, the design, the
metric, the analysis, and the falsification conditions *ahead of* the campaign, so that the
result cannot be steered by the data (the anti-HARKing commitment stated in
`documents/paper/aaai_paper_plan.md`). The outcome is to be reported whatever it is. This file
is to be committed before any job is submitted.

**Scope.** This document governs the **learning half** (reinforcement learning) only, and adds a
**second learner** to it. It does not amend, extend, or reinterpret the oracle half. The
pre-registration of the off-sheet control
(`2026-07-02_preregistration_offsheet_h2_replication.md`), the plan
(`documents/paper/aaai_paper_plan.md`), and the benchmark report
(`documents/reports/2026-06-30_h1h2_representation_benchmark/`) are untouched by it.

---

## 1. Background and motivation

The learning half's existing evidence base is **single-learner**. Every reinforcement-learning
number reported to date comes from the continuous-time value gradient
(`ContinuousValueGradient`, `src/agents/value_gradient_jax.py`), across five studies at five
seeds and three representations (`data/main_unified/`). The continuous-time actor-critic
(`CTACSignatureJAX`, `src/agents/signatures_jax.py`) has, by contrast, an evidence base of nine
runs on one cell (`delayed_oscillator_high_gap`) at three seeds, recorded as PNG images with no
`.npz`, `.pkl` or `summary.yaml`, and therefore **not regenerable** (measured 2026-07-16).

Until 2026-07-16 the actor-critic was additionally **signature-only**: `make_representation` was
called in exactly one agent, and `CTACSignatureJAX` instantiated `SlidingSignatureJAX` directly,
so "actor-critic on raw-history" was unrunnable. That port is now done (commit `2ae3b73`,
verified behaviour-preserving at `max|old − new| = 0.0`), which makes the present question
askable for the first time.

**The question this campaign exists to answer.** Every ordering the project reports — history
helps ($H_1$), signature versus history ($H_2$) — has so far been measured through one learning
algorithm. An ordering that holds for one learner and reverses for another is a property of the
**optimiser**, not of the **representation**, and no representational conclusion may rest on it.
The value gradient and the actor-critic differ in the object they optimise: the value gradient
extracts the control analytically from the critic's vertical derivative
($u = \tfrac12 R^{-1} B^\top \partial_x V$, `value_gradient_jax.py:199-205`), whereas the
actor-critic carries a separate parametrised actor trained against the critic. If the
representation ordering is a representational fact, it must survive that substitution.

**Prior expectation, declared as prior.** The group's settled mechanism (the $L^2$ value fit
leaves $\partial_x V$ unidentified on $\ker G$; see
`documents/analysis/signature_value_gradient_conditioning/FINDINGS.md`) is a statement about
**gradient extraction from a fitted critic**. It applies directly to the value gradient. Whether
it transfers to a learner with an explicit actor is **not known** and is not predicted by this
document beyond the operational predictions of Section 3.

---

## 2. Hypothesis ($H_4$) — the claim under test

> **$H_4$ (learner-invariance of the representation ordering).** At matched cell, representation,
> seed, window, initial path, horizon and divergence rule, the **sign** of the $H_1$ contrast and
> the **sign** of the $H_2$ contrast measured under the continuous-time actor-critic agree with
> those measured under the continuous-time value gradient.

$H_4$ is a statement about **agreement between two learners**, not about which representation
wins. It is deliberately neutral on the direction of $H_2$: the oracle half has refuted the
*representational* reading of an $H_2$ win, and this document does not re-open that question.
$H_4$ is falsifiable in either direction and its verdict is reported per cell.

**What $H_4$ can and cannot attribute — stated now, not after the fact.** The two learners are
distinct algorithms, not one algorithm under a single intervention. They are matched on everything
enumerated in Section 6, but they necessarily differ in the *learning rule itself* (analytic
control extraction from the critic's vertical derivative against an explicitly parametrised actor),
and they retain the residual differences that Section 6 records as open or deliberately retained.
Consequently:

- A **sign agreement** across learners is evidence that the ordering does not depend on the choice
  of learner within this pair — the intended positive content of $H_4$.
- A **sign disagreement** establishes that the ordering is **learner-dependent**, and therefore that
  no representational claim may rest on it. It does **not** identify *which* difference is
  responsible. Attributing a disagreement to a specific mechanism would require a further controlled
  experiment, which this campaign does not perform and this document does not pre-register.

$H_4$ is therefore a test of **robustness across two learners as implemented and as matched by
Section 6**, and is reported in exactly those terms. Any stronger attribution drawn after seeing the
results would be HARKing and is forbidden by this document.

**Notation.** Let $I(a, c, r, s)$ denote the closed-loop cost of learner $a \in \{\text{value
gradient}, \text{actor-critic}\}$ on cell $c$, representation $r \in \{\text{markovian},
\text{raw\_history}, \text{signature}\}$, seed $s \in \{0,1,2,3,4\}$. Let $\tilde I(a,c,r)$ be the
**median over the five seeds** and $\mathrm{CI}(a,c,r)$ its 95% bootstrap confidence interval.
Define the two contrasts

$$\Delta_1(a,c) = \tilde I(a,c,\text{raw\_history}) - \tilde I(a,c,\text{markovian}), \qquad
  \Delta_2(a,c) = \tilde I(a,c,\text{signature}) - \tilde I(a,c,\text{raw\_history}).$$

$H_1$ holds on $(a,c)$ when $\Delta_1(a,c) < 0$ with disjoint 95% confidence intervals; $H_2$
holds on $(a,c)$ when $\Delta_2(a,c) < 0$ with disjoint 95% confidence intervals. A contrast whose
confidence intervals overlap is recorded as **undecided**, never as a win for either side.

---

## 3. Operational predictions (fixed now)

- **P1 ($H_1$ replication, four delayed cells).** On `linear_dde`, `hopfield_linear`,
  `hopfield_nonlinear` and `mg_chaotic`, both learners give $\Delta_1(a,c) < 0$.
- **P2 ($H_1$ falsification control, `markovian`).** On the `markovian` cell neither learner gives
  a decided $H_1$ effect: the 95% confidence intervals of $\tilde I(a,\text{markovian},
  \text{raw\_history})$ and $\tilde I(a,\text{markovian},\text{markovian})$ overlap. This cell is
  the control, and its prediction is a **tautology of the construction, not a measured weakness**:
  the plant is `double_integrator`, whose delayed coupling $A_1 = 0$ and whose delay $\tau = 0$
  **exactly**, so the history window carries no information the Markovian state lacks. A decided
  $H_1$ win on this cell indicates a **harness defect**, not a scientific finding, and under
  Section 7 the campaign is then repaired and re-run rather than reported.
- **P3 ($H_4$ proper — sign agreement).** On each of the five cells,
  $\operatorname{sign}\Delta_1(\text{actor-critic},c) = \operatorname{sign}\Delta_1(\text{value
  gradient},c)$ and $\operatorname{sign}\Delta_2(\text{actor-critic},c) =
  \operatorname{sign}\Delta_2(\text{value gradient},c)$, restricted to cells where **both** learners
  decide the contrast (disjoint confidence intervals under both). Cells where either learner
  leaves a contrast undecided are reported as **undecided**, and are excluded from the $H_4$
  verdict rather than counted as agreement.
- **P4 (the $\varepsilon$-pair controlled experiment).** The `hopfield_linear` /
  `hopfield_nonlinear` pair is the **only controlled experiment in the suite**: the two plants have
  the **same linearisation** because $\varphi_\varepsilon'(0) = 1$, measured **bit-identical**, and
  they share a measured non-Markovianity gate value (history-kernel ratio $0.300$, $H_1$ cost gap
  $+54.86\%$ at the deployment initial condition). Therefore, for each learner, $\Delta_1$ agrees
  between the two cells within overlapping 95% confidence intervals ($H_1$ is held fixed by
  construction), whilst $\Delta_2$ is **free to differ** ($\varepsilon$ moves the nonlinearity in
  the delayed term). P4 is a prediction about $\Delta_1$ only; no direction is pre-registered for
  $\Delta_2$ on this pair.

---

## 4. Falsification conditions (fixed now)

- **$H_4$ is falsified on a cell** when both learners decide a contrast and their signs **disagree**
  — that is, $\Delta_k(\text{value gradient},c)$ and $\Delta_k(\text{actor-critic},c)$ have opposite
  signs, each with disjoint 95% confidence intervals over the five seeds, for $k \in \{1,2\}$. That
  outcome means the ordering is a property of the optimiser and **no representational claim may be
  made from that cell by either half of the project**.
- **P4 is falsified** when, for either learner, $\Delta_1$ differs between `hopfield_linear` and
  `hopfield_nonlinear` with disjoint 95% confidence intervals. Because the linearisation is
  measured bit-identical, such a difference is evidence of a **defect in the harness or in the
  $\varepsilon$ knob**, not of a scientific effect, and triggers Section 7 rather than a report.
- **P2 is falsified** by any decided $H_1$ win on the `markovian` cell, which likewise triggers
  Section 7.

Each cell is falsified or not **independently**. A per-cell falsification does not invalidate the
other cells.

---

## 5. Design (fixed now)

**Pipeline.** The learning half only: `main_unified.py`, dispatched by
`experiments/h1h2_evaluation/jeanzay/rl_launch.sh` and
`experiments/h1h2_evaluation/jeanzay/rl_array.slurm`. No oracle, no off-sheet axis, and no
conditioning statistic is computed or reported by this campaign.

**Learners (2).** `AGENT=value_gradient` (`ContinuousValueGradient`) and `AGENT=signatures`
(`CTACSignatureJAX`). **Both halves are run in this campaign**, rather than comparing the
actor-critic against the existing `main_unified/` value-gradient studies, so that the baseline is
matched **by construction** rather than by argument: the existing studies contain Dadebo and not
Hopfield, and were run before the window, divergence and initial-path corrections of Section 6.

**Cells (5).** Selected against the measured linearised delayed-LQR gate of
`run/study/hopfield_delay_impact.py`, validated first against all five recorded reference points.
The gate reports a history-kernel ratio (initial-condition **invariant**, and therefore the
quantity that carries the verdict) and an $H_1$ cost gap (initial-condition **dependent**,
measured $+6.26\%$ to $+39.09\%$ across initial conditions on one platoon plant, and therefore
quoted at the deployment initial condition only).

| Cell | Plant | Ratio / gap (measured) | Role |
|---|---|---|---|
| `markovian` | `double_integrator` | $0.000$ / $+0.00\%$ | Falsification control; $A_1 = 0$, $\tau = 0$ exactly. Exact CARE oracle. |
| `linear_dde` | `linear_dde_scalar` | $0.346$ / $+117.34\%$ | Linear-in-delay, scalar; largest measured gap. |
| `hopfield_linear` | `hopfield_linear` | $0.300$ / $+54.86\%$ | Linear-in-delay, multichannel ($n=2$). |
| `hopfield_nonlinear` | `hopfield_nonlinear` | $0.300$ / $+54.86\%$ | Nonlinear-in-delay; identical linearisation to the row above. |
| `mg_chaotic` | `MG_1D_chaotic` | $0.115$ / $+89.09\%$ | Nonlinear-in-delay, scalar, chaotic. |

Deliberately excluded as redundant, all three qualifying on the gate and all three remaining
dispatchable in `rl_array.slurm`: `platoon` ($0.398$ / $+78.03\%$; redundant with
`hopfield_linear` on the linear-multichannel axis and the most expensive cell, raw-history
dimension $3320$), `hopfield_duffing` (redundant with `hopfield_nonlinear`), `mg_limit_cycle`
(redundant with `mg_chaotic`). Excluded as **disqualified**: `dadebo_cstr`, measured near-Markovian
at $0.009$ / $+1.03\%$ against a qualification floor of about $0.10$. It is present in
`main_unified`'s existing five-seed benchmark and **must not carry an $H_1$ claim**; its runs stay
on disk and its disqualification is recorded here.

**Representations (3).** `markovian`, `raw_history` (degree 2), `signature` (depth 2). The depth
and degree are passed **explicitly on the command line** by `rl_array.slurm:86-91` for both
learners, so the `depth: 3` default in `conf/agent/signatures.yaml:58` is overridden and does not
enter this campaign (measured 2026-07-16).

**Seeds (5).** Master seeds $0,1,2,3,4$, the explicit seed axis. Per the shared-seed policy, each
(cell, representation) at a given seed derives the same per-role seeds, so a difference between
representations reflects the intervention and not RNG noise.

**Grid.** $5 \text{ cells} \times 3 \text{ representations} \times 5 \text{ seeds} \times 2
\text{ learners} = \mathbf{150}$ tasks, $75$ per learner, dispatched as two arrays.
$N_{\text{episodes}} = 1000$.

**Windows (fixed now, per cell).** `markovian` $10$, `linear_dde` $8$, `hopfield_linear` $5$,
`hopfield_nonlinear` $5$, `mg_chaotic` $71$; made live for both learners by
`agent.signature.force_signature_window=true`. Every window covers its delay: taps needed against
window supplied is $0/10$ on `markovian`, $5/8$ on `linear_dde`, $68/71$ on `mg_chaotic` (measured;
the recorded figures $4/7$ and $24/27$ belong to `platoon` and `mg_limit_cycle`, which are **not**
campaign cells), and $5/5$ on both Hopfield cells, where the window spans the delay **exactly** at
$\tau = 0.5$, $\Delta t = 0.1$. The three pre-existing cells take the value gradient's
**previous effective windows**, so the value-gradient half stays comparable with the existing
studies and only the actor-critic moves; the two Hopfield cells, having no prior value-gradient
data, take the oracle-referenced harness's window ($\text{round}(\tau/\Delta t) + 1 = 5$ at
$\tau = 0.5$, $\Delta t = 0.1$), which spans exactly the delay.

**Compute.** `cpu_p1` + `qos_cpu-t3` (billed CPU; JAX on CPU, no GPU), 8 cores per task,
4 h wall-clock. The smoke test uses `qos_cpu-dev` with `DEBUG=true`. Aggregation and replotting
from the saved `eval.pkl` is a separate `prepost` step (non-billed).

**Output root.** `$SCRATCH/rl_campaigns/RL_and_signatures/`, **never `$WORK`**. Measured cause:
the previous RL array (job 544311) wrote to `$WORK`, hit the inode quota with
`OSError: [Errno 122] Disk quota exceeded`, and produced 80 SLURM log files and **zero** run
directories. `$SCRATCH` is purged periodically, so results are to be rapatriated. SLURM logs go to
a dedicated `slurm/` subfolder of the experiment directory. The experiment group name carries the
learner, because `aggregate_representation_study` groups by (env, kind, capacity) and **not** by
agent, so the two learners must never share a group.

---

## 6. Confound control (fixed now)

$H_4$ compares two learners, so **every difference between the learners that is not the learning
algorithm is a confound**. The following are the differences measured to date. A confound closed by
moving the **actor-critic** is preferred, because the value gradient must stay **bit-unchanged**:
the existing five-seed $H_1$/$H_2$ value-gradient results depend on it, and the actor-critic has no
prior data on these cells to protect.

| # | Asymmetry | Measured state | Resolution |
|---|---|---|---|
| 1 | **Window** | `force_signature_window:false` made both learners discard the array's window and derive their own by different formulae — the value gradient $\lceil \tau/\Delta t \rceil + 3$, the actor-critic $\lceil \tau/\Delta t \rceil + 1$. Measured value gradient / actor-critic: markovian $10/10$, linear_dde $8/6$, platoon $7/6$, mg_limit_cycle $27/25$, mg_chaotic $71/69$. On `kind=signature` this divergence leaves **no trace in any recorded dimension** and is undetectable after the fact. | **Closed.** The array passes an explicit per-cell window with `force_signature_window=true`, set to the value gradient's current windows, so the value gradient is bit-unchanged and only the actor-critic moves. Verified identical, 15/15 cell-by-learner combinations. |
| 2 | **Divergence threshold** | Actor-critic $50.0$ against value gradient $100.0$. Fires on `mg_chaotic`: the actor-critic cut all three episodes at $\lVert x \rVert > 50$ (peak $97.46$) whilst the value gradient cut once at $> 100$. A truncated episode's cost is not a completed episode's cost. | **Closed.** The array passes `agent.training.divergence_threshold=100.0` explicitly to both learners — the value gradient's default and the value used by the existing five-seed runs. |
| 3 | **Initial condition** | The actor-critic burnt `burning_steps + window_size` zero-control steps with the episode clock running (markovian $150 \to 135$, linear_dde $100 \to 87$, mg_chaotic $240 \to 164$ controlled steps; first control at $t = 19$ of $60$). The initial condition of a delay differential equation is a **path** $\varphi \in C([-\tau,0],\mathbb{R}^{d_X})$, not a point, and the zero-control preheat is the wrong object. | **Closed** (commit `67020c4`). The actual initial path $\varphi$ is loaded to capacity and the preheat is retired. Verified against the final bytes: zero preheat rows on all five cells, first control at $t = 0$, exact controlled-step parity, and the loaded window equals the environment's own history buffer to $0$ exactly. $\varphi$ measured **constant** on all five cells, so the front-padding is the true $\varphi$. |
| 4 | **Action clipping** | **OPEN — and measured to bind.** The array does **not** pass `agent.training.clip_action`, so each learner falls through to its own default: `conf/agent/value_gradient.yaml:14` `clip_action: null` against `conf/agent/signatures.yaml:16` `clip_action: 10.0`. The asymmetry is **structural, and the two learners clip different objects**: `value_gradient_jax.py:197` **guards** the clip and applies it to $\mu$, the greedy control, **before** the exploration noise is added at `:247`, so the value gradient's *final action is unbounded* even when the flag is set; `signatures_jax.py:374-375` forms $\mu + \text{noise}$ and clips the **total action**, **unguarded**, at three sites (`:375`, `:748`, `:994`). A clipped control is not the greedy control $u^\star = -\tfrac12 R^{-1} g^\top \nabla V$. | **NOT YET CLOSED — the campaign must not be submitted until it is.** Measured under the array's own invocation at seed 0 over **20 episodes** (not the campaign's 1000): the clip **never binds** on `linear_dde`, `hopfield_linear` or `hopfield_nonlinear` (max $\lvert u \rvert \approx 0.36$–$0.44$ against a bound of $10$), nor on `markovian`/`signature` (max $0.349$) — but it binds on **`markovian`/`raw_history` at $152/2973 = 5.113\%$ of steps, with a pre-clip $\lvert u \rvert$ reaching $2014.44$**. `mg_chaotic` is **not measured**. The value gradient's own $\lvert u \rvert$ is **not measured** and must be, since an actor-critic-only probe cannot certify an actor-critic-versus-value-gradient property. Consequence as measured: on the falsification-control cell the actor-critic is hard-bounded at $10$ whilst the value gradient runs unbounded. Passing `clip_action=10.0` to both would **not** equalise them (the same value clips different objects), and passing `null` to the actor-critic raises (`-None`); the minimal fix is to add the value gradient's guard to the actor-critic's three sites and set its `clip_action: null`, leaving the value gradient bit-unchanged. That the pre-clip magnitude reaches $2014$ indicates the clip performs real stabilisation on that cell, so removing it may alter actor-critic stability there; the divergence rule ($\lVert x \rVert > 100$) still terminates such episodes. |
| 5 | **Initial-path dtype** | **Open, measured.** `value_gradient_jax._fill_buffer_initial` appends the initial history as `float32` in an otherwise `float64` pipeline (code-review finding F-E1). Measured consequence: $\max\lvert \text{AC} - \text{VG} \rvert = 0$ on markovian and platoon, and $1.192093\times10^{-8}$ on linear_dde, mg_limit_cycle and mg_chaotic — exactly the binary32 representation error of $0.8$, vanishing when the actor-critic's window is rounded to `float32`, which identifies the value gradient's cast as the sole cause. | **Retained deliberately, and declared.** The cast **stays**, by explicit decision, because the existing five-seed value-gradient results must remain bit-reproducible. The two learners therefore start from initial paths differing by $\approx 1.2\times10^{-8}$ on three cells. This is declared here as a **known residual asymmetry** of the design, bounded and measured, and is not to be discovered later and reported as a finding. |

**A note on what the port does and does not equalise.** Both learners construct their features
through the same `make_representation` factory, so the three representations are the same code
objects in both. That is a statement about the **factory**, not about the **arguments**: the two
learners previously passed different window arguments to it (row 1), and the resulting feature maps
were not the same objects. Row 1 is what makes that sentence true for this campaign; it is not true
of the code in general, and it must not be quoted as though it were.

---

## 7. Metric and analysis (fixed now)

- **Per task:** the closed-loop cost $I$ at the pre-registered evaluation initial condition and
  horizon, written to a **per-task** artefact (`eval.pkl`, carrying `cost_reduction_pct`) under the
  task's own run directory. No shared file is written by more than one task, so there is no race.
- **Aggregate per (learner, cell, representation):** the **median** over the five seeds and its
  **95% bootstrap confidence interval on the median**. The median is the pre-registered central
  statistic, not the mean, because the raw cost is heavy-tailed. This matches the oracle half's
  pre-registered choice.
- **Divergent seeds** ($I = \infty$ or NaN, or an episode terminated by the divergence rule) are
  reported as a **separate count** per (learner, cell, representation). They are neither dropped nor
  averaged into a NaN. A cell at which the divergent count differs between the two learners has its
  $H_4$ verdict reported **together with that count**, since a divergence-rate difference is itself
  a candidate explanation for a sign disagreement.
- **Verdict per cell:** P1 ∧ P2 ∧ P3 ∧ P4 as stated in Section 3; the conditions of Section 4
  falsify. Contrasts with overlapping confidence intervals are **undecided** and are reported as
  such.
- **The threshold-free comparison is the cost median and its confidence interval.** No success-rate
  threshold decides any verdict; a success rate, if reported, is descriptive only.
- **Reproducibility.** Every figure is to be regenerable from the saved artefacts without re-running
  training. Any number that cannot be read off a concrete run is reported as **not measured** and
  left empty; no slot is filled by extrapolation from another run.

---

## 8. What is NOT changed after seeing results

The learners, the cells, the representations, the seeds, the windows, the horizons, the evaluation
initial conditions, the metric (median $I$ with bootstrap confidence interval), the per-cell verdict
rule, and the falsification conditions are all fixed by this document. Only the reported numbers are
filled in afterwards.

If the pipeline reveals a defect — in particular if the `markovian` falsification control (P2) or
the $\varepsilon$-pair control (P4) violates its predicted outcome — the defect is documented, the
fix is committed, and the campaign is **re-run**. A verdict is not read off a run already known to
be defective. Such a re-run is a documented correction of a design or implementation defect, **not**
a post-hoc adjustment of the analysis to obtain a desired result; the precedent and the standard are
the amendment recorded in `2026-07-02_preregistration_offsheet_h2_replication.md`.

---

## 9. Scope limit carried over from the oracle half

This campaign may report **measured reinforcement-learning performance** ("the signature critic
attains lower closed-loop cost than the degree-two history critic on cells $X$ and $Y$, under both
learners"). It may **not** assert that the signature is a **richer representation** of history: that
claim is refuted by the oracle half's pre-registered off-sheet control and is the subject of the
second paper. $H_4$ is deliberately constructed to be neutral on the direction of $H_2$ and to test
only agreement between learners, so no result of this campaign can be read as reinstating the
representational claim.
