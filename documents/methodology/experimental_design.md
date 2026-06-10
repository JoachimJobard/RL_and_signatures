# Experimental design (pre-registration)

This note fixes the design of the experiments **before** they are run, so the
analysis cannot be steered by the results. It states the hypotheses, the design
factors, the predicted outcomes (with falsification conditions), the fairness
protocol, the validation/oracle ladder, the analytic baselines, and the
correctness safeguards. It is the contract the implementation and the analysis
must respect.

References:
- I. Perez Arribas, *Derivatives pricing using signature payoffs* (2018) — the
  linear-approximation theorem (Thm 4.2) at the core of the signature approach.
- K. Doya, *Reinforcement Learning in Continuous Time and Space* (2000) — the
  continuous-time value-gradient control used as the learning backbone.
- V. Kolmanovskii, A. Myshkis, *Applied Theory of Functional Differential
  Equations* (1992), §6.2 — the linear-quadratic regulator for delay systems
  (the analytic oracle for the linear delayed environment).

All three are in `documents/references/` (Arribas, Doya) or the Scientific Library
(Kolmanovskii–Myshkis).

## 1. Objective and hypotheses

The study tests two hypotheses about *non-Markovian* (history-dependent) control
when the environment dynamics are themselves non-Markovian (delay differential
equations).

- **H1 (non-Markovian modelling helps).** When the dynamics are non-Markovian, a
  policy/value that is a functional of the *history* $x_t = \{x(t+\theta):
  \theta\in[-h,0]\}$ attains strictly lower control cost than one that is a
  function of the *current state* $x(t)$ alone.
- **H2 (signature representation helps).** When the dynamics are non-Markovian and
  the optimal control is a *nonlinear* functional of the history, a
  signature-based representation of the history attains lower control cost — with
  the value a **linear functional of $\Phi$ (matched hypothesis class) at matched
  capacity** — than a basic ("raw history") representation, e.g. the discretised path
  or its polynomial features.

H2 is stated with the value linear in $\Phi$ at matched capacity deliberately: the
claim is about the representation's inductive bias, not about model size or a
nonlinear parameterisation of $\Phi$ (see §4).

## 2. Design factors

### 2.1 Representation factor $\Phi$ (the treatment)

A single map from the observed history window to features $\Phi$, with the value a
**linear functional** $V=\theta^\top\Phi$ (see §4):

1. **Markovian** — $\Phi_{\mathrm{mk}}(x_t) = x(t)$ (current state only); for the
   quadratic baseline, the monomials $x_i x_j$.
2. **Raw history** — $\Phi_{\mathrm{raw}}(x_t)$ = the discretised history
   $(x(t), x(t-\delta), \dots, x(t-h))$ and, for degree $m$, its monomials up to
   degree $m$ (so $\operatorname{span}\Phi$ contains the degree-$m$ polynomial
   functionals of the path).
3. **Signature** — $\Phi_{\mathrm{sig}}(x_t) = S^m(\hat x_t)$, the depth-$m$
   signature of the Arribas-augmented path $\hat x_t = (t,\, x_t,\, (x_0/T)\,t)$
   (`src/utils/dynamic_signature.py`).

### 2.2 Algorithm backbone (held fixed — not a factor)

The **value-gradient (critic-only)** agent (Doya / `value_gradient_jax.py`) is the
fixed learning backbone for H1/H2. Rationale: it is the most faithful instantiation
of Arribas — the *value* is a functional of the history, a **linear-in-$\Phi$
critic** instantiates Thm 4.2 directly, and the control follows in closed form from
$\partial V/\partial x$ (Doya), so there is no separate actor to confound the
comparison. The actor-only vs actor-critic comparison is a **separate, orthogonal
study** (different estimator bias/variance) run with the representation held fixed;
it is *not* crossed with H1/H2.

#### 2.2.1 The control law as a vertical (functional) derivative

The value is a *functional* of the history segment $x_t\in C[-h,0]$, so "$\partial
V/\partial x(t)$" in Doya's control law must be read as the **vertical (Dupire)
functional derivative** $\partial_x V$ — the derivative w.r.t. the *current value*
$\psi(0)=x(t)$ holding the past fixed. This is the object the delayed-HJB requires:
the control enters the dynamics only through the instantaneous rate of the current
state ($\dot x(t)=A\psi(0)+A_1\psi(-h)+Bu$), so the only $u$-dependent term in the
HJB operator $L_uV$ (Kolmanovskii §6.1, eq. 1.4) is $\langle \partial_x V, Bu\rangle$,
and stationarity in $u$ (with control cost $u^\top R u$) gives

$$u^\star = -\tfrac{1}{2}R^{-1}B^\top\,\partial_x V(x_t).$$

Crucially the control still depends on the whole history, because $V$ *couples* the
present with the past. For the linear-quadratic Bellman functional (Kolmanovskii eq.
2.4), $\partial_x V = 2P\,\psi(0) + 2\int_{-h}^0 Q(\theta)\psi(\theta)\,d\theta$,
recovering eq. 2.7: the cross-kernel $Q$ falls out of the *endpoint* derivative.

> **TODO (notation convention — value functional 2.4).** When eq. 2.4 is written
> out in full here, fix and document the cross-term convention. The value is the
> quadratic form of a self-adjoint operator on $M_2=\mathbb{R}^n\times
> L^2([-h,0];\mathbb{R}^n)$ with symmetric block
> $\begin{psmallmatrix}P & Q\\ Q^{*} & R\end{psmallmatrix}$. Its two off-diagonal
> blocks $(1,2)$ and $(2,1)$ are mutual transposes; since each cross term is a
> scalar, $\psi(0)^\top Q\psi(\theta)=\psi(\theta)^\top Q^\top\psi(0)$, so the two
> add to $2\,\psi(0)^\top\!\int_{-h}^0 Q(t,\theta)\psi(\theta)\,d\theta$.
> Kolmanovskii lists the term **and its transpose separately** (no explicit 2);
> the factor-2 single-term form used at line 84 above is the collapsed equivalent.
> The "2" is exactly what makes $\partial_{\psi(0)}V=2[P\psi(0)+\int Q\psi]$ cancel
> the $\tfrac12$ in eq. 2.7 to give a unit-coefficient control law. Pick **one**
> convention and carry it consistently into the kernel Riccati system (2.5)–(2.6):
> with the factor-2 form $Q$ appears once and must not be double-counted; with the
> split form each off-diagonal block contributes. The only genuine error is mixing
> them (keeping both transposed terms *and* a factor of 2). Verify against the book
> typography directly — the Kolmanovskii–Myshkis PDF is not yet in
> `documents/references/`, so this note rests on the algebraic identity, not a
> direct read of eq. 2.4.

In the implementation, with the value a linear functional $V=\theta^\top\Phi(x_t)$ and
the window holding the discretised path ($\text{window}[-1]=x(t)$), the control gradient is the
**discrete vertical derivative** $\theta^\top\,\partial\Phi/\partial\,\text{window}[-1]$
(perturb the last sample, hold the earlier ones fixed) — which is what
`value_gradient_jax` already takes as `grad_path[-1]`. Because $\Phi$ contains
present–past cross-terms (degree-$\ge 2$ raw-history products, or the signature's
iterated integrals that mix the endpoint with the path), this endpoint derivative
carries the history — the discrete image of the $\int Q\psi$ term. This makes the H1
mechanism exact: the Markovian representation has $\partial\Phi/\partial\,\text{history}=0$
(no cross-kernel), so its $\partial_x V$ *structurally cannot* depend on the past,
whereas the raw-history and signature representations have non-zero history
sensitivity (verified numerically in `test/representations/`).

*Caveat (claim strength).* This is a finite-dimensional surrogate of the vertical
functional derivative; its convergence to the exact $\partial_x V$ as $dt\to 0$
depends on the regularity of $V$ (Lipschitz kernels) and the quadrature of the
$\int Q\psi$ term, and is **not proven here** — only the existence, finiteness, and
non-vanishing of the surrogate are verified (the prerequisite for a well-defined,
non-trivial control).

### 2.3 Environment factor (the regime, with predicted outcomes)

Environments span a `{Markovian, delayed} × {linear, nonlinear}` grid. The
predicted outcome in each cell is a **falsification condition**: an outcome
contrary to the prediction signals a confound or an implementation error, not
support for the hypothesis.

| Cell | Environment (example) | Optimal control is… | H1 prediction | H2 prediction |
|---|---|---|---|---|
| Markovian, nonlinear | non-delayed nonlinear oscillator | function of $x(t)$ | **null** (history must NOT help — negative control) | n/a |
| Linear delayed | linear DDE (delayed-LQR; `delay_jax`) | *linear* functional of $x_t$ (Kolmanovskii 2.7) | **positive** | **null** (signature must NOT beat raw history — control) |
| Nonlinear delayed (primary) | well-conditioned controlled Mackey–Glass / bounded nonlinear delayed oscillator | *nonlinear* functional of $x_t$ | **positive** | **positive** |
| Nonlinear delayed, stiff (stress) | Dadebo CSTR (`chemical_process`) | *nonlinear* functional of $x_t$ | **positive** | robustness only |

**Why the linear delayed cell is a control, not a treatment.** Kolmanovskii §6.2
gives, for linear delayed dynamics with quadratic cost, a value that is a
*quadratic* functional of the history (eq. 2.4) and an optimal control that is a
*linear* functional of the history (eq. 2.7):
$$u^\star(t,x_t) = -N_1^{-1} B^\top\!\Big[P(t)\,x(t) + \int_{-h}^0 Q(t,\theta)\,x(t+\theta)\,d\theta\Big].$$
A value taken as a linear/quadratic functional of the *raw* discretised history
already represents these exactly, so the signature's nonlinear iterated-integral
features are not needed: **H2 is predicted null here.** If the signature wins on a
linear delayed plant, suspect over-parameterisation, capacity mismatch, or leakage.

**Why the signature can win on the nonlinear delayed cell.** By Arribas Thm 4.2,
linear functionals of the signature of the augmented path are dense in the
*continuous (nonlinear) functionals* of the path, whereas a linear functional of the
raw history spans only linear functionals (degree-$m$ polynomial at degree $m$).
When the optimal control is a genuinely nonlinear functional of the history (the
nonlinear delayed cell), a linear functional of the signature can approximate it and
a linear functional of the raw history cannot — at matched depth/degree. **This cell
is where the study lives** and must therefore be the best-conditioned environment
available (the
stiff Dadebo CSTR is demoted to a stress test; a well-conditioned nonlinear
delayed environment is added as the primary H2 cell).

## 3. The validation / oracle ladder

Run in order; do not interpret a learned comparison until the relevant rungs pass.

1. **Analytic oracle.** Closed-form optimal control/value: ordinary LQR for the
   Markovian linear cell, delayed-LQR (§5) for the linear delayed cell. Purpose:
   validate the environment, cost, integrator, and evaluation harness; establish
   the **performance ceiling** (the achievable optimum). Every learned model is
   reported as **normalised sub-optimality relative to this ceiling**, which makes
   the three environments commensurable.
2. **Oracle critic + learned actor** (and dual). Purpose: decompose error into
   representation/critic error vs policy-optimisation error — essential for
   diagnosing the Dadebo instability.
3. **Linear-on-true-features.** Fit the linear/quadratic functional on the *known*
   sufficient features (the discretised history for the linear cell). Purpose:
   confirm the optimiser recovers the oracle when the representation is provably
   adequate. Failure here implicates the optimiser, not the representation.
4. **The representation comparison** (Markovian / raw / signature), now
   interpretable because the ceiling and the optimiser are pinned.

## 4. Fairness protocol (equality of models)

The **only** quantity that varies across the H1/H2 arms is the representation map
$\Phi$. Everything else is held identical:

1. **Hypothesis class** — the value is a linear functional of $\Phi$,
   $V=\theta^\top\Phi$, for every arm (Arribas Thm 4.2 is about *linear* functionals
   of the signature; a nonlinear parameterisation of $\Phi$ would conflate
   "representation" with "nonlinear approximation" and break the theory link).
2. **Capacity — fixed hypothesis class, sweep capacity** (the chosen convention).
   Keep the value linear in $\Phi$ on every arm and **sweep the complexity knob**:
   signature depth $m\in\{1,2,3,4\}$ and raw-history polynomial degree / window
   resolution. Plot performance against the **feature dimension** $\dim\Phi$, so a
   "signature wins" conclusion is read off at *matched* $\dim\Phi$ rather than
   asserted. The depth/degree sweep is also a direct test of Arribas (error should
   fall with $m$ until the target functional's complexity is met, then plateau).
   *Anchor:* the span of the depth-2 signature contains the quadratic path
   functionals — exactly the delayed-LQR value class (Kolmanovskii 2.4) — so on the
   linear cell, depth-2 signature and degree-2 raw history should both match the
   oracle value; this equivalence is an explicit unit test.
3. **Common random numbers.** Same master seed per (variant, seed) cell (shared-seed
   plumbing already in place): identical model init, sampler trajectory, and
   exploration noise across arms. H1/H2 are reported as **paired** differences.
4. **Matched information set.** All non-Markovian arms see the *same* history window
   $h$ at the *same* sampling cadence; only $\Phi$ differs. The Markovian arm sees
   $x(t)$. Equalise the window/cadence first, then vary $\Phi$.
5. **Everything else identical**: optimiser and its $dt$-scaling, exploration
   process and $\sigma$ schedule, episode/step budget, $Q,R$, integrator and $dt$,
   burn-in, best-state restoration.

Precise H2 claim, given the above: *with the value a linear functional of $\Phi$
(matched hypothesis class) at matched capacity,
the signature representation attains lower normalised sub-optimality (or higher
sample-efficiency) than the raw-history representation on nonlinear delayed
systems.*

## 5. Analytic baselines — the delayed-LQR oracle (Kolmanovskii §6.2)

The linear delayed cell has a closed-form optimum, used as the ceiling and as the
fairness anchor (it is also the best linear-on-raw-history controller).

**Recommended implementation — augmented finite-dimensional LQR (option A).**
Discretise the history on the agents' grid $x(t), x(t-\delta), \dots, x(t-h)$ ($K$
taps), stack into $\xi\in\mathbb R^{nK}$, write the method-of-steps linear map
$\xi_{k+1} = \mathcal A\,\xi_k + \mathcal B\,u_k$, and solve the standard
discrete-time Riccati equation. The resulting gain is a linear functional of the
discretised history — a discrete form of Kolmanovskii (2.7) — and, by construction,
the optimal linear-on-raw-history controller (the fairness anchor).

**Alternatives** (more faithful, more work): the §6.2.2 exact solution for
$G=0,N_2=0$ via the fundamental matrix $Z$ ($\dot Z = AZ$, $Z(0)=I$), $B_2$ (method
of steps), and the $P_1$ Riccati ODE (eqs. 2.8–2.9); or the infinite-horizon
stationary kernels for constant $A,A_1,B,N$.

**Tests of the oracle implementation** (independent, increasing strength):
- *No-delay limit:* as $A_1\to 0$, $Q,R\to 0$, $P\to$ the ordinary Riccati solution,
  $u^\star\to -N_1^{-1}B^\top P x$ (continuity unit test).
- *Bellman residual:* substitute the value (2.4) and control (2.7) into the
  optimality identity and check it is $\approx 0$ on sampled trajectories.
- *Grid convergence:* option A's closed-loop cost converges to the continuous
  solution as $K\to\infty$ ($\delta\to 0$), monotonically.
- *Independent optimiser:* agree with a direct-transcription numerical optimal
  control on held-out initial histories, to solver tolerance.
- *Closed-loop stability:* the closed-loop characteristic roots lie in the left
  half-plane.

## 6. Training-stability protocol (Dadebo CSTR)

The CSTR is the least well-conditioned cell (stiff Arrhenius terms, 4-D,
state-dependent gain $B(x)$). To prevent it from confounding H2:
- It is a **stress/robustness** cell, **not** primary H2 evidence; a
  well-conditioned nonlinear delayed environment carries the primary claim.
- Use the oracle ladder (rung 2) to separate optimisation pathology from
  representation: if oracle-critic + learned-actor is also unstable, the problem is
  conditioning/optimisation, not the signature.
- Condition the problem: state/reward normalisation; sub-step reward integration
  (not the end-of-step point sample); target network + Polyak (already present in
  the value-gradient agent); opt-in global-norm gradient clipping **enabled only for
  this env and reported**. Consider a smaller $dt$ / stiffer integrator for Dadebo
  and verify via the grid-convergence check.

## 7. Metrics and analysis

- **Primary metric:** normalised sub-optimality $\;(J_{\text{agent}} -
  J_{\text{oracle}}) / |J_{\text{oracle}}|\;$ on a fixed held-out set of initial
  histories (commensurable across environments).
- **Secondary:** sample-efficiency (sub-optimality vs training budget), final state
  error, control effort.
- **Statistics:** explicit seed axis on top of every (variant, env) cell; H1/H2 as
  **paired** differences under common random numbers; report effect sizes with
  confidence intervals, not just point estimates.
- **Replot contract:** every figure regenerable from saved metrics (the
  `replot=<run_dir>` path), so the analysis is auditable without re-running.

### 7.1 Decomposing the residual on exact-capacity arms (the optimization floor)

The value function is parameterised as a **linear functional of the feature map**,
$V_\theta(x_t)=\theta^\top\Phi(x_t)$. On the linear delayed cell, the degree-2
monomials of the discretised history and the depth-2 path signature both span the
**quadratic functionals of the history segment**, and the delayed-LQR value is such a
quadratic functional (Kolmanovskii eq. 2.4; the depth-2-signature ≈
degree-2-raw-history equivalence is an explicit unit test, §4). Hence for these
*exact-capacity* arms $V^\star\in\operatorname{span}\Phi$, the approximation error
$\inf_\theta\lVert V_\theta-V^\star\rVert=0$ **by construction**, and the residual
normalised sub-optimality $\rho=(J(\hat\theta)-J^\star)/|J^\star|$ is **optimization
error**, not approximation error. It is the sum of three sub-terms:

1. **Projected semi-gradient TD fixed-point bias.** Under `semi_gradient: true` the
   parameter converges to the solution of a *projected* Bellman equation (the
   semi-gradient TD fixed point), which need not equal the true value parameter
   $\theta^\star$ even when $V^\star\in\operatorname{span}\Phi$. A systematic bias,
   independent of optimisation budget.
2. **Finite-budget non-convergence.** With $N$ episodes and step size $\eta$
   (`critic_lr`), $\hat\theta_N$ is an iterate, not the fixed point.
3. **Policy-improvement error.** The control is the greedy law
   $u=-\tfrac12 R^{-1}B^\top\,\partial_x V_{\hat\theta}$; an error in the vertical
   derivative $\partial_x V_{\hat\theta}$ yields a control law different from the
   oracle's history-feedback law $u^\star$ (Kolmanovskii 2.7).

**Empirical findings** (`run/study/diagnose_optimization_floor.py`; one seed of each
arm on `delayed_velocity_oscillator`; claim strength: measured). The diagnostic rolls
the delayed-LQR oracle through the *same* `collect_evaluation_data` path and reports:

- **Representation is adequate (confirmed).** The agent's control is *exactly* affine
  in the history window ($R^2=1.000$ for both exact-capacity arms), as a quadratic
  $V_\theta$ requires. The under-capacity contrast — degree-1 raw history, whose
  *linear* $V_\theta$ has a state-independent vertical derivative — produces a
  near-constant control ($|u|\approx0.02$, no state feedback): a representation
  failure ($\rho=1.31$), cleanly separated from the optimization floor.
- **The residual is transient-localised (confirmed).** $95$–$102\%$ of the cumulative
  excess $E(t)=J(\hat\theta){-}J^\star$ accrues before the oracle's settling time; the
  tail is neutral.
- **It is not a magnitude error.** raw-history deg-2 matches the oracle control
  amplitude ($\max|u|$ $3.70$ vs $3.53$); the depth-2 signature *under-actuates*
  ($1.84$ vs $3.53$) yet attains lower cost ($\rho=0.056$ vs $0.119$).
- **It is a feedback-law error.** Applying the oracle law $K^\star$ to the agent's
  *own* windows and comparing to the agent's realised control gives an RMS
  discrepancy of $0.65$–$0.73$ (relative to the oracle control RMS) — concentrated in
  the transient. So sub-terms (1)/(3) (a biased fixed point and the induced
  policy-improvement error), not finite budget (2), dominate.

**Methodological caveats.** (i) The multi-tap feedback *kernel* is **not identifiable**
from a single closed-loop trajectory: the $dt$-spaced taps are near-collinear (design
condition number $\sim10^4$–$10^5$), so a recovered $\hat K$ is the inflated min-norm
solution and $\lVert\hat K-K^\star\rVert$ is meaningless — the identifiable quantity is
the on-trajectory action discrepancy above. (ii) The forward-Euler oracle is
marginally conservative on the RK4 plant (the depth-2 signature beats it in the tail
while under-actuating), so the action discrepancy is an *upper bound* on the agent's
law error.

**Implication for the floor.** Because the dominant error is the feedback law and the
biased fixed point — not finite budget — the productive levers are those that change
the learned $V$'s vertical derivative: the target network / Polyak coefficient
(`tau_polyak`) and a *less biased TD target* (integral / $n$-step TD, or an off-policy
LSTD fit on the exact features), rather than `critic_lr` / episode count alone.

## 8. Correctness safeguards

- **Oracles as tests:** unit tests asserting oracle sub-optimality $\approx 0$, the
  no-delay Riccati limit, and the depth-2-signature ≈ quadratic-raw-history value
  equivalence on the linear cell.
- **Provenance:** each run's `run_context.yaml` ties the numbers to the exact
  command and commit; `data/main_unified/<experiment_group>/` groups an ablation.
- **Representation correctness:** the signature augmentation matches Arribas exactly
  (Phase-1 fix) and runs in float64 (removes a precision confound for the iterated
  integrals) — both load-bearing for a study about a signature method.

## 9. Design matrix → job arrays

The matrix `representation × environment × (depth/degree) × seed` is encoded as a
variants file (`run/study/representation_study_variants.txt`) for
`experiment_array_launcher.sh` (one task per cell), submitted once per seed reusing
the same `--experiment-group` so all seeds accumulate under
`data/main_unified/<experiment_group>/`. The finalize/aggregation step
(`run/study/aggregate_representation_study.py`) discovers the runs, computes the
delayed-LQR oracle ceiling for the linear cells, and writes the normalised
sub-optimality (with 95% CIs across seeds), a `summary.yaml`, a machine-readable
`aggregation_data.json` (so the figure regenerates without recomputation), and the
`comparison` figure into the group directory — run it on the non-billed `prepost`
partition via the launcher's `--finalize`.

## 10. Open items

- Choice of the well-conditioned nonlinear delayed environment for the primary H2
  cell (controlled Mackey–Glass vs a bounded nonlinear delayed oscillator).
- Whether to use the augmented-LQR oracle (option A) or the §6.2.2 exact $Z,B_2,P_1$
  construction for the delayed-LQR ceiling.
- The orthogonal actor-only vs actor-critic study (separate from H1/H2).
