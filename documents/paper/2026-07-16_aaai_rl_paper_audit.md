# AAAI RL-first paper — audit, theory route, and evidence inventory

*Status: report of the 2026-07-16 session. Records (i) the decisions taken with the author,
(ii) the audited defects of the AAAI draft, (iii) the recommended theory route and notation,
(iv) the measured evidence inventory. Companion to `aaai_paper_plan.md` (v0, 2026-07-02), which
covers the **oracle half**; this document covers the **RL half**. Neither the oracle/H1--H2
documents nor the pre-registration are modified by this report.*

---

## 0. Decisions taken (2026-07-16)

| Decision | Choice |
|---|---|
| Split | **Two papers.** Paper 1 (AAAI) = the RL results. Paper 2 = oracle / H1--H2 / conditioning. |
| Paper 1 scope | RL half only: **no oracle, no off-sheet axis, no conditioning statistics.** |
| Voice | **Impersonal**, per the project register rule. AAAI prescribes nothing on voice (verified: `aaai2027.sty` and the author kit are silent) — the earlier claim that "AAAI convention is *we*" was descriptive, asserted without checking, and does not oppose the project rule. |
| Notation | Align to the thesis master lexicon (Section 3). |
| Functional HJB | Stated in the core paper, proved in the appendix, adapted to the delay case on Lipschitz paths. |
| Actor-critic | **Campaign to be run** (it does not currently exist for raw-history — Section 4). |

### The one constraint carried over from the oracle half

Paper 1 must **not claim a representational advantage** of the signature. The oracle half's
pre-registered 5-seed off-sheet control refutes that claim, and it is the subject of Paper 2.
Paper 1 may report measured RL performance ("the signature critic attains lower closed-loop cost
than the degree-two history critic on cells X, Y"); it may not assert that the signature is a
richer representation of history. This costs one sentence of discipline, not a section.

Rationale for the scope being legitimate rather than selective: the off-sheet control is an
artefact of the **oracle half** (supervised least-squares fit to $V^\star$ with an $n_{\mathrm{off}}$
budget parameter). The RL half has no off-sheet budget parameter — data arrive from rollouts plus
exploration — so the $\rho_{\mathrm{off}} \ge 2$ crossing does not transfer mechanically. A paper
reporting what the agent does is not concealing a refutation of a claim it does not make.

---

## 1. Draft defects, ranked (`paper_draft_aaai_template/AnonymousSubmission2027.tex`)

Verified against the file at Overleaf commit `fe5da43`.

### Blocking

1. **Sign error in the main theorem.** Under the draft's own convention
   ($r = x^\top Q x + u^\top R u$, $R$ negative definite, supremum), $\partial_u(u^\top R u) = +2Ru$,
   so the stationarity condition is $2Ru + g^\top \nabla V = 0$ and
   $u^\star = -\tfrac{1}{2}R^{-1}g^\top\nabla V$. The draft's proof line reads
   $-2Ru + g^\top\nabla V = 0$ and concludes $u^\star = +\tfrac{1}{2}R^{-1}g^\top\nabla V$.
   Scalar check at $d = m = 1$, $g = 1$, $R = -1$: the objective $-u^2 + pu$ maximises at $u = p/2$;
   the draft's formula returns $-p/2$, the **anti-optimal** control. The implementation
   (`value_gradient_jax.py:199-205`) is correct; only the paper is wrong. The added
   Kolmanovskii property uses the opposite convention ($Q,R$ positive, minimisation), so the
   document currently carries two incompatible conventions.
2. **Type error on the central object.** `Theorem [Functional HJB]` types
   $V^\star : [-h,0] \to \mathbb{R}$ and $u^\star : [-h,0] \to \mathbb{R}^p$. A map $[-h,0]\to\mathbb{R}$
   is a function of one real variable, not a functional on a path space, and cannot be evaluated at
   $x_t$ — which the next line does. Correct: $V^\star : \mathrm{Lip}([-\tau,0],\mathbb{R}^{d_X}) \to \mathbb{R}$.
3. **The augmentation breaks the central lemma — measured, not conjectured.** The augmentation
   $\tilde x : s \mapsto (x(s), s, x(a)s/(b-a))$ has $a = t - \tau$, so the ramp channel's slope is
   *redrawn* as the window slides; $\tilde x_{t+\Delta t}$ is not a sub-path concatenation of
   $\tilde x_t$ and Chen's identity does not apply. Numerical test ($x(t)=\sin 2t + 0.3t$, $\tau=1$,
   $t=0.4$, $D=3$, $N=3$, 4000 nodes, $\Delta t = 10^{-5}$): residual **1.14, not converging**,
   against $6.2\times10^{-5}$ (converging) on the un-augmented window. Word-by-word the failure is
   perfectly structured — every word over the state and time channels passes at $10^{-5}$; **every
   word containing the ramp channel fails**, and at word $(2,)$ the residual $1.0247$ equals
   $\dot x(t-\tau) = 1.0247$ to five digits. The draft's sentence "the augmentation merely appends
   smooth coordinates; it does not affect the algebraic structure of the proof" is false.
   *Fixes (both needed):* (i) use the fixed window length $\tau$ as denominator; (ii) drop the
   sliding-window lemma from the main text and prove the theorem from the tip derivative alone —
   the tip derivative **does** survive (verified, residual $1.2\times10^{-6}$), because tip
   extension does not move the window's left endpoint, so the ramp slope is frozen and Chen applies.

### High

4. **State space.** $\mathcal{C}^1$ is untenable on three counts: measurable controls give $\dot x$
   only almost everywhere; the tip extension has a corner and **leaves $\mathcal{C}^1$**, so the
   draft's own key limit is formally undefined; and $\mathcal{C}^1$-compactness requires a bounded
   control set, contradicting the unconstrained argmax of the value-gradient theorem (a circularity
   between the universality theorem and the greedy-control theorem). Use
   $\mathrm{Lip}([-\tau,0],\mathbb{R}^{d_X})$, with $\mathcal{K}$ compact in the **1-variation
   topology**. Justification is intrinsic: the plant's own solutions under measurable bounded
   control are Lipschitz and no better. The signature requires finite $p$-variation with $p<2$ and
   is **not** defined on all of $C([-\tau,0],\mathbb{R}^{d_X})$ — continuity alone is insufficient.
5. **Augmented-path dimension.** $(x(s), s, x(a)s/(b-a))$ has $d + 1 + d = 2d+1$ components, not
   $d+2$ ($d+2 = 2d+1$ only at $d=1$). Fixed at `fe5da43`; every downstream dimension count must be
   re-derived.
6. **"Continuous Actor-Critic methods are well-known to be unstable"** — uncited (violating the
   practice-claim rule), unsupported by the bibliography, and **contradicted by the repository's own
   oracle ladder**, which measures the full learned actor-critic converging at $J = 2.82 \pm 0.08$
   (3 seeds) with the residual dominated by policy/gradient extraction (replacing the actor:
   $2.82 \to 1.83$; replacing the critic: $2.82 \to 2.56$) — the $\ker G$ gap, not instability.
7. **Window sign inconsistency.** Line 250 defines $x_t : s \mapsto x(t-s)$, $s\in[-h,0]$ — the
   *future* window $[t, t+h]$. Lines 286/303/469 correctly use $x(t+s)$.
8. **"Cauchy-Schwartz theorem"** for ODE uniqueness: it is Cauchy–Lipschitz (Picard–Lindelöf);
   Cauchy–Schwarz is an inner-product inequality, and is misspelt. Continuity gives Peano existence,
   not uniqueness — contradicted by the draft's own next theorem, which assumes Lipschitz.
9. **The Riesz step is unnecessary and incorrect.** $v \mapsto \langle\phi, S(x_t)\otimes v\rangle$
   is a linear form on the finite-dimensional $\mathbb{R}^d$; no Riesz theorem and no path Hilbert
   space is needed. $\mathcal{C}^1$ on a compact interval is Banach, not Hilbert; "the Hilbert space
   of piecewise $\mathcal{C}^1$ paths" does not exist as written.
10. **The $C^0$-to-$C^1$ gap.** The universality theorem gives sup-norm approximation of *values*;
    the control reads $V$ entirely through $D\Phi^\top\nabla\Psi$. Uniform closeness of two
    functionals implies nothing about their vertical derivatives. This is the group's own settled
    finding (the $L^2$ value fit leaves $\partial_x V$ unidentified on $\ker G$). The guarantee must
    be stated **conditionally on HJB-residual smallness**, never as "good value fit $\Rightarrow$
    near-optimal control".

### Mechanical

11. Undefined `\ref{eq:HJB}`, `\ref{eq:HJB-sig}`, `\ref{app:ascoli}`; `\label` after an unnumbered
    `\section` makes `\ref{sec:full_proof_vg}` print a wrong number with no LaTeX warning;
    bibliography path breaks the build from the file's own directory (all 23 citations render `[?]`);
    hard-coded `\tag{4}`; text-mode `sup_{u \in \mathcal{U}}` (missing `\sup`); "Linear Quadratic
    **Regressor**" → *Regulator*; "**Delayed** Differential Equation" → *Delay Differential Equation*;
    abstract is `(TO BE WRITTEN)`; Experiments/Discussion/Conclusion empty; "Estimating the Value
    Function" is an empty subsection; zero `\includegraphics` despite `Figures/figure{1,2}.pdf`
    existing; the anonymous submission lists author names.
12. **~16 symbol collisions**, traceable to the absent macro layer: $\phi$ (weight tensor / initial
    history), $\theta$ (weight tensor / integration variable / thesis weights), $N$ (truncation level
    / Riccati kernel), $n$ (word length / output dimension / number of delays), $h$ (delay / initial
    time), and others. `:=` used twice (proscribed); `\begin{proof}[Proof sketch]` (proscribed);
    ~25 first-person occurrences.

---

## 2. Theory route

**The functional HJB is not a contribution — it is a citation.** It dates to Vinter & Kwong (1981);
the draft itself says its theorem is "adapted from Kolmanovskii". Placing it in a contributions list
invites a correct rejection.

**The key identification** (the draft's own `.bib` contains `dupireFunctionalItoCalculus2019` and the
body cites it **zero** times; collected-and-not-engaged reads worse than ignorance):

- $L_u V(x_t)$ **is** Dupire's total functional derivative (deterministic, bounded-variation case);
  equivalently the Driver–Krasovskii derivative along a retarded FDE, equivalently the coinvariant
  (ci-) total derivative of Kim/Lukoyanov.
- $\nabla_{x(t)}V(x_t)$ **is** Dupire's vertical derivative — precisely, the $v$-**linear part** of
  the tip derivative, which is **affine**, not linear:
  $S(\tilde x_t)\otimes(v,1,c) = \underbrace{S(\tilde x_t)\otimes(0,1,c)}_{\text{horizontal}}
  + \underbrace{S(\tilde x_t)\otimes(v,0,0)}_{\text{vertical}\cdot v}$.
- $V(x_t) = \langle\phi, S^N(\tilde x_t)\rangle$ is a **truncated functional Taylor expansion** with
  $\phi_\alpha = \Delta_\alpha V$ (Dupire & Tissot-Daguette, *Functional Expansions*,
  arXiv:2212.13628, Theorem 3.10) — strictly stronger than a density statement.

**Recommended route:** Lukoyanov's coinvariant framework as the cited setting (deterministic,
first-order, retarded-FDE-specific, sliding window, Lipschitz paths — every structural choice of the
draft is already a hypothesis there); Dupire–Tissot-Daguette for the $\nabla_{x(t)}V$ identification;
Kolmanovskii's Krasovskii derivative for the proof; the $M^2$ semigroup lift **only** for the
linear-quadratic Riccati oracle. The $M^2$ route is not merely inconvenient but *incompatible*:
iterated integrals are not continuous for the $L^2$ topology, so $S(x_t)$ is not well defined there.

**Deliverable = a verification theorem**, not a well-posedness theory. Fifteen lines: chain rule →
differential inequality → transversality. Transversality and attainment are **real hypotheses**, not
formalities, and the draft omits both.

**The strongest theoretical sentence available:** $V^\star$ is generically only locally Lipschitz on
nonlinear delay plants and is characterised solely as a minimax (equivalently viscosity) solution —
but **every member of the hypothesis class is ci-smooth with closed-form derivatives**, so classical
verification applies to the ansatz precisely where it fails for the true value. Two citations.

### Recommended structural move: state the theorem representation-agnostically

For any feature map $\Phi$ with an affine tip derivative and any $\Psi \in C^1$, with $V = \Psi\circ\Phi$:

$$u^\star = -\tfrac{1}{2}R^{-1} g(x_t)^\top D\Phi(x_t)^\top \nabla\Psi(\Phi(x_t))$$

Markovian features recover Doya (2000) exactly; raw-history gives the gradient with respect to the
most recent collocation node only; the signature gives $D\Phi(x_t)v = S(\tilde x_t)\otimes\iota(v)$.
The three baselines become instantiations of one theorem. The $(D\Phi,\nabla\Psi)$ factorisation also
means a **neural head on the signature costs nothing theoretically** — one autodiff call, no new
theorem. The code already demonstrates the point: `value_gradient_jax.py:199-205` takes
`grad_path[-1]` by generic `jax.grad`; nothing in it knows it is differentiating a signature.

### On polynomial heads (asked, and answered in the negative)

The shuffle identity gives a two-sided sandwich, both strict:
$\mathcal{L}_N \subsetneq \mathcal{P}_{p,N} \subsetneq \mathcal{L}_{pN}$.
A polynomial head **is** richer than the draft's fixed-$N$ linear model, but adds nothing beyond
*deepening the truncation* — the $D^N$ cost the signature was adopted to avoid. By Radford's theorem
(the shuffle algebra is a polynomial algebra on Lyndon words), $\bigcup_p \mathcal{P}_{p,N}
\subsetneq \bigcup_M \mathcal{L}_M$: **unbounded degree does not close the gap**. Measured span
saturates beyond $p=2$, with deficit equal to the number of Lyndon words of length $>N$.
*Recommendation:* omit; if mentioned, one remark citing Ree/Reutenauer — not a boxed proposition.

### Prior art absent from `aaai2027.bib`

Kalsi, Lyons & Perez Arribas (SIAM J. Financial Math. 11(2):470–493, 2020) already parametrise the
value as a linear signature functional. Cohen et al., *Exponentially Fading Memory Signature*
(arXiv:2507.03700, July 2025), Equation 3.32, gives $\mathrm{d}X^\lambda = -\Lambda X^\lambda\,
\mathrm{d}t + X^\lambda \otimes \mathrm{d}X$ — the identical injection-plus-forgetting structure —
and Drobac et al. (arXiv:2510.12337) publish the discrete sliding-window update by the draft's exact
mechanism. **The draft's lemma is the $\Delta t \to 0$ limit of a published algorithm.** None of
Kalsi, Bank–Riedel, Abi Jaber, Lukoyanov, Vinter, Drobac is in the `.bib`; Ohnishi, Dupire, Holt and
Hambly are in it and cited zero times.

**Ohnishi et al. 2024** is safe as motivation but not as the plan states it: its plant is **Markovian
by explicit assumption**; what is path-dependent is the **cost** (trajectory following). The present
work is its exact complement — instantaneous cost, path-dependent **plant**. Its robustness claim is
empirical (no theorem) and concerns rollout error under model misspecification, **not** stability
with respect to a Markovian approximation of the state. Note also it appeared at **L4DC 2024**, the
venue targeted for Paper 2.

---

## 3. Notation — aligned to the thesis master lexicon

Source: `~/Documents/work/phd/latex_documents/thesis/2024_07_06_phd_thesis_hosseinkhan/utils/new_commands_maths.tex` (108 macros).

| Object | Macro | Glyph |
|---|---|---|
| State point / process | `\statePoint` / `\stateProcess` | $x$ / $X$ |
| Control point / process | `\controlPoint` / `\controlProcess` | $u$ / $U$ |
| History process / space | `\historyProcess` / `\historySpace` | $H$ / $\mathscr{H}$ |
| **Time delay** | `\timeDelayPoint` | $\tau$ |
| Continuous time | `\continuousTimePoint` | $t$ |
| Objective | `\objectiveFunction` | $J$ |
| Discount factor | `\discountFactor` | $\gamma$ |
| State / control dimension | `\stateDimension` / `\controlDimension` | $d_X$ / $d_U$ |
| State / control space | `\stateSpace` / `\controlSpace` | $\mathcal{X}$ / $\mathcal{U}$ |
| Admissible controls | `\admissibleControlSpace` | $\mathscr{A}_{\mathcal{U}}$ |
| Weights / weight space | `\weights` / `\weightSpace` | $\theta$ / $\Theta$ |

### The live collision, and the resolution

The thesis sets $\tau$ = **delay**. The draft sets $\tau$ = **discount time constant** and $h$ =
delay — and the thesis has `\historyPoint` = $h$, a third meaning. The **code sides with the
thesis**: `HOPFIELD_TAU`, MG `tau=6`, "the tau sweep", `delay: 0.5`. So the same glyph means delay in
the repository and discount in the paper. Doya is the source of the draft's usage; Doya is not the
project lexicon.

**Resolution:** $\tau$ = delay (thesis + code + DDE literature agree); the discount is written as a
**rate** $\gamma \in (0,+\infty)$, reusing the thesis glyph, so the HJB reads
$\gamma V^\star(\xi) = \sup\{\cdots\}$ rather than $\tfrac{1}{\tau}V^\star(\xi)$. Kills the collision,
reuses two thesis macros, costs one preamble line.

**Build the macro layer before any rename pass.** A body-wide textual substitution on raw glyphs is
exactly the operation that previously corrupted `A = -I` into `A = -J`.

---

## 4. Evidence inventory (measured, 2026-07-16)

### The RL half — what exists

| Artefact | Location | Seeds | Representations | Status |
|---|---|---|---|---|
| `main_unified/` value-gradient benchmark | local `data/main_unified/` (10 sub-studies); cluster `$WORK/.../data/main_unified/` (19, canonical) | **5** | markovian / raw_history / signature | **usable — the paper's backbone** |
| `linear_cell_study` | local | 5 | 3 reps | usable |
| `delayed_oscillator_high_gap_study` | local | 5 | 3 reps | usable |
| `mackey_glass_limit_cycle_study` | local | 5 | 3 reps | usable |
| `mackey_glass_chaotic_study` | local | 5 | 3 reps | usable |
| `delayed_velocity_study_v2` | local | 5 | 3 reps | usable |
| `budget_sweep_high_gap_{b800,b2400,b7200}` | cluster only | 2 | markovian + raw_history **only** | H1 budget axis; **no signature** |
| `oracle_ladder_high_gap` (actor-critic) | local | 3 | **signature only** | PNG-only, not regenerable |
| `h1h2_rl_lspi/20260617_150750_array` (job 544311) | cluster | — | — | **FAILED — zero data** |

**Caveat carried:** `main_unified` contains **Dadebo, not Hopfield**.

### Why the RL array failed — and the lesson for the AC campaign

```
OSError: [Errno 122] Disk quota exceeded:
  '.../data/h1h2_rl_lspi/20260617_150750_array/hopfield_duffing_seed3'
```

Job 544311 wrote to **`$WORK`**, hit the inode quota, and produced 80 slurm log files and zero data
directories. **The AC campaign must write to `$SCRATCH`**
(`/lustre/fsn1/projects/rech/oym/ucd32aq`), not `$WORK`.

Surviving 1-seed `_debug_` arrays (indicative only, **not** CI-backed):

| Cell | markovian $J$ | raw_history $J$ | signature $J$ | H1 | H2 |
|---|---|---|---|---|---|
| `linear_dde` | 0.4232 | 0.1809 | 0.1277 | holds | holds |
| `hopfield_linear` | 0.1343 | 0.1082 | 0.1110 | holds | **fails** |
| `hopfield_duffing` | 0.1467 | 0.1177 | 0.1154 | holds | fails (tie) |

### The actor-critic gap (the campaign to be run)

Verified: `make_representation` is called in **exactly one agent** — `value_gradient_jax.py:87` —
and `kind: signature | raw_history | markovian` appears only in `conf/agent/value_gradient.yaml:46`.
`CTACSignatureJAX` instantiates `SlidingSignatureJAX` directly at `signatures_jax.py:166` and never
touches the factory. **The actor-critic supports the signature only.** Its evidence base is 9 runs,
one cell (`delayed_oscillator_high_gap`), 3 seeds, PNG-only — no `.npz`, `.pkl` or `summary.yaml`,
so its numbers are not regenerable.

**Work required:** port `CTACSignatureJAX` onto `make_representation`, add `kind:` to the
`conf/agent/CTAC_*.yaml` configs, and run 3 representations × 5 seeds across the cells. Route to
`cpu_p1` (JAX-CPU); finalize on `prepost`. Output to `$SCRATCH`. SLURM logs to a `slurm/` subfolder.

### Residual code defect

`value_gradient_jax.py:_fill_buffer_initial` (~line 157-171) appends the initial history as
`dtype=np.float32` despite the float64 pipeline that finding F-E1 was meant to fix (commits
`07b8d4a`, `7b1153e`). The initial-condition path is still narrowed to float32.

---

## 5. Open actions

1. **Run the actor-critic campaign** (Section 4) — to `$SCRATCH`.
2. Fix the draft defects of Section 1, macro layer first (Section 3).
3. Re-derive every reported number from the artefacts; several figures quoted in discussion do not
   match the 5-seed files (H1 margins measured at 58.6 / 63.7 / 22.8 / 40.2 / 99.4 per cell; the
   signature feature dimension is 12 / 30 / 462 per cell, **not** a fixed ~30).
4. Decide whether to rapatriate the 9 cluster-only `main_unified` sub-studies (assessment: not
   load-bearing).
