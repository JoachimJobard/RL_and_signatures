# AAAI paper plan — value-gradient representations for delayed control

*Working document, to be refined. Status: draft v0 (2026-07-02). Grounded in the H1/H2
benchmark report (`documents/reports/2026-06-30_h1h2_representation_benchmark/`) and the
companion note (`../../latex_documents/notes/2026_06_17_signature_density_vs_conditioning/`).*

**Governing principle — no HARKing.** The pre-registration contains **only** the operational
hypotheses $H_1/H_2$ as closed-loop-cost comparisons ($I[\text{raw}]<I[\text{markovian}]$;
$I[\text{sig}]<I[\text{raw}]$). The paper reports the *apparent* $H_2$ result honestly and then the
controls that overturn it; every claim carries its epistemic status (proven / measured /
conjectured).

Two boundaries this imposes:
- **The $H_2^{\mathrm{pop}}$ vs $H_2^{(n)}$ split is post hoc**, formed after observing the collapse.
  It is a standard approximation/estimation (bias/variance) decomposition used to *explain* the
  pre-registered $H_2$ — presented as a **post-hoc interpretation and a conceptual contribution**,
  never as a hypothesis we set out to test. Retro-fitting it into the pre-registration would be
  HARKing and is forbidden.
- **A mathematical statement needs a proof; a "law" is not a theorem.** The identifiability co-rank
  bound is a Proposition (proven); the realised threshold $n_{\mathrm{off}}\approx c\,\dim\phi$,
  $c\approx2$, is an *empirical scaling*, flagged as measured and never placed in a theorem
  environment or called a "law/theorem".

---

## 1. Thesis and positioning

**One-sentence contribution.** For continuous-time value-gradient (Doya) control of delayed
systems, *history* representations robustly help, but the *path-signature structure* confers no
representational advantage over a plain polynomial of the history window; its apparent advantage
is a finite-sample conditioning effect, which we formalise, prove the mechanism of, and verify
under controls.

**Framing.** A negative result with a *proven mechanism* and a *positive re-characterisation*.
Lead with the positive message (*density $\neq$ realisability for control*; *sample efficiency, not
expressivity*); the refutation of $H_2$ is a corollary. Do **not** frame as "signatures are bad".

**Venue.** AAAI main technical track (target). Better-fit alternatives to keep in mind: L4DC,
NeurIPS (main or Datasets & Benchmarks). If AAAI, foreground theory + methodology so it does not
read as a purely empirical negative.

**Reviewer risks and pre-emptions.**
- *"Negative result."* → Lead with the off-sheet-budget scaling (Prop 4) and the
  $H_2^{\mathrm{pop}}/H_2^{(n)}$ separation; the refutation is downstream.
- *"Signatures need channel reduction (Morrill et al. 2021)."* → Cite; our novelty is the
  control-specific gradient / kernel-of-Gram mechanism and the effective-rank bound, absent there.
- *"Oracle labels + plain LS is not real RL."* → The oracle half is the controlled instrument;
  the learned LSPI / capacity-sweep benchmark is the deployment-realistic confirmation. Both shown.
- *"Only these plants."* → Graded suite + two nonlinear families + generic theory.

---

## 2. Contributions (claim list, in order)

1. **Delayed value-gradient control (the enabling framework).** The extension of Doya's
   continuous-time value-gradient law $u=\tfrac12 R^{-1}B^\top\partial_x V$ from ODE (Markov) systems
   to **delayed / non-Markovian** dynamics: the value as a functional of the history (augmented
   state), and the continuous-time delayed oracle — delayed-LQR by Chebyshev collocation of the
   history generator, and a Pontryagin two-point BVP for nonlinear delayed plants. This is what
   makes $H_1/H_2$ askable at all. (Builds on the thesis framework, Jobard 2026; credit explicitly.)
2. **Theory**: effective-rank / Veronese bound; kernel-of-Gram gradient indeterminacy; Lyapunov
   exploration-invariance; the identifiability co-rank bound (proven) + the measured off-sheet
   scaling (empirical). See §3 for the exact claim-strength split.
3. **Empirics**: a confound-controlled, (partly) pre-registered program on a graded suite —
   $H_1$ holds, $H_2$ fails under every control, the scaling verified across two nonlinear families.
4. **Post-hoc formalisation (conceptual contribution, labelled as such).** The
   approximation/estimation decomposition of $H_2$ into $H_2^{\mathrm{pop}}$ (representational) and
   $H_2^{(n)}$ (finite-sample) that explains the collapse. Not a pre-registered hypothesis (§1).
5. **Methodology**: gated-oracle labelling + on/off-sheet design + capacity/budget/conditioning
   controls + pre-registration — a template for representation studies in RL.

---

## 3. Theoretical results (with claim strength explicit)

- **Theorem 1 (attribution).** Density of linear functionals of the signature (Stone–Weierstrass
  via shuffle + Hambly–Lyons). Delimits what density does *not* give (norm, rate, derivative).
- **Proposition 1 (Veronese effective-rank bound, PROVEN).**
  $\operatorname{rank}\operatorname{Cov}(\mathbb S_{\le L}(\underline x^{(z)})) \le \binom{k+L}{L}$,
  independent of the ambient dimension $N(d,L)$. The engine.
- **Proposition 2 (gradient indeterminacy, PROVEN).** With $\ker G_n\neq\{0\}$, $R^2=1$ is
  compatible with an arbitrary deployed control gradient off-manifold.
- **Proposition 3 (Lyapunov scale-invariance, PROVEN).** Isotropic exploration leaves the
  effective rank of a regulated plant unchanged — the degeneracy is structural.
- **Proposition 4 (identifiability co-rank bound, PROVEN).** By rank subadditivity,
  $\operatorname{rank}(G_n)\le r^\star + n_{\mathrm{off}}$ (on-sheet effective rank $r^\star$ plus the
  off-sheet count, since each off-sheet row adds at most one to the rank), so full rank requires
  $n_{\mathrm{off}}\ge\dim\phi-r^\star$. Hence the off-sheet budget for gradient identifiability is
  $O(1)$ for the fixed-dimension signature and $O(\tau^2)$ for degree-two raw-history. *This is the
  theorem; it is the whole of the "law" that is provable.*
- **Empirical finding (measured, NOT a theorem).** The *realised* threshold is
  $n_{\mathrm{off}}\approx c\,\dim\phi$ with $c\approx2$ (the off/dim $\gtrsim2$ transition), beyond
  the bare-rank $c=1$, reflecting the non-genericity of small perturbations and conditioning.
  Reported as a measured scaling — never a theorem.
- **Conjecture 1 (approximation–conditioning trade-off).** Interior optimal depth/degree
  $L^\star(n,d,k)$ from the bias/variance decomposition. Conjecture with empirical support
  (capacity sweep), not a theorem.

*(Optional theory of the framework, Contribution 1: state the delayed value-gradient consistency —
that on the augmented state, Doya's law applied to the collocated/Pontryagin value reproduces the
delayed-optimal control (the report's consistency check, provable at $\varepsilon=0$ where the BVP
reduces to the delayed-LQR). Decide whether to include as a Proposition or leave as construction.)*

---

## 4. Numerical results

**Suite (graded, pre-registered).** markovian (negative control) · linear DDE · delayed Hopfield
{linear, tanh, Duffing} · Mackey–Glass {limit cycle, chaotic} · [$N{=}5$ platoon, high state-dim —
**candidate for cut**, see below]. Dadebo CSTR demoted with the measured kernel-ratio (an honest
null).

**Main-text (5 items):**
1. **Table 1 — $H_1/H_2$ verdicts**, 5 seeds ± CI, three axes ($R^2$, gradient cosine, cost $I$).
2. **Figure 1 — the overturn.** $\tau$-sweep (apparent $H_2$) → three controls (×4 data, ridge,
   off-sheet rebalancing) collapse it; annotate the on/off *balance* sub-finding.
3. **Figure 2 — the mechanism.** Effective-rank-vs-$\tau$ (flat $\approx3.4$ while $\dim$ grows
   $28\times$).
4. **Figure 3 — the off-sheet scaling (empirical).** off/dim $\in\{1,2,4\}$ cost curves for Hopfield
   $\tau{=}3$ and both Mackey–Glass cells: signature flat, raw-history crosses below at
   off/dim $\gtrsim2$. Verifies the co-rank bound (Prop 4) and its measured constant $c\approx2$.
   4b. **Table — no-off-data (a headline sub-result, must be in the paper).** On-sheet-only vs
   on${+}$off closed-loop cost per cell: *with no off-sheet data at all* the signature attains its
   full cost on both Mackey–Glass cells ($J=0.015$–$0.019$) and linear\_dde — exactly where
   raw-history **diverges** ($J=118$–$785$) — yet the signature itself **diverges** on the platoon
   (and Hopfield). Message: the signature is markedly more sample-efficient (over-parameterisation
   ratio ~12 fixed vs raw-history's 83–2180) but **not exempt** from the off-manifold identifiability
   obstruction ($\dim\phi_S>$ on-manifold rank ⇒ $\ker G\neq\{0\}$; stabilising off-manifold gradient
   is the cell-dependent accident of Prop 2). Directly measured; in the report as
   `find:sig-nooff` / `tab:sig-nooff`.
5. **Figure 4 — the high-state-dimension reversal (CANDIDATE FOR CUT).** Platoon controlled
   trajectory: signature sustains an undamped oscillation (cond $\sim10^{14}$, $\gamma{=}0.32$);
   raw-history tracks. *If the platoon is dropped, this figure goes with it and the $O(d^L)$ reversal
   becomes theory-only (Prop 1: signature dimension grows with channels) — the empirical core is
   then the delay-length $O(\tau^2)$ story alone, which is cleaner. The $O(\tau^2)$-vs-$O(d^L)$
   "duality" is then stated as one measured axis (delay) + one theoretical axis (state dimension),
   not two empirical axes.*

**Appendix.** On-sheet-only divergence (kernel-of-Gram, 5-seed); comprehensive *learned*
capacity-sweep benchmark + the $H_1$ under-convergence reconciliation (budget sweep, oracle ladder,
trick ablation); Duffing $\kappa$-sweep; per-cell controlled-trajectory + control-signal gallery;
full reproduction (scripts, run dirs, seeds).

---

## 5. Section-by-section outline (~7 pages + refs)

1. **Introduction** (¾ pg) — belief, control question, pre-registered $H_1/H_2$, one-line result,
   contributions.
2. **Problem setup** (¾ pg) — delayed plant, Doya law, three representations, gated-oracle
   labelling, on/off-sheet data, three metrics.
3. **Hypotheses and protocol** (½ pg) — the pre-registered *operational* $H_1/H_2$
   (closed-loop-cost comparisons only), the suite, the confound controls, the claim-strength policy.
   The $H_2^{\mathrm{pop}}/H_2^{(n)}$ split does **not** appear here (post hoc, item 6).
4. **Theory** (1¼ pg) — Prop 1–4 (sketch proofs) + the empirical off-sheet scaling + Conjecture 1;
   *density $\neq$ realisability for control*.
5. **Results** (2 pg) — $H_1$ (Table 1); the $H_2$ overturn (Fig 1); mechanism (Fig 2); the off-sheet
   scaling (Fig 3); [high-dim reversal (Fig 4) — only if platoon kept]; learned-benchmark
   confirmation (1 para → appendix).
6. **Interpretation (post hoc, labelled)** (½ pg) — the approximation/estimation decomposition of the
   pre-registered $H_2$ into $H_2^{\mathrm{pop}}$ (measured false) and $H_2^{(n)}$ (regime-dependent),
   explaining the flip (report §5.1). Explicitly *not* a pre-registered hypothesis.
7. **Related work** (½ pg) — signatures in ML/RL (Morrill, Kidger, Lyons–Oberhauser), Doya
   continuous-time RL **and its delayed extension**, delayed/POMDP RL, representation-in-RL,
   evaluation rigor / registered reports.
8. **Discussion & limitations** (½ pg) — when history/signature reps help; the $O(\tau^2)$
   (measured) vs $O(d^L)$ (theoretical) axes; practitioner guidance (off-manifold data +
   conditioning > richer algebra); limits (LQR-style oracle, finite suite, platoon scoping).

---

## 6. Title candidates

- "Density is not Realisability: When Path-Signature Representations Help Continuous-Time Control
  of Delayed Systems"
- "History Helps, Structure Does Not: A Controlled Study of Value-Gradient Representations for
  Delayed Control"
- "Sample Efficiency, not Expressivity: Signatures for Delayed Value-Gradient Control"

---

## 7. What is already done vs to run before submission

**Done (in the report):** the full $H_1/H_2$ suite (oracle half, 5-seed canonical table); the
$\tau$-sweep and the three controls (single-seed); Duffing + Mackey–Glass off-sheet (complete);
effective-rank-vs-$\tau$; on-sheet-only 5-seed; the learned capacity-sweep benchmark + robustness
(oracle ladder, trick ablation, budget sweep); the trajectory gallery; the four proven propositions
(companion note); the $H_2$ formalisation (report §5.1).

**To run (make the sweeps CI-backed):**
- Promote the $\tau$-sweep and the three controls to **5 seeds** (the pre-registered off-sheet
  replication, commit `8b1b97b`, covers this — finish it).
- Complete platoon + linear-DDE in the learned benchmark at 5 seeds (currently partial).
- Re-run effective-rank across seeds to put a CI on $\approx3.4$.
- (Optional) Hopfield on-sheet-only at $\tau{=}3$ + the linear arm, for a fuller Table.

---

## 8. Open framing questions to refine

- Lead contribution: the **scaling result** (sample efficiency: Prop 4 + measured $c$) vs the
  **separation** ($H_2^{\mathrm{pop}}$ vs $H_2^{(n)}$, post hoc) vs the **delayed value-gradient
  framework** vs the **methodology**? (Current lean: the framework + theory as spine, the scaling as
  takeaway.)
- How much of the *learned* comprehensive benchmark to keep in main text vs appendix (it carries
  the deployment-realism, but the oracle half is the cleaner instrument).
- Whether to include the interior-optimum conjecture ($L^\star$) at all, or defer to a follow-up.
- Target the *methodology* (pre-registration + confound controls for RL representation studies) as
  a co-equal contribution, or keep it secondary.
- Author-facing: relationship to the companion note (scaffold) — cite as separate, or fold the
  proofs in.

---

## 9. Immediate next artefacts (on request)

- Contributions paragraph + abstract.
- One-page pre-registration document (hypotheses, suite, metrics, decision rules) to lock claims
  before the remaining 5-seed runs.
- Related-work scaffold with the key citations.
