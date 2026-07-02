# AAAI paper plan — value-gradient representations for delayed control

*Working document, to be refined. Status: draft v0 (2026-07-02). Grounded in the H1/H2
benchmark report (`documents/reports/2026-06-30_h1h2_representation_benchmark/`) and the
companion note (`../../latex_documents/notes/2026_06_17_signature_density_vs_conditioning/`).*

**Governing principle — no HARKing.** The hypotheses $H_1/H_2$ are stated as *pre-registered*
(see the pre-registration commit / registry). The paper reports the *apparent* $H_2$ result
honestly and then the controls that overturn it; every claim carries its epistemic status
(proven / measured / conjectured). The one non-proven quantitative element (the constant
$c\approx2$ in the sample-efficiency law) is flagged as empirical, never dressed as a theorem.

---

## 1. Thesis and positioning

**One-sentence contribution.** For continuous-time value-gradient (Doya) control of delayed
systems, *history* representations robustly help, but the *path-signature structure* confers no
representational advantage over a plain polynomial of the history window; its apparent advantage
is a finite-sample conditioning effect, which we formalise, prove the mechanism of, and verify
under controls.

**Framing.** A negative result with a *proven mechanism* and a *positive re-characterisation*.
Lead with the positive law (*density $\neq$ realisability for control*; *sample efficiency, not
expressivity*); the refutation of $H_2$ is a corollary. Do **not** frame as "signatures are bad".

**Venue.** AAAI main technical track (target). Better-fit alternatives to keep in mind: L4DC,
NeurIPS (main or Datasets & Benchmarks). If AAAI, foreground theory + methodology so it does not
read as a purely empirical negative.

**Reviewer risks and pre-emptions.**
- *"Negative result."* → Lead with the sample-efficiency law and the $H_2^{\mathrm{pop}}/H_2^{(n)}$
  separation; the refutation is downstream.
- *"Signatures need channel reduction (Morrill et al. 2021)."* → Cite; our novelty is the
  control-specific gradient / kernel-of-Gram mechanism and the effective-rank law, absent there.
- *"Oracle labels + plain LS is not real RL."* → The oracle half is the controlled instrument;
  the learned LSPI / capacity-sweep benchmark is the deployment-realistic confirmation. Both shown.
- *"Only these plants."* → Graded suite + two nonlinear families + generic theory.

---

## 2. Contributions (claim list, in order)

1. **Formalisation** separating the representational hypothesis $H_2^{\mathrm{pop}}$ from the
   finite-sample $H_2^{(n)}$, with the approximation/estimation decomposition. (Report §5.1.)
2. **Theory**: effective-rank / Veronese bound; kernel-of-Gram gradient indeterminacy; Lyapunov
   exploration-invariance; the sample-efficiency law (off-sheet budget $\propto$ feature dimension).
3. **Empirics**: a confound-controlled, (partly) pre-registered program on a graded suite —
   $H_1$ holds, $H_2$ fails under every control, the law verified across two nonlinear families.
4. **Methodology**: gated-oracle labelling + on/off-sheet design + capacity/budget/conditioning
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
- **Proposition 4 (sample-efficiency law — SPLIT claim strength).**
  *Provable:* co-rank identity $n_{\mathrm{off}} \ge \dim\phi - r^\star$ for full rank, hence
  $n_{\mathrm{off}}^{\mathrm{sig}}=O(1)$ vs $n_{\mathrm{off}}^{\mathrm{raw}}=O(\tau^2)$.
  *Empirical (labelled):* the constant $c\approx2$ and the non-genericity of small perturbations.
  Present as Proposition + separate Empirical finding.
- **Conjecture 1 (approximation–conditioning trade-off).** Interior optimal depth/degree
  $L^\star(n,d,k)$ from the bias/variance decomposition. Conjecture with empirical support
  (capacity sweep), not a theorem.

---

## 4. Numerical results

**Suite (graded, pre-registered).** markovian (negative control) · linear DDE · delayed Hopfield
{linear, tanh, Duffing} · $N{=}5$ platoon (high state-dim) · Mackey–Glass {limit cycle, chaotic}.
Dadebo CSTR demoted with the measured kernel-ratio (an honest null).

**Main-text (5 items):**
1. **Table 1 — $H_1/H_2$ verdicts**, 5 seeds ± CI, three axes ($R^2$, gradient cosine, cost $I$).
2. **Figure 1 — the overturn.** $\tau$-sweep (apparent $H_2$) → three controls (×4 data, ridge,
   off-sheet rebalancing) collapse it; annotate the on/off *balance* sub-finding.
3. **Figure 2 — the mechanism.** Effective-rank-vs-$\tau$ (flat $\approx3.4$ while $\dim$ grows
   $28\times$).
4. **Figure 3 — the sample-efficiency law.** off/dim $\in\{1,2,4\}$ cost curves for Hopfield
   $\tau{=}3$ and both Mackey–Glass cells: signature flat, raw-history crosses below at
   off/dim $\gtrsim2$.
5. **Figure 4 — the high-state-dim reversal.** Platoon controlled trajectory: signature sustains
   an undamped oscillation (cond $\sim10^{14}$, $\gamma{=}0.32$); raw-history tracks.

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
3. **Formalising the question** (½ pg) — $H_2^{\mathrm{pop}}$ vs $H_2^{(n)}$ + decomposition
   (report §5.1, trimmed).
4. **Theory** (1¼ pg) — Prop 1–4 (sketch proofs), the law, Conjecture 1; *density $\neq$
   realisability for control*.
5. **Experimental protocol** (½ pg) — suite, pre-registration, confound controls, claim-strength
   policy.
6. **Results** (2 pg) — $H_1$ (Table 1); overturn (Fig 1); mechanism (Fig 2); the law (Fig 3);
   high-dim reversal (Fig 4); learned-benchmark confirmation (1 para → appendix).
7. **Related work** (½ pg) — signatures in ML/RL (Morrill, Kidger, Lyons–Oberhauser), Doya
   continuous-time RL, delayed/POMDP RL, representation-in-RL, evaluation rigor / registered
   reports.
8. **Discussion & limitations** (½ pg) — when history/signature reps help; $O(\tau^2)$ vs $O(d^L)$
   duality; practitioner guidance (off-manifold data + conditioning > richer algebra); limits
   (LQR-style oracle, finite suite).

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

- Lead contribution: the **law** (sample efficiency) vs the **separation** ($H_2^{\mathrm{pop}}$
  vs $H_2^{(n)}$) vs the **methodology**? (Current lean: the separation as spine, the law as
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
