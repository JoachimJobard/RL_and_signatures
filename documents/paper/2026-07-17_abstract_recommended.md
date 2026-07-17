# AAAI-2027 — the recommended abstract

*Drafted 2026-07-17 against the measured evidence of `2026-07-17_abstract_evidence_base.md` and the
scope decisions taken with the project owner the same day. Abstract text locks **22 July**
(21 July AoE); paper **29 July**. The draft `.tex` is not touched by this file.*

**Status: awaiting adversarial verification** (workflow `wf_983b0e03-509` — fabrication auditor,
risk adversary, hostile reviewer, register auditor). Not yet submitted.

---

## Title

> **A Representation-Agnostic Verification Theorem for Continuous-Time Control of Delay Systems**

The earlier candidate, *"... for **Value-Gradient** Control of Delay Systems"*, is withdrawn: it
privileges one of the three learners, which contradicts the decision that they appear as three
mechanisms instantiating one law with no hierarchy between them.

## Abstract — 198 words (counted, not estimated)

The value of a delay system is a functional of its history window, not a function of the current
state. A verification theorem is proved for that setting: for any feature map $\Phi$ on the window
with an affine tip derivative and any $C^1$ head $\Psi$, the value $V=\Psi\circ\Phi$ delivers the
greedy control in closed form,
$u^\star=-\tfrac12 R^{-1}g(x_t)^\top D\Phi(x_t)^\top\nabla\Psi(\Phi(x_t))$. The functional
Hamilton–Jacobi–Bellman equation on which the argument rests is cited, not claimed. The statement is
representation-agnostic: three feature maps ordinarily presented as competitors are instantiations
of it. Markovian features reproduce Doya's (2000) law exactly; a discretised window returns the
gradient with respect to its most recent node; the path signature returns
$D\Phi(x_t)v=S(\tilde x_t)\otimes\iota(v)$. Because the law factorises into a representation
Jacobian and a head gradient, three learners sharing no common mechanism instantiate it unchanged:
analytic extraction from the critic's vertical derivative, a bootstrapped actor, and a Monte-Carlo
policy gradient. The prediction that history-dependent features lower the cost is measured on five
delay plants at five seeds. It holds on four and fails on a linear cell, where Markovian features
attain a lower suboptimality against an exact delayed-LQR oracle.

---

## Claim ledger

| Claim | Strength | Evidence | Survives every risk? |
|---|---|---|---|
| The verification theorem and its closed-form control | **proven** | the contribution; audit §2 | **yes** — no run can refute it |
| Functional HJB cited, not claimed | citation | Vinter & Kwong 1981 | yes |
| Markovian recovers Doya (2000) exactly | **proven** (special case) | audit §2 | yes |
| Discretised window → gradient at the most recent node | **proven** | audit §2; tip derivative verified numerically at residual $1.2\times10^{-6}$ | yes |
| Signature → $D\Phi(x_t)v=S(\tilde x_t)\otimes\iota(v)$ | **proven** | audit §2 | yes — a statement of the theorem, **not** an H2 claim |
| Three learners instantiate the law | **design fact** | `value_gradient.yaml`; `signatures.yaml` (`actor_target: td`); `policy_gradient.yaml` (`actor_target: monte_carlo`) | yes — true at HEAD, needs no run |
| H1 measured, five plants, five seeds, holds on four | **measured** | `data/main_unified/`, 140 runs, ci95 recorded | yes — already banked |
| Fails on `linear_cell`, Markovian lower against an exact oracle | **measured** | ρ = 0.127 (5/5) vs best history 0.284 | yes |

**Every row survives every risk.** If the campaign never launches, if the actor-critic and policy
gradient produce nothing further, if the functional extension stalls at what is proven today — every
sentence above remains true on 29 July.

## Deliberately absent, and why

1. **Any claim that H1 holds for the actor-critic or the policy gradient.** Never measured; for the
   actor-critic it was *structurally impossible* before `2ae3b73` (signature-only; H1 needs
   markovian).
2. **Any policy-gradient performance number.** `cost_reduction_pct` is unmeasured on every cell, and
   its own audit returns `estimator_is_correct = NO`.
3. **Any claim that the three learners agree.** The campaign has never been launched. The abstract
   states the three as a *design*; the paper may report agreement by the 29th without the abstract
   needing an edit.
4. **Any epistemic hierarchy between the learners** — no "strictly stronger", no learner-invariance,
   no H4. Owner's decision, 2026-07-17.
5. **Any representational advantage for the signature.** H2 is out of scope
   (*"we dont care about h2 in this paper"*) — Paper 2's subject.
6. **A number for the Markovian falsification cell.** `double_integrator` has never been run outside
   `_debug_`, so the by-construction failure is not asserted as measured. Running it at five seeds
   would let the abstract add "and fails there, as measured"; it currently cannot.

## Open for the author

1. Whether the final sentence should name `linear_cell` explicitly or keep "a linear cell".
2. Whether to run the Markovian control cell at five seeds before the 22nd, promoting item 6 above
   from by-construction to measured.
3. TL;DR (optional field), Primary Topic, Secondary Topics — none chosen.
4. The Reproducibility Checklist — unanswered.
