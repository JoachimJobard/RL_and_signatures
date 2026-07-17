# AAAI-2027 abstract — candidates and recommended title

> ## ⚠ SUPERSEDED IN PART — 2026-07-17, later the same day
>
> Three defects, each established by measurement. See
> **`2026-07-17_abstract_evidence_base.md`** for the corrected evidence base.
>
> 1. **The claim table below states "5 studies × 5 seeds × 3 representations". That is wrong.**
>    `data/main_unified/` holds **140 runs** — five plants × five seeds × **five to seven** feature
>    maps spanning three *families* (`raw_deg1/2/3`, `sig_depth2/3/4` are distinct variants).
>    Worse, the metric is **not commensurable** across the five: the three linear plants are scored
>    as suboptimality against an exact delayed-LQR oracle, the two Mackey–Glass plants as raw cost.
> 2. **"H1 holds on delay plants" is too strong.** Measured: H1 holds on **four of five** plants and
>    **fails outright on `linear_cell`**, where Markovian features win at rho = 0.127 (5/5) against a
>    best history variant of 0.284.
> 3. **The candidates are framed defensively around H2, which is now out of scope entirely**
>    (owner, 2026-07-17: *"we dont care about h2 in this paper"*). Every clause spent not-claiming a
>    signature advantage is wasted. The signature is one instantiation among three, nothing more.
>
> Also decided 2026-07-17: all three learners appear, as **three mechanisms instantiating one law,
> with no claim of epistemic hierarchy between them**. The `5b410d6` "strictly stronger" argument is
> **agent-authored and unendorsed** — do not reintroduce it (it is the same class of object as H4,
> which was reverted at the owner's instruction in `49db500`).
>
> Candidates 3–6 below are retained for their phrasing, not their claims.

*Drafted 2026-07-17. Abstract due 21 July; full paper 28 July. Scope per
`2026-07-16_aaai_rl_paper_audit.md` (commit `008c0f5`): Paper 1 = the RL half only.
The draft `.tex` is not touched by this file.*

---

## 1. What the abstracts may and may not say (checked against the repository)

| Claim | Status | Source |
|---|---|---|
| The verification theorem, representation-agnostic | **proven** (the contribution) | audit §2 |
| Markovian features recover Doya (2000) | **proven** (special case) | audit §2 |
| Functional HJB | **cited, not claimed** (Vinter & Kwong 1981) | audit §2 |
| H1 holds on delay plants | **measured, 5 studies × 5 seeds × 3 representations, value gradient only** | `data/main_unified/` (verified: seeds 0–4, reps markovian/raw/sig on all five) |
| H1 fails on the Markovian cell | **true by construction, NOT measured** | `conf/env/double_integrator.yaml`: `A1: numpy.zeros`, `delay: numpy.zeros` |
| Three learners instantiate the law | **implemented** (design fact) | `value_gradient.yaml`; `signatures.yaml` (`actor_target: td`, Doya Eq 20); `policy_gradient.yaml` (`actor_target: monte_carlo`, commit `5b410d6`) |
| The learners **agree** | **NOT measured — campaign not launched, policy gradient never run** | [[ac-campaign-state]] |
| The signature is a richer representation | **forbidden** — refuted by the oracle half, belongs to Paper 2 | audit §0 |

**Two slots are deliberately empty in every candidate below:** no cross-learner agreement is
asserted (unmeasured), and no number is attached to the Markovian control cell (never run outside
`_debug_`). Each candidate states the learners as a *design* whose purpose is to isolate the law from
the optimiser — which is true now — and can absorb measured agreement by the 28 July deadline
without rewriting.

Notation throughout: $\tau$ = **delay**; the discount is a **rate** $\gamma$, so the HJB reads
$\gamma V^\star = \sup\{\cdots\}$.

---

## 2. Recommended title

> **A Representation-Agnostic Verification Theorem for Value-Gradient Control of Delay Systems**

It foregrounds the contribution (the theorem and its representation-agnosticism), promises nothing
the draft cannot deliver, and avoids the signature-positive framing the evidence does not support.

Alternates:
- *One Law, Three Representations: Value-Gradient Control of Delay Systems* — punchier; "one law"
  carries the unification, but reads more like a workshop title.
- *Value-Gradient Control of Delay Systems: Markovian, History and Signature Critics as One Theorem*
  — most explicit, least elegant.
- **Reject:** anything of the form *"Continuous RL **using Signatures**"* (the current draft title).
  It promises a signature result the paper must not claim.

---

## 3. Candidate A — theorem-forward (recommended pairing with the title above) · 181 words

Continuous-time control of a delay system requires the value to be a functional of the history
window rather than a function of the current state. A verification theorem is proved for that
setting: for any feature map $\Phi$ on the window admitting an affine tip derivative and any $C^1$
head $\Psi$, the value $V=\Psi\circ\Phi$ delivers the greedy control in closed form,
$u^\star=-\tfrac12 R^{-1}g(x_t)^\top D\Phi(x_t)^\top\nabla\Psi(\Phi(x_t))$. The statement is
representation-agnostic, and three feature maps ordinarily presented as competitors are recovered as
instantiations of it: Markovian features reproduce Doya's (2000) law exactly; a discretised history
window returns the gradient with respect to its most recent node; the path signature returns
$D\Phi(x_t)v=S(\tilde x_t)\otimes\iota(v)$. The functional Hamilton–Jacobi–Bellman equation on which
the argument rests is cited, not claimed. The construction is verified on five delay plants,
measured at five seeds and three representations under an analytic value-gradient learner, and is
equipped with a Markovian cell of delay exactly zero on which, by construction, it must fail. Three
learners instantiate the same law by different mechanisms, so that agreement across them would
isolate the law from the optimiser.

---

## 4. Candidate B — factorisation-forward · 176 words

The value of a delay system is a functional of its history window, and any learned controller must
extract a control from that functional's derivative. A verification theorem is proved which
factorises that extraction: for a feature map $\Phi$ with an affine tip derivative and a $C^1$ head
$\Psi$, the greedy control of $V=\Psi\circ\Phi$ is
$u^\star=-\tfrac12 R^{-1}g(x_t)^\top D\Phi(x_t)^\top\nabla\Psi(\Phi(x_t))$. The factorisation into a
representation Jacobian $D\Phi$ and a head gradient $\nabla\Psi$ separates what the features do from
what the head does. Two consequences follow. First, the Markovian, discretised-history and
path-signature critics are instantiations of one theorem rather than competing constructions, with
Doya's (2000) law recovered exactly in the Markovian case. Second, a nonlinear head costs nothing
theoretically: $\nabla\Psi$ is one automatic-differentiation call, so no separate theorem is required
for a neural critic. The functional Hamilton–Jacobi–Bellman equation underlying the argument is
cited rather than claimed. The instantiations are verified on five delay plants at five seeds and
three representations, and a Markovian cell of delay exactly zero is carried as the control on which
the construction requires failure.

---

## 5. Candidate C — falsifiability-forward · 186 words

A learned controller for a delay system reads its value functional only through a derivative, so a
guarantee about values is not a guarantee about controls. A verification theorem is proved that
addresses the derivative directly: for any feature map $\Phi$ on the history window with an affine
tip derivative and any $C^1$ head $\Psi$, the greedy control of $V=\Psi\circ\Phi$ is
$u^\star=-\tfrac12 R^{-1}g(x_t)^\top D\Phi(x_t)^\top\nabla\Psi(\Phi(x_t))$, with the functional
Hamilton–Jacobi–Bellman equation cited rather than claimed. Being representation-agnostic, the
theorem predicts the behaviour of each of its instantiations — Markovian features recovering Doya
(2000) exactly, a discretised window returning the gradient at its most recent node, the path
signature returning $D\Phi(x_t)v=S(\tilde x_t)\otimes\iota(v)$ — including where they must fail. The
predictions are examined on five delay plants, measured at five seeds and three representations
under a value-gradient learner, and against a Markovian control plant whose delay is exactly zero
and whose optimal value depends on the current state alone: there the history representations can
buy nothing, by construction rather than by measurement. Three learners instantiate the same law,
so that agreement across them would separate the law from the optimiser.

---

## 6. Candidate D — learner-invariance-forward · 181 words

Whether a learned continuous-time controller succeeds on a delay plant may be a property of the
control law or an artefact of the optimiser that fits it. A verification theorem is proved which
makes the first hypothesis testable: for any feature map $\Phi$ on the history window with an affine
tip derivative and any $C^1$ head $\Psi$, the greedy control of $V=\Psi\circ\Phi$ is
$u^\star=-\tfrac12 R^{-1}g(x_t)^\top D\Phi(x_t)^\top\nabla\Psi(\Phi(x_t))$; the Markovian,
discretised-history and path-signature critics are its instantiations, the first recovering Doya
(2000) exactly. The functional Hamilton–Jacobi–Bellman equation on which the proof rests is cited,
not claimed. Because the law is fixed independently of the optimiser, it can be instantiated by
learners that share nothing else: analytic extraction from the critic's vertical derivative, a
continuous actor-critic whose actor is bootstrapped through the critic, and a Monte-Carlo policy
gradient whose actor signal carries no bootstrap. Agreement across mechanisms would then be evidence
about the law rather than about any one optimiser. The value-gradient instantiation is measured on
five delay plants at five seeds and three representations, against a Markovian control cell of delay
exactly zero.

---

## 7. Assessment

**A** is the safest and matches the recommended title; it states the contribution first and the
evidence second, and its final sentence introduces the learners without promising their agreement.

**B** is the most technically distinctive — the $(D\Phi,\nabla\Psi)$ factorisation and the free
neural head are the genuinely new content — but it spends words on a consequence (the neural head)
that the paper does not currently exercise.

**C** is the most defensible against a sceptical reviewer, because it foregrounds the $C^0$-to-$C^1$
gap (audit §1 item 10) and the falsification control. Its risk is sounding defensive.

**D** carries the learner design furthest, and is the only candidate that names all three mechanisms.
Its risk is that a reviewer reads the agreement as promised; the conditional ("would be evidence") is
doing real work and must not be softened into an assertion.

**Recommendation: A, with B's factorisation sentence grafted in if a line can be found** — the
factorisation is the contribution that distinguishes this from Doya (2000), and A currently implies
it without stating it.

---

## 8. Open, for the author

1. If the actor-critic and policy-gradient campaign lands before 28 July, one clause of A/D can be
   promoted from *"would isolate"* to a measured statement. Until then it stays conditional.
2. The Markovian control cell is cited by construction only. Running it at five seeds would let the
   abstract say "and fails there, as measured" — currently it cannot.
3. Candidates B and D reference the neural head and the three mechanisms respectively; both are
   implemented but neither is exercised by a reported experiment.
