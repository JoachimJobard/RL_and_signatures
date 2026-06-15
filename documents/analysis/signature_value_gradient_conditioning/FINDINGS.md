# Why the signature value-gradient critic fails on the high-dimensional platoon

**Question.** On the connected-cruise-control platoon
([`src/envs/platoon.py`](../../../src/envs/platoon.py); state dimension $2N=10$,
$N=5$ vehicles), the **markovian** value-gradient critic learns a good controller
($I=0.039$ vs the delayed-LQR optimum $0.028$ and no-control $0.075$), but the
**signature** critic does not: semi-gradient TD diverges, and LSTD — plain or
whitened — cannot recover a usable critic (best $I\approx3.5$, worse than no
control). Is the failure due to (1) the signature **features** (conditioning),
(2) **representation** capacity / non-Markovianity, or (3) the **control law**
(vertical derivative)?

**Method.** Data analysis of the signature features along trajectories
([`run/study/diagnose_signature_conditioning.py`](../../../run/study/diagnose_signature_conditioning.py)).
Roll the delayed-LQR oracle out from one initial condition and from 24 diverse
initial conditions; for each representation compute the features along the
trajectories and measure: the feature-Gram condition number and **effective rank**
(participation ratio $(\sum\lambda)^2/\sum\lambda^2$); the **value-fit $R^2$** of a
ridge least-squares critic on the discounted return-to-go (representation test); and
the closed-loop cost of the value-gradient control extracted from that ideal
least-squares critic (control test).

**Results** (regime $\alpha=3,\beta=1,\sigma=0.2$; string-stable, bounded paths).

| representation | dim | value-fit $R^2$ | eff. rank (1 IC) | eff. rank (24 ICs) | Gram cond |
|---|---|---|---|---|---|
| markovian deg-2 | 65 | 1.000 | 1.4 | **6.3** | $10^{9}$ |
| raw-history deg-2 | 4185 | 1.000 | 1.5 | **10.0** | $10^{13}$ |
| signature depth-2 | 462 | 1.000 | 1.0 | **1.1** | $10^{15}$ |
| signature depth-3 | 9723 | 1.000 | 1.0 | **1.1** | $10^{15}$ |

![Feature conditioning](conditioning_spectra.png)

**Conclusion — it is a signature-specific feature-collinearity problem, not
representation.**

1. **Not representation / non-Markovianity.** The value-fit $R^2=1.0$ for *every*
   representation, signature included: a linear functional of the signature
   represents the value function essentially exactly. The signature *can* express the
   optimal value.

2. **Not (merely) data excitation.** With a single initial condition every
   representation is effective-rank $\approx1$. Exciting with 24 diverse initial
   conditions raises the markovian and raw-history effective rank to **6–10** (their
   linear critics become identifiable), but the **signature stays at effective rank
   $\approx1$** with condition number $\sim10^{15}$ regardless of excitation. The
   signature features are *intrinsically* near-collinear — dominated by a single
   direction (the monotone time-augmentation channel) on top of the algebraic
   **shuffle** redundancies of iterated integrals (e.g.
   $S^{(i,j)}+S^{(j,i)}=S^{(i)}S^{(j)}$). The linear-in-$\Phi$ critic is therefore
   **unidentifiable**, which is exactly why semi-gradient TD diverges (deadly triad on
   a rank-1 Gram) and LSTD — whose normal equations invert that same near-singular
   Gram — cannot recover a usable critic; per-feature whitening makes it worse because
   it amplifies the near-null directions.

3. **Caveat on the control test.** The ideal-critic (least-squares) value-gradient
   control costs are all worse than no-control, but that test is confounded: the
   least-squares critic is fit on the oracle's *narrow on-trajectory* state
   distribution, so its vertical derivative $\partial_x V$ is unreliable off that
   distribution (where the greedy control probes it). It does not bear on the
   conditioning finding.

**Implications / candidate fixes** (not yet validated):
- **Log-signature** — the minimal, non-redundant coordinates of the free Lie algebra;
  removes the shuffle redundancies and should raise the usable feature rank.
- **Per-channel signature normalisation / time-channel rescaling** so no single
  augmentation channel dominates the Gram spectrum.
- These are representation-level fixes; the env itself is sound (history genuinely
  matters here: the delayed-LQR optimum beats the markovian-LQR by $+105\%$), and the
  markovian critic already learns it, so the cell is a valid H1/H2 testbed *once the
  signature critic is made identifiable*.

Raw numbers and the figure regenerate via
`uv run python run/study/diagnose_signature_conditioning.py`
(outputs to `data/diagnose_signature_conditioning/`; the committed copy here is a
snapshot).

## Update (correction): conditioning is real but NOT the operative barrier

Implementing the prescribed fix — **centred features** in the LSTD critic (running
feature mean subtracted; the gauge fix) — and testing it on the platoon signature
**did not** make the agent learn: across regularisations the closed-loop cost stayed
$I\approx12$–$124$, far worse than no-control ($0.075$) and than the markovian critic
($0.039$). Centring **does** restore the feature-Gram effective rank ($1.07\to6.17$, as
the analysis predicted), so the conditioning diagnosis is correct *as far as it goes* —
but conditioning was **necessary, not sufficient**: it is not what blocks the control.

The binding barrier is the **value-gradient control law itself**, i.e. the **vertical
derivative** $\partial_x V=\theta^\top\partial_x\Phi$. Evidence: the *ideal* critic — a
least-squares fit with $R^2=1.0$ (perfect value on the data) — still yields bad control
($I=11$ for signature depth-2). A perfect value fit does not imply a usable gradient,
because $\partial_x V$ probes the critic **off** the (thin, low-dimensional) trajectory
manifold on which it was fit, where a high-dimensional signature critic extrapolates
unreliably; the markovian critic's low-dimensional polynomial gradient generalises off
that manifold and so controls well. So there are **two distinct issues**: (1) feature
conditioning (real, fixed by centring), and (2) the off-manifold vertical-derivative of
the value-gradient control law (the operative one, untouched by centring).

Implications for next steps (none yet validated): the operative barrier (2) is a property
of the *value-gradient* control with high-dimensional signature features on thin-manifold
data, not of the representation's capacity ($R^2=1$) nor (only) its conditioning. Candidate
directions: an explicit **actor** (actor-critic) that learns the control directly and does
not differentiate the signature critic; off-manifold **excitation** of the endpoint so
$\partial_x V$ is constrained where the control queries it; or a lower-dimensional
signature (depth-1, smaller window) whose vertical derivative is better behaved.

## Second correction: the "ideal-critic" / vertical-derivative test is itself confounded

Quantifying barrier (2) directly (cosine between the value-gradient control
$u=\tfrac12R^{-1}B^\top\partial_xV$ from the ideal least-squares critic and the oracle
control $u^\star=-K\xi$, on- and off-manifold, vs depth and window) gave cosine $\approx0$
for **every** representation at **every** perturbation level — **including markovian at
$\eta=0$**. But markovian *does* learn a good controller in the RL loop ($I=0.039$). The
only consistent reading: an $R^2=1$ **least-squares value fit has a gradient nearly
orthogonal to the control-relevant gradient** — matching a function in $L^2$ does not match
its derivative. Hence the "ideal-critic control test" is **not a valid probe of a
representation's control adequacy** (it fails markovian too), so it does **not** establish
a signature-specific barrier (2); and value-fit $R^2$ says nothing about the control
gradient. The robust, valid distinction that remains between markovian (learns) and
signature (fails) is the **conditioning of the learning dynamics**: the markovian TD/LSTD
iteration is well-conditioned (Gram rank 6, cond $10^6$) and converges, the signature's is
not (rank 1, cond $10^{15}$). Centring fixes the *feature Gram* but not the LSTD control,
which points at the LSTD **drift matrix** $M=\mathbb{E}[\phi((\phi'-\phi)/dt-\phi/\tau)^\top]$
(distinct from the Gram) and/or the least-squares-policy-iteration loop — not yet isolated.
Net: the validated facts are the feature-conditioning gap (signature rank 1 vs markovian
rank 6, unaffected by excitation) and that conditioning fixes alone (centring, whitening,
Tikhonov) do not yet recover signature control; the precise learning-dynamics mechanism and
a working fix are open. This is flagged to prevent over-claiming either barrier.

## Resolution (MWE): the value-fit identifies the critic only modulo $\ker G$, and the control reads the unidentified part

A minimal working example settles the mechanism. See
[`run/study/mwe_history_representation_control.py`](../../../run/study/mwe_history_representation_control.py):
a plain 2-D double-integrator LQR with **no delay** (markovian is exactly optimal; the true
value is the known quadratic $V^\star(x)=-x^\top P x$, $u^\star=-Kx$). No learning loop — the
critic is solved in closed form (ridge-fit to the *true* value along the oracle rollout, so
value-fit $R^2=1$ by construction and the critic is not the variable). The value-gradient
control $u=\tfrac12R^{-1}B^\top\partial_xV$ is then read off each representation.

**Result (hypothesis H0 — the window-machinery alone, no history to exploit).**

| representation | dim | value-fit $R^2$ | $\lVert u\rVert/\lVert u^\star\rVert$ | $\cos_{L^2}(u,u^\star)$ | closed-loop $I$ | vs no-control ($0.075\to 6.0$) |
|---|---|---|---|---|---|---|
| markovian deg-2 | 5 | 1.000 | 0.98 | 1.000 | **1.3055** (= oracle) | OK |
| raw-history deg-2 | 189 | 1.000 | 5.12 | 0.253 | 209.0 | worse than no-control |
| signature depth-2 | 30 | 1.000 | 17.30 | $-0.061$ | 1621.8 | worse than no-control |

The failure is **active harm by a large, misdirected gradient** ($\lVert u\rVert/\lVert u^\star\rVert=17$,
$\cos\approx0$), not a vanishing one (which would recover no-control). It is robust across
$R$, $x_0$, window length, and depth, and it hits raw-history too — so it is **not**
history-dependence, **not** representational capacity ($R^2=1$), and **not** signature-specific
algebra.

**The mechanism (functional-analytic).** Let $\Gamma=\gamma([0,L])$ be the oracle trajectory (a
regular $C^1$ curve), $\mu$ its occupation measure (support $=\Gamma$, a Lebesgue-null 1-D set),
$H=L^2(\mu)$, and $E_\mu:\mathbb R^d\to H$, $E_\mu\theta=\theta^\top\phi$ the evaluation operator
(synthesis). Its adjoint is analysis, $E_\mu^\ast f=\mathbb E_\mu[\phi f]$, and the feature Gram is
the composition $G=E_\mu^\ast E_\mu$, $G_{k\ell}=\langle\phi_k,\phi_\ell\rangle_H$. Because
$\lVert E_\mu\theta\rVert_H^2=\theta^\top G\theta$ and $G\succeq0$,
$$\ker E_\mu=\ker G=:\mathcal N .$$
The least-squares fit is the projection $\widehat V=\Pi_{\mathcal F}V^\star$,
$\mathcal F=\operatorname{span}\{\phi_k\}\subset H$, with normal equations $G\theta=E_\mu^\ast V^\star$;
the minimiser set is the affine space $\theta^\star+\mathcal N$ ($\theta^\star$ any exact global
representer). The loss is **exactly constant** on it: for $\nu\in\mathcal N$,
$$L(\theta+\nu)=L(\theta)+2\!\!\int(\theta^\top\phi-V^\star)\,\nu^\top\phi\,d\mu+\nu^\top G\nu=L(\theta),$$
both added terms vanishing because $\nu^\top\phi=0$ $\mu$-a.e. So the data fixes $\theta$ only
**modulo $\mathcal N$** — $\nu=\widehat\theta-\theta^\star\in\mathcal N$ is a gauge indeterminacy, not a
bias.

**Lemma (the fit pins only the tangential derivative).** For $\nu\in\mathcal N$,
$\nu^\top\phi(\gamma(s))\equiv0$; differentiating in $s$ gives
$\langle D\phi(\gamma(s))^\top\nu,\ \gamma'(s)\rangle=0$. Hence the induced gradient error
$e(x)=D\phi(x)^\top\nu$ is **normal** to $\Gamma$ and otherwise free (the $n-1$ normal components
are unconstrained). Side by side:
$$\nu^\top G\,\nu=0\quad(\text{loss frozen})\qquad\text{but}\qquad D\phi^\top\nu\neq0\quad(\text{control moves}),$$
no contradiction because $\ker G\not\subseteq\ker(D\phi^\top)$. The value-gradient law consumes
exactly the unidentified normal derivative; markovian escapes because $d=5$ is fully excited by
the curve ($\mathcal N=\{0\}$, $\operatorname{cond}G\sim10^6$), whereas the signature has $d=30$
with $\operatorname{rank}_{\mathrm{eff}}G\approx1$ ($\operatorname{cond}\sim10^{15}$: shuffle
redundancy + scale decay), so $\mathcal N$ is effectively $\approx29$-dimensional.

**Consequences.**
- **Not a finite-sample problem.** The empirical $G$ converges to the population $G=\mathbb E_\mu[\phi\phi^\top]$, whose kernel $\mathcal N$ **persists**; sampling the *same* curve more densely does not remove it. The deficiency is of the **support dimension** (the normal bundle is never sampled), not the sample count.
- **Training does not self-correct.** TD/LSTD solve the *projected* fixed point $V_\theta=\Pi_\mu\mathcal T V_\theta$; the projector annihilates $\mathcal N$ and the normal derivative at **every** iteration, and the $L^2(\mu)$ contraction is blind to the gradient. Every policy's occupation measure is again 1-D, so the deficiency is reproduced, not annealed. Action-noise exploration (white, or the RBF Gaussian-process mode) only thickens $\mu$ to an $O(\sigma)$ tube through $\operatorname{range}(B)$, at $O(\sigma^{-2})$ gradient variance — a bias–variance wall; the smooth/GP mode, being low-frequency, covers the normal bundle *less*.

**Validated fix (derivative matching).** Refitting with the analytic endpoint-gradient constrained,
$$\textstyle\min_\theta\ \lVert\theta^\top\phi-V^\star\rVert_{L^2(\mu)}^2+\gamma\sum_t\lVert J_t^\top\theta-\partial_xV^\star(x_t)\rVert^2,\qquad J_t=\partial_{x(t)}\phi,$$
i.e. replacing the $L^2(\mu)$ norm by the Sobolev norm $\lVert V\rVert_{L^2(\mu)}^2+\gamma\lVert\nabla V\rVert_{L^2(\mu)}^2$ under which $\nabla$ is bounded by construction, **snaps both signature and raw-history to the exact oracle** ($I=1.3055$, $\cos=1.000$, $\lVert u\rVert/\lVert u^\star\rVert=1.00$) at $\gamma\in\{1,100\}$. This proves the barrier is the *fitting objective*, not the representation: matching $V$ in $L^2$ does not match $\partial_xV$.

**Scope.** Shown on E0 (no delay) with an analytic gradient target. The transfer claim — that a
gradient-constrained critic or an explicit actor (which never differentiates the critic) fixes the
*platoon RL* — is the next test (E1 scalar delayed LQR, then the platoon agent), not yet
demonstrated. The earlier "conditioning" framing is **superseded**: centring repaired the Gram of
$V$ but never touched $\partial_xV$, which is why it did not restore control; and the truncated-SVD
partial success is explained as shrinking $\mathcal N$ (the gradient null space).

### Trajectory-count sweep (E0): more on-policy data saturates *below* the optimum

Does adding oracle trajectories — past the point where the number of samples exceeds the
number of features — restore the value-gradient control on E0? See
[`run/study/mwe_e0_trajectory_sweep.py`](../../../run/study/mwe_e0_trajectory_sweep.py)
(analytic target $V^\star=-x^\top Px$; pool oracle rollouts from $n_{ic}$ ICs over the plane;
deploy from a canonical $x_0$).

| rep | $n_{ic}$ | $n_{\text{pts}}$ | $\operatorname{cond}$ | eff. rank | $\cos_{L^2}$ | $I_{\text{cl}}$ |
|---|---|---|---|---|---|---|
| markovian $d{=}5$ | 1 | 120 | $10^6$ | 1.2 | 1.000 | 1.3055 (oracle) |
| markovian $d{=}5$ | 256 | 30720 | $42$ | 1.9 | 1.000 | 1.3055 |
| signature $d{=}30$ | 1 | 120 | $10^{18}$ | 1.4 | $-0.06$ | 1621.8 |
| signature $d{=}30$ | 4 | 480 | $10^{18}$ | 1.5 | 0.25 | 3.30 |
| signature $d{=}30$ | 16 | 1920 | $10^{18}$ | 3.9 | 0.59 | 1.90 |
| signature $d{=}30$ | 256 | 30720 | $10^{18}$ | **5.2** | **0.63** | **1.82** |
| raw-history $d{=}189$ | 256 | 30720 | $10^{20}$ | **2.0** | **0.62** | **2.12** |

**Adding trajectories removes the catastrophic blow-up but saturates short of the oracle.**
Signature improves $I:1621\to1.82$ and $\cos:-0.06\to0.63$, then **stops** by $n_{ic}\approx16$
($n_{ic}=64,256$ are identical). Two consequences:

1. **"Samples $\gg$ features" is not the cure.** At $n_{ic}=256$ there are $30720$ points versus
   $d=30$/$189$ — a $10^3\times$ excess — and it changes nothing past the plateau, because the
   **effective rank saturates far below $d$** (signature $\to5.2$ of $30$; raw-history $\to2.0$ of
   $189$) and $\operatorname{cond}$ stays $\sim10^{18}$ regardless of $n_{ic}$.

2. **The saturation has a structural cause.** The oracle flow is autonomous and deterministic, so
   the history is a function of the current state and **every window lies on a manifold of
   dimension $=$ the state dimension** (here $2$), for any window length or trajectory count. On
   E0 this manifold is exactly the $2$-D linear subspace $\mathcal M=\operatorname{im}(L)$,
   $L=[\,e^{-M(L-1)\,dt};\dots;e^{-M\,dt};I\,]$, $M=A-BK$. The high-dimensional window features are
   therefore intrinsically collinear on $\mathcal M$ — their image is $\approx5$-D (signature) /
   $\approx2$-D (raw-history), the eff-rank ceiling. Adding trajectories fills $\mathcal M$ in its
   **tangent** directions, which pins the derivative of $V_\theta$ *along* $\mathcal M$ (this lifts
   $\cos$ from $0$ to $0.63$). But the control law needs $\partial V/\partial x(t)$: the change in
   value when the **current state is moved with its past held fixed**. In the data the present and
   its history move *together* along the flow (the past is the backward image of the present), so a
   perturbation that moves the current state alone points **off** $\mathcal M$ — and no on-policy
   trajectory ever leaves $\mathcal M$ to constrain it. Markovian escapes because its features
   depend only on $x(t)$: its manifold is the state space itself, so moving the current state stays
   on it (no off-manifold direction), and one trajectory already makes its $5$ features full-rank
   ($\cos=1$, $I=$ oracle).

**Confirmation (tangent/normal decomposition).**
[`run/study/mwe_e0_tangent_normal.py`](../../../run/study/mwe_e0_tangent_normal.py) decomposes the
window-gradient error $E=\nabla_{\mathrm{win}}V_\theta-\nabla_{\mathrm{win}}V^\star$ at the deployed
states into the part along $\mathcal M$ and the part across it. The result is unambiguous:
$\lVert P_{\mathcal M}E\rVert\approx0$ (the along-manifold derivative is recovered *exactly*) and the
entire error lies in the normal complement (normal energy fraction $=1.000$), while $94\%$ of the
directions the control differentiates along — moving the current state with the past frozen — are
themselves normal to $\mathcal M$. With strictly on-manifold (full-history) windows this normal
error is **independent of $n_{ic}$**, confirming that on-policy trajectories cannot touch it; the
small $\cos$ gain in the sweep above came only from the front-padded (flow-inconsistent, hence
off-manifold) windows incidentally leaking a little normal information.

So "add more trajectories" and "add the gradient constraint" are **not interchangeable**: the
former fixes the along-manifold derivative, only the latter (derivative matching, $\cos\to1$, $I\to$
oracle on E0) fixes the across-manifold one. On-policy data is trapped on a state-dimensional
manifold; the value-gradient law reads a derivative pointing off it.
