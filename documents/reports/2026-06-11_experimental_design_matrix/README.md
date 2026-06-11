# Experimental design matrix — progress report (2026-06-11)

**Branch:** `scientific-workflow-refactor` · **HEAD at writing:** `69e12b7`
**Scope:** changes to the design matrix (`representation × environment × capacity × seed`)
and the results obtained so far. This is a *progress* note; the pre-registered design
contract it implements is [`documents/methodology/experimental_design.md`](../../methodology/experimental_design.md).

This directory is self-contained for sharing: the comparison figure and the
machine-readable aggregates are committed alongside this note (the canonical
`data/` tree is git-ignored).

| File | Content |
|---|---|
| [`linear_cell_study_comparison.png`](linear_cell_study_comparison.png) | Normalised sub-optimality $\rho$ vs feature dimension $\dim\Phi$, linear delayed cell |
| [`linear_cell_study_summary.yaml`](linear_cell_study_summary.yaml) | Per-cell mean / std / SEM / 95% CI across 5 seeds (human-readable) |
| [`linear_cell_study_aggregation_data.json`](linear_cell_study_aggregation_data.json) | Same aggregates, machine-readable (figure regenerates from this) |

---

## 1. What the matrix is

The study crosses three factors over an environment grid (methodology §2):

- **Representation $\Phi$** (the treatment): `markovian` (current state), `raw_history`
  (discretised history + degree-$m$ monomials), `signature` (depth-$m$ path signature).
- **Capacity** knob: raw-history polynomial degree $\in\{1,2,3\}$ and signature depth
  $\in\{2,3,4\}$, so a "signature wins" claim is read off at *matched* $\dim\Phi$.
- **Environment regime**: the `{Markovian, delayed} × {linear, nonlinear}` grid, each
  cell carrying a *predicted* outcome that doubles as a falsification condition.

The value is a **linear functional** $V=\theta^\top\Phi$ on every variant (Arribas
Thm 4.2), so the only thing that varies is $\Phi$; everything else (optimiser, seeds,
window, integrator) is held identical under common random numbers.

## 2. Changes since the last report

The work this period **expanded the environment axis of the matrix and added the
oracle-ladder decomposition** (methodology §3, rung 2). Five commits:

1. **High-gap delayed oscillator** — a *strong, learnable* H1 cell
   (`conf/env/delayed_oscillator_high_gap.yaml`, `d20c7bd`).
   Linear DDE with strong delayed position+velocity feedback ($\tau=0.2$, 4 delay taps).
   Motivation: the first learnable linear cell (`delayed_velocity_oscillator`) had only
   an $\sim18\%$ markovian-oracle gap, too weak to separate H1. A parameter search (max
   gap subject to open-loop boundedness) placed this env at a **$\sim54.5\%$
   markovian-oracle gap** while staying open-loop bounded ($\max|x|\approx2.9$ from
   $x_0=[1,1]$, so the critic-only agent sees finite TD targets) and delayed-LQR
   stabilisable (closed-loop spectral radius $\approx0.94$). The gap is genuine
   sub-optimality, not divergence.

2. **Chaotic Mackey–Glass stress cell** — nonlinear-delayed robustness test
   (`conf/env/MG_1D_chaotic.yaml`, `run/study/mackey_glass_chaotic_variants.txt`,
   `ce5a6bc`). Canonical chaotic regime $n=10,\ \tau=17$ (above the chaotic onset
   $\tau\approx16$); bounded chaotic attractor, control target the
   delay-destabilised equilibrium $x^\star=1$. Role: **stress / robustness** cell, not
   primary H2 evidence — it tests whether the signature advantage (to be established on
   the $\tau=6$ limit cycle) survives chaos. At $\tau=17,\ dt=0.25$ the window needs
   $\sim70$ taps, so raw-history degree $\geq3$ is infeasible ($\sim57\mathrm{k}$
   monomials); the capacity sweep here is therefore **signature-focused** (the signature
   dimension is independent of window length).

3. **Least-squares value-fit check** — optimisation-vs-representation diagnostic
   (`run/study/least_squares_value_fit_check.py`, `722ff21`). Directly probes whether
   the residual sub-optimality on exact-capacity variants is approximation error
   (representation) or optimisation error (methodology §7.1).

4. **Delayed-LQR oracle wired into the actor-critic agent**
   (`src/agents/signatures_jax.py`, `a27487d`). The closed-form history-functional
   $K^\star/V^\star$ can now replace the agent's actor or critic via
   `agent.algorithm.{actor_oracle,critic_oracle}`. Verified $J_{\text{agent}} =
   J_{\text{oracle}}$ exactly when the oracle actor is wired in.

5. **Oracle-ladder variants** — error decomposition on a linear delayed cell
   (`run/study/oracle_ladder_variants.txt`, `69e12b7`). Three arms — `full_learned`,
   `oracle_critic` (isolates actor / policy-optimisation error), `oracle_actor`
   (validation, $\rho\approx0$ by construction) — that decompose the full agent's
   sub-optimality into representation/critic vs actor components.

**Status of the new cells:** wired and committed, **not yet run** — the only study with
results to date is the linear-cell representation comparison (§3), on `env=delay_jax`.
The high-gap oscillator, chaotic Mackey–Glass, and oracle-ladder runs are pending.

## 3. Results so far — linear delayed cell (control)

**Run:** `data/main_unified/linear_cell_study/`, `env=delay_jax` (linear DDE,
delay $=1.0$, window $=20$ taps, $1000$ episodes), value-gradient backbone, 5 seeds ×
5 arms ($n=25$). Metric: normalised sub-optimality
$\rho=(J_{\text{agent}}-J_{\text{oracle}})/|J_{\text{oracle}}|$ vs the delayed-LQR
ceiling. Claim strength: **measured**, single environment, 5 seeds.

| Representation | Capacity | $\dim\Phi$ | $\rho$ (mean) | 95% CI |
|---|---|---:|---:|---:|
| markovian   | deg 2 | 5   | **0.127** | ±0.016 |
| raw_history | deg 1 | 42  | 0.711 | ±0.002 |
| raw_history | deg 2 | 945 | 0.322 | ±0.190 |
| signature   | depth 2 | 30 | 0.284 | ±0.074 |
| signature   | depth 3 | 155 | 3.86 | ±1.92 |

![Representation comparison on the linear delayed cell](linear_cell_study_comparison.png)

Reading the cell against its **predicted-null** H2 prediction (on a *linear* delayed
plant the optimal control is a linear functional of the history, so the signature's
nonlinear iterated-integral features must *not* help):

- **H2 null is consistent with the data.** At matched capacity (depth-2 signature
  $\dim 30$ vs degree-2 raw history $\dim 945$, both spanning the quadratic path
  functionals that contain the delayed-LQR value), $\rho_{\text{sig}}=0.284\,(\pm0.074)$
  vs $\rho_{\text{raw}}=0.322\,(\pm0.190)$ — **overlapping CIs, no signature advantage**.
  The signature does *not* beat raw history here, as required for the control cell.
- **Over-parameterised signature degrades.** Depth-3 ($\dim 155$) blows up to
  $\rho=3.86\,(\pm1.92)$ — the methodology's explicit warning ("if the signature wins on
  a linear delayed plant, suspect over-parameterisation") instead manifests as a large
  *loss*, confirming the inductive-bias mismatch rather than a leakage artefact.
- **Under-capacity raw history fails as predicted.** Degree-1 raw history (linear $V$)
  cannot represent the quadratic delayed-LQR value and sits at $\rho=0.711\,(\pm0.002)$ —
  a clean representation failure, not an optimisation floor.
- **Caveat — weak H1 separation on this cell.** The markovian baseline attains the
  *lowest* $\rho=0.127$ here, i.e. on `delay_jax` the current-state representation is
  already near-best and history does not help. This is exactly the weak-separation
  problem that motivated the **high-gap delayed oscillator** (change 1): a cell where
  the markovian–oracle gap is large ($\sim54.5\%$) so H1 can actually be tested. The
  present linear-cell numbers should therefore be read as a **pipeline/control
  validation**, not as an H1 result.

## 4. Caveats and next steps

- **Single environment, single seed-block.** The table is one linear cell at 5 seeds;
  treat it as a control/sanity result, not a hypothesis test.
- **Pending runs** that close the matrix: (i) high-gap oscillator for a genuine H1
  separation; (ii) a well-conditioned nonlinear-delayed cell (limit-cycle Mackey–Glass)
  for the *primary* H2 claim, with the chaotic $\tau=17$ cell as the robustness stress
  test; (iii) the oracle-ladder decomposition to attribute residual error to
  actor vs critic.
- **Figure cosmetics.** The committed PNG has a legend/axis-label overlap (the `dim Φ`
  annotation collides with the legend); regenerate via the aggregator's replot path
  before using it in a write-up.
- **Open design items** remain as listed in methodology §10 (choice of the primary
  nonlinear-delayed env; augmented-LQR vs exact $Z,B_2,P_1$ oracle construction).
