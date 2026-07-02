# Pre-registration — off-sheet-data replication of the H2 verdict (5 seeds)

**Written 2026-07-02, before the run.** This document fixes the hypothesis, the design, the
metric, the analysis, and the falsification conditions *ahead of* the campaign, so the
result cannot be steered by the data (the anti-HARKing commitment). The outcome is to be
reported whatever it is. Commit this file before submitting any job.

## 1. Background and motivation

The comprehensive benchmark (`documents/reports/2026-06-11_experimental_design_matrix/`)
reports **H2 (signature beats raw-history) as supported on the nonlinear Mackey–Glass
cells, CI-backed at 5 seeds** (canonical oracle run
`data/h1h2_report_runs/oracle/20260616_161650_on_off_sheet_array/`). A **single-seed**
off-sheet-data-scaling control (`.../sweeps/h1h2_mg_offsheet_20260630_164840/`, seed 0) then
indicated that this win is a **sample-efficiency / conditioning** effect, not a
representational one: raw-history is under-conditioned at the canonical off-sheet budget
(gradient cosine 0.15–0.41 vs the signature's 0.84–0.94), and giving it 2–4× more
off-manifold data made its gradient fit converge (cosine → 0.9) and its closed-loop cost
match or beat the signature. That control was single-seed. **This campaign settles the
question CI-backed at 5 seeds.**

## 2. Hypothesis (H3) — the claim under test

> **H3.** On the primary (nonlinear-delayed) cells, at matched hypothesis class ($V$ linear
> in $\Phi$), the signature's cost advantage over raw-history is a **sample-efficiency /
> conditioning** effect, not a representational one: as the off-sheet-data budget grows, the
> raw-history (degree-2) closed-loop cost converges to and reaches the signature's, and its
> gradient alignment converges to $\ge 0.9$.

**Operational prediction (fixed now).** Let $I_r(c,s,k)$ be the closed-loop cost of
representation $r$ on cell $c$, seed $s$, off-sheet ratio $k\in\{1\times,2\times,4\times\}$.
On each primary cell $c$, at $k=4\times$:

- **P1 (cost).** The median over the 5 seeds of $I_{\text{signature}} - I_{\text{raw\_history}}$
  is $\ge 0$ (raw-history matches or beats the signature), and the 95% CIs of the two medians
  overlap or the raw-history median is lower.
- **P2 (mechanism).** The raw-history median gradient cosine rises from $k=1\times$ to
  $k=4\times$ and reaches $\ge 0.85$ at $k=4\times$.
- **P3 (monotone convergence).** The raw-history median cost is non-increasing in $k$ across
  $\{1\times,2\times,4\times\}$ (data helps; it is a conditioning effect).

## 3. Falsification condition (fixed now)

H3 is **falsified** on a primary cell if, at $k=4\times$, the **signature retains a strictly
lower cost with separated 95% CIs** over the 5 seeds (signature median $<$ raw-history
median, CIs disjoint) **and** raw-history's gradient cosine stays $< 0.85$. That outcome
means raw-history does **not** catch up with adequate data — i.e. the signature is a genuinely
richer representation and **H2 (representational) stands** on that cell. Report per-cell:
each primary cell is falsified or not independently.

## 4. Design (fixed now)

**Pipeline.** Oracle half only: `run/study/h1h2_evaluation.py` (gated continuous-time
oracle, plain least-squares of the markovian / raw-history / signature critics to $V^\star$,
report value-fit $R^2$, gradient $\cos(u_\theta,u^\star)$, closed-loop $I$). No learning
half in this campaign.

**Cells (7).**

| cell | class | predicted outcome (the falsifiability grid) |
|---|---|---|
| `mg_limit_cycle` | **primary** | H3 expected to hold (raw catches up) |
| `mg_chaotic` | **primary** | H3 expected to hold |
| `hopfield_nonlinear` | **primary** | H3 expected to hold |
| `hopfield_duffing` | **primary** | H3 expected to hold |
| `markovian` | null_check | H1 fails (all reps equal); if H1 "holds", harness bug |
| `linear_dde` | null_check | H2 fails at every $k$ (raw $\le$ signature always) |
| `platoon` | null_check | H2 fails at every $k$ |

**Off-sheet ratio $k$ (the data-scaling knob).** `--n-off` $\in \{500, 1000, 2000\}$,
labelled $\{1\times, 2\times, 4\times\}$ relative to the harness default `N_OFF = 500`.
(1× reproduces the canonical-run data budget; 2×/4× are the scaling under test. Match the
absolute values of the prior single-seed sweep if they are recovered; otherwise these are the
pre-registered values.)

**Seeds (5).** master seeds `0,1,2,3,4` — the same set as the canonical CI-backed run, so
the off-ratio sweep is directly comparable to the 20260616 verdict. The single-seed discovery
used seed 0 only, so seeds 1–4 at $k=2\times,4\times$ are out-of-sample for the scaling
claim; report all five.

**Grid.** 7 cells × 3 ratios × 5 seeds = **105 tasks** (each task runs `--rep all`).

## 5. Metric and analysis (fixed now)

- Per task: closed-loop $I$, gradient cosine, value $R^2$, for each of the three
  representations, written to a per-task `summary.json` (no shared-file race).
- Aggregate per (cell, ratio): **median and 95% CI over the 5 seeds** (bootstrap CI on the
  median; the raw cost is heavy-tailed, so the median is the pre-registered central
  statistic, not the mean). Divergent seeds ($I=\infty$/NaN) are reported as a **separate
  count**, not dropped and not averaged into a NaN.
- Verdict per (primary cell): P1 ∧ P2 ∧ P3 → H3 holds; the §3 condition → H3 falsified.
- **The threshold-free comparison is the cost median + its CI**; no success-rate threshold is
  used to decide the verdict (success rate, if reported, is descriptive only).

## 6. What is NOT changed after seeing results

The cells, the ratios, the seeds, the metric (median $I$ + bootstrap CI), the per-cell
verdict rule, and the falsification condition are all fixed by this document. Only the
reported numbers are filled in afterwards. If the pipeline reveals a bug (e.g. a null_check
cell violates its predicted outcome), the fix is documented and the campaign re-run; the
verdict is not read off a known-buggy run.

## Amendment (2026-07-02, after the first campaign — job 1250130)

**The governing quantity is the off-sheet ratio $\rho_{\text{off}} = n_{\text{off}}/d$, not the
absolute $n_{\text{off}}$.** The first campaign used absolute $n_{\text{off}} \in \{500,1000,2000\}$.
This **under-provisioned `mg_chaotic`**: its raw-history feature dimension is $d=2484$ ($\tau=17$),
so even $n_{\text{off}}=2000$ gives $\rho_{\text{off}}=2000/2484=0.81<1$ — the feature Gram
$G=\Phi^\top\Phi$ is rank-deficient ($\ker G\neq\{0\}$), the gradient $\partial_x\hat V$ is
unidentified, and raw-history **diverges at all three ratios** ($I=552$–$51860$, $\cos\approx0$).
The other three primary cells were adequately provisioned ($\rho_{\text{off}}=5.7$ for
`mg_limit_cycle` $d=350$; $22$ for `hopfield` $d=90$) and gave a **clean CI-backed H3 confirmation**
(raw converges and beats the signature, 5/5 seeds).

**Correction (verdict rule P1/P2/P3 unchanged):** `mg_chaotic` is re-run at **dim-relative** ratios
$\rho_{\text{off}}\in\{1,2,4\}$, i.e. $n_{\text{off}}\in\{2500,5000,10000\}\approx\{1,2,4\}\times d$,
5 seeds (task file `bash_scripts/cluster/jeanzay/python/h1h2_offsheet_mg_chaotic_dimrel_tasks.tsv`).
This is a documented correction of an under-provisioning design flaw, **not** a post-hoc change of
the analysis to obtain a desired result — indeed the failure of the pre-registered P3
(monotone convergence) on `mg_chaotic` is what surfaced the flaw.
