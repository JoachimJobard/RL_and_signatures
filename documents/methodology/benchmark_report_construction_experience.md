# Constructing a benchmark report — experience handoff

*Written by an agent (Claude) that has just finished building an a-posteriori
benchmark report in a sibling repository (`learning_moments_kinetics`, the
report `documents/reports/closure_aposteriori_benchmark_report.tex`). This note
is shared so that the same structure, conventions, and — above all — the same
rules of rigour can be reused to build the analogous report for the
`RL_and_signatures` work. It is descriptive of what was actually done, including
the mistakes that had to be corrected; treat it as a checklist, not a template
to fill blindly.*

---

## 1. What the report is for

The report benchmarks several methods (analytic, fitted, learned) against a
reference, on a small number of well-defined test cases, and reports both the
quantity of interest and the structural properties (here: accuracy and
hyperbolicity). The reader must be able to (i) reconstruct every number from the
saved artefacts, and (ii) tell at a glance what is *proven*, what is *measured*,
and what is *conjectured*. For `RL_and_signatures`, substitute the corresponding
objects: the reference is the ground-truth value / optimal policy / analytic
benchmark; the methods are the signature-based estimators / agents; the test
cases are the chosen environments or option-pricing instances; the structural
property is whatever plays the role hyperbolicity played here (stability,
variance, a no-arbitrage or monotonicity constraint, convergence order).

## 2. Document structure (the skeleton that worked)

1. **Common framework** — everything shared across test cases, stated once:
   - the governing object and the reference solution (as `Definition`s);
   - the problem actually solved (the estimator / the learning problem), with the
     hypothesis class written explicitly;
   - the methods compared (one `Definition` each, including any *oracle* baseline
     — a deliberately strong, non-deployable adversary chosen with hindsight);
   - the learning problem: data, objective, optimiser, model selection;
   - the metrics, each given a single-letter symbol and defined once.
2. **One section per test case**, each with the identical sub-skeleton:
   - the problem (setup, initial/boundary data or instance) as a `Definition`;
   - **measured results** as tables, one row per severity parameter (never only an
     average — see the averaging trap in §5);
   - visualisation (a focused figure plus a full companion figure);
   - analysis as `Empirical finding` / `Observation` environments.
3. **A mechanism / discussion section** — the conceptual core. If a statement can
   be proven, it is a `Proposition` with a `proof`; the measured evidence for it
   is a *separate* `Empirical finding`. Keep these two apart (see §4).
4. **Appendices** — a-priori / diagnostic views, the data manifolds, and a
   **Reproduction** section listing the exact scripts, datasets, and run
   directories that produced every figure and number.

A "Synthesis" section was tried and then **removed**: it was uncited, and its
sole content (a parameter-averaged table) contradicted the report's own point
that the average obscures the per-parameter story. Lesson: do not add a summary
section that re-aggregates in a way the body argues against.

## 3. Claim-strength discipline (theorem environments)

This is the single most important convention. Reserve the environments by
*epistemic status*, not by importance:

- `Definition`, `Proposition`/`Lemma`/`Corollary` + `proof` — **proven** mathematical
  statements only. A `Proposition` without a real proof must be downgraded.
- `Empirical finding` — a result **read off measured tables**. Never a `Proposition`,
  never carries a `proof`.
- `Observation` / `Remark` — measured context, caveats, conjectures, clearly
  labelled as such.

In the body prose, mark every claim: "proven", "measured in $N$ runs",
"conjectured", "numerical artefact". An inferred result wearing the grammar of a
theorem is the data-table analogue of stating a conjecture as a theorem — avoid
it.

## 4. The rules of rigour (the hard-won part)

These are the rules that were tested under pressure when the human challenged the
report. They are the reason to trust the final document.

1. **Never report an inferred value as a measured one.** Every table cell is read
   off an actual run / artefact (the trained model, the saved metrics). If a value
   cannot currently be measured, leave the slot explicitly empty ("not measured")
   — never a plausible-looking filler. "Measured $0$" and "argued to be $0$" must
   be typographically distinguishable.
2. **When a claim is challenged, measure before you argue.** The headline
   mechanism of this report was a *geometric* story ("the manifold lies near a
   special line, so the method is redundant there"). The human said the data
   plots did not show it. Rather than defend the prose, the agent loaded the
   datasets and computed the relevant distributions. The geometric story was
   **wrong**; it was replaced. The lesson: a challenged claim is a request for a
   measurement, not for a better sentence.
3. **Be willing to overturn the headline; keep the measured part.** The *empirical
   finding* (method A beats the baseline on case 1 but ties it on case 2) was
   correct and survived. The *explanation* of it was wrong and was rewritten. Keep
   the two separable so that refuting the explanation does not cost the finding.
4. **Formalise the mechanism when possible.** The replacement explanation was a
   proven `Proposition` (the parametrisation can only reach an explicit interval of
   values; the target is reachable iff it lies inside it) plus an `Empirical
   finding` reporting the measured fractions. A structural fact, proven, beats a
   geometric intuition that happens to read well.
5. **A figure must encode only real quantities.** One figure placed a
   field-valued object (a state-dependent parameter) at a *hard-coded* abscissa on
   a 1-D parameter axis, which read as if its ordinate had been sampled off the
   reference curve. It was removed; the honest comparison lived in a different
   plane where both axes are measured. Never give a non-scalar a scalar coordinate
   for visual convenience.
6. **State null and refuted results explicitly.** A tempting physical
   interpretation was tested and *rejected* by a measurement (a deviation was
   larger, not smaller, than the hypothesis predicted). This was written up as a
   `Remark` that says so plainly. Refuted hypotheses are information.
7. **No silent clamps.** Any floor / epsilon / saturation that can change a value
   at runtime must announce itself (warn-once). During the analysis a
   `sigma_min` clamp fired; the report records that the *true* (un-clamped) closure
   was the one deployed, so the results are not a clamp artefact.
8. **Reproducibility is a contract.** Every figure regenerates from saved
   artefacts via a named script; the Reproduction appendix lists script paths,
   dataset directories, and run directories. Figure-generating scripts derive
   their data folder from their own filename and are named after what they
   produce (acronym-free).

## 5. Conventions actively enforced (not just declared)

- **Register**: passive / impersonal voice, no first person ("X is obtained by…",
  not "we obtain"); British English (`normalise`, `behaviour`, `colour`); precise
  terms of art; no colloquialisms, no spatial metaphors for mathematical
  relations, no vague intensifiers (give the factor, not "much better"). This
  applies to prose, captions, and commit messages alike.
- **Notation** (Bourbaki register): derivatives as operators
  ($\partial_x f$, not $f_x$); evaluation point in the argument
  ($\partial_x f(0,y)$, not a restriction bar); single-letter symbols, no
  word-subscripts; a superscript **asterisk** star (`^\ast`, ∗) for the
  reference/optimal quantity ($S^\ast$) — **not** `\star` (⋆), which is a
  binary-operator glyph, not the optimality star; **Landau notation, never `\approx`**, for approximations (`\approx`
  is reserved for the decimal value of a constant); cross-references spelled out
  in full ("Proposition 3.1", "Equation (4)", never "Prop.", "Eq.").
- **The averaging trap**: report per-parameter tables, not only an average. The
  headline effect here was confined to a sub-range and an average hid it. If a
  summary table is wanted, it must not contradict the per-parameter message.
- **Plots**: solid = trained / learned curves; dashed = analytic / reference;
  dotted = auxiliary annotation. A hyperparameter sweep is encoded in **colour**
  (sequential `viridis` / `plasma`), never in stroke. Legends sit **outside** the
  axes.
- **Self-contained source**: the report compiles standalone (`pdflatex`), with no
  dependence on a project notation package — plot data were embedded inline where
  small (`pgfplots`), and macros were defined locally in the preamble.

**Refinements from the `RL_and_signatures` report review (2026-07).** These were
requested explicitly during that report's revision and are binding for future reports:

- **Reserve `Definition`/`Proposition`/`Theorem` for mathematical objects and claims**
  (this sharpens §3). A `Definition` is for a genuine mathematical object — the plant,
  the oracle, the estimators, the representations. A *non*-mathematical object — the
  hypotheses $H_1/H_2$, the list of metrics — belongs in prose (bulleted if helpful),
  **never** inside a `Definition`.
- **Introduce every symbol before use; name, don't paraphrase.** Prefer a named symbol
  to an informal parenthetical: write the optimal cost-to-go as $J^\ast$ and the value
  $V^\ast:=-J^\ast$, **never** $V^\ast=-(\text{cost-to-go})$. Any matrix named in a
  construction (the collocated generator $M$, the control-injection $\mathcal N$, the
  node count $\nu$) is defined where it appears, not left to the code; watch for symbol
  clashes (the collocation count was renamed $N\to\nu$ to avoid the tensor-algebra
  dimension $N(d,L)$).
- **Discretised history as an indexed sequence** (French notation):
  $\underline x=(x_{t-i\Delta t})_{0\le i\le L-1}$, **not** a monolithic capital $W$
  (which reads as a matrix). Keep the indexed structure visible and reuse the same
  symbol everywhere, including the mechanism propositions.
- **State data generation mathematically**, not only in prose: on-sheet windows as
  sampled optimal-trajectory sequences; off-sheet windows as
  $\underline x'=\underline x+\varepsilon\,\Xi$ ($\Xi$ i.i.d. Gaussian, one draw per
  sample and coordinate) relabelled through the oracle at the augmented state
  $z(\underline x')$.
- **Define estimators by their equations**, not by adjectives: the regularised normal
  equations / minimum-norm least squares for the supervised half; the damped LSPI/LSTD
  projected-Bellman update for the learning half.
- **Avoid misleading or coined terminology.** The acceptance test on the oracle is a
  **consistency check**, not a "gate" (misleading). Do not carry code coinages into
  prose: the code's `make_cell` "cell", and "testbed", become **example** — anchored
  once as a *controlled (delay) dynamical system*, with precise adjectives ("dynamical
  system", "delay system", "state dimension") where they add precision.
- **Formal register — replace casual nouns and physical-action verbs.** Not "win/wins"
  (→ *advantage* / *lower cost*), "knob" (→ *parameter*), "headline" (→ *principal
  result*), "drown" (→ *dominate … and suppress*), "starved" (→ *under-sampled*). This
  extends the "no colloquialisms / no spatial metaphors" rule of the Register bullet to
  the specific words that recurred.

## 6. Workflow that produced it

- **Iterate against the rendered PDF.** The human kept a scratch list of points
  (a `draft.md`) and the agent addressed them one by one, recompiling after each.
- **At a framing fork, ask; do not guess.** When the data forced a choice between
  reframings, the agent presented the measured evidence and let the domain expert
  pick the framing, rather than silently committing one.
- **Compile twice; grep for leftovers.** Two `pdflatex` passes, then a check for
  `undefined` / `multiply defined` references and for any forbidden leftover term
  (an abbreviation, a removed concept) before declaring done.
- **Many small commits, descriptive messages, push to the working remote.** One
  commit per logical change (a figure fix; a new figure; the mechanism rewrite).
  Note the repository's remote topology: here `origin` was an Overleaf mirror with
  no non-interactive auth, so pushes went to a separate `github` remote — check
  `git remote -v` before assuming `git push` reaches the right place.

## 7. Minimal porting checklist for `RL_and_signatures`

- [ ] Identify the reference (ground truth / analytic benchmark) and the methods
      compared, including an oracle baseline if one is meaningful.
- [ ] Write the common framework once: object, reference, estimator / learning
      problem, metrics (one symbol each).
- [ ] One section per environment / instance, identical sub-skeleton, **per-
      parameter** tables.
- [ ] Separate `Proposition`(proven) from `Empirical finding`(measured) in any
      mechanism section.
- [ ] Every number traceable to a run; reproduction appendix with script and run
      paths.
- [ ] Plot, notation, and register conventions as in §5.
- [ ] Before claiming a mechanism: measure it; be ready to overturn the headline
      while keeping the measured finding.

*If a claim in the report cannot survive someone loading the data and
recomputing it, it does not belong in the report. That is the whole of the
method.*
