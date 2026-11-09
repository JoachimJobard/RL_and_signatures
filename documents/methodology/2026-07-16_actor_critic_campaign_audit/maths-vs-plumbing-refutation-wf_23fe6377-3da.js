export const meta = {
  name: 'maths-vs-plumbing-refutation',
  description: 'Deduplicate the 94 cached audit items, refute each by execution against the campaign invocation, and return a launch decision',
  phases: [
    { title: 'Merge', detail: 'dedupe 94 items from five lenses into canonical defects' },
    { title: 'Refute', detail: 'three diverse skeptics per defect: reproduce, campaign-reachability, agent-asymmetry' },
    { title: 'Report', detail: 'ranked register plus an explicit launch decision' },
  ],
}

const REPO = '/home/rho/Documents/work/phd/git_repositories/RL_and_signatures'
const PY = `${REPO}/.venv/bin/python`
const ITEMS = args.itemsFile

const CONTEXT = `
REPOSITORY: ${REPO}  (branch scientific-workflow-refactor, HEAD 0e457c2)
PYTHON: ${PY} — there is NO bare 'python' on PATH. Always run from ${REPO}.

PURPOSE. An actor-critic campaign of 150 SLURM tasks (5 cells x 3 representations x 5 seeds x 2
agents) is about to be launched on Jean Zay. This audit decides whether it is safe to launch. A
defect that changes a reported number, or that differs BETWEEN THE TWO AGENTS at fixed
representation, wastes all 150 tasks. That is the decision this work serves.

THE CAMPAIGN'S EXACT INVOCATION, quoted verbatim from experiments/h1h2_evaluation/jeanzay/rl_array.slurm:125-133:

    python main_unified.py \\
        agent="$AGENT" env="$ENV" $REPARG \\
        agent.signature.window_size="$WIN" \\
        agent.signature.force_signature_window=true \\
        agent.training.divergence_threshold=100.0 \\
        agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$MAXT" \\
        eval.x0_test="$X0" eval.T_sim="$TSIM" \\
        seed="$SEED" experiment_group="$EXPERIMENT_GROUP" \\
        run_tag="\${AGENT}_\${CELL}_\${REP}" debug="$DEBUG" wandb.mode=offline

with AGENT in {value_gradient, signatures}, and REPARG one of:
    markovian)   agent.signature.kind=markovian   agent.signature.degree=2
    raw_history) agent.signature.kind=raw_history agent.signature.degree=2
    signature)   agent.signature.kind=signature   agent.signature.depth=2

THE FIVE CAMPAIGN CELLS and their windows (cell -> ENV, WIN, MAXT, TSIM, X0):
    markovian          -> double_integrator,  WIN=10, MAXT=15, TSIM=15, X0=[1.0,0.5]
    linear_dde         -> linear_dde_scalar,  WIN=8,  MAXT=20, TSIM=20, X0=[0.8]
    hopfield_linear    -> hopfield_linear,    WIN=5,  MAXT=15, TSIM=15, X0=[0.5,0.0]
    hopfield_nonlinear -> hopfield_nonlinear, WIN=5,  MAXT=15, TSIM=15, X0=[0.5,0.0]
    mg_chaotic         -> MG_1D_chaotic,      WIN=71, MAXT=60, TSIM=60, X0=[0.8]

GROUND TRUTH ALREADY MEASURED THIS SESSION — do not re-derive, but you MAY falsify:
- Every agent config sets training.scale = 1.0, and conf/agent/signatures.yaml:72 sets
  network.normalize_entries = false. signatures_jax.py:93 sets training.scale =
  training.divergence_threshold ONLY when normalize_entries is true. Therefore any missing
  1/scale chain-rule factor in the control law is LATENT, NOT ARMED, in this campaign
  (scale = 1 makes the factor unity). It remains a foot-gun a future config would arm.
- The array FORCES agent.signature.depth=2 for the signature representation, so the
  depth: 3 default in conf/agent/signatures.yaml is OVERRIDDEN and irrelevant to the campaign.
- The array does NOT pass agent.training.clip_action. Each agent therefore uses its own default:
  conf/agent/value_gradient.yaml:14 clip_action: null (NO clipping) versus
  conf/agent/signatures.yaml:16 clip_action: 10.0 (clipping at 10).
  This is a CANDIDATE FOURTH AGENT-ASYMMETRY CONFOUND. One cached auditor reported clip_action
  binding on 6.0% of actions at the shipped value; that figure is UNVERIFIED — measure it.

THE THREE CONFOUNDS ALREADY CLOSED (do not re-report them as open; DO report any residue):
  window (force_signature_window=true with explicit per-cell WIN), divergence_threshold (100.0
  passed explicitly to both agents), and the actor-critic preheat (retired; the DDE's initial
  path phi is now loaded).

THE MATHEMATICS THE CODE IS SUPPOSED TO IMPLEMENT.
- Plant: a controlled delay differential equation xdot(t) = f(x_t) + g(x_t) u(t), state
  x_t : s -> x(t+s), s in [-tau, 0], an element of a path space. Its initial condition is a PATH phi.
- Greedy control (Doya 2000, functional case): u* = -(1/2) R^{-1} g(x_t)^T grad_{x(t)} V(x_t),
  where grad_{x(t)}V is the derivative with respect to the window ENDPOINT (Dupire's vertical
  derivative).
- Critic: continuous-time temporal difference, delta = r - V/tau_discount + dV/dt.

CLASSIFICATION (exactly one per defect):
  (A) MATHEMATICALLY INERT — provably cannot change the answer. Acceptable with a comment.
  (B) CHANGES THE ANSWER — a clamp/floor/normalisation/regulariser that can fire on real inputs.
  (C) WRONG OBJECT — the code computes a different mathematical object from the one it claims.
  (D) NECESSARY DISCRETISATION — legitimate, but the arithmetic must be right.

A CONSTRAINT THAT OVERRIDES ANY RECOMMENDATION: the VALUE GRADIENT MUST STAY BIT-UNCHANGED. The
project owner's existing 5-seed H1/H2 value-gradient results must remain reproducible, INCLUDING
the float32 cast in value_gradient_jax._fill_buffer_initial, which stays for now by explicit
decision. This audit is a REGISTER, not a purge. Every recommendation must state whether applying
it would invalidate the existing value-gradient results.

RULES:
- READ-ONLY. Do NOT modify ANY tracked file. Do NOT git commit, push, or submit any SLURM job.
  Scratch scripts go to a temp directory only; clean up any _debug_ directories you create.
- GROUND EVERY CLAIM IN EXECUTION or a quoted file line. "Never report an inferred value as a
  measured one" — if a clamp's firing rate is claimed, MEASURE it; otherwise write "not measured".
- Verify the EFFECT, not the artefact. A config default is not proof of runtime behaviour: resolve
  it. An AC-vs-AC probe cannot certify an AC-vs-VG property.
- Register: impersonal voice, British English, precise terminology, no first person.
`

phase('Merge')

const MERGED = {
  type: 'object',
  additionalProperties: false,
  required: ['defects', 'merge_notes'],
  properties: {
    defects: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'name', 'location', 'claim', 'classification', 'severity', 'n_sources'],
        properties: {
          id: { type: 'string', description: 'short kebab-case stable id' },
          name: { type: 'string' },
          location: { type: 'string', description: 'file:line' },
          claim: { type: 'string', description: 'the defect in two or three precise sentences, merged from all sources' },
          classification: { type: 'string', description: 'A INERT | B CHANGES-THE-ANSWER | C WRONG-OBJECT | D DISCRETISATION' },
          severity: { type: 'string' },
          n_sources: { type: 'number', description: 'how many of the 94 raw items merged into this defect' },
        },
      },
    },
    merge_notes: { type: 'string' },
  },
}

const merged = await agent(`${CONTEXT}

Five independent auditors swept this repository under five lenses (clamps, control law, window,
signature augmentation, estimator). Two runs of that sweep produced 94 raw items in total, with
heavy duplication across lenses — the same defect (for instance the window's unexplained "+3"
taps) is reported by several lenses in different words. The raw items are at:

  ${ITEMS}

Read that file in full. Merge the 94 raw items into a canonical, DEDUPLICATED list of DISTINCT
defects. Two items are the same defect when they concern the same code path and the same
mathematical object, however differently worded. Preserve the strongest and most specific
formulation, and merge the evidence.

Assign each a short stable kebab-case id. Set n_sources to the number of raw items merged.
Do not drop an item merely because it is unverified — verification is the next phase's job.
Do not invent defects absent from the file.
Order the output by severity, most severe first.`,
  { label: 'merge:dedupe', phase: 'Merge', schema: MERGED, effort: 'high' })

const defects = merged?.defects ?? []
log(`${defects.length} canonical defects merged from 94 raw items; refuting each under three lenses.`)

phase('Refute')

const VERDICT = {
  type: 'object',
  additionalProperties: false,
  required: ['exists', 'armed_in_campaign', 'agent_asymmetric', 'blocks_launch', 'evidence', 'corrected_classification', 'vg_bit_risk'],
  properties: {
    exists: { type: 'boolean', description: 'true only if reproduced by execution or an exact quoted line at HEAD' },
    armed_in_campaign: { type: 'string', description: 'ARMED | LATENT | UNREACHABLE, with the measured reason' },
    agent_asymmetric: { type: 'string', description: 'Does it differ between AGENT=value_gradient and AGENT=signatures at fixed representation, or across the three representations? YES/NO plus the measured evidence.' },
    blocks_launch: { type: 'boolean', description: 'true if launching 150 tasks now would produce a corrupted or confounded number because of this defect' },
    evidence: { type: 'string', description: 'exactly what was executed or quoted, and the numbers obtained. Write "not measured" where not measured.' },
    corrected_classification: { type: 'string', description: 'A|B|C|D, corrected if the auditor mis-classified' },
    vg_bit_risk: { type: 'string', description: 'Would fixing this change the value gradient bit-for-bit and thus invalidate the existing 5-seed results? YES/NO/NOT-APPLICABLE plus why.' },
  },
}

const REFUTE_LENSES = [
  { key: 'reproduce', ask: `LENS ONE — DOES IT EXIST AT ALL? Attempt to REFUTE the defect by execution. Default to exists=false
unless you reproduce it against the code at HEAD. Quote the exact lines. If the auditor's line
numbers are stale or the described code does not exist, say so and set exists=false. If the
auditor asserted a number (a firing rate, a residual, a dimension), RE-MEASURE it: an auditor's
figure is an assertion until reproduced.` },

  { key: 'campaign', ask: `LENS TWO — IS IT ARMED IN THIS CAMPAIGN? The only configuration that matters is the array's exact
invocation quoted above: the five cells, depth=2, force_signature_window=true with explicit
per-cell windows, divergence_threshold=100.0, both agents. RESOLVE the config rather than reading
a default — Hydra composition and any code that overrides a value at runtime both matter. An
unreachable defect is NOT a defect for the launch decision, but record it at LOW as a foot-gun a
future config would arm, and say precisely which flag would arm it.` },

  { key: 'asymmetry', ask: `LENS THREE — IS IT A CONFOUND? This is the decisive lens for the launch decision. Establish, BY
EXECUTION, whether the defect acts DIFFERENTLY on AGENT=value_gradient than on AGENT=signatures at
FIXED representation, or differently across the three representations at fixed agent. A defect that
acts identically on both agents cancels in the comparison; a defect that acts on one and not the
other is a confound and corrupts the campaign's central claim.
Note carefully: an actor-critic-versus-actor-critic probe CANNOT certify an actor-critic-versus-
value-gradient property. Compare the two AGENTS. The clip_action asymmetry (null versus 10.0) is
the prime suspect: if it is real, MEASURE the binding rate per agent per cell on a short run, and
report the fraction of actions clipped and the most extreme pre-clip |u|.` },
]

const verdicts = await pipeline(
  defects,
  (d) => parallel(REFUTE_LENSES.map(l => () =>
    agent(`${CONTEXT}

A canonical defect, merged from ${d.n_sources} independent auditor report(s):

${JSON.stringify(d, null, 2)}

The full raw auditor items behind it are in ${ITEMS} if you need their detail.

${l.ask}

READ-ONLY. Do not modify any tracked file, commit, push, or submit any job.
Ground every claim in execution or a quoted line. Write "not measured" rather than a plausible
filler. Impersonal voice, British English.`,
      { label: `refute:${l.key}:${d.id}`, phase: 'Refute', schema: VERDICT, effort: 'high' })
  )).then(vs => ({ defect: d, verdicts: vs.filter(Boolean) }))
)

const settled = verdicts.filter(Boolean)
const surviving = settled.filter(v => v.verdicts.some(x => x.exists))
const blocking = settled.filter(v => v.verdicts.some(x => x.blocks_launch && x.exists))
log(`${surviving.length} of ${defects.length} defects reproduced; ${blocking.length} flagged as launch-blocking by at least one skeptic.`)

phase('Report')

const REPORT = {
  type: 'object',
  additionalProperties: false,
  required: ['headline', 'launch_decision', 'launch_rationale', 'blocking_defects', 'wrong_object', 'changes_the_answer', 'inert_but_undocumented', 'necessary_discretisations', 'false_alarms', 'markdown_report'],
  properties: {
    headline: { type: 'string', description: 'The single most important thing the owner must know, in one sentence' },
    launch_decision: { type: 'string', description: 'LAUNCH | FIX-FIRST | LAUNCH-WITH-CAVEAT' },
    launch_rationale: { type: 'string', description: 'Why, in terms of measured evidence only' },
    blocking_defects: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['id', 'why_blocking', 'minimal_fix', 'vg_bit_risk'], properties: { id: { type: 'string' }, why_blocking: { type: 'string' }, minimal_fix: { type: 'string', description: 'the smallest change that closes it, ideally one flag in the array' }, vg_bit_risk: { type: 'string' } } } },
    wrong_object: { type: 'array', items: { type: 'string' } },
    changes_the_answer: { type: 'array', items: { type: 'string' } },
    inert_but_undocumented: { type: 'array', items: { type: 'string' } },
    necessary_discretisations: { type: 'array', items: { type: 'string' } },
    false_alarms: { type: 'array', items: { type: 'string' } },
    markdown_report: { type: 'string', description: 'The full report as a self-contained Markdown document body, no title, ready to write to documents/methodology/. Impersonal, British, precise. Tables where they help. Every number traceable and marked measured or not measured.' },
  },
}

const report = await agent(`${CONTEXT}

Every canonical defect, each attacked by three independent skeptics under distinct lenses
(does it exist / is it armed in this campaign / is it an agent-asymmetric confound):

${JSON.stringify(settled, null, 2)}

Write the review the project owner asked for: "a review of those plumbing things ... we want to
stick with the maths as most as possible."

THE DECISION THIS REPORT SERVES: whether to launch 150 SLURM tasks now. State launch_decision as
LAUNCH, FIX-FIRST, or LAUNCH-WITH-CAVEAT, and justify it from measured evidence alone. A defect
blocks the launch only if a skeptic REPRODUCED it AND it is ARMED in the array's invocation AND it
either corrupts a reported number or acts asymmetrically between the two agents. Rank the blocking
defects by how cheaply they close — a defect closed by one flag in rl_array.slurm is not a reason
to delay a campaign, it is a one-line edit.

Organise the body by CLASSIFICATION, not by file. A defect a skeptic refuted goes in false_alarms,
not in the body. Do not pad. Be blunt about what is merely tidy versus what corrupts a result.
Respect the standing constraint: the value gradient must stay bit-unchanged, so mark every
recommendation with whether it would invalidate the existing 5-seed value-gradient results.
Every number measured and traceable; write "not measured" where it is not, never a plausible filler.
State claim strength in plain words. Impersonal voice, British English, no first person.`,
  { label: 'report:launch-decision', phase: 'Report', schema: REPORT, effort: 'max' })

return {
  report,
  n_defects: defects.length,
  n_reproduced: surviving.length,
  n_blocking: blocking.length,
}
