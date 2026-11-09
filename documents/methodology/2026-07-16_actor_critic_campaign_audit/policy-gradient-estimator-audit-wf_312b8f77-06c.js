export const meta = {
  name: 'policy-gradient-estimator-audit',
  description: 'Adversarially audit the newly-written Monte-Carlo policy gradient: its estimator, its config inheritance, and the tests that certify it',
  phases: [
    { title: 'Attack', detail: 'four independent lenses over the new estimator' },
    { title: 'Verdict', detail: 'synthesis: is the estimator correct and are its tests adequate' },
  ],
}

const REPO = '/home/rho/Documents/work/phd/git_repositories/RL_and_signatures'
const PY = `${REPO}/.venv/bin/python`

const CONTEXT = `
REPOSITORY: ${REPO} (branch scientific-workflow-refactor)
PYTHON: ${PY} -- there is NO bare 'python' on PATH. Always run from ${REPO}.

WHAT IS UNDER AUDIT. Commit 5b410d6 added a THIRD learner: a Monte-Carlo policy gradient
(REINFORCE with a value baseline), selected by \`algorithm.actor_target: monte_carlo\`, configured
by \`conf/agent/policy_gradient.yaml\`, implemented inside \`src/agents/signatures_jax.py\`, and
tested by \`test/agents/test_policy_gradient.py\` (13 tests, all passing).

WHY THIS AUDIT EXISTS, stated plainly: the code AND its tests were written by the same author in
the same sitting, so they share his blind spots by construction. Thirteen green tests are that
author marking his own homework. Earlier the same day that author shipped a test whose tolerance
was hard-coded rather than derived, and it only surfaced because it happened to fail; and he
reported a clamp's binding rate from a 20-episode probe against a 1000-episode campaign, calling
"not observed" a null result. Assume that standard of care and attack accordingly.

THE MATHEMATICS IT CLAIMS TO IMPLEMENT.
- Plant: a controlled delay differential equation xdot(t) = f(x_t) + g(x_t) u(t); the state x_t is
  the path on [-tau, 0]; the environment's reward is a RATE, r = -(x'Qx + u'Ru).
- Objective: J = integral of r over the episode. The rest of the code forms it as reward * dt.
- Policy: Gaussian, u = mu(s; w) + n, with n ~ N(0, sigma^2). The score with respect to the mean
  is grad_mu log N(u; mu, sigma^2) = (u - mu)/sigma^2 = noise/sigma^2.
- REINFORCE: the actor ascends J via E[score * A_t], with A_t = R_t - V(s_t) and the return-to-go
  R_t = int_t^T r ds, estimated as sum_{k>=t} r_k dt. The baseline V(s_t) must depend on the STATE
  only, never on the action, or it biases the gradient.
- The reference algorithm for the sibling actor-critic is Doya (2000) Equation 20,
  wdot^A = eta^A delta(t) n(t) dA/dw, implemented at signatures_jax.py:633-636.

RELEVANT MEASURED CONTEXT (established this session; you may falsify any of it):
- Both existing learners are matched on: window (force_signature_window=true, explicit per-cell
  WIN), divergence_threshold (100.0), initial path phi, clip_action (null, both), critic_lr (1e-3),
  noise.length_scale (0.002), noise.sigma (0.1), noise.schedule (constant), discount.tau (1.0).
- discount.discounted is FALSE and algorithm.lstd is FALSE in the campaign, so discount.tau is
  inert on every live path.
- The exploration noise is numerically WHITE at the shipped length_scale (adjacent-step
  correlation 0.000 at dt=0.1), despite noise.smooth=true. Doya's own noise is low-pass filtered
  (Ornstein-Uhlenbeck, tau_n = 1.0) and therefore correlated. This is a KNOWN deviation.
- UNRESOLVED and inherited: the value gradient bootstraps V_next from a Polyak-averaged TARGET
  network (value_gradient_jax.py:261, tau_polyak 0.01); the actor-critic (and hence this policy
  gradient) bootstraps from the LIVE critic. Do not re-report this as new; DO report anything it
  implies for the new estimator specifically.

RULES:
- READ-ONLY. Do NOT modify any tracked file. Do not git commit, push, or submit any SLURM job.
  Scratch work goes to a temp directory; clean up any _debug_ directories you create.
- GROUND EVERY CLAIM IN EXECUTION or an exactly quoted line. Never report an inferred value as a
  measured one; write "not measured" rather than a plausible filler.
- A probe's null is only as strong as its horizon. If you assert a quantity is zero/absent, state
  the horizon over which you looked.
- Register: impersonal voice, British English, precise terminology, no first person.
`

const FINDING = {
  type: 'object',
  additionalProperties: false,
  required: ['findings', 'summary'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'location', 'claim', 'severity', 'evidence', 'is_real', 'fix'],
        properties: {
          name: { type: 'string' },
          location: { type: 'string', description: 'file:line' },
          claim: { type: 'string', description: 'the defect, precisely' },
          severity: { type: 'string', description: 'critical | high | medium | low' },
          evidence: { type: 'string', description: 'exactly what was executed or quoted, with numbers. "not measured" where not measured.' },
          is_real: { type: 'boolean', description: 'true only if reproduced by execution or exact quotation' },
          fix: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

phase('Attack')

const LENSES = [
  { key: 'estimator', ask: `LENS ONE -- IS THE ESTIMATOR THE POLICY GRADIENT IT CLAIMS TO BE?
Read \`_make_monte_carlo_actor_update_fn\`, \`monte_carlo_returns\` and \`_monte_carlo_actor_update\`
in src/agents/signatures_jax.py. Establish BY EXECUTION:
- Is the SIGN right? The loss is \`-advantage * sum(log_prob_grad * mu)\` and the optimiser performs
  gradient DESCENT. Work out whether the resulting parameter step ASCENDS the return J. The reward
  is NEGATIVE (a cost), so a sign error would look like plausible-but-anti-optimal learning. Verify
  numerically against a case with a known answer -- do not reason it out on paper alone.
- Is the return-to-go the right Riemann sum? \`jnp.flip(jnp.cumsum(jnp.flip(rates))) * dt\` claims
  R_t = sum_{k>=t} r_k dt. Check the ALIGNMENT: is R_t the return from the step whose sig_t/noise
  are paired with it, or is it off by one (i.e. does index t of the returns correspond to index t
  of the recorded sig/noise)? An off-by-one here regresses each action against the next action's
  return and is undetectable from a training curve.
- Is the baseline action-independent? V(s_t) is recorded as \`ctx.V_t\`. Establish whether ctx.V_t
  is computed from the state BEFORE the action was applied. If it were computed after, it would
  depend on the action and BIAS the gradient. Quote the lines.
- Does \`jax.vmap(lambda s: actor.apply(actor_params, s))(sig_batch)\` agree with the unbatched
  \`actor.apply(actor_params, sig)\` used by the temporal-difference path? Test it numerically.` },

  { key: 'buffer', ask: `LENS TWO -- THE EPISODE BUFFER AND THE INTERACTION WITH TRUNCATION.
The Monte-Carlo path records one entry per step and consumes them at episode end.
- The divergence cut (signatures_jax.py:_is_episode_done, ||x|| > 100) ENDS AN EPISODE EARLY. What
  does the return-to-go mean for a truncated episode? R_t is then the integral to the CUT, not to
  the horizon T. Is that the intended object? A truncated episode's return is not the episode's
  return; establish whether the estimator silently treats one as the other, and what that does to
  the actor's signal on a diverging cell. MEASURE it on double_integrator/raw_history, where the
  actor-critic is known to diverge (peak ||x|| 133, cut fires).
- Is the buffer emptied on EVERY path that starts an episode? Find every place an episode begins
  (including any evaluation rollout, and any early \`continue\`/exception path) and establish
  whether \`_on_episode_start\` is reached in all of them. A leak attributes one episode's rewards
  to another episode's actions.
- Does the EVALUATION rollout (\`_evaluate_noiseless\`) append to the buffer? It calls the action
  path; if it also records, evaluation would contaminate the actor's training signal. Check.
- Memory: the buffer holds every step of an episode. At mg_chaotic (window 71, 240 steps, feature
  dimension possibly in the hundreds) x 1000 episodes, is there an unbounded growth or a leak
  across episodes? Measure the buffer length at episode end.` },

  { key: 'tests', ask: `LENS THREE -- ARE THE TESTS ADEQUATE, OR DO THEY CERTIFY NOTHING?
Read test/agents/test_policy_gradient.py (13 tests, all passing). Attack the TESTS, not the code.
- Which of the 13 would still pass if the estimator's SIGN were flipped? Try it: actually mutate a
  copy of the source in a temp directory, flip the sign, and run the suite against it. Any test
  that still passes is not testing what its name claims. Report the count.
- Which would still pass with an OFF-BY-ONE in the return alignment? Same method: mutate and run.
- Which would still pass if the baseline were made action-DEPENDENT?
- Which would still pass if the buffer were never reset between episodes?
- The unbiasedness test does NOT exercise the agent's code at all -- it re-implements the estimator
  in numpy inside the test and checks THAT against the analytic gradient. So it verifies the
  AUTHOR'S ALGEBRA, not the SHIPPED CODE. Confirm or refute this reading, and if confirmed, say
  plainly what it means: the property could hold in the test and fail in the agent.
This lens's output is the most valuable if it is blunt. A test that cannot fail is worse than no
test, because it is cited as evidence.` },

  { key: 'science', ask: `LENS FOUR -- DOES THIS LEARNER DELIVER THE SCIENTIFIC CLAIM MADE FOR IT?
The claim in commit 5b410d6 is that the two existing learners "both obtain their control by reading
the value function", so an H1 verdict surviving a learner that "never bootstraps" is "a statement
about the REPRESENTATION -- strictly stronger".
- Is that claim TRUE of the implementation as written? The critic is still used as the baseline
  V(s_t), and the critic reads the representation. So the representation enters the actor's signal
  through the baseline as well as through the actor's own input. Does that undermine the "no
  bootstrap" claim? Distinguish carefully: a baseline affects VARIANCE, not the expectation --
  but establish whether that is true HERE, given the critic is learned and hence correlated with
  the trajectory it is baselining. State whether the claim needs weakening, and to what exactly.
- The exploration noise is white (measured). The actor learns ONLY through the noise
  (score = noise/sigma^2). Establish, by execution rather than by argument, whether the
  Monte-Carlo actor's learning signal survives white noise, and how it compares with the
  temporal-difference actor's. NOTE: an earlier analysis claiming the TD actor's signal "vanishes"
  under white noise was WITHDRAWN when the algebra showed Vdot divides by dt and restores the
  scale. Do not repeat that error; do the algebra AND the measurement.
- One update per episode against ~150-240 steps per episode, versus the actor-critic's per-step
  update: is the effective learning rate comparable, or is this learner handicapped by a factor of
  the episode length purely by construction? That would be a confound in any three-learner
  comparison. Quantify it.` },
]

const attacks = await parallel(LENSES.map(l => () =>
  agent(`${CONTEXT}\n\n${l.ask}\n\nGROUND EVERY FINDING IN EXECUTION or an exactly quoted line.
READ-ONLY. Default to is_real=false unless you reproduce it.`,
    { label: `pg-audit:${l.key}`, phase: 'Attack', schema: FINDING, effort: 'high' })
))

const all = attacks.filter(Boolean).flatMap((a, i) => a.findings.map(f => ({ ...f, lens: LENSES[i]?.key })))
log(`${all.length} findings across ${attacks.filter(Boolean).length} lenses; ${all.filter(f => f.is_real).length} reproduced.`)

phase('Verdict')

const REPORT = {
  type: 'object',
  additionalProperties: false,
  required: ['headline', 'estimator_is_correct', 'tests_are_adequate', 'blocking', 'markdown_report'],
  properties: {
    headline: { type: 'string' },
    estimator_is_correct: { type: 'string', description: 'YES | NO | UNPROVEN, with the measured reason' },
    tests_are_adequate: { type: 'string', description: 'what the 13 tests do and do not certify, bluntly' },
    blocking: { type: 'array', items: { type: 'object', additionalProperties: false, required: ['name', 'why', 'fix'], properties: { name: { type: 'string' }, why: { type: 'string' }, fix: { type: 'string' } } } },
    markdown_report: { type: 'string', description: 'self-contained Markdown body, no title. Impersonal, British, precise. Every number measured or marked "not measured".' },
  },
}

const report = await agent(`${CONTEXT}

Findings from four independent adversarial lenses over the new estimator:
${JSON.stringify(all, null, 2)}

Write the verdict. Answer two questions without hedging:
1. IS THE ESTIMATOR CORRECT? YES / NO / UNPROVEN, from measured evidence alone.
2. ARE ITS 13 TESTS ADEQUATE, or do they certify less than they appear to? Be blunt: a test that
   cannot fail is worse than no test, because it is cited as evidence.
A finding no lens reproduced goes in neither list. Rank the blocking items by how cheaply they
close. Impersonal voice, British English, no first person.`,
  { label: 'pg-audit:verdict', phase: 'Verdict', schema: REPORT, effort: 'max' })

return { report, n_findings: all.length, n_real: all.filter(f => f.is_real).length }
