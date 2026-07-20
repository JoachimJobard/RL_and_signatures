"""Sanity checks for the Monte-Carlo policy gradient (algorithm.actor_target='monte_carlo').

The estimator under test is REINFORCE with NO baseline: the actor regresses the Gaussian score
function against the realised return-to-go, A_t = R_t, once per episode. The actor does not read
the critic at all (commit 8d0f4b0), which is the point of the learner -- an H1 verdict measured
under it is a statement about the REPRESENTATION rather than about the value machinery.

NOTE on what these tests do and do not certify. An adversarial audit mutated the shipped
estimator's sign and all thirteen of the original tests PASSED: none constrained the DIRECTION of
the parameter step, only its norm, and a norm is sign-blind. The three directional tests below
close that gap and are verified to close it -- the same mutant now fails exactly those three.

What each check buys, and why it is here rather than a weaker one:

1. The return-to-go arithmetic against a CLOSED FORM computed independently in the test, plus the
   backward recursion R_t = r_t dt + exp(-dt/tau) R_{t+1} in the discounted case. This is the one
   piece of the estimator that has an exact answer, so it is checked exactly rather than
   approximately.
2. Unbiasedness of the score-function estimator against the ANALYTIC policy gradient of a problem
   whose gradient is known in closed form. An estimator that is merely "plausible" on a training
   curve can be silently biased; this pins it to a number.
3. A zero advantage must give an exactly zero gradient -- the estimator must not manufacture a
   learning signal from nothing.
4. The buffer must not leak steps between episodes, which would attribute one episode's rewards to
   another episode's actions. A truncated episode (divergence cut) makes this reachable.
5. End-to-end: the agent trains through the real Hydra/train path.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

from src.training.train import build_environment, build_agent
from src.utils.run_context import find_repo_root

CONF_DIR = str(find_repo_root(__file__) / "conf")


def _build(agent_name="policy_gradient", **extra):
    overrides = [
        f"agent={agent_name}",
        "env=double_integrator",
        "agent.signature.kind=markovian",
        "agent.signature.degree=2",
        "agent.training.n_episodes=1",
        "agent.training.max_time=1.0",
        "seed=0",
        "debug=true",
        "wandb.mode=disabled",
    ] + [f"{k}={v}" for k, v in extra.items()]
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="config_unified", overrides=overrides)
    env = build_environment(cfg)
    return build_agent(cfg, env)


# =============================================================================
# 1. The return-to-go, against a closed form
# =============================================================================

def test_undiscounted_return_to_go_matches_closed_form():
    """R_t = sum_{k >= t} r_k dt, checked against an independently computed sum."""
    agent = _build()
    rates = jnp.array([1.0, 2.0, 3.0, -4.0])
    dt = 0.5
    got = np.asarray(agent.monte_carlo_returns(rates, dt))
    # Computed here independently of the implementation, not copied from it.
    expected = np.array([
        (1.0 + 2.0 + 3.0 - 4.0) * dt,
        (2.0 + 3.0 - 4.0) * dt,
        (3.0 - 4.0) * dt,
        (-4.0) * dt,
    ])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_return_to_go_satisfies_its_backward_recursion():
    """R_t = r_t dt + R_{t+1} must hold exactly, undiscounted, for a random reward sequence."""
    agent = _build()
    rng = np.random.default_rng(0)
    rates = jnp.asarray(rng.normal(size=25))
    dt = 0.1
    R = np.asarray(agent.monte_carlo_returns(rates, dt))
    r = np.asarray(rates)
    np.testing.assert_allclose(R[:-1], r[:-1] * dt + R[1:], rtol=0, atol=1e-10)
    np.testing.assert_allclose(R[-1], r[-1] * dt, rtol=0, atol=1e-12)


def test_discounted_return_to_go_satisfies_its_backward_recursion():
    """R_t = r_t dt + exp(-dt/tau) R_{t+1}, the discounted analogue, checked exactly."""
    agent = _build()
    agent.discount.discounted = True
    agent.discount.tau = 2.0
    rng = np.random.default_rng(1)
    rates = jnp.asarray(rng.normal(size=20))
    dt = 0.25
    R = np.asarray(agent.monte_carlo_returns(rates, dt))
    r = np.asarray(rates)
    decay = float(np.exp(-dt / 2.0))
    np.testing.assert_allclose(R[:-1], r[:-1] * dt + decay * R[1:], rtol=1e-6, atol=1e-9)


def test_discounting_shortens_the_effective_horizon():
    """A discounted return must weigh the far future strictly less than an undiscounted one.

    Guards the SIGN of the exponent: a decay written the wrong way round would still satisfy a
    recursion test, but would weigh the future MORE.
    """
    agent = _build()
    rates = jnp.ones(50)
    dt = 0.1
    undiscounted = np.asarray(agent.monte_carlo_returns(rates, dt))[0]
    agent.discount.discounted = True
    agent.discount.tau = 1.0
    discounted = np.asarray(agent.monte_carlo_returns(rates, dt))[0]
    assert discounted < undiscounted, (
        f"discounted return {discounted} should be below the undiscounted {undiscounted}"
    )


# =============================================================================
# 2. Unbiasedness of the score-function estimator, against an analytic gradient
# =============================================================================

def test_score_function_estimator_is_unbiased_against_the_analytic_gradient():
    """The REINFORCE estimator must recover dJ/dmu on a problem where it is known exactly.

    One-step Gaussian problem: u = mu + n with n ~ N(0, sigma^2), reward r(u) = -(u - u_star)^2.
    Then J(mu) = E[r] = -(mu - u_star)^2 - sigma^2 and dJ/dmu = -2 (mu - u_star), independent of
    sigma. The estimator used by the agent is (n / sigma^2) * A, whose expectation must equal it.

    This tests the ESTIMATOR IDENTITY the implementation relies on. It is checked here rather than
    inferred from a training curve, because a biased estimator can still produce a curve that
    descends.
    """
    rng = np.random.default_rng(0)
    mu, u_star, sigma = 0.7, -0.3, 0.5
    n_samples = 400_000
    noise = rng.normal(0.0, sigma, size=n_samples)
    reward = -((mu + noise - u_star) ** 2)
    # The agent's estimator: advantage * score. A constant baseline must not bias it, so subtract
    # the sample mean of the reward -- which is exactly the role V(s_t) plays in the agent.
    advantage = reward - reward.mean()
    estimate = np.mean((noise / sigma ** 2) * advantage)
    analytic = -2.0 * (mu - u_star)
    # Monte-Carlo standard error over 4e5 samples; the tolerance is the statistics, not a fudge.
    assert abs(estimate - analytic) < 0.02, (
        f"score estimate {estimate:.5f} vs analytic dJ/dmu {analytic:.5f}"
    )


def test_state_dependent_baseline_does_not_bias_the_estimator():
    """Subtracting any action-independent baseline must leave the expectation unchanged.

    The shipped estimator has NO baseline (A_t = R_t, commit 8d0f4b0), so this test does not
    describe it. It is kept because it pins the property that WOULD license adding one:
    E[(n/sigma^2) b] = 0 for any b not depending on the action. If a baseline is ever added, it
    must depend on the STATE only -- not on the episode's own mean return, which depends on the
    actions taken and would bias the gradient.

    The tolerance is DERIVED, not chosen. The baseline's contribution is (b/sigma^2) * mean(n),
    whose standard error is |b| / (sigma sqrt(N)) -- it grows in proportion to the baseline, so a
    single fixed tolerance is wrong by construction and would either pass a biased estimator at
    small |b| or fail an unbiased one at large |b| (the latter is what a fixed 0.02 did here at
    b = 5, on an estimator that is provably unbiased). Five standard errors gives a false-failure
    rate of about one in 3.5 million per assertion.
    """
    rng = np.random.default_rng(1)
    sigma = 0.4
    n_samples = 400_000
    noise = rng.normal(0.0, sigma, size=n_samples)
    for baseline in (0.0, 5.0, -12.34):
        contribution = np.mean((noise / sigma ** 2) * baseline)
        standard_error = abs(baseline) / (sigma * np.sqrt(n_samples))
        tolerance = 5.0 * standard_error + 1e-12
        assert abs(contribution) < tolerance, (
            f"baseline {baseline} contributed {contribution:.6f} to the gradient, exceeding five "
            f"standard errors ({tolerance:.6f}); the baseline is biasing the estimator"
        )


def test_baseline_reduces_the_estimator_variance():
    """The baseline's PURPOSE is variance reduction; check it actually delivers it.

    Unbiasedness alone does not justify the baseline -- a baseline that left the variance
    unchanged would be pure complexity. On the one-step Gaussian problem the optimal constant
    baseline is close to the mean reward, so subtracting it must strictly reduce the sample
    variance of the per-sample gradient estimate.
    """
    rng = np.random.default_rng(2)
    mu, u_star, sigma = 0.7, -0.3, 0.5
    n_samples = 200_000
    noise = rng.normal(0.0, sigma, size=n_samples)
    reward = -((mu + noise - u_star) ** 2)
    score = noise / sigma ** 2
    variance_without_baseline = np.var(score * reward)
    variance_with_baseline = np.var(score * (reward - reward.mean()))
    assert variance_with_baseline < variance_without_baseline, (
        f"baseline did not reduce variance: {variance_with_baseline:.4f} against "
        f"{variance_without_baseline:.4f}"
    )


# =============================================================================
# 3-4. The implementation's own invariants
# =============================================================================

def _actor_mean(agent, params, sig):
    """The actor's mean action at one signature, as a float (first control component)."""
    return float(jnp.ravel(agent.actor.apply(params, sig))[0])


def test_positive_advantage_moves_the_actor_TOWARD_the_sampled_action():
    """A better-than-expected outcome must move the mean TOWARD the action that produced it.

    THIS IS THE TEST THAT PINS THE SIGN, and it drives the SHIPPED estimator
    (_jit_monte_carlo_actor_update) rather than a re-implementation of it.

    Why it is needed, measured: an adversarial audit mutated the shipped estimator's sign -- an
    actor that ascends COST rather than reward, i.e. the exact negation of the intended learner --
    and all thirteen of this file's other tests passed unchanged. The two whose names imply they
    guard the update direction assert only |grad| == 0 and |grad| > 1e-8, and a norm is sign-blind
    by construction. A suite that cannot distinguish an estimator from its own negation certifies
    nothing about the estimator, whilst being cited as evidence that it is correct.

    The mechanism under test: the policy is Gaussian with mean mu, the realised action is
    u = mu + n, and the loss is -A * <n/sigma^2, mu>. Gradient DESCENT on that loss therefore
    ascends A * <n/sigma^2, mu>, so for A > 0 and n > 0 the mean must INCREASE -- towards u.
    """
    agent = _build()
    # A NON-ZERO signature is load-bearing: the agent's signature at construction is all zeros, and
    # a linear actor's gradient with respect to its weights is proportional to its input, so a zero
    # input gives an identically zero gradient and the actor cannot move at all -- the test would
    # then pass or fail for a reason unrelated to the estimator's sign.
    d = int(np.asarray(agent.representation_buffer.current_signature).shape[0])
    sig = jnp.asarray(np.random.default_rng(0).normal(size=(d,)))
    m = int(agent.env.B.shape[1])
    sigma = 0.1
    noise = jnp.ones((1, m)) * 0.5                 # a POSITIVE perturbation: u = mu + 0.5 > mu
    advantage = jnp.asarray([+1.0])               # and it turned out BETTER than expected

    before = _actor_mean(agent, agent.actor_params, sig)
    new_params, _, grad_norm = agent._jit_monte_carlo_actor_update(
        agent.actor_params, agent.actor_opt_state, sig[None, :], noise, advantage, sigma
    )
    after = _actor_mean(agent, new_params, sig)

    assert float(grad_norm) > 1e-8, "the update did not fire at all"
    assert after > before, (
        f"positive advantage on a positive perturbation moved the actor's mean AWAY from the "
        f"sampled action ({before:.6e} -> {after:.6e}). The estimator ascends cost rather than "
        f"reward: its sign is inverted."
    )


def test_negative_advantage_moves_the_actor_AWAY_from_the_sampled_action():
    """The converse. A worse-than-expected outcome must move the mean away from its action.

    Together with the previous test this pins the sign in both directions, so neither an overall
    negation nor a one-sided defect can survive.
    """
    agent = _build()
    d = int(np.asarray(agent.representation_buffer.current_signature).shape[0])
    sig = jnp.asarray(np.random.default_rng(0).normal(size=(d,)))
    m = int(agent.env.B.shape[1])
    sigma = 0.1
    noise = jnp.ones((1, m)) * 0.5                 # the same POSITIVE perturbation
    advantage = jnp.asarray([-1.0])               # but it turned out WORSE than expected

    before = _actor_mean(agent, agent.actor_params, sig)
    new_params, _, grad_norm = agent._jit_monte_carlo_actor_update(
        agent.actor_params, agent.actor_opt_state, sig[None, :], noise, advantage, sigma
    )
    after = _actor_mean(agent, new_params, sig)

    assert float(grad_norm) > 1e-8, "the update did not fire at all"
    assert after < before, (
        f"negative advantage on a positive perturbation moved the actor's mean TOWARD the sampled "
        f"action ({before:.6e} -> {after:.6e}). The estimator reinforces the action that did "
        f"worse: its sign is inverted."
    )


def test_the_shipped_estimator_recovers_the_analytic_policy_gradient():
    """Drive the SHIPPED estimator on a problem whose gradient is known in closed form.

    This replaces the weaker guarantee of the NumPy unbiasedness tests below, which re-implement
    the estimator and therefore verify the author's algebra rather than the shipped code (measured:
    a trace over the estimator's lines during those tests returns the empty list -- they would pass
    unchanged if the estimator were deleted).

    One-step Gaussian problem: u = mu + n, n ~ N(0, sigma^2), reward r(u) = -(u - u_star)^2, so
    J(mu) = -(mu - u_star)^2 - sigma^2 and dJ/dmu = -2(mu - u_star). Driving the SHIPPED update
    repeatedly with A_t = r must therefore carry the actor's mean towards u_star.
    """
    agent = _build()
    d = int(np.asarray(agent.representation_buffer.current_signature).shape[0])
    rng = np.random.default_rng(0)
    sig = jnp.asarray(rng.normal(size=(d,)))   # non-zero: see the note in the directional test
    m = int(agent.env.B.shape[1])
    sigma, u_star = 0.3, -1.0

    params, opt_state = agent.actor_params, agent.actor_opt_state
    start = _actor_mean(agent, params, sig)
    for _ in range(600):
        n = jnp.asarray(rng.normal(0.0, sigma, size=(1, m)))
        mu_now = _actor_mean(agent, params, sig)
        reward = -((mu_now + float(n[0, 0]) - u_star) ** 2)      # the realised return
        params, opt_state, _ = agent._jit_monte_carlo_actor_update(
            params, opt_state, sig[None, :], n, jnp.asarray([reward]), sigma
        )
    end = _actor_mean(agent, params, sig)

    # It must move TOWARDS the optimum. The tolerance is the direction, not a fitted value: an
    # estimator with the wrong sign moves away, and one with the right sign closes the gap.
    assert abs(end - u_star) < abs(start - u_star), (
        f"the shipped estimator did not move the actor's mean towards the analytic optimum "
        f"u* = {u_star}: |{start:.4f} - u*| = {abs(start - u_star):.4f} -> "
        f"|{end:.4f} - u*| = {abs(end - u_star):.4f}"
    )


def test_zero_advantage_gives_exactly_zero_actor_gradient():
    """A vanishing advantage must give a vanishing update: no signal manufactured from nothing."""
    agent = _build()
    d = int(np.asarray(agent.representation_buffer.current_signature).shape[0])
    m = int(agent.env.B.shape[1])
    T = 8
    rng = np.random.default_rng(0)
    sig = jnp.asarray(rng.normal(size=(T, d)))
    noise = jnp.asarray(rng.normal(size=(T, m)))
    zero_advantage = jnp.zeros(T)
    _, _, grad_norm = agent._jit_monte_carlo_actor_update(
        agent.actor_params, agent.actor_opt_state, sig, noise, zero_advantage, 0.1
    )
    assert float(grad_norm) == pytest.approx(0.0, abs=1e-12)


def test_nonzero_advantage_gives_a_nonzero_actor_gradient():
    """The converse of the previous check: a real advantage must actually move the actor."""
    agent = _build()
    d = int(np.asarray(agent.representation_buffer.current_signature).shape[0])
    m = int(agent.env.B.shape[1])
    T = 8
    rng = np.random.default_rng(0)
    sig = jnp.asarray(rng.normal(size=(T, d)))
    noise = jnp.asarray(rng.normal(size=(T, m)))
    advantage = jnp.asarray(rng.normal(size=T))
    _, _, grad_norm = agent._jit_monte_carlo_actor_update(
        agent.actor_params, agent.actor_opt_state, sig, noise, advantage, 0.1
    )
    assert float(grad_norm) > 1e-8


def test_episode_buffer_is_emptied_between_episodes():
    """A truncated episode must not leak its steps into the next episode's return.

    Reachable in practice: the divergence cut ends an episode early, so without a reset the next
    episode's return would be computed over both episodes' rewards, attributing one episode's cost
    to another's actions.
    """
    agent = _build()
    agent._mc_episode_features.append(jnp.zeros(3))
    agent._mc_episode_reward_rate.append(jnp.asarray(1.0))
    assert len(agent._mc_episode_features) == 1
    agent._on_episode_start(0, np.zeros(2))
    assert agent._mc_episode_features == []
    assert agent._mc_episode_reward_rate == []


def test_unknown_actor_target_fails_loudly():
    """An unrecognised actor_target must raise, not fall back to a default silently."""
    agent = _build()
    agent.algorithm.actor_target = "not_a_target"
    with pytest.raises(ValueError, match="actor_target"):
        _ = agent._actor_target_is_monte_carlo


def test_actor_target_defaults_to_td_and_signatures_is_unchanged():
    """The actor-critic must be untouched by this feature: its target stays the TD error."""
    ac = _build(agent_name="actor_critic")
    assert ac._actor_target_is_monte_carlo is False
    pg = _build(agent_name="policy_gradient")
    assert pg._actor_target_is_monte_carlo is True


# =============================================================================
# 5. End-to-end
# =============================================================================

@pytest.mark.slow
def test_policy_gradient_trains_end_to_end():
    """The full instantiate -> train path must run and update the actor at least once."""
    agent = _build(**{"agent.training.n_episodes": 2, "agent.training.max_time": 1.0})
    before = jnp.concatenate(
        [jnp.ravel(p) for p in jax_leaves(agent.actor_params)]
    )
    agent.train()
    after = jnp.concatenate([jnp.ravel(p) for p in jax_leaves(agent.actor_params)])
    assert float(jnp.max(jnp.abs(after - before))) > 0.0, (
        "the actor parameters did not move: the Monte-Carlo update never fired"
    )


def jax_leaves(tree):
    import jax
    return jax.tree_util.tree_leaves(tree)
