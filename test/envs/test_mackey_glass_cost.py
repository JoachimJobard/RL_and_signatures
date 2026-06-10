"""Tests for the Mackey-Glass running cost (code_review F-D1).

The cost must be the quadratic tracking + control-EFFORT cost
(x - x_target)' Q (x - x_target) + u' R u, with NO control-rate (delta_u) term.
The earlier code penalised delta_u, which makes the reward depend on the previous
action; the regression test below checks that it does not.
"""

import jax.numpy as jnp
import numpy as np

from src.envs.mackey_glass_1D import MackeyGlass1DEnv


def _make_env():
    return MackeyGlass1DEnv(
        delay=17.0, step_size=1.0, resolution=5,
        Q=jnp.array([[1.0]]), R=jnp.array([[0.1]]), x_target=0.0,
    )


def test_reward_is_quadratic_tracking_plus_effort():
    env = _make_env()
    state = env.reset(rng_key=None, x0=jnp.array([0.8]))
    u = jnp.array([0.3])
    new_state, x, reward = env.step(state, u)
    expected_cost = float((x - env.x_target).T @ env.Q @ (x - env.x_target)
                          + u.T @ env.R @ u)
    assert np.isclose(float(reward), -expected_cost, atol=1e-8)


def test_reward_independent_of_last_u():
    # With an effort cost (no delta_u term) the reward depends only on (x, u), not
    # on state.last_u. Two states identical except for last_u, stepped with the same
    # action, must yield the same next state and the same reward.
    env = _make_env()
    base = env.reset(rng_key=None, x0=jnp.array([0.8]))
    state_a = base._replace(last_u=jnp.array([0.0]))
    state_b = base._replace(last_u=jnp.array([5.0]))
    u = jnp.array([0.3])
    _, x_a, reward_a = env.step(state_a, u)
    _, x_b, reward_b = env.step(state_b, u)
    assert np.allclose(np.asarray(x_a), np.asarray(x_b))      # dynamics ignore last_u
    assert np.isclose(float(reward_a), float(reward_b), atol=1e-10)  # cost ignores last_u
