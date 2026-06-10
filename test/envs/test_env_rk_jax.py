"""Tests for the RK4 DDE integrator base class (code_review F-D2).

On a no-delay linear ODE the integrator must reproduce the analytic solution
exp(A t) x0 to high accuracy, and the reward must be the negated quadratic cost
-(x' Q x + u' R u).
"""

import jax.numpy as jnp
import numpy as np

from src.envs.env_rk_jax import JAXDDEEnv


def test_rk4_matches_analytic_solution_linear_ode():
    # dx/dt = A x, A = -1, x0 = 1  =>  x(t) = exp(-t).
    env = JAXDDEEnv(
        A=jnp.array([[-1.0]]), B=jnp.array([[0.0]]), A1=None, delay=None,
        Q=jnp.array([[1.0]]), R=jnp.array([[1.0]]), step_size=0.1, resolution=10,
    )
    state = env.reset(rng_key=None, x0=jnp.array([1.0]))
    t_total = 1.0
    n_steps = int(round(t_total / env.step_size))
    for _ in range(n_steps):
        state, x, _ = env.step(state, jnp.array([0.0]))
    assert np.isclose(float(x[0]), np.exp(-1.0), atol=1e-6)


def test_reward_is_negated_quadratic_cost():
    env = JAXDDEEnv(
        A=jnp.array([[-0.5]]), B=jnp.array([[1.0]]), A1=None, delay=None,
        Q=jnp.array([[2.0]]), R=jnp.array([[0.5]]), step_size=0.1, resolution=5,
    )
    state = env.reset(rng_key=None, x0=jnp.array([1.0]))
    u = jnp.array([0.7])
    state, x, reward = env.step(state, u)
    expected = -float(x.T @ env.Q @ x + u.T @ env.R @ u)
    assert np.isclose(float(reward), expected, atol=1e-10)


def test_buffer_size_covers_delay():
    env = JAXDDEEnv(
        A=jnp.zeros((1, 1)), B=jnp.array([[1.0]]), A1=jnp.array([[0.1]]),
        delay=jnp.array([1.0]), Q=jnp.array([[1.0]]), R=jnp.array([[1.0]]),
        step_size=0.1, resolution=10,
    )
    # buffer must hold at least ceil(max_delay / solver_step_size) + 1 samples
    assert env.buffer_size >= int(np.ceil(1.0 / env.solver_step_size)) + 1
