"""Delayed-LQR oracle as an evaluable agent + the normalised-sub-optimality metric.

The oracle applies the analytic delayed-LQR feedback (``src.solvers.delayed_lqr``)
to the *same* environment the learned agents control, through the *same* evaluation
code (``collect_evaluation_data``), so the oracle cost ``J_oracle`` is computed
identically to the agents' ``J``. Every learned model is then reported as the
**normalised sub-optimality** ``(J - J_oracle) / |J_oracle|`` relative to this
ceiling — the analysis convention fixed in the experimental-design pre-registration.

Caveat: the gain is synthesised for the forward-Euler discretisation of the plant
(``delayed_lqr`` option A) but applied to the RK4-integrated environment, so it is a
near-optimal (not exactly optimal) ceiling; it converges to the true optimum as the
control step shrinks. A learned model marginally beating the oracle is therefore
possible and is itself a useful diagnostic.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from src.envs.env_rk_jax import JAXDDEEnv, JAXEnvWrapper
from src.solvers.delayed_lqr import DelayedLQR, augmented_discrete_lqr


def delayed_lqr_for_env(env: JAXDDEEnv, discount_beta: float = 1.0) -> DelayedLQR:
    """Synthesise the delayed-LQR oracle from a linear delayed environment's
    matrices (A, A1, B, Q, R), its max delay, and its control step.

    ``discount_beta`` (per-step, in ``(0, 1]``) selects the objective: ``1.0`` gives
    the undiscounted infinite-horizon oracle; ``exp(-gamma dt)`` gives the discounted
    oracle on the SAME objective as an agent discounting at rate ``gamma``,
    which is the fairness anchor for a discounted comparison."""
    return augmented_discrete_lqr(
        np.array(env.A), np.array(env.A1), np.array(env.B),
        np.array(env.Q), np.array(env.R),
        delay=float(env.max_delay), dt=float(env.step_size),
        discount_beta=discount_beta,
    )


class OracleDelayedLQRAgent:
    """Evaluable agent that applies the delayed-LQR feedback (a linear functional of
    the discretised history). Implements just the surface the evaluation code uses:
    ``env``, ``wrapper``, ``training.scale``, ``_fill_buffer_initial``,
    ``update_buffer``, ``get_eval_action``, ``get_value``."""

    def __init__(self, env: JAXDDEEnv, lqr: DelayedLQR, scale: float = 1.0) -> None:
        self.env = env
        self.lqr = lqr
        self.wrapper = JAXEnvWrapper(env)
        self.training = SimpleNamespace(scale=scale)
        self.discretization_state = 0.01
        self._history: deque = deque(maxlen=lqr.k_taps + 1)

    def _fill_buffer_initial(self) -> None:
        assert self.wrapper.state is not None
        x = np.array(self.wrapper.state.x).reshape(self.env.N)
        self._history.clear()
        for _ in range(self.lqr.k_taps + 1):
            self._history.append(x)

    def update_buffer(self, x: np.ndarray) -> None:
        self._history.appendleft(np.array(x).reshape(self.env.N))

    def get_eval_action(self, x_scaled: np.ndarray) -> np.ndarray:
        # History window, newest state first: [x(t), x(t-dt), ..., x(t-K dt)].
        window = np.array(self._history)
        return np.asarray(self.lqr.control(window)).reshape(-1)

    def get_value(self) -> float:
        return 0.0


def oracle_total_cost(
    env: JAXDDEEnv,
    lqr: DelayedLQR,
    x0: np.ndarray,
    T_sim: float,
    seed: int = 456,
    burning_steps: int = 0,
) -> float:
    """Closed-loop integrated cost ``J_oracle`` of the delayed-LQR oracle on ``env``,
    computed by the same eval code as the learned agents (so it is commensurable)."""
    from src.training.evaluate import EvaluableAgent, collect_evaluation_data

    agent = OracleDelayedLQRAgent(env, lqr)
    data = collect_evaluation_data(
        cast("EvaluableAgent", agent), np.asarray(x0), T_sim,
        seed=seed, burning_steps=burning_steps,
    )
    return float(data["eval_metrics"]["eval/total_cost_agent"])


def normalised_suboptimality(j_agent: float, j_oracle: float) -> float:
    """``(J_agent - J_oracle) / |J_oracle|`` — 0 at the ceiling, >0 above it."""
    if j_oracle == 0.0:
        return float("nan")
    return (j_agent - j_oracle) / abs(j_oracle)
