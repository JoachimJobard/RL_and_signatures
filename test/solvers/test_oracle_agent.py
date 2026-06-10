"""Tests for the delayed-LQR oracle agent and the normalised-sub-optimality metric."""

import math

import numpy as np

from src.envs.env_rk_jax import JAXDDEEnv
from src.solvers.oracle_agent import (
    OracleDelayedLQRAgent,
    delayed_lqr_for_env,
    normalised_suboptimality,
)
from src.training.evaluate import collect_evaluation_data


def test_normalised_suboptimality():
    assert normalised_suboptimality(1.0, 1.0) == 0.0
    assert normalised_suboptimality(2.0, 1.0) == 1.0
    assert math.isnan(normalised_suboptimality(1.0, 0.0))


def _unstable_linear_delayed_env() -> JAXDDEEnv:
    # Lightly unstable delayed oscillator: no-control diverges, so the oracle's
    # stabilisation is unambiguous.
    return JAXDDEEnv(
        A=np.array([[0.0, 1.0], [-0.5, 0.1]]),
        B=np.array([[0.0], [1.0]]),
        A1=np.array([[0.1, 0.0], [0.0, 0.1]]),
        delay=np.array([0.5, 0.5]),
        Q=np.eye(2),
        R=np.array([[0.1]]),
        step_size=0.05,
        resolution=4,
    )


def test_oracle_stabilises_linear_delayed_env():
    env = _unstable_linear_delayed_env()
    lqr = delayed_lqr_for_env(env)
    assert lqr.closed_loop_spectral_radius() < 1.0  # stabilising gain

    agent = OracleDelayedLQRAgent(env, lqr)
    data = collect_evaluation_data(agent, np.array([1.0, 1.0]), T_sim=5.0)
    m = data["eval_metrics"]

    assert np.isfinite(m["eval/total_cost_agent"])
    # The oracle controls the state better than doing nothing on an unstable plant.
    assert m["eval/final_error_agent"] < m["eval/final_error_no_control"]
    assert m["eval/cost_reduction_pct"] > 0.0


def test_oracle_is_its_own_ceiling():
    env = _unstable_linear_delayed_env()
    lqr = delayed_lqr_for_env(env)
    agent = OracleDelayedLQRAgent(env, lqr)
    j = collect_evaluation_data(agent, np.array([1.0, 1.0]), T_sim=5.0)["eval_metrics"][
        "eval/total_cost_agent"
    ]
    assert normalised_suboptimality(j, j) == 0.0
