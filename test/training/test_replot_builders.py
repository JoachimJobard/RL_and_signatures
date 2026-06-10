"""The figure builders must be pure functions of saved data (no agent), so
figures are rebuildable and restyleable a posteriori."""

import numpy as np
import plotly.graph_objects as go

from src.training.evaluate import (
    plot_agent_vs_no_control_from_data,
    plot_multiple_trajectories_from_data,
)


def _synthetic_eval_data(n_state: int = 1, T: int = 11) -> dict:
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 1.0, T)
    states_agent = rng.standard_normal((T, n_state))
    states_no_ctrl = rng.standard_normal((T, n_state))
    actions = rng.standard_normal((T - 1, 1))
    cost_agent = np.abs(rng.standard_normal(T - 1))
    cost_no_ctrl = np.abs(rng.standard_normal(T - 1))
    return {
        "times": times,
        "states_agent": states_agent,
        "states_no_ctrl": states_no_ctrl,
        "actions": actions,
        "cost_agent": cost_agent,
        "cost_no_ctrl": cost_no_ctrl,
        "cum_cost_agent": np.cumsum(cost_agent),
        "cum_cost_no_ctrl": np.cumsum(cost_no_ctrl),
        "x0": np.zeros(n_state),
        "x_ref": np.zeros(n_state),
        "has_target": False,
    }


def test_agent_vs_no_control_builds_from_data():
    fig = plot_agent_vs_no_control_from_data(_synthetic_eval_data(n_state=2))
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0  # traces were added from the data alone


def test_agent_vs_no_control_title_override():
    fig = plot_agent_vs_no_control_from_data(_synthetic_eval_data(), title="My custom title")
    assert fig.layout.title.text == "My custom title"


def test_multiple_trajectories_builds_from_data():
    multi = {"trajectories": [_synthetic_eval_data(), _synthetic_eval_data()],
             "n_trajectories": 2, "metrics": {}}
    fig = plot_multiple_trajectories_from_data(multi, title="multi")
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "multi"
    assert len(fig.data) > 0
