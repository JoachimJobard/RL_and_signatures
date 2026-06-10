"""Tests for the shared Plotly plot-style helpers and a styled figure."""

import numpy as np
import plotly.graph_objects as go

from src.utils.plot_style import (
    sequential_colors,
    apply_external_legend,
    add_formula_textbox,
    STROKE_TRAINED,
    STROKE_REFERENCE,
    STROKE_AUXILIARY,
)
from src.training.evaluate import plot_training_metrics


def test_sequential_colors_count_and_distinct():
    cols = sequential_colors(5)
    assert len(cols) == 5
    assert len(set(cols)) == 5
    assert all(c.startswith("#") for c in cols)


def test_sequential_colors_edge_cases():
    assert sequential_colors(0) == []
    assert len(sequential_colors(1)) == 1


def test_stroke_constants_are_distinct():
    assert len({STROKE_TRAINED, STROKE_REFERENCE, STROKE_AUXILIARY}) == 3


def test_external_legend_is_below_axes():
    fig = go.Figure(go.Scatter(y=[1, 2, 3], name="a"))
    apply_external_legend(fig)
    assert fig.layout.legend.y is not None and fig.layout.legend.y < 0
    assert fig.layout.showlegend is True


def test_formula_textbox_added_below_axes():
    fig = go.Figure(go.Scatter(y=[1, 2, 3]))
    add_formula_textbox(fig, r"$c = x^\top Q x + u^\top R u$")
    texts = [a.text for a in fig.layout.annotations]
    assert any("Q x" in t for t in texts)
    box = [a for a in fig.layout.annotations if "Q x" in a.text][0]
    assert box.yref == "paper" and box.y < 0


def test_plot_training_metrics_is_styled():
    metrics = {
        "cost_episodic": np.linspace(5, 1, 50),
        "loss_episodic": np.linspace(1, 0.1, 50),
        "gradient_critic": np.abs(np.random.default_rng(0).standard_normal(50)),
    }
    fig = plot_training_metrics(metrics)
    # External legend below the axes
    assert fig.layout.legend.y < 0
    # A formula text box (paper-referenced, below the axes) is present
    assert any(a.yref == "paper" and a.y < 0 and "delta" in a.text.lower()
               for a in fig.layout.annotations)
