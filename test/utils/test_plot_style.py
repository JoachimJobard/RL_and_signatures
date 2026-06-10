"""Tests for the shared Matplotlib plot-style helpers and a styled figure."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    # Matplotlib linestyles: solid / dashed / dotted.
    assert (STROKE_TRAINED, STROKE_REFERENCE, STROKE_AUXILIARY) == ("-", "--", ":")


def test_external_legend_is_below_axes():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], label="a")
    legend = apply_external_legend(fig, y=0.06)
    assert legend is not None
    # Figure-level legend (external), not attached to the axes.
    assert legend in fig.legends
    # Anchored in the lower strip of the figure (below the axes box).
    anchor = legend.get_bbox_to_anchor()
    y_fig = fig.transFigure.inverted().transform((anchor.x0, anchor.y0))[1]
    assert y_fig < ax.get_position().y0
    plt.close(fig)


def test_formula_textbox_added_below_axes():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    art = add_formula_textbox(fig, r"$c = x^\top Q x + u^\top R u$")
    assert art is not None and art in fig.texts
    # Positioned in figure coordinates, below the axes box.
    assert art.get_position()[1] < ax.get_position().y0
    plt.close(fig)


def test_plot_training_metrics_is_styled():
    metrics = {
        "cost_episodic": np.linspace(5, 1, 50),
        "loss_episodic": np.linspace(1, 0.1, 50),
        "gradient_critic": np.abs(np.random.default_rng(0).standard_normal(50)),
    }
    fig = plot_training_metrics(metrics)
    # External figure-level legend is present.
    assert len(fig.legends) >= 1
    # A formula text box mentioning the TD-error delta is present below the figure.
    assert any("delta" in t.get_text().lower() for t in fig.texts)
    plt.close(fig)
