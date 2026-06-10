"""Shared plotting style helpers (Plotly).

Centralises the project plot conventions so every figure is consistent:

- **Stroke convention.** Trained / predicted curves are solid; analytical
  references and baselines (e.g. the uncontrolled trajectory) are dashed;
  auxiliary annotation lines (targets, set-points, markers) are dotted.
- **External legend.** Legends are placed outside the plotting area (below the
  axes), never on top of the data.
- **Explanatory text box.** A formula box giving the analytical expression for the
  plotted quantities is placed below the figure.
- **Sequential colour for sweeps.** A sweep over an index/hyperparameter is encoded
  in colour (viridis), not in stroke.

LaTeX in titles/labels/annotations uses ``$...$`` and is rendered by MathJax;
``_save_figure`` in ``main_unified`` writes HTML with ``include_mathjax="cdn"`` so
the maths renders in the saved figure.
"""

from __future__ import annotations

import matplotlib
from matplotlib.colors import to_hex

# Plotly line-dash values for the stroke convention.
STROKE_TRAINED = "solid"     # trained models / predicted / controlled (agent) curves
STROKE_REFERENCE = "dash"    # analytical references / baselines (e.g. uncontrolled system)
STROKE_AUXILIARY = "dot"     # auxiliary annotations (targets, set-points, constants)


def sequential_colors(n: int, cmap_name: str = "viridis") -> list[str]:
    """Return ``n`` hex colours sampled evenly from a sequential colormap.

    Use to encode a sweep (over an index, seed, or hyperparameter) in colour.
    """
    if n <= 0:
        return []
    cmap = matplotlib.colormaps[cmap_name]
    if n == 1:
        return [to_hex(cmap(0.5))]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def apply_external_legend(fig, *, title: str | None = None, bottom_margin: int = 160):
    """Place the legend outside the axes (centered below them)."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.14,
            xanchor="center", x=0.5,
            title_text=title or "",
        ),
        margin=dict(b=bottom_margin),
        showlegend=True,
    )
    return fig


def add_formula_textbox(fig, text: str, *, y: float = -0.34, bottom_margin: int = 230):
    """Add an explanatory formula text box below the axes (paper coordinates)."""
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=y,
        xanchor="center", yanchor="top",
        showarrow=False, align="center",
        font=dict(size=11),
        bordercolor="gray", borderwidth=1, borderpad=6,
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_layout(margin=dict(b=bottom_margin))
    return fig
