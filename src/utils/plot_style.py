"""Shared plotting style helpers (Matplotlib).

Centralises the project plot conventions so every figure is consistent:

- **Stroke convention.** Trained / predicted curves are solid (``"-"``); analytical
  references and baselines (e.g. the uncontrolled trajectory) are dashed (``"--"``);
  auxiliary annotation lines (targets, set-points, markers) are dotted (``":"``).
- **External legend.** Legends are placed outside the plotting area (centered below
  the panels, as a figure-level legend), never on top of the data.
- **Explanatory text box.** A formula box giving the analytical expression for the
  plotted quantities is placed below the figure.
- **Sequential colour for sweeps.** A sweep over an index / hyperparameter is encoded
  in colour (viridis), not in stroke.

Layout-overlap detection (``check_layout``) and the build/save convenience wrappers
(``prepare_figure`` for builders that *return* a figure, ``finalize_figure`` for
scripts that *save* one) are ported from the ``constrained_learning_option_pricing``
``_figure_layout`` utilities. ``check_layout`` emits non-fatal ``warnings.warn`` for
common defects — a legend / title / axis label spilling past the canvas, or the
bottom formula box overlapping the x-axis tick labels or a legend — so layout
regressions surface as visible warnings rather than silently clipped figures.

LaTeX in titles / labels / annotations uses ``$...$`` and is rendered by Matplotlib's
mathtext; no external MathJax dependency.
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (headless cluster / wandb logging).
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# Matplotlib linestyles for the stroke convention.
STROKE_TRAINED = "-"      # trained models / predicted / controlled (agent) curves
STROKE_REFERENCE = "--"   # analytical references / baselines (e.g. uncontrolled system)
STROKE_AUXILIARY = ":"    # auxiliary annotations (targets, set-points, constants)


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


# =============================================================================
# External legend + formula box (figure-level artists)
# =============================================================================

def apply_external_legend(fig, handles=None, labels=None, *, title: str | None = None,
                          y: float = 0.10, ncol: int | None = None, fontsize: int = 8):
    """Place a single figure-level legend centered below the panels.

    Returns the legend artist (or ``None`` if there is nothing to show). When
    ``handles``/``labels`` are omitted, they are collected from the figure's first
    axes. ``y`` is the legend's lower-edge position in figure coordinates; reserve
    room below the panels with :func:`prepare_figure`'s ``reserve_bottom`` first.
    """
    if handles is None or labels is None:
        h_l = [ax.get_legend_handles_labels() for ax in fig.axes]
        handles, labels = ([], [])
        seen: set[str] = set()
        for hs, ls in h_l:
            for h, l in zip(hs, ls):
                if l and l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    if not labels:
        return None
    if ncol is None:
        ncol = min(len(labels), 5)
    return fig.legend(handles, labels, loc="lower center",
                      bbox_to_anchor=(0.5, y), ncol=max(1, ncol),
                      fontsize=fontsize, frameon=True, title=title)


def add_formula_textbox(fig, text: str, *, y: float = 0.015, fontsize: int = 8):
    """Add an explanatory formula text box below the figure; return the artist.

    Placed at ``(0.5, y)`` in figure coordinates with a rounded grey box. Reserve
    bottom room with :func:`prepare_figure`'s ``reserve_bottom`` first.
    """
    if not text:
        return None
    return fig.text(0.5, y, text, ha="center", va="bottom", fontsize=fontsize,
                    wrap=True,
                    bbox=dict(boxstyle="round", facecolor="#f5f5f5",
                              edgecolor="#bbbbbb"))


# =============================================================================
# Layout-overlap detection (ported from constrained_learning _figure_layout.py)
# =============================================================================

def _bb(artist, renderer):
    """Window extent of an artist, or None if it has no drawable extent."""
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return None
    return bb if (bb is not None and bb.width > 0 and bb.height > 0) else None


def _overlaps(a, b, eps: float = 1.0) -> bool:
    return not (a.x1 < b.x0 + eps or b.x1 < a.x0 + eps
                or a.y1 < b.y0 + eps or b.y1 < a.y0 + eps)


def check_layout(fig, fname: str, *, legends=(), formula=None, axes=()):
    """Emit warnings for common figure-layout defects.

    1. **Cropping risk** — any legend / title / x- or y-axis label whose box spills
       past the figure canvas (the spill means the layout under-reserves space).
    2. **Hidden x-axis** — the bottom formula box overlapping the x-axis tick labels
       or the x-axis label, which would hide them.
    3. **Formula hidden by a legend** — a tall (multi-line) formula box rising into a
       bottom-anchored legend and disappearing behind it.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    eps = 2.0

    named = [("legend", lg) for lg in legends if lg is not None]
    if formula is not None:
        named.append(("formula box", formula))
    for ax in axes:
        named += [("title", ax.title), ("x-axis label", ax.xaxis.label),
                  ("y-axis label", ax.yaxis.label)]
    for kind, art in named:
        bb = _bb(art, r)
        if bb is None:
            continue
        if (bb.x0 < fb.x0 - eps or bb.x1 > fb.x1 + eps
                or bb.y0 < fb.y0 - eps or bb.y1 > fb.y1 + eps):
            warnings.warn(f"[{fname}] {kind} spills past the figure area "
                          f"(cropping risk) — reserve more margin/legend space.",
                          stacklevel=2)

    if formula is not None:
        fbb = _bb(formula, r)
        for ax in axes:
            for t in list(ax.get_xticklabels()) + [ax.xaxis.label]:
                tb = _bb(t, r)
                if fbb is not None and tb is not None and _overlaps(fbb, tb):
                    warnings.warn(f"[{fname}] the formula box overlaps the x-axis "
                                  f"labels (may hide them) — increase the reserved "
                                  f"bottom margin.", stacklevel=2)
                    break
        for lg in legends:
            lb = _bb(lg, r)
            if fbb is not None and lb is not None and _overlaps(fbb, lb):
                warnings.warn(f"[{fname}] the formula box overlaps a legend (may "
                              f"hide the formula) — raise the legend anchor or "
                              f"reserve more bottom margin.", stacklevel=2)
                break


# =============================================================================
# Build / save convenience wrappers
# =============================================================================

def prepare_figure(fig, *, fname: str = "figure", handles=None, labels=None,
                   formula: str | None = None, axes=(), reserve_bottom: float = 0.20,
                   legend_y: float | None = None, formula_y: float = 0.015,
                   legend_ncol: int | None = None, legend_fontsize: int = 8,
                   formula_fontsize: int = 8, legend_title: str | None = None):
    """Reserve bottom room, attach the external legend + formula box, and check the
    layout. Returns the (legend, formula) artists. For builders that *return* the
    figure (logged to wandb and saved elsewhere); use :func:`finalize_figure` when a
    script saves directly.
    """
    fig.subplots_adjust(bottom=reserve_bottom)
    if legend_y is None:
        # Sit the legend in the upper part of the reserved strip, above the formula.
        legend_y = max(formula_y + 0.04, reserve_bottom * 0.35)
    legend = apply_external_legend(fig, handles, labels, title=legend_title,
                                   y=legend_y, ncol=legend_ncol,
                                   fontsize=legend_fontsize)
    formula_art = add_formula_textbox(fig, formula, y=formula_y,
                                      fontsize=formula_fontsize) if formula else None
    check_layout(fig, fname, legends=[legend] if legend else [],
                 formula=formula_art, axes=axes)
    return legend, formula_art


def finalize_figure(fig, path, *, handles=None, labels=None, formula: str | None = None,
                    axes=(), reserve_bottom: float = 0.20, dpi: int = 150,
                    legend_ncol: int | None = None, legend_fontsize: int = 8,
                    formula_fontsize: int = 8):
    """Prepare the figure (legend + formula + layout check), then save it including
    all external artists, and close it."""
    fname = os.path.basename(str(path))
    legend, formula_art = prepare_figure(
        fig, fname=fname, handles=handles, labels=labels, formula=formula, axes=axes,
        reserve_bottom=reserve_bottom, legend_ncol=legend_ncol,
        legend_fontsize=legend_fontsize, formula_fontsize=formula_fontsize)
    extra = [a for a in (legend, formula_art) if a is not None]
    fig.savefig(path, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra)
    plt.close(fig)
