"""Comparison grid -- one 3x3 grid per cell (learner rows x representation columns) of the controlled
state on the clean fixed initial condition at the representative seed, re-simulated from checkpoints.
Panel title = relative optimality eta. Solid = agent (blue if eta>0, red if it fails), dashed = no
control, dotted = target. This is the diagnostic grid behind the dashboard; it is the per-run
illustration, complementary to the F2 distribution.

Divergent excursions are masked out of frame (the eta title carries the failure) so a blown-up
trajectory does not collapse the axis. Emits one PNG per cell.

Usage: python .../fig_comparison_grid.py <out_dir> [cell_name]   (all cells if none given)
"""
import os
import sys
import numpy as np
import campaign as C
import _figlib as F

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", "figures_out")
only = sys.argv[2] if len(sys.argv) > 2 else None
GOOD, BAD = "#1f77b4", "#c1352b"

for name, cell in C.CELLS.items():
    if only and name != only:
        continue
    fig, ax = F.plt.subplots(3, 3, figsize=(13.5, 10.2), sharex=True)
    fig.suptitle(f"{cell['title']}  ·  held-out seed {C.REPRESENTATIVE_SEED}  ·  clean fixed initial "
                 f"condition  ·  controlled state per learner × representation", fontsize=12.5, y=0.995)
    for r, learner in enumerate(C.LEARNERS):
        for c, kind in enumerate(C.KINDS):
            a = ax[r, c]
            rd = F.find_run(C.DATA_ROOT, cell["group_glob"], learner, kind, C.REPRESENTATIVE_SEED)
            if rd is None:
                a.text(0.5, 0.5, "missing", ha="center", va="center", color="0.6"); continue
            ed, _ = F.reeval_run(rd, cell["x0"], cell["burn_steps"])
            t, xa, xn = F.state0(ed)
            yt = F.target_of(ed)
            eta = F.eta_of(ed, cell["oracle_cost_reduction"])
            col = GOOD if (np.isfinite(eta) and eta > 0) else BAD
            lo = min(float(np.min(xn)), yt); hi = max(float(np.max(xn)), yt)
            span = (hi - lo) if hi > lo else 1.0; pad = 0.35 * span + 0.15
            ylo, yhi = lo - pad, hi + pad
            xap = np.where((xa >= ylo) & (xa <= yhi), xa, np.nan)
            a.plot(t, xap, "-", lw=1.5, color=col)
            a.plot(t, xn, "--", lw=1.0, color="0.6")
            a.axhline(yt, ls=":", lw=1, color="k")
            a.set_ylim(ylo, yhi)
            a.set_title(f"η={eta:.0f}%" if (np.isfinite(eta) and eta > -1000) else "η≪0",
                        color=col, fontsize=12, fontweight="bold", pad=3)
            if r == 0:
                a.annotate(C.KIND_LABEL[kind], xy=(0.5, 1.28), xycoords="axes fraction", ha="center",
                           fontsize=12, color="0.25")
            if c == 0:
                a.set_ylabel(C.LEARNER_LABEL[learner], fontsize=12, color="0.25")
            if r == 2:
                a.set_xlabel("time")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    F.save(fig, f"{out_dir}/grid_{name}.png")
