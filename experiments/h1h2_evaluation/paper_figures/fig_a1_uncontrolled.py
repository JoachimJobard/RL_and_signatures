"""A1 -- Uncontrolled regimes. One panel per cell: the uncontrolled state trajectory (control = 0) from
the cell's fixed evaluation initial condition, with the target. Shows where each cell lives and what the
no-control cost J0 measures: the undamped harmonic oscillation, the decaying delayed transient of the
linear DDE, the developed Mackey-Glass limit cycle. This is setup (a property of the plant), not a
result. Emits PNG + CSV (time, no_control_state per cell).

Usage: python .../fig_a1_uncontrolled.py <out_dir>   (default out_dir from PAPER_FIGURES_OUT)
"""
import os
import sys
import numpy as np
import campaign as C
import _figlib as F

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", "figures_out")

fig, axes = F.plt.subplots(1, len(C.CELLS), figsize=(4.6 * len(C.CELLS), 3.4))
csv_rows, longest = {}, 0
for ax, (name, cell) in zip(np.atleast_1d(axes), C.CELLS.items()):
    # Any run of the cell gives the same no-control trajectory (it does not depend on the learner).
    rd = F.find_run(C.DATA_ROOT, cell["group_glob"], "value_gradient", "signature", C.REPRESENTATIVE_SEED)
    if rd is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center"); continue
    ed, _ = F.reeval_run(rd, cell["x0"], cell["burn_steps"])
    t, _, xn = F.state0(ed)
    yt = F.target_of(ed)
    ax.plot(t, xn, "-", lw=1.6, color=C.NOCTRL_COLOUR, label="no control")
    ax.axhline(yt, ls=":", lw=1.0, color=C.TARGET_COLOUR, label="target")
    ax.set_title(cell["title"], fontsize=10)
    ax.set_xlabel("time")
    ax.set_ylabel("state" if name == list(C.CELLS)[0] else "")
    csv_rows[name] = (t, xn); longest = max(longest, len(t))
axes_list = np.atleast_1d(axes)
axes_list[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=True)

# CSV: time_<cell>, state_<cell> columns, padded to the longest trajectory.
header, cols = [], []
for name, (t, xn) in csv_rows.items():
    header += [f"time_{name}", f"no_control_{name}"]
    cols += [t, xn]
rows = []
for i in range(longest):
    row = []
    for c in cols:
        row.append(f"{c[i]:.6g}" if i < len(c) else "")
    rows.append(row)
fig.tight_layout(rect=[0, 0.04, 1, 1])
F.save(fig, f"{out_dir}/figA1_uncontrolled.png", rows=rows, header=header)
