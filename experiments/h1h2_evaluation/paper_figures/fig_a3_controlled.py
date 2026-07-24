"""A3 -- Controlled trajectories. One panel per cell: a representative controlled run (the signature
value-gradient controller, the reliable one) against the analytic oracle, the no-control baseline and
the target, all from the cell's fixed evaluation initial condition. Illustrates that the signature
controller drives the delayed state to the target, where the uncontrolled system oscillates.

Stroke convention: solid = trained agent, dashed = analytic delayed-LQR oracle, dotted = no-control
baseline and target line. Emits PNG + CSV (time, agent, oracle, no_control per cell).

Usage: python .../fig_a3_controlled.py <out_dir>
"""
import os
import sys
import numpy as np
import campaign as C
import _figlib as F

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", "figures_out")
SEED = C.REPRESENTATIVE_SEED
LEARNER, KIND = "value_gradient", "signature"  # the reliable controller carries the illustration

fig, axes = F.plt.subplots(1, len(C.CELLS), figsize=(4.6 * len(C.CELLS), 3.4))
csv_header, csv_cols, longest = [], [], 0
for ax, (name, cell) in zip(np.atleast_1d(axes), C.CELLS.items()):
    rd = F.find_run(C.DATA_ROOT, cell["group_glob"], LEARNER, KIND, SEED)
    if rd is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center"); continue
    ed, _ = F.reeval_run(rd, cell["x0"], cell["burn_steps"])
    t, xa, xn = F.state0(ed)
    yt = F.target_of(ed)
    eta = F.eta_of(ed, cell["oracle_cost_reduction"])
    ax.plot(t, xa, "-", lw=1.6, color=C.AGENT_COLOUR, label="agent (signature)")
    # analytic oracle trajectory, re-evaluated on the same IC (closed-form controller; no checkpoint)
    orc = F.find_oracle(C.DATA_ROOT, cell["group_glob"])
    xo = None
    if orc is not None:
        edo, _ = F.reeval_run(orc, cell["x0"], cell["burn_steps"], load_checkpoint=False)
        to, xo, _ = F.state0(edo)
        ax.plot(to, xo, "--", lw=1.3, color=C.ORACLE_COLOUR, label="oracle")
    ax.plot(t, xn, ":", lw=1.3, color=C.NOCTRL_COLOUR, label="no control")
    ax.axhline(yt, ls=":", lw=0.9, color=C.TARGET_COLOUR)
    ax.set_title(f"{cell['title']}   (η = {eta:.0f}%)", fontsize=9.5)
    ax.set_xlabel("time")
    ax.set_ylabel("state" if name == list(C.CELLS)[0] else "")
    csv_header += [f"time_{name}", f"agent_{name}", f"oracle_{name}", f"no_control_{name}"]
    xo_pad = xo if xo is not None else np.full_like(xa, np.nan)
    csv_cols += [t, xa, xo_pad, xn]; longest = max(longest, len(t))
axes_list = np.atleast_1d(axes)
axes_list[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=True)

rows = []
for i in range(longest):
    rows.append([f"{c[i]:.6g}" if i < len(c) else "" for c in csv_cols])
fig.tight_layout(rect=[0, 0.04, 1, 1])
F.save(fig, f"{out_dir}/figA3_controlled.png", rows=rows, header=csv_header)
