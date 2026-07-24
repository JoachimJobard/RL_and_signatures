"""A2 -- Learning curves. One panel per cell: the noiseless evaluation metric versus training episode,
one curve per representation, at the representative seed. Illustrates convergence, patience early
stopping (each curve ends where its run stopped) and the divergence guard (a diverged run is marked).
Justifies the adaptive-budget protocol. Emits PNG + CSV.

The evaluation metric is the agent's own logged noiseless eval (eval_reward for actor-critic / policy
gradient, eval_cost for value gradient -- both are returns, higher is better), read from
training_metrics.pkl. Nothing is recomputed.

Usage: python .../fig_a2_learning_curves.py <out_dir>
"""
import os
import sys
import glob
import pickle
import numpy as np
import campaign as C
import _figlib as F

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PAPER_FIGURES_OUT", "figures_out")
SEED = C.REPRESENTATIVE_SEED

fig, axes = F.plt.subplots(1, len(C.CELLS), figsize=(4.6 * len(C.CELLS), 3.4))
csv_header, csv_cols, longest = [], [], 0
for ax, (name, cell) in zip(np.atleast_1d(axes), C.CELLS.items()):
    for kind in C.KINDS:
        rd = F.find_run(C.DATA_ROOT, cell["group_glob"], "value_gradient", kind, SEED)
        if rd is None:
            continue
        m = pickle.load(open(f"{rd}/training_metrics.pkl", "rb"))
        ep = np.asarray(m.get("eval_episodes", []), dtype=float)
        val = np.asarray(m.get("eval_reward", m.get("eval_cost", [])), dtype=float)
        n = min(len(ep), len(val))
        if n == 0:
            continue
        ep, val = ep[:n], val[:n]
        diverged = bool(m.get("diverged", False))
        ax.plot(ep, val, "-", lw=1.4, color=C.KIND_COLOUR[kind], label=C.KIND_LABEL[kind])
        if diverged:
            ax.plot(ep[-1], val[-1], "x", ms=7, color=C.KIND_COLOUR[kind])  # divergence-abort marker
        csv_header += [f"episode_{name}_{kind}", f"eval_{name}_{kind}"]
        csv_cols += [ep, val]; longest = max(longest, n)
    ax.set_title(cell["title"], fontsize=10)
    ax.set_xlabel("episode")
    ax.set_ylabel("noiseless eval (return)" if name == list(C.CELLS)[0] else "")
axes_list = np.atleast_1d(axes)
axes_list[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=True)
axes_list[0].text(0.02, -0.42, "value-gradient learner; × = divergence-abort", transform=axes_list[0].transAxes,
                  fontsize=7.5, color="0.4")

rows = []
for i in range(longest):
    rows.append([f"{c[i]:.6g}" if i < len(c) else "" for c in csv_cols])
fig.tight_layout(rect=[0, 0.06, 1, 1])
F.save(fig, f"{out_dir}/figA2_learning_curves.png", rows=rows, header=csv_header)
