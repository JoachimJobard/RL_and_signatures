"""Generate the consolidated H1/H2 results-summary figure for the 2026-06-11 report.

Self-contained: the aggregate values are transcribed from the per-group
``summary.yaml`` files on Jean Zay (``data/main_unified/<group>/summary.yaml``),
quoted here as literals so the figure regenerates without cluster access. This is a
*summary* visualisation (point aggregates over 5 seeds; the oracle-ladder rung is
preliminary at 1-2 seeds) — it does not recompute anything.

Run: ``.venv/bin/python make_results_summary_figure.py`` (writes ``results_summary.png``
next to this script). Lower is better in every panel.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).with_name("results_summary.png")

# Consistent colours per representation across panels.
C_MARKOVIAN = "#bcbd22"   # current-state only
C_RAW = "#1f77b4"         # raw-history (discretised window + monomials)
C_SIG = "#6a3d9a"         # path signature
C_LADDER = "#2ca02c"      # oracle-ladder rungs

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# --- Panel A: H1, high-gap linear cell (metric = normalised sub-optimality rho) ---
axA = axes[0, 0]
labels = ["markovian", "signature\n(depth 3)"]
vals = [0.380, 0.157]
bars = axA.bar(labels, vals, color=[C_MARKOVIAN, C_SIG], width=0.6)
for b, v in zip(bars, vals):
    axA.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
             ha="center", va="bottom", fontsize=10)
axA.set_ylabel(r"normalised sub-optimality $\rho$")
axA.set_title("H1 (strong): history halves $\\rho$\n"
              "high-gap linear cell (54.5% oracle gap)", fontsize=11)
axA.set_ylim(0, 0.45)
axA.annotate("lower is better", xy=(0.02, 0.95), xycoords="axes fraction",
             fontsize=8, style="italic", va="top")

# --- Panel B: H2, Mackey-Glass limit cycle tau=6 (metric = raw cost J, log scale) ---
axB = axes[0, 1]
labels = ["markovian", "raw-history\n(best stable)", "signature"]
vals = [np.nan, 0.28, 0.05]  # markovian fails (no finite usable cost)
colors = [C_MARKOVIAN, C_RAW, C_SIG]
plot_vals = [v if np.isfinite(v) else 0 for v in vals]
bars = axB.bar(labels, plot_vals, color=colors, width=0.6)
axB.set_yscale("log")
axB.set_ylim(0.02, 50)
for b, v in zip(bars, vals):
    if np.isfinite(v):
        axB.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=10)
    else:
        axB.text(b.get_x() + b.get_width() / 2, 0.025, "fails",
                 ha="center", va="bottom", fontsize=10, color="firebrick")
axB.set_ylabel(r"raw closed-loop cost $J$ (log)")
axB.set_title("H2 (representational): signature succeeds\n"
              r"Mackey-Glass limit cycle $\tau=6$", fontsize=11)
axB.annotate("raw-history unstable across capacity:\n0.28 / 15 / NaN",
             xy=(0.5, 0.93), xycoords="axes fraction", ha="center", va="top",
             fontsize=8, style="italic")

# --- Panel C: H2 robustness, Mackey-Glass chaotic tau=17 (raw cost J, log scale) ---
axC = axes[1, 0]
labels = ["markovian", "raw-history\n(deg 2)", "signature"]
vals = [43.8, 0.30, 0.088]
bars = axC.bar(labels, vals, color=[C_MARKOVIAN, C_RAW, C_SIG], width=0.6)
axC.set_yscale("log")
axC.set_ylim(0.05, 100)
for b, v in zip(bars, vals):
    axC.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.3g}",
             ha="center", va="bottom", fontsize=10)
axC.set_ylabel(r"raw closed-loop cost $J$ (log)")
axC.set_title("H2 robust to chaos: signature wins\n"
              r"Mackey-Glass chaotic $\tau=17$ (219$\times$ fewer features)",
              fontsize=11)

# --- Panel D: oracle-ladder error decomposition, high-gap cell (rho) ---
axD = axes[1, 1]
labels = ["full\nlearned", "oracle\ncritic", "oracle\nactor"]
vals = [0.520, 0.374, 0.000]
bars = axD.bar(labels, vals, color=C_LADDER, width=0.6)
for b, v in zip(bars, vals):
    axD.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=10)
# Reference: critic-only value-gradient agent on the same cell (Panel A signature).
axD.axhline(0.157, color=C_SIG, linestyle="--", linewidth=1.5,
            label=r"critic-only value-gradient ($\rho=0.157$)")
axD.set_ylabel(r"normalised sub-optimality $\rho$")
axD.set_title("Oracle-ladder decomposition (preliminary, 1-2 seeds)\n"
              "actor-critic error dominated by the actor", fontsize=11)
axD.set_ylim(0, 0.58)
axD.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8, frameon=True)

fig.suptitle("Signature-RL: H1/H2 results summary (2026-06-11) — lower is better",
             fontsize=13)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
