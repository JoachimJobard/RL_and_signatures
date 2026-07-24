#!/usr/bin/env python3
"""Figure 4 — relative optimality eta across environments, learners, and representations.

Reads the per-seed relative optimality table (``f2_relopt_10seed.csv``: columns cell, learner,
representation, seed, oracle_cost_reduction, cost_reduction, eta_pct) and draws, for every
(environment, learner) panel, the three representations side by side: one dot per seed, a box for the
quartiles, a bar for the median, whiskers to the extreme SUCCESSFUL seeds. Seeds with eta <= 0 (no
better than no control) or a non-finite cost are drawn off-scale below a separator; the count k/10
reports the reliable seeds (eta > 0). The magnitude of a failure — an artefact of the small no-control
cost — is deliberately NOT plotted.

The MG_dt0.25 cell is excluded by construction (superseded by MG_dt0.05); see the campaign README.

Reusable by subset. The paper's full Figure 4 is the default (3 learners x 3 representations); the
value-gradient + {markovian, signature} variant for the VG+signature paper is

    python3 fig_f4_relopt.py --learners value_gradient \
        --representations markovian signature --row-axis learners \
        --csv <path>/f2_relopt_10seed.csv --out figF4_vg_marksig.png

Usage:
    python3 fig_f4_relopt.py [--csv CSV] [--learners ...] [--representations ...]
                             [--row-axis {cells,learners}] [--out PNG] [--stats-csv CSV]
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Environments in order of increasing delay. (csv_name, display title, delay). MG_dt0.25 is omitted.
CELLS = [
    ("harmonic", "Harmonic oscillator", 0),
    ("linear_dde", "Linear DDE", 1),
    ("MG_dt0.05", "Mackey–Glass", 6),
]
LEARNER_LABEL = {
    "value_gradient": "value gradient",
    "policy_gradient": "policy gradient",
    "actor_critic": "actor–critic",
}
REP_LABEL = {"markovian": "markovian", "raw_history": "raw history", "signature": "signature"}
REP_COLOUR = {"markovian": "#6E6E6E", "raw_history": "#D95F02", "signature": "#1B66B4"}

FAIL_Y = -0.085          # fixed off-scale row for failed seeds (magnitude not plotted)
SEP_Y = 0.0              # separator: eta = 0 (no better than no control)
Y_BOTTOM, Y_TOP = -0.16, 1.06


def load(csv_path: str) -> dict:
    """eta (fraction) per (cell, learner, representation) -> list of (seed, eta)."""
    d: dict = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                eta = float(r["eta_pct"]) / 100.0
            except (TypeError, ValueError):
                eta = float("nan")
            d[(r["cell"], r["learner"], r["representation"])].append((int(r["seed"]), eta))
    return d


def summarise(vals: list) -> dict:
    good = sorted(e for _, e in vals if math.isfinite(e) and e > 0)
    n, k = len(vals), len([e for _, e in vals if math.isfinite(e) and e > 0])
    out = {"n": n, "k": k, "good": good}
    if good:
        # Quartiles by linear interpolation (numpy-free).
        def q(p):
            if len(good) == 1:
                return good[0]
            idx = p * (len(good) - 1)
            lo = int(math.floor(idx))
            frac = idx - lo
            hi = min(lo + 1, len(good) - 1)
            return good[lo] + frac * (good[hi] - good[lo])

        out.update(med=q(0.5), q1=q(0.25), q3=q(0.75), lo=good[0], hi=good[-1])
    return out


def draw_panel(ax, d, cell_csv, learner, reps, jitter=0.055):
    xs = list(range(len(reps)))
    for x, rep in zip(xs, reps):
        s = summarise(d.get((cell_csv, learner, rep), []))
        colour = REP_COLOUR[rep]
        # Box + median + whiskers over the SUCCESSFUL seeds.
        if s["k"] >= 1:
            stat = [{
                "med": s["med"], "q1": s["q1"], "q3": s["q3"],
                "whislo": s["lo"], "whishi": s["hi"], "fliers": [],
            }]
            bp = ax.bxp(stat, positions=[x], widths=0.42, showfliers=False,
                        patch_artist=True, manage_ticks=False)
            for box in bp["boxes"]:
                box.set(facecolor=colour, alpha=0.20, edgecolor=colour, linewidth=1.3)
            for med in bp["medians"]:
                med.set(color=colour, linewidth=2.2)
            for w in bp["whiskers"] + bp["caps"]:
                w.set(color=colour, linewidth=1.1)
            # Per-seed dots (deterministic jitter by rank, no RNG).
            good = s["good"]
            for i, e in enumerate(good):
                jx = x + jitter * ((i / max(len(good) - 1, 1)) - 0.5) * 2
                ax.plot(jx, e, "o", ms=3.4, color=colour, mec="white", mew=0.4, zorder=5)
        # Failed seeds: off-scale row, magnitude NOT plotted.
        n_fail = s["n"] - s["k"]
        if n_fail:
            for i in range(n_fail):
                jx = x + jitter * ((i / max(n_fail - 1, 1)) - 0.5) * 2
                ax.plot(jx, FAIL_Y, "x", ms=4.2, color=colour, mew=1.2, zorder=5)
        # k/10 under the x-axis.
        ax.text(x, Y_BOTTOM + 0.012, f"{s['k']}/{s['n']}", ha="center", va="bottom",
                fontsize=8, color=colour, fontweight="bold")

    ax.axhline(SEP_Y, color="#999999", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([REP_LABEL[r] for r in reps], fontsize=8.5)
    ax.set_xlim(-0.6, len(reps) - 0.4)
    ax.set_ylim(Y_BOTTOM, Y_TOP)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="path to f2_relopt_10seed.csv (default: campaign snapshot)")
    ap.add_argument("--learners", nargs="+", default=["value_gradient", "policy_gradient", "actor_critic"])
    ap.add_argument("--representations", nargs="+", default=["markovian", "raw_history", "signature"])
    ap.add_argument("--row-axis", choices=["cells", "learners"], default="cells",
                    help="'cells' (paper: rows=envs, cols=learners) or 'learners' (rows=learners, cols=envs)")
    ap.add_argument("--hide-row-label", action="store_true",
                    help="show only the metric symbol on the y-axis, not the row (learner/cell) name")
    ap.add_argument("--out", default="figF4_relopt.png")
    ap.add_argument("--stats-csv", default=None, help="also write the per-group summary stats here")
    args = ap.parse_args()

    csv_path = args.csv
    if csv_path is None:
        import campaign  # local import so the script also runs standalone with --csv
        csv_path = os.path.join(campaign.DATA_ROOT, "f2_relopt_10seed.csv")
    d = load(csv_path)

    learners, reps = args.learners, args.representations
    if args.row_axis == "cells":
        row_items = [(c[0], c[1]) for c in CELLS]            # (csv, title)
        col_items = [(l, LEARNER_LABEL.get(l, l)) for l in learners]
        cell_of = lambda ri, ci: (row_items[ri][0], col_items[ci][0])
    else:
        row_items = [(l, LEARNER_LABEL.get(l, l)) for l in learners]
        col_items = [(c[0], c[1]) for c in CELLS]
        cell_of = lambda ri, ci: (col_items[ci][0], row_items[ri][0])

    nrow, ncol = len(row_items), len(col_items)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.35 * ncol + 1.1, 2.55 * nrow + 0.7),
                             squeeze=False, sharey=True)

    for ri in range(nrow):
        for ci in range(ncol):
            ax = axes[ri][ci]
            cell_csv, learner = cell_of(ri, ci)
            draw_panel(ax, d, cell_csv, learner, reps)
            if ri == 0:
                ax.set_title(col_items[ci][1], fontsize=10.5, pad=8)
            if ci == 0:
                ylab = "η" if args.hide_row_label else f"{row_items[ri][1]}\nη"
                ax.set_ylabel(ylab, fontsize=9.5)

    # Legend: representation colours + the two glyphs.
    handles = [Line2D([0], [0], marker="o", ls="", color=REP_COLOUR[r], mec="white",
                      label=REP_LABEL[r]) for r in reps]
    handles += [
        Line2D([0], [0], marker="o", ls="", color="#444", mec="white", label="reliable seed (η>0)"),
        Line2D([0], [0], marker="x", ls="", color="#444", label="failed seed (η≤0), off-scale"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8.2,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Relative optimality η  (1 = delayed-LQR oracle, 0 = no control)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"saved {args.out}  ({nrow}x{ncol} panels; learners={learners}; reps={reps})")

    if args.stats_csv:
        with open(args.stats_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "learner", "representation", "k", "n", "median", "q1", "q3",
                        "whisker_lo", "whisker_hi"])
            for c in CELLS:
                for learner in learners:
                    for rep in reps:
                        s = summarise(d.get((c[0], learner, rep), []))
                        row = [c[0], learner, rep, s["k"], s["n"]]
                        row += [f"{s[k]:.4f}" if s.get(k) is not None and s["k"] else ""
                                for k in ("med", "q1", "q3", "lo", "hi")]
                        w.writerow(row)
        print(f"saved {args.stats_csv}")


if __name__ == "__main__":
    main()
