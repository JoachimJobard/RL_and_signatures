"""Aggregate per-task H1/H2 summaries (one per cell x seed) into the cross-seed table.

The finalize step of the H1/H2 job array. Reads every ``<cell>_seed<S>/summary.json`` under a
shared parent directory, groups by (cell, representation), reports mean +/- 95% CI of value R^2,
gradient cos, and closed-loop I across seeds, and reads off the H1/H2 verdicts with CIs.

  H1 := raw_history vs markovian      H2 := signature vs raw_history

Usage:
    uv run python run/study/h1h2_aggregate.py --parent data/h1h2_evaluation/<run_dir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPS = ["markovian", "raw_history", "signature"]


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="dir containing <cell>_seed<S>/summary.json")
    args = ap.parse_args()
    parent = Path(args.parent)
    files = sorted(parent.glob("*_seed*/summary.json"))
    if not files:
        raise SystemExit(f"no summaries under {parent}")

    # cell -> rep -> metric -> [values across seeds]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    cells = []
    for f in files:
        s = json.loads(f.read_text())
        cell = s["cell"]
        if cell not in cells:
            cells.append(cell)
        for rep, r in s["reps"].items():
            for k in ("r2", "cos", "I"):
                data[cell][rep][k].append(r[k])

    for cell in cells:
        seeds = max(len(data[cell][rep]["I"]) for rep in data[cell])
        print(f"\n=== cell '{cell}'  (n_seeds={seeds}) ===")
        print(f"{'rep':>12}{'valR2':>16}{'gradcos':>16}{'I_cl':>18}")
        for rep in REPS:
            if rep not in data[cell]:
                continue
            d = data[cell][rep]
            r2, cos, I = (np.mean(d["r2"]), np.mean(d["cos"]), np.mean(d["I"]))
            print(f"{rep:>12}{r2:>9.3f}+/-{ci95(d['r2']):<5.3f}"
                  f"{cos:>9.3f}+/-{ci95(d['cos']):<5.3f}{I:>11.4f}+/-{ci95(d['I']):<6.4f}")

        def mean_I(rep): return float(np.mean(data[cell][rep]["I"])) if rep in data[cell] else np.nan
        if all(r in data[cell] for r in REPS):
            iM, iR, iS = mean_I("markovian"), mean_I("raw_history"), mean_I("signature")
            h1 = (iM - iR) / abs(iM) if np.isfinite(iM) else np.nan
            h2 = (iR - iS) / abs(iR) if np.isfinite(iR) else np.nan
            print(f"  H1 (raw_history > markovian): {'HOLDS' if h1 > 0.03 else 'fails'} "
                  f"(mean I {iR:.4f} vs {iM:.4f})")
            print(f"  H2 (signature > raw_history): {'HOLDS' if h2 > 0.03 else 'fails'} "
                  f"(mean I {iS:.4f} vs {iR:.4f})")


if __name__ == "__main__":
    main()
