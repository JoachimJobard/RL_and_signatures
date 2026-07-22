"""Controlled-dynamics boards for the 5-seed final runs.

For each cell (one per group directory) and each learner, overlay the closed-loop trajectories of
the three representations (markovian / raw_history / signature) on a single 2x2 board -- controlled
state, control, error-from-target, cumulative cost -- via the library figure
``src.training.evaluate.plot_representation_comparison_from_data`` (repo stroke convention: trained
solid per-representation colour, uncontrolled baseline dashed, target dotted). Reads the saved
``eval.pkl`` of each run, so it recomputes nothing and works equally on the selection groups
(seed 42) or the final groups (seeds 0-4). One representative seed per board (the trajectory is a
single realisation; the seed-averaged quantitative comparison lives in the %opt table).

Usage:
  python final_runs_boards.py --groups <group_dir> [<group_dir> ...] --out <dir> [--seed 0]
                              [--learners value_gradient actor_critic policy_gradient]
Each <group_dir> is one cell's run directory (e.g. .../final_MG_1D_chaotic_<ts>/).
"""
import argparse
import pickle
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from src.training.evaluate import plot_representation_comparison_from_data

KINDS = ["markovian", "raw_history", "signature"]  # fixed board order
RUN_RE = re.compile(
    r"_(value_gradient|policy_gradient|actor_critic)_(markovian|raw_history|signature)_lr[\d.]+_seed(\d+)$"
)


def cell_of(group_dir: Path) -> str:
    """final_<cell>_<timestamp> -> <cell> (timestamp is <date>_<time>, two trailing fields)."""
    name = group_dir.name
    stem = name[len("final_"):] if name.startswith("final_") else name
    return re.sub(r"_\d{8}_\d{6}$", "", stem)


def discover(group_dir: Path, seed: int) -> dict[str, dict[str, dict]]:
    """learner -> kind -> eval_data, for the requested seed (skips missing eval.pkl)."""
    out: dict[str, dict[str, dict]] = {}
    for run_dir in sorted(group_dir.glob(f"*_seed{seed}")):
        m = RUN_RE.search(run_dir.name)
        if not m or int(m.group(3)) != seed:
            continue
        eval_pkl = run_dir / "eval.pkl"
        if not eval_pkl.exists():
            continue
        with open(eval_pkl, "rb") as fh:
            out.setdefault(m.group(1), {})[m.group(2)] = pickle.load(fh)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learners", nargs="+",
                    default=["value_gradient", "actor_critic", "policy_gradient"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for group_dir in args.groups:
        cell = cell_of(group_dir)
        by_learner = discover(group_dir, args.seed)
        for learner in args.learners:
            reps = by_learner.get(learner, {})
            variants = [(kind, reps[kind]) for kind in KINDS if kind in reps]
            if len(variants) < 2:
                print(f"[skip] {cell} / {learner}: only {len(variants)} representation(s) with eval.pkl at seed {args.seed}")
                continue
            fig = plot_representation_comparison_from_data(
                variants, title=f"{cell} -- {learner} (seed {args.seed})")
            path = args.out / f"board_{cell}_{learner}_seed{args.seed}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[saved] {path}  ({len(variants)} representations)")
            n_saved += 1
    print(f"\n{n_saved} board(s) written to {args.out}")


if __name__ == "__main__":
    main()
