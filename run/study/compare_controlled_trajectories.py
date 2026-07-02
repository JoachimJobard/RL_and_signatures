"""Overlay closed-loop trajectories across representations for one study cell.

For an experiment group (one representation x capacity x seed per run dir), load each
run's saved ``eval.pkl`` (the controlled / uncontrolled trajectory arrays produced by
:func:`src.training.evaluate.collect_evaluation_data`) and overlay the controlled
trajectories of every representation on shared axes, for a single seed. This is the
cross-representation comparison the per-variant ``figure_agent_vs_no_control.png`` does
not provide: it answers "how does the controlled trajectory differ across markovian /
raw-history / signature representations on the same plant".

The figure is rebuilt purely from saved data (never retrains), consistent with the
repo's replot contract; if a group's ``eval.pkl`` files are not present locally (e.g.
only the PNGs were rapatriated from the cluster), pull them first, for example:

    rsync -avzP --include='*/' --include='eval.pkl' --include='config.yaml' \\
        --exclude='*' \\
        jeanzay-any:'$WORK/git_repositories/RL_and_signatures/data/main_unified/mackey_glass_limit_cycle_study/' \\
        data/main_unified/mackey_glass_limit_cycle_study/

Usage:
    uv run python run/study/compare_controlled_trajectories.py <group-dir> \\
        [--seed 0] [--out FILE.png] [--title "..."]
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from src.training.evaluate import plot_representation_comparison_from_data


# Order representations as markovian -> raw-history (by degree) -> signature (by depth),
# so the overlay reads from least to most expressive feature map.
_KIND_ORDER = {"markovian": 0, "raw_history": 1, "signature": 2}


def _load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore


def _variant_key_and_label(cfg: dict) -> tuple[tuple[int, int], str]:
    """Return a sort key and a human-readable label from a run's config.

    The label uses the established capacity vocabulary: ``deg`` for the polynomial
    degree of the markovian / raw-history monomial basis, ``depth`` for the
    signature truncation level.
    """
    sig = cfg["agent"]["signature"]
    kind = str(sig.get("kind", "signature"))
    if kind == "signature":
        capacity = int(sig["depth"])
        label = f"signature depth{capacity}"
    else:
        capacity = int(sig.get("degree", 2))
        label = f"{kind} deg{capacity}"
    return (_KIND_ORDER.get(kind, 99), capacity), label


def discover_variants(group_dir: Path, seed: int) -> list[tuple[str, dict]]:
    """Load ``(label, eval_data)`` for every run of the given seed, sorted by
    representation expressiveness then capacity."""
    seed_suffix = re.compile(rf"_seed{seed}$")
    found: list[tuple[tuple[int, int], str, dict]] = []
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir() and p.name != "slurm"):
        if not seed_suffix.search(run_dir.name):
            continue
        cfg_path, eval_path = run_dir / "config.yaml", run_dir / "eval.pkl"
        if not (cfg_path.exists() and eval_path.exists()):
            print(f"[skip] {run_dir.name}: missing config.yaml or eval.pkl")
            continue
        cfg = _load_yaml(cfg_path)
        with open(eval_path, "rb") as f:
            eval_data = pickle.load(f)
        key, label = _variant_key_and_label(cfg)
        found.append((key, label, eval_data))
    found.sort(key=lambda t: t[0])
    return [(label, data) for _, label, data in found]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("group_dir", type=Path, help="experiment group directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed whose runs are overlaid (default: 0)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output PNG path (default: <group>/representation_trajectories_seed<seed>.png)")
    parser.add_argument("--title", type=str, default=None, help="figure title override")
    args = parser.parse_args()

    if not args.group_dir.is_dir():
        parser.error(f"not a directory: {args.group_dir}")

    variants = discover_variants(args.group_dir, args.seed)
    if not variants:
        parser.error(
            f"no runs with eval.pkl for seed {args.seed} under {args.group_dir}. "
            "If only PNGs were rapatriated, pull the eval.pkl files first (see module docstring)."
        )
    print(f"[overlay] {len(variants)} representations (seed {args.seed}): "
          + ", ".join(lab for lab, _ in variants))

    title = args.title or (f"Controlled trajectories by representation — "
                           f"{args.group_dir.name} (seed {args.seed})")
    fig = plot_representation_comparison_from_data(variants, title=title)

    out = args.out or (args.group_dir / f"representation_trajectories_seed{args.seed}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[overlay] wrote {out}")

    # Provenance: which run dirs fed the figure, so the number traces back to inputs.
    manifest = out.with_suffix(".inputs.json")
    with open(manifest, "w") as f:
        json.dump({"group": str(args.group_dir), "seed": args.seed,
                   "variants": [lab for lab, _ in variants]}, f, indent=2)
    print(f"[overlay] wrote {manifest}")


if __name__ == "__main__":
    main()
