"""Aggregate a representation-study experiment group into the H1/H2 comparison.

Reads every run directory under ``data/main_unified/<experiment-group>/``, extracts
each run's evaluation cost and its (representation, capacity, environment, seed)
from the saved ``config.yaml`` / ``eval.pkl``, computes the delayed-LQR oracle
ceiling for the *linear* environments, and reports the **normalised sub-optimality**
``(J - J_oracle)/|J_oracle|`` (or the raw cost for nonlinear environments, which
have no analytic oracle) aggregated across seeds with 95% confidence intervals.
Outputs a combined summary, a machine-readable aggregation file (so the figure
regenerates without recomputation), and the comparison figure — all into the group
directory.

Usage:
    uv run python run/study/aggregate_representation_study.py <experiment-group-dir>

Design and predicted outcomes: documents/methodology/experimental_design.md.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Records and discovery
# =============================================================================

@dataclass
class RunRecord:
    run_dir: str
    kind: str            # markovian | raw_history | signature
    capacity: int        # degree (polynomial) or depth (signature)
    env_name: str
    seed: int
    j_agent: float
    feature_dim: int
    is_linear: bool      # linear env -> analytic delayed-LQR oracle available


@dataclass
class CellAggregate:
    env_name: str
    kind: str
    capacity: int
    feature_dim: int
    metric_is_suboptimality: bool
    n_seeds: int
    mean: float
    std: float
    sem: float
    ci95: float
    values: list[float]


def _load_yaml(path: Path) -> dict:
    # resolve=False: the saved config.yaml keeps Hydra interpolations (e.g.
    # ${now:...} in wandb.name) that cannot resolve outside a Hydra run; the keys
    # this script reads (agent.signature, env.environment_params, seed, eval.T_sim)
    # are all concrete, so leaving interpolations unresolved is correct.
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore


def discover_runs(group_dir: Path) -> list[RunRecord]:
    """Build a RunRecord for every completed run dir under ``group_dir``."""
    from src.envs.env_rk_jax import JAXDDEEnv
    from src.representations.factory import make_representation
    import hydra

    records: list[RunRecord] = []
    env_cache: dict[str, Any] = {}
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir() and p.name != "slurm"):
        cfg_path, eval_path = run_dir / "config.yaml", run_dir / "eval.pkl"
        if not (cfg_path.exists() and eval_path.exists()):
            continue
        cfg = _load_yaml(cfg_path)
        with open(eval_path, "rb") as f:
            eval_data = pickle.load(f)

        sig = cfg["agent"]["signature"]
        kind = str(sig.get("kind", "signature"))
        capacity = int(sig["depth"] if kind == "signature" else sig.get("degree", 2))
        env_params = cfg["env"]["environment_params"]
        env_name = str(env_params["_target_"].split(".")[-1])

        # Instantiate the env (cached) to get its state dimension and linearity.
        cache_key = json.dumps(env_params, sort_keys=True, default=str)
        if cache_key not in env_cache:
            env_cache[cache_key] = hydra.utils.instantiate(env_params)
        env = env_cache[cache_key]
        is_linear = type(env).__name__ == "JAXDDEEnv" and isinstance(env, JAXDDEEnv)

        rep = make_representation(
            kind, window_length=int(sig["window_size"]) + 1, n_state=int(env.N),
            depth=int(sig["depth"]), degree=int(sig.get("degree", 2)),
        )
        records.append(RunRecord(
            run_dir=str(run_dir), kind=kind, capacity=capacity, env_name=env_name,
            seed=int(cfg["seed"]),
            j_agent=float(eval_data["eval_metrics"]["eval/total_cost_agent"]),
            feature_dim=int(rep.feature_dim), is_linear=is_linear,
        ))
    return records


def compute_oracle_costs(group_dir: Path) -> dict[str, float]:
    """For each LINEAR env+x0 in the group, the delayed-LQR oracle cost J_oracle,
    keyed by ``f"{env_name}|{x0}"`` (matching :func:`_oracle_key`)."""
    from src.solvers.oracle_agent import delayed_lqr_for_env, oracle_total_cost
    import hydra

    costs: dict[str, float] = {}
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        cfg_path, eval_path = run_dir / "config.yaml", run_dir / "eval.pkl"
        if not (cfg_path.exists() and eval_path.exists()):
            continue
        cfg = _load_yaml(cfg_path)
        env_params = cfg["env"]["environment_params"]
        env = hydra.utils.instantiate(env_params)
        if type(env).__name__ != "JAXDDEEnv":
            continue
        with open(eval_path, "rb") as f:
            x0 = np.asarray(pickle.load(f)["x0"], dtype=float)
        key = _oracle_key(str(env_params["_target_"].split(".")[-1]), x0)
        if key not in costs:
            lqr = delayed_lqr_for_env(env)
            costs[key] = oracle_total_cost(env, lqr, x0, float(cfg["eval"]["T_sim"]))
    return costs


def _oracle_key(env_name: str, x0: np.ndarray) -> str:
    return f"{env_name}|" + ",".join(f"{v:.4f}" for v in np.asarray(x0).reshape(-1))


# =============================================================================
# Aggregation (pure)
# =============================================================================

def normalised_suboptimality(j_agent: float, j_oracle: float) -> float:
    if j_oracle == 0.0:
        return float("nan")
    return (j_agent - j_oracle) / abs(j_oracle)


def aggregate(records: list[RunRecord], oracle_by_env: dict[str, float] | None) -> list[CellAggregate]:
    """Group records by (env, representation, capacity) and aggregate the metric
    across seeds. The metric is normalised sub-optimality where an oracle cost is
    provided for the env, else the raw cost."""
    oracle_by_env = oracle_by_env or {}
    groups: dict[tuple[str, str, int], list[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.env_name, r.kind, r.capacity), []).append(r)

    cells: list[CellAggregate] = []
    for (env_name, kind, capacity), recs in sorted(groups.items()):
        use_oracle = all(r.is_linear for r in recs) and env_name in oracle_by_env
        if use_oracle:
            j_oracle = oracle_by_env[env_name]
            values = [normalised_suboptimality(r.j_agent, j_oracle) for r in recs]
        else:
            values = [r.j_agent for r in recs]
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 0 else 0.0
        cells.append(CellAggregate(
            env_name=env_name, kind=kind, capacity=capacity,
            feature_dim=int(recs[0].feature_dim), metric_is_suboptimality=use_oracle,
            n_seeds=int(n), mean=float(np.mean(arr)) if n else float("nan"),
            std=std, sem=sem, ci95=float(1.96 * sem), values=[float(v) for v in arr],
        ))
    return cells


# =============================================================================
# Figure (pure, data-driven -> rebuildable)
# =============================================================================

def build_comparison_figure(cells: list[CellAggregate]):
    """One panel per environment: metric (sub-optimality or cost) vs feature
    dimension, a series per representation kind, with 95% CI error bars
    (Matplotlib). All series are trained-agent results and therefore solid; the
    representation kind is encoded in colour."""
    import matplotlib.pyplot as plt
    from src.utils.plot_style import STROKE_TRAINED, prepare_figure

    envs = sorted({c.env_name for c in cells})
    kind_color = {"signature": "#440154", "raw_history": "#21918c", "markovian": "#fde725"}
    n_panels = max(1, len(envs))
    fig, axes = plt.subplots(1, n_panels, figsize=(max(5.6, 4.8 * n_panels), 4.6),
                             squeeze=False)
    axes = axes[0]
    # Reserve room for the long normalised-sub-optimality y-axis label (it otherwise
    # spills past the canvas on a single narrow panel — flagged by check_layout).
    fig.subplots_adjust(left=0.16 if n_panels == 1 else 0.09, right=0.97, wspace=0.32)

    handles_by_kind: dict[str, Any] = {}
    any_log = False
    for col, env in enumerate(envs):
        ax = axes[col]
        sub = any(c.metric_is_suboptimality for c in cells if c.env_name == env)
        finite = [c for c in cells if c.env_name == env and np.isfinite(c.mean)]
        # Raw-cost (nonlinear) panels span orders of magnitude -> log-y; sub-optimality
        # panels stay linear (they can be ~0 or negative). Log-y needs strictly positive
        # means, so guard on it.
        ylog = (not sub) and bool(finite) and all(c.mean > 0 for c in finite)
        any_log = any_log or ylog
        for kind in ("signature", "raw_history", "markovian"):
            cs = sorted((c for c in cells
                         if c.env_name == env and c.kind == kind and np.isfinite(c.mean)),
                        key=lambda c: c.feature_dim)
            if not cs:
                continue
            means = np.array([c.mean for c in cs])
            errs = np.array([c.ci95 for c in cs])
            if ylog:
                # Clip the lower error so mean - lower stays > 0 on the log axis.
                yerr = np.vstack([np.minimum(errs, means * (1.0 - 1e-6)), errs])
            else:
                yerr = errs
            line = ax.errorbar(
                [c.feature_dim for c in cs], means, yerr=yerr, fmt=f"o{STROKE_TRAINED}",
                color=kind_color.get(kind), capsize=3, label=kind)
            handles_by_kind.setdefault(kind, line)
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(env)
        ax.set_xlabel(r"$\dim\Phi$")
        ax.set_ylabel(r"$(J-J_{\mathrm{oracle}})/|J_{\mathrm{oracle}}|$" if sub else r"$J$")
        ax.grid(True, alpha=0.3, which="both")
    if not envs:
        axes[0].set_title("(no data)")

    fig.suptitle("Representation comparison (value-gradient backbone)", fontsize=12)
    handles = list(handles_by_kind.values())
    labels = list(handles_by_kind.keys())
    prepare_figure(
        fig, fname="comparison", axes=list(axes),
        handles=handles or None, labels=labels or None,
        reserve_bottom=0.32, legend_y=0.14, legend_fontsize=8,
        formula=(r"Value linear in $\Phi$, swept capacity $\dim\Phi$. Linear cells: "
                 r"normalised sub-optimality vs the delayed-LQR oracle (linear $y$); "
                 r"nonlinear cells: raw cost $J$ (log $y$). 95% CIs over seeds; "
                 r"divergent (non-finite) variants omitted."),
    )
    return fig


# =============================================================================
# Figure rebuild (no recomputation)
# =============================================================================

def load_cells(group_dir: Path) -> list[CellAggregate]:
    """Reload the aggregated cells from ``aggregation_data.json`` (written by a prior
    full run), so the comparison figure rebuilds without re-discovering runs or
    recomputing the oracle — the repo's figure-rebuild contract."""
    data = json.loads((group_dir / "aggregation_data.json").read_text())
    return [CellAggregate(**d) for d in data]


def save_comparison_figure(cells: list[CellAggregate], group_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig = build_comparison_figure(cells)
    fig.savefig(str(group_dir / "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Driver
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_dir", type=Path, help="data/main_unified/<experiment-group>/")
    parser.add_argument("--replot", action="store_true",
                        help="rebuild comparison.png from the existing "
                             "aggregation_data.json, without re-discovering runs or "
                             "recomputing the oracle (pure plotting; runs anywhere).")
    args = parser.parse_args()
    group_dir = args.group_dir
    if not group_dir.is_dir():
        raise SystemExit(f"Not a directory: {group_dir}")

    if args.replot:
        agg_json = group_dir / "aggregation_data.json"
        if not agg_json.exists():
            raise SystemExit(f"--replot needs {agg_json} (run a full aggregation first).")
        cells = load_cells(group_dir)
        save_comparison_figure(cells, group_dir)
        print(f"[aggregate] rebuilt comparison.png from {agg_json.name} "
              f"({len(cells)} cells) in {group_dir}")
        return

    print(f"[aggregate] discovering runs under {group_dir} ...")
    records = discover_runs(group_dir)
    print(f"[aggregate] {len(records)} run(s) found.")
    if not records:
        raise SystemExit("No completed runs (config.yaml + eval.pkl) found.")

    oracle_by_env = compute_oracle_costs(group_dir)
    if oracle_by_env:
        print(f"[aggregate] oracle ceiling computed for: {sorted(oracle_by_env)}")
    cells = aggregate(records, {k.split('|')[0]: v for k, v in oracle_by_env.items()})

    # Persist: machine-readable aggregation (for replot) + human summary + figure.
    (group_dir / "aggregation_data.json").write_text(
        json.dumps([asdict(c) for c in cells], indent=2))
    import yaml
    (group_dir / "summary.yaml").write_text(yaml.safe_dump(
        {"n_runs": len(records), "cells": [asdict(c) for c in cells]}, sort_keys=False))

    save_comparison_figure(cells, group_dir)
    print(f"[aggregate] wrote summary.yaml, aggregation_data.json, comparison.png in {group_dir}")


if __name__ == "__main__":
    main()
