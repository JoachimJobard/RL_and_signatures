"""Shared helpers for the paper figures. Reuses the tested build_agent + collect_evaluation_data, so a
figure never re-implements env instantiation or evaluation. All re-evaluation is done from the saved
checkpoint on the cell's clean fixed initial condition (campaign.CELLS[...]['x0'], ['burn_steps']);
nothing is retrained.

Every figure script writes BOTH a vector-friendly PNG and a CSV of the plotted quantities, so the paper
can either include the PNG or rebuild the figure in pgfplots from the CSV (the F2 pattern).
"""
import os
import sys
import glob
import csv as _csv
import numpy as np
from omegaconf import OmegaConf
import hydra

# The scripts run from the repo root (the launcher cd's there via _environment.sh); make the package
# importable either way.
sys.path.insert(0, os.getcwd())
from src.training.train import build_agent  # noqa: E402
from src.training.evaluate import collect_evaluation_data  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6,
})


def find_run(data_root, group_glob, learner, kind, seed):
    """Return the run directory for (learner, kind, seed) within a cell's group glob, or None."""
    hits = sorted(glob.glob(f"{data_root}/{group_glob}/*_{learner}_{kind}_*_seed{seed}"))
    return hits[0] if hits else None


def find_oracle(data_root, group_glob):
    """Return the analytic-oracle run directory for a cell (the delayed-LQR reference), or None."""
    hits = sorted(glob.glob(f"{data_root}/{group_glob}/*oracle_reference*"))
    return hits[0] if hits else None


def reeval_run(run_dir, x0, burn_steps, load_checkpoint=True):
    """Rebuild the agent from run_dir/config.yaml, restore its checkpoint, and evaluate on the fixed IC.
    Returns the collect_evaluation_data dict (times, states_agent, states_no_ctrl, cum_cost_*, x_ref,
    eval_metrics, ...). load_checkpoint=False builds the agent without restoring params (analytic
    oracle: the controller is closed-form and needs no learned weights)."""
    cfg = OmegaConf.load(f"{run_dir}/config.yaml")
    env = hydra.utils.instantiate(cfg.env.environment_params)
    agent = build_agent(cfg, env)
    if load_checkpoint and os.path.exists(f"{run_dir}/checkpoint_agent.pkl"):
        agent.load(f"{run_dir}/checkpoint_agent.pkl")
    ed = collect_evaluation_data(agent, np.asarray(x0, dtype=float), float(cfg.eval.T_sim),
                                 burning_steps=burn_steps)
    return ed, cfg


def state0(ed):
    """First component of a trajectory array, as a 1-D series aligned with ed['times']."""
    t = np.asarray(ed["times"]).ravel()
    return t, np.asarray(ed["states_agent"]).reshape(len(t), -1)[:, 0], \
        np.asarray(ed["states_no_ctrl"]).reshape(len(t), -1)[:, 0]


def target_of(ed):
    xr = ed.get("x_ref")
    xr = np.ravel(np.asarray(xr)) if xr is not None else None
    return float(xr[0]) if (xr is not None and xr.size) else 0.0


def eta_of(ed, oracle_cost_reduction):
    cr = ed["eval_metrics"]["eval/cost_reduction_pct"]
    return (cr / oracle_cost_reduction * 100.0) if np.isfinite(cr) else float("nan")


def save(fig, out_path_png, rows=None, header=None):
    """Save the figure to PNG and, if rows given, the plotted data to a sibling CSV."""
    os.makedirs(os.path.dirname(out_path_png), exist_ok=True)
    fig.savefig(out_path_png)
    plt.close(fig)
    print(f"saved {out_path_png}")
    if rows is not None:
        out_csv = os.path.splitext(out_path_png)[0] + ".csv"
        with open(out_csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            if header:
                w.writerow(header)
            w.writerows(rows)
        print(f"saved {out_csv}")
