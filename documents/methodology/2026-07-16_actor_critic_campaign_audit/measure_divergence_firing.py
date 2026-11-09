"""Measure whether the divergence cut fires, and whether it fires ASYMMETRICALLY across
representations -- the same failure mode that action clipping had.

The cut is `jnp.linalg.norm(x) > training.divergence_threshold` inside _is_episode_done
(signatures_jax.py:477, value_gradient_jax.py:628). It carries no log line, so a run cannot
be told from its output whether the cut fired at all. This probe wraps _is_episode_done in
memory (no tracked file is modified) and records, per episode, whether termination was caused
by the divergence bound or by reaching the horizon.

A cut that fires on one representation and not another does not cancel in the H1 contrast:
the two arms then receive different amounts of training experience.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from hydra import compose, initialize_config_dir

REPO = Path("/home/rho/Documents/work/phd/git_repositories/RL_and_signatures")
sys.path.insert(0, str(REPO))

from src.training.train import build_environment, build_agent  # noqa: E402

CELL = sys.argv[1]
REP = sys.argv[2]
AGENT = sys.argv[3] if len(sys.argv) > 3 else "signatures"
N_EPISODES = int(sys.argv[4]) if len(sys.argv) > 4 else 20

TABLE = {
    "double_integrator": (10, 15.0),
    "linear_dde_scalar": (8, 20.0),
    "hopfield_linear": (5, 15.0),
    "hopfield_nonlinear": (5, 15.0),
    "MG_1D_chaotic": (71, 60.0),
}
ENV_OF = {
    "markovian": "double_integrator",
    "linear_dde": "linear_dde_scalar",
    "hopfield_linear": "hopfield_linear",
    "hopfield_nonlinear": "hopfield_nonlinear",
    "mg_chaotic": "MG_1D_chaotic",
}
env_name = ENV_OF[CELL]
win, maxt = TABLE[env_name]
rep_arg = (
    ["agent.signature.kind=signature", "agent.signature.depth=2"]
    if REP == "signature"
    else [f"agent.signature.kind={REP}", "agent.signature.degree=2"]
)

with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
    cfg = compose(
        config_name="config_unified",
        overrides=[
            f"agent={AGENT}",
            f"env={env_name}",
            *rep_arg,
            f"agent.signature.window_size={win}",
            "agent.signature.force_signature_window=true",
            "agent.training.divergence_threshold=100.0",
            f"agent.training.n_episodes={N_EPISODES}",
            f"agent.training.max_time={maxt}",
            "seed=0",
            "debug=true",
            "wandb.mode=offline",
        ],
    )

env = build_environment(cfg)
agent = build_agent(cfg, env)

thr = float(cfg.agent.training.divergence_threshold)
_orig = agent._is_episode_done
peak_norm = [0.0]
n_div_calls = [0]
n_calls = [0]


def _wrapped(x, time_only: bool = False):
    try:
        nx = float(jnp.linalg.norm(jnp.asarray(x)))
        peak_norm[0] = max(peak_norm[0], nx)
        n_calls[0] += 1
        if (not time_only) and nx > thr:
            n_div_calls[0] += 1
    except Exception:
        pass
    return _orig(x, time_only) if time_only is not False else _orig(x)


agent._is_episode_done = _wrapped
agent.train()

print()
print(f"AGENT={AGENT} cell={CELL} rep={REP} threshold={thr}")
print(f"termination checks           : {n_calls[0]}")
print(f"checks where the cut FIRED   : {n_div_calls[0]}")
print(f"peak ||x|| reached           : {peak_norm[0]:.6f}")
print(f"MEASURED_DIV_FIRED={n_div_calls[0]}")
print(f"MEASURED_PEAK_NORM={peak_norm[0]:.6f}")
