"""Measure whether the actor-critic's clip_action binds, under the campaign's own invocation.

Read-only with respect to tracked files: the agent is built through the repository's own
build_environment / build_agent, and the jitted action function is wrapped IN MEMORY after
construction to record the pre-clip action. Nothing on disk is modified.

The pre-clip action is mu + noise, which is exactly what signatures_jax.py:374 forms before
signatures_jax.py:375 clips it. The clip binds on a step iff |mu + noise|_inf > clip_action.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path("/home/rho/Documents/work/phd/git_repositories/RL_and_signatures")
sys.path.insert(0, str(REPO))

from src.training.train import build_environment, build_agent  # noqa: E402

CELL = sys.argv[1] if len(sys.argv) > 1 else "hopfield_linear"
REP = sys.argv[2] if len(sys.argv) > 2 else "raw_history"
N_EPISODES = int(sys.argv[3]) if len(sys.argv) > 3 else 20

# The array's per-cell window / horizon table (rl_array.slurm:73-81).
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
            "agent=signatures",
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

clip = cfg.agent.training.clip_action
print(f"cell={CELL} env={env_name} rep={REP} win={win} n_episodes={N_EPISODES}")
print(f"RESOLVED agent.training.clip_action = {clip!r}   (the value the campaign would use)")
print(f"RESOLVED agent.training.scale       = {cfg.agent.training.scale!r}")
print(f"RESOLVED network.normalize_entries  = {cfg.agent.network.normalize_entries!r}")

env = build_environment(cfg)
agent = build_agent(cfg, env)

# Wrap the jitted action function in memory. Returns (action, mu, noise, key, noise_state);
# the pre-clip action formed at signatures_jax.py:374 is mu + noise.
_orig = agent._jit_select_action
rec_pre: list[float] = []
rec_post: list[float] = []


def _wrapped(*a, **k):
    action, mu, noise, key, ns = _orig(*a, **k)
    pre = jnp.asarray(mu) + jnp.asarray(noise)
    rec_pre.append(float(jnp.max(jnp.abs(pre))))
    rec_post.append(float(jnp.max(jnp.abs(jnp.asarray(action)))))
    return action, mu, noise, key, ns


agent._jit_select_action = _wrapped

agent.train()

pre = np.asarray(rec_pre)
post = np.asarray(rec_post)
n = pre.size
if n == 0:
    print("NO ACTIONS RECORDED — the wrap did not intercept; result is NOT MEASURED")
    raise SystemExit(2)

bound = float(clip) if clip is not None else np.inf
bind = pre > bound + 1e-12
print()
print(f"steps recorded            : {n}")
print(f"max |mu + noise| (pre-clip): {pre.max():.6f}")
print(f"max |action|    (post-clip): {post.max():.6f}")
print(f"clip bound                 : {bound}")
print(f"steps where the clip BINDS : {int(bind.sum())} / {n} = {100.0 * bind.mean():.3f}%")
if bind.any():
    print(f"most extreme pre-clip |u|  : {pre[bind].max():.6f}")
print()
print(f"MEASURED_BINDING_PCT={100.0 * bind.mean():.4f}")
print(f"MEASURED_MAX_PRECLIP={pre.max():.6f}")
