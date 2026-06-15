"""WHERE in representation space is the value gradient badly estimated? (real platoon)

Train the linear-in-features critic (off-manifold cheat, linearised label, min-norm),
then PROBE the control gradient u = 1/2 R^-1 B^T dV/dx against the oracle u* = -K xi over
a grid of two interpretable axes:
  - OFF-MANIFOLD distance eta : perturb the current state with the history frozen
    (the direction the control reads but on-policy data never samples);
  - DISTANCE-FROM-ORIGIN ||x|| : probe windows from oracle rollouts at several IC scales
    (the nonlinearity axis -- the linearised value is exact only near the origin).

For each probe we record the magnitude blow-up |u_rep|/|u*|, the cosine, and the relative
error ||u_rep - u*||/||u*||, then bin by (||x||, eta). The map LOCALISES the bad gradient:
the hypothesis is that it is small on-manifold near the origin and grows with BOTH eta
(off-manifold, geometry) and ||x|| (far from origin, nonlinearity).

Usage:
    uv run python run/study/mwe_platoon_gradient_error_map.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REGIME = dict(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
TF = 8.0
WIN_LEN = 5
KIND, DEG = "raw_history", 2
N_IC_TRAIN = 16
AUG_PER = 4
EPS_TRAIN = 0.2
PROBE_SCALES = [0.3, 1.0, 3.0, 10.0]   # IC scales -> spread ||x||
ETAS = [0.0, 0.05, 0.2, 0.5]     # off-manifold distance
PROBE_SUB = 5
M_PERT = 3
IC_SCALE = 0.4
SEED = 0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R), float(env.max_delay), env.step_size)


def canonical_x0(n):
    x0 = np.zeros(n); x0[1] = 0.5
    return x0


def diverse_ics(n, n_ic, seed, scale=1.0):
    rng = np.random.default_rng(seed); ics = [scale * canonical_x0(n)]
    for _ in range(n_ic - 1):
        x0 = np.zeros(n); x0[1::2] = scale * IC_SCALE * rng.standard_normal(n // 2); ics.append(x0)
    return ics


def rollout(env, gain, k, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws = []; t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws)


def xi_of(W):
    return W[::-1].reshape(-1)


def main():
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); n = env.N
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T
    rep = make_representation(KIND, window_length=WIN_LEN, n_state=n, depth=DEG, degree=DEG)
    feat = jax.jit(rep.feature_fn)

    # train: on-manifold + off-manifold cheat, linearised label, min-norm
    rng = np.random.default_rng(SEED)
    onman = np.concatenate([rollout(env, gain, k, np.asarray(ic))
                            for ic in diverse_ics(n, N_IC_TRAIN, SEED)], 0)
    aug = [onman]
    for _ in range(AUG_PER):
        aug.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(aug, 0)
    Vtr = np.array([-(xi_of(W) @ P @ xi_of(W)) for W in Wtr])
    Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
    theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    u_of = lambda W: half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1]

    # probe windows across IC scales: EARLY/transient windows keep ||x|| large.
    probe = []
    for s in PROBE_SCALES:
        for ic in diverse_ics(n, 4, SEED + 1, scale=s):
            probe.append(rollout(env, gain, k, np.asarray(ic))[:20])
    probe = np.concatenate(probe, 0)

    # for each probe window and eta, record (||x||, eta, mag_ratio, cos, grad_rel_err, value_rel_err)
    rows = []
    for W in probe:
        xnorm = float(np.linalg.norm(W[-1]))
        for eta in ETAS:
            for _ in range(M_PERT if eta > 0 else 1):
                Wp = W.copy()
                if eta > 0:
                    Wp[-1] = Wp[-1] + eta * rng.standard_normal(n)
                ur = u_of(Wp); us = -gain @ xi_of(Wp)
                nus = np.linalg.norm(us) + 1e-12
                Vth = float(np.dot(theta, np.asarray(feat(jnp.asarray(Wp)))))
                Vstar = float(-(xi_of(Wp) @ P @ xi_of(Wp)))
                rows.append((xnorm, eta, np.linalg.norm(ur) / nus,
                             float(ur @ us / (np.linalg.norm(ur) * nus + 1e-12)),
                             np.linalg.norm(ur - us) / nus,
                             abs(Vth - Vstar) / (abs(Vstar) + 1e-12)))
    rows = np.array(rows)

    # bin by ||x|| terciles x eta
    xq = np.quantile(rows[:, 0], [0, 1 / 3, 2 / 3, 1.0])
    print(f"\nGRADIENT-ERROR MAP | rep={KIND} d{DEG} (trained: cheat eps={EPS_TRAIN})")
    print(f"mean ||u*|| baseline ~ {np.mean([np.linalg.norm(-gain@xi_of(W)) for W in probe]):.4f}")
    print(f"\n|u_rep|/|u*|   (magnitude blow-up):")
    print(f"{'||x|| bin':>16} " + "".join(f"eta={e:<7.2f}" for e in ETAS))
    for b in range(3):
        lo, hi = xq[b], xq[b + 1]
        sel = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
        cells = [np.mean(rows[sel & (rows[:, 1] == e), 2]) if np.any(sel & (rows[:, 1] == e)) else np.nan
                 for e in ETAS]
        print(f"[{lo:5.2f},{hi:5.2f}]   " + "".join(f"{c:>11.1f}" for c in cells))
    print(f"\ncosine(u_rep, u*)   (gradient DIRECTION):")
    print(f"{'||x|| bin':>16} " + "".join(f"eta={e:<7.2f}" for e in ETAS))
    for b in range(3):
        lo, hi = xq[b], xq[b + 1]
        sel = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
        cells = [np.mean(rows[sel & (rows[:, 1] == e), 3]) if np.any(sel & (rows[:, 1] == e)) else np.nan
                 for e in ETAS]
        print(f"[{lo:5.2f},{hi:5.2f}]   " + "".join(f"{c:>11.3f}" for c in cells))

    print(f"\nVALUE relative error |V_theta - V*|/|V*|   (compare: is the VALUE bad where the gradient is?):")
    print(f"{'||x|| bin':>16} " + "".join(f"eta={e:<7.2f}" for e in ETAS))
    for b in range(3):
        lo, hi = xq[b], xq[b + 1]
        sel = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
        cells = [np.mean(rows[sel & (rows[:, 1] == e), 5]) if np.any(sel & (rows[:, 1] == e)) else np.nan
                 for e in ETAS]
        print(f"[{lo:5.2f},{hi:5.2f}]   " + "".join(f"{c:>11.4f}" for c in cells))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_gradient_error_map"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, scales=PROBE_SCALES, etas=ETAS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
