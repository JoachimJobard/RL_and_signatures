"""REAL platoon: CHEAT -- inject off-manifold data directly and see if it solves the
value-gradient history control (oracle label, min-norm least squares).

Exploration and data amount both saturated: the control direction reaches cos~0.93 but
the magnitude stays ~20-40x the oracle and the closed loop fails, because on-policy /
exploration data lives in an O(sigma)-thin tube around the state-manifold and never
constrains the gradient NORMAL to it. This script bypasses that: for each on-manifold
window it adds copies with the CURRENT STATE perturbed by eps*noise and the HISTORY
FROZEN -- windows off the manifold, in exactly the direction the control reads -- each
labeled with the analytic value V*(xi') = -xi'^T P_aug xi' of the perturbed augmented
state. This is not realizable on-line (you cannot freeze the past while moving the
present in the real env), hence "cheat"; it is the upper bound that tells us whether the
off-manifold gradient information is the whole missing ingredient.

Prediction (from the E0 history-fix): any eps>0 should drive cos->1, |u|/|u*|->1, and
I->oracle, with conditioning unchanged -- confirming the issue is purely the missing
off-manifold gradient, not data amount, exploration, actuation, or conditioning.

Usage:
    uv run python run/study/mwe_real_platoon_offmanifold_cheat.py [--debug]
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
TF = 15.0
N_IC = 16
AUG_PER = 4
EPS = [0.0, 0.05, 0.2, 0.5]
IC_SCALE = 0.4
WIN_LEN = 5                      # = k_taps+1: window is exactly the augmented state xi
REPS = [("signature", dict(depth=2)), ("raw_history", dict(degree=2))]
SEED = 0
CLIP = 3.0


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


def diverse_ics(n, n_ic, seed):
    rng = np.random.default_rng(seed); ics = [canonical_x0(n)]
    for _ in range(n_ic - 1):
        x0 = np.zeros(n); x0[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(x0)
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
    o_win = [x0.copy() for _ in range(k + 1)]; windows = []; t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        windows.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(windows)


def xi_of(W, k):
    return W[::-1][:k + 1].reshape(-1)


def label(W, P, k):
    xi = xi_of(W, k); return -(xi @ P @ xi)


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def closed_loop_cost(env, feat, theta, x0, half_RinvBT):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        u = np.clip(half_RinvBT @ np.array(grad(jnp.asarray(buf.buffer.to_array())))[-1], -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); I += -float(r) * env.step_size
        buf.append(np.array(x).reshape(-1).astype(np.float32))
    return I


def main():
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); n = env.N
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    x0c = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T

    dep_W = rollout(env, gain, k, x0c)
    u_star = np.array([-gain @ xi_of(W, k) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))
    I_nc, I_or = 0.0753, 0.0283

    onman = np.concatenate([rollout(env, gain, k, np.asarray(ic))
                            for ic in diverse_ics(n, N_IC, SEED)], 0)
    rng = np.random.default_rng(SEED)

    feats = {kd: jax.jit(make_representation(kd, window_length=WIN_LEN, n_state=n, **kw).feature_fn)
             for kd, kw in REPS}
    dims = {kd: int(make_representation(kd, window_length=WIN_LEN, n_state=n, **kw).feature_dim)
            for kd, kw in REPS}

    print(f"\nREAL platoon OFF-MANIFOLD CHEAT | dim={n} win={WIN_LEN}(=xi) | "
          f"min-norm lstsq, oracle label | aug/window={AUG_PER}")
    print(f"reference: no-control I={I_nc} oracle I*={I_or} | mean||u*||={mag_star:.4f}")
    print(f"{'rep':>12}{'mode':>14}{'eps':>6}{'erank':>7}{'R2':>7}{'cosL2':>8}"
          f"{'|u|/|u*|':>10}{'I_cl':>11}  verdict")

    for kd, _ in REPS:
      feat = feats[kd]; d = dims[kd]
      for mode in ("current_only", "full_window"):
        for eps in EPS:
            W = onman.copy()
            if eps > 0:
                aug = [onman]
                for _ in range(AUG_PER):
                    Wp = onman.copy()
                    if mode == "current_only":          # perturb present, freeze past
                        Wp[:, -1] = Wp[:, -1] + eps * rng.standard_normal(onman[:, -1].shape)
                    else:                                # perturb the ENTIRE window (full xi-space)
                        Wp = onman + eps * rng.standard_normal(onman.shape)
                    aug.append(Wp)
                W = np.concatenate(aug, 0)
            V = np.array([label(w, P, k) for w in W])
            Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in W])
            erank = conditioning(Phi)
            theta = np.linalg.lstsq(Phi, V, rcond=None)[0]
            r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
            th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
            u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(Wd)))[-1] for Wd in dep_W])
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            ratio = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (mag_star + 1e-12))
            I_cl = closed_loop_cost(env, feat, theta, x0c, half_RinvBT)
            ok = (I_cl < I_nc) and (cos > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{kd:>12}{mode:>14}{eps:>6.2f}{erank:>7.2f}{r2:>7.3f}"
                  f"{cos:>8.3f}{ratio:>10.2f}{I_cl:>11.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_offmanifold_cheat"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, eps=EPS, aug_per=AUG_PER), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
