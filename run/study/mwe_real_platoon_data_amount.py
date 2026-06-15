"""REAL platoon: does AUGMENTING the amount of exploration data fix the value-gradient
history control? (oracle label, min-norm least squares, no ridge.)

The exploration study left raw-history at cos~0.9 but |u|/|u*|~25x (a magnitude
blow-up) and failing. One suspicion: with d=4185 features and only ~3500 windows the
fit is UNDERDETERMINED (n<d), so the min-norm solution leaves a large gradient null
space. This sweeps the amount of off-manifold exploration data (number of ICs x noise
realisations) at fixed sigma, and asks whether driving n well past d -- more off-
manifold coverage -- brings the control magnitude down to the oracle and recovers it,
or whether it SATURATES (more data insufficient -> a structural fix is needed).

Window = 5 taps = the augmented state xi (so V*(xi)=-xi^T P xi is exactly degree-2 in
the window and raw-history deg-2 represents it; dim stays tractable so n>d is reachable).

Usage:
    uv run python run/study/mwe_real_platoon_data_amount.py [--debug]
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
SIGMA = 0.3
N_NOISE = 2
N_ICS = [4, 16, 48]              # data-amount axis
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


def rollout(env, gain, k, x0, sigma=0.0, rng=None):
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
        if sigma > 0 and rng is not None:
            u = u + sigma * rng.standard_normal(u.shape)
        windows.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(windows)


def xi_of(W, k):
    return W[::-1][:k + 1].reshape(-1)


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
    env = make_env(); n, dt = env.N, env.step_size
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    x0c = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T

    nc_W = rollout(env, gain * 0.0, k, x0c)
    dep_W = rollout(env, gain, k, x0c)
    # reference costs via a fresh oracle/no-control rollout cost using rewards is omitted;
    # reuse the canonical bars measured elsewhere:
    I_nc, I_or = 0.0753, 0.0283
    u_star = np.array([-gain @ xi_of(W, k) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))

    print(f"\nREAL platoon DATA-AMOUNT sweep | dim={n} win={WIN_LEN}(=xi) sigma={SIGMA} | "
          f"min-norm lstsq, oracle label")
    print(f"reference: no-control I={I_nc} oracle I*={I_or} | mean||u*||={mag_star:.4f}")
    print(f"{'rep':>12}{'dim':>6}{'n_ic':>6}{'n_pts':>8}{'n/d':>6}{'R2':>7}{'cosL2':>8}"
          f"{'|u|/|u*|':>10}{'I_cl':>11}  verdict")

    feats = {kind: jax.jit(make_representation(kind, window_length=WIN_LEN, n_state=n, **kw).feature_fn)
             for kind, kw in REPS}
    dims = {kind: int(make_representation(kind, window_length=WIN_LEN, n_state=n, **kw).feature_dim)
            for kind, kw in REPS}

    for n_ic in N_ICS:
        ics = diverse_ics(n, n_ic, SEED)
        Ws = []
        for ic in ics:
            for s in range(N_NOISE):
                rg = np.random.default_rng(1000 + s + 7 * int(SIGMA * 100) + 999 * n_ic)
                Ws.append(rollout(env, gain, k, np.asarray(ic), sigma=SIGMA, rng=rg))
        Wall = np.concatenate(Ws, 0)
        V = np.array([-(xi_of(W, k) @ P @ xi_of(W, k)) for W in Wall])
        for kind, _ in REPS:
            feat = feats[kind]; d = dims[kind]
            Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wall])
            theta = np.linalg.lstsq(Phi, V, rcond=None)[0]
            r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
            th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
            u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            ratio = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (mag_star + 1e-12))
            I_cl = closed_loop_cost(env, feat, theta, x0c, half_RinvBT)
            ok = (I_cl < I_nc) and (cos > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{kind:>12}{d:>6}{n_ic:>6}{len(Phi):>8}{len(Phi)/d:>6.1f}{r2:>7.3f}"
                  f"{cos:>8.3f}{ratio:>10.2f}{I_cl:>11.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_data_amount"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, sigma=SIGMA, n_ics=N_ICS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
