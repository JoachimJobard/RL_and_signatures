"""REAL platoon (src/envs/platoon.py, N=5, dim 10, nonlinear, delayed): does
exploration recover the value-gradient signature control, and does the VALUE LABEL
decide it? Full actuation (all 5 vehicles, as the env is); sweep exploration sigma.

Pipeline (no TD loop -- the critic is solved once, so the only variables are
exploration and the label):
  - roll the delayed-LQR oracle + action noise sigma from several ICs -> off-manifold
    windows (exploration);
  - fit the IDEAL signature critic on those windows with one of two labels:
       oracle_label : V*(W) = -xi^T P_aug xi   (analytic delayed-LQR value; CLEAN)
       rtg_label    : discounted return-to-go of the noisy rollout (REALISTIC, noisy);
  - deploy the value-gradient control u = 1/2 R^-1 B^T dV/dx(t) (no noise) from the
    canonical IC and measure cos(u,u*) and closed-loop cost I.

Hypotheses carried over from the simple-platoon studies, tested here on the real env:
  (1) with the CLEAN label, exploration recovers the control (geometry is solvable);
  (2) with the NOISY return-to-go label, the high-dim critic overfits and the control
      degrades/diverges -- the bottleneck is value-label quality, not exploration.

Usage:
    uv run python run/study/mwe_real_platoon_exploration.py [--debug]
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
TAU = 10.0                       # discount for the return-to-go label
SIGMAS = [0.3]                   # fixed good exploration level (best cos from the sweep)
N_IC = 6
N_NOISE = 2
IC_SCALE = 0.4
REPS = [("markovian", dict(degree=2)), ("raw_history", dict(degree=2))]  # no signature
SEED = 0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R),
                                  float(env.max_delay), env.step_size)


def canonical_x0(n):
    x0 = np.zeros(n); x0[1] = 0.5
    return x0


def diverse_ics(n, n_ic, seed):
    rng = np.random.default_rng(seed)
    ics = [canonical_x0(n)]
    for _ in range(n_ic - 1):
        x0 = np.zeros(n); x0[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(x0)
    return ics


def rollout(env, gain, k, x0, win_len, sigma=0.0, rng=None):
    """Oracle (+ optional action noise) rollout. Returns rep windows (T,win_len,N) and
    per-step rewards (T,)."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]
    windows, rewards = [], []
    t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        if sigma > 0 and rng is not None:
            u = u + sigma * rng.standard_normal(u.shape)
        windows.append(np.array(buf.buffer.to_array()).reshape(win_len, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rewards.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(windows), np.array(rewards)


def xi_of(W, k):
    return W[::-1][:k + 1].reshape(-1)        # [x_t, x_{t-1}, ..., x_{t-k}] flattened


def labels_oracle(Ws, P, k):
    return np.array([-(xi_of(W, k) @ P @ xi_of(W, k)) for W in Ws])


def labels_rtg(rewards):
    g = np.exp(-REGIME["step_size"] / TAU); V = np.zeros(len(rewards)); acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] * REGIME["step_size"] + g * acc; V[i] = acc
    return V


def closed_loop_cost(env, feat_fn, theta, win_len, x0, half_RinvBT):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat_fn(p))))
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    CLIP = 3.0           # the agent clips actions; without it the nonlinear plant just diverges
    while t < TF - 1e-9:
        eg = np.array(grad(jnp.asarray(buf.buffer.to_array())))[-1]
        u = np.clip(half_RinvBT @ eg, -CLIP, CLIP)
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
    win_len = int(np.ceil(float(env.max_delay) / dt)) + 3 + 1
    x0c = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T

    nc_W, nc_r = rollout(env, gain * 0.0, k, x0c, win_len)     # no control
    dep_W, dep_r = rollout(env, gain, k, x0c, win_len)         # oracle deployment
    I_nc = float(np.sum(-nc_r) * dt); I_or = float(np.sum(-dep_r) * dt)
    u_star = np.array([-gain @ xi_of(W, k) for W in dep_W])

    ics = diverse_ics(n, N_IC, SEED)
    print(f"\nREAL platoon | N={REGIME['n_vehicles']} dim={n} ctrl={env.B.shape[1]} "
          f"win={win_len} | NO signature -- markovian vs raw_history")
    print(f"reference bars : no-control I={I_nc:.4f} | oracle I*={I_or:.4f}")
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))
    print(f"oracle control magnitude on deployment trajectory: mean||u*|| = {mag_star:.4f}")
    print(f"{'sigma':>6}{'rep':>13}{'dim':>6}{'label':>13}{'R2':>7}{'cosL2':>8}"
          f"{'mean|u|':>9}{'|u|/|u*|':>10}{'I_cl':>11}  verdict")

    for sg in SIGMAS:
        # exploration data is representation-independent: build once, reuse for every rep.
        Ws, RW = [], []
        for ic in ics:
            n_noise = N_NOISE if sg > 0 else 1
            for s in range(n_noise):
                rg = np.random.default_rng(1000 + s + 7 * int(sg * 100))
                ww, rr = rollout(env, gain, k, np.asarray(ic), win_len, sigma=sg, rng=rg)
                Ws.append(ww); RW.append(rr)
        Wall = np.concatenate(Ws, 0)
        Vo = np.concatenate([labels_oracle(ww, P, k) for ww in Ws])
        Vr = np.concatenate([labels_rtg(rr) for rr in RW])

        for kind, kw in REPS:
            rep = make_representation(kind, window_length=win_len, n_state=n, **kw)
            feat = jax.jit(rep.feature_fn)
            Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wall])
            d = Phi.shape[1]
            for lab, V in [("oracle_label", Vo)]:           # oracle label only
                theta = np.linalg.lstsq(Phi, V, rcond=None)[0]   # NO ridge: min-norm least squares
                r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
                th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
                u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
                cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
                mag_rep = float(np.mean(np.linalg.norm(u_rep, axis=1)))
                ratio = mag_rep / (mag_star + 1e-12)
                I_cl = closed_loop_cost(env, feat, theta, win_len, x0c, half_RinvBT)
                ok = (I_cl < I_nc) and (cos > 0.9)
                v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
                print(f"{sg:>6.2f}{kind:>13}{d:>6}{lab:>13}{r2:>7.3f}{cos:>8.3f}"
                      f"{mag_rep:>9.3f}{ratio:>10.2f}{I_cl:>11.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_real_platoon"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, sigmas=SIGMAS, reps=[r[0] for r in REPS],
                                                      I_nc=I_nc, I_oracle=I_or), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
