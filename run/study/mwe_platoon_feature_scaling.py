"""Does CENTER-AND-REDUCE (standardising the features) fix the gradient blow-up?

The value gradient on the platoon is a ~24x magnitude blow-up with the value perfect and
the direction nearly right. Standardising features phi_k -> (phi_k - mu_k)/sigma_k is an
affine reparametrisation: it leaves the value fit (R^2) and the gradient of a FIXED
function unchanged, but it changes which MINIMUM-NORM solution lstsq selects (min ||theta||
in standardised coords = sigma^2-weighted min-norm in raw coords), so it CAN move the
gradient. This compares raw vs centered vs standardised features for the value-gradient
control, on the cheat-trained critic.

Usage:
    uv run python run/study/mwe_platoon_feature_scaling.py [--debug]
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
WIN_LEN = 5
N_IC = 16
AUG_PER = 4
EPS_TRAIN = 0.2
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
        x0 = np.zeros(n); x0[1::2] = 0.4 * rng.standard_normal(n // 2); ics.append(x0)
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


def closed_loop_cost(env, control_fn, x0):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        u = np.clip(control_fn(np.array(buf.buffer.to_array())), -CLIP, CLIP)
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
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T

    dep_W = rollout(env, gain, k, canonical_x0(n))
    u_star = np.array([-gain @ xi_of(W) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))
    I_nc, I_or = 0.0753, 0.0283

    rng = np.random.default_rng(SEED)
    onman = np.concatenate([rollout(env, gain, k, np.asarray(ic))
                            for ic in diverse_ics(n, N_IC, SEED)], 0)
    aug = [onman]
    for _ in range(AUG_PER):
        aug.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(aug, 0)
    Vtr = np.array([-(xi_of(W) @ P @ xi_of(W)) for W in Wtr])

    print(f"\nFEATURE SCALING for the value gradient | cheat eps={EPS_TRAIN} | min-norm lstsq")
    print(f"reference: no-control {I_nc} oracle {I_or} | mean||u*||={mag_star:.4f}")
    print(f"{'rep':>12}{'scaling':>14}{'R2':>7}{'cosL2':>8}{'|u|/|u*|':>10}{'I_cl':>11}  verdict")

    for kd, kw in REPS:
        rep = make_representation(kd, window_length=WIN_LEN, n_state=n, **kw)
        feat = jax.jit(rep.feature_fn); d = int(rep.feature_dim)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
        mu = Phi.mean(0); sd = Phi.std(0); sd = np.where(sd < 1e-12, 1.0, sd)
        mu_j, sd_j = jnp.asarray(mu), jnp.asarray(sd)

        for scaling in ("raw", "centered", "standardised"):
            if scaling == "raw":
                theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
                Vfn = lambda W, th=jnp.asarray(theta): jnp.dot(th, feat(W))
            elif scaling == "centered":
                theta = np.linalg.lstsq(Phi - mu, Vtr - Vtr.mean(), rcond=None)[0]
                Vfn = lambda W, th=jnp.asarray(theta): jnp.dot(th, feat(W) - mu_j)
            else:  # standardised (center AND reduce)
                theta = np.linalg.lstsq((Phi - mu) / sd, Vtr, rcond=None)[0]
                Vfn = lambda W, th=jnp.asarray(theta): jnp.dot(th, (feat(W) - mu_j) / sd_j)
            pred = np.array([float(Vfn(jnp.asarray(w))) for w in Wtr])
            r2 = 1 - np.sum((Vtr - pred) ** 2) / (np.sum((Vtr - Vtr.mean()) ** 2) + 1e-12)
            grad = jax.jit(jax.grad(lambda W: Vfn(W)))
            u_of = lambda W: half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1]
            u_rep = np.array([u_of(W) for W in dep_W])
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            ratio = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (mag_star + 1e-12))
            I_cl = closed_loop_cost(env, u_of, canonical_x0(n))
            ok = (I_cl < I_nc) and (cos > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{kd:>12}{scaling:>14}{r2:>7.3f}{cos:>8.3f}{ratio:>10.2f}{I_cl:>11.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_feature_scaling"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, eps=EPS_TRAIN), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
