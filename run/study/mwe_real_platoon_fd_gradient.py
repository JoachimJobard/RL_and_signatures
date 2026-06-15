"""Case B: does the value-gradient control work with the TRUE (finite-difference)
gradient, bypassing the critic entirely? (real platoon)

Every failure so far used a FITTED critic and differentiated it. This removes the
critic: at each deployment state we estimate dV/dx(t) by CENTRAL-DIFFERENCING the exact
deterministic nonlinear return-to-go (perturb the current state +/- h with the history
frozen, roll the oracle forward, difference), then form the value-gradient control
u = 1/2 R^-1 B^T dV/dx and compare to the oracle u* = -K xi.

Decision:
  - if u_FD matches the oracle (cos ~ 1, |u|/|u*| ~ 1)  => the ~24x blow-up was the
    CRITIC's fit/extrapolation; the control law + true gradient is fine (vindicates a
    model-based / Sobolev gradient, or finite-difference-in-the-loop);
  - if u_FD is still blown up  => even the true gradient gives bad control (a deeper
    problem with the value-gradient law on the nonlinear plant).

A linearised finite-difference of -xi^T P xi is included as a sanity check that the
difference machinery and direction handling are correct.

Usage:
    uv run python run/study/mwe_real_platoon_fd_gradient.py [--debug]
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
H_LABEL = 120
H_FD = 1e-2
DEP_SUB = 8            # subsample deployment states (each costs 2n forward rollouts)
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
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, rs, snaps = [], [], []; t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        snaps.append((np.array(w.state.buffer.data), int(w.state.buffer.ptr), np.array(w.state.x)))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), np.array(rs), snaps


def state_from_snap(env, snap):
    import jax.numpy as jnp
    from src.utils.solver_buffer_jax import BufferState
    from src.envs.env_rk_jax import EnvState
    data, ptr, x = snap
    buf = BufferState(data=jnp.asarray(data), ptr=int(ptr), capacity=env.buffer_size, dim=env.N)
    return EnvState(x=jnp.asarray(x), t=0.0, buffer=buf, last_u=jnp.zeros(env.B.shape[1]))


def perturb(snap, W, delta):
    data, ptr, x = snap
    data = data.copy(); j = (ptr - 1) % data.shape[0]; data[j] = data[j] + delta
    W = W.copy(); W[-1] = W[-1] + delta
    return (data, ptr, x + delta), W


def nonlinear_return(env, gain, k, snap, W, H, Q, R, jit_step):
    import jax.numpy as jnp
    state = state_from_snap(env, snap)
    o_win = [W[-1 - i].copy() for i in range(k + 1)]; rtg = 0.0
    for _ in range(H):
        u = -gain @ np.concatenate(o_win[:k + 1])
        state, xn, r = jit_step(state, jnp.asarray(u))
        rtg += float(r); o_win = [np.asarray(xn).reshape(-1)] + o_win[:-1]
    return rtg


def xi_of(W):
    return W[::-1].reshape(-1)


def fd_grad_nonlinear(env, gain, k, snap, W, Q, R, jit_step, n):
    g = np.zeros(n)
    for j in range(n):
        e = np.zeros(n); e[j] = H_FD
        sp, Wp = perturb(snap, W, e); sm, Wm = perturb(snap, W, -e)
        Vp = nonlinear_return(env, gain, k, sp, Wp, H_LABEL, Q, R, jit_step)
        Vm = nonlinear_return(env, gain, k, sm, Wm, H_LABEL, Q, R, jit_step)
        g[j] = (Vp - Vm) / (2 * H_FD)
    return g


def fd_grad_linear(W, P, k, n):
    g = np.zeros(n)
    for j in range(n):
        e = np.zeros(n); e[j] = H_FD
        Wp = W.copy(); Wp[-1] = Wp[-1] + e; Wm = W.copy(); Wm[-1] = Wm[-1] - e
        Vp = -(xi_of(Wp) @ P @ xi_of(Wp)); Vm = -(xi_of(Wm) @ P @ xi_of(Wm))
        g[j] = (Vp - Vm) / (2 * H_FD)
    return g


def main():
    import jax
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); n = env.N; Q, R = np.array(env.Q), np.array(env.R)
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    half_RinvBT = 0.5 * np.linalg.inv(R) @ np.array(env.B).T
    jit_step = jax.jit(env.step)

    dep_W, dep_r, dep_snaps = rollout(env, gain, k, canonical_x0(n))
    idx = list(range(0, len(dep_W), DEP_SUB))
    u_star = np.array([-gain @ xi_of(dep_W[i]) for i in idx])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))

    print(f"\nCase B: TRUE finite-difference value-gradient (no critic) | real platoon "
          f"| h={H_FD} H={H_LABEL} | {len(idx)} states")
    print(f"mean||u*|| = {mag_star:.4f}")

    u_lin = np.array([half_RinvBT @ fd_grad_linear(dep_W[i], P, k, n) for i in idx])
    u_nl = np.array([half_RinvBT @ fd_grad_nonlinear(env, gain, k, dep_snaps[i], dep_W[i], Q, R, jit_step, n)
                     for i in idx])

    for name, U in [("linearised FD (sanity)", u_lin), ("NONLINEAR FD (the test)", u_nl)]:
        cos = float(np.sum(U * u_star) / (np.linalg.norm(U) * np.linalg.norm(u_star) + 1e-12))
        mag = float(np.mean(np.linalg.norm(U, axis=1)) / (mag_star + 1e-12))
        print(f"  {name:<26} cos(u, u*)={cos:6.3f}   |u|/|u*|={mag:7.2f}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_fd_gradient"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, h_fd=H_FD, H=H_LABEL), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
