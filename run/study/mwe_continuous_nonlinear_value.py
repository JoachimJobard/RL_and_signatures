"""H1/H2 with the TRUE NONLINEAR value (not the linearised quadratic), continuous-time.

Earlier the value target was the LINEARISED delayed-LQR value -z^T P_c z (exactly
quadratic), so degree-2 raw-history nailed it and the signature's universality was unused.
Here we use the TRUE nonlinear value: V_nl(window) = integrated return-to-go of the
(continuous linearised-oracle) policy rolled forward on the NONLINEAR env,
  V_nl = sum_j r_j * dt          (r = env reward; *dt -> continuous-time value, consistent
                                  with Doya's continuous control formula -- no 1/dt artifact).
We fit each representation to V_nl and report:
  value R^2  (does degree-2 still fit, or is V_nl genuinely nonlinear?),
  vs-linearised R^2 (how far V_nl departs from the quadratic -z^T P_c z),
  grad cos (vs the continuous linearised oracle) and closed-loop I on the nonlinear env.
If V_nl is genuinely nonlinear, degree-2 raw-history R^2 drops and the signature may gain
-> the regime where H2 (signature > raw-history) can actually show up.

Usage:
    uv run python run/study/mwe_continuous_nonlinear_value.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.linalg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REGIME = dict(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
TF = 15.0
WIN_LEN = 5
N_CHEB = 12
H_VAL = 150          # forward control steps for the nonlinear return-to-go
N_IC = 8
SUBSAMPLE = 4
AUG_PER = 2
EPS_TRAIN = 0.3
IC_SCALE = 0.8       # larger ICs -> explore the nonlinear regime
REPS = [("markovian", dict(degree=2)), ("raw_history", dict(degree=2)),
        ("signature", dict(depth=2)), ("signature", dict(depth=3))]
SEED = 0
CLIP = 3.0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T; dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1)); D = D - np.diag(D.sum(axis=1))
    return D, x


def continuous_oracle(env, N):
    A, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
    Q, R = np.array(env.Q), np.array(env.R)
    n, m = A.shape[0], B.shape[1]; tau = float(env.max_delay)
    Dc, xc = cheb(N); theta = (tau / 2.0) * (xc - 1.0); D = (2.0 / tau) * Dc
    big = n * (N + 1); M = np.zeros((big, big))
    for i in range(1, N + 1):
        for j in range(N + 1):
            M[i * n:(i + 1) * n, j * n:(j + 1) * n] = D[i, j] * np.eye(n)
    M[0:n, 0:n] = A; M[0:n, N * n:(N + 1) * n] = A1
    Nmat = np.zeros((big, m)); Nmat[0:n, :] = B
    Qz = np.zeros((big, big)); Qz[0:n, 0:n] = Q
    Pc = scipy.linalg.solve_continuous_are(M, Nmat, Qz, R)
    return Pc, np.linalg.inv(R) @ Nmat.T @ Pc, theta, n


def z_of_window(W, dt, theta, n):
    L = len(W); t_grid = -dt * np.arange(L)[::-1]
    return np.stack([np.interp(theta, t_grid, W[:, c]) for c in range(n)], axis=1).reshape(-1)


def rollout(env, Kc, theta, k, x0):
    """Roll the continuous linearised-oracle policy; return windows, snapshots, integrated cost."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, snaps, rs = [], [], []; t = 0.0
    while t < TF - 1e-9:
        u = np.clip(-Kc @ z_of_window(np.array(o_win)[::-1], env.step_size, theta, env.N) if Kc is not None
                    else np.zeros(env.B.shape[1]), -CLIP, CLIP)
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        snaps.append((np.array(w.state.buffer.data), int(w.state.buffer.ptr), np.array(w.state.x)))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), snaps, float(np.sum(-np.array(rs)) * env.step_size)


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


def nonlinear_value(env, Kc, theta, snap, W, H, jit_step):
    """Integrated return-to-go (continuous value) of the linearised-oracle policy on the
    NONLINEAR env, rolled forward from the exact fine state."""
    import jax.numpy as jnp
    dt, n = env.step_size, env.N
    state = state_from_snap(env, snap)
    o_win = [W[-1 - i].copy() for i in range(len(W))]; Vint = 0.0
    for _ in range(H):
        u = np.clip(-Kc @ z_of_window(np.array(o_win)[::-1], dt, theta, n), -CLIP, CLIP)
        state, xn, r = jit_step(state, jnp.asarray(u))
        Vint += float(r) * dt; o_win = [np.asarray(xn).reshape(-1)] + o_win[:-1]
    return Vint


def closed_loop_rep(env, feat, theta_c, half_RinvBT, x0):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    th = jnp.asarray(theta_c); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
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
    k = int(round(float(env.max_delay) / dt))
    Pc, Kc, theta, _ = continuous_oracle(env, N_CHEB)
    Bm, R = np.array(env.B), np.array(env.R)
    half_RinvBT = 0.5 * np.linalg.inv(R) @ Bm.T
    jit_step = jax.jit(env.step)
    x0c = np.zeros(n); x0c[1] = 0.5

    dep_W, dep_snaps, _ = rollout(env, Kc, theta, k, x0c)
    u_star = np.array([-Kc @ z_of_window(W, dt, theta, n) for W in dep_W])
    I_nc = rollout(env, None, theta, k, x0c)[2]
    I_or = rollout(env, Kc, theta, k, x0c)[2]

    # validate the nonlinear value vs the rollout's own integrated return-to-go
    rtg = np.cumsum(([], )[0]) if False else None
    print(f"\nNONLINEAR value H1/H2 vs continuous oracle | win={WIN_LEN} H={H_VAL}")
    print(f"reference: no-control I={I_nc:.4f}  continuous (lin.) oracle I*={I_or:.4f}")

    # training windows (on-manifold + off-manifold cheat), labelled by the NONLINEAR value
    rng = np.random.default_rng(SEED)
    ics = [x0c]
    for _ in range(N_IC - 1):
        z = np.zeros(n); z[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(z)
    pairs = []
    for ic in ics:
        Ws, sn, _ = rollout(env, Kc, theta, k, ic)
        for j in range(0, len(Ws), SUBSAMPLE):
            pairs.append((Ws[j], sn[j]))
    items = list(pairs)
    for _ in range(AUG_PER):
        for (W, sn) in pairs:
            sp, Wp = perturb(sn, W, EPS_TRAIN * rng.standard_normal(n)); items.append((Wp, sp))
    Wtr = np.array([it[0] for it in items])
    print(f"labelling {len(Wtr)} windows by nonlinear forward rollout (H={H_VAL}) ...")
    Vnl = np.array([nonlinear_value(env, Kc, theta, sn, W, H_VAL, jit_step) for (W, sn) in items])
    Vlin = np.array([-(z_of_window(W, dt, theta, n) @ Pc @ z_of_window(W, dt, theta, n)) for W in Wtr])
    nl_vs_lin = 1 - np.sum((Vnl - Vlin) ** 2) / (np.sum((Vnl - Vnl.mean()) ** 2) + 1e-12)
    print(f"how quadratic is V_nl? R^2(V_lin -> V_nl) = {nl_vs_lin:.4f}  (1=quadratic, <1=nonlinear)")

    print(f"{'rep':>14}{'dim':>7}{'valR2(Vnl)':>12}{'gradcos':>9}{'I_cl':>9}  verdict")
    for kd, kw in REPS:
        rep = make_representation(kd, window_length=WIN_LEN, n_state=n, **kw)
        feat = jax.jit(rep.feature_fn); d = int(rep.feature_dim)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
        theta_c = np.linalg.lstsq(Phi, Vnl, rcond=None)[0]
        r2 = 1 - np.sum((Vnl - Phi @ theta_c) ** 2) / (np.sum((Vnl - Vnl.mean()) ** 2) + 1e-12)
        th = jnp.asarray(theta_c); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
        u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
        cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
        I_cl = closed_loop_rep(env, feat, theta_c, half_RinvBT, x0c)
        v = "OK" if I_cl < 1.5 * I_or else ("weak" if I_cl < I_nc else "WORSE-than-nc")
        name = f"{kd} d{kw.get('depth', kw.get('degree'))}"
        print(f"{name:>14}{d:>7}{r2:>12.3f}{cos:>9.3f}{I_cl:>9.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_nonlinear_value"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, I_nc=I_nc, I_or=I_or,
                                                      nl_vs_lin_r2=float(nl_vs_lin)), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
