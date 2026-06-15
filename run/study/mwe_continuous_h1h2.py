"""H1/H2 against the CONTINUOUS-time delayed-LQR oracle -- convention-clean.

Uses the Chebyshev-collocation continuous oracle (mwe_continuous_delayed_oracle): value
V(z) = -z^T P_c z, control u* = -K_c z, for which Doya's continuous formula
u = 1/2 R^-1 B^T dV/dx is EXACT (no dt). We fit each representation's critic to the
continuous value (window history interpolated to the collocation points), extract the
control with Doya's continuous formula (NO rescaling -- magnitudes are correct here), and
report the two convention-free axes:
  value R^2   = representation quality (H1: history > markovian; H2: signature vs raw-history)
  gradient cos(u_rep, u*) and closed-loop I = extraction quality.

Usage:
    uv run python run/study/mwe_continuous_h1h2.py [--debug]
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
N_IC = 16
AUG_PER = 4
EPS_TRAIN = 0.2
REPS = [("markovian", dict(degree=2)), ("raw_history", dict(degree=2)), ("signature", dict(depth=2))]
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
    Kc = np.linalg.inv(R) @ Nmat.T @ Pc
    return Pc, Kc, theta, n


def z_of_window(W, dt, theta, n):
    L = len(W); t_grid = -dt * np.arange(L)[::-1]      # chronological [-(L-1)dt..0]
    z = np.stack([np.interp(theta, t_grid, W[:, c]) for c in range(n)], axis=1)
    return z.reshape(-1)


def rollout(env, control_owin, k, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, rs = [], []; t = 0.0
    while t < TF - 1e-9:
        u = np.clip(control_owin(o_win), -CLIP, CLIP)
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), float(np.sum(-np.array(rs)) * env.step_size)


def closed_loop_rep(env, feat, theta_c, half_RinvBT, k, x0):
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

    def u_cont(o_win):                       # continuous oracle control from a control-cadence history
        W = np.array(o_win)[::-1]            # chronological current-last
        return -Kc @ z_of_window(W, dt, theta, n)

    x0c = np.zeros(n); x0c[1] = 0.5
    dep_W, _ = rollout(env, u_cont, k, x0c)
    u_star = np.array([-Kc @ z_of_window(W, dt, theta, n) for W in dep_W])
    I_nc = rollout(env, lambda ow: np.zeros(Bm.shape[1]), k, x0c)[1]
    I_or = rollout(env, u_cont, k, x0c)[1]

    # training windows (on-manifold continuous-oracle rollouts + off-manifold cheat)
    rng = np.random.default_rng(SEED)
    ics = [x0c] + [np.zeros(n) for _ in range(N_IC - 1)]
    for j in range(1, N_IC):
        ics[j][1::2] = 0.4 * rng.standard_normal(n // 2)
    onman = np.concatenate([rollout(env, u_cont, k, ic)[0] for ic in ics], 0)
    aug = [onman]
    for _ in range(AUG_PER):
        aug.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(aug, 0)
    Vtr = np.array([-(z_of_window(W, dt, theta, n) @ Pc @ z_of_window(W, dt, theta, n)) for W in Wtr])

    print(f"\nH1/H2 vs CONTINUOUS oracle (Chebyshev N={N_CHEB}) | win={WIN_LEN} | Doya formula, NO rescale")
    print(f"reference: no-control I={I_nc:.4f}  continuous oracle I*={I_or:.4f}")
    print(f"{'rep':>12}{'dim':>6}{'value R2':>10}{'grad cos':>10}{'I_cl':>10}  verdict")
    for kd, kw in REPS:
        rep = make_representation(kd, window_length=WIN_LEN, n_state=n, **kw)
        feat = jax.jit(rep.feature_fn)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr]); d = Phi.shape[1]
        theta_c = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
        r2 = 1 - np.sum((Vtr - Phi @ theta_c) ** 2) / (np.sum((Vtr - Vtr.mean()) ** 2) + 1e-12)
        th = jnp.asarray(theta_c); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
        u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
        cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
        I_cl = closed_loop_rep(env, feat, theta_c, half_RinvBT, k, x0c)
        v = "OK" if I_cl < 1.5 * I_or else ("weak" if I_cl < I_nc else "WORSE-than-nc")
        print(f"{kd:>12}{d:>6}{r2:>10.3f}{cos:>10.3f}{I_cl:>10.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_continuous_h1h2"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, N_cheb=N_CHEB, I_nc=I_nc, I_or=I_or), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
