"""H2 (signature > raw-history) at MATCHED dimension, vs the continuous oracle.

H1 (history > markovian) held in mwe_continuous_h1h2, but raw-history beat the signature
there at UNMATCHED dimension (1325 vs 462). H2 is the fair test: at matched feature
dimension and matched window (same history), does the signature's structured feature set
beat the naive window? We compare, all on the win-5 history and the continuous-oracle
value:
  - signature depth-2            (d = 462, structured)
  - raw-history degree-2 full     (d = 1325, reference -- more features)
  - raw-history degree-2 -> PCA 462  (the STRONGEST matched-dim raw baseline: the best
    462-dim linear subspace of the raw polynomial features)
H2 holds iff signature-462 beats raw-history-PCA-462 (lower I / higher grad cos).

Usage:
    uv run python run/study/mwe_continuous_h2_matched.py [--debug]
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
MATCH_DIM = 462
N_IC = 12
AUG_PER = 2
EPS_TRAIN = 0.2
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


def closed_loop(env, Vfn_grad, half_RinvBT, x0):
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
        u = np.clip(half_RinvBT @ np.array(Vfn_grad(jnp.asarray(buf.buffer.to_array())))[-1], -CLIP, CLIP)
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
    u_cont = lambda ow: -Kc @ z_of_window(np.array(ow)[::-1], dt, theta, n)

    x0c = np.zeros(n); x0c[1] = 0.5
    dep_W, _ = rollout(env, u_cont, k, x0c)
    u_star = np.array([-Kc @ z_of_window(W, dt, theta, n) for W in dep_W])
    I_nc = rollout(env, lambda ow: np.zeros(Bm.shape[1]), k, x0c)[1]
    I_or = rollout(env, u_cont, k, x0c)[1]

    rng = np.random.default_rng(SEED)
    ics = [x0c]
    for _ in range(N_IC - 1):
        z = np.zeros(n); z[1::2] = 0.4 * rng.standard_normal(n // 2); ics.append(z)
    onman = np.concatenate([rollout(env, u_cont, k, ic)[0] for ic in ics], 0)
    aug = [onman]
    for _ in range(AUG_PER):
        aug.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(aug, 0)
    Vtr = np.array([-(z_of_window(W, dt, theta, n) @ Pc @ z_of_window(W, dt, theta, n)) for W in Wtr])

    sig = make_representation("signature", window_length=WIN_LEN, n_state=n, depth=2)
    raw = make_representation("raw_history", window_length=WIN_LEN, n_state=n, degree=2)
    fsig, fraw = jax.jit(sig.feature_fn), jax.jit(raw.feature_fn)

    def fit_eval(name, dim, Phi, Vfn_builder):
        theta_c = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
        r2 = 1 - np.sum((Vtr - Phi @ theta_c) ** 2) / (np.sum((Vtr - Vtr.mean()) ** 2) + 1e-12)
        Vgrad = Vfn_builder(theta_c)
        u_rep = np.array([half_RinvBT @ np.array(Vgrad(jnp.asarray(W)))[-1] for W in dep_W])
        cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
        I = closed_loop(env, Vgrad, half_RinvBT, x0c)
        v = "OK" if I < 1.5 * I_or else ("weak" if I < I_nc else "WORSE-than-nc")
        print(f"{name:>24}{dim:>6}{r2:>9.3f}{cos:>9.3f}{I:>9.4f}  {v}")

    print(f"\nH2 MATCHED-DIMENSION vs continuous oracle | win={WIN_LEN}")
    print(f"reference: no-control I={I_nc:.4f}  continuous oracle I*={I_or:.4f}")
    print(f"{'rep':>24}{'dim':>6}{'valR2':>9}{'gradcos':>9}{'I_cl':>9}  verdict")

    Psig = np.stack([np.asarray(fsig(jnp.asarray(w))) for w in Wtr])
    fit_eval("signature d2", Psig.shape[1], Psig,
             lambda th: jax.jit(jax.grad(lambda p: jnp.dot(jnp.asarray(th), fsig(p)))))

    Praw = np.stack([np.asarray(fraw(jnp.asarray(w))) for w in Wtr])
    fit_eval("raw_history d2 (full)", Praw.shape[1], Praw,
             lambda th: jax.jit(jax.grad(lambda p: jnp.dot(jnp.asarray(th), fraw(p)))))

    # raw-history -> top-MATCH_DIM PCA subspace (unsupervised; high-variance, not value-relevant)
    mu = Praw.mean(0); Pc_ = Praw - mu
    _, _, Vt = np.linalg.svd(Pc_, full_matrices=False)
    U = Vt[:MATCH_DIM].T; mu_j, U_j = jnp.asarray(mu), jnp.asarray(U)
    fit_eval(f"raw d2 PCA-{MATCH_DIM} (unsup.)", MATCH_DIM, Pc_ @ U,
             lambda th: jax.jit(jax.grad(lambda p: jnp.dot(jnp.asarray(th), U_j.T @ (fraw(p) - mu_j)))))

    # raw-history -> top-MATCH_DIM features by |corr with V| (SUPERVISED matched-dim baseline)
    corr = np.array([abs(np.corrcoef(Praw[:, j], Vtr)[0, 1]) if Praw[:, j].std() > 1e-9 else 0.0
                     for j in range(Praw.shape[1])])
    idx = np.argsort(-corr)[:MATCH_DIM]; idx_j = jnp.asarray(idx)
    fit_eval(f"raw d2 corr-{MATCH_DIM} (superv.)", MATCH_DIM, Praw[:, idx],
             lambda th: jax.jit(jax.grad(lambda p: jnp.dot(jnp.asarray(th), fraw(p)[idx_j]))))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_h2_matched"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, match_dim=MATCH_DIM, I_nc=I_nc, I_or=I_or), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
