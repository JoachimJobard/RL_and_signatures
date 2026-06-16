"""Does regularising the ill-posed value-gradient EXTRACTION recover control on the
ROUGH (true nonlinear) target?

Diagnosis (mwe_target_difference_analysis): the raw_history deg-2 fit to the nonlinear
value V_nl is an ill-conditioned inverse, cond(Phi)~1e17. The fit theta = sum_i
(u_i^T y / sigma_i) v_i amplifies the rough target's energy in the small-sigma modes by
1/sigma_i, which flips the gradient (cos 0.997 on V_lin -> -0.226 on V_nl).

This script applies the two SVD-prescribed regularisers to the SAME rough V_nl fit and
sweeps their strength:
  - truncated SVD (rank k): keep modes i<k, coeff = (u_i^T y)/sigma_i, else 0.
  - ridge / Tikhonov (lambda): coeff = sigma_i (u_i^T y)/(sigma_i^2 + lambda), the
    bounded filter sigma_i/(sigma_i^2+lambda) replacing the unbounded 1/sigma_i.
For each strength we report value R^2 (on V_nl), gradient cos vs the continuous oracle,
and closed-loop cost I. Prediction: an intermediate k / lambda restores cos -> ~1 and
I -> oracle -- the sweet spot where the value content is kept but the roughness floor
(small-sigma modes) is dropped. The full fit (k=d, lambda=0) is the broken baseline; the
smooth-target V_lin full fit is the "works" reference.

Usage:
    uv run python run/study/mwe_truncation_ridge_sweep.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.linalg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REGIME = dict(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
TF = 15.0
WIN_LEN = 5
N_CHEB = 12
H_VAL = 120          # Monte-Carlo horizon (steps) for the nonlinear value target
N_IC = 8
SUBSAMPLE = 3
AUG_PER = 3
EPS_TRAIN = 0.3
IC_SCALE = 0.8
SEED = 0
CLIP = 3.0

# regularisation sweeps
TRUNC_RANKS = [2, 5, 10, 20, 40, 80, 160, 320, 640, 1000]   # full handled separately
RIDGE_LAMBDAS = np.logspace(-12.0, 2.0, 15)


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
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, snaps = [], []; t = 0.0
    while t < TF - 1e-9:
        u = np.clip(-Kc @ z_of_window(np.array(o_win)[::-1], env.step_size, theta, env.N), -CLIP, CLIP)
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        snaps.append((np.array(w.state.buffer.data), int(w.state.buffer.ptr), np.array(w.state.x)))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), snaps


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
    import jax.numpy as jnp
    dt, n = env.step_size, env.N
    state = state_from_snap(env, snap)
    o_win = [W[-1 - i].copy() for i in range(len(W))]; Vint = 0.0
    for _ in range(H):
        u = np.clip(-Kc @ z_of_window(np.array(o_win)[::-1], dt, theta, n), -CLIP, CLIP)
        state, xn, r = jit_step(state, jnp.asarray(u))
        Vint += float(r) * dt; o_win = [np.asarray(xn).reshape(-1)] + o_win[:-1]
    return Vint


def closed_loop(env, Vgrad, half_RinvBT, k, x0):
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
        u = np.clip(half_RinvBT @ np.array(Vgrad(jnp.asarray(buf.buffer.to_array())))[-1], -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); I += -float(r) * env.step_size
        buf.append(np.array(x).reshape(-1).astype(np.float32))
    return I


def main():
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-cheat", action="store_true",
                    help="train on ON-MANIFOLD oracle windows only (drop the off-manifold perturbations)")
    ap.add_argument("--n-ic", type=int, default=N_IC, help="number of initial conditions (rollouts)")
    ap.add_argument("--subsample", type=int, default=SUBSAMPLE, help="keep every k-th window of each rollout")
    ap.add_argument("--rep", choices=["raw_history", "signature"], default="raw_history",
                    help="feature representation used for the critic fit")
    ap.add_argument("--order", type=int, default=2,
                    help="polynomial degree (raw_history) or signature depth")
    args = ap.parse_args()
    aug_per = 0 if args.no_cheat else AUG_PER
    n_ic, subsample = args.n_ic, args.subsample
    rep_kw = dict(degree=args.order) if args.rep == "raw_history" else dict(depth=args.order)
    cheat_tag = "nocheat" if args.no_cheat else "cheat"
    tag = f"{args.rep}_o{args.order}_{cheat_tag}"
    env = make_env(); n, dt = env.N, env.step_size
    k = int(round(float(env.max_delay) / dt))
    Pc, Kc, theta, _ = continuous_oracle(env, N_CHEB)
    Bm, R = np.array(env.B), np.array(env.R)
    half_RinvBT = 0.5 * np.linalg.inv(R) @ Bm.T
    jit_step = jax.jit(env.step)
    x0c = np.zeros(n); x0c[1] = 0.5
    Vlin_of = lambda W: -(z_of_window(W, dt, theta, n) @ Pc @ z_of_window(W, dt, theta, n))

    # deployment trajectory + oracle control (the cos reference)
    dep_W, _ = rollout(env, Kc, theta, k, x0c)
    u_star = np.array([-Kc @ z_of_window(W, dt, theta, n) for W in dep_W])
    I_nc = closed_loop(env, lambda W: np.zeros((WIN_LEN, n)), half_RinvBT, k, x0c)
    # oracle closed loop via z_of_window control (recompute as reference bar)
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import RepresentationBuffer
    def oracle_cl():
        rep = make_representation("markovian", window_length=WIN_LEN, n_state=n, degree=1)
        buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=n)
        w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0c), t0=0.0)
        buf.reset()
        for _ in range(WIN_LEN):
            buf.append(x0c.astype(np.float32))
        I, t = 0.0, 0.0
        while t < TF - 1e-9:
            W = np.array(buf.buffer.to_array()).reshape(WIN_LEN, n)
            u = np.clip(-Kc @ z_of_window(W, dt, theta, n), -CLIP, CLIP)
            t, x, r = w.step(w.state, jnp.array(u)); I += -float(r) * dt
            buf.append(np.array(x).reshape(-1).astype(np.float32))
        return I
    I_or = oracle_cl()

    # training set: on-manifold (oracle rollouts, sub-sampled) + off-manifold "cheat" perturbations
    rng = np.random.default_rng(SEED)
    ics = [x0c]
    for _ in range(n_ic - 1):
        z = np.zeros(n); z[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(z)
    pairs = []
    for ic in ics:
        Ws, sn = rollout(env, Kc, theta, k, ic)
        for j in range(0, len(Ws), subsample):
            pairs.append((Ws[j], sn[j]))
    items = list(pairs)
    for _ in range(aug_per):
        for (W, s) in pairs:
            sp, Wp = perturb(s, W, EPS_TRAIN * rng.standard_normal(n)); items.append((Wp, sp))
    Wtr = np.array([it[0] for it in items])
    Vlin = np.array([Vlin_of(W) for W in Wtr])
    Vnl = np.array([nonlinear_value(env, Kc, theta, s, W, H_VAL, jit_step) for (W, s) in items])

    rep = make_representation(args.rep, window_length=WIN_LEN, n_state=n, **rep_kw)
    feat = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)        # Phi = U diag(S) Vt
    d = len(S)
    c_nl = U.T @ Vnl                                          # nonlinear-target energy per mode
    c_lin = U.T @ Vlin

    def eval_theta(theta_c, y):
        """value R^2 (vs y), gradient cos vs oracle, closed-loop I, for coeff vector theta_c."""
        pred = Phi @ theta_c
        r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-12)
        th = jnp.asarray(theta_c)
        Vgrad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
        u_rep = np.array([half_RinvBT @ np.array(Vgrad(jnp.asarray(W)))[-1] for W in dep_W])
        cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
        I = closed_loop(env, Vgrad, half_RinvBT, k, x0c)
        return r2, cos, I

    def theta_trunc(c, kk):
        coeff = np.zeros(d); coeff[:kk] = c[:kk] / S[:kk]
        return Vt.T @ coeff

    def theta_ridge(c, lam):
        return Vt.T @ (S * c / (S ** 2 + lam))

    print(f"\n=== TRUNCATION / RIDGE SWEEP on the rough nonlinear target ({args.rep} o{args.order}) ===")
    print(f"n_train={len(Wtr)}  dim={d}  n/d={len(Wtr)/d:.1f}  cond(Phi)={S[0]/S[-1]:.2e}")
    print(f"reference: no-control I={I_nc:.4f}   continuous oracle I*={I_or:.4f}")

    # broken baseline: full fit to V_nl ; works reference: full fit to V_lin (smooth)
    r2_f, cos_f, I_f = eval_theta(Vt.T @ (c_nl / S), Vnl)
    r2_l, cos_l, I_l = eval_theta(Vt.T @ (c_lin / S), Vlin)
    print(f"\n{'fit':>28}{'valR2':>9}{'gradcos':>9}{'I_cl':>10}")
    print(f"{'V_lin full (smooth, ref)':>28}{r2_l:>9.3f}{cos_l:>9.3f}{I_l:>10.4f}")
    print(f"{'V_nl full (rough, broken)':>28}{r2_f:>9.3f}{cos_f:>9.3f}{I_f:>10.4f}")

    print(f"\n--- truncated SVD (rank k), target = V_nl ---")
    print(f"{'k':>8}{'valR2':>9}{'gradcos':>9}{'I_cl':>10}")
    trunc = []
    ranks = [r for r in TRUNC_RANKS if r < d] + [d]
    for kk in ranks:
        r2, cos, I = eval_theta(theta_trunc(c_nl, kk), Vnl)
        trunc.append((kk, r2, cos, I)); print(f"{kk:>8}{r2:>9.3f}{cos:>9.3f}{I:>10.4f}")

    print(f"\n--- ridge (lambda), target = V_nl ---")
    print(f"{'lambda':>10}{'valR2':>9}{'gradcos':>9}{'I_cl':>10}")
    ridge = []
    for lam in RIDGE_LAMBDAS:
        r2, cos, I = eval_theta(theta_ridge(c_nl, lam), Vnl)
        ridge.append((lam, r2, cos, I)); print(f"{lam:>10.2e}{r2:>9.3f}{cos:>9.3f}{I:>10.4f}")

    # sweet spots
    tk = np.array([t[0] for t in trunc]); tcos = np.array([t[2] for t in trunc]); tI = np.array([t[3] for t in trunc])
    rl = np.array([r[0] for r in ridge]); rcos = np.array([r[2] for r in ridge]); rI = np.array([r[3] for r in ridge])
    finite_tI = np.where(np.isfinite(tI), tI, np.inf); finite_rI = np.where(np.isfinite(rI), rI, np.inf)
    print(f"\nbest truncation: k={tk[np.argmin(finite_tI)]}  I={finite_tI.min():.4f}  (cos={tcos[np.argmin(finite_tI)]:.3f})")
    print(f"best ridge:      lambda={rl[np.argmin(finite_rI)]:.2e}  I={finite_rI.min():.4f}  (cos={rcos[np.argmin(finite_rI)]:.3f})")

    # ---- figure: cos and I vs regularisation strength ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    Icap = lambda v: np.where(np.isfinite(v), v, np.nan)
    # (a) truncation
    a = ax[0]; a2 = a.twinx()
    a.semilogx(tk, tcos, "-", color="tab:blue", marker="o", ms=4, label="grad cos")
    a2.semilogx(tk, Icap(tI), "-", color="tab:red", marker="s", ms=4, label="$I_{cl}$")
    a2.axhline(I_or, ls="--", color="black", lw=1.0, label="oracle $I^*$")
    a2.axhline(I_nc, ls="--", color="gray", lw=1.0, label="no-control $I$")
    a.set_xlabel("truncation rank $k$"); a.set_ylabel("gradient cos", color="tab:blue")
    a2.set_ylabel("closed-loop $I$", color="tab:red"); a.set_title("(a) truncated SVD on $V_{nl}$")
    a.set_ylim(-1.05, 1.05); a2.set_ylim(0, max(I_nc * 1.6, np.nanmin(Icap(tI)) * 3 + 1e-6))
    # (b) ridge
    b = ax[1]; b2 = b.twinx()
    b.semilogx(rl, rcos, "-", color="tab:blue", marker="o", ms=4, label="grad cos")
    b2.semilogx(rl, Icap(rI), "-", color="tab:red", marker="s", ms=4, label="$I_{cl}$")
    b2.axhline(I_or, ls="--", color="black", lw=1.0)
    b2.axhline(I_nc, ls="--", color="gray", lw=1.0)
    b.set_xlabel("ridge $\\lambda$"); b.set_ylabel("gradient cos", color="tab:blue")
    b2.set_ylabel("closed-loop $I$", color="tab:red"); b.set_title("(b) ridge / Tikhonov on $V_{nl}$")
    b.set_ylim(-1.05, 1.05); b2.set_ylim(0, max(I_nc * 1.6, np.nanmin(Icap(rI)) * 3 + 1e-6))
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"target $V_{{nl}}$ | training data: {tag}", y=1.02, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"/tmp/truncation_ridge_sweep_{tag}.png", dpi=130, bbox_inches="tight")
    print(f"\nsaved figure /tmp/truncation_ridge_sweep_{tag}.png")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_trunc_ridge_{tag}"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "truncation_ridge_sweep.png", dpi=130)
    (out / "summary.json").write_text(json.dumps(dict(
        regime=REGIME, training_data=tag, n_train=len(Wtr), dim=d, cond=float(S[0] / S[-1]),
        I_nc=I_nc, I_or=I_or, full_Vnl=dict(r2=r2_f, cos=cos_f, I=I_f),
        full_Vlin=dict(r2=r2_l, cos=cos_l, I=I_l),
        truncation=[dict(k=int(t[0]), r2=t[1], cos=t[2], I=t[3]) for t in trunc],
        ridge=[dict(lam=float(r[0]), r2=r[1], cos=r[2], I=r[3]) for r in ridge],
    ), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
