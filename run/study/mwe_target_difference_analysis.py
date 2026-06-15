"""In-depth analysis: WHAT changes between the linearised target V_lin and the nonlinear
target V_nl, and HOW a 0.07% value difference flips the value-gradient.

Both targets on the same windows; representation = raw_history deg-2 (the one that went
cos 0.997 -> -0.226). We decompose in the SVD basis of the feature matrix Phi = U S V^T:
  - the LEAST-SQUARES fit is theta = sum_i (u_i^T V_target / sigma_i) v_i, so the fit
    coefficient along singular direction i is amplified by 1/sigma_i;
  - V_lin is SMOOTH -> its energy u_i^T V_lin concentrates in low-i (large sigma);
  - dV = V_nl - V_lin is ROUGH -> spreads to high-i (small sigma), where 1/sigma_i blows
    it up. We quantify all of this and plot it.

Usage:
    uv run python run/study/mwe_target_difference_analysis.py [--debug]
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
H_VAL = 150
N_IC = 8
SUBSAMPLE = 4
AUG_PER = 2
EPS_TRAIN = 0.3
IC_SCALE = 0.8
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


def roughness(v):
    """RMS of successive differences / RMS of (centred) values -- high = jagged."""
    return float(np.sqrt(np.mean(np.diff(v) ** 2)) / (np.std(v) + 1e-12))


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
    Vlin_of = lambda W: -(z_of_window(W, dt, theta, n) @ Pc @ z_of_window(W, dt, theta, n))

    # deployment (time-ordered) for the smoothness view
    dep_W, dep_sn = rollout(env, Kc, theta, k, x0c)
    u_star = np.array([-Kc @ z_of_window(W, dt, theta, n) for W in dep_W])
    Vlin_dep = np.array([Vlin_of(W) for W in dep_W])
    Vnl_dep = np.array([nonlinear_value(env, Kc, theta, sn, W, H_VAL, jit_step)
                        for W, sn in zip(dep_W, dep_sn)])
    dV_dep = Vnl_dep - Vlin_dep

    # training set (on-manifold + cheat) for the fit / SVD
    rng = np.random.default_rng(SEED)
    ics = [x0c]
    for _ in range(N_IC - 1):
        z = np.zeros(n); z[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(z)
    pairs = []
    for ic in ics:
        Ws, sn = rollout(env, Kc, theta, k, ic)
        for j in range(0, len(Ws), SUBSAMPLE):
            pairs.append((Ws[j], sn[j]))
    items = list(pairs)
    for _ in range(AUG_PER):
        for (W, s) in pairs:
            sp, Wp = perturb(s, W, EPS_TRAIN * rng.standard_normal(n)); items.append((Wp, sp))
    Wtr = np.array([it[0] for it in items])
    Vlin = np.array([Vlin_of(W) for W in Wtr])
    Vnl = np.array([nonlinear_value(env, Kc, theta, s, W, H_VAL, jit_step) for (W, s) in items])
    dV = Vnl - Vlin

    raw = make_representation("raw_history", window_length=WIN_LEN, n_state=n, degree=2)
    feat = jax.jit(raw.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
    U, S, Vt = np.linalg.svd(Phi, full_matrices=False)        # Phi = U diag(S) Vt

    c_lin = U.T @ Vlin                                        # energy of each target in singular basis
    c_dV = U.T @ dV
    th_lin = (c_lin / S)                                      # fit coeff along v_i (amplified by 1/sigma)
    th_dV = (c_dV / S)

    # gradient consequence on deployment
    theta_lin = Vt.T @ (c_lin / S); theta_nl = Vt.T @ ((U.T @ Vnl) / S)
    th_l, th_n = jnp.asarray(theta_lin), jnp.asarray(theta_nl)
    g_l = jax.jit(jax.grad(lambda p: jnp.dot(th_l, feat(p))))
    g_n = jax.jit(jax.grad(lambda p: jnp.dot(th_n, feat(p))))
    u_l = np.array([half_RinvBT @ np.array(g_l(jnp.asarray(W)))[-1] for W in dep_W])
    u_n = np.array([half_RinvBT @ np.array(g_n(jnp.asarray(W)))[-1] for W in dep_W])
    cos_l = float(np.sum(u_l * u_star) / (np.linalg.norm(u_l) * np.linalg.norm(u_star) + 1e-12))
    cos_n = float(np.sum(u_n * u_star) / (np.linalg.norm(u_n) * np.linalg.norm(u_star) + 1e-12))

    half = len(S) // 2
    print("\n=== TARGET DIFFERENCE ANALYSIS (raw_history d2) ===")
    print(f"R^2(V_lin -> V_nl)                 = {1 - np.sum(dV**2)/np.sum((Vnl-Vnl.mean())**2):.5f}")
    print(f"||dV||/||V_nl-mean||  (RMS)         = {np.std(dV)/ (np.std(Vnl)+1e-12):.4f}")
    print(f"roughness (succ.diff/std), V_lin    = {roughness(Vlin_dep):.4f}")
    print(f"roughness (succ.diff/std), dV       = {roughness(dV_dep):.4f}   <- jaggedness of the difference")
    print(f"energy of V_lin in BOTTOM-half sing = {np.sum(c_lin[half:]**2)/np.sum(c_lin**2):.4f}")
    print(f"energy of dV    in BOTTOM-half sing = {np.sum(c_dV[half:]**2)/np.sum(c_dV**2):.4f}   <- dV spreads to small sigma")
    print(f"||theta_nl|| / ||theta_lin||        = {np.linalg.norm(theta_nl)/np.linalg.norm(theta_lin):.2f}")
    print(f"||dtheta|| / ||theta_lin||          = {np.linalg.norm(theta_nl-theta_lin)/np.linalg.norm(theta_lin):.2f}")
    print(f"gradient cos:  V_lin = {cos_l:.3f}   V_nl = {cos_n:.3f}")
    print(f"cond(Phi) = sigma_max/sigma_min     = {S[0]/S[-1]:.2e}")

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    i = np.arange(len(S))
    ax[0, 0].semilogy(i, S, "k-"); ax[0, 0].set_title("(a) singular spectrum $\\sigma_i$ of $\\Phi$")
    ax[0, 0].set_xlabel("singular index $i$"); ax[0, 0].set_ylabel("$\\sigma_i$")
    ax[0, 1].semilogy(i, np.abs(c_lin) + 1e-18, label="$|u_i^T V_{lin}|$ (smooth)")
    ax[0, 1].semilogy(i, np.abs(c_dV) + 1e-18, label="$|u_i^T\\,\\Delta V|$ (rough)")
    ax[0, 1].set_title("(b) where each target lives in the basis"); ax[0, 1].set_xlabel("singular index $i$")
    ax[0, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax[1, 0].semilogy(i, np.abs(th_lin) + 1e-18, label="$|c_i^{lin}/\\sigma_i|$")
    ax[1, 0].semilogy(i, np.abs(th_dV) + 1e-18, label="$|c_i^{\\Delta V}/\\sigma_i|$ (blows up)")
    ax[1, 0].set_title("(c) FITTED coeff = target/$\\sigma_i$ (the amplification)")
    ax[1, 0].set_xlabel("singular index $i$")
    ax[1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    tt = np.arange(len(dV_dep))
    ax[1, 1].plot(tt, Vlin_dep, "k--", label="$V_{lin}(t)$")
    sc = np.std(Vlin_dep) / (np.std(dV_dep) + 1e-12)
    ax[1, 1].plot(tt, dV_dep * sc, "-", color="crimson", lw=0.7, label="$\\Delta V(t)\\times%.0f$ (rescaled)" % sc)
    ax[1, 1].set_title("(d) along the trajectory: $V_{lin}$ smooth, $\\Delta V$ jagged")
    ax[1, 1].set_xlabel("time step"); ax[1, 1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0.03, 1, 1]); fig.subplots_adjust(hspace=0.5)
    fig.savefig("/tmp/target_difference_analysis.png", dpi=130)
    print("\nsaved figure /tmp/target_difference_analysis.png")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_target_diff"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "target_difference_analysis.png", dpi=130)
    (out / "summary.json").write_text(json.dumps(dict(cos_lin=cos_l, cos_nl=cos_n), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
