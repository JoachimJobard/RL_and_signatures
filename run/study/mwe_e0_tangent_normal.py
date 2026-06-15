"""E0: is the residual control error the NORMAL-to-manifold part of the gradient?

Follows run/study/mwe_e0_trajectory_sweep.py, which showed that pooling oracle
trajectories lifts cos(u_rep, u*) from 0 to ~0.63 and then SATURATES -- the data is
trapped on the 2-D window-manifold M = im(L), L = [e^{-M(L-1)dt};...;e^{-M dt};I],
M_dyn = A-BK. Hypothesis (point 3): adding trajectories pins the TANGENTIAL derivative
of V_theta on M, but the control reads the gradient w.r.t. the window ENDPOINT TAP,
which has a NORMAL component to M that no on-policy trajectory constrains.

Test. Fix n_ic. Decompose the window-gradient error
    E = grad_win( theta^T phi )  -  grad_win( V* ),     V*(W) = -W[-1]^T P W[-1],
(both in R^{L*N}) into tangent and normal parts w.r.t. M:
    tangent_err = || P_M E ||,   normal_err = || (I - P_M) E ||,
with P_M the orthogonal projector onto the 2-D data manifold (top-2 right singular
vectors of the pooled FULL-history windows). Prediction: tangent_err -> 0 as n_ic
grows (V_theta matches V* along M, so its tangential derivative is pinned), while
normal_err stays large (the endpoint-tap normal derivative is free). Also report the
fraction of the endpoint-tap directions that is normal to M -- i.e. how much of what
the control reads is structurally unconstrained.

Usage:
    uv run python run/study/mwe_e0_tangent_normal.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from datetime import datetime
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.linalg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.representations.factory import make_representation  # noqa: E402
from src.utils.run_context import script_data_dir  # noqa: E402

A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
Q = np.eye(2); R = np.array([[0.1]])
N_STATE = 2; WINDOW_LENGTH = 9; DT = 0.05; HORIZON = 6.0
X0_DEPLOY = np.array([1.0, 0.0])
N_ICS = [1, 4, 16, 64, 256]
N_IC_FIXED = 64
IC_SCALE = 1.5
KIND, DEG = "signature", 2
SEED = 0


def lqr():
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return P, np.linalg.inv(R) @ B.T @ P


def rollout(control_fn, n_steps, x0):
    win = deque([x0.copy() for _ in range(WINDOW_LENGTH)], maxlen=WINDOW_LENGTH)
    x = x0.copy(); states = [x.copy()]
    for _ in range(n_steps):
        u = np.asarray(control_fn(np.stack(list(win)))).reshape(-1)
        if not np.all(np.isfinite(u)) or np.linalg.norm(x) > 1e6:
            break
        d = lambda xx: A @ xx + (B @ u).reshape(-1)
        k1 = d(x); k2 = d(x + .5 * DT * k1); k3 = d(x + .5 * DT * k2); k4 = d(x + DT * k3)
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4); win.append(x.copy()); states.append(x.copy())
    return np.stack(states)


def full_windows(states):
    """Only windows with FULL real history (no front-padding) -> they lie exactly on
    the manifold im(L). Returns (n, WINDOW_LENGTH, N_STATE)."""
    out = []
    for k in range(WINDOW_LENGTH - 1, states.shape[0]):
        out.append(states[k - WINDOW_LENGTH + 1:k + 1])
    return np.stack(out) if out else np.empty((0, WINDOW_LENGTH, N_STATE))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    P, K = lqr(); n_steps = int(HORIZON / DT)
    rng = np.random.default_rng(SEED)
    LN = WINDOW_LENGTH * N_STATE

    rep = make_representation(kind=KIND, window_length=WINDOW_LENGTH, n_state=N_STATE,
                              depth=DEG, degree=DEG)
    feat = jax.jit(rep.feature_fn)

    ic_bank = [X0_DEPLOY] + [IC_SCALE * rng.standard_normal(N_STATE) for _ in range(max(N_ICS))]

    def oracle_windows(n_ic):
        Ws = [full_windows(rollout(lambda w: -K @ w[-1], n_steps, np.asarray(x0)))
              for x0 in ic_bank[:n_ic]]
        return np.concatenate(Ws, 0)

    # Manifold M: top-2 right singular vectors of the densely-covered full-history windows.
    ref = oracle_windows(max(N_ICS)).reshape(-1, LN)
    _, _, Vt = np.linalg.svd(ref, full_matrices=False)
    U2 = Vt[:2].T                       # (LN, 2) orthonormal basis of M
    P_M = U2 @ U2.T                      # projector onto the 2-D manifold
    # how much of the endpoint-tap directions is NORMAL to M:
    Dtap = np.zeros((LN, N_STATE)); Dtap[(WINDOW_LENGTH - 1) * N_STATE:, :] = np.eye(N_STATE)
    nu_tap = 1.0 - np.trace(Dtap.T @ P_M @ Dtap) / N_STATE

    # deployment full-history windows (where the control is queried)
    dep_W = full_windows(rollout(lambda w: -K @ w[-1], n_steps, X0_DEPLOY))

    def grad_win_theta(theta):
        th = jnp.asarray(theta)
        g = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
        return lambda W: np.asarray(g(jnp.asarray(W))).reshape(-1)

    def grad_win_star(W):
        G = np.zeros((WINDOW_LENGTH, N_STATE)); G[-1] = -2.0 * P @ W[-1]
        return G.reshape(-1)

    print(f"\nE0 tangent/normal decomposition | rep={KIND} d={int(rep.feature_dim)} "
          f"| L={WINDOW_LENGTH} dt={DT}")
    print(f"manifold dim(M)=2 in R^{LN}; endpoint-tap NORMAL fraction nu_tap={nu_tap:.3f} "
          f"(fraction of what the control reads that is normal to M)")
    print(f"\n{'n_ic':>6}{'||E||':>9}{'tan_err':>9}{'nor_err':>9}{'nor_frac':>9}  "
          f"(tan_err=||P_M E||, nor_err=||(I-P_M) E||, nor_frac=normal energy / total)")

    results = []
    for n_ic in N_ICS:
        Wf = oracle_windows(n_ic); Vf = -np.einsum("ti,ij,tj->t", Wf[:, -1], P, Wf[:, -1])
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wf])
        d = Phi.shape[1]; lam = 1e-8 * np.trace(Phi.T @ Phi) / d
        theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ Vf)
        gth = grad_win_theta(theta)
        E = np.stack([gth(W) - grad_win_star(W) for W in dep_W])         # (T, LN)
        Et, En = E @ P_M, E - E @ P_M
        tan = float(np.mean(np.linalg.norm(Et, axis=1)))
        nor = float(np.mean(np.linalg.norm(En, axis=1)))
        nfrac = float(np.sum(En ** 2) / (np.sum(E ** 2) + 1e-30))
        Enorm = float(np.mean(np.linalg.norm(E, axis=1)))
        marker = "  <-- fixed n_ic" if n_ic == N_IC_FIXED else ""
        print(f"{n_ic:>6}{Enorm:>9.3f}{tan:>9.3f}{nor:>9.3f}{nfrac:>9.3f}{marker}")
        results.append(dict(n_ic=n_ic, E_norm=Enorm, tan_err=tan, nor_err=nor, nor_frac=nfrac))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_tangent_normal"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(rep=KIND, dim=int(rep.feature_dim), nu_tap=float(nu_tap),
             n_ic_fixed=N_IC_FIXED, results=results), indent=2, default=str))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
