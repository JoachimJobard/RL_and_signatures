"""A CONTINUOUS-TIME delayed-LQR oracle for the platoon, via Chebyshev collocation.

The discrete augmented LQR (augmented_discrete_lqr) discretises with Bd = dt*B, so
Doya's CONTINUOUS control u = 1/2 R^-1 B^T dV/dx mismatches its value by 1/dt. To test
Doya faithfully we need a CONTINUOUS-time value. For a delayed system the continuous
value is infinite-dimensional (a functional of the history), so we discretise the
HISTORY (not time) by pseudospectral collocation:

  state of the DDE = the history phi(theta), theta in [-tau, 0];
  generator:  (A phi)(theta) = phi'(theta)  for theta in [-tau,0),
              (A phi)(0)      = A phi(0) + A1 phi(-tau) + B u   (boundary = the dynamics).
On Chebyshev points theta_0=0 > ... > theta_N=-tau with differentiation matrix D, this
is a CONTINUOUS-TIME finite LTI system  z' = M z + Nmat u,  z=[phi(theta_0),...],
z_0 = x(t). Solve the CONTINUOUS ARE -> P_c, V(z) = -z^T P_c z, u* = -R^-1 Nmat^T P_c z.
Then Doya's continuous formula u = 1/2 R^-1 B^T dV/dz_0 EQUALS u* exactly -- no dt, no
rescale. This script builds it, verifies the match, and runs it closed-loop on the
nonlinear env (vs the discrete oracle).

Usage:
    uv run python run/study/mwe_continuous_delayed_oracle.py [--debug]
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
N_CHEB = 12          # collocation points for the history
CLIP = 3.0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def disc_oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R), float(env.max_delay), env.step_size)


def cheb(N):
    """Trefethen Chebyshev differentiation matrix D and points x in [-1,1], x_0=1..x_N=-1."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def build_continuous_oracle(env, N):
    A, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
    Q, R = np.array(env.Q), np.array(env.R)
    n, m = A.shape[0], B.shape[1]; tau = float(env.max_delay)
    Dc, xc = cheb(N)
    theta = (tau / 2.0) * (xc - 1.0)          # theta_0=0, theta_N=-tau
    D = (2.0 / tau) * Dc                       # d/dtheta on [-tau,0]
    big = n * (N + 1)
    M = np.zeros((big, big))
    for i in range(1, N + 1):                  # transport at interior/lower points
        for j in range(N + 1):
            M[i * n:(i + 1) * n, j * n:(j + 1) * n] = D[i, j] * np.eye(n)
    M[0:n, 0:n] = A                            # boundary (theta=0) = the dynamics
    M[0:n, N * n:(N + 1) * n] = A1
    Nmat = np.zeros((big, m)); Nmat[0:n, :] = B
    Qz = np.zeros((big, big)); Qz[0:n, 0:n] = Q
    Pc = scipy.linalg.solve_continuous_are(M, Nmat, Qz, R)
    Kc = np.linalg.inv(R) @ Nmat.T @ Pc
    return dict(M=M, Nmat=Nmat, Pc=Pc, Kc=Kc, theta=theta, n=n, m=m, N=N, B=B, R=R)


def z_from_history(o_win, dt, theta, n):
    """Interpolate the control-cadence history o_win (newest first) onto Chebyshev theta."""
    k = len(o_win) - 1
    t_grid = -dt * np.arange(k + 1)            # [0, -dt, ..., -k dt], newest first
    ti = t_grid[::-1]; vi = np.array(o_win)[::-1]   # increasing time
    z = np.stack([np.interp(theta, ti, vi[:, c]) for c in range(n)], axis=1)  # (N+1, n)
    return z.reshape(-1)


def closed_loop(env, control_xi, k, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    o_win = [x0.copy() for _ in range(k + 1)]; I, t = 0.0, 0.0
    while t < TF - 1e-9:
        u = np.clip(control_xi(o_win), -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * env.step_size; o_win = [x] + o_win[:-1]
    return I


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); n, dt = env.N, env.step_size
    dl = disc_oracle(env); kd, gain_d = dl.k_taps, dl.gain
    oc = build_continuous_oracle(env, N_CHEB)
    Pc, Kc, theta, B, R = oc["Pc"], oc["Kc"], oc["theta"], oc["B"], oc["R"]
    Rinv = np.linalg.inv(R)

    # spectral sanity: closed-loop generator must be stable
    eig = np.linalg.eigvals(oc["M"] - oc["Nmat"] @ Kc)
    print(f"\nCONTINUOUS-TIME delayed-LQR oracle (Chebyshev N={N_CHEB}) | dim z = {n*(N_CHEB+1)}")
    print(f"closed-loop generator max Re(eig) = {np.max(eig.real):.4f}  (<0 = stabilising)")

    x0 = np.zeros(n); x0[1] = 0.5

    # verify Doya's continuous formula == continuous oracle on a sample history cloud
    rng = np.random.default_rng(0); cos_l, mag_l = [], []
    for _ in range(200):
        ow = [0.3 * rng.standard_normal(n) for _ in range(kd + 1)]
        z = z_from_history(ow, dt, theta, n)
        u_oracle = -Kc @ z
        u_doya = Rinv @ B.T @ (Pc @ z)[:n] * (-1.0)   # 1/2 R^-1 B^T dV/dx0, V=-z^T Pc z -> -R^-1 B^T (Pc z)|0
        cos_l.append(u_oracle @ u_doya / (np.linalg.norm(u_oracle) * np.linalg.norm(u_doya) + 1e-12))
        mag_l.append(np.linalg.norm(u_doya) / (np.linalg.norm(u_oracle) + 1e-12))
    print(f"Doya continuous formula vs continuous oracle: cos={np.mean(cos_l):.4f}  |u|/|u*|={np.mean(mag_l):.4f}"
          f"  (expect 1.0 / 1.0 -- no dt mismatch)")

    # closed-loop costs on the nonlinear env
    I_nc = closed_loop(env, lambda ow: np.zeros(oc["m"]), kd, x0)
    I_disc = closed_loop(env, lambda ow: -gain_d @ np.concatenate(ow[:kd + 1]), kd, x0)
    I_cont = closed_loop(env, lambda ow: -Kc @ z_from_history(ow, dt, theta, n), kd, x0)
    print(f"\nclosed-loop cost on the NONLINEAR env:")
    print(f"  no-control                : I={I_nc:.4f}")
    print(f"  discrete oracle (existing): I={I_disc:.4f}")
    print(f"  CONTINUOUS oracle (new)   : I={I_cont:.4f}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_continuous_oracle"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, N_cheb=N_CHEB,
                                                      I_nc=I_nc, I_disc=I_disc, I_cont=I_cont), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
