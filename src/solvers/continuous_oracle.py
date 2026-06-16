"""Continuous-time LQR oracles for value-gradient (Doya) control.

Two builders, both producing a *continuous-time* value $V^\\star$ whose gradient, fed to
Doya's law $u=\\tfrac12 R^{-1}B^\\top\\partial_x V^\\star$, reproduces the optimal control
exactly (no $\\Delta t$ rescale -- see documents/analysis/continuous_time_oracle/):

  * ``build_markovian_oracle``: standard continuous ARE for a no-delay LTI (the H1
    negative-control cell). $V^\\star(x)=-x^\\top P x$, $u^\\star=-Kx$.
  * ``build_delayed_oracle``: continuous delayed-LQR by Chebyshev collocation of the
    history generator (platoon, linear-DDE, Mackey-Glass linearisation). $V^\\star(z)=
    -\\delta z^\\top P_c\\,\\delta z$ on the augmented (collocated history) state,
    $u^\\star=-K_c\\,\\delta z$.

``z_from_window`` maps a control-cadence history window to the augmented state $z$.

The derivation and the validation protocol (the magnitude gate, the Lyapunov/integral
cross-checks) are in documents/analysis/continuous_time_oracle/continuous_time_oracle_derivation.md.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg


def cheb(N: int):
    """Trefethen Chebyshev differentiation matrix D and nodes x in [-1,1], x_0=1 .. x_N=-1."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def build_markovian_oracle(A, B, Q, R):
    """Standard continuous-time LQR (no delay). V*(x) = -x^T P x, u* = -K x."""
    A, B, Q, R = map(np.asarray, (A, B, Q, R))
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return dict(kind="markovian", P=P, K=K, n=A.shape[0], A=A, B=B, Q=Q, R=R)


def build_delayed_oracle(A0, A1, B, Q, R, tau, N):
    """Continuous delayed-LQR by Chebyshev collocation of the history on [-tau, 0].

    Returns Pc (value matrix), Kc (feedback gain) on the augmented state z in R^{n(N+1)},
    the collocation nodes theta (theta_0=0 .. theta_N=-tau), and the augmented (M, Nmat).
    """
    A0, A1, B, Q, R = map(np.asarray, (A0, A1, B, Q, R))
    n, m = A0.shape[0], B.shape[1]
    Dc, xc = cheb(N)
    theta = (tau / 2.0) * (xc - 1.0)               # theta_0 = 0, theta_N = -tau
    D = (2.0 / tau) * Dc                            # d/dtheta on [-tau, 0]  (the 2/tau scaling)
    big = n * (N + 1)
    M = np.zeros((big, big))
    for i in range(1, N + 1):                       # transport rows
        for j in range(N + 1):
            M[i * n:(i + 1) * n, j * n:(j + 1) * n] = D[i, j] * np.eye(n)
    M[0:n, 0:n] = A0                                # boundary (theta = 0) = the dynamics
    M[0:n, N * n:(N + 1) * n] = A1
    Nmat = np.zeros((big, m)); Nmat[0:n, :] = B
    Qz = np.zeros((big, big)); Qz[0:n, 0:n] = Q
    Pc = scipy.linalg.solve_continuous_are(M, Nmat, Qz, R)
    Kc = np.linalg.inv(R) @ Nmat.T @ Pc
    return dict(kind="delayed", Pc=Pc, Kc=Kc, theta=theta, M=M, Nmat=Nmat, n=n, m=m, N=N,
                A0=A0, A1=A1, B=B, Q=Q, R=R, tau=float(tau))


def z_from_window(W, dt, theta, n):
    """Map a control-cadence window W (shape (L, n), chronological: oldest first, current last)
    to the augmented state z by interpolating each channel onto the Chebyshev nodes theta."""
    W = np.asarray(W)
    L = len(W)
    t_grid = -dt * np.arange(L)[::-1]               # increasing time: [-(L-1)dt, ..., -dt, 0]
    return np.stack([np.interp(theta, t_grid, W[:, c]) for c in range(n)], axis=1).reshape(-1)


def doya_gate(oracle, dt, n_samples=200, scale=0.3, seed=0):
    """Magnitude gate: Doya's u = 1/2 R^-1 B^T dV/dx must equal the oracle control on a
    state cloud (cos = 1, |u|/|u*| = 1). Returns (cos, mag). See derivation note Prop. 5.2."""
    rng = np.random.default_rng(seed)
    R, B = oracle["R"], oracle["B"]
    Rinv = np.linalg.inv(R)
    cos_l, mag_l = [], []
    if oracle["kind"] == "markovian":
        P, K, n = oracle["P"], oracle["K"], oracle["n"]
        for _ in range(n_samples):
            x = scale * rng.standard_normal(n)
            u_or = -K @ x
            u_do = -Rinv @ B.T @ (P @ x)            # 1/2 R^-1 B^T d(-x^T P x)/dx
            cos_l.append(u_or @ u_do / (np.linalg.norm(u_or) * np.linalg.norm(u_do) + 1e-12))
            mag_l.append(np.linalg.norm(u_do) / (np.linalg.norm(u_or) + 1e-12))
    else:
        Pc, Kc, n, N = oracle["Pc"], oracle["Kc"], oracle["n"], oracle["N"]
        big = n * (N + 1)
        for _ in range(n_samples):
            dz = scale * rng.standard_normal(big)
            u_or = -Kc @ dz
            u_do = -Rinv @ B.T @ (Pc @ dz)[:n]
            cos_l.append(u_or @ u_do / (np.linalg.norm(u_or) * np.linalg.norm(u_do) + 1e-12))
            mag_l.append(np.linalg.norm(u_do) / (np.linalg.norm(u_or) + 1e-12))
    return float(np.mean(cos_l)), float(np.mean(mag_l))
