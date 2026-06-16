"""Nonlinear continuous-time optimal-control oracle for Mackey-Glass (Pontryagin).

Solves the continuous-time two-point boundary value problem (state + costate) on the
collocation-augmented state, with the linearised P_c as terminal cost. The costate
lambda(t) = dV*/dz IS the value gradient (maximum principle), so u* = -1/2 R^-1 B^T lambda_0
= Doya's law. See documents/analysis/continuous_time_oracle/ (Prop. 6.1-6.3).

``build_pontryagin_dataset`` produces labelled samples (window, V*, u*) for the H1/H2 harness
by solving the BVP from a cloud of initial histories (the method of characteristics) and
sampling control-cadence windows along each optimal trajectory.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.integrate import solve_bvp


def mg_g(y, n):  return y / (1.0 + y ** n)
def mg_gp(y, n): return (1.0 + (1.0 - n) * y ** n) / (1.0 + y ** n) ** 2


def solve_pontryagin(z0, D, Pc, A_cl, xs, mu, p, n_exp, R, T, n_mesh=120):
    """Continuous-time Pontryagin TPBVP on the augmented state. Returns (sol, zstar)."""
    m = D.shape[0]; Rs = float(R); zstar = xs * np.ones(m); DT = D.T

    def rhs(t, y):
        z, lam = y[:m], y[m:]
        u = -0.5 * lam[0] / Rs
        Fz = D @ z
        Fz0 = -mu * z[0] + p * mg_g(z[-1], n_exp) + u
        Fz = np.vstack([Fz0[None, :], Fz[1:]])
        dq = np.zeros_like(z); dq[0] = 2.0 * (z[0] - xs)              # Q = 1
        JTlam = DT @ lam
        corr = -D[0, :][:, None] * lam[0][None, :]
        corr[0] += (-mu) * lam[0]
        corr[-1] += (p * mg_gp(z[-1], n_exp)) * lam[0]
        lamdot = -(dq + (JTlam + corr))
        return np.vstack([Fz, lamdot])

    def bc(ya, yb):
        return np.concatenate([ya[:m] - z0, yb[m:] - 2.0 * Pc @ (yb[:m] - zstar)])

    tmesh = np.linspace(0.0, T, n_mesh)
    Phi = scipy.linalg.expm(A_cl * (tmesh[1] - tmesh[0]))
    dz = (z0 - zstar).copy(); Zg = [dz.copy()]
    for _ in range(n_mesh - 1):
        dz = Phi @ dz; Zg.append(dz.copy())
    Zg = np.array(Zg).T
    Yg = np.vstack([zstar[:, None] + Zg, 2.0 * Pc @ Zg])
    return solve_bvp(rhs, bc, tmesh, Yg, max_nodes=20000, tol=1e-6), zstar


def value_along(sol, m, xs, Pc, R):
    """Cost-to-go V*_cost(z(t)) = remaining running cost + linearised terminal cost, with the
    control u*(t) = -1/2 R^-1 lambda_0. Returns (t, z0_traj, u, Vcost)."""
    t = sol.x; z, lam = sol.y[:m], sol.y[m:]
    u = -0.5 * lam[0] / float(R)
    run = (z[0] - xs) ** 2 + float(R) * u ** 2
    seg = 0.5 * (run[:-1] + run[1:]) * np.diff(t)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    dzT = z[:, -1] - xs
    Vcost = tail + float(dzT @ Pc @ dzT)
    return t, z[0], u, Vcost


def build_pontryagin_dataset(oracle, mu, p, n_exp, xs, dt, win, ics, T=20.0, stride=3):
    """Solve the BVP from each initial history in ``ics`` and sample (window, V*, u*) along the
    optimal trajectory. ``oracle`` is build_delayed_oracle(...) for the MG linearisation.

    Returns (W_list [k,(win,1)], Vstar [k], Ustar [k,1]) in REWARD convention (V* = -Vcost)."""
    D, Pc, Kc, Nmat, M, theta = (oracle["D"], oracle["Pc"], oracle["Kc"], oracle["Nmat"],
                                 oracle["M"], oracle["theta"])
    R = float(oracle["R"][0, 0]); m = D.shape[0]
    A_cl = M - Nmat @ Kc
    theta_inc = theta[::-1]                                  # increasing: -tau .. 0
    Ws, Vs, Us = [], [], []
    for z0 in ics:
        sol, zstar = solve_pontryagin(z0, D, Pc, A_cl, xs, mu, p, n_exp, R, T)
        if not sol.success:
            continue
        t, z0_traj, u, Vcost = value_along(sol, m, xs, Pc, R)
        # continuous x(t): IC history for t<0 (interp the IC collocation values), BVP z0 for t>=0
        def x_at(tt):
            return np.where(tt < 0.0, np.interp(tt, theta_inc, z0[::-1]),
                            np.interp(np.clip(tt, 0.0, t[-1]), t, z0_traj))
        ks = np.arange(0, int((t[-1]) / dt) + 1)[::stride]
        for k in ks:
            tc = k * dt
            offs = tc - dt * np.arange(win)[::-1]            # [tc-(win-1)dt, ..., tc]
            W = x_at(offs).reshape(win, 1)
            Ws.append(W)
            Vs.append(-float(np.interp(tc, t, Vcost)))       # reward convention
            Us.append(np.array([float(np.interp(tc, t, u))]))
    return Ws, np.array(Vs), np.array(Us)
