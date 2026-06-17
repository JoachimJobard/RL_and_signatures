"""Nonlinear continuous-time optimal-control oracle for the delayed Hopfield network.

Multichannel generalisation of ``mackey_glass_pontryagin`` (which is scalar-state specific:
a single delayed scalar nonlinearity g(z_{-1}) with control on z_0). Here the plant is the
n-channel delay-coupled Hopfield network

    x'(t) = -x(t) + W phi_eps( x(t-tau) ) + B u ,
    phi_eps(xi) = (1-eps) xi + eps tanh(kappa xi)/kappa ,

so the only nonlinearity is the delayed coupling W phi_eps(x_tau), entering the boundary
(theta = 0) block of the Chebyshev-collocated augmented state z in R^{n(N+1)} laid out in
blocks z = [z_0, ..., z_N] (z_i = state at node theta_i; theta_0 = 0, theta_N = -tau), the
SAME layout as ``continuous_oracle.build_delayed_oracle``.

We solve the continuous-time Pontryagin two-point BVP on (z, lambda) with the linearised
terminal cost Pc; the costate lambda(t) = dV*/dz is the value gradient (maximum principle),
so u* = -1/2 R^-1 B^T lambda_0 is Doya's law. The augmented transport (rows 1..N) and the
linearised boundary (A0 = -I, A1 = W) are reused from the linear oracle's M, Nmat; only the
boundary block swaps the linear W z_N for the nonlinear W phi_eps(z_N), with the matching
costate Jacobian W (diag phi'_eps(z_N) - I).

ACCEPTANCE (verified in __main__): at eps = 0 the BVP reduces EXACTLY to the linear
delayed-LQR oracle (u*, dV* match Kc, Pc), and the free response (u = 0) matches the env
integrator -- the two correctness gates that caught the MG 2/tau scaling bug.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.integrate import solve_bvp


def phi_eps(xd, eps, kappa, nonlinearity="tanh"):
    if nonlinearity == "cubic":
        return xd + kappa * xd ** 3                          # Duffing/hardening, non-saturating
    return (1.0 - eps) * xd + eps * np.tanh(kappa * xd) / kappa


def phip_eps(xd, eps, kappa, nonlinearity="tanh"):
    if nonlinearity == "cubic":
        return 1.0 + 3.0 * kappa * xd ** 2                   # phi' grows with amplitude (history kept)
    return (1.0 - eps) + eps * (1.0 / np.cosh(kappa * xd) ** 2)


def solve_pontryagin_hopfield(z0, oracle, W, eps, kappa, T=15.0, n_mesh=140, nonlinearity="tanh",
                              damping=0.0):
    """Continuous-time Pontryagin TPBVP for the delayed Hopfield plant on the augmented
    state. ``oracle`` is build_delayed_oracle(A0=-I, A1=W, B, Q, R, tau, N). Returns
    (sol, zstar) with zstar = 0 (origin equilibrium)."""
    M, Nmat, Pc, Kc = oracle["M"], oracle["Nmat"], oracle["Pc"], oracle["Kc"]
    B, R, Q = oracle["B"], oracle["R"], oracle["Q"]
    n, N = oracle["n"], oracle["N"]
    big = n * (N + 1)
    Wm = np.asarray(W)
    half_RinvBT = 0.5 * np.linalg.inv(np.asarray(R)) @ np.asarray(B).T   # (m, n)
    Qn = np.asarray(Q)
    zstar = np.zeros(big)

    def rhs(t, y):
        z, lam = y[:big], y[big:]
        z0b, zN = z[:n], z[big - n:big]
        lam0 = lam[:n]
        u = -half_RinvBT @ lam0                                   # (m, P)
        Fz = M @ z + Nmat @ u
        Fz[:n] += Wm @ (phi_eps(zN, eps, kappa, nonlinearity) - zN)   # swap linear -> nonlinear at boundary
        Fz[:n] += -damping * z0b ** 3                            # instantaneous cubic confinement (Duffing)
        dq = np.zeros_like(z); dq[:n] = 2.0 * (Qn @ z0b)          # d/dz0 of (z0^T Q z0), zstar=0
        JTlam = M.T @ lam
        WT_lam0 = Wm.T @ lam0                                     # (n, P)
        corr = np.zeros_like(lam)
        corr[big - n:big] = phip_eps(zN, eps, kappa, nonlinearity) * WT_lam0 - WT_lam0   # E^T lambda at block N
        corr[:n] += -3.0 * damping * z0b ** 2 * lam0             # d/dz0 of -damping z0^3 (block-0 Jacobian)
        lamdot = -(dq + JTlam + corr)
        return np.vstack([Fz, lamdot])

    def bc(ya, yb):
        return np.concatenate([ya[:big] - z0, yb[big:] - 2.0 * Pc @ (yb[:big] - zstar)])

    tmesh = np.linspace(0.0, T, n_mesh)
    A_cl = M - Nmat @ Kc                                          # linear closed loop for the initial guess
    Phi = scipy.linalg.expm(A_cl * (tmesh[1] - tmesh[0]))
    dz = (z0 - zstar).copy(); Zg = [dz.copy()]
    for _ in range(n_mesh - 1):
        dz = Phi @ dz; Zg.append(dz.copy())
    Zg = np.array(Zg).T
    Yg = np.vstack([zstar[:, None] + Zg, 2.0 * Pc @ Zg])
    return solve_bvp(rhs, bc, tmesh, Yg, max_nodes=40000, tol=1e-6), zstar


def value_along_hopfield(sol, oracle):
    """Cost-to-go V*_cost(z(t)) = remaining running cost + linearised terminal cost, with
    u*(t) = -1/2 R^-1 B^T lambda_0(t). Returns (t, z0_traj [n, P], u [m, P], Vcost [P])."""
    n, N = oracle["n"], oracle["N"]
    big = n * (N + 1)
    B, R, Q, Pc = oracle["B"], oracle["R"], oracle["Q"], oracle["Pc"]
    half_RinvBT = 0.5 * np.linalg.inv(np.asarray(R)) @ np.asarray(B).T
    Qn, Rn = np.asarray(Q), np.asarray(R)
    t = sol.x; z, lam = sol.y[:big], sol.y[big:]
    z0b, lam0 = z[:n], lam[:n]
    u = -half_RinvBT @ lam0                                       # (m, P)
    run = np.einsum("ip,ij,jp->p", z0b, Qn, z0b) + np.einsum("ip,ij,jp->p", u, Rn, u)
    seg = 0.5 * (run[:-1] + run[1:]) * np.diff(t)
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
    zT = z[:, -1]
    Vcost = tail + float(zT @ Pc @ zT)
    return t, z0b, u, Vcost


def build_hopfield_pontryagin_dataset(oracle, W, eps, kappa, dt, win, ics_x0, T=15.0, stride=1,
                                      nonlinearity="tanh", damping=0.0):
    """Solve the BVP from each constant-history initial state x0 in ``ics_x0`` (n-vectors) and
    sample (window [win, n], V*, u* [m]) along the optimal trajectory. Returns
    (W_list, Vstar [k], Ustar [k, m]) in REWARD convention (V* = -Vcost)."""
    n, N = oracle["n"], oracle["N"]
    Ws, Vs, Us = [], [], []
    for x0 in ics_x0:
        x0 = np.asarray(x0, dtype=float).reshape(n)
        z0 = np.tile(x0, N + 1)                                   # constant history on [-tau, 0]
        sol, _ = solve_pontryagin_hopfield(z0, oracle, W, eps, kappa, T=T, nonlinearity=nonlinearity,
                                           damping=damping)
        if not sol.success:
            continue
        t, z0b, u, Vcost = value_along_hopfield(sol, oracle)       # z0b [n, P], u [m, P]
        tmax = t[-1]

        def x_at(tt):                                             # (n,) at scalar time tt
            if tt < 0.0:
                return x0
            return np.array([np.interp(min(tt, tmax), t, z0b[c]) for c in range(n)])

        ks = np.arange(0, int(tmax / dt) + 1)[::stride]
        for k in ks:
            tc = k * dt
            offs = tc - dt * np.arange(win)[::-1]                 # [tc-(win-1)dt, ..., tc]
            Wmat = np.stack([x_at(o) for o in offs], axis=0)      # (win, n)
            Ws.append(Wmat)
            Vs.append(-float(np.interp(tc, t, Vcost)))            # reward convention
            Us.append(np.array([float(np.interp(tc, t, u[c])) for c in range(u.shape[0])]))
    return Ws, np.array(Vs), np.array(Us)


# TESTING: the two correctness gates ==============================================
if __name__ == "__main__":
    from src.envs.delayed_hopfield_network import DelayedHopfieldNetwork, rotational_coupling
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.solvers.continuous_oracle import build_delayed_oracle, z_from_window
    import jax, jax.numpy as jnp

    W = rotational_coupling(rho=2.0, theta=np.pi / 2.0)
    tau, dt, Ncheb = 0.5, 0.1, 12
    A0, A1, B = -np.eye(2), W, np.eye(2)
    Q, R = np.eye(2), 0.1 * np.eye(2)
    oracle = build_delayed_oracle(A0, A1, B, Q, R, tau, Ncheb)
    Kc, Pc, theta, n = oracle["Kc"], oracle["Pc"], oracle["theta"], 2
    big = 2 * (Ncheb + 1)

    # ---- GATE 1: eps=0 must reduce to the linear delayed-LQR oracle (u*, dV*) ----
    rng = np.random.default_rng(0)
    cos_u, mag_u, cos_g = [], [], []
    for _ in range(8):
        z0 = 0.3 * rng.standard_normal(big)
        sol, _ = solve_pontryagin_hopfield(z0, oracle, W, eps=0.0, kappa=1.0, T=15.0)
        if not sol.success:
            print("  eps=0 BVP failed"); continue
        t, z0b, u, _ = value_along_hopfield(sol, oracle)
        u_bvp = u[:, 0]                                           # control at t=0
        u_lin = -Kc @ z0                                         # linear-oracle control
        cos_u.append(u_bvp @ u_lin / (np.linalg.norm(u_bvp) * np.linalg.norm(u_lin) + 1e-12))
        mag_u.append(np.linalg.norm(u_bvp) / (np.linalg.norm(u_lin) + 1e-12))
        # value gradient: lambda_0(0) should equal 2 Pc z0 restricted to block 0 via Doya
        lam0 = sol.y[big:big + n, 0]
        g_lin = (2.0 * Pc @ z0)[:n]
        cos_g.append(lam0 @ g_lin / (np.linalg.norm(lam0) * np.linalg.norm(g_lin) + 1e-12))
    print(f"GATE 1 (eps=0 -> linear oracle): cos(u)={np.mean(cos_u):.4f} |u|/|u*|={np.mean(mag_u):.4f} "
          f"cos(grad)={np.mean(cos_g):.4f}  (all must be ~1.000)")

    # ---- GATE 2: free response (u=0) of the BVP rhs must match the env integrator ----
    # Integrate the open-loop nonlinear plant from a constant history, compare to the env.
    for eps in (0.0, 1.0):
        env = DelayedHopfieldNetwork(W=W, delay=tau, eps=eps, step_size=dt, resolution=4)
        x0 = np.array([0.4, -0.2])
        w = JAXEnvWrapper(env, rng_key=0); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
        xs_env = [x0.copy()]
        for _ in range(int(2.0 / dt)):
            _, x, _ = w.step(w.state, jnp.zeros(2)); xs_env.append(np.array(x))
        xs_env = np.array(xs_env)
        # collocation free integration of x' = -x + W phi(x_tau) from constant history x0
        from scipy.integrate import solve_ivp
        z0 = np.tile(x0, Ncheb + 1)
        M = oracle["M"]
        def free(t, z):
            zN = z[big - n:]
            Fz = M @ z; Fz[:n] += W @ (phi_eps(zN, eps, 1.0) - zN)
            return Fz
        sol = solve_ivp(free, [0, 2.0], z0, t_eval=np.arange(0, 2.0 + 1e-9, dt), rtol=1e-8, atol=1e-10)
        x_colloc = sol.y[:n, :].T
        err = np.max(np.abs(x_colloc[:len(xs_env)] - xs_env[:len(x_colloc)]))
        print(f"GATE 2 (free response eps={eps}): max|x_colloc - x_env| over T=2 = {err:.2e}  (small => consistent)")
