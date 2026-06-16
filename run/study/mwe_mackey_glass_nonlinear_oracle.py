"""The NONLINEAR continuous-time optimal-control oracle for Mackey-Glass (Pontryagin).

The linearised oracle (mwe_mackey_glass_continuous_oracle.py) gives a QUADRATIC V*, on
which raw-history degree-2 is exact -> H2 would fail as on the platoon. The H2 signal needs
the genuinely non-polynomial value, i.e. the OPTIMAL value of the NONLINEAR plant. We keep
time continuous (collocate only the history) and solve the continuous-time Pontryagin
two-point boundary value problem on the augmented state z in R^{N+1}:

  dynamics   z' = F(z,u):  row0 = -mu z0 + p g(z_N) + u  (g(y)=y/(1+y^n)),  rows i = (D z)_i
  Hamiltonian H = q(z) + u^T R u + lambda^T F,   q(z) = Q (z0 - x*)^2,  lambda = dV*/dz
  stationarity  u* = -1/2 R^-1 B^T lambda_0          <-- this IS Doya's law (costate = value grad)
  costate    lambda' = -(dq/dz + J(z)^T lambda),     J(z) = D with row0 = [-mu,0,..,0, p g'(z_N)]
  BCs        z(0) = z0,   lambda(T) = 2 P_c (z(T) - z*)   (linearised LQR terminal cost = inf-horizon tail)

The costate lambda(t) = dV*/dz EXACTLY (maximum principle), so the magnitude gate
|u|/|u*| = 1 holds by construction -- we VERIFY it by finite-differencing V*(z0) and
comparing to lambda(0). We also (a) cross-check the optimal control on the true nonlinear
env, and (b) measure how NON-quadratic V* is (R^2 of the linearised value vs V*_nl) -- the
H2 prerequisite that failed on the platoon (99.9% quadratic).

Usage:
    uv run python run/study/mwe_mackey_glass_nonlinear_oracle.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.integrate import solve_bvp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REGIME = dict(delay=6.0, mu=0.1, n=10, p=0.2, x_target=1.0, step_size=0.25, resolution=5)
Q = np.array([[1.0]]); R = np.array([[0.1]])
N_CHEB = 10          # collocation points (modest -> BVP dim 2(N+1) manageable, less stiff)
T_HOR = 20.0         # horizon; linear closed loop (rate ~0.45) is settled, P_c handles the tail
X0_CONST = 0.8       # constant-history IC (lets us cross-check against the real env cleanly)


def make_env():
    from src.envs.mackey_glass_1D import MackeyGlass1DEnv
    return MackeyGlass1DEnv(delay=REGIME["delay"], step_size=REGIME["step_size"],
                            resolution=REGIME["resolution"], Q=Q, R=R, n=REGIME["n"],
                            p=REGIME["p"], mu=REGIME["mu"], x_target=REGIME["x_target"])


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T; dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1)); D = D - np.diag(D.sum(axis=1))
    return D, x


def mg_g(y, n):  return y / (1.0 + y ** n)
def mg_gp(y, n): return (1.0 + (1.0 - n) * y ** n) / (1.0 + y ** n) ** 2


def linear_oracle(D, mu, p, n, xs, tau):
    """Linearised continuous delayed-LQR about x*: returns Pc, Kc, M (for warm start / terminal cost)."""
    m = D.shape[0]
    M = D.copy(); M[0, :] = 0.0; M[0, 0] = -mu; M[0, -1] = p * mg_gp(xs, n)
    Nmat = np.zeros((m, 1)); Nmat[0, 0] = 1.0
    Qz = np.zeros((m, m)); Qz[0, 0] = Q[0, 0]
    Pc = scipy.linalg.solve_continuous_are(M, Nmat, Qz, R)
    Kc = np.linalg.inv(R) @ Nmat.T @ Pc
    return Pc, Kc, M, Nmat


def solve_pontryagin(z0, D, Pc, A_cl, xs, mu, p, n, T, n_mesh=120):
    """Continuous-time Pontryagin TPBVP on the augmented state. Returns the solution object."""
    m = D.shape[0]
    Rs = R[0, 0]; zstar = xs * np.ones(m)
    DT = D.T

    def rhs(t, y):                                  # y = [z; lambda], shape (2m, n_mesh)
        z, lam = y[:m], y[m:]
        u = -0.5 * lam[0] / Rs                      # u* = -1/2 R^-1 B^T lambda_0
        Fz = D @ z
        Fz0 = -mu * z[0] + p * mg_g(z[-1], n) + u
        Fz = np.vstack([Fz0[None, :], Fz[1:]])
        # lambda' = -(dq/dz + J(z)^T lambda),  J = D with row0 replaced
        dq = np.zeros_like(z); dq[0] = 2.0 * Q[0, 0] * (z[0] - xs)
        JTlam = DT @ lam                            # then correct row-0 of J (= column-0 effect on J^T)
        corr = -D[0, :][:, None] * lam[0][None, :]  # (m, n_mesh): (J[0,:]-D[0,:])*lam_0 minus the g' part
        corr[0] += (-mu) * lam[0]
        corr[-1] += (p * mg_gp(z[-1], n)) * lam[0]
        JTlam = JTlam + corr
        lamdot = -(dq + JTlam)
        return np.vstack([Fz, lamdot])

    def bc(ya, yb):
        z0a, zTb, lamTb = ya[:m], yb[:m], yb[m:]
        return np.concatenate([z0a - z0, lamTb - 2.0 * Pc @ (zTb - zstar)])

    # warm start: linearised optimal trajectory dz(t)=expm(A_cl t)dz0, lambda=2 Pc dz
    tmesh = np.linspace(0.0, T, n_mesh)
    Phi = scipy.linalg.expm(A_cl * (tmesh[1] - tmesh[0]))
    dz = (z0 - zstar).copy(); Zg = [dz.copy()]
    for _ in range(n_mesh - 1):
        dz = Phi @ dz; Zg.append(dz.copy())
    Zg = np.array(Zg).T                              # (m, n_mesh)
    Yg = np.vstack([zstar[:, None] + Zg, 2.0 * Pc @ Zg])
    return solve_bvp(rhs, bc, tmesh, Yg, max_nodes=20000, tol=1e-6), zstar


def value_along(sol, m, xs, mu, p, n, T):
    """V*(z(t)) = remaining cost-to-go along the optimal trajectory + linearised terminal cost."""
    t = sol.x; z, lam = sol.y[:m], sol.y[m:]
    u = -0.5 * lam[0] / R[0, 0]
    run = Q[0, 0] * (z[0] - xs) ** 2 + R[0, 0] * u ** 2          # running cost q + u^T R u
    # cumulative integral from t to T (trapezoid), reversed
    dtv = np.diff(t)
    seg = 0.5 * (run[:-1] + run[1:]) * dtv
    tail = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])   # remaining running cost from each t
    return t, z, lam, u, tail


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--n-cheb", type=int, default=N_CHEB)
    ap.add_argument("--delay", type=float, default=REGIME["delay"], help="tau (6=limit cycle, 17=chaotic)")
    args = ap.parse_args()
    REGIME["delay"] = args.delay
    env = make_env(); dt = env.step_size; tau = float(env.max_delay)
    mu, p, n, xs = REGIME["mu"], REGIME["p"], REGIME["n"], REGIME["x_target"]
    n_cheb = args.n_cheb
    Dc, _ = cheb(n_cheb); D = (2.0 / tau) * Dc      # d/dtheta on [-tau,0] (NOT the raw [-1,1] matrix)
    m = n_cheb + 1
    Pc, Kc, M, Nmat = linear_oracle(D, mu, p, n, xs, tau)
    A_cl = M - Nmat @ Kc

    print(f"\nMACKEY-GLASS NONLINEAR (Pontryagin) oracle | tau={tau} N_cheb={n_cheb} dim z={m}  T={T_HOR}")
    z0 = X0_CONST * np.ones(m)
    sol, zstar = solve_pontryagin(z0, D, Pc, A_cl, xs, mu, p, n, T_HOR)
    print(f"solve_bvp: success={sol.success}  nodes={sol.x.size}  max_residual={np.max(sol.rms_residuals):.2e}")

    t, z, lam, u, tail = value_along(sol, m, xs, mu, p, n, T_HOR)
    dzT = z[:, -1] - zstar
    Vterm = float(dzT @ Pc @ dzT)
    Vnl = tail + Vterm                               # V*_nl(z(t)) cost-to-go
    dz = z - zstar[:, None]
    Vlin = np.einsum("it,ij,jt->t", dz, Pc, dz)      # linearised value dz^T Pc dz at the same states

    # (1) regulation: z0(t) -> x*
    print(f"\n(1) regulation: x(0)={z[0,0]:.4f} -> x(T)={z[0,-1]:.4f}  (target x*={xs}); "
          f"max|u*|={np.max(np.abs(u)):.3f} (clip {5.0})")

    # (2) cross-check the optimal control on the TRUE nonlinear env (constant-history IC)
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array([X0_CONST]), t0=0.0)
    tt, xx = 0.0, [X0_CONST]
    while tt < T_HOR - 1e-9:
        ut = float(np.interp(tt, t, u))
        tt, xnext, _ = w.step(w.state, jnp.array([ut])); xx.append(float(np.array(xnext).reshape(-1)[0]))
    x_env = np.array(xx); t_env = dt * np.arange(len(xx))
    x_bvp_on_env = np.interp(t_env, t, z[0])
    diff = np.abs(x_env - x_bvp_on_env)
    env_mismatch = float(np.max(diff)); t_at = float(t_env[np.argmax(diff)])
    pre = float(np.max(diff[t_env < tau])); post = float(np.max(diff[t_env >= tau]))
    print(f"(2) optimal u* on the REAL nonlinear env vs BVP trajectory: max|x_env - x_bvp|={env_mismatch:.3e} "
          f"at t={t_at:.2f}  (t<tau: {pre:.2e}, t>=tau: {post:.2e})")
    # free-response (u=0) model-vs-env, to isolate model fidelity from open-loop control fragility
    from scipy.integrate import solve_ivp
    def F_free(tt, zz):
        f = D @ zz; f0 = -mu * zz[0] + p * mg_g(zz[-1], n)
        return np.concatenate([[f0], f[1:]])
    sol_free = solve_ivp(F_free, [0, float(t_env[-1])], z0, t_eval=t_env, rtol=1e-8, atol=1e-10)
    w2 = JAXEnvWrapper(env, rng_key=42); w2.reset(jax.random.PRNGKey(0), x0=jnp.array([X0_CONST]), t0=0.0)
    tt2, xx2 = 0.0, [X0_CONST]
    while tt2 < T_HOR - 1e-9:
        tt2, xn2, _ = w2.step(w2.state, jnp.array([0.0])); xx2.append(float(np.array(xn2).reshape(-1)[0]))
    x_free_env = np.array(xx2); free_mismatch = float(np.max(np.abs(x_free_env - sol_free.y[0])))
    print(f"    free response (u=0) collocation model vs real env: max|x_env - x_model|={free_mismatch:.3e}")
    print(f"    trajectory dump (controlled):    t |   x_env | x_bvp | x_free_env | x_free_model")
    for tc in [0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 19.5]:
        ix = int(np.argmin(np.abs(t_env - tc)))
        print(f"      t={tc:5.1f}:  {x_env[ix]:7.4f}  {x_bvp_on_env[ix]:7.4f}    "
              f"{x_free_env[ix]:7.4f}     {sol_free.y[0][ix]:7.4f}")

    # (3) how NON-quadratic is V*?  (platoon was 99.9% quadratic -> H2 failed)
    ss_res = np.sum((Vnl - Vlin) ** 2); ss_tot = np.sum((Vnl - Vnl.mean()) ** 2)
    r2_lin = 1.0 - ss_res / (ss_tot + 1e-18)
    rel_nl = float(np.std(Vnl - Vlin) / (np.std(Vnl) + 1e-18))
    print(f"(3) non-quadraticity: R^2(linearised value -> V*_nl)={r2_lin:.5f}   "
          f"||V_nl - V_lin||/||V_nl||(RMS)={rel_nl:.4f}")
    print(f"    (platoon was R^2=0.9993 -> H2 FAILED; lower R^2 here = genuine nonlinearity = H2 signal)")

    # (4) GATE: costate(0) == dV*/dz0 by finite difference (re-solve BVP at perturbed IC)
    eps = 1e-4; fd = np.zeros(m); idx = [0, m // 2, m - 1]
    for j in idx:
        zp = z0.copy(); zp[j] += eps
        sp, _ = solve_pontryagin(zp, D, Pc, A_cl, xs, mu, p, n, T_HOR)
        tp, zp_, lamp, up, tailp = value_along(sp, m, xs, mu, p, n, T_HOR)
        Vp = tailp[0] + float((zp_[:, -1] - zstar) @ Pc @ (zp_[:, -1] - zstar))
        zm = z0.copy(); zm[j] -= eps
        smn, _ = solve_pontryagin(zm, D, Pc, A_cl, xs, mu, p, n, T_HOR)
        tm, zm_, lamm, um, tailm = value_along(smn, m, xs, mu, p, n, T_HOR)
        Vm = tailm[0] + float((zm_[:, -1] - zstar) @ Pc @ (zm_[:, -1] - zstar))
        fd[j] = (Vp - Vm) / (2 * eps)
    lam0 = lam[:, 0]
    gate = {j: (float(fd[j]), float(lam0[j]), abs(fd[j] - lam0[j]) / (abs(lam0[j]) + 1e-12)) for j in idx}
    print(f"\n(4) GATE costate(0) == dV*/dz0 (finite difference): component (FD, lambda0, rel.err)")
    for j in idx:
        print(f"    z[{j:>2}]: FD={gate[j][0]:+.5f}  lambda0={gate[j][1]:+.5f}  rel.err={gate[j][2]:.2e}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_mg_nonlinear_oracle"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(
        regime=REGIME, N_cheb=N_CHEB, T=T_HOR, success=bool(sol.success),
        r2_lin=float(r2_lin), rel_nl=rel_nl, env_mismatch=env_mismatch,
        gate={str(j): gate[j] for j in idx}), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
