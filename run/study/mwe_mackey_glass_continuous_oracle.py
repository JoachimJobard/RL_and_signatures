"""A CONTINUOUS-TIME delayed-LQR oracle for the Mackey-Glass limit-cycle cell.

The env config (conf/env/MG_1D_limit_cycle.yaml) states "there is no analytic oracle for
this nonlinear plant". This builds one, the convention-clean way: keep TIME continuous and
discretise only the HISTORY by Chebyshev collocation (as for the platoon,
mwe_continuous_delayed_oracle.py), so Doya's continuous control u = 1/2 R^-1 B^T dV/dx is
EXACT -- no dt, no magnitude rescale.

Plant:  dx/dt = -mu x + p x(t-tau)/(1 + x(t-tau)^n) + u,  regulated to x* = (p/mu-1)^{1/n}=1
(the delay-destabilised equilibrium the limit cycle orbits). Linearising about x*:
  A = -mu = -0.1,  A1 = f'(x*) = p(2-n)/4 = -0.4,  B = 1.
Collocating the history generator on Chebyshev nodes gives a CONTINUOUS-TIME LTI
z' = M z + N u; solve the CONTINUOUS ARE -> P_c, V(dz) = -dz^T P_c dz (dz = z - x*),
u* = -R^-1 N^T P_c dz. This is the LINEARISED-value oracle (the H1 reference; the
nonlinear-value/Pontryagin layer for H2 is a separate, later step).

ACCEPTANCE GATE (the whole point -- no magnitude issue): Doya's formula applied to V must
reproduce u* with cos = 1.000 AND |u|/|u*| = 1.000. We also confirm the oracle actually
suppresses the nonlinear limit cycle closed-loop, and that K_c converges in N_cheb.

Usage:
    uv run python run/study/mwe_mackey_glass_continuous_oracle.py [--debug]
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

# MG_1D_limit_cycle regime (verified in conf/env/MG_1D_limit_cycle.yaml)
REGIME = dict(delay=6.0, mu=0.1, n=10, p=0.2, x_target=1.0, step_size=0.25, resolution=5)
Q = np.array([[1.0]])
R = np.array([[0.1]])
TF = 40.0            # ~10 limit-cycle periods (eval T_sim in the cell config)
N_CHEB_SWEEP = [10, 14, 18, 22]
CLIP = 5.0           # clip_action in the cell config
X0 = 0.8             # eval x0_test in the cell config


def make_env():
    from src.envs.mackey_glass_1D import MackeyGlass1DEnv
    return MackeyGlass1DEnv(delay=REGIME["delay"], step_size=REGIME["step_size"],
                            resolution=REGIME["resolution"], Q=Q, R=R, n=REGIME["n"],
                            p=REGIME["p"], mu=REGIME["mu"], x_target=REGIME["x_target"])


def linearise():
    """Jacobian of the MG RHS about x*: A = -mu, A1 = f'(x*), B = 1 (scalar state)."""
    mu, p, n = REGIME["mu"], REGIME["p"], REGIME["n"]
    xs = REGIME["x_target"]
    # f(y) = p y/(1+y^n);  f'(y) = p (1 + (1-n) y^n)/(1+y^n)^2
    fp = p * (1.0 + (1.0 - n) * xs ** n) / (1.0 + xs ** n) ** 2
    A = np.array([[-mu]]); A1 = np.array([[fp]]); B = np.array([[1.0]])
    return A, A1, B, xs


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T; dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1)); D = D - np.diag(D.sum(axis=1))
    return D, x


def build_continuous_oracle(A, A1, B, Q, R, tau, N):
    n, m = A.shape[0], B.shape[1]
    Dc, xc = cheb(N)
    theta = (tau / 2.0) * (xc - 1.0)          # theta_0=0, theta_N=-tau
    D = (2.0 / tau) * Dc                       # d/dtheta on [-tau, 0]
    big = n * (N + 1)
    M = np.zeros((big, big))
    for i in range(1, N + 1):                  # transport rows
        for j in range(N + 1):
            M[i * n:(i + 1) * n, j * n:(j + 1) * n] = D[i, j] * np.eye(n)
    M[0:n, 0:n] = A                            # boundary (theta=0) = the dynamics
    M[0:n, N * n:(N + 1) * n] = A1
    Nmat = np.zeros((big, m)); Nmat[0:n, :] = B
    Qz = np.zeros((big, big)); Qz[0:n, 0:n] = Q
    Pc = scipy.linalg.solve_continuous_are(M, Nmat, Qz, R)
    Kc = np.linalg.inv(R) @ Nmat.T @ Pc
    return dict(M=M, Nmat=Nmat, Pc=Pc, Kc=Kc, theta=theta, n=n, m=m, N=N)


def verify_value_independently(oc, Q, R, h=0.01, T=120.0, n_samples=6, seed=0):
    """Cross-check V*(dz) = dz^T Pc dz WITHOUT the Riccati equation.

    The cost-to-go of the LQR-optimal closed loop dz' = (M - N Kc) dz, with running cost
    dz^T(Qz + Kc^T R Kc)dz, equals dz_0^T Pc dz_0 by definition. We (a) integrate that cost
    numerically forward in time (independent of solve_continuous_are), and (b) solve the
    closed-loop Lyapunov equation (a different routine), and compare both to the Riccati Pc.
    """
    M, Nmat, Pc, Kc = oc["M"], oc["Nmat"], oc["Pc"], oc["Kc"]
    big = M.shape[0]
    Qz = np.zeros((big, big)); Qz[0:1, 0:1] = Q
    A_cl = M - Nmat @ Kc
    S = Qz + Kc.T @ R @ Kc                      # running cost matrix in dz
    # (b) Lyapunov: A_cl^T P + P A_cl + S = 0  ->  P must equal Pc
    P_lyap = scipy.linalg.solve_continuous_lyapunov(A_cl.T, -S)
    lyap_relerr = float(np.linalg.norm(P_lyap - Pc) / np.linalg.norm(Pc))
    # (a) direct forward integration of the cost-to-go for random dz0
    Phi = scipy.linalg.expm(A_cl * h)           # exact propagator over step h (not a Riccati/Lyapunov solve)
    nsteps = int(T / h)
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n_samples):
        dz0 = 0.2 * rng.standard_normal(big)
        V_ric = float(dz0 @ Pc @ dz0)
        dz = dz0.copy(); V_sim = 0.0
        for _ in range(nsteps):
            V_sim += 0.5 * float(dz @ S @ dz) * h      # trapezoid
            dz = Phi @ dz
            V_sim += 0.5 * float(dz @ S @ dz) * h
        rows.append((V_ric, V_sim, abs(V_sim - V_ric) / (abs(V_ric) + 1e-12)))
    return lyap_relerr, rows


def z_from_history(o_win, dt, theta, n):
    """Interpolate control-cadence history o_win (newest first) onto Chebyshev theta nodes."""
    k = len(o_win) - 1
    t_grid = -dt * np.arange(k + 1)            # [0, -dt, ..., -k dt], newest first
    ti = t_grid[::-1]; vi = np.array(o_win)[::-1]
    z = np.stack([np.interp(theta, ti, vi[:, c]) for c in range(n)], axis=1)
    return z.reshape(-1)


def closed_loop(env, control_owin, k, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array([x0]), t0=0.0)
    o_win = [np.array([x0]) for _ in range(k + 1)]; I, t = 0.0, 0.0
    xs = []
    while t < TF - 1e-9:
        u = np.clip(control_owin(o_win), -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf"), np.array(xs)
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * env.step_size; o_win = [x] + o_win[:-1]; xs.append(float(x[0]))
    return I, np.array(xs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); dt = env.step_size; tau = float(env.max_delay)
    A, A1, B, xs = linearise()
    k = int(round(tau / dt))                   # control-cadence history taps spanning tau
    Rinv = np.linalg.inv(R)

    print(f"\nMACKEY-GLASS continuous-time delayed-LQR oracle | tau={tau} dt={dt} x*={xs}")
    print(f"linearisation about x*: A={A[0,0]:.3f}  A1={A1[0,0]:.3f}  B={B[0,0]:.3f}  "
          f"(history taps k={k})")

    # no-control reference (the uncontrolled limit cycle)
    I_nc, x_nc = closed_loop(env, lambda ow: np.zeros(1), k, X0)

    print(f"\n{'N_cheb':>7}{'|Kc|':>10}{'maxRe(eig)':>12}{'gate cos':>10}{'gate |u|/|u*|':>15}{'I_cl':>10}")
    results = []
    rng = np.random.default_rng(0)
    for N in N_CHEB_SWEEP:
        oc = build_continuous_oracle(A, A1, B, Q, R, tau, N)
        Pc, Kc, theta = oc["Pc"], oc["Kc"], oc["theta"]
        eig = np.linalg.eigvals(oc["M"] - oc["Nmat"] @ Kc)
        # magnitude gate: Doya's formula vs the continuous oracle on a history cloud about x*
        cos_l, mag_l = [], []
        for _ in range(200):
            ow = [xs + 0.15 * rng.standard_normal(1) for _ in range(k + 1)]
            dz = z_from_history(ow, dt, theta, 1) - xs
            u_oracle = -Kc @ dz
            u_doya = -Rinv @ B.T @ (Pc @ dz)[:1]      # 1/2 R^-1 B^T dV/dx, V=-dz^T Pc dz
            cos_l.append(float(u_oracle @ u_doya / (np.linalg.norm(u_oracle) * np.linalg.norm(u_doya) + 1e-12)))
            mag_l.append(float(np.linalg.norm(u_doya) / (np.linalg.norm(u_oracle) + 1e-12)))
        ctrl = lambda ow, Kc=Kc, theta=theta: -Kc @ (z_from_history(ow, dt, theta, 1) - xs)
        I_cl, _ = closed_loop(env, ctrl, k, X0)
        results.append(dict(N=N, Knorm=float(np.linalg.norm(Kc)), maxre=float(eig.real.max()),
                            cos=float(np.mean(cos_l)), mag=float(np.mean(mag_l)), I=I_cl))
        print(f"{N:>7}{np.linalg.norm(Kc):>10.4f}{eig.real.max():>12.4f}"
              f"{np.mean(cos_l):>10.4f}{np.mean(mag_l):>15.4f}{I_cl:>10.4f}")

    # INDEPENDENT value check: cross-validate Pc (= V*) without the Riccati equation
    oc18 = build_continuous_oracle(A, A1, B, Q, R, tau, 18)
    lyap_relerr, rows = verify_value_independently(oc18, Q, R)
    print(f"\nINDEPENDENT value check (N=18) -- V*(dz)=dz^T Pc dz vs forward cost integral:")
    print(f"{'V_riccati':>14}{'V_simulated':>14}{'rel.err':>12}")
    for vr, vs, re in rows:
        print(f"{vr:>14.6f}{vs:>14.6f}{re:>12.2e}")
    print(f"closed-loop Lyapunov cross-check ||P_lyap - Pc||/||Pc|| = {lyap_relerr:.2e}")

    print(f"\nreference: no-control (uncontrolled limit cycle) I={I_nc:.4f}")
    print(f"          uncontrolled x range over the run: [{x_nc.min():.3f}, {x_nc.max():.3f}]  (expect ~[0.83,1.14])")
    best = min(results, key=lambda r: r["I"])
    print(f"\nGATE: cos={best['cos']:.4f}  |u|/|u*|={best['mag']:.4f}  (BOTH must be 1.000 -- no magnitude issue)")
    print(f"continuous oracle suppresses the cycle: I={best['I']:.4f} vs no-control {I_nc:.4f} "
          f"(reduction {100*(1-best['I']/I_nc):.1f}%)")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_mg_continuous_oracle"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, I_nc=I_nc, sweep=results), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
