"""Minimal working example: does the signature window-representation, by itself,
corrupt the value-gradient control law when history is DYNAMICALLY IRRELEVANT?

Hypothesis H0. On a plain linear-quadratic regulator with NO delay (markovian is
exactly optimal; the true value is the known quadratic V(x) = -x^T P x), the
signature still ingests a sliding WINDOW of the recent state path and maps it
through the Arribas time-augmentation + iterated integrals. H0 asks whether that
window-machinery alone -- with no non-Markovian effect to exploit and the TRUE
value handed to the critic -- produces a value-gradient control that is worse than
no control. If yes, the platoon failure is a property of how the signature encodes
the window, not of history-dependence.

Design (atomic; no learning loop, the critic is removed as a variable).
  E0  : 2-D double integrator, Q = I, R = r I. Continuous ARE gives P, K = R^-1 B^T P,
        the analytic value V(x) = -x^T P x (Doya reward-to-go sign) and oracle
        u*(x) = -K x. The codebase control law u = 1/2 R^-1 B^T dV/dx then reproduces
        u* EXACTLY for any representation that represents V with the correct gradient.
  IDEAL critic (value fit): ridge-fit theta to the TRUE value -x_t^T P x_t along the
        oracle rollout (value-fit R^2 ~ 1 by construction; the critic is not the variable).
  Representations: markovian deg-2 (positive control -- MUST pass, V is exactly its
        span), raw_history deg-2 (high-dim but non-redundant), signature depth-m.

Measurements / modes.
  M1  control magnitude  ||u_rep|| vs ||u*|| along the oracle path
  M2  direction          L^2(mu) cosine of u_rep vs u*
  M3  closed-loop cost   roll out each control; compare to no-control and oracle
  DERIVATIVE MATCHING    re-fit theta with an added term gamma * sum_t ||J_t^T theta - g_t||^2
        constraining the ENDPOINT gradient to the analytic g_t = dV/dx = -2 P x_t.
        If this rescues the signature, it proves the barrier is the unconstrained
        gradient of an L^2 value fit (matching V in L^2 does not match dV/dx).
  SWEEP                  repeat M1/M2/M3 over R, x0, window length, depth to show the
        failure is robust, not a single-config artefact.

Usage:
    uv run python run/study/mwe_history_representation_control.py [--debug]
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


# ----------------------------------------------------------------- fixed E0 pieces
A = np.array([[0.0, 1.0], [0.0, 0.0]])          # double integrator
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
N_STATE = 2
DT = 0.05
HORIZON = 6.0


def lqr(Rm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = scipy.linalg.solve_continuous_are(A, B, Q, Rm)
    K = np.linalg.inv(Rm) @ B.T @ P
    return P, K


def rollout(control_fn, n_steps, x0, window_length):
    win = deque([x0.copy() for _ in range(window_length)], maxlen=window_length)
    x = x0.copy()
    states, controls = [x.copy()], []
    for _ in range(n_steps):
        window = np.stack(list(win))
        u = np.asarray(control_fn(window)).reshape(-1)
        if not np.all(np.isfinite(u)) or np.linalg.norm(x) > 1e6:
            for _ in range(n_steps - len(controls)):
                controls.append(u if np.all(np.isfinite(u)) else np.zeros(B.shape[1]))
                states.append(states[-1])
            break
        d = lambda xx: A @ xx + (B @ u).reshape(-1)
        k1 = d(x); k2 = d(x + 0.5 * DT * k1); k3 = d(x + 0.5 * DT * k2); k4 = d(x + DT * k3)
        x = x + DT / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        win.append(x.copy()); states.append(x.copy()); controls.append(u.copy())
    return np.stack(states), np.stack(controls)


def windows_along(states, window_length):
    T = states.shape[0]
    out = np.empty((T, window_length, N_STATE))
    for k in range(T):
        lo = k - window_length + 1
        if lo >= 0:
            out[k] = states[lo:k + 1]
        else:
            pad = np.repeat(states[:1], -lo, axis=0)
            out[k] = np.concatenate([pad, states[:k + 1]], axis=0)
    return out


def quad_cost(states, controls, Rm):
    cs = np.einsum("ti,ij,tj->t", states[:-1], Q, states[:-1])
    cu = np.einsum("ti,ij,tj->t", controls, Rm, controls)
    return float(np.sum((cs + cu) * DT))


def evaluate(kind, depth, R_scalar, x0, window_length, gamma=0.0):
    """Fit the ideal critic for one representation and return M1/M2/M3. If gamma>0,
    add the analytic endpoint-gradient (derivative-matching) term to the fit."""
    Rm = np.array([[R_scalar]])
    Rinv = np.linalg.inv(Rm)
    P, K = lqr(Rm)
    n_steps = int(HORIZON / DT)

    oracle_states, oracle_controls = rollout(lambda w: -K @ w[-1], n_steps, x0, window_length)
    nc_states, nc_controls = rollout(lambda w: np.zeros(B.shape[1]), n_steps, x0, window_length)
    I_oracle = quad_cost(oracle_states, oracle_controls, Rm)
    I_nc = quad_cost(nc_states, nc_controls, Rm)

    mu_w = windows_along(oracle_states[:-1], window_length)
    u_star = (-(K @ oracle_states[:-1].T)).T

    rep = make_representation(kind=kind, window_length=window_length, n_state=N_STATE,
                              depth=depth, degree=depth)
    feat_fn = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat_fn(jnp.asarray(w))) for w in mu_w])           # (T,d)
    V_target = -np.einsum("ti,ij,tj->t", oracle_states[:-1], P, oracle_states[:-1])
    d = Phi.shape[1]
    G = Phi.T @ Phi
    rhs = Phi.T @ V_target
    lam = 1e-8 * np.trace(G) / d

    if gamma > 0.0:
        jac_fn = jax.jit(jax.jacfwd(rep.feature_fn))                              # (d,L,n)
        J = np.stack([np.asarray(jac_fn(jnp.asarray(w)))[:, -1, :] for w in mu_w])  # (T,d,n)
        g = -2.0 * (P @ oracle_states[:-1].T).T                                   # (T,n) analytic dV/dx
        G = G + gamma * np.einsum("tdi,tei->de", J, J)
        rhs = rhs + gamma * np.einsum("tdi,ti->d", J, g)

    theta = np.linalg.solve(G + lam * np.eye(d), rhs)
    V_hat = Phi @ theta
    r2 = 1.0 - np.sum((V_target - V_hat) ** 2) / np.sum((V_target - V_target.mean()) ** 2)

    theta_j = jnp.asarray(theta)
    end_grad = jax.jit(lambda w: jax.grad(lambda ww: jnp.dot(theta_j, feat_fn(ww)))(w)[-1])
    u_rep_fn = lambda w: 0.5 * Rinv @ B.T @ np.asarray(end_grad(jnp.asarray(w)))

    u_rep = np.stack([u_rep_fn(w) for w in mu_w])
    mag = float(np.mean(np.linalg.norm(u_rep, axis=1)) /
                (np.mean(np.linalg.norm(u_star, axis=1)) + 1e-12))
    cos = float(np.sum(u_rep * u_star) /
                (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
    cl_states, cl_controls = rollout(u_rep_fn, n_steps, x0, window_length)
    I_cl = quad_cost(cl_states, cl_controls, Rm)
    return dict(kind=kind, dim=d, r2=float(r2), mag=mag, cos=cos,
                I_cl=I_cl, I_nc=I_nc, I_oracle=I_oracle, beats_nc=bool(I_cl < I_nc))


def verdict(r):
    if r["beats_nc"] and r["cos"] > 0.9:
        return "OK"
    return "WORSE-than-nc" if not r["beats_nc"] else "weak"


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'rep':<12}{'dim':>6}{'R2':>8}{'|u|/|u*|':>10}{'cosL2':>9}{'I_cl':>11}{'I_nc':>9}  verdict")
    for r in rows:
        print(f"{r['kind']:<12}{r['dim']:>6}{r['r2']:>8.3f}{r['mag']:>10.2f}"
              f"{r['cos']:>9.3f}{r['I_cl']:>11.4f}{r['I_nc']:>9.4f}  {verdict(r)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    base = dict(R_scalar=0.1, x0=np.array([1.0, 0.0]), window_length=9)
    all_results = {}

    # --- baseline: standard value fit (gamma=0) -----------------------------------
    base_rows = [evaluate("markovian", 2, **base),
                 evaluate("raw_history", 2, **base),
                 evaluate("signature", 2, **base)]
    print_table("[A] baseline -- standard L^2 value fit", base_rows)
    all_results["baseline"] = base_rows

    # --- derivative matching: constrain dV/dx to the analytic gradient -------------
    dm_rows = []
    for g in (1.0, 100.0):
        for kind, depth in (("raw_history", 2), ("signature", 2)):
            r = evaluate(kind, depth, gamma=g, **base)
            r["kind"] = f"{kind} g={g:g}"
            dm_rows.append(r)
    print_table("[B] derivative-matching fit (constrain dV/dx = -2Px)", dm_rows)
    all_results["derivative_matching"] = dm_rows

    # --- robustness sweep of the standard-fit FAILURE (signature) -----------------
    sweep = []
    for Rv in (0.01, 0.1, 1.0):
        sweep.append(("R", Rv, evaluate("signature", 2, R_scalar=Rv,
                      x0=base["x0"], window_length=base["window_length"])))
    for x0 in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, -1.0])):
        sweep.append(("x0", tuple(x0), evaluate("signature", 2, R_scalar=base["R_scalar"],
                      x0=x0, window_length=base["window_length"])))
    for L in (5, 9, 17):
        sweep.append(("L", L, evaluate("signature", 2, R_scalar=base["R_scalar"],
                      x0=base["x0"], window_length=L)))
    for dp in (2, 3):
        sweep.append(("depth", dp, evaluate("signature", dp, R_scalar=base["R_scalar"],
                      x0=base["x0"], window_length=base["window_length"])))
    print("\n[C] robustness sweep -- signature, standard fit (markovian passes everywhere)")
    print(f"{'axis':<8}{'value':<12}{'dim':>6}{'R2':>8}{'|u|/|u*|':>10}{'cosL2':>9}{'I_cl':>11}{'I_nc':>9}  verdict")
    for axis, val, r in sweep:
        print(f"{axis:<8}{str(val):<12}{r['dim']:>6}{r['r2']:>8.3f}{r['mag']:>10.2f}"
              f"{r['cos']:>9.3f}{r['I_cl']:>11.4f}{r['I_nc']:>9.4f}  {verdict(r)}")
    all_results["sweep"] = [dict(axis=a, value=str(v), **r) for a, v, r in sweep]

    debug_prefix = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = script_data_dir(__file__) / f"{debug_prefix}{ts}_double_integrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
