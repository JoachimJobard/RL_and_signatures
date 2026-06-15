"""E0: does increasing the number of (oracle) trajectories past the number of
features fix the value-gradient control? (no platoon, analytic target.)

E0 = 2-D double-integrator LQR, no delay; analytic value V*(x) = -x^T P x is the
fit target (so there is NO discounting / return-to-go confound, unlike the platoon).
We pool oracle rollouts from n_ic initial conditions sampled over the plane, fit the
ideal linear critic theta on the pooled (window, V*) data, then deploy the
value-gradient control u = 1/2 R^-1 B^T dV/dx(t) closed-loop from a canonical x0 and
measure direction (cos to u*), magnitude, and closed-loop cost.

The structural question. The oracle flow is autonomous + deterministic, so the
history is a function of the current state: every window lives on a 2-D manifold
(= state dim), regardless of n_ic or window length. The control needs the gradient
w.r.t. the window ENDPOINT TAP, which generically has a component NORMAL to that
manifold. So either (a) the high-dim window features become full-rank on the 2-D
manifold and the gradient is identified (control restored as n_ic grows), or (b) they
stay rank-deficient and adding trajectories never fixes it. This script decides it.

Usage:
    uv run python run/study/mwe_e0_trajectory_sweep.py [--debug]
"""
from __future__ import annotations

import argparse
import json
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

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
R = np.array([[0.1]])
N_STATE = 2
WINDOW_LENGTH = 9
DT = 0.05
HORIZON = 6.0
X0_DEPLOY = np.array([1.0, 0.0])
N_ICS = [1, 4, 16, 64, 256]
IC_SCALE = 1.5
REPS = [("markovian", 2), ("signature", 2), ("raw_history", 2)]
SEED = 0


def lqr():
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return P, K


def rollout(control_fn, n_steps, x0):
    from collections import deque
    win = deque([x0.copy() for _ in range(WINDOW_LENGTH)], maxlen=WINDOW_LENGTH)
    x = x0.copy(); states = [x.copy()]; controls = []
    for _ in range(n_steps):
        u = np.asarray(control_fn(np.stack(list(win)))).reshape(-1)
        if not np.all(np.isfinite(u)) or np.linalg.norm(x) > 1e6:
            for _ in range(n_steps - len(controls)):
                controls.append(np.zeros(B.shape[1])); states.append(states[-1])
            break
        d = lambda xx: A @ xx + (B @ u).reshape(-1)
        k1 = d(x); k2 = d(x + .5 * DT * k1); k3 = d(x + .5 * DT * k2); k4 = d(x + DT * k3)
        x = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        win.append(x.copy()); states.append(x.copy()); controls.append(u.copy())
    return np.stack(states), np.stack(controls)


def windows_along(states):
    T = states.shape[0]; out = np.empty((T, WINDOW_LENGTH, N_STATE))
    for k in range(T):
        lo = k - WINDOW_LENGTH + 1
        out[k] = states[lo:k + 1] if lo >= 0 else np.concatenate(
            [np.repeat(states[:1], -lo, axis=0), states[:k + 1]], axis=0)
    return out


def quad_cost(states, controls):
    cs = np.einsum("ti,ij,tj->t", states[:-1], Q, states[:-1])
    cu = np.einsum("ti,ij,tj->t", controls, R, controls)
    return float(np.sum((cs + cu) * DT))


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev[-1] / ev[0]), float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    P, K = lqr(); Rinv = np.linalg.inv(R); n_steps = int(HORIZON / DT)
    half_RinvBT = 0.5 * Rinv @ B.T
    rng = np.random.default_rng(SEED)

    # deployment (canonical) oracle trajectory: where control is queried.
    dep_states, dep_controls = rollout(lambda w: -K @ w[-1], n_steps, X0_DEPLOY)
    I_oracle = quad_cost(dep_states, dep_controls)
    nc_states, nc_controls = rollout(lambda w: np.zeros(B.shape[1]), n_steps, X0_DEPLOY)
    I_nc = quad_cost(nc_states, nc_controls)
    dep_w = windows_along(dep_states[:-1])
    u_star = (-(K @ dep_states[:-1].T)).T

    # a fixed bank of ICs; n_ic uses the first n.
    ic_bank = [X0_DEPLOY] + [IC_SCALE * rng.standard_normal(N_STATE) for _ in range(max(N_ICS))]

    print(f"\nE0 trajectory sweep | double integrator | win L={WINDOW_LENGTH} dt={DT} T={HORIZON}")
    print(f"reference bars : no-control I={I_nc:.4f} | oracle I*={I_oracle:.4f}")

    results = []
    for kind, deg in REPS:
        rep = make_representation(kind=kind, window_length=WINDOW_LENGTH, n_state=N_STATE,
                                  depth=deg, degree=deg)
        feat = jax.jit(rep.feature_fn)
        print(f"\n=== {kind} (dim {int(rep.feature_dim)}) ===")
        print(f"{'n_ic':>6}{'n_pts':>8}{'cond':>10}{'erank':>8}{'R2':>7}"
              f"{'|u|/|u*|':>10}{'cosL2':>8}{'I_cl':>11}  verdict")
        for n_ic in N_ICS:
            W_list, V_list = [], []
            for x0 in ic_bank[:n_ic]:
                xs, _ = rollout(lambda w: -K @ w[-1], n_steps, np.asarray(x0))
                W_list.append(windows_along(xs[:-1]))
                V_list.append(-np.einsum("ti,ij,tj->t", xs[:-1], P, xs[:-1]))
            Wf = np.concatenate(W_list, 0); Vf = np.concatenate(V_list)
            Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wf])
            d = Phi.shape[1]; lam = 1e-8 * np.trace(Phi.T @ Phi) / d
            cond, erank = conditioning(Phi)
            theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ Vf)
            r2 = 1 - np.sum((Vf - Phi @ theta) ** 2) / (np.sum((Vf - Vf.mean()) ** 2) + 1e-12)
            th = jnp.asarray(theta)
            grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
            u_rep = np.stack([half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1] for w in dep_w])
            mag = float(np.mean(np.linalg.norm(u_rep, 1)) / (np.mean(np.linalg.norm(u_star, 1)) + 1e-12))
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            cl_s, cl_c = rollout(lambda w, th=th, feat=feat, grad=grad:
                                 half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1], n_steps, X0_DEPLOY)
            I_cl = quad_cost(cl_s, cl_c)
            ok = (I_cl < I_nc) and (cos > 0.9)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{n_ic:>6}{len(Phi):>8}{cond:>10.1e}{erank:>8.2f}{r2:>7.3f}"
                  f"{mag:>10.2f}{cos:>8.3f}{I_cl:>11.4f}  {v}")
            results.append(dict(kind=kind, dim=d, n_ic=n_ic, n_pts=len(Phi),
                                cond=cond, erank=erank, r2=float(r2), mag=mag, cos=cos, I_cl=I_cl))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_e0_sweep"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(I_nc=I_nc, I_oracle=I_oracle, n_ics=N_ICS, results=results), indent=2, default=str))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
