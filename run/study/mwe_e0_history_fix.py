"""E0 / H0: FIX the history representation by off-manifold excitation.

raw_history (degree-2 monomials of the whole history window) fails the value-gradient
control on E0 not because it cannot represent the value (R^2=1) but because on-policy
data lives on the state-dimensional manifold M = im(L): the control needs the
derivative of V as the CURRENT STATE is moved with the PAST HELD FIXED, a direction
normal to M that no trajectory ever samples (see
documents/analysis/signature_value_gradient_conditioning/FINDINGS.md).

The fix follows from the diagnosis: add OFF-MANIFOLD training windows -- perturb the
current state of each on-trajectory window while freezing its history -- and label
them with the analytic value V*(W) = -W_now^T P W_now (on E0 the value depends only on
the current state, so any window, consistent or not, has a known target). This is the
state-space-excitation analogue the trajectory sweep could not provide (more
trajectories only fill M's TANGENT directions; this fills the NORMAL one).

We sweep the off-manifold perturbation radius eps. Prediction: eps=0 (on-manifold
only) leaves raw_history failing; any eps>0 constrains the normal derivative and snaps
the control to the oracle. markovian is the control: it depends only on the current
state, so the augmentation is already on its manifold and it is unaffected.

Usage:
    uv run python run/study/mwe_e0_history_fix.py [--debug]
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
N_IC = 16            # on-trajectory ICs (enough to cover the manifold's tangent)
AUG_PER = 4          # off-manifold perturbations added per on-trajectory window
EPS = [0.0, 0.05, 0.2, 0.5]
IC_SCALE = 1.5
REPS = [("markovian", 2), ("raw_history", 2)]
SEED = 0


def lqr():
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return P, np.linalg.inv(R) @ B.T @ P


def rollout(control_fn, n_steps, x0):
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


def full_windows(states):
    return np.stack([states[k - WINDOW_LENGTH + 1:k + 1]
                     for k in range(WINDOW_LENGTH - 1, states.shape[0])])


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

    ic_bank = [X0_DEPLOY] + [IC_SCALE * rng.standard_normal(N_STATE) for _ in range(N_IC)]
    onman = np.concatenate([full_windows(rollout(lambda w: -K @ w[-1], n_steps, np.asarray(x0))[0])
                            for x0 in ic_bank[:N_IC]], 0)               # (T, L, N) on-manifold

    dep_W = full_windows(rollout(lambda w: -K @ w[-1], n_steps, X0_DEPLOY)[0])
    u_star = np.stack([-K @ W[-1] for W in dep_W])
    nc_states, nc_controls = rollout(lambda w: np.zeros(B.shape[1]), n_steps, X0_DEPLOY)
    dep_states, dep_controls = rollout(lambda w: -K @ w[-1], n_steps, X0_DEPLOY)
    I_nc, I_oracle = quad_cost(nc_states, nc_controls), quad_cost(dep_states, dep_controls)

    def value_target(W):                       # V*(W) = -W_now^T P W_now  (any W)
        return -np.einsum("ti,ij,tj->t", W[:, -1], P, W[:, -1])

    print(f"\nE0 history fix via off-manifold excitation | win L={WINDOW_LENGTH} | n_ic={N_IC} "
          f"aug/window={AUG_PER}")
    print(f"reference bars : no-control I={I_nc:.4f} | oracle I*={I_oracle:.4f}")

    results = []
    for kind, deg in REPS:
        rep = make_representation(kind=kind, window_length=WINDOW_LENGTH, n_state=N_STATE,
                                  depth=deg, degree=deg)
        feat = jax.jit(rep.feature_fn)
        print(f"\n=== {kind} (dim {int(rep.feature_dim)}) ===")
        print(f"{'eps':>6}{'n_pts':>8}{'cond':>10}{'erank':>8}{'R2':>7}"
              f"{'|u|/|u*|':>10}{'cosL2':>8}{'I_cl':>11}  verdict")
        for eps in EPS:
            W = onman.copy()
            if eps > 0.0:
                aug = []
                for _ in range(AUG_PER):
                    Wp = onman.copy()
                    Wp[:, -1] = Wp[:, -1] + eps * rng.standard_normal(onman[:, -1].shape)
                    aug.append(Wp)
                W = np.concatenate([onman] + aug, 0)
            V = value_target(W)
            Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in W])
            d = Phi.shape[1]; lam = 1e-8 * np.trace(Phi.T @ Phi) / d
            cond, erank = conditioning(Phi)
            theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ V)
            r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
            th = jnp.asarray(theta)
            grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
            u_rep = np.stack([half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1] for w in dep_W])
            mag = float(np.mean(np.linalg.norm(u_rep, 1)) / (np.mean(np.linalg.norm(u_star, 1)) + 1e-12))
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            cl_s, cl_c = rollout(lambda w, grad=grad: half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1],
                                 n_steps, X0_DEPLOY)
            I_cl = quad_cost(cl_s, cl_c)
            ok = (I_cl < I_nc) and (cos > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{eps:>6.2f}{len(Phi):>8}{cond:>10.1e}{erank:>8.2f}{r2:>7.3f}"
                  f"{mag:>10.2f}{cos:>8.3f}{I_cl:>11.4f}  {v}")
            results.append(dict(kind=kind, eps=eps, n_pts=len(Phi), cond=cond, erank=erank,
                                r2=float(r2), mag=mag, cos=cos, I_cl=I_cl))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_history_fix"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(n_ic=N_IC, aug_per=AUG_PER, eps=EPS, I_nc=I_nc, I_oracle=I_oracle,
             results=results), indent=2, default=str))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
