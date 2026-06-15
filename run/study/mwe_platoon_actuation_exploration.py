"""Simple platoon: how do ACTUATION and EXPLORATION govern whether the value-gradient
history critic can identify its control gradient?

Mechanism under test. The value-gradient control reads dV/dx|_H (current state moved,
past frozen) -- a direction off the state-manifold that on-policy data does not
constrain. Exploration (action noise) is the only on-line source of off-manifold data,
but it injects through the actuators B, and the control itself only needs B^T dV/dx.
So two competing effects:
  - exploration covers off-manifold directions in ~range(B) (action noise -> state via B);
  - the control needs the gradient contracted by B^T.
Whether realistic exploration suffices, and how it scales with actuation (full vs
under) and noise sigma, is what this measures.

Simple platoon (no delay -> markovian-OPTIMAL, so V*(x)=-x^T P x is analytic and the
history critic's failure is purely the off-manifold/window effect = H0). N vehicles,
state per vehicle (gap error e_i, velocity error v_i):
    e_i' = v_{i-1} - v_i        (v_0 = 0 lead reference)
    v_i' = alpha (v_{i-1}-v_i) + u_i      (velocity coupling -> controllable from the lead)
B actuates the v-rows of the chosen vehicles. We fit the IDEAL raw_history critic
(target V*(W) = -W_now^T P W_now) on oracle-plus-noise rollouts (the exploration model),
then deploy the value-gradient control deterministically and measure cos(u,u*) and cost.

modes:
  baseline  : sigma=0 (on-manifold only)
  explore   : action-noise rollouts (realistic exploration, off-manifold via B)
  ideal_aug : full-state off-manifold excitation (perturb current state, all directions,
              past frozen) -- the upper bound, independent of actuation

Usage:
    uv run python run/study/mwe_platoon_actuation_exploration.py [--debug]
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

N_VEH = 3
ALPHA = 1.0
R_SCALAR = 0.1
WINDOW_LENGTH = 5
DT = 0.05
HORIZON = 6.0
N_IC = 8
N_NOISE = 2          # noise realisations per IC for the explore mode
AUG_PER = 4          # perturbations per window for ideal_aug
SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.3, 1.0]   # include small sigma -> under-exploration test
IC_SCALE = 1.0
KIND, DEG = "raw_history", 2
SEED = 0


def build_platoon(actuated):
    """Damped spring-mass chain (lead attached to a wall, free right end): a stable,
    single-input-CONTROLLABLE 'platoon'. Mass i has position p_i, velocity v_i;
        p_i' = v_i,
        v_i' = k (p_{i-1} - 2 p_i + p_{i+1}) - d v_i + u_i   (springs propagate the
    control down the chain, so it is controllable even from the lead alone)."""
    n = 2 * N_VEH
    k, d = 1.0, 0.5
    A = np.zeros((n, n))
    for i in range(N_VEH):
        p, v = 2 * i, 2 * i + 1
        A[p, v] = 1.0
        A[v, v] = -d
        left = -k if i == N_VEH - 1 else -2.0 * k   # wall/interior vs free right end
        A[v, p] += left
        if i > 0:
            A[v, 2 * (i - 1)] += k
        if i < N_VEH - 1:
            A[v, 2 * (i + 1)] += k
    B = np.zeros((n, len(actuated)))
    for col, j in enumerate(actuated):
        B[2 * j + 1, col] = 1.0
    Q = np.eye(n)
    R = R_SCALAR * np.eye(len(actuated))
    return A, B, Q, R


def lqr(A, B, Q, R):
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    return P, np.linalg.inv(R) @ B.T @ P


def rollout(A, B, control_fn, n_steps, x0, sigma=0.0, rng=None):
    n = A.shape[0]
    win = deque([x0.copy() for _ in range(WINDOW_LENGTH)], maxlen=WINDOW_LENGTH)
    x = x0.copy(); states = [x.copy()]; controls = []
    for _ in range(n_steps):
        u = np.asarray(control_fn(np.stack(list(win)))).reshape(-1)
        if sigma > 0 and rng is not None:
            u = u + sigma * rng.standard_normal(u.shape)
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


def quad_cost(states, controls, Q, R):
    cs = np.einsum("ti,ij,tj->t", states[:-1], Q, states[:-1])
    cu = np.einsum("ti,ij,tj->t", controls, R, controls)
    return float(np.sum((cs + cu) * DT))


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev[-1] / ev[0]), float(ev.sum() ** 2 / (ev ** 2).sum())


def evaluate(A, B, Q, R, P, K, fit_windows, dep_W, u_star, x0, n_steps):
    n = A.shape[0]
    half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
    rep = make_representation(kind=KIND, window_length=WINDOW_LENGTH, n_state=n, degree=DEG)
    feat = jax.jit(rep.feature_fn)
    V = -np.einsum("ti,ij,tj->t", fit_windows[:, -1], P, fit_windows[:, -1])
    Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in fit_windows])
    d = Phi.shape[1]; lam = 1e-8 * np.trace(Phi.T @ Phi) / d
    cond, erank = conditioning(Phi)
    theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ V)
    th = jnp.asarray(theta)
    grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    u_rep = np.stack([half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1] for w in dep_W])
    cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
    mag = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (np.mean(np.linalg.norm(u_star, axis=1)) + 1e-12))
    cl_s, cl_c = rollout(A, B, lambda w, grad=grad: half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1],
                         n_steps, x0)
    I_cl = quad_cost(cl_s, cl_c, Q, R)
    return dict(dim=d, cond=cond, erank=erank, cos=cos, mag=mag, I_cl=I_cl)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    n_steps = int(HORIZON / DT)
    rng = np.random.default_rng(SEED)
    n = 2 * N_VEH
    x0 = np.zeros(n); x0[1] = 0.5            # lead-velocity perturbation
    ic_bank = [x0] + [IC_SCALE * rng.standard_normal(n) for _ in range(N_IC)]

    ACTUATIONS = [("full", list(range(N_VEH))), ("under(lead)", [0])]

    print(f"\nsimple platoon actuation x exploration | N={N_VEH} state_dim={n} "
          f"win={WINDOW_LENGTH} | rep={KIND} d2")
    for name, act in ACTUATIONS:
        A, B, Q, R = build_platoon(act)
        try:
            P, K = lqr(A, B, Q, R)
        except Exception as e:
            print(f"\n=== actuation={name}: LQR failed ({e}) ==="); continue
        nc_s, nc_c = rollout(A, B, lambda w: np.zeros(B.shape[1]), n_steps, x0)
        dep_s, dep_c = rollout(A, B, lambda w: -K @ w[-1], n_steps, x0)
        I_nc = quad_cost(nc_s, nc_c, Q, R); I_or = quad_cost(dep_s, dep_c, Q, R)
        dep_W = full_windows(dep_s); u_star = np.stack([-K @ W[-1] for W in dep_W])
        onman = np.concatenate([full_windows(rollout(A, B, lambda w: -K @ w[-1], n_steps,
                                np.asarray(ic))[0]) for ic in ic_bank[:N_IC]], 0)

        print(f"\n=== actuation={name}  (n_ctrl={B.shape[1]}) | no-control I={I_nc:.4f} "
              f"oracle I*={I_or:.4f} ===")
        print(f"{'mode':<12}{'sigma':>7}{'n_pts':>8}{'cond':>9}{'erank':>7}"
              f"{'|u|/|u*|':>10}{'cosL2':>8}{'I_cl':>10}  verdict")

        def show(mode, sigma, fitW):
            m = evaluate(A, B, Q, R, P, K, fitW, dep_W, u_star, x0, n_steps)
            ok = (m['I_cl'] < I_nc) and (m['cos'] > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if m['I_cl'] >= I_nc else "weak")
            print(f"{mode:<12}{sigma:>7.2f}{len(fitW):>8}{m['cond']:>9.1e}{m['erank']:>7.2f}"
                  f"{m['mag']:>10.2f}{m['cos']:>8.3f}{m['I_cl']:>10.4f}  {v}")
            return dict(actuation=name, mode=mode, sigma=sigma, **m)

        rows = [show("baseline", 0.0, onman)]
        for sg in SIGMAS[1:]:
            # explore: realistic action-noise rollouts (off-manifold via B)
            ex = [onman]
            for ic in ic_bank[:N_IC]:
                for s in range(N_NOISE):
                    rg = np.random.default_rng(1000 + s + 7 * int(sg * 100))
                    ex.append(full_windows(rollout(A, B, lambda w: -K @ w[-1], n_steps,
                              np.asarray(ic), sigma=sg, rng=rg)[0]))
            rows.append(show("explore", sg, np.concatenate(ex, 0)))
            # ideal_aug: full-state off-manifold excitation (upper bound)
            aug = [onman]
            for _ in range(AUG_PER):
                Wp = onman.copy(); Wp[:, -1] = Wp[:, -1] + sg * rng.standard_normal(onman[:, -1].shape)
                aug.append(Wp)
            rows.append(show("ideal_aug", sg, np.concatenate(aug, 0)))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_actuation_exploration"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(N_VEH=N_VEH, sigmas=SIGMAS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
