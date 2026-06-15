"""Simple platoon: is the bottleneck the VALUE LABEL, not actuation/exploration?

The actuation x exploration study showed: given the analytic oracle value V*(W) =
-W_now^T P W_now as the label at each (exploration-generated, off-manifold) window,
the value-gradient history critic recovers the oracle control under FULL or UNDER
actuation, with even sigma=0.02. So the off-manifold GEOMETRY is solved by exploration.
But real RL has no oracle value -- it labels states with the realized Monte-Carlo
return-to-go of the (noisy) behaviour policy. This script keeps the SAME off-manifold
states and swaps only the LABEL:

  oracle_label : V*(W) = -W_now^T P_gamma W_now           (analytic, clean)
  rtg_label    : discounted return-to-go along the noisy rollout (realistic, noisy,
                 single-sample, = value of the behaviour policy, not V*)

Discounted setting (rho=1/tau) so the two targets are the same object in expectation:
P_gamma solves the discounted ARE (A - rho/2 I, B, Q, R); the greedy control of
-x^T P_gamma x is u* = -K_gamma x; return-to-go converges to -x^T P_gamma x in
expectation. The hypothesis: oracle_label works (geometry solved), rtg_label degrades
(label noise corrupts the high-dim critic's gradient) -- relocating the bottleneck from
actuation/exploration to value-label quality.

Usage:
    uv run python run/study/mwe_platoon_value_target.py [--debug]
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
R_SCALAR = 0.1
WINDOW_LENGTH = 5
DT = 0.05
HORIZON = 8.0
TAU = 2.0            # discount time-constant (rho = 1/TAU); short tail at HORIZON
N_IC = 6
N_NOISE = 2
SIGMAS = [0.1, 0.3]
IC_SCALE = 1.0
KIND, DEG = "raw_history", 2
SEED = 0


def build_platoon(actuated):
    n = 2 * N_VEH; k, d = 1.0, 0.5
    A = np.zeros((n, n))
    for i in range(N_VEH):
        p, v = 2 * i, 2 * i + 1
        A[p, v] = 1.0; A[v, v] = -d
        A[v, p] += -k if i == N_VEH - 1 else -2.0 * k
        if i > 0:
            A[v, 2 * (i - 1)] += k
        if i < N_VEH - 1:
            A[v, 2 * (i + 1)] += k
    B = np.zeros((n, len(actuated)))
    for col, j in enumerate(actuated):
        B[2 * j + 1, col] = 1.0
    return A, B, np.eye(n), R_SCALAR * np.eye(len(actuated))


def disc_lqr(A, B, Q, R):
    rho = 1.0 / TAU
    P = scipy.linalg.solve_continuous_are(A - 0.5 * rho * np.eye(A.shape[0]), B, Q, R)
    return P, np.linalg.inv(R) @ B.T @ P


def rollout(A, B, control_fn, n_steps, x0, sigma=0.0, rng=None):
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


def windows_and_labels(states, controls, P, Q, R):
    gamma = np.exp(-DT / TAU); T = len(controls)
    r = -(np.einsum("ti,ij,tj->t", states[:T], Q, states[:T])
          + np.einsum("ti,ij,tj->t", controls, R, controls))
    rtg = np.zeros(T); acc = 0.0
    for t in range(T - 1, -1, -1):
        acc = r[t] * DT + gamma * acc; rtg[t] = acc
    Ws, Vo, Vr = [], [], []
    for k in range(WINDOW_LENGTH - 1, T):
        Ws.append(states[k - WINDOW_LENGTH + 1:k + 1])
        Vo.append(-states[k] @ P @ states[k]); Vr.append(rtg[k])
    return np.array(Ws), np.array(Vo), np.array(Vr)


def quad_cost(states, controls, Q, R):
    cs = np.einsum("ti,ij,tj->t", states[:-1], Q, states[:-1])
    cu = np.einsum("ti,ij,tj->t", controls, R, controls)
    return float(np.sum((cs + cu) * DT))


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def fit_eval(A, B, Q, R, fitW, fitV, dep_W, u_star, x0, n_steps, half_RinvBT, n):
    rep = make_representation(kind=KIND, window_length=WINDOW_LENGTH, n_state=n, degree=DEG)
    feat = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in fitW])
    d = Phi.shape[1]; lam = 1e-8 * np.trace(Phi.T @ Phi) / d
    theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ fitV)
    r2 = 1 - np.sum((fitV - Phi @ theta) ** 2) / (np.sum((fitV - fitV.mean()) ** 2) + 1e-12)
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    u_rep = np.stack([half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1] for w in dep_W])
    cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
    cl_s, cl_c = rollout(A, B, lambda w, grad=grad: half_RinvBT @ np.array(grad(jnp.asarray(w)))[-1],
                         n_steps, x0)
    return dict(r2=float(r2), cos=cos, I_cl=quad_cost(cl_s, cl_c, Q, R))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    n_steps = int(HORIZON / DT); rng = np.random.default_rng(SEED)
    n = 2 * N_VEH
    x0 = np.zeros(n); x0[1] = 0.5
    ics = [x0] + [IC_SCALE * rng.standard_normal(n) for _ in range(N_IC)]

    print(f"\nsimple platoon: VALUE-TARGET quality | N={N_VEH} state={n} win={WINDOW_LENGTH} "
          f"tau={TAU} | rep={KIND} d2")
    for name, act in [("full", list(range(N_VEH))), ("under(lead)", [0])]:
        A, B, Q, R = build_platoon(act); P, K = disc_lqr(A, B, Q, R)
        half_RinvBT = 0.5 * np.linalg.inv(R) @ B.T
        nc = rollout(A, B, lambda w: np.zeros(B.shape[1]), n_steps, x0)
        dep = rollout(A, B, lambda w: -K @ w[-1], n_steps, x0)
        I_nc, I_or = quad_cost(*nc, Q, R), quad_cost(*dep, Q, R)
        dep_W = np.array([dep[0][k - WINDOW_LENGTH + 1:k + 1]
                          for k in range(WINDOW_LENGTH - 1, len(dep[0]) - 1)])
        u_star = np.stack([-K @ W[-1] for W in dep_W])

        print(f"\n=== actuation={name} (n_ctrl={B.shape[1]}) | no-control I={I_nc:.4f} "
              f"oracle I*={I_or:.4f} ===")
        print(f"{'sigma':>6}{'label':>13}{'erank':>7}{'R2':>7}{'cosL2':>8}{'I_cl':>10}  verdict")
        for sg in SIGMAS:
            Ws, Vo, Vr = [], [], []
            for ic in ics[:N_IC]:
                for s in range(N_NOISE):
                    rg = np.random.default_rng(1000 + s + 7 * int(sg * 100))
                    st, ct = rollout(A, B, lambda w: -K @ w[-1], n_steps, np.asarray(ic), sigma=sg, rng=rg)
                    w, vo, vr = windows_and_labels(st, ct, P, Q, R)
                    Ws.append(w); Vo.append(vo); Vr.append(vr)
            Ws = np.concatenate(Ws); Vo = np.concatenate(Vo); Vr = np.concatenate(Vr)
            erank = conditioning(np.stack([np.asarray(jax.jit(make_representation(
                kind=KIND, window_length=WINDOW_LENGTH, n_state=n, degree=DEG).feature_fn)(jnp.asarray(w)))
                for w in Ws[:300]]))
            for lab, V in [("oracle_label", Vo), ("rtg_label", Vr)]:
                m = fit_eval(A, B, Q, R, Ws, V, dep_W, u_star, x0, n_steps, half_RinvBT, n)
                ok = (m['I_cl'] < I_nc) and (m['cos'] > 0.95)
                v = "OK" if ok else ("WORSE-than-nc" if m['I_cl'] >= I_nc else "weak")
                print(f"{sg:>6.2f}{lab:>13}{erank:>7.2f}{m['r2']:>7.3f}{m['cos']:>8.3f}"
                      f"{m['I_cl']:>10.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_value_target"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(N_VEH=N_VEH, tau=TAU, sigmas=SIGMAS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
