"""E0: does a NEURAL-NET critic show the same value-good / gradient-bad dissociation,
and is the magnitude blow-up milder than a linear rich-basis critic?

Prediction (from the unboundedness of differentiation + SGD's spectral bias): the
dissociation (value R^2 ~ 1 while the gradient is imperfect) is structural and survives
the switch to a NN, but the wild magnitude blow-up of min-norm-in-a-rich-basis should be
much milder for an MLP, because SGD biases toward smooth/low-frequency functions (which
have bounded gradients). E0 is the clean test: 2-D double-integrator LQR, V*(x)=-x^T P x
analytic, u*=-Kx, so we can score the gradient directly.

Critics (both take the WINDOW, like the signature/raw-history critics):
  - linear raw_history deg-2 (min-norm lstsq)  -- the rich-basis baseline;
  - small MLP (2x64 tanh) trained by Adam on the value MSE.
Each fit ON-MANIFOLD only and WITH the off-manifold cheat; scored by value R^2, gradient
cosine, magnitude ratio, and closed-loop cost.

Usage:
    uv run python run/study/mwe_e0_nn_critic.py [--debug]
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
import optax
import flax.linen as fnn
import scipy.linalg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.representations.factory import make_representation  # noqa: E402
from src.utils.run_context import script_data_dir  # noqa: E402

A = np.array([[0.0, 1.0], [0.0, 0.0]]); B = np.array([[0.0], [1.0]])
Q = np.eye(2); R = np.array([[0.1]])
N_STATE = 2; WINDOW_LENGTH = 9; DT = 0.05; HORIZON = 6.0
X0_DEPLOY = np.array([1.0, 0.0])
N_IC = 16; AUG_PER = 4; EPS_CHEAT = 0.2; IC_SCALE = 1.5
MLP_STEPS = 4000
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
    return np.stack([states[k - WINDOW_LENGTH + 1:k + 1] for k in range(WINDOW_LENGTH - 1, len(states))])


def quad_cost(states, controls):
    cs = np.einsum("ti,ij,tj->t", states[:-1], Q, states[:-1])
    cu = np.einsum("ti,ij,tj->t", controls, R, controls)
    return float(np.sum((cs + cu) * DT))


class MLP(fnn.Module):
    @fnn.compact
    def __call__(self, x):
        x = fnn.tanh(fnn.Dense(64)(x))
        x = fnn.tanh(fnn.Dense(64)(x))
        return fnn.Dense(1)(x)[..., 0]


def train_mlp(Wtr, Vtr, key):
    X = jnp.asarray(Wtr.reshape(len(Wtr), -1)); y = jnp.asarray(Vtr)
    model = MLP(); params = model.init(key, X[:1])
    opt = optax.adam(1e-3); opt_state = opt.init(params)

    def loss(p):
        return jnp.mean((model.apply(p, X) - y) ** 2)

    @jax.jit
    def step(p, os):
        l, g = jax.value_and_grad(loss)(p)
        upd, os = opt.update(g, os); return optax.apply_updates(p, upd), os, l
    for _ in range(MLP_STEPS):
        params, opt_state, _ = step(params, opt_state)
    Vfn = lambda W: model.apply(params, W.reshape(-1))
    return Vfn


def score(Vfn_jax, half_RinvBT, dep_W, u_star, P):
    grad = jax.jit(jax.grad(lambda W: Vfn_jax(W)))
    u_rep = np.stack([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
    cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
    mag = float(np.mean(np.linalg.norm(u_rep, 1)) / (np.mean(np.linalg.norm(u_star, 1)) + 1e-12))
    Vt = np.array([float(Vfn_jax(jnp.asarray(W))) for W in dep_W])
    Vs = np.array([-(W[-1] @ P @ W[-1]) for W in dep_W])
    r2 = 1 - np.sum((Vs - Vt) ** 2) / (np.sum((Vs - Vs.mean()) ** 2) + 1e-12)
    return r2, cos, mag


def closed_loop(Vfn_jax, half_RinvBT, n_steps):
    grad = jax.jit(jax.grad(lambda W: Vfn_jax(W)))
    cf = lambda W: half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1]
    s, c = rollout(cf, n_steps, X0_DEPLOY)
    return quad_cost(s, c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    P, K = lqr(); Rinv = np.linalg.inv(R); half_RinvBT = 0.5 * Rinv @ B.T
    n_steps = int(HORIZON / DT); rng = np.random.default_rng(SEED)

    dep_states, dep_c = rollout(lambda w: -K @ w[-1], n_steps, X0_DEPLOY)
    nc_s, nc_c = rollout(lambda w: np.zeros(B.shape[1]), n_steps, X0_DEPLOY)
    I_nc, I_or = quad_cost(nc_s, nc_c), quad_cost(dep_states, dep_c)
    dep_W = full_windows(dep_states[:-1]); u_star = np.stack([-K @ W[-1] for W in dep_W])

    ics = [X0_DEPLOY] + [IC_SCALE * rng.standard_normal(N_STATE) for _ in range(N_IC)]
    onman = np.concatenate([full_windows(rollout(lambda w: -K @ w[-1], n_steps, np.asarray(ic))[0])
                            for ic in ics], 0)

    def with_cheat(W):
        out = [W]
        for _ in range(AUG_PER):
            Wp = W.copy(); Wp[:, -1] = Wp[:, -1] + EPS_CHEAT * rng.standard_normal(W[:, -1].shape)
            out.append(Wp)
        return np.concatenate(out, 0)

    rep = make_representation("raw_history", window_length=WINDOW_LENGTH, n_state=N_STATE, degree=2)
    feat = jax.jit(rep.feature_fn)

    print(f"\nE0 NN critic vs linear rich basis | win={WINDOW_LENGTH} | MLP 2x64 ({MLP_STEPS} steps)")
    print(f"reference: no-control I={I_nc:.4f} oracle I*={I_or:.4f}")
    print(f"{'critic':>16}{'data':>14}{'valueR2':>9}{'cosL2':>8}{'|u|/|u*|':>10}{'I_cl':>10}  verdict")

    for data_name, Wtr in [("on-manifold", onman), ("with cheat", with_cheat(onman))]:
        Vtr = np.array([-(W[-1] @ P @ W[-1]) for W in Wtr])
        # linear raw_history (min-norm)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr])
        theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]; th = jnp.asarray(theta)
        Vlin = lambda W, th=th: jnp.dot(th, feat(W))
        # MLP
        key = jax.random.PRNGKey(SEED)
        Vmlp = train_mlp(Wtr, Vtr, key)
        for cname, Vfn in [("linear raw_hist", Vlin), ("MLP", Vmlp)]:
            r2, cos, mag = score(Vfn, half_RinvBT, dep_W, u_star, P)
            I = closed_loop(Vfn, half_RinvBT, n_steps)
            ok = (I < I_nc) and (cos > 0.95)
            v = "OK" if ok else ("WORSE-than-nc" if I >= I_nc else "weak")
            print(f"{cname:>16}{data_name:>14}{r2:>9.3f}{cos:>8.3f}{mag:>10.2f}{I:>10.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = script_data_dir(__file__) / f"{debug}{ts}_nn_critic"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(win=WINDOW_LENGTH, mlp_steps=MLP_STEPS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
