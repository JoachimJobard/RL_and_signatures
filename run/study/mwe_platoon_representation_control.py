"""H0 on the real platoon: does a high-dimensional WINDOW representation, fit by
L^2 value-matching, corrupt the value-gradient control -- and does derivative
matching rescue it? Signature dropped on purpose; the contrast is the cleanest
honest pair.

  markovian deg-2   : features of the CURRENT state only -> low-dim, fully excited.
  raw_history deg-2 : degree-2 monomials of the whole flattened history WINDOW ->
                      high-dim, near-collinear along one trajectory. (Degree 2 so it
                      can represent the quadratic delayed-LQR value: value-fit R^2~1,
                      so the critic is not the variable -- exactly as in the E0 MWE.)

Env = connected-cruise-control platoon (src/envs/platoon.py), with its analytic
delayed-LQR oracle (gain, value, control). No learning loop: the IDEAL critic is the
ridge least-squares fit to the oracle's discounted return-to-go along the oracle
rollout. The value-gradient control u = 1/2 R^-1 B^T dV/dx(t) is read off; we measure

  M1  ||u_rep|| / ||u*||      magnitude (blow-up?)
  M2  L^2(mu) cosine to u*    direction (mis-aligned?)
  M3  closed-loop cost I      vs no-control and the oracle

then refit with DERIVATIVE MATCHING. The control needs only B^T dV/dx, and
u* = 1/2 R^-1 B^T dV/dx  =>  B^T dV/dx = 2 R u*, with u*_t = -gain @ xi_t known from
the oracle -- so the matching target is available WITHOUT the Riccati P. If matching
the endpoint gradient to the oracle control rescues raw_history, the E0 fitting-
objective mechanism transfers to the platoon.

Usage:
    uv run python run/study/mwe_platoon_representation_control.py [--debug]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REGIME = dict(n_vehicles=5, alpha=3.0, beta=1.0, delay=0.2, step_size=0.05, resolution=4)
TAU = 10.0          # discount of the return-to-go value target (matches the agent)
TF = 15.0
DM_SUBSAMPLE = 4    # use every k-th window for the (expensive) derivative-matching Jacobians
CLIP = 3.0          # closed-loop action safeguard (the agent's clip), to avoid NaN blow-ups


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R),
                                  float(env.max_delay), env.step_size)


def canonical_x0(n):
    x0 = np.zeros(n)
    x0[1] = 0.5      # the canonical lead-velocity perturbation
    return x0


def oracle_rollout(env, gain, k, x0, win_len):
    """One oracle rollout. Returns the representation windows (T, win_len, N),
    the oracle controls u*_t (T, m), and the rewards (T,)."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]      # oracle history, newest first
    windows, us_star, rs = [], [], []
    t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        windows.append(np.array(buf.buffer.to_array()).reshape(win_len, env.N))
        us_star.append(u.copy())
        t, x, r = w.step(w.state, jnp.array(u))
        x = np.array(x).reshape(-1)
        rs.append(float(r))
        buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(windows), np.array(us_star), np.array(rs)


def nocontrol_cost(env, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        t, x, r = w.step(w.state, jnp.zeros(env.B.shape[1]))
        I += -float(r) * env.step_size
    return I


def return_to_go(rs, dt, tau):
    gamma = np.exp(-dt / tau)
    V = np.zeros(len(rs)); acc = 0.0
    for i in range(len(rs) - 1, -1, -1):
        acc = rs[i] * dt + gamma * acc; V[i] = acc
    return V


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev[-1] / ev[0]), float(ev.sum() ** 2 / (ev ** 2).sum())


def closed_loop_cost(env, feat_fn, theta, win_len, x0):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    th = jnp.asarray(theta)
    grad_fn = jax.jit(jax.grad(lambda p: jnp.dot(th, feat_fn(p))))
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T
    w = JAXEnvWrapper(env, rng_key=42)
    w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        path = jnp.asarray(buf.buffer.to_array())
        end_grad = np.array(grad_fn(path))[-1]
        u = np.clip(half_RinvBT @ end_grad, -CLIP, CLIP)
        t, x, r = w.step(w.state, jnp.array(u))
        I += -float(r) * env.step_size
        buf.append(np.array(x).reshape(-1).astype(np.float32))
    return I


def evaluate(env, kind, degree, windows, us_star, V, half_RinvBT, win_len, x0, gamma_dm=0.0):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    rep = make_representation(kind, window_length=win_len, n_state=env.N, degree=degree)
    feat_fn = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat_fn(jnp.asarray(w))) for w in windows])      # (T,d)
    d = Phi.shape[1]
    G = Phi.T @ Phi
    rhs = Phi.T @ V
    lam = 1e-6 * np.trace(G) / d
    cond, erank = conditioning(Phi)

    if gamma_dm > 0.0:
        jac_fn = jax.jit(jax.jacfwd(rep.feature_fn))                            # (d,L,N)
        idx = range(0, len(windows), DM_SUBSAMPLE)
        for i in idx:
            J = np.asarray(jac_fn(jnp.asarray(windows[i])))[:, -1, :]           # (d,N)
            C = half_RinvBT @ J.T                                               # (m,d)
            G = G + gamma_dm * (C.T @ C)
            rhs = rhs + gamma_dm * (C.T @ us_star[i])

    theta = np.linalg.solve(G + lam * np.eye(d), rhs)
    r2 = 1.0 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)

    th = jnp.asarray(theta)
    grad_fn = jax.jit(jax.grad(lambda p: jnp.dot(th, feat_fn(p))))
    u_rep = np.stack([half_RinvBT @ np.array(grad_fn(jnp.asarray(w)))[-1] for w in windows])
    mag = float(np.mean(np.linalg.norm(u_rep, axis=1)) /
                (np.mean(np.linalg.norm(us_star, axis=1)) + 1e-12))
    cos = float(np.sum(u_rep * us_star) /
                (np.linalg.norm(u_rep) * np.linalg.norm(us_star) + 1e-12))
    I_cl = closed_loop_cost(env, feat_fn, theta, win_len, x0)
    return dict(kind=kind, dim=d, r2=float(r2), cond=cond, erank=erank,
                mag=mag, cos=cos, I_cl=I_cl)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    env = make_env(); n, dt = env.N, env.step_size
    lqr = oracle(env); k = lqr.k_taps
    win_len = int(np.ceil(float(env.max_delay) / dt)) + 3 + 1
    x0 = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T

    windows, us_star, rs = oracle_rollout(env, lqr.gain, k, x0, win_len)
    V = return_to_go(rs, dt, TAU)
    I_oracle = float(np.sum(-rs) * dt)
    I_nc = nocontrol_cost(env, x0)

    REPS = [("markovian", 2), ("raw_history", 2)]
    rows_std, rows_dm = [], []
    for kind, deg in REPS:
        rows_std.append(evaluate(env, kind, deg, windows, us_star, V, half_RinvBT, win_len, x0))
        rows_dm.append(evaluate(env, kind, deg, windows, us_star, V, half_RinvBT, win_len, x0,
                                gamma_dm=1.0))

    def show(title, rows):
        print(f"\n{title}")
        print(f"{'rep':<14}{'dim':>6}{'cond':>10}{'erank':>8}{'R2':>7}"
              f"{'|u|/|u*|':>10}{'cosL2':>8}{'I_cl':>10}  verdict")
        for r in rows:
            ok = (r['I_cl'] < I_nc) and (r['cos'] > 0.9)
            v = "OK" if ok else ("WORSE-than-nc" if r['I_cl'] >= I_nc else "weak")
            print(f"{r['kind']:<14}{r['dim']:>6}{r['cond']:>10.1e}{r['erank']:>8.2f}"
                  f"{r['r2']:>7.3f}{r['mag']:>10.2f}{r['cos']:>8.3f}{r['I_cl']:>10.4f}  {v}")

    print(f"\nplatoon H0 MWE | N={n} m={env.B.shape[1]} | win={win_len} dt={dt} TF={TF}")
    print(f"reference bars : no-control I={I_nc:.4f} | oracle I*={I_oracle:.4f}")
    show("[A] standard L^2 value fit (ideal critic)", rows_std)
    show("[B] + derivative matching (B^T dV/dx -> 2 R u*)", rows_dm)

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_platoon"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(regime=REGIME, I_nc=I_nc, I_oracle=I_oracle,
             standard=rows_std, derivative_matching=rows_dm), indent=2, default=str))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
