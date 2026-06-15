"""Does sampling x0 in a small NEIGHBOURHOOD restore the value-gradient control?
(oracle method, platoon).

The single-trajectory ideal critic fails because the support is a 1-D curve: the
value-fit pins theta only modulo ker E_mu and leaves the NORMAL derivative free
(see documents/analysis/signature_value_gradient_conditioning/FINDINGS.md). The
oracle lets us test the fix exactly: roll the OPTIMAL policy from n_ic initial
conditions in a ball of radius eps around the canonical x0, pool the (exact,
deterministic) return-to-go targets, fit the ideal critic on the pooled data, then
DEPLOY closed-loop from the canonical x0. Sweeping eps answers whether a *small*
neighbourhood suffices or a finite radius is needed -- a tiny eps excites the normal
directions only at O(eps), so the pooled Gram may stay ill-conditioned.

Targets remain V* (optimal value): changing the INITIAL CONDITION keeps the optimal
policy, hence the same optimal value function on an enlarged support -- only changing
the POLICY would change the value.

Usage:
    uv run python run/study/mwe_platoon_neighbourhood_sweep.py [--debug]
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
TAU = 10.0
TF = 15.0
CLIP = 3.0
N_IC = 8
RADII = [0.0, 0.02, 0.1, 0.5]      # 0.0 = single-IC baseline (all rollouts identical)
REPS = [("markovian", 2), ("raw_history", 2)]
SEED = 0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R),
                                  float(env.max_delay), env.step_size)


def canonical_x0(n):
    x0 = np.zeros(n); x0[1] = 0.5
    return x0


def oracle_rollout(env, gain, k, x0, win_len):
    """One oracle rollout: rep windows (T,win_len,N), oracle controls (T,m), rewards (T,)."""
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
    o_win = [x0.copy() for _ in range(k + 1)]
    windows, us_star, rs = [], [], []
    t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        windows.append(np.array(buf.buffer.to_array()).reshape(win_len, env.N))
        us_star.append(u.copy())
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(windows), np.array(us_star), np.array(rs)


def nocontrol_cost(env, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        t, x, r = w.step(w.state, jnp.zeros(env.B.shape[1])); I += -float(r) * env.step_size
    return I


def return_to_go(rs, dt, tau):
    g = np.exp(-dt / tau); V = np.zeros(len(rs)); acc = 0.0
    for i in range(len(rs) - 1, -1, -1):
        acc = rs[i] * dt + g * acc; V[i] = acc
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
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        end_grad = np.array(grad_fn(jnp.asarray(buf.buffer.to_array())))[-1]
        u = np.clip(half_RinvBT @ end_grad, -CLIP, CLIP)
        t, x, r = w.step(w.state, jnp.array(u)); I += -float(r) * env.step_size
        buf.append(np.array(x).reshape(-1).astype(np.float32))
    return I


def fit_and_eval(env, kind, degree, fit_windows, fit_V, eval_windows, eval_us_star,
                 half_RinvBT, win_len, x0_eval):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    rep = make_representation(kind, window_length=win_len, n_state=env.N, degree=degree)
    feat_fn = jax.jit(rep.feature_fn)
    Phi = np.stack([np.asarray(feat_fn(jnp.asarray(w))) for w in fit_windows])
    d = Phi.shape[1]
    lam = 1e-6 * np.trace(Phi.T @ Phi) / d
    cond, erank = conditioning(Phi)
    theta = np.linalg.solve(Phi.T @ Phi + lam * np.eye(d), Phi.T @ fit_V)
    r2 = 1.0 - np.sum((fit_V - Phi @ theta) ** 2) / (np.sum((fit_V - fit_V.mean()) ** 2) + 1e-12)
    th = jnp.asarray(theta)
    grad_fn = jax.jit(jax.grad(lambda p: jnp.dot(th, feat_fn(p))))
    u_rep = np.stack([half_RinvBT @ np.array(grad_fn(jnp.asarray(w)))[-1] for w in eval_windows])
    mag = float(np.mean(np.linalg.norm(u_rep, axis=1)) /
                (np.mean(np.linalg.norm(eval_us_star, axis=1)) + 1e-12))
    cos = float(np.sum(u_rep * eval_us_star) /
                (np.linalg.norm(u_rep) * np.linalg.norm(eval_us_star) + 1e-12))
    I_cl = closed_loop_cost(env, feat_fn, theta, win_len, x0_eval)
    return dict(dim=d, cond=cond, erank=erank, r2=float(r2), mag=mag, cos=cos, I_cl=I_cl)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    env = make_env(); n, dt = env.N, env.step_size
    lqr = oracle(env); k = lqr.k_taps
    win_len = int(np.ceil(float(env.max_delay) / dt)) + 3 + 1
    x0c = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(np.array(env.R)) @ np.array(env.B).T
    rng = np.random.default_rng(SEED)

    # Deployment trajectory (canonical IC) -- where the control is queried/evaluated.
    eval_w, eval_us, eval_rs = oracle_rollout(env, lqr.gain, k, x0c, win_len)
    I_oracle = float(np.sum(-eval_rs) * dt)
    I_nc = nocontrol_cost(env, x0c)

    print(f"\nplatoon neighbourhood sweep | N={n} | win={win_len} dt={dt} TF={TF} | n_ic={N_IC}")
    print(f"reference bars : no-control I={I_nc:.4f} | oracle I*={I_oracle:.4f}")

    results = []
    for kind, deg in REPS:
        print(f"\n=== {kind} (deg {deg}) ===")
        print(f"{'radius':>8}{'dim':>7}{'cond':>10}{'erank':>8}{'R2':>7}"
              f"{'|u|/|u*|':>10}{'cosL2':>8}{'I_cl':>10}  verdict")
        for eps in RADII:
            ics = [x0c] + [x0c + eps * rng.standard_normal(n) for _ in range(N_IC - 1)]
            rolls = [oracle_rollout(env, lqr.gain, k, x0, win_len) for x0 in ics]
            fit_w = np.concatenate([w for w, _, _ in rolls], axis=0)
            fit_V = np.concatenate([return_to_go(r, dt, TAU) for _, _, r in rolls])
            m = fit_and_eval(env, kind, deg, fit_w, fit_V, eval_w, eval_us,
                             half_RinvBT, win_len, x0c)
            ok = (m['I_cl'] < I_nc) and (m['cos'] > 0.9)
            v = "OK" if ok else ("WORSE-than-nc" if m['I_cl'] >= I_nc else "weak")
            print(f"{eps:>8.2f}{m['dim']:>7}{m['cond']:>10.1e}{m['erank']:>8.2f}{m['r2']:>7.3f}"
                  f"{m['mag']:>10.2f}{m['cos']:>8.3f}{m['I_cl']:>10.4f}  {v}")
            results.append(dict(kind=kind, radius=eps, **m))

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_neighbourhood"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(
        dict(regime=REGIME, n_ic=N_IC, radii=RADII, I_nc=I_nc, I_oracle=I_oracle,
             results=results), indent=2, default=str))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
