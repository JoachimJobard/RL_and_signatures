"""Two clean controls for the value-gradient story on the platoon:

(a) The ~20-25x magnitude blow-up is a CONTINUOUS-vs-DISCRETE control-law convention,
    not a gradient pathology. The continuous value-gradient law u = -R^-1 B^T (P xi)[:n]
    (Doya, continuous time) overshoots the DISCRETE delayed-LQR oracle by a near-constant
    factor (cos~1). We confirm by rescaling u to the oracle magnitude and showing the
    closed loop recovers near-oracle -- i.e. the DIRECTION was fine, only the scalar was
    wrong. This uses the analytic gradient (no critic), so it isolates the control law.

(b) The genuine, convention-free control statistic is the gradient DIRECTION cos(u_rep, u*).
    For each representation we fit the critic (linearised label, on-manifold + off-manifold
    cheat) and report value R^2 (the representation result) vs gradient cos (the extraction
    result). Value fits for all; the gradient direction is where the critic's off-manifold
    unreliability shows -- the honest "value fit vs gradient fit" table.

Usage:
    uv run python run/study/mwe_platoon_gradient_direction.py [--debug]
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
TF = 15.0
WIN_LEN = 5
N_IC = 16
AUG_PER = 4
EPS_TRAIN = 0.2
REPS = [("markovian", dict(degree=2)), ("raw_history", dict(degree=2)), ("signature", dict(depth=2))]
SEED = 0
CLIP = 3.0


def make_env():
    from src.envs.platoon import PlatoonEnv
    return PlatoonEnv(**REGIME)


def oracle(env):
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    return augmented_discrete_lqr(np.array(env.A), np.array(env.A1), np.array(env.B),
                                  np.array(env.Q), np.array(env.R), float(env.max_delay), env.step_size)


def canonical_x0(n):
    x0 = np.zeros(n); x0[1] = 0.5
    return x0


def diverse_ics(n, n_ic, seed):
    rng = np.random.default_rng(seed); ics = [canonical_x0(n)]
    for _ in range(n_ic - 1):
        x0 = np.zeros(n); x0[1::2] = 0.4 * rng.standard_normal(n // 2); ics.append(x0)
    return ics


def rollout(env, gain, k, x0):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, rs = [], []; t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        Ws.append(np.array(buf.buffer.to_array()).reshape(WIN_LEN, env.N))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), np.array(rs)


def xi_of(W):
    return W[::-1].reshape(-1)


def closed_loop_xi(env, gain, k, x0, control_xi):
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    o_win = [x0.copy() for _ in range(k + 1)]; I, t = 0.0, 0.0
    while t < TF - 1e-9:
        xi = np.concatenate(o_win[:k + 1])
        u = np.clip(control_xi(xi), -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        I += -float(r) * env.step_size; o_win = [x] + o_win[:-1]
    return I


def closed_loop_window(env, feat, theta, half_RinvBT, x0, s):
    """Closed loop with the fitted critic's value-gradient control, RESCALED by scalar s."""
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=WIN_LEN, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=WIN_LEN, n_state=env.N)
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(WIN_LEN):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        u = np.clip(s * half_RinvBT @ np.array(grad(jnp.asarray(buf.buffer.to_array())))[-1], -CLIP, CLIP)
        if not np.all(np.isfinite(u)) or abs(I) > 1e8:
            return float("inf")
        t, x, r = w.step(w.state, jnp.array(u)); I += -float(r) * env.step_size
        buf.append(np.array(x).reshape(-1).astype(np.float32))
    return I


def main():
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    env = make_env(); n = env.N
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    Rinv = np.linalg.inv(np.array(env.R)); Bm = np.array(env.B)
    half_RinvBT = 0.5 * Rinv @ Bm.T
    x0c = canonical_x0(n)

    dep_W, dep_r = rollout(env, gain, k, x0c)
    nc = rollout(env, gain * 0.0, k, x0c)
    I_nc = float(np.sum(-nc[1]) * env.step_size); I_or = float(np.sum(-dep_r) * env.step_size)
    u_star = np.array([-gain @ xi_of(W) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))

    # continuous-formula control (analytic gradient, no critic): u = -R^-1 B^T (P xi)[:n]
    def u_cont(xi):
        return -Rinv @ Bm.T @ (P @ xi)[:n]
    u_c = np.array([u_cont(xi_of(W)) for W in dep_W])
    cos_c = float(np.sum(u_c * u_star) / (np.linalg.norm(u_c) * np.linalg.norm(u_star) + 1e-12))
    mag_c = float(np.mean(np.linalg.norm(u_c, axis=1)) / (mag_star + 1e-12))
    s = mag_star / (np.mean(np.linalg.norm(u_c, axis=1)) + 1e-12)   # rescale to oracle magnitude

    print(f"\n(a) MAGNITUDE = continuous-vs-discrete CONVENTION (analytic gradient, no critic)")
    print(f"reference: no-control I={I_nc:.4f} oracle I*={I_or:.4f}")
    print(f"  continuous-formula control: cos(u,u*)={cos_c:.3f}  |u|/|u*|={mag_c:.1f}  (near-constant overshoot)")
    print(f"  closed-loop, unscaled    : I={closed_loop_xi(env, gain, k, x0c, u_cont):.4f}")
    print(f"  closed-loop, RESCALED 1/{1/s:.1f} to oracle magnitude: "
          f"I={closed_loop_xi(env, gain, k, x0c, lambda xi: s * u_cont(xi)):.4f}  (-> near oracle confirms)")

    # (b) value fit vs gradient DIRECTION across representations
    rng = np.random.default_rng(SEED)
    onman = np.concatenate([rollout(env, gain, k, np.asarray(ic))[0]
                            for ic in diverse_ics(n, N_IC, SEED)], 0)
    aug = [onman]
    for _ in range(AUG_PER):
        aug.append(onman + EPS_TRAIN * rng.standard_normal(onman.shape))
    Wtr = np.concatenate(aug, 0)
    Vtr = np.array([-(xi_of(W) @ P @ xi_of(W)) for W in Wtr])

    print(f"\n(b) VALUE FIT vs GRADIENT DIRECTION + magnitude-rescaled closed loop "
          f"(no-control {I_nc:.4f}, oracle {I_or:.4f})")
    print(f"{'rep':>12}{'dim':>6}{'value R2':>10}{'grad cos':>10}{'I rescaled':>12}  verdict")
    for kd, kw in REPS:
        rep = make_representation(kd, window_length=WIN_LEN, n_state=n, **kw)
        feat = jax.jit(rep.feature_fn)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wtr]); d = Phi.shape[1]
        theta = np.linalg.lstsq(Phi, Vtr, rcond=None)[0]
        r2 = 1 - np.sum((Vtr - Phi @ theta) ** 2) / (np.sum((Vtr - Vtr.mean()) ** 2) + 1e-12)
        th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
        u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
        cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
        s = mag_star / (np.mean(np.linalg.norm(u_rep, axis=1)) + 1e-12)   # rescale to oracle magnitude
        I_res = closed_loop_window(env, feat, theta, half_RinvBT, x0c, s)
        v = "OK" if I_res < 1.5 * I_or else ("weak" if I_res < I_nc else "WORSE-than-nc")
        print(f"{kd:>12}{d:>6}{r2:>10.3f}{cos:>10.3f}{I_res:>12.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_gradient_direction"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, cos_c=cos_c, mag_c=mag_c), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
