"""REAL platoon: does a NONLINEAR value target remove the magnitude blow-up?

The off-manifold cheat with the LINEARISED label V*=-xi^T P_aug xi left raw-history at
~23x the oracle magnitude on the nonlinear platoon, while the SAME cheat on the exact
linear-delayed system worked (~1.15x). Conclusion: the obstruction is nonlinearity --
the linearised quadratic label is the wrong function off the origin. This tests the
predicted fix: replace the label by the realised return on the NONLINEAR plant.

Because the platoon dynamics and the oracle policy are DETERMINISTIC, the return-to-go
from any (possibly off-manifold) augmented state xi is EXACT -- one rollout, zero
variance -- so it avoids the label-noise that was fatal earlier. For each window we set
the env history from it (direct buffer construction) and roll the deterministic oracle
forward, accumulating cost: that is V^{pi_oracle}_nonlinear(xi). We then fit the SAME
critic to the linearised vs the nonlinear label on the SAME windows and compare.

Decision: if the nonlinear label drops |u|/|u*| toward 1 and I toward the oracle, the
linearised label was the whole nonlinearity problem (degree-2 critic may suffice if the
nonlinear value is near-quadratic with the RIGHT Hessian); if it stays blown up, a
higher-capacity critic (degree 3) is also needed.

Usage:
    uv run python run/study/mwe_real_platoon_nonlinear_label.py [--debug]
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
H_LABEL = 120                    # forward control steps for the nonlinear return-to-go label
N_IC = 8
SUBSAMPLE = 4                    # label every k-th window (forward rollouts are the cost)
AUG_PER = 2
EPS_CHEAT = 0.05
IC_SCALE = 0.4
REPS = [("signature", dict(depth=3)), ("raw_history", dict(degree=3))]
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
        x0 = np.zeros(n); x0[1::2] = IC_SCALE * rng.standard_normal(n // 2); ics.append(x0)
    return ics


def rollout(env, gain, k, x0):
    """Deterministic oracle rollout. Returns windows (T,K+1,N), rewards (T,), and the
    EXACT fine env state at each step as (buffer_data, ptr) snapshots -- so a label can
    be computed by rolling forward from the *true* fine history (no interpolation)."""
    import jax, jax.numpy as jnp
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.representations.factory import make_representation, RepresentationBuffer
    win_len = k + 1
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    o_win = [x0.copy() for _ in range(k + 1)]; Ws, rs, snaps = [], [], []; t = 0.0
    while t < TF - 1e-9:
        u = -gain @ np.concatenate(o_win[:k + 1])
        Ws.append(np.array(buf.buffer.to_array()).reshape(win_len, env.N))
        snaps.append((np.array(w.state.buffer.data), int(w.state.buffer.ptr), np.array(w.state.x)))
        t, x, r = w.step(w.state, jnp.array(u)); x = np.array(x).reshape(-1)
        rs.append(float(r)); buf.append(x.astype(np.float32)); o_win = [x] + o_win[:-1]
    return np.array(Ws), np.array(rs), snaps


def state_from_snap(env, snap):
    import jax.numpy as jnp
    from src.utils.solver_buffer_jax import BufferState
    from src.envs.env_rk_jax import EnvState
    data, ptr, x = snap
    buf = BufferState(data=jnp.asarray(data), ptr=int(ptr), capacity=env.buffer_size, dim=env.N)
    return EnvState(x=jnp.asarray(x), t=0.0, buffer=buf, last_u=jnp.zeros(env.B.shape[1]))


def perturb(snap, W, delta):
    """Off-manifold cheat at the FINE level: move the current state by delta, freeze the
    past (the most-recent fine buffer entry and the window endpoint move with it)."""
    data, ptr, x = snap
    data = data.copy(); j = (ptr - 1) % data.shape[0]; data[j] = data[j] + delta
    W = W.copy(); W[-1] = W[-1] + delta
    return (data, ptr, x + delta), W


def nonlinear_return(env, gain, k, snap, W, H, Q, R, jit_step):
    """Exact V^{pi_oracle}_nonlinear = -(cost-to-go of the deterministic oracle rolled
    forward from the TRUE fine state on the NONLINEAR plant)."""
    import jax.numpy as jnp
    state = state_from_snap(env, snap)
    o_win = [W[-1 - i].copy() for i in range(k + 1)]; reward_to_go = 0.0
    for _ in range(H):
        u = -gain @ np.concatenate(o_win[:k + 1])
        state, xn, r = jit_step(state, jnp.asarray(u))   # r = env reward (uses next state)
        reward_to_go += float(r)
        o_win = [np.asarray(xn).reshape(-1)] + o_win[:-1]
    return reward_to_go            # = -(cost-to-go), same convention as V_lin = -xi^T P xi


def xi_of(W):
    return W[::-1].reshape(-1)


def closed_loop_cost(env, feat, theta, win_len, x0, half_RinvBT):
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation, RepresentationBuffer
    from src.envs.env_rk_jax import JAXEnvWrapper
    rep = make_representation("markovian", window_length=win_len, n_state=env.N, degree=1)
    buf = RepresentationBuffer(rep, window_length=win_len, n_state=env.N)
    th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
    w = JAXEnvWrapper(env, rng_key=42); w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0)
    buf.reset()
    for _ in range(win_len):
        buf.append(x0.astype(np.float32))
    I, t = 0.0, 0.0
    while t < TF - 1e-9:
        u = np.clip(half_RinvBT @ np.array(grad(jnp.asarray(buf.buffer.to_array())))[-1], -CLIP, CLIP)
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
    env = make_env(); n = env.N; Q, R = np.array(env.Q), np.array(env.R)
    lqr = oracle(env); k, P, gain = lqr.k_taps, lqr.P, lqr.gain
    win_len = k + 1; x0c = canonical_x0(n)
    half_RinvBT = 0.5 * np.linalg.inv(R) @ np.array(env.B).T
    jit_step = jax.jit(env.step)

    dep_W, dep_r, dep_snaps = rollout(env, gain, k, x0c)
    nc_W, nc_r, _ = rollout(env, gain * 0.0, k, x0c)
    I_or = float(np.sum(-dep_r) * env.step_size); I_nc = float(np.sum(-nc_r) * env.step_size)
    u_star = np.array([-gain @ xi_of(W) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))

    # validation: exact nonlinear_return vs the rollout's reward-to-go (must now match).
    print("\nvalidation (nonlinear_return vs direct reward-to-go, on-manifold):")
    for i in [10, 60, 120]:
        nl = nonlinear_return(env, gain, k, dep_snaps[i], dep_W[i], H_LABEL, Q, R, jit_step)
        print(f"  i={i:3d}  nonlinear_return={nl:8.4f}  rollout_rtg(H)={float(np.sum(dep_r[i:i+H_LABEL])):8.4f}")

    # collect (window, snapshot) pairs; build on-manifold + off-manifold (fine-level cheat).
    rng = np.random.default_rng(SEED)
    pairs = []
    for ic in diverse_ics(n, N_IC, SEED):
        Ws, _, snaps = rollout(env, gain, k, np.asarray(ic))
        for j in range(0, len(Ws), SUBSAMPLE):
            pairs.append((Ws[j], snaps[j]))
    items = list(pairs)
    for _ in range(AUG_PER):
        for (W, snap) in pairs:
            snp, Wp = perturb(snap, W, EPS_CHEAT * rng.standard_normal(n))
            items.append((Wp, snp))
    Wall = np.array([it[0] for it in items])
    print(f"\nlabelling {len(Wall)} windows by forward rollout (H={H_LABEL}) ...")
    V_lin = np.array([-(xi_of(W) @ P @ xi_of(W)) for W in Wall])
    V_nl = np.array([nonlinear_return(env, gain, k, snap, W, H_LABEL, Q, R, jit_step)
                     for (W, snap) in items])

    print(f"\nREAL platoon NONLINEAR vs LINEARISED label | dim={n} win={win_len} | "
          f"min-norm lstsq | n_pts={len(Wall)}")
    print(f"reference: no-control I={I_nc:.4f} oracle I*={I_or:.4f} | mean||u*||={mag_star:.4f}")
    print(f"{'rep':>12}{'dim':>6}{'label':>12}{'R2':>7}{'cosL2':>8}{'|u|/|u*|':>10}{'I_cl':>11}  verdict")

    for kd, kw in REPS:
        rep = make_representation(kd, window_length=win_len, n_state=n, **kw)
        feat = jax.jit(rep.feature_fn); d = int(rep.feature_dim)
        Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in Wall])
        for lab, V in [("linearised", V_lin), ("nonlinear", V_nl)]:
            theta = np.linalg.lstsq(Phi, V, rcond=None)[0]
            r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
            th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
            u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(W)))[-1] for W in dep_W])
            cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
            ratio = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (mag_star + 1e-12))
            I_cl = closed_loop_cost(env, feat, theta, win_len, x0c, half_RinvBT)
            ok = (I_cl < I_nc) and (cos > 0.9)
            v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
            print(f"{kd:>12}{d:>6}{lab:>12}{r2:>7.3f}{cos:>8.3f}{ratio:>10.2f}{I_cl:>11.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_nonlinear_label"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, H=H_LABEL, eps=EPS_CHEAT), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
