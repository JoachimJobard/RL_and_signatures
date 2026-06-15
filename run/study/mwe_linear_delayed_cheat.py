"""LINEAR delayed surrogate: does the off-manifold cheat solve the value-gradient
history control when the system is exactly linear (history matters, NO nonlinearity)?

On the real (nonlinear) platoon the off-manifold cheat FAILED: even with arbitrary
off-manifold data and clean labels, raw-history's control stayed ~23x the oracle
magnitude and diverged. Two confounds were tangled there: history-dependence AND
nonlinearity (the value label was the LINEARISED delayed-LQR value). This isolates
them: simulate the platoon's linearised matrices as an EXACT linear delayed system via
its augmented Markov form
    xi_{k+1} = A_aug xi_k + B_aug u_k,
where xi = [x_k, x_{k-1}, ..., x_{k-K}] is the history. Now V*(xi) = -xi^T P_aug xi and
u* = -K xi are EXACTLY optimal -- no linearisation error. History still genuinely
matters (delayed-LQR beats markovian). Run the same cheat (perturb current-state-only
or the full window off-manifold, label with V*) and a min-norm fit.

Decision:
  - cheat WORKS here (cos->1, |u|/|u*|->1, I->oracle)  => the real platoon's residual
    obstruction is NONLINEARITY;
  - cheat STILL FAILS (magnitude blow-up persists)      => the obstruction is
    HISTORY-DEPENDENCE itself (present/past cross-term entanglement in the features).

Usage:
    uv run python run/study/mwe_linear_delayed_cheat.py [--debug]
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
N_STEPS = 200
N_IC = 16
AUG_PER = 4
EPS = [0.0, 0.05, 0.2, 0.5]
IC_SCALE = 0.4
REPS = [("signature", dict(depth=2)), ("raw_history", dict(degree=2))]
SEED = 0


def linear_system():
    from src.envs.platoon import PlatoonEnv
    from src.solvers.delayed_lqr import augmented_discrete_lqr
    env = PlatoonEnv(**REGIME)
    A, A1, B = np.array(env.A), np.array(env.A1), np.array(env.B)
    Q, R = np.array(env.Q), np.array(env.R)
    lqr = augmented_discrete_lqr(A, A1, B, Q, R, float(env.max_delay), env.step_size)
    return lqr, Q, R, env.N


def window_of(xi, K, N):
    return xi.reshape(K + 1, N)[::-1]          # chronological, current state last


def xi_of(W):
    return W[::-1].reshape(-1)


def simulate(Aaug, Baug, Q, R, N, K, xi0, u_fn, n_steps):
    xi = xi0.copy(); Ws, cost = [], 0.0
    for _ in range(n_steps):
        W = window_of(xi, K, N); u = np.asarray(u_fn(W)).reshape(-1)
        if not np.all(np.isfinite(u)) or np.linalg.norm(xi) > 1e8:
            return np.array(Ws), float("inf")
        x = xi[:N]; cost += float(x @ Q @ x + u @ R @ u)
        Ws.append(W); xi = Aaug @ xi + Baug @ u
    return np.array(Ws), cost


def conditioning(Phi):
    G = Phi.T @ Phi / len(Phi)
    ev = np.clip(np.linalg.eigvalsh((G + G.T) / 2), 1e-18, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def main():
    import jax, jax.numpy as jnp
    from src.representations.factory import make_representation
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    lqr, Q, R, N = linear_system()
    Aaug, Baug, P, gain, K = lqr.Aaug, lqr.Baug, lqr.P, lqr.gain, lqr.k_taps
    WIN_LEN = K + 1
    half_RinvBT = 0.5 * np.linalg.inv(R) @ np.array(Baug)[:N].T   # B (current-state block)

    rng = np.random.default_rng(SEED)
    def ic_xi(x0): return np.tile(x0, K + 1)
    x0c = np.zeros(N); x0c[1] = 0.5
    ics = [x0c]
    for _ in range(N_IC - 1):
        x0 = np.zeros(N); x0[1::2] = IC_SCALE * rng.standard_normal(N // 2); ics.append(x0)

    u_oracle = lambda W: -gain @ xi_of(W)
    dep_W, I_or = simulate(Aaug, Baug, Q, R, N, K, ic_xi(x0c), u_oracle, N_STEPS)
    _, I_nc = simulate(Aaug, Baug, Q, R, N, K, ic_xi(x0c), lambda W: np.zeros(R.shape[0]), N_STEPS)
    u_star = np.array([-gain @ xi_of(W) for W in dep_W])
    mag_star = float(np.mean(np.linalg.norm(u_star, axis=1)))

    onman = np.concatenate([simulate(Aaug, Baug, Q, R, N, K, ic_xi(ic), u_oracle, N_STEPS)[0]
                            for ic in ics], 0)
    feats = {kd: jax.jit(make_representation(kd, window_length=WIN_LEN, n_state=N, **kw).feature_fn)
             for kd, kw in REPS}

    print(f"\nLINEAR delayed surrogate (exact LQR-delay) | dim={N} K={K} win={WIN_LEN} | "
          f"min-norm lstsq, oracle label")
    print(f"reference: no-control I={I_nc:.4f} oracle I*={I_or:.4f} | mean||u*||={mag_star:.4f}")
    print(f"{'rep':>12}{'mode':>14}{'eps':>6}{'erank':>7}{'R2':>7}{'cosL2':>8}"
          f"{'|u|/|u*|':>10}{'I_cl':>12}  verdict")

    for kd, _ in REPS:
        feat = feats[kd]
        for mode in ("current_only", "full_window"):
            for eps in EPS:
                W = onman.copy()
                if eps > 0:
                    aug = [onman]
                    for _ in range(AUG_PER):
                        Wp = onman.copy()
                        if mode == "current_only":
                            Wp[:, -1] = Wp[:, -1] + eps * rng.standard_normal(onman[:, -1].shape)
                        else:
                            Wp = onman + eps * rng.standard_normal(onman.shape)
                        aug.append(Wp)
                    W = np.concatenate(aug, 0)
                V = np.array([-(xi_of(w) @ P @ xi_of(w)) for w in W])
                Phi = np.stack([np.asarray(feat(jnp.asarray(w))) for w in W])
                erank = conditioning(Phi)
                theta = np.linalg.lstsq(Phi, V, rcond=None)[0]
                r2 = 1 - np.sum((V - Phi @ theta) ** 2) / (np.sum((V - V.mean()) ** 2) + 1e-12)
                th = jnp.asarray(theta); grad = jax.jit(jax.grad(lambda p: jnp.dot(th, feat(p))))
                u_rep = np.array([half_RinvBT @ np.array(grad(jnp.asarray(Wd)))[-1] for Wd in dep_W])
                cos = float(np.sum(u_rep * u_star) / (np.linalg.norm(u_rep) * np.linalg.norm(u_star) + 1e-12))
                ratio = float(np.mean(np.linalg.norm(u_rep, axis=1)) / (mag_star + 1e-12))
                _, I_cl = simulate(Aaug, Baug, Q, R, N, K, ic_xi(x0c),
                                   lambda Wd, grad=grad: half_RinvBT @ np.array(grad(jnp.asarray(Wd)))[-1],
                                   N_STEPS)
                ok = (I_cl < I_nc) and (cos > 0.95)
                v = "OK" if ok else ("WORSE-than-nc" if I_cl >= I_nc else "weak")
                print(f"{kd:>12}{mode:>14}{eps:>6.2f}{erank:>7.2f}{r2:>7.3f}"
                      f"{cos:>8.3f}{ratio:>10.2f}{I_cl:>12.4f}  {v}")

    debug = "_debug_" if args.debug else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    from src.utils.run_context import script_data_dir
    out = script_data_dir(__file__) / f"{debug}{ts}_linear_delayed"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(dict(regime=REGIME, eps=EPS), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
