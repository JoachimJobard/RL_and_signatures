"""Why does H1 fail (raw_history deg2 < markovian) on the high-gap linear cell?

The claim is conditioning/optimization, not representation. This script MEASURES the
conditioning: on the cell's on-trajectory windows it forms each representation's feature
covariance (Gram) C = E[(Phi-mean)(Phi-mean)'] and reports its condition number kappa =
lambda_max/lambda_min and effective rank. Linear semi-gradient TD converges at a rate set
by the eigenvalue spread of C: the slow (small-eigenvalue) directions need ~kappa times
more updates, so kappa(raw)/kappa(markovian) predicts the extra episode budget raw_history
needs to catch markovian. A large ratio with a representation that PROVABLY spans V*
(R^2=1.0, see least_squares_value_fit_check.py) ⇒ the H1 failure at a matched budget is
under-convergence in the ill-conditioned directions, not a representation deficit.

Usage: uv run python run/study/feature_conditioning_check.py [--env delayed_oscillator_high_gap]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import hydra
from omegaconf import OmegaConf

from src.envs.env_rk_jax import JAXEnvWrapper
from src.representations.factory import make_representation
from src.solvers.oracle_agent import delayed_lqr_for_env


def collect_windows(env, lqr, L, x0s, T_sim, jitter, seed=0):
    rng = np.random.default_rng(seed)
    n = env.N
    wins = []
    for x0 in x0s:
        w = JAXEnvWrapper(env)
        w.reset(jax.random.PRNGKey(0), x0=jnp.array(x0), t0=0.0,
                history_function=lambda t: jnp.array(x0))
        feat = np.tile(np.asarray(x0, float).reshape(n), (L, 1))
        for _ in range(int(T_sim / env.step_size)):
            xi = feat[::-1][:lqr.k_taps + 1].reshape(-1)
            u = np.asarray(lqr.control(xi.reshape(-1, n))).reshape(-1)
            wins.append(feat + jitter * rng.standard_normal(feat.shape))
            _, x_next, _ = w.step(w.state, jnp.array(u))
            feat = np.vstack([feat[1:], np.asarray(x_next).reshape(n)[None, :]])
    return np.array(wins)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="delayed_oscillator_high_gap")
    ap.add_argument("--window-size", type=int, default=5)
    args = ap.parse_args()

    cfg = OmegaConf.load(Path("conf/env") / f"{args.env}.yaml")
    env = hydra.utils.instantiate(cfg.environment_params)
    n, L = env.N, args.window_size + 1
    lqr = delayed_lqr_for_env(env)
    x0s = [np.ones(n), -np.ones(n), np.array([1.0, -1.0])[:n], np.array([0.5, 0.5])[:n]]
    W = collect_windows(env, lqr, L, x0s, 15.0, jitter=0.1)

    reps = {
        "markovian deg2": make_representation("markovian", window_length=L, n_state=n, degree=2),
        "raw_history deg2": make_representation("raw_history", window_length=L, n_state=n, degree=2),
        "signature depth2": make_representation("signature", window_length=L, n_state=n, depth=2),
        "signature depth3": make_representation("signature", window_length=L, n_state=n, depth=3),
    }

    print(f"\nenv={args.env}  N_windows={len(W)}")
    print("kappa = cond(feature covariance); semi-gradient TD convergence time ~ kappa.")
    hdr = (f"{'representation':18s} {'dim':>4s} {'kappa':>11s} {'eff.rank':>9s} "
           f"{'lambda_min':>11s} {'budget x vs mk':>15s}")
    print(hdr); print("-" * len(hdr))
    kappa_mk = None
    for name, rep in reps.items():
        Phi = np.array([np.asarray(rep.feature_fn(jnp.array(w_))) for w_ in W])  # (N, dim)
        Phi = Phi - Phi.mean(axis=0, keepdims=True)
        C = (Phi.T @ Phi) / len(Phi)
        ev = np.linalg.eigvalsh(C)
        ev = np.clip(ev, 0, None)
        lam_max, lam_min = float(ev[-1]), float(ev[ev > 1e-14][0]) if np.any(ev > 1e-14) else 0.0
        kappa = lam_max / lam_min if lam_min > 0 else np.inf
        eff_rank = float(ev.sum() ** 2 / (ev ** 2).sum())  # participation ratio
        if "markovian" in name:
            kappa_mk = kappa
        ratio = kappa / kappa_mk if kappa_mk else 1.0
        print(f"{name:18s} {rep.feature_dim:>4d} {kappa:>11.2e} {eff_rank:>9.1f} "
              f"{lam_min:>11.2e} {ratio:>14.1f}x")

    print("\nReading: kappa(raw_history)/kappa(markovian) = the factor by which raw-history's")
    print("ill-conditioned (near-collinear cross-term) directions slow semi-gradient TD ->")
    print("~that factor more episodes to converge. At a matched budget raw-history is under-")
    print("converged in those directions, so it loses to markovian even though it spans V*")
    print("exactly (R^2=1.0). That is the H1 failure mechanism on the high-gap LINEAR cell;")
    print("on the nonlinear MG cells markovian cannot represent the value at all, so history")
    print("wins regardless of conditioning.")


if __name__ == "__main__":
    main()
