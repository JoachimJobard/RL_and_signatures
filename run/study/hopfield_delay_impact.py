"""Analytic delay-impact gate for the delayed Hopfield network (H1 characterisation).

Measures *how much the history matters* on the delay-coupled Hopfield network
(``src/envs/delayed_hopfield_network.py``) by the same linearised delayed-LQR technique
used as the linear-cell oracle and as the Dadebo gate (``run/study/dadebo_delay_impact.py``).
Because phi_eps'(0) = 1 for every eps, the linearisation A = -I, A1 = W, B is
eps-INDEPENDENT, so this single gate certifies the entire eps sweep.

For each delay tau:
  * solve the augmented method-of-steps delayed-LQR and report the history-kernel ratio
    ||K_hist|| / ||K_now|| and the markovian-oracle cost gap (J_markovian - J_delayed)/|J_delayed|
    on the linearised plant (the H1 effect size);
  * rung-2 cross-check: apply the linearised optimal markovian / delayed gains to the actual
    NONLINEAR plant (eps = 1) and compare closed-loop cost -- separating the structural delay
    impact from optimisation quality.

A near-zero ratio/gap (the Dadebo verdict, 0.009 / +1%) would disqualify the cell; a
substantial ratio (cf. platoon 0.17 / +12.5%) certifies it as a genuine H1/H2 testbed.

Usage:
    uv run python run/study/hopfield_delay_impact.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

# Linearisation (eps-independent): A = -I, A1 = W, B = I.  Defaults match the env.
from src.envs.delayed_hopfield_network import rotational_coupling   # noqa: E402

W = rotational_coupling(rho=2.0, theta=np.pi / 2.0)                 # eig = +/- 2i
N = W.shape[0]
A = -np.eye(N)
B = np.eye(N)
Q = np.eye(N)
R = 0.1 * np.eye(N)
DT = 0.1
X0 = np.array([0.5, 0.0])
TAU_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
RUNG2_TAU = [(0.40, 15.0), (0.50, 15.0)]


def analytic_sweep() -> list[dict]:
    from src.solvers.delayed_lqr import augmented_discrete_lqr, markovian_oracle_gap
    rows = []
    for tau in TAU_GRID:
        lqr = augmented_discrete_lqr(A, A1_(), B, Q, R, tau, DT)
        k_now = float(np.linalg.norm(lqr.gain[:, :N]))
        k_hist = float(np.linalg.norm(lqr.gain[:, N:]))
        gap, j_or, j_mk = markovian_oracle_gap(A, A1_(), B, Q, R, X0, tau, DT, 200)
        rows.append(dict(tau=tau, k_taps=int(lqr.k_taps),
                         kernel_ratio=(k_hist / k_now) if k_now > 0 else float("nan"),
                         h1_cost_gap=float(gap), rho_cl=float(lqr.closed_loop_spectral_radius()),
                         j_delayed_lqr=float(j_or), j_markovian_lqr=float(j_mk)))
    return rows


def A1_():
    return np.asarray(W, dtype=float)


def nonlinear_rung2() -> list[dict]:
    import jax
    import jax.numpy as jnp
    from src.envs.delayed_hopfield_network import DelayedHopfieldNetwork
    from src.envs.env_rk_jax import JAXEnvWrapper
    from src.solvers.delayed_lqr import augmented_discrete_lqr

    def closed_loop(env, gain, k, tf):
        w = JAXEnvWrapper(env, rng_key=42)
        w.reset(jax.random.PRNGKey(0), x0=jnp.asarray(X0), t0=0.0)
        win = [X0.copy() for _ in range(k + 1)]
        cost, t = 0.0, 0.0
        while t < tf - 1e-9:
            u = -gain @ np.concatenate(win[:k + 1])
            t, _, r = w.step(w.state, jnp.asarray(u))
            cost += -float(r) * DT
            win = [np.asarray(w.state.x).reshape(-1)] + win[:-1]
        return cost

    rows = []
    for tau, tf in RUNG2_TAU:
        env = DelayedHopfieldNetwork(delay=tau, eps=1.0, step_size=DT, resolution=4)
        mk = augmented_discrete_lqr(A, np.zeros_like(A), B, Q, R, 0.0, DT)
        dl = augmented_discrete_lqr(A, A1_(), B, Q, R, tau, DT)
        j_nc = closed_loop(env, np.zeros((N, N)), 0, tf)
        j_mk = closed_loop(env, mk.gain, 0, tf)
        j_dl = closed_loop(env, dl.gain, dl.k_taps, tf)
        rows.append(dict(tau=tau, tf=tf, j_no_control=j_nc, j_markovian_lqr=j_mk,
                         j_delayed_lqr=j_dl, reduction_markovian=(j_nc - j_mk) / j_nc,
                         reduction_delayed=(j_nc - j_dl) / j_nc,
                         history_gain_over_markovian=(j_mk - j_dl) / j_dl))
    return rows


def main():
    from src.utils.run_context import script_data_dir
    out_dir = script_data_dir(__file__)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = analytic_sweep()
    print(f"\nDelayed Hopfield network: analytic delay-impact gate (W eig = {np.round(np.linalg.eigvals(W),3)})")
    print(f"{'tau':>6}{'k_taps':>8}{'kernel_ratio':>14}{'H1_gap[%]':>11}{'rho_cl':>9}")
    for r in rows:
        print(f"{r['tau']:>6.2f}{r['k_taps']:>8}{r['kernel_ratio']:>14.3f}"
              f"{100*r['h1_cost_gap']:>11.2f}{r['rho_cl']:>9.3f}")

    rung2 = nonlinear_rung2()
    print(f"\nRung-2 (linear gains on the NONLINEAR eps=1 plant):")
    print(f"{'tau':>6}{'red_mk[%]':>11}{'red_dl[%]':>11}{'hist_gain[%]':>13}")
    for r in rung2:
        print(f"{r['tau']:>6.2f}{100*r['reduction_markovian']:>11.1f}"
              f"{100*r['reduction_delayed']:>11.1f}{100*r['history_gain_over_markovian']:>13.1f}")

    (out_dir / "delay_impact_summary.json").write_text(
        json.dumps(dict(W=W.tolist(), analytic=rows, rung2_nonlinear=rung2), indent=2))
    print(f"\n[delay-impact] wrote delay_impact_summary.json in {out_dir}")


if __name__ == "__main__":
    main()
