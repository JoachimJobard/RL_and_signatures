"""Analytic delay-impact diagnostic for the Dadebo CSTR (H1 characterisation).

Measures *how much the history matters* on the Dadebo two-stage CSTR
(``src/envs/chemical_process.py``) WITHOUT any reinforcement learning, by the same
delayed-LQR optimal-control technique used as the linear-cell oracle
(``src/solvers/delayed_lqr.py``):

1. **Linearise** the nonlinear CSTR dynamics ``f(x, x(t-tau), u)`` at the regulation
   setpoint ``x*=0, u*=0`` (JAX autodiff) to obtain ``A = df/dx``, ``A1 =
   df/dx_delayed``, ``B = df/du``. The setpoint is a genuine equilibrium
   (``f(0,0,0)=0``); ``eig(A)`` has an unstable mode (tank 1, +3.0) so control is
   genuinely required, and ``A1`` couples the delayed tank-1 state into tank 2.
2. **Analytic H1 effect size vs delay.** For each ``tau`` solve the augmented
   method-of-steps delayed-LQR (``augmented_discrete_lqr``) and report
   ``markovian_oracle_gap`` = ``(J_markovian - J_delayed)/|J_delayed|`` on the
   linearised plant, plus the history-kernel ratio
   ``||K_history|| / ||K_current||``. A large gap / ratio means history matters; a
   near-zero one means the plant is near-Markovian and H1 is predicted null.
3. **Rung-2 cross-check on the NONLINEAR plant.** Apply the (linearised) optimal
   markovian-LQR and delayed-LQR gains to the actual nonlinear environment and
   compare their closed-loop cost. This separates the *structural* delay impact from
   the *optimisation* quality of any learned controller: if the optimal markovian-LQR
   already attains the delayed-LQR cost on the nonlinear plant, then a learned-agent
   markovian/history gap is an optimisation artefact, not a structural H1 effect.

Empirical conclusion (measured 2026-06-11): the analytic H1 gap is tiny in the
paper's delay range (+1.0% at tau=0.20) and grows only modestly with tau (+7.7% at
tau=1.50, +11.1% at tau=2.00); on the nonlinear plant the optimal markovian-LQR
matches or beats the delayed-LQR at every tested tau (e.g. 94.9% vs 94.2% cost
reduction at tau=1.5). The Dadebo CSTR is therefore *structurally near-Markovian* at
all studied delays, so the null/weak H1 observed in the RL grids is the CORRECT
outcome, and a large markovian-vs-history gap in a learned agent (e.g. the tau=1.5,
tf=4 single-seed run where the markovian critic under-converged) is an optimisation
artefact rather than evidence for H1.

Usage:
    uv run python run/study/dadebo_delay_impact.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Headless backend before pyplot is imported transitively.
import matplotlib
matplotlib.use("Agg")


# The paper's tabulated delays (Dadebo & Luus 1992, Table V) and a few larger,
# beyond-paper delays to expose the (still modest) growth of the history effect.
TAU_GRID = [0.05, 0.10, 0.20, 0.40, 0.60, 1.00, 1.50, 2.00]
PAPER_TAU = (0.05, 0.40)         # band where a ground-truth DP optimum exists
DT = 0.01                        # control step (matches the RL runs)
X0 = np.array([0.15, -0.03, 0.1, 0.0])   # paper initial condition
N_STEPS = 200                    # tf=2.0 at dt=0.01 for the linearised cost integral


def linearise_cstr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobians (A, A1, B) of the CSTR dynamics at the setpoint x*=0, u*=0."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    def f(x, xd, u):
        R1 = (x[0] + 0.5) * jnp.exp(25 * x[1] / (x[1] + 2))
        R2 = (x[2] + 0.25) * jnp.exp(25 * x[3] / (x[3] + 2))
        return jnp.array([
            0.5 - x[0] - R1,
            -2 * (x[1] + 0.25) - u[0] * (x[1] + 0.25) + R1,
            xd[0] - x[2] - R2 + 0.25,
            xd[1] - 2 * x[3] - u[1] * (x[3] + 0.25) + R2 - 0.25,
        ])

    z, u0 = jnp.zeros(4), jnp.zeros(2)
    A = np.asarray(jax.jacobian(f, 0)(z, z, u0))
    A1 = np.asarray(jax.jacobian(f, 1)(z, z, u0))
    B = np.asarray(jax.jacobian(f, 2)(z, z, u0))
    return A, A1, B


def analytic_sweep(A, A1, B, Q, R) -> list[dict]:
    """Per-delay analytic H1 effect size on the linearised plant."""
    from src.solvers.delayed_lqr import augmented_discrete_lqr, markovian_oracle_gap

    rows: list[dict] = []
    for tau in TAU_GRID:
        lqr = augmented_discrete_lqr(A, A1, B, Q, R, tau, DT)
        n = A.shape[0]
        k_now = float(np.linalg.norm(lqr.gain[:, :n]))
        k_hist = float(np.linalg.norm(lqr.gain[:, n:]))
        gap, j_oracle, j_markov = markovian_oracle_gap(A, A1, B, Q, R, X0, tau, DT, N_STEPS)
        rows.append({
            "tau": tau, "k_taps": int(lqr.k_taps),
            "h1_cost_gap": float(gap),
            "kernel_ratio": float(k_hist / k_now) if k_now > 0 else float("nan"),
            "j_delayed_lqr": float(j_oracle), "j_markovian_lqr": float(j_markov),
        })
    return rows


def nonlinear_rung2(A, A1, B, Q, R, cases) -> list[dict]:
    """Apply the linearised optimal markovian/delayed LQR gains to the NONLINEAR
    plant and compare closed-loop cost (oracle-ladder rung 2)."""
    import jax
    import jax.numpy as jnp
    from src.envs.chemical_process import ChemicalReactionEnv
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
            x_now = np.asarray(w.state.x).reshape(-1)
            win = [x_now] + win[:-1]
        return cost

    rows: list[dict] = []
    for tau, tf in cases:
        env = ChemicalReactionEnv(delay=tau, step_size=DT, resolution=4,
                                  Q=jnp.eye(4), R=0.1 * jnp.eye(2))
        mk = augmented_discrete_lqr(A, np.zeros_like(A), B, Q, R, 0.0, DT)
        dl = augmented_discrete_lqr(A, A1, B, Q, R, tau, DT)
        j_nc = closed_loop(env, np.zeros((2, 4)), 0, tf)
        j_mk = closed_loop(env, mk.gain, 0, tf)
        j_dl = closed_loop(env, dl.gain, dl.k_taps, tf)
        rows.append({
            "tau": tau, "tf": tf, "j_no_control": j_nc,
            "j_markovian_lqr": j_mk, "reduction_markovian": (j_nc - j_mk) / j_nc,
            "j_delayed_lqr": j_dl, "reduction_delayed": (j_nc - j_dl) / j_nc,
            "history_gain_over_markovian": (j_mk - j_dl) / j_dl,
        })
    return rows


def build_figure(rows: list[dict], out_dir: Path):
    import matplotlib.pyplot as plt
    from src.utils.plot_style import STROKE_TRAINED, prepare_figure

    taus = [r["tau"] for r in rows]
    gaps = [100.0 * r["h1_cost_gap"] for r in rows]
    ratios = [r["kernel_ratio"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    l1 = ax.plot(taus, gaps, f"o{STROKE_TRAINED}", color="#440154",
                 label=r"analytic H1 cost-gap $(J_{\mathrm{mk}}-J_{\mathrm{opt}})/J_{\mathrm{opt}}$")[0]
    ax.set_xlabel(r"delay $\tau$")
    ax.set_ylabel(r"H1 cost-gap [\%] (optimal markovian vs delayed LQR)")
    ax.axvspan(*PAPER_TAU, color="0.85", zorder=0,
               label=r"paper delay range (Table V)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    l2 = ax2.plot(taus, ratios, f"s{STROKE_TRAINED}", color="#21918c",
                  label=r"history/current gain ratio $\|K_{\mathrm{hist}}\|/\|K_{\mathrm{now}}\|$")[0]
    ax2.set_ylabel(r"history-kernel ratio $\|K_{\mathrm{hist}}\|/\|K_{\mathrm{now}}\|$")

    band = ax.patches[0]
    handles = [l1, l2, band]
    labels = [h.get_label() for h in handles]
    fig.suptitle("Dadebo CSTR: analytic delay impact (linearised delayed-LQR)", fontsize=12)
    prepare_figure(
        fig, fname="dadebo_delay_impact", axes=[ax, ax2],
        handles=handles, labels=labels, reserve_bottom=0.30, legend_y=0.12,
        legend_fontsize=8,
        formula=(r"Linearise at $x^\star=0,u^\star=0$; both LQR controllers optimal. A small "
                 r"gap means the plant is near-Markovian (H1 null). Even at $\tau=2$ the "
                 r"optimal history controller beats the optimal markovian one by $<12\%$."),
    )
    fig.savefig(str(out_dir / "dadebo_delay_impact.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from src.utils.run_context import script_data_dir
    out_dir = script_data_dir(__file__)
    out_dir.mkdir(parents=True, exist_ok=True)

    A, A1, B = linearise_cstr()
    Q, R = np.eye(4), 0.1 * np.eye(2)
    eigs = np.linalg.eigvals(A)

    sweep = analytic_sweep(A, A1, B, Q, R)
    rung2 = nonlinear_rung2(A, A1, B, Q, R, cases=[(0.20, 2.0), (1.50, 4.0)])

    print("Linearised CSTR at x*=0, u*=0:")
    print("  eig(A) =", np.round(eigs, 3))
    print("\nAnalytic H1 effect size vs delay (linearised plant):")
    print(f"  {'tau':>5} {'k_taps':>6} {'H1 cost-gap':>12} {'||Khist||/||Know||':>18}")
    for r in sweep:
        print(f"  {r['tau']:>5.2f} {r['k_taps']:>6d} {100*r['h1_cost_gap']:>11.2f}% "
              f"{r['kernel_ratio']:>18.4f}")
    print("\nRung-2 cross-check on the NONLINEAR plant (optimal LQR controllers):")
    for r in rung2:
        print(f"  tau={r['tau']}, tf={r['tf']}: markovian-LQR {100*r['reduction_markovian']:.1f}% "
              f"vs delayed-LQR {100*r['reduction_delayed']:.1f}%  "
              f"(history gain {100*r['history_gain_over_markovian']:+.1f}%)")

    summary = {
        "linearisation": {"A": A.tolist(), "A1": A1.tolist(), "B": B.tolist(),
                          "eig_A": [complex(e).real for e in eigs]},
        "Q": Q.tolist(), "R": R.tolist(), "dt": DT, "x0": X0.tolist(),
        "analytic_sweep": sweep, "nonlinear_rung2": rung2,
    }
    (out_dir / "delay_impact_summary.json").write_text(json.dumps(summary, indent=2))
    build_figure(sweep, out_dir)
    print(f"\n[delay-impact] wrote delay_impact_summary.json and dadebo_delay_impact.png in {out_dir}")


if __name__ == "__main__":
    main()
