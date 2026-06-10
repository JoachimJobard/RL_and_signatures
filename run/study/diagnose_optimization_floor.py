"""Decompose the residual sub-optimality of an *exact-capacity* linear-cell variant.

On the linear delayed cell, the spans of the degree-2 raw-history and depth-2
signature feature maps both contain the delayed-LQR value functional (Kolmanovskii
eq. 2.4) — the value is taken linear in the features, V = theta . Phi — so the residual
normalised sub-optimality ``rho = (J - J_oracle)/|J_oracle|`` is *optimization* error,
not representation error (see documents/methodology/experimental_design.md §7.1). This
script isolates which sub-term dominates, for one or more run directories, by:

1. **Transient vs tail split.** Roll the delayed-LQR oracle through the *same*
   ``collect_evaluation_data`` path as the agent, form the cumulative-excess curve
   ``E(t) = cumcost_agent(t) - cumcost_oracle(t)``, and report the fraction of the
   total excess ``E(T)`` accrued within the oracle's settling time ``t_settle`` (the
   transient) vs after it (the tail). A transient-dominated excess with matched
   control amplitude localises the residual to a feedback-gain error.
2. **Feedback-gain recovery.** For a quadratic value the greedy control is linear in
   the history window, so an effective gain ``K_hat`` is recoverable by least squares
   of the recorded actions on the oracle-horizon window (newest state first). The
   same regression applied to the oracle's own actions reproduces ``lqr.gain`` (a
   pipeline self-check). Report the relative gain error ``||K_hat - K*|| / ||K*||``,
   the linear-in-window fit ``R^2`` (control linearity / horizon adequacy), and the
   constant offset (a non-zero offset with near-zero gain is the signature of a
   *representation* failure, e.g. degree-1 raw history whose linear value yields a
   state-independent control — the under-capacity contrast variant).

Usage:
    uv run python run/study/diagnose_optimization_floor.py RUN_DIR [RUN_DIR ...] --out OUT_DIR

Each RUN_DIR must contain eval.pkl and config.yaml (linear JAXDDEEnv only).
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class FloorDiagnosis:
    run: str
    variant: str
    j_agent: float
    j_oracle: float
    rho: float                    # normalised sub-optimality
    t_settle: float               # oracle settling time
    excess_total: float           # E(T) = J_agent - J_oracle
    excess_transient_fraction: float   # E(t_settle) / E(T)
    max_abs_u_agent: float
    max_abs_u_oracle: float
    action_discrepancy: float     # RMS(u_agent - K*·window_agent) / RMS(u_oracle): IDENTIFIABLE
    gain_fit_r2: float            # R^2 of the linear-in-window action fit (control linearity)
    offset_norm: float            # ||constant control offset||
    window_cond: float            # condition number of the regression design (kernel identifiability)
    k_taps: int


def _load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore


def _windows(states: np.ndarray, x0: np.ndarray, k_taps: int) -> np.ndarray:
    """Newest-first history windows xi_i = [x_i, x_{i-1}, ..., x_{i-K}] flattened,
    one per action index i, padding pre-trajectory taps with the constant x0
    (fix_initial_state ⇒ the history before t=0 is x0)."""
    n_actions = states.shape[0] - 1  # actions[i] drives states[i] -> states[i+1]
    n = states.shape[1]
    X = np.zeros((n_actions, n * (k_taps + 1)))
    for i in range(n_actions):
        taps = []
        for j in range(k_taps + 1):
            idx = i - j
            taps.append(states[idx] if idx >= 0 else x0)
        X[i] = np.concatenate(taps)
    return X


def _fit_gain(states: np.ndarray, actions: np.ndarray, x0: np.ndarray, k_taps: int):
    """Least-squares fit u_i ≈ -K_hat @ xi_i + c. Returns (K_hat, offset c, R^2)."""
    X = _windows(states, x0, k_taps)
    Y = np.asarray(actions, dtype=float).reshape(X.shape[0], -1)
    A = np.hstack([X, np.ones((X.shape[0], 1))])          # augment for the offset
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)           # (n*(K+1)+1, m)
    B_lin, c = coef[:-1], coef[-1]                         # u ≈ X @ B_lin + c
    K_hat = -B_lin.T                                       # u = -K_hat @ xi (+c)
    resid = Y - A @ coef
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return K_hat, np.asarray(c).reshape(-1), r2


def _settling_time(times: np.ndarray, states: np.ndarray, x_ref: np.ndarray,
                   x0: np.ndarray, frac: float = 0.05) -> tuple[float, int]:
    """First time the oracle error norm drops below ``frac * ||x0 - x_ref||`` and
    stays below for the remainder; returns (t_settle, index)."""
    err = np.linalg.norm(states - x_ref, axis=1)
    thresh = frac * float(np.linalg.norm(x0 - x_ref))
    below = err <= thresh
    idx = len(times) - 1
    for i in range(len(below)):
        if below[i] and below[i:].all():
            idx = i
            break
    return float(times[idx]), idx


def diagnose_run(run_dir: Path) -> FloorDiagnosis:
    import hydra
    from src.envs.env_rk_jax import JAXDDEEnv
    from src.solvers.oracle_agent import delayed_lqr_for_env, OracleDelayedLQRAgent
    from src.training.evaluate import collect_evaluation_data

    cfg = _load_yaml(run_dir / "config.yaml")
    with open(run_dir / "eval.pkl", "rb") as f:
        ev = pickle.load(f)

    env_params = cfg["env"]["environment_params"]
    env = hydra.utils.instantiate(env_params)
    if not isinstance(env, JAXDDEEnv) or type(env).__name__ != "JAXDDEEnv":
        raise SystemExit(f"{run_dir.name}: not a linear JAXDDEEnv (no analytic oracle).")

    x0 = np.asarray(ev["x0"], dtype=float)
    x_ref = np.asarray(ev["x_ref"], dtype=float)
    T_sim = float(cfg["eval"]["T_sim"])
    lqr = delayed_lqr_for_env(env)

    # Oracle rolled through the SAME eval path (greedy, deterministic) → commensurable.
    oracle_data = collect_evaluation_data(
        OracleDelayedLQRAgent(env, lqr), x0, T_sim,
        seed=int(cfg.get("seed", 0)), burning_steps=int(cfg["eval"].get("burning_steps", 0)))

    times = np.asarray(ev["times"], dtype=float)
    cum_agent = np.asarray(ev["cum_cost_agent"], dtype=float)
    cum_oracle = np.asarray(oracle_data["cum_cost_agent"], dtype=float)
    m = min(len(cum_agent), len(cum_oracle))
    excess = cum_agent[:m] - cum_oracle[:m]

    j_agent = float(ev["eval_metrics"]["eval/total_cost_agent"])
    j_oracle = float(oracle_data["eval_metrics"]["eval/total_cost_agent"])
    rho = (j_agent - j_oracle) / abs(j_oracle)

    t_settle, idx_settle = _settling_time(
        times[:m], np.asarray(oracle_data["states_agent"])[:m], x_ref, x0)
    e_total = float(excess[-1])
    e_settle = float(excess[min(idx_settle, m - 1)])
    transient_fraction = e_settle / e_total if e_total != 0 else float("nan")

    # Effective feedback gain by regression (control linearity / offset diagnosis).
    # NOTE: the multi-tap kernel is NOT uniquely identifiable from a single closed-loop
    # trajectory — the dt-spaced taps are near-collinear (window_cond is huge), so the
    # min-norm K_hat is inflated and ||K_hat - K*|| is uninformative. The identifiable
    # gain-fidelity quantity is the on-trajectory action discrepancy below.
    K_hat, offset, r2 = _fit_gain(
        np.asarray(ev["states_agent"]), np.asarray(ev["actions"]), x0, lqr.k_taps)
    K_orc_hat, _, r2_orc = _fit_gain(
        np.asarray(oracle_data["states_agent"]), np.asarray(oracle_data["actions"]),
        x0, lqr.k_taps)
    Kstar = lqr.gain
    window_cond = float(np.linalg.cond(_windows(np.asarray(ev["states_agent"]), x0, lqr.k_taps)))
    # Pipeline self-check: regressing the oracle's own actions must reproduce lqr.gain.
    pipeline_err = float(np.linalg.norm(K_orc_hat - Kstar) / np.linalg.norm(Kstar))

    # IDENTIFIABLE gain-fidelity metric: apply the oracle law K* to the AGENT's own
    # windows and compare to the agent's actual actions. RMS(u_agent - u*_on_agent) /
    # RMS(u_oracle) — how far the agent's realised control is from what the oracle
    # would do facing the same history, free of the kernel non-identifiability.
    Xw = _windows(np.asarray(ev["states_agent"]), x0, lqr.k_taps)
    u_star_on_agent = -(Xw @ Kstar.T)                     # (n_actions, m)
    u_agent = np.asarray(ev["actions"], dtype=float).reshape(Xw.shape[0], -1)
    u_oracle = np.asarray(oracle_data["actions"], dtype=float)
    rms_or = float(np.sqrt(np.mean(u_oracle ** 2)))
    action_discrepancy = float(np.sqrt(np.mean((u_agent - u_star_on_agent) ** 2)) / rms_or) \
        if rms_or > 0 else float("nan")

    variant = str(cfg["agent"]["signature"].get("kind", "?")) + (
        f"_d{cfg['agent']['signature'].get('depth')}" if cfg["agent"]["signature"].get("kind") == "signature"
        else f"_deg{cfg['agent']['signature'].get('degree')}")

    diag = FloorDiagnosis(
        run=run_dir.name, variant=variant, j_agent=j_agent, j_oracle=j_oracle, rho=rho,
        t_settle=t_settle, excess_total=e_total,
        excess_transient_fraction=transient_fraction,
        max_abs_u_agent=float(np.max(np.abs(ev["actions"]))),
        max_abs_u_oracle=float(np.max(np.abs(oracle_data["actions"]))),
        action_discrepancy=action_discrepancy, gain_fit_r2=r2,
        offset_norm=float(np.linalg.norm(offset)), window_cond=window_cond,
        k_taps=int(lqr.k_taps),
    )
    # Attach arrays for plotting (not part of the dataclass record).
    diag._plot = dict(  # type: ignore[attr-defined]
        times=times[:m], excess=excess, t_settle=t_settle,
        u_agent=u_agent.reshape(-1), u_star_on_agent=u_star_on_agent.reshape(-1),
        cost_agent=np.asarray(ev["cost_agent"]),
        cost_oracle=np.asarray(oracle_data["cost_agent"]),
        pipeline_err=pipeline_err, r2_orc=r2_orc,
    )
    return diag


def build_figure(diags: list[FloorDiagnosis], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from src.utils.plot_style import (STROKE_TRAINED, STROKE_REFERENCE,
                                       sequential_colors, prepare_figure)
    colors = sequential_colors(len(diags))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    for d, c in zip(diags, colors):
        p = d._plot  # type: ignore[attr-defined]
        axes[0].plot(p["times"], p["excess"], STROKE_TRAINED, color=c, label=d.variant)
        axes[0].axvline(p["t_settle"], color=c, ls=STROKE_REFERENCE, alpha=0.4)
        # Identifiable comparison: agent's realised control (solid) vs the oracle law
        # applied to the agent's OWN windows (dashed) — same colour per variant.
        t_u = p["times"][:len(p["u_agent"])]
        axes[1].plot(t_u, p["u_agent"], STROKE_TRAINED, color=c, alpha=0.9, label=d.variant)
        axes[1].plot(t_u, p["u_star_on_agent"], STROKE_REFERENCE, color=c, alpha=0.6)
        t = p["times"][:len(p["cost_agent"])]
        axes[2].plot(t, p["cost_agent"], STROKE_TRAINED, color=c, alpha=0.8, label=d.variant)
    p0 = diags[0]._plot  # type: ignore[attr-defined]
    t0 = p0["times"][:len(p0["cost_oracle"])]
    axes[2].plot(t0, p0["cost_oracle"], STROKE_REFERENCE, color="k", alpha=0.8,
                 label="oracle")

    axes[0].set_title("Cumulative excess $E(t)=J_{\\rm agent}-J_{\\rm oracle}$")
    axes[0].set_xlabel(r"$t$"); axes[0].set_ylabel(r"$E(t)$"); axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Agent control vs oracle-law-on-agent-window")
    axes[1].set_xlabel(r"$t$"); axes[1].set_ylabel(r"$u(t)$")
    axes[1].grid(True, alpha=0.3)
    axes[2].set_title("Instantaneous cost rate")
    axes[2].set_xlabel(r"$t$"); axes[2].set_ylabel(r"$c(t)$"); axes[2].grid(True, alpha=0.3)

    prepare_figure(
        fig, fname="optimization_floor_diagnosis", axes=list(axes), reserve_bottom=0.24,
        legend_fontsize=7,
        formula=(r"Exact-capacity variants span the delayed-LQR value class, so the "
                 r"residual is optimization error. Panel 1 dashed vertical = oracle "
                 r"settling time. Panel 2: solid = agent's realised $u$, dashed = "
                 r"oracle law $K^\star$ on the agent's own window (identifiable; the "
                 r"raw kernel is not, window cond $\gg 1$)."))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    diags = [diagnose_run(rd) for rd in args.run_dirs]

    hdr = (f"{'variant':16s} {'rho':>8s} {'E_total':>9s} {'transient%':>10s} "
           f"{'|u|ag/|u|or':>12s} {'u_discrep':>9s} {'R2':>7s} {'offset':>7s} {'winCond':>9s}")
    print("\n" + hdr); print("-" * len(hdr))
    for d in diags:
        print(f"{d.variant:16s} {d.rho:8.4f} {d.excess_total:9.4f} "
              f"{100*d.excess_transient_fraction:9.1f}% "
              f"{d.max_abs_u_agent:5.2f}/{d.max_abs_u_oracle:<5.2f} "
              f"{d.action_discrepancy:9.3f} {d.gain_fit_r2:7.3f} {d.offset_norm:7.4f} "
              f"{d.window_cond:9.1e}")
    print(f"\n[self-check] oracle-action regression reproduces lqr.gain to "
          f"rel-err {diags[0]._plot['pipeline_err']:.2e} (R^2={diags[0]._plot['r2_orc']:.4f})")

    (args.out / "optimization_floor_diagnosis.json").write_text(
        json.dumps([asdict(d) for d in diags], indent=2))
    build_figure(diags, args.out / "optimization_floor_diagnosis.png")
    print(f"[diagnose] wrote optimization_floor_diagnosis.json/.png in {args.out}")


if __name__ == "__main__":
    main()
