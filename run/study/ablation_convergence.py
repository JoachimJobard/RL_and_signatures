"""Convergence-timing analysis for the trick-ablation study.

For each run in an experiment group (one trick variant x seed), read the per-episode
noiseless-evaluation cost logged during training (``metrics_history['eval_cost']`` =
integrated reward = -J, with ``eval_episodes`` the episode index), convert it to the
normalised sub-optimality vs the LQR oracle ``rho(ep) = (J_agent(ep) - J_oracle)/
|J_oracle|``, average over seeds per variant, and report the **episodes to reach a
near-optimal controller** (first episode with rho below a threshold). Produces a
rho-vs-episode figure (log-y) and a summary table, so the minimal episode budget and
the effect of each removable trick can be read off.

Usage:
    uv run python run/study/ablation_convergence.py GROUP_DIR [--threshold 0.1] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class VariantConvergence:
    variant: str
    n_seeds: int
    episodes_to_threshold: float   # mean episode index where rho first < threshold (nan if never)
    final_rho_mean: float
    final_rho_ci95: float
    threshold: float


def _load_yaml(path: Path) -> dict:
    from omegaconf import OmegaConf
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)  # type: ignore


def _oracle_cost(cfg: dict, cache: dict) -> float:
    """LQR-oracle integrated cost for this run's env + eval IC (cached by env+x0)."""
    import hydra
    from src.solvers.oracle_agent import delayed_lqr_for_env, oracle_total_cost
    env_params = cfg["env"]["environment_params"]
    x0 = np.asarray(cfg["eval"]["x0_test"], dtype=float)
    T_sim = float(cfg["eval"]["T_sim"])
    key = (json.dumps(env_params, sort_keys=True, default=str),
           ",".join(f"{v:.4f}" for v in x0), T_sim)
    if key not in cache:
        env = hydra.utils.instantiate(env_params)
        lqr = delayed_lqr_for_env(env)
        cache[key] = oracle_total_cost(env, lqr, x0, T_sim)
    return cache[key]


def _first_below(episodes: np.ndarray, rho: np.ndarray, thr: float) -> float:
    idx = np.where(rho < thr)[0]
    return float(episodes[idx[0]]) if len(idx) else float("nan")


def collect(group_dir: Path):
    """Return {variant: {'episodes': arr, 'rho_seeds': [arr,...]}} and the oracle cache."""
    by_variant: dict[str, dict] = {}
    cache: dict = {}
    for run_dir in sorted(p for p in group_dir.iterdir() if p.is_dir() and p.name != "slurm"):
        cfg_path, m_path = run_dir / "config.yaml", run_dir / "training_metrics.pkl"
        if not (cfg_path.exists() and m_path.exists()):
            continue
        cfg = _load_yaml(cfg_path)
        with open(m_path, "rb") as f:
            metrics = pickle.load(f)
        ec_raw, ep_raw = metrics.get("eval_cost"), metrics.get("eval_episodes")
        if ec_raw is None or ep_raw is None or len(ec_raw) == 0 or len(ep_raw) == 0:
            continue
        eval_cost = np.asarray(ec_raw, dtype=float)   # = integrated reward = -J
        episodes = np.asarray(ep_raw, dtype=float)
        j_oracle = _oracle_cost(cfg, cache)
        rho = (-eval_cost - j_oracle) / abs(j_oracle)
        variant = str(cfg.get("run_tag") or "?")
        d = by_variant.setdefault(variant, {"episodes": episodes, "rho_seeds": []})
        # Align lengths across seeds (defensive; same eval_interval ⇒ same grid).
        n = min(len(d["episodes"]), len(episodes))
        d["episodes"] = d["episodes"][:n]
        d["rho_seeds"] = [r[:n] for r in d["rho_seeds"]] + [rho[:n]]
    return by_variant, cache


def build_figure(by_variant: dict, threshold: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from src.utils.plot_style import STROKE_TRAINED, STROKE_AUXILIARY, sequential_colors, prepare_figure
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sequential_colors(max(1, len(by_variant)))
    for (variant, d), c in zip(sorted(by_variant.items()), colors):
        R = np.vstack(d["rho_seeds"])
        mean = R.mean(axis=0)
        ax.plot(d["episodes"], np.maximum(mean, 1e-4), STROKE_TRAINED, color=c, label=variant)
    ax.axhline(threshold, color="k", ls=STROKE_AUXILIARY, alpha=0.6,
               label=f"threshold={threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("episode")
    ax.set_ylabel(r"$\rho = (J_{\rm agent}-J_{\rm oracle})/|J_{\rm oracle}|$")
    ax.set_title("Convergence to near-optimal (markovian, simple linear plant)")
    ax.grid(True, alpha=0.3, which="both")
    prepare_figure(fig, fname="ablation_convergence", axes=[ax], reserve_bottom=0.20,
                   legend_fontsize=8,
                   formula=(r"Normalised sub-optimality vs the LQR oracle, mean over seeds; "
                            r"episodes-to-threshold = time to a near-optimal controller."))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.group_dir
    out.mkdir(parents=True, exist_ok=True)

    by_variant, _ = collect(args.group_dir)
    if not by_variant:
        raise SystemExit("No runs with eval_cost history found.")

    records = []
    for variant, d in sorted(by_variant.items()):
        R = np.vstack(d["rho_seeds"])
        ep = d["episodes"]
        t2 = [_first_below(ep, R[i], args.threshold) for i in range(R.shape[0])]
        final = R[:, -1]
        n = R.shape[0]
        sem = float(np.std(final, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        records.append(VariantConvergence(
            variant=variant, n_seeds=n,
            episodes_to_threshold=float(np.nanmean(t2)),
            final_rho_mean=float(np.mean(final)), final_rho_ci95=float(1.96 * sem),
            threshold=args.threshold))

    hdr = f"{'variant':16s} {'seeds':>5s} {'ep->near-opt':>13s} {'final rho':>11s}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in sorted(records, key=lambda r: r.episodes_to_threshold):
        e = "never" if np.isnan(r.episodes_to_threshold) else f"{r.episodes_to_threshold:.0f}"
        print(f"{r.variant:16s} {r.n_seeds:5d} {e:>13s} "
              f"{r.final_rho_mean:7.3f}±{r.final_rho_ci95:.3f}")

    (out / "ablation_convergence.json").write_text(json.dumps([asdict(r) for r in records], indent=2))
    build_figure(by_variant, args.threshold, out / "ablation_convergence.png")
    print(f"\n[ablation] wrote ablation_convergence.json/.png in {out}")


if __name__ == "__main__":
    main()
