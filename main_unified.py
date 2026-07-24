"""
Unified main entry point for all CTAC experiments with Weights & Biases logging.

This replaces: main.py, main_signatures.py, main_base_jax.py, main_CSAC.py

Usage:
    python main_unified.py                                    # Default config
    python main_unified.py agent=actor_critic                   # continuous-time actor-critic
    python main_unified.py agent=base_jax                     # Vanilla CTAC
    python main_unified.py agent=value_gradient               # Value Gradient
    python main_unified.py agent.depth=4 env=mackey_glass     # Custom params
    python main_unified.py wandb.mode=disabled                # Local testing
    python main_unified.py debug=true agent.training.n_episodes=2   # Exploratory smoke run
    python main_unified.py replot=data/main_unified/<run_dir>      # Rebuild figures from saved data

Outputs follow the scientific-workflow convention (see
documents/methodology/scientific_workflow.md): canonical artefacts (run context,
resolved config, checkpoint, evaluation data, training history, figures) are
written to

    data/main_unified/<timestamp>_<agent>_<env>_seed<seed>/

with a ``_debug_`` prefix when ``debug=true``. The run directory is derived from
this script's filename via ``run_context.resolve_run_dir``.
"""

import pickle
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import numpy as np
import jax
import wandb

from src.training.train import train
from src.training.evaluate import (
    log_training_metrics,
    plot_training_metrics,
    get_training_summary,
    collect_evaluation_data,
    collect_multiple_trajectories_data,
    plot_agent_vs_no_control_from_data,
    plot_multiple_trajectories_from_data,
    save_evaluation_data,
    load_evaluation_data,
    save_training_metrics,
    load_training_metrics,
    make_training_snapshot_callback,
    get_statistics_visited_states,
    EvaluableAgent,
)
from src.utils.run_context import (
    resolve_run_dir,
    derive_seeds,
    capture_run_context,
    format_run_context,
)

# Enable float64 for JAX
jax.config.update("jax_enable_x64", True)

# Smoke-test guard: real runs use thousands of episodes. A run with fewer episodes
# than this threshold is refused unless flagged as exploratory (debug=true), so a
# smoke run cannot silently land in the real-run namespace.
SMOKE_TEST_N_EPISODES_THRESHOLD = 100

# Roles whose seeds are derived from the single master seed (cfg.seed).
SEED_ROLES = ("agent", "eval_init", "eval_x0_fallback")


def _training_cfg(cfg: DictConfig) -> Any:
    """Return the agent's training block, tolerating both the nested
    (``training:``) and legacy flat (``training_params:``) config schemas."""
    block = cfg.agent.get("training", None)
    if block is None:
        block = cfg.agent.get("training_params", None)
    return block if block is not None else {}


def _save_figure(fig: Any, path_without_ext: Path) -> None:
    """Persist a Matplotlib figure to disk as PNG (so figures survive without wandb).

    The builders already place the external legend and formula box via
    ``src.utils.plot_style.prepare_figure`` and run the layout-overlap check; saving
    with ``bbox_inches="tight"`` keeps those figure-level artists from being clipped.
    """
    if fig is None:
        return
    if hasattr(fig, "savefig"):  # Matplotlib
        fig.savefig(str(path_without_ext) + ".png", dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)


def run_experiment(cfg: DictConfig, run_dir: Path, derived_seeds: dict) -> None:
    """Run a complete experiment: train, evaluate, and persist artefacts.

    Artefacts (checkpoint, evaluation data, training history, figures) are written
    to ``run_dir`` and, when wandb is enabled, also logged to Weights & Biases.
    """

    # =========================================================================
    # Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    # Lightweight on-disk training snapshot (wandb-free): every snapshot_interval
    # episodes it overwrites run_dir/training_snapshot.png (controlled state,
    # control signal, eval-cost vs episode) + a .npz for replotting. Watch it live
    # with: open run_dir/training_snapshot.png.
    x0_eval = np.array(cfg.eval.x0_test)
    T_sim_eval = cfg.eval.T_sim
    eval_interval = cfg.eval.get("snapshot_interval", 100)
    eval_burning_steps = cfg.eval.get("burning_steps", 0)

    snapshot_cb = make_training_snapshot_callback(
        x0=x0_eval,
        T_sim=T_sim_eval,
        run_dir=run_dir,
        interval=eval_interval,
        burning_steps=eval_burning_steps,
    )

    agent, metrics = train(cfg, eval_callback=snapshot_cb)
    # The trained agent satisfies both the TrainableAgent and EvaluableAgent
    # protocols at runtime; expose it under the evaluation interface for the
    # evaluation/plotting helpers below (no runtime change — same object).
    agent_eval = cast(EvaluableAgent, agent)

    # Log scalars (subsampled for efficiency)
    log_interval = cfg.eval.get("log_interval", 20)
    log_training_metrics(metrics, log_interval=log_interval)

    # Training figure
    fig_training = plot_training_metrics(metrics)
    wandb.log({"Training Metrics": fig_training})
    _save_figure(fig_training, run_dir / "figure_training_metrics")

    for k, v in get_training_summary(metrics).items():
        wandb.run.summary[k] = v  # type: ignore

    # =========================================================================
    # Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    x0_test = np.array(cfg.eval.x0_test)
    T_sim = cfg.eval.T_sim

    # Ensure x0_test matches environment dimension
    if len(x0_test) != agent.env.N:
        print(f"  Warning: x0_test dim ({len(x0_test)}) != env dim ({agent.env.N})")
        print("  Using random initial state")
        rng = np.random.default_rng(derived_seeds["eval_x0_fallback"])
        x0_test = rng.standard_normal(agent.env.N)

    # --- Collect evaluation data ONCE: it is the single source for both the saved
    #     artefacts and the figures, so every figure is rebuildable a posteriori. ---
    print("  - Collecting evaluation data (agent vs no control)...")
    eval_data = collect_evaluation_data(
        agent_eval, x0_test, T_sim, burning_steps=eval_burning_steps,
    )
    eval_metrics = eval_data["eval_metrics"]

    print("  - Evaluating multiple trajectories...")
    n_eval = cfg.eval.get("n_eval_trajectories", 5)
    rng = np.random.default_rng(derived_seeds["eval_init"])
    x0_list = [x0_test] + [
        rng.standard_normal(agent.env.N) * np.linalg.norm(x0_test)
        for _ in range(n_eval - 1)
    ]
    multi_data = collect_multiple_trajectories_data(
        agent_eval, x0_list, T_sim, burning_steps=eval_burning_steps,
    )
    multi_metrics = multi_data["metrics"]

    # --- Save the data FIRST, so the figures are provably rebuildable from it ---
    if cfg.get("save_eval_data", True):
        eval_to_save = dict(eval_data)
        eval_to_save["training_metrics"] = {
            "cost_episodic": np.array(metrics.get("cost_episodic", [])),
            "loss_episodic": np.array(metrics.get("loss_episodic", [])),
            "gradient_critic": np.array(metrics.get("gradient_critic", [])),
        }
        eval_to_save["config"] = OmegaConf.to_container(cfg, resolve=True)
        save_evaluation_data(eval_to_save, run_dir / "eval.pkl")

        multi_to_save = dict(multi_data)
        multi_to_save["config"] = OmegaConf.to_container(cfg, resolve=True)
        save_evaluation_data(multi_to_save, run_dir / "multi.pkl")
        save_training_metrics(metrics, run_dir / "training_metrics.pkl")

    # --- Build the figures FROM the (saved) data — identical to what replot produces ---
    # A DIVERGED run (guard-aborted) can leave non-finite values in the evaluation trajectory whose
    # axis limits break Matplotlib's tick locator (ValueError: arange: cannot compute length in
    # ticker._raw_ticks). The data is already persisted above, so a figure failure must NOT abort the
    # run: it is logged and skipped so the cost print, the wandb summary, and the checkpoint below
    # still execute and the task exits cleanly. Such a figure is regenerable via --replot. This guards
    # only the plotting; a divergence itself is already reported by the guard and the negative cost.
    try:
        fig_comparison = plot_agent_vs_no_control_from_data(eval_data)
        wandb.log({"Agent vs No Control": fig_comparison})
        _save_figure(fig_comparison, run_dir / "figure_agent_vs_no_control")

        fig_multi = plot_multiple_trajectories_from_data(multi_data)
        wandb.log({"Multiple Trajectories": fig_multi})
        _save_figure(fig_multi, run_dir / "figure_multiple_trajectories")

        if metrics.get("state_counts", None) is not None:
            fig_visiting = get_statistics_visited_states(metrics, agent.discretization_state)  # type: ignore
            wandb.log({"Visited States Distribution": fig_visiting})
            _save_figure(fig_visiting, run_dir / "figure_visited_states")
    except Exception as figure_error:
        print(f"    [WARNING] figure generation failed (non-fatal; eval.pkl/multi.pkl already "
              f"saved, regenerable via --replot): {type(figure_error).__name__}: {figure_error}")

    for k, v in eval_metrics.items():
        wandb.run.summary[k] = v  # type: ignore
    for k, v in multi_metrics.items():
        wandb.run.summary[k] = v  # type: ignore
    print(f"    Cost reduction: {eval_metrics['eval/cost_reduction_pct']:.1f}%")
    print(f"    Final error (agent): {eval_metrics['eval/final_error_agent']:.4f}")
    print(f"    Mean cost: {multi_metrics['eval/multi_cost_mean']:.4f} "
          f"± {multi_metrics['eval/multi_cost_std']:.4f}")

    # --- Checkpoint (stable filename so --replot / resume can find it) ---
    if cfg.get("save_checkpoint", True):
        print("\n  - Saving checkpoint...")
        checkpoint_path = run_dir / "checkpoint_agent.pkl"
        if hasattr(agent, "save"):
            agent.save(str(checkpoint_path))  # type: ignore
        else:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({
                    "agent_params": getattr(agent, "actor_params", None),
                    "critic_params": getattr(agent, "critic_params", None),
                }, f)
        print(f"    Saved to {checkpoint_path}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


def run_replot(run_dir: Path) -> None:
    """Rebuild EVERY figure from the saved data, without retraining.

    All figures are pure functions of the saved arrays (eval.pkl, multi.pkl,
    training_metrics.pkl), so they can be regenerated and restyled a posteriori
    (colours, titles, labels) by editing the builders and re-running this — the
    underlying data is never lost. Builders accept style overrides (e.g. a
    ``title=`` argument) for ad-hoc restyling.
    """
    print(f"\n[replot] Rebuilding figures from {run_dir}")
    n = 0

    metrics_path = run_dir / "training_metrics.pkl"
    if metrics_path.exists():
        metrics = load_training_metrics(metrics_path)
        _save_figure(plot_training_metrics(metrics), run_dir / "figure_training_metrics"); n += 1
        if metrics.get("state_counts", None) is not None:
            disc = metrics.get("discretization_state", 0.01)
            _save_figure(get_statistics_visited_states(metrics, disc),
                         run_dir / "figure_visited_states"); n += 1

    eval_path = run_dir / "eval.pkl"
    if eval_path.exists():
        eval_data = load_evaluation_data(eval_path)
        _save_figure(plot_agent_vs_no_control_from_data(eval_data),
                     run_dir / "figure_agent_vs_no_control"); n += 1

    multi_path = run_dir / "multi.pkl"
    if multi_path.exists():
        multi_data = load_evaluation_data(multi_path)
        _save_figure(plot_multiple_trajectories_from_data(multi_data),
                     run_dir / "figure_multiple_trajectories"); n += 1

    if n == 0:
        raise FileNotFoundError(
            f"No saved data (eval.pkl / multi.pkl / training_metrics.pkl) in {run_dir}; "
            "cannot replot. Pass the run directory of a completed run."
        )
    print(f"[replot] Rebuilt {n} figure(s) in {run_dir}")


@hydra.main(config_path="conf", config_name="config_unified", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point with reproducible run context and wandb integration."""

    # --- Replot mode: rebuild figures from an existing run directory and exit ---
    replot_target = cfg.get("replot", None)
    if replot_target:
        run_replot(Path(hydra.utils.get_original_cwd()) / replot_target
                   if not Path(replot_target).is_absolute() else Path(replot_target))
        return

    # --- Resolve seeds: single master seed -> deterministic per-role seeds ---
    master_seed = int(cfg.seed)
    derived_seeds = derive_seeds(master_seed, SEED_ROLES)

    # --- Smoke-test guard ---
    debug = bool(cfg.get("debug", False))
    n_episodes = int(_training_cfg(cfg).get("n_episodes", 0))
    # An analytic-oracle run performs NO policy learning: actor_oracle substitutes the closed-form
    # optimal control for the actor, so its episode count is a simulation budget, not a training
    # budget, and the smoke-test threshold is meaningless for it. Without this exemption the oracle
    # reference line that every sweep manifest carries (20 episodes at debug=false) fails the guard,
    # which has now cost two campaigns a separate hand-run oracle pass. Flagging the oracle
    # debug=true instead is NOT the fix: that would bury the campaign's %opt denominator in a
    # _debug_-prefixed directory, excluded from aggregation and swept by the debug cleanup.
    algorithm_block = cfg.agent.get("algorithm", None) or {}
    is_analytic_oracle_run = bool(algorithm_block.get("actor_oracle", False))
    if not debug and not is_analytic_oracle_run and 0 < n_episodes < SMOKE_TEST_N_EPISODES_THRESHOLD:
        raise SystemExit(
            f"Refusing a real run with n_episodes={n_episodes} "
            f"(< {SMOKE_TEST_N_EPISODES_THRESHOLD}). Set debug=true to flag it as exploratory."
        )

    # --- Canonical run directory ---
    # data/main_unified/[<experiment_group>/]<ts>_<config_tag>_seed<seed>/
    # config_tag defaults to <agent>_<env>; an ablation/sweep task overrides run_tag
    # (per-variant label) and experiment_group (shared parent) so its folder is
    # distinctly named and grouped — see the job-array launcher.
    choices = HydraConfig.get().runtime.choices
    agent_name = choices.get("agent", "agent")
    env_name = choices.get("env", "env")
    run_tag = cfg.get("run_tag", None)
    experiment_group = cfg.get("experiment_group", None)
    config_tag = str(run_tag) if run_tag else f"{agent_name}_{env_name}"
    run_dir = resolve_run_dir(__file__, config_tag, seed=master_seed, debug=debug,
                              subdir=str(experiment_group) if experiment_group else None)

    # Log the signature window the agents ACTUALLY run. When force_signature_window is false the
    # window is auto-derived from the plant delay (signature_window_size); derive it HERE, at config
    # time, and write it into cfg BEFORE the run context and config.yaml are saved -- otherwise they
    # record the config default (e.g. 40), never the derived value (e.g. 23). This is also the single
    # source of the window: both learners consume it, removing the old value-gradient(+3)/actor-
    # critic(+1) split, a cross-learner confound at fixed representation.
    _sig_cfg = cfg.agent.get("signature", None)
    if _sig_cfg is not None and not bool(_sig_cfg.get("force_signature_window", False)):
        from src.training.train import build_environment
        from src.representations.factory import signature_window_size
        cfg.agent.signature.window_size = signature_window_size(build_environment(cfg))

    # --- Self-contained run context: persist + log ---
    context = capture_run_context(
        master_seed=master_seed,
        derived_seeds=derived_seeds,
        # to_container returns a broad union; the top-level config resolves to a
        # mapping at runtime, so narrow it for capture_run_context's signature.
        hyperparameters=cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)),
        extra={"run_dir": str(run_dir), "agent": agent_name, "env": env_name},
    )
    (run_dir / "run_context.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(context)))
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    print(format_run_context(context))
    print(f"\nRun directory: {run_dir}")
    print(f"Follow progress (this run's log is on stdout): {run_dir}\n")

    # --- Weights & Biases ---
    wandb_cfg = cfg.wandb
    wandb.init(
        project=wandb_cfg.project_name,
        name=wandb_cfg.name,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        mode=wandb_cfg.mode,
    )

    try:
        run_experiment(cfg, run_dir, derived_seeds)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
