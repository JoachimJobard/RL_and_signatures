"""
Unified main entry point for all CTAC experiments with Weights & Biases logging.

This replaces: main.py, main_signatures.py, main_base_jax.py, main_CSAC.py

Usage:
    python main_unified.py                                    # Default config
    python main_unified.py agent=signatures                   # Signature-based CTAC
    python main_unified.py agent=base_jax                     # Vanilla CTAC
    python main_unified.py agent=csac                         # Soft Actor-Critic
    python main_unified.py agent=value_gradient               # Value Gradient
    python main_unified.py agent.depth=4 env=mackey_glass     # Custom params
    python main_unified.py wandb.mode=disabled                # Local testing
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
from pathlib import Path

from src.training.train import train
from src.training.evaluate import (
    log_training_metrics,
    plot_training_metrics,
    get_training_summary,
    compare_with_no_control,
    evaluate_multiple_trajectories,
    collect_evaluation_data,
    collect_multiple_trajectories_data,
    save_evaluation_data,
    save_training_metrics,
    make_eval_callback,
    get_statistics_visited_states,
)


def run_experiment(cfg: DictConfig) -> None:
    """
    Run complete experiment: train + evaluate + log to wandb.
    
    This function orchestrates:
    1. Training the agent
    2. Logging training metrics to wandb
    3. Evaluating on test trajectories
    4. Saving evaluation data for external plotting
    """
    
    # =========================================================================
    # Training
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    # Set up periodic trajectory evaluation callback (wandb slider)
    x0_eval = np.array(cfg.eval.x0_test)
    T_sim_eval = cfg.eval.T_sim
    eval_interval = cfg.eval.get('snapshot_interval', 100)
    eval_burning_steps = cfg.eval.get('burning_steps', 0)
    
    # Build and attach the callback before training
    snapshot_cb = make_eval_callback(
        x0=x0_eval,
        T_sim=T_sim_eval,
        eval_interval=eval_interval,
        burning_steps=eval_burning_steps,
    )
    
    agent, metrics = train(cfg, eval_callback=snapshot_cb)
    
    # Log scalars (subsampled for efficiency)
    log_interval = cfg.eval.get('log_interval', 20)
    log_training_metrics(metrics, log_interval=log_interval)
    
    # Log training figure
    fig_training = plot_training_metrics(metrics)
    wandb.log({"Training Metrics": fig_training})
    
    # Training summary
    for k, v in get_training_summary(metrics).items():
        wandb.run.summary[k] = v  # type: ignore
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Get evaluation parameters
    x0_test = np.array(cfg.eval.x0_test)
    T_sim = cfg.eval.T_sim
    
    # Ensure x0_test matches environment dimension
    if len(x0_test) != agent.env.N:
        print(f"  Warning: x0_test dim ({len(x0_test)}) != env dim ({agent.env.N})")
        print("  Using random initial state")
        rng = np.random.default_rng(cfg.seed)
        x0_test = rng.standard_normal(agent.env.N)
    
    # Agent vs No Control comparison
    print("  - Comparing with no control...")
    fig_comparison, eval_metrics = compare_with_no_control(
        agent,
        x0_test,
        T_sim,
        burning_steps=eval_burning_steps,
    ) #type: ignore
    wandb.log({"Agent vs No Control": fig_comparison})
    
    #visiting statistics
    if metrics.get('state_counts', None) is not None:
        fig_visiting = get_statistics_visited_states(metrics, agent.discretization_state) #type: ignore
        wandb.log({"Visited States Distribution": fig_visiting})

    for k, v in eval_metrics.items():
        wandb.run.summary[k] = v  # type: ignore
    
    print(f"    Cost reduction: {eval_metrics['eval/cost_reduction_pct']:.1f}%")
    print(f"    Final error (agent): {eval_metrics['eval/final_error_agent']:.4f}")
    
    # Multiple trajectories evaluation
    print("  - Evaluating multiple trajectories...")
    n_eval = cfg.eval.get('n_eval_trajectories', 5)
    rng = np.random.default_rng(cfg.seed + 1000)
    x0_list = [x0_test] + [
        rng.standard_normal(agent.env.N) * np.linalg.norm(x0_test)
        for _ in range(n_eval - 1)
    ]
    
    fig_multi, multi_metrics = evaluate_multiple_trajectories(
        agent,
        x0_list,
        T_sim,
        burning_steps=eval_burning_steps,
    ) #type: ignore
    wandb.log({"Multiple Trajectories": fig_multi})
    
    for k, v in multi_metrics.items():
        wandb.run.summary[k] = v  # type: ignore
    
    print(f"    Mean cost: {multi_metrics['eval/multi_cost_mean']:.4f} ± {multi_metrics['eval/multi_cost_std']:.4f}")
    
    # =========================================================================
    # Save Checkpoint & Evaluation Data
    # =========================================================================
    if cfg.get('save_checkpoint', True):
        print("\n  - Saving checkpoint...")
        checkpoint_path = cfg.get('checkpoint_path', 'checkpoint_agent.pkl')
        if hasattr(agent, 'save'):
            agent.save(checkpoint_path) #type: ignore
        else:
            import pickle
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'agent_params': getattr(agent, 'actor_params', None),
                    'critic_params': getattr(agent, 'critic_params', None),
                }, f)
            print(f"    Saved to {checkpoint_path}")
    
    if cfg.get('save_eval_data', True):
        print("  - Saving evaluation data...")
        original_cwd = hydra.utils.get_original_cwd()
        eval_data_dir = Path(original_cwd) / cfg.get('eval_data_dir', 'eval_outputs')
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = cfg.get('eval_data_name', 'run')
        
        # Single trajectory data
        eval_data = collect_evaluation_data(
            agent,
            x0_test,
            T_sim,
            burning_steps=eval_burning_steps,
        ) #type: ignore
        eval_data['training_metrics'] = {
            'cost_episodic': np.array(metrics.get('cost_episodic', [])),
            'loss_episodic': np.array(metrics.get('loss_episodic', [])),
            'gradient_critic': np.array(metrics.get('gradient_critic', [])),
        }
        eval_data['config'] = OmegaConf.to_container(cfg, resolve=True)
        save_evaluation_data(eval_data, eval_data_dir / f"{base_name}_eval.pkl")
        
        # Multiple trajectories data
        multi_data = collect_multiple_trajectories_data(
            agent,
            x0_list,
            T_sim,
            burning_steps=eval_burning_steps,
        ) #type: ignore
        multi_data['config'] = OmegaConf.to_container(cfg, resolve=True)
        save_evaluation_data(multi_data, eval_data_dir / f"{base_name}_multi.pkl")
        
        # Training metrics (cost, loss, gradients, weights per episode)
        save_training_metrics(metrics, eval_data_dir / f"{base_name}_training_metrics.pkl")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


@hydra.main(config_path="conf", config_name="config_unified", version_base=None)
def main(cfg: DictConfig):
    """Main entry point with wandb integration."""
    
    # Initialize wandb
    wandb_cfg = cfg.wandb
    wandb.init(
        project=wandb_cfg.project_name,
        name=wandb_cfg.name,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        mode=wandb_cfg.mode
    )
    
    try:
        run_experiment(cfg)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
