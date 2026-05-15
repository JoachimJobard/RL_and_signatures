#!/usr/bin/env python
"""
MG Delay Comparison - Launch all experiments via Hydra multirun on Slurm.

Each method has different agent configs, so we launch 5 separate Hydra multiruns.
Hydra + Submitit handles parallel job submission to Slurm automatically.

Usage:
    python run_comparison.py                        # All 5 methods, seeds 1-10
    python run_comparison.py --seeds 1 3            # Seeds 1-3 only
    python run_comparison.py --methods 1 4          # Only methods 1 and 4
    python run_comparison.py --seeds 1 5 --methods 2 3
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Shared parameters ────────────────────────────────────────────────────────

COMMON = {
    "env": "MG_1D",
    "env.environment_params.x_target": "0.",
    "env.environment_params.delay": "8, 17, 30",
    "eval.x0_test": "[0.1]",
    "eval.T_sim": "1400",
    "eval.snapshot_interval": "100",
    "wandb.group": "comparison_delay_burning_more_ep_MG_15_05",
    "wandb.mode": "online",
    "eval_data_dir": "outputs/comparison_delay_burning_more_ep_MG_15_05",
    # "agent.algorithm.burning_steps": "100",
    # "eval.burning_steps": "400",

}

# ── Per-method configurations ────────────────────────────────────────────────

EXPERIMENTS = {
    1: {
        "name": "Signatures",
        "overrides": {
            "agent": "CTAC_sig_MG_1D",
            "agent.training.actor_lr": "1e-4",
            "agent.training.critic_lr": "1e-3",
            "agent.training.max_time": "1000",
            "agent.training.n_episodes": "500",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "constant",
            "agent.signature.depth": "2,3,4",
            "agent.signature.time_augmentation": "false",
            "agent.signature.origin_augmentation": "true",
            "agent.algorithm.fix_initial_state": "true",
            "wandb.name": "signature_delay_${env.environment_params.delay}_depth_${agent.signature.depth}_seed_${seed}",
            "eval_data_name": "signature_delay_${env.environment_params.delay}_depth_${agent.signature.depth}_seed_${seed}",
        },
    },
    2: {
        "name": "No Signatures",
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.fix_initial_state": "true",
            "agent.algorithm.delayed_state": "false",
            "agent.training.actor_lr": "1e-4",
            "agent.training.critic_lr": "1e-3",
            "agent.training.max_time": "1000",
            "agent.training.n_episodes": "500",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "constant",
            "wandb.name": "no_signature_delay_${env.environment_params.delay}_seed_${seed}",
            "eval_data_name": "no_signature_delay_${env.environment_params.delay}_seed_${seed}",
        },
    },
    3: {
        "name": "Augmented State",
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.fix_initial_state": "true",
            "agent.algorithm.delayed_state": "true",
            "agent.training.actor_lr": "1e-4",
            "agent.training.critic_lr": "1e-3",
            "agent.training.max_time": "1000",
            "agent.training.n_episodes": "500",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "constant",
            "wandb.name": "augmented_state_delay_${env.environment_params.delay}_seed_${seed}",
            "eval_data_name": "augmented_state_delay_${env.environment_params.delay}_seed_${seed}",
        },
    },
    4: {
        "name": "Value Gradient",
        "overrides": {
            "agent": "value_gradient",
            "agent.training.critic_lr": "1e-4",
            "agent.training.max_time": "1000",
            "agent.training.n_episodes": "500",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "decaying",
            "agent.signature.depth": "2,3,4",
            "agent.signature.time_augmentation": "false",
            "agent.signature.origin_augmentation": "true",
            "agent.algorithm.fix_initial_state": "true",
            "wandb.name": "vg_delay_${env.environment_params.delay}_depth_${agent.signature.depth}_seed_${seed}",
            "eval_data_name": "vg_delay_${env.environment_params.delay}_depth_${agent.signature.depth}_seed_${seed}",
        },
    },
    5: {
        "name": "Whole State Delay",
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.fix_initial_state": "true",
            "agent.algorithm.delayed_state": "false",
            "agent.algorithm.whole_state_delay": "true",
            "agent.training.actor_lr": "1e-4",
            "agent.training.critic_lr": "1e-3",
            "agent.training.max_time": "1000",
            "agent.training.n_episodes": "500",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "constant",
            "wandb.name": "whole_state_delay_${env.environment_params.delay}_seed_${seed}",
            "eval_data_name": "whole_state_delay_${env.environment_params.delay}_seed_${seed}",
        },
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_cmd(overrides: dict, seeds: str, delays: str = "30") -> list[str]:
    """Build the Hydra multirun command for one experiment.
    
    Uses plain 'python' since the venv is activated by Slurm setup block.
    """
    cmd = ["python", "main_unified.py", "-m", "launcher=slurm"]
    all_overrides = {**COMMON, **overrides}
    all_overrides["seed"] = seeds
    for k, v in all_overrides.items():
        cmd.append(f"{k}={v}")
    return cmd


def run_experiment(exp_id: int, seeds: str) -> subprocess.Popen:
    """Launch one experiment as a subprocess (non-blocking)."""
    exp = EXPERIMENTS[exp_id]
    cmd = build_cmd(exp["overrides"], seeds)
    print(f"  [{exp_id}] {exp['name']}")
    print(f"      cmd: {' '.join(cmd)}")
    print()
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MG Delay Comparison on Slurm")
    parser.add_argument("--seeds", type=int, nargs=2, default=[1, 10],
                        metavar=("START", "END"), help="Seed range (default: 1 10)")
    parser.add_argument("--methods", type=int, nargs="+", default=list(EXPERIMENTS),
                        metavar="ID", help="Which experiments to run (default: all)")
    args = parser.parse_args()

    seeds = ",".join(str(s) for s in range(args.seeds[0], args.seeds[1] + 1))

    print("=" * 60)
    print("  MG Comparison - Delay Analysis on Slurm")
    print("=" * 60)
    print(f"  Seeds: {seeds}")
    print(f"  Methods: {args.methods}")
    print()

    # Launch all methods in parallel (each submits its own Slurm jobs)
    processes: list[tuple[int, subprocess.Popen]] = []
    for exp_id in args.methods:
        if exp_id not in EXPERIMENTS:
            print(f"  [WARNING] Unknown method {exp_id}, skipping")
            continue
        proc = run_experiment(exp_id, seeds)
        processes.append((exp_id, proc))

    # Wait for all submissions to complete
    failed = []
    for exp_id, proc in processes:
        rc = proc.wait()
        name = EXPERIMENTS[exp_id]["name"]
        if rc == 0:
            print(f"  [{exp_id}] {name}: submitted ✓")
        else:
            print(f"  [{exp_id}] {name}: FAILED (exit code {rc})")
            failed.append(exp_id)

    print()
    if failed:
        print(f"Some experiments failed: {failed}")
        sys.exit(1)
    else:
        print("All experiments submitted to Slurm!")
        print("Monitor with: squeue -u $USER")
        sys.exit(0)

if __name__ == "__main__":
    main()
