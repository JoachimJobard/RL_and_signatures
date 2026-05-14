#!/usr/bin/env python
"""
Chemical Process Comparison - launch all 5 experiment families via Hydra multirun.

This script consolidates:
- run_exp1_signatures.sh
- run_exp2_no_signatures.sh
- run_exp3_augmented_state.sh
- run_exp4_value_gradient.sh
- run_exp5_whole_state.sh

Usage:
    python run_comparison_chemical.py
    python run_comparison_chemical.py --methods 1 4
    python run_comparison_chemical.py --seeds 100 120
    python run_comparison_chemical.py --methods 2 5 --seeds 101 105
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Shared parameters across all experiments.
COMMON = {
    "env": "chemical_reaction_4D",
    "env.environment_params.x_target": "0.",
    "eval.x0_test": "[0.15,-0.03,0.1,0.0]",
    "eval.T_sim": "2",
    "eval.snapshot_interval": "100",
    "wandb.group": "comparison_signature_chemical_process_14_05",
    "wandb.mode": "online",
    "agent.algorithm.fix_initial_state": "true",
    "agent.algorithm.preheat": "true",
    "agent.algorithm.burning_steps": "5",
    "agent.training.max_time": "3",
}

EXPERIMENTS = {
    1: {
        "name": "Signatures",
        "default_seed_range": (100, 150),
        "overrides": {
            "agent": "CTAC_sig_chemical",
            "agent.training.critic_lr": "1e-3",
            "agent.training.actor_lr": "5e-2",
            "agent.signature.depth": "2,3,4",
            "agent.training.n_episodes": "1000",
            "agent.discount.V_bad": "-0.1",
            "agent.discount.V_target": "0.",
            "agent.network.std_init": "0.2",
            "agent.training.eval_interval": "20",
            "agent.algorithm.actor_update_frequency": "2",
            "agent.noise.sigma": "0.1",
            "agent.noise.schedule": "constant",
            "agent.signature.time_augmentation": "false",
            "agent.signature.origin_augmentation": "true",
            "eval_data_dir": "outputs/comparison_signatures_chemical_process_14_05",
            "wandb.name": "signature_depth_${agent.signature.depth}_seed_${seed}",
            "eval_data_name": "signature_depth_${agent.signature.depth}_seed_${seed}",
        },
    },
    2: {
        "name": "No Signatures",
        "default_seed_range": (100, 110),
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.delayed_state": "false",
            "agent.training.critic_lr": "1e-3",
            "agent.training.actor_lr": "1e-2",
            "agent.training.n_episodes": "1500",
            "agent.noise.sigma": "0.2",
            "agent.discount.V_bad": "-0.1",
            "agent.discount.V_target": "0.",
            "agent.algorithm.actor_update_frequency": "2",
            "agent.network.std_init": "0.2",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "decaying",
            "eval_data_dir": "outputs/comparison_signatures_chemical_process_14_05",
            "wandb.name": "no_signature_seed_${seed}",
            "eval_data_name": "base_no_delay_seed_${seed}",
        },
    },
    3: {
        "name": "Augmented State",
        "default_seed_range": (100, 150),
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.delayed_state": "true",
            "agent.training.critic_lr": "1e-3",
            "agent.training.actor_lr": "1e-2",
            "agent.training.n_episodes": "1500",
            "agent.noise.sigma": "0.2",
            "agent.discount.V_bad": "-0.1",
            "agent.discount.V_target": "0.",
            "agent.network.std_init": "0.1",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "decaying",
            "agent.algorithm.actor_update_frequency": "2",
            "eval_data_dir": "outputs/comparison_signatures_chemical_process_14_05",
            "wandb.name": "no_signature_augmented_state_seed_${seed}",
            "eval_data_name": "base_augmented_state_seed_${seed}",
        },
    },
    4: {
        "name": "Value Gradient",
        "default_seed_range": (100, 150),
        "overrides": {
            "agent": "value_gradient",
            "agent.training.n_episodes": "1300",
            "agent.training.critic_lr": "1e-3",
            "agent.discount.discounted": "false",
            "agent.noise.sigma": "0.4",
            "agent.signature.depth": "4",
            "agent.signature.time_augmentation": "false",
            "agent.signature.origin_augmentation": "true",
            "agent.discount.V_bad": "-0.1",
            "agent.discount.V_target": "1",
            "agent.network.std_init": "0.1",
            "agent.training.eval_interval": "10",
            "agent.noise.schedule": "linear_decay",
            "eval_data_dir": "outputs/comparison_signatures_chemical_process_14_05",
            "wandb.name": "vg_signature_depth_${agent.signature.depth}_seed_${seed}",
            "eval_data_name": "value_gradient_signature_depth_${agent.signature.depth}_seed_${seed}",
        },
    },
    5: {
        "name": "Whole State Delay",
        "default_seed_range": (100, 150),
        "overrides": {
            "agent": "CTAC_jax",
            "agent.algorithm.delayed_state": "false",
            "agent.algorithm.whole_state_delay": "true",
            "agent.training.critic_lr": "1e-3",
            "agent.training.actor_lr": "1e-2",
            "agent.training.n_episodes": "1500",
            "agent.noise.sigma": "0.2",
            "agent.discount.V_bad": "-0.1",
            "agent.discount.V_target": "0.",
            "agent.network.std_init": "0.1",
            "agent.training.eval_interval": "20",
            "agent.noise.schedule": "decaying",
            "agent.algorithm.actor_update_frequency": "2",
            "eval_data_dir": "outputs/comparison_signatures_chemical_process_14_05",
            "wandb.name": "whole_state_delay_seed_${seed}",
            "eval_data_name": "whole_state_delay_seed_${seed}",
        },
    },
}


def build_seed_csv(seed_range: tuple[int, int]) -> str:
    start, end = seed_range
    if end < start:
        raise ValueError(f"Invalid seed range: {start}..{end}")
    return ",".join(str(seed) for seed in range(start, end + 1))


def build_cmd(overrides: dict[str, str], seed_csv: str) -> list[str]:
    cmd = ["uv", "run", "python", "main_unified.py", "-m", "launcher=slurm"]
    all_overrides = {**COMMON, **overrides, "seed": seed_csv}
    for key, value in all_overrides.items():
        cmd.append(f"{key}={value}")
    return cmd


def run_experiment(exp_id: int, seed_range: tuple[int, int]) -> subprocess.Popen[bytes]:
    exp = EXPERIMENTS[exp_id]
    seed_csv = build_seed_csv(seed_range)
    cmd = build_cmd(exp["overrides"], seed_csv)

    print(f"  [{exp_id}] {exp['name']}")
    print(f"      seeds: {seed_range[0]}..{seed_range[1]}")
    print(f"      cmd: {' '.join(cmd)}")
    print()

    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chemical process comparison via Hydra multirun")
    parser.add_argument(
        "--methods",
        type=int,
        nargs="+",
        default=list(EXPERIMENTS.keys()),
        metavar="ID",
        help="Methods to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Override seed range for all selected methods",
    )

    args = parser.parse_args()

    print("=" * 64)
    print("  Chemical Process Comparison")
    print("=" * 64)
    print(f"  Methods: {args.methods}")
    if args.seeds is None:
        print("  Seed ranges: per-method defaults from bash scripts")
    else:
        print(f"  Seed range override: {args.seeds[0]}..{args.seeds[1]}")
    print()

    processes: list[tuple[int, subprocess.Popen[bytes]]] = []

    for exp_id in args.methods:
        if exp_id not in EXPERIMENTS:
            print(f"  [WARNING] Unknown method {exp_id}, skipping")
            continue

        seed_range = tuple(args.seeds) if args.seeds is not None else EXPERIMENTS[exp_id]["default_seed_range"]
        proc = run_experiment(exp_id, seed_range)
        processes.append((exp_id, proc))

    if not processes:
        print("No valid methods to run.")
        sys.exit(1)

    failed: list[int] = []
    for exp_id, proc in processes:
        return_code = proc.wait()
        exp_name = EXPERIMENTS[exp_id]["name"]
        if return_code == 0:
            print(f"  [{exp_id}] {exp_name}: submitted/finished ok")
        else:
            print(f"  [{exp_id}] {exp_name}: FAILED (exit code {return_code})")
            failed.append(exp_id)

    print()
    if failed:
        print(f"Some experiments failed: {failed}")
        sys.exit(1)

    print("All selected experiment families launched successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()
