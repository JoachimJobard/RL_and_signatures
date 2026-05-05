#!/usr/bin/env python
"""
Harmonic Oscillator - Riccati Sanity Check with Value Gradient.

Runs the value gradient agent on the harmonic_oscillator_rk environment
(2D LQR, no delay) to verify that the learned value function converges
to the analytical Riccati solution.

The environment has:
  A = [[0, -1], [1, 0]],  B = [[0], [1]],  Q = I,  R = 0.1

Usage:
    python run_riccati_test.py                   # seeds 1-5
    python run_riccati_test.py --seeds 1 10      # seeds 1-10
    python run_riccati_test.py --local           # run locally instead of Slurm
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Shared parameters ─────────────────────────────────────────────────────────

COMMON = {
    "env": "harmonic_oscillator_rk_jax",
    "eval.x0_test": "[1.0,0.0]",
    "eval.T_sim": "20.0",
    "eval.snapshot_interval": "50",
    "wandb.group": "harmonic_oscillator_riccati",
    "wandb.mode": "online",
    "eval_data_dir": "outputs/harmonic_oscillator_riccati_two",
    "agent.signature.window_size": "100",  
    "agent.signature.force_signature_window": "true",
}

# ── Value Gradient configuration ──────────────────────────────────────────────

VG_OVERRIDES = {
    "agent": "value_gradient",
    "agent.training.critic_lr": "1e-4",
    "agent.training.max_time": "20.0",
    "agent.training.n_episodes": "500",
    "agent.training.eval_interval": "10",
    "agent.noise.schedule": "decaying",
    "agent.noise.sigma": "0.1",
    "agent.algorithm.fix_initial_state": "true",
    "agent.algorithm.burning_steps": "5",
    "eval.burning_steps": "0",
    "wandb.name": "vg_harmonic_seed_${seed}",
    "eval_data_name": "vg_harmonic_seed_${seed}",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_cmd(seeds: str, launcher: str = "slurm") -> list[str]:
    cmd = ["python", "main_unified.py", "-m", f"launcher={launcher}"]
    all_overrides = {**COMMON, **VG_OVERRIDES}
    all_overrides["seed"] = seeds
    for k, v in all_overrides.items():
        cmd.append(f"{k}={v}")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Harmonic oscillator Riccati sanity check (value gradient)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs=2, default=[1, 5],
        metavar=("START", "END"), help="Seed range (default: 1 5)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Run locally instead of submitting to Slurm",
    )
    args = parser.parse_args()

    seeds = ",".join(str(s) for s in range(args.seeds[0], args.seeds[1] + 1))
    launcher = "local" if args.local else "slurm"

    cmd = build_cmd(seeds, launcher)

    print("=" * 60)
    print("  Harmonic Oscillator - Riccati Sanity Check")
    print("=" * 60)
    print(f"  Seeds:    {seeds}")
    print(f"  Launcher: {launcher}")
    print(f"  Cmd:      {' '.join(cmd)}")
    print()

    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
    rc = proc.wait()

    if rc == 0:
        if launcher == "slurm":
            print("Submitted to Slurm! Monitor with: squeue -u $USER")
        else:
            print("Experiment completed.")
        sys.exit(0)
    else:
        print(f"FAILED (exit code {rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
