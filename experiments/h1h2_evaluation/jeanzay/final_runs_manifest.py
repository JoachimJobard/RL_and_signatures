"""Emit the per-cell task manifest for the 5-seed final runs from the frozen learning rates.

One line per (learner, representation, seed) at the frozen alpha0 read from frozen_alpha0.yaml,
plus one representation-agnostic oracle line (the delayed-LQR / linearised-delayed optimum, run
once -- it is deterministic and seed-independent). The seed axis is SHARED across every learner
and representation (identical initial conditions per seed), so a between-representation difference
reflects the representation, not RNG. The line index IS the SLURM array task id.

Columns (tab-separated): learner  kind  actor_lr  critic_lr  seed
  value_gradient : the swept rate is critic_lr (no actor)      -> actor_lr = "na"
  policy_gradient: the swept rate is actor_lr (REINFORCE)      -> critic_lr = "na"
  actor_critic   : both actor_lr (swept) and critic_lr (co-parameter)
  oracle         : learner="oracle"; actor_lr = critic_lr = "na"

Usage:  python final_runs_manifest.py <cell> "<seed0 seed1 ...>" <path/to/frozen_alpha0.yaml>
"""
import sys
import yaml

LEARNERS = ["value_gradient", "policy_gradient", "actor_critic"]
KINDS = ["markovian", "raw_history", "signature"]


def main() -> None:
    cell, seeds_str, frozen_path = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = seeds_str.split()
    frozen = yaml.safe_load(open(frozen_path))
    if cell not in frozen:
        raise SystemExit(f"cell {cell!r} absent from {frozen_path}")
    lines = []
    for learner in LEARNERS:
        for kind in KINDS:
            cfg = frozen[cell][learner][kind]
            actor_lr = cfg.get("actor_lr", "na")
            critic_lr = cfg.get("critic_lr", "na")
            for seed in seeds:
                lines.append(f"{learner}\t{kind}\t{actor_lr}\t{critic_lr}\t{seed}")
    # Oracle: deterministic, one run, seed only labels its output folder.
    lines.append(f"oracle\tmarkovian\tna\tna\t{seeds[0]}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
