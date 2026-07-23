"""Emit the per-cell learning-rate selection manifests (one TSV per cell).

Pre-registered selection sweep (see ../SELECTION_AND_FINAL_PROTOCOL.md). One line per
(learner, representation, learning-rate-grid-point, seed). The exploration-noise standard deviation
sigma is NOT a manifest column: it is the array-level SIGMA env var, so the same grid is launched
once per sigma in {0.3, 0.5} (representation-neutral per-cell sigma selection happens at aggregation).

The learning-rate grid is IDENTICAL across representations within a learner, so no representation is
advantaged by a finer grid. It is also identical across cells (the cell only sets env/gamma), so the
three per-cell TSVs share the same grid; they are written separately because the launcher reads
selection_tasks_<cell>.tsv per cell.

Columns (tab-separated, same schema as the final worker): learner  kind  actor_lr  critic_lr  seed
  value_gradient : swept critic_lr (no actor)                 -> actor_lr  = "na"
  policy_gradient: swept actor_lr  (no critic)                -> critic_lr = "na"
  actor_critic   : 2-D grid (actor_lr x critic_lr)
  oracle         : one representation-agnostic line per cell (delayed-LQR optimum; seed labels it)

Usage:  python selection_manifest.py "<seed0 seed1 ...>"
        (defaults to "40 41 42" -- the selection seeds, disjoint from the test seeds 0..4)
"""
import os
import sys

CELLS = ["harmonic_oscillator", "linear_dde_scalar", "MG_1D_limit_cycle"]
KINDS = ["markovian", "raw_history", "signature"]

# Learning-rate grids (identical across representations within a learner; run at each sigma).
VG_CRITIC_LR = [0.03, 0.1, 0.3, 1]          # value_gradient: the swept rate is critic_lr
PG_ACTOR_LR = [0.01, 0.1, 1, 10]            # policy_gradient: the swept rate is actor_lr
AC_ACTOR_LR = [0.3, 3, 30]                  # actor_critic: 2-D grid ...
AC_CRITIC_LR = [0.01, 1]                     # ... x critic_lr


def grid_for(learner: str) -> list[tuple[str, str]]:
    """(actor_lr, critic_lr) grid points for the learner, as manifest strings ("na" where absent)."""
    if learner == "value_gradient":
        return [("na", str(lr)) for lr in VG_CRITIC_LR]
    if learner == "policy_gradient":
        return [(str(lr), "na") for lr in PG_ACTOR_LR]
    if learner == "actor_critic":
        return [(str(a), str(c)) for a in AC_ACTOR_LR for c in AC_CRITIC_LR]
    raise ValueError(learner)


LEARNERS = ["value_gradient", "policy_gradient", "actor_critic"]


def main() -> None:
    seeds_str = sys.argv[1] if len(sys.argv) > 1 else "40 41 42"
    seeds = seeds_str.split()
    here = os.path.dirname(os.path.abspath(__file__))
    total = 0
    for cell in CELLS:
        rows = []
        for learner in LEARNERS:
            for kind in KINDS:
                for actor_lr, critic_lr in grid_for(learner):
                    for seed in seeds:
                        rows.append(f"{learner}\t{kind}\t{actor_lr}\t{critic_lr}\t{seed}")
        # Oracle: deterministic, seed-independent; one line, seed only labels its output folder.
        rows.append(f"oracle\tmarkovian\tna\tna\t{seeds[0]}")
        path = os.path.join(here, f"selection_tasks_{cell}.tsv")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{cell}: {len(rows)} tasks -> {os.path.basename(path)}")
        total += len(rows)
    n_configs = sum(len(grid_for(l)) for l in LEARNERS) * len(KINDS)
    print(f"per cell: {n_configs} lr-configs x {len(seeds)} seeds + 1 oracle = {n_configs * len(seeds) + 1}")
    print(f"total selection tasks (seeds {seeds}, per sigma): {total}  --  x2 sigma = {2 * total}")


if __name__ == "__main__":
    main()
