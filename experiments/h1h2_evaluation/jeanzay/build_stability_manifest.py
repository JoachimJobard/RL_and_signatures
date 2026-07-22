"""Build the conservative-learning-rate stability-search manifests (one per cell).

Motivation: with the seed-42 frozen alpha0, several raw_history/signature combos that were stable
at seed 42 diverge on the harder seeds (seed 1 is the worst -- 5 combos diverge there). The cause
is an aggressive learning rate driving a poorly-learned critic to divergence. This search fixes the
HARD seed (1) and sweeps a CONSERVATIVE (lower) learning-rate grid to find a stable set. The grid is
IDENTICAL for raw_history and signature within a learner, so neither representation is advantaged.

Both representations are searched for every affected (cell, learner) pair -- even the one that did
not itself diverge -- so the signature-vs-raw comparison is retuned symmetrically.

Columns (tab-separated, same schema as the final worker): learner  kind  actor_lr  critic_lr  seed
  value_gradient : swept critic_lr (no actor)        -> actor_lr = "na"
  actor_critic   : swept actor_lr, critic_lr fixed low (the aggressive knob is the actor here)
  policy_gradient: swept actor_lr                     -> critic_lr = "na"
"""
SEED = 1  # the hardest seed (most raw/sig divergences in the 1000-episode final run)
REPS = ["raw_history", "signature"]

# Conservative grids (lower than the frozen alpha0, which drove divergence).
VG_CRITIC_LR = [0.003, 0.01, 0.03, 0.1]      # frozen was 0.3-1
AC_ACTOR_LR = [0.03, 0.1, 0.3, 1.0]          # frozen was 3-10; critic held low
AC_CRITIC_LR = 0.01
PG_ACTOR_LR = [0.003, 0.01, 0.03, 0.1]       # frozen was 0.01-3

# (cell, learner) pairs with a diverging raw_history or signature combo on the final run.
AFFECTED = {
    "harmonic_oscillator": ["actor_critic"],
    "linear_dde_scalar": ["actor_critic", "policy_gradient"],
    "MG_1D_limit_cycle": ["value_gradient"],
    "MG_1D_chaotic": ["value_gradient", "policy_gradient"],
}


def lines_for(learner: str) -> list[tuple[str, str]]:
    """(actor_lr, critic_lr) pairs for the learner's conservative grid."""
    if learner == "value_gradient":
        return [("na", str(lr)) for lr in VG_CRITIC_LR]
    if learner == "actor_critic":
        return [(str(lr), str(AC_CRITIC_LR)) for lr in AC_ACTOR_LR]
    if learner == "policy_gradient":
        return [(str(lr), "na") for lr in PG_ACTOR_LR]
    raise ValueError(learner)


def main() -> None:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    total = 0
    for cell, learners in AFFECTED.items():
        rows = []
        for learner in learners:
            for kind in REPS:
                for actor_lr, critic_lr in lines_for(learner):
                    rows.append(f"{learner}\t{kind}\t{actor_lr}\t{critic_lr}\t{SEED}")
        path = os.path.join(here, f"stability_tasks_{cell}.tsv")
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
        print(f"{cell}: {len(rows)} tasks -> {os.path.basename(path)}")
        total += len(rows)
    print(f"total stability-search tasks (seed {SEED}): {total}")


if __name__ == "__main__":
    main()
