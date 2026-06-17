#!/bin/bash
# =============================================================================
# Jean Zay -- H1/H2 RL-LSPI (damped least-squares policy iteration) array worker
# =============================================================================
# The learning half via the STABLE least-squares solver. Maps SLURM_ARRAY_TASK_ID -> (cell, seed)
# from CELLS_STR x SEEDS_STR (cell outer, seed inner) and runs:
#     python run/study/h1h2_rl_lspi.py --cell <cell> --seed <seed> --out-dir $EXPDIR
# which writes $EXPDIR/<cell>_seed<seed>/summary.json (all three representations; per-cell tau in
# the evaluator's CFG). CPU-only (numpy LSTD + JAX-CPU env rollouts).
#
# Required env (via --export): PATH_CONTENT_ROOT EXPDIR CELLS_STR SEEDS_STR
# =============================================================================

#SBATCH --job-name=h1h2_rl_lspi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in PATH_CONTENT_ROOT EXPDIR CELLS_STR SEEDS_STR; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: required environment variable $var is not defined" >&2
        exit 1
    fi
done

# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

read -ra CELLS <<< "$CELLS_STR"; read -ra SEEDS <<< "$SEEDS_STR"
N_CELLS=${#CELLS[@]}; N_SEEDS=${#SEEDS[@]}
N_TASKS=$(( N_CELLS * N_SEEDS ))
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((N_TASKS-1)))" >&2
    exit 1
fi
CELL=${CELLS[$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))]}
SEED=${SEEDS[$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))]}
echo "Task [$SLURM_ARRAY_TASK_ID/$((N_TASKS-1))]: cell=$CELL seed=$SEED -> $EXPDIR"
echo

sleep "$(( SLURM_ARRAY_TASK_ID * 3 ))"   # stagger starts

python run/study/h1h2_rl_lspi.py --cell "$CELL" --seed "$SEED" --out-dir "$EXPDIR"
STATUS=$?
echo
echo "Task $SLURM_ARRAY_TASK_ID (cell=$CELL seed=$SEED) finished with exit code $STATUS"
exit $STATUS
