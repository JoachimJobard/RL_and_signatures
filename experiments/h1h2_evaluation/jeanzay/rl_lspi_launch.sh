#!/bin/bash
# =============================================================================
# Jean Zay -- launcher for the H1/H2 RL-LSPI array + finalize
# =============================================================================
# Submits a job array over (cell x seed) running damped LSPI (h1h2_rl_lspi.py, all 3 reps per task)
# on cpu_p1, then an aggregation job on the non-billed prepost partition (afterok). CPU-only.
#
# Usage (Jean Zay login node, after git pull + uv sync):
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_lspi_launch.sh           # full
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_lspi_launch.sh --smoke    # 2 cells, seed 0
#
# Routing (~/.claude/CLAUDE.md): compute array cpu_p1 / qos_cpu-t3 (smoke qos_cpu-dev); finalize prepost.
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"

SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"
if [[ -n "$SMOKE" ]]; then
    CELLS=(markovian linear_dde); SEEDS=(0); QOS="qos_cpu-dev"; TIME="00:30:00"; PREFIX="_debug_"
else
    CELLS=(markovian linear_dde platoon mg_limit_cycle mg_chaotic); SEEDS=(0 1 2 3 4)
    QOS="qos_cpu-t3"; TIME="04:00:00"; PREFIX=""
fi
CELLS_STR="${CELLS[*]}"; SEEDS_STR="${SEEDS[*]}"
N_TASKS=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))

TS=$(date +%Y%m%d_%H%M%S)
EXPDIR="$PATH_CONTENT_ROOT/data/h1h2_rl_lspi/${PREFIX}${TS}_array"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPDIR=$EXPDIR,CELLS_STR=$CELLS_STR,SEEDS_STR=$SEEDS_STR"
WORKER_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"

echo "cells = ${CELLS_STR} | seeds = ${SEEDS_STR} | tasks = $N_TASKS"
echo "EXPDIR = $EXPDIR"

# compute array: cpu_p1 (CPU-only: numpy LSTD + JAX-CPU rollouts). 8 cores/task.
ARRAY_JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=cpu_p1 --qos="$QOS" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=8 --hint=nomultithread \
    --time="$TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$WORKER_DIR/rl_lspi_array.slurm")
echo "submitted RL-LSPI array job $ARRAY_JOB ($N_TASKS tasks, cpu_p1 / $QOS)"

# finalize on prepost (non-billed): aggregate the per-task summaries.
FIN_JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=prepost \
    --dependency=afterok:"$ARRAY_JOB" \
    --ntasks=1 --cpus-per-task=2 --time=00:20:00 \
    --output="$SLURM_LOG_DIR/finalize-%j.out" --error="$SLURM_LOG_DIR/finalize-%j.err" \
    --export=ALL,"$EXPORTS" \
    "$WORKER_DIR/h1h2_finalize.slurm")
echo "submitted finalize job $FIN_JOB (afterok:$ARRAY_JOB, prepost)"
echo
echo "results: $EXPDIR/aggregate.txt   (after both jobs finish)"
