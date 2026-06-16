#!/bin/bash
# =============================================================================
# Jean Zay — launcher for the H1/H2 RL (value-gradient LEARNING) array
# =============================================================================
# Submits a job array over (cell x representation x seed) running the value-gradient
# agent (main_unified.py) on the same five cells as the oracle-referenced harness. The
# learning counterpart of h1h2_launch.sh; CPU-only (JAX on CPU), no GPU.
#
# Shared/fixed seeding: SEEDS is the explicit seed axis (per the shared-seed policy each
# (cell, rep) at a given seed uses the same derived per-role seeds). The smoke run pins a
# single fixed seed (0).
#
# Usage (Jean Zay login node, after git pull + uv sync):
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh           # full
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh --smoke   # 2 cells x 3 reps x seed 0
#
# Partition / QoS (see ~/.claude/CLAUDE.md): cpu_p1 + qos_cpu-t3 (billed CPU); smoke uses
# qos_cpu-dev. Replotting/aggregation of the saved eval.pkl is a separate prepost step.
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set — are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"

REPS=(markovian raw_history signature)
SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"

if [[ -n "$SMOKE" ]]; then
    CELLS=(markovian linear_dde); SEEDS=(0); N_EPISODES=5
    QOS="qos_cpu-dev"; TIME="00:30:00"; DEBUG="true"
    GROUP="_debug_h1h2_rl_$(date +%Y%m%d_%H%M%S)"
else
    CELLS=(markovian linear_dde platoon mg_limit_cycle mg_chaotic); SEEDS=(0 1 2 3 4); N_EPISODES=1000
    QOS="qos_cpu-t3"; TIME="04:00:00"; DEBUG="false"
    GROUP="h1h2_rl_$(date +%Y%m%d_%H%M%S)"
fi
N_TASKS=$(( ${#CELLS[@]} * ${#REPS[@]} * ${#SEEDS[@]} ))

EXPDIR="$PATH_CONTENT_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,CELLS_STR=${CELLS[*]},REPS_STR=${REPS[*]},SEEDS_STR=${SEEDS[*]},N_EPISODES=$N_EPISODES,DEBUG=$DEBUG"

echo "cells = ${CELLS[*]} | reps = ${REPS[*]} | seeds = ${SEEDS[*]} | tasks = $N_TASKS"
echo "experiment_group = $GROUP  (n_episodes=$N_EPISODES)"

# Compute array: cpu_p1 (CPU-only: JAX on CPU). 8 cores/task for JAX + signature threading.
ARRAY_JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=cpu_p1 --qos="$QOS" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=8 --hint=nomultithread \
    --time="$TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/rl_array.slurm")
echo "submitted RL array job $ARRAY_JOB ($N_TASKS tasks, cpu_p1 / $QOS)"
echo
echo "runs land in: $EXPDIR/<timestamp>_<cell>_<rep>_seed<seed>/  (eval.pkl has cost_reduction_pct)"
echo "watch: squeue -u \$USER"
