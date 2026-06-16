#!/bin/bash
# =============================================================================
# Jean Zay — launcher for the H1/H2 evaluation array + finalize
# =============================================================================
# Submits a job array over (cell x seed) on the CPU partition, then an aggregation
# job on the non-billed prepost partition (afterok dependency).
#
# The whole pipeline is CPU-only: the oracle is scipy (CARE / solve_bvp), the env is
# JAX on CPU, the critic fits are numpy least-squares. No GPU is requested.
#
# Usage (on a Jean Zay login node, after `git pull` + `uv sync`):
#   ACCOUNT=<projid>@cpu  bash experiments/h1h2_evaluation/jeanzay/h1h2_launch.sh
#   ACCOUNT=<projid>@cpu  bash experiments/h1h2_evaluation/jeanzay/h1h2_launch.sh --smoke
#
# Partition / QoS routing (see ~/.claude/CLAUDE.md):
#   - compute array : cpu_p1 + qos_cpu-t3   (billed CPU; <=20 h, here 2 h/task)
#                     smoke uses qos_cpu-dev (<=2 h, <=10 jobs, fast scheduling)
#   - finalize      : prepost               (NON-billed post-processing of saved metrics)
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set — are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"

SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"

if [[ -n "$SMOKE" ]]; then
    CELLS=(markovian linear_dde); SEEDS=(0)
    QOS="qos_cpu-dev"; TIME="00:30:00"; PREFIX="_debug_"
else
    CELLS=(markovian linear_dde platoon mg_limit_cycle mg_chaotic); SEEDS=(0 1 2 3 4)
    QOS="qos_cpu-t3"; TIME="02:00:00"; PREFIX=""
fi
CELLS_STR="${CELLS[*]}"; SEEDS_STR="${SEEDS[*]}"
N_TASKS=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))

# Off-manifold data strategy (the ker-G ablation): both (default) | on_only | rigorous.
# 'rigorous' adds N_OFF off-sheet WINDOW perturbations per MG task, each labelled by its own BVP
# (costly: ~N_OFF extra BVP solves per MG cell/seed). Override via env: DATA_MODE=... N_OFF=...
DATA_MODE="${DATA_MODE:-both}"
N_OFF="${N_OFF:-500}"
# rigorous mode does ~N_OFF extra BVP solves per MG task -> give it a longer wall (still < 20h t3 cap).
[[ "$DATA_MODE" == "rigorous" && -z "$SMOKE" ]] && TIME="06:00:00"

# Output folder derives from the harness name (per repo convention); SLURM logs live in slurm/.
TS=$(date +%Y%m%d_%H%M%S)
EXPDIR="$PATH_CONTENT_ROOT/data/h1h2_evaluation/${PREFIX}${TS}_${DATA_MODE}_array"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPDIR=$EXPDIR,CELLS_STR=$CELLS_STR,SEEDS_STR=$SEEDS_STR,DATA_MODE=$DATA_MODE,N_OFF=$N_OFF"
WORKER_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"

echo "cells = ${CELLS_STR} | seeds = ${SEEDS_STR} | tasks = $N_TASKS"
echo "EXPDIR = $EXPDIR"

# --- compute array: cpu_p1 (CPU-only work; no GPU). 8 cores/task for numpy+JAX-CPU threading. ---
ARRAY_JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=cpu_p1 --qos="$QOS" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=8 --hint=nomultithread \
    --time="$TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$WORKER_DIR/h1h2_array.slurm")
echo "submitted array job $ARRAY_JOB ($N_TASKS tasks, cpu_p1 / $QOS)"

# --- finalize: prepost (non-billed; reads saved summaries, aggregates). ---
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
echo "watch:   squeue -u \$USER"
echo "results: $EXPDIR/aggregate.txt   (after both jobs finish)"
