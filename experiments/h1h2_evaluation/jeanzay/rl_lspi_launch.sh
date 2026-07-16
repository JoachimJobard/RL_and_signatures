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
#
# OUTPUT ROOT -- read this before changing it. Runs are written under $SCRATCH, NOT under the
# repository ($WORK). $WORK has an inode quota far too small for a job array's output: job 544311
# (an rl_lspi array, this very launcher) died mid-array with `OSError: [Errno 122] Disk quota
# exceeded` writing to $WORK and produced ZERO run directories against 80 slurm logs. $SCRATCH is
# purged periodically, so rapatriate what matters.
#
# REDIRECT MECHANISM -- deliberately NOT the same as rl_launch.sh's, because the workers differ.
# rl_lspi_array.slurm passes an explicit `--out-dir $EXPDIR` to run/study/h1h2_rl_lspi.py, and that
# script takes `Path(args.out_dir) / tag` WITHOUT consulting script_data_dir (h1h2_rl_lspi.py:165).
# Exporting RL_SIGNATURES_DATA_ROOT here would therefore be a dead flag: --out-dir already wins and
# the env var would never be read. The redirect is carried by EXPDIR itself, re-based onto
# DATA_ROOT below. EXPDIR keeps the `<root>/data/<script_stem>/` shape script_data_dir would have
# produced under the override, so both halves of the campaign share one on-disk layout.
# RL_SIGNATURES_DATA_ROOT is still honoured as the DATA_ROOT *input* (same knob as rl_launch.sh).
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"
# Runs go to $SCRATCH; only the slurm logs stay beside the run directories they describe.
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set -- are you on Jean Zay?}/rl_campaigns/$NAME_PROJECT}"

SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"
if [[ -n "$SMOKE" ]]; then
    CELLS=(hopfield_linear hopfield_duffing); SEEDS=(0); QOS="qos_cpu-dev"; TIME="00:30:00"; PREFIX="_debug_"
else
    CELLS=(markovian linear_dde hopfield_linear hopfield_nonlinear hopfield_duffing platoon mg_limit_cycle mg_chaotic); SEEDS=(0 1 2 3 4)
    QOS="qos_cpu-t3"; TIME="04:00:00"; PREFIX=""
fi
CELLS_STR="${CELLS[*]}"; SEEDS_STR="${SEEDS[*]}"
N_TASKS=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))

# Output folder derives from the harness name (per repo convention), under DATA_ROOT (NOT the
# repository) -- the `data/h1h2_rl_lspi/` suffix mirrors script_data_dir(run/study/h1h2_rl_lspi.py).
TS=$(date +%Y%m%d_%H%M%S)
EXPDIR="$DATA_ROOT/data/h1h2_rl_lspi/${PREFIX}${TS}_array"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPDIR=$EXPDIR,CELLS_STR=$CELLS_STR,SEEDS_STR=$SEEDS_STR"
WORKER_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"

echo "cells = ${CELLS_STR} | seeds = ${SEEDS_STR} | tasks = $N_TASKS"
echo "data root = $DATA_ROOT   (\$SCRATCH, not \$WORK -- see the header)"
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
