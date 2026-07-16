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
#
# OUTPUT ROOT — read this before changing it. Runs are written under $SCRATCH, NOT under the
# repository ($WORK). $WORK has an inode quota far too small for a job array's output: job 544311
# (an rl_lspi array, whose launcher resolved EXPDIR under $WORK exactly as this one did) died
# mid-array with `OSError: [Errno 122] Disk quota exceeded` and produced ZERO run directories
# against 80 slurm logs. $SCRATCH is purged periodically, so rapatriate what matters.
#
# REDIRECT MECHANISM — deliberately NOT the same as rl_launch.sh's, because the workers differ.
# h1h2_array.slurm passes an explicit `--out-dir $EXPDIR` to run/study/h1h2_evaluation.py, and that
# script takes `Path(args.out_dir) / tag` WITHOUT consulting script_data_dir (h1h2_evaluation.py:454);
# h1h2_finalize.slurm likewise reads `--parent $EXPDIR`. Exporting RL_SIGNATURES_DATA_ROOT here would
# therefore be a dead flag: --out-dir already wins and the env var would never be read. The redirect
# is carried by EXPDIR itself, re-based onto DATA_ROOT below. EXPDIR keeps the
# `<root>/data/<script_stem>/` shape script_data_dir would have produced under the override, so the
# whole campaign shares one on-disk layout. RL_SIGNATURES_DATA_ROOT is still honoured as the
# DATA_ROOT *input* (same knob as rl_launch.sh).
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set — are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"
# Runs go to $SCRATCH; only the slurm logs stay beside the run directories they describe.
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set — are you on Jean Zay?}/rl_campaigns/$NAME_PROJECT}"

SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"

if [[ -n "$SMOKE" ]]; then
    CELLS=(hopfield_linear hopfield_duffing); SEEDS=(0)   # smoke exercises the new cells (analytic + BVP)
    QOS="qos_cpu-dev"; TIME="00:30:00"; PREFIX="_debug_"
else
    CELLS=(markovian linear_dde hopfield_linear hopfield_nonlinear hopfield_duffing platoon mg_limit_cycle mg_chaotic); SEEDS=(0 1 2 3 4)
    QOS="qos_cpu-t3"; TIME="02:00:00"; PREFIX=""
fi
CELLS_STR="${CELLS[*]}"; SEEDS_STR="${SEEDS[*]}"
N_TASKS=$(( ${#CELLS[@]} * ${#SEEDS[@]} ))

# Uniform on-sheet / off-sheet sampling (the ker-G axis): on_sheet | on_off_sheet (default).
# on_off_sheet adds N_OFF off-sheet WINDOW perturbations per task, each RE-LABELLED through the
# cell's oracle (analytic for linear cells, one Pontryagin BVP each for MG). Override: DATA_MODE=...
DATA_MODE="${DATA_MODE:-on_off_sheet}"
N_OFF="${N_OFF:-500}"
# on_off_sheet does ~N_OFF extra BVP solves per MG task -> give it a longer wall (still < 20h t3 cap).
[[ "$DATA_MODE" == "on_off_sheet" && -z "$SMOKE" ]] && TIME="06:00:00"

# Output folder derives from the harness name (per repo convention); SLURM logs live in slurm/.
# The root is DATA_ROOT (NOT the repository) and the `data/h1h2_evaluation/` suffix mirrors
# script_data_dir(run/study/h1h2_evaluation.py).
TS=$(date +%Y%m%d_%H%M%S)
EXPDIR="$DATA_ROOT/data/h1h2_evaluation/${PREFIX}${TS}_${DATA_MODE}_array"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPDIR=$EXPDIR,CELLS_STR=$CELLS_STR,SEEDS_STR=$SEEDS_STR,DATA_MODE=$DATA_MODE,N_OFF=$N_OFF"
WORKER_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"

echo "cells = ${CELLS_STR} | seeds = ${SEEDS_STR} | tasks = $N_TASKS"
echo "data root = $DATA_ROOT   (\$SCRATCH, not \$WORK — see the header)"
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
