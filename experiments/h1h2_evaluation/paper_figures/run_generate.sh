#!/bin/bash
# Submit generate_all.slurm on the non-billed prepost partition, resolving $WORK from the login shell.
# Run through a login shell: bash experiments/h1h2_evaluation/paper_figures/run_generate.sh
set -euo pipefail
: "${WORK:?run through a login shell}"
REPO="${PATH_CONTENT_ROOT:-$WORK/git_repositories/RL_and_signatures}"
SNAP="${PAPER_FIGURES_DATA_ROOT:-$WORK/paper_data/aaai2027_h1_clean_20260724}"
OUT="${PAPER_FIGURES_OUT:-$SNAP/paper_figures_out}"
LOGD="$OUT/slurm"; mkdir -p "$LOGD"
JOB=$(sbatch --parsable --account=akz@cpu --partition=prepost --time=01:00:00 \
  --ntasks=1 --cpus-per-task=2 --hint=nomultithread \
  --output="$LOGD/figs-%j.out" --error="$LOGD/figs-%j.err" \
  --export=ALL,PATH_CONTENT_ROOT="$REPO",PAPER_FIGURES_DATA_ROOT="$SNAP",PAPER_FIGURES_OUT="$OUT" \
  "$REPO/experiments/h1h2_evaluation/paper_figures/generate_all.slurm")
echo "paper figures job: $JOB  (out=$OUT)"; echo "$JOB" > /tmp/figs_jobid
