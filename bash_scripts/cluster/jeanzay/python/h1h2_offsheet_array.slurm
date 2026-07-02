#!/bin/bash
# =============================================================================
# Jean Zay — job-array worker: off-sheet H2 replication (5 seeds), oracle half.
# =============================================================================
# One array task per line of h1h2_offsheet_5seed_tasks.tsv (105 tasks =
# 7 cells x 3 off-ratios x 5 seeds). Each TSV line is  <subdir>\t<h1h2 args>;
# the task runs:
#     python run/study/h1h2_evaluation.py <args> --out-dir $RUN_ROOT/<subdir>
# so the same-cell/same-seed different-ratio runs land in DISTINCT dirs
# (the script's own tag cell_datamode_seedN does NOT encode n_off, so without the
# per-ratio subdir they would collide).
#
# Pre-registration: documents/methodology/2026-07-02_preregistration_offsheet_h2_replication.md
#   -- commit that file (and this launcher) BEFORE submitting: the remote sees only
#      pushed code, and the dated commit is the anti-HARKing provenance.
#
# Partition / QoS (Jean Zay CPU BVP job; see ~/.claude/CLAUDE.md):
#   cpu_p1 + account akz@cpu + qos_cpu-t3. Each MG/Hopfield task fans the Pontryagin
#   BVP labelling across --cpus-per-task cores (BVP_NPROC). Linear cells are instant.
#   Outputs route to $SCRATCH (WORK inode quota); code/venv stay on $WORK.
#
# !!! NOT cluster-validated (prepared off-cluster). BEFORE the full 105-task array,
#     smoke ONE task on the fast dev QoS:
#       RUN_ROOT=$SCRATCH/h1h2_offsheet_smoke \
#       sbatch --array=0-0 --qos=qos_cpu-dev --time=00:20:00 \
#         --output=$SCRATCH/h1h2_offsheet_smoke/slurm/smoke-%A_%a.out \
#         --error=$SCRATCH/h1h2_offsheet_smoke/slurm/smoke-%A_%a.err \
#         --export=ALL,NAME_PROJECT,RUN_ROOT,TASKS_FILE \
#         bash_scripts/cluster/jeanzay/python/h1h2_offsheet_array.slurm
#     then inspect $RUN_ROOT/mg_limit_cycle_off1x/*/summary.json before scaling up.
#
# Full submission (after the smoke passes, and after `mkdir -p $RUN_ROOT/slurm`):
#   RUN_ROOT=$SCRATCH/h1h2_offsheet_5seed_$(date +%Y%m%d) \
#   TASKS_FILE=$WORK/git_repositories/RL_and_signatures/bash_scripts/cluster/jeanzay/python/h1h2_offsheet_5seed_tasks.tsv \
#   sbatch --output=$RUN_ROOT/slurm/task-%A_%a.out --error=$RUN_ROOT/slurm/task-%A_%a.err \
#          --export=ALL,NAME_PROJECT,RUN_ROOT,TASKS_FILE \
#          bash_scripts/cluster/jeanzay/python/h1h2_offsheet_array.slurm
# =============================================================================

#SBATCH --job-name=h1h2_offsheet
#SBATCH --account=akz@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --array=0-104
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

: "${NAME_PROJECT:=RL_and_signatures}"
: "${RUN_ROOT:?RUN_ROOT is not set — export it (route to \$SCRATCH) before sbatch}"
: "${TASKS_FILE:?TASKS_FILE is not set — path to h1h2_offsheet_5seed_tasks.tsv}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORK/git_repositories/$NAME_PROJECT}"

# Activate the uv venv + print diagnostics (same helper as the main worker).
# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

# Select this task's TSV line (skip blank/#-comment lines).
mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "$TASKS_FILE")
N=${#TASKS[@]}
if (( SLURM_ARRAY_TASK_ID >= N )); then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((N-1)))" >&2
    exit 1
fi
LINE="${TASKS[$SLURM_ARRAY_TASK_ID]}"
SUBDIR="${LINE%%$'\t'*}"          # before the tab
ARGS="${LINE#*$'\t'}"            # after the tab
OUT_DIR="$RUN_ROOT/$SUBDIR"
mkdir -p "$OUT_DIR"
echo "Task [$SLURM_ARRAY_TASK_ID/$((N-1))]: $ARGS  ->  $OUT_DIR"

# Fan the per-window BVP labelling across the allocated cores.
export BVP_NPROC="${SLURM_CPUS_PER_TASK:-1}"

cd "$PATH_CONTENT_ROOT"
# shellcheck disable=SC2086  # intentional word-splitting of the argparse string
python run/study/h1h2_evaluation.py $ARGS --out-dir "$OUT_DIR"
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
