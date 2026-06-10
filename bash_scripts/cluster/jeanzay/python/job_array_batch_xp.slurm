#!/bin/bash
# =============================================================================
# Jean Zay — generic job-array worker (one task per variant)
# =============================================================================
# Submitted by experiment_array_launcher.sh. Each array task selects line
# SLURM_ARRAY_TASK_ID of a "variants file" (one Hydra-override string per line)
# and runs:
#     python <PATH_PYTHON_SCRIPT> <COMMON_OVERRIDES> <variant line> \
#            seed=<SEED> experiment_group=<EXPERIMENT_GROUP> wandb.mode=<WANDB_MODE>
#
# Required environment variables (broadcast via sbatch --export):
#   NAME_PROJECT PATH_CONTENT_ROOT PATH_VENV_BIN PATH_PYTHON_SCRIPT
#   VARIANTS_FILE COMMON_OVERRIDES SEED EXPERIMENT_GROUP WANDB_MODE
# =============================================================================

#SBATCH --job-name=rl_sig_array
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in NAME_PROJECT PATH_CONTENT_ROOT PATH_VENV_BIN PATH_PYTHON_SCRIPT \
           VARIANTS_FILE SEED EXPERIMENT_GROUP WANDB_MODE; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: required environment variable $var is not defined" >&2
        exit 1
    fi
done

# SLURM copies this script to a spool dir, so BASH_SOURCE points there, not at the
# repo. Locate _environment.sh via the exported (absolute) PATH_CONTENT_ROOT.
# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

# Effective variant lines: drop blank lines and #-comments.
mapfile -t VARIANTS < <(grep -vE '^\s*(#|$)' "$VARIANTS_FILE")
N_VARIANTS=${#VARIANTS[@]}
if (( SLURM_ARRAY_TASK_ID >= N_VARIANTS )); then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((N_VARIANTS-1)))" >&2
    exit 1
fi
VARIANT_OVERRIDES="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
echo "Variant [$SLURM_ARRAY_TASK_ID/$((N_VARIANTS-1))]: $VARIANT_OVERRIDES"
echo

# Stagger task starts to avoid filesystem / wandb-init contention.
SLEEP_TIME=$(( SLURM_ARRAY_TASK_ID * 5 ))
echo "Sleeping ${SLEEP_TIME}s to stagger task starts..."
sleep "$SLEEP_TIME"

# shellcheck disable=SC2086  # intentional word-splitting of the override strings
python "$PATH_PYTHON_SCRIPT" \
    $COMMON_OVERRIDES \
    $VARIANT_OVERRIDES \
    seed="$SEED" \
    experiment_group="$EXPERIMENT_GROUP" \
    wandb.mode="$WANDB_MODE"
STATUS=$?
echo
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
