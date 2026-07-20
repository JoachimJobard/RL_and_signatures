#!/bin/bash
# =============================================================================
# Jean Zay -- launch the CONTROLLABILITY / SNR sweep (held-out seed 42), NON-BILLED
# =============================================================================
# Submits ONE job array over (learner x cell x representation x lr_scale) at seed 42, to a
# non-billed partition, to fix the campaign's shared learning rate alpha_0 and confirm each cell is
# controllable with actor-gradient SNR above ~1. Scored on cost_reduction_pct and the logged SNR;
# NOT on H1 (choosing hyperparameters to make H1 hold would be HARKing -- see the pre-registration).
#
#   3 learners x 5 cells x 3 reps x 3 lr_scales = 135 tasks. N_EPISODES defaults to 500 (a
#   controllability read, not the campaign's 1000). Seed 42 is held out from the campaign (0..4).
#
# Partition (NON-BILLED): SWEEP_PARTITION, default prepost (on-label light CPU). compil / visu are
# idle non-billed alternatives (off-label). No project hours are deducted on any of these.
#
# Usage (Jean Zay login, after git pull):
#   SWEEP_PARTITION=compil bash experiments/h1h2_evaluation/jeanzay/sweep_launch.sh
# =============================================================================
set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
SWEEP_PARTITION="${SWEEP_PARTITION:-prepost}"
# Walltime: default 08:00:00 fits prepost/compil (20 h cap) with generous headroom for the
# mg_chaotic cell (340-tap windows -- the only slow one; a killed task produces NO eval.pkl, since
# evaluation runs after training). NOTE: visu caps at 04:00:00, so route mg_chaotic to prepost/compil.
SWEEP_TIME="${SWEEP_TIME:-08:00:00}"
ACCOUNT="${ACCOUNT:-}"                       # optional on non-billed partitions

LEARNERS=(value_gradient actor_critic policy_gradient)
CELLS=(markovian linear_dde hopfield_nonlinear mg_chaotic platoon)
REPS=(markovian raw_history signature)
LR_SCALES=(0.1 1.0 10.0)                      # alpha_0 in {1e-4, 1e-3, 1e-2}
N_EPISODES="${N_EPISODES:-500}"
DEBUG="${DEBUG:-false}"

N_TASKS=$(( ${#LEARNERS[@]} * ${#CELLS[@]} * ${#REPS[@]} * ${#LR_SCALES[@]} ))
# ARRAY_RANGE lets a canary submit a single task (e.g. ARRAY_RANGE=0-0) before the full grid;
# defaults to the whole grid. The decode in the worker is absolute, so any sub-range is valid.
ARRAY_RANGE="${ARRAY_RANGE:-0-$(( N_TASKS - 1 ))}"
GROUP="sweep_ctrl_snr_$(date +%Y%m%d_%H%M%S)"
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP"
EXPORTS+=",LEARNERS_STR=${LEARNERS[*]},CELLS_STR=${CELLS[*]},REPS_STR=${REPS[*]},LR_SCALES_STR=${LR_SCALES[*]}"
EXPORTS+=",N_EPISODES=$N_EPISODES,DEBUG=$DEBUG,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"

echo "sweep: $N_TASKS tasks (learners=${#LEARNERS[@]} x cells=${#CELLS[@]} x reps=${#REPS[@]} x lr=${#LR_SCALES[@]})"
echo "partition = $SWEEP_PARTITION (non-billed) | n_episodes = $N_EPISODES | seed = 42"
echo "group = $GROUP  ->  $EXPDIR"

# --account is optional on the non-billed partitions; pass it only if set.
ACCOUNT_ARG=(); [[ -n "$ACCOUNT" ]] && ACCOUNT_ARG=(--account="$ACCOUNT")

JOB=$(sbatch --parsable \
    "${ACCOUNT_ARG[@]}" \
    --partition="$SWEEP_PARTITION" \
    --array="$ARRAY_RANGE" \
    --ntasks=1 --cpus-per-task=4 --hint=nomultithread \
    --time="$SWEEP_TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/sweep_array.slurm")
echo "submitted sweep array job $JOB ($N_TASKS tasks, $SWEEP_PARTITION, non-billed)"
echo "runs land in: $EXPDIR/<timestamp>_<agent>_<cell>_<rep>_lr<scale>_seed42/"
echo "each run's metrics carry cost_reduction_pct (eval.pkl) and per-episode actor_snr"
echo "watch: squeue -u \$USER -p $SWEEP_PARTITION"
