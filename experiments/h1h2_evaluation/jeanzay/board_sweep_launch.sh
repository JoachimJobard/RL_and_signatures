#!/bin/bash
# =============================================================================
# Jean Zay -- launch the learner x representation board sweep (NON-BILLED). akz@cpu.
# =============================================================================
# 3 learners x 3 kinds x 5 lr = 45 tasks. One consistent sweep for a shareable board comparing
# AC / PG / VG across markovian / signature / raw_history on the discounted oscillator.
#
# Usage (Jean Zay login, after git pull):
#   bash experiments/h1h2_evaluation/jeanzay/board_sweep_launch.sh
# =============================================================================
set -euo pipefail
NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
SWEEP_PARTITION="${SWEEP_PARTITION:-visu}"
SWEEP_TIME="${SWEEP_TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-akz@cpu}"

read -ra LEARNERS <<< "${LEARNERS:-actor_critic policy_gradient value_gradient}"
read -ra KINDS    <<< "${KINDS:-markovian signature raw_history}"
N_EPISODES="${N_EPISODES:-600}"
N_TASKS=$(( ${#LEARNERS[@]} * ${#KINDS[@]} * 5 ))     # 5 = per-learner lr grid size (fixed in the worker)

GROUP="board_sweep_$(date +%Y%m%d_%H%M%S)"
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
EXPORTS+=",LEARNERS_STR=${LEARNERS[*]},KINDS_STR=${KINDS[*]},N_EPISODES=$N_EPISODES"

echo "board sweep: $N_TASKS tasks (learners=${#LEARNERS[@]} x kinds=${#KINDS[@]} x lr=5)"
echo "partition = $SWEEP_PARTITION (non-billed) | n_episodes = $N_EPISODES | seed = 42 | group = $GROUP"

JOB=$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$SWEEP_PARTITION" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=4 --hint=nomultithread \
    --time="$SWEEP_TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/board_sweep_array.slurm")
echo "submitted board-sweep array job $JOB ($N_TASKS tasks, $SWEEP_PARTITION, non-billed)"
echo "runs land in: $EXPDIR/<ts>_<learner>_<kind>_lr<>_seed42/"
echo "watch: squeue -u \$USER -p $SWEEP_PARTITION"