#!/bin/bash
# =============================================================================
# Jean Zay -- launch the optimiser-decoupling study (NON-BILLED). akz@cpu.
# =============================================================================
# 3 kinds x 2 actor_opt x 2 critic_opt x 5 alpha0 = 60 tasks. CONTROLLED: a single shared alpha0 grid
# and identical fixed params for every config, so only the optimiser role and the cell vary.
#
# Partition: NON-BILLED (SWEEP_PARTITION, default visu -- empty 80-CPU nodes, no queue). akz@cpu.
#
# Usage (Jean Zay login, after git pull):
#   bash experiments/h1h2_evaluation/jeanzay/optimizer_decoupling_launch.sh
#   # override axes: KINDS="signature raw_history" ALRS="1 10 100" bash .../optimizer_decoupling_launch.sh
# =============================================================================
set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
SWEEP_PARTITION="${SWEEP_PARTITION:-visu}"       # visu: empty nodes, no queue; 4 h cap is ample
SWEEP_TIME="${SWEEP_TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-akz@cpu}"

read -ra KINDS <<< "${KINDS:-markovian signature raw_history}"
read -ra AOPTS <<< "${AOPTS:-adam sgd}"
read -ra COPTS <<< "${COPTS:-adam sgd}"
read -ra ALRS  <<< "${ALRS:-1 3 10 30 100}"       # shared alpha0 grid (matched budget across configs)
N_EPISODES="${N_EPISODES:-600}"

N_TASKS=$(( ${#KINDS[@]} * ${#AOPTS[@]} * ${#COPTS[@]} * ${#ALRS[@]} ))
GROUP="opt_decoupling_$(date +%Y%m%d_%H%M%S)"
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
EXPORTS+=",KINDS_STR=${KINDS[*]},AOPTS_STR=${AOPTS[*]},COPTS_STR=${COPTS[*]},ALRS_STR=${ALRS[*]},N_EPISODES=$N_EPISODES"

echo "optimiser-decoupling: $N_TASKS tasks (kinds=${#KINDS[@]} x aopt=${#AOPTS[@]} x copt=${#COPTS[@]} x alpha0=${#ALRS[@]})"
echo "partition = $SWEEP_PARTITION (non-billed) | n_episodes = $N_EPISODES | seed = 42 | fixed alpha0-grid + params"
echo "group = $GROUP  ->  $EXPDIR"

JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$SWEEP_PARTITION" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=4 --hint=nomultithread \
    --time="$SWEEP_TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/optimizer_decoupling_array.slurm")
echo "submitted optimiser-decoupling array job $JOB ($N_TASKS tasks, $SWEEP_PARTITION, non-billed)"
echo "runs land in: $EXPDIR/<ts>_<kind>_A-<aopt>_C-<copt>_alr<>_seed42/"
echo "watch: squeue -u \$USER -p $SWEEP_PARTITION   |   results: grep -r 'Cost reduction' $SLURM_LOG_DIR"