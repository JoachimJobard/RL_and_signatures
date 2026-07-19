#!/bin/bash
# =============================================================================
# Jean Zay -- launch the FOCUSED markovian actor-signal study (AC + PG), NON-BILLED
# =============================================================================
# 2 learners x 4 sigma x 3 two-timescale pairs = 24 tasks on the markovian cell, seed 42, with full
# per-episode diagnostics (monitor_snr). Goal: find the (sigma, actor_lr, critic_lr) that gives the
# AC/PG a strong, stable signal on the simplest plant -- addressing the cold-critic early transient.
#
# Partition: NON-BILLED (SWEEP_PARTITION, default visu -- empty 80-CPU nodes, no queue). akz@cpu.
#
# Usage (Jean Zay login, after git pull):
#   SWEEP_PARTITION=visu bash experiments/h1h2_evaluation/jeanzay/markovian_signal_launch.sh
# =============================================================================
set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
SWEEP_PARTITION="${SWEEP_PARTITION:-visu}"       # visu: empty nodes, no queue; 4 h cap is ample
SWEEP_TIME="${SWEEP_TIME:-03:00:00}"
ACCOUNT="${ACCOUNT:-akz@cpu}"
PLANT="${PLANT:-double_integrator}"                  # double_integrator (unstable) | harmonic_oscillator (marginally stable)

# value_gradient included as the model-informed reference (should stabilise either plant); AC + PG
# are the model-free learners under test.
read -ra LEARNERS <<< "${LEARNERS:-value_gradient signatures policy_gradient}"
read -ra SIGMAS   <<< "${SIGMAS:-0.1 0.3 0.5 1.0}"
read -ra TS       <<< "${TS:-1e-3:1e-3 1e-3:1e-2 1e-2:1e-2}"   # "actor_lr:critic_lr" (fast critic = larger critic_lr)
read -ra ROLLOUTS <<< "${ROLLOUTS:-1}"                          # rollouts_per_update K (batch size)
read -ra DECAY    <<< "${DECAY:-0.8}"                           # actor RM decay power p (lr_decay_power)
read -ra HORIZONS <<< "${HORIZONS:-15}"                         # episode horizon T (max_time = T_sim)
N_EPISODES="${N_EPISODES:-2000}"

N_TASKS=$(( ${#LEARNERS[@]} * ${#SIGMAS[@]} * ${#TS[@]} * ${#ROLLOUTS[@]} * ${#DECAY[@]} * ${#HORIZONS[@]} ))
GROUP="signal_study_${PLANT}_$(date +%Y%m%d_%H%M%S)"
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP"
EXPORTS+=",LEARNERS_STR=${LEARNERS[*]},SIGMAS_STR=${SIGMAS[*]},TS_STR=${TS[*]},ROLLOUTS_STR=${ROLLOUTS[*]},DECAY_STR=${DECAY[*]},HORIZONS_STR=${HORIZONS[*]}"
EXPORTS+=",N_EPISODES=$N_EPISODES,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT,PLANT=$PLANT"

echo "signal study on PLANT=$PLANT: $N_TASKS tasks (learners=${#LEARNERS[@]} x sigma=${#SIGMAS[@]} x timescale=${#TS[@]} x rollouts=${#ROLLOUTS[@]} x decay=${#DECAY[@]} x horizon=${#HORIZONS[@]})"
echo "partition = $SWEEP_PARTITION (non-billed) | n_episodes = $N_EPISODES | seed = 42"
echo "group = $GROUP  ->  $EXPDIR"

JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$SWEEP_PARTITION" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=4 --hint=nomultithread \
    --time="$SWEEP_TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/markovian_signal_array.slurm")
echo "submitted markovian-signal array job $JOB ($N_TASKS tasks, $SWEEP_PARTITION, non-billed)"
echo "runs land in: $EXPDIR/<ts>_<agent>_sigma<>_alr<>_clr<>_seed42/"
echo "each run's training_metrics.pkl carries per-episode: actor_update_snr, actor_coupling,"
echo "  actor_adv_mean/std, actor_n_eff, actor_grad_signal (+ actor_snr per-step) from episode 0"
echo "watch: squeue -u \$USER -p $SWEEP_PARTITION"
