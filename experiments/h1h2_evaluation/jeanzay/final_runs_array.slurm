#!/bin/bash
# =============================================================================
# Jean Zay -- 5-seed FINAL runs, manifest-driven, non-billed visu. akz@cpu.
# =============================================================================
# The learning rate per (learner, representation) is FROZEN (frozen_alpha0.yaml, selected by
# %opt on the held-out seed 42). This worker fixes that rate and sweeps the SEED axis instead:
# one array task = one line of the per-cell manifest = one (learner, kind, actor_lr, critic_lr,
# seed) tuple. The seed is SHARED across learners and representations, so a between-representation
# difference is the representation, not RNG. Fixed exactly as in the selection sweep: sigma=0.5,
# discounted objective, monitor_snr off, actor_optimizer=adam, decay powers, one rollout/update.
#
# Env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP RL_SIGNATURES_DATA_ROOT MANIFEST_FILE
#   PLANT MAX_TIME T_SIM GAMMA N_EPISODES. Partition: NON-BILLED visu.
# =============================================================================
set -euo pipefail
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

# One core per task; pin every numeric backend to a single thread so dense packing cannot oversubscribe.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"

# Manifest line index == array task id (0-based -> NR==id+1).
LINE=$(awk -v i="$SLURM_ARRAY_TASK_ID" 'NR==i+1' "$MANIFEST_FILE")
[ -z "$LINE" ] && { echo "no manifest line for task $SLURM_ARRAY_TASK_ID in $MANIFEST_FILE"; exit 2; }
IFS=$'\t' read -r LEARNER KIND ACTOR_LR CRITIC_LR SEED <<< "$LINE"

# Oracle: delayed-LQR / linearised-delayed optimum, representation-agnostic, deterministic.
if [ "$LEARNER" = "oracle" ]; then
  echo "Task $SLURM_ARRAY_TASK_ID: ORACLE reference (delayed-LQR optimum) on env=$PLANT gamma=$GAMMA seed=$SEED"
  python main_unified.py agent=actor_critic env=$PLANT agent.signature.kind=markovian \
    ++agent.algorithm.actor_oracle=true ++agent.discount.discounted=true ++agent.discount.gamma=$GAMMA \
    agent.training.n_episodes=20 agent.training.max_time=$MAX_TIME eval.T_sim=$T_SIM \
    ++agent.training.monitor_snr=false seed=$SEED experiment_group=$EXPERIMENT_GROUP wandb.mode=disabled \
    debug=false run_tag="oracle_reference"
  ST=$?; echo "Task $SLURM_ARRAY_TASK_ID (oracle) finished with exit code $ST"; exit $ST
fi

# The swept rate that labels the run: critic_lr for VG (no actor), actor_lr for PG/AC.
if [ "$LEARNER" = "value_gradient" ]; then SWEPT=$CRITIC_LR; else SWEPT=$ACTOR_LR; fi
echo "Task $SLURM_ARRAY_TASK_ID: $LEARNER $KIND actor_lr=$ACTOR_LR critic_lr=$CRITIC_LR seed=$SEED (env=$PLANT, gamma=$GAMMA)"

COMMON="agent=$LEARNER env=$PLANT agent.signature.kind=$KIND \
  ++agent.noise.sigma=${SIGMA:-0.5} ++agent.discount.discounted=true ++agent.discount.gamma=$GAMMA \
  agent.training.n_episodes=$N_EPISODES agent.training.max_time=$MAX_TIME eval.T_sim=$T_SIM \
  ++agent.training.monitor_snr=${MONITOR_SNR:-false} ++agent.training.patience=${PATIENCE:-0} ++agent.training.eval_interval=${EVAL_INTERVAL:-50} \
  ++agent.training.rollouts_per_update=1 seed=$SEED experiment_group=$EXPERIMENT_GROUP wandb.mode=disabled"
case $LEARNER in
  actor_critic)
    EXTRA="++agent.training.actor_optimizer=adam ++agent.training.critic_optimizer=adam ++agent.training.actor_lr=$ACTOR_LR ++agent.training.critic_lr=$CRITIC_LR ++agent.training.lr_decay_power=0.8 ++agent.training.critic_lr_decay_power=0.6";;
  policy_gradient) EXTRA="++agent.training.actor_optimizer=adam ++agent.training.actor_lr=$ACTOR_LR ++agent.training.lr_decay_power=0.8";;
  value_gradient)  EXTRA="++agent.training.critic_lr=$CRITIC_LR ++agent.training.critic_lr_decay_power=0.6";;
esac

# actor_critic sweeps a 2-D (actor_lr, critic_lr) grid: append critic_lr so two configs sharing
# actor_lr but differing in critic_lr cannot collide on the same run directory. VG/PG keep the
# single-rate tag (backward-compatible with the _lr<rate> aggregation pattern).
if [ "$LEARNER" = "actor_critic" ]; then RUN_TAG="${LEARNER}_${KIND}_lr${SWEPT}_c${CRITIC_LR}"; else RUN_TAG="${LEARNER}_${KIND}_lr${SWEPT}"; fi
python main_unified.py $COMMON $EXTRA run_tag="$RUN_TAG"
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
