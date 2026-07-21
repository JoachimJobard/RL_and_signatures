#!/bin/bash
# =============================================================================
# Jean Zay -- learner x representation board sweep (AC / PG / VG), non-billed visu
# =============================================================================
# One consistent sweep for a shareable board: the three learners on the three representations,
# same discounted oscillator objective, everything fixed except the learner, the cell, and a
# per-learner learning-rate grid (Adam lives on a different scale for the actor vs the critic).
#   learner in {actor_critic (Adam actor + RM-TD critic), policy_gradient (Adam actor, REINFORCE),
#               value_gradient (Adam critic, control = grad V -- no actor)}
#   kind    in {markovian, signature, raw_history}
#   lr grid : actor_lr {1,3,10,30,100} for AC/PG ; critic_lr {1e-3,1e-2,1e-1,1,10} for VG
# FIXED: sigma=0.5, discounted gamma=1, RM decays, n_episodes, horizon, seed 42, actor_feature_map=raw.
#
# Partition: NON-BILLED visu. Env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP LEARNERS_STR
#   KINDS_STR N_EPISODES RL_SIGNATURES_DATA_ROOT
# =============================================================================
set -euo pipefail
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

read -ra LEARNERS <<< "$LEARNERS_STR"
read -ra KINDS    <<< "$KINDS_STR"
NL=${#LEARNERS[@]}; NK=${#KINDS[@]}; NR=5
i=$SLURM_ARRAY_TASK_ID
li=$(( i / (NK*NR) )); r=$(( i % (NK*NR) )); ki=$(( r / NR )); ri=$(( r % NR ))
LEARNER=${LEARNERS[$li]}; KIND=${KINDS[$ki]}
if [ "$LEARNER" = "value_gradient" ]; then LRVALS=(0.001 0.01 0.1 1 10); else LRVALS=(1 3 10 30 100); fi
LR=${LRVALS[$ri]}

echo "Task $SLURM_ARRAY_TASK_ID: learner=$LEARNER kind=$KIND lr=$LR (oscillator, discounted gamma=1, seed 42)"

COMMON="agent=$LEARNER env=harmonic_oscillator agent.signature.kind=$KIND \
  ++agent.noise.sigma=0.5 ++agent.discount.discounted=true ++agent.discount.tau=1.0 \
  agent.training.n_episodes=$N_EPISODES agent.training.max_time=5.0 eval.T_sim=5.0 \
  ++agent.training.monitor_snr=false seed=42 experiment_group=$EXPERIMENT_GROUP wandb.mode=disabled"
case $LEARNER in
  actor_critic)    EXTRA="++agent.training.actor_optimizer=adam ++agent.training.critic_optimizer=sgd ++agent.training.actor_lr=$LR ++agent.training.critic_lr=1e-2 ++agent.training.lr_decay_power=0.8 ++agent.training.critic_lr_decay_power=0.6";;
  policy_gradient) EXTRA="++agent.training.actor_optimizer=adam ++agent.training.actor_lr=$LR ++agent.training.lr_decay_power=0.8";;
  value_gradient)  EXTRA="++agent.training.critic_lr=$LR ++agent.training.critic_lr_decay_power=0.6";;
esac

python main_unified.py $COMMON $EXTRA run_tag="${LEARNER}_${KIND}_lr${LR}"
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
