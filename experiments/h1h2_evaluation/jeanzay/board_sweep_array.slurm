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

# Each task uses ~1 core (measured: TotalCPU ~= Elapsed on 8 alloc CPUs -- the episode loop is
# sequential and per-step ops are tiny). Pin every numeric backend to a single thread so packing
# many 1-CPU tasks per node cannot oversubscribe threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_cpu_multi_thread_eigen=false"

read -ra LEARNERS <<< "$LEARNERS_STR"
read -ra KINDS    <<< "$KINDS_STR"
NL=${#LEARNERS[@]}; NK=${#KINDS[@]}
# NR = lr-grid size. LR_GRID (space-separated, propagated via --export=ALL) overrides the
# per-learner defaults below with a common custom grid (e.g. small lr to probe a stable band).
if [ -n "${LR_GRID:-}" ]; then read -ra _LRG <<< "$LR_GRID"; NR=${#_LRG[@]}; else NR=5; fi
NGRID=$(( NL*NK*NR ))

# Env is parameterised (oscillator defaults preserved). For a DDE the reference optimum is
# NOT -K_gamma x (state feedback) but the delayed-LQR oracle -- run once as the last array index.
PLANT="${PLANT:-harmonic_oscillator}"; MAX_TIME="${MAX_TIME:-5.0}"; T_SIM="${T_SIM:-5.0}"; GAMMA="${GAMMA:-1.0}"
# x0 is passed WITHOUT commas (sbatch --export splits on commas): underscore-separated,
# optional brackets; rebuild the hydra list here. e.g. X0_TEST=0.8_0.0 -> eval.x0_test=[0.8,0.0]
X0ARG=""; [ -n "${X0_TEST:-}" ] && X0ARG="eval.x0_test=[$(echo "$X0_TEST" | tr -d '[]' | tr '_' ',')]"
FWARG="";  [ "${FORCE_WINDOW:-false}" = "true" ] && FWARG="agent.signature.force_signature_window=true"

# Oracle reference (last index): delayed-LQR optimum, representation-agnostic, seed 42.
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NGRID" ]; then
  echo "Task $SLURM_ARRAY_TASK_ID: ORACLE reference (delayed-LQR optimum) on env=$PLANT"
  python main_unified.py agent=actor_critic env=$PLANT agent.signature.kind=markovian \
    ++agent.algorithm.actor_oracle=true ++agent.discount.discounted=true ++agent.discount.gamma=$GAMMA \
    agent.training.n_episodes=20 agent.training.max_time=$MAX_TIME eval.T_sim=$T_SIM $X0ARG \
    ++agent.training.monitor_snr=false seed=42 experiment_group=$EXPERIMENT_GROUP wandb.mode=disabled \
    debug=true run_tag="oracle_reference"
  ST=$?; echo "Task $SLURM_ARRAY_TASK_ID (oracle) finished with exit code $ST"; exit $ST
fi

i=$SLURM_ARRAY_TASK_ID
li=$(( i / (NK*NR) )); r=$(( i % (NK*NR) )); ki=$(( r / NR )); ri=$(( r % NR ))
LEARNER=${LEARNERS[$li]}; KIND=${KINDS[$ki]}
if [ -n "${LR_GRID:-}" ]; then read -ra LRVALS <<< "$LR_GRID"
elif [ "$LEARNER" = "value_gradient" ]; then LRVALS=(0.001 0.01 0.1 1 10)
else LRVALS=(1 3 10 30 100); fi
LR=${LRVALS[$ri]}

echo "Task $SLURM_ARRAY_TASK_ID: learner=$LEARNER kind=$KIND lr=$LR (env=$PLANT, discounted gamma=$GAMMA, seed 42)"

COMMON="agent=$LEARNER env=$PLANT agent.signature.kind=$KIND $FWARG \
  ++agent.noise.sigma=0.5 ++agent.discount.discounted=true ++agent.discount.gamma=$GAMMA \
  agent.training.n_episodes=$N_EPISODES agent.training.max_time=$MAX_TIME eval.T_sim=$T_SIM $X0ARG \
  ++agent.training.monitor_snr=false ++agent.training.rollouts_per_update=${ROLLOUTS:-1} seed=42 experiment_group=$EXPERIMENT_GROUP wandb.mode=disabled"
case $LEARNER in
  actor_critic)
    _CLR="${CRITIC_LR:-1e-2}"; [ "${TIE_CRITIC_LR:-0}" = "1" ] && _CLR="$LR"   # x=y: critic_lr = actor_lr (single alpha0)
    EXTRA="++agent.training.actor_optimizer=adam ++agent.training.critic_optimizer=adam ++agent.training.actor_lr=$LR ++agent.training.critic_lr=$_CLR ++agent.training.lr_decay_power=0.8 ++agent.training.critic_lr_decay_power=0.6";;
  policy_gradient) EXTRA="++agent.training.actor_optimizer=adam ++agent.training.actor_lr=$LR ++agent.training.lr_decay_power=0.8";;
  value_gradient)  EXTRA="++agent.training.critic_lr=$LR ++agent.training.critic_lr_decay_power=0.6";;
esac

python main_unified.py $COMMON $EXTRA run_tag="${LEARNER}_${KIND}_lr${LR}"
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
