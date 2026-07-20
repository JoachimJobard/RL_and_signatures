#!/bin/bash
# =============================================================================
# Jean Zay -- optimiser-decoupling study: actor/critic optimiser x representation
# =============================================================================
# Tests the actor/critic asymmetry (actor = true gradient, Adam applies; critic = semi-gradient TD
# fixed point, where Adam preconditions a non-gradient). CONTROLLED comparison: everything fixed
# except the three axes below, so a difference is attributable to the optimiser choice / cell alone.
#   kind            in {markovian, signature, raw_history}   -- representation (well- vs ill-conditioned)
#   actor_optimizer in {adam, sgd}                           -- actor role
#   critic_optimizer in {adam, sgd}                          -- critic role (sgd = classical RM-TD)
#   alpha0 (actor_lr) over a SHARED grid                     -- same grid for every config (matched budget,
#                                                               so no per-config alpha0 cherry-picking)
# FIXED across all tasks: critic_lr, sigma, RM decays (0.8 actor / 0.6 critic), n_episodes, horizon,
# seed 42, env=harmonic_oscillator, actor_feature_map=raw (default).
#
# Partition: NON-BILLED visu (empty 80-CPU nodes, no queue; qos_prepost caps the account at node=3
# shared across prepost/compil/visu, so tasks pack densely on visu's free nodes). Walltime fits the 4 h cap.
#
# Env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP KINDS_STR AOPTS_STR COPTS_STR ALRS_STR
#   N_EPISODES RL_SIGNATURES_DATA_ROOT [CRITIC_LR SIGMA PDECAY CDECAY HORIZON]
# =============================================================================
set -euo pipefail

# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"   # activates .venv (NOT bare cray-python)
jeanzay_print_diagnostics

read -ra KINDS <<< "$KINDS_STR"
read -ra AOPTS <<< "$AOPTS_STR"
read -ra COPTS <<< "$COPTS_STR"
read -ra ALRS  <<< "$ALRS_STR"
NK=${#KINDS[@]}; NA=${#AOPTS[@]}; NC=${#COPTS[@]}; NAL=${#ALRS[@]}

i=$SLURM_ARRAY_TASK_ID
ki=$(( i / (NA*NC*NAL) ));  r=$(( i % (NA*NC*NAL) ))
ai=$(( r / (NC*NAL) ));     r=$(( r % (NC*NAL) ))
ci=$(( r / NAL ));          ali=$(( r % NAL ))
KIND=${KINDS[$ki]}; AOPT=${AOPTS[$ai]}; COPT=${COPTS[$ci]}; ALR=${ALRS[$ali]}

echo "Task $SLURM_ARRAY_TASK_ID: kind=$KIND actor_opt=$AOPT critic_opt=$COPT alpha0=$ALR (oscillator, seed 42, fixed params)"

python main_unified.py \
    agent=actor_critic env=harmonic_oscillator \
    agent.signature.kind="$KIND" \
    ++agent.training.actor_optimizer="$AOPT" ++agent.training.critic_optimizer="$COPT" \
    ++agent.training.actor_lr="$ALR" ++agent.training.critic_lr="${CRITIC_LR:-1e-2}" \
    ++agent.noise.sigma="${SIGMA:-0.5}" \
    ++agent.training.lr_decay_power="${PDECAY:-0.8}" \
    ++agent.training.critic_lr_decay_power="${CDECAY:-0.6}" \
    ++agent.training.monitor_snr=true \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="${HORIZON:-5.0}" \
    eval.T_sim="${HORIZON:-5.0}" \
    seed=42 experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${KIND}_A-${AOPT}_C-${COPT}_alr${ALR}" wandb.mode=disabled
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
