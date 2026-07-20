#!/bin/bash
# =============================================================================
# Jean Zay -- FOCUSED markovian-cell actor-signal study (AC + PG), held-out seed 42
# =============================================================================
# Purpose: SOLVE the weak actor signal on the SIMPLEST cell (markovian double-integrator, markovian
# representation). If the actor-critic / policy-gradient cannot obtain a strong, stable signal here,
# nothing downstream matters. The per-episode diagnostics (monitor_snr=true) resolve the BEGINNING of
# training, where the pathology lives: a cold critic emits exploding TD-error advantages that kick
# the actor before it warms. Axes therefore target that transient + exploration:
#   learner  in {actor_critic (AC), policy_gradient (PG)}
#   sigma    in {0.1, 0.3, 0.5, 1.0}                     -- exploration strength (advantage coupling)
#   (actor_lr, critic_lr) in {(1e-3,1e-3),(1e-3,1e-2),(1e-2,1e-2)}  -- two-timescale (fast critic)
# = 2 x 4 x 3 = 24 tasks. 2000 episodes (longer than the campaign's 1000, to see convergence once
# the critic warms). Scored on the diagnostics + cost_reduction, NOT on H1. critic_lr is inert for
# PG's actor (no critic in its update) -- harmless.
#
# Partition: NON-BILLED (SWEEP_PARTITION, default visu -- empty 80-CPU nodes, no queue). qos_prepost
# caps the account at node=3, shared across prepost/compil/visu; visu's nodes are empty so tasks pack
# densely. Walltime fits visu's 4 h cap.
#
# Env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP LEARNERS_STR SIGMAS_STR TS_STR
#   N_EPISODES RL_SIGNATURES_DATA_ROOT
# =============================================================================
set -euo pipefail

# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"   # activates .venv (NOT bare cray-python)
jeanzay_print_diagnostics

read -ra LEARNERS <<< "$LEARNERS_STR"
read -ra SIGMAS   <<< "$SIGMAS_STR"
read -ra TS       <<< "$TS_STR"     # entries "actor_lr:critic_lr", e.g. "1e-3:1e-2"
read -ra ROLLOUTS <<< "${ROLLOUTS_STR:-1}"   # rollouts_per_update K (batch size); default 1
read -ra DECAY    <<< "${DECAY_STR:-0.8}"    # actor RM decay power p (lr_decay_power); default 0.8
read -ra HORIZONS <<< "${HORIZONS_STR:-15}"  # episode horizon T (max_time = T_sim); default 15
read -ra CDECAYS  <<< "${CDECAYS_STR:-0.6}"  # critic RM decay power (critic_lr_decay_power); default 0.6
NL=${#LEARNERS[@]}; NS=${#SIGMAS[@]}; NT=${#TS[@]}; NK=${#ROLLOUTS[@]}; ND=${#DECAY[@]}; NH=${#HORIZONS[@]}; NCD=${#CDECAYS[@]}

i=$SLURM_ARRAY_TASK_ID
li=$(( i / (NS*NT*NK*ND*NH*NCD) )); r=$(( i % (NS*NT*NK*ND*NH*NCD) ))
si=$(( r / (NT*NK*ND*NH*NCD) ));    r=$(( r % (NT*NK*ND*NH*NCD) ))
ti=$(( r / (NK*ND*NH*NCD) ));       r=$(( r % (NK*ND*NH*NCD) ))
ki=$(( r / (ND*NH*NCD) ));          r=$(( r % (ND*NH*NCD) ))
di=$(( r / (NH*NCD) ));             r=$(( r % (NH*NCD) ))
hi=$(( r / NCD ));                  cdi=$(( r % NCD ))
AGENT=${LEARNERS[$li]}; SIGMA=${SIGMAS[$si]}; PAIR=${TS[$ti]}; K=${ROLLOUTS[$ki]}; PDECAY=${DECAY[$di]}; HORIZON=${HORIZONS[$hi]}; CDECAY=${CDECAYS[$cdi]}
ALR=${PAIR%%:*}; CLR=${PAIR##*:}

PLANT="${PLANT:-double_integrator}"   # double_integrator (unstable) or harmonic_oscillator (marginally stable)
echo "Task $SLURM_ARRAY_TASK_ID: agent=$AGENT env=$PLANT sigma=$SIGMA actor_lr=$ALR critic_lr=$CLR rollouts=$K decay_p=$PDECAY (markovian, seed 42)"

# markovian representation (natural for a Markov plant), dt=0.05. PLANT selects the plant; the two LQR
# plants differ ONLY in A (open-loop stability), isolating the instability as the cause.
python main_unified.py \
    agent="$AGENT" env="$PLANT" \
    agent.signature.kind=markovian agent.signature.degree=2 \
    agent.signature.window_size=10 agent.signature.force_signature_window=true \
    ++agent.training.tau_polyak=1.0 \
    ++agent.training.actor_lr="$ALR" ++agent.training.critic_lr="$CLR" \
    ++agent.training.monitor_snr="${MONITOR:-true}" \
    ++agent.noise.sigma="$SIGMA" \
    ++agent.training.rollouts_per_update="$K" \
    ++agent.training.lr_decay_power="$PDECAY" \
    ++agent.training.critic_lr_decay_power="$CDECAY" \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$HORIZON" \
    eval.x0_test="[1.0,0.5]" eval.T_sim="$HORIZON" \
    seed=42 experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${AGENT}_sigma${SIGMA}_alr${ALR}_clr${CLR}_K${K}_p${PDECAY}_T${HORIZON}_cd${CDECAY}" wandb.mode=disabled
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
