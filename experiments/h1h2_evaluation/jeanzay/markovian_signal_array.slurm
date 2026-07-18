#!/bin/bash
# =============================================================================
# Jean Zay -- FOCUSED markovian-cell actor-signal study (AC + PG), held-out seed 42
# =============================================================================
# Purpose: SOLVE the weak actor signal on the SIMPLEST cell (markovian double-integrator, markovian
# representation). If the actor-critic / policy-gradient cannot obtain a strong, stable signal here,
# nothing downstream matters. The per-episode diagnostics (monitor_snr=true) resolve the BEGINNING of
# training, where the pathology lives: a cold critic emits exploding TD-error advantages that kick
# the actor before it warms. Axes therefore target that transient + exploration:
#   learner  in {signatures (AC), policy_gradient (PG)}
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
NL=${#LEARNERS[@]}; NS=${#SIGMAS[@]}; NT=${#TS[@]}

i=$SLURM_ARRAY_TASK_ID
li=$(( i / (NS*NT) ));  r=$(( i % (NS*NT) ))
si=$(( r / NT ));       ti=$(( r % NT ))
AGENT=${LEARNERS[$li]}; SIGMA=${SIGMAS[$si]}; PAIR=${TS[$ti]}
ALR=${PAIR%%:*}; CLR=${PAIR##*:}

echo "Task $SLURM_ARRAY_TASK_ID: agent=$AGENT sigma=$SIGMA actor_lr=$ALR critic_lr=$CLR (markovian, seed 42)"

# markovian cell = double_integrator + markovian representation (natural for a Markov plant), dt=0.05.
python main_unified.py \
    agent="$AGENT" env=double_integrator \
    agent.signature.kind=markovian agent.signature.degree=2 \
    agent.signature.window_size=10 agent.signature.force_signature_window=true \
    ++agent.training.tau_polyak=1.0 \
    ++agent.training.actor_lr="$ALR" ++agent.training.critic_lr="$CLR" \
    ++agent.training.monitor_snr=true \
    ++agent.noise.sigma="$SIGMA" \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time=15 \
    eval.x0_test="[1.0,0.5]" eval.T_sim=15 \
    seed=42 experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${AGENT}_sigma${SIGMA}_alr${ALR}_clr${CLR}" wandb.mode=disabled
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
