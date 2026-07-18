#!/bin/bash
# =============================================================================
# Jean Zay -- CONTROLLABILITY / SNR sweep worker (held-out seed 42)
# =============================================================================
# One task per (learner, cell, representation, lr_scale). Purpose: fix the shared learning
# rate alpha_0 for the campaign and confirm each cell is controllable AND that the actor-gradient
# SNR clears ~1, BEFORE the 5-seed campaign. Scored on cost_reduction_pct (a controller exists) and
# the logged actor SNR -- NOT on whether H1 holds (that would be tuning toward the hypothesis).
# Seed 42 is HELD OUT from the campaign's seeds 0..4, so nothing chosen here leaks into the campaign
# results.
#
# Partition: NON-BILLED. Set via SWEEP_PARTITION (default prepost -- the on-label light-CPU
# partition). compil / visu are idle alternatives (off-label; both non-billed). A short exploratory
# sweep is exactly the light-CPU work these partitions are for. No hours are deducted.
#
# Env (via --export, from sweep_launch.sh): PATH_CONTENT_ROOT EXPERIMENT_GROUP LEARNERS_STR CELLS_STR
#   REPS_STR LR_SCALES_STR N_EPISODES DEBUG RL_SIGNATURES_DATA_ROOT
# =============================================================================
set -euo pipefail

# Activate the project venv (.venv) + set PYTHONPATH / thread hygiene / cwd via the SAME shared helper
# the proven rl_array.slurm sources. NOTE: a bare `module load cray-python` yields a Python WITHOUT
# the project dependencies (hydra/jax/flax) -- the first submission (job 2045141) failed fast on
# every task with `ModuleNotFoundError: No module named 'hydra'` for exactly that reason.
# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics   # host + python/JAX/devices, so the SLURM log is self-contained

read -ra LEARNERS <<< "$LEARNERS_STR"
read -ra CELLS    <<< "$CELLS_STR"
read -ra REPS     <<< "$REPS_STR"
read -ra LRS      <<< "$LR_SCALES_STR"
NL=${#LEARNERS[@]}; NC=${#CELLS[@]}; NR=${#REPS[@]}; NLR=${#LRS[@]}

# Decode SLURM_ARRAY_TASK_ID -> (learner, cell, rep, lr) in row-major order.
i=$SLURM_ARRAY_TASK_ID
li=$(( i / (NC*NR*NLR) ));            r=$(( i % (NC*NR*NLR) ))
ci=$(( r / (NR*NLR) ));               r=$(( r % (NR*NLR) ))
ri=$(( r / NLR ));                    lri=$(( r % NLR ))
AGENT=${LEARNERS[$li]}; CELL=${CELLS[$ci]}; REP=${REPS[$ri]}; LRSCALE=${LRS[$lri]}

# Per-cell env / window / horizon -- identical table to rl_array.slurm (uniform dt=0.05).
case "$CELL" in
    markovian)          ENV=double_integrator; WIN=10;  MAXT=15; TSIM=15; X0="[1.0,0.5]" ;;
    linear_dde)         ENV=linear_dde_scalar; WIN=20;  MAXT=20; TSIM=20; X0="[0.8]" ;;
    hopfield_nonlinear) ENV=hopfield_nonlinear; WIN=10; MAXT=15; TSIM=15; X0="[0.5,0.0]" ;;
    mg_chaotic)         ENV=MG_1D_chaotic;     WIN=340; MAXT=60; TSIM=60; X0="[0.8]" ;;
    platoon)            ENV=platoon_h1h2;      WIN=4;   MAXT=15; TSIM=15; X0="[0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" ;;
    *) echo "Error: unknown cell '$CELL'" >&2; exit 1 ;;
esac
case "$REP" in
    markovian)   REPARG="agent.signature.kind=markovian agent.signature.degree=2" ;;
    raw_history) REPARG="agent.signature.kind=raw_history agent.signature.degree=2" ;;
    signature)   REPARG="agent.signature.kind=signature agent.signature.depth=2" ;;
    *) echo "Error: unknown rep '$REP'" >&2; exit 1 ;;
esac

# lr_scale multiplies BOTH base rates (alpha_0 of the RM schedule). The schedule shape (p) is fixed
# by the configs; only alpha_0 is swept. critic_lr applies to all three learners; actor_lr only to
# the two with an actor (harmless override for the value gradient, which ignores it).
BASE_CRITIC_LR=1e-3; BASE_ACTOR_LR=1e-3
CRITIC_LR=$(python -c "print(${BASE_CRITIC_LR}*${LRSCALE})")
ACTOR_LR=$(python -c "print(${BASE_ACTOR_LR}*${LRSCALE})")

echo "Task $SLURM_ARRAY_TASK_ID: agent=$AGENT cell=$CELL rep=$REP lr_scale=$LRSCALE (critic_lr=$CRITIC_LR) env=$ENV win=$WIN seed=42"

# shellcheck disable=SC2086
python main_unified.py \
    agent="$AGENT" env="$ENV" $REPARG \
    agent.signature.window_size="$WIN" agent.signature.force_signature_window=true \
    ++agent.training.tau_polyak=1.0 \
    ++agent.training.critic_lr="$CRITIC_LR" ++agent.training.actor_lr="$ACTOR_LR" \
    ++agent.training.monitor_snr=true \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$MAXT" \
    eval.x0_test="$X0" eval.T_sim="$TSIM" \
    seed=42 experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${AGENT}_${CELL}_${REP}_lr${LRSCALE}" debug="$DEBUG" wandb.mode=disabled
STATUS=$?
echo "Task $SLURM_ARRAY_TASK_ID finished with exit code $STATUS"
exit $STATUS
