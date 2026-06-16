#!/bin/bash
# =============================================================================
# Jean Zay — H1/H2 RL (value-gradient LEARNING) job-array worker
# =============================================================================
# The learning counterpart of the oracle-referenced harness: instead of fitting a
# critic to the known V*, the value-gradient agent LEARNS the critic online (continuous
# TD) and derives u = 1/2 R^-1 B^T dV/dx. One task per (cell, representation, seed).
#
# Maps SLURM_ARRAY_TASK_ID -> (cell, rep, seed) from CELLS_STR x REPS_STR x SEEDS_STR
# (cell outer, rep middle, seed inner) and runs main_unified.py on the SAME plants as the
# oracle harness (double_integrator / linear_dde_scalar / platoon_h1h2 / MG cells).
#
# Required env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP CELLS_STR REPS_STR
#                              SEEDS_STR N_EPISODES DEBUG
# =============================================================================

#SBATCH --job-name=h1h2_rl_array
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in PATH_CONTENT_ROOT EXPERIMENT_GROUP CELLS_STR REPS_STR SEEDS_STR N_EPISODES DEBUG; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: required environment variable $var is not defined" >&2
        exit 1
    fi
done

# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

read -ra CELLS <<< "$CELLS_STR"; read -ra REPS <<< "$REPS_STR"; read -ra SEEDS <<< "$SEEDS_STR"
NC=${#CELLS[@]}; NR=${#REPS[@]}; NS=${#SEEDS[@]}
N_TASKS=$(( NC * NR * NS ))
if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
    echo "Error: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (0..$((N_TASKS-1)))" >&2
    exit 1
fi
ci=$(( SLURM_ARRAY_TASK_ID / (NR * NS) )); rem=$(( SLURM_ARRAY_TASK_ID % (NR * NS) ))
ri=$(( rem / NS )); si=$(( rem % NS ))
CELL=${CELLS[$ci]}; REP=${REPS[$ri]}; SEED=${SEEDS[$si]}

# Per-cell env + signature window (must cover the delay) + horizon + eval IC. These match the
# oracle-referenced harness cells in run/study/h1h2_evaluation.py.
case "$CELL" in
    markovian)      ENV=double_integrator; WIN=5;  MAXT=15; TSIM=15; X0="[1.0,0.5]" ;;
    linear_dde)     ENV=linear_dde_scalar; WIN=8;  MAXT=20; TSIM=20; X0="[0.8]" ;;
    platoon)        ENV=platoon_h1h2;      WIN=5;  MAXT=15; TSIM=15; X0="[0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" ;;
    mg_limit_cycle) ENV=MG_1D_limit_cycle; WIN=28; MAXT=40; TSIM=40; X0="[0.8]" ;;
    mg_chaotic)     ENV=MG_1D_chaotic;     WIN=72; MAXT=60; TSIM=60; X0="[0.8]" ;;
    *) echo "Error: unknown cell '$CELL'" >&2; exit 1 ;;
esac

# Per-representation kind + capacity (natural config: H1 = raw vs markovian, H2 = signature vs raw).
case "$REP" in
    markovian)   REPARG="agent.signature.kind=markovian agent.signature.degree=2" ;;
    raw_history) REPARG="agent.signature.kind=raw_history agent.signature.degree=2" ;;
    signature)   REPARG="agent.signature.kind=signature agent.signature.depth=2" ;;
    *) echo "Error: unknown rep '$REP'" >&2; exit 1 ;;
esac

echo "Task [$SLURM_ARRAY_TASK_ID/$((N_TASKS-1))]: cell=$CELL rep=$REP seed=$SEED (env=$ENV win=$WIN)"
echo
sleep "$(( SLURM_ARRAY_TASK_ID * 3 ))"   # stagger starts

# Critic solver = PLAIN LSTD (the TD analog of the oracle half's plain least-squares fit to V*):
# least-squares solve of the continuous-TD fixed point E[phi*delta]=0, minimal Tikhonov, NO truncation
# (lstd_rank=0), growing batch (lstd_forget=1.0 = a single batch fit). Uniform across all reps.
LSTD_ARGS="agent.algorithm.lstd=true agent.algorithm.lstd_reg=1e-3 agent.algorithm.lstd_rank=0 agent.algorithm.lstd_forget=1.0"

# shellcheck disable=SC2086  # intentional word-splitting of the override strings
python main_unified.py \
    agent=value_gradient env="$ENV" $REPARG \
    agent.signature.window_size="$WIN" \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$MAXT" \
    eval.x0_test="$X0" eval.T_sim="$TSIM" \
    $LSTD_ARGS \
    seed="$SEED" experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${CELL}_${REP}" debug="$DEBUG" wandb.mode=offline
STATUS=$?
echo
echo "Task $SLURM_ARRAY_TASK_ID (cell=$CELL rep=$REP seed=$SEED) finished with exit code $STATUS"
exit $STATUS
