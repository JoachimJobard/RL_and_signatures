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
# AGENT selects the learner: `value_gradient` (ContinuousValueGradient, no actor; the
# default and the agent behind data/main_unified) or `signatures` (CTACSignatureJAX, the
# continuous-time actor-critic). Both consume the SAME feature maps through the SAME
# `make_representation` factory, so an actor-critic-versus-value-gradient comparison at
# fixed representation is not confounded by two implementations of the representation.
#
# Required env (via --export): PATH_CONTENT_ROOT EXPERIMENT_GROUP CELLS_STR REPS_STR
#                              SEEDS_STR N_EPISODES DEBUG AGENT RL_SIGNATURES_DATA_ROOT
# =============================================================================

#SBATCH --job-name=h1h2_rl_array
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in PATH_CONTENT_ROOT EXPERIMENT_GROUP CELLS_STR REPS_STR SEEDS_STR N_EPISODES DEBUG AGENT RL_SIGNATURES_DATA_ROOT; do
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

echo "Task [$SLURM_ARRAY_TASK_ID/$((N_TASKS-1))]: agent=$AGENT cell=$CELL rep=$REP seed=$SEED (env=$ENV win=$WIN)"
echo "Run output root: $RL_SIGNATURES_DATA_ROOT (NOT the repository; see run_context.script_data_dir)"
echo
sleep "$(( SLURM_ARRAY_TASK_ID * 3 ))"   # stagger starts

# Critic solver = SEMI-GRADIENT TD (the default, lstd=false). LSTD/LSPI -- the least-squares TD
# mirror of the oracle half's lstsq -- is UNSTABLE for online value-gradient control here: online
# LSTD diverges (deadly triad, even on the trivial markovian cell) and batch LSPI oscillates on the
# delayed/oscillatory cells (see run/study/mwe_lspi_*). Semi-gradient TD's slow incremental updates
# are the only stable online option (markovian 56%, linear_dde 81%, MG 86% cost reduction).
# The two halves therefore use different estimators by necessity: supervised lstsq to the FIXED V*
# (oracle half) vs incremental TD to the BOOTSTRAPPED value (RL half).
#
# The run tag carries the agent so that value-gradient and actor-critic runs are distinguishable
# in `ls`. Keeping the two agents in SEPARATE experiment groups is nonetheless mandatory:
# aggregate_representation_study.discover_runs groups by (env, kind, capacity) and NOT by agent,
# so co-locating them in one group would silently average an actor-critic run with a
# value-gradient run of the same representation.

# shellcheck disable=SC2086  # intentional word-splitting of the override strings
python main_unified.py \
    agent="$AGENT" env="$ENV" $REPARG \
    agent.signature.window_size="$WIN" \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$MAXT" \
    eval.x0_test="$X0" eval.T_sim="$TSIM" \
    seed="$SEED" experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${AGENT}_${CELL}_${REP}" debug="$DEBUG" wandb.mode=offline
STATUS=$?
echo
echo "Task $SLURM_ARRAY_TASK_ID (cell=$CELL rep=$REP seed=$SEED) finished with exit code $STATUS"
exit $STATUS
