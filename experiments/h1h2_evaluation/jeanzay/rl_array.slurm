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
# default and the agent behind data/main_unified) or `actor_critic` (ContinuousTimeActorCritic, the
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
#
# WIN below is the RESOLVED window: the value that each agent actually uses, made LIVE by
# `agent.signature.force_signature_window=true` on the python invocation. Without that flag the
# window_size passed here was DISCARDED by both agents, each deriving its own from the delay by a
# DIFFERENT formula -- ContinuousValueGradient._compute_window_size (value_gradient_jax.py:56-61)
# adds 3, ContinuousTimeActorCritic._init_signatures (actor_critic_jax.py:161-166) adds 1 -- so the two agents
# ran on different histories (measured, WIN -> VG/AC: markovian 10/10, linear_dde 8/6, platoon 7/6,
# mg_limit_cycle 27/25, mg_chaotic 71/69). On kind=signature that divergence leaves NO trace in any
# recorded dimension (the signature dim depends only on channel count and depth), so it is
# undetectable after the fact. For the five cells that predate this campaign the values are the
# value-gradient's PREVIOUS effective windows, so the value-gradient half stays bit-unchanged and
# comparable with the existing mackey_glass studies (27, 71); only the actor-critic moves, and it has
# no prior data on those cells. Every window covers its delay (taps needed 0/5/4/24/68 against
# 10/8/7/27/71).
#
# The two Hopfield cells have NO prior value-gradient data, so their window is taken from the
# oracle-referenced harness instead (run/study/h1h2_evaluation.py: win = round(tau/dt) + 1, i.e.
# window_size 5 at tau = 0.5, dt = 0.1), which spans exactly the delay rather than over-covering it,
# and keeps the two halves of the project on identical windows as well as identical plants. Their
# horizon (tf = 15.0) and evaluation initial condition (x0c = [0.5, 0.0]) are likewise the harness's.
# WIN is now round(tau/dt) at the UNIFORM control dt = 0.05 (env configs, 2026-07-17). The number
# of taps therefore SCALES WITH THE DELAY, by design: a longer delay is a longer history and must be
# represented by more samples. This is the physically honest discretisation, and it is deliberately
# NOT equalised across cells -- equalising the window would hide the regime the study is about. The
# signature's dimension is independent of the tap count (depth-N of d channels = d + d^2 + ... +
# d^N), so it is bounded even at 340 taps; raw_history degree-2 grows as taps^2 and explodes on the
# long-delay cells, where its critic Gram becomes rank-deficient. That divergence is a genuine H1
# result (a history representation broken by its own dimension), surfaced loudly now that trimming
# is off and the finiteness guards are in. MAXT/TSIM are time horizons (dt-invariant); steps per
# episode = MAXT/dt, so mg_chaotic is ~1200 steps at 340 taps -- the heaviest cell by far.
case "$CELL" in
    markovian)          ENV=double_integrator; WIN=10;  MAXT=15; TSIM=15; X0="[1.0,0.5]" ;;   # tau=0: WIN arbitrary (redundant history of a Markov plant; H1 must fail)
    linear_dde)         ENV=linear_dde_scalar; WIN=20;  MAXT=20; TSIM=20; X0="[0.8]" ;;        # tau=1.0 / dt=0.05
    hopfield_linear)    ENV=hopfield_linear;   WIN=10;  MAXT=15; TSIM=15; X0="[0.5,0.0]" ;;    # tau=0.5 / dt=0.05
    hopfield_nonlinear) ENV=hopfield_nonlinear; WIN=10; MAXT=15; TSIM=15; X0="[0.5,0.0]" ;;    # tau=0.5 / dt=0.05
    hopfield_duffing)   ENV=hopfield_duffing;  WIN=10;  MAXT=15; TSIM=15; X0="[0.5,0.0]" ;;    # tau=0.5 / dt=0.05
    platoon)            ENV=platoon_h1h2;      WIN=4;   MAXT=15; TSIM=15; X0="[0.0,0.5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]" ;;  # tau=0.2 / dt=0.05
    mg_limit_cycle)     ENV=MG_1D_limit_cycle; WIN=120; MAXT=40; TSIM=40; X0="[0.8]" ;;        # tau=6.0 / dt=0.05
    mg_chaotic)         ENV=MG_1D_chaotic;     WIN=340; MAXT=60; TSIM=60; X0="[0.8]" ;;        # tau=17.0 / dt=0.05 -- the long-delay stress cell
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

# DIVERGENCE THRESHOLD is passed explicitly so that BOTH agents cut episodes at the SAME state
# norm. Relying on the per-agent defaults would compare two different episode-termination rules:
# conf/agent/value_gradient.yaml sets 100.0 while conf/agent/actor_critic.yaml sets 50.0. Measured on
# mg_chaotic/raw_history seed 0, that mismatch cut ALL THREE actor-critic episodes at ||x||>50
# (first at episode 0, ||x||=50.75, peak 97.46) against a single cut at >100 for the value gradient,
# i.e. the actor-critic was charged for excursions the value gradient was allowed to complete.
# 100.0 is chosen because it is the value-gradient default AND the value used by the existing 5-seed
# value-gradient runs, so the value-gradient half stays comparable with its own history; the
# actor-critic's 50.0 is the outlier and has no prior data on these cells. The threshold only
# affects the mg_chaotic tasks: on mg_limit_cycle the cut never fires (peak ||x|| 1.15 actor-critic
# / 1.23 value gradient, against 50).
#
# shellcheck disable=SC2086  # intentional word-splitting of the override strings
python main_unified.py \
    agent="$AGENT" env="$ENV" $REPARG \
    agent.signature.window_size="$WIN" \
    agent.signature.force_signature_window=true \
    ++agent.training.tau_polyak=1.0 \
    agent.training.n_episodes="$N_EPISODES" agent.training.max_time="$MAXT" \
    eval.x0_test="$X0" eval.T_sim="$TSIM" \
    seed="$SEED" experiment_group="$EXPERIMENT_GROUP" \
    run_tag="${AGENT}_${CELL}_${REP}" debug="$DEBUG" wandb.mode=offline
STATUS=$?
echo
echo "Task $SLURM_ARRAY_TASK_ID (cell=$CELL rep=$REP seed=$SEED) finished with exit code $STATUS"
exit $STATUS
