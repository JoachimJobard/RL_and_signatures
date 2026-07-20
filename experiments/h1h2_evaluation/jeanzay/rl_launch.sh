#!/bin/bash
# =============================================================================
# Jean Zay — launcher for the H1/H2 RL (value-gradient LEARNING) array
# =============================================================================
# Submits a job array over (cell x representation x seed) running the value-gradient
# agent (main_unified.py) on the same five cells as the oracle-referenced harness. The
# learning counterpart of h1h2_launch.sh; CPU-only (JAX on CPU), no GPU.
#
# Shared/fixed seeding: SEEDS is the explicit seed axis (per the shared-seed policy each
# (cell, rep) at a given seed uses the same derived per-role seeds). The smoke run pins a
# single fixed seed (0).
#
# AGENT selects the learner (default `value_gradient`; `actor_critic` = the continuous-time
# actor-critic ContinuousTimeActorCritic). Both agents build their features through the same
# `make_representation` factory, so the three representations are the same objects in both.
#
# Usage (Jean Zay login node, after git pull + uv sync):
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh                    # value-gradient, full
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh --smoke            # 2 cells x 3 reps x seed 0
#   ACCOUNT=<projid>@cpu AGENT=actor_critic bash .../rl_launch.sh                                   # actor-critic, full
#   ACCOUNT=<projid>@cpu AGENT=actor_critic bash .../rl_launch.sh --smoke                           # actor-critic smoke
#
# Partition / QoS (see ~/.claude/CLAUDE.md): cpu_p1 + qos_cpu-t3 (billed CPU); smoke uses
# qos_cpu-dev. Replotting/aggregation of the saved eval.pkl is a separate prepost step.
#
# OUTPUT ROOT — read this before changing it. Runs are written under $SCRATCH, NOT under the
# repository ($WORK), via RL_SIGNATURES_DATA_ROOT (see run_context.script_data_dir). $WORK has an
# inode quota far too small for a job array's output: job 544311 (the previous h1h2 RL array) died
# mid-array with `OSError: [Errno 122] Disk quota exceeded` writing to $WORK and produced ZERO run
# directories against 80 slurm logs. $SCRATCH is purged periodically, so rapatriate what matters.
# =============================================================================

set -euo pipefail

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set — are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:?Set ACCOUNT to your Jean Zay CPU account, e.g. ACCOUNT=abc@cpu}"
AGENT="${AGENT:-value_gradient}"
# The three implemented learners. policy_gradient was added in 5b410d6 and was NOT accepted here
# until now, so the third learner could not be dispatched at all -- an omission caught by the
# abstract-evidence audit, which observed that "three algorithms work" could not be a measurement
# over three algorithms when the launcher rejects one of them.
case "$AGENT" in
    value_gradient|actor_critic|policy_gradient) ;;
    *) echo "Error: AGENT must be 'value_gradient', 'actor_critic' or 'policy_gradient' (got '$AGENT')" >&2; exit 1 ;;
esac
# Runs go to $SCRATCH; only the slurm logs stay beside the repo-independent group dir.
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set — are you on Jean Zay?}/rl_campaigns/$NAME_PROJECT}"

REPS=(markovian raw_history signature)
SMOKE=""
[[ "${1:-}" == "--smoke" ]] && SMOKE="1"

if [[ -n "$SMOKE" ]]; then
    CELLS=(markovian linear_dde); SEEDS=(0); N_EPISODES=5
    QOS="qos_cpu-dev"; TIME="00:30:00"; DEBUG="true"
    GROUP="_debug_h1h2_rl_${AGENT}_$(date +%Y%m%d_%H%M%S)"
else
    # H1 test suite, selected on METHODOLOGICAL grounds only: span the two axes along which
    # "history helps a non-Markovian plant" can be tested, and cover each point once. The selection
    # criterion is the design, NOT the presence of prior value-gradient data on disk (which would
    # let the available evidence dictate the experiment). Each cell also qualifies on the measured
    # linearised delayed-LQR gate (run/study/hopfield_delay_impact.py, history-kernel ratio); the
    # ratio is initial-condition-invariant and carries the verdict, the +% gap varies with the
    # initial condition and is quoted at the deployment condition only.
    #
    #   AXIS 1 -- degree of non-Markovianity (the gate, 0 = Markov):
    #   AXIS 2 -- linear vs nonlinear in the delay; AXIS 3 -- state dimension.
    #
    #   markovian     0.000  dim 2   MARKOV control: A1 = 0 and tau = 0 EXACTLY, so history is
    #                                irrelevant and H1 MUST fail by construction (a falsification
    #                                control, not a finding). Exact CARE oracle.
    #   mg_chaotic    0.115  dim 1   nonlinear-in-delay, scalar, chaotic.
    #   hopfield_nonlinear 0.300 dim 2  NONLINEAR-in-delay, multichannel (n = 2). Fills the
    #                                nonlinear x multichannel cell no other plant covers (mg is
    #                                nonlinear but scalar). Its linear twin hopfield_linear is the
    #                                H2 instrument (eps moves H2 with H1 held fixed) and is dropped:
    #                                for H1 the pair is one plant, so keeping both tests nothing
    #                                extra. Swap to hopfield_linear if an exact delayed-LQR oracle
    #                                on this cell is wanted instead of nonlinear coverage.
    #   linear_dde    0.346  dim 1   linear-in-delay, scalar; A = 0 so the dynamics are ENTIRELY
    #                                delayed feedback -- the strongly-non-Markov linear end.
    #   platoon       0.398  dim 10  linear-in-delay, HIGH-DIMENSIONAL (connected-cruise-control,
    #                                5 vehicles). Adds the state-dimension axis the rest of the
    #                                suite lacks, and stress-tests raw_history's conditioning: its
    #                                feature dimension is ~3320, deep in the n < d / ker-G regime,
    #                                so a raw_history divergence here is a genuine H1 finding (with
    #                                trimming off and NaN guards on, it surfaces loudly). Earlier it
    #                                was excluded as "redundant with hopfield on the linear-
    #                                multichannel axis" -- true on that axis, FALSE on dimension
    #                                (hopfield is 2-D, platoon is 10-D), which is why it is included.
    #
    # Dropped: hopfield_linear (H2 twin, see above); hopfield_duffing, mg_limit_cycle (redundant on
    # every axis this suite spans). DISQUALIFIED: dadebo_cstr (near-Markovian, gate 0.009, must not
    # carry an H1 claim). All remain dispatchable in rl_array.slurm.
    #
    # Task count: this launcher submits ONE learner per invocation, so 5 cells x 3 reps x 5 seeds
    # = 75 tasks per learner; the full three-learner study (value_gradient + actor_critic +
    # policy_gradient) is 225 tasks across three submissions.
    CELLS=(markovian mg_chaotic hopfield_nonlinear linear_dde platoon); SEEDS=(0 1 2 3 4); N_EPISODES=1000
    QOS="qos_cpu-t3"; TIME="04:00:00"; DEBUG="false"
    GROUP="h1h2_rl_${AGENT}_$(date +%Y%m%d_%H%M%S)"
fi
N_TASKS=$(( ${#CELLS[@]} * ${#REPS[@]} * ${#SEEDS[@]} ))

# The group name carries the agent: aggregate_representation_study groups by
# (env, kind, capacity) and NOT by agent, so the two learners must never share a group.
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
SLURM_LOG_DIR="$EXPDIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,CELLS_STR=${CELLS[*]},REPS_STR=${REPS[*]},SEEDS_STR=${SEEDS[*]},N_EPISODES=$N_EPISODES,DEBUG=$DEBUG,AGENT=$AGENT,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"

echo "agent = $AGENT"
echo "cells = ${CELLS[*]} | reps = ${REPS[*]} | seeds = ${SEEDS[*]} | tasks = $N_TASKS"
echo "experiment_group = $GROUP  (n_episodes=$N_EPISODES)"
echo "data root = $DATA_ROOT   (\$SCRATCH, not \$WORK — see the header)"

# Compute array: cpu_p1 (CPU-only: JAX on CPU). 8 cores/task for JAX + signature threading.
ARRAY_JOB=$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition=cpu_p1 --qos="$QOS" \
    --array=0-$(( N_TASKS - 1 )) \
    --ntasks=1 --cpus-per-task=8 --hint=nomultithread \
    --time="$TIME" \
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err" \
    --export=ALL,"$EXPORTS" \
    "$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay/rl_array.slurm")
echo "submitted RL array job $ARRAY_JOB ($N_TASKS tasks, cpu_p1 / $QOS)"
echo
echo "runs land in: $EXPDIR/<timestamp>_${AGENT}_<cell>_<rep>_seed<seed>/  (eval.pkl has cost_reduction_pct)"
echo "watch: squeue -u \$USER"
