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
# AGENT selects the learner (default `value_gradient`; `signatures` = the continuous-time
# actor-critic CTACSignatureJAX). Both agents build their features through the same
# `make_representation` factory, so the three representations are the same objects in both.
#
# Usage (Jean Zay login node, after git pull + uv sync):
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh                    # value-gradient, full
#   ACCOUNT=<projid>@cpu bash experiments/h1h2_evaluation/jeanzay/rl_launch.sh --smoke            # 2 cells x 3 reps x seed 0
#   ACCOUNT=<projid>@cpu AGENT=signatures bash .../rl_launch.sh                                   # actor-critic, full
#   ACCOUNT=<projid>@cpu AGENT=signatures bash .../rl_launch.sh --smoke                           # actor-critic smoke
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
case "$AGENT" in
    value_gradient|signatures) ;;
    *) echo "Error: AGENT must be 'value_gradient' or 'signatures' (got '$AGENT')" >&2; exit 1 ;;
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
    # Minimal spanning set: each axis of the design is covered exactly once, at 30 tasks per cell
    # (3 representations x 5 seeds x 2 agents). Selected against the measured linearised
    # delayed-LQR gate of run/study/hopfield_delay_impact.py (history-kernel ratio / H1 cost gap);
    # the ratio is initial-condition-invariant and therefore carries the verdict, while the gap
    # varies with the initial condition (measured +6.26% to +39.09% across initial conditions on one
    # platoon plant) and is quoted at the deployment condition only.
    #   markovian           0.000 / +0.00%   falsification control: A1 = 0 and tau = 0 exactly, so
    #                                        H1 MUST fail. Exact CARE oracle (no discretisation).
    #   linear_dde          0.346 / +117.34% linear-in-delay, scalar; the largest measured gap.
    #   hopfield_linear     0.300 / +54.86%  linear-in-delay, multichannel (n = 2), conditioned.
    #   hopfield_nonlinear  0.300 / +54.86%  NONLINEAR-in-delay; identical linearisation to the arm
    #                                        above because phi_eps'(0) = 1, measured bit-identical.
    #                                        The pair is the suite's only controlled experiment: eps
    #                                        moves H2 with H1 held fixed by construction.
    #   mg_chaotic          0.115 / +89.09%  nonlinear-in-delay, scalar, chaotic.
    # Excluded deliberately: platoon (0.398 / +78.03%; redundant with hopfield_linear on the
    # linear-multichannel axis and the most expensive cell, raw-history dimension 3320);
    # hopfield_duffing (redundant with hopfield_nonlinear); mg_limit_cycle (redundant with
    # mg_chaotic). All three qualify on the gate and remain dispatchable in rl_array.slurm.
    # Excluded as DISQUALIFIED: dadebo_cstr, measured near-Markovian at 0.009 / +1.03% against a
    # qualification floor of about 0.10 -- it is present in main_unified's existing five-seed
    # benchmark and must not carry an H1 claim.
    CELLS=(markovian linear_dde hopfield_linear hopfield_nonlinear mg_chaotic); SEEDS=(0 1 2 3 4); N_EPISODES=1000
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
