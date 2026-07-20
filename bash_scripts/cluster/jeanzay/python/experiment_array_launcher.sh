#!/bin/bash
# =============================================================================
# Jean Zay — job-array launcher for RL_and_signatures ablations / sweeps
# =============================================================================
# Submits ONE SLURM array task per line of a "variants file" (each line is a set
# of Hydra overrides for main_unified.py), so independent variants/seeds run in
# parallel at the same compute cost as a sequential job. An optional finalize
# step (--dependency=afterok) aggregates afterwards on the non-billed prepost
# partition.
#
# Variants file format (see python/variants_example.txt):
#   - one Hydra-override string per line, e.g.:
#         agent=actor_critic env=MG_1D run_tag=depth2 agent.signature.depth=2
#   - blank lines and lines starting with '#' are ignored.
#   - all tasks are grouped under data/main_unified/<experiment-group>/ and each
#     task's folder is labelled by its run_tag.
#
# Partition / QoS routing (Jean Zay; see ~/.claude/CLAUDE.md):
#   --mode cpu  (default): cpu_p1, account akz@cpu, qos_cpu-t3  — light RL jobs,
#                cheaper and more parallel; JAX runs on CPU (small networks).
#   --mode gpu : gpu_p13 (V100) via account akz@v100, qos_gpu-t3, 1 GPU/task —
#                only when a variant genuinely needs a GPU.
#   Finalize   : prepost (NON-billed), account akz@cpu — pure replot/aggregation.
# For a smoke test, pass --qos qos_cpu-dev / qos_gpu-dev (max 2 h, schedules fast)
# and put --debug inside --common so the run dirs are flagged _debug_.
#
# Example:
#   bash bash_scripts/cluster/jeanzay/python/experiment_array_launcher.sh \
#       --variants-file bash_scripts/cluster/jeanzay/python/variants_example.txt \
#       --experiment-group mg_depth_ablation \
#       --common "agent.training.n_episodes=2000" \
#       --seed 0
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
NAME_PROJECT="RL_and_signatures"
PYTHON_SCRIPT_REL="main_unified.py"
VARIANTS_FILE=""                 # required
EXPERIMENT_GROUP=""              # default: <variants-basename>_<timestamp>
COMMON_OVERRIDES=""              # Hydra overrides shared by every task
SEED=0
WANDB_MODE="offline"             # compute nodes: offline is safe (sync later from login)
MODE="cpu"                       # cpu | gpu
FINALIZE_CMD=""                  # optional; '{GROUP_DIR}' substituted post-submit
PARTITION=""                     # optional override (e.g. prepost/visu — see PARTITIONS.md)

S_BATCH_TIME="04:00:00"
S_BATCH_TIME_FINALIZE="00:30:00"
S_BATCH_CPU_PER_TASK=10
S_BATCH_GPUS=1
S_BATCH_QOS=""                   # default chosen from MODE below
S_BATCH_ACCOUNT=""               # default chosen from MODE below

# ── Argument parsing ─────────────────────────────────────────────────────────
while (( $# )); do
    case "$1" in
        --variants-file)    VARIANTS_FILE="$2";    shift 2 ;;
        --experiment-group) EXPERIMENT_GROUP="$2"; shift 2 ;;
        --common)           COMMON_OVERRIDES="$2"; shift 2 ;;
        --seed)             SEED="$2";             shift 2 ;;
        --wandb-mode)       WANDB_MODE="$2";       shift 2 ;;
        --mode)             MODE="$2";             shift 2 ;;
        --partition)        PARTITION="$2";        shift 2 ;;
        --finalize)         FINALIZE_CMD="$2";     shift 2 ;;
        -A|--account)       S_BATCH_ACCOUNT="$2";  shift 2 ;;
        --qos)              S_BATCH_QOS="$2";      shift 2 ;;
        --time)             S_BATCH_TIME="$2";     shift 2 ;;
        --time-finalize)    S_BATCH_TIME_FINALIZE="$2"; shift 2 ;;
        --cpus-per-task)    S_BATCH_CPU_PER_TASK="$2";  shift 2 ;;
        --gpus)             S_BATCH_GPUS="$2";     shift 2 ;;
        --python-script)    PYTHON_SCRIPT_REL="$2"; shift 2 ;;
        -h|--help)          sed -n '2,46p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; sed -n '2,46p' "$0"; exit 1 ;;
    esac
done

[[ -z "$VARIANTS_FILE" ]] && { echo "Error: --variants-file is required." >&2; exit 1; }

# Mode-dependent SLURM routing (each directive's choice is documented above).
case "$MODE" in
    cpu) : "${S_BATCH_ACCOUNT:=akz@cpu}";  : "${S_BATCH_QOS:=qos_cpu-t3}" ;;
    gpu) : "${S_BATCH_ACCOUNT:=akz@v100}"; : "${S_BATCH_QOS:=qos_gpu-t3}" ;;
    *)   echo "Error: --mode must be cpu or gpu (got $MODE)." >&2; exit 1 ;;
esac

# Optional partition override (e.g. --partition prepost for light, non-billed
# diagnostics; see PARTITIONS.md). The non-billed partitions take no --qos, so drop it.
PARTITION_FLAG=()
QOS_FLAG=(--qos="$S_BATCH_QOS")
if [[ -n "$PARTITION" ]]; then
    PARTITION_FLAG=(--partition="$PARTITION")
    case "$PARTITION" in
        prepost|visu|archive|compil|compil_h100) QOS_FLAG=() ;;  # non-billed: no qos
    esac
fi

# ── Locate the project on $WORK ──────────────────────────────────────────────
WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="$WORKDIR/git_repositories/$NAME_PROJECT"
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT/$PYTHON_SCRIPT_REL"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/.venv/bin/activate"
PATH_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATH_WORKER_SLURM="$PATH_PARENT/job_array_batch_xp.slurm"
# Resolve the variants file relative to the project root if not absolute.
[[ "$VARIANTS_FILE" != /* ]] && VARIANTS_FILE="$PATH_CONTENT_ROOT/$VARIANTS_FILE"

for path in "$PATH_CONTENT_ROOT" "$PATH_PYTHON_SCRIPT" "$PATH_VENV_BIN" \
            "$PATH_WORKER_SLURM" "$VARIANTS_FILE"; do
    [[ -e "$path" ]] || { echo "Error: missing path $path" >&2; exit 1; }
done

# Default experiment group from the variants filename + a timestamp.
if [[ -z "$EXPERIMENT_GROUP" ]]; then
    EXPERIMENT_GROUP="$(basename "$VARIANTS_FILE" .txt)_$(date +%Y%m%d_%H%M%S)"
fi

# Count effective variant lines (skip blanks and comments).
N_VARIANTS=$(grep -cvE '^\s*(#|$)' "$VARIANTS_FILE")
(( N_VARIANTS > 0 )) || { echo "Error: 0 variants in $VARIANTS_FILE" >&2; exit 1; }
N_LAST=$(( N_VARIANTS - 1 ))

# SLURM logs go in a slurm/ subfolder of the ablation directory (CLAUDE.md rule).
GROUP_DIR="$PATH_CONTENT_ROOT/data/main_unified/$EXPERIMENT_GROUP"
SLURM_LOG_DIR="$GROUP_DIR/slurm"
mkdir -p "$SLURM_LOG_DIR"

echo "Project root   : $PATH_CONTENT_ROOT"
echo "Script         : $PATH_PYTHON_SCRIPT"
echo "Variants file  : $VARIANTS_FILE  ($N_VARIANTS tasks)"
echo "Experiment grp : $EXPERIMENT_GROUP  -> $GROUP_DIR"
echo "Mode/account   : $MODE / $S_BATCH_ACCOUNT / $S_BATCH_QOS"
echo "Common over.   : ${COMMON_OVERRIDES:-<none>}  | seed=$SEED | wandb=$WANDB_MODE"
echo

# ── Submit the array ─────────────────────────────────────────────────────────
GRES_FLAG=()
[[ "$MODE" == "gpu" ]] && GRES_FLAG=(--gres=gpu:"$S_BATCH_GPUS")

# Pass the worker variables via the submitting ENVIRONMENT (--export=ALL), NOT as an
# inline --export=VAR=val,VAR=val list: SLURM's inline --export grammar is
# comma-delimited, so a value that itself contains commas — e.g. a Hydra list override
# COMMON_OVERRIDES="... eval.x0_test=[0.15,-0.03,0.1,0.0] ..." — is silently truncated at
# the first inner comma (the remainder is mis-parsed as further VAR=... pairs). Exporting
# the variables into this shell and forwarding the whole environment with --export=ALL
# preserves commas verbatim. (Verified failure mode on 2026-06-11: Hydra raised
# "no viable alternative at input '[0.15'" because it received only "eval.x0_test=[0.15".)
export NAME_PROJECT PATH_CONTENT_ROOT PATH_VENV_BIN PATH_PYTHON_SCRIPT \
       VARIANTS_FILE COMMON_OVERRIDES SEED EXPERIMENT_GROUP WANDB_MODE

TRAIN_JOB_ID=$(sbatch --parsable \
    --job-name="rlsig_${EXPERIMENT_GROUP}" \
    --array=0-"$N_LAST" \
    --output="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.out" \
    --error="$SLURM_LOG_DIR/slurm-TRAIN-%A_%a.err" \
    --export=ALL \
    --account="$S_BATCH_ACCOUNT" \
    "${QOS_FLAG[@]+"${QOS_FLAG[@]}"}" \
    "${PARTITION_FLAG[@]+"${PARTITION_FLAG[@]}"}" \
    --time="$S_BATCH_TIME" \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --nodes=1 --ntasks-per-node=1 \
    --hint=nomultithread \
    "${GRES_FLAG[@]+"${GRES_FLAG[@]}"}" \
    "$PATH_WORKER_SLURM")
echo "TRAIN array job id: $TRAIN_JOB_ID  (--array=0-$N_LAST)"

# ── Optional finalize on the non-billed prepost partition ────────────────────
FINALIZE_JOB_ID=""
if [[ -n "$FINALIZE_CMD" ]]; then
    FINALIZE_CMD_RESOLVED="${FINALIZE_CMD//\{GROUP_DIR\}/$GROUP_DIR}"
    # prepost is NOT billed; akz@cpu still required because the user has several accounts.
    FINALIZE_JOB_ID=$(sbatch --parsable \
        --job-name="rlsig_${EXPERIMENT_GROUP}_finalize" \
        --dependency="afterok:${TRAIN_JOB_ID}" \
        --output="$SLURM_LOG_DIR/slurm-FINALIZE.out" \
        --error="$SLURM_LOG_DIR/slurm-FINALIZE.err" \
        --partition=prepost \
        --account=akz@cpu \
        --time="$S_BATCH_TIME_FINALIZE" \
        --cpus-per-task=4 \
        --nodes=1 --ntasks-per-node=1 \
        --hint=nomultithread \
        --wrap "cd '$PATH_CONTENT_ROOT' && source '$PATH_VENV_BIN' && export PYTHONPATH=\$PYTHONPATH:'$PATH_CONTENT_ROOT' && $FINALIZE_CMD_RESOLVED")
    echo "FINALIZE job id   : $FINALIZE_JOB_ID  (prepost, afterok)"
fi

cat <<EOF_SUMMARY

──────────────────────────────────────────────────────────────────────
  Submitted $N_VARIANTS tasks under experiment group: $EXPERIMENT_GROUP
  Results   : $GROUP_DIR
  Watch     : squeue -u "\$USER"
              tail -f $SLURM_LOG_DIR/slurm-TRAIN-${TRAIN_JOB_ID}_0.out
──────────────────────────────────────────────────────────────────────
EOF_SUMMARY
