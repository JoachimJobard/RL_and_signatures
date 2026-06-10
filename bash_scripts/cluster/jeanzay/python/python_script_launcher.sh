#!/bin/bash
# =============================================================================
# Jean Zay — single-run launcher (CPU).
# =============================================================================
# Submits one main_unified.py run on the default CPU partition.
#
# Partition / QoS routing (Jean Zay; see ~/.claude/CLAUDE.md):
#   cpu_p1 (default CPU, 40 cores/node) via account akz@cpu, qos_cpu-t3 (<=20 h).
#   For a short smoke test pass --qos qos_cpu-dev (max 2 h, schedules fast) and
#   put debug=true in --args so the run dir is flagged _debug_.
#
# Example:
#   bash bash_scripts/cluster/jeanzay/python/python_script_launcher.sh \
#       --args "agent=signatures env=MG_1D agent.training.n_episodes=2000 wandb.mode=offline"
# =============================================================================

set -euo pipefail

NAME_PROJECT="RL_and_signatures"
PYTHON_SCRIPT_REL="main_unified.py"
ARGS_PYTHON_SCRIPT=""
S_BATCH_ACCOUNT="akz@cpu"
S_BATCH_QOS="qos_cpu-t3"
S_BATCH_TIME="04:00:00"
S_BATCH_CPU_PER_TASK=10

while (( $# )); do
    case "$1" in
        -p|--python-script) PYTHON_SCRIPT_REL="$2"; shift 2 ;;
        -a|--args)          ARGS_PYTHON_SCRIPT="$2"; shift 2 ;;
        -A|--account)       S_BATCH_ACCOUNT="$2";   shift 2 ;;
        --qos)              S_BATCH_QOS="$2";        shift 2 ;;
        --time)             S_BATCH_TIME="$2";       shift 2 ;;
        --cpus-per-task)    S_BATCH_CPU_PER_TASK="$2"; shift 2 ;;
        -h|--help)          sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="$WORKDIR/git_repositories/$NAME_PROJECT"
PATH_PYTHON_SCRIPT="$PATH_CONTENT_ROOT/$PYTHON_SCRIPT_REL"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/.venv/bin/activate"
PATH_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PATH_WORKER_SLURM="$PATH_PARENT/run_python_script.slurm"
SLURM_LOG_DIR="$WORKDIR/logs/$NAME_PROJECT"
mkdir -p "$SLURM_LOG_DIR"

for path in "$PATH_CONTENT_ROOT" "$PATH_PYTHON_SCRIPT" "$PATH_VENV_BIN" "$PATH_WORKER_SLURM"; do
    [[ -e "$path" ]] || { echo "Error: missing path $path" >&2; exit 1; }
done

JOB_ID=$(sbatch --parsable \
    --job-name="rlsig_cpu" \
    --output="$SLURM_LOG_DIR/slurm-%j.out" \
    --error="$SLURM_LOG_DIR/slurm-%j.err" \
    --export=ALL,NAME_PROJECT="$NAME_PROJECT",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",PATH_VENV_BIN="$PATH_VENV_BIN",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT" \
    --account="$S_BATCH_ACCOUNT" \
    --qos="$S_BATCH_QOS" \
    --time="$S_BATCH_TIME" \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --nodes=1 --ntasks-per-node=1 \
    --hint=nomultithread \
    "$PATH_WORKER_SLURM")

echo "Submitted CPU job $JOB_ID (account=$S_BATCH_ACCOUNT qos=$S_BATCH_QOS)"
echo "Follow: tail -f $SLURM_LOG_DIR/slurm-$JOB_ID.out"
