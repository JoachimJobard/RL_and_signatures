#!/bin/bash
# =============================================================================
# Jean Zay — single-run launcher (GPU).
# =============================================================================
# Submits one main_unified.py run on a V100 GPU.
#
# Partition / QoS routing (Jean Zay; see ~/.claude/CLAUDE.md):
#   gpu_p13 (default GPU, V100) via account akz@v100, qos_gpu-t3 (<=20 h),
#   1 GPU. For a short smoke test pass --qos qos_gpu-dev (max 2 h). Use the GPU
#   path only when the workload genuinely needs it — the RL models here are small
#   and usually run fine on CPU (python_script_launcher.sh), which is cheaper.
#   Requires a CUDA-enabled JAX in the venv (see README one-time setup).
#
# Example:
#   bash bash_scripts/cluster/jeanzay/python/python_script_launcher_gpu.sh \
#       --args "agent=value_gradient env=MG_1D agent.training.n_episodes=2000 wandb.mode=offline"
# =============================================================================

set -euo pipefail

NAME_PROJECT="RL_and_signatures"
PYTHON_SCRIPT_REL="main_unified.py"
ARGS_PYTHON_SCRIPT=""
S_BATCH_ACCOUNT="akz@v100"
S_BATCH_QOS="qos_gpu-t3"
S_BATCH_TIME="04:00:00"
S_BATCH_CPU_PER_TASK=10
S_BATCH_GPUS=1

while (( $# )); do
    case "$1" in
        -p|--python-script) PYTHON_SCRIPT_REL="$2"; shift 2 ;;
        -a|--args)          ARGS_PYTHON_SCRIPT="$2"; shift 2 ;;
        -A|--account)       S_BATCH_ACCOUNT="$2";   shift 2 ;;
        --qos)              S_BATCH_QOS="$2";        shift 2 ;;
        --time)             S_BATCH_TIME="$2";       shift 2 ;;
        --cpus-per-task)    S_BATCH_CPU_PER_TASK="$2"; shift 2 ;;
        --gpus)             S_BATCH_GPUS="$2";       shift 2 ;;
        -h|--help)          sed -n '2,22p' "$0"; exit 0 ;;
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
    --job-name="rlsig_gpu" \
    --output="$SLURM_LOG_DIR/slurm-%j.out" \
    --error="$SLURM_LOG_DIR/slurm-%j.err" \
    --export=ALL,NAME_PROJECT="$NAME_PROJECT",PATH_CONTENT_ROOT="$PATH_CONTENT_ROOT",PATH_VENV_BIN="$PATH_VENV_BIN",PATH_PYTHON_SCRIPT="$PATH_PYTHON_SCRIPT",ARGS_PYTHON_SCRIPT="$ARGS_PYTHON_SCRIPT" \
    --account="$S_BATCH_ACCOUNT" \
    --qos="$S_BATCH_QOS" \
    --time="$S_BATCH_TIME" \
    --cpus-per-task="$S_BATCH_CPU_PER_TASK" \
    --gres=gpu:"$S_BATCH_GPUS" \
    --nodes=1 --ntasks-per-node=1 \
    --hint=nomultithread \
    "$PATH_WORKER_SLURM")

echo "Submitted GPU job $JOB_ID (account=$S_BATCH_ACCOUNT qos=$S_BATCH_QOS gpus=$S_BATCH_GPUS)"
echo "Follow: tail -f $SLURM_LOG_DIR/slurm-$JOB_ID.out"
