#!/bin/bash
# =============================================================================
# Jean Zay — interactive compute shell for quick code-path checks.
# =============================================================================
# Drops you on a compute node with the project venv activated.
#
# Partition / QoS routing (Jean Zay; see ~/.claude/CLAUDE.md):
#   default (CPU): prepost partition — NON-billed, up to 20 h, ideal for light
#                  CPU checks / replots without spending the allocation.
#   --gpu:         gpu_p13 (V100) via account akz@v100, qos_gpu-dev (max 2 h,
#                  high priority) — for GPU code-path checks.
#
# Usage:
#   bash bash_scripts/cluster/jeanzay/python/run_interactive_job.sh          # CPU (prepost)
#   bash bash_scripts/cluster/jeanzay/python/run_interactive_job.sh --gpu    # 1 V100, qos_gpu-dev
# =============================================================================

set -euo pipefail

NAME_PROJECT="RL_and_signatures"
TIME="01:00:00"
USE_GPU=0

while (( $# )); do
    case "$1" in
        --gpu)  USE_GPU=1; shift ;;
        --time) TIME="$2"; shift 2 ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="$WORKDIR/git_repositories/$NAME_PROJECT"
PATH_VENV_BIN="$PATH_CONTENT_ROOT/.venv/bin/activate"
[[ -f "$PATH_VENV_BIN" ]] || { echo "Error: venv not found at $PATH_VENV_BIN (run 'uv sync' once)." >&2; exit 1; }

if (( USE_GPU )); then
    echo "Requesting 1 V100 on gpu_p13 (account akz@v100, qos_gpu-dev, $TIME)..."
    srun --pty \
        --account=akz@v100 --qos=qos_gpu-dev \
        --gres=gpu:1 --cpus-per-task=10 \
        --nodes=1 --ntasks-per-node=1 --hint=nomultithread \
        --time="$TIME" \
        bash --rcfile <(echo "source '$PATH_VENV_BIN'; cd '$PATH_CONTENT_ROOT'; export PYTHONPATH=\$PYTHONPATH:'$PATH_CONTENT_ROOT'")
else
    echo "Requesting a CPU shell on prepost (non-billed, $TIME)..."
    srun --pty \
        --partition=prepost --account=akz@cpu \
        --cpus-per-task=4 \
        --nodes=1 --ntasks-per-node=1 --hint=nomultithread \
        --time="$TIME" \
        bash --rcfile <(echo "source '$PATH_VENV_BIN'; cd '$PATH_CONTENT_ROOT'; export PYTHONPATH=\$PYTHONPATH:'$PATH_CONTENT_ROOT'")
fi
