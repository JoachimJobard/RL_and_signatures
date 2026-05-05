#!/bin/bash

set -e

# Defaults: tuned for your cluster. Override with env vars if needed.
PARTITION="${PARTITION:-tau}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1440}"
CPUS="${CPUS:-4}"
MEM_GB="${MEM_GB:-16}"
GPUS="${GPUS:-1}"

if [ "$1" = "status" ]; then
    if command -v sinfo >/dev/null 2>&1; then
        sinfo -p "$PARTITION" -o "%20P %5a %.10l %16F %N"
    else
        echo "sinfo not found on this machine. Run this on the cluster login node."
    fi
    echo
    if command -v squeue >/dev/null 2>&1; then
        squeue -u "$USER"
    else
        echo "squeue not found on this machine."
    fi
    exit 0
fi

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  ./run_sweep.sh status"
    echo "  ./run_sweep.sh agent=signatures,base_jax seed=1,2,3"
    echo
    echo "Optional env vars: PARTITION TIMEOUT_MIN CPUS MEM_GB GPUS"
    exit 1
fi

CMD=(
    python main_unified.py -m launcher=slurm
    hydra.launcher.partition="$PARTITION"
    hydra.launcher.timeout_min="$TIMEOUT_MIN"
    hydra.launcher.cpus_per_task="$CPUS"
    hydra.launcher.mem_gb="$MEM_GB"
)

if [ "$GPUS" = "0" ]; then
    CMD+=(hydra.launcher.gpus_per_node=null)
else
    CMD+=(hydra.launcher.gpus_per_node="$GPUS")
fi

CMD+=("$@")

echo "Launching on partition=$PARTITION (cpus=$CPUS mem=${MEM_GB}GB gpus=$GPUS timeout=${TIMEOUT_MIN}min)"
"${CMD[@]}"
