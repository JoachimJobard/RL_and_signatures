#!/bin/bash
# =============================================================================
# Master Script: Launch all experiments in parallel
# =============================================================================
# This script starts all 4 experiments as background jobs and monitors them.
# Each experiment runs in its own process with output redirected to log files.
#
# Usage:
#   ./run_all_parallel.sh           # Launch all 4 experiments
#   ./run_all_parallel.sh 1 3       # Launch only experiments 1 and 3
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  MG Comparison - Parallel Run"
echo "=============================================="
echo ""

# Determine which experiments to run
if [ $# -eq 0 ]; then
    EXPERIMENTS=(1 2 3 4)
else
    EXPERIMENTS=("$@")
fi

# Experiment metadata
declare -A EXP_NAMES=(
    [1]="Signatures (depth sweep)"
    [2]="No Signatures (state-based)"
    [3]="Augmented State (delayed)"
    [4]="Value Gradient"
)

declare -A EXP_SCRIPTS=(
    [1]="run_exp1_signatures.sh"
    [2]="run_exp2_no_signatures.sh"
    [3]="run_exp3_augmented_state.sh"
    [4]="run_exp4_value_gradient.sh"
)

# Start experiments
declare -A PIDS
for EXP in "${EXPERIMENTS[@]}"; do
    SCRIPT="${EXP_SCRIPTS[$EXP]}"
    LOG_FILE="${LOG_DIR}/exp${EXP}_$(date +%Y%m%d_%H%M%S).log"
    
    if [ ! -f "${SCRIPT_DIR}/${SCRIPT}" ]; then
        echo -e "${RED}[ERROR]${NC} Script not found: ${SCRIPT}"
        continue
    fi
    
    echo -e "${GREEN}[START]${NC} Experiment ${EXP}: ${EXP_NAMES[$EXP]}"
    echo "        Logging to: ${LOG_FILE}"
    
    # Make script executable and launch in background
    chmod +x "${SCRIPT_DIR}/${SCRIPT}"
    "${SCRIPT_DIR}/${SCRIPT}" > "$LOG_FILE" 2>&1 &
    PIDS[$EXP]=$!
    
    echo "        PID: ${PIDS[$EXP]}"
    echo ""
done

echo "=============================================="
echo "  All experiments launched"
echo "=============================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f ${LOG_DIR}/exp1_*.log"
echo "  tail -f ${LOG_DIR}/exp2_*.log"
echo "  tail -f ${LOG_DIR}/exp3_*.log"
echo "  tail -f ${LOG_DIR}/exp4_*.log"
echo ""
echo "Check status:"
echo "  ps -p ${!PIDS[@]}"
echo ""
echo "Kill all experiments:"
echo "  kill ${!PIDS[@]}"
echo ""
echo "=============================================="
