#!/bin/bash
# =============================================================================
# Master Script: Launch all experiments sequentially
# =============================================================================
# This script runs all 4 experiments one after another.
# Each experiment runs with its own seed loop and output is logged.
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
echo "  2D System Comparison (delay_jax_unstable)"
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
    [1]="Signatures (depth 2)"
    [2]="No Signatures (standard CTAC)"
    [3]="Augmented State (delayed)"
    [4]="Value Gradient"
)

declare -A EXP_SCRIPTS=(
    [1]="run_exp1_signatures.sh"
    [2]="run_exp2_no_signatures.sh"
    [3]="run_exp3_augmented_state.sh"
    [4]="run_exp4_value_gradient.sh"
)

# Run experiments sequentially
for EXP in "${EXPERIMENTS[@]}"; do
    SCRIPT="${EXP_SCRIPTS[$EXP]}"
    LOG_FILE="${LOG_DIR}/exp${EXP}_$(date +%Y%m%d_%H%M%S).log"
    
    if [ ! -f "${SCRIPT_DIR}/${SCRIPT}" ]; then
        echo -e "${RED}[ERROR]${NC} Script not found: ${SCRIPT}"
        continue
    fi
    
    echo -e "${GREEN}[START]${NC} Experiment ${EXP}: ${EXP_NAMES[$EXP]}"
    echo "        Logging to: ${LOG_FILE}"
    echo ""
    
    # Make script executable and run
    chmod +x "${SCRIPT_DIR}/${SCRIPT}"
    
    if "${SCRIPT_DIR}/${SCRIPT}" > "$LOG_FILE" 2>&1; then
        echo -e "${GREEN}[DONE]${NC} Experiment ${EXP}: ${EXP_NAMES[$EXP]}"
    else
        EXIT_CODE=$?
        echo -e "${RED}[FAIL]${NC} Experiment ${EXP}: ${EXP_NAMES[$EXP]} (exit code: $EXIT_CODE)"
        echo "        Check log: ${LOG_FILE}"
    fi
    echo ""
done

echo "=============================================="
echo "  All experiments completed"
echo "=============================================="
echo ""
echo "Logs available in: ${LOG_DIR}"
ls -lah "${LOG_DIR}/"
