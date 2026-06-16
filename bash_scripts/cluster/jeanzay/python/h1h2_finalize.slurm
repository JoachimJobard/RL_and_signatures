#!/bin/bash
# =============================================================================
# Jean Zay — H1/H2 finalize / aggregation (runs after the array, on prepost)
# =============================================================================
# Submitted by h1h2_launch.sh with --dependency=afterok on the array. Reads every
# $EXPDIR/<cell>_seed<seed>/summary.json, aggregates across seeds (mean +/- 95% CI of
# R^2 / cos / I) and reads off the H1/H2 verdicts. Pure post-processing of saved
# tensors -> the non-billed `prepost` partition (per the Jean Zay routing rules).
#
# Required env (broadcast via --export): PATH_CONTENT_ROOT EXPDIR
# =============================================================================

#SBATCH --job-name=h1h2_finalize
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in PATH_CONTENT_ROOT EXPDIR; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: required environment variable $var is not defined" >&2
        exit 1
    fi
done

# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

python run/study/h1h2_aggregate.py --parent "$EXPDIR" | tee "$EXPDIR/aggregate.txt"
echo
echo "Aggregated table written to $EXPDIR/aggregate.txt"
