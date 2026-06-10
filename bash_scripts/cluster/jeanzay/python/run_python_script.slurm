#!/bin/bash
# =============================================================================
# Jean Zay — single-run worker: runs one main_unified.py invocation in the venv.
# Invoked by python_script_launcher.sh (CPU) or python_script_launcher_gpu.sh.
#
# Required environment variables (broadcast via sbatch --export):
#   NAME_PROJECT PATH_CONTENT_ROOT PATH_VENV_BIN PATH_PYTHON_SCRIPT ARGS_PYTHON_SCRIPT
# =============================================================================

#SBATCH --job-name=rl_sig_single
#SBATCH --mail-type=ALL
#SBATCH --mail-user=remy.hosseinkhan@polytechnique.edu

set -uo pipefail

for var in NAME_PROJECT PATH_CONTENT_ROOT PATH_VENV_BIN PATH_PYTHON_SCRIPT; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: required environment variable $var is not defined" >&2
        exit 1
    fi
done

# SLURM copies this script to a spool dir, so BASH_SOURCE points there, not at the
# repo. Locate _environment.sh via the exported (absolute) PATH_CONTENT_ROOT.
# shellcheck source=/dev/null
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"
jeanzay_print_diagnostics

# shellcheck disable=SC2086  # intentional word-splitting of the override string
echo "Running: python $PATH_PYTHON_SCRIPT ${ARGS_PYTHON_SCRIPT:-}"
echo
python "$PATH_PYTHON_SCRIPT" ${ARGS_PYTHON_SCRIPT:-}
STATUS=$?
echo
echo "Finished with exit code $STATUS"
exit $STATUS
