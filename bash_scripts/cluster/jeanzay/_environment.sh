#!/bin/bash
# =============================================================================
# Jean Zay — shared environment helper for the RL_and_signatures launchers
# =============================================================================
# Sourced by the SLURM worker scripts (and usable on the login node) to:
#   - locate the project checkout on $WORK,
#   - activate the uv-managed virtual environment (.venv),
#   - print a self-contained diagnostic (host, SLURM vars, Python/JAX/devices).
#
# It deliberately does NOT run any heavy computation, so it is safe to source on
# a login node (the "no compute on login nodes" rule).
#
# Override points (export before sourcing, or rely on the defaults):
#   NAME_PROJECT      default: RL_and_signatures
#   PATH_CONTENT_ROOT default: $WORK/git_repositories/$NAME_PROJECT
#   PATH_VENV_BIN     default: $PATH_CONTENT_ROOT/.venv/bin/activate
# =============================================================================

NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
WORKDIR="${WORK:?WORK env var is not set — are you on Jean Zay?}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-$WORKDIR/git_repositories/$NAME_PROJECT}"
PATH_VENV_BIN="${PATH_VENV_BIN:-$PATH_CONTENT_ROOT/.venv/bin/activate}"

if [[ ! -f "$PATH_VENV_BIN" ]]; then
    echo "Error: venv activate script not found at $PATH_VENV_BIN" >&2
    echo "       Create it once on the login node with: cd $PATH_CONTENT_ROOT && uv sync" >&2
    echo "       (see bash_scripts/cluster/jeanzay/README.md)" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck source=/dev/null
source "$PATH_VENV_BIN"
cd "$PATH_CONTENT_ROOT" || { echo "Error: cannot cd into $PATH_CONTENT_ROOT" >&2; return 1 2>/dev/null || exit 1; }

# Make `import src...` work regardless of the install mode.
export PYTHONPATH="${PYTHONPATH:-}:$PATH_CONTENT_ROOT"
# JAX/threading hygiene on shared nodes.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TQDM_DISABLE=1

jeanzay_print_diagnostics() {
    echo "=============================================================="
    echo "Project       : $NAME_PROJECT"
    echo "Content root  : $PATH_CONTENT_ROOT"
    echo "Venv          : $PATH_VENV_BIN"
    echo "Host          : ${SLURMD_NODENAME:-$(hostname)}"
    echo "Python        : $(python --version 2>&1)  ($(command -v python))"
    python - <<'PYEOF' 2>/dev/null || echo "(jax not importable)"
import jax
jax.config.update("jax_enable_x64", True)
print(f"JAX           : {jax.__version__} | backend {jax.default_backend()} | devices {jax.devices()}")
PYEOF
    for v in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION \
             SLURM_JOB_QOS SLURM_ARRAY_TASK_ID SLURM_CPUS_PER_TASK SLURM_JOB_GPUS \
             SLURM_JOB_NODELIST SLURM_MEM_PER_NODE; do
        echo "$v = ${!v:-}"
    done
    echo "=============================================================="
    echo
}
