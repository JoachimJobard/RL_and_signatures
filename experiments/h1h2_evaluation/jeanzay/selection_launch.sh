#!/bin/bash
# =============================================================================
# Jean Zay -- launch the pre-registered SELECTION sweep (learning-rate grid x sigma grid). akz@cpu.
# =============================================================================
# See ../SELECTION_AND_FINAL_PROTOCOL.md. One SLURM array per (cell, sigma). Each array = the full
# learning-rate grid x 9 combinations x 3 seeds (126 tasks) + 1 oracle line, at cap n_episodes=2000
# with patience=10 (ADAPTIVE budget: each combination trains to its own plateau, best checkpoint
# restored on stop) and monitor_snr=true (the SNR stability gate is recorded). sigma in {0.3,0.5} is
# the array-level SIGMA env var -- shared across the nine combinations of a cell -- so the grid is
# launched once per sigma, in distinct run groups (representation-neutral per-cell sigma selection
# happens at aggregation, never per combination).
#
# Partition routing (audited against the global routing table):
#   * harmonic_oscillator -> visu (NON-BILLED, 4h cap): fast ODE-control tasks clear quickly even
#     under the shared qos_prepost node cap; no billed hours needed for the cheap cell.
#   * linear_dde_scalar, MG_1D_limit_cycle -> cpu_p1 (BILLED, qos_cpu-t3, 20h cap): many tasks and a
#     large independent quota, so the bulk of the sweep runs at throughput without queueing behind
#     the visu node cap. --time is the 2000-episode cap scaled from measured 1000-episode Elapsed
#     plus margin; adaptive early-stopping means most tasks finish well before it, and diverging
#     markovian/raw_history combos abort early (guard) and free their slots.
#
# Usage (Jean Zay login, after git pull):
#   bash experiments/h1h2_evaluation/jeanzay/selection_launch.sh
#   DRYRUN=1 bash .../selection_launch.sh        # print sbatch lines, do not submit, create nothing
#   SIGMAS="0.5" bash .../selection_launch.sh     # single sigma
# =============================================================================
set -euo pipefail
NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:-akz@cpu}"
SEEDS="${SEEDS:-40 41 42}"       # the selection seeds (disjoint from the test seeds 0..4)
SIGMAS="${SIGMAS:-0.3 0.5}"      # per-cell exploration grid; one array per sigma
DRYRUN="${DRYRUN:-0}"
# Optional space-separated PLANT filter, e.g. ONLY_CELLS="MG_1D_limit_cycle_dt0p05". Empty = every
# cell in CELLS. Added so a single cell can be (re)swept without resubmitting the cells whose
# selection is already frozen -- relaunching those would silently create a second, competing group.
ONLY_CELLS="${ONLY_CELLS:-}"
JZ_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"
TS=$(date +%Y%m%d_%H%M%S)

# Budget fixed a priori, identical for every cell and (later) the final runs.
N_EPISODES=2000; PATIENCE=10; MONITOR_SNR=true; EVAL_INTERVAL=50

# Per-cell block: PLANT  MAX_TIME  T_SIM  GAMMA  WALLTIME  PARTITION  QOS ("-" = none)
CELLS=(
  "harmonic_oscillator 10 10 0.5  04:00:00 visu   -"
  "linear_dde_scalar   10 10 0.5  08:00:00 cpu_p1 qos_cpu-t3"
  "MG_1D_limit_cycle   85 85 0.10 15:00:00 cpu_p1 qos_cpu-t3"
)

# sbatch with bounded retry: Jean Zay's submit RPC intermittently returns "Resource temporarily
# unavailable" under scheduler load. Retry a few times before giving up, so one transient error does
# not abort the whole multi-array launch (which would leave a partial submission at a fixed TS).
submit() {
  local attempt jobid
  for attempt in 1 2 3 4 5 6; do
    if jobid=$("$@" 2>/dev/null); then printf '%s' "$jobid"; return 0; fi
    echo "  submit attempt $attempt hit a transient scheduler error; retrying in 20s" >&2
    sleep 20
  done
  return 1
}

# Regenerate the per-cell selection manifests for the requested seeds (idempotent for the defaults).
python "$JZ_DIR/selection_manifest.py" "$SEEDS"

echo "SELECTION sweep | seeds=[$SEEDS] | sigma in {$SIGMAS} | cap n_ep=$N_EPISODES patience=$PATIENCE snr=$MONITOR_SNR"
for SIGMA in $SIGMAS; do
  STAG=$(echo "$SIGMA" | tr -d '.')      # 0.3 -> "03" for the group name
  for spec in "${CELLS[@]}"; do
    read -r PLANT MAX_TIME T_SIM GAMMA WALLTIME CELL_PART CELL_QOS <<< "$spec"
    if [ -n "$ONLY_CELLS" ] && ! grep -qw -- "$PLANT" <<< "$ONLY_CELLS"; then continue; fi
    SRC="$JZ_DIR/selection_tasks_${PLANT}.tsv"
    [ -s "$SRC" ] || { echo "skip $PLANT (no/empty selection manifest)"; continue; }

    GROUP="selection_${PLANT}_s${STAG}_${TS}"
    EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
    SLURM_LOG_DIR="$EXPDIR/slurm"
    MANIFEST="$EXPDIR/tasks.tsv"
    NT=$(wc -l < "$SRC")

    EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
    EXPORTS+=",MANIFEST_FILE=$MANIFEST,PLANT=$PLANT,MAX_TIME=$MAX_TIME,T_SIM=$T_SIM,GAMMA=$GAMMA,N_EPISODES=$N_EPISODES"
    EXPORTS+=",SIGMA=$SIGMA,PATIENCE=$PATIENCE,MONITOR_SNR=$MONITOR_SNR,EVAL_INTERVAL=$EVAL_INTERVAL"

    SBATCH=(sbatch --parsable --account="$ACCOUNT" --partition="$CELL_PART")
    [ "$CELL_QOS" != "-" ] && SBATCH+=(--qos="$CELL_QOS")
    SBATCH+=(--array=0-$((NT - 1)) --ntasks=1 --cpus-per-task=1 --hint=nomultithread --time="$WALLTIME"
      --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err"
      --export=ALL,"$EXPORTS" "$JZ_DIR/final_runs_array.slurm")

    BILLED=$([ "$CELL_PART" = "visu" ] && echo "non-billed" || echo "BILLED")
    if [ "$DRYRUN" = "1" ]; then
      echo "[dryrun] $PLANT sigma=$SIGMA -> $CELL_PART ($BILLED): $NT tasks, time=$WALLTIME, group=$GROUP"
      echo "         ${SBATCH[*]}"
    else
      mkdir -p "$SLURM_LOG_DIR"                     # only for a real submission
      cp "$SRC" "$MANIFEST"                          # copy manifest into run dir for provenance
      if JOB=$(submit "${SBATCH[@]}"); then
        echo "submitted $PLANT sigma=$SIGMA -> $CELL_PART ($BILLED): job $JOB ($NT tasks, time=$WALLTIME, group=$GROUP)"
      else
        echo "FAILED to submit $PLANT sigma=$SIGMA after retries -- leaving its (empty) group dir; rerun to retry" >&2
      fi
      sleep "${SUBMIT_SPACING:-25}"   # space submissions: a burst of six 127-task arrays can re-choke a recovering slurmdbd
    fi
  done
done
echo "watch: squeue -u \$USER"
