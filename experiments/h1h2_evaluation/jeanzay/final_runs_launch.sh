#!/bin/bash
# =============================================================================
# Jean Zay -- launch the 5-seed FINAL runs (NON-BILLED visu, akz@cpu).
# =============================================================================
# One array PER CELL (per-cell --time so the fast cells keep short walltime -> backfill priority).
# Each array = 3 learners x 3 representations x 5 seeds (45) + 1 oracle = 46 tasks, at the FROZEN
# alpha0 (frozen_alpha0.yaml). The seed axis is shared across learners/representations.
#
# Non-billed routing: visu (empty 80-CPU nodes). The qos_prepost cap is 3 nodes (~240 CPU) shared
# across prepost/compil/visu; 4x46 = 184 one-CPU tasks fit in a single wave. --time is set per cell
# from the measured selection Elapsed + margin (harmonic 6m, linear_dde 45m outlier, MG_limit 4m,
# MG_chaotic 12m) -- short walltime schedules via backfill instead of waiting behind long jobs.
#
# Usage (Jean Zay login, after git pull):
#   bash experiments/h1h2_evaluation/jeanzay/final_runs_launch.sh
#   SEEDS="0 1 2 3 4" DRYRUN=1 bash .../final_runs_launch.sh   # print sbatch lines, do not submit
# =============================================================================
set -euo pipefail
NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
PARTITION="${PARTITION:-visu}"          # non-billed
ACCOUNT="${ACCOUNT:-akz@cpu}"
SEEDS="${SEEDS:-0 1 2 3 4}"             # 5 fresh seeds, disjoint from the seed-42 selection
DRYRUN="${DRYRUN:-0}"
JZ_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"
TS=$(date +%Y%m%d_%H%M%S)

# Per-cell block: PLANT  MAX_TIME  T_SIM  GAMMA  N_EPISODES  WALLTIME
# (values read off the selection logs; walltime = measured Elapsed max + margin)
CELLS=(
  "harmonic_oscillator 10 10 0.5  300 00:30:00"
  "linear_dde_scalar   10 10 0.5  300 01:00:00"
  "MG_1D_limit_cycle   85 85 0.10 100 00:30:00"
  "MG_1D_chaotic       85 85 0.10 100 00:30:00"
)

echo "5-seed FINAL runs | seeds = [$SEEDS] | partition = $PARTITION (non-billed) | frozen alpha0"
for spec in "${CELLS[@]}"; do
  read -r PLANT MAX_TIME T_SIM GAMMA N_EPISODES WALLTIME <<< "$spec"
  SRC="$JZ_DIR/final_tasks_${PLANT}.tsv"
  [ -f "$SRC" ] || { echo "MISSING manifest $SRC -- run final_runs_manifest.py first"; exit 1; }

  GROUP="final_${PLANT}_${TS}"
  EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
  SLURM_LOG_DIR="$EXPDIR/slurm"; mkdir -p "$SLURM_LOG_DIR"
  MANIFEST="$EXPDIR/tasks.tsv"; cp "$SRC" "$MANIFEST"     # copy into run dir for provenance
  NT=$(wc -l < "$MANIFEST")

  EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
  EXPORTS+=",MANIFEST_FILE=$MANIFEST,PLANT=$PLANT,MAX_TIME=$MAX_TIME,T_SIM=$T_SIM,GAMMA=$GAMMA,N_EPISODES=$N_EPISODES"

  SBATCH=(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION"
    --array=0-$((NT - 1)) --ntasks=1 --cpus-per-task=1 --hint=nomultithread --time="$WALLTIME"
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err"
    --export=ALL,"$EXPORTS" "$JZ_DIR/final_runs_array.slurm")

  if [ "$DRYRUN" = "1" ]; then
    echo "[dryrun] $PLANT: $NT tasks, time=$WALLTIME, group=$GROUP"; echo "         ${SBATCH[*]}"
  else
    JOB=$("${SBATCH[@]}")
    echo "submitted $PLANT: job $JOB ($NT tasks, time=$WALLTIME, gamma=$GAMMA, n_ep=$N_EPISODES, group=$GROUP)"
  fi
done
echo "watch: squeue -u \$USER -p $PARTITION"
