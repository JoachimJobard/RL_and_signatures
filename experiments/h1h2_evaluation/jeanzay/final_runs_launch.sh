#!/bin/bash
# =============================================================================
# Jean Zay -- launch the 5-seed FINAL runs. Split across two partitions. akz@cpu.
# =============================================================================
# One array PER CELL. Each array = 3 learners x 3 representations x 5 seeds (45) + 1 oracle = 46
# tasks, at the FROZEN alpha0 (frozen_alpha0.yaml), n_episodes=1000 (full campaign budget). The
# seed axis is shared across learners/representations.
#
# TWO-PARTITION routing so no single partition hits its node cap:
#   * visu   (NON-BILLED, 4h max) -- the SHORT cells (harmonic, MG_limit). The qos_prepost node cap
#            (3 nodes ~240 CPU) is shared across prepost/compil/visu, so visu alone cannot absorb all
#            184 tasks under contention; it takes the cheap-to-finish cells for free.
#   * cpu_p1 (BILLED, qos_cpu-t3, 20h max) -- the LONG cells (linear_dde raw_history ~2.5h,
#            MG_chaotic signature ~2h). cpu_p1 has a large independent quota (~40960 CPU), so the
#            long poles run uncapped and do not queue behind the visu node cap. Cost ~90 CPU-hours.
# --time = measured selection Elapsed scaled to 1000 episodes + margin. Diverging markovian/
# raw_history abort early (guard) and free their slots. Expected wall-clock ~2.5h.
#
# Usage (Jean Zay login, after git pull):
#   bash experiments/h1h2_evaluation/jeanzay/final_runs_launch.sh
#   DRYRUN=1 bash .../final_runs_launch.sh          # print sbatch lines, do not submit
# =============================================================================
set -euo pipefail
NAME_PROJECT="${NAME_PROJECT:-RL_and_signatures}"
PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:-${WORK:?WORK not set -- are you on Jean Zay?}/git_repositories/$NAME_PROJECT}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-${SCRATCH:?SCRATCH not set}/rl_campaigns/$NAME_PROJECT}"
ACCOUNT="${ACCOUNT:-akz@cpu}"
SEEDS="${SEEDS:-0 1 2 3 4}"             # informational echo; the actual seeds live in the manifest
DRYRUN="${DRYRUN:-0}"
# Manifest family: "final" (5-seed final runs) or "stability" (conservative-lr search on the hard
# seed). Selects <MANIFEST_TAG>_tasks_<cell>.tsv and names the run group <GROUP_PREFIX>_<cell>_<ts>.
MANIFEST_TAG="${MANIFEST_TAG:-final}"
GROUP_PREFIX="${GROUP_PREFIX:-final}"
JZ_DIR="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/jeanzay"
TS=$(date +%Y%m%d_%H%M%S)

# Per-cell block: PLANT  MAX_TIME  T_SIM  GAMMA  N_EPISODES  WALLTIME  PARTITION  QOS ("-" = none)
CELLS=(
  "harmonic_oscillator 10 10 0.5  1000 01:30:00 visu   -"
  "MG_1D_limit_cycle   85 85 0.10 1000 02:00:00 visu   -"
  "linear_dde_scalar   10 10 0.5  1000 04:00:00 cpu_p1 qos_cpu-t3"
  "MG_1D_chaotic       85 85 0.10 1000 04:00:00 cpu_p1 qos_cpu-t3"
)

echo "5-seed FINAL runs | seeds = [$SEEDS] | n_episodes=1000 | visu (non-billed) + cpu_p1 (billed) | frozen alpha0"
for spec in "${CELLS[@]}"; do
  read -r PLANT MAX_TIME T_SIM GAMMA N_EPISODES WALLTIME CELL_PART CELL_QOS <<< "$spec"
  SRC="$JZ_DIR/${MANIFEST_TAG}_tasks_${PLANT}.tsv"
  # Skip a cell that has no (or an empty) manifest for this family -- lets a partial search or fix
  # target only a subset of cells without failing on the others.
  [ -s "$SRC" ] || { echo "skip $PLANT (no/empty $MANIFEST_TAG manifest)"; continue; }

  GROUP="${GROUP_PREFIX}_${PLANT}_${TS}"
  EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"
  SLURM_LOG_DIR="$EXPDIR/slurm"; mkdir -p "$SLURM_LOG_DIR"
  MANIFEST="$EXPDIR/tasks.tsv"; cp "$SRC" "$MANIFEST"     # copy into run dir for provenance
  NT=$(wc -l < "$MANIFEST")

  EXPORTS="PATH_CONTENT_ROOT=$PATH_CONTENT_ROOT,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
  EXPORTS+=",MANIFEST_FILE=$MANIFEST,PLANT=$PLANT,MAX_TIME=$MAX_TIME,T_SIM=$T_SIM,GAMMA=$GAMMA,N_EPISODES=$N_EPISODES"

  SBATCH=(sbatch --parsable --account="$ACCOUNT" --partition="$CELL_PART")
  [ "$CELL_QOS" != "-" ] && SBATCH+=(--qos="$CELL_QOS")
  SBATCH+=(--array=0-$((NT - 1)) --ntasks=1 --cpus-per-task=1 --hint=nomultithread --time="$WALLTIME"
    --output="$SLURM_LOG_DIR/slurm-%A_%a.out" --error="$SLURM_LOG_DIR/slurm-%A_%a.err"
    --export=ALL,"$EXPORTS" "$JZ_DIR/final_runs_array.slurm")

  BILLED=$([ "$CELL_PART" = "visu" ] && echo "non-billed" || echo "BILLED")
  if [ "$DRYRUN" = "1" ]; then
    echo "[dryrun] $PLANT -> $CELL_PART ($BILLED): $NT tasks, time=$WALLTIME, group=$GROUP"
    echo "         ${SBATCH[*]}"
  else
    JOB=$("${SBATCH[@]}")
    echo "submitted $PLANT -> $CELL_PART ($BILLED): job $JOB ($NT tasks, time=$WALLTIME, n_ep=$N_EPISODES, group=$GROUP)"
  fi
done
echo "watch: squeue -u \$USER"
