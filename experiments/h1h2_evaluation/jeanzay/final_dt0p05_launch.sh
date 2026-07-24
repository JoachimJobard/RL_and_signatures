#!/bin/bash
# Jean Zay -- launch the 10-seed FINAL for the uniform-cadence MG cell (MG_1D_limit_cycle_dt0p05,
# step_size 0.05), at the CLEAN-eta-frozen learning rates (frozen_dt0p05_clean.yaml, sigma = 0.3).
#
# The run is TRAINED at eval.x0_test = [1.0] -- the same training initial condition as the dt=0.25
# finals -- so that the only difference between the two campaigns is the control cadence. The
# reported relative optimality eta is NOT read from the native eval (which sits at the equilibrium,
# J0 ~ 0): it is recomputed afterwards from each checkpoint on the developed limit cycle, via
# experiments/h1h2_evaluation/reeval/reeval_group.py with x0 = 0.8, burn = 2000 (100 time units at
# dt = 0.05), exactly as the dt=0.25 finals were re-evaluated. Both campaigns are therefore evaluated
# on the cycle from identical training, isolating the cadence.
#
# Partition: cpu_p1 / qos_cpu-t3 (BILLED, 20 h cap). Walltime 15:00:00 from the measured timing probe
# (job 145536): the slowest arm is value-gradient + signature at 15.9 s/episode -> 8.8 h for the
# 2000-episode cap, so 15 h holds and qos_cpu-t3 suffices (no need for qos_cpu-t4).
#
# Usage (Jean Zay login, after sync):  bash .../final_dt0p05_launch.sh   (DRYRUN=1 to print only)
set -euo pipefail
: "${WORK:?WORK not set -- run through a login shell}"; : "${SCRATCH:?SCRATCH not set}"
REPO="${PATH_CONTENT_ROOT:-$WORK/git_repositories/RL_and_signatures}"
DATA_ROOT="${RL_SIGNATURES_DATA_ROOT:-$SCRATCH/rl_campaigns/RL_and_signatures}"
JZ="$REPO/experiments/h1h2_evaluation/jeanzay"
DRYRUN="${DRYRUN:-0}"
TS=$(date +%Y%m%d_%H%M%S)
GROUP="final_dt0p05_clean_MG_1D_limit_cycle_dt0p05_${TS}"
EXPDIR="$DATA_ROOT/data/main_unified/$GROUP"; LOGD="$EXPDIR/slurm"
SRC="$JZ/final_tasks_MG_1D_limit_cycle_dt0p05.tsv"
[ -s "$SRC" ] || { echo "missing manifest $SRC"; exit 1; }

EXPORTS="PATH_CONTENT_ROOT=$REPO,EXPERIMENT_GROUP=$GROUP,RL_SIGNATURES_DATA_ROOT=$DATA_ROOT"
EXPORTS+=",MANIFEST_FILE=$EXPDIR/tasks.tsv,PLANT=MG_1D_limit_cycle_dt0p05,MAX_TIME=85,T_SIM=85,GAMMA=0.10"
EXPORTS+=",N_EPISODES=2000,SIGMA=0.3,PATIENCE=10,EVAL_INTERVAL=50,MONITOR_SNR=false,EVAL_X0=[1.0]"
NT=$(wc -l < "$SRC")
SBATCH=(sbatch --parsable --account=akz@cpu --partition=cpu_p1 --qos=qos_cpu-t3
  --array=0-$((NT - 1)) --ntasks=1 --cpus-per-task=1 --hint=nomultithread --time=15:00:00
  --output="$LOGD/slurm-%A_%a.out" --error="$LOGD/slurm-%A_%a.err"
  --export=ALL,"$EXPORTS" "$JZ/final_runs_array.slurm")

if [ "$DRYRUN" = "1" ]; then
  echo "[dryrun] dt0p05 FINAL -> cpu_p1 (BILLED): $NT tasks, group=$GROUP"
  echo "         ${SBATCH[*]}"
else
  mkdir -p "$LOGD"; cp "$SRC" "$EXPDIR/tasks.tsv"
  JOB=$("${SBATCH[@]}")
  echo "submitted dt0p05 FINAL -> cpu_p1 (BILLED): job $JOB ($NT tasks, group=$GROUP)"
fi
