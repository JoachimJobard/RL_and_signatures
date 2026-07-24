#!/bin/bash
# Regenerate every paper figure from the campaign snapshot, on the non-billed prepost partition
# (post-processing of saved checkpoints = replot-class work). Submit through a LOGIN shell so $WORK
# and the module environment are set:
#   ssh jeanzay 'PAPER_FIGURES_DATA_ROOT=$WORK/paper_data/<snapshot> \
#                PAPER_FIGURES_OUT=$WORK/paper_data/<snapshot>/paper_figures_out \
#                sbatch --account=akz@cpu --partition=prepost --time=01:00:00 --ntasks=1 \
#                       --cpus-per-task=2 --hint=nomultithread \
#                       experiments/h1h2_evaluation/paper_figures/generate_all.slurm'
# (run_generate.sh wraps this.)
set -uo pipefail
export PATH_CONTENT_ROOT="${PATH_CONTENT_ROOT:?}"
source "$PATH_CONTENT_ROOT/bash_scripts/cluster/jeanzay/_environment.sh"   # cd's to the repo root
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export PAPER_FIGURES_DATA_ROOT="${PAPER_FIGURES_DATA_ROOT:?set PAPER_FIGURES_DATA_ROOT to the snapshot}"
OUT="${PAPER_FIGURES_OUT:?set PAPER_FIGURES_OUT to an output directory}"
mkdir -p "$OUT"
FIG="$PATH_CONTENT_ROOT/experiments/h1h2_evaluation/paper_figures"
echo "data=$PAPER_FIGURES_DATA_ROOT  out=$OUT"
python "$FIG/fig_a1_uncontrolled.py"     "$OUT"
python "$FIG/fig_a2_learning_curves.py"  "$OUT"
python "$FIG/fig_a3_controlled.py"       "$OUT"
python "$FIG/fig_comparison_grid.py"     "$OUT"
python "$FIG/extract_f2_relopt.py"       "$OUT/f2_relopt_10seed.csv"
echo "ALL_DONE"; ls -la "$OUT"
