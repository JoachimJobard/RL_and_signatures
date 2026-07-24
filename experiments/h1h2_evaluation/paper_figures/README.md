# Paper figures — reusable generation from a campaign snapshot

Every empirical figure of the AAAI submission is generated here, from a campaign data snapshot, by
re-evaluating the saved checkpoints on each cell's clean fixed initial condition. Nothing is retrained:
a figure loads `checkpoint_agent.pkl` (the trained controller) plus `config.yaml`, re-simulates the
controlled trajectory with the tested `collect_evaluation_data`, and plots it. This is the same
re-evaluation used to correct the confounded metric (validated bit-faithful).

## Regenerating for a NEW campaign — the one thing to edit

Edit **`campaign.py`** only. It holds the cell table: for each cell, its `title`, `group_glob` (the run
groups, relative to `DATA_ROOT`), `oracle_cost_reduction` (the η denominator), and the clean evaluation
initial condition (`x0`, `burn_steps`). Point these at the new campaign's groups and re-run
`generate_all.slurm`. No group name, oracle value, IC recipe, or seed is hard-coded anywhere else.

```bash
# from a Jean Zay login shell, after syncing the repo:
bash experiments/h1h2_evaluation/paper_figures/run_generate.sh          # submits generate_all.slurm on prepost
# or explicitly:
PAPER_FIGURES_DATA_ROOT=$WORK/paper_data/<snapshot> \
PAPER_FIGURES_OUT=$WORK/paper_data/<snapshot>/paper_figures_out \
  sbatch --account=akz@cpu --partition=prepost --time=01:00:00 --ntasks=1 --cpus-per-task=2 \
         --hint=nomultithread experiments/h1h2_evaluation/paper_figures/generate_all.slurm
```

Outputs land in `PAPER_FIGURES_OUT` as a PNG **and** a CSV per figure; pull them to the paper's
`figures/data/`. The paper can include the PNG directly or rebuild the figure in pgfplots from the CSV.

## Files

| File | Produces | Reads |
|---|---|---|
| `campaign.py` | the cell table + colours/labels (the config) | — |
| `_figlib.py` | shared helpers: `find_run`, `reeval_run`, plotting, `save` (PNG+CSV) | `src.training` |
| `fig_a1_uncontrolled.py` | **A1** uncontrolled regime per cell (setup) | no-control trajectory |
| `fig_a2_learning_curves.py` | **A2** noiseless-eval vs episode, 3 representations | `training_metrics.pkl` |
| `fig_a3_controlled.py` | **A3** agent vs oracle vs no-control vs target | checkpoints (agent + oracle) |
| `fig_comparison_grid.py` | one 3×3 grid per cell (the dashboard grids) | checkpoints |
| `extract_f2_relopt.py` | **F2** per-seed η CSV | `reeval_clean/summary.tsv` |
| `generate_all.slurm` / `run_generate.sh` | run all of the above on prepost | — |

## Conventions (consistent across every figure)

- Metric: relative optimality **η = (J₀ − J)/(J₀ − J★)** ∈ (−∞, 1]; a percentage is the rendering 100 η.
- Representation colour (categorical): markovian grey, raw history orange, signature blue.
- Stroke: **solid** = trained agent, **dashed** = analytic delayed-LQR oracle, **dotted** = no-control
  baseline and target line.
- Every re-evaluation is on the cell's fixed, seed-independent initial condition (so a seed varies only
  the training RNG). The delayed cells are placed on their developed attractor via `burn_steps`.

## What a checkpoint is

`checkpoint_agent.pkl` in each run directory holds the trained network parameters at the best-eval point
(best-checkpoint restore). It is the trained controller, frozen. The analytic oracle has none (its
control is closed-form), so `fig_a3` builds the oracle agent without loading a checkpoint
(`load_checkpoint=False`).
