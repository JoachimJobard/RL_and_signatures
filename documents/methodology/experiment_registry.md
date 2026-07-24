# Experiment registry — index

_Derived artefact: auto-generated from `experiment_registry.jsonl` by `render_registry_index.py`. Do not edit by hand — append to the JSONL ledger and re-run the generator._

Records: 9 (5 dated).

## Index

| Date | Kind | ID | Summary |
|---|---|---|---|
| — | prior_run | `oracle_20260616_canonical` | canonical baseline for the campaign |
| — | prior_run | `mg_offsheet_20260630_discovery` | single-seed discovery motivating the 5-seed campaign |
| — | preregistration | `prereg_offsheet_h2_20260702` | H3: signature advantage is sample-efficiency, not representation |
| — | planned_campaign | `campaign_offsheet_5seed_20260702` | PREPARED_NOT_SUBMITTED |
| 2026-07-17 | preregistration | `prereg_h1_20260717` | H1: the history/signature representation advantage grows with delay; the instantaneous (markovian) state is sufficient at delay 0 |
| 2026-07-23 | campaign_run | `campaign_h1_primary_20260723` | folder name 'final_dt0.25' is a BATCH label, NOT a cadence: harmonic and linear_dde run at dt=0.05; only the MG group inside is dt=0.25 |
| 2026-07-24 | selection_sweep | `selection_h1_dt0p05_20260724` | pre-registered learning-rate/sigma selection for the dt=0.05 MG cell, on HELD-OUT seeds; appears in NO final evaluation |
| 2026-07-24 | campaign_run | `campaign_h1_mg_dt0p05_20260724` | cadence-robustness re-run; the USED Mackey-Glass cell, supersedes the dt=0.25 MG |
| 2026-07-24 | dataset_snapshot | `snapshot_h1_aaai2027_20260724` | AUTHORITATIVE dataset for the AAAI-2027 H1 submission; byte-verified; $SCRATCH originals purged |

## Records

### `oracle_20260616_canonical`

- **kind**: prior_run
- **role**: canonical baseline for the campaign
- **verdict**: H2 HOLDS on MG (CI-backed); H1 negative control OK
- **pipeline**: h1h2_evaluation (oracle/least-squares)
- **cells**: markovian, linear_dde, platoon, mg_limit_cycle, mg_chaotic
- **n_seeds**: 5
- **path**: data/h1h2_report_runs/oracle/20260616_161650_on_off_sheet_array

### `mg_offsheet_20260630_discovery`

- **kind**: prior_run
- **role**: single-seed discovery motivating the 5-seed campaign
- **verdict**: raw_history matches/beats signature at off>=2x (contests H2-representational)
- **pipeline**: h1h2_evaluation (off-sheet scaling)
- **cells**: mg_limit_cycle, mg_chaotic
- **n_seeds**: 1
- **ratios**: 1x, 2x, 4x
- **path**: data/h1h2_report_runs/sweeps/h1h2_mg_offsheet_20260630_164840

### `prereg_offsheet_h2_20260702`

- **kind**: preregistration
- **hypothesis**: H3: signature advantage is sample-efficiency, not representation
- **path**: documents/methodology/2026-07-02_preregistration_offsheet_h2_replication.md

### `campaign_offsheet_5seed_20260702`

- **kind**: planned_campaign
- **status**: PREPARED_NOT_SUBMITTED
- **pipeline**: h1h2_evaluation (oracle)
- **launcher**: bash_scripts/cluster/jeanzay/python/h1h2_offsheet_array.slurm
- **tasks_file**: bash_scripts/cluster/jeanzay/python/h1h2_offsheet_5seed_tasks.tsv
- **tasks**: 105
- **cells_primary**: mg_limit_cycle, mg_chaotic, hopfield_nonlinear, hopfield_duffing
- **cells_null_check**: markovian, linear_dde, platoon
- **seeds**: 0, 1, 2, 3, 4
- **ratios**:
  - `1x`: 500
  - `2x`: 1000
  - `4x`: 2000
- **output_root**: $SCRATCH/h1h2_offsheet_5seed_<date> (Jean Zay)
- **submitted_by**: None
- **job_ids**: 

### `prereg_h1_20260717`

- **kind**: preregistration
- **date**: 2026-07-17
- **hypothesis**: H1: the history/signature representation advantage grows with delay; the instantaneous (markovian) state is sufficient at delay 0
- **cells**: harmonic, linear_dde, MG_limit_cycle
- **learners**: value_gradient, policy_gradient, actor_critic
- **representations**: markovian, raw_history, signature
- **path**: documents/methodology/2026-07-17_preregistration_h1_representation_learners.md
- **metric**: eta = (J0 - J)/(J0 - Jstar), cost-minimisation

### `campaign_h1_primary_20260723`

- **kind**: campaign_run
- **date**: 2026-07-23
- **pipeline**: main_unified (continuous-time RL, JAX)
- **cluster**: Jean Zay (akz@cpu)
- **commit**: 56effcd
- **branch**: pg-rollout-speedup
- **launcher**: experiments/h1h2_evaluation/jeanzay/final_runs_launch.sh + final_runs_array.slurm
- **manifests**: final_tasks_harmonic_oscillator.tsv, final_tasks_linear_dde_scalar.tsv, final_tasks_MG_1D_limit_cycle.tsv
- **cells**:
  - `harmonic`:
    - `dt`: 0.05
    - `T`: 10
    - `gamma`: 0.5
    - `delay`: 0
    - `state_dim`: 2
    - `window`: 10
  - `linear_dde`:
    - `dt`: 0.05
    - `T`: 10
    - `gamma`: 0.5
    - `delay`: 1
    - `state_dim`: 1
    - `window`: 23
  - `MG_limit_cycle`:
    - `dt`: 0.25
    - `T`: 85
    - `gamma`: 0.1
    - `delay`: 6
    - `state_dim`: 1
    - `window`: 27
    - `status`: SUPERSEDED by campaign_h1_mg_dt0p05_20260724; dropped from the paper
- **learners**: value_gradient, policy_gradient, actor_critic
- **representations**: markovian, raw_history, signature
- **n_seeds**: 10
- **seeds**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **seed_groups**:
  - `_223541`: seeds 0-4
  - `_230752`: seeds 5-9
- **sigma**: 0.3
- **n_episodes**: 2000
- **frozen_lr**: frozen_final.yaml
- **output_snapshot**: $WORK/paper_data/aaai2027_h1_clean_20260724/final_dt0.25/
- **output_original**: $SCRATCH/rl_campaigns/RL_and_signatures/data/main_unified/ (purged)
- **caveat**: folder name 'final_dt0.25' is a BATCH label, NOT a cadence: harmonic and linear_dde run at dt=0.05; only the MG group inside is dt=0.25

### `selection_h1_dt0p05_20260724`

- **kind**: selection_sweep
- **date**: 2026-07-24
- **role**: pre-registered learning-rate/sigma selection for the dt=0.05 MG cell, on HELD-OUT seeds; appears in NO final evaluation
- **pipeline**: main_unified (lr x sigma grid, adaptive budget)
- **commit**: bb7fc6b
- **launcher**: experiments/h1h2_evaluation/jeanzay/selection_launch.sh + sweep_array.slurm
- **held_out_seeds**: 40, 41, 42
- **sigma_variants**:
  - `s03`: 0.3
  - `s05`: 0.5
- **groups**: selection_MG_1D_limit_cycle_dt0p05_s03_20260724_045042, selection_MG_1D_limit_cycle_dt0p05_s05_20260724_045042
- **output_snapshot**: $WORK/paper_data/aaai2027_h1_clean_20260724/selection_dt0.05/
- **note**: two-stage: %opt sweep on the held-out seeds, then conservative stability re-tune checked on the hardest eval seed (seed 1); sigma=0.3 frozen (frozen_dt0p05_clean.yaml)

### `campaign_h1_mg_dt0p05_20260724`

- **kind**: campaign_run
- **date**: 2026-07-24
- **role**: cadence-robustness re-run; the USED Mackey-Glass cell, supersedes the dt=0.25 MG
- **verdict**: value_gradient+signature useful 10/10 (eta median ~0.89); markovian & raw_history fail; actor_critic+signature ~0.97 but 4/10 NaN-abort
- **pipeline**: main_unified (continuous-time RL, JAX)
- **cluster**: Jean Zay (akz@cpu), cpu_p1 / qos_cpu-t3 (billed, 15h)
- **commit**: 56effcd
- **branch**: pg-rollout-speedup
- **launcher**: experiments/h1h2_evaluation/jeanzay/final_dt0p05_launch.sh + final_runs_array.slurm
- **manifest**: final_tasks_MG_1D_limit_cycle_dt0p05.tsv
- **cell**: MG_1D_limit_cycle_dt0p05
- **cell_params**:
  - `dt`: 0.05
  - `T`: 85
  - `gamma`: 0.1
  - `delay`: 6
  - `state_dim`: 1
  - `window`: 123
- **learners**: value_gradient, policy_gradient, actor_critic
- **representations**: markovian, raw_history, signature
- **n_seeds**: 10
- **seeds**: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- **sigma**: 0.3
- **n_episodes**: 2000
- **eval_x0**: 0.8
- **frozen_lr**: frozen_dt0p05_clean.yaml
- **group**: final_dt0p05_clean_MG_1D_limit_cycle_dt0p05_20260724_112745
- **output_snapshot**: $WORK/paper_data/aaai2027_h1_clean_20260724/final_dt0.05/

### `snapshot_h1_aaai2027_20260724`

- **kind**: dataset_snapshot
- **date**: 2026-07-24
- **role**: AUTHORITATIVE dataset for the AAAI-2027 H1 submission; byte-verified; $SCRATCH originals purged
- **commit**: 56effcd
- **branch**: pg-rollout-speedup
- **cells_used**: harmonic, linear_dde, MG_dt0.05
- **path**: $WORK/paper_data/aaai2027_h1_clean_20260724/ (Jean Zay, rech/oym/ucd32aq)
- **contents**: final_dt0.25/, final_dt0.05/, selection_dt0.05/, oracle_audit/, f2_relopt_10seed.csv, frozen_final.yaml, frozen_dt0p05_clean.yaml, paper_figures_out/, README.md
- **metric**: eta = (J0 - J)/(J0 - Jstar), cost-minimisation; percentage = 100*eta
- **oracle_denominators**:
  - `harmonic`: 94.28
  - `linear_dde`: 94.06
  - `MG_dt0.25`: 99.38
  - `MG_dt0.05`: 99.2
- **evaluation**: clean fixed-IC re-eval from best-eval checkpoints (reeval_clean/summary.tsv), correcting the random-per-seed-IC fallback
- **figures_infra**: experiments/h1h2_evaluation/paper_figures/ (commit f94fa7e; campaign.py CELLS is the authoritative cell selection)
- **caveats**: 'final_dt0.25/' is a batch label: harmonic & linear_dde inside run at dt=0.05, MG dt=0.25 SUPERSEDED and dropped; f2_relopt_10seed.csv still carries a MG_dt0.25 cell to be filtered (use cell in {harmonic, linear_dde, MG_dt0.05}), actor_critic+signature has ~4/10 NaN-abort seeds - report eta with the abort rate
