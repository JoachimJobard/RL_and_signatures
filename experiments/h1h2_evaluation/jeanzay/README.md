# H1 campaign launcher artefacts — which files are authoritative

**Authoritative final learning rates for the AAAI-2027 paper:**

- `frozen_final.yaml` — harmonic_oscillator and linear_dde_scalar.
- `frozen_dt0p05_clean.yaml` — Mackey–Glass, the **dt = 0.05** cadence actually reported.

These match the paper's Appendix C, Table 2, and the reported per-run configs in the
authoritative snapshot on Jean Zay,
`$WORK/paper_data/aaai2027_h1_clean_20260724/` (registry id
`snapshot_h1_aaai2027_20260724`; see `documents/methodology/experiment_registry.md`).

## Superseded — do not read as final selections

- `frozen_alpha0.yaml` — **SUPERSEDED.** The 2026-07-22 selection on the single
  held-out seed 42 (%opt): the exact single-seed confound the 2026-07-23
  pre-registration corrects. Its rates differ from the reported runs — e.g. harmonic
  value_gradient / markovian is `0.3` here but `1` in `frozen_final.yaml`. Retained
  only as the historical launch record read by `final_runs_manifest.py` and
  `final_runs_launch.sh`; it is not the paper's selection.
- `final_tasks_*.tsv`, `selection_tasks_*.tsv`, `stability*_tasks_*.tsv` — generated
  launch manifests for specific (superseded or intermediate) sweeps. Launch records,
  not results.

## Caveat on `frozen_final.yaml`

Its `MG_1D_limit_cycle` block is the **dt = 0.25** cadence, which the paper does
**not** use. The reported Mackey–Glass cell is the dt = 0.05 re-run; its selection is
`frozen_dt0p05_clean.yaml`. Use that block for Mackey–Glass, `frozen_final.yaml` for
the other two environments.

## Where the reported data lives

Not in this repository. The reported artefacts (per-run configs, `reeval_clean`
summaries, the F2 CSV) are on Jean Zay under
`$WORK/paper_data/aaai2027_h1_clean_20260724/`. Everything under the repo's gitignored
`data/` is exploratory or pre-H1 scratch, not paper data.
