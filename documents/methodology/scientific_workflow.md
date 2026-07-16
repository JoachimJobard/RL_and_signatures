# Scientific workflow conventions

This note records the reproducibility and organisation conventions adopted for
the repository. They are adapted to this project's JAX + Hydra + Weights & Biases
stack from a shared set of practices used across the group's research codebases.
The helpers that implement them live in `src/utils/run_context.py`.

## 1. Output folders derive from the script filename

Every script that produces a top-level output folder derives it from
`Path(__file__).stem`, via `run_context.script_data_dir(__file__)` →
`<data_root>/data/<script_stem>/`, where `<data_root>` is the repository root by
default. Renaming the script therefore moves its data folder automatically; a
hardcoded literal such as `Path("data/some_name")` is forbidden because it
silently drifts from the filename.

### 1.1 Redirecting the data root (`RL_SIGNATURES_DATA_ROOT`)

`<data_root>` is not unconditionally the repository root: setting the environment
variable `RL_SIGNATURES_DATA_ROOT` to an **absolute** path redirects it. Only the
root moves — the `data/<script_stem>/` suffix is appended unchanged, so the
filename-derived folder name, and every convention built on it (the timestamped
sub-folder, the `_debug_` prefix, the `_seed<seed>` suffix), is preserved. The
override is therefore invisible to the conventions of this section; it changes
only where the tree is rooted.

The override exists because a project filesystem may carry an inode quota too
small for a job array's output. On Jean Zay the repository lives on `$WORK`,
whose inode quota is far below what an array produces, while `$SCRATCH` is sized
for run output. Writing an array to `$WORK` is what aborted job 544311 mid-array
with `OSError: [Errno 122] Disk quota exceeded`: it left 80 SLURM logs and **zero**
run directories. `$SCRATCH` is purged periodically, so results worth keeping are
to be rapatriated.

Two constraints are enforced in `script_data_dir`:

- A leading `~` is expanded, after which a **non-absolute** value is rejected with
  a `ValueError` rather than resolved against the process working directory. The
  cluster workers `cd` into the repository checkout before invoking python
  (`bash_scripts/cluster/jeanzay/_environment.sh`), so a relative value would
  resolve back inside the repository — exactly the write the override exists to
  prevent, and silently so.
- An empty (exported-but-unset) value is treated as absent and falls back to the
  repository root, since `Path("").resolve()` yields the working directory.

The override is one knob, but its propagation to a worker depends on how that
worker's entry point resolves its output, and the two mechanisms in use must not
be confused:

| Launcher | Entry point | Mechanism |
| --- | --- | --- |
| `rl_launch.sh` | `main_unified.py` (no `--out-dir`) | `RL_SIGNATURES_DATA_ROOT` exported to the worker; `script_data_dir` reads it |
| `h1h2_launch.sh` | `run/study/h1h2_evaluation.py` | explicit `--out-dir $EXPDIR`, with `EXPDIR` re-based onto the data root by the launcher |
| `rl_lspi_launch.sh` | `run/study/h1h2_rl_lspi.py` | explicit `--out-dir $EXPDIR`, with `EXPDIR` re-based onto the data root by the launcher |

Where an entry point receives `--out-dir`, it takes `Path(args.out_dir) / tag` and
never consults `script_data_dir`; exporting the environment variable to such a
worker would be a dead flag. Those launchers instead re-base `EXPDIR` itself,
keeping the `<data_root>/data/<script_stem>/` shape the override would have
produced, so that the campaign shares a single on-disk layout regardless of which
mechanism carried the redirect.

Individual runs use `run_context.resolve_run_dir(__file__, config_tag, seed=...,
debug=...)`, which appends a timestamped, config- and seed-tagged sub-folder:

```
data/<script_stem>/<debug_prefix><timestamp>_<config_tag>[_seed<seed>]/
```

The seed appears **in the folder name** (not only inside the metadata) so that
several seeds of one variant are distinguishable at a glance and so an aggregator
can discover them by name.

## 2. The `--debug` flag and smoke-test guard

Every experiment script exposes a `--debug` flag that prepends `_debug_` to the
run folder. Because `_` sorts after digits, exploratory runs cluster at the
bottom of `ls` and can be wiped en masse:

```bash
find data -type d -name '_debug_*' -prune -exec rm -rf {} +
```

Test / smoke runs MUST pass `--debug`; real runs (the ones whose results are kept,
plotted, or cited) MUST NOT. Scripts additionally guard against an unflagged
under-sized run (a `SMOKE_TEST_*` threshold that aborts via `parser.error` unless
`--debug` is set), so a smoke run cannot silently land in the real-run namespace.

## 3. Deterministic seeding (single master seed)

Randomness is seeded from one user-controlled master `seed`. Per-role seeds are
derived deterministically with `run_context.derive_seed(master_seed, role)` for
explicit, semantic roles (`"model_init"`, `"sampler"`, `"eval"`, …) — never from
cosmetic metadata (a variant's display name, colour, or iteration budget). The
default across a comparison is **shared seeding**, so observed differences between
variants reflect the intervention rather than RNG noise; decorrelated seeds are a
deliberate, explicitly-keyed opt-in. Variance estimates use an explicit seed axis
layered on top of the comparison.

## 4. Self-contained, reproducible run logs

At startup each run logs, and persists next to its artefacts, a captured run
context (`run_context.capture_run_context` → `format_run_context`): the full
command line, the Python / JAX / Flax / Optax / signax / Hydra / wandb versions,
the accelerator (device kind, count, x64 flag), the resolved hyperparameters, the
master seed and every derived seed, and (where applicable) the model parameter
count. The log is therefore a self-contained reproduction recipe.

## 5. Save run statistics so figures regenerate without recomputation

Persist the artefacts needed to rebuild every figure without re-running the
expensive computation: the fitted parameters, a machine-readable run-metadata
file, and the per-iteration history. Provide a replot path that rebuilds all
plots from those saved artefacts; replotting reads saved tensors/metrics and
never recomputes them.

## 6. Library / experiment separation

`src/` is the importable library (unit-tested, depended upon); `run/` and
`main_unified.py` are study-specific drivers. Logic worth reusing or testing is
promoted into `src/`.

## 7. Sweeps and ablations

Default to independent parallel tasks (one per variant / seed / grid point) over
a single sequential job, aggregated afterwards. Each task accepts a variant/config
selector and an output-directory override so all tasks write under one shared
parent, and writes a per-task summary file rather than racing on a shared one. A
finalize step aggregates the per-task summaries into the canonical comparison
plots and a combined summary. On Jean Zay this is realised with `sbatch` job
arrays (Phase 5).

## 8. Plot conventions

Plots use LaTeX-rendered mathematics, an explanatory text box giving the formula
for the symbols used, and a legend placed **outside** the axes. Trained/predicted
curves are solid; analytical references are dashed; auxiliary annotation lines are
dotted. Hyperparameter sweeps are encoded in colour (sequential palette), not in
stroke. (Phase 4 brings the existing plot utilities up to this standard.)

## 9. Gradient clipping

Gradient clipping is opt-in and **off by default**. When `training.clip_gradient`
is a positive number, `run`/agent optimizers (built via `src/utils/optim.py::
build_adam`) apply direction-preserving global-norm clipping; otherwise none.
