# Contributing

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) (see `pyproject.toml`,
`.python-version` pins Python 3.12):

```bash
uv sync                 # create .venv and install the project + dependencies
uv run python -c "import jax; print(jax.__version__, jax.devices())"
```

JAX float64 is enabled at the entry point (`main_unified.py`), so the whole
pipeline runs in double precision.

## Project layout

```
src/                         # importable library (unit-tested, depended upon)
    agents/
        signatures_jax.py    # CTACSignatureJAX: continuous-time actor-critic over signatures
        base_jax.py          # CTACJAX: vanilla (state-based) continuous-time actor-critic
        value_gradient_jax.py# value-gradient / HJB agent (target network)
        CSAC_jax.py          # QUARANTINED, unsupported (outside thesis scope)
    envs/                    # delay differential equation environments (RK4 method-of-steps)
        env_rk_jax.py        # base linear DDE + RK4 + ring-buffer history
        mackey_glass_1D.py   # Mackey–Glass
        chemical_process.py  # chemical CSTR (Dadebo)
    networks/                # Flax actor/critic modules
    utils/
        dynamic_signature.py # path-signature feature map (Perez Arribas 2018 augmentation)
        run_context.py       # output-dir convention, seeding, run-context logging
        optim.py             # build_adam (optional global-norm gradient clipping)
        solver_buffer_jax.py # delay history ring buffer
        experience_replay_buffer.py
conf/                        # Hydra configs (agent/, env/, launcher/)
run/                         # experiment / comparison drivers
    diagnostics/             # standalone diagnostics (e.g. float32-vs-float64 signature)
main_unified.py              # training + evaluation entry point
data/                        # generated experiment output (git-ignored)
    <script_stem>/<timestamp>_<config_tag>[_seed<S>]/
documents/
    references/              # Doya 2000, Perez Arribas 2018, the master thesis
    methodology/             # design notes, code review, scientific-workflow conventions
test/                        # unit tests mirroring src/ (forthcoming, Phase 3)
```

**Library vs experiment.** `src/` holds the reusable, tested library code; `run/`
and `main_unified.py` are study-specific drivers. Logic worth reusing or testing
should be promoted into `src/`.

## Running an experiment

The entry point is Hydra-configured:

```bash
uv run python main_unified.py                       # default config
uv run python main_unified.py agent=value_gradient env=mackey_glass
uv run python main_unified.py agent.signature.depth=4 wandb.mode=disabled
```

Standalone diagnostics follow the output-directory convention directly:

```bash
uv run python run/diagnostics/compare_signature_float32_vs_float64.py --debug
```

## Experiment conventions

These are enforced by `src/utils/run_context.py` and described in detail in
[`documents/methodology/scientific_workflow.md`](documents/methodology/scientific_workflow.md):

- **Output folders derive from the script filename** via `script_data_dir(__file__)`
  / `resolve_run_dir(__file__, config_tag, seed=..., debug=...)`; never hardcode a
  data path. Runs land in `data/<script_stem>/<timestamp>_<config_tag>[_seed<S>]/`.
- **`--debug`** prepends `_debug_` to the run folder; exploratory/smoke runs MUST
  use it. Wipe them with `find data -type d -name '_debug_*' -prune -exec rm -rf {} +`.
- **Single master seed** (`seed`), with deterministic per-role seeds via
  `derive_seed(master_seed, role)`. Shared seeding across a comparison is the
  default; decorrelation is an explicit opt-in.
- **Self-contained logging**: log the captured run context (`capture_run_context`)
  — command line, library/runtime versions, accelerator, hyperparameters, seeds —
  at startup, and persist it next to the artefacts so a run reproduces from its
  folder alone.
- **Replot without recomputation**: persist the metrics/history needed to rebuild
  every figure, and rebuild plots from those saved artefacts.
- **Gradient clipping** is opt-in (`training.clip_gradient`, global-norm) and off
  by default.

## Tests

Unit and integration tests live under `test/` mirroring `src/`:

```bash
uv run pytest                       # full suite (~8 s)
uv run pytest test/utils            # a subset
```

Coverage includes the run-context helper, `build_adam` clipping, the delay-history
interpolation, signature properties (size, depth-1 identity, translation
invariance, float32-vs-float64), instantiation of every `conf/env/*.yaml`, the
Mackey–Glass cost, the RK4 integrator, and a one-episode training smoke test for
each supported agent. New library code should come with a test mirroring its
module path.

## Cluster (Jean Zay)

Standalone `sbatch` launchers live under `bash_scripts/cluster/jeanzay/` (see its
`README.md` for one-time setup and the partition/QoS routing). Highlights:

- `python/python_script_launcher.sh` (CPU) / `python_script_launcher_gpu.sh` (GPU)
  submit a single `main_unified.py` run.
- `python/experiment_array_launcher.sh` runs an ablation/sweep as a SLURM **job
  array**, one task per line of a variants file (each line = Hydra overrides);
  tasks group under `data/main_unified/<experiment-group>/` and SLURM logs land
  in that group's `slurm/` sub-folder.
- `python/run_interactive_job.sh` opens an interactive compute shell.

These replace the Inria-`tau` Hydra submitit launcher (`conf/launcher/slurm.yaml`).

## Code style

Documentation and comments in English; mathematics in LaTeX in Markdown.
Favour many small, descriptively-titled commits. Before handing work to a remote
runner (a cluster job, CI), commit and push first — the remote only sees what is
on the upstream branch.
