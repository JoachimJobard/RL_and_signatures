# RL and signatures — continuous-time reinforcement learning for delayed systems

Signature-based continuous-time reinforcement learning for non-Markovian
(delayed) dynamical systems. The project extends K. Doya, *Reinforcement
Learning in Continuous Time and Space* (2000) to a path-signature representation
of the delayed state history, following the signature construction of
I. Perez Arribas, *Derivatives pricing using signature payoffs* (2018).

The reference documents (Doya 2000, Perez Arribas 2018, and the master thesis
that this code accompanies) are in [`documents/references/`](documents/references/).

## What is here

- **`src/`** — the importable library: agents (continuous-time actor-critic with
  and without signatures, and a value-gradient agent), environments (delay
  differential equations: Mackey–Glass, a chemical CSTR, delayed oscillators),
  the path-signature representation, networks, and shared utilities.
- **`run/`** — experiment / comparison drivers.
- **`conf/`** — Hydra configuration (agents, environments, launchers).
- **`main_unified.py`** — the single training + evaluation entry point.
- **`documents/`** — references and methodology notes (including the code review).

## Quick start

Install with [uv](https://docs.astral.sh/uv/) and run a local experiment:

```bash
uv sync
uv run python main_unified.py wandb.mode=disabled            # default signature agent
uv run python main_unified.py agent=value_gradient env=mackey_glass
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the project layout, the experiment
conventions (output directories, the `--debug` flag, seeding, logging), and the
cluster workflow, and
[`documents/methodology/scientific_workflow.md`](documents/methodology/scientific_workflow.md)
for the rationale behind those conventions.

## Status

A correctness review and a scientific-workflow reorganisation are in progress;
see [`documents/methodology/code_review.md`](documents/methodology/code_review.md)
for the audit and the applied fixes. The soft actor-critic agent (`CSAC_jax.py`)
is a **quarantined, unsupported** extension outside the thesis scope.
