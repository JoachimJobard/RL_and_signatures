"""Reproducible run context: output-directory convention, seeding, and logging.

This module centralises the scientific-workflow conventions for the project so
that every experiment is reproducible and auditable from its log alone:

- ``script_data_dir`` / ``resolve_run_dir`` derive the output folder from the
  *script filename* (never a hardcoded string), append a timestamped, config- and
  seed-tagged sub-folder, and prepend ``_debug_`` for exploratory runs.
- ``derive_seed`` / ``derive_seeds`` turn a single user-controlled master seed
  into deterministic per-role seeds, so two variants with identical
  hyperparameters share initial conditions and sampler trajectories (the
  shared-seed policy).
- ``capture_run_context`` / ``format_run_context`` record the full command line,
  library/runtime versions, accelerator, resolved hyperparameters and seeds.

The module is deliberately cheap to import: heavy libraries (JAX, Flax, Optax)
are imported lazily inside ``capture_run_context`` only, so it is safe to import
from login-node / init-only code paths.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


# =============================================================================
# Output-directory convention (folder derived from the script filename)
# =============================================================================

def find_repo_root(start: str | Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing ``pyproject.toml``
    is found and return it.

    Raises:
        FileNotFoundError: if no ancestor contains ``pyproject.toml`` (the script
            is being run from outside the repository checkout).
    """
    if start is None:
        start = Path(__file__)
    start = Path(start).resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError(
        f"Could not locate repository root (no pyproject.toml found above {start})."
    )


def script_data_dir(script_file: str | Path) -> Path:
    """Return ``<data_root>/data/<script_stem>/`` for the calling script.

    Every script that writes experiment data MUST derive its top-level output
    directory from this helper by passing ``__file__``, so the folder name on
    disk is mechanically tied to the running script and cannot drift if the
    script is renamed. The directory is *not* created (side-effect-free).

    ``<data_root>`` is the repository root by default. It may be redirected by
    setting the environment variable ``RL_SIGNATURES_DATA_ROOT`` to an absolute
    path; the ``data/<script_stem>/`` suffix is appended unchanged, so the
    filename-derived folder name is preserved. This exists for clusters whose
    project filesystem has an inode quota too small for experiment output: on
    Jean Zay the repository lives on ``$WORK`` but runs must be written to
    ``$SCRATCH``. Writing a job array to ``$WORK`` is what aborted job 544311
    mid-array (``OSError: [Errno 122] Disk quota exceeded``), losing every task.

    A leading ``~`` is expanded. An override that is set but *not* absolute is
    rejected loudly rather than resolved against the process working directory:
    the cluster workers ``cd`` into the repository checkout before invoking
    python (``bash_scripts/cluster/jeanzay/_environment.sh``), so a relative
    value would silently resolve back inside the repository — precisely the
    write the override exists to prevent. An empty (exported-but-unset) value is
    treated as absent and falls back to the repository root.

    Raises:
        ValueError: if ``RL_SIGNATURES_DATA_ROOT`` is set to a non-absolute path.
    """
    script_path = Path(script_file).resolve()
    data_root_override = os.environ.get("RL_SIGNATURES_DATA_ROOT", "").strip()
    if data_root_override:
        root = Path(data_root_override).expanduser()
        if not root.is_absolute():
            raise ValueError(
                "RL_SIGNATURES_DATA_ROOT must be an absolute path, got "
                f"{data_root_override!r}. A relative value would resolve against the "
                "process working directory (the repository checkout on the cluster "
                "workers), which is the $WORK inode-quota write this override exists "
                "to prevent. Set it to an absolute path such as $SCRATCH/rl_campaigns, "
                "or unset it to write under the repository root."
            )
        root = root.resolve()
    else:
        root = find_repo_root(script_path)
    return root / "data" / script_path.stem


def run_timestamp() -> str:
    """A filesystem-friendly local timestamp ``YYYYMMDD_HHMMSS`` for run folders."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(
    script_file: str | Path,
    config_tag: str,
    *,
    seed: int | None = None,
    debug: bool = False,
    timestamp: str | None = None,
    subdir: str | None = None,
    create: bool = True,
) -> Path:
    """Build (and optionally create) the canonical run directory.

    Layout: ``data/<script_stem>/[<subdir>/]<debug_prefix><timestamp>_<config_tag>[_seed<seed>]/``.
    The ``_debug_`` prefix (for ``debug=True``) sorts exploratory runs to the
    bottom of ``ls`` and lets them be wiped en masse; the seed suffix keeps
    multi-seed runs of one variant distinguishable at a glance; ``subdir`` groups
    all tasks of one ablation/sweep under a shared parent directory.

    Args:
        script_file: pass ``__file__`` from the running script.
        config_tag: short, descriptive tag identifying the variant/config.
        seed: master seed; appended as ``_seed<seed>`` when not ``None``.
        debug: prepend ``_debug_`` to flag an exploratory run.
        timestamp: override the timestamp (default: :func:`run_timestamp`).
        subdir: optional grouping sub-folder (e.g. an ablation/experiment-group name).
        create: ``mkdir(parents=True)`` the directory before returning it.
    """
    ts = timestamp if timestamp is not None else run_timestamp()
    debug_prefix = "_debug_" if debug else ""
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    parent = script_data_dir(script_file)
    if subdir:
        parent = parent / subdir
    run_dir = parent / f"{debug_prefix}{ts}_{config_tag}{seed_suffix}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# =============================================================================
# Deterministic seeding (single master seed -> per-role seeds)
# =============================================================================

def derive_seed(master_seed: int, role: str) -> int:
    """Deterministically derive a 32-bit per-role seed from the master seed.

    The derivation is a stable hash of ``f"{master_seed}:{role}"`` (independent
    of process hash randomization and of Python version), so a given
    ``(master_seed, role)`` always yields the same seed. ``role`` must be an
    explicit, semantic tag (e.g. ``"model_init"``, ``"sampler"``, ``"eval"``) —
    never cosmetic metadata such as a variant's display name or colour.
    """
    digest = hashlib.sha256(f"{master_seed}:{role}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def derive_seeds(master_seed: int, roles: Iterable[str]) -> dict[str, int]:
    """Map each role to its derived seed (see :func:`derive_seed`)."""
    return {role: derive_seed(master_seed, role) for role in roles}


# =============================================================================
# Self-contained run context (command line, versions, device, seeds)
# =============================================================================

def _package_versions() -> dict[str, str]:
    """Best-effort version strings for the key numerical/ML libraries."""
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "jax", "jaxlib", "flax", "optax", "signax",
                 "hydra", "omegaconf", "wandb"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 — a missing optional dep is not fatal here
            versions[name] = "not installed"
    return versions


def _jax_devices() -> dict[str, Any]:
    """Best-effort accelerator description (device kind, count, x64 flag)."""
    info: dict[str, Any] = {}
    try:
        import jax

        devices = jax.devices()
        info["device_count"] = len(devices)
        info["devices"] = [str(d) for d in devices]
        info["default_backend"] = jax.default_backend()
        info["x64_enabled"] = bool(getattr(jax.config, "jax_enable_x64", False))
    except Exception:  # noqa: BLE001
        info["devices"] = "jax not available"
    return info


def capture_run_context(
    *,
    master_seed: int | None = None,
    derived_seeds: Mapping[str, int] | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect everything needed to reproduce and audit a run.

    Returns a JSON/YAML-serialisable dict; callers should both log it (see
    :func:`format_run_context`) and persist it next to the run's artefacts.
    """
    context: dict[str, Any] = {
        "command_line": " ".join(sys.argv),
        "timestamp": run_timestamp(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "accelerator": _jax_devices(),
    }
    if master_seed is not None:
        context["master_seed"] = master_seed
    if derived_seeds is not None:
        context["derived_seeds"] = dict(derived_seeds)
    if hyperparameters is not None:
        context["hyperparameters"] = dict(hyperparameters)
    if extra is not None:
        context["extra"] = dict(extra)
    return context


def format_run_context(context: Mapping[str, Any]) -> str:
    """Render a captured run context as a flat, human-readable block."""
    lines = ["=== run context ==="]

    def _emit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, sub in value.items():
                _emit(f"{prefix}{key}.", sub)
        else:
            lines.append(f"{prefix[:-1]:<28}: {value}")

    for key, value in context.items():
        _emit(f"{key}.", value)
    return "\n".join(lines)
