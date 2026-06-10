"""Tests for the reproducible run-context helper."""

from pathlib import Path

from src.utils.run_context import (
    derive_seed,
    derive_seeds,
    resolve_run_dir,
    script_data_dir,
    capture_run_context,
    format_run_context,
)


def test_script_data_dir_derives_from_filename():
    d = script_data_dir("run/diagnostics/my_experiment.py")
    assert d.name == "my_experiment"
    assert d.parent.name == "data"


def test_resolve_run_dir_layout(tmp_path):
    # create=False so no directory is written under the repo's data/
    run_dir = resolve_run_dir(
        "run/foo.py", "depth3_mg", seed=7, debug=False,
        timestamp="20260101_000000", create=False,
    )
    assert run_dir.name == "20260101_000000_depth3_mg_seed7"
    assert run_dir.parent.name == "foo"


def test_resolve_run_dir_debug_prefix():
    run_dir = resolve_run_dir(
        "run/foo.py", "tag", seed=0, debug=True,
        timestamp="20260101_000000", create=False,
    )
    assert run_dir.name.startswith("_debug_")


def test_resolve_run_dir_no_seed_suffix():
    run_dir = resolve_run_dir(
        "run/foo.py", "tag", seed=None, debug=False,
        timestamp="20260101_000000", create=False,
    )
    assert run_dir.name == "20260101_000000_tag"


def test_derive_seed_is_deterministic():
    assert derive_seed(42, "model_init") == derive_seed(42, "model_init")


def test_derive_seed_distinct_roles_distinct_seeds():
    seeds = derive_seeds(42, ["model_init", "sampler", "eval"])
    assert len(set(seeds.values())) == 3


def test_derive_seed_distinct_masters_distinct_seeds():
    assert derive_seed(0, "sampler") != derive_seed(1, "sampler")


def test_derive_seed_in_uint32_range():
    s = derive_seed(123456, "role")
    assert 0 <= s < 2 ** 32


def test_capture_and_format_run_context():
    ctx = capture_run_context(
        master_seed=3,
        derived_seeds={"agent": 11},
        hyperparameters={"depth": 3},
    )
    assert ctx["master_seed"] == 3
    assert ctx["derived_seeds"] == {"agent": 11}
    text = format_run_context(ctx)
    assert "master_seed" in text and "command_line" in text
