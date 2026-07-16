"""Tests for the reproducible run-context helper."""

from pathlib import Path

import pytest

from src.utils.run_context import (
    derive_seed,
    derive_seeds,
    find_repo_root,
    resolve_run_dir,
    script_data_dir,
    capture_run_context,
    format_run_context,
)


def test_script_data_dir_derives_from_filename():
    d = script_data_dir("run/diagnostics/my_experiment.py")
    assert d.name == "my_experiment"
    assert d.parent.name == "data"


def test_script_data_dir_default_root_is_the_repository(monkeypatch):
    monkeypatch.delenv("RL_SIGNATURES_DATA_ROOT", raising=False)
    d = script_data_dir("run/diagnostics/my_experiment.py")
    assert d == find_repo_root(Path(__file__)) / "data" / "my_experiment"


def test_script_data_dir_env_override_redirects_root_keeping_stem(monkeypatch, tmp_path):
    # The Jean Zay case: the repository lives on $WORK but runs must be written to
    # $SCRATCH. Only the root moves; the filename-derived suffix is preserved.
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", str(tmp_path))
    d = script_data_dir("run/diagnostics/my_experiment.py")
    assert d == tmp_path / "data" / "my_experiment"


def test_script_data_dir_env_override_tolerates_trailing_slash(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", f"{tmp_path}/")
    assert script_data_dir("run/foo.py") == tmp_path / "data" / "foo"


def test_script_data_dir_empty_env_override_falls_back_to_repository(monkeypatch):
    # An unset-but-exported variable (RL_SIGNATURES_DATA_ROOT=) must not redirect the
    # root to the process working directory, which is what Path("").resolve() returns.
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", "")
    assert script_data_dir("run/foo.py") == find_repo_root(Path(__file__)) / "data" / "foo"


def test_script_data_dir_relative_env_override_is_rejected(monkeypatch):
    # _environment.sh does `cd "$PATH_CONTENT_ROOT"` before invoking python, so a
    # relative override would resolve INSIDE the repository -- the $WORK inode-quota
    # write the override exists to prevent (job 544311). It must raise, not resolve.
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", "scratch/rl_campaigns")
    with pytest.raises(ValueError, match="absolute"):
        script_data_dir("run/foo.py")


def test_script_data_dir_dot_env_override_is_rejected(monkeypatch):
    # The degenerate case: "." resolves to the working directory silently.
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", ".")
    with pytest.raises(ValueError, match="absolute"):
        script_data_dir("run/foo.py")


def test_script_data_dir_env_override_expands_user(monkeypatch):
    monkeypatch.setenv("HOME", "/home/somebody")
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", "~/rl_campaigns")
    assert script_data_dir("run/foo.py") == Path("/home/somebody/rl_campaigns/data/foo")


def test_script_data_dir_whitespace_env_override_falls_back_to_repository(monkeypatch):
    # A whitespace-only value is an unset variable in practice, not a relative path.
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", "   ")
    assert script_data_dir("run/foo.py") == find_repo_root(Path(__file__)) / "data" / "foo"


def test_resolve_run_dir_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("RL_SIGNATURES_DATA_ROOT", str(tmp_path))
    run_dir = resolve_run_dir(
        "run/foo.py", "tag", seed=3, debug=False,
        timestamp="20260101_000000", create=False,
    )
    assert run_dir == tmp_path / "data" / "foo" / "20260101_000000_tag_seed3"


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


def test_resolve_run_dir_subdir_groups_under_parent():
    # An ablation/sweep groups all tasks under a shared sub-folder.
    run_dir = resolve_run_dir(
        "run/foo.py", "depth2", seed=0, debug=False, subdir="my_ablation",
        timestamp="20260101_000000", create=False,
    )
    assert run_dir.name == "20260101_000000_depth2_seed0"
    assert run_dir.parent.name == "my_ablation"
    assert run_dir.parent.parent.name == "foo"


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
