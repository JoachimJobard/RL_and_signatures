"""Tests for the representation-study aggregation logic (pure functions)."""

import importlib.util
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

from src.utils.run_context import find_repo_root

# run/ is not an installed package; load the aggregation module from its path.
_spec = importlib.util.spec_from_file_location(
    "agg_study", find_repo_root(__file__) / "run" / "study" / "aggregate_representation_study.py"
)
agg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = agg  # register so dataclasses can resolve __module__
_spec.loader.exec_module(agg)  # type: ignore
RunRecord, aggregate, build_comparison_figure = (
    agg.RunRecord, agg.aggregate, agg.build_comparison_figure
)
CellAggregate, load_cells, save_comparison_figure = (
    agg.CellAggregate, agg.load_cells, agg.save_comparison_figure
)


def _rec(kind, capacity, env, seed, j, fdim, linear):
    return RunRecord(run_dir="d", kind=kind, capacity=capacity, env_name=env,
                     seed=seed, j_agent=j, feature_dim=fdim, is_linear=linear)


def test_aggregate_normalised_suboptimality_for_linear_env():
    recs = [_rec("signature", 2, "JAXDDEEnv", s, 2.0 + 0.1 * s, 30, True) for s in range(3)]
    cells = aggregate(recs, oracle_by_env={"JAXDDEEnv": 1.0})
    assert len(cells) == 1
    c = cells[0]
    assert c.metric_is_suboptimality is True and c.n_seeds == 3
    # mean of (2.0-1)/1, (2.1-1)/1, (2.2-1)/1 = 1.1
    assert abs(c.mean - 1.1) < 1e-9
    assert c.ci95 >= 0.0


def test_aggregate_raw_cost_for_nonlinear_env():
    recs = [_rec("signature", 2, "MackeyGlass1DEnv", s, 5.0, 30, False) for s in range(2)]
    cells = aggregate(recs, oracle_by_env={})
    assert cells[0].metric_is_suboptimality is False
    assert cells[0].mean == 5.0


def test_aggregate_groups_by_kind_and_capacity():
    recs = [
        _rec("signature", 2, "E", 0, 1.0, 10, False),
        _rec("signature", 3, "E", 0, 1.0, 20, False),
        _rec("raw_history", 2, "E", 0, 1.0, 14, False),
    ]
    assert len(aggregate(recs, {})) == 3


def test_build_comparison_figure_from_cells():
    cells = aggregate(
        [_rec("signature", 2, "JAXDDEEnv", 0, 2.0, 30, True),
         _rec("raw_history", 2, "JAXDDEEnv", 0, 2.5, 14, True)],
        {"JAXDDEEnv": 1.0},
    )
    fig = build_comparison_figure(cells)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) >= 1
    assert any(ax.lines or ax.containers for ax in fig.axes)
    plt.close(fig)


def test_replot_from_aggregation_json_roundtrip(tmp_path):
    import json
    from dataclasses import asdict
    cells = aggregate(
        [_rec("signature", 2, "JAXDDEEnv", 0, 2.0, 30, True),
         _rec("raw_history", 2, "JAXDDEEnv", 0, 2.5, 14, True)],
        {"JAXDDEEnv": 1.0},
    )
    # Persist exactly as the full pipeline does, then rebuild from JSON alone.
    (tmp_path / "aggregation_data.json").write_text(
        json.dumps([asdict(c) for c in cells]))
    reloaded = load_cells(tmp_path)
    assert [asdict(c) for c in reloaded] == [asdict(c) for c in cells]
    save_comparison_figure(reloaded, tmp_path)
    assert (tmp_path / "comparison.png").exists()
