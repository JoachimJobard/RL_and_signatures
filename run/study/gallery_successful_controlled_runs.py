"""Build a browsable gallery of controlled-run figures where the control task is achieved.

For every experiment group under ``data/main_unified/`` with an ``aggregation_data.json``,
this reads the per-seed control metric of each (environment, representation, capacity)
cell, classifies each seed as "task achieved" against a threshold, and emits a single
static HTML gallery organised system -> representation -> seed. Each gallery entry embeds
that run's ``figure_agent_vs_no_control.png`` (state dynamics, control signal u(t),
cumulative and instantaneous cost, error from target) via a plain ``<img>`` tag -- no
Plotly, no JavaScript charting.

Success criterion (per seed):
  - suboptimality cells (linear plants, metric rho = (J - J_oracle)/|J_oracle|):
      achieved  <=>  rho < ``--rho-max`` (default 0.5, i.e. within 50% of the oracle)
  - raw-cost cells (nonlinear plants, no closed-form oracle, metric = raw cost J):
      achieved  <=>  J finite and J < ``--j-max`` (default 1.0; separates set-point
      tracking, J ~ 0.05-0.5, from collapse to a spurious fixed point, J ~ 30+)
These are heuristics; tune them on the command line. Per-seed values come from each
group's ``aggregation_data.json`` (``values`` list, index = seed), which is authoritative
even when the per-run ``eval.pkl`` was not rapatriated from the cluster.

Usage:
    uv run python run/study/gallery_successful_controlled_runs.py \\
        [--data-root data/main_unified] [--out <file.html>] \\
        [--rho-max 0.5] [--j-max 1.0] [--include-failures]
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


# Map the aggregation "kind" + capacity to the run-directory variant token.
def _variant_token(kind: str, capacity: int) -> str:
    return {
        "markovian": f"markovian_deg{capacity}",
        "raw_history": f"raw_deg{capacity}",
        "signature": f"sig_depth{capacity}",
    }.get(kind, kind)  # oracle-ladder kinds (full_learned, ...) pass through unchanged


def _pretty_kind(kind: str) -> str:
    return {"markovian": "markovian", "raw_history": "raw-history",
            "signature": "signature"}.get(kind, kind)


def _find_figure(group_dir: Path, kind: str, capacity: int, seed: int) -> Path | None:
    token = _variant_token(kind, capacity)
    hits = sorted(group_dir.glob(f"*_{token}_seed{seed}/figure_agent_vs_no_control.png"))
    return hits[0] if hits else None


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def classify_group(group_dir: Path, rho_max: float, j_max: float) -> list[dict]:
    """Return one record per (kind, capacity, seed) with its metric, achieved flag,
    and figure path (if present)."""
    agg = group_dir / "aggregation_data.json"
    if not agg.exists():
        return []
    records = []
    for cell in json.loads(agg.read_text()):
        kind, cap = cell["kind"], cell["capacity"]
        subopt = bool(cell["metric_is_suboptimality"])
        for seed, val in enumerate(cell["values"]):
            nan = _is_nan(val)
            if nan:
                achieved = False
            elif subopt:
                achieved = val < rho_max
            else:
                achieved = val < j_max
            records.append({
                "kind": kind, "capacity": cap, "seed": seed,
                "metric": "rho" if subopt else "J",
                "value": None if nan else float(val),
                "achieved": achieved,
                "figure": _find_figure(group_dir, kind, cap, seed),
            })
    return records


_CSS = """
body{font-family:system-ui,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
h1{margin:0 0 4px} .crit{color:#555;font-size:14px;margin:0 0 20px}
h2{margin:28px 0 6px;border-bottom:2px solid #ccc;padding-bottom:4px}
h3{margin:18px 0 8px;color:#333}
.grid{display:flex;flex-wrap:wrap;gap:14px}
figure{margin:0;border:1px solid #ddd;border-radius:6px;background:#fff;padding:8px;
       box-shadow:0 1px 3px rgba(0,0,0,.06)}
figure img{width:520px;max-width:44vw;height:auto;display:block}
figcaption{font-size:13px;margin-top:6px;color:#222}
.ok{color:#137333;font-weight:600} .bad{color:#a50e0e}
.miss{color:#999;font-style:italic;width:520px;max-width:44vw}
.count{color:#555;font-weight:400;font-size:14px}
"""


def build_html(groups: dict[str, list[dict]], *, rho_max: float, j_max: float,
               include_failures: bool) -> str:
    parts = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
             "<title>Controlled runs where the task is achieved</title>",
             f"<style>{_CSS}</style></head><body>",
             "<h1>Controlled trajectories &mdash; task-achieved runs</h1>",
             (f"<p class='crit'>Per-seed success criterion: linear cells "
              f"&rho;&lt;{rho_max} (within {rho_max*100:.0f}% of the delayed-LQR oracle); "
              f"nonlinear cells raw cost J&lt;{j_max}. Each panel is that run's "
              f"agent-vs-no-control figure (state, control u(t), cumulative &amp; "
              f"instantaneous cost, error). "
              f"{'Failures shown greyed.' if include_failures else 'Only achieved runs shown.'}"
              "</p>")]

    for group in sorted(groups):
        recs = groups[group]
        shown = [r for r in recs if r["achieved"] or include_failures]
        if not shown:
            continue
        n_ok = sum(r["achieved"] for r in recs)
        parts.append(f"<h2>{html.escape(group)} "
                     f"<span class='count'>&mdash; {n_ok}/{len(recs)} seeds achieved</span></h2>")
        # group by representation (kind, capacity), ordered markovian->raw->signature
        order = {"markovian": 0, "raw_history": 1, "signature": 2}
        keys = sorted({(r["kind"], r["capacity"]) for r in shown},
                      key=lambda kc: (order.get(kc[0], 9), kc[1]))
        for kind, cap in keys:
            cap_label = (f"depth {cap}" if kind == "signature" else f"deg {cap}")
            row = sorted((r for r in shown if r["kind"] == kind and r["capacity"] == cap),
                         key=lambda r: r["seed"])
            parts.append(f"<h3>{_pretty_kind(kind)} &mdash; {cap_label}</h3>")
            parts.append("<div class='grid'>")
            for r in row:
                mv = "NaN" if r["value"] is None else f"{r['value']:.4f}"
                cls = "ok" if r["achieved"] else "bad"
                cap_txt = (f"seed {r['seed']} &mdash; {r['metric']}={mv} "
                           f"<span class='{cls}'>[{'achieved' if r['achieved'] else 'failed'}]</span>")
                if r["figure"] is None:
                    parts.append(f"<div class='miss'>{cap_txt}<br>(figure missing)</div>")
                else:
                    src = html.escape(r["figure"].resolve().as_uri())
                    parts.append(f"<figure><img src='{src}' loading='lazy'>"
                                 f"<figcaption>{cap_txt}</figcaption></figure>")
            parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=Path("data/main_unified"))
    p.add_argument("--out", type=Path, default=None,
                   help="output HTML (default: <data-root>/gallery_successful_controlled_runs.html)")
    p.add_argument("--rho-max", type=float, default=0.5)
    p.add_argument("--j-max", type=float, default=1.0)
    p.add_argument("--include-failures", action="store_true",
                   help="also show non-achieved seeds (greyed), for contrast")
    args = p.parse_args()

    if not args.data_root.is_dir():
        p.error(f"not a directory: {args.data_root}")

    # Validation / smoke scaffolds carry an aggregation_data.json but no run figures
    # locally; skip them so the gallery shows only real study cells.
    SKIP = {"smoke_test", "study_pipeline_validation"}
    groups: dict[str, list[dict]] = {}
    for group_dir in sorted(d for d in args.data_root.iterdir() if d.is_dir()):
        if group_dir.name in SKIP:
            continue
        recs = classify_group(group_dir, args.rho_max, args.j_max)
        if recs:
            groups[group_dir.name] = recs

    if not groups:
        p.error(f"no groups with aggregation_data.json under {args.data_root}")

    out = args.out or (args.data_root / "gallery_successful_controlled_runs.html")
    out.write_text(build_html(groups, rho_max=args.rho_max, j_max=args.j_max,
                              include_failures=args.include_failures))

    total = sum(len(v) for v in groups.values())
    ok = sum(r["achieved"] for v in groups.values() for r in v)
    missing = sum(1 for v in groups.values() for r in v if r["achieved"] and r["figure"] is None)
    print(f"[gallery] {len(groups)} groups, {ok}/{total} seeds achieved "
          f"({missing} achieved runs missing a figure)")
    print(f"[gallery] wrote {out}")


if __name__ == "__main__":
    main()
