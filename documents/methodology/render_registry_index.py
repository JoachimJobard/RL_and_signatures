#!/usr/bin/env python3
"""Render the append-only experiment ledger into a human-readable Markdown index.

The authoritative record is ``experiment_registry.jsonl`` (one JSON object per line). This script
renders it to ``experiment_registry.md`` — a chronological index table plus a per-record detail
section — so the ledger is browsable without opening JSON. The Markdown is a DERIVED artefact: never
edit it by hand; append to the JSONL and re-run this generator.

Usage (from anywhere):
    python3 documents/methodology/render_registry_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "experiment_registry.jsonl"
INDEX = HERE / "experiment_registry.md"

# Fields rendered first, in this order, when present; any remaining keys follow in file order.
FIELD_ORDER = [
    "kind", "date", "status", "role", "hypothesis", "verdict",
    "pipeline", "cluster", "commit", "branch",
    "launcher", "manifest", "manifests", "tasks_file", "tasks",
    "cells", "cell", "cell_params", "cells_primary", "cells_null_check", "cells_used",
    "learners", "representations",
    "n_seeds", "seeds", "seed_groups", "held_out_seeds", "sigma", "sigma_variants",
    "n_episodes", "eval_x0", "ratios",
    "frozen_lr", "group", "groups",
    "path", "output_snapshot", "output_original", "output_root", "contents",
    "metric", "oracle_denominators", "evaluation", "figures_infra",
    "note", "caveat", "caveats", "submitted_by", "job_ids",
]

# First non-empty of these becomes the one-line summary in the index table.
SUMMARY_FIELDS = ["role", "verdict", "hypothesis", "status", "note", "caveat"]


def _cell(text: str) -> str:
    """Make a value safe for a Markdown table cell."""
    return str(text).replace("\n", " ").replace("|", r"\|").strip()


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fmt_value(v, indent: int = 0) -> list[str]:
    """Render a JSON value as one or more Markdown lines (nested dicts/lists become sub-bullets)."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(v, dict):
        for k, sub in v.items():
            if isinstance(sub, (dict, list)):
                lines.append(f"{pad}- `{k}`:")
                lines.extend(_fmt_value(sub, indent + 1))
            else:
                lines.append(f"{pad}- `{k}`: {_fmt_scalar(sub)}")
    elif isinstance(v, list):
        if all(not isinstance(x, (dict, list)) for x in v):
            lines.append(f"{pad}- {', '.join(_fmt_scalar(x) for x in v)}")
        else:
            for x in v:
                if isinstance(x, (dict, list)):
                    lines.extend(_fmt_value(x, indent))
                else:
                    lines.append(f"{pad}- {_fmt_scalar(x)}")
    else:
        lines.append(f"{pad}- {_fmt_scalar(v)}")
    return lines


def _summary(rec: dict) -> str:
    for f in SUMMARY_FIELDS:
        if rec.get(f):
            val = rec[f]
            if isinstance(val, list):
                val = val[0] if val else ""
            return _cell(val)
    return ""


def render(records: list[dict]) -> str:
    out: list[str] = []
    out.append("# Experiment registry — index\n")
    out.append(
        "_Derived artefact: auto-generated from `experiment_registry.jsonl` by "
        "`render_registry_index.py`. Do not edit by hand — append to the JSONL ledger and re-run "
        "the generator._\n"
    )
    dated = [r for r in records if r.get("date")]
    out.append(f"Records: {len(records)} ({len(dated)} dated).\n")

    # Chronological index (records without a date sort first, preserving file order among them).
    order = sorted(range(len(records)), key=lambda i: (records[i].get("date") or "", i))

    out.append("## Index\n")
    out.append("| Date | Kind | ID | Summary |")
    out.append("|---|---|---|---|")
    for i in order:
        r = records[i]
        out.append(
            f"| {_cell(r.get('date', '—'))} | {_cell(r.get('kind', '—'))} "
            f"| `{_cell(r.get('id', '?'))}` | {_summary(r)} |"
        )
    out.append("")

    out.append("## Records\n")
    for i in order:
        r = records[i]
        out.append(f"### `{r.get('id', '?')}`\n")
        keys = [k for k in FIELD_ORDER if k in r] + [
            k for k in r if k not in FIELD_ORDER and k != "id"
        ]
        for k in keys:
            v = r[k]
            if isinstance(v, (dict, list)) and not (
                isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v)
            ):
                out.append(f"- **{k}**:")
                out.extend("  " + line for line in _fmt_value(v))
            elif isinstance(v, list):
                out.append(f"- **{k}**: {', '.join(_fmt_scalar(x) for x in v)}")
            else:
                out.append(f"- **{k}**: {_fmt_scalar(v)}")
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    records = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
    INDEX.write_text(render(records))
    print(f"rendered {len(records)} records -> {INDEX.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
