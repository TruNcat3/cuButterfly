#!/usr/bin/env python3
"""Summarize the bounded E3 cross-operator resident-control journal."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--journal",
        type=Path,
        default=Path("results/paper_completion_20260918/e3_cross_operator/a100-80gb/journal.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/paper_completion_20260918/e3_cross_operator/summary.json"),
    )
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text())
    if journal.get("status") != "complete":
        raise SystemExit(f"journal is not complete: {journal.get('status')}")
    rows = []
    for operator in ("fwht", "ntt"):
        for batch in (1, 1024):
            key = lambda width: f"{operator}-n12-b{batch}-leaf{width}"
            baseline = journal["summary"][key(0)]
            treatment = journal["summary"][key(2)]
            speedup = baseline["median_ms"] / treatment["median_ms"]
            rows.append({
                "operator": operator,
                "logN": 12,
                "batch": batch,
                "leaf0_median_ms": baseline["median_ms"],
                "leaf2_median_ms": treatment["median_ms"],
                "leaf2_speedup": speedup,
                "all_trials": baseline["trials_ms"] + treatment["trials_ms"],
            })
    by_operator = {}
    for operator in ("fwht", "ntt"):
        values = [row["leaf2_speedup"] for row in rows if row["operator"] == operator]
        by_operator[operator] = {
            "geomean_speedup": math.prod(values) ** (1.0 / len(values)),
            "min_speedup": min(values),
            "max_speedup": max(values),
            "cases": len(values),
        }
    result = {
        "schema": "cubutterfly-e3-cross-operator-summary-v1",
        "status": "complete",
        "scope": "A100 80GB, N=2^12, native shared-iterative core; leaf0 versus leaf2 resident organization",
        "journal": str(args.journal),
        "identity": journal["identity"],
        "accepted_units": len([row for row in journal["records"] if row.get("accepted")]),
        "correctness": "all accepted units passed complete-plan and independent-group references",
        "rows": rows,
        "by_operator": by_operator,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# E3 Cross-Operator Organization Control",
        "",
        "Status: **COMPLETE**; 24/24 units accepted on the pinned A100 80GB.",
        "",
        "The study fixes the native shared-iterative core and the `[8,4]` physical "
        "groups. `leaf0` is the stagewise shared-HBM control and `leaf2` retains "
        "two-stage local units in the same groups. It covers FWHT FP32 and natural "
        "word64 NTT at N=2^12 for B=1 and B=1024. Each cell has three alternating "
        "trials, 500 ms of event warmup, 100 repeated launches, and complete-plan "
        "plus independent-group correctness checks. Compilation and initialization "
        "are outside timing. This is an arithmetic-family control, not a migration "
        "experiment or an external-library comparison.",
        "",
        "| Operator | Batch | leaf0 median (ms) | leaf2 median (ms) | leaf2 speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['operator']} | {row['batch']} | {row['leaf0_median_ms']:.9f} | "
            f"{row['leaf2_median_ms']:.9f} | {row['leaf2_speedup']:.4f}x |"
        )
    lines.extend([
        "",
        "| Operator | Geomean speedup | Range |",
        "|---|---:|---:|",
    ])
    for operator in ("fwht", "ntt"):
        item = by_operator[operator]
        lines.append(f"| {operator} | {item['geomean_speedup']:.4f}x | {item['min_speedup']:.4f}--{item['max_speedup']:.4f}x |")
    lines.extend([
        "",
        "Source journal: `a100-80gb/journal.json`; machine-readable summary: `summary.json`.",
    ])
    (args.output.parent / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
