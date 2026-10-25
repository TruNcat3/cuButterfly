#!/usr/bin/env python3
"""Summarize precision/inverse timing without mixing semantic contracts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze(source: Path, output: Path) -> dict[str, Any]:
    report = json.loads(source.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in report["rows"]:
        sample = row.get("sample") or {}
        if not row.get("correct") or not sample:
            continue
        identity = row["identity"]
        groups[(identity["operator"], identity["precision_regime"], identity["direction"])].append(
            float(sample["kernel_ms"])
        )
    summary = []
    for key in sorted(groups):
        values = sorted(groups[key])
        summary.append({
            "operator": key[0], "precision_regime": key[1], "direction": key[2],
            "count": len(values), "kernel_ms": values[len(values) // 2],
        })
    by_pair = {(row["operator"], row["precision_regime"], row["direction"]): row for row in summary}
    for row in summary:
        inverse = by_pair.get((row["operator"], row["precision_regime"], "inverse"))
        forward = by_pair.get((row["operator"], row["precision_regime"], "forward"))
        if inverse and forward and forward["kernel_ms"] > 0:
            row["inverse_over_forward"] = inverse["kernel_ms"] / forward["kernel_ms"]
    result = {
        "schema": "cubutterfly-extended-semantics-analysis-v1",
        "source": str(source.resolve()),
        "run_summary": report["summary"],
        "correctness_complete": report["summary"]["failed"] == 0,
        "groups": summary,
        "interpretation": "These are internal CT lowering timings. No GS or external-library claim is made by this report.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source.resolve(), args.output.resolve())
    print(json.dumps({"groups": len(result["groups"]), "correctness_complete": result["correctness_complete"]}, sort_keys=True))
    return 0 if result["correctness_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
