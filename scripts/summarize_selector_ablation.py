#!/usr/bin/env python3
"""Summarize paired public measured-selector traces.

The two inputs must be traces from the same semantic workload and build.  The
script deliberately reports the measured candidate population and the
selected mapping; it does not infer a missing mapping or turn an incomplete
trace into a performance claim.
"""

import argparse
import json
import math
from pathlib import Path


def _canonical_mapping(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, str):
        try:
            return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return value
    return str(value)


def _read_trace(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError("{}:{} is not JSON: {}".format(path, line_number, error))
        if not isinstance(row, dict):
            raise ValueError("{}:{} is not a JSON object".format(path, line_number))
        rows.append(row)
    if not rows:
        raise ValueError("trace is empty: {}".format(path))
    return rows


def _summarize_trace(path):
    rows = _read_trace(path)
    searches = {}
    for row in rows:
        search_id = row.get("search_id")
        if search_id is None:
            raise ValueError("trace row has no search_id: {}".format(path))
        searches.setdefault(search_id, []).append(row)

    summaries = []
    for search_id, search_rows in sorted(searches.items(), key=lambda item: item[0]):
        inventory = [row for row in search_rows if row.get("phase") == "inventory"]
        if len(inventory) != 1:
            raise ValueError(
                "expected one inventory event for search_id {}, found {} in {}".format(
                    search_id, len(inventory), path
                )
            )
        inventory = inventory[0]
        measured = []
        for row in search_rows:
            if row.get("phase") not in ("candidate_result", "result"):
                continue
            accepted = row.get("outcome") == "accepted" or row.get("status") == "accepted"
            kernel_ms = row.get("kernel_ms")
            if accepted and isinstance(kernel_ms, (int, float)) and math.isfinite(kernel_ms):
                measured.append((float(kernel_ms), row.get("mapping")))
        measured.sort(key=lambda item: item[0])
        best = measured[0] if measured else (None, None)
        summary_rows = [row for row in search_rows if row.get("phase") == "summary"]
        summary = summary_rows[-1] if summary_rows else {}
        best_mapping = summary.get("best_mapping")
        if best_mapping is None:
            best_mapping = best[1]
        summaries.append(
            {
                "search_id": search_id,
                "search": inventory.get("search"),
                "candidate_count": inventory.get("candidate_count"),
                "restricted_candidate_count": inventory.get("restricted_candidate_count"),
                "disabled_candidate_count": inventory.get("disabled_candidate_count"),
                "scheduled_candidate_count": inventory.get("scheduled_candidate_count"),
                "search_budget": inventory.get("search_budget"),
                "disabled_mechanisms": inventory.get("disabled_mechanisms", []),
                "accepted_count": len(measured),
                "rejected_count": sum(
                    1
                    for row in search_rows
                    if row.get("phase") in ("candidate_result", "result")
                    and (row.get("outcome") == "rejected" or row.get("status") == "rejected")
                ),
                "best_kernel_ms": best[0],
                "best_mapping": _canonical_mapping(best_mapping),
            }
        )
    return {"path": str(path), "searches": summaries}


def _pair(unrestricted, restricted):
    if len(unrestricted["searches"]) != len(restricted["searches"]):
        raise ValueError("paired traces contain different search counts")
    pairs = []
    for left, right in zip(unrestricted["searches"], restricted["searches"]):
        if left.get("search") != right.get("search"):
            raise ValueError("paired traces contain different search families")
        left_ms = left.get("best_kernel_ms")
        right_ms = right.get("best_kernel_ms")
        regret = None
        if left_ms is not None and right_ms is not None and left_ms > 0:
            regret = right_ms / left_ms - 1.0
        pairs.append(
            {
                "search": left.get("search"),
                "unrestricted": left,
                "restricted": right,
                "selected_mapping_same": (
                    left.get("best_mapping") is not None
                    and left.get("best_mapping") == right.get("best_mapping")
                ),
                "restricted_regret_vs_unrestricted_best": regret,
            }
        )
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unrestricted", type=Path)
    parser.add_argument("restricted", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    unrestricted = _summarize_trace(args.unrestricted)
    restricted = _summarize_trace(args.restricted)
    result = {
        "schema": "cubutterfly-selector-ablation-summary-v1",
        "unrestricted": unrestricted,
        "restricted": restricted,
        "paired": _pair(unrestricted, restricted),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
