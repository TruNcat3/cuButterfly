#!/usr/bin/env python3
"""Normalize an explicit stage-calibration point set.

The stage calibrator intentionally accepts only complete semantic requests
with ``mapping_json``.  Repository campaign files are easier to review when
they keep the mapping as an object, so this CPU-only adapter performs the
lossless normalization once and records the selected scope.  It does not
enumerate candidates, query CUDA, or infer missing mappings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "cubutterfly-stage-calibration-points-v1"
NORMALIZED_SCHEMA = "cubutterfly-stage-points-v2"


def _read(path: Path) -> Any:
    value = json.loads(path.read_text())
    if isinstance(value, dict):
        value = value.get("points", value.get("workloads", value))
    if not isinstance(value, list):
        raise ValueError("input must be a JSON list or an object containing points")
    return value


def normalize(points: list[dict[str, Any]], *, operators: set[str] | None = None,
              backends: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    omitted: list[dict[str, Any]] = []
    for position, original in enumerate(points):
        if not isinstance(original, dict):
            raise ValueError(f"point {position} must be an object")
        point = dict(original)
        mapping = point.pop("mapping", None)
        if mapping is None:
            mapping = point.get("mapping_json")
            if isinstance(mapping, str):
                try:
                    mapping = json.loads(mapping)
                except json.JSONDecodeError as error:
                    raise ValueError(f"point {position} mapping_json is invalid JSON") from error
        if not isinstance(mapping, dict):
            raise ValueError(f"point {position} requires a mapping object or mapping_json object")
        operator = str(point.get("operator", "fft"))
        backend = str(mapping.get("backend", point.get("backend", "")))
        if operators and operator not in operators:
            omitted.append({"index": position, "reason": "operator-filter", "operator": operator})
            continue
        if backends and backend not in backends:
            omitted.append({"index": position, "reason": "backend-filter", "backend": backend})
            continue
        if "logN" not in point and "log_n" not in point and "N" not in point:
            raise ValueError(f"point {position} requires logN, log_n or N")
        try:
            batch = int(point.get("batch", 1))
        except (TypeError, ValueError) as error:
            raise ValueError(f"point {position} batch must be an integer") from error
        if batch < 1:
            raise ValueError(f"point {position} batch must be positive")
        point["batch"] = batch
        point.setdefault("operator", operator)
        point.setdefault("precision", "fp32")
        point["mapping_json"] = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
        key = json.dumps({key: point[key] for key in sorted(point)}, sort_keys=True,
                         separators=(",", ":"))
        if key in seen:
            omitted.append({"index": position, "reason": "duplicate"})
            continue
        seen.add(key)
        selected.append(point)
    summary = {
        "schema": SCHEMA,
        "normalized_schema": NORMALIZED_SCHEMA,
        "input_count": len(points),
        "selected_count": len(selected),
        "omitted_count": len(omitted),
        "omitted": omitted,
        "selection_policy": "explicit points only; no candidate enumeration or inferred defaults",
    }
    return selected, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operators", nargs="*")
    parser.add_argument("--backends", nargs="*")
    args = parser.parse_args()
    points = _read(args.input)
    selected, summary = normalize(points,
                                  operators=set(args.operators) if args.operators else None,
                                  backends=set(args.backends) if args.backends else None)
    document = dict(schema=NORMALIZED_SCHEMA, points=selected, summary=summary,
                    source={"path": str(args.input.resolve()),
                            "sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
