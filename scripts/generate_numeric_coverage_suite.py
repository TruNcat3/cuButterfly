#!/usr/bin/env python3
import argparse
import json
import math
import pathlib

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv, shape_key
from build_numeric_piecewise_selector import (apply_followup_boundaries, contract_key,
                                               is_stable, measured_points,
                                               overlay_points)
from generate_numeric_regime_suite import expand


def adaptive_shapes(points, minimum_quick_margin=0.15, minimum_full_margin=0.05):
    groups = {}
    for point in points:
        groups.setdefault(contract_key(point), []).append(point)
    boundary_shapes = set()
    for values in groups.values():
        values.sort(key=lambda item: int(item["batch"]))
        for left, right in zip(values, values[1:]):
            if left["winner"] != right["winner"]:
                boundary_shapes.update((shape_key(left), shape_key(right)))

    selected = []
    for values in groups.values():
        for index, point in enumerate(values):
            if (point.get("evidence") == "full" or point.get("evidence_boundary", False)
                    or shape_key(point) in boundary_shapes
                    or is_stable(point, minimum_quick_margin, minimum_full_margin)):
                continue
            neighbors = values[max(0, index - 1):index] + values[index + 1:index + 2]
            consistent = [neighbor for neighbor in neighbors
                          if neighbor["winner"] == point["winner"]
                          and not neighbor.get("evidence_boundary", False)]
            if not consistent:
                continue
            span = max(abs(math.log2(int(neighbor["batch"]) / int(point["batch"])))
                       for neighbor in consistent)
            selected.append((span, point))
    selected.sort(key=lambda item: (-item[0], contract_key(item[1]), int(item[1]["batch"])))
    return [point for _, point in selected]


def generate(suite, points, max_shapes=0):
    selected = adaptive_shapes(points)
    if max_shapes > 0:
        selected = selected[:max_shapes]
    selected_keys = {shape_key(point) for point in selected}
    cases = []
    for case in suite["cases"]:
        key = (case["operator"], case["precision"], case.get("accumulation", "native"),
               int(case["logN"]), int(case["batch"]))
        if key not in selected_keys:
            continue
        copy = dict(case)
        copy["tier"] = "full"
        copy["coverage_reason"] = "weak non-boundary quick anchor with a same-winner neighbor"
        cases.append(copy)
    case_shapes = {(case["operator"], case["precision"], case.get("accumulation", "native"),
                    int(case["logN"]), int(case["batch"])) for case in cases}
    missing = selected_keys - case_shapes
    if missing:
        raise ValueError(f"selected shapes are absent from generated suite: {sorted(missing)[:3]}")
    cell_ids = {case["group"].rsplit("-b", 1)[0] for case in cases}
    cells = [cell for cell in suite.get("cells", []) if cell["id"] in cell_ids]
    source_shapes = [{
        "operator": point["operator"], "precision": point["precision"],
        "accumulation": point.get("accumulation", "native"),
        "logN": int(point["logN"]), "batch": int(point["batch"]),
        "quick_winner": point["winner"], "quick_margin": point["winner_margin"],
    } for point in selected]
    return {
        "schema_version": 1, "study": "numeric-regime-coverage-expansion-v0.5",
        "hardware": suite["hardware"], "protocols": suite["protocols"],
        "binaries": suite["binaries"], "source_shapes": source_shapes,
        "cells": cells, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate full timing for weak non-boundary selector anchors.")
    parser.add_argument("--space", type=pathlib.Path,
                        default=pathlib.Path("config/v100_numeric_regime_space.json"))
    parser.add_argument("--quick-raw", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_regime_quick_raw.csv"))
    parser.add_argument("--quick-manifest", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_regime_suite.json"))
    parser.add_argument("--calibration-raw", type=pathlib.Path)
    parser.add_argument("--calibration-manifest", type=pathlib.Path)
    parser.add_argument("--followups", type=pathlib.Path)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    suite = expand(json.loads(args.space.read_text()))
    quick_manifest = json.loads(args.quick_manifest.read_text())
    points = measured_points(aggregate_samples(read_csv(args.quick_raw), quick_manifest))
    if bool(args.calibration_raw) != bool(args.calibration_manifest):
        raise ValueError("--calibration-raw and --calibration-manifest must be provided together")
    if args.calibration_raw:
        calibration_manifest = json.loads(args.calibration_manifest.read_text())
        calibration = aggregate_samples(read_csv(args.calibration_raw), calibration_manifest)
        points = overlay_points(points, measured_points(calibration, evidence="full"))
    if args.followups:
        points = apply_followup_boundaries(points, read_csv(args.followups))
    document = generate(suite, points, args.max_shapes)
    if not document["cases"]:
        raise ValueError("no coverage-expansion cases were selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"shapes={len(document['source_shapes'])} cases={len(document['cases'])}")


if __name__ == "__main__":
    main()
