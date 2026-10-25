#!/usr/bin/env python3
"""Prepare a small, current-build mechanism cohort without querying a GPU.

The cohort isolates stage organization while keeping the arithmetic core and
semantic request fixed.  It is intentionally finite: the paper needs a
controlled cross-operator witness, not an unbounded parameter sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _mapping(operator: str, partition: list[int]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "kind": "ntt" if operator == "ntt" else "butterfly",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "stage_partition": partition,
        "tile_threads": 128,
    }
    if operator == "ntt":
        value.update({"dataflow_layout": "linear", "modular_multiply": "shoup"})
    else:
        value.update({"fft_core": "scalar", "local_exchange": "shared", "shared_layout": "linear"})
    return value


def build_study() -> dict[str, object]:
    # Two medium/large points per operator expose both launch-bound and
    # multi-stage behavior while keeping the cohort short enough for an
    # isolated diagnostic run.
    requests = [
        ("fft", "fp32", 12, 4),
        ("fft", "fp32", 18, 1),
        ("fwht", "fp32", 12, 4),
        ("fwht", "fp32", 18, 1),
        ("ntt", "word64", 12, 4),
        ("ntt", "word64", 18, 1),
    ]
    cases = []
    for operator, precision, log_n, batch in requests:
        if operator == "ntt":
            workload = {
                "operator": operator,
                "precision": precision,
                "logN": log_n,
                "batch": batch,
                "placement": "out-of-place",
                "direction": "forward",
                "normalization": "none",
                "input_order": "natural",
                "output_order": "natural",
                # This is the same verified 59-bit prime used by the current
                # word64 benchmark contract; the runner rejects composite Q.
                "modulus": 576460756061519873,
            }
        else:
            workload = {
                "operator": operator,
                "precision": precision,
                "logN": log_n,
                "batch": batch,
                "placement": "out-of-place",
                "direction": "forward",
                "normalization": "none",
            }
        stagewise = [1] * log_n
        resident = [8] * (log_n // 8)
        if log_n % 8:
            resident = [8] * (log_n // 8) + [log_n % 8]
        variants = [
            {"id": "stagewise", "mapping": _mapping(operator, stagewise)},
            {"id": "resident8", "mapping": _mapping(operator, resident)},
        ]
        cases.append({
            "id": f"{operator}-{precision}-n{log_n}-b{batch}",
            "experiment": "E2-current-stage-organization",
            "workload": workload,
            "variants": variants,
            "comparisons": [{
                "baseline": "stagewise",
                "treatment": "resident8",
                "changed_axes": ["stage_partition"],
            }],
        })
    return {
        "schema": "cubutterfly-controlled-experiment-v1",
        "protocol": {"trials": 3, "warmup": 10, "repeat": 20, "verify_batches": 0},
        "scope": (
            "Current build, same scalar radix2 arithmetic core and explicit semantic request; "
            "only stage_partition changes. Complete-call CUDA-event timing excludes compile, "
            "allocation and warmup. This cohort is a controlled mechanism witness, not a claim "
            "that resident8 wins for every operator or size."
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != document:
            raise SystemExit("existing study differs; choose a new output path")
    else:
        args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "cases": len(document["cases"]), "variants": 2 * len(document["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
