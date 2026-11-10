#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser(description="Reduce GPU hardware microbenchmarks into a capability profile.")
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with args.input.open() as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("hardware capability input is empty")

    metric_fields = {
        "global_copy_GB_s": "global_feedback_bytes_per_second",
        "butterfly_Gbutterfly_s": "equivalent_butterflies_per_second",
        "shared_exchange_GB_s": "interstage_shared_bytes_per_second",
        "barrier_GCTA_s": "cta_barriers_per_second",
    }
    capabilities = {
        output_name: statistics.median(float(row[input_name]) for row in rows) * 1.0e9
        for input_name, output_name in metric_fields.items()
    }
    stage_pipeline = {
        f"Us{stage_space}_butterflies_per_second": statistics.median(
            float(row[f"pipeline_us{stage_space}_Gbutterfly_s"]) for row in rows
        )
        * 1.0e9
        for stage_space in (1, 2, 4, 8)
    }
    if any(row.get("pipeline_correct") != "1" for row in rows):
        raise ValueError("at least one stage-pipeline trial failed cross-Us verification")

    profile = {
        "device": rows[0]["device"],
        "compute_capability": rows[0]["compute_capability"],
        "sm_count": int(rows[0]["sm_count"]),
        "trials": len(rows),
        "conditions": {
            name: int(rows[0][name]) for name in ("points", "blocks", "threads", "iterations", "warmup", "repeat")
        },
        "capabilities": capabilities,
        "measured_stage_pipeline": stage_pipeline,
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")


if __name__ == "__main__":
    main()
