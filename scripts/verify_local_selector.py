#!/usr/bin/env python3
"""Replay promoted mappings through the common automatic selector."""
import argparse
import csv
import io
import json
import pathlib
import subprocess

from local_selector_data import select_points, ENUM_FIELDS, INT_FIELDS
from run_comprehensive_suite import require_exclusive_gpu


def verify(build_dir, records, output):
    results = []
    for point in select_points({"candidates": records}):
        expected = point["samples"][0]
        ntt = expected["operator"] == "ntt"
        command = [str(build_dir / ("cuntt_bench" if ntt else "cubutterfly_bench")), "--auto-select", "--verify", "--csv",
                   "--warmup", "3", "--repeat", "5"]
        fields = ("word_bits", "modulus", "input_order", "output_order", "logN", "batch") if ntt else (
            "operator", "precision", "placement", "logN", "batch", "accumulation", "normalization", "element_stride", "batch_stride")
        for field in fields:
            if field in expected:
                command += ["--" + field.replace("_", "-"), expected[field]]
        if expected.get("direction") == "inverse" or (ntt and expected.get("inverse") == "1"):
            command.append("--inverse")
        if expected.get("stage_matrices"):
            for matrix in expected["stage_matrices"].split("x"):
                command += ["--stage-matrix", matrix.replace(":", ",")]
        require_exclusive_gpu()
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        require_exclusive_gpu()
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        if len(rows) != 1:
            raise ValueError("selector replay did not produce exactly one result")
        actual = rows[0]
        if ntt:
            actual.update(operator="ntt", precision="word"+actual["word_bits"], placement=actual["output_order"])
        if actual.get("correct") != "1" or actual.get("selected_implementation") != point["name"]:
            raise ValueError(f"selector replay failed for {point['name']}: {actual}")
        if expected.get("mapping_json"):
            if json.loads(actual.get("mapping_json", "{}")) != json.loads(expected["mapping_json"]):
                raise ValueError(f"complete mapping replay mismatch for {point['name']}")
        fields = [] if ntt else [*ENUM_FIELDS, *INT_FIELDS, "stages_per_decomposition", "segment_threads", "segment_ept",
                  "boundary_twiddles", "boundary_layouts", "boundary_residencies", "group_threads", "group_ept",
                  "segment_cores", "group_cores",
                  "operator", "precision", "direction", "accumulation", "placement", "stage_matrices", "stage_overlap"]
        for field in fields:
            if field in expected and actual.get(field) != expected[field]:
                raise ValueError(f"selector mapping mismatch for {point['name']}/{field}: {actual.get(field)} != {expected[field]}")
        results.append({"name": point["name"], "correct": True, "mapping_matches": True, "sample": actual})
        output.write_text(json.dumps(results, indent=2) + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--calibration", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    results = verify(args.build_dir.resolve(), json.loads(args.calibration.read_text()), args.output)
    print(f"verified {len(results)} registry mappings through automatic selection")


if __name__ == "__main__":
    main()
