#!/usr/bin/env python3
"""Create a compact, explicit summary of core/organization control results."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.results.read_text())
    study_cases = {}
    study_path = document.get("study")
    if study_path and Path(study_path).exists():
        study_cases = {case["id"]: case for case in json.loads(Path(study_path).read_text())["cases"]}
    rows = []
    for case in document["cases"]:
        study_case = study_cases.get(case["id"], {})
        contract = study_case.get("semantic_contract", case.get("semantic_contract", {}))
        measurement = case.get("measurement", {})
        mapping = {}
        if measurement.get("mapping_json"):
            mapping = json.loads(measurement["mapping_json"])
        rows.append({
            "id": case["id"],
            "operator": study_case.get("operator", case.get("operator")),
            "N": contract.get("N"),
            "batch": contract.get("batch"),
            "core": study_case.get("core", case.get("core", "")) or contract.get("core", "") or ";".join(
                item.get("core", "") for item in mapping.get("subgraph_mappings", [])
            ),
            "backend": mapping.get("backend", measurement.get("backend", "")),
            "stage_partition": "+".join(str(item) for item in mapping.get("stage_partition", [])),
            "kernel_ms": measurement.get("kernel_ms"),
            "correct": measurement.get("correct"),
            "status": case.get("status"),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "summarized", "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
