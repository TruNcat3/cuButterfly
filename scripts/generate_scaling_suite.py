#!/usr/bin/env python3
import argparse
import json
import pathlib


WORKLOAD_FIELDS = ("operator", "precision", "logN", "modulus_bits", "direction",
                   "element_stride", "batch_padding")


def expand(document):
    if document.get("schema_version") != 1:
        raise ValueError("unsupported scaling-space schema")
    required = {"hardware", "max_points", "protocols", "binaries", "workloads"}
    missing = required - set(document)
    if missing:
        raise ValueError(f"scaling space lacks {sorted(missing)}")
    cases = []
    seen = set()
    for workload in document["workloads"]:
        for field in ("id", "operator", "precision", "logN", "batches", "implementations"):
            if field not in workload:
                raise ValueError(f"scaling workload lacks {field}")
        quick = [int(value) for value in workload["batches"].get("quick", [])]
        full = [int(value) for value in workload["batches"].get("full", [])]
        if not quick or len(set(quick + full)) != len(quick + full):
            raise ValueError(f"workload {workload['id']} has empty or duplicate batch levels")
        if any(value <= 0 for value in quick + full):
            raise ValueError(f"workload {workload['id']} has non-positive batch")
        if max(quick + full) * (1 << int(workload["logN"])) > int(document["max_points"]):
            raise ValueError(f"workload {workload['id']} exceeds max_points")
        references = sum(bool(item.get("reference")) for item in workload["implementations"])
        if references > 1:
            raise ValueError(f"workload {workload['id']} has multiple references")
        for batch in quick + full:
            tier = "quick" if batch in quick else "full"
            group = f"{workload['id']}-batch{batch}"
            for implementation in workload["implementations"]:
                case_id = f"{workload['id']}_{implementation['id']}_b{batch}"
                if case_id in seen:
                    raise ValueError(f"duplicate generated case {case_id}")
                seen.add(case_id)
                case = {
                    "id": case_id,
                    "tier": tier,
                    "group": group,
                    "runner": implementation["runner"],
                    "implementation": implementation["name"],
                    "reference": bool(implementation.get("reference")),
                    "batch": batch,
                    "args": implementation["args"],
                }
                for field in WORKLOAD_FIELDS:
                    if field in workload:
                        case[field] = workload[field]
                cases.append(case)
    return {
        "schema_version": 1,
        "hardware": document["hardware"],
        "protocols": document["protocols"],
        "binaries": document["binaries"],
        "cases": cases,
        "external_evidence": document.get("external_evidence", []),
    }


def main():
    parser = argparse.ArgumentParser(description="Expand an orthogonal logN/batch scaling space.")
    parser.add_argument("--spec", type=pathlib.Path, default=pathlib.Path("config/v100_scaling_space.json"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--check", type=pathlib.Path, help="Fail if this generated manifest is stale.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    suite = expand(json.loads(args.spec.read_text()))
    rendered = json.dumps(suite, indent=2) + "\n"
    if args.check:
        if args.check.read_text() != rendered:
            raise ValueError(f"generated scaling manifest is stale: {args.check}")
        print(f"validated generated manifest cases={len(suite['cases'])}")
        return
    if args.validate_only:
        print(f"validated workloads={len(json.loads(args.spec.read_text())['workloads'])} cases={len(suite['cases'])}")
        return
    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
