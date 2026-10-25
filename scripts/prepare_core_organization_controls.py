#!/usr/bin/env python3
"""Prepare the current-build core-by-organization control cohort.

The manifest contains only legal, explicitly named command-line variants.  It
does not infer support for a Cartesian product that the lowering rejects.
"""

import argparse
import json
from pathlib import Path


def build_study():
    cases = []

    # FWHT has a legal scalar radix-2 shared organization and a separate
    # temporal core family.  The temporal family is intentionally limited to
    # logN<=10 by the current generated kernel contract.
    for stage_partition in ("1,1,1,1,1,1,1,1,1,1,1,1,1,1", "8,6"):
        cases.append({
            "id": "fwht-shared-radix2-stage-" + stage_partition.replace(",", "-"),
            "operator": "fwht",
            "runner": "cubutterfly_bench",
            "args": ["--operator", "fwht", "--backend", "shared-iterative",
                     "--logN", "14", "--batch", "4", "--stage-partition", stage_partition,
                     "--compute-unit", "radix2", "--shared-layout", "linear",
                     "--tile-threads", "128", "--precision", "fp32", "--warmup", "2",
                     "--repeat", "10", "--verify", "--verify-batches", "4", "--csv"],
            "semantic_contract": {"N": 16384, "batch": 4, "precision": "fp32",
                                  "operator": "fwht", "core": "scalar-radix2"},
            "organization": {"backend": "shared-iterative", "stage_partition": stage_partition},
        })
    for compute_unit in ("radix2", "radix4"):
        cases.append({
            "id": "fwht-temporal-{}-n1024-b16".format(compute_unit),
            "operator": "fwht",
            "runner": "cubutterfly_bench",
            "args": ["--operator", "fwht", "--backend", "temporal-tile", "--logN", "10",
                     "--batch", "16", "--compute-unit", compute_unit, "--tile-threads", "128",
                     "--precision", "fp32", "--warmup", "2", "--repeat", "10", "--verify",
                     "--verify-batches", "16", "--csv"],
            "semantic_contract": {"N": 1024, "batch": 16, "precision": "fp32",
                                  "operator": "fwht", "organization": "temporal-tile"},
            "core": compute_unit,
        })

    # The current specialized NTT templates bind their core and stage
    # partition together at the 10+10 support point.  Use that explicit
    # support domain instead of presenting a generic N=2^14 mapping field as
    # evidence that a different kernel was executed.
    for core in ("homogeneous-warp128-static-io-radix2",
                 "homogeneous-warp128-vector-radix4-static-io"):
        cases.append({
            "id": "ntt-{}-n1048576-b1".format(core),
            "operator": "ntt",
            "runner": "cuntt_bench",
            "args": ["--logN", "20", "--batch", "1", "--backend", "hierarchical-dataflow",
                     "--stage-partition", "10,10", "--segment-cores", core,
                     "--segment-threads", "128", "--segment-units", "8", "--word-bits", "64",
                     "--modulus", "576460756061519873", "--warmup", "2", "--repeat", "10",
                     "--verify", "--verify-batches", "1", "--csv"],
            "semantic_contract": {"N": 1048576, "batch": 1, "precision": "uint64",
                                  "operator": "ntt", "modulus": 576460756061519873},
            "organization": {"backend": "hierarchical-dataflow", "stage_partition": "10,10"},
            "core": core,
        })
    return {
        "schema": "cubutterfly-core-organization-control-v1",
        "build_policy": "current-build-only; every row records executable support rather than inferred coverage",
        "protocol": {"warmup": 2, "repeat": 10, "verify": True, "timed_region": "kernel"},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    study = build_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and json.loads(args.output.read_text()) != study:
        raise SystemExit("existing study differs; choose a new output path")
    args.output.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "prepared", "cases": len(study["cases"])}))


if __name__ == "__main__":
    main()
