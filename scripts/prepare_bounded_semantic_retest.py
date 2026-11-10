#!/usr/bin/env python3
"""Prepare a bounded semantic matrix and exclusion audit without GPU work."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from prepare_semantic_retest import configuration as original_configuration
from semantic_campaign_manifest import generate_manifest, semantic_id, validate_config


DEFAULT_MAX_ELEMENTS = 1 << 20
RANK_POINT_CAPS = {1: 1 << 20, 2: 1024 * 1024, 3: 64 * 64 * 64}


def prepare_configuration(max_elements=DEFAULT_MAX_ELEMENTS):
    """Keep semantic dimensions intact; declare eligible scaling batches explicitly."""
    if type(max_elements) is not int or max_elements < 1:
        raise ValueError("max_elements must be a positive integer")
    original = original_configuration()
    config = {"schema": original["schema"], "populations": []}
    audit = []
    purposes = {}
    for population in original["populations"]:
        if population["name"] != "size-batch-scaling":
            config["populations"].append(copy.deepcopy(population))
            purposes[population["name"]] = "functional-correctness-and-support"
            continue
        for shape in population["shapes"]:
            points = math.prod(shape)
            point_cap = RANK_POINT_CAPS[len(shape)]
            eligible = []
            for batch in population["batches"]:
                reasons = []
                if points > point_cap:
                    reasons.append("rank-point-cap")
                if points * batch > max_elements:
                    reasons.append("total-logical-element-cap")
                if not reasons:
                    eligible.append(batch)
                audit.append({"shape": list(shape), "batch": batch,
                              "points_per_transform": points,
                              "total_logical_elements": points * batch,
                              "rank_point_cap": point_cap,
                              "decision": "excluded" if reasons else "retained",
                              "reasons": reasons})
            if eligible:
                bounded = copy.deepcopy(population)
                bounded["name"] = "size-batch-scaling-" + "x".join(map(str, shape))
                bounded["shapes"] = [list(shape)]
                bounded["batches"] = eligible
                config["populations"].append(bounded)
                purposes[bounded["name"]] = "bounded-scaling-correctness-and-performance"
    validate_config(config)
    policy = {
        "schema": "cubutterfly-bounded-semantic-policy-v1",
        "max_scaling_logical_elements": max_elements,
        "rank_point_caps": {str(rank): cap for rank, cap in RANK_POINT_CAPS.items()},
        "candidate_batches": [1, 16, 256, 4096],
        "budget_scope": "size-batch-scaling populations only",
        "reference_budget_note": "Default element cap matches the current all-element oracle cap. NTT operation limits remain separate.",
        "selection_basis": "shape and batch only; no support or measured-performance filtering",
        "batch_policy": "Exclude whole requested combinations; never shrink a requested batch.",
        "population_purposes": purposes,
        "non_performance_populations": ["rank-boundary"],
        "shape_batch_audit": audit,
        "retained_shape_batch_combinations": sum(row["decision"] == "retained" for row in audit),
        "excluded_shape_batch_combinations": sum(row["decision"] == "excluded" for row in audit),
        "source_config_sha256": semantic_id(original),
        "config_sha256": semantic_id(config),
    }
    return config, policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-elements", type=int, default=DEFAULT_MAX_ELEMENTS,
                        help="total logical element budget for each scaling shape/batch")
    args = parser.parse_args(argv)
    try:
        config, policy = prepare_configuration(args.max_elements)
    except ValueError as error:
        parser.error(str(error))
    paths = {name: args.output / (name + ".json") for name in ("config", "policy", "manifest")}
    if any(path.exists() for path in paths.values()):
        parser.error("use a fresh config/policy/manifest output")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in (("config", config), ("policy", policy)):
        with paths[name].open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    manifest = generate_manifest(config, args.library, config_path=paths["config"])
    with paths["manifest"].open("x") as stream:
        json.dump(manifest, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(json.dumps({"metrics": manifest["metrics"],
                      "retained_shape_batch_combinations": policy["retained_shape_batch_combinations"],
                      "excluded_shape_batch_combinations": policy["excluded_shape_batch_combinations"]}, indent=2))


if __name__ == "__main__":
    main()
