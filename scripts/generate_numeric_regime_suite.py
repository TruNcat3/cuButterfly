#!/usr/bin/env python3
import argparse
import json
import math
import pathlib

from residency_features import derive_residency_features


def _baseline_coverage(spec, operator, precision):
    matches = []
    for baseline in spec["baselines"]:
        if baseline["operator"] == operator and precision in baseline["precisions"]:
            matches.append({key: value for key, value in baseline.items() if key != "precisions"})
    if not matches:
        return [{"implementation": "none", "status": "not-declared",
                 "reason": "no baseline policy was declared for this semantic cell"}]
    return matches


def _local_stages(mapping, log_n):
    value = mapping.get("local_stages", log_n)
    if value == "balanced":
        return min(10, (log_n + 1) // 2)
    return min(int(value), log_n)


def _mapping_args(operator, contract, mapping, log_n, stage_matrix=None):
    if operator == "ntt":
        return [
            "--backend", mapping["backend"], "--compute-unit", mapping["compute_unit"],
            "--threads-per-block", str(mapping["threads"]),
            "--word-bits", str(contract["storage_bits"]),
            "--cross-twiddle", "fused", "--output-order", "natural",
        ]
    local_stages = _local_stages(mapping, log_n)
    args = [
        "--operator", operator, "--precision", contract["precision"],
        "--accumulation", contract["accumulation"], "--backend", mapping["backend"],
        "--compute-unit", mapping["compute_unit"], "--tile-threads", str(mapping["threads"]),
        "--normalization", "none",
    ]
    if mapping["backend"] in ("online-reorder", "hierarchical"):
        args += ["--local-stages", str(local_stages)]
    if mapping["backend"] == "online-reorder":
        args += ["--reorder-columns", "1"]
    if stage_matrix:
        args += ["--stage-matrix", stage_matrix]
    return args


def _resource_estimate(spec, operator, contract, mapping, log_n, batch, value_lanes):
    hardware = spec["hardware"]
    word_bytes = contract["storage_bits"] // 8
    state_words = mapping["state_words_per_thread"] * value_lanes
    register_scale = max(1, math.ceil(contract["compute_bits"] / 32))
    registers = mapping["base_registers"] + state_words * register_scale
    local_stages = _local_stages(mapping, log_n)
    tile_values = min(1 << log_n, mapping["threads"] * max(2, mapping["state_words_per_thread"]))
    shared_bytes = tile_values * word_bytes * value_lanes
    if mapping["backend"] == "temporal-tile":
        ctas_per_transform = 1
    else:
        ctas_per_transform = max(1, 1 << max(0, log_n - local_stages - 1))
    total_ctas = batch * ctas_per_transform
    features = derive_residency_features(
        hardware, mapping["threads"], registers, shared_bytes, total_ctas,
        temporal_state_words_per_thread=state_words)
    return features, ctas_per_transform, registers, shared_bytes


def _batch_levels(spec, operator, contract, mapping, log_n, value_lanes):
    features, ctas_per_transform, _, _ = _resource_estimate(
        spec, operator, contract, mapping, log_n, 1, value_lanes)
    concurrency = spec["hardware"]["sm_count"] * features["resident_ctas_per_sm"]
    max_batch = max(1, spec["max_points"] // (1 << log_n))
    levels = {1: "quick"}
    for tier in ("quick", "full"):
        for wave in spec["wave_targets"][tier]:
            center = max(1, math.ceil(wave * concurrency / ctas_per_transform))
            neighborhood = (center - 1, center, center + 1) if wave == 1.0 else (center,)
            for batch in neighborhood:
                if 1 <= batch <= max_batch:
                    old = levels.get(batch)
                    levels[batch] = "quick" if old == "quick" or tier == "quick" else "full"
    levels[max_batch] = levels.get(max_batch, "full")
    return sorted(levels.items())


def expand(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported numeric-regime schema")
    required = {"hardware", "max_points", "protocols", "binaries", "lengths",
                "wave_targets", "contracts", "operators", "mappings", "baselines"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"numeric-regime space lacks {sorted(missing)}")
    cases = []
    cells = []
    seen = set()
    lengths = [(value, "quick") for value in spec["lengths"]["quick"]]
    lengths += [(value, "full") for value in spec["lengths"]["full"]]
    for operator in spec["operators"]:
        for contract in spec["contracts"]:
            if operator["id"] not in contract["operators"]:
                continue
            coverage = _baseline_coverage(spec, operator["id"], contract["precision"])
            for log_n, length_tier in lengths:
                if not operator.get("min_logN", 1) <= log_n <= operator.get("max_logN", 30):
                    continue
                cell_id = f"{operator['id']}-{contract['id']}-log{log_n}"
                cells.append({
                    "id": cell_id, "operator": operator["id"], "contract": contract["id"],
                    "logN": log_n, "baseline_coverage": coverage,
                })
                mappings = [mapping for mapping in spec["mappings"]
                            if operator["id"] in mapping["operators"]
                            and mapping["min_logN"] <= log_n <= mapping["max_logN"]]
                batch_levels = {}
                for mapping in mappings:
                    for batch, tier in _batch_levels(
                            spec, operator["id"], contract, mapping, log_n, operator["value_lanes"]):
                        old = batch_levels.get(batch)
                        batch_levels[batch] = "quick" if old == "quick" or tier == "quick" else "full"
                for mapping in mappings:
                    for batch, batch_tier in sorted(batch_levels.items()):
                        case_id = f"{cell_id}-{mapping['id']}-b{batch}"
                        if case_id in seen:
                            raise ValueError(f"duplicate case {case_id}")
                        seen.add(case_id)
                        resources, ctas_per_transform, registers, shared_bytes = _resource_estimate(
                            spec, operator["id"], contract, mapping, log_n, batch, operator["value_lanes"])
                        tier = "quick" if length_tier == "quick" and batch_tier == "quick" else "full"
                        case = {
                            "id": case_id, "tier": tier,
                            "group": f"{cell_id}-b{batch}", "runner": operator.get("runner", "butterfly"),
                            "implementation": f"cuButterfly-{mapping['id']}",
                            "operator": operator["id"], "precision": contract["precision"],
                            "accumulation": contract["accumulation"], "logN": log_n, "batch": batch,
                            "storage_bits": contract["storage_bits"], "compute_bits": contract["compute_bits"],
                            "accumulator_bits": contract["accumulator_bits"],
                            "emulated_native": contract["emulated_native"],
                            "coefficient_policy": operator["coefficient_policy"],
                            "arithmetic_pipeline": operator["arithmetic_pipeline"],
                            "mapping_id": mapping["id"], "ctas_per_transform": ctas_per_transform,
                            "estimated_registers_per_thread": registers,
                            "estimated_shared_bytes_per_cta": shared_bytes,
                            "args": _mapping_args(operator["id"], contract, mapping, log_n,
                                                  operator.get("stage_matrix")),
                        }
                        if "modulus_bits" in contract:
                            case["modulus_bits"] = contract["modulus_bits"]
                        working_set = batch * (1 << log_n) * operator["value_lanes"] * (
                            contract["storage_bits"] // 8) * 2
                        case["working_set_bytes"] = working_set
                        case["working_set_over_l2"] = working_set / spec["hardware"]["l2_bytes"]
                        case.update(resources)
                        cases.append(case)
    return {
        "schema_version": 1, "study": "numeric-regime-cliffs-v0.5",
        "hardware": spec["hardware"], "protocols": spec["protocols"],
        "binaries": spec["binaries"], "cells": cells, "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate the numeric-contract and batch-cliff experiment suite.")
    parser.add_argument("--spec", type=pathlib.Path, default=pathlib.Path("config/v100_numeric_regime_space.json"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--check", type=pathlib.Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    suite = expand(json.loads(args.spec.read_text()))
    rendered = json.dumps(suite, indent=2) + "\n"
    if args.check:
        if args.check.read_text() != rendered:
            raise ValueError(f"generated numeric-regime manifest is stale: {args.check}")
        print(f"validated cases={len(suite['cases'])} cells={len(suite['cells'])}")
    elif args.validate_only:
        print(f"validated cases={len(suite['cases'])} cells={len(suite['cells'])}")
    elif args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        raise ValueError("--output is required unless --validate-only or --check is used")


if __name__ == "__main__":
    main()
