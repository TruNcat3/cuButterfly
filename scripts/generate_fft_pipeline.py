#!/usr/bin/env python3
import argparse
import json
import math
import pathlib

from fft_design_space import classify_online_point, load_codegen_points, load_space


def load_pipeline(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 2:
        raise ValueError("unsupported FFT pipeline schema")
    for field in ("target", "processing_units", "mapping_spaces", "selection", "protocol"):
        if field not in document:
            raise ValueError(f"FFT pipeline lacks {field}")
    ids = [unit["id"] for unit in document["processing_units"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate processing-unit id")
    for mapping in document["mapping_spaces"]:
        decomposition = mapping.get("decomposition_space", {})
        for field in ("decomposition_count", "stages_per_decomposition", "compiled_decomposition_counts"):
            if field not in decomposition:
                raise ValueError(f"mapping {mapping['id']} lacks decomposition_space.{field}")
    return document


def ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator


def stage_partitions(log_n, decomposition_count, minimum, maximum):
    def visit(remaining, decompositions_left, prefix):
        if decompositions_left == 0:
            if remaining == 0:
                yield tuple(prefix)
            return
        lower = max(minimum, remaining - maximum * (decompositions_left - 1))
        upper = min(maximum, remaining - minimum * (decompositions_left - 1))
        for stages in range(lower, upper + 1):
            yield from visit(remaining - stages, decompositions_left - 1, [*prefix, stages])

    yield from visit(log_n, decomposition_count, [])


def mapping_stage_partitions(mapping, compiled_only=False):
    decomposition = mapping["decomposition_space"]
    count_range = decomposition["decomposition_count"]
    stage_range = decomposition["stages_per_decomposition"]
    compiled_counts = set(decomposition["compiled_decomposition_counts"])
    for decomposition_count in range(count_range["min"], count_range["max"] + 1):
        if compiled_only and decomposition_count not in compiled_counts:
            continue
        for stages in stage_partitions(mapping["logN"], decomposition_count,
                                       stage_range["min"], stage_range["max"]):
            yield stages


def enumerate_decomposition_topologies(document):
    topologies = []
    for mapping in document["mapping_spaces"]:
        compiled_counts = set(mapping["decomposition_space"]["compiled_decomposition_counts"])
        for stages in mapping_stage_partitions(mapping):
            decomposition_count = len(stages)
            compiled = decomposition_count in compiled_counts
            topologies.append({
                "id": f"{mapping['id']}_d{decomposition_count}_" + "x".join(map(str, stages)),
                "mapping_id": mapping["id"], "logN": mapping["logN"],
                "decomposition_count": decomposition_count, "stages_per_decomposition": list(stages),
                "boundary_count": decomposition_count - 1,
                "status": "compiled-runtime" if compiled else "requires-multi-pass-runtime",
                "reason": "" if compiled else "decomposition-count-not-implemented",
            })
    return topologies


def pass_resources(dimension_log_n, other_log_n, threads, ept, batch, hardware, unit):
    size = 1 << dimension_log_n
    units = threads * ept // size
    blocks = ceil_div(batch * (1 << other_log_n), units)
    shared = threads * ept * 8
    register_spec = unit.get("estimated_registers_per_thread", {})
    registers = int(register_spec.get(f"logN{dimension_log_n}", register_spec.get("default", 0)))
    limits = [hardware["max_ctas_per_sm"], hardware["max_threads_per_sm"] // threads,
              hardware["max_shared_bytes_per_cta"] // shared]
    if registers:
        limits.append(hardware["registers_per_sm"] // (registers * threads))
    resident_ctas = max(1, min(limits))
    waves = blocks / (hardware["sm_count"] * resident_ctas)
    return {
        "units_per_cta": units,
        "blocks": blocks,
        "threads": threads,
        "ept": ept,
        "tile_shared_bytes": shared,
        "estimated_registers_per_thread": registers,
        "estimated_resident_ctas_per_sm": resident_ctas,
        "estimated_waves_per_sm": waves,
        "issued_thread_slots": blocks * threads,
    }


def score_candidate(point, hardware, prefix_unit, suffix_unit, selection):
    prefix = pass_resources(point["prefix_log_n"], point["suffix_log_n"], point["prefix_threads"],
                            point["prefix_ept"], point["batch"], hardware, prefix_unit)
    suffix = pass_resources(point["suffix_log_n"], point["prefix_log_n"], point["suffix_threads"],
                            point["suffix_ept"], point["batch"], hardware, suffix_unit)
    weighted_prefix = selection["prefix_work_weight"] * prefix["issued_thread_slots"]
    weighted_suffix = selection["suffix_work_weight"] * suffix["issued_thread_slots"]
    work = weighted_prefix + weighted_suffix
    imbalance = max(weighted_prefix, weighted_suffix) / min(weighted_prefix, weighted_suffix)
    underfill = max(0.0, 1.0 - min(prefix["estimated_waves_per_sm"], suffix["estimated_waves_per_sm"]))
    score = work * (1.0 + selection["imbalance_penalty"] * math.log2(imbalance)
                    + selection["underfill_penalty"] * underfill)
    if point["cross_twiddle"] == "recurrence":
        score *= 0.98
    point["prefix_resources"] = prefix
    point["suffix_resources"] = suffix
    point["estimated_work_imbalance"] = imbalance
    point["static_score"] = score


def compiled_segment_options(stage_log_n, unit, codegen_points, hardware, batch, total_log_n):
    size = 1 << stage_log_n
    options = []
    for threads in unit["threads"]:
        for ept in unit["elements_per_thread"]:
            if (threads > hardware["max_threads_per_cta"] or ept > size or
                    threads * ept < size or (threads * ept) % size or
                    threads * ept * 8 > hardware["max_shared_bytes_per_cta"]):
                continue
            if stage_log_n >= 11:
                compiled = threads * ept == size
            else:
                compiled = (stage_log_n, threads, ept) in codegen_points
            if not compiled:
                continue
            resources = pass_resources(stage_log_n, total_log_n - stage_log_n, threads, ept,
                                       batch, hardware, unit)
            options.append({"processing_unit": unit["id"], "core": unit["core"],
                            "exchange": unit["exchange"], "threads": threads, "ept": ept,
                            "resources": resources})
    return sorted(options, key=lambda item: (item["resources"]["issued_thread_slots"],
                                              -item["resources"]["estimated_waves_per_sm"],
                                              item["threads"], item["ept"]))


def score_multisegment(point, selection):
    weighted = []
    for segment, mapping in enumerate(point["segment_mappings"]):
        weight = selection["prefix_work_weight"] if segment == 0 else selection["suffix_work_weight"]
        weighted.append(weight * mapping["resources"]["issued_thread_slots"])
    work = sum(weighted)
    imbalance = max(weighted) / min(weighted)
    underfill = max(0.0, 1.0 - min(mapping["resources"]["estimated_waves_per_sm"]
                                   for mapping in point["segment_mappings"]))
    point["estimated_work_imbalance"] = imbalance
    point["static_score"] = work * (1.0 + selection["imbalance_penalty"] * math.log2(imbalance)
                                     + selection["underfill_penalty"] * underfill)


def multisegment_runtime_args(point):
    threads = ",".join(str(mapping["threads"]) for mapping in point["segment_mappings"])
    ept = ",".join(str(mapping["ept"]) for mapping in point["segment_mappings"])
    twiddles = ",".join(boundary["cross_twiddle"] for boundary in point["boundaries"])
    layouts = ",".join(boundary["layout"] for boundary in point["boundaries"])
    return [
        "--operator", "fft", "--precision", point["precision"], "--backend", "online-reorder",
        "--fft-core", "cufftdx-block", "--stage-partition",
        ",".join(map(str, point["stages_per_decomposition"])),
        "--segment-threads", threads, "--segment-ept", ept,
        "--boundary-twiddle", twiddles, "--boundary-layout", layouts,
        "--placement", point["placement"], "--normalization", point["normalization"],
    ]


def multisegment_candidate_id(point):
    stages = "x".join(map(str, point["stages_per_decomposition"]))
    threads = "x".join(str(mapping["threads"]) for mapping in point["segment_mappings"])
    ept = "x".join(str(mapping["ept"]) for mapping in point["segment_mappings"])
    twiddles = "".join("r" if boundary["cross_twiddle"] == "recurrence" else "t"
                       for boundary in point["boundaries"])
    return f"{point['mapping_id']}_b{point['batch']}_d{point['decomposition_count']}_{stages}_t{threads}_e{ept}_{twiddles}"


def runtime_args(point):
    return [
        "--operator", "fft", "--precision", point["precision"], "--backend", "online-reorder",
        "--fft-core", "cufftdx-block", "--stage-partition",
        ",".join(map(str, point["stages_per_decomposition"])),
        "--reorder-columns", "1", "--cross-twiddle", point["cross_twiddle"],
        "--direct-boundary", point["direct_boundary"],
        "--prefix-threads", str(point["prefix_threads"]), "--suffix-threads", str(point["suffix_threads"]),
        "--prefix-ept", str(point["prefix_ept"]), "--suffix-ept", str(point["suffix_ept"]),
        "--placement", point["placement"], "--normalization", point["normalization"],
    ]


def candidate_id(point):
    twiddle = "rec" if point["cross_twiddle"] == "recurrence" else "tbl"
    boundary_suffixes = {"direct-strided": "", "tiled-transpose": "_tx",
                         "prefix-tiled-transpose": "_ptx"}
    boundary = boundary_suffixes[point["direct_boundary"]]
    return (f"{point['mapping_id']}_b{point['batch']}_p{point['prefix_log_n']}s{point['suffix_log_n']}_"
            f"t{point['prefix_threads']}x{point['suffix_threads']}_e{point['prefix_ept']}x{point['suffix_ept']}_{twiddle}{boundary}")


def matches_required(point, required):
    if len(required["stage_partition"]) != 2 or len(required["segment_mappings"]) != 2:
        raise ValueError("current runtime incumbent must describe exactly two decomposition segments")
    prefix, suffix = required["segment_mappings"]
    fields = {
        "stages_per_decomposition": required["stage_partition"],
        "prefix_threads": prefix["threads"], "suffix_threads": suffix["threads"],
        "prefix_ept": prefix["ept"], "suffix_ept": suffix["ept"],
        "cross_twiddle": required["cross_twiddle"],
        "direct_boundary": required.get("direct_boundary", "direct-strided"),
    }
    return all(point[field] == value for field, value in fields.items())


def enumerate_pipeline(document, architecture, codegen_points):
    hardware = architecture["hardware_profiles"][document["target"]]
    compiled_units = [unit for unit in document["processing_units"] if unit["status"] == "compiled"]
    planned_units = [unit for unit in document["processing_units"] if unit["status"] != "compiled"]

    def find_unit(dimension_log_n):
        matches = [unit for unit in compiled_units
                   if unit["dimension_log_n"]["min"] <= dimension_log_n <= unit["dimension_log_n"]["max"]]
        if len(matches) > 1:
            raise ValueError(f"multiple compiled processing units cover logN={dimension_log_n}")
        return matches[0] if matches else None

    runnable = []
    backlog = []
    for mapping in document["mapping_spaces"]:
        for batch in mapping["batches"]:
            shape = []
            planned_shape = []
            for stages in mapping_stage_partitions(mapping, compiled_only=True):
                if len(stages) != 2:
                    units = [find_unit(stage) for stage in stages]
                    if any(unit is None for unit in units):
                        planned_shape.append({
                            "mapping_id": mapping["id"], "batch": batch,
                            "decomposition_count": len(stages), "stages_per_decomposition": list(stages),
                            "status": "requires-new-kernel", "reason": "segment-processing-unit",
                        })
                        continue
                    options = [compiled_segment_options(stage, unit, codegen_points, hardware,
                                                        batch, mapping["logN"])
                               for stage, unit in zip(stages, units)]
                    if any(not segment_options for segment_options in options):
                        planned_shape.append({
                            "mapping_id": mapping["id"], "batch": batch,
                            "decomposition_count": len(stages), "stages_per_decomposition": list(stages),
                            "status": "awaiting-codegen", "reason": "segment-specialization-not-selected",
                        })
                        continue
                    mapping_variants = min(3, max(len(segment_options) for segment_options in options))
                    twiddle_variants = [
                        ["table"] * (len(stages) - 1),
                        ["recurrence"] * (len(stages) - 1),
                        ["recurrence" if boundary % 2 == 0 else "table"
                         for boundary in range(len(stages) - 1)],
                    ]
                    for variant in range(mapping_variants):
                        segment_mappings = [segment_options[min(variant, len(segment_options) - 1)]
                                            for segment_options in options]
                        for twiddles in twiddle_variants:
                            boundaries = [{"cross_twiddle": twiddle, "layout": "direct-strided"}
                                          for twiddle in twiddles]
                            point = {
                                **{key: value for key, value in mapping.items()
                                   if key not in ("batches", "decomposition_space", "required_candidates")},
                                "mapping_id": mapping["id"], "batch": batch,
                                "decomposition_count": len(stages),
                                "stages_per_decomposition": list(stages),
                                "segment_mappings": segment_mappings, "boundaries": boundaries,
                                "processing_unit": "+".join(item["id"] for item in units),
                                "prefix_log_n": stages[0], "suffix_log_n": stages[-1],
                                "prefix_threads": segment_mappings[0]["threads"],
                                "suffix_threads": segment_mappings[-1]["threads"],
                                "prefix_ept": segment_mappings[0]["ept"],
                                "suffix_ept": segment_mappings[-1]["ept"],
                                "cross_twiddle": (twiddles[0] if len(set(twiddles)) == 1 else "mixed"),
                                "direct_boundary": "direct-strided", "status": "compiled", "reason": "",
                            }
                            score_multisegment(point, document["selection"])
                            point["id"] = multisegment_candidate_id(point)
                            point["args"] = multisegment_runtime_args(point)
                            shape.append(point)
                    continue
                prefix_log_n, suffix_log_n = stages
                prefix_unit = find_unit(prefix_log_n)
                suffix_unit = find_unit(suffix_log_n)
                if prefix_unit is None or suffix_unit is None:
                    planned = {
                        **{key: value for key, value in mapping.items() if key not in ("batches", "decomposition_space", "required_candidates")},
                        "mapping_id": mapping["id"], "batch": batch, "prefix_log_n": prefix_log_n,
                        "suffix_log_n": suffix_log_n, "status": "requires-new-kernel",
                        "reason": "large-dimension-processing-unit",
                        "candidate_processing_units": [item["id"] for item in planned_units],
                    }
                    planned_shape.append(planned)
                    continue
                twiddles = sorted(set(prefix_unit["cross_twiddle"]) & set(suffix_unit["cross_twiddle"]))
                boundaries = ["direct-strided"]
                if suffix_log_n >= 11 and "tiled-transpose" in suffix_unit.get("direct_boundaries", []):
                    boundaries.append("tiled-transpose")
                if prefix_log_n >= 11 and "prefix-tiled-transpose" in prefix_unit.get("direct_boundaries", []):
                    boundaries.append("prefix-tiled-transpose")
                for twiddle in twiddles:
                    for boundary in boundaries:
                        for prefix_threads in prefix_unit["threads"]:
                            for suffix_threads in suffix_unit["threads"]:
                                for prefix_ept in prefix_unit["elements_per_thread"]:
                                    for suffix_ept in suffix_unit["elements_per_thread"]:
                                        point = {
                                            **{key: value for key, value in mapping.items() if key not in ("batches", "decomposition_space", "required_candidates")},
                                            "mapping_id": mapping["id"], "batch": batch,
                                            "decomposition_count": 2,
                                            "stages_per_decomposition": [prefix_log_n, suffix_log_n],
                                            "processing_unit": f"{prefix_unit['id']}+{suffix_unit['id']}",
                                            "prefix_processing_unit": prefix_unit["id"],
                                            "suffix_processing_unit": suffix_unit["id"], "prefix_log_n": prefix_log_n,
                                            "suffix_log_n": suffix_log_n, "prefix_threads": prefix_threads,
                                            "suffix_threads": suffix_threads, "prefix_ept": prefix_ept,
                                            "suffix_ept": suffix_ept, "cross_twiddle": twiddle,
                                            "direct_boundary": boundary,
                                            "segment_mappings": [
                                                {"processing_unit": prefix_unit["id"], "core": prefix_unit["core"],
                                                 "exchange": prefix_unit["exchange"], "threads": prefix_threads,
                                                 "ept": prefix_ept},
                                                {"processing_unit": suffix_unit["id"], "core": suffix_unit["core"],
                                                 "exchange": suffix_unit["exchange"], "threads": suffix_threads,
                                                 "ept": suffix_ept},
                                            ],
                                            "boundaries": [{"cross_twiddle": twiddle, "layout": boundary}],
                                        }
                                        status, reason = classify_online_point(point, hardware, codegen_points)
                                        if status != "compiled":
                                            continue
                                        point["status"] = status
                                        point["reason"] = reason
                                        score_candidate(point, hardware, prefix_unit, suffix_unit, document["selection"])
                                        point["id"] = candidate_id(point)
                                        point["args"] = runtime_args(point)
                                        shape.append(point)
            # Preserve factorization/twiddle diversity before filling by score.
            selected = []
            for required in mapping.get("required_candidates", []):
                match = next((point for point in shape if matches_required(point, required)), None)
                if match is None:
                    raise ValueError(f"required candidate is not compiled for {mapping['id']} batch={batch}: {required}")
                selected.append(match)
            for decomposition_count in sorted({point["decomposition_count"] for point in shape}):
                selected.append(min((point for point in shape
                                     if point["decomposition_count"] == decomposition_count),
                                    key=lambda item: item["static_score"]))
            for stages in sorted({tuple(point["stages_per_decomposition"]) for point in shape
                                  if point["decomposition_count"] == 2}):
                selected.append(min((point for point in shape
                                     if tuple(point["stages_per_decomposition"]) == stages),
                                    key=lambda item: item["static_score"]))
            d2_boundary_keys = {
                (tuple(point["stages_per_decomposition"]), point["direct_boundary"])
                for point in shape if point["decomposition_count"] == 2
            }
            for stages, boundary in sorted(d2_boundary_keys):
                selected.append(min((point for point in shape
                                     if tuple(point["stages_per_decomposition"]) == stages and
                                     point["direct_boundary"] == boundary),
                                    key=lambda item: item["static_score"]))
            multi_mapping_keys = {
                (point["decomposition_count"],
                 tuple((item["threads"], item["ept"]) for item in point["segment_mappings"]))
                for point in shape if point["decomposition_count"] > 2
            }
            for decomposition_count, mapping_key in sorted(multi_mapping_keys):
                selected.append(min(
                    (point for point in shape
                     if point["decomposition_count"] == decomposition_count and
                     tuple((item["threads"], item["ept"]) for item in point["segment_mappings"]) == mapping_key),
                    key=lambda item: item["static_score"]))
            diversity = {}
            for point in sorted(shape, key=lambda item: item["static_score"]):
                key = (point["decomposition_count"], tuple(point["stages_per_decomposition"]),
                       point["cross_twiddle"], point["direct_boundary"])
                if key not in diversity:
                    diversity[key] = point
            selected = list({point["id"]: point for point in selected}.values())
            selected_ids = {point["id"] for point in selected}
            selected.extend(point for point in diversity.values() if point["id"] not in selected_ids)
            limit = int(mapping.get("max_runnable_candidates_per_shape",
                                    document["selection"]["max_runnable_candidates_per_shape"]))
            selected_ids = {point["id"] for point in selected}
            selected.extend(point for point in sorted(shape, key=lambda item: item["static_score"])
                            if point["id"] not in selected_ids)
            selected = selected[:limit]
            for point in selected:
                point["selection_pool_size"] = len(shape)
                point["selection_limit"] = limit
            runnable.extend(sorted(selected, key=lambda item: item["static_score"]))
            backlog.extend(planned_shape)
    return runnable, backlog


def main():
    parser = argparse.ArgumentParser(description="Generate processing-unit, mapping, and benchmark layers for FFT selection.")
    parser.add_argument("--pipeline", type=pathlib.Path, default=pathlib.Path("config/v100_fft_pipeline.json"))
    parser.add_argument("--architecture", type=pathlib.Path, default=pathlib.Path("config/fft_architecture_space.json"))
    parser.add_argument("--codegen", type=pathlib.Path, default=pathlib.Path("config/v100_fft_codegen.json"))
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=pathlib.Path)
    output_group.add_argument("--check", type=pathlib.Path)
    args = parser.parse_args()
    document = load_pipeline(args.pipeline)
    architecture = load_space(args.architecture)
    codegen = load_codegen_points(args.codegen)
    runnable, backlog = enumerate_pipeline(document, architecture, codegen)
    topologies = enumerate_decomposition_topologies(document)
    output = {
        "schema_version": 2, "target": document["target"], "protocol": document["protocol"],
        "processing_units": document["processing_units"], "runnable_candidates": runnable,
        "backlog_candidates": backlog, "decomposition_topologies": topologies,
    }
    rendered = json.dumps(output, indent=2) + "\n"
    if args.check:
        if not args.check.exists() or args.check.read_text() != rendered:
            raise ValueError(f"generated FFT candidate manifest is stale: {args.check}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(f"generated runnable={len(runnable)} backlog={len(backlog)}")


if __name__ == "__main__":
    main()
