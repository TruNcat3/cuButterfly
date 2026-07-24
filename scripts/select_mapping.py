#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics


SEMANTIC_FIELDS = (
    "operator", "precision", "direction", "normalization", "placement", "logN",
    "element_stride", "output_order", "modulus_bits",
)
EXTERNAL_BACKENDS = {"cufft", "vkfft"}


def parse_args_list(values):
    parsed = {}
    index = 0
    while index < len(values):
        token = values[index]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            if index + 1 < len(values) and not values[index + 1].startswith("--"):
                parsed[key] = values[index + 1]
                index += 2
            else:
                parsed[key] = "1"
                index += 1
        else:
            index += 1
    return parsed


def load_cases(manifest_path):
    document = json.loads(manifest_path.read_text())
    return {
        case["id"]: {**case, "parameters": parse_args_list(case.get("args", []))}
        for case in document["cases"]
    }


def load_rows(summary_path, cases, internal_only=True):
    with summary_path.open() as source:
        rows = list(csv.DictReader(source))
    output = []
    for row in rows:
        case = cases.get(row["case_id"])
        if case is None or row.get("correct") != "1":
            continue
        parameters = case["parameters"]
        backend = parameters.get("backend", row.get("backend", ""))
        if internal_only and (backend in EXTERNAL_BACKENDS or not row["implementation"].startswith(("cuButterfly", "cuNTT"))):
            continue
        output.append({
            **row,
            "batch": int(row["performance_batch"]),
            "logN_int": int(row["logN"]),
            "kernel_ms": float(row["median_kernel_ms"]),
            "throughput": float(row["billion_points_s"]),
            "parameters": parameters,
            "mapping_family": backend,
            "processing_unit": processing_unit(row, parameters),
        })
    return output


def processing_unit(row, parameters):
    if parameters.get("fft_core"):
        return parameters["fft_core"]
    if parameters.get("local_exchange") == "warp-register":
        return "warp-register"
    return parameters.get("compute_unit", row.get("compute_unit", "auto"))


def semantic_key(row):
    return tuple(row.get(field, "") for field in SEMANTIC_FIELDS)


def shape_key(row):
    return semantic_key(row) + (row["batch"],)


def interpolate_log_latency(samples, target_batch):
    points = sorted((math.log2(row["batch"]), math.log(row["kernel_ms"])) for row in samples)
    if not points:
        return None, "measurement-required"
    if len(points) == 1:
        return math.exp(points[0][1]), "single-point"
    target = math.log2(target_batch)
    lower = [point for point in points if point[0] < target]
    upper = [point for point in points if point[0] > target]
    if lower and upper:
        left, right = lower[-1], upper[0]
        confidence = "interpolated"
        anchor = left
    elif lower:
        left, right = points[-2], points[-1]
        confidence = "extrapolated-high"
        anchor = right
    else:
        left, right = points[0], points[1]
        confidence = "extrapolated-low"
        anchor = left
    slope = (right[1] - left[1]) / (right[0] - left[0])
    # Latency cannot improve indefinitely with more work, and a batch doubling
    # cannot require more than twice the work in this fixed-contract model.
    slope = min(1.0, max(0.0, slope))
    predicted = anchor[1] + slope * (target - anchor[0])
    return math.exp(predicted), confidence


def hardware_context(row, hardware):
    points = (1 << row["logN_int"]) * row["batch"]
    word_bytes = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8,
                  "uint32": 4, "uint64": 8}.get(row["precision"], 4)
    working_set = points * word_bytes * 2
    sm_count = int(hardware["sm_count"])
    parameters = row["parameters"]
    threads = int(parameters.get("tile_threads", parameters.get("prefix_threads", 0)) or 0)
    return {
        "total_points": points,
        "working_set_bytes": working_set,
        "working_set_over_l2": working_set / int(hardware["l2_bytes"]),
        "batch_per_sm": row["batch"] / sm_count,
        "configured_threads": threads,
        "legal": threads <= int(hardware["threads_per_sm"]),
    }


def explanation(row, context, confidence):
    factors = [confidence]
    if context["batch_per_sm"] < 1.0:
        factors.append("batch concurrency below one transform per SM")
    else:
        factors.append("batch concurrency covers all SMs")
    if context["working_set_over_l2"] <= 1.0:
        factors.append("estimated two-buffer working set fits L2")
    else:
        factors.append(f"estimated working set is {context['working_set_over_l2']:.1f}x L2")
    if row["mapping_family"] in ("online-reorder", "compact-stage"):
        factors.append("boundary permutation is fused with the required store")
    if row["processing_unit"] == "warp-register":
        factors.append("local exchange is register-resident")
    return "; ".join(factors)


def rank_shape(target_rows, all_rows, hardware, excluded_shape=None):
    ranked = []
    for target in target_rows:
        training = [row for row in all_rows
                    if semantic_key(row) == semantic_key(target)
                    and row["implementation"] == target["implementation"]
                    and (excluded_shape is None or shape_key(row) != excluded_shape)]
        predicted_ms, confidence = interpolate_log_latency(training, target["batch"])
        context = hardware_context(target, hardware)
        if not context["legal"] or predicted_ms is None:
            continue
        ranked.append({
            "implementation": target["implementation"],
            "mapping_family": target["mapping_family"],
            "processing_unit": target["processing_unit"],
            "predicted_kernel_ms": predicted_ms,
            "confidence": confidence,
            "reason": explanation(target, context, confidence),
            "parameters": target["parameters"],
        })
    ranked.sort(key=lambda row: row["predicted_kernel_ms"])
    return ranked


def evaluate(rows, hardware, top_k=3):
    groups = {}
    for row in rows:
        groups.setdefault(shape_key(row), []).append(row)
    records = []
    for key, candidates in sorted(groups.items()):
        if len(candidates) < 2:
            continue
        ranked = rank_shape(candidates, rows, hardware, excluded_shape=key)
        if len(ranked) != len(candidates):
            continue
        actual = sorted(candidates, key=lambda row: row["kernel_ms"])
        winner = actual[0]["implementation"]
        predicted = ranked[0]["implementation"]
        chosen_ms = next(row["kernel_ms"] for row in candidates if row["implementation"] == predicted)
        regret = chosen_ms / actual[0]["kernel_ms"]
        top_names = [row["implementation"] for row in ranked[:top_k]]
        top3_names = [row["implementation"] for row in ranked[:3]]
        records.append({
            "operator": candidates[0]["operator"],
            "precision": candidates[0]["precision"],
            "logN": candidates[0]["logN_int"],
            "batch": candidates[0]["batch"],
            "candidates": len(candidates),
            "measured_winner": winner,
            "predicted_winner": predicted,
            "top1_correct": int(predicted == winner),
            "topk_contains_winner": int(winner in top_names),
            "top3_contains_winner": int(winner in top3_names),
            "regret": regret,
            "confidence": ranked[0]["confidence"],
        })
    if not records:
        raise ValueError("no multi-candidate shapes can be evaluated")
    regrets = [row["regret"] for row in records]
    metrics = {
        "method": "leave-one-complete-shape-out log-latency interpolation",
        "scope": "V100 calibrated internal candidates",
        "evaluated_shapes": len(records),
        "top_k": top_k,
        "top1_accuracy": statistics.mean(row["top1_correct"] for row in records),
        "topk_recall": statistics.mean(row["topk_contains_winner"] for row in records),
        "top3_recall": statistics.mean(row["top3_contains_winner"] for row in records),
        "geomean_regret": math.exp(statistics.mean(math.log(value) for value in regrets)),
        "worst_regret": max(regrets),
        "mean_candidates": statistics.mean(row["candidates"] for row in records),
        "mean_returned": statistics.mean(min(top_k, row["candidates"]) for row in records),
    }
    metrics["search_space_reduction"] = 1.0 - metrics["mean_returned"] / metrics["mean_candidates"]
    metrics["top1_search_space_reduction"] = 1.0 - 1.0 / metrics["mean_candidates"]
    return records, metrics


def write_evaluation(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in records:
            formatted = dict(row)
            formatted["regret"] = f"{row['regret']:.6f}"
            writer.writerow(formatted)


def main():
    parser = argparse.ArgumentParser(description="Select a calibrated cuButterfly mapping for a covered V100 workload.")
    parser.add_argument("--summary", type=pathlib.Path, default=pathlib.Path("results/v100_scaling_full_summary.csv"))
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_scaling_suite.json"))
    parser.add_argument("--hardware", type=pathlib.Path, default=pathlib.Path("configs/hardware/v100_sxm2_16gb.json"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--include-external", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluation-output", type=pathlib.Path)
    parser.add_argument("--metrics-output", type=pathlib.Path)
    parser.add_argument("--operator")
    parser.add_argument("--precision")
    parser.add_argument("--logN", type=int)
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    cases = load_cases(args.manifest)
    rows = load_rows(args.summary, cases, not args.include_external)
    hardware = json.loads(args.hardware.read_text())
    if args.evaluate:
        records, metrics = evaluate(rows, hardware, args.top_k)
        if args.evaluation_output:
            write_evaluation(args.evaluation_output, records)
        rendered = json.dumps(metrics, indent=2) + "\n"
        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(rendered)
        print(rendered, end="")
        return
    required = (args.operator, args.precision, args.logN, args.batch)
    if any(value is None for value in required):
        raise ValueError("selection requires --operator, --precision, --logN, and --batch")
    candidates = [row for row in rows if row["operator"] == args.operator
                  and row["precision"] == args.precision and row["logN_int"] == args.logN
                  and row["batch"] == args.batch]
    if not candidates:
        raise ValueError("workload is outside the calibrated manifest; measurement is required")
    ranked = rank_shape(candidates, rows, hardware, excluded_shape=shape_key(candidates[0]))
    print(json.dumps({"hardware": hardware["name"], "candidates": ranked[:args.top_k]}, indent=2))


if __name__ == "__main__":
    main()
