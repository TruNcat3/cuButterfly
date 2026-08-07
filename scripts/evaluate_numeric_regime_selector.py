#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv, shape_key


def batch_region(row):
    waves = float(row["grid_waves"])
    if waves < 0.5:
        return "launch-limited"
    if waves < 2.0:
        return "crossover"
    return "saturated"


def holdout_value(row, dimension):
    if dimension == "numeric-contract":
        return f"{row['precision']}:{row.get('accumulation', 'native')}"
    if dimension == "arithmetic-policy":
        return f"{row['arithmetic_pipeline']}:{row['coefficient_policy']}"
    if dimension == "batch-region":
        return batch_region(row)
    if dimension == "operator":
        return row["operator"]
    raise ValueError(f"unknown holdout dimension {dimension}")


def distance(left, right):
    numeric = (
        abs(int(left["logN"]) - int(right["logN"])) / 12.0 +
        abs(math.log2(int(left["batch"])) - math.log2(int(right["batch"]))) / 16.0 +
        abs(int(left["storage_bits"]) - int(right["storage_bits"])) / 64.0 +
        abs(int(left["compute_bits"]) - int(right["compute_bits"])) / 64.0 +
        abs(int(left["accumulator_bits"]) - int(right["accumulator_bits"])) / 64.0 +
        abs(float(left["working_set_over_l2"]) - float(right["working_set_over_l2"])) /
        max(1.0, float(left["working_set_over_l2"]), float(right["working_set_over_l2"]))
    )
    categorical = 0.0
    for field, penalty in (("operator", 1.0), ("arithmetic_pipeline", 0.75),
                           ("coefficient_policy", 0.5), ("limiting_resource", 0.25)):
        categorical += penalty * (left.get(field) != right.get(field))
    return numeric + categorical


def predict(candidate, training, neighbors=3):
    compatible = [row for row in training if row["mapping_id"] == candidate["mapping_id"]]
    nearest = sorted(((distance(candidate, row), row) for row in compatible), key=lambda item: item[0])[:neighbors]
    if not nearest:
        return None
    weighted = []
    for metric, row in nearest:
        weight = 1.0 / max(metric, 1.0e-6)
        # Normalize latency to points*stages before transporting it to the target shape.
        source_work = int(row["batch"]) * (1 << int(row["logN"])) * int(row["logN"])
        target_work = int(candidate["batch"]) * (1 << int(candidate["logN"])) * int(candidate["logN"])
        weighted.append((weight, row["median_ms"] * target_work / source_work))
    return sum(weight * value for weight, value in weighted) / sum(weight for weight, _ in weighted)


def evaluate(rows, dimensions=("numeric-contract", "arithmetic-policy", "batch-region", "operator"), top_k=3):
    records = []
    for dimension in dimensions:
        values = sorted({holdout_value(row, dimension) for row in rows})
        for value in values:
            training = [row for row in rows if holdout_value(row, dimension) != value]
            held = [row for row in rows if holdout_value(row, dimension) == value]
            shapes = {}
            for row in held:
                shapes.setdefault(shape_key(row), []).append(row)
            for key, candidates in shapes.items():
                if len(candidates) < 2:
                    continue
                predictions = [(predict(candidate, training), candidate) for candidate in candidates]
                if any(predicted is None for predicted, _ in predictions):
                    continue
                predictions.sort(key=lambda item: item[0])
                actual = sorted(candidates, key=lambda item: item["median_ms"])
                winner = actual[0]["mapping_id"]
                chosen = predictions[0][1]
                top_names = [candidate["mapping_id"] for _, candidate in predictions[:top_k]]
                records.append({
                    "holdout_dimension": dimension, "holdout_value": value,
                    "operator": key[0], "precision": key[1], "accumulation": key[2],
                    "logN": key[3], "batch": key[4], "measured_winner": winner,
                    "predicted_winner": chosen["mapping_id"],
                    "top1_correct": int(chosen["mapping_id"] == winner),
                    "topk_contains_winner": int(winner in top_names),
                    "regret": chosen["median_ms"] / actual[0]["median_ms"],
                })
    if not records:
        raise ValueError("no complete held-out regimes had comparable candidates")
    regrets = [record["regret"] for record in records]
    metrics = {
        "method": "complete-regime holdout with descriptor nearest neighbors",
        "evaluated_shapes": len(records), "top_k": top_k,
        "top1_accuracy": statistics.mean(record["top1_correct"] for record in records),
        "topk_recall": statistics.mean(record["topk_contains_winner"] for record in records),
        "geomean_regret": math.exp(statistics.mean(math.log(value) for value in regrets)),
        "worst_regret": max(regrets),
        "evaluated_by_holdout": {
            dimension: sum(record["holdout_dimension"] == dimension for record in records)
            for dimension in dimensions
        },
    }
    thresholds = {"topk_recall": 0.95, "geomean_regret": 1.02, "worst_regret": 1.10}
    metrics["promotion_thresholds"] = thresholds
    complete_coverage = all(metrics["evaluated_by_holdout"][dimension] > 0 for dimension in dimensions)
    metrics["complete_holdout_coverage"] = complete_coverage
    metrics["runtime_selector_status"] = (
        "promotable" if complete_coverage and metrics["topk_recall"] >= thresholds["topk_recall"]
        and metrics["geomean_regret"] <= thresholds["geomean_regret"]
        and metrics["worst_regret"] <= thresholds["worst_regret"]
        else "measurement-required")
    return records, metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate mapping selection on complete held-out regimes.")
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--records", type=pathlib.Path, required=True)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = aggregate_samples(read_csv(args.raw), manifest)
    records, metrics = evaluate(rows, top_k=args.top_k)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
