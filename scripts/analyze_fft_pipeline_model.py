#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics


SIGNATURE_FIELDS = (
    "mapping_id", "decomposition_count", "stages_per_decomposition", "segment_threads",
    "segment_ept", "boundary_twiddles", "boundary_layouts", "boundary_residencies",
    "execution_group_stages", "group_threads", "group_ept", "cross_twiddle", "direct_boundary",
)


def rank_values(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def pearson(left, right):
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def evaluate(rows, top_k_values=(1, 3, 10)):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    evaluations = []
    for group, candidates in sorted(groups.items()):
        ranked = sorted(candidates, key=lambda row: float(row["static_score"]))
        measured = sorted(candidates, key=lambda row: float(row["median_kernel_ms"]))
        winner = measured[0]
        winner_rank = next(index for index, row in enumerate(ranked, 1)
                           if row["candidate_id"] == winner["candidate_id"])
        result = {
            "group": group,
            "logN": int(winner["logN"]),
            "batch": int(winner["batch"]),
            "candidate_count": len(candidates),
            "measured_winner": winner["candidate_id"],
            "measured_winner_ms": float(winner["median_kernel_ms"]),
            "winner_static_rank": winner_rank,
            "spearman": pearson(
                rank_values([float(row["static_score"]) for row in candidates]),
                rank_values([float(row["median_kernel_ms"]) for row in candidates])),
        }
        for top_k in top_k_values:
            limit = min(top_k, len(ranked))
            selected = min(ranked[:limit], key=lambda row: float(row["median_kernel_ms"]))
            result[f"top{top_k}_candidate"] = selected["candidate_id"]
            result[f"top{top_k}_regret"] = (
                float(selected["median_kernel_ms"]) / float(winner["median_kernel_ms"]))
            result[f"top{top_k}_recall"] = int(winner_rank <= limit)
        evaluations.append(result)

    metrics = {"evaluated_shapes": len(evaluations)}
    for top_k in top_k_values:
        regrets = [row[f"top{top_k}_regret"] for row in evaluations]
        metrics[f"top{top_k}_exact_recall"] = statistics.mean(
            row[f"top{top_k}_recall"] for row in evaluations)
        metrics[f"top{top_k}_geomean_regret"] = math.exp(
            statistics.mean(math.log(value) for value in regrets))
        metrics[f"top{top_k}_worst_regret"] = max(regrets)
        metrics[f"top{top_k}_candidate_fraction"] = statistics.mean(
            min(top_k, row["candidate_count"]) / row["candidate_count"] for row in evaluations)
    metrics["mean_spearman"] = statistics.mean(row["spearman"] for row in evaluations)
    return evaluations, metrics


def candidate_signature(row):
    return tuple(row.get(field, "") for field in SIGNATURE_FIELDS)


def interpolate_latency(samples, batch):
    points = sorted((math.log2(int(row["batch"])), math.log(float(row["median_kernel_ms"])))
                    for row in samples)
    if len(points) == 1:
        return math.exp(points[0][1])
    target = math.log2(batch)
    if target <= points[0][0]:
        left, right = points[:2]
    elif target >= points[-1][0]:
        left, right = points[-2:]
    else:
        left, right = next((left, right) for left, right in zip(points, points[1:])
                           if left[0] <= target <= right[0])
    prediction = left[1] + (right[1] - left[1]) * (target - left[0]) / (right[0] - left[0])
    return math.exp(prediction)


def evaluate_calibrated(rows, top_k_values=(1, 3, 10)):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    evaluations = []
    for group, held_out in sorted(groups.items()):
        log_n = held_out[0]["logN"]
        batch = int(held_out[0]["batch"])
        training = [row for row in rows if row["logN"] == log_n and row["group"] != group]
        by_signature = {}
        for row in training:
            by_signature.setdefault(candidate_signature(row), []).append(row)
        predicted = [
            (interpolate_latency(by_signature[candidate_signature(row)], batch), row)
            for row in held_out if candidate_signature(row) in by_signature
        ]
        predicted.sort(key=lambda item: item[0])
        winner = min(held_out, key=lambda row: float(row["median_kernel_ms"]))
        winner_rank = next((index for index, (_, row) in enumerate(predicted, 1)
                            if row["candidate_id"] == winner["candidate_id"]), None)
        result = {
            "group": group, "logN": int(log_n), "batch": batch,
            "candidate_count": len(held_out), "calibrated_coverage": len(predicted),
            "measured_winner": winner["candidate_id"],
            "measured_winner_ms": float(winner["median_kernel_ms"]),
            "winner_calibrated_rank": winner_rank or 0,
        }
        for top_k in top_k_values:
            limit = min(top_k, len(predicted))
            selected = min((row for _, row in predicted[:limit]),
                           key=lambda row: float(row["median_kernel_ms"]))
            result[f"top{top_k}_regret"] = (
                float(selected["median_kernel_ms"]) / float(winner["median_kernel_ms"]))
            result[f"top{top_k}_recall"] = int(winner_rank is not None and winner_rank <= limit)
        evaluations.append(result)

    metrics = {
        "evaluated_shapes": len(evaluations),
        "mean_candidate_coverage": statistics.mean(
            row["calibrated_coverage"] / row["candidate_count"] for row in evaluations),
    }
    for top_k in top_k_values:
        regrets = [row[f"top{top_k}_regret"] for row in evaluations]
        metrics[f"top{top_k}_exact_recall"] = statistics.mean(
            row[f"top{top_k}_recall"] for row in evaluations)
        metrics[f"top{top_k}_geomean_regret"] = math.exp(
            statistics.mean(math.log(value) for value in regrets))
        metrics[f"top{top_k}_worst_regret"] = max(regrets)
        metrics[f"top{top_k}_candidate_fraction"] = statistics.mean(
            min(top_k, row["calibrated_coverage"]) / row["candidate_count"] for row in evaluations)
    return evaluations, metrics


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, metrics, calibrated_rows, calibrated_metrics):
    lines = [
        "# V100 FFT Static-Model Evaluation", "",
        "Candidates are ordered only by the generator's static resource score; measured",
        "latency is then used to compute selection regret. Lower regret is better.", "",
        "| logN | Batch | Candidates | Winner static rank | Top-1 regret | Top-3 regret | Top-10 regret | Spearman |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['logN']} | {row['batch']} | {row['candidate_count']} | "
            f"{row['winner_static_rank']} | {row['top1_regret']:.4f}x | "
            f"{row['top3_regret']:.4f}x | {row['top10_regret']:.4f}x | "
            f"{row['spearman']:.3f} |")
    lines += ["", "## Aggregate", "",
              "| Budget | Exact recall | Geomean regret | Worst regret | Candidate fraction |",
              "|:--|--:|--:|--:|--:|"]
    for top_k in (1, 3, 10):
        lines.append(
            f"| Top-{top_k} | {metrics[f'top{top_k}_exact_recall']:.1%} | "
            f"{metrics[f'top{top_k}_geomean_regret']:.4f}x | "
            f"{metrics[f'top{top_k}_worst_regret']:.4f}x | "
            f"{metrics[f'top{top_k}_candidate_fraction']:.1%} |")
    lines += ["", f"Mean per-shape Spearman correlation: {metrics['mean_spearman']:.3f}.", ""]
    lines += [
        "## Leave-One-Batch-Out Calibration", "",
        "For each held-out shape, the calibrated selector sees only the other batches",
        "at the same length and interpolates or extrapolates log-latency for matching",
        "physical mappings. It never consumes timing from the target shape.", "",
        "| logN | Batch | Covered | Winner rank | Top-1 regret | Top-3 regret | Top-10 regret |",
        "|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in calibrated_rows:
        lines.append(
            f"| {row['logN']} | {row['batch']} | {row['calibrated_coverage']}/{row['candidate_count']} | "
            f"{row['winner_calibrated_rank']} | {row['top1_regret']:.4f}x | "
            f"{row['top3_regret']:.4f}x | {row['top10_regret']:.4f}x |")
    lines += ["", "| Budget | Exact recall | Geomean regret | Worst regret | Candidate fraction |",
              "|:--|--:|--:|--:|--:|"]
    for top_k in (1, 3, 10):
        lines.append(
            f"| Top-{top_k} | {calibrated_metrics[f'top{top_k}_exact_recall']:.1%} | "
            f"{calibrated_metrics[f'top{top_k}_geomean_regret']:.4f}x | "
            f"{calibrated_metrics[f'top{top_k}_worst_regret']:.4f}x | "
            f"{calibrated_metrics[f'top{top_k}_candidate_fraction']:.1%} |")
    lines += ["", f"Mean mapping coverage: {calibrated_metrics['mean_candidate_coverage']:.1%}.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate the FFT pipeline static model against measurements.")
    parser.add_argument("summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--calibrated-output", type=pathlib.Path)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--markdown", type=pathlib.Path)
    args = parser.parse_args()
    with args.summary.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError("FFT pipeline summary is empty")
    evaluations, metrics = evaluate(rows)
    calibrated_evaluations, calibrated_metrics = evaluate_calibrated(rows)
    write_csv(args.output, evaluations)
    if args.calibrated_output:
        write_csv(args.calibrated_output, calibrated_evaluations)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps({"static": metrics, "calibrated": calibrated_metrics}, indent=2) + "\n")
    if args.markdown:
        write_markdown(args.markdown, evaluations, metrics, calibrated_evaluations, calibrated_metrics)


if __name__ == "__main__":
    main()
