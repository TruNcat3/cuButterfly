#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv, shape_key


def measured_points(rows, evidence="quick"):
    groups = {}
    for row in rows:
        groups.setdefault(shape_key(row), []).append(row)
    points = []
    for key, candidates in groups.items():
        ordered = sorted(candidates, key=lambda item: item["median_ms"])
        winner, runner_up = ordered[:2]
        points.append({
            **winner,
            "winner": winner["mapping_id"],
            "runner_up": runner_up["mapping_id"],
            "winner_margin": runner_up["median_ms"] / winner["median_ms"] - 1.0,
            "trial_ranges_non_overlapping": winner["trial_max_ms"] < runner_up["trial_min_ms"],
            "evidence": evidence,
            "candidates": ordered,
        })
    return points


def overlay_points(base, calibration):
    combined = {shape_key(point): point for point in base}
    combined.update({shape_key(point): point for point in calibration})
    return list(combined.values())


def apply_followup_boundaries(points, records):
    unstable = {
        (record["operator"], record["precision"], record.get("accumulation", "native"),
         int(record["logN"]), int(record["batch"]))
        for record in records
        if record.get("status") in ("confirmed-reversed", "not-confirmed")
    }
    for point in points:
        point["evidence_boundary"] = shape_key(point) in unstable
    return points


def contract_key(row):
    return (row["operator"], row["precision"], row.get("accumulation", "native"), int(row["logN"]))


def is_stable(point, minimum_anchor_margin, minimum_full_margin=0.05):
    required_margin = minimum_full_margin if point.get("evidence") == "full" else minimum_anchor_margin
    return (point["winner_margin"] >= required_margin
            and point["trial_ranges_non_overlapping"]
            and not point.get("evidence_boundary", False))


def build_model(points, minimum_anchor_margin=0.15, minimum_full_margin=0.05):
    groups = {}
    for point in points:
        groups.setdefault(contract_key(point), []).append(point)

    contracts = []
    for key, contract_points in sorted(groups.items()):
        contract_points.sort(key=lambda item: int(item["batch"]))
        anchors = [{
            "batch": int(point["batch"]),
            "mapping": point["winner"],
            "runner_up": point["runner_up"],
            "winner_margin": point["winner_margin"],
            "evidence": point.get("evidence", "quick"),
            "evidence_boundary": point.get("evidence_boundary", False),
            "stable": is_stable(point, minimum_anchor_margin, minimum_full_margin),
        } for point in contract_points]
        intervals = []
        for left, right in zip(anchors, anchors[1:]):
            stable = left["stable"] and right["stable"] and left["mapping"] == right["mapping"]
            candidates = sorted({left["mapping"], left["runner_up"],
                                 right["mapping"], right["runner_up"]})
            intervals.append({
                "batch_min": left["batch"], "batch_max": right["batch"],
                "decision": "auto-select" if stable else "measurement-required",
                "mapping": left["mapping"] if stable else "",
                "candidates": candidates,
                "reason": ("same stable winner at both batch anchors" if stable else
                           "winner, margin, or trial-separation boundary"),
            })
        contracts.append({
            "operator": key[0], "precision": key[1], "accumulation": key[2], "logN": key[3],
            "anchors": anchors, "intervals": intervals,
        })
    return {
        "schema_version": 1,
        "method": "piecewise batch interpolation with explicit abstention",
        "minimum_anchor_margin": minimum_anchor_margin,
        "minimum_full_anchor_margin": minimum_full_margin,
        "scope": "exact V100 numeric contract and measured length only",
        "unseen_contract_policy": "measurement-required",
        "out_of_range_policy": "measurement-required",
        "contracts": contracts,
    }


def select(model, operator, precision, accumulation, log_n, batch):
    contract = next((item for item in model["contracts"]
                     if item["operator"] == operator and item["precision"] == precision
                     and item["accumulation"] == accumulation and item["logN"] == log_n), None)
    if contract is None:
        return {"decision": "measurement-required", "mapping": "", "candidates": [],
                "reason": "unseen numeric contract or length"}
    anchor = next((item for item in contract["anchors"] if item["batch"] == batch), None)
    if anchor is not None:
        candidates = sorted({anchor["mapping"], anchor["runner_up"]})
        if anchor["stable"]:
            return {"decision": "auto-select", "mapping": anchor["mapping"],
                    "candidates": candidates, "reason": "stable measured batch anchor"}
        return {"decision": "measurement-required", "mapping": "",
                "candidates": candidates, "reason": "measured batch anchor is a boundary"}
    interval = next((item for item in contract["intervals"]
                     if item["batch_min"] < batch < item["batch_max"]), None)
    if interval is not None:
        return {key: interval[key] for key in ("decision", "mapping", "candidates", "reason")}
    return {"decision": "measurement-required", "mapping": "", "candidates": [],
            "reason": "batch is outside calibrated intervals"}


def evaluate_leave_one_batch_out(points, minimum_anchor_margin=0.15, minimum_full_margin=0.05):
    groups = {}
    for point in points:
        groups.setdefault(contract_key(point), []).append(point)
    for values in groups.values():
        values.sort(key=lambda item: int(item["batch"]))

    records = []
    for target in points:
        peers = [point for point in groups[contract_key(target)] if point is not target
                 and is_stable(point, minimum_anchor_margin, minimum_full_margin)]
        lower = [point for point in peers if int(point["batch"]) < int(target["batch"])]
        upper = [point for point in peers if int(point["batch"]) > int(target["batch"])]
        left = max(lower, key=lambda item: int(item["batch"])) if lower else None
        right = min(upper, key=lambda item: int(item["batch"])) if upper else None
        decision = "measurement-required"
        predicted = ""
        reason = "target is not bracketed by stable anchors"
        regret = ""
        if target.get("evidence_boundary", False):
            reason = "focused full timing marks target as an unstable boundary"
        elif left is not None and right is not None:
            if left["winner"] == right["winner"]:
                predicted = left["winner"]
                chosen = next((candidate for candidate in target["candidates"]
                               if candidate["mapping_id"] == predicted), None)
                if chosen is not None:
                    decision = "auto-select"
                    reason = "same stable winner at both held-out batch anchors"
                    regret = chosen["median_ms"] / target["median_ms"]
                else:
                    predicted = ""
                    reason = "bracketing winner is unavailable at held-out shape"
            else:
                reason = "bracketing anchors select different mappings"
        records.append({
            "operator": target["operator"], "precision": target["precision"],
            "accumulation": target.get("accumulation", "native"),
            "logN": int(target["logN"]), "batch": int(target["batch"]),
            "decision": decision, "measured_winner": target["winner"],
            "predicted_winner": predicted, "regret": regret, "reason": reason,
        })

    selected = [record for record in records if record["decision"] == "auto-select"]
    regrets = [float(record["regret"]) for record in selected]
    if not selected:
        raise ValueError("no held-out batch is bracketed by same-winner stable anchors")
    metrics = {
        "method": "leave-one-batch-out piecewise interpolation",
        "evaluated_shapes": len(records),
        "auto_selected_shapes": len(selected),
        "abstained_shapes": len(records) - len(selected),
        "auto_selection_coverage": len(selected) / len(records),
        "top1_accuracy_when_selected": statistics.mean(
            record["predicted_winner"] == record["measured_winner"] for record in selected),
        "geomean_regret_when_selected": math.exp(statistics.mean(math.log(value) for value in regrets)),
        "worst_regret_when_selected": max(regrets),
        "minimum_anchor_margin": minimum_anchor_margin,
        "minimum_full_anchor_margin": minimum_full_margin,
        "gates": {"geomean_regret": 1.02, "worst_regret": 1.10},
    }
    metrics["calibrated_region_status"] = (
        "validated" if metrics["geomean_regret_when_selected"] <= metrics["gates"]["geomean_regret"]
        and metrics["worst_regret_when_selected"] <= metrics["gates"]["worst_regret"]
        else "measurement-required")
    metrics["global_runtime_status"] = "measurement-required"
    return records, metrics


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_report(path, metrics):
    lines = [
        "# V100 Numeric Piecewise Selector",
        "",
        "The model auto-selects only inside batch intervals bracketed by two stable anchors with the same winner.",
        "Unknown numeric contracts, unmeasured lengths, extrapolation, and boundary intervals require measurement.",
        "",
        "| Metric | Result |",
        "|:--|--:|",
        f"| Evaluated shapes | {metrics['evaluated_shapes']} |",
        f"| Auto-selected leave-one-batch-out shapes | {metrics['auto_selected_shapes']} |",
        f"| Auto-selection coverage | {metrics['auto_selection_coverage']:.2%} |",
        f"| Top-1 accuracy when selected | {metrics['top1_accuracy_when_selected']:.2%} |",
        f"| Geometric-mean regret when selected | {metrics['geomean_regret_when_selected']:.4f}x |",
        f"| Worst regret when selected | {metrics['worst_regret_when_selected']:.4f}x |",
        f"| Calibrated-region status | {metrics['calibrated_region_status']} |",
        f"| Global runtime status | {metrics['global_runtime_status']} |",
        "",
        "Coverage is reported explicitly and is not treated as a global selector claim.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build an abstaining piecewise numeric-regime selector.")
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--records", type=pathlib.Path, required=True)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--followups", type=pathlib.Path,
                        help="Focused follow-up analysis; reversed/unconfirmed points become boundaries.")
    parser.add_argument("--minimum-anchor-margin", type=float, default=0.15)
    parser.add_argument("--minimum-full-anchor-margin", type=float, default=0.05)
    parser.add_argument("--calibration-raw", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--calibration-manifest", type=pathlib.Path, action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = aggregate_samples(read_csv(args.raw), manifest)
    points = measured_points(rows)
    if len(args.calibration_raw) != len(args.calibration_manifest):
        raise ValueError("each --calibration-raw requires one --calibration-manifest")
    for raw_path, manifest_path in zip(args.calibration_raw, args.calibration_manifest):
        calibration_manifest = json.loads(manifest_path.read_text())
        calibration_rows = aggregate_samples(read_csv(raw_path), calibration_manifest)
        points = overlay_points(points, measured_points(calibration_rows, evidence="full"))
    if args.followups:
        points = apply_followup_boundaries(points, read_csv(args.followups))
    if not points:
        raise ValueError("no comparable measured shapes matched the manifest")
    model = build_model(points, args.minimum_anchor_margin, args.minimum_full_anchor_margin)
    records, metrics = evaluate_leave_one_batch_out(
        points, args.minimum_anchor_margin, args.minimum_full_anchor_margin)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=2) + "\n")
    write_csv(args.records, records)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    write_report(args.report, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
