#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def aggregate_samples(rows, manifest):
    cases = {case["id"]: case for case in manifest["cases"]}
    samples = {}
    for row in rows:
        case_id = row.get("suite_case_id", row.get("case_id", ""))
        if case_id not in cases or row.get("preflight_correct", row.get("correct", "1")) not in ("1", "-1"):
            continue
        latency = row.get("kernel_ms", row.get("median_kernel_ms", ""))
        if not latency:
            continue
        samples.setdefault(case_id, []).append(float(latency))
    output = []
    for case_id, values in samples.items():
        case = cases[case_id]
        output.append({
            **case, "case_id": case_id, "median_ms": statistics.median(values),
            "trial_min_ms": min(values), "trial_max_ms": max(values), "trials": len(values),
        })
    return output


def shape_key(row):
    return (row["operator"], row["precision"], row.get("accumulation", "native"),
            int(row["logN"]), int(row["batch"]))


def winners(rows):
    groups = {}
    for row in rows:
        groups.setdefault(shape_key(row), []).append(row)
    result = {}
    for key, candidates in groups.items():
        ordered = sorted(candidates, key=lambda item: item["median_ms"])
        winner = dict(ordered[0])
        winner["runner_up_ms"] = ordered[1]["median_ms"] if len(ordered) > 1 else math.nan
        winner["winner_margin"] = (ordered[1]["median_ms"] / ordered[0]["median_ms"] - 1.0
                                    if len(ordered) > 1 else math.nan)
        result[key] = winner
    return result


def _significance(winner, loser, minimum_margin):
    margin = loser["median_ms"] / winner["median_ms"] - 1.0
    non_overlapping = winner["trial_max_ms"] < loser["trial_min_ms"]
    return margin, non_overlapping, margin >= minimum_margin and non_overlapping


def detect_events(rows, minimum_margin=0.02):
    selected = winners(rows)
    events = []

    by_contract_length = {}
    by_contract_batch = {}
    for key, row in selected.items():
        operator, precision, accumulation, log_n, batch = key
        by_contract_length.setdefault((operator, precision, accumulation, log_n), []).append(row)
        by_contract_batch.setdefault((operator, precision, accumulation, batch), []).append(row)

    for axis, groups, order_field in (
            ("batch", by_contract_length, "batch"), ("length", by_contract_batch, "logN")):
        for group, points in groups.items():
            points.sort(key=lambda item: int(item[order_field]))
            for left, right in zip(points, points[1:]):
                if left["mapping_id"] == right["mapping_id"]:
                    continue
                right_loser = next((row for row in rows if shape_key(row) == shape_key(right)
                                    and row["mapping_id"] == left["mapping_id"]), None)
                if right_loser is None:
                    continue
                margin, non_overlap, confirmed = _significance(right, right_loser, minimum_margin)
                events.append({
                    "event_type": "winner-crossover", "axis": axis,
                    "operator": right["operator"], "precision": right["precision"],
                    "accumulation": right.get("accumulation", "native"),
                    "logN": right["logN"], "batch": right["batch"],
                    "from_mapping": left["mapping_id"], "to_mapping": right["mapping_id"],
                    "relative_margin": margin, "trial_ranges_non_overlapping": int(non_overlap),
                    "classification": "confirmed" if confirmed else "ambiguous",
                    "reason": f"winner changes along {axis}",
                })

    by_mapping = {}
    for row in rows:
        key = (row["operator"], row["precision"], row.get("accumulation", "native"),
               row["mapping_id"], int(row["batch"]))
        by_mapping.setdefault(key, []).append(row)
    for group, points in by_mapping.items():
        points.sort(key=lambda item: int(item["logN"]))
        for left, right in zip(points, points[1:]):
            if int(left["resident_ctas_per_sm"]) != int(right["resident_ctas_per_sm"]):
                events.append({
                    "event_type": "resource-capacity", "axis": "length",
                    "operator": right["operator"], "precision": right["precision"],
                    "accumulation": right.get("accumulation", "native"),
                    "logN": right["logN"], "batch": right["batch"],
                    "from_mapping": left["mapping_id"], "to_mapping": right["mapping_id"],
                    "relative_margin": "", "trial_ranges_non_overlapping": "",
                    "classification": "predicted",
                    "reason": (f"resident CTAs/SM {left['resident_ctas_per_sm']} -> "
                               f"{right['resident_ctas_per_sm']} ({right['limiting_resource']})"),
                })

    for key, points in by_contract_length.items():
        points.sort(key=lambda item: int(item["batch"]))
        for left, right in zip(points, points[1:]):
            for threshold in (0.5, 1.0, 2.0, 4.0):
                if float(left["grid_waves"]) < threshold <= float(right["grid_waves"]):
                    events.append({
                        "event_type": "grid-wave", "axis": "batch", "operator": right["operator"],
                        "precision": right["precision"], "accumulation": right.get("accumulation", "native"),
                        "logN": right["logN"], "batch": right["batch"],
                        "from_mapping": left["mapping_id"], "to_mapping": right["mapping_id"],
                        "relative_margin": "", "trial_ranges_non_overlapping": "",
                        "classification": "predicted", "reason": f"crosses {threshold:g} grid waves",
                    })
            if float(left["working_set_over_l2"]) < 1.0 <= float(right["working_set_over_l2"]):
                events.append({
                    "event_type": "working-set", "axis": "batch", "operator": right["operator"],
                    "precision": right["precision"], "accumulation": right.get("accumulation", "native"),
                    "logN": right["logN"], "batch": right["batch"],
                    "from_mapping": left["mapping_id"], "to_mapping": right["mapping_id"],
                    "relative_margin": "", "trial_ranges_non_overlapping": "",
                    "classification": "predicted", "reason": "estimated two-buffer working set crosses L2",
                })

    numeric_groups = {}
    for key, row in selected.items():
        numeric_groups.setdefault((row["operator"], int(row["logN"]), int(row["batch"])), []).append(row)
    for _, points in numeric_groups.items():
        mapping_names = {point["mapping_id"] for point in points}
        if len(mapping_names) > 1:
            points.sort(key=lambda item: (item["storage_bits"], item["compute_bits"], item["accumulator_bits"]))
            events.append({
                "event_type": "numeric-pipeline", "axis": "numeric-contract",
                "operator": points[0]["operator"], "precision": "multiple", "accumulation": "multiple",
                "logN": points[0]["logN"], "batch": points[0]["batch"],
                "from_mapping": points[0]["mapping_id"], "to_mapping": points[-1]["mapping_id"],
                "relative_margin": "", "trial_ranges_non_overlapping": "",
                "classification": "observed",
                "reason": "best mapping changes while shape and operator remain fixed",
            })
    return events, selected


def regime_document(selected, events):
    event_shapes = {(event["operator"], str(event["logN"]), str(event["batch"])) for event in events}
    regimes = []
    for key, row in sorted(selected.items()):
        operator, precision, accumulation, log_n, batch = key
        regimes.append({
            "operator": operator, "precision": precision, "accumulation": accumulation,
            "logN": log_n, "batch": batch, "selected_mapping": row["mapping_id"],
            "median_ms": row["median_ms"], "winner_margin": row["winner_margin"],
            "limiting_resource": row["limiting_resource"], "grid_waves": row["grid_waves"],
            "working_set_over_l2": row["working_set_over_l2"],
            "boundary_neighborhood": int((operator, str(log_n), str(batch)) in event_shapes),
        })
    return {"schema_version": 1, "method": "numeric-and-resource-boundary classification", "regimes": regimes}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["event_type"]
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, events, regimes):
    counts = {}
    for event in events:
        counts[event["event_type"]] = counts.get(event["event_type"], 0) + 1
    lines = ["# Numeric-Regime Cliff Report", "", f"Measured regimes: {len(regimes['regimes'])}.", "",
             "| Event | Count |", "|:--|--:|"]
    lines += [f"| {name} | {count} |" for name, count in sorted(counts.items())]
    lines += ["", "A winner crossover is confirmed only when the median gap is at least 2% and trial ranges do not overlap.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Detect batch, length, resource, and numeric-contract cliffs.")
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--events", type=pathlib.Path, required=True)
    parser.add_argument("--regimes", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-margin", type=float, default=0.02)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = aggregate_samples(read_csv(args.raw), manifest)
    if not rows:
        raise ValueError("no valid samples matched the manifest")
    events, selected = detect_events(rows, args.minimum_margin)
    regimes = regime_document(selected, events)
    write_csv(args.events, events)
    args.regimes.parent.mkdir(parents=True, exist_ok=True)
    args.regimes.write_text(json.dumps(regimes, indent=2) + "\n")
    write_markdown(args.report, events, regimes)
    print(f"samples={len(rows)} regimes={len(selected)} events={len(events)}")


if __name__ == "__main__":
    main()
