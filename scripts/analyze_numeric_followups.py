#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv


def evaluate_followups(rows, manifest, minimum_margin=0.02):
    index = {}
    for row in rows:
        key = (row["operator"], row["precision"], row.get("accumulation", "native"),
               int(row["logN"]), int(row["batch"]), row["mapping_id"])
        index[key] = row
    records = []
    for event in manifest.get("source_events", []):
        base = (event["operator"], event["precision"], event["accumulation"],
                int(event["logN"]), int(event["batch"]))
        before = index.get(base + (event["from_mapping"],))
        after = index.get(base + (event["to_mapping"],))
        if before is None or after is None:
            status = "missing"
            full_winner = ""
            margin = ""
            separated = ""
        else:
            ordered = sorted((before, after), key=lambda row: row["median_ms"])
            winner, loser = ordered
            margin = loser["median_ms"] / winner["median_ms"] - 1.0
            separated = int(winner["trial_max_ms"] < loser["trial_min_ms"])
            full_winner = winner["mapping_id"]
            confirmed = margin >= minimum_margin and bool(separated)
            if confirmed and full_winner == event["to_mapping"]:
                status = "confirmed-same-direction"
            elif confirmed:
                status = "confirmed-reversed"
            else:
                status = "not-confirmed"
        records.append({
            "event_id": event["id"], "axis": event["axis"], "operator": event["operator"],
            "precision": event["precision"], "accumulation": event["accumulation"],
            "logN": event["logN"], "batch": event["batch"],
            "quick_from_mapping": event["from_mapping"], "quick_to_mapping": event["to_mapping"],
            "quick_relative_margin": event["relative_margin"], "full_winner": full_winner,
            "full_relative_margin": margin, "trial_ranges_non_overlapping": separated,
            "status": status,
        })
    return records


def summary(records):
    counts = {}
    by_operator = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        operator_counts = by_operator.setdefault(record["operator"], {})
        operator_counts[record["status"]] = operator_counts.get(record["status"], 0) + 1
    return {"events": len(records), "status_counts": counts, "by_operator": by_operator}


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_markdown(path, report):
    lines = ["# V100 Numeric-Regime Confirmed Follow-Ups", "", "| Status | Events |", "|:--|--:|"]
    lines += [f"| {name} | {count} |" for name, count in sorted(report["status_counts"].items())]
    lines += ["", "A full-protocol event requires at least a 2% median gap and non-overlapping five-trial ranges.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Check quick crossover directions under focused full timing.")
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-margin", type=float, default=0.02)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = aggregate_samples(read_csv(args.raw), manifest)
    records = evaluate_followups(rows, manifest, args.minimum_margin)
    if not records:
        raise ValueError("manifest has no source crossover events")
    report = summary(records)
    write_csv(args.output, records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
