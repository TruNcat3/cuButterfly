#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv, shape_key
from build_numeric_piecewise_selector import measured_points


def analyze(rows, manifest, minimum_margin=0.05):
    points = {shape_key(point): point for point in measured_points(rows, evidence="full")}
    records = []
    for source in manifest.get("source_shapes", []):
        key = (source["operator"], source["precision"], source.get("accumulation", "native"),
               int(source["logN"]), int(source["batch"]))
        point = points.get(key)
        if point is None:
            status = "missing"
            winner = ""
            margin = ""
            separated = ""
        else:
            winner = point["winner"]
            margin = point["winner_margin"]
            separated = int(point["trial_ranges_non_overlapping"])
            stable = margin >= minimum_margin and bool(separated)
            if not stable:
                status = "near-tie"
            elif winner == source["quick_winner"]:
                status = "stable-same-direction"
            else:
                status = "stable-reversed"
        records.append({
            "operator": source["operator"], "precision": source["precision"],
            "accumulation": source.get("accumulation", "native"),
            "logN": int(source["logN"]), "batch": int(source["batch"]),
            "quick_winner": source["quick_winner"], "quick_margin": source["quick_margin"],
            "full_winner": winner, "full_margin": margin,
            "trial_ranges_non_overlapping": separated, "status": status,
        })
    return records


def summarize(records):
    counts = {}
    by_operator = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        operator = by_operator.setdefault(record["operator"], {})
        operator[record["status"]] = operator.get(record["status"], 0) + 1
    return {"shapes": len(records), "status_counts": counts, "by_operator": by_operator}


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_report(path, report):
    lines = ["# V100 Numeric Selector Coverage Expansion", "",
             "Weak, non-crossover quick anchors were repeated with the full five-trial protocol.", "",
             "| Result | Shapes |", "|:--|--:|"]
    lines += [f"| {name} | {count} |" for name, count in sorted(report["status_counts"].items())]
    lines += ["", "A stable full anchor requires at least a 5% median gap and non-overlapping trial ranges.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Classify full timing used to expand selector coverage.")
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--summary", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--minimum-margin", type=float, default=0.05)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = aggregate_samples(read_csv(args.raw), manifest)
    records = analyze(rows, manifest, args.minimum_margin)
    if not records:
        raise ValueError("manifest contains no coverage source shapes")
    report = summarize(records)
    write_csv(args.output, records)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n")
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
