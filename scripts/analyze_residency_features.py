#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib

from residency_features import derive_residency_features


IDENTITY_FIELDS = ("operator", "precision", "core", "coefficient_policy")


def main():
    parser = argparse.ArgumentParser(description="Derive portable GPU residency/cliff features for generated kernels.")
    parser.add_argument("--hardware", required=True, type=pathlib.Path)
    parser.add_argument("--resources", required=True, type=pathlib.Path)
    parser.add_argument("--target-points", type=int, default=1 << 22)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()
    hardware = json.loads(args.hardware.read_text())
    with args.resources.open() as source:
        resources = list(csv.DictReader(source))

    resources.sort(key=lambda row: (tuple(row[field] for field in IDENTITY_FIELDS), int(row["logN"])))
    previous_by_family = {}
    records = []
    for row in resources:
        family = tuple(row[field] for field in IDENTITY_FIELDS)
        log_n = int(row["logN"])
        threads = int(row["threads"])
        batch = int(row.get("batch") or max(1, args.target_points // (1 << log_n)))
        features = derive_residency_features(
            hardware,
            threads,
            int(row["registers_per_thread"]),
            int(row["shared_bytes"]),
            batch * int(row.get("ctas_per_transform") or 1),
            temporal_state_words_per_thread=(1 << log_n) // threads,
            previous=previous_by_family.get(family),
        )
        previous_by_family[family] = features
        records.append({**row, "batch": batch, **features})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
