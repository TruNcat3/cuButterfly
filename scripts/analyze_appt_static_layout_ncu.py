#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


LABEL = re.compile(r"(static|natural)_fw(8|16|32)_u(32|64)_b(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    rows = {row["label"]: row for row in csv.DictReader(args.summary.open())}
    records = []
    for label, row in rows.items():
        match = LABEL.fullmatch(label)
        if not match:
            continue
        contract, fragment, bits, batch = match.groups()
        baseline = rows[f"v06_u{bits}_b{batch}"]
        stores = float(row["global_store_sectors"])
        base_stores = float(baseline["global_store_sectors"])
        records.append({
            "bits": int(bits), "batch": int(batch), "contract": contract,
            "fragment": int(fragment), "time_us": float(row["time_us"]),
            "store_ratio": stores / base_stores,
            "load_ratio": float(row["global_load_sectors"]) /
                          float(baseline["global_load_sectors"]),
            "dram_pct": float(row["dram_peak_pct"]),
            "registers": float(row["registers_per_thread"]),
            "local_sectors": float(row["local_load_sectors"]) +
                             float(row["local_store_sectors"]),
        })
    lines = ["# APPT Static Layout NCU", "",
             "V100 base-clock replay; sector ratios use the matched v0.6 10+10 kernel.", "",
             "| bits | batch | contract | fragment | us | load/v0.6 | store/v0.6 | DRAM peak | regs | local sectors |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(records, key=lambda item: (item["bits"], item["batch"], item["contract"], item["fragment"])):
        lines.append(f"| {row['bits']} | {row['batch']} | {row['contract']} | {row['fragment']} | {row['time_us']:.1f} | {row['load_ratio']:.3f} | {row['store_ratio']:.3f} | {row['dram_pct']:.1f}% | {row['registers']:.0f} | {row['local_sectors']:.0f} |")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
