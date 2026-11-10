#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize HybridDataflow benchmark CSV files")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with path.open(newline="") as handle:
            record = next(csv.DictReader(handle))
        record.setdefault("role_stages", "0")
        record.setdefault("target_ctas_per_sm", "0")
        record.setdefault("token_interleave", "1")
        record["case"] = path.stem
        rows.append(record)
    rows.sort(key=lambda row: (int(row["word_bits"]), int(row["logN"]), int(row["batch"]), row["case"]))

    fields = ["case", "backend", "word_bits", "logN", "batch", "stage_space", "role_stages", "target_ctas_per_sm", "data_space", "data_time", "token_interleave",
              "pipeline_buffers", "dataflow_layout", "dataflow_state", "kernel_ms", "kernel_ntt_s", "correct"]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    lines = ["# HybridDataflow benchmark", "", "| case | bits | logN | batch | backend | Us | Ur | Rb | Ud | Td | Ti | kernel ms | correct |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['case']} | {row['word_bits']} | {row['logN']} | {row['batch']} | "
                     f"{row['backend']} | {row['stage_space']} | {row['role_stages']} | {row['target_ctas_per_sm']} | "
                     f"{row['data_space']} | {row['data_time']} | {row['token_interleave']} | "
                     f"{float(row['kernel_ms']):.6f} | {row['correct']} |")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["word_bits"], row["logN"], row["batch"])].append(row)
    lines += ["", "## Best generated point versus baseline", "",
              "| bits | logN | batch | baseline | baseline ms | best Us | best Ur | best Rb | best Td | dataflow ms | baseline/dataflow |",
              "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in sorted(groups, key=lambda item: tuple(map(int, item))):
        group = groups[key]
        dataflow = [row for row in group if row["backend"] == "hybrid-dataflow"]
        baselines = [row for row in group if row["backend"] not in ("hybrid-dataflow", "stage-pipeline")]
        if not dataflow or not baselines:
            continue
        best = min(dataflow, key=lambda row: float(row["kernel_ms"]))
        baseline = min(baselines, key=lambda row: float(row["kernel_ms"]))
        ratio = float(baseline["kernel_ms"]) / float(best["kernel_ms"])
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {baseline['backend']} | "
                     f"{float(baseline['kernel_ms']):.6f} | {best['stage_space']} | {best['role_stages']} | "
                     f"{best['target_ctas_per_sm']} | {best['data_time']} | "
                     f"{float(best['kernel_ms']):.6f} | {ratio:.3f}x |")
    args.markdown.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
