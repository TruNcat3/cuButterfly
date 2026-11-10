#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import statistics

from analyze_numeric_regime_cliffs import aggregate_samples, read_csv, shape_key


def role(row):
    name = row["implementation"]
    if "-base-" in name:
        return "base"
    if "-searched-" in name:
        return "searched"
    if name in ("cuFFT", "Dao-AILab-FHT") or name.startswith("GPU-NTT"):
        return "library"
    return "other"


def classify(ratio):
    if ratio >= 1.03:
        return "faster"
    if ratio >= 0.97:
        return "parity"
    return "slower"


def build_main(current, archived):
    rows = list(current)
    present = {(row["group"], role(row)) for row in rows}
    for row in archived:
        if row["operator"] == "ntt" and role(row) == "library" and (row["group"], "library") not in present:
            rows.append({**row, "evidence_source": "archived-matching-protocol"})
    for row in rows:
        row.setdefault("evidence_source", "current-randomized-full-protocol")
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], {})[role(row)] = row
    output = []
    for group, entries in sorted(groups.items()):
        if not {"library", "base", "searched"}.issubset(entries):
            continue
        library, base, searched = entries["library"], entries["base"], entries["searched"]
        library_ms = float(library["median_kernel_ms"])
        base_ms = float(base["median_kernel_ms"])
        searched_ms = float(searched["median_kernel_ms"])
        versus_library = library_ms / searched_ms
        output.append({
            "group": group, "operator": searched["operator"], "precision": searched["precision"],
            "logN": int(searched["logN"]), "N": int(searched["N"]), "batch": int(searched["batch"]),
            "output_order": searched.get("output_order", "natural"),
            "library": library["implementation"], "library_ms": library_ms,
            "base": base["implementation"], "base_ms": base_ms,
            "searched": searched["implementation"], "searched_ms": searched_ms,
            "searched_speedup_vs_base": base_ms / searched_ms,
            "searched_throughput_vs_library": versus_library,
            "searched_class_vs_library": classify(versus_library),
            "library_stability": library["stability_class"],
            "base_stability": base["stability_class"],
            "searched_stability": searched["stability_class"],
            "library_evidence_source": library["evidence_source"],
        })
    order = {"fft": 0, "fwht": 1, "ntt": 2}
    output.sort(key=lambda row: (order.get(row["operator"], 99), row["logN"], row["batch"]))
    return output


def base_candidate(candidates):
    preferences = ("temporal-r2-t128", "online-r2-t256", "ntt-hybrid-r2-t256")
    by_name = {candidate["mapping_id"]: candidate for candidate in candidates}
    for mapping in preferences:
        if mapping in by_name:
            return by_name[mapping]
    return next((candidate for candidate in candidates if "r2" in candidate["mapping_id"]), None)


def build_internal(rows):
    groups = {}
    for row in rows:
        groups.setdefault(shape_key(row), []).append(row)
    records = []
    for key, candidates in sorted(groups.items()):
        base = base_candidate(candidates)
        if base is None:
            continue
        searched = min(candidates, key=lambda item: item["median_ms"])
        records.append({
            "operator": key[0], "precision": key[1], "accumulation": key[2],
            "logN": key[3], "batch": key[4], "base_mapping": base["mapping_id"],
            "base_ms": base["median_ms"], "searched_mapping": searched["mapping_id"],
            "searched_ms": searched["median_ms"],
            "searched_speedup_vs_base": base["median_ms"] / searched["median_ms"],
        })
    return records


def aggregate_internal(records):
    groups = {}
    for row in records:
        groups.setdefault((row["operator"], row["precision"], row["accumulation"]), []).append(row)
    output = []
    for key, values in sorted(groups.items()):
        speedups = [row["searched_speedup_vs_base"] for row in values]
        output.append({
            "operator": key[0], "precision": key[1], "accumulation": key[2],
            "shapes": len(values),
            "geomean_search_speedup": math.exp(statistics.mean(math.log(value) for value in speedups)),
            "median_search_speedup": statistics.median(speedups),
            "max_search_speedup": max(speedups),
        })
    return output


def main_metrics(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["operator"], []).append(row)
    output = {}
    for operator, values in sorted(groups.items()):
        library_ratios = [row["searched_throughput_vs_library"] for row in values]
        base_ratios = [row["searched_speedup_vs_base"] for row in values]
        output[operator] = {
            "shapes": len(values),
            "geomean_searched_vs_library": math.exp(statistics.mean(math.log(value) for value in library_ratios)),
            "geomean_searched_vs_base": math.exp(statistics.mean(math.log(value) for value in base_ratios)),
            "faster": sum(row["searched_class_vs_library"] == "faster" for row in values),
            "parity": sum(row["searched_class_vs_library"] == "parity" for row in values),
            "slower": sum(row["searched_class_vs_library"] == "slower" for row in values),
        }
    all_library = [row["searched_throughput_vs_library"] for row in rows]
    all_base = [row["searched_speedup_vs_base"] for row in rows]
    return {
        "shapes": len(rows), "by_operator": output,
        "overall_geomean_searched_vs_library": math.exp(statistics.mean(math.log(value) for value in all_library)),
        "overall_geomean_searched_vs_base": math.exp(statistics.mean(math.log(value) for value in all_base)),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, metrics, internal_summary):
    lines = [
        "# V100 Three-Way Library Comparison", "",
        "All local rows use 1000 warmups, 100 repetitions, five randomized process trials, and correctness checks.",
        "GPU-NTT rows are imported from the archived matching-protocol run on the same V100; they were not interleaved with this refresh.",
        "Ratios above one mean the searched cuButterfly/cuNTT configuration has higher throughput.", "",
        "| Operator | Numeric | logN | Batch | High-performance library | Library ms | Base ms | Searched ms | Search/base | Search/library | Result | Stability |",
        "|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|",
    ]
    for row in rows:
        numeric = row["precision"]
        lines.append(
            f"| {row['operator']} | {numeric} | {row['logN']} | {row['batch']:,} | {row['library']} | "
            f"{row['library_ms']:.6f} | {row['base_ms']:.6f} | {row['searched_ms']:.6f} | "
            f"{row['searched_speedup_vs_base']:.3f}x | {row['searched_throughput_vs_library']:.3f}x | "
            f"{row['searched_class_vs_library']} | {row['library_stability']} / "
            f"{row['base_stability']} / {row['searched_stability']} |")
    lines += ["", "## Aggregate", "", "| Operator | Shapes | Search/base geomean | Search/library geomean | Faster | Parity | Slower |",
              "|:--|--:|--:|--:|--:|--:|--:|"]
    for operator, item in metrics["by_operator"].items():
        lines.append(f"| {operator} | {item['shapes']} | {item['geomean_searched_vs_base']:.3f}x | "
                     f"{item['geomean_searched_vs_library']:.3f}x | {item['faster']} | {item['parity']} | {item['slower']} |")
    lines += ["", "## Extended Internal Coverage", "",
              "These full-protocol adaptive shapes quantify search gain where no exact external-library row is joined.",
              "They are not external superiority claims.", "",
              "| Operator | Numeric | Accumulation | Shapes | Search/base geomean | Median | Maximum |",
              "|:--|:--|:--|--:|--:|--:|--:|"]
    for row in internal_summary:
        lines.append(f"| {row['operator']} | {row['precision']} | {row['accumulation']} | {row['shapes']} | "
                     f"{row['geomean_search_speedup']:.3f}x | {row['median_search_speedup']:.3f}x | "
                     f"{row['max_search_speedup']:.3f}x |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create library/base/searched cuButterfly comparison tables.")
    parser.add_argument("current", type=pathlib.Path)
    parser.add_argument("--archived-baselines", type=pathlib.Path, required=True)
    parser.add_argument("--internal-raw", type=pathlib.Path, required=True)
    parser.add_argument("--internal-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--internal-output", type=pathlib.Path, required=True)
    parser.add_argument("--metrics", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    args = parser.parse_args()
    current = read_csv(args.current)
    archived = read_csv(args.archived_baselines)
    main_rows = build_main(current, archived)
    internal_manifest = json.loads(args.internal_manifest.read_text())
    internal_samples = aggregate_samples(read_csv(args.internal_raw), internal_manifest)
    internal = build_internal(internal_samples)
    internal_summary = aggregate_internal(internal)
    metrics = main_metrics(main_rows)
    write_csv(args.output, main_rows)
    write_csv(args.internal_output, internal)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    write_markdown(args.report, main_rows, metrics, internal_summary)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
