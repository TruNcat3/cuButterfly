#!/usr/bin/env python3
import argparse
import csv
import json
import pathlib
import statistics


SEMANTIC_FIELDS = (
    "operator", "precision", "direction", "normalization", "placement", "logN", "N",
    "performance_batch", "element_stride", "batch_stride", "word_bits", "modulus_bits",
    "inverse", "output_order",
)

CONFIG_FIELDS = (
    "backend", "compute_unit", "fft_core", "local_exchange", "cross_twiddle",
)


def read_rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def classify(ratio):
    if ratio >= 1.03:
        return "faster"
    if ratio >= 0.97:
        return "parity"
    return "slower"


def summarize(rows, max_relative_range=0.03):
    if not rows:
        raise ValueError("comprehensive-suite input is empty")
    invalid = [row["suite_case_id"] for row in rows if row.get("preflight_correct") not in ("1", "-1")]
    if invalid:
        raise ValueError(f"correctness preflight failed or is missing: {sorted(set(invalid))}")
    by_case = {}
    for row in rows:
        by_case.setdefault(row["suite_case_id"], []).append(row)
    cases = []
    for case_id, samples in by_case.items():
        times = [float(sample["kernel_ms"]) for sample in samples]
        ordered_times = sorted(times)
        central_times = ordered_times[1:-1] if len(ordered_times) >= 5 else ordered_times
        median_time = statistics.median(times)
        transform_count = int(first_value(samples, "performance_batch", "0"))
        point_count = int(first_value(samples, "N", "0")) * transform_count
        relative_range = (max(times) - min(times)) / median_time
        central_relative_range = (max(central_times) - min(central_times)) / median_time
        if central_relative_range > max_relative_range:
            stability_class = "unstable"
        elif relative_range > max_relative_range:
            stability_class = "stable-with-outlier"
        else:
            stability_class = "stable"
        first = samples[0]
        cases.append({
            "case_id": case_id,
            "group": first["suite_group"],
            "implementation": first["implementation"],
            "reference": int(first["reference"]),
            "trials": len(samples),
            "median_kernel_ms": median_time,
            "min_kernel_ms": min(times),
            "max_kernel_ms": max(times),
            "relative_range": relative_range,
            "central_relative_range": central_relative_range,
            "stability_class": stability_class,
            "correct": int(all(sample.get("preflight_correct") == "1" for sample in samples)),
            **{field: first.get(field, "") for field in SEMANTIC_FIELDS},
            **{field: first.get(field, "") for field in CONFIG_FIELDS},
            "million_transforms_s": transform_count / (median_time * 1000.0),
            "billion_points_s": point_count / (median_time * 1.0e6),
        })
    by_group = {}
    for case in cases:
        by_group.setdefault(case["group"], []).append(case)
    output = []
    for group, group_cases in sorted(by_group.items()):
        references = [case for case in group_cases if case["reference"]]
        if len(references) > 1:
            raise ValueError(f"group {group} has multiple references")
        basis = references[0] if references else min(group_cases, key=lambda case: case["median_kernel_ms"])
        basis_kind = "declared-reference" if references else "group-fastest"
        for case in sorted(group_cases, key=lambda item: item["median_kernel_ms"]):
            ratio = basis["median_kernel_ms"] / case["median_kernel_ms"]
            output.append({
                **case,
                "comparison_basis": basis_kind,
                "basis_implementation": basis["implementation"],
                "basis_kernel_ms": basis["median_kernel_ms"],
                "throughput_vs_basis": ratio,
                "performance_class": classify(ratio),
            })
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for field in ("median_kernel_ms", "min_kernel_ms", "max_kernel_ms", "basis_kernel_ms"):
                formatted[field] = f"{row[field]:.9f}"
            for field in (
                "relative_range", "central_relative_range", "throughput_vs_basis",
                "million_transforms_s", "billion_points_s",
            ):
                formatted[field] = f"{row[field]:.6f}"
            writer.writerow(formatted)


def first_value(rows, field, default=""):
    for row in rows:
        value = row.get(field, "")
        if value != "":
            return value
    return default


def layout_label(row):
    stride = row.get("element_stride") or "1"
    batch_stride = row.get("batch_stride") or "n/a"
    placement = row.get("placement") or "n/a"
    order = row.get("output_order")
    suffix = f", {order}" if order else ""
    return f"{placement}, es={stride}, bs={batch_stride}{suffix}"


def numeric_label(row):
    modulus_bits = row.get("modulus_bits")
    if modulus_bits:
        return f"{row['precision']}/{modulus_bits}-bit"
    return row["precision"]


def semantic_label(row):
    normalization = row.get("normalization")
    if normalization and normalization != "none":
        return f"{row['direction']}, norm={normalization}"
    return row["direction"]


def design_label(row):
    parts = []
    for field in ("backend", "fft_core", "local_exchange", "compute_unit"):
        value = row.get(field)
        if value and value not in parts and value not in ("auto", "scalar", "shared"):
            parts.append(value)
    return " / ".join(parts) if parts else "library"


def write_markdown(path, rows, external_evidence):
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    lines = [
        "# Single-GPU Comprehensive Summary", "",
        "Every row uses the listed transform length and batch. `Mtransform/s` and",
        "`Gpoint/s` are derived from the median resident kernel time. Ratios above one",
        "mean higher throughput than `Basis`; groups without a declared reference use",
        "their fastest internal implementation as the basis.", "",
    ]
    operator_order = ("fft", "ntt", "fwht", "xor-zeta")
    for operator in operator_order:
        operator_rows = [row for row in rows if row["operator"] == operator]
        if not operator_rows:
            continue
        lines += [f"## {operator.upper()} Results", "",
                  "| Numeric | Semantics | logN | N | Batch | Layout | Implementation | Design point | Median ms | Mtransform/s | Gpoint/s | vs basis | Basis | Stability |",
                  "|:--|:--|--:|--:|--:|:--|:--|:--|--:|--:|--:|--:|:--|:--|"]
        for row in sorted(operator_rows, key=lambda item: (int(item["logN"]), item["group"], item["median_kernel_ms"])):
            lines.append(
                f"| {numeric_label(row)} | {semantic_label(row)} | {row['logN']} | "
                f"{int(row['N']):,} | {int(row['performance_batch']):,} | {layout_label(row)} | "
                f"{row['implementation']} | {design_label(row)} | {row['median_kernel_ms']:.6f} | "
                f"{row['million_transforms_s']:.3f} | {row['billion_points_s']:.3f} | "
                f"{row['throughput_vs_basis']:.3f}x | {row['basis_implementation']} | "
                f"{row['stability_class']} |")
        lines.append("")
    lines += ["## External Evidence", ""]
    if external_evidence:
        for item in external_evidence:
            lines.append(
                f"- **{item['operator']} / {item['implementation']}**: {item['status']}; "
                f"{item['reason']}. Existing evidence: `{item['existing_result']}`.")
    else:
        lines.append("- None recorded.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize the single-GPU comprehensive comparison suite.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/v100_comprehensive_suite.json"))
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument("--max-relative-range", type=float, default=0.03)
    parser.add_argument("--require-stable", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(read_rows(path))
    summary = summarize(rows, args.max_relative_range)
    write_csv(args.output, summary)
    if args.markdown:
        manifest = json.loads(args.manifest.read_text())
        write_markdown(args.markdown, summary, manifest.get("external_evidence", []))
    unstable = [row["case_id"] for row in summary if row["stability_class"] == "unstable"]
    if args.require_stable and unstable:
        raise ValueError(f"unstable benchmark cases exceed relative range {args.max_relative_range}: {unstable}")


if __name__ == "__main__":
    main()
