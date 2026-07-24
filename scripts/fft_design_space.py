#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import pathlib
from collections import Counter


REQUIRED_AXES = {
    "log_n", "decomposition", "precision", "direction", "placement",
    "boundary", "reorder", "cross_twiddle", "dimension",
}


def load_space(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported FFT design-space schema")
    missing = REQUIRED_AXES - set(document.get("axes", {}))
    if missing:
        raise ValueError(f"missing FFT design-space axes: {sorted(missing)}")
    dimension = document["axes"]["dimension"]
    for name in ("core", "spatial_scope", "stage_fusion", "temporal_reuse", "threads",
                 "elements_per_thread", "units_per_cta", "exchange", "shared_padding", "pipeline_buffers"):
        if name not in dimension or not dimension[name]:
            raise ValueError(f"missing dimension axis: {name}")
    if not document.get("families"):
        raise ValueError("FFT design space has no implementation families")
    return document


def load_codegen_points(path):
    selection = json.loads(path.read_text())
    expanded = {
        (log_n, threads, ept)
        for family in selection.get("families", {}).values()
        for log_n in family.get("dimension_log_n", [])
        for threads, ept in family.get("thread_ept_pairs", [])
    }
    explicit = {
        tuple(point)
        for family in selection.get("families", {}).values()
        for point in family.get("points", [])
    }
    return expanded | explicit


def classify_online_point(point, hardware, compiled_points=None):
    prefix_n = 1 << point["prefix_log_n"]
    suffix_n = 1 << point["suffix_log_n"]
    reasons = []
    for label, size, threads, ept in (
        ("prefix", prefix_n, point["prefix_threads"], point["prefix_ept"]),
        ("suffix", suffix_n, point["suffix_threads"], point["suffix_ept"]),
    ):
        if threads > hardware["max_threads_per_cta"]:
            reasons.append(f"{label}-threads")
        if ept > size or threads * ept < size or (threads * ept) % size:
            reasons.append(f"{label}-coverage")
        units = threads * ept // size if not reasons or not reasons[-1].startswith(label) else 0
        point[f"{label}_units_per_cta"] = units
        shared = threads * ept * 8
        point[f"{label}_tile_shared_bytes"] = shared
        if shared > hardware["max_shared_bytes_per_cta"]:
            reasons.append(f"{label}-shared")
    if reasons:
        return "hardware-infeasible", ";".join(reasons)
    if point["prefix_log_n"] < 3 or point["suffix_log_n"] < 3:
        return "requires-new-kernel", "mixed-small-dimension-core"
    if point["prefix_log_n"] > 12 or point["suffix_log_n"] > 12:
        return "requires-new-kernel", "large-dimension-layout"
    for label in ("prefix", "suffix"):
        if point[f"{label}_log_n"] >= 11 and point[f"{label}_units_per_cta"] != 1:
            return "requires-new-kernel", f"{label}-direct-single-unit"
    compiled_points = compiled_points or set()
    prefix_key = (point["prefix_log_n"], point["prefix_threads"], point["prefix_ept"])
    suffix_key = (point["suffix_log_n"], point["suffix_threads"], point["suffix_ept"])
    if prefix_key in compiled_points and suffix_key in compiled_points:
        return "compiled", ""
    return "awaiting-codegen", "specialization-not-selected"


def enumerate_online(document, profile, log_ns, thread_values, ept_values, twiddles, compiled_points=None):
    hardware = document["hardware_profiles"][profile]
    for log_n in log_ns:
        for prefix_log_n in range(1, log_n):
            suffix_log_n = log_n - prefix_log_n
            for prefix_threads, suffix_threads, prefix_ept, suffix_ept, twiddle in itertools.product(
                    thread_values, thread_values, ept_values, ept_values, twiddles):
                point = {
                    "family": "cufftdx-online",
                    "logN": log_n,
                    "prefix_log_n": prefix_log_n,
                    "suffix_log_n": suffix_log_n,
                    "prefix_core": "cufftdx-block" if prefix_log_n <= 10 else "cufftdx-direct",
                    "suffix_core": "cufftdx-block" if suffix_log_n <= 10 else "cufftdx-direct",
                    "prefix_spatial_scope": "cta",
                    "suffix_spatial_scope": "cta",
                    "prefix_stage_fusion": "full-dimension",
                    "suffix_stage_fusion": "full-dimension",
                    "prefix_temporal_reuse": "resident-codelet",
                    "suffix_temporal_reuse": "resident-codelet",
                    "prefix_threads": prefix_threads,
                    "suffix_threads": suffix_threads,
                    "prefix_ept": prefix_ept,
                    "suffix_ept": suffix_ept,
                    "cross_twiddle": twiddle,
                    "boundary": "global-scratch",
                    "reorder": "online-tiled" if max(prefix_log_n, suffix_log_n) <= 10 else "direct-strided-or-transpose",
                }
                point["status"], point["reason"] = classify_online_point(point, hardware, compiled_points)
                yield point


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate and enumerate the canonical FFT architecture space.")
    parser.add_argument("--spec", type=pathlib.Path,
                        default=pathlib.Path("config/fft_architecture_space.json"))
    parser.add_argument("--profile", default="v100-sm70")
    parser.add_argument("--codegen-spec", type=pathlib.Path,
                        default=pathlib.Path("config/v100_fft_codegen.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--logNs", nargs="+", type=int, default=(16, 18, 20))
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--ept", nargs="+", type=int)
    parser.add_argument("--twiddles", nargs="+", choices=("table", "recurrence"), default=("table", "recurrence"))
    parser.add_argument("--status", nargs="+", choices=("compiled", "awaiting-codegen", "requires-new-kernel", "hardware-infeasible"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--summary", type=pathlib.Path)
    args = parser.parse_args()

    document = load_space(args.spec)
    if args.profile not in document["hardware_profiles"]:
        raise ValueError(f"unknown hardware profile: {args.profile}")
    if args.validate_only:
        return
    dimension = document["axes"]["dimension"]
    compiled_points = load_codegen_points(args.codegen_spec)
    threads = args.threads or dimension["threads"]
    ept = args.ept or dimension["elements_per_thread"]
    rows = list(enumerate_online(document, args.profile, args.logNs, threads, ept, args.twiddles, compiled_points))
    if args.status:
        rows = [row for row in rows if row["status"] in args.status]
    if not rows:
        raise ValueError("FFT design-space selection is empty")
    if args.output:
        write_csv(args.output, rows)
    counts = Counter(row["status"] for row in rows)
    summary = {
        "schema_version": document["schema_version"],
        "model": document["model"],
        "profile": args.profile,
        "selected_points": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "axes": {"logNs": args.logNs, "threads": threads, "ept": ept, "twiddles": args.twiddles},
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    else:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
