#!/usr/bin/env python3
import argparse
import csv
import pathlib


def read_rows(path):
    with path.open() as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description="Combine NTT, FWHT, FFT, and Boolean-zeta mapping summaries.")
    parser.add_argument("--ntt", required=True, type=pathlib.Path)
    parser.add_argument("--operators", required=True, type=pathlib.Path)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    ntt = read_rows(args.ntt)
    operators = read_rows(args.operators)
    output = []

    ntt_pipeline = [row for row in ntt if row["backend"] == "stage-pipeline" and row["stage_handoff"] == "named-barrier"]
    ntt_best = min(ntt_pipeline, key=lambda row: float(row["median_kernel_ms"]))
    ntt_us1 = next(row for row in ntt_pipeline if row["stage_space"] == "1")
    ntt_baseline = next(row for row in ntt if row["backend"] == "tile256")
    output.append({
        "operator": "ntt64",
        "value_bytes": 8,
        "has_twiddle": 1,
        "bit_reverse_input": 1,
        "best_stage_space": ntt_best["stage_space"],
        "us1_kernel_ms": ntt_us1["median_kernel_ms"],
        "best_pipeline_ms": ntt_best["median_kernel_ms"],
        "pipeline_speedup_vs_us1": ntt_best["speedup_vs_us1"],
        "baseline": "tile256",
        "baseline_ms": ntt_baseline["median_kernel_ms"],
        "pipeline_efficiency_vs_baseline": ntt_best["speedup_vs_tile256"],
    })

    metadata = {
        "fwht": (4, 0, 0, "temporal-tile"),
        "structured-2x2": (4, 1, 0, "temporal-tile"),
        "subset-zeta": (4, 0, 0, "temporal-tile"),
        "superset-zeta": (4, 0, 0, "temporal-tile"),
        "xor-zeta": (4, 0, 0, "temporal-tile"),
        "fft": (8, 1, 1, "cufft"),
    }
    for operator, (value_bytes, has_twiddle, bit_reverse, baseline_name) in metadata.items():
        pipeline = [row for row in operators if row["operator"] == operator and row["backend"] == "stage-pipeline"]
        best = min(pipeline, key=lambda row: float(row["median_kernel_ms"]))
        us1 = next(row for row in pipeline if row["stage_space"] == "1")
        baseline = next(row for row in operators if row["operator"] == operator and row["backend"] == baseline_name)
        output.append({
            "operator": operator,
            "value_bytes": value_bytes,
            "has_twiddle": has_twiddle,
            "bit_reverse_input": bit_reverse,
            "best_stage_space": best["stage_space"],
            "us1_kernel_ms": us1["median_kernel_ms"],
            "best_pipeline_ms": best["median_kernel_ms"],
            "pipeline_speedup_vs_us1": best["speedup_vs_us1"],
            "baseline": baseline_name,
            "baseline_ms": baseline["median_kernel_ms"],
            "pipeline_efficiency_vs_baseline": f"{float(baseline['median_kernel_ms']) / float(best['median_kernel_ms']):.6f}",
        })

    fields = list(output[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
