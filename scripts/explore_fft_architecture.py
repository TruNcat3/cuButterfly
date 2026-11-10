#!/usr/bin/env python3
import argparse
import csv
import pathlib
import statistics
import subprocess

from fft_design_space import load_codegen_points, load_space


def run(binary, log_n, batch, local_stages=None, twiddle="table", prefix_threads=0,
        suffix_threads=0, warmup=10, repeat=30):
    command = [
        str(binary), "--operator", "fft", "--precision", "fp32", "--logN", str(log_n),
        "--batch", str(batch), "--placement", "out-of-place", "--normalization", "none",
        "--warmup", str(warmup), "--repeat", str(repeat), "--csv",
    ]
    if local_stages is None:
        command += ["--backend", "cufft"]
    else:
        command += [
            "--backend", "online-reorder", "--fft-core", "cufftdx-block",
            "--local-stages", str(local_stages), "--reorder-columns", "1",
            "--cross-twiddle", twiddle, "--tile-threads", "32",
            "--prefix-threads", str(prefix_threads), "--suffix-threads", str(suffix_threads),
        ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected output from {' '.join(command)}")
    return rows[0]


def config_key(row):
    return (row["logN"], row["backend"], row["local_stages"], row["cross_twiddle"],
            row["prefix_threads"], row["suffix_threads"], row.get("prefix_ept", "8"),
            row.get("suffix_ept", "8"))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Hierarchical search of the two-dimensional FFT architecture mapping.")
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/cubutterfly_bench"))
    parser.add_argument("--space-spec", type=pathlib.Path,
                        default=pathlib.Path("config/fft_architecture_space.json"))
    parser.add_argument("--codegen-spec", type=pathlib.Path,
                        default=pathlib.Path("config/v100_fft_codegen.json"))
    parser.add_argument("--logNs", nargs="+", type=int, default=(16, 18, 20))
    parser.add_argument("--local-stages", nargs="+", type=int)
    parser.add_argument("--thread-options", nargs="+", type=int, choices=(128, 256, 512, 1024),
                        default=None)
    parser.add_argument("--target-points", type=int, default=1 << 22)
    parser.add_argument("--split-finalists", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--confirm-trials", type=int, default=5)
    parser.add_argument("--output-prefix", type=pathlib.Path,
                        default=pathlib.Path("results/fft_architecture_explore_v100"))
    args = parser.parse_args()

    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {args.binary}")
    space = load_space(args.space_spec)
    family = next(family for family in space["families"] if family["name"] == "cufftdx-online")
    compiled_points = load_codegen_points(args.codegen_spec)
    thread_options = args.thread_options or sorted({threads for _, threads, ept in compiled_points if ept == 8})
    dimension_range = family["dimension_log_n"]
    local_stage_options = args.local_stages or range(dimension_range["min"], dimension_range["max"] + 1)
    search_rows = []
    confirmation_rows = []
    summaries = []

    for log_n in args.logNs:
        batch = max(1, args.target_points // (1 << log_n))
        coarse = []
        for local_stages in local_stage_options:
            remaining = log_n - local_stages
            if not (3 <= local_stages <= 10 and 3 <= remaining <= 10):
                continue
            for twiddle in ("table", "recurrence"):
                row = run(args.binary, log_n, batch, local_stages, twiddle, 512, 512,
                          args.warmup, args.repeat)
                row = {"search_stage": "coarse", "trial": 1, **row}
                coarse.append(row)
                search_rows.append(row)

        coarse.sort(key=lambda row: float(row["kernel_ms"]))
        fine = []
        for seed in coarse[:args.split_finalists]:
            local_stages = int(seed["local_stages"])
            prefix_n = 1 << local_stages
            suffix_n = 1 << (log_n - local_stages)
            for prefix_threads in thread_options:
                if prefix_threads * 8 < prefix_n:
                    continue
                for suffix_threads in thread_options:
                    if suffix_threads * 8 < suffix_n:
                        continue
                    row = run(args.binary, log_n, batch, local_stages, seed["cross_twiddle"],
                              prefix_threads, suffix_threads, args.warmup, args.repeat)
                    row = {"search_stage": "fine", "trial": 1, **row}
                    fine.append(row)
                    search_rows.append(row)

        best_by_config = {}
        for row in coarse + fine:
            key = config_key(row)
            if key not in best_by_config or float(row["kernel_ms"]) < float(best_by_config[key]["kernel_ms"]):
                best_by_config[key] = row
        finalists = sorted(best_by_config.values(), key=lambda row: float(row["kernel_ms"]))[:args.finalists]

        confirmed_configs = [(None, "table", 0, 0)] + [
            (int(row["local_stages"]), row["cross_twiddle"], int(row["prefix_threads"]),
             int(row["suffix_threads"])) for row in finalists
        ]
        for local_stages, twiddle, prefix_threads, suffix_threads in confirmed_configs:
            samples = []
            for trial in range(1, args.confirm_trials + 1):
                row = run(args.binary, log_n, batch, local_stages, twiddle, prefix_threads,
                          suffix_threads, args.warmup, args.repeat)
                row = {"search_stage": "confirm", "trial": trial, **row}
                samples.append(row)
                confirmation_rows.append(row)
            representative = samples[0]
            summaries.append({
                "logN": log_n,
                "batch": batch,
                "backend": representative["backend"],
                "local_stages": representative["local_stages"],
                "remaining_stages": 0 if local_stages is None else log_n - local_stages,
                "cross_twiddle": representative["cross_twiddle"],
                "prefix_threads": representative["prefix_threads"],
                "suffix_threads": representative["suffix_threads"],
                "prefix_ept": representative["prefix_ept"],
                "suffix_ept": representative["suffix_ept"],
                "prefix_units_per_cta": representative["prefix_units_per_cta"],
                "suffix_units_per_cta": representative["suffix_units_per_cta"],
                "trials": len(samples),
                "median_kernel_ms": f"{statistics.median(float(row['kernel_ms']) for row in samples):.6f}",
                "min_kernel_ms": f"{min(float(row['kernel_ms']) for row in samples):.6f}",
                "max_kernel_ms": f"{max(float(row['kernel_ms']) for row in samples):.6f}",
            })

    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_search.csv"), search_rows)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_confirm.csv"), confirmation_rows)
    write_csv(args.output_prefix.with_name(args.output_prefix.name + "_summary.csv"), summaries)


if __name__ == "__main__":
    main()
