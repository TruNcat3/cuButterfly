#!/usr/bin/env python3
import argparse
import csv
import pathlib
import subprocess

from fft_design_space import load_codegen_points


def base_configurations(operator, precision, log_n, tile_threads, hierarchical_local_stages, reorder_column_options):
    if log_n <= 10:
        for compute_unit in ("radix2", "radix4", "radix8"):
            for threads in tile_threads:
                yield {
                    "backend": "temporal-tile",
                    "compute_unit": compute_unit,
                    "tile_threads": threads,
                    "local_stages": 0,
                    "reorder_columns": 0,
                    "warp_stages": 0,
                    "pipeline_warps": 0,
                    "stage_space": 0,
                }
    if log_n >= 6:
        for local_stages in hierarchical_local_stages:
            if local_stages >= log_n:
                continue
            for backend in ("hierarchical", "online-reorder"):
                if backend == "online-reorder" and log_n - local_stages > 10:
                    continue
                columns = reorder_column_options if backend == "online-reorder" else (0,)
                remaining_n = 1 << (log_n - local_stages)
                local_n = 1 << local_stages
                for reorder_columns in columns:
                    if reorder_columns and (reorder_columns > local_n or reorder_columns * remaining_n > 1024):
                        continue
                    for compute_unit in ("radix2", "radix4", "radix8"):
                        for threads in tile_threads:
                            yield {
                                "backend": backend,
                                "compute_unit": compute_unit,
                                "tile_threads": threads,
                                "local_stages": local_stages,
                                "reorder_columns": reorder_columns,
                                "warp_stages": 0,
                                "pipeline_warps": 0,
                                "stage_space": 0,
                            }
    if log_n == 8:
        for warp_stages in range(6):
            yield {
                "backend": "warp-hybrid",
                "compute_unit": "radix2",
                "tile_threads": 0,
                "local_stages": 0,
                "reorder_columns": 0,
                "warp_stages": warp_stages,
                "pipeline_warps": 0,
                "stage_space": 0,
            }
        if not (operator == "fft" and precision == "fp64"):
            for pipeline_warps in (4, 8):
                for stage_space in (1, 2, 4, 8):
                    if stage_space <= pipeline_warps:
                        yield {
                            "backend": "stage-pipeline",
                            "compute_unit": "radix2",
                            "tile_threads": 0,
                            "local_stages": 0,
                            "reorder_columns": 0,
                            "warp_stages": 0,
                            "pipeline_warps": pipeline_warps,
                            "stage_space": stage_space,
                        }
    if operator == "fft":
        yield {
            "backend": "cufft",
            "compute_unit": "auto",
            "tile_threads": 0,
            "local_stages": 0,
            "reorder_columns": 0,
            "warp_stages": 0,
            "pipeline_warps": 0,
            "stage_space": 0,
        }


def configurations(operator, precision, log_n, tile_threads, hierarchical_local_stages, reorder_column_options,
                   resident_thread_options, prefix_thread_options, suffix_thread_options,
                   prefix_ept_options, suffix_ept_options):
    if precision == "fp16-fp32":
        if operator == "fft" and log_n == 3:
            yield {
                "backend": "temporal-tile", "compute_unit": "radix8", "complex_multiply": "four-mul", "local_exchange": "shared",
                "fft_core": "wmma-dft8", "tile_threads": 256, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                "pipeline_warps": 0, "stage_space": 0,
            }
        return
    for config in base_configurations(operator, precision, log_n, tile_threads, hierarchical_local_stages, reorder_column_options):
        multiply_options = ("four-mul", "gauss3") if operator == "fft" and config["backend"] != "cufft" else ("four-mul",)
        for complex_multiply in multiply_options:
            yield {**config, "complex_multiply": complex_multiply, "local_exchange": "shared", "fft_core": "scalar"}
    if operator == "fwht" and precision == "fp32" and 3 <= log_n <= 15:
        threads = 1 if log_n == 3 else 2 if log_n == 4 else 4 if log_n == 5 else 8 if log_n == 6 else 16 if log_n == 7 else 32 if log_n <= 9 else 128 if log_n == 10 else 256
        yield {
            "backend": "temporal-tile", "compute_unit": "radix2", "complex_multiply": "four-mul", "local_exchange": "warp-register",
            "fft_core": "scalar", "tile_threads": threads, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
            "pipeline_warps": 0, "stage_space": 0,
        }
    if operator == "fft" and precision == "fp32":
        generated_cores = []
        if log_n == 3:
            generated_cores.append("thread-dft8")
        if 3 <= log_n <= 10:
            generated_cores.append("cta-dft8")
        for fft_core in generated_cores:
            threads_options = (128,) if fft_core == "thread-dft8" else tuple(t for t in tile_threads if t * 8 >= 1 << log_n)
            for threads in threads_options:
                yield {
                    "backend": "temporal-tile", "compute_unit": "radix8", "complex_multiply": "four-mul", "local_exchange": "shared",
                    "fft_core": fft_core, "tile_threads": threads, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                    "pipeline_warps": 0, "stage_space": 0,
                }
        if 3 <= log_n <= 10:
            yield {
                "backend": "temporal-tile", "compute_unit": "auto", "complex_multiply": "four-mul", "local_exchange": "shared",
                "fft_core": "cufftdx-block", "tile_threads": 32, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                "pipeline_warps": 0, "stage_space": 0,
            }
        if 11 <= log_n <= 14:
            direct_threads = tuple(t for t in resident_thread_options if log_n != 14 or t != 256)
            for threads in direct_threads:
                yield {
                    "backend": "temporal-tile", "compute_unit": "auto", "complex_multiply": "four-mul",
                    "cross_twiddle": "table", "local_exchange": "shared", "fft_core": "cufftdx-direct",
                    "tile_threads": threads, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                    "pipeline_warps": 0, "stage_space": 0,
                }
        for local_stages in hierarchical_local_stages:
            remaining_stages = log_n - local_stages
            if 3 <= local_stages <= 10 and 3 <= remaining_stages <= 10:
                for cross_twiddle in ("table", "recurrence"):
                    prefix_n = 1 << local_stages
                    suffix_n = 1 << remaining_stages
                    for prefix_threads in prefix_thread_options:
                        for prefix_ept in prefix_ept_options:
                            if prefix_ept > prefix_n or prefix_threads * prefix_ept < prefix_n or \
                                    (prefix_threads * prefix_ept) % prefix_n:
                                continue
                            for suffix_threads in suffix_thread_options:
                                for suffix_ept in suffix_ept_options:
                                    if suffix_ept > suffix_n or suffix_threads * suffix_ept < suffix_n or \
                                            (suffix_threads * suffix_ept) % suffix_n:
                                        continue
                                    yield {
                                        "backend": "online-reorder", "compute_unit": "auto", "complex_multiply": "four-mul",
                                        "cross_twiddle": cross_twiddle, "local_exchange": "shared", "fft_core": "cufftdx-block",
                                        "tile_threads": 32, "prefix_threads": prefix_threads, "suffix_threads": suffix_threads,
                                        "prefix_ept": prefix_ept, "suffix_ept": suffix_ept,
                                        "local_stages": local_stages, "reorder_columns": 1, "warp_stages": 0,
                                        "pipeline_warps": 0, "stage_space": 0,
                                    }
        resident_local_stages = {12: 6, 14: 7}.get(log_n)
        if resident_local_stages is not None and resident_local_stages in hierarchical_local_stages:
            for threads in resident_thread_options:
                for cross_twiddle in ("table", "recurrence"):
                    yield {
                        "backend": "online-reorder", "compute_unit": "auto", "complex_multiply": "four-mul",
                        "cross_twiddle": cross_twiddle, "local_exchange": "shared", "fft_core": "cufftdx-resident",
                        "tile_threads": threads, "local_stages": resident_local_stages, "reorder_columns": 1, "warp_stages": 0,
                        "pipeline_warps": 0, "stage_space": 0,
                    }
        if 7 <= log_n <= 10:
            yield {
                "backend": "temporal-tile", "compute_unit": "auto", "complex_multiply": "four-mul", "local_exchange": "shared",
                "fft_core": "turbofft-generated", "tile_threads": 32, "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                "pipeline_warps": 0, "stage_space": 0,
            }
    if operator == "fft" and precision == "fp64":
        if 3 <= log_n <= 10:
            yield {
                "backend": "temporal-tile", "compute_unit": "auto", "complex_multiply": "four-mul",
                "local_exchange": "shared", "fft_core": "cufftdx-block", "tile_threads": 32,
                "local_stages": 0, "reorder_columns": 0, "warp_stages": 0,
                "pipeline_warps": 0, "stage_space": 0,
            }
        if log_n == 16 and 8 in hierarchical_local_stages:
            for prefix_threads in prefix_thread_options:
                for prefix_ept in prefix_ept_options:
                    if prefix_threads not in (128, 256, 512) or prefix_ept not in (4, 8):
                        continue
                    for suffix_threads in suffix_thread_options:
                        for suffix_ept in suffix_ept_options:
                            if suffix_threads not in (128, 256, 512) or suffix_ept not in (4, 8):
                                continue
                            yield {
                                "backend": "online-reorder", "compute_unit": "auto",
                                "complex_multiply": "four-mul", "cross_twiddle": "table",
                                "local_exchange": "shared", "fft_core": "cufftdx-block",
                                "tile_threads": 32, "prefix_threads": prefix_threads,
                                "suffix_threads": suffix_threads, "prefix_ept": prefix_ept,
                                "suffix_ept": suffix_ept, "local_stages": 8, "reorder_columns": 1,
                                "warp_stages": 0, "pipeline_warps": 0, "stage_space": 0,
                            }


def run(binary, operator, precision, config, log_n, direction, normalization, placement, batch, element_stride, batch_stride, warmup, repeat):
    command = [
        str(binary), "--operator", operator, "--backend", config["backend"],
        "--compute-unit", config["compute_unit"], "--precision", precision, "--logN", str(log_n),
        "--complex-multiply", config["complex_multiply"],
        "--cross-twiddle", config.get("cross_twiddle", "table"),
        "--local-exchange", config["local_exchange"],
        "--fft-core", config["fft_core"],
        "--normalization", normalization, "--placement", placement,
        "--batch", str(batch), "--warmup", str(warmup), "--repeat", str(repeat), "--csv",
        "--batch-stride", str(batch_stride),
        "--element-stride", str(element_stride),
    ]
    if direction == "inverse":
        command.append("--inverse")
    if config["backend"] == "temporal-tile":
        command += ["--tile-threads", str(config["tile_threads"])]
    elif config["backend"] in ("hierarchical", "online-reorder"):
        command += ["--tile-threads", str(config["tile_threads"]), "--local-stages", str(config["local_stages"])]
        if config["backend"] == "online-reorder":
            command += ["--reorder-columns", str(config["reorder_columns"])]
            if config.get("prefix_threads"):
                command += ["--prefix-threads", str(config["prefix_threads"])]
            if config.get("suffix_threads"):
                command += ["--suffix-threads", str(config["suffix_threads"])]
            if config.get("prefix_ept"):
                command += ["--prefix-ept", str(config["prefix_ept"])]
            if config.get("suffix_ept"):
                command += ["--suffix-ept", str(config["suffix_ept"])]
    elif config["backend"] == "warp-hybrid":
        command += ["--warp-stages", str(config["warp_stages"])]
    elif config["backend"] == "stage-pipeline":
        command += [
            "--stage-space", str(config["stage_space"]),
            "--pipeline-warps", str(config["pipeline_warps"]),
            "--stage-handoff", "named-barrier",
        ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected benchmark output for {' '.join(command)}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description="Sweep cuButterfly kernel forms and mapping parameters.")
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/cubutterfly_bench"))
    parser.add_argument("--fft-codegen-spec", type=pathlib.Path,
                        default=pathlib.Path("config/v100_fft_codegen.json"))
    parser.add_argument("--operators", nargs="+", choices=("fwht", "fft", "xor-zeta"), default=("fwht", "fft", "xor-zeta"))
    parser.add_argument("--backends", nargs="+", choices=("temporal-tile", "hierarchical", "online-reorder", "warp-hybrid", "stage-pipeline", "cufft"),
                        default=("temporal-tile", "hierarchical", "online-reorder", "warp-hybrid", "stage-pipeline", "cufft"))
    parser.add_argument("--compute-units", nargs="+", choices=("radix2", "radix4", "radix8"), default=("radix2", "radix4", "radix8"))
    parser.add_argument("--complex-multiplies", nargs="+", choices=("four-mul", "gauss3"), default=("four-mul", "gauss3"))
    parser.add_argument("--cross-twiddles", nargs="+", choices=("table", "recurrence"), default=("table",))
    parser.add_argument("--local-exchanges", nargs="+", choices=("shared", "warp-register"), default=("shared", "warp-register"))
    parser.add_argument("--fft-cores", nargs="+",
                        choices=("scalar", "thread-dft8", "cta-dft8", "wmma-dft8", "cufftdx-block", "cufftdx-direct", "cufftdx-resident", "turbofft-generated"),
                        default=("scalar", "thread-dft8", "cta-dft8", "wmma-dft8"))
    parser.add_argument("--logNs", nargs="+", type=int, default=(8,))
    parser.add_argument("--directions", nargs="+", choices=("forward", "inverse"), default=("forward",))
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "fp64", "fp16-fp32"), default=("fp32",))
    parser.add_argument("--normalizations", nargs="+", choices=("none", "inverse"), default=("inverse",))
    parser.add_argument("--placements", nargs="+", choices=("in-place", "out-of-place"), default=("out-of-place",))
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--target-points", type=int, default=0, help="Override batch per length to process approximately this many points.")
    parser.add_argument("--batch-padding", type=int, default=0, help="Add this many elements between adjacent transforms.")
    parser.add_argument("--element-stride", type=int, default=1)
    parser.add_argument("--tile-thread-options", nargs="+", type=int, choices=(32, 64, 128, 256), default=(32, 64, 128, 256))
    parser.add_argument("--resident-thread-options", nargs="+", type=int, choices=(256, 512, 1024), default=(512,))
    parser.add_argument("--prefix-thread-options", nargs="+", type=int, choices=(128, 256, 512, 1024), default=(512,))
    parser.add_argument("--suffix-thread-options", nargs="+", type=int, choices=(128, 256, 512, 1024), default=(512,))
    parser.add_argument("--prefix-ept-options", nargs="+", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=(8,))
    parser.add_argument("--suffix-ept-options", nargs="+", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=(8,))
    parser.add_argument("--hierarchical-local-stages", nargs="+", type=int, choices=range(5, 11), default=(5, 6, 8, 10))
    parser.add_argument("--reorder-column-options", nargs="+", type=int,
                        choices=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024), default=(1,))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {args.binary}")
    compiled_fft_points = load_codegen_points(args.fft_codegen_spec)
    records = []
    for operator in args.operators:
        precisions = ("uint32",) if operator == "xor-zeta" else args.precisions
        for precision in precisions:
            for log_n in args.logNs:
                if not 1 <= log_n <= 20:
                    raise ValueError("cuButterfly logN sweep values must be in [1, 20]")
                batch = max(1, args.target_points // (1 << log_n)) if args.target_points else args.batch
                transform_extent = ((1 << log_n) - 1) * args.element_stride + 1
                batch_stride = transform_extent + args.batch_padding
                for direction in args.directions:
                    normalizations = ("none",) if operator == "xor-zeta" else args.normalizations
                    for normalization in normalizations:
                        for placement in args.placements:
                            for config in configurations(operator, precision, log_n, args.tile_thread_options,
                                                         args.hierarchical_local_stages, args.reorder_column_options,
                                                         args.resident_thread_options, args.prefix_thread_options,
                                                         args.suffix_thread_options, args.prefix_ept_options,
                                                         args.suffix_ept_options):
                                if config["backend"] not in args.backends:
                                    continue
                                if config["compute_unit"] != "auto" and config["compute_unit"] not in args.compute_units:
                                    continue
                                if operator == "fft" and config["complex_multiply"] not in args.complex_multiplies:
                                    continue
                                if config.get("cross_twiddle", "table") not in args.cross_twiddles:
                                    continue
                                if config["local_exchange"] not in args.local_exchanges:
                                    continue
                                if config["fft_core"] not in args.fft_cores:
                                    continue
                                if config["fft_core"] == "cufftdx-block" and config["backend"] == "online-reorder":
                                    prefix_key = (config["local_stages"], config["prefix_threads"], config["prefix_ept"])
                                    suffix_key = (log_n - config["local_stages"], config["suffix_threads"], config["suffix_ept"])
                                    if prefix_key not in compiled_fft_points or suffix_key not in compiled_fft_points:
                                        continue
                                for trial in range(1, args.trials + 1):
                                    row = run(args.binary, operator, precision, config, log_n, direction, normalization, placement, batch,
                                              args.element_stride, batch_stride, args.warmup, args.repeat)
                                    row = {"trial": trial, **row}
                                    records.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
