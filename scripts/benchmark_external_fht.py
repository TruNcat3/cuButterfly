#!/usr/bin/env python3
import argparse
import csv
import math
import pathlib


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dao-AILab fast-hadamard-transform with cuButterfly-compatible fields.")
    parser.add_argument("--logNs", nargs="+", type=int, default=(8, 10, 12, 15))
    parser.add_argument("--dtypes", nargs="+", choices=("fp16", "bf16", "fp32"), default=("fp16", "bf16", "fp32"))
    parser.add_argument("--target-points", type=int, default=1 << 24)
    parser.add_argument("--batch", type=int, help="Use one explicit batch for every requested length.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--normalization", choices=("none", "inverse"), default="none")
    parser.add_argument("--sm70-patched", action="store_true", help="Acknowledge the recorded local sm_70 gencode patch.")
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    try:
        import torch
        from fast_hadamard_transform import hadamard_transform
    except ImportError as error:
        raise SystemExit("install PyTorch and Dao-AILab/fast-hadamard-transform before running this adapter") from error
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available through PyTorch")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (7, 5) and not ((major, minor) == (7, 0) and args.sm70_patched):
        raise SystemExit("upstream targets sm_75+; pass --sm70-patched only after applying patches/dao_fast_hadamard_sm70.patch")

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    records = []
    for log_n in args.logNs:
        if not 3 <= log_n <= 15:
            raise ValueError("upstream power-of-two FWHT supports logN in [3, 15]")
        n = 1 << log_n
        batch = args.batch if args.batch is not None else max(1, args.target_points // n)
        if batch < 1:
            raise ValueError("batch must be positive")
        scale = 1.0 / n if args.normalization == "inverse" else 1.0
        for dtype_name in args.dtypes:
            values = torch.randn((batch, n), device="cuda", dtype=dtype_map[dtype_name])
            check_input = values[: min(batch, 2)].clone()
            check_scale = 1.0 / math.sqrt(n)
            check_output = hadamard_transform(hadamard_transform(check_input, check_scale), check_scale)
            expected = check_input.float()
            max_error = (check_output.float() - expected).abs().max().item()
            tolerance = 2.0e-3 * max(1.0, expected.abs().max().item()) if dtype_name == "fp32" else 5.0e-2 * max(1.0, expected.abs().max().item())
            correct = max_error <= tolerance
            if not correct:
                raise RuntimeError(f"round-trip verification failed for logN={log_n} {dtype_name}: error={max_error} tolerance={tolerance}")
            for _ in range(args.warmup):
                hadamard_transform(values, scale)
            torch.cuda.synchronize()
            for trial in range(1, args.trials + 1):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.repeat):
                    hadamard_transform(values, scale)
                stop.record()
                stop.synchronize()
                kernel_ms = start.elapsed_time(stop) / args.repeat
                transforms_s = batch / (kernel_ms / 1000.0)
                records.append({
                    "trial": trial,
                    "device": torch.cuda.get_device_name(),
                    "compute_capability": f"{major}.{minor}",
                    "operator": "fwht",
                    "precision": dtype_name,
                    "direction": "forward",
                    "normalization": args.normalization,
                    "placement": "out-of-place",
                    "backend": "dao-fast-hadamard-transform-sm70-local" if (major, minor) == (7, 0) else "dao-fast-hadamard-transform",
                    "compute_unit": "upstream-warp-register",
                    "logN": log_n,
                    "N": n,
                    "batch": batch,
                    "element_stride": 1,
                    "batch_stride": n,
                    "kernel_ms": f"{kernel_ms:.6f}",
                    "transforms_s": f"{transforms_s:.6f}",
                    "Gbutterfly_s": f"{transforms_s * (n // 2) * log_n / 1.0e9:.6f}",
                    "max_roundtrip_error": f"{max_error:.6f}",
                    "correct": int(correct),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
