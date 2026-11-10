#!/usr/bin/env python3
import argparse
import csv
import pathlib
import subprocess


MAPPINGS = {
    14: (7, 128, 4, 128, 4),
    15: (7, 128, 4, 128, 8),
    16: (8, 128, 8, 128, 8),
    17: (8, 128, 8, 256, 8),
    18: (9, 256, 8, 256, 8),
}


def command(binary, log_n, batch, backend, warmup, repeat, verify=False):
    args = [
        str(binary), "--operator", "fft", "--precision", "fp64",
        "--backend", backend, "--logN", str(log_n), "--batch", str(batch),
        "--placement", "out-of-place", "--normalization", "none",
        "--warmup", str(warmup), "--repeat", str(repeat), "--csv",
    ]
    if backend == "online-reorder":
        local, prefix_threads, prefix_ept, suffix_threads, suffix_ept = MAPPINGS[log_n]
        args += [
            "--fft-core", "cufftdx-block", "--local-stages", str(local),
            "--prefix-threads", str(prefix_threads), "--prefix-ept", str(prefix_ept),
            "--suffix-threads", str(suffix_threads), "--suffix-ept", str(suffix_ept),
            "--cross-twiddle", "recurrence", "--shared-layout", "xor-swizzle",
        ]
    if verify:
        args.append("--verify")
    return args


def run(args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    rows = list(csv.DictReader(result.stdout.splitlines()))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected benchmark output for {' '.join(args)}")
    return rows[0]


def main():
    parser = argparse.ArgumentParser(description="Measure FP64 FFT length/batch robustness against cuFFT.")
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/cubutterfly_bench"))
    parser.add_argument("--lengths", nargs="+", type=int, default=tuple(MAPPINGS))
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32, 64))
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", "-o", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.binary.is_file():
        raise FileNotFoundError(f"benchmark binary not found: {args.binary}")
    unsupported = set(args.lengths) - set(MAPPINGS)
    if unsupported:
        raise ValueError(f"unsupported FP64 robustness lengths: {sorted(unsupported)}")
    if any(batch < 1 for batch in args.batches):
        raise ValueError("batches must be positive")

    for log_n in args.lengths:
        run(command(args.binary, log_n, 1, "online-reorder", 1, 1, verify=True))

    records = []
    for log_n in args.lengths:
        for batch in args.batches:
            for trial in range(1, args.trials + 1):
                backends = ("online-reorder", "cufft")
                if (log_n + batch + trial) & 1:
                    backends = tuple(reversed(backends))
                for backend in backends:
                    row = run(command(args.binary, log_n, batch, backend, args.warmup, args.repeat))
                    records.append({"trial": trial, **row})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=records[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
