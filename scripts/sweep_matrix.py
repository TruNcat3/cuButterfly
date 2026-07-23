#!/usr/bin/env python3

import argparse
import csv
import io
import random
import statistics
import subprocess
from pathlib import Path


MILLER_RABIN_BASES = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    power = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power += 1
    for base in MILLER_RABIN_BASES:
        if base % value == 0:
            continue
        result = pow(base, odd_part, value)
        if result in (1, value - 1):
            continue
        for _ in range(1, power):
            result = result * result % value
            if result == value - 1:
                break
        else:
            return False
    return True


def ntt_prime(bits: int, log_n: int) -> int:
    step = 1 << log_n
    multiplier = ((1 << bits) - 2) // step
    while multiplier > 0:
        candidate = multiplier * step + 1
        if candidate.bit_length() != bits:
            break
        if is_prime(candidate):
            return candidate
        multiplier -= 1
    raise RuntimeError(f"could not find a {bits}-bit prime congruent to 1 mod 2^{log_n}")


def parse_csv(output: str) -> dict[str, str]:
    lines = [line for line in output.splitlines() if line.startswith("device,") or line.startswith('"')]
    if len(lines) != 2:
        raise RuntimeError(f"benchmark did not produce one CSV record:\n{output}")
    return next(csv.DictReader(io.StringIO("\n".join(lines))))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep cuNTT Hybrid2D length, modulus width, and compute unit.")
    parser.add_argument("--bench", type=Path, default=Path("build/cuntt_bench"))
    parser.add_argument("--output", type=Path, default=Path("results/hybrid2d_matrix.csv"))
    parser.add_argument("--logN", type=int, nargs="+", default=[12, 14, 16, 18, 20])
    parser.add_argument("--bits", type=int, nargs="+", default=[30, 40, 50, 60])
    parser.add_argument("--units", nargs="+", choices=["radix2", "radix4", "radix8"], default=["radix2", "radix4", "radix8"])
    parser.add_argument("--cross-twiddle", choices=["first", "second", "fused", "fused-barrett"], default="first")
    parser.add_argument("--mod-multiplies", nargs="+", choices=["shoup", "barrett"], default=["shoup", "barrett"])
    parser.add_argument("--word-bits", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--target-points-log", type=int, default=22)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cases = []
    mod_multiply_options = ["barrett"] if args.cross_twiddle == "fused-barrett" else args.mod_multiplies
    for log_n in args.logN:
        batch = max(1, (1 << args.target_points_log) // (1 << log_n))
        for bits in args.bits:
            modulus = ntt_prime(bits, log_n)
            valid_word_bits = [word_bits for word_bits in args.word_bits if word_bits == 64 or bits < 31]
            for word_bits in valid_word_bits:
                for unit in args.units:
                    for mod_multiply in mod_multiply_options:
                        cases.append((log_n, bits, word_bits, unit, mod_multiply, batch, modulus))

    raw_records = []
    records_by_case = {case[:5]: [] for case in cases}
    for trial in range(args.trials):
        trial_cases = list(cases)
        random.Random(0xC017 + trial).shuffle(trial_cases)
        for log_n, bits, word_bits, unit, mod_multiply, batch, modulus in trial_cases:
            command = [
                str(args.bench), "--logN", str(log_n), "--batch", str(batch),
                "--backend", "hybrid2d", "--compute-unit", unit,
                "--cross-twiddle", args.cross_twiddle,
                "--mod-multiply", mod_multiply,
                "--word-bits", str(word_bits),
                "--modulus", str(modulus), "--warmup", str(args.warmup),
                "--repeat", str(args.repeat), "--csv",
            ]
            if args.verify:
                command.append("--verify")
            completed = subprocess.run(command, check=True, text=True, capture_output=True)
            record = parse_csv(completed.stdout)
            record["trial"] = str(trial + 1)
            raw_records.append(record)
            records_by_case[(log_n, bits, word_bits, unit, mod_multiply)].append(record)
            print(
                f"trial={trial + 1} logN={log_n:2d} modulus_bits={bits:2d} word_bits={word_bits:2d} "
                f"unit={unit:6s} mod_multiply={mod_multiply:7s} kernel_ms={float(record['kernel_ms']):.6f}",
                flush=True,
            )

    metric_fields = ("h2d_ms", "kernel_ms", "d2h_ms", "kernel_ntt_s", "end_to_end_ntt_s", "kernel_points_s")
    records = []
    for case in cases:
        samples = records_by_case[case[:5]]
        record = {key: value for key, value in samples[0].items() if key != "trial"}
        for field in metric_fields:
            record[field] = f"{statistics.median(float(sample[field]) for sample in samples):.6f}"
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    raw_output = args.output.with_name(args.output.stem + "_raw" + args.output.suffix)
    with raw_output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=raw_records[0].keys())
        writer.writeheader()
        writer.writerows(raw_records)
    print(f"wrote {len(records)} medians to {args.output} and {len(raw_records)} samples to {raw_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
