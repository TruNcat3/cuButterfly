#!/usr/bin/env python3
"""Pair isolated native and cuButterfly CKKS stage-only measurements.

The benchmark intentionally keeps only one stage backend resident per process
for large batches.  This tool combines the two exact-contract CSV rows without
using either process's eager time as the other's denominator.
"""

import argparse
import csv
import math
from pathlib import Path


def read_one(path: Path, backend: str):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one completed measurement row, found {len(rows)}")
    row = rows[0]
    if row.get("stage_backend") != backend:
        raise ValueError(f"{path}: stage_backend={row.get('stage_backend')!r}, expected {backend!r}")
    if row.get("exact_ciphertext") != "1" or row.get("mismatch_count") != "0":
        raise ValueError(f"{path}: ciphertext correctness failed")
    for field in ("native_ms", "cubutterfly_ms"):
        value = float(row[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: invalid {field}={row[field]!r}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True,
                        help="isolated --stage-backend native CSV")
    parser.add_argument("--cu", type=Path, required=True,
                        help="isolated --stage-backend cubutterfly CSV")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = read_one(args.native, "native")
    cu = read_one(args.cu, "cubutterfly")

    identity_fields = ("scene", "type", "N", "batch", "mode", "ntt_execution",
                       "graph_replays_per_sample", "parameter_set", "keyswitch_method")
    for field in identity_fields:
        if native.get(field) != cu.get(field):
            raise ValueError(f"request mismatch for {field}: {native.get(field)!r} != {cu.get(field)!r}")
    native_eager = float(native["native_ms"])
    native_stage = float(native["cubutterfly_ms"])
    cu_eager = float(cu["native_ms"])
    cu_stage = float(cu["cubutterfly_ms"])
    result = {
        "scene": native["scene"],
        "type": native["type"],
        "N": native["N"],
        "batch": native["batch"],
        "mode": native["mode"],
        "ntt_execution": native["ntt_execution"],
        "parameter_set": native["parameter_set"],
        "keyswitch_method": native["keyswitch_method"],
        "native_eager_ms": native_eager,
        "native_stage_ms": native_stage,
        "cu_eager_ms": cu_eager,
        "cu_stage_ms": cu_stage,
        "native_stage_speedup": native_eager / native_stage,
        "cu_stage_speedup": cu_eager / cu_stage,
        "cu_vs_native_stage": native_stage / cu_stage,
        "exact_native": native["exact_ciphertext"],
        "exact_cu": cu["exact_ciphertext"],
        "native_mismatch": native["mismatch_count"],
        "cu_mismatch": cu["mismatch_count"],
        "native_stage_nodes": native["cubutterfly_kernel_nodes"],
        "cu_stage_nodes": cu["cubutterfly_kernel_nodes"],
        "polynomials": native["polynomials"],
        "repetitions": native["repetitions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)
    print(",".join(f"{key}={value}" for key, value in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
