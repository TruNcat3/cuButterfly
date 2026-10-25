#!/usr/bin/env python3
"""Build a bounded, CPU-only semantic coverage matrix.

The matrix describes public requests; it does not claim that every row has a
GPU lowering or run a benchmark.  It is deliberately factorized so a missing
combination cannot be hidden by an operator-specific list.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


SCHEMA = "cubutterfly-semantic-coverage-v1"
OPERATORS = ("fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "structured-2x2")
FLOAT_PRECISIONS = ("fp32", "fp64")
NTT_PRECISIONS = ("word32", "word64")
ORDERS = ("natural", "bit-reversed")
PLACEMENTS = ("out-of-place", "in-place")
RANKS = (1, 2, 3)
SHAPES = {1: (8,), 2: (4, 8), 3: (2, 4, 8)}
BATCHES = (1, 4, 32)


def _orders(rank: int, value: str, compact_axis: bool = False):
    # The compact real axis has no independent half-spectrum reversal.  The
    # matrix still records it explicitly as natural on that axis.
    values = [value] * rank
    if compact_axis:
        values[-1] = "natural"
    return values


def rows():
    result = []
    for operator in OPERATORS:
        precisions = NTT_PRECISIONS if operator == "ntt" else FLOAT_PRECISIONS
        directions = ("forward", "inverse")
        fft_types = ("c2c",)
        if operator == "fft":
            fft_types = ("c2c", "r2c", "c2r")
        for precision, rank, batch, direction, placement, order, fft_type in itertools.product(
                precisions, RANKS, BATCHES, directions, PLACEMENTS, ORDERS, fft_types):
            if operator != "fft" and fft_type != "c2c":
                continue
            if operator == "fft" and fft_type == "r2c" and direction != "forward":
                continue
            if operator == "fft" and fft_type == "c2r" and direction != "inverse":
                continue
            shape = SHAPES[rank]
            compact_axis = operator == "fft" and fft_type != "c2c"
            result.append({
                "id": f"{operator}-r{rank}-{precision}-b{batch}-{direction}-"
                      f"{placement}-{order}-{fft_type}",
                "schema": SCHEMA,
                "operator": operator,
                "precision": precision,
                "rank": rank,
                "shape": list(shape),
                "batch": batch,
                "direction": direction,
                "placement": placement,
                "input_order": _orders(rank, order, compact_axis),
                "output_order": _orders(rank, order, compact_axis),
                "fft_type": fft_type,
                "execution_modes": ["composite", "xd"] if rank > 1 else ["composite"],
                "status": "cpu-manifest",
            })
    return result


def build():
    data = rows()
    return {"schema": SCHEMA, "run_mode": "cpu-dry-run", "rows": data,
            "axes": {"operator": OPERATORS, "precision": FLOAT_PRECISIONS + NTT_PRECISIONS,
                     "rank": RANKS, "batch": BATCHES, "direction": directions(),
                     "placement": PLACEMENTS, "order": ORDERS}}


def directions():
    return ("forward", "inverse")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
