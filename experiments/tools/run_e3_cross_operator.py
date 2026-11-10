#!/usr/bin/env python3
"""Prepare and run the non-migration E3 cross-operator organization control.

This is a small, reproducible wrapper around the already audited public
resident-mapping runner.  It changes only the arithmetic family and batch
axis: FWHT FP32 and natural-order NTT word64 at N=2^12, with leaf0/leaf2
shared-iterative organizations.  Compilation is a separate CPU-only mode;
the GPU mode uses the pinned A100 target and the runner's exclusivity,
correctness, warmup, and thermal guards.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/paper_completion_20260918/public_resident_promotion/run.py"
HERE = ROOT / "results/paper_completion_20260918/e3_cross_operator"


def load_runner():
    spec = importlib.util.spec_from_file_location("public_resident_runner", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load audited runner: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    # Register the dynamic module so any imported helper retains a normal
    # module identity while the audited runner is reused below.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.HERE = HERE
    module.CACHE = HERE / "modules"
    return module


def cases():
    for operator, precision in (("fwht", "fp32"), ("ntt", "word64")):
        for batch in (1, 1024):
            for width in (0, 2):
                mapping = dict(
                    schema_version=1,
                    backend="shared-iterative",
                    stage_partition=[8, 4],
                    local_stage_partitions=[],
                    exchange_chunks=[],
                )
                if operator == "ntt":
                    mapping.update(threads_per_block=128, dataflow_layout="hermes-xor")
                else:
                    mapping.update(tile_threads=128, local_stages=8, shared_layout="writer-aligned")
                if width:
                    mapping["local_stage_partitions"] = [[width] * (stage // width) for stage in (8, 4)]
                point = dict(
                    operator=operator,
                    logN=12,
                    batch=batch,
                    precision=precision,
                    direction="forward",
                    normalization="none",
                    placement="out-of-place",
                    mapping_json=json.dumps(mapping, sort_keys=True),
                )
                if operator == "ntt":
                    point.update(modulus=576460756061519873, output_order="natural", input_order="natural")
                yield dict(
                    id=f"{operator}-n12-b{batch}-leaf{width}",
                    family=operator,
                    variant=f"leaf{width}",
                    point=point,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "run"))
    parser.add_argument("--target", choices=("a100-40gb", "a100-80gb"), default="a100-80gb")
    parser.add_argument("--workers", type=int, default=1,
                        help="compatibility option; the four-module E3 preparation is intentionally sequential")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    runner.cases = cases
    if args.mode == "compile":
        # Keep this bounded preparation independent of multiprocessing start
        # methods: the E3 pool has only four unique modules and deterministic
        # sequential compilation makes the provenance easier to audit.
        rows = []
        for case in cases():
            row = runner.compile_one(case)
            rows.append(row)
            print("compiled", row["id"], flush=True)
        runner.save(HERE / "compilation.json", {"status": "compiled", "modules": rows})
    else:
        runner.run(args.target)


if __name__ == "__main__":
    main()
