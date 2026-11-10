#!/usr/bin/env python3
"""Bounded explicit-core experiment, intended inside run_isolated_gpu.py.

CPU references, compilation and capture are excluded from CUDA-event timings.
The CUDA-process guard owns isolation; this runner records each command/result.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=os.environ.get("XD_REPORT_DIR"))
    parser.add_argument("--phase", choices=("all", "correctness", "performance", "profile"), default="all")
    args = parser.parse_args()
    if args.output is None:
        parser.error("provide --output or use the exclusive GPU runner")
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    config = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    sources = ["apps/cubutterfly_boundary_bench.cu", "src/shared_iterative.cuh",
               "src/operator_traits.cuh", "src/butterfly.cu", "src/mixed_dataflow.cpp"]
    identity = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "binary": str(binary), "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256((binary.parent/"libcuntt.so").read_bytes()).hexdigest(),
        "build_fingerprint_header": (binary.parent/"generated/build_fingerprint.hpp").read_text(),
        "config": config, "phase": args.phase,
        "compile_mode": os.environ.get("CUBUTTERFLY_COMPILE_MODE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                                         "--format=csv,noheader"], text=True),
        "cases": [],
    }
    rows = []
    def record(label, command_args, reject=False, profile=False):
        command = [str(binary)] + list(map(str, command_args))
        if profile:
            command = ["/usr/local/cuda/bin/ncu", "--csv", "--page", "raw",
                       "--cache-control", "none", "--clock-control", "none",
                       "--launch-skip", "4", "--launch-count", "2", "--metrics",
                       ",".join(config["ncu_metrics"])] + command
        completed = subprocess.run(command, text=True, capture_output=True, timeout=300)
        (args.output/f"{label}.csv").write_text(completed.stdout)
        (args.output/f"{label}.stderr").write_text(completed.stderr)
        accepted = (completed.returncode == 1 and "exceeds static hardware resources" in completed.stderr
                    if reject else completed.returncode == 0)
        identity["cases"].append(dict(label=label, command=command, returncode=completed.returncode,
                                      expected_resource_rejection=reject, diagnostic_only=profile, accepted=accepted))
        (args.output/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
        if not accepted:
            raise RuntimeError(f"{label}: failed, inspect saved CSV and stderr")
        if not reject and not profile:
            observed = list(csv.DictReader(io.StringIO(completed.stdout)))
            expected_modes = {"strided", "materialized", "producer"}
            if command_args[0] == "fft":
                expected_modes.add("cufft")
            if {r["mode"] for r in observed} != expected_modes:
                raise RuntimeError(f"{label}: incomplete mode comparison")
            rows.extend(dict(experiment=label, **r) for r in observed)
        print(f"PASS {label}", flush=True)
    for phase in ("correctness", "performance"):
        if args.phase not in ("all", phase):
            continue
        spec = config[phase]
        thread_options = spec["threads"] if isinstance(spec["threads"], list) else [spec["threads"]]
        for op, precision, tiles, inverse, threads in itertools.product(
                config["operators"], config["precisions"], spec["tiles"], spec["inverse"], thread_options):
            label = f"{phase}-{op}-{precision}-t{tiles}-inv{inverse}-threads{threads}"
            record(label, [op, precision, spec["logN"], spec["batch"], spec["split"], tiles,
                           threads, inverse, spec["repeat"], spec["trials"], "all"])
    if args.phase == "all":
        for i, case in enumerate(config.get("resource_rejections", [])):
            record(f"resource-rejection-{i}", case, reject=True)
    if args.phase == "profile":
        for i, case in enumerate(config["profile_cases"]):
            record(f"profile-{i}", case, profile=True)
    if rows:
        with (args.output/"summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
