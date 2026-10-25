#!/usr/bin/env python3
"""Bounded core/organization experiment, not public-library autotuning.

GPU timing finishes before CPU FP32 reference/error processing. Raw outputs and
all elementwise absolute errors are retained, without an acceptance tolerance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import scipy
from scipy import fft


def command(args):
    return subprocess.check_output(args, text=True).strip()


def gpu_state(uuid, retry=3):
    details = command([
        "nvidia-smi", "-i", uuid,
        "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]).split(", ")
    apps = command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits",
    ])
    if len(details) != 6 or details[0] != uuid:
        raise RuntimeError("GPU identity query did not match the requested UUID")
    if any(line.split(",")[0] == uuid for line in apps.splitlines()):
        raise RuntimeError(f"GPU is busy; remaining measurements were not started: {details}")
    # NVML utilization retains the previous sampling window after our child
    # process exits. Let that finite window expire, while still rejecting any
    # new process immediately. This is not an unbounded idle-GPU waiter.
    if int(details[4]):
        if retry:
            time.sleep(1)
            return gpu_state(uuid, retry-1)
        raise RuntimeError(f"GPU activity did not settle: {details}")
    return dict(zip(("uuid", "name", "memory_mib", "memory_used_mib", "utilization_pct", "driver"), details))


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def errors(output, reference):
    finite = np.isfinite(output)
    absolute = np.abs(output.astype(np.complex128) - reference.astype(np.complex128))
    valid = absolute[np.isfinite(absolute)]
    result = {"elements": int(output.size), "nonfinite": int(output.size - finite.sum()),
              "absolute_error_quantiles": None, "relative_l2": None}
    if valid.size:
        result["absolute_error_quantiles"] = dict(zip(
            ("min", "median", "p95", "p99", "max"),
            map(float, np.quantile(valid, [0, .5, .95, .99, 1]))))
    if np.isfinite(absolute).all():
        norm = np.linalg.norm(reference.astype(np.complex128).ravel())
        result["relative_l2"] = float(np.linalg.norm(absolute.ravel()) / norm) if norm else None
    return result, absolute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="exclusive GPU UUID")
    parser.add_argument("--binary", type=Path, default=Path("build-e13-cufftdx/stockham_subgraph_bench"))
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--variants", nargs="+", type=int, choices=range(4), default=list(range(4)))
    parser.add_argument("--patterns", nargs="+", choices=["random", "impulse", "tone"], default=["random"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--chain", type=int, default=32)
    args = parser.parse_args()
    if min(args.batches) < 1 or max(args.batches) > 4096:
        parser.error("this bounded local-core experiment supports B=1..4096")
    if not 1 <= args.trials <= 101 or not 1 <= args.chain <= 1024:
        parser.error("trials must be 1..101 and graph chain 1..1024")
    repo = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve(strict=True)
    output = args.output.resolve()
    state = gpu_state(args.gpu)
    output.mkdir(parents=True, exist_ok=False)
    sources = ["apps/stockham_subgraph_bench.cu", "src/xd_stockham_pair_fft.cu",
               "src/xd_stockham_pair_fft.hpp", "src/fft_radix4_codelet.cuh",
               "src/xd_pair_fft.cu", "scripts/run_stage_pair_experiment.py"]
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "gpu": state,
        "protocol": "serial_graph_chain_v2", "shape": [32, 32], "dtype": "complex64",
        "normalization": "unnormalized in both directions", "input_order": "natural",
        "output_order": "natural", "reference": "scipy.fft CPU complex64, workers=1",
        "threshold_policy": "report only; no finite-error acceptance threshold",
        "compile_mode": os.environ.get("CUBUTTERFLY_COMPILE_MODE", "research"),
        "source_sha256": {p: digest(repo / p) for p in sources},
        "binary": str(binary), "binary_sha256": digest(binary),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "batches": args.batches, "variants": args.variants, "patterns": args.patterns,
        "trials": args.trials, "chain": args.chain, "status": "measuring", "rows": [],
    }
    save(output / "summary.json", manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    env.setdefault("CUBUTTERFLY_COMPILE_MODE", "research")
    pending = []
    try:
        for batch in args.batches:
            for inverse in [0, 1]:
                for pattern in args.patterns:
                    for variant in args.variants:
                        gpu_state(args.gpu)
                        name = f"b{batch}_i{inverse}_v{variant}_{pattern}"
                        prefix = output / name
                        cmd = [str(binary), str(batch), "5", "20", str(inverse), str(variant),
                               str(prefix), str(args.trials), str(args.chain), pattern]
                        run = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
                        prefix.with_suffix(".stdout").write_text(run.stdout)
                        prefix.with_suffix(".stderr").write_text(run.stderr)
                        if run.returncode not in [0, 2]:
                            raise RuntimeError(f"{name} failed ({run.returncode}): {run.stderr[-2000:]}")
                        row = json.loads(run.stdout)
                        row.update(case=name, command=cmd, exit_code=run.returncode,
                                   status="measured_pending_cpu_reference")
                        manifest["rows"].append(row)
                        pending.append((prefix, row))
                        save(output / "summary.json", manifest)
                        print(f"{name}: {row['speedup_vs_cufft']:.3f}x vs cuFFT", flush=True)
        manifest["status"] = "cpu_reference"
        save(output / "summary.json", manifest)
    except Exception as error:
        manifest["status"] = "interrupted"
        manifest["failure"] = str(error)
    # Finish checking already measured rows even if the GPU became unavailable.
    for prefix, row in pending:
        count = row["planes"] * 1024
        data = np.fromfile(str(prefix) + ".input.c64", dtype=np.complex64)
        if data.size != count:
            raise RuntimeError(f"invalid input dump for {row['case']}")
        data = data.reshape(row["planes"], 32, 32)
        transform = fft.ifft2 if row["inverse"] else fft.fft2
        reference = transform(data, axes=(-2, -1), norm="forward" if row["inverse"] else "backward", workers=1)
        if reference.dtype != np.complex64 or not np.isfinite(reference).all():
            raise RuntimeError("CPU reference must remain finite and in the requested precision")
        distributions = {}
        for provider, metrics in row["providers"].items():
            values = np.fromfile(str(prefix) + f".{provider}.c64", dtype=np.complex64)
            if values.size != count:
                raise RuntimeError(f"invalid {provider} dump for {row['case']}")
            metrics["cpu_fp32_error"], absolute = errors(values.reshape(data.shape), reference)
            distributions[provider] = absolute
        distributions["cpu_reference"] = reference
        np.savez_compressed(str(prefix) + ".errors.npz", **distributions)
        row["status"] = "error_distribution_reported"
        save(output / "summary.json", manifest)
    if manifest["status"] != "interrupted":
        manifest["status"] = "measured_and_errors_reported"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(output / "summary.json", manifest)
    print(f"{manifest['status']}: {output / 'summary.json'}", flush=True)
    return 1 if manifest["status"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
