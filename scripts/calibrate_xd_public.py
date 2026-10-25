#!/usr/bin/env python3
"""Workload-local public MEASURE+DEFAULT real xD calibration and cache retest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = (((64, 64), 8), ((32, 32), 64), ((8, 8, 8), 16))
SEED = 20261012
WARMUP, REPEATS, TRIALS = 20, 50, 3


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_library_identity(path, expected):
    if sha256(path) != expected:
        raise RuntimeError("library changed during calibration/retest")


def checkpoint(path, document):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


@contextmanager
def policy(value):
    name = "CUBUTTERFLY_E13_ALGORITHM_POLICY"
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def cache_gate(calibrated, replayed):
    """Require both public plans and the entire physical/semantic identity."""
    def indexed(records):
        if len(records) != 2 or {r["direction"] for r in records} != {"forward", "inverse"}:
            raise ValueError("exactly one forward and inverse plan are required")
        return {r["direction"]: r for r in records}
    before, after = indexed(calibrated), indexed(replayed)
    for direction in ("forward", "inverse"):
        first, second = before[direction], after[direction]
        if second["selection_source"] != "user_cache":
            raise ValueError(f"{direction}: DEFAULT did not select user_cache")
        for record in (first, second):
            if not isinstance(record.get("mapping", {}).get("physical"), dict):
                raise ValueError(f"{direction}: missing physical mapping")
        if first["mapping"] != second["mapping"]:
            raise ValueError(f"{direction}: calibration/replay mapping mismatch")
        if first["transform"] != second["transform"]:
            raise ValueError(f"{direction}: transform mismatch")
    return {"passed": True, "selection_source": "user_cache", "identical_mapping": True}


def exclusive_gpu(uuid, own_pid=None):
    own_pid = os.getpid() if own_pid is None else own_pid
    output = subprocess.check_output([
        "nvidia-smi", f"--id={uuid}", "--query-compute-apps=gpu_uuid,pid,process_name",
        "--format=csv,noheader,nounits"], text=True)
    foreign = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",", 2)]
        if len(fields) != 3 or fields[0] != uuid:
            raise RuntimeError(f"cannot validate GPU process row: {line}")
        if int(fields[1]) != own_pid:
            foreign.append(line)
    if foreign:
        raise RuntimeError("GPU has other compute processes: " + "; ".join(foreign))
    return {"gpu_uuid": uuid, "own_pid_ignored": own_pid, "foreign_compute_processes": []}


def errors(actual, expected):
    import torch
    actual = actual.detach().cpu().to(expected.dtype)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = actual - expected
    denominator = float(torch.linalg.vector_norm(expected.reshape(-1)))
    return {"finite": finite,
            "max_abs": float(difference.abs().max()) if finite else None,
            "relative_l2": (float(torch.linalg.vector_norm(difference.reshape(-1))) / denominator
                            if finite and denominator else None),
            "float_pass_tolerance": None}


def input_exact(before, after):
    import torch
    return bool(torch.equal(before.view(torch.uint8), after.view(torch.uint8)))


def inputs(shape, batch, device):
    import torch
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    cpu = torch.randn((batch, *shape), generator=generator, dtype=torch.float32)
    dims = tuple(range(-len(shape), 0))
    reference_forward = torch.fft.rfftn(cpu.double(), dim=dims, norm="forward")
    half = reference_forward.to(torch.complex64).contiguous()
    reference_inverse = torch.fft.irfftn(half.to(torch.complex128), s=shape, dim=dims, norm="forward")
    return cpu.to(device), half.to(device), reference_forward, reference_inverse


def validate(adapter, shape, batch, device):
    import torch
    x, half, forward_ref, inverse_ref = inputs(shape, batch, device)
    x_before, half_before = x.clone(), half.clone()
    forward = adapter.rfftn(x)
    inverse = adapter.irfftn(half)
    dims = tuple(range(-len(shape), 0))
    torch_forward = torch.fft.rfftn(x, dim=dims, norm="forward")
    torch_inverse = torch.fft.irfftn(half, s=shape, dim=dims, norm="forward")
    report = {"forward": errors(forward, forward_ref), "inverse": errors(inverse, inverse_ref),
              "torch_fp32": {"forward": errors(torch_forward, forward_ref),
                             "inverse": errors(torch_inverse, inverse_ref)},
              "input_exact": {"forward": input_exact(x_before, x),
                              "inverse": input_exact(half_before, half)}}
    return report


def valid(report):
    return all(report[d]["finite"] and report["torch_fp32"][d]["finite"]
               and report["input_exact"][d] for d in ("forward", "inverse"))


def elapsed(function):
    import torch
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        result = function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / REPEATS


def compare(adapter, shape, batch, device, uuid):
    import torch
    x, half, _, _ = inputs(shape, batch, device)
    x_before, half_before = x.clone(), half.clone()
    dims = tuple(range(-len(shape), 0))
    torch_forward = lambda: torch.fft.rfftn(x, dim=dims, norm="forward")
    torch_inverse = lambda: torch.fft.irfftn(half, s=shape, dim=dims, norm="forward")
    calls = {
        "forward": (lambda: adapter.rfftn(x), torch_forward),
        "inverse": (lambda: adapter.irfftn(half), torch_inverse),
        "pair": (lambda: adapter.irfftn(adapter.rfftn(x)),
                 lambda: torch.fft.irfftn(torch_forward(), s=shape, dim=dims, norm="forward")),
    }
    guard_before = exclusive_gpu(uuid)
    results = {}
    for operation, functions in calls.items():
        samples = {"public": [], "torch": []}
        orders = []
        for trial in range(TRIALS):
            order = (0, 1) if trial % 2 == 0 else (1, 0)
            orders.append([("public", "torch")[index] for index in order])
            for index in order:
                samples[("public", "torch")[index]].append(elapsed(functions[index]))
        public, baseline = (statistics.median(samples[key]) for key in ("public", "torch"))
        results[operation] = {"public_us": public, "torch_us": baseline,
                              "speedup_torch_over_public": baseline / public,
                              "trials_us": samples, "orders": orders, "measured": True}
    torch.cuda.synchronize()
    guard_after = exclusive_gpu(uuid)
    if not input_exact(x_before, x) or not input_exact(half_before, half):
        raise RuntimeError("timing changed a public input")
    return {"operations": results, "guard_before": guard_before, "guard_after": guard_after,
            "input_exact": True}


def write_csv(path, document):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shape", "batch", "status", "operation", "public_us", "torch_us", "speedup_torch_over_public"))
        for row in document["cases"]:
            for operation, timing in row.get("timing", {}).get("operations", {}).items():
                writer.writerow(("x".join(map(str, row["shape"])), row["batch"], row["status"], operation,
                                 timing["public_us"], timing["torch_us"], timing["speedup_torch_over_public"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="fresh JSON checkpoint path")
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--dry-run", action="store_true", help="print bounded protocol without importing CUDA")
    args = parser.parse_args(argv)
    output, library = args.output.resolve(), args.library.resolve()
    cache, csv_path = output.with_suffix(".cache.json"), output.with_suffix(".csv")
    for target in (output, cache, csv_path, output.with_name(output.name + ".tmp")):
        if target.exists():
            parser.error(f"refusing existing output: {target}")
    document = {"scope": "workload-local public MEASURE+DEFAULT cache calibration; not installation stage cost model or all xD",
                "status": "prepared", "library": str(library), "gpu_uuid": args.gpu_uuid,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "cache_path": str(cache), "seed": SEED, "dtype": "fp32", "norm": "forward",
                "compile_mode": "research",
                "protocol": {"method": "CUDA events around full public adapter calls including allocation",
                             "warmup": WARMUP, "repeats": REPEATS, "trials": TRIALS,
                             "pair": "actual consecutive forward/inverse calls", "reuse_buffers": False,
                             "input": "same seeded GPU-resident real input and FP32 compact spectrum from CPU FP64 reference",
                             "integer_output_exact": "not applicable to floating-point R2C/C2R; input bytes checked separately",
                             "excluded": ["reference", "initialization", "compilation", "plan construction"],
                             "float_pass_tolerance": None},
                "cases": [{"shape": list(shape), "batch": batch, "status": "pending"} for shape, batch in CASES]}
    if args.dry_run:
        print(json.dumps(document, indent=2))
        return 0
    if not library.is_file():
        parser.error(f"library not found: {library}")
    if not output.parent.is_dir():
        parser.error("output parent must already exist")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["CUBUTTERFLY_COMPILE_MODE"] = "research"
    import torch
    sys.path.insert(0, str(ROOT / "results" / "e13_assets"))
    from cubutterfly_real_fft import RealFFTAdapter
    document["library_sha256"] = sha256(library)
    document["torch_version"] = torch.__version__
    document["guard_initial"] = exclusive_gpu(args.gpu_uuid)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    document["gpu_name"] = properties.name
    observed = str(getattr(properties, "uuid", ""))
    if observed and observed.removeprefix("GPU-") != args.gpu_uuid.removeprefix("GPU-"):
        raise RuntimeError(f"torch UUID {observed} differs from requested GPU {args.gpu_uuid}")
    checkpoint(output, document)
    try:
        with policy("measure"):
            for row in document["cases"]:
                require_library_identity(library, document["library_sha256"])
                exclusive_gpu(args.gpu_uuid)
                start = time.monotonic()
                adapter = RealFFTAdapter(library, row["shape"], row["batch"], device, cache_path=cache)
                try:
                    row["calibration"] = {"policy": "measure", "plans": adapter.plan_metadata(),
                                          "plan_construction_seconds": time.monotonic() - start,
                                          "validation": validate(adapter, row["shape"], row["batch"], device)}
                finally:
                    torch.cuda.synchronize()
                    adapter.close()
                row["status"] = "calibrated"
                checkpoint(output, document)
        document["calibrated_cache_sha256"] = sha256(cache)
        document["status"] = "all-three-calibrated"
        checkpoint(output, document)
        with policy("default"):
            for row in document["cases"]:
                require_library_identity(library, document["library_sha256"])
                adapter = RealFFTAdapter(library, row["shape"], row["batch"], device, cache_path=cache)
                try:
                    row["replay"] = {"policy": "default", "plans": adapter.plan_metadata()}
                    try:
                        row["cache_gate"] = cache_gate(row["calibration"]["plans"], row["replay"]["plans"])
                    except ValueError as exc:
                        row["status"], row["cache_gate"] = "cache-gate-failed", {"passed": False, "error": str(exc)}
                        continue
                    row["replay"]["validation"] = validate(adapter, row["shape"], row["batch"], device)
                    if not valid(row["calibration"]["validation"]) or not valid(row["replay"]["validation"]):
                        row["status"] = "finite-or-input-gate-failed"
                        continue
                    row["timing"] = compare(adapter, row["shape"], row["batch"], device, args.gpu_uuid)
                    row["status"] = "measured"
                finally:
                    torch.cuda.synchronize()
                    adapter.close()
                    checkpoint(output, document)
        require_library_identity(library, document["library_sha256"])
        document["final_cache_sha256"] = sha256(cache)
        document["status"] = "complete" if all(r["status"] == "measured" for r in document["cases"]) else "incomplete"
    except Exception as exc:
        document["status"], document["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        checkpoint(output, document)
        write_csv(csv_path, document)
    print(json.dumps({"status": document["status"], "output": str(output), "csv": str(csv_path),
                      "cases": [{"shape": r["shape"], "batch": r["batch"], "status": r["status"],
                                 "operations": r.get("timing", {}).get("operations", {})} for r in document["cases"]]}, indent=2))
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
