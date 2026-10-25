#!/usr/bin/env python3
"""Bounded public xD coverage and replay check; no implicit floating threshold.

Preparation, plan selection and graph capture are excluded from timing. Both
backends reuse output buffers. This is a diagnostic matrix, not the complete
application comparison; the filter runner measures padding and crop separately.
"""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(torch, call, repeats, samples):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / repeats)
    return {"median_us": statistics.median(values), "samples_us": values}


def error(torch, output, reference):
    import numpy as np
    difference = (output.to(torch.complex128) - reference.to(torch.complex128)).abs()
    finite = bool(torch.isfinite(output).all().item())
    return {"finite": finite, "max_abs": float(difference.max().item()),
            "relative_l2": float((difference.norm() / reference.norm().clamp_min(1e-300)).item()),
            "absolute_quantiles": dict(zip(("0.5", "0.9", "0.99", "1.0"),
                map(float, np.quantile(difference.cpu().numpy(), [.5, .9, .99, 1.0]))))}


def run_case(args, torch, api, library, shape, batch, precision, kind, inverse, family):
    real_dtype = torch.float64 if precision == "fp64" else torch.float32
    complex_dtype = torch.complex128 if precision == "fp64" else torch.complex64
    generator = torch.Generator(device="cuda").manual_seed(20261013 + sum(shape) + batch)
    if kind == "real":
        real = torch.randn((batch, *shape), generator=generator, device="cuda", dtype=real_dtype)
        values = torch.fft.rfftn(real, dim=(-2, -1)) if inverse else real
        reference = (torch.fft.irfftn(values, s=shape, dim=(-2, -1)) if inverse
                     else torch.fft.rfftn(values, dim=(-2, -1)))
        transform = api.Transform(storage=getattr(api.DataType, precision.upper()),
            compute=getattr(api.ComputeType, precision.upper()), extents=list(shape), batch=batch,
            inverse=inverse, fft_type=api.FftType.C2R if inverse else api.FftType.R2C,
            execution_mode=api.ExecutionMode.XD)
        baseline = torch.fft.irfftn if inverse else torch.fft.rfftn
    else:
        values = torch.randn((batch, *shape), generator=generator, device="cuda", dtype=complex_dtype)
        baseline = torch.fft.ifftn if inverse else torch.fft.fftn
        reference = baseline(values, dim=(-2, -1))
        transform = api.Transform(storage=getattr(api.DataType, "COMPLEX_" + precision.upper()),
            compute=getattr(api.ComputeType, precision.upper()), extents=list(shape), batch=batch,
            inverse=inverse, execution_mode=api.ExecutionMode.XD)
    original = values.clone()
    # Same-precision CPU FFT is the numerical reference. GPU vendor outputs
    # are a performance comparator, not ground truth; torch CPU preserves the
    # requested FP32/FP64 precision (unlike NumPy's default FFT promotion).
    reference_cpu = baseline(values.cpu(), s=shape, dim=(-2, -1))
    assert reference_cpu.dtype == reference.dtype
    reference = reference_cpu.to(device=values.device)
    output, torch_output = torch.empty_like(reference), torch.empty_like(reference)
    options = api.PlanOptions(policy=api.AlgorithmPolicy.EXPLICIT if family else api.AlgorithmPolicy.MEASURE,
                              mapping="xd:" + family if family else "", execution_mode=api.ExecutionMode.XD)
    started = time.perf_counter()
    with api.Plan(library, transform, options, stream=torch.cuda.current_stream().cuda_stream,
                  cache_path=args.cache) as plan:
        workspace = plan.workspace_size()
        if workspace:
            plan.set_workspace(torch.empty(workspace, device="cuda", dtype=torch.uint8), workspace)
        setup = time.perf_counter() - started
        plan.execute(values, output)
        torch.cuda.synchronize()
        row = {"shape": shape, "batch": batch, "precision": precision, "kind": kind,
               "inverse": inverse, "requested_family": family or "public-measure",
               "diagnostic_forced_candidate": bool(family), "setup_seconds": setup,
               "mapping": json.loads(plan.mapping_json()), "selection_reason": plan.selection_reason(),
               "error": error(torch, output, reference), "input_unchanged": torch.equal(values, original),
               "reference": {"backend": "torch.fft CPU", "dtype": str(reference_cpu.dtype),
                             "precision_promotion": False},
               "accuracy_status": "distribution_only_no_accepted_tolerance"}
        selected_source = ct.c_int()
        library._call("PlanGetSelectionSource", plan.native_handle, ct.byref(selected_source))
        row["selection_source"] = selected_source.value
        if not family and selected_source.value != 2:
            args.failed_row = row
            raise RuntimeError("MEASURE did not perform runtime measurement; inspect selection_reason "
                               "and tuning_budget_bytes before requiring measured-cache replay")
        call = lambda: plan.execute(values, output)
        torch_call = lambda: baseline(values, s=shape, dim=(-2, -1), out=torch_output)
        methods = [("cubutterfly", call), ("torch", torch_call)]
        if args.process_trial_index % 2:
            methods.reverse()
        row["method_order"] = [name for name, _ in methods]
        for name, function in methods:
            row[name] = measure(torch, function, args.repeats, args.samples)
        row["torch_error_against_cpu"] = error(torch, torch_output, reference)
        row["speedup_torch_over_cubutterfly"] = row["torch"]["median_us"] / row["cubutterfly"]["median_us"]
        # Public DEFAULT must consume the same complete-workload cache entry.
        if not family:
            with api.Plan(library, transform, stream=torch.cuda.current_stream().cuda_stream,
                          cache_path=args.cache) as replay:
                size = replay.workspace_size()
                if size:
                    replay.set_workspace(torch.empty(size, dtype=torch.uint8, device="cuda"), size)
                replay_output = torch.empty_like(output)
                replay.execute(values, replay_output)
                torch.cuda.synchronize()
                source = ct.c_int()
                library._call("PlanGetSelectionSource", replay.native_handle, ct.byref(source))
                row["cache_replay"] = {"source": source.value,
                    "same_mapping": json.loads(replay.mapping_json()) == row["mapping"],
                    "same_output": torch.equal(output, replay_output)}
                if source.value != 1 or not all(row["cache_replay"][key] for key in ("same_mapping", "same_output")):
                    row["cache_replay"]["mapping"] = json.loads(replay.mapping_json())
                    row["cache_replay"]["output_difference"] = error(torch, replay_output, output)
                    args.failed_row = row
                    raise RuntimeError("DEFAULT did not replay the measured public plan")
        if not row["input_unchanged"] or not row["error"]["finite"]:
            raise RuntimeError("non-finite output or input mutation: " + json.dumps(row))
        return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("xd_coverage.json"))
    p.add_argument("--cache", type=Path)
    p.add_argument("--matrix", choices=("coverage", "filter"), default="coverage")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--process-trial-index", type=int, default=0)
    p.add_argument("--batches", type=int, nargs="+")
    p.add_argument("--direction", choices=("both", "forward", "inverse"), default="both")
    p.add_argument("--tune-max-bytes", type=int,
                   default=int(os.environ.get("CUBUTTERFLY_XD_TUNE_MAX_BYTES", 8 * 1024**3)),
                   help="explicit measurement allocation budget for this bounded matrix (default 8 GiB)")
    args = p.parse_args()
    if min(args.repeats, args.samples) < 1:
        p.error("repeats and samples must be positive")
    if args.tune_max_bytes <= 0:
        p.error("tune-max-bytes must be positive")
    os.environ["CUBUTTERFLY_XD_TUNE_MAX_BYTES"] = str(args.tune_max_bytes)
    if not args.output.is_absolute():
        args.output = Path(os.environ.get("XD_REPORT_DIR", ".")) / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache or args.output.with_suffix(".cache")
    import torch
    import cubutterfly as api
    torch.cuda.set_device(0)
    library = api.Library(args.library)
    query = library._dll.cubutterflyPlanGetSelectionSource
    query.argtypes, query.restype = [ct.c_void_p, ct.POINTER(ct.c_int)], ct.c_int
    props = torch.cuda.get_device_properties(0)
    report = {"library": str(args.library.resolve()), "library_sha256": digest(args.library),
              "runner_sha256": digest(__file__), "gpu": props.name, "uuid": str(props.uuid),
              "memory_bytes": props.total_memory, "torch": torch.__version__,
              "tuning_budget_bytes": args.tune_max_bytes,
              "scope": "public complete xD; output buffers reused; setup excluded",
              "matrix": args.matrix, "rows": [], "status": "running"}
    if args.matrix == "coverage":
        cases = [(shape, 2, precision, "complex", inv, family)
                 for shape in ((2048, 32), (32, 2048), (4096, 16), (16, 4096))
                 for precision in ("fp32", "fp64") for inv in (False, True)
                 for family in ("axis-tiled-cufftdx", "axis-tiled-cufftdx-ept16",
                                "axis-packed-cufftdx", "axis-packed-cufftdx-ept16")]
    else:
        cases = [((2048, 2048), b, "fp32", "real", inverse, None)
                 for b in (1, 16) for inverse in (False, True)]
    cases = [case for case in cases
             if (args.batches is None or case[1] in args.batches)
             and (args.direction == "both" or case[4] == (args.direction == "inverse"))]
    if not cases:
        p.error("filters selected no matrix cases")
    try:
        for case in cases:
            row = run_case(args, torch, api, library, *case)
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ("shape", "batch", "precision", "inverse", "requested_family", "speedup_torch_over_cubutterfly")}), flush=True)
        report["status"] = "measured_errors_reported"
    except Exception as exc:
        report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        if hasattr(args, "failed_row"):
            report["failed_row"] = args.failed_row
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
