#!/usr/bin/env python3
"""Eight packed real FFT cases with same-precision CPU torch.fft references.

This reports numeric distributions without an acceptance tolerance. Mapping,
input preservation and repeatability of the recorded execution remain checked.
No performance timing or additional search matrix is included.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

SHAPES = ((2048, 33), (64, 8, 33))
BATCH = 3
SCALE = 0.875


def prepare(torch, shape, precision, inverse):
    real_dtype = torch.float64 if precision == "fp64" else torch.float32
    complex_dtype = torch.complex128 if precision == "fp64" else torch.complex64
    dimensions = tuple(range(-len(shape), 0))
    seed = 20261013 + sum(shape) + BATCH
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((BATCH, *shape), generator=generator, dtype=real_dtype, device="cpu")
    values = torch.fft.rfftn(real, dim=dimensions) if inverse else real
    reference = (torch.fft.irfftn(values, s=shape, dim=dimensions) if inverse
                 else torch.fft.rfftn(values, dim=dimensions)) * SCALE
    if values.dtype != (complex_dtype if inverse else real_dtype):
        raise RuntimeError("CPU input precision mismatch")
    if reference.dtype != (real_dtype if inverse else complex_dtype):
        raise RuntimeError("CPU FFT reference promoted precision")
    return values, reference, dimensions, seed


def errors(torch, measured, reference):
    accumulator = torch.complex128 if reference.is_complex() else torch.float64
    difference = (measured.cpu().to(accumulator) - reference.to(accumulator)).abs()
    denominator = reference.to(accumulator).abs().square().sum().sqrt().item()
    numerator = difference.square().sum().sqrt().item()
    relative_l2 = numerator / denominator if denominator else (0.0 if not numerator else None)
    quantiles = torch.quantile(difference.flatten(), torch.tensor(
        [0.5, 0.9, 0.99, 0.999, 1.0], dtype=torch.float64))
    scalar = lambda value: value if value is not None and math.isfinite(value) else None
    return {"finite": bool(torch.isfinite(measured).all().item()),
            "relative_l2": scalar(relative_l2), "max_abs": scalar(difference.max().item()),
            "absolute_quantiles": dict(zip(("p50", "p90", "p99", "p999", "max"),
                                            map(scalar, quantiles.tolist())))}


def same_bytes(torch, left, right):
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def physical_mapping(mapping):
    physical = mapping["physical"]
    if physical.get("axis_exchange_policy") != 4:
        raise RuntimeError("requested packed core did not select exchange policy 4")
    axes = physical["axes"]
    if not any(axis.get("packed") for axis in axes):
        raise RuntimeError("packed real case did not execute a permutation boundary")
    for axis in axes:
        if (axis.get("core") != "cufftdx-axis-tiled" or
                axis.get("core_suffix") != 1 or axis.get("core_exchange_policy") != 0):
            raise RuntimeError("packed real case did not execute the contiguous cuFFTDx core")
    return physical


def packed_replay(shell, shape, precision, inverse, boundary_workspace):
    # The existing EXPLICIT JSON interface verifies this full physical record
    # against the lowered plan. No new family parser or library API is needed.
    compact = list(shape)
    compact[-1] = compact[-1] // 2 + 1
    compact_bytes = BATCH * math.prod(compact) * (16 if precision == "fp64" else 8)
    axis_order = list(range(len(shape) - 1))
    physical_order = list(range(len(shape)))
    dimensions = compact.copy()
    axes = []
    for logical_axis in axis_order:
        position = physical_order.index(logical_axis)
        length = dimensions[position]
        suffix = math.prod(dimensions[position + 1:])
        packed = suffix != 1
        permutation = [i for i in range(len(shape)) if i != position] + [position] if packed else []
        batch_lines = BATCH * math.prod(dimensions) // length
        lines_per_block = 1 if length > 1024 else min(32, min(256 * 8,
            2048 if precision == "fp64" else 4096) // length)
        axes.append({"axis": logical_axis, "length": length, "stride": 1,
            "batch": batch_lines, "idist": length, "prefix_loops": 1, "packed": packed,
            "core": "cufftdx-axis-tiled", "elements_per_thread": min(length, 8),
            "ffts_per_block": lines_per_block, "cooperative_lanes": 0,
            "twiddle_policy": "core-defined", "pair_rows": 0, "pair_planes": 0,
            "paired_axis": 0, "axis_suffix": suffix,
            "boundary": "permute_nd" if packed else "none", "input_axis_order": physical_order.copy(),
            "permutation": permutation, "io_exchange": "packed-contiguous-register" if packed else "contiguous-register",
            "core_exchange_policy": 0, "core_suffix": 1, "fft_launches": 1,
            "permutation_launches": int(packed)})
        if packed:
            dimensions = [dimensions[i] for i in permutation]
            physical_order = [physical_order[i] for i in permutation]
    restore = [physical_order.index(i) for i in range(len(shape))]
    if restore == list(range(len(shape))):
        restore = []
    # The rank-1 public boundary plan has exactly the same R2C/C2R batch.
    # Its C2R workspace includes the compact preservation buffer already.
    real_work_aligned = boundary_workspace - (compact_bytes if inverse else 0)
    data_offset = real_work_aligned + (compact_bytes if inverse else 0)
    workspace = ((data_offset + 255) // 256) * 256 + compact_bytes
    permutations = sum(axis["packed"] for axis in axes) + int(bool(restore))
    physical = {"kind": "xd-real-fft", "rank": len(shape), "axis_exchange_policy": 4,
        "batch": BATCH, "inverse": inverse, "batch_tile": 0, "batch_chunks": 1,
        "normalize_inverse": inverse, "fp64": precision == "fp64", "complex_only": False,
        "packed_axes": True, "c2r_input_copy_elision": inverse,
        "c2r_input_boundary": "direct-compact" if inverse else "not-applicable",
        "cost": {"prefix_launches": len(axes), "permutation_launches": permutations,
            "restore_permutation_launches": int(bool(restore)), "scale_kernel_launches": int(inverse),
            "c2r_d2d_copy_launches": 0, "real_boundary_launches": 1,
            "total_launches": len(axes) + permutations + int(inverse) + 1,
            "workspace_bytes": workspace, "compact_boundary_bytes": compact_bytes,
            "source": "lowering-instance"},
        "fused_axes": True, "axis_ept": 8, "register_lanes": 4, "pair_layout": 0,
        "paired_axes": False, "pair_groups": 0, "axis_core": "cufftdx-block", "pair_core": "thread",
        "layout_policy": "retain-axis-layout-v1", "output_scale": SCALE, "axis_order": axis_order,
        "extents": list(shape), "compact_last_extent": compact[-1],
        "axis_resources": [{"core": "cufftdx-block", "ept": 8, "register_lanes": 4,
                            "block_cap": 0, "exchange": 4} for _ in axis_order],
        "axes": axes, "output_permutation": restore}
    return {"schema_version": shell["schema_version"], "kind": shell["kind"],
            "contract": shell["contract"], "target": shell["target"],
            "lowering": ("xd-compact-c2r" if inverse else "xd-compact-r2c") + "[axis-packed-cufftdx]",
            "physical": physical}


def run_case(torch, api, library, args, shape, precision, inverse):
    values_cpu, reference, dimensions, seed = prepare(torch, shape, precision, inverse)
    values = values_cpu.cuda()
    original = values.clone()
    output = torch.empty(reference.shape, dtype=reference.dtype, device="cuda")
    transform = api.Transform(storage=getattr(api.DataType, precision.upper()),
        compute=getattr(api.ComputeType, precision.upper()), extents=list(shape), batch=BATCH,
        inverse=inverse, normalize_inverse=True, fft_output_scale=SCALE,
        fft_type=api.FftType.C2R if inverse else api.FftType.R2C,
        execution_mode=api.ExecutionMode.XD)
    stream = torch.cuda.current_stream().cuda_stream
    with api.Plan(library, transform, stream=stream, cache_path=args.cache) as selected:
        shell = json.loads(selected.mapping_json())
    if args.physical_mappings:
        matches = [physical for physical in args.physical_records
            if physical["extents"] == list(shape) and physical["fp64"] == (precision == "fp64")
            and physical["inverse"] == inverse]
        if len(matches) != 1:
            raise RuntimeError("expected exactly one exported physical mapping for this case")
        physical = dict(matches[0])
        # Public semantic normalization clears the inverse-only flag on R2C.
        physical["normalize_inverse"] = inverse
        boundary_workspace = physical["cost"]["workspace_bytes"] - physical["cost"]["compact_boundary_bytes"]
        replay = packed_replay(shell, shape, precision, inverse, boundary_workspace)
        if replay["physical"] != physical:
            raise RuntimeError("exported physical mapping disagrees with the bounded case configuration")
        replay["physical"] = physical
    else:
        boundary = api.Transform(storage=transform.storage, compute=transform.compute,
            extents=[shape[-1]], batch=BATCH * math.prod(shape[:-1]), inverse=inverse,
            normalize_inverse=True, fft_output_scale=SCALE, fft_type=transform.fft_type)
        with api.Plan(library, boundary, stream=stream, cache_path=args.cache) as boundary_plan:
            boundary_workspace = boundary_plan.workspace_size()
        replay = packed_replay(shell, shape, precision, inverse, boundary_workspace)
    options = api.PlanOptions(policy=api.AlgorithmPolicy.EXPLICIT,
        mapping=json.dumps(replay), execution_mode=api.ExecutionMode.XD)
    with api.Plan(library, transform, options, stream=torch.cuda.current_stream().cuda_stream,
                  cache_path=args.cache) as plan:
        mapping = json.loads(plan.mapping_json())
        physical_mapping(mapping)
        size = plan.workspace_size()
        if size:
            plan.set_workspace(torch.empty(size, dtype=torch.uint8, device="cuda"), size)
        plan.execute(values, output)
        torch.cuda.synchronize()
        first = output.cpu()
        first_preserved = same_bytes(torch, values, original)
        # Rebind a fresh workspace to cover the existing ownership boundary.
        if size:
            plan.set_workspace(torch.empty(size, dtype=torch.uint8, device="cuda"), size)
        plan.execute(values, output)
        torch.cuda.synchronize()
        repeated = output.cpu()
        unchanged = first_preserved and same_bytes(torch, values, original)
        if not unchanged:
            raise RuntimeError("public packed real plan mutated the caller input")
        if not same_bytes(torch, first, repeated):
            raise RuntimeError("public packed real plan changed output after workspace rebinding")
        native = (torch.fft.irfftn(values, s=shape, dim=dimensions) if inverse
                  else torch.fft.rfftn(values, dim=dimensions)) * SCALE
        torch.cuda.synchronize()
        if not same_bytes(torch, values, original):
            raise RuntimeError("native comparison mutated the shared input")
        return {"shape": list(shape), "batch": BATCH, "precision": precision,
                "inverse": inverse, "normalize_inverse": True, "output_scale": SCALE,
                "seed": seed, "input_generator": "CPU torch.randn in storage precision",
                "input_dtype": str(values_cpu.dtype), "reference_dtype": str(reference.dtype),
                "main_oracle": "same-storage-and-compute-precision CPU torch.fft",
                "forced_diagnostic": "public complete mapping JSON EXPLICIT",
                "mapping": mapping, "algorithm": plan.algorithm_name(),
                "selection_reason": plan.selection_reason(), "input_unchanged": unchanged,
                "workspace_rebind_same_output": True,
                "cubutterfly_vs_cpu": errors(torch, first, reference),
                "native_gpu_vs_cpu": errors(torch, native, reference),
                "accuracy_status": "distribution_only_no_accepted_tolerance"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--physical-mappings", type=Path,
        help="exact physical JSON lines exported by the matching library's XdRealFftPlan")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.physical_records = [json.loads(line) for line in args.physical_mappings.read_text().splitlines()
                            if line.strip()] if args.physical_mappings else []
    if not args.prepare_only and (args.library is None or args.output is None):
        parser.error("GPU execution requires --library and --output")
    import torch
    torch.set_num_threads(1)
    cases = [(shape, precision, inverse) for shape in SHAPES
             for precision in ("fp32", "fp64") for inverse in (False, True)]
    if args.prepare_only:
        for shape, precision, inverse in cases:
            values, reference, _, seed = prepare(torch, shape, precision, inverse)
            assert torch.isfinite(reference).all()
            replay = packed_replay({"schema_version": 1, "kind": "xd", "contract": {}, "target": {}},
                shape, precision, inverse, BATCH * math.prod(shape[:-1]) * (shape[-1] // 2 + 1) *
                (16 if precision == "fp64" else 8) if inverse else 0)
            physical_mapping(replay)
            print(json.dumps({"shape": shape, "precision": precision, "inverse": inverse,
                "seed": seed, "input_dtype": str(values.dtype),
                "reference_dtype": str(reference.dtype), "cpu_prepared": True}))
        return
    import cubutterfly as api
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    library = api.Library(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache or args.output.with_suffix(".cache")
    report = {"library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__, "gpu": props.name, "uuid": str(props.uuid),
        "physical_mapping_source": str(args.physical_mappings.resolve()) if args.physical_mappings else "reconstructed",
        "physical_mapping_source_sha256": hashlib.sha256(args.physical_mappings.read_bytes()).hexdigest()
            if args.physical_mappings else None,
        "memory_bytes": props.total_memory, "scope": "eight existing packed real boundary cases",
        "main_oracle": "same-precision CPU torch.fft", "rows": [], "status": "running"}
    try:
        for case in cases:
            row = run_case(torch, api, library, args, *case)
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in (
                "shape", "precision", "inverse", "cubutterfly_vs_cpu", "native_gpu_vs_cpu")}), flush=True)
        report["status"] = "completed_distributions_reported_no_numeric_gate"
    except Exception as exc:
        report.update(status="structural_or_execution_failure", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
