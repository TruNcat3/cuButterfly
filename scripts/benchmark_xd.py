#!/usr/bin/env python3
"""xD resident acceptance benchmark through the public C API.

The script intentionally reuses the ctypes contract and references from
``tests/check_xd.py``.  Plan construction, coefficient/JIT work, workspace
binding, and warmup are outside the measured region.  CUDA graph capture keeps
the repeated C-API launch sequence out of the host timing path: each graph
contains 20 transform calls and the report stores the median replay latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import check_xd as xd  # noqa: E402  (the test module is the API contract)


FFT_SHAPES: tuple[tuple[int, ...], ...] = (
    (16, 16),
    (32, 64),
    (64, 64),
    (128, 128),
    (8, 8, 8),
    (16, 32, 32),
)
REAL_SHAPES: tuple[tuple[int, ...], ...] = ((8, 16), (4, 8, 8))
FFT_BATCHES = (1, 16)


def _shape_name(shape: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _repo_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {}
    try:
        identity["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        paths = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).split(b"\0")
        digest = hashlib.sha256()
        for raw_path in paths:
            if not raw_path:
                continue
            path = ROOT / os.fsdecode(raw_path)
            if not path.is_file():
                continue
            digest.update(raw_path)
            digest.update(b"\0")
            digest.update(path.read_bytes())
        identity["repo_sha256"] = digest.hexdigest()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
        identity["working_tree_dirty"] = bool(status.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        identity["identity_error"] = str(exc)
    return identity


def _gpu_identity(torch: Any) -> dict[str, Any]:
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    result: dict[str, Any] = {
        "index": index,
        "model": str(properties.name),
        "vram_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "multiprocessors": int(properties.multi_processor_count),
    }
    uuid = getattr(properties, "uuid", None)
    if uuid is not None:
        result["uuid"] = str(uuid)

    # CUDA_VISIBLE_DEVICES remaps the logical torch index, so querying
    # nvidia-smi with ``--id=<index>`` can describe a different physical GPU.
    # Prefer the UUID returned by torch; fall back to an unambiguous visible
    # device token, and omit redundant nvidia-smi fields when neither exists.
    query_id: str | None = None
    if uuid is not None and str(uuid).strip():
        query_id = str(uuid).strip()
        if not query_id.startswith("GPU-"):
            query_id = "GPU-" + query_id
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            query_id = str(index)
        else:
            tokens = [token.strip() for token in visible.split(",")]
            if 0 <= index < len(tokens) and tokens[index] and tokens[index] != "-1":
                query_id = tokens[index]
    try:
        if query_id is None:
            raise ValueError("physical GPU mapping is unavailable")
        query = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={query_id}",
                "--query-gpu=uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if query:
            fields = [field.strip() for field in query.split(",", 2)]
            if len(fields) >= 3:
                result.setdefault("uuid", fields[0])
                result.setdefault("model_nvidia_smi", fields[1])
                result["vram_mib_nvidia_smi"] = int(fields[2])
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    return result


def _metadata(plan: xd.Plan) -> dict[str, Any]:
    raw_mapping = plan.metadata.get("mapping_json", "")
    try:
        mapping: Any = json.loads(raw_mapping)
    except (TypeError, json.JSONDecodeError):
        mapping = raw_mapping
    return {
        "algorithm": plan.metadata.get("algorithm"),
        "selection_reason": plan.metadata.get("selection_reason"),
        "selection_source": plan.metadata.get("selection_source"),
        "workspace_bytes": int(plan.metadata.get("workspace_bytes", 0)),
        "mapping_json": mapping,
    }


def _max_error(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    if actual_cpu.numel() == 0:
        return {"max_abs": 0.0, "scale": 1.0}
    # CUDA/CPU UInt subtraction is not implemented consistently across
    # PyTorch versions.  All acceptance integers here fit exactly in fp64,
    # so cast only integer operands before computing an absolute difference.
    if not (actual_cpu.is_floating_point() or actual_cpu.is_complex()):
        actual_cpu = actual_cpu.to(torch.float64)
        expected_cpu = expected_cpu.to(torch.float64)
    difference = torch.abs(actual_cpu - expected_cpu)
    return {
        "max_abs": float(difference.max().item()),
        "scale": max(1.0, float(torch.abs(expected_cpu).max().item())),
    }


def _check_error(error: dict[str, float], *, atol: float, rtol: float, label: str) -> None:
    tolerance = float(atol + rtol * error["scale"])
    error["tolerance"] = tolerance
    if error["max_abs"] > tolerance:
        raise AssertionError(
            f"{label}: max_abs={error['max_abs']:g}, tolerance={tolerance:g}"
        )


def _graph_median_ms(
    torch: Any,
    callback: Callable[[], None],
    stream: Any,
    *,
    warmup: int,
    replays: int,
    capture_calls: int,
) -> dict[str, Any]:
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("torch.cuda.CUDAGraph is unavailable")
    if capture_calls != 20:
        raise ValueError("xD acceptance requires exactly 20 calls per CUDA graph capture")

    # The caller supplies one non-default stream shared by the C API handle,
    # warmup, capture, and replay.  Recent PyTorch releases reject capture on
    # the legacy default stream; using a second stream here would silently
    # leave ctypes-launched kernels outside the graph.
    with torch.cuda.stream(stream):
        for _ in range(max(0, warmup)):
            callback()
    torch.cuda.synchronize()

    # This pre-capture call materializes lazy library state and allocator
    # ownership before capture.  It is deliberately outside timing.
    with torch.cuda.stream(stream):
        callback()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(capture_calls):
                callback()

    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup // 2)):
            graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(max(1, replays)):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        with torch.cuda.stream(stream):
            graph.replay()
        stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)) / capture_calls)
    return {
        "graph_used": True,
        "graph_capture_stream": "shared_non_default",
        "graph_capture_calls": capture_calls,
        "graph_replays": max(1, replays),
        "samples_ms": samples,
        "median_ms": float(statistics.median(samples)),
    }


def _event_median_ms(
    torch: Any,
    callback: Callable[[], None],
    stream: Any,
    *,
    warmup: int,
    repeat: int,
) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        callback()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        callback()
        stop.record(stream)
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    return {
        "graph_used": False,
        "samples_ms": samples,
        "median_ms": float(statistics.median(samples)),
    }


def _make_fft_source(torch: Any, shape: Sequence[int], batch: int, dtype: Any) -> Any:
    full_shape = (int(batch), *[int(value) for value in shape])
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    return torch.complex(
        torch.randn(full_shape, dtype=real_dtype, device="cuda"),
        torch.randn(full_shape, dtype=real_dtype, device="cuda"),
    )


def _fft_spec(shape: Sequence[int], batch: int, storage: int, compute: int, mode: int) -> dict[str, Any]:
    return {
        "op": xd.OP_FFT,
        "storage": storage,
        "compute": compute,
        "shape": tuple(int(value) for value in shape),
        "batch": int(batch),
        "mode": mode,
        "direction": xd.FORWARD,
        "normalize_inverse": True,
    }


def _benchmark_fft_case(
    api: xd.Api,
    handle: xd.Handle,
    torch: Any,
    shape: Sequence[int],
    batch: int,
    *,
    dtype: Any,
    storage: int,
    compute: int,
    warmup: int,
    replays: int,
    capture_calls: int,
    use_graph: bool,
) -> dict[str, Any]:
    source = _make_fft_source(torch, shape, batch, dtype)
    dims = tuple(range(1, len(shape) + 1))
    expected = torch.fft.fftn(source, dim=dims, norm="backward")
    torch.cuda.synchronize()
    result: dict[str, Any] = {
        "kind": "fft_c2c",
        "shape": list(shape),
        "batch": int(batch),
        "dtype": str(dtype),
        "protocol": {
            "warmup": int(warmup),
            "graph_capture_calls": int(capture_calls),
            "graph_replays": int(replays),
            "timing": "median_cuda_event_ms_per_transform",
            "plan_construction_outside_timing": True,
        },
        "plans": [],
    }

    for mode, mode_name in ((xd.EXEC_XD, "xd"), (xd.EXEC_AUTO, "auto"),
                            (xd.EXEC_COMPOSITE, "composite")):
        output = torch.empty_like(source)
        plan = xd.make_plan(
            api, handle, torch,
            **_fft_spec(shape, batch, storage, compute, mode),
        )
        try:
            plan.execute(source, output)
            torch.cuda.synchronize()
            error = _max_error(torch, output, expected)
            _check_error(
                error,
                atol=5e-4 if dtype == torch.complex64 else 5e-11,
                rtol=5e-4 if dtype == torch.complex64 else 5e-11,
                label=f"FFT {mode_name}",
            )
            callback = lambda plan=plan: plan.execute(source, output)
            timing = (
                _graph_median_ms(
                    torch, callback, torch.cuda.current_stream(), warmup=warmup,
                    replays=replays, capture_calls=capture_calls,
                )
                if use_graph else _event_median_ms(
                    torch, callback, torch.cuda.current_stream(), warmup=warmup,
                    repeat=max(1, replays * capture_calls),
                )
            )
            result["plans"].append({
                "mode": mode_name,
                "status": "passed",
                "correctness": error,
                **_metadata(plan),
                **timing,
            })
        finally:
            plan.close()

    torch_output: Any = None

    def torch_callback() -> None:
        nonlocal torch_output
        torch_output = torch.fft.fftn(source, dim=dims, norm="backward")

    torch_timing = (
        _graph_median_ms(
            torch, torch_callback, torch.cuda.current_stream(), warmup=warmup,
            replays=replays, capture_calls=capture_calls,
        )
        if use_graph else _event_median_ms(
            torch, torch_callback, torch.cuda.current_stream(), warmup=warmup,
            repeat=max(1, replays * capture_calls),
        )
    )
    torch_error = _max_error(torch, torch_output, expected)
    _check_error(
        torch_error,
        atol=1e-6 if dtype == torch.complex64 else 1e-12,
        rtol=1e-6 if dtype == torch.complex64 else 1e-12,
        label="torch FFT reference",
    )
    result["torch"] = {
        "vendor": "PyTorch/cuFFT",
        "status": "passed",
        "correctness": torch_error,
        **torch_timing,
    }
    result["status"] = "passed"
    return result


def _random_uint(torch: Any, shape: Sequence[int], dtype: Any) -> Any:
    # uint tensors do not have a portable torch.randn implementation.
    return torch.randint(0, 1 << 30, tuple(shape), dtype=torch.int64, device="cuda").to(dtype)


def _real_source(torch: Any, shape: Sequence[int], batch: int) -> Any:
    return torch.randn((int(batch), *[int(value) for value in shape]),
                       dtype=torch.float32, device="cuda")


def _run_plan_once(api: xd.Api, handle: xd.Handle, torch: Any,
                   spec: dict[str, Any], source: Any) -> tuple[Any, dict[str, Any]]:
    output = torch.empty_like(source)
    plan = xd.make_plan(api, handle, torch, **spec)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        return output, _metadata(plan)
    finally:
        plan.close()


def _non_fft_case(
    api: xd.Api,
    handle: xd.Handle,
    torch: Any,
    *,
    op: int,
    op_name: str,
    shape: Sequence[int],
    batch: int,
    storage: int,
    compute: int,
    source: Any,
    matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    modulus: int | None = None,
) -> dict[str, Any]:
    base = {
        "op": op,
        "storage": storage,
        "compute": compute,
        "shape": tuple(int(value) for value in shape),
        "batch": int(batch),
        "direction": xd.FORWARD,
        "normalize_inverse": True,
    }
    if matrices is not None:
        base["matrices"] = matrices
    if modulus is not None:
        base["modulus"] = int(modulus)
        base["word_bits"] = 32 if storage == xd.DATA_UINT32 else 64

    outputs: dict[str, Any] = {}
    plans: dict[str, xd.Plan] = {}
    plan_metadata: dict[str, Any] = {}
    for mode, mode_name in ((xd.EXEC_XD, "xd"), (xd.EXEC_COMPOSITE, "composite")):
        output = torch.empty_like(source)
        plan = xd.make_plan(api, handle, torch, **dict(base, mode=mode))
        plans[mode_name] = plan
        plan.execute(source, output)
        torch.cuda.synchronize()
        outputs[mode_name] = output
        plan_metadata[mode_name] = _metadata(plan)

    try:
        if op == xd.OP_FWHT:
            expected = xd._fwht_reference(
                torch, source.cpu(), range(1, len(shape) + 1),
            ).to("cuda")
            atol, rtol = 5e-4, 5e-4
        elif op in (xd.OP_SUBSET_ZETA, xd.OP_SUPERSET_ZETA):
            import numpy as np

            expected_cpu = xd._zeta_reference(
                source.cpu().numpy(), (batch, *shape), op, False,
            )
            expected = torch.from_numpy(np.asarray(expected_cpu)).to("cuda").reshape(source.shape)
            atol, rtol = 0.0, 0.0
        elif op == xd.OP_NTT:
            import numpy as np

            expected_values = xd._ntt_reference(
                source.cpu().reshape(-1).tolist(), shape, batch, int(modulus), False,
            )
            expected = torch.tensor(
                expected_values, dtype=source.dtype, device="cuda",
            ).reshape(source.shape)
            atol, rtol = 0.0, 0.0
        elif op == xd.OP_STRUCTURED_2X2:
            expected = xd._structured_reference(
                torch, source.cpu(), shape, matrices, False,
            ).to("cuda")
            atol, rtol = 7e-4, 7e-4
        else:
            raise ValueError(f"unsupported non-FFT operator {op}")

        errors: dict[str, Any] = {}
        for mode_name, output in outputs.items():
            error = _max_error(torch, output, expected)
            _check_error(error, atol=atol, rtol=rtol, label=f"{op_name} {mode_name}")
            errors[mode_name] = error
        difference = _max_error(torch, outputs["xd"], outputs["composite"])
        _check_error(difference, atol=atol, rtol=rtol, label=f"{op_name} xD/composite")

        # Keep each plan alive while timing its public execute call.  The
        # caller has bound both plans to the shared non-default stream.
        for mode_name, plan in plans.items():
            plan_metadata[mode_name]["timing"] = _graph_median_ms(
                torch,
                lambda plan=plan, output=outputs[mode_name]: plan.execute(source, output),
                torch.cuda.current_stream(),
                warmup=10,
                replays=5,
                capture_calls=20,
            )
        return {
            "kind": op_name,
            "shape": list(shape),
            "batch": int(batch),
            "status": "passed",
            "correctness": errors,
            "xd_vs_composite": difference,
            "plans": plan_metadata,
        }
    finally:
        for plan in plans.values():
            plan.close()


def _structured_matrices(shape: Sequence[int]) -> list[list[list[float]]]:
    matrices: list[list[list[float]]] = []
    for axis, extent in enumerate(shape):
        stages = int(extent).bit_length() - 1
        axis_matrices: list[list[float]] = []
        for stage in range(stages):
            scale = 1.0 + 0.03125 * (axis + 1) + 0.015625 * (stage + 1)
            axis_matrices.append([
                scale, 0.125 * (stage + 1),
                -0.0625 * (axis + 1), 1.0 / scale,
            ])
        matrices.append(axis_matrices)
    return matrices


def _run_non_fft_cases(api: xd.Api, handle: xd.Handle, torch: Any,
                       quick: bool) -> list[dict[str, Any]]:
    shape_cases = ((8, 16),) if quick else REAL_SHAPES
    results: list[dict[str, Any]] = []
    for shape in shape_cases:
        batch = 4 if len(shape) == 3 else 16
        full_shape = (batch, *shape)
        try:
            source = _real_source(torch, shape, batch)
            results.append(_non_fft_case(
                api, handle, torch, op=xd.OP_FWHT, op_name="fwht",
                shape=shape, batch=batch, storage=xd.DATA_FP32,
                compute=xd.COMPUTE_FP32, source=source,
            ))
        except Exception as exc:
            results.append({"kind": "fwht", "shape": list(shape), "batch": batch,
                            "status": "failed", "error": str(exc),
                            "traceback": traceback.format_exc()})

        uint_source = _random_uint(torch, full_shape, torch.uint32)
        for op, name in ((xd.OP_SUBSET_ZETA, "subset_zeta"),
                         (xd.OP_SUPERSET_ZETA, "superset_zeta")):
            try:
                results.append(_non_fft_case(
                    api, handle, torch, op=op, op_name=name,
                    shape=shape, batch=batch, storage=xd.DATA_UINT32,
                    compute=xd.COMPUTE_UINT32, source=uint_source,
                ))
            except Exception as exc:
                results.append({"kind": name, "shape": list(shape), "batch": batch,
                                "status": "failed", "error": str(exc),
                                "traceback": traceback.format_exc()})

        # 998244353 admits roots for every power-of-two axis in these cases.
        # PyTorch 2.5 has no CUDA remainder kernel for UInt32.  Keep the
        # generated values in a supported signed type for host-independent
        # reduction, then cast to each NTT storage type below.
        ntt_source = torch.remainder(uint_source.to(torch.int64), 998244353)
        for storage, compute, dtype_name in ((xd.DATA_UINT32, xd.COMPUTE_UINT32, "uint32"),
                                             (xd.DATA_UINT64, xd.COMPUTE_UINT64, "uint64")):
            source_for_ntt = ntt_source.to(torch.uint64 if storage == xd.DATA_UINT64 else torch.uint32)
            try:
                results.append(_non_fft_case(
                    api, handle, torch, op=xd.OP_NTT, op_name=f"ntt_{dtype_name}",
                    shape=shape, batch=batch, storage=storage, compute=compute,
                    source=source_for_ntt, modulus=998244353,
                ))
            except Exception as exc:
                results.append({"kind": f"ntt_{dtype_name}", "shape": list(shape),
                                "batch": batch, "status": "failed", "error": str(exc),
                                "traceback": traceback.format_exc()})

        matrices = _structured_matrices(shape)
        try:
            results.append(_non_fft_case(
                api, handle, torch, op=xd.OP_STRUCTURED_2X2,
                op_name="structured_2x2", shape=shape, batch=batch,
                storage=xd.DATA_FP32, compute=xd.COMPUTE_FP32,
                source=source, matrices=matrices,
            ))
        except Exception as exc:
            results.append({"kind": "structured_2x2", "shape": list(shape),
                            "batch": batch, "status": "failed", "error": str(exc),
                            "traceback": traceback.format_exc()})
    return results


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="path to libcuntt.so")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output", "-o", type=pathlib.Path)
    parser.add_argument("--quick", action="store_true", help="run a reduced shape matrix")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--capture-calls", type=int, default=20)
    parser.add_argument("--no-graph", action="store_true", help="use event timing without CUDA graphs")
    parser.add_argument("--skip-non-fft", action="store_true")
    parser.add_argument("--skip-fft", action="store_true",
                        help="run only the non-FFT operator matrix")
    return parser.parse_args(argv)


def _write_report(report: dict[str, Any], path: pathlib.Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report: dict[str, Any] = {
        "script": str(pathlib.Path(__file__).resolve()),
        "timestamp": time.time(),
        "repository": _repo_identity(),
        "protocol": {
            "quick": bool(args.quick),
            "warmup": int(args.warmup),
            "graph_capture_calls": int(args.capture_calls),
            "graph_replays": int(args.replays),
            "graph_required_by_default": True,
            "comparison": "same input; torch.fft.fftn norm=backward",
            "skip_fft": bool(args.skip_fft),
            "skip_non_fft": bool(args.skip_non_fft),
        },
        "fft": [],
        "non_fft": [],
    }
    try:
        import torch
    except Exception as exc:
        report.update(status="error", error=f"torch import failed: {exc}")
        _write_report(report, args.output)
        return 2
    if not torch.cuda.is_available():
        report.update(status="error", error="CUDA is unavailable")
        _write_report(report, args.output)
        return 2
    if args.device is not None:
        torch.cuda.set_device(args.device)
    report["gpu"] = _gpu_identity(torch)
    report["torch"] = str(torch.__version__)
    report["torch_cuda"] = str(torch.version.cuda)

    failures = 0
    try:
        library = xd._find_library(args.library)
        report["library"] = str(library)
        api = xd.Api(library)
        # Bind the C API handle to the same non-default stream used by every
        # plan call and CUDA graph operation.  A separate capture stream would
        # make ctypes launches bypass the graph.
        benchmark_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        stream = int(benchmark_stream.cuda_stream)
        shapes = ((16, 16), (8, 8, 8)) if args.quick else FFT_SHAPES
        with xd.Handle(api, stream) as handle:
            with torch.cuda.stream(benchmark_stream):
                if not args.skip_fft:
                    for shape in shapes:
                        for batch in ((1,) if args.quick else FFT_BATCHES):
                            for dtype, storage, compute, dtype_name in (
                                (torch.complex64, xd.DATA_COMPLEX_FP32, xd.COMPUTE_FP32, "complex64"),
                                (torch.complex128, xd.DATA_COMPLEX_FP64, xd.COMPUTE_FP64, "complex128"),
                            ):
                                name = f"fft_{_shape_name(shape)}_b{batch}_{dtype_name}"
                                try:
                                    row = _benchmark_fft_case(
                                        api, handle, torch, shape, batch,
                                        dtype=dtype, storage=storage, compute=compute,
                                        warmup=max(0, args.warmup), replays=max(1, args.replays),
                                        capture_calls=args.capture_calls,
                                        use_graph=not args.no_graph,
                                    )
                                    row["name"] = name
                                    report["fft"].append(row)
                                except Exception as exc:
                                    failures += 1
                                    report["fft"].append({
                                        "name": name, "kind": "fft_c2c", "shape": list(shape),
                                        "batch": batch, "dtype": dtype_name, "status": "failed",
                                        "error": str(exc), "traceback": traceback.format_exc(),
                                    })
                if not args.skip_non_fft:
                    report["non_fft"] = _run_non_fft_cases(api, handle, torch, args.quick)
    except Exception as exc:
        failures += 1
        report["harness_error"] = str(exc)
        report["harness_traceback"] = traceback.format_exc()

    failures += sum(1 for row in report["non_fft"] if row.get("status") == "failed")
    report["status"] = "failed" if failures else "passed"
    report["summary"] = {
        "fft_cases": len(report["fft"]),
        "non_fft_cases": len(report["non_fft"]),
        "failed": failures,
        "passed": sum(row.get("status") == "passed" for row in report["fft"] + report["non_fft"]),
    }
    _write_report(report, args.output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
