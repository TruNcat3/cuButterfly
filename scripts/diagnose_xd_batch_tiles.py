#!/usr/bin/env python3
"""Measure complete xD FFT calls with explicit batch tiles.

Each family/tile row constructs its plans before correctness and timing.  One
complete callback walks every batch chunk in order; CUDA graph timing captures
20 such callbacks and replays the graph seven times.  Individual chunks are
never timed, so the result describes the cross-axis plan plus its batch
schedule rather than a sum of independent launches.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_xd as bench  # noqa: E402
import check_xd as xd  # noqa: E402
from diagnose_xd_fft import NativeCufft  # noqa: E402


FAMILIES = (
    "axis-tiled-cufftdx",
    "axis-tiled-register",
    "pair-tiled-cooperative-register",
    "cufft-native",
)
GRAPH_CAPTURE_CALLS = 20
GRAPH_REPLAYS = 7
GRAPH_WARMUP = 10


class UnsupportedFamily(RuntimeError):
    """The requested family is unavailable for this shape/precision/target."""


class NativePlan:
    """A movable-pointer wrapper around one native cuFFT batch plan."""

    def __init__(self, torch: Any, shape: Sequence[int], batch: int,
                 fp64: bool, stream: Any, source: Any, output: Any):
        self._native = NativeCufft(
            torch, shape, batch, fp64, False, stream, source, output,
        )
        self._torch = torch
        self.mapping = {
            "kind": "native-cufft",
            "backend": "libcufft.so.11",
            "shape": [int(value) for value in shape],
            "batch": int(batch),
            "precision": "fp64" if fp64 else "fp32",
            "direction": "forward",
        }
        self.metadata = {
            "algorithm": "cufft-native",
            "selection_reason": "independent native cuFFT plan",
            "selection_source": None,
            "workspace_bytes": int(self._native.workspace.numel()),
            "mapping_json": self.mapping,
        }

    def execute(self, source: Any, output: Any) -> None:
        # cuFFT plans are batch-size specific, but their input/output pointers
        # may change between sequential chunks on the same stream.
        self._native.source = source
        self._native.output = output
        self._native()

    def close(self) -> None:
        self._native.close()


class ApiPlan:
    """A public C-API plan with metadata retained for every chunk."""

    def __init__(self, plan: xd.Plan, metadata: dict[str, Any]):
        self.plan = plan
        self.metadata = metadata
        self.mapping = metadata["mapping_json"]

    def execute(self, source: Any, output: Any) -> None:
        self.plan.execute(source, output)

    def close(self) -> None:
        self.plan.close()


@dataclass
class ChunkBinding:
    index: int
    start: int
    stop: int
    source: Any
    output: Any
    owner: Any


def _chunk_ranges(batch: int, tile: int) -> list[tuple[int, int]]:
    if tile == 0:
        return [(0, batch)]
    return [(start, min(start + tile, batch))
            for start in range(0, batch, tile)]


def _is_unsupported(error: BaseException) -> bool:
    return isinstance(error, xd.UnsupportedCase) or (
        isinstance(error, xd.CApiError) and error.status == xd.STATUS_NOT_SUPPORTED
    )


def _make_api_plan(api: xd.Api, handle: xd.Handle, torch: Any,
                   shape: Sequence[int], batch: int, precision: str,
                   family: str) -> ApiPlan:
    fp64 = precision == "fp64"
    plan = xd.make_plan(
        api,
        handle,
        torch,
        op=xd.OP_FFT,
        storage=xd.DATA_COMPLEX_FP64 if fp64 else xd.DATA_COMPLEX_FP32,
        compute=xd.COMPUTE_FP64 if fp64 else xd.COMPUTE_FP32,
        shape=tuple(int(value) for value in shape),
        batch=int(batch),
        mode=xd.EXEC_XD,
        direction=xd.FORWARD,
        normalize_inverse=True,
        mapping="xd:" + family,
    )
    return ApiPlan(plan, bench._metadata(plan))


def _make_owner(api: xd.Api, handle: xd.Handle, torch: Any,
                shape: Sequence[int], batch: int, precision: str,
                family: str, source: Any, output: Any, stream: Any) -> Any:
    if family == "cufft-native":
        try:
            return NativePlan(
                torch, shape, batch, precision == "fp64", stream, source, output,
            )
        except OSError as error:
            raise UnsupportedFamily(str(error)) from error
    try:
        return _make_api_plan(api, handle, torch, shape, batch, precision, family)
    except BaseException as error:
        if _is_unsupported(error):
            raise UnsupportedFamily(str(error)) from error
        raise


def _prepare_bindings(api: xd.Api, handle: xd.Handle, torch: Any,
                      source: Any, output: Any, shape: Sequence[int],
                      batch: int, tile: int, precision: str,
                      family: str, stream: Any) -> tuple[list[ChunkBinding], list[Any]]:
    ranges = _chunk_ranges(batch, tile)
    sizes = list(dict.fromkeys(stop - start for start, stop in ranges))
    owners: dict[int, Any] = {}
    owned: list[Any] = []
    try:
        # Every distinct plan batch size is built before the first callback.
        for size in sizes:
            owner = _make_owner(
                api, handle, torch, shape, size, precision, family,
                source[:size], output[:size], stream,
            )
            owners[size] = owner
            owned.append(owner)
        bindings = [
            ChunkBinding(
                index=index,
                start=start,
                stop=stop,
                source=source[start:stop],
                output=output[start:stop],
                owner=owners[stop - start],
            )
            for index, (start, stop) in enumerate(ranges)
        ]
        return bindings, owned
    except BaseException:
        for owner in owned:
            owner.close()
        raise


def _chunk_metadata(bindings: Sequence[ChunkBinding]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    seen_sizes: dict[int, int] = {}
    for binding in bindings:
        size = binding.stop - binding.start
        reuse_index = seen_sizes.get(size, 0)
        seen_sizes[size] = reuse_index + 1
        owner_metadata = binding.owner.metadata
        metadata.append({
            "chunk": int(binding.index),
            "start": int(binding.start),
            "stop": int(binding.stop),
            "batch": int(size),
            "plan_reuse_index": int(reuse_index),
            "algorithm": owner_metadata.get("algorithm"),
            "selection_source": owner_metadata.get("selection_source"),
            "workspace_bytes": int(owner_metadata.get("workspace_bytes", 0)),
            "mapping": owner_metadata.get("mapping_json"),
        })
    return metadata


def _profile_graph(torch: Any, callback: Callable[[], None], stream: Any,
                   label: str) -> dict[str, Any]:
    """Profile graph replays while retaining the complete-call granularity."""
    if not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("torch.cuda.CUDAGraph is unavailable")
    with torch.cuda.stream(stream):
        for _ in range(GRAPH_WARMUP):
            callback()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            callback()

    cudart = torch.cuda.cudart()
    profiler_started = False
    range_pushed = False
    try:
        cudart.cudaProfilerStart()
        profiler_started = True
        torch.cuda.nvtx.range_push(label)
        range_pushed = True
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize()
    finally:
        if range_pushed:
            torch.cuda.nvtx.range_pop()
        if profiler_started:
            cudart.cudaProfilerStop()
    return {
        "graph_used": True,
        "profile_only": True,
        "graph_capture_calls": 1,
        "graph_replays": 1,
        "measurement_unit": "one_complete_tiled_call",
        "timing": "cuda_profiler_range; no event latency reported",
    }


def _time_complete_call(torch: Any, callback: Callable[[], None], stream: Any,
                        profile_only: bool, label: str) -> dict[str, Any]:
    if profile_only:
        return _profile_graph(torch, callback, stream, label)
    timing = bench._graph_median_ms(
        torch,
        callback,
        stream,
        warmup=GRAPH_WARMUP,
        replays=GRAPH_REPLAYS,
        capture_calls=GRAPH_CAPTURE_CALLS,
    )
    timing["measurement_unit"] = "one_complete_tiled_call"
    timing["block_timing"] = "not_collected"
    samples_per_transform = timing.pop("samples_ms")
    per_transform_ms = timing.pop("median_ms")
    timing["samples_complete_call_ms"] = [
        sample * GRAPH_CAPTURE_CALLS for sample in samples_per_transform
    ]
    timing["median_graph_ms_per_transform"] = per_transform_ms
    timing["median_complete_call_ms"] = per_transform_ms * GRAPH_CAPTURE_CALLS
    return timing


def _check_correctness(torch: Any, output: Any, expected_torch: Any,
                       expected_oracle: Any, precision: str) -> dict[str, Any]:
    atol = 5e-4 if precision == "fp32" else 5e-11
    rtol = atol
    torch_error = bench._max_error(torch, output, expected_torch)
    oracle_error = bench._max_error(torch, output, expected_oracle)
    bench._check_error(torch_error, atol=atol, rtol=rtol, label="torch reference")
    bench._check_error(oracle_error, atol=atol, rtol=rtol, label="FP64 oracle")
    return {
        "torch_reference": torch_error,
        "fp64_oracle": oracle_error,
        "oracle_policy": "FP64 FFT cast to output precision" if precision == "fp32"
            else "FP64 FFT",
    }


def _run_case(api: xd.Api, handle: xd.Handle, torch: Any,
              source: Any, source_before: Any,
              expected_torch: Any, expected_oracle: Any,
              shape: Sequence[int], batch: int, tile: int, precision: str,
              family: str, stream: Any, profile_only: bool) -> dict[str, Any]:
    output = torch.empty_like(source)
    bindings, owners = _prepare_bindings(
        api, handle, torch, source, output, shape, batch, tile, precision,
        family, stream,
    )

    def execute_complete() -> None:
        for binding in bindings:
            binding.owner.execute(binding.source, binding.output)

    try:
        execute_complete()
        torch.cuda.synchronize()
        correctness = _check_correctness(
            torch, output, expected_torch, expected_oracle, precision,
        )
        if not torch.equal(source, source_before):
            raise AssertionError("tiled FFT modified its input")
        timing = _time_complete_call(
            torch,
            execute_complete,
            stream,
            profile_only,
            f"xd_batch_tiles/{family}/tile{tile}",
        )
        torch.cuda.synchronize()
        if not torch.equal(source, source_before):
            raise AssertionError("tiled FFT modified its input during timing")
        return {
            "status": "passed",
            "family": family,
            "tile": int(tile),
            "batch": int(batch),
            "chunk_count": len(bindings),
            "chunk_plan_batches": sorted({binding.stop - binding.start for binding in bindings}),
            "chunks": _chunk_metadata(bindings),
            "correctness": correctness,
            "input_unchanged": True,
            "timing": timing,
            "workspace_policy": "one plan/workspace per distinct chunk size; serial same-stream reuse",
        }
    finally:
        for owner in owners:
            owner.close()


def _loaded_cuda_libraries() -> list[str]:
    try:
        lines = Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return sorted({
        line.split()[-1]
        for line in lines
        if "libcufft" in line or "libcudart" in line
    })


def _cufft_version() -> int | None:
    try:
        library = ct.CDLL("libcufft.so.11")
        version = ct.c_int()
        status = library.cufftGetVersion(ct.byref(version))
        return int(version.value) if status == 0 else None
    except OSError:
        return None


def _mapping_runtime_fingerprint(mapping: Any) -> str | None:
    if not isinstance(mapping, dict):
        return None
    target = mapping.get("target")
    if isinstance(target, dict):
        value = target.get("runtime_fingerprint")
        if value:
            return str(value)
    return None


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--shape", required=True, type=int, nargs="+")
    parser.add_argument("--batch", required=True, type=int)
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--lowerings", nargs="+", default=list(FAMILIES),
                        help="explicit family names passed to the xD C API")
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 4, 8, 10])
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.shape or any(value <= 0 for value in args.shape):
        parser.error("--shape values must be positive")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if not args.tiles or any(tile < 0 for tile in args.tiles):
        parser.error("--tiles values must be nonnegative; 0 means whole batch")
    args.tiles = list(dict.fromkeys(args.tiles))
    args.library = args.library.resolve()
    return args


def _write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "arguments": {
            "library": str(args.library),
            "shape": list(args.shape),
            "batch": int(args.batch),
            "precision": args.precision,
            "lowerings": list(args.lowerings),
            "tiles": list(args.tiles),
            "profile_only": bool(args.profile_only),
        },
        "protocol": {
            "schedule": "one callback executes every batch chunk; each plan completes all axes",
            "whole_tile": 0,
            "tail_policy": "positive tiles split batch and create one plan per distinct full/tail size",
            "graph_capture_calls": GRAPH_CAPTURE_CALLS,
            "graph_replays": GRAPH_REPLAYS,
            "graph_warmup": GRAPH_WARMUP,
            "profile_only_graph": "one complete callback capture and one replay",
            "block_timing": "disabled; only complete-call timing is reported",
            "plan_construction_outside_timing": True,
            "workspace_policy": "same non-default stream; serial plan/workspace reuse",
            "input_policy": "one fixed random input shared by every family and tile",
            "correctness_policy": "torch reference plus FP64 oracle before timing",
            "selected_lowering": "not requested; every family is explicit",
        },
        "rows": [],
    }

    try:
        import torch
    except Exception as error:
        report.update(status="error", error=f"torch import failed: {error}")
        _write_report(report, args.output)
        return 2
    if not torch.cuda.is_available():
        report.update(status="error", error="CUDA is unavailable")
        _write_report(report, args.output)
        return 2

    try:
        library = args.library
        if not library.is_file():
            raise FileNotFoundError(f"library does not exist: {library}")
        report["library"] = str(library)
        report["library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
        report["repository"] = bench._repo_identity()
        report["gpu"] = bench._gpu_identity(torch)
        report["torch"] = str(torch.__version__)
        report["torch_cuda"] = str(torch.version.cuda)
        report["compile_mode"] = os.getenv("CUBUTTERFLY_COMPILE_MODE")
        report["cufft_version"] = _cufft_version()

        api = xd.Api(library)
        stream = torch.cuda.Stream(device=torch.cuda.current_device())
        with torch.cuda.stream(stream), xd.Handle(api, int(stream.cuda_stream)) as handle:
            torch.manual_seed(42)
            fp64 = args.precision == "fp64"
            dtype = torch.complex128 if fp64 else torch.complex64
            source = bench._make_fft_source(torch, args.shape, args.batch, dtype)
            source_before = source.clone()
            dimensions = tuple(range(1, len(args.shape) + 1))
            expected_torch = torch.fft.fftn(source, dim=dimensions, norm="backward")
            # Use a high-precision oracle for FP32 so cancellation-sensitive
            # cases do not pass solely because the reference rounded early.
            expected_oracle = torch.fft.fftn(
                source.to(torch.complex128), dim=dimensions, norm="backward",
            )
            torch.cuda.synchronize()

            for family in args.lowerings:
                for tile in args.tiles:
                    print(f"{family} tile={tile} starting", flush=True)
                    try:
                        row = _run_case(
                            api,
                            handle,
                            torch,
                            source,
                            source_before,
                            expected_torch,
                            expected_oracle,
                            args.shape,
                            args.batch,
                            tile,
                            args.precision,
                            family,
                            stream,
                            args.profile_only,
                        )
                    except UnsupportedFamily as error:
                        row = {
                            "status": "unsupported",
                            "family": family,
                            "tile": int(tile),
                            "batch": int(args.batch),
                            "reason": str(error),
                        }
                    except Exception as error:
                        row = {
                            "status": "failed",
                            "family": family,
                            "tile": int(tile),
                            "batch": int(args.batch),
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        }
                    report["rows"].append(row)
                    print(
                        family,
                        f"tile={tile}",
                        row["status"],
                        row.get("timing", {}).get("median_complete_call_ms", ""),
                        row.get("error", row.get("reason", "")),
                        flush=True,
                    )

            if not torch.equal(source, source_before):
                raise AssertionError("shared source changed across batch-tile cases")
    except Exception as error:
        report["harness_error"] = str(error)
        report["harness_traceback"] = traceback.format_exc()

    report["loaded_cuda_libraries"] = _loaded_cuda_libraries()
    fingerprints = set()
    for row in report["rows"]:
        for chunk in row.get("chunks", []):
            fingerprint = _mapping_runtime_fingerprint(chunk.get("mapping"))
            if fingerprint:
                fingerprints.add(fingerprint)
    report["build_gpu_fingerprint"] = {
        "library_sha256": report.get("library_sha256"),
        "git_commit": report.get("repository", {}).get("git_commit"),
        "working_tree_dirty": report.get("repository", {}).get("working_tree_dirty"),
        "gpu": report.get("gpu"),
        "runtime_fingerprints": sorted(fingerprints),
        "loaded_cufft_soname": "libcufft.so.11",
        "cufft_version": report.get("cufft_version"),
    }
    failed = sum(row.get("status") == "failed" for row in report["rows"])
    unsupported = sum(row.get("status") == "unsupported" for row in report["rows"])
    passed = sum(row.get("status") == "passed" for row in report["rows"])
    if "harness_error" in report:
        report["status"] = "failed"
    else:
        report["status"] = "passed" if failed == 0 else "failed"
    report["summary"] = {
        "rows": len(report["rows"]),
        "passed": passed,
        "unsupported": unsupported,
        "failed": failed,
    }
    _write_report(report, args.output)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
