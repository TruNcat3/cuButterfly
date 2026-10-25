#!/usr/bin/env python3
"""Benchmark compact real FFT paths against the legacy adapter and cuFFT.

The adapter in ``results/e13_assets/cubutterfly_real_fft.py`` has two
deliberate modes: the default native compact R2C/C2R implementation and the
``native_real=False`` C2C composition retained for regression comparisons.
This script exercises both modes with the same real input as PyTorch's
``torch.fft.rfftn``/``irfftn`` reference.  Plan construction, workspace
allocation, warmup, and CUDA graph capture are reported separately and are
excluded from steady-state CUDA event measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "results" / "e13_assets"
if str(ASSET_DIR) not in sys.path:
    sys.path.insert(0, str(ASSET_DIR))

from cubutterfly_real_fft import RealFFTAdapter  # noqa: E402


DEFAULT_CASES: tuple[tuple[tuple[int, ...], int], ...] = (
    ((64, 64), 8),
    ((128, 128), 4),
    ((64, 128), 16),
    ((32, 32), 64),
)
QUICK_CASES: tuple[tuple[tuple[int, ...], int], ...] = (
    ((64, 64), 8),
    ((32, 32), 64),
)
DEFAULT_RANK3_CASES: tuple[tuple[tuple[int, ...], int], ...] = (
    ((8, 8, 8), 16),
    ((8, 16, 32), 4),
)
QUICK_RANK3_CASES: tuple[tuple[tuple[int, ...], int], ...] = (
    ((8, 8, 8), 16),
)


def _shape_name(shape: Sequence[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {}
    try:
        identity["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        digest = hashlib.sha256()
        paths = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT, stderr=subprocess.DEVNULL,
        ).split(b"\0")
        for raw_path in paths:
            if not raw_path:
                continue
            path = ROOT / os.fsdecode(raw_path)
            if path.is_file():
                digest.update(raw_path)
                digest.update(b"\0")
                digest.update(path.read_bytes())
        identity["repo_sha256"] = digest.hexdigest()
        identity["working_tree_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        identity["identity_error"] = str(exc)
    return identity


def _discover_library(requested: Optional[Path]) -> Path:
    if requested is not None:
        path = requested if requested.is_absolute() else ROOT / requested
        if not path.is_file():
            raise FileNotFoundError(f"cuButterfly library not found: {path}")
        return path.resolve()

    candidates: list[Path] = []
    for variable in ("CUBUTTERFLY_LIB", "LIBCUNTT"):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend((
        ROOT / "build-e13-shared" / "libcuntt.so",
        ROOT / "build" / "libcuntt.so",
        ROOT / "build-debug" / "libcuntt.so",
        ROOT / "build-release" / "libcuntt.so",
    ))
    candidates.extend(sorted(ROOT.glob("build*/libcuntt.so")))
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else ROOT / candidate
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "unable to discover libcuntt.so; pass --library or set CUBUTTERFLY_LIB "
        f"(searched: {searched})"
    )


def _gpu_identity() -> dict[str, Any]:
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
                "nvidia-smi", f"--id={query_id}",
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


def _exception(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _max_abs(torch_value: torch.Tensor) -> float:
    if torch_value.numel() == 0:
        return 0.0
    return float(torch.abs(torch_value.detach()).max().item())


def _error(actual: torch.Tensor, expected: torch.Tensor,
           *, atol: float, rtol: float) -> dict[str, Any]:
    difference = actual.detach() - expected.detach()
    max_abs = _max_abs(difference)
    scale = max(1.0, _max_abs(expected))
    tolerance = float(atol + rtol * scale)
    return {
        "max_abs": max_abs,
        "scale": scale,
        "atol": float(atol),
        "rtol": float(rtol),
        "tolerance": tolerance,
        "passed": bool(max_abs <= tolerance),
    }


def _input_check(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    max_abs = _max_abs(after.detach() - before.detach())
    return {"max_abs": max_abs, "passed": bool(max_abs == 0.0)}


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _graph_timing(
    callback: Callable[[], Any],
    device: torch.device,
    *,
    warmup: int,
    trials: int,
    capture_calls: int,
    python_wall: bool,
) -> dict[str, Any]:
    if capture_calls != 20:
        raise ValueError("each CUDA graph must capture exactly 20 calls")
    # The caller constructs the adapter on one shared non-default stream.
    # Reusing that stream here keeps ctypes launches inside the graph; making a
    # fresh capture stream would leave the adapter's plan on another stream.
    stream = torch.cuda.current_stream(device)

    with torch.cuda.stream(stream):
        for _ in range(max(0, warmup)):
            callback()
    _synchronize(device)

    # Materialize lazy FFT state and allocator ownership before graph capture.
    with torch.cuda.stream(stream):
        callback()
    _synchronize(device)
    capture_start = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(capture_calls):
                callback()
    _synchronize(device)
    capture_wall_ms = (time.perf_counter() - capture_start) * 1000.0

    with torch.cuda.stream(stream):
        for _ in range(max(1, warmup // 2)):
            graph.replay()
    _synchronize(device)

    event_samples: list[float] = []
    wall_samples: list[float] = []
    replay_count = max(1, trials)
    for _ in range(replay_count):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter() if python_wall else 0.0
        begin.record(stream)
        with torch.cuda.stream(stream):
            graph.replay()
        end.record(stream)
        end.synchronize()
        event_samples.append(float(begin.elapsed_time(end)) / capture_calls)
        if python_wall:
            wall_samples.append(
                (time.perf_counter() - wall_start) * 1000.0 / capture_calls
            )

    result: dict[str, Any] = {
        "graph_used": True,
        "graph_capture_stream": "shared_non_default",
        "graph_capture_calls": capture_calls,
        "graph_replays": replay_count,
        "warmup_calls": max(0, warmup),
        "timing_unit": "milliseconds per call",
        "samples_ms": event_samples,
        "median_ms": float(statistics.median(event_samples)),
        "min_ms": float(min(event_samples)),
        "max_ms": float(max(event_samples)),
        "graph_capture_wall_ms_excluded": capture_wall_ms,
    }
    if python_wall:
        result["python_wall_samples_ms"] = wall_samples
        result["python_wall_median_ms"] = float(statistics.median(wall_samples))
        result["python_wall_min_ms"] = float(min(wall_samples))
        result["python_wall_max_ms"] = float(max(wall_samples))
    return result


def _adapter_metadata(adapter: RealFFTAdapter) -> dict[str, Any]:
    plans = adapter.plan_metadata()
    for plan, raw in zip((adapter._forward, adapter._inverse), plans):
        raw["workspace_bytes"] = (
            int(plan.workspace.numel()) if plan.workspace is not None else 0
        )
    return {
        "native_real": bool(adapter.native_real),
        "reuse_buffers": bool(adapter.reuse_buffers),
        "plans": plans,
        "forward_workspace_bytes": (
            int(adapter._forward.workspace.numel())
            if adapter._forward.workspace is not None else 0
        ),
        "inverse_workspace_bytes": (
            int(adapter._inverse.workspace.numel())
            if adapter._inverse.workspace is not None else 0
        ),
    }


def _run_path(
    *,
    name: str,
    source: torch.Tensor,
    expected_half: torch.Tensor,
    expected_full: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    warmup: int,
    trials: int,
    capture_calls: int,
    python_wall: bool,
    adapter: Optional[RealFFTAdapter],
) -> dict[str, Any]:
    rank = len(shape)
    dims = tuple(range(1, rank + 1))
    if adapter is None:
        forward_fn: Callable[[torch.Tensor], torch.Tensor] = lambda value: torch.fft.rfftn(
            value, dim=dims, norm="forward"
        )
        inverse_fn: Callable[[torch.Tensor], torch.Tensor] = lambda value: torch.fft.irfftn(
            value, s=shape, dim=dims, norm="forward"
        )
        metadata: dict[str, Any] = {
            "backend": "PyTorch torch.fft / cuFFT",
            "native_real": True,
            "planning": "lazy torch.fft/cuFFT state; warmup and graph capture excluded",
        }
    else:
        forward_fn = adapter.rfftn
        inverse_fn = adapter.irfftn
        metadata = {
            "backend": "cuButterfly RealFFTAdapter",
            **_adapter_metadata(adapter),
        }

    source_before = source.clone()
    half = forward_fn(source)
    _synchronize(device)
    forward_input = _input_check(source_before, source)
    forward_error = _error(half, expected_half, atol=5e-4, rtol=5e-4)

    half_before = half.clone()
    recovered = inverse_fn(half)
    _synchronize(device)
    inverse_input = _input_check(half_before, half)
    inverse_error = _error(recovered, expected_full, atol=5e-4, rtol=5e-4)
    source_after = _input_check(source_before, source)
    correctness = {
        "forward": forward_error,
        "inverse": inverse_error,
        "forward_source_unchanged": forward_input,
        "inverse_spectrum_unchanged": inverse_input,
        "source_unchanged_after_roundtrip": source_after,
    }
    correctness["passed"] = bool(
        forward_error["passed"] and inverse_error["passed"]
        and forward_input["passed"] and inverse_input["passed"]
        and source_after["passed"]
    )

    # The cloned spectrum is stable across the separate inverse timing graph;
    # round-trip timing dynamically chains the two public calls.
    half_for_timing = half.detach().clone()
    state: dict[str, Any] = {}

    def forward_call() -> None:
        state["half"] = forward_fn(source)

    def inverse_call() -> None:
        state["full"] = inverse_fn(half_for_timing)

    def roundtrip_call() -> None:
        state["half"] = forward_fn(source)
        state["full"] = inverse_fn(state["half"])

    timing = {
        "forward": _graph_timing(
            forward_call, device, warmup=warmup, trials=trials,
            capture_calls=capture_calls, python_wall=python_wall,
        ),
        "inverse": _graph_timing(
            inverse_call, device, warmup=warmup, trials=trials,
            capture_calls=capture_calls, python_wall=python_wall,
        ),
        "roundtrip": _graph_timing(
            roundtrip_call, device, warmup=warmup, trials=trials,
            capture_calls=capture_calls, python_wall=python_wall,
        ),
    }
    return {
        "name": name,
        "status": "passed" if correctness["passed"] else "failed",
        "plan_creation_wall_ms": None,
        "plan_construction_excluded_from_timing": True,
        "metadata": metadata,
        "correctness": correctness,
        "timing": timing,
        "timing_scope": (
            "public adapter or torch.fft calls; CUDA event values are per call; "
            "roundtrip contains one forward and one inverse per graph call"
        ),
    }


def _make_case(
    shape: tuple[int, ...],
    batch: int,
    *,
    library: Path,
    device: torch.device,
    warmup: int,
    trials: int,
    capture_calls: int,
    python_wall: bool,
    include_legacy: bool,
) -> dict[str, Any]:
    source = torch.randn((batch, *shape), dtype=torch.float32, device=device)
    dims = tuple(range(1, len(shape) + 1))
    expected_half = torch.fft.rfftn(source, dim=dims, norm="forward")
    expected_full = torch.fft.irfftn(
        expected_half, s=shape, dim=dims, norm="forward"
    )
    _synchronize(device)
    case: dict[str, Any] = {
        "shape": list(shape),
        "shape_name": _shape_name(shape),
        "batch": int(batch),
        "dtype": "torch.float32 -> torch.complex64",
        "norm": "forward",
        "paths": [],
    }
    path_specs = [("compact_native", True)]
    if include_legacy:
        path_specs.append(("legacy_composite", False))

    for path_name, native_real in path_specs:
        adapter: Optional[RealFFTAdapter] = None
        plan_start = time.perf_counter()
        try:
            adapter = RealFFTAdapter(
                library,
                shape,
                batch,
                device,
                reuse_buffers=True,
                native_real=native_real,
            )
            _synchronize(device)
            plan_wall_ms = (time.perf_counter() - plan_start) * 1000.0
            path = _run_path(
                name=path_name,
                source=source,
                expected_half=expected_half,
                expected_full=expected_full,
                shape=shape,
                device=device,
                warmup=warmup,
                trials=trials,
                capture_calls=capture_calls,
                python_wall=python_wall,
                adapter=adapter,
            )
            path["plan_creation_wall_ms"] = plan_wall_ms
        except Exception as exc:
            path = {
                "name": path_name,
                "status": "failed",
                "plan_creation_wall_ms": (
                    (time.perf_counter() - plan_start) * 1000.0
                ),
                "plan_construction_excluded_from_timing": True,
                "error": _exception(exc),
            }
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as cleanup_exc:
                    # Preserve the primary benchmark result; close errors are
                    # retained as diagnostics without masking it.
                    path.setdefault("cleanup_error", _exception(cleanup_exc))
        case["paths"].append(path)

    try:
        path = _run_path(
            name="torch_cufft",
            source=source,
            expected_half=expected_half,
            expected_full=expected_full,
            shape=shape,
            device=device,
            warmup=warmup,
            trials=trials,
            capture_calls=capture_calls,
            python_wall=python_wall,
            adapter=None,
        )
        path["plan_creation_wall_ms"] = None
    except Exception as exc:
        path = {
            "name": "torch_cufft",
            "status": "failed",
            "plan_creation_wall_ms": None,
            "plan_construction_excluded_from_timing": True,
            "error": _exception(exc),
        }
    case["paths"].append(path)
    del source, expected_half, expected_full
    _synchronize(device)
    case["status"] = "passed" if all(
        item.get("status") == "passed" for item in case["paths"]
    ) else "failed"
    return case


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native compact real FFT, legacy C2C composition, and "
            "PyTorch/cuFFT with 20-call CUDA graph captures."
        )
    )
    parser.add_argument("--library", type=Path, default=None,
                        help="path to libcuntt.so (otherwise auto-discover)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "e13_assets" / "benchmark_xd_real.json")
    parser.add_argument("--quick", action="store_true",
                        help="run the reduced rank-2/rank-3 matrix")
    parser.add_argument("--rank3", "--include-rank3", dest="rank3",
                        action="store_true", help="also run compact rank-3 cases")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", "--replays", dest="trials", type=int, default=7)
    parser.add_argument("--capture-calls", type=int, default=20,
                        help="calls per graph; must remain 20 for this protocol")
    parser.add_argument("--python-wall", action="store_true",
                        help="also record host wall time for graph replays")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report: dict[str, Any] = {
        "test": "xd_real_fft_benchmark",
        "protocol_version": 1,
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "repo": _repo_identity(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_requested": args.device,
        "protocol": {
            "norm": "forward",
            "dtype": "float32/complex64",
            "warmup": int(args.warmup),
            "trials": int(args.trials),
            "graph_capture_calls": int(args.capture_calls),
            "python_wall_requested": bool(args.python_wall),
            "plan_construction_outside_timing": True,
            "input_reused_across_paths": True,
            "correctness_checks": [
                "forward compact spectrum",
                "inverse reconstructed real tensor",
                "forward source unchanged",
                "inverse spectrum unchanged",
                "source unchanged after roundtrip",
            ],
        },
        "cases": [],
    }
    try:
        if args.capture_calls != 20:
            raise ValueError("--capture-calls must be exactly 20")
        if args.warmup < 0 or args.trials <= 0:
            raise ValueError("--warmup must be non-negative and --trials positive")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; use --help for the protocol")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must select a CUDA device")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu"] = _gpu_identity()
        library = _discover_library(args.library)
        report["library"] = {
            "path": str(library),
            "sha256": _sha256(library),
        }
        report["adapter"] = {
            "path": str((ASSET_DIR / "cubutterfly_real_fft.py").resolve()),
            "sha256": _sha256(ASSET_DIR / "cubutterfly_real_fft.py"),
        }

        # Construct plans and run every adapter/torch operation on one shared
        # non-default stream.  The adapter captures the current stream during
        # plan creation, so graph capture must reuse this exact stream.
        benchmark_stream = torch.cuda.Stream(device=device)
        report["protocol"]["graph_capture_stream"] = "shared_non_default"
        cases = QUICK_CASES if args.quick else DEFAULT_CASES
        rank3_cases = QUICK_RANK3_CASES if args.quick else DEFAULT_RANK3_CASES
        with torch.cuda.stream(benchmark_stream):
            for shape, batch in cases:
                report["cases"].append(_make_case(
                    tuple(shape), batch, library=library, device=device,
                    warmup=args.warmup, trials=args.trials,
                    capture_calls=args.capture_calls,
                    python_wall=args.python_wall, include_legacy=True,
                ))
            if args.rank3:
                for shape, batch in rank3_cases:
                    report["cases"].append(_make_case(
                        tuple(shape), batch, library=library, device=device,
                        warmup=args.warmup, trials=args.trials,
                        capture_calls=args.capture_calls,
                        python_wall=args.python_wall, include_legacy=False,
                    ))
        report["status"] = (
            "passed" if all(case.get("status") == "passed" for case in report["cases"])
            else "failed"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = _exception(exc)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
