#!/usr/bin/env python3
"""Check the finite operator-coverage gap manifest through the public C ABI.

The default invocation is a CPU-only manifest audit.  ``--execute`` is the
explicit GPU mode: it initializes a torch CUDA context and current stream,
creates plans through the public ``libcuntt.so`` C ABI, checks outputs against
independent references, and records CUDA-event timings.  A candidate that
returns ``NOT_SUPPORTED`` is a defect, not a skipped pass.  Only rows whose
manifest state is explicitly ``unsupported`` may produce
``expected-unsupported``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes as ct
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence
import zlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_LARGE_BYTES = 1 << 30
CHECKER_SCHEMA = "operator-coverage-gap-check-v1"
# A large anchor already owns its input, output, and workspace allocations.
# Keep correctness temporaries bounded to a small, fixed number of elements.
LARGE_VALIDATION_BLOCK_ELEMENTS = 1 << 20


class MemoryLimitError(RuntimeError):
    def __init__(self, message: str, *, workspace_bytes: int | None = None):
        super().__init__(message)
        self.workspace_bytes = workspace_bytes


def _load_xd_module() -> Any:
    """Load the existing C-ABI consumer without changing it."""
    path = ROOT / "tests" / "check_xd.py"
    spec = importlib.util.spec_from_file_location("cubutterfly_check_xd", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "operator-coverage-gap-v1":
        raise ValueError(f"unsupported manifest schema: {document.get('schema')!r}")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest has no rows")
    execution = document.get("execution", {})
    if execution.get("mode") != "cpu-dry-run" or execution.get("gpu_execution") is not False:
        raise ValueError("manifest must declare the CPU-only generator boundary")
    if any(not isinstance(row, dict) or not row.get("id") for row in rows):
        raise ValueError("manifest rows must be non-empty objects with ids")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("manifest contains duplicate row ids")
    counts = document.get("counts", {})
    if counts.get("rows") != len(rows):
        raise ValueError("manifest row count does not match its counts block")
    actual_sections = dict(sorted(Counter(row.get("section") for row in rows).items()))
    if counts.get("by_section") != actual_sections:
        raise ValueError("manifest section counts do not match its rows")
    actual_expectations = dict(sorted(
        Counter(row.get("expected_execution") for row in rows).items()))
    if counts.get("by_expected_execution") != actual_expectations:
        raise ValueError("manifest execution counts do not match its rows")
    actual_states = dict(sorted(
        Counter(row.get("coverage_state") for row in rows).items()))
    if counts.get("by_coverage_state") != actual_states:
        raise ValueError("manifest coverage-state counts do not match its rows")
    allowed_expectations = {"candidate", "packed-only", "unsupported"}
    for row in rows:
        expected = row.get("expected_execution")
        if expected not in allowed_expectations:
            raise ValueError(f"{row['id']} has unknown expected_execution={expected!r}")
        if row.get("coverage_state") == "unsupported" and expected != "unsupported":
            raise ValueError(f"{row['id']} marks unsupported coverage as executable")
        if expected == "unsupported" and row.get("coverage_state") != "unsupported":
            raise ValueError(f"{row['id']} expects unsupported without an unsupported state")
        if row.get("section") == "real_public_api":
            if expected != "candidate" or row.get("execution_mode") != "auto":
                raise ValueError(f"{row['id']} real public API row must be an AUTO candidate")
    return document


def _shape_product(shape: Iterable[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _span_elements(shape: Sequence[int], strides: Sequence[int],
                   batch_stride: int, batch: int) -> int:
    extent = 1 + sum((int(size) - 1) * int(stride)
                     for size, stride in zip(shape, strides))
    return (int(batch) - 1) * int(batch_stride) + extent


def _row_estimated_bytes(row: dict[str, Any]) -> int:
    input_bytes = int(row.get("input_bytes", 0))
    output_bytes = int(row.get("output_bytes", 0))
    if row.get("placement") == "in-place":
        return max(input_bytes, output_bytes)
    # The runner owns both buffers, so max(input, output) would understate
    # the allocation gate for out-of-place rows.
    return input_bytes + output_bytes


def _stable_seed(row_id: str) -> int:
    return zlib.crc32(row_id.encode("utf-8")) & 0xFFFFFFFF


def _select_rows(manifest: dict[str, Any], sections: Sequence[str] | None,
                 include_large: bool, only_candidates: bool,
                 row_ids: Sequence[str] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = set(sections or ())
    wanted_ids = set(row_ids or ())
    selected: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in manifest["rows"]:
        reason = None
        if wanted_ids and row["id"] not in wanted_ids:
            reason = "row-filter"
        elif wanted and row["section"] not in wanted:
            reason = "section-filter"
        elif not include_large and row["section"] == "large_n_batch":
            reason = "large-filter"
        elif only_candidates and row["expected_execution"] != "candidate":
            reason = "non-candidate-filter"
        if reason is None:
            selected.append(row)
        else:
            filtered.append({"id": row["id"], "section": row["section"], "status": "filtered", "reason": reason})
    return selected, filtered


def _counter_summary(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("status", "unknown") for row in rows)
    return {key: int(counts[key]) for key in sorted(counts)}


def _dry_run_report(manifest: dict[str, Any], selected: Sequence[dict[str, Any]],
                    filtered: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "id": row["id"],
            "section": row["section"],
            "expected_execution": row["expected_execution"],
            "coverage_state": row["coverage_state"],
            "status": "dry-run",
            "run_mode": "cpu-dry-run",
        }
        for row in selected
    ]
    return {
        "schema": CHECKER_SCHEMA,
        "mode": "cpu-dry-run",
        "manifest_schema": manifest["schema"],
        "manifest_counts": manifest.get("counts", {}),
        "execution": {"gpu_execution": False, "benchmark_invocations": 0,
                       "library_loads": 0, "device_queries": 0},
        "selection": {"selected": len(selected), "filtered": len(filtered),
                       "filtered_rows": list(filtered)},
        "rows": rows,
        "summary": {"selected": len(rows), "filtered": len(filtered),
                     "statuses": _counter_summary(rows)},
    }


def _storage_constants(xd: Any) -> dict[str, int]:
    return {
        "fp16": 0, "bf16": 1, "fp32": getattr(xd, "DATA_FP32", 2),
        "fp64": getattr(xd, "DATA_FP64", 3),
        "complex_fp16": 4, "complex_bf16": 5,
        "complex_fp32": getattr(xd, "DATA_COMPLEX_FP32", 6),
        "complex_fp64": getattr(xd, "DATA_COMPLEX_FP64", 7),
        "uint32": getattr(xd, "DATA_UINT32", 8),
        "uint64": getattr(xd, "DATA_UINT64", 9),
    }


def _operator_constants(xd: Any) -> dict[str, int]:
    return {
        "fft": xd.OP_FFT, "ntt": xd.OP_NTT, "fwht": xd.OP_FWHT,
        "subset-zeta": xd.OP_SUBSET_ZETA, "superset-zeta": xd.OP_SUPERSET_ZETA,
        "structured-2x2": xd.OP_STRUCTURED_2X2,
    }


def _compute_constant(xd: Any, value: str) -> int:
    return {
        "native": getattr(xd, "COMPUTE_NATIVE", 0),
        "fp32": xd.COMPUTE_FP32,
        "fp64": xd.COMPUTE_FP64,
        "uint32": getattr(xd, "COMPUTE_UINT32", 3),
        "uint64": getattr(xd, "COMPUTE_UINT64", 4),
        "modular": getattr(xd, "COMPUTE_UINT64", 4),
    }[value]


def _precision_dtype(torch: Any, precision: str) -> Any:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        "fp64": torch.float64,
        "word32": torch.int32,
        "word64": torch.int64,
    }[precision]


def _matrix_rows(row: dict[str, Any], rank: int) -> list[list[list[float]]] | None:
    if row["operator"] != "structured-2x2":
        return None
    values = [float(value) for value in row.get("stage_matrix", "1,0.25,-0.5,1").split(",")]
    if len(values) != 4:
        raise ValueError(f"{row['id']} has an invalid stage_matrix")
    return [[[values[0], values[1], values[2], values[3]]] for _ in range(rank)]


def _setup_plan(xd: Any, api: Any, handle: Any, torch: Any,
                row: dict[str, Any], *, workspace_limit: int | None = None,
                output_order_override: int | None = None) -> Any:
    """Create a public C-ABI plan, including the optional NTT order anchor."""
    constants = _storage_constants(xd)
    operators = _operator_constants(xd)
    shape = tuple(int(value) for value in row["shape"])
    descriptor = api.create_descriptor()
    plan_value = ct.c_void_p()
    try:
        api.check("cubutterflySetOperator", api.cubutterflySetOperator(
            descriptor, operators[row["operator"]]))
        api.check("cubutterflySetDataType", api.cubutterflySetDataType(
            descriptor, constants[row["storage"]], _compute_constant(xd, row["compute"])))
        api.check("cubutterflySetShape", api.cubutterflySetShape(
            descriptor, len(shape), xd._ct_sizes(shape), int(row["batch"])))
        mode = xd.EXEC_AUTO
        api.check("cubutterflySetExecutionMode", api.cubutterflySetExecutionMode(
            descriptor, mode))
        if row["operator"] == "fft":
            fft_type = {"c2c": xd.FFT_C2C, "r2c": xd.FFT_R2C, "c2r": xd.FFT_C2R}[row["fft_type"]]
            if api.fft_type_fn is None and fft_type != xd.FFT_C2C:
                raise xd.UnsupportedCase("library has no cubutterflySetFftTransformType")
            if api.fft_type_fn is not None:
                api.check("cubutterflySetFftTransformType", api.fft_type_fn(descriptor, fft_type))
            if float(row.get("output_scale", 1.0)) != 1.0:
                if api.fft_output_scale_fn is None:
                    raise xd.UnsupportedCase("library has no cubutterflySetFftOutputScale")
                api.check("cubutterflySetFftOutputScale", api.fft_output_scale_fn(
                    descriptor, float(row["output_scale"])))
        api.check("cubutterflySetStrides", api.cubutterflySetStrides(
            descriptor, xd._ct_sizes(row["input_strides"]), int(row["input_batch_stride"]),
            xd._ct_sizes(row["output_strides"]), int(row["output_batch_stride"])))
        api.check("cubutterflySetDirection", api.cubutterflySetDirection(
            descriptor, xd.INVERSE if row["direction"] == "inverse" else xd.FORWARD,
            int(row["direction"] == "inverse")))
        api.check("cubutterflySetPlacement", api.cubutterflySetPlacement(
            descriptor, xd.IN_PLACE if row["placement"] == "in-place" else xd.OUT_OF_PLACE))
        api.check("cubutterflySetLengthMode", api.cubutterflySetLengthMode(
            descriptor, xd.LENGTH_ZERO_EXTENDED if row["length_mode"] != "standard" else xd.LENGTH_STANDARD))
        api.check("cubutterflySetAlgorithmPolicy", api.cubutterflySetAlgorithmPolicy(
            descriptor, xd.POLICY_DEFAULT, None))
        if row["operator"] == "ntt":
            api.check("cubutterflySetModulus", api.cubutterflySetModulus(
                descriptor, int(row["modulus"]), int(row["word_bits"])))
            output_order = row["output_order"] if output_order_override is None else output_order_override
            order_value = {"natural": 0, "bit-reversed": 2}[output_order]
            api.check("cubutterflySetNttLayouts", api.cubutterflySetNttLayouts(
                descriptor, 0, order_value))
        matrices = _matrix_rows(row, len(shape))
        if matrices is not None:
            for axis, axis_matrices in enumerate(matrices):
                array = xd._matrix_array(axis_matrices)
                api.check("cubutterflySetStageMatrices", api.cubutterflySetStageMatrices(
                    descriptor, axis, ct.cast(array, ct.c_void_p), len(axis_matrices)))
        api.check("cubutterflyCreatePlan", api.cubutterflyCreatePlan(
            handle.value, descriptor, ct.byref(plan_value)))
        workspace_bytes = api.query_size(api.cubutterflyPlanGetWorkspaceSize,
                                         plan_value, "cubutterflyPlanGetWorkspaceSize")
        input_bytes = api.query_size(api.cubutterflyPlanGetInputSize,
                                     plan_value, "cubutterflyPlanGetInputSize")
        output_bytes = api.query_size(api.cubutterflyPlanGetOutputSize,
                                      plan_value, "cubutterflyPlanGetOutputSize")
        if workspace_limit is not None and workspace_bytes > workspace_limit:
            raise MemoryLimitError(
                f"workspace_bytes={workspace_bytes} exceeds memory limit={workspace_limit}",
                workspace_bytes=workspace_bytes)
        workspace = None
        if workspace_bytes:
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
            api.check("cubutterflyPlanSetWorkspace", api.cubutterflyPlanSetWorkspace(
                plan_value, xd._pointer_tensor(workspace), workspace_bytes))
        metadata = {
            "workspace_bytes": workspace_bytes,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "algorithm": api.query_text(api.cubutterflyPlanGetAlgorithmName,
                                         plan_value, "cubutterflyPlanGetAlgorithmName"),
            "selection_reason": api.query_text(api.cubutterflyPlanGetSelectionReason,
                                                plan_value, "cubutterflyPlanGetSelectionReason"),
            "mapping_json": api.query_text(api.cubutterflyPlanGetMappingJson,
                                            plan_value, "cubutterflyPlanGetMappingJson"),
        }
        source = ct.c_int()
        api.check("cubutterflyPlanGetSelectionSource",
                  api.cubutterflyPlanGetSelectionSource(plan_value, ct.byref(source)))
        metadata["selection_source"] = int(source.value)
        plan = xd.Plan(api, handle, plan_value, workspace, metadata)
        plan_value = ct.c_void_p()
        return plan
    except Exception:
        if plan_value:
            api.cubutterflyDestroyPlan(plan_value)
        raise
    finally:
        api.cubutterflyDestroyDescriptor(descriptor)


def _radix_root(length: int, modulus: int) -> int:
    if (modulus - 1) % length:
        raise ValueError(f"modulus {modulus} does not support N={length}")
    for candidate in range(2, 256):
        root = pow(candidate, (modulus - 1) // length, modulus)
        if pow(root, length, modulus) == 1 and (length == 1 or pow(root, length // 2, modulus) != 1):
            return root
    raise ValueError(f"no primitive N-th root for N={length}, p={modulus}")


def _reverse_bits(value: int, bits: int) -> int:
    result = 0
    for bit in range(bits):
        result = (result << 1) | ((value >> bit) & 1)
    return result


def _ntt_radix2(values: Sequence[int], modulus: int, inverse: bool) -> list[int]:
    n = len(values)
    if n == 0 or n & (n - 1):
        raise ValueError("radix-2 NTT requires a power-of-two length")
    bits = n.bit_length() - 1
    result = [int(values[_reverse_bits(index, bits)]) % modulus for index in range(n)]
    root = _radix_root(n, modulus)
    if inverse:
        root = pow(root, modulus - 2, modulus)
    length = 2
    while length <= n:
        half = length // 2
        step = pow(root, n // length, modulus)
        for start in range(0, n, length):
            twiddle = 1
            for offset in range(half):
                left = result[start + offset]
                right = result[start + offset + half] * twiddle % modulus
                result[start + offset] = (left + right) % modulus
                result[start + offset + half] = (left - right) % modulus
                twiddle = twiddle * step % modulus
        length <<= 1
    if inverse:
        scale = pow(n, modulus - 2, modulus)
        result = [value * scale % modulus for value in result]
    return result


def _ntt_reference(row: dict[str, Any], values: Sequence[int]) -> list[int]:
    n = int(row["N"])
    batch = int(row["batch"])
    modulus = int(row["modulus"])
    inverse = row["direction"] == "inverse"
    result: list[int] = []
    for batch_index in range(batch):
        line = _ntt_radix2(values[batch_index * n:(batch_index + 1) * n], modulus, inverse)
        if row.get("output_order") == "bit-reversed":
            bits = n.bit_length() - 1
            line = [line[_reverse_bits(index, bits)] for index in range(n)]
        result.extend(line)
    return result


def _zeta_reference(row: dict[str, Any], values: Sequence[int]) -> list[int]:
    n = _shape_product(row["shape"])
    batch = int(row["batch"])
    inverse = row["direction"] == "inverse"
    subset = row["operator"] == "subset-zeta"
    result = [int(value) & 0xFFFFFFFF for value in values]
    for batch_index in range(batch):
        base = batch_index * n
        width = 1
        while width < n:
            for start in range(0, n, width * 2):
                for offset in range(width):
                    left = base + start + offset
                    right = left + width
                    if subset:
                        if inverse:
                            result[right] = (result[right] - result[left]) & 0xFFFFFFFF
                        else:
                            result[right] = (result[right] + result[left]) & 0xFFFFFFFF
                    else:
                        if inverse:
                            result[left] = (result[left] - result[right]) & 0xFFFFFFFF
                        else:
                            result[left] = (result[left] + result[right]) & 0xFFFFFFFF
            width <<= 1
    return result


def _random_complex(torch: Any, shape: Sequence[int], seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    real = torch.randn(tuple(shape), dtype=torch.float32, generator=generator)
    imag = torch.randn(tuple(shape), dtype=torch.float32, generator=generator)
    return torch.complex(real, imag)


def _logical_random(torch: Any, shape: Sequence[int], dtype: Any, seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), dtype=dtype, generator=generator)


def _logical_offsets(shape: Sequence[int], strides: Sequence[int], batch: int,
                     batch_stride: int):
    for batch_index in range(int(batch)):
        for index in itertools.product(*(range(int(size)) for size in shape)):
            offset = batch_index * int(batch_stride) + sum(
                int(position) * int(stride) for position, stride in zip(index, strides))
            yield (batch_index, *index), offset


def _scatter_strided(torch: Any, logical: Any, *, shape: Sequence[int],
                     strides: Sequence[int], batch_stride: int, device: str) -> Any:
    span = _span_elements(shape, strides, batch_stride, logical.shape[0])
    storage = torch.zeros(span, dtype=logical.dtype, device=device)
    for index, offset in _logical_offsets(shape, strides, logical.shape[0], batch_stride):
        storage[offset] = logical[index].to(device=device)
    return storage


def _gather_logical(torch: Any, storage: Any, *, shape: Sequence[int],
                    strides: Sequence[int], batch_stride: int, batch: int) -> Any:
    host = storage.detach().cpu()
    result = torch.empty((int(batch), *tuple(int(value) for value in shape)), dtype=host.dtype)
    for index, offset in _logical_offsets(shape, strides, batch, batch_stride):
        result[index] = host[offset]
    return result


def _error_metrics(torch: Any, actual: Any, expected: Any, *, tolerance: float,
                   label: str, relative: bool = False) -> dict[str, float]:
    actual = actual.detach().cpu()
    expected = expected.detach().cpu()
    diff = torch.abs(actual - expected)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    if relative:
        denom = float(torch.linalg.vector_norm(expected.reshape(-1)).item())
        numerator = float(torch.linalg.vector_norm((actual - expected).reshape(-1)).item())
        relative_l2 = numerator / max(denom, 1.0e-30)
        if relative_l2 > tolerance:
            raise AssertionError(f"{label}: relative_l2={relative_l2:g}, tolerance={tolerance:g}")
        return {"max_abs": max_abs, "relative_l2": relative_l2, "tolerance": tolerance}
    scale = max(1.0, float(torch.abs(expected).max().item()))
    bound = tolerance * (1.0 + scale)
    if max_abs > bound:
        raise AssertionError(f"{label}: max_abs={max_abs:g}, tolerance={bound:g}")
    return {"max_abs": max_abs, "scale": scale, "tolerance": bound}


def _validation_block_ranges(elements: int, block_elements: int = LARGE_VALIDATION_BLOCK_ELEMENTS):
    """Yield complete, non-overlapping ranges for bounded output validation."""
    elements = int(elements)
    block_elements = int(block_elements)
    if elements < 0:
        raise ValueError(f"elements must be non-negative, got {elements}")
    if block_elements <= 0:
        raise ValueError(f"block_elements must be positive, got {block_elements}")
    for start in range(0, elements, block_elements):
        yield start, min(start + block_elements, elements)


def _validate_constant_output(torch: Any, output: Any, expected_value: Any, *,
                              tolerance: float | None = None,
                              block_elements: int = LARGE_VALIDATION_BLOCK_ELEMENTS,
                              label: str = "constant output") -> dict[str, Any]:
    """Check every output element against one value with bounded temporaries.

    Large-anchor outputs are contiguous allocations owned by the plan runner.
    ``view(-1)`` is intentional: unlike ``reshape(-1)``, it cannot silently
    materialize a full copy for a non-contiguous tensor.  Each arithmetic or
    comparison temporary is limited to ``block_elements`` values.
    """
    elements = int(output.numel())
    flat = output.view(-1)
    blocks = 0
    max_abs = 0.0
    for start, stop in _validation_block_ranges(elements, block_elements):
        blocks += 1
        block = flat[start:stop]
        if tolerance is None:
            if bool(torch.ne(block, expected_value).any().item()):
                raise AssertionError(f"{label} mismatch")
            continue
        block_max = float(torch.abs(block - expected_value).max().item())
        if not math.isfinite(block_max):
            raise AssertionError(f"{label} mismatch: non-finite output")
        max_abs = max(max_abs, block_max)
    stats = {
        "validation_block_elements": int(block_elements),
        "validation_blocks": blocks,
    }
    if tolerance is None:
        stats["exact"] = True
        return stats
    if max_abs > float(tolerance):
        raise AssertionError(
            f"{label} mismatch: max_abs={max_abs:g}, tolerance={float(tolerance):g}")
    stats["max_abs"] = max_abs
    return stats


def _time_plan(torch: Any, plan: Any, input_tensor: Any, output_tensor: Any,
               *, warmup: int, repeat: int, stream: Any) -> float:
    for _ in range(warmup):
        plan.execute(input_tensor, output_tensor)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeat):
        plan.execute(input_tensor, output_tensor)
    stop.record(stream)
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repeat


def _execute_low_precision(xd: Any, api: Any, handle: Any, torch: Any,
                           row: dict[str, Any], *, workspace_limit: int | None,
                           warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    shape = tuple(int(value) for value in row["shape"])
    batch = int(row["batch"])
    logical_shape = (batch, *shape)
    precision = row["precision"]
    scalar_dtype = _precision_dtype(torch, precision)
    seed = _stable_seed(row["id"])
    if row["operator"] == "fft":
        source_float = _logical_random(torch, (*logical_shape, 2), torch.float32, seed)
        raw_cpu = source_float.to(dtype=scalar_dtype)
        source_reference = torch.complex(raw_cpu[..., 0].float(), raw_cpu[..., 1].float())
        source = raw_cpu.reshape(-1).to(device="cuda")
        output = torch.empty_like(source)
        expected = (torch.fft.fftn if row["direction"] == "forward" else torch.fft.ifftn)(
            source_reference,
            dim=tuple(range(1, len(shape) + 1)), norm="backward")
        label = "low-precision C2C FFT"
        relative = True
        tolerance = 0.40
        decode = lambda value: torch.complex(value.reshape(*logical_shape, 2)[..., 0].float(),
                                              value.reshape(*logical_shape, 2)[..., 1].float())
    else:
        source_cpu = _logical_random(torch, logical_shape, torch.float32, seed).to(dtype=scalar_dtype)
        source = source_cpu.reshape(-1).to(device="cuda")
        output = torch.empty_like(source)
        source_reference = source_cpu.float()
        if row["operator"] == "fwht":
            expected = xd._fwht_reference(torch, source_reference, range(1, len(shape) + 1))
            if row["direction"] == "inverse":
                expected = expected / _shape_product(shape)
            label = "low-precision FWHT"
        else:
            matrices = _matrix_rows(row, len(shape))
            expected = xd._structured_reference(torch, source_reference, shape, matrices, row["direction"] == "inverse")
            label = "low-precision structured-2x2"
        decode = lambda value: value.reshape(*logical_shape).float()
        relative = True
        tolerance = 0.40
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        actual = decode(output)
        error = _error_metrics(torch, actual, expected, tolerance=tolerance,
                               label=label, relative=relative)
        timing_input = source
        timing_output = output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {"correctness": error, "kernel_ms": kernel_ms,
                "algorithm": plan.metadata["algorithm"],
                "workspace_bytes": plan.metadata["workspace_bytes"]}
    finally:
        plan.close()


def _execute_real(xd: Any, api: Any, handle: Any, torch: Any,
                  row: dict[str, Any], *, workspace_limit: int | None,
                  warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    shape = tuple(int(value) for value in row["shape"])
    batch = int(row["batch"])
    real_dtype = _precision_dtype(torch, row["precision"])
    compact_shape = shape[:-1] + (shape[-1] // 2 + 1,)
    seed = _stable_seed(row["id"])
    if row["fft_type"] == "r2c":
        source_cpu = _logical_random(torch, (batch, *shape), real_dtype, seed)
        source = source_cpu.to(device="cuda")
        output = torch.empty((batch, *compact_shape), dtype=torch.complex64 if real_dtype == torch.float32 else torch.complex128, device="cuda")
        expected = torch.fft.rfftn(source_cpu, dim=tuple(range(1, len(shape) + 1)), norm="backward")
        tolerance = 5.0e-4 if real_dtype == torch.float32 else 5.0e-11
        label = "R2C"
    else:
        source_cpu = _random_complex(torch, (batch, *compact_shape), seed)
        if real_dtype == torch.float64:
            source_cpu = source_cpu.to(dtype=torch.complex128)
        source = source_cpu.to(device="cuda")
        output = torch.empty((batch, *shape), dtype=real_dtype, device="cuda")
        expected = torch.fft.irfftn(source_cpu, s=shape, dim=tuple(range(1, len(shape) + 1)), norm="backward")
        tolerance = 8.0e-4 if real_dtype == torch.float32 else 8.0e-11
        label = "C2R"
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        error = _error_metrics(torch, output, expected, tolerance=tolerance, label=label)
        timing_input = source
        timing_output = output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {"correctness": error, "kernel_ms": kernel_ms,
                "algorithm": plan.metadata["algorithm"],
                "workspace_bytes": plan.metadata["workspace_bytes"]}
    finally:
        plan.close()


def _execute_ntt(xd: Any, api: Any, handle: Any, torch: Any,
                 row: dict[str, Any], *, workspace_limit: int | None,
                 warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    n = int(row["N"])
    batch = int(row["batch"])
    modulus = int(row["modulus"])
    storage_dtype = torch.int32 if int(row["word_bits"]) == 32 else torch.int64
    values = [(index * 17 + 5) % modulus for index in range(n * batch)]
    source = torch.tensor(values, dtype=storage_dtype, device="cuda")
    output = torch.empty_like(source)
    expected = _ntt_reference(row, values)
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        actual = [int(value) for value in output.detach().cpu().tolist()]
        if int(row["word_bits"]) == 32:
            actual = [value & 0xFFFFFFFF for value in actual]
        if actual != expected:
            mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
            raise AssertionError(f"NTT mismatch at index {mismatch}: actual={actual[mismatch]}, expected={expected[mismatch]}")
        timing_input = source
        timing_output = output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {"correctness": {"elements": len(expected), "exact": True},
                "kernel_ms": kernel_ms, "algorithm": plan.metadata["algorithm"],
                "workspace_bytes": plan.metadata["workspace_bytes"]}
    finally:
        plan.close()


def _execute_zeta(xd: Any, api: Any, handle: Any, torch: Any,
                  row: dict[str, Any], *, workspace_limit: int | None,
                  warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    count = int(row["batch"]) * _shape_product(row["shape"])
    values = [(index * 17 + 5) & 0xFFFFFFFF for index in range(count)]
    source = torch.tensor(values, dtype=torch.int32, device="cuda")
    output = torch.empty_like(source)
    expected = _zeta_reference(row, values)
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        actual = [int(value) & 0xFFFFFFFF for value in output.detach().cpu().tolist()]
        if actual != expected:
            mismatch = next(index for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1])
            raise AssertionError(f"zeta mismatch at index {mismatch}: actual={actual[mismatch]}, expected={expected[mismatch]}")
        timing_input = source
        timing_output = output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {"correctness": {"elements": count, "exact": True},
                "kernel_ms": kernel_ms, "algorithm": plan.metadata["algorithm"],
                "workspace_bytes": plan.metadata["workspace_bytes"]}
    finally:
        plan.close()


def _execute_rank_layout(xd: Any, api: Any, handle: Any, torch: Any,
                         row: dict[str, Any], *, workspace_limit: int | None,
                         warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    shape = tuple(int(value) for value in row["shape"])
    batch = int(row["batch"])
    logical_shape = (batch, *shape)
    seed = _stable_seed(row["id"])
    if row["operator"] == "fft":
        logical = _random_complex(torch, logical_shape, seed)
        expected = (torch.fft.fftn if row["direction"] == "forward" else torch.fft.ifftn)(
            logical, dim=tuple(range(1, len(shape) + 1)), norm="backward")
        tolerance = 5.0e-4
    else:
        logical = _logical_random(torch, logical_shape, torch.float32, seed)
        if row["operator"] == "fwht":
            expected = xd._fwht_reference(torch, logical, range(1, len(shape) + 1))
        else:
            expected = xd._structured_reference(
                torch, logical, shape, _matrix_rows(row, len(shape)),
                row["direction"] == "inverse")
        tolerance = 7.0e-4
    source = _scatter_strided(torch, logical, shape=shape,
                              strides=row["input_strides"],
                              batch_stride=row["input_batch_stride"], device="cuda")
    if row["placement"] == "in-place":
        output = source
    else:
        output = torch.zeros(_span_elements(shape, row["output_strides"],
                                             row["output_batch_stride"], batch),
                             dtype=source.dtype, device="cuda")
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        actual = _gather_logical(torch, output, shape=shape,
                                 strides=row["output_strides"],
                                 batch_stride=row["output_batch_stride"], batch=batch)
        error = _error_metrics(torch, actual, expected, tolerance=tolerance,
                               label=f"{row['operator']} rank/layout")
        timing_input = source
        timing_output = source if row["placement"] == "in-place" else output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {"correctness": error, "kernel_ms": kernel_ms,
                "algorithm": plan.metadata["algorithm"],
                "workspace_bytes": plan.metadata["workspace_bytes"]}
    finally:
        plan.close()


def _execute_large_anchor(xd: Any, api: Any, handle: Any, torch: Any,
                          row: dict[str, Any], *, workspace_limit: int | None,
                          warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    """Run large rows with an analytic impulse, without a full host oracle.

    Small rows deliberately use random host references.  The large anchors
    instead use one unit impulse per batch: an unnormalized forward FFT or
    NTT must produce an all-ones spectrum.  This checks every output element
    while keeping the reference O(batch) and validation temporaries bounded,
    rather than materializing a second N-element host array (or a Python
    integer list for NTT).
    """
    shape = tuple(int(value) for value in row["shape"])
    batch = int(row["batch"])
    n = _shape_product(shape)
    elements = batch * n
    if row["operator"] == "ntt":
        dtype = torch.int32 if int(row["word_bits"]) == 32 else torch.int64
        source = torch.zeros((batch, n), dtype=dtype, device="cuda")
        source[:, 0] = 1
        output = torch.empty_like(source)
        expected_value = 1
        tolerance = None
    else:
        dtype = torch.complex64 if row["precision"] == "fp32" else torch.complex128
        source = torch.zeros((batch, n), dtype=dtype, device="cuda")
        source[:, 0] = 1
        output = torch.empty_like(source)
        expected_value = 1.0 + 0.0j
        tolerance = 2.0e-4 if row["precision"] == "fp32" else 2.0e-11
    plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
    try:
        plan.execute(source, output)
        torch.cuda.synchronize()
        if row["operator"] == "ntt":
            validation = _validate_constant_output(
                torch, output, expected_value,
                label="large NTT impulse reference")
            correctness = {"elements": elements, **validation,
                           "reference": "unit-impulse-per-batch"}
        else:
            validation = _validate_constant_output(
                torch, output, expected_value, tolerance=tolerance,
                label="large FFT impulse reference")
            correctness = {"elements": elements, **validation,
                           "tolerance": float(tolerance),
                           "reference": "unit-impulse-per-batch"}
        timing_input = source
        timing_output = output
        kernel_ms = _time_plan(torch, plan, timing_input, timing_output,
                               warmup=warmup, repeat=repeat, stream=stream)
        return {
            "correctness": correctness,
            "input_pattern": "unit impulse at logical index 0 per batch",
            "full_elements": elements,
            "kernel_ms": kernel_ms,
            "algorithm": plan.metadata["algorithm"],
            "workspace_bytes": plan.metadata["workspace_bytes"],
        }
    finally:
        plan.close()


def _execute_row(xd: Any, api: Any, handle: Any, torch: Any,
                 row: dict[str, Any], *, workspace_limit: int | None,
                 warmup: int, repeat: int, stream: Any) -> dict[str, Any]:
    operator = row["operator"]
    if row["section"] == "unsupported_real_low_precision":
        try:
            plan = _setup_plan(xd, api, handle, torch, row, workspace_limit=workspace_limit)
        except xd.CApiError as exc:
            if exc.status == xd.STATUS_NOT_SUPPORTED:
                return {"status": "expected-unsupported", "status_code": exc.status,
                        "status_text": exc.message}
            raise
        else:
            plan.close()
            raise AssertionError("manifest-marked unsupported row created a plan")
    if row["section"] == "low_precision_public_api":
        return _execute_low_precision(xd, api, handle, torch, row,
                                      workspace_limit=workspace_limit, warmup=warmup,
                                      repeat=repeat, stream=stream)
    if row["section"] == "real_public_api":
        return _execute_real(xd, api, handle, torch, row,
                             workspace_limit=workspace_limit, warmup=warmup,
                             repeat=repeat, stream=stream)
    if row["section"] == "large_n_batch":
        return _execute_large_anchor(xd, api, handle, torch, row,
                                     workspace_limit=workspace_limit,
                                     warmup=warmup, repeat=repeat, stream=stream)
    if operator == "ntt":
        return _execute_ntt(xd, api, handle, torch, row,
                            workspace_limit=workspace_limit, warmup=warmup,
                            repeat=repeat, stream=stream)
    if operator in ("subset-zeta", "superset-zeta"):
        return _execute_zeta(xd, api, handle, torch, row,
                             workspace_limit=workspace_limit, warmup=warmup,
                             repeat=repeat, stream=stream)
    return _execute_rank_layout(xd, api, handle, torch, row,
                                workspace_limit=workspace_limit, warmup=warmup,
                                repeat=repeat, stream=stream)


def _classify_c_api_error(xd: Any, exc: Exception, *, expected_unsupported: bool) -> str:
    """Keep API capability failures distinct from runtime/environment errors."""
    status = int(getattr(exc, "status", -1))
    not_supported = int(getattr(xd, "STATUS_NOT_SUPPORTED", 3))
    if status == not_supported:
        return "expected-unsupported" if expected_unsupported else "defect-unsupported"
    if status == int(getattr(xd, "STATUS_ALLOC_FAILED", 4)):
        return "memory-limit"
    environment_statuses = {
        int(getattr(xd, "STATUS_CUDA_ERROR", 5)),
        int(getattr(xd, "STATUS_BACKEND_ERROR", 6)),
        int(getattr(xd, "STATUS_INTERNAL_ERROR", 7)),
    }
    if status in environment_statuses:
        return "environment-error"
    return "failed"


def _run_gpu(manifest: dict[str, Any], selected: Sequence[dict[str, Any]],
             filtered: Sequence[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"schema": CHECKER_SCHEMA, "mode": "gpu-execute",
                "status": "environment-error", "error": f"torch import failed: {exc}",
                "manifest_counts": manifest.get("counts", {}),
                "execution": {"gpu_execution": False, "benchmark_invocations": 0,
                               "library_loads": 0, "device_queries": 0},
                "rows": [], "summary": {"selected": len(selected), "filtered": len(filtered)}}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if args.device is not None:
            torch.cuda.set_device(args.device)
        torch.cuda.init()
        stream = torch.cuda.current_stream()
        stream_handle = int(stream.cuda_stream)
        xd = _load_xd_module()
        library = Path(args.library) if args.library else xd._find_library(None)
        api = xd.Api(library)
        with xd.Handle(api, stream_handle) as handle:
            results: list[dict[str, Any]] = []
            for row in selected:
                result: dict[str, Any] = {
                    "id": row["id"], "section": row["section"],
                    "expected_execution": row["expected_execution"],
                    "estimated_bytes": _row_estimated_bytes(row),
                }
                estimated_bytes = _row_estimated_bytes(row)
                if args.memory_limit_bytes:
                    result["estimated_io_bytes"] = estimated_bytes
                    result["workspace_budget_bytes"] = max(
                        0, args.memory_limit_bytes - estimated_bytes)
                if args.memory_limit_bytes and estimated_bytes > args.memory_limit_bytes:
                    result.update({"status": "memory-limit",
                                   "error": f"estimated IO bytes={estimated_bytes} exceeds memory limit={args.memory_limit_bytes}",
                                   "memory_limit_bytes": args.memory_limit_bytes})
                    results.append(result)
                    continue
                try:
                    details = _execute_row(
                        xd, api, handle, torch, row,
                        workspace_limit=(
                            max(0, args.memory_limit_bytes - estimated_bytes)
                            if args.memory_limit_bytes else None),
                        warmup=args.warmup, repeat=args.repeat, stream=stream)
                    result.update(details)
                    result.setdefault("status", "passed")
                except xd.CApiError as exc:
                    status = _classify_c_api_error(
                        xd, exc,
                        expected_unsupported=row["expected_execution"] == "unsupported")
                    result.update({
                        "status": status,
                        "status_code": exc.status, "status_text": exc.message,
                        "error": str(exc),
                    })
                    if status == "memory-limit":
                        result["memory_limit_bytes"] = args.memory_limit_bytes
                except MemoryLimitError as exc:
                    result.update({"status": "memory-limit", "error": str(exc),
                                   "workspace_bytes": exc.workspace_bytes,
                                   "memory_limit_bytes": args.memory_limit_bytes})
                except xd.UnsupportedCase as exc:
                    result.update({"status": "environment-error", "error": str(exc)})
                except Exception as exc:
                    message = str(exc).lower()
                    status = "memory-limit" if (
                        "out of memory" in message or
                        "memory allocation" in message or
                        "cudaerrormemoryallocation" in message
                    ) else "failed"
                    result.update({"status": status, "error": str(exc)})
                    if status == "memory-limit":
                        result["memory_limit_bytes"] = args.memory_limit_bytes
                results.append(result)
        statuses = _counter_summary(results)
        bad = sum(statuses.get(name, 0)
                  for name in ("failed", "defect-unsupported", "environment-error"))
        if statuses.get("environment-error", 0) and not sum(
                statuses.get(name, 0) for name in ("failed", "defect-unsupported")):
            report_status = "environment-error"
        else:
            report_status = "failed" if bad else "passed"
        return {
            "schema": CHECKER_SCHEMA,
            "mode": "gpu-execute",
            "status": report_status,
            "manifest_counts": manifest.get("counts", {}),
            "library": str(library),
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
            "torch": torch.__version__,
            "execution": {"gpu_execution": True,
                           "benchmark_invocations": sum("kernel_ms" in row for row in results),
                           "library_loads": 1, "device_queries": 1,
                           "warmup": args.warmup, "repeat": args.repeat},
            "selection": {"selected": len(selected), "filtered": len(filtered),
                           "filtered_rows": list(filtered)},
            "rows": results,
            "summary": {"selected": len(results), "filtered": len(filtered),
                         "statuses": statuses},
        }
    except Exception as exc:
        return {
            "schema": CHECKER_SCHEMA,
            "mode": "gpu-execute",
            "status": "environment-error",
            "error": str(exc),
            "manifest_counts": manifest.get("counts", {}),
            "execution": {"gpu_execution": False, "benchmark_invocations": 0,
                           "library_loads": 0, "device_queries": 0},
            "selection": {"selected": len(selected), "filtered": len(filtered),
                           "filtered_rows": list(filtered)},
            "rows": [],
            "summary": {"selected": len(selected), "filtered": len(filtered),
                         "statuses": {"environment-error": 1}},
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="initialize CUDA and execute selected rows")
    parser.add_argument("--device", type=int)
    parser.add_argument("--section", action="append", dest="sections",
                        help="restrict rows to one section; repeat or use comma-separated values")
    parser.add_argument("--row", action="append", dest="rows",
                        help="restrict execution to one or more manifest row ids; repeat or use comma-separated values")
    parser.add_argument("--include-large", action="store_true",
                        help="include large_n_batch rows (filtered by default)")
    parser.add_argument("--only-candidates", action="store_true",
                        help="exclude explicit packed-only and unsupported contract rows")
    parser.add_argument("--memory-limit-bytes", type=int, default=0,
                        help="mark rows whose IO/workspace exceeds this limit as memory-limit")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("warmup must be >= 0 and repeat must be positive")
    if args.memory_limit_bytes < 0:
        parser.error("memory limit must be non-negative")
    sections = []
    for value in args.sections or []:
        sections.extend(item.strip() for item in value.split(",") if item.strip())
    args.sections = sections or None
    rows = []
    for value in args.rows or []:
        rows.extend(item.strip() for item in value.split(",") if item.strip())
    args.rows = rows or None
    if args.include_large and args.execute and args.memory_limit_bytes == 0:
        args.memory_limit_bytes = DEFAULT_MAX_LARGE_BYTES
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = _load_json(manifest_path)
    selected, filtered = _select_rows(manifest, args.sections,
                                      args.include_large, args.only_candidates,
                                      args.rows)
    report = (_run_gpu(manifest, selected, filtered, args)
              if args.execute else _dry_run_report(manifest, selected, filtered))
    report["manifest"] = str(manifest_path)
    report["created_unix"] = time.time()
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.get("status") in ("failed", "environment-error"):
        return 2 if report.get("status") == "environment-error" else 1
    statuses = report.get("summary", {}).get("statuses", {})
    return 1 if statuses.get("failed", 0) or statuses.get("defect-unsupported", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
