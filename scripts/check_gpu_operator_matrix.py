#!/usr/bin/env python3
"""Execute and validate a bounded GPU matrix for every public xD operator.

The benchmark campaign records timings; this command records correctness from
the actual C-ABI output.  It uses the same workload inventory as
``benchmark_all_operators_xd.py`` and the descriptor/plan helpers from
``tests/check_xd.py``.  ``--dry-run`` is intentionally CPU-only and is safe to
use while a GPU run is being scheduled.

The default coverage and scale profiles each contain 240 selected cells:
six public operators, their legal precisions, four rank-2/rank-3 shapes, three
batches, and two directions.  ``--include-resident`` adds a second 240-cell
set with the explicit ``xd:resident-t128`` mapping.  A skipped or missing
cell is never counted as a correctness pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"
for path in (TESTS, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Reuse the campaign inventory instead of maintaining a second matrix that
# could silently drift from benchmark_all_operators_xd.py.
from benchmark_all_operators_xd import (  # noqa: E402
    DIRECTIONS,
    OPERATORS,
    PROFILE_BATCHES,
    PROFILE_SHAPES,
    _case_id,
)
from check_xd import (  # noqa: E402
    Api,
    CApiError,
    COMPUTE_FP32,
    COMPUTE_FP64,
    COMPUTE_UINT32,
    COMPUTE_UINT64,
    DATA_COMPLEX_FP32,
    DATA_COMPLEX_FP64,
    DATA_FP32,
    DATA_FP64,
    DATA_UINT32,
    DATA_UINT64,
    EXEC_XD,
    FORWARD,
    Handle,
    INVERSE,
    OP_FFT,
    OP_FWHT,
    OP_NTT,
    OP_STRUCTURED_2X2,
    OP_SUBSET_ZETA,
    OP_SUPERSET_ZETA,
    STATUS_NOT_SUPPORTED,
    UnsupportedCase,
    _assert_close,
    _fwht_reference,
    _primitive_root_for_length,
    _structured_reference,
    _zeta_reference,
    make_plan,
)


DEFAULT_LIBRARY = ROOT / "build-e13-shared" / "libcuntt.so"
DEFAULT_MODULUS = 257
DEFAULT_MATRIX = (1.0, 0.25, -0.5, 1.0)


@dataclass(frozen=True)
class GpuCase:
    operator: str
    precision: str
    shape: tuple[int, ...]
    batch: int
    inverse: bool
    configuration: str = "selected"
    source_row: dict[str, Any] | None = None

    @property
    def direction(self) -> int:
        return INVERSE if self.inverse else FORWARD

    @property
    def name(self) -> str:
        return _case_id(self.operator, self.precision, self.shape, self.batch,
                        self.inverse, self.configuration)


def _sha256(path: pathlib.Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape_text(shape: Iterable[int]) -> str:
    return "x".join(str(int(value)) for value in shape)


def _operator_constants(name: str) -> tuple[int, int, int]:
    """Return C-ABI operator, storage/compute defaults by operator name."""
    if name == "fft":
        return OP_FFT, DATA_COMPLEX_FP32, DATA_FP32
    if name == "fwht":
        return OP_FWHT, DATA_FP32, DATA_FP32
    if name == "ntt":
        return OP_NTT, DATA_UINT32, DATA_UINT32
    if name == "subset-zeta":
        return OP_SUBSET_ZETA, DATA_UINT32, DATA_UINT32
    if name == "superset-zeta":
        return OP_SUPERSET_ZETA, DATA_UINT32, DATA_UINT32
    if name == "structured-2x2":
        return OP_STRUCTURED_2X2, DATA_FP32, DATA_FP32
    raise ValueError(f"unknown operator {name!r}")


def _precision_types(torch: Any, operator: str, precision: str) -> tuple[int, int, Any]:
    if operator == "fft":
        if precision == "fp32":
            return DATA_COMPLEX_FP32, COMPUTE_FP32, torch.complex64
        if precision == "fp64":
            return DATA_COMPLEX_FP64, COMPUTE_FP64, torch.complex128
    if precision == "fp32":
        return DATA_FP32, COMPUTE_FP32, torch.float32
    if precision == "fp64":
        return DATA_FP64, COMPUTE_FP64, torch.float64
    if precision == "uint32":
        return DATA_UINT32, COMPUTE_UINT32, torch.int32
    if precision == "uint64":
        return DATA_UINT64, COMPUTE_UINT64, torch.int64
    raise ValueError(f"unsupported precision {precision!r} for {operator}")


def _structured_matrices(
        shape: Sequence[int], matrix: Sequence[float] = DEFAULT_MATRIX,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Match plan_bench's one matrix per logical axis/stage contract."""
    matrices: list[tuple[tuple[float, ...], ...]] = []
    for length in shape:
        stages = int(math.log2(length))
        if (1 << stages) != int(length):
            raise ValueError("structured matrix axes must be powers of two")
        # A count-one stage list is expanded by the public API/reference for
        # every radix-2 stage, exactly as apps/cubutterfly_plan_bench.cpp does.
        if len(matrix) != 4:
            raise ValueError("structured matrix must contain four coefficients")
        matrices.append((tuple(float(value) for value in matrix),))
    return tuple(matrices)


def _seed(case: GpuCase, base_seed: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{case.operator}:{case.precision}:{case.shape}:"
        f"{case.batch}:{case.inverse}:{case.configuration}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _gpu_random(torch: Any, case: GpuCase, seed: int) -> Any:
    _, _, dtype = _precision_types(torch, case.operator, case.precision)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    shape = (case.batch, *case.shape)
    if dtype.is_complex:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=real_dtype, generator=generator, device="cuda")
        imag = torch.randn(shape, dtype=real_dtype, generator=generator, device="cuda")
        return torch.complex(real, imag)
    return torch.randn(shape, dtype=dtype, generator=generator, device="cuda")


def _integer_source(np: Any, case: GpuCase, modulus: int, seed: int) -> Any:
    count = case.batch
    for axis in case.shape:
        count *= int(axis)
    generator = np.random.default_rng(int(seed))
    values = generator.integers(0, int(modulus), size=count, dtype=np.uint64)
    return values.astype(np.uint64, copy=False)


def _ntt_numpy_reference(np: Any, values: Any, shape: Sequence[int], batch: int,
                         modulus: int, inverse: bool) -> Any:
    """Vectorized modular DFT reference for GPU NTT output checking."""
    work = np.asarray(values, dtype=np.uint64).reshape((batch, *shape)).copy()
    for axis, length in enumerate(shape, start=1):
        moved = np.moveaxis(work, axis, -1)
        flat = np.ascontiguousarray(moved).reshape(-1, int(length))
        root = _primitive_root_for_length(int(length), int(modulus))
        if inverse:
            root = pow(root, int(modulus) - 2, int(modulus))
        exponents = np.arange(int(length), dtype=np.int64)[:, None] * np.arange(
            int(length), dtype=np.int64)[None, :]
        twiddle = np.asarray(
            [[pow(root, int(exponent), int(modulus)) for exponent in row]
             for row in exponents],
            dtype=np.int64,
        )
        transformed = (flat.astype(np.int64) @ twiddle) % int(modulus)
        if inverse:
            transformed = (transformed * pow(int(length), int(modulus) - 2,
                                             int(modulus))) % int(modulus)
        flat[...] = transformed.astype(np.uint64, copy=False)
        work = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return work.reshape(-1)


def _integer_error(np: Any, actual: Any, expected: Any, label: str) -> dict[str, Any]:
    actual_u = np.asarray(actual, dtype=np.uint64)
    expected_u = np.asarray(expected, dtype=np.uint64)
    mismatch = np.flatnonzero(actual_u != expected_u)
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"{label}: mismatch at flat index {index}: "
            f"actual={int(actual_u[index])}, expected={int(expected_u[index])}")
    return {"exact": True, "max_abs": 0}


def _assert_finite(torch: Any, value: Any, label: str) -> int:
    """Reject NaN/Inf before tolerance comparison (NaN would otherwise pass)."""
    if value.is_complex():
        finite = torch.isfinite(value.real) & torch.isfinite(value.imag)
    else:
        finite = torch.isfinite(value)
    count = int(finite.detach().cpu().sum().item())
    if count != int(value.numel()):
        raise AssertionError(
            f"{label}: non-finite output/expected values "
            f"({int(value.numel()) - count}/{int(value.numel())})")
    return count


def _mapping_json_text(value: Any) -> tuple[str, Any]:
    """Return stable text and parsed payload for a mapping JSON value.

    Timing reports store the C-ABI string verbatim. Accepting a dict as well
    keeps input-run compatibility with hand-authored/mock reports while
    normalizing it before passing an explicit mapping to the C API.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("mapping_json is empty")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("mapping_json is not valid JSON") from exc
    else:
        payload = value
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if not isinstance(payload, dict):
        raise ValueError("mapping_json must encode a JSON object")
    return text, payload


def _requested_mapping(case: GpuCase) -> tuple[str | None, str | None, Any | None]:
    """Resolve the mapping request and source payload for one correctness case."""
    if case.source_row is not None:
        source_value = case.source_row.get("mapping_json")
        if source_value is None or (isinstance(source_value, str) and not source_value.strip()):
            # Preserve the legacy configuration request for old reports, but
            # keep mapping evidence explicitly unavailable below.
            fallback = "xd:resident-t128" if case.configuration == "resident-t128" else None
            return fallback, None, None
        source_text, source_payload = _mapping_json_text(source_value)
        return source_text, source_text, source_payload
    if case.configuration == "resident-t128":
        return "xd:resident-t128", None, None
    return None, None, None


def _record_base(case: GpuCase, library_sha256: str | None) -> dict[str, Any]:
    source_mapping: str | None = None
    source_mapping_error: str | None = None
    if case.source_row is not None and case.source_row.get("mapping_json") is not None:
        try:
            source_mapping, _ = _mapping_json_text(case.source_row.get("mapping_json"))
        except ValueError as exc:
            source_mapping_error = str(exc)
    record: dict[str, Any] = {
        "id": case.name,
        "operator": case.operator,
        "precision": case.precision,
        "shape": list(case.shape),
        "batch": case.batch,
        "inverse": case.inverse,
        "direction": "inverse" if case.inverse else "forward",
        "configuration": case.configuration,
        "library_sha256": library_sha256,
        "algorithm": None,
        "mapping_json": None,
        "source_mapping_json": source_mapping,
        "mapping_replay_verified": None,
        "mapping_evidence": {
            "status": ("invalid" if source_mapping_error else
                       "unavailable" if case.source_row is not None else "not_requested"),
            "bound": False,
            "exact_match": None,
            "reason": (source_mapping_error or
                       "source row has no mapping_json; timing mapping cannot be replayed"
                       if case.source_row is not None else
                       "no timing source row; correctness mapping is not a timing binding"),
        },
        "error": None,
        "elements_checked": 0,
        "actual_result_checked": False,
    }
    if case.source_row is not None:
        record["source_status"] = case.source_row.get("status")
        record["source_id"] = case.source_row.get("id")
    return record


def _run_fft(torch: Any, case: GpuCase, source: Any, output: Any) -> dict[str, Any]:
    axes = tuple(range(1, len(case.shape) + 1))
    expected = (torch.fft.ifftn if case.inverse else torch.fft.fftn)(
        source, dim=axes, norm="backward")
    finite_actual = _assert_finite(torch, output, "FFT output")
    finite_expected = _assert_finite(torch, expected, "FFT reference")
    error = _assert_close(
        output, expected,
        atol=5e-4 if case.precision == "fp32" else 5e-11,
        rtol=5e-4 if case.precision == "fp32" else 5e-11,
        label=f"FFT {'inverse' if case.inverse else 'forward'}",
    )
    return {"reference": "torch.fft", "error": error,
            "finite_elements": min(finite_actual, finite_expected)}


def _run_fwht(torch: Any, case: GpuCase, source: Any, output: Any) -> dict[str, Any]:
    axes = tuple(range(1, len(case.shape) + 1))
    # Keep the reference independent from the cuButterfly plan, but execute
    # its radix-2 stages on the GPU.  The older CPU helper is intentionally
    # simple and becomes quadratic in wall time for the 64^3 x B=16 scale
    # anchors because it copies every stage through Python.
    expected = source.clone()
    for axis in axes:
        moved = expected.movedim(axis, -1).contiguous()
        n = int(moved.shape[-1])
        if n == 0 or n & (n - 1):
            raise ValueError("FWHT reference requires power-of-two axes")
        rows = moved.reshape(-1, n)
        half = 1
        while half < n:
            for start in range(0, n, 2 * half):
                left = rows[:, start:start + half].clone()
                right = rows[:, start + half:start + 2 * half].clone()
                rows[:, start:start + half] = left + right
                rows[:, start + half:start + 2 * half] = left - right
            half <<= 1
        expected = moved.reshape(expected.movedim(axis, -1).shape).movedim(-1, axis)
    if case.inverse:
        expected = expected / float(math.prod(case.shape))
    finite_actual = _assert_finite(torch, output, "FWHT output")
    finite_expected = _assert_finite(torch, expected, "FWHT reference")
    error = _assert_close(
        output, expected,
        atol=5e-4 if case.precision == "fp32" else 5e-11,
        rtol=5e-4 if case.precision == "fp32" else 5e-11,
        label=f"FWHT {'inverse' if case.inverse else 'forward'}",
    )
    return {"reference": "independent-gpu-radix2-fwht", "error": error,
            "finite_elements": min(finite_actual, finite_expected)}


def _run_ntt(np: Any, case: GpuCase, source: Any, output: Any,
             modulus: int) -> dict[str, Any]:
    source_cpu = source.detach().cpu().numpy().astype(np.uint64, copy=False)
    expected = _ntt_numpy_reference(np, source_cpu, case.shape, case.batch,
                                    modulus, case.inverse)
    actual = output.detach().cpu().numpy().astype(np.uint64, copy=False)
    error = _integer_error(np, actual, expected, f"NTT {'inverse' if case.inverse else 'forward'}")
    return {"reference": "numpy-vectorized-modular-dft", "modulus": modulus,
            "word_bits": 32 if case.precision == "uint32" else 64, "error": error}


def _run_zeta(np: Any, case: GpuCase, source: Any, output: Any) -> dict[str, Any]:
    # PyTorch 1.13 has no uint32 CUDA dtype in this acceptance image; the
    # C-ABI pointer is nevertheless uint32. Preserve the bit pattern through
    # an explicit int32 -> uint32 view on both input and output.
    source_i32 = np.ascontiguousarray(source.detach().cpu().numpy(), dtype=np.int32)
    source_cpu = source_i32.view(np.uint32)
    op = OP_SUBSET_ZETA if case.operator == "subset-zeta" else OP_SUPERSET_ZETA
    expected = _zeta_reference(source_cpu, (case.batch, *case.shape), op, case.inverse)
    actual_i32 = np.ascontiguousarray(output.detach().cpu().numpy(), dtype=np.int32)
    actual = actual_i32.view(np.uint32)
    error = _integer_error(
        np, actual, expected,
        f"{case.operator} {'inverse' if case.inverse else 'forward'}",
    )
    return {"reference": "check_xd._zeta_reference", "word_bits": 32,
            "error": error, "finite_elements": int(actual.size)}


def _run_structured(torch: Any, case: GpuCase, source: Any, output: Any,
                    matrices: Sequence[Sequence[Sequence[float]]]) -> dict[str, Any]:
    expected = _structured_reference(torch, source, case.shape, matrices, case.inverse)
    finite_actual = _assert_finite(torch, output, "structured output")
    finite_expected = _assert_finite(torch, expected, "structured reference")
    error = _assert_close(
        output, expected,
        atol=8e-4 if case.precision == "fp32" else 8e-11,
        rtol=8e-4 if case.precision == "fp32" else 8e-11,
        label=f"structured-2x2 {'inverse' if case.inverse else 'forward'}",
    )
    return {"reference": "check_xd._structured_reference", "error": error,
            "finite_elements": min(finite_actual, finite_expected)}


def _run_gpu_case(api: Api, handle: Handle, torch: Any, np: Any, case: GpuCase,
                  library_sha256: str | None, base_seed: int,
                  modulus: int, matrix: Sequence[float]) -> dict[str, Any]:
    started = time.perf_counter()
    record = _record_base(case, library_sha256)
    plan = None
    try:
        storage, compute, dtype = _precision_types(torch, case.operator, case.precision)
        op, _, _ = _operator_constants(case.operator)
        source: Any
        if case.operator in ("ntt", "subset-zeta", "superset-zeta"):
            source_cpu = _integer_source(np, case, modulus, _seed(case, base_seed))
            host_dtype = np.int32 if dtype == torch.int32 else np.int64
            source = torch.from_numpy(source_cpu.astype(host_dtype, copy=False)).to(device="cuda")
        else:
            source = _gpu_random(torch, case, _seed(case, base_seed))
        source_before = source.clone()
        output = torch.empty_like(source)
        matrices = None
        if case.operator == "structured-2x2":
            matrices = _structured_matrices(case.shape, matrix)
        mapping, source_mapping_text, source_mapping_payload = _requested_mapping(case)
        if source_mapping_text is not None:
            record["source_mapping_json"] = source_mapping_text
            record["mapping_evidence"] = {
                "status": "pending",
                "bound": True,
                "exact_match": None,
                "source_mapping_json": source_mapping_text,
                "reason": "source mapping will be replayed as an explicit plan mapping",
            }
        elif mapping is not None:
            record["mapping_evidence"]["requested_mapping"] = mapping
        spec: dict[str, Any] = {
            "op": op,
            "storage": storage,
            "compute": compute,
            "shape": case.shape,
            "batch": case.batch,
            "mode": EXEC_XD,
            "direction": case.direction,
            "mapping": mapping,
        }
        if case.operator == "ntt":
            spec.update(modulus=modulus, word_bits=32 if case.precision == "uint32" else 64)
        if matrices is not None:
            spec["matrices"] = matrices
        plan = make_plan(api, handle, torch, **spec)
        metadata = dict(plan.metadata)
        record.update({
            "algorithm": metadata.get("algorithm"),
            "mapping_json": metadata.get("mapping_json"),
            "selection_reason": metadata.get("selection_reason"),
            "selection_source": metadata.get("selection_source"),
            "workspace_bytes": metadata.get("workspace_bytes"),
            "input_bytes": metadata.get("input_bytes"),
            "output_bytes": metadata.get("output_bytes"),
        })
        observed_mapping = metadata.get("mapping_json")
        if source_mapping_payload is not None:
            observed_mapping_text, observed_mapping_payload = _mapping_json_text(
                observed_mapping)
            exact_match = source_mapping_payload == observed_mapping_payload
            record["mapping_replay_verified"] = exact_match
            record["mapping_evidence"] = {
                "status": "verified" if exact_match else "mismatch",
                "bound": True,
                "exact_match": exact_match,
                "source_mapping_json": source_mapping_text,
                "observed_mapping_json": observed_mapping_text,
                "reason": ("created plan metadata matches the source mapping"
                           if exact_match else
                           "created plan metadata differs from the source mapping"),
            }
            if not exact_match:
                raise AssertionError("source mapping replay mismatch")
        elif case.source_row is not None:
            record["mapping_evidence"] = {
                **record["mapping_evidence"],
                "observed_mapping_json": observed_mapping,
                "reason": "source row has no mapping_json; correctness result is unbound",
            }
        plan.execute(source, output)
        torch.cuda.synchronize()
        record["elements_checked"] = int(output.numel())
        if case.operator == "fft":
            details = _run_fft(torch, case, source, output)
        elif case.operator == "fwht":
            details = _run_fwht(torch, case, source, output)
        elif case.operator == "ntt":
            details = _run_ntt(np, case, source, output, modulus)
        elif case.operator in ("subset-zeta", "superset-zeta"):
            details = _run_zeta(np, case, source, output)
        elif case.operator == "structured-2x2":
            details = _run_structured(torch, case, source, output, matrices or ())
        else:  # pragma: no cover - inventory validates this before execution
            raise ValueError(f"unsupported operator {case.operator!r}")
        record.update(details)
        record["input_unchanged"] = bool(torch.equal(source, source_before))
        if not record["input_unchanged"]:
            raise AssertionError("out-of-place GPU execution modified its input")
        record["actual_result_checked"] = True
        record["status"] = "passed"
    except UnsupportedCase as exc:
        record.update({"status": "skipped", "reason": str(exc)})
    except CApiError as exc:
        if exc.status == STATUS_NOT_SUPPORTED:
            record.update({"status": "skipped", "reason": str(exc),
                           "status_code": exc.status})
        else:
            record.update({"status": "failed", "error": str(exc),
                           "status_code": exc.status})
    except (AssertionError, RuntimeError, ValueError, OverflowError) as exc:
        record.update({"status": "failed", "error": str(exc)})
    finally:
        if plan is not None:
            plan.close()
    record["elapsed_s"] = time.perf_counter() - started
    return record


def _matrix_cases(profile: str, selected: Sequence[str] | None = None,
                  include_resident: bool = False) -> list[GpuCase]:
    selected_set = set(selected or ())
    operators = [item for item in OPERATORS
                 if not selected_set or item["name"] in selected_set]
    configurations = ("selected", "resident-t128") if include_resident else ("selected",)
    cases: list[GpuCase] = []
    for item in operators:
        for precision in item["precisions"]:
            for shape in PROFILE_SHAPES[profile]:
                for batch in PROFILE_BATCHES[profile]:
                    for inverse in DIRECTIONS:
                        for configuration in configurations:
                            cases.append(GpuCase(item["name"], precision, tuple(shape),
                                                 int(batch), bool(inverse), configuration))
    return cases


def _load_input_cases(path: pathlib.Path, include_resident: bool) -> list[GpuCase]:
    source = path / "rows.json" if path.is_dir() else path
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a rows.json list")
    cases: list[GpuCase] = []
    legal_precisions = {
        item["name"]: set(item["precisions"]) for item in OPERATORS
    }
    for row in payload:
        try:
            configuration = str(row.get("configuration", "selected"))
            if configuration == "resident-t128" and not include_resident:
                continue
            if configuration not in ("selected", "resident-t128"):
                raise ValueError(f"unknown configuration {configuration!r}")
            operator = str(row["operator"])
            precision = str(row["precision"])
            if operator not in legal_precisions:
                raise ValueError(f"unknown operator {operator!r}")
            if precision not in legal_precisions[operator]:
                raise ValueError(
                    f"precision {precision!r} is illegal for {operator!r}")
            raw_shape = row["shape"]
            if isinstance(raw_shape, str):
                shape = tuple(int(x) for x in raw_shape.split("x"))
            else:
                shape = tuple(int(x) for x in raw_shape)
            if len(shape) not in (2, 3) or any(value <= 0 for value in shape):
                raise ValueError("shape must be positive rank-2 or rank-3")
            batch = int(row["batch"])
            if batch <= 0:
                raise ValueError("batch must be positive")
            cases.append(GpuCase(
                operator, precision, shape,
                batch, bool(row.get("inverse", False)), configuration,
                source_row=dict(row),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid input-run row: {row!r}: {exc}") from exc
    return cases


def _parse_matrix(value: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("matrix must contain four numbers") from exc
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError(
            "matrix must contain four finite comma-separated numbers")
    return values  # type: ignore[return-value]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _validate_modulus(modulus: int, cases: Sequence[GpuCase]) -> None:
    ntt_lengths = {
        int(length)
        for case in cases
        if case.operator == "ntt"
        for length in case.shape
    }
    if not ntt_lengths:
        return
    if modulus <= max(ntt_lengths) or modulus >= (1 << 64):
        raise ValueError(
            f"NTT modulus must be in (max_axis={max(ntt_lengths)}, 2**64)")
    if not _is_prime(modulus):
        raise ValueError(f"NTT modulus={modulus} is not prime")
    incompatible = sorted(length for length in ntt_lengths
                          if (modulus - 1) % length != 0)
    if incompatible:
        raise ValueError(
            f"NTT modulus={modulus} has no primitive roots for axes {incompatible}")


def _input_scope(path: pathlib.Path, cases: Sequence[GpuCase],
                 include_resident: bool) -> dict[str, Any]:
    source = path / "rows.json" if path.is_dir() else path
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else []
    config_counts: dict[str, int] = {}
    for row in rows:
        configuration = (
            str(row.get("configuration", "selected"))
            if isinstance(row, dict) else "invalid"
        )
        config_counts[configuration] = config_counts.get(configuration, 0) + 1
    mapping_rows = sum(
        isinstance(row, dict) and bool(row.get("mapping_json")) for row in rows
    )
    return {
        "source": str(source),
        "source_rows": len(rows),
        "loaded_rows": len(cases),
        "filtered_rows": len(rows) - len(cases),
        "source_configurations": config_counts,
        "source_mapping_rows": mapping_rows,
        "source_mapping_missing_rows": len(rows) - mapping_rows,
        "include_resident": include_resident,
        "scope": "all source rows" if include_resident else "selected rows only",
        "mapping_binding": (
            "source mapping replay is available for rows with mapping_json; "
            "rows without it are correctness-only and unbound"
        ),
    }


def _unexecuted_cases(cases: Sequence[GpuCase], library_sha256: str | None,
                     status: str, reason: str) -> list[dict[str, Any]]:
    """Materialize case-level status when setup prevents any execution."""
    return [dict(_record_base(case, library_sha256), status=status, reason=reason)
            for case in cases]


def _load_checkpoint(path: pathlib.Path | None, cases: Sequence[GpuCase],
                     library_sha256: str | None, profile: str,
                     modulus: int, matrix: Sequence[float]) -> dict[str, dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if (payload.get("library_sha256") != library_sha256 or
            payload.get("profile") != profile or
            payload.get("modulus") != modulus or
            payload.get("structured_matrix") != list(matrix)):
        return {}
    expected_ids = {case.name for case in cases}
    expected_cases = {case.name: case for case in cases}
    records = payload.get("cases", [])
    if not isinstance(records, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = record.get("id")
        case = expected_cases.get(case_id)
        source_mapping_required = bool(
            case is not None and case.source_row is not None and
            case.source_row.get("mapping_json")
        )
        # A pre-evidence checkpoint must not silently turn an old correctness
        # result into proof that the timing mapping was replayed.
        replay_evidence_present = (
            not source_mapping_required or
            record.get("mapping_replay_verified") is not None
        )
        if source_mapping_required and replay_evidence_present:
            try:
                _, requested = _mapping_json_text(case.source_row["mapping_json"])
                _, recorded = _mapping_json_text(record.get("source_mapping_json"))
                replay_evidence_present = requested == recorded
                if record.get("status") == "passed":
                    _, observed = _mapping_json_text(record.get("mapping_json"))
                    replay_evidence_present = (replay_evidence_present and requested == observed
                                               and record.get("mapping_replay_verified") is True
                                               and record.get("actual_result_checked") is True)
            except (TypeError, ValueError):
                replay_evidence_present = False
        if (case_id in expected_ids and
                record.get("status") in ("passed", "failed", "skipped") and
                replay_evidence_present):
            result[str(case_id)] = record
    return result


def _write_checkpoint(path: pathlib.Path | None, cases: Sequence[GpuCase],
                      records: dict[str, dict[str, Any]], library_sha256: str | None,
                      profile: str, modulus: int, matrix: Sequence[float]) -> None:
    if path is None:
        return
    payload = {
        "schema": 1,
        "status": "complete" if all(case.name in records for case in cases)
        else "in-progress",
        "library_sha256": library_sha256,
        "profile": profile,
        "modulus": modulus,
        "structured_matrix": list(matrix),
        "completed": sum(case.name in records for case in cases),
        "requested": len(cases),
        "cases": [records[case.name] for case in cases if case.name in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _summary(cases: Sequence[GpuCase], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {status: sum(row.get("status") == status for row in records)
              for status in ("passed", "failed", "skipped", "missing", "incomplete")}
    return {
        "requested": len(cases),
        "observed": len(records),
        "passed": counts["passed"],
        "failed": counts["failed"],
        "skipped": counts["skipped"],
        "missing": counts["missing"],
        "incomplete": (counts["incomplete"] + counts["skipped"] + counts["missing"] +
                       max(0, len(cases) - len(records))),
        "mapping_replay_verified": sum(
            row.get("mapping_replay_verified") is True for row in records
        ),
        "mapping_replay_mismatched": sum(
            row.get("mapping_replay_verified") is False for row in records
        ),
        "mapping_replay_unavailable": sum(
            (row.get("mapping_evidence") or {}).get("status") == "unavailable"
            for row in records
        ),
    }


def _mapping_summary(cases: Sequence[GpuCase],
                     records: Sequence[dict[str, Any]],
                     required: bool) -> dict[str, Any]:
    """Summarize whether correctness rows are bound to timing mappings."""
    source_rows = sum(case.source_row is not None for case in cases)
    source_with_mapping = sum(
        case.source_row is not None and bool(case.source_row.get("mapping_json"))
        for case in cases
    )
    verified = sum(row.get("mapping_replay_verified") is True for row in records)
    mismatched = sum(row.get("mapping_replay_verified") is False for row in records)
    unavailable = sum(
        (row.get("mapping_evidence") or {}).get("status") == "unavailable"
        for row in records
    )
    if mismatched:
        status = "mismatch"
    elif source_rows == 0:
        status = "not_requested"
    elif unavailable or source_with_mapping < source_rows:
        status = "unavailable"
    elif verified == source_with_mapping:
        status = "verified"
    else:
        status = "incomplete"
    return {
        "status": status,
        "required": required,
        "source_rows": source_rows,
        "source_rows_with_mapping": source_with_mapping,
        "source_rows_without_mapping": source_rows - source_with_mapping,
        "verified": verified,
        "mismatched": mismatched,
        "unavailable": unavailable,
        "bound_timing_claim_allowed": status == "verified",
    }


def _status_for_summary(summary: dict[str, Any]) -> str:
    if summary["failed"]:
        return "failed"
    if (summary["missing"] or summary["skipped"] or summary["incomplete"] or
            summary["observed"] == 0):
        return "incomplete"
    return "passed"


def _runtime_metadata(torch: Any, compile_mode: str) -> dict[str, Any]:
    device_index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_index": device_index,
        "device_name": str(torch.cuda.get_device_name(device_index)),
        "device_capability": list(torch.cuda.get_device_capability(device_index)),
        "memory_bytes": int(properties.total_memory),
        "compile_mode": compile_mode,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=pathlib.Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--profile", choices=tuple(PROFILE_SHAPES), default="coverage")
    parser.add_argument("--operators", default="all",
                        help="comma-separated operator subset")
    parser.add_argument("--include-resident", "--resident-control", action="store_true",
                        help="also validate explicit xd:resident-t128 cases")
    parser.add_argument("--input-run", type=pathlib.Path,
                        help="rows.json (or its directory) from benchmark_all_operators_xd.py")
    parser.add_argument(
        "--require-source-mapping", action="store_true",
        help="require every input-run row to carry mapping_json for exact replay",
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--compile-mode", choices=("research", "auto", "precompiled"),
        default=os.environ.get("CUBUTTERFLY_COMPILE_MODE", "research"),
    )
    parser.add_argument("--modulus", type=int, default=DEFAULT_MODULUS)
    parser.add_argument("--matrix", type=_parse_matrix,
                        default=DEFAULT_MATRIX,
                        help="structured-2x2 matrix m00,m01,m10,m11")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        help="resume completed case records from this JSON path")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress per-case progress on stderr")
    parser.add_argument("--dry-run", action="store_true",
                        help="construct/report cases without importing torch or touching CUDA")
    parser.add_argument("--output", type=pathlib.Path,
                        help="write the JSON report to this path")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)
    requested = [value.strip() for value in args.operators.split(",")
                 if value.strip() and value.strip() != "all"]
    known = {item["name"] for item in OPERATORS}
    unknown = sorted(set(requested) - known)
    if unknown:
        parser.error(f"unknown operators: {', '.join(unknown)}")
    if args.modulus <= 1 or args.modulus >= (1 << 64):
        parser.error("--modulus must be in (1, 2**64)")
    if args.max_cases is not None and args.max_cases < 0:
        parser.error("--max-cases must be nonnegative")
    args.selected_operators = requested or None
    args.library = args.library.resolve()
    if args.input_run is not None:
        args.input_run = args.input_run.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.require_source_mapping and args.input_run is None:
        parser.error("--require-source-mapping requires --input-run")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_scope: dict[str, Any] | None = None
    if args.input_run is not None:
        try:
            cases = _load_input_cases(args.input_run, args.include_resident)
            input_scope = _input_scope(args.input_run, cases, args.include_resident)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            report = {"status": "error", "error": str(exc), "input_run": str(args.input_run)}
            text = json.dumps(report, indent=args.indent, sort_keys=True)
            print(text)
            return 2
    else:
        cases = _matrix_cases(args.profile, args.selected_operators, args.include_resident)
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if input_scope is not None:
        input_scope["requested_after_max_cases"] = len(cases)
    checkpoint_profile = "input-run" if args.input_run is not None else args.profile

    if args.require_source_mapping:
        missing_mapping = [case.name for case in cases
                           if case.source_row is None or
                           not case.source_row.get("mapping_json")]
        if missing_mapping:
            report = {
                "status": "error",
                "error": (
                    "--require-source-mapping found input rows without mapping_json"
                ),
                "input_run": str(args.input_run),
                "missing_mapping_rows": missing_mapping,
                "mapping_evidence": {
                    "status": "unavailable",
                    "required": True,
                    "bound_timing_claim_allowed": False,
                },
            }
            text = json.dumps(report, indent=args.indent, sort_keys=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n", encoding="utf-8")
            print(text)
            return 2

    try:
        _validate_modulus(args.modulus, cases)
    except ValueError as exc:
        report = {"status": "error", "error": str(exc),
                  "profile": checkpoint_profile, "input_run": str(args.input_run)
                  if args.input_run is not None else None}
        text = json.dumps(report, indent=args.indent, sort_keys=True)
        print(text)
        return 2

    library_sha256 = _sha256(args.library)
    report: dict[str, Any] = {
        "script": str(pathlib.Path(__file__).resolve()),
        "profile": checkpoint_profile,
        "input_run": str(args.input_run) if args.input_run is not None else None,
        "input_scope": input_scope,
        "library": str(args.library),
        "library_sha256": library_sha256,
        "modulus": args.modulus,
        "structured_matrix": list(args.matrix),
        "compile_mode": args.compile_mode,
        "gpu_used": False,
        "runtime": None,
        "checkpoint": None,
        "correctness": {
            "actual_gpu_results_checked": False,
            "timing_is_not_used_for_status": True,
            "mapping_evidence": None,
            "require_source_mapping": args.require_source_mapping,
            "api_helpers": ["Api", "Handle", "make_plan", "Plan"],
            "reference_helpers": ["_fwht_reference", "_zeta_reference",
                                  "_structured_reference", "torch.fft"],
        },
        "matrix": {
            "requested_cells": len(cases),
            "operators": sorted({case.operator for case in cases}),
            "shapes": [list(shape) for shape in sorted({case.shape for case in cases})],
            "batches": sorted({case.batch for case in cases}),
            "directions": ["forward", "inverse"],
            "configurations": sorted({case.configuration for case in cases}),
        },
        "cases": [],
    }
    if args.dry_run:
        report["cases"] = [dict(_record_base(case, library_sha256), status="dry-run",
                                 actual_result_checked=False)
                            for case in cases]
        report["status"] = "dry-run"
        report["summary"] = {"requested": len(cases), "passed": 0, "failed": 0,
                              "skipped": 0, "missing": 0, "incomplete": 0,
                              "observed": len(cases), "dry_run": True}
    else:
        try:
            import numpy as np
            import torch
        except Exception as exc:
            report["status"] = "incomplete"
            report["reason"] = f"runtime import failed: {exc}"
            report["cases"] = _unexecuted_cases(
                cases, library_sha256, "skipped", report["reason"])
        else:
            if not torch.cuda.is_available():
                report["status"] = "incomplete"
                report["reason"] = "CUDA is unavailable"
                report["cases"] = _unexecuted_cases(
                    cases, library_sha256, "skipped", report["reason"])
            elif not args.library.is_file():
                report["status"] = "incomplete"
                report["reason"] = f"library not found: {args.library}"
                report["cases"] = _unexecuted_cases(
                    cases, library_sha256, "missing", report["reason"])
            else:
                if args.device is not None:
                    torch.cuda.set_device(args.device)
                report["runtime"] = _runtime_metadata(torch, args.compile_mode)
                resumed = _load_checkpoint(
                    args.checkpoint, cases, library_sha256, checkpoint_profile,
                    args.modulus, args.matrix)
                records = dict(resumed)
                pending = [case for case in cases if case.name not in records]
                report["checkpoint"] = {
                    "path": str(args.checkpoint) if args.checkpoint is not None else None,
                    "resumed": len(resumed), "pending": len(pending),
                }
                if not args.no_progress and resumed:
                    print(f"resumed {len(resumed)}/{len(cases)} cases", file=sys.stderr)
                if pending:
                    try:
                        api = Api(args.library)
                        stream = int(torch.cuda.current_stream().cuda_stream)
                        with Handle(api, stream) as handle:
                            for case in pending:
                                row = _run_gpu_case(
                                    api, handle, torch, np, case, library_sha256,
                                    args.seed, args.modulus, args.matrix)
                                records[case.name] = row
                                _write_checkpoint(
                                    args.checkpoint, cases, records, library_sha256,
                                    checkpoint_profile, args.modulus, args.matrix)
                                if not args.no_progress:
                                    print(
                                        f"[{len(records)}/{len(cases)}] {case.name} "
                                        f"{row.get('status')}", file=sys.stderr,
                                        flush=True)
                    except (OSError, RuntimeError, FileNotFoundError) as exc:
                        report["status"] = "incomplete"
                        report["reason"] = str(exc)
                        remaining = [case for case in cases if case.name not in records]
                        for row in _unexecuted_cases(
                                remaining, library_sha256, "missing", report["reason"]):
                            records[row["id"]] = row
                _write_checkpoint(
                    args.checkpoint, cases, records, library_sha256,
                    checkpoint_profile, args.modulus, args.matrix)
                report["cases"] = [records[case.name] for case in cases
                                   if case.name in records]
                report["correctness"]["actual_gpu_results_checked"] = any(
                    row.get("actual_result_checked", False) for row in report["cases"])
                report["gpu_used"] = bool(report["runtime"] if "runtime" in report else False)

    if "summary" not in report:
        report["summary"] = _summary(cases, report.get("cases", []))
    elif report.get("status") != "dry-run":
        report["summary"] = _summary(cases, report.get("cases", []))
    report["mapping_evidence"] = _mapping_summary(
        cases, report.get("cases", []), args.require_source_mapping
    )
    report["correctness"]["mapping_evidence"] = report["mapping_evidence"]
    if report.get("status") not in ("dry-run", "error"):
        report["status"] = _status_for_summary(report["summary"])

    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report.get("status") == "failed":
        return 1
    if report.get("status") == "error":
        return 2
    if report.get("status") == "incomplete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
