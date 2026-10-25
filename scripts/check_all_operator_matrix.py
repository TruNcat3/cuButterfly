#!/usr/bin/env python3
"""Run a bounded, output-checked CPU matrix for the public xD operators.

This is deliberately a correctness harness, not a timing harness.  It runs
all requested cells in one Python process using CPU tensors/arrays, computes
the actual result, and compares that result with the independent reference
helpers already used by :mod:`tests.check_xd`.  The C-ABI ``Api``/``Plan``
classes and CUDA-only extended entry points are imported for contract
provenance, but are not instantiated: this command never allocates CUDA
memory or claims to validate a GPU lowering.

The default ``coverage`` profile is the practical rank-2/rank-3 matrix.  The
``scale`` and ``full`` profiles add 32^3 and 64^3 cells separately.  Direct
CPU NTT references are quadratic in each transformed line, so NTT scale cells
are reported as ``skipped`` rather than turning an impractical run into a
false pass or silently using timing as correctness.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

# Keep all semantic constants and the non-FFT references in one place.  In
# particular, do not copy the zeta/NTT/FWHT reference implementations here.
from check_xd import (  # noqa: E402
    Api,
    Plan,
    DATA_COMPLEX_FP32,
    DATA_COMPLEX_FP64,
    DATA_FP32,
    DATA_FP64,
    DATA_UINT32,
    DATA_UINT64,
    FORWARD,
    INVERSE,
    OP_FFT,
    OP_FWHT,
    OP_NTT,
    OP_STRUCTURED_2X2,
    OP_SUBSET_ZETA,
    OP_SUPERSET_ZETA,
    _assert_close,
    _fwht_reference,
    _ntt_reference,
    _primitive_root_for_length,
    _prod,
    _structured_reference,
    _zeta_reference,
    _embedding_fwht_case,
    _embedding_rank3_fwht_case,
    _inplace_fft_case,
    _inplace_fwht_case,
    _inplace_rank2_fwht_case,
    _real_fft_case,
    _strided_fwht_case,
    _strided_rank3_fwht_case,
)


PROFILE_SHAPES: dict[str, tuple[tuple[int, ...], ...]] = {
    "coverage": ((8, 8), (32, 32), (4, 4, 4), (8, 8, 8)),
    "scale": ((32, 32, 32), (64, 64, 64)),
}
PROFILE_BATCHES: dict[str, tuple[int, ...]] = {
    "coverage": (1, 4, 16),
    # Scale is intentionally bounded to one representative batch.  The
    # required B=1/4/16 matrix is in the default coverage profile.
    "scale": (1,),
}
OPERATORS: tuple[str, ...] = (
    "fft", "fwht", "ntt", "subset-zeta", "superset-zeta", "structured-2x2",
)
PRECISIONS: dict[str, tuple[str, ...]] = {
    "fft": ("fp32", "fp64"),
    "fwht": ("fp32", "fp64"),
    "ntt": ("uint32", "uint64"),
    # The public C ABI intentionally restricts both zeta operators to uint32.
    "subset-zeta": ("uint32",),
    "superset-zeta": ("uint32",),
    "structured-2x2": ("fp32", "fp64"),
}

_MATRICES: tuple[tuple[tuple[float, float, float, float], ...], ...] = (
    ((1.10, 0.20, -0.10, 0.90), (0.80, -0.30, 0.25, 1.20),
     (1.02, -0.08, 0.12, 0.94)),
    ((0.95, 0.15, 0.05, 1.05), (1.20, -0.20, 0.10, 0.85),
     (0.75, 0.40, -0.15, 1.10)),
    ((1.05, 0.10, -0.05, 0.97), (0.90, -0.25, 0.15, 1.08),
     (1.15, 0.05, 0.20, 0.92)),
)
_MASK32 = (1 << 32) - 1
_NTT_MODULUS = 97


@dataclass(frozen=True)
class MatrixCase:
    operator: str
    precision: str
    shape: tuple[int, ...]
    batch: int
    inverse: bool

    @property
    def direction(self) -> str:
        return "inverse" if self.inverse else "forward"

    @property
    def name(self) -> str:
        shape = "x".join(str(value) for value in self.shape)
        return (f"{self.operator}_{self.precision}_{shape}_b{self.batch}_"
                f"{('inv' if self.inverse else 'fwd')}")


class SkipCase(RuntimeError):
    """A valid matrix cell that this CPU runner cannot execute economically."""


def _matrices_for_shape(
        shape: Sequence[int],
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    """Expand the deterministic stage-matrix seed to every radix-2 stage."""
    result: list[tuple[tuple[float, float, float, float], ...]] = []
    for axis, length in enumerate(shape):
        source = _MATRICES[axis % len(_MATRICES)]
        stages = int(math.log2(length))
        if 1 << stages != length:
            raise ValueError("structured CPU matrix requires power-of-two axes")
        result.append(tuple(source[index % len(source)] for index in range(stages)))
    return tuple(result)


def _import_runtime() -> tuple[Any | None, Any | None, str | None]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on the host image
        return None, None, f"numpy import failed: {exc}"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on the host image
        return np, None, f"torch import failed: {exc}"
    return np, torch, None


def _seed(case: MatrixCase, base: int) -> int:
    digest = hashlib.sha256(
        f"{base}:{case.operator}:{case.precision}:{case.shape}:{case.batch}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF


def _torch_random(torch: Any, case: MatrixCase, seed: int) -> Any:
    if case.precision == "fp32":
        dtype = torch.float32 if case.operator != "fft" else torch.complex64
    elif case.precision == "fp64":
        dtype = torch.float64 if case.operator != "fft" else torch.complex128
    else:
        raise ValueError(f"not a floating-point precision: {case.precision}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shape = (case.batch, *case.shape)
    if dtype.is_complex:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=real_dtype, generator=generator, device="cpu")
        imag = torch.randn(shape, dtype=real_dtype, generator=generator, device="cpu")
        return torch.complex(real, imag)
    return torch.randn(shape, dtype=dtype, generator=generator, device="cpu")


def _integer_source(np: Any, case: MatrixCase, seed: int) -> Any:
    count = case.batch * _prod(case.shape)
    # Keep values small enough for a portable legal uint32/uint64 modulus while
    # retaining a nontrivial pattern across batches and all logical axes.
    values = (np.arange(count, dtype=np.uint64) * 17 + seed + 3) % _NTT_MODULUS
    return values.astype(np.uint64, copy=False)


def _fwht_actual(torch: Any, value: Any, axes: Sequence[int]) -> Any:
    """A CPU execution path kept separate from check_xd's reference helper."""
    out = value.clone()
    for axis in axes:
        moved = out.movedim(axis, -1).contiguous()
        n = moved.shape[-1]
        rows = moved.reshape(-1, n)
        half = 1
        while half < n:
            for start in range(0, n, 2 * half):
                left = rows[:, start:start + half].clone()
                right = rows[:, start + half:start + 2 * half].clone()
                rows[:, start:start + half] = left + right
                rows[:, start + half:start + 2 * half] = left - right
            half <<= 1
        # ``moved`` has the axis at the end; restore its pre-move shape before
        # moving the transformed axis back to its logical position.
        out = moved.reshape(moved.shape).movedim(-1, axis)
    return out


def _ntt_line_actual(values: Sequence[int], modulus: int, inverse: bool) -> list[int]:
    """Direct DFT form, independent of check_xd._ntt_line's implementation."""
    n = len(values)
    primitive = _primitive_root_for_length(n, modulus)
    root = pow(primitive, modulus - 2, modulus) if inverse else primitive
    result: list[int] = []
    for k in range(n):
        acc = 0
        for j, value in enumerate(values):
            acc = (acc + int(value) * pow(root, j * k, modulus)) % modulus
        if inverse:
            acc = (acc * pow(n, modulus - 2, modulus)) % modulus
        result.append(acc)
    return result


def _ntt_actual(np: Any, values: Any, shape: Sequence[int], batch: int,
                modulus: int, inverse: bool) -> Any:
    work = np.asarray(values, dtype=np.uint64).reshape((batch, *shape)).copy()
    for axis, n in enumerate(shape, start=1):
        moved = np.moveaxis(work, axis, -1)
        flat = np.ascontiguousarray(moved).reshape(-1, n)
        for row in range(flat.shape[0]):
            flat[row, :] = np.asarray(
                _ntt_line_actual(flat[row, :], modulus, inverse), dtype=np.uint64)
        work = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return work.reshape(-1)


def _zeta_actual(np: Any, values: Any, shape: Sequence[int], operator: str,
                 inverse: bool) -> Any:
    work = np.asarray(values, dtype=np.uint64).reshape((-1, *shape)).copy()
    for axis, n in enumerate(shape, start=1):
        moved = np.moveaxis(work, axis, -1)
        flat = np.ascontiguousarray(moved).reshape(-1, n)
        half = 1
        while half < n:
            for start in range(0, n, 2 * half):
                left = flat[:, start:start + half].copy()
                right = flat[:, start + half:start + 2 * half].copy()
                if operator == "subset-zeta":
                    if inverse:
                        right = (right - left) & _MASK32
                    else:
                        right = (right + left) & _MASK32
                else:
                    if inverse:
                        left = (left - right) & _MASK32
                    else:
                        left = (left + right) & _MASK32
                flat[:, start:start + half] = left
                flat[:, start + half:start + 2 * half] = right
            half <<= 1
        work = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return work.reshape(-1)


def _structured_actual(np: Any, value: Any, shape: Sequence[int],
                       matrices: Sequence[Sequence[Sequence[float]]],
                       inverse: bool) -> Any:
    """Execute the stage matrices with NumPy, separate from the torch oracle."""
    # ``np.asarray(copy=...)`` is only available in newer NumPy releases;
    # ``np.array`` keeps this CPU harness usable with the repository image.
    work = np.array(value.detach().cpu().numpy(), copy=True)
    for axis, n in enumerate(shape):
        tensor_axis = axis + 1
        moved = np.moveaxis(work, tensor_axis, -1)
        flat = np.ascontiguousarray(moved).reshape(-1, n)
        half = 1
        stage = 0
        stage_matrices = matrices[axis]
        while half < n:
            matrix = np.asarray(
                stage_matrices[0] if len(stage_matrices) == 1 else stage_matrices[stage],
                dtype=work.dtype,
            ).reshape(2, 2)
            if inverse:
                matrix = np.linalg.inv(matrix)
            for start in range(0, n, 2 * half):
                left = flat[:, start:start + half].copy()
                right = flat[:, start + half:start + 2 * half].copy()
                flat[:, start:start + half] = matrix[0, 0] * left + matrix[0, 1] * right
                flat[:, start + half:start + 2 * half] = matrix[1, 0] * left + matrix[1, 1] * right
            half <<= 1
            stage += 1
        work = np.moveaxis(flat.reshape(moved.shape), -1, tensor_axis)
    return work


def _float_error(torch: Any, actual: Any, expected: Any, *, atol: float,
                 rtol: float, label: str) -> dict[str, float]:
    return _assert_close(actual, expected, atol=atol, rtol=rtol, label=label)


def _integer_error(np: Any, actual: Any, expected: Any, label: str) -> dict[str, Any]:
    actual_u = np.asarray(actual, dtype=np.uint64)
    expected_u = np.asarray(expected, dtype=np.uint64)
    mismatch = np.flatnonzero(actual_u != expected_u)
    if mismatch.size:
        index = int(mismatch[0])
        raise AssertionError(
            f"{label}: mismatch at flat index {index}: "
            f"actual={int(actual_u[index])}, expected={int(expected_u[index])}")
    return {"exact": True, "elements": int(actual_u.size), "max_abs": 0}


def _run_fft(np: Any, torch: Any, case: MatrixCase, seed: int) -> dict[str, Any]:
    source = _torch_random(torch, case, seed)
    axes = tuple(range(1, len(case.shape) + 1))
    source_np = source.numpy()
    expected_np = (np.fft.ifftn if case.inverse else np.fft.fftn)(
        source_np, axes=axes)
    actual = (torch.fft.ifftn if case.inverse else torch.fft.fftn)(
        source, dim=axes, norm="backward")
    expected = torch.from_numpy(expected_np.astype(source.numpy().dtype, copy=False))
    tolerance = (3e-5, 3e-5) if case.precision == "fp32" else (5e-11, 5e-11)
    error = _float_error(torch, actual, expected, atol=tolerance[0], rtol=tolerance[1],
                         label=f"FFT {case.direction}")
    return {"actual_execution": "torch.fft.cpu", "reference": "numpy.fft",
            "error": error}


def _run_fwht(np: Any, torch: Any, case: MatrixCase, seed: int) -> dict[str, Any]:
    source = _torch_random(torch, case, seed)
    axes = tuple(range(1, len(case.shape) + 1))
    actual = _fwht_actual(torch, source, axes)
    expected = _fwht_reference(torch, source, axes)
    if case.inverse:
        scale = float(_prod(case.shape))
        actual = actual / scale
        expected = expected / scale
    tolerance = (4e-4, 4e-4) if case.precision == "fp32" else (5e-11, 5e-11)
    error = _float_error(torch, actual, expected, atol=tolerance[0], rtol=tolerance[1],
                         label=f"FWHT {case.direction}")
    return {"actual_execution": "cpu-radix2", "reference": "check_xd._fwht_reference",
            "error": error}


def _run_ntt(np: Any, case: MatrixCase, seed: int) -> dict[str, Any]:
    if max(case.shape) > 32:
        raise SkipCase("CPU direct NTT reference is quadratic; scale axis exceeds 32")
    source = _integer_source(np, case, seed)
    actual = _ntt_actual(np, source, case.shape, case.batch, _NTT_MODULUS, case.inverse)
    expected = np.asarray(
        _ntt_reference(source, case.shape, case.batch, _NTT_MODULUS, case.inverse),
        dtype=np.uint64,
    )
    error = _integer_error(np, actual, expected, f"NTT {case.direction}")
    return {"actual_execution": "cpu-direct-dft", "reference": "check_xd._ntt_reference",
            "modulus": _NTT_MODULUS, "word_bits": 32 if case.precision == "uint32" else 64,
            "error": error}


def _run_zeta(np: Any, case: MatrixCase, seed: int) -> dict[str, Any]:
    source = _integer_source(np, case, seed)
    actual = _zeta_actual(np, source, case.shape, case.operator, case.inverse)
    op = OP_SUBSET_ZETA if case.operator == "subset-zeta" else OP_SUPERSET_ZETA
    expected = _zeta_reference(source, (case.batch, *case.shape), op, case.inverse)
    error = _integer_error(np, actual, expected, f"{case.operator} {case.direction}")
    return {"actual_execution": "cpu-radix2-zeta", "reference": "check_xd._zeta_reference",
            "word_bits": 32 if case.precision == "uint32" else 64, "error": error}


def _run_structured(np: Any, torch: Any, case: MatrixCase, seed: int) -> dict[str, Any]:
    source = _torch_random(torch, case, seed)
    matrices = _matrices_for_shape(case.shape)
    actual_np = _structured_actual(np, source, case.shape, matrices, case.inverse)
    actual = torch.from_numpy(actual_np)
    expected = _structured_reference(torch, source, case.shape, matrices, case.inverse)
    tolerance = (8e-5, 8e-5) if case.precision == "fp32" else (8e-11, 8e-11)
    error = _float_error(torch, actual, expected, atol=tolerance[0], rtol=tolerance[1],
                         label=f"structured-2x2 {case.direction}")
    return {"actual_execution": "cpu-numpy-stage-matrices",
            "reference": "check_xd._structured_reference", "error": error}


def _run_case(np: Any, torch: Any, case: MatrixCase, base_seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "id": case.name,
        "operator": case.operator,
        "precision": case.precision,
        "shape": list(case.shape),
        "batch": case.batch,
        "direction": case.direction,
        "actual_result_checked": False,
    }
    try:
        seed = _seed(case, base_seed)
        if case.operator == "fft":
            if torch is None:
                raise SkipCase("torch is required for CPU FFT")
            details = _run_fft(np, torch, case, seed)
        elif case.operator == "fwht":
            if torch is None:
                raise SkipCase("torch is required for CPU FWHT")
            details = _run_fwht(np, torch, case, seed)
        elif case.operator == "ntt":
            details = _run_ntt(np, case, seed)
        elif case.operator in ("subset-zeta", "superset-zeta"):
            details = _run_zeta(np, case, seed)
        elif case.operator == "structured-2x2":
            if torch is None:
                raise SkipCase("torch is required for CPU structured operator")
            details = _run_structured(np, torch, case, seed)
        else:  # pragma: no cover - matrix construction validates this first
            raise ValueError(f"unknown operator {case.operator!r}")
        record.update(details)
        record["actual_result_checked"] = True
        record["status"] = "passed"
    except SkipCase as exc:
        record.update({"status": "skipped", "reason": str(exc)})
    except (AssertionError, RuntimeError, ValueError, OverflowError) as exc:
        record.update({"status": "failed", "error": str(exc)})
    record["elapsed_s"] = time.perf_counter() - started
    return record


def _case_matrix(profile: str, operators: Sequence[str] | None,
                 precisions: Sequence[str] | None, shapes: Sequence[tuple[int, ...]] | None,
                 batches: Sequence[int] | None, selected: str | None) -> list[MatrixCase]:
    if profile == "full":
        profile_names = ("coverage", "scale")
    else:
        profile_names = (profile,)
    shape_values = tuple(shapes) if shapes is not None else tuple(
        shape for name in profile_names for shape in PROFILE_SHAPES[name]
    )
    batch_values = tuple(batches) if batches is not None else tuple(
        batch for name in profile_names for batch in PROFILE_BATCHES[name]
    )
    # Preserve order while de-duplicating the full profile's shared 32^3 cell.
    shape_values = tuple(dict.fromkeys(shape_values))
    batch_values = tuple(dict.fromkeys(batch_values))
    op_values = tuple(operators) if operators else OPERATORS
    unknown = sorted(set(op_values) - set(OPERATORS))
    if unknown:
        raise ValueError(f"unknown operator(s): {', '.join(unknown)}")
    result: list[MatrixCase] = []
    for operator in op_values:
        precision_values = tuple(precisions) if precisions else PRECISIONS[operator]
        unknown_precision = sorted(set(precision_values) - set(PRECISIONS[operator]))
        if unknown_precision:
            raise ValueError(
                f"precision(s) {unknown_precision!r} are not legal for {operator}")
        for precision, shape, batch, inverse in itertools.product(
                precision_values, shape_values, batch_values, (False, True)):
            case = MatrixCase(operator, precision, tuple(shape), int(batch), bool(inverse))
            if selected and selected not in case.name:
                continue
            result.append(case)
    return result


def _parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    if len(shape) not in (2, 3) or any(n <= 0 for n in shape):
        raise argparse.ArgumentTypeError("shape must be positive rank-2 or rank-3")
    return shape


def _extended_cpu_checks(np: Any, torch: Any, base_seed: int) -> list[dict[str, Any]]:
    """Run CPU equivalents and identify the existing CUDA-only entry points.

    These checks are intentionally separate from the six-operator matrix.  A
    report can therefore say exactly which layout/in-place/embedding/real
    semantics were exercised by CPU and which existing C-ABI helpers still
    require a GPU run.
    """
    if torch is None:
        return [{"name": name, "status": "not-run", "reason": "torch unavailable",
                 "gpu_entrypoint": helper}
                for name, helper in (
                    ("layout", "_strided_fwht_case/_strided_rank3_fwht_case"),
                    ("in-place", "_inplace_fwht_case/_inplace_fft_case"),
                    ("embedding", "_embedding_fwht_case/_embedding_rank3_fwht_case"),
                    ("real", "_real_fft_case"),
                )]

    checks: list[dict[str, Any]] = []

    def add(name: str, helper: str, fn: Any) -> None:
        started = time.perf_counter()
        try:
            details = fn()
            checks.append({"name": name, "status": "passed", "cpu_equivalent": True,
                           "gpu_entrypoint": helper, **details,
                           "elapsed_s": time.perf_counter() - started})
        except (AssertionError, RuntimeError, ValueError) as exc:
            checks.append({"name": name, "status": "failed", "cpu_equivalent": True,
                           "gpu_entrypoint": helper, "error": str(exc),
                           "elapsed_s": time.perf_counter() - started})

    add("layout", "_strided_fwht_case/_strided_rank3_fwht_case", lambda: _cpu_layout_check(torch))
    add("in-place", "_inplace_fwht_case/_inplace_fft_case", lambda: _cpu_inplace_check(torch))
    add("embedding", "_embedding_fwht_case/_embedding_rank3_fwht_case",
        lambda: _cpu_embedding_check(torch))
    add("real", "_real_fft_case", lambda: _cpu_real_check(np, torch, base_seed))
    return checks


def _cpu_layout_check(torch: Any) -> dict[str, Any]:
    base = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 4, 8)
    source = base.transpose(1, 2)
    actual = _fwht_actual(torch, source, (1, 2))
    expected = _fwht_reference(torch, source.contiguous(), (1, 2))
    error = _float_error(torch, actual, expected, atol=1e-4, rtol=1e-4, label="layout")
    return {"shape": [8, 4], "strides": list(source.stride()), "error": error}


def _cpu_inplace_check(torch: Any) -> dict[str, Any]:
    source = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)
    expected = _fwht_reference(torch, source.clone(), (1,))
    work = source.clone()
    work.copy_(_fwht_actual(torch, work, (1,)))
    error = _float_error(torch, work, expected, atol=1e-4, rtol=1e-4, label="in-place")
    return {"shape": [8], "error": error}


def _cpu_embedding_check(torch: Any) -> dict[str, Any]:
    logical = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    padded = torch.zeros((2, 4, 8), dtype=torch.float32)
    padded[:, :3, :5] = logical
    actual = _fwht_actual(torch, padded, (1, 2))
    expected = _fwht_reference(torch, padded, (1, 2))
    error = _float_error(torch, actual, expected, atol=1e-4, rtol=1e-4, label="embedding")
    return {"logical_shape": [3, 5], "physical_shape": [4, 8], "error": error}


def _cpu_real_check(np: Any, torch: Any, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = torch.randn((2, 4, 6), dtype=torch.float32, generator=generator, device="cpu")
    actual = torch.fft.rfftn(source, dim=(1, 2), norm="backward")
    expected = torch.from_numpy(np.fft.rfftn(source.numpy(), axes=(1, 2)))
    error = _float_error(torch, actual, expected, atol=4e-5, rtol=4e-5, label="real R2C")
    recovered = torch.fft.irfftn(actual, s=(4, 6), dim=(1, 2), norm="backward")
    inverse_error = _float_error(torch, recovered, source, atol=4e-5, rtol=4e-5,
                                 label="real C2R")
    return {"shape": [4, 6], "forward": error, "inverse": inverse_error,
            "gpu_entrypoint_note": "_real_fft_case remains CUDA-only"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("coverage", "scale", "full"),
                        default="coverage")
    parser.add_argument("--scale", action="store_true",
                        help="alias for --profile scale")
    parser.add_argument("--operator", action="append", dest="operators",
                        help="restrict to one operator; may be repeated")
    parser.add_argument("--precision", action="append", dest="precisions",
                        help="restrict precision; may be repeated")
    parser.add_argument("--shape", action="append", type=_parse_shape,
                        help="override shapes, e.g. 8x8 or 4x4x4")
    parser.add_argument("--batch", action="append", type=int,
                        help="override batches; may be repeated")
    parser.add_argument("--case", dest="selected",
                        help="run cells whose id contains this substring")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="bound execution after matrix construction")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--no-extended", action="store_true",
                        help="omit CPU layout/in-place/embedding/real checks")
    parser.add_argument("--dry-run", action="store_true",
                        help="emit the planned matrix without executing it")
    parser.add_argument("--output", type=pathlib.Path,
                        help="write the JSON report to this path")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv if argv is not None else sys.argv[1:])
    profile = "scale" if args.scale else args.profile
    try:
        cases = _case_matrix(profile, args.operators, args.precisions, args.shape,
                             args.batch, args.selected)
    except ValueError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=args.indent))
        return 2
    if args.max_cases is not None:
        if args.max_cases < 0:
            print(json.dumps({"status": "error", "error": "--max-cases must be nonnegative"},
                             indent=args.indent))
            return 2
        cases = cases[:args.max_cases]

    report: dict[str, Any] = {
        "script": str(pathlib.Path(__file__).resolve()),
        "profile": profile,
        "backend": "torch-cpu+numpy-cpu",
        "gpu_used": False,
        "correctness": {
            "actual_results_checked": False,
            "timing_is_not_used_for_status": True,
            "reference_module": "tests/check_xd.py",
            "reference_helpers": [
                "_fwht_reference", "_ntt_reference", "_zeta_reference",
                "_structured_reference",
            ],
            "c_abi_helpers_imported": ["Api", "Plan"],
        },
        "matrix": {
            "operators": list(OPERATORS),
            "shapes": [list(case.shape) for case in cases],
            "batches": sorted({case.batch for case in cases}),
            "directions": ["forward", "inverse"],
            "requested_cells": len(cases),
        },
        "cases": [],
        "extended": [],
    }
    if args.dry_run:
        report["cases"] = [{"id": case.name, "operator": case.operator,
                             "precision": case.precision, "shape": list(case.shape),
                             "batch": case.batch, "direction": case.direction,
                             "status": "planned", "actual_result_checked": False}
                            for case in cases]
        report["status"] = "planned"
        report["summary"] = {"planned": len(cases), "passed": 0, "failed": 0, "skipped": 0}
    else:
        np, torch, import_error = _import_runtime()
        if np is None:
            report["status"] = "skipped"
            report["summary"] = {"planned": len(cases), "passed": 0, "failed": 0,
                                  "skipped": len(cases)}
            report["reason"] = import_error
        else:
            report["cases"] = [_run_case(np, torch, case, args.seed) for case in cases]
            report["correctness"]["actual_results_checked"] = any(
                row.get("actual_result_checked", False) for row in report["cases"])
            if not args.no_extended:
                report["extended"] = _extended_cpu_checks(np, torch, args.seed)
            passed = sum(row.get("status") == "passed" for row in report["cases"])
            failed = sum(row.get("status") == "failed" for row in report["cases"])
            skipped = sum(row.get("status") == "skipped" for row in report["cases"])
            extended_failed = sum(row.get("status") == "failed" for row in report["extended"])
            report["status"] = "failed" if failed or extended_failed else "passed"
            report["summary"] = {
                "planned": len(cases), "passed": passed, "failed": failed + extended_failed,
                "skipped": skipped, "extended_passed": sum(
                    row.get("status") == "passed" for row in report["extended"]),
                "extended_failed": extended_failed,
            }
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
