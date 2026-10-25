#!/usr/bin/env python3
"""C-ABI correctness and benchmark checks for the rank-N cuButterfly API.

The test intentionally talks to ``libcuntt.so`` through ctypes.  The public
Python package is not part of the contract being checked here: torch owns the
device allocations while this file owns descriptor/plan construction and C
API execution.  Plan construction, workspace allocation, and calibration are
kept outside the timing region.

The script is useful while the xD API is still being integrated.  Cases whose
transform-type symbol is not present in an older shared object are reported as
``skipped`` rather than silently falling back to torch.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import glob
import json
import os
import pathlib
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


# Public C API enum values.  Keep these explicit: this file is a consumer
# contract and must not import private C++ headers or reproduce C++ layouts.
STATUS_SUCCESS = 0
STATUS_NOT_SUPPORTED = 3
SELECTION_USER_CACHE = 1

OP_FFT = 0
OP_NTT = 1
OP_FWHT = 2
OP_SUBSET_ZETA = 3
OP_SUPERSET_ZETA = 4
OP_STRUCTURED_2X2 = 5

DATA_FP32 = 2
DATA_FP64 = 3
DATA_COMPLEX_FP32 = 6
DATA_COMPLEX_FP64 = 7
DATA_UINT32 = 8
DATA_UINT64 = 9

COMPUTE_FP32 = 1
COMPUTE_FP64 = 2
COMPUTE_UINT32 = 3
COMPUTE_UINT64 = 4

FORWARD = 0
INVERSE = 1
OUT_OF_PLACE = 0
IN_PLACE = 1
LENGTH_STANDARD = 0
LENGTH_ZERO_EXTENDED = 1

EXEC_AUTO = 0
EXEC_XD = 1
EXEC_COMPOSITE = 2

FFT_C2C = 0
FFT_R2C = 1
FFT_C2R = 2

POLICY_DEFAULT = 0
POLICY_MEASURE = 1
POLICY_EXPLICIT = 2

MATRIX_FIELDS = ("m00", "m01", "m10", "m11")

_LOCAL_FAMILIES = (
    "axis-tiled-local",
    "axis-tiled-local-cap16",
    "axis-tiled-local-recurrence",
)


class CApiError(RuntimeError):
    """A non-success status returned by the public API."""

    def __init__(self, call: str, status: int, message: str):
        super().__init__(f"{call}: status={status} ({message})")
        self.call = call
        self.status = int(status)
        self.message = message


class UnsupportedCase(RuntimeError):
    pass


@dataclass
class CaseResult:
    name: str
    status: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, **self.details}


class Api:
    """ctypes declarations for the stable C ABI."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        try:
            self.lib = ct.CDLL(str(path), mode=getattr(ct, "RTLD_GLOBAL", 0))
        except OSError as exc:
            raise RuntimeError(f"cannot load {path}: {exc}") from exc

        self._bind("cubutterflyGetStatusString", ct.c_char_p, [ct.c_int])
        self._bind("cubutterflyCreate", ct.c_int, [ct.POINTER(ct.c_void_p)])
        self._bind("cubutterflyDestroy", ct.c_int, [ct.c_void_p])
        self._bind("cubutterflySetStream", ct.c_int, [ct.c_void_p, ct.c_void_p])
        self._bind("cubutterflySetCachePath", ct.c_int, [ct.c_void_p, ct.c_char_p])

        self._bind("cubutterflyCreateDescriptor", ct.c_int, [ct.POINTER(ct.c_void_p)])
        self._bind("cubutterflyDestroyDescriptor", ct.c_int, [ct.c_void_p])
        self._bind("cubutterflySetOperator", ct.c_int, [ct.c_void_p, ct.c_int])
        self._bind("cubutterflySetDataType", ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int])
        self._bind("cubutterflySetShape", ct.c_int,
                   [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_size_t), ct.c_size_t])
        self._bind("cubutterflySetExecutionMode", ct.c_int, [ct.c_void_p, ct.c_int])
        self.fft_type_fn = self._bind_optional(
            "cubutterflySetFftTransformType", ct.c_int, [ct.c_void_p, ct.c_int])
        self.fft_output_scale_fn = self._bind_optional(
            "cubutterflySetFftOutputScale", ct.c_int, [ct.c_void_p, ct.c_double])
        self._bind("cubutterflySetStrides", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_size_t), ct.c_size_t,
                    ct.POINTER(ct.c_size_t), ct.c_size_t])
        self._bind("cubutterflySetDirection", ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int])
        self._bind("cubutterflySetPlacement", ct.c_int, [ct.c_void_p, ct.c_int])
        self._bind("cubutterflySetLengthMode", ct.c_int, [ct.c_void_p, ct.c_int])
        self._bind("cubutterflySetAlgorithmPolicy", ct.c_int,
                   [ct.c_void_p, ct.c_int, ct.c_char_p])
        self._bind("cubutterflySetModulus", ct.c_int,
                   [ct.c_void_p, ct.c_uint64, ct.c_uint32])
        self._bind("cubutterflySetNttLayouts", ct.c_int, [ct.c_void_p, ct.c_int, ct.c_int])
        self._bind("cubutterflySetStageMatrices", ct.c_int,
                   [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_size_t])

        self._bind("cubutterflyCreatePlan", ct.c_int,
                   [ct.c_void_p, ct.c_void_p, ct.POINTER(ct.c_void_p)])
        self._bind("cubutterflyDestroyPlan", ct.c_int, [ct.c_void_p])
        self._bind("cubutterflyPlanGetWorkspaceSize", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanGetInputSize", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanGetOutputSize", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanGetPhysicalExtents", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_size_t), ct.c_uint32])
        self._bind("cubutterflyPlanGetSelectionSource", ct.c_int,
                   [ct.c_void_p, ct.POINTER(ct.c_int)])
        self._bind("cubutterflyPlanGetAlgorithmName", ct.c_int,
                   [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanGetSelectionReason", ct.c_int,
                   [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanGetMappingJson", ct.c_int,
                   [ct.c_void_p, ct.c_char_p, ct.POINTER(ct.c_size_t)])
        self._bind("cubutterflyPlanSetWorkspace", ct.c_int,
                   [ct.c_void_p, ct.c_void_p, ct.c_size_t])
        self._bind("cubutterflyExecute", ct.c_int,
                   [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p])

    def _bind(self, name: str, restype: Any, argtypes: list[Any]) -> Any:
        fn = getattr(self.lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
        setattr(self, name, fn)
        return fn

    def _bind_optional(self, name: str, restype: Any, argtypes: list[Any]) -> Any | None:
        try:
            fn = getattr(self.lib, name)
        except AttributeError:
            return None
        fn.restype = restype
        fn.argtypes = argtypes
        setattr(self, name, fn)
        return fn

    def status_text(self, status: int) -> str:
        raw = self.cubutterflyGetStatusString(int(status))
        return raw.decode("utf-8", "replace") if raw else "unknown"

    def check(self, call: str, status: int) -> None:
        if int(status) != STATUS_SUCCESS:
            raise CApiError(call, int(status), self.status_text(status))

    def create_handle(self, stream: int) -> ct.c_void_p:
        handle = ct.c_void_p()
        self.check("cubutterflyCreate", self.cubutterflyCreate(ct.byref(handle)))
        try:
            self.check("cubutterflySetStream", self.cubutterflySetStream(
                handle, ct.c_void_p(int(stream))))
        except Exception:
            self.cubutterflyDestroy(handle)
            raise
        return handle

    def create_descriptor(self) -> ct.c_void_p:
        descriptor = ct.c_void_p()
        self.check("cubutterflyCreateDescriptor",
                   self.cubutterflyCreateDescriptor(ct.byref(descriptor)))
        return descriptor

    def query_size(self, fn: Any, plan: ct.c_void_p, name: str) -> int:
        value = ct.c_size_t()
        self.check(name, fn(plan, ct.byref(value)))
        return int(value.value)

    def query_text(self, fn: Any, plan: ct.c_void_p, name: str) -> str:
        size = ct.c_size_t(0)
        self.check(name, fn(plan, None, ct.byref(size)))
        buffer = ct.create_string_buffer(int(size.value))
        self.check(name, fn(plan, buffer, ct.byref(size)))
        return buffer.value.decode("utf-8", "replace")


class Handle:
    def __init__(self, api: Api, stream: int):
        self.api = api
        self.value = api.create_handle(stream)

    def close(self) -> None:
        if self.value:
            self.api.cubutterflyDestroy(self.value)
            self.value = ct.c_void_p()

    def __enter__(self) -> "Handle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class Plan:
    def __init__(self, api: Api, handle: Handle, value: ct.c_void_p,
                 workspace: Any, metadata: dict[str, Any]):
        self.api = api
        self.handle = handle
        self.value = value
        self.workspace = workspace
        self.metadata = metadata

    def execute(self, input_tensor: Any, output_tensor: Any) -> None:
        self.api.check("cubutterflyExecute", self.api.cubutterflyExecute(
            self.handle.value, self.value,
            ct.c_void_p(int(input_tensor.data_ptr())),
            ct.c_void_p(int(output_tensor.data_ptr()))))

    def close(self) -> None:
        if self.value:
            self.api.cubutterflyDestroyPlan(self.value)
            self.value = ct.c_void_p()

    def __enter__(self) -> "Plan":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _next_power_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _dtype_for_storage(torch: Any, storage: int) -> Any:
    return {
        DATA_FP32: torch.float32,
        DATA_FP64: torch.float64,
        DATA_COMPLEX_FP32: torch.complex64,
        DATA_COMPLEX_FP64: torch.complex128,
        DATA_UINT32: torch.uint32,
        DATA_UINT64: torch.uint64,
    }[storage]


def _element_bytes(storage: int) -> int:
    return {
        DATA_FP32: 4,
        DATA_FP64: 8,
        DATA_COMPLEX_FP32: 8,
        DATA_COMPLEX_FP64: 16,
        DATA_UINT32: 4,
        DATA_UINT64: 8,
    }[storage]


def _pointer_tensor(tensor: Any) -> ct.c_void_p:
    return ct.c_void_p(int(tensor.data_ptr()))


def _ct_sizes(values: Sequence[int]) -> Any:
    return (ct.c_size_t * len(values))(*[int(value) for value in values])


def _matrix_array(matrices: Sequence[Sequence[float]]) -> Any:
    class Matrix2x2(ct.Structure):
        _fields_ = [(name, ct.c_double) for name in MATRIX_FIELDS]

    return (Matrix2x2 * len(matrices))(*[
        Matrix2x2(*[float(value) for value in matrix]) for matrix in matrices
    ])


def make_plan(
    api: Api,
    handle: Handle,
    torch: Any,
    *,
    op: int,
    storage: int,
    compute: int,
    shape: Sequence[int],
    batch: int,
    mode: int = EXEC_AUTO,
    direction: int = FORWARD,
    normalize_inverse: bool = True,
    placement: int = OUT_OF_PLACE,
    length_mode: int = LENGTH_STANDARD,
    input_strides: Sequence[int] | None = None,
    output_strides: Sequence[int] | None = None,
    input_batch_stride: int = 0,
    output_batch_stride: int = 0,
    modulus: int | None = None,
    word_bits: int = 64,
    matrices: Sequence[Sequence[Sequence[float]]] | None = None,
    mapping: str | None = None,
    fft_type: int = FFT_C2C,
    fft_output_scale: float = 1.0,
    input_orders: Sequence[int] | None = None,
    output_orders: Sequence[int] | None = None,
) -> Plan:
    import cubutterfly as public
    import operator

    if isinstance(direction, bool) or isinstance(placement, bool):
        raise TypeError("direction and placement must be integer enum values")
    direction = operator.index(direction)
    placement = operator.index(placement)
    if direction not in (FORWARD, INVERSE):
        raise ValueError("direction must be FORWARD or INVERSE")
    if placement not in (OUT_OF_PLACE, IN_PLACE):
        raise ValueError("placement must be OUT_OF_PLACE or IN_PLACE")
    try:
        library = public.Library(api.path)
    except AttributeError as exc:
        raise UnsupportedCase(f"library lacks the public descriptor API: {exc}") from exc
    transform = public.Transform(
        op=op, storage=storage, compute=compute, extents=list(shape), batch=batch,
        execution_mode=mode, inverse=direction == INVERSE, normalize_inverse=normalize_inverse,
        in_place=placement == IN_PLACE, length_mode=length_mode, fft_type=fft_type,
        fft_output_scale=fft_output_scale,
        input_strides=list(input_strides) if input_strides is not None else [],
        output_strides=list(output_strides) if output_strides is not None else [],
        input_batch_stride=input_batch_stride, output_batch_stride=output_batch_stride,
        modulus=modulus if modulus is not None else public.Transform().modulus,
        word_bits=word_bits, stage_matrices=list(matrices) if matrices is not None else [],
        input_orders=list(input_orders) if input_orders is not None else [],
        output_orders=list(output_orders) if output_orders is not None else [])
    options = public.PlanOptions(
        policy=public.AlgorithmPolicy.EXPLICIT if mapping is not None else public.AlgorithmPolicy.DEFAULT,
        mapping=mapping if mapping is not None else "")
    descriptor = None
    plan_value = ct.c_void_p()
    try:
        try:
            descriptor = public.Descriptor(library, transform, options)
            library._call("CreatePlan", handle.value, descriptor.native_handle, ct.byref(plan_value))
        except public.Error as exc:
            raise CApiError("public descriptor/plan", exc.status, str(exc)) from exc
        workspace_bytes = api.query_size(api.cubutterflyPlanGetWorkspaceSize,
                                         plan_value, "cubutterflyPlanGetWorkspaceSize")
        workspace = None
        if workspace_bytes:
            workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
            api.check("cubutterflyPlanSetWorkspace", api.cubutterflyPlanSetWorkspace(
                plan_value, _pointer_tensor(workspace), workspace_bytes))
        input_bytes = api.query_size(api.cubutterflyPlanGetInputSize,
                                     plan_value, "cubutterflyPlanGetInputSize")
        output_bytes = api.query_size(api.cubutterflyPlanGetOutputSize,
                                      plan_value, "cubutterflyPlanGetOutputSize")
        metadata: dict[str, Any] = {
            "workspace_bytes": workspace_bytes,
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "mapping_json": api.query_text(api.cubutterflyPlanGetMappingJson,
                                            plan_value, "cubutterflyPlanGetMappingJson"),
            "algorithm": api.query_text(api.cubutterflyPlanGetAlgorithmName,
                                         plan_value, "cubutterflyPlanGetAlgorithmName"),
            "selection_reason": api.query_text(api.cubutterflyPlanGetSelectionReason,
                                                plan_value, "cubutterflyPlanGetSelectionReason"),
        }
        source = ct.c_int()
        api.check("cubutterflyPlanGetSelectionSource",
                  api.cubutterflyPlanGetSelectionSource(plan_value, ct.byref(source)))
        metadata["selection_source"] = int(source.value)
        return Plan(api, handle, plan_value, workspace, metadata)
    except Exception:
        if plan_value:
            api.cubutterflyDestroyPlan(plan_value)
        raise
    finally:
        if descriptor is not None:
            descriptor.close()


def _fwht_reference(torch: Any, value: Any, axes: Sequence[int]) -> Any:
    """Independent radix-2 butterfly reference on a torch CPU tensor."""
    out = value.clone()
    for axis in axes:
        moved = out.movedim(axis, -1).contiguous()
        n = moved.shape[-1]
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
        out = moved.reshape(out.movedim(axis, -1).shape).movedim(-1, axis)
    return out


def _zeta_reference(array: Any, shape: Sequence[int], op: int,
                    inverse: bool) -> Any:
    import numpy as np

    work = np.array(array, dtype=np.uint64, copy=True).reshape(shape)
    for axis, n in enumerate(shape[1:], start=1):
        moved = np.moveaxis(work, axis, -1)
        # ``moveaxis`` is commonly non-contiguous.  Its reshape can therefore
        # be a copy, so reconstruct the moved view from the mutated flat copy.
        flat = np.array(moved, copy=True).reshape(-1, n)
        half = 1
        while half < n:
            for start in range(0, n, 2 * half):
                left = flat[:, start:start + half].copy()
                right = flat[:, start + half:start + 2 * half].copy()
                if op == OP_SUBSET_ZETA:
                    if inverse:
                        right = (right - left) & 0xFFFFFFFF
                    else:
                        right = (right + left) & 0xFFFFFFFF
                else:
                    if inverse:
                        left = (left - right) & 0xFFFFFFFF
                    else:
                        left = (left + right) & 0xFFFFFFFF
                flat[:, start:start + half] = left
                flat[:, start + half:start + 2 * half] = right
            half <<= 1
        work = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return work.reshape(-1).astype(np.uint32, copy=False)


def _primitive_root_for_length(length: int, modulus: int) -> int:
    for candidate in range(2, modulus):
        root = pow(candidate, (modulus - 1) // length, modulus)
        if pow(root, length, modulus) == 1 and pow(root, length // 2, modulus) != 1:
            return root
    raise ValueError(f"no primitive root for n={length}, p={modulus}")


def _ntt_line(values: Sequence[int], modulus: int, inverse: bool) -> list[int]:
    n = len(values)
    root = _primitive_root_for_length(n, modulus)
    if inverse:
        root = pow(root, modulus - 2, modulus)
    result = []
    for k in range(n):
        result.append(sum(int(values[j]) * pow(root, j * k, modulus)
                         for j in range(n)) % modulus)
    if inverse:
        scale = pow(n, modulus - 2, modulus)
        result = [(value * scale) % modulus for value in result]
    return result


def _ntt_reference(values: Sequence[int], shape: Sequence[int], batch: int,
                   modulus: int, inverse: bool) -> list[int]:
    import numpy as np

    work = np.array(values, dtype=np.uint64).reshape((batch, *shape))
    for axis, n in enumerate(shape, start=1):
        moved = np.moveaxis(work, axis, -1)
        flat = np.array(moved, copy=True).reshape(-1, n)
        for row in range(flat.shape[0]):
            flat[row, :] = np.asarray(_ntt_line(flat[row, :], modulus, inverse), dtype=np.uint64)
        work = np.moveaxis(flat.reshape(moved.shape), -1, axis)
    return [int(value) for value in work.reshape(-1)]


def _structured_reference(torch: Any, value: Any, shape: Sequence[int],
                          matrices: Sequence[Sequence[Sequence[float]]],
                          inverse: bool) -> Any:
    out = value.clone()
    for axis, n in enumerate(shape):
        tensor_axis = axis + 1  # axis zero is the batch dimension in ``value``.
        moved = out.movedim(tensor_axis, -1).contiguous()
        rows = moved.reshape(-1, n)
        stage_matrices = matrices[axis]
        half = 1
        stage = 0
        while half < n:
            m = stage_matrices[0] if len(stage_matrices) == 1 else stage_matrices[stage]
            m00, m01, m10, m11 = [float(x) for x in m]
            if inverse:
                det = m00 * m11 - m01 * m10
                m00, m01, m10, m11 = m11 / det, -m01 / det, -m10 / det, m00 / det
            left = rows[:, 0:n].clone()
            for start in range(0, n, 2 * half):
                a = rows[:, start:start + half].clone()
                b = rows[:, start + half:start + 2 * half].clone()
                rows[:, start:start + half] = m00 * a + m01 * b
                rows[:, start + half:start + 2 * half] = m10 * a + m11 * b
            half <<= 1
            stage += 1
        out = moved.reshape(out.movedim(tensor_axis, -1).shape).movedim(-1, tensor_axis)
    return out


def _max_error(actual: Any, expected: Any) -> tuple[float, float]:
    import torch

    a = actual.detach().cpu()
    e = expected.detach().cpu()
    if a.numel() == 0:
        return 0.0, 0.0
    diff = torch.abs(a - e)
    max_abs = float(diff.max().item())
    scale = max(1.0, float(torch.abs(e).max().item()))
    return max_abs, scale


def _assert_close(actual: Any, expected: Any, *, atol: float, rtol: float,
                  label: str = "value") -> dict[str, float]:
    max_abs, scale = _max_error(actual, expected)
    if max_abs > atol + rtol * scale:
        raise AssertionError(f"{label}: max_abs={max_abs:g}, tolerance={atol + rtol * scale:g}")
    return {"max_abs": max_abs, "tolerance": atol + rtol * scale}


def _tensor_from_flat(torch: Any, values: Sequence[int], dtype: Any, device: str = "cuda") -> Any:
    return torch.tensor(list(values), dtype=dtype, device=device)


def _make_random(torch: Any, shape: Sequence[int], dtype: Any, device: str = "cuda") -> Any:
    if dtype.is_complex:
        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(tuple(shape), dtype=real_dtype, device=device)
        imag = torch.randn(tuple(shape), dtype=real_dtype, device=device)
        return torch.complex(real, imag)
    return torch.randn(tuple(shape), dtype=dtype, device=device)


def _logical_from_strided(torch: Any, storage: Any, shape: Sequence[int], batch: int,
                          strides: Sequence[int], batch_stride: int) -> Any:
    result = torch.empty((batch, *shape), dtype=storage.dtype, device=storage.device)
    flat = storage.reshape(-1)
    for batch_index in range(batch):
        for index in __import__("itertools").product(*(range(n) for n in shape)):
            offset = batch_index * batch_stride + sum(i * s for i, s in zip(index, strides))
            result[(batch_index, *index)] = flat[offset]
    return result


def _run_plan(api: Api, handle: Handle, torch: Any, spec: dict[str, Any],
              input_tensor: Any, output_tensor: Any) -> tuple[Plan, dict[str, Any]]:
    plan = make_plan(api, handle, torch, **spec)
    plan.execute(input_tensor, output_tensor)
    torch.cuda.synchronize()
    return plan, plan.metadata


def _result(name: str, fn: Callable[[], dict[str, Any]]) -> CaseResult:
    started = time.perf_counter()
    try:
        details = fn()
        details.setdefault("elapsed_host_s", time.perf_counter() - started)
        return CaseResult(name, "passed", details)
    except UnsupportedCase as exc:
        return CaseResult(name, "skipped", {"reason": str(exc)})
    except CApiError as exc:
        if exc.status == STATUS_NOT_SUPPORTED:
            return CaseResult(name, "skipped", {
                "reason": str(exc), "status_code": exc.status,
            })
        return CaseResult(name, "failed", {"error": str(exc)})
    except (AssertionError, RuntimeError, ValueError, OverflowError) as exc:
        return CaseResult(name, "failed", {"error": str(exc)})


def _fft_case(api: Api, handle: Handle, torch: Any, shape: Sequence[int],
              batch: int, dtype: Any, storage: int, mode: int,
              mapping: str | None = None) -> dict[str, Any]:
    full_shape = (batch, *shape)
    source = _make_random(torch, full_shape, dtype)
    source_before = source.clone()
    expected_forward = torch.fft.fftn(source, dim=tuple(range(1, len(shape) + 1)), norm="backward")
    forward_spec = dict(op=OP_FFT, storage=storage, compute=COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32,
                        shape=shape, batch=batch, mode=mode, direction=FORWARD,
                        mapping=mapping)
    forward_out = torch.empty_like(source)
    forward, meta = _run_plan(api, handle, torch, forward_spec, source, forward_out)
    forward_error = _assert_close(forward_out, expected_forward,
                                  atol=3e-4 if dtype == torch.complex64 else 3e-11,
                                  rtol=3e-4 if dtype == torch.complex64 else 3e-11,
                                  label="FFT forward")
    if not torch.equal(source, source_before):
        raise AssertionError("out-of-place FFT modified its input")
    forward.close()
    inverse_spec = dict(forward_spec, direction=INVERSE)
    inverse_out = torch.empty_like(source)
    spectrum_before = forward_out.clone()
    inverse, inverse_meta = _run_plan(api, handle, torch, inverse_spec, forward_out, inverse_out)
    inverse_error = _assert_close(inverse_out, source,
                                  atol=4e-4 if dtype == torch.complex64 else 5e-11,
                                  rtol=4e-4 if dtype == torch.complex64 else 5e-11,
                                  label="FFT inverse")
    if not torch.equal(forward_out, spectrum_before):
        raise AssertionError("out-of-place inverse FFT modified its input")
    inverse.close()
    if mapping and (mapping.startswith("xd:axis-tiled-") or
                    mapping.startswith("xd:pair-tiled-")):
        for result in (meta, inverse_meta):
            resolved = json.loads(result["mapping_json"])
            if resolved.get("lowering") != mapping[3:]:
                raise AssertionError(
                    f"forced family {mapping!r} resolved as {resolved.get('lowering')!r}")
    if mapping and mapping.startswith("xd:pair-tiled-"):
        for result in (meta, inverse_meta):
            if json.loads(result["mapping_json"])["physical"].get("pair_groups", 0) < 1:
                raise AssertionError("forced pair candidate did not fuse any axes")
    if mapping and "cooperative" in mapping:
        for result in (meta, inverse_meta):
            axes = json.loads(result["mapping_json"])["physical"]["axes"]
            if not any(axis.get("core") == "register-cooperative-pair" for axis in axes):
                raise AssertionError("forced cooperative pair did not use its core")
    if mapping and mapping.startswith("xd:axis-tiled-register"):
        family = mapping[3:]
        expected_lanes = (8 if family.endswith("-l8") else
                          16 if family.endswith("-l16") else None)
        required_length = 256 if 256 in shape else None
        for result in (meta, inverse_meta):
            axes = json.loads(result["mapping_json"])["physical"]["axes"]
            if not any(axis.get("core") == "register-cooperative-axis" for axis in axes):
                raise AssertionError("forced register axis did not use its core")
            _assert_register_axis_mapping_contract(
                result, "forced register axis", expected_lanes, required_length)
    if mapping and mapping.startswith("xd:"):
        family = mapping[3:]
        expected_lanes = (8 if family.endswith("-l8") else
                          16 if family.endswith("-l16") else None)
        for result in (meta, inverse_meta):
            physical = json.loads(result["mapping_json"])["physical"]
            if expected_lanes is not None and physical.get("register_lanes") != expected_lanes:
                raise AssertionError(
                    f"forced family {family!r} used register_lanes="
                    f"{physical.get('register_lanes')!r}")
            if expected_lanes is not None and (
                    family.startswith("axis-tiled-register-") or
                    family.startswith("pair-tiled-cooperative-register-")):
                register_axes = [
                    axis for axis in physical.get("axes", [])
                    if axis.get("core") == "register-cooperative-axis"
                ]
                if not any(axis.get("cooperative_lanes") == expected_lanes
                           for axis in register_axes):
                    raise AssertionError(
                        f"forced family {family!r} has no register axis with "
                        f"cooperative_lanes={expected_lanes}")
            if family.endswith("-layout1") and physical.get("pair_layout") != 1:
                raise AssertionError(
                    f"forced family {family!r} did not use pair_layout=1")
            expected_cap = (8 if "-cap8" in family else
                            16 if "-cap16" in family else None)
            if expected_cap is not None:
                if physical.get("axis_ffts_per_block_cap") != expected_cap:
                    raise AssertionError(
                        f"forced family {family!r} used axis_ffts_per_block_cap="
                        f"{physical.get('axis_ffts_per_block_cap')!r}, expected {expected_cap}")
                if any(axis.get("ffts_per_block", 0) > expected_cap
                       for axis in physical.get("axes", [])):
                    raise AssertionError(
                        f"forced family {family!r} exceeded its ffts_per_block cap")
            if family.startswith("axis-tiled-cufftdx"):
                expected_ept = (4 if "-ept4" in family else
                                16 if "-ept16" in family else 8)
                if physical.get("axis_ept") != expected_ept:
                    raise AssertionError(
                        f"forced family {family!r} used axis_ept="
                        f"{physical.get('axis_ept')!r}, expected {expected_ept}")
            expected_exchange = (3 if family.endswith("-unrolled32") else
                                 1 if family.endswith("-direct") else
                                 2 if family.endswith("-unrolled") else None)
            if expected_exchange is not None:
                if physical.get("axis_exchange_policy") != expected_exchange:
                    raise AssertionError(
                        f"forced family {family!r} used axis_exchange_policy="
                        f"{physical.get('axis_exchange_policy')!r}, expected {expected_exchange}")
                expected_io = ("direct-strided-register" if expected_exchange == 1
                               else "unrolled-staged" if expected_exchange == 2
                               else "unrolled-staged-u32")
                io_values = [axis.get("io_exchange") for axis in physical.get("axes", [])]
                if expected_io not in io_values:
                    raise AssertionError(
                        f"forced family {family!r} has no {expected_io!r} axis")
                if any(value not in (expected_io, "contiguous-register")
                       for value in io_values):
                    raise AssertionError(
                        f"forced family {family!r} has an unexpected io_exchange value")
    if mapping and mapping[3:] in _LOCAL_FAMILIES:
        family = mapping[3:]
        expected_cap = 16 if family.endswith("-cap16") else 0
        required_length = (256 if 256 in shape else
                           128 if 128 in shape else None)
        expected_twiddle_kind = ("recurrence" if family.endswith("-recurrence")
                                 else "constant")
        for result in (meta, inverse_meta):
            _assert_local_axis_mapping_contract(
                result, f"forced local family {family!r}", expected_cap,
                required_length, expected_twiddle_kind=expected_twiddle_kind)
    return {"shape": list(shape), "batch": batch, "dtype": str(dtype), "mode": mode,
            "forward": forward_error, "inverse": inverse_error,
            "algorithm": meta["algorithm"], "inverse_algorithm": inverse_meta["algorithm"]}


def _assert_real_register_mapping_contract(metadata: dict[str, Any],
                                           label: str) -> None:
    """A compact real register candidate must have a real register axis."""
    mapping = json.loads(metadata["mapping_json"])
    physical = mapping.get("physical", {})
    if physical.get("axis_core") == "cooperative-register":
        raise AssertionError(f"{label} advertised a cooperative register core")
    if any(axis.get("core") == "register-cooperative-axis"
           for axis in physical.get("axes", [])):
        raise AssertionError(f"{label} used a register axis with suffix one")
    if "[axis-tiled-register]" in metadata["algorithm"]:
        raise AssertionError(f"{label} selected the register algorithm without a register axis")

    marker = "; candidates="
    reason = metadata["selection_reason"]
    if marker not in reason:
        raise AssertionError(f"{label} selection reason has no candidate list")
    try:
        candidates = json.loads(reason.rsplit(marker, 1)[1])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssertionError(f"{label} selection candidate list is not JSON") from exc
    if any(candidate.get("axis_core") == "cooperative-register"
           for candidate in candidates):
        raise AssertionError(f"{label} measured a fake cooperative register candidate")


def _assert_register_axis_mapping_contract(metadata: dict[str, Any], label: str,
                                           expected_lanes: int | None = None,
                                           required_length: int | None = None) -> None:
    """Validate register-axis metadata without imposing pair-core rules."""
    mapping = json.loads(metadata["mapping_json"])
    physical = mapping.get("physical", {})
    axes = [axis for axis in physical.get("axes", [])
            if axis.get("core") == "register-cooperative-axis"]
    if not axes:
        raise AssertionError(f"{label} has no register-cooperative-axis metadata")
    observed_required_length = required_length is None
    for axis in axes:
        length = axis.get("length")
        expected_twiddle = ("constant-roots-256" if length == 256 else
                            "constant-roots-128" if length in (64, 128) else None)
        if expected_twiddle is None or axis.get("twiddle_policy") != expected_twiddle:
            raise AssertionError(
                f"{label} has invalid twiddle_policy={axis.get('twiddle_policy')!r} "
                f"for register length={length!r}")
        if expected_lanes is not None and axis.get("cooperative_lanes") != expected_lanes:
            raise AssertionError(
                f"{label} has cooperative_lanes={axis.get('cooperative_lanes')!r}, "
                f"expected {expected_lanes}")
        observed_required_length = observed_required_length or length == required_length
    if not observed_required_length:
        raise AssertionError(
            f"{label} has no register axis of required length={required_length}")


def _assert_local_axis_mapping_contract(metadata: dict[str, Any], label: str,
                                        expected_cap: int = 0,
                                        required_length: int | None = None,
                                        expected_top_core: str | None = "thread-shared",
                                        expected_twiddle_kind: str = "constant") -> None:
    """Validate the per-axis metadata emitted by the shared local core."""
    if expected_twiddle_kind not in ("constant", "recurrence"):
        raise ValueError(f"{label} has an invalid expected twiddle kind")
    mapping = json.loads(metadata["mapping_json"])
    physical = mapping.get("physical", {})
    if expected_top_core is not None and physical.get("axis_core") != expected_top_core:
        raise AssertionError(
            f"{label} used axis_core={physical.get('axis_core')!r}, "
            f"expected {expected_top_core}")
    if not physical.get("fused_axes") or not physical.get("packed_axes"):
        raise AssertionError(f"{label} did not retain fused packed axes")
    if physical.get("paired_axes") or physical.get("pair_groups", 0):
        raise AssertionError(f"{label} unexpectedly selected a pair core")
    if expected_top_core is not None and physical.get("axis_ffts_per_block_cap", 0) != expected_cap:
        raise AssertionError(
            f"{label} used axis_ffts_per_block_cap="
            f"{physical.get('axis_ffts_per_block_cap', 0)!r}, expected {expected_cap}")

    resources = physical.get("axis_resources", [])
    if not isinstance(resources, list) or not any(
            resource.get("core") == "thread-shared" for resource in resources):
        raise AssertionError(f"{label} has no thread-shared axis resource")
    local_axes = [axis for axis in physical.get("axes", [])
                  if axis.get("core") == "thread-shared-axis"]
    if not local_axes:
        raise AssertionError(f"{label} has no thread-shared-axis metadata")
    if required_length is not None and not any(
            axis.get("length") == required_length for axis in local_axes):
        raise AssertionError(
            f"{label} has no local axis of required length={required_length}")

    resource_by_axis = {
        index: resource for index, resource in enumerate(resources)
    }
    for resource in resources:
        if (resource.get("core") != "thread-shared" and
                "local_twiddle_policy" in resource):
            raise AssertionError(
                f"{label} serialized local_twiddle_policy on a nonlocal resource")

    fp64 = bool(physical.get("fp64"))
    for axis in local_axes:
        length = axis.get("length")
        if length not in (64, 128, 256):
            raise AssertionError(f"{label} has invalid local length={length!r}")
        expected_lines = 8 if expected_cap != 16 else 16
        if fp64 and length == 256:
            expected_lines = 8
        if axis.get("ffts_per_block") != expected_lines:
            raise AssertionError(
                f"{label} local length={length} used ffts_per_block="
                f"{axis.get('ffts_per_block')!r}, expected {expected_lines}")
        expected_ept = 8 if length == 64 else length // 16
        if axis.get("elements_per_thread") != expected_ept:
            raise AssertionError(
                f"{label} local length={length} has elements_per_thread="
                f"{axis.get('elements_per_thread')!r}")
        expected_factors = [8, 8] if length == 64 else [16, length // 16]
        if axis.get("local_factors") != expected_factors:
            raise AssertionError(
                f"{label} local length={length} has local_factors="
                f"{axis.get('local_factors')!r}")
        if axis.get("io_exchange") != "soa-xor-local":
            raise AssertionError(
                f"{label} local axis has io_exchange={axis.get('io_exchange')!r}")
        resource = resource_by_axis.get(int(axis.get("axis", -1)))
        if resource is None or resource.get("core") != "thread-shared":
            raise AssertionError(
                f"{label} local logical axis={axis.get('axis')!r} has no thread-shared resource")
        expected_local_policy = 1 if expected_twiddle_kind == "recurrence" else 0
        actual_local_policy = resource.get("local_twiddle_policy", 0)
        if actual_local_policy != expected_local_policy:
            raise AssertionError(
                f"{label} local length={length} has local_twiddle_policy="
                f"{actual_local_policy!r}, expected {expected_local_policy}")
        if expected_local_policy == 0 and "local_twiddle_policy" in resource:
            raise AssertionError(
                f"{label} policy-zero resource serialized local_twiddle_policy")
        expected_twiddle_root = "recurrence-root" if expected_twiddle_kind == "recurrence" else "constant-roots"
        expected_twiddle = f"{expected_twiddle_root}-{length}"
        if axis.get("twiddle_policy") != expected_twiddle:
            raise AssertionError(
                f"{label} local length={length} has twiddle_policy="
                f"{axis.get('twiddle_policy')!r}")


def _assert_local_mixed_mapping_contract(metadata: dict[str, Any], label: str) -> None:
    """Require one local axis and one independent Block axis in a mixed replay."""
    mapping = json.loads(metadata["mapping_json"])
    physical = mapping.get("physical", {})
    resources = physical.get("axis_resources", [])
    if len(resources) != 2:
        raise AssertionError(f"{label} does not serialize two logical resources")
    resource_by_axis = {axis: resource.get("core")
                        for axis, resource in enumerate(resources)}
    if resource_by_axis != {0: "thread-shared", 1: "cufftdx-block"}:
        raise AssertionError(
            f"{label} has unexpected axis resource cores={resource_by_axis!r}")
    axis_by_logical = {int(axis.get("axis")): axis
                       for axis in physical.get("axes", [])}
    if axis_by_logical.get(0, {}).get("core") != "thread-shared-axis":
        raise AssertionError(f"{label} logical axis 0 is not thread-shared")
    if axis_by_logical.get(1, {}).get("core") != "cufftdx-axis-tiled":
        raise AssertionError(f"{label} logical axis 1 is not Block-tiled")
    _assert_local_axis_mapping_contract(
        metadata, label, expected_cap=0, required_length=128,
        expected_top_core=None)


def _register_axis_lane4_256_unavailable_case(api: Api, handle: Handle,
                                              torch: Any) -> dict[str, Any]:
    """The explicit lane-4 family must reject the unsupported 256-point axis."""
    source = _make_random(torch, (1, 256, 32), torch.complex64)
    spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                shape=(256, 32), batch=1, mode=EXEC_XD, direction=FORWARD,
                mapping="xd:axis-tiled-register")
    try:
        plan = make_plan(api, handle, torch, **spec)
    except CApiError as exc:
        if exc.status != STATUS_NOT_SUPPORTED:
            raise
        return {"shape": [256, 32], "register_lanes": 4,
                "configuration_available": False}
    metadata = plan.metadata
    plan.close()
    physical = json.loads(metadata["mapping_json"]).get("physical", {})
    register_axes = [axis for axis in physical.get("axes", [])
                     if axis.get("core") == "register-cooperative-axis"]
    if any(axis.get("length") == 256 and axis.get("cooperative_lanes") == 4
           for axis in register_axes):
        raise AssertionError("lane-4 register axis unexpectedly accepted length 256")
    raise AssertionError("lane-4 256 register-axis mapping was accepted")


def _real_fft_case(api: Api, handle: Handle, torch: Any, shape: Sequence[int],
                   batch: int, real_dtype: Any, storage: int,
                   output_scale: float = 1.0,
                   mapping_validator: Callable[[dict[str, Any], str], None] | None = None,
                   mapping: str | None = None
                   ) -> dict[str, Any]:
    if api.fft_type_fn is None:
        raise UnsupportedCase("library has no cubutterflySetFftTransformType")
    complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
    input_shape = (batch, *shape)
    compact_shape = (batch, *shape[:-1], shape[-1] // 2 + 1)
    source = torch.randn(input_shape, dtype=real_dtype, device="cuda")
    expected = torch.fft.rfftn(source, dim=tuple(range(1, len(shape) + 1)), norm="backward") * output_scale
    spec = dict(op=OP_FFT, storage=storage, compute=COMPUTE_FP64 if real_dtype == torch.float64 else COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_AUTO if len(shape) == 1 else EXEC_XD, direction=FORWARD,
                fft_type=FFT_R2C, fft_output_scale=output_scale, mapping=mapping)
    output = torch.empty(compact_shape, dtype=complex_dtype, device="cuda")
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    if mapping_validator is not None:
        mapping_validator(meta, "R2C")
    error = _assert_close(output, expected,
                          atol=5e-4 if real_dtype == torch.float32 else 5e-11,
                          rtol=5e-4 if real_dtype == torch.float32 else 5e-11,
                          label="R2C")
    plan.close()

    half_input = output.clone()
    c2r_spec = dict(spec, direction=INVERSE, fft_type=FFT_C2R)
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, c2r_spec, half_input, recovered)
    if mapping_validator is not None:
        mapping_validator(inverse_meta, "C2R")
    inverse_expected = torch.fft.irfftn(half_input, s=shape,
                                         dim=tuple(range(1, len(shape) + 1)), norm="backward") * output_scale
    inverse_error = _assert_close(recovered, inverse_expected,
                                  atol=8e-4 if real_dtype == torch.float32 else 8e-11,
                                  rtol=8e-4 if real_dtype == torch.float32 else 8e-11,
                                  label="C2R")
    if not torch.equal(half_input, output):
        raise AssertionError("C2R modified its half-spectrum input")
    inverse.close()
    return {"shape": list(shape), "batch": batch, "dtype": str(real_dtype),
            "r2c": error, "c2r": inverse_error, "algorithm": meta["algorithm"],
            "inverse_algorithm": inverse_meta["algorithm"]}


def _fwht_case(api: Api, handle: Handle, torch: Any, shape: Sequence[int],
               batch: int, mode: int = EXEC_XD,
               mapping: str | None = None) -> dict[str, Any]:
    source = torch.randn((batch, *shape), dtype=torch.float32, device="cuda")
    expected = _fwht_reference(torch, source.cpu(), range(1, len(shape) + 1)).to("cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=mode, direction=FORWARD,
                mapping=mapping)
    output = torch.empty_like(source)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    error = _assert_close(output, expected, atol=4e-4, rtol=4e-4, label="FWHT")
    plan.close()
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, dict(spec, direction=INVERSE),
                                      output, recovered)
    inverse_error = _assert_close(recovered, source, atol=5e-4, rtol=5e-4,
                                  label="FWHT inverse")
    inverse.close()
    return {"shape": list(shape), "batch": batch, "mode": mode, "forward": error,
            "inverse": inverse_error, "algorithm": meta["algorithm"],
            "inverse_algorithm": inverse_meta["algorithm"]}


def _zeta_case(api: Api, handle: Handle, torch: Any, op: int,
               mapping: str | None = None) -> dict[str, Any]:
    shape = (4, 8)
    batch = 2
    import numpy as np

    source_cpu = np.asarray([(index * 17 + 5) & 0xFFFFFFFF
                             for index in range(batch * _prod(shape))], dtype=np.uint32)
    expected_cpu = _zeta_reference(source_cpu, (batch, *shape), op, False)
    # PyTorch 1.13 does not expose NumPy uint32 tensors through
    # from_numpy.  The C ABI consumes the same four-byte bit pattern, so use
    # signed int32 storage for this compatibility path and compare modulo
    # 2**32 below.
    source = torch.from_numpy(source_cpu.astype(np.int32)).to("cuda")
    output = torch.empty_like(source)
    spec = dict(op=op, storage=DATA_UINT32, compute=COMPUTE_UINT32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD,
                mapping=mapping)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    actual = output.cpu().numpy().astype(np.uint32)
    if not np.array_equal(actual, expected_cpu):
        raise AssertionError(f"zeta output mismatch at {int(np.flatnonzero(actual != expected_cpu)[0])}")
    plan.close()
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, dict(spec, direction=INVERSE),
                                      output, recovered)
    if not np.array_equal(recovered.cpu().numpy().astype(np.uint32), source_cpu):
        raise AssertionError("zeta inverse roundtrip mismatch")
    inverse.close()
    return {"shape": list(shape), "batch": batch,
            "operator": "subset-zeta" if op == OP_SUBSET_ZETA else "superset-zeta",
            "algorithm": meta["algorithm"], "inverse_algorithm": inverse_meta["algorithm"]}


def _ntt_case(api: Api, handle: Handle, torch: Any, storage: int,
              compute: int, mapping: str | None = None,
              normalize_inverse: bool = True) -> dict[str, Any]:
    import numpy as np

    shape = (4, 8)
    batch = 2
    modulus = 97
    source_cpu = np.asarray([(index * index + 7 * index + 3) % modulus
                             for index in range(batch * _prod(shape))], dtype=np.uint64)
    expected_cpu = np.asarray(_ntt_reference(source_cpu, shape, batch, modulus, False), dtype=np.uint64)
    # Keep the pointer width expected by the C ABI while supporting the
    # PyTorch 1.13 runtime used by the acceptance image, which has no uint32
    # or uint64 dtype exposed to from_numpy.
    host_dtype = np.int32 if storage == DATA_UINT32 else np.int64
    source = torch.from_numpy(source_cpu.astype(host_dtype)).to("cuda")
    output = torch.empty_like(source)
    spec = dict(op=OP_NTT, storage=storage, compute=compute, shape=shape,
                batch=batch, mode=EXEC_XD, direction=FORWARD, modulus=modulus,
                word_bits=32 if storage == DATA_UINT32 else 64, mapping=mapping,
                normalize_inverse=normalize_inverse)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    if not np.array_equal(output.cpu().numpy().astype(np.uint64), expected_cpu):
        raise AssertionError("NTT forward does not match direct host DFT")
    plan.close()

    recovered = torch.empty_like(source)
    inverse_spec = dict(spec, direction=INVERSE)
    inverse, inverse_meta = _run_plan(api, handle, torch, inverse_spec, output, recovered)
    if not np.array_equal(recovered.cpu().numpy(), source.cpu().numpy()):
        raise AssertionError("NTT inverse roundtrip mismatch")
    inverse.close()
    return {"shape": list(shape), "batch": batch, "storage": storage,
            "modulus": modulus, "normalize_inverse": normalize_inverse,
            "algorithm": meta["algorithm"],
            "inverse_algorithm": inverse_meta["algorithm"]}


def _structured_case(api: Api, handle: Handle, torch: Any,
                     mapping: str | None = None) -> dict[str, Any]:
    shape = (4, 8)
    batch = 2
    matrices = [
        ((1.10, 0.20, -0.10, 0.90), (0.80, -0.30, 0.25, 1.20)),
        ((0.95, 0.15, 0.05, 1.05), (1.20, -0.20, 0.10, 0.85), (0.75, 0.40, -0.15, 1.10)),
    ]
    source = torch.randn((batch, *shape), dtype=torch.float32, device="cuda")
    expected = _structured_reference(torch, source.cpu(), shape, matrices, False).to("cuda")
    spec = dict(op=OP_STRUCTURED_2X2, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD,
                matrices=matrices, mapping=mapping)
    output = torch.empty_like(source)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    error = _assert_close(output, expected, atol=6e-4, rtol=6e-4,
                          label="structured forward")
    plan.close()
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, dict(spec, direction=INVERSE),
                                      output, recovered)
    inverse_error = _assert_close(recovered, source, atol=8e-4, rtol=8e-4,
                                  label="structured roundtrip")
    inverse.close()
    return {"shape": list(shape), "batch": batch, "forward": error,
            "roundtrip": inverse_error, "algorithm": meta["algorithm"],
            "inverse_algorithm": inverse_meta["algorithm"]}


def _strided_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (4, 8)
    batch = 2
    strides = (10, 1)
    batch_stride = 43
    source_storage = torch.randn((batch * batch_stride + 8,), dtype=torch.float32, device="cuda")
    output_storage = torch.full_like(source_storage, float("nan"))
    logical_source = _logical_from_strided(torch, source_storage, shape, batch, strides, batch_stride)
    expected = _fwht_reference(torch, logical_source.cpu(), range(1, len(shape) + 1)).to("cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_COMPOSITE,
                direction=FORWARD, input_strides=strides, output_strides=strides,
                input_batch_stride=batch_stride, output_batch_stride=batch_stride)
    plan, meta = _run_plan(api, handle, torch, spec, source_storage, output_storage)
    actual = _logical_from_strided(torch, output_storage, shape, batch, strides, batch_stride)
    error = _assert_close(actual, expected, atol=5e-4, rtol=5e-4,
                          label="strided FWHT")
    plan.close()
    return {"shape": list(shape), "batch": batch, "strides": list(strides), **error,
            "algorithm": meta["algorithm"]}


def _strided_rank3_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    """Exercise positive, non-contiguous rank-three xD strides."""
    shape = (4, 2, 8)
    batch = 2
    strides = (24, 10, 1)
    batch_stride = 101
    source_storage = torch.randn((batch * batch_stride + 8,), dtype=torch.float32, device="cuda")
    output_storage = torch.full_like(source_storage, float("nan"))
    logical_source = _logical_from_strided(torch, source_storage, shape, batch, strides, batch_stride)
    expected = _fwht_reference(torch, logical_source.cpu(), range(1, len(shape) + 1)).to("cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD,
                input_strides=strides, output_strides=strides,
                input_batch_stride=batch_stride, output_batch_stride=batch_stride)
    plan, meta = _run_plan(api, handle, torch, spec, source_storage, output_storage)
    actual = _logical_from_strided(torch, output_storage, shape, batch, strides, batch_stride)
    error = _assert_close(actual, expected, atol=5e-4, rtol=5e-4,
                          label="rank-three strided FWHT")
    plan.close()
    return {"shape": list(shape), "batch": batch, "strides": list(strides), **error,
            "algorithm": meta["algorithm"]}


def _inplace_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (8,)
    batch = 2
    source = torch.randn((batch, *shape), dtype=torch.float32, device="cuda")
    original = source.clone()
    expected = _fwht_reference(torch, original.cpu(), (1,)).to("cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_COMPOSITE,
                direction=FORWARD, placement=IN_PLACE)
    plan, meta = _run_plan(api, handle, torch, spec, source, source)
    error = _assert_close(source, expected, atol=5e-4, rtol=5e-4,
                          label="in-place FWHT")
    plan.close()
    return {"shape": list(shape), "batch": batch, **error,
            "algorithm": meta["algorithm"]}


def _inplace_rank2_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (4, 8)
    batch = 2
    source = torch.randn((batch, *shape), dtype=torch.float32, device="cuda")
    original = source.clone()
    expected = _fwht_reference(torch, original.cpu(), range(1, len(shape) + 1)).to("cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD,
                direction=FORWARD, placement=IN_PLACE)
    plan, meta = _run_plan(api, handle, torch, spec, source, source)
    error = _assert_close(source, expected, atol=5e-4, rtol=5e-4,
                          label="rank-two in-place FWHT")
    plan.close()
    return {"shape": list(shape), "batch": batch, **error,
            "algorithm": meta["algorithm"]}


def _inplace_fft_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (4, 8)
    batch = 2
    source = _make_random(torch, (batch, *shape), torch.complex64)
    expected_forward = torch.fft.fftn(source, dim=(1, 2), norm="backward")
    forward_spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                        shape=shape, batch=batch, mode=EXEC_XD,
                        direction=FORWARD, placement=IN_PLACE)
    forward, meta = _run_plan(api, handle, torch, forward_spec, source, source)
    forward_error = _assert_close(source, expected_forward, atol=4e-4, rtol=4e-4,
                                  label="in-place FFT forward")
    forward.close()

    expected_inverse = torch.fft.ifftn(expected_forward, dim=(1, 2), norm="backward")
    inverse, inverse_meta = _run_plan(api, handle, torch,
                                      dict(forward_spec, direction=INVERSE), source, source)
    inverse_error = _assert_close(source, expected_inverse, atol=5e-4, rtol=5e-4,
                                  label="in-place FFT inverse")
    inverse.close()
    return {"shape": list(shape), "batch": batch, "forward": forward_error,
            "inverse": inverse_error, "algorithm": meta["algorithm"],
            "inverse_algorithm": inverse_meta["algorithm"]}


def _embedding_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    logical_shape = (3, 5)
    physical_shape = tuple(_next_power_two(value) for value in logical_shape)
    batch = 2
    source = torch.randn((batch, *logical_shape), dtype=torch.float32, device="cuda")
    padded = torch.zeros((batch, *physical_shape), dtype=torch.float32)
    padded[(slice(None),) + tuple(slice(0, n) for n in logical_shape)] = source.cpu()
    expected = _fwht_reference(torch, padded, range(1, len(physical_shape) + 1)).to("cuda")
    output = torch.empty((batch, *physical_shape), dtype=torch.float32, device="cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=logical_shape, batch=batch, mode=EXEC_COMPOSITE,
                direction=FORWARD, length_mode=LENGTH_ZERO_EXTENDED)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    error = _assert_close(output, expected, atol=6e-4, rtol=6e-4,
                          label="zero-extended FWHT")
    plan.close()
    return {"logical_shape": list(logical_shape), "physical_shape": list(physical_shape),
            "batch": batch, **error, "algorithm": meta["algorithm"]}


def _embedding_rank3_fwht_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    logical_shape = (3, 3, 5)
    physical_shape = tuple(_next_power_two(value) for value in logical_shape)
    batch = 2
    source = torch.randn((batch, *logical_shape), dtype=torch.float32, device="cuda")
    padded = torch.zeros((batch, *physical_shape), dtype=torch.float32)
    logical_slices = (slice(None),) + tuple(slice(0, n) for n in logical_shape)
    padded[logical_slices] = source.cpu()
    axes = range(1, len(physical_shape) + 1)
    expected = _fwht_reference(torch, padded, axes).to("cuda")
    output = torch.empty((batch, *physical_shape), dtype=torch.float32, device="cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=logical_shape, batch=batch, mode=EXEC_XD,
                direction=FORWARD, length_mode=LENGTH_ZERO_EXTENDED)
    plan, meta = _run_plan(api, handle, torch, spec, source, output)
    error = _assert_close(output, expected, atol=7e-4, rtol=7e-4,
                          label="rank-three zero-extended FWHT")
    plan.close()

    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch,
                                      dict(spec, direction=INVERSE), output, recovered)
    # The inverse operates on the padded power-of-two domain.  Restrict the
    # independently transformed result back to the logical embedding.
    inverse_padded = (_fwht_reference(torch, expected.cpu(), axes) /
                      float(_prod(physical_shape))).to("cuda")
    inverse_error = _assert_close(recovered, inverse_padded[logical_slices],
                                  atol=8e-4, rtol=8e-4,
                                  label="rank-three embedding inverse")
    inverse.close()
    return {"logical_shape": list(logical_shape), "physical_shape": list(physical_shape),
            "batch": batch, "forward": error, "inverse": inverse_error,
            "algorithm": meta["algorithm"], "inverse_algorithm": inverse_meta["algorithm"]}


def _mapping_replay_case(api: Api, handle: Handle, torch: Any,
                         fft_type: int | None = None, fused: bool = False,
                         lowering: str | None = None,
                         dtype: Any | None = None,
                         batch: int = 2,
                         shape_override: Sequence[int] | None = None) -> dict[str, Any]:
    if shape_override is not None:
        shape = tuple(shape_override)
    elif lowering:
        shape = ((128, 256) if lowering == "axis-tiled-local-recurrence" else
                 (128, 3) if lowering in _LOCAL_FAMILIES else
                 (16, 32, 32) if lowering.startswith("pair-tiled-") else
                 (8, 16, 32))
    else:
        shape = (4, 2, 8)
    axis_cap = bool(lowering and lowering.startswith("axis-tiled-cufftdx") and "-cap" in lowering)
    axis_exchange = bool(lowering and lowering.startswith("axis-tiled-cufftdx") and
                         ("-direct" in lowering or "-unrolled" in lowering))
    axis_ept4 = bool(lowering and lowering.startswith("axis-tiled-cufftdx") and
                     "-ept4" in lowering)
    if shape_override is None and lowering and (axis_cap or axis_exchange or
                     axis_ept4 or
                     "cooperative" in lowering or
                     lowering.startswith("axis-tiled-register")):
        shape = (64, 64) if axis_cap or axis_exchange else (64, 64, 64)
    source = torch.randn((batch, *shape), dtype=torch.float32, device="cuda")
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD)
    first_out = torch.empty_like(source)
    if fft_type is not None:
        spec.update(op=OP_FFT, fft_type=fft_type)
        spectrum = torch.fft.rfftn(source, dim=tuple(range(1, len(shape) + 1)))
        if fft_type == FFT_C2C:
            complex_dtype = dtype if dtype is not None else torch.complex64
            source = _make_random(torch, (batch, *shape), complex_dtype)
            first_out = torch.empty_like(source)
            spec.update(
                storage=(DATA_COMPLEX_FP64 if complex_dtype == torch.complex128
                         else DATA_COMPLEX_FP32),
                compute=(COMPUTE_FP64 if complex_dtype == torch.complex128
                         else COMPUTE_FP32),
                mapping=("xd:" + lowering if lowering else
                         "xd:axis-tiled-cufftdx" if fused else "xd:packed-axis-cufft"))
        elif fft_type == FFT_R2C:
            first_out = torch.empty_like(spectrum)
            spec.update(mode=EXEC_COMPOSITE)  # Real composition keeps packed-axis candidates.
        else:
            first_out = torch.empty_like(source)
            source = spectrum
            spec.update(mode=EXEC_COMPOSITE, direction=INVERSE)
    source_before = source.clone()
    first, metadata = _run_plan(api, handle, torch, spec, source, first_out)
    if not torch.equal(source, source_before):
        raise AssertionError("first mapping replay execution modified its input")
    mapping_text = metadata["mapping_json"]
    first_mapping = json.loads(mapping_text)
    first.close()
    replay_out = torch.empty_like(first_out)
    replay, replay_meta = _run_plan(api, handle, torch, dict(spec, mapping=mapping_text),
                                   source, replay_out)
    replay_mapping = json.loads(replay_meta["mapping_json"])
    if lowering and lowering.startswith("axis-tiled-cufftdx"):
        physical = first_mapping.get("physical", {})
        expected_ept = (4 if "-ept4" in lowering else
                        16 if "-ept16" in lowering else 8)
        if physical.get("axis_ept") != expected_ept:
            raise AssertionError(
                f"replay family {lowering!r} used axis_ept={physical.get('axis_ept')!r}, "
                f"expected {expected_ept}")
        expected_exchange = (3 if lowering.endswith("-unrolled32") else
                             1 if lowering.endswith("-direct") else
                             2 if lowering.endswith("-unrolled") else None)
        if expected_exchange is not None:
            if physical.get("axis_exchange_policy") != expected_exchange:
                raise AssertionError(
                    f"replay family {lowering!r} used axis_exchange_policy="
                    f"{physical.get('axis_exchange_policy')!r}, expected {expected_exchange}")
            expected_io = ("direct-strided-register" if expected_exchange == 1 else
                           "unrolled-staged" if expected_exchange == 2 else
                           "unrolled-staged-u32")
            io_values = [axis.get("io_exchange")
                         for axis in physical.get("axes", [])]
            if expected_io not in io_values:
                raise AssertionError(
                    f"replay family {lowering!r} has no {expected_io!r} axis")
    if lowering in _LOCAL_FAMILIES:
        expected_cap = 16 if lowering.endswith("-cap16") else 0
        required_length = (256 if 256 in shape else
                           128 if 128 in shape else None)
        expected_twiddle_kind = ("recurrence" if lowering.endswith("-recurrence")
                                 else "constant")
        _assert_local_axis_mapping_contract(
            {"mapping_json": mapping_text},
            f"local family {lowering!r} replay", expected_cap, required_length,
            expected_twiddle_kind=expected_twiddle_kind)
        _assert_local_axis_mapping_contract(
            replay_meta,
            f"local family {lowering!r} replay result", expected_cap, required_length,
            expected_twiddle_kind=expected_twiddle_kind)
    if lowering and lowering.startswith("axis-tiled-register"):
        expected_lanes = (8 if lowering.endswith("-l8") else
                          16 if lowering.endswith("-l16") else None)
        required_length = 256 if 256 in shape else None
        _assert_register_axis_mapping_contract(
            {"mapping_json": mapping_text}, "register-axis replay",
            expected_lanes, required_length)
        _assert_register_axis_mapping_contract(
            replay_meta, "register-axis replay result",
            expected_lanes, required_length)
    if lowering == "pair-tiled-cufftdx-ept16":
        physical = first_mapping.get("physical", {})
        unpaired = [axis for axis in physical.get("axes", [])
                    if axis.get("core") == "cufftdx-axis-tiled"]
        if len(unpaired) != 1 or unpaired[0].get("elements_per_thread") != 16:
            raise AssertionError("pair ept16 rank-3 replay did not use EPT16 on its unpaired axis")
    if json.dumps(first_mapping, sort_keys=True, separators=(",", ":")) != json.dumps(
            replay_mapping, sort_keys=True, separators=(",", ":")):
        raise AssertionError("mapping replay changed the resolved mapping")
    if not torch.equal(source, source_before):
        raise AssertionError("replayed mapping execution modified its input")
    tolerance = (5e-11, 5e-11) if source.dtype == torch.complex128 else (5e-4, 5e-4)
    error = _assert_close(replay_out, first_out, atol=tolerance[0], rtol=tolerance[1],
                          label="mapping replay output")
    replay.close()
    return {"mapping_exact": True, "input_unchanged": True, **error,
            "algorithm": replay_meta["algorithm"]}


_MIXED_EPT4_FAMILY = "axis-tiled-cufftdx-ept4-cap8-unrolled"
_MIXED_EPT16_FAMILY = "axis-tiled-cufftdx-ept16-cap8-unrolled"


def _mixed_resource_from_physical(physical: dict[str, Any]) -> dict[str, Any]:
    """Copy the uniform plan's resource fields into one logical-axis entry."""
    return {
        "core": physical["axis_core"],
        "ept": int(physical["axis_ept"]),
        "register_lanes": int(physical.get("register_lanes", 4)),
        "block_cap": int(physical.get("axis_ffts_per_block_cap", 0)),
        "exchange": int(physical.get("axis_exchange_policy", 0)),
    }


def _uniform_mixed_seed(api: Api, handle: Handle, torch: Any,
                        shape: Sequence[int], batch: int, dtype: Any,
                        direction: int, family: str,
                        required_axes: Sequence[int] | None = None) -> dict[str, Any]:
    storage = DATA_COMPLEX_FP64 if dtype == torch.complex128 else DATA_COMPLEX_FP32
    compute = COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32
    spec = dict(op=OP_FFT, storage=storage, compute=compute, shape=shape,
                batch=batch, mode=EXEC_XD, direction=direction,
                mapping="xd:" + family)
    plan = make_plan(api, handle, torch, **spec)
    try:
        mapping = json.loads(plan.metadata["mapping_json"])
    finally:
        plan.close()
    if mapping.get("lowering") != family:
        raise AssertionError(
            f"uniform mixed seed resolved as {mapping.get('lowering')!r}, expected {family!r}")
    physical = mapping.get("physical", {})
    axes = physical.get("axes", [])
    if "axis_resources" in physical:
        raise AssertionError("uniform xD mapping serialized an empty axis_resources vector")
    if (not physical.get("fused_axes") or physical.get("paired_axes") or
            len(axes) != len(shape)):
        raise UnsupportedCase("uniform mixed seed did not produce one fused block axis per logical axis")
    by_axis = {int(axis["axis"]): axis for axis in axes}
    required = set(range(len(shape))) if required_axes is None else set(required_axes)
    if not required.issubset(by_axis):
        raise AssertionError("uniform mixed seed omitted a requested logical axis")
    if any(by_axis[axis].get("core") != "cufftdx-axis-tiled" for axis in required):
        raise UnsupportedCase("uniform mixed seed has a non-block or fallback required axis")
    if required_axes is not None:
        for axis, record in by_axis.items():
            if axis not in required and record.get("packed"):
                raise AssertionError("uniform mixed seed changed the contiguous fallback layout")
            if (axis not in required and
                    (record.get("core") != "cufft-rank1-strided" or
                     record.get("stride") != 1)):
                raise UnsupportedCase(
                    "uniform mixed seed did not keep the 17-point axis on contiguous cuFFT")
    return mapping


def _uniform_local_seed(api: Api, handle: Handle, torch: Any,
                        shape: Sequence[int], batch: int, dtype: Any,
                        direction: int, family: str = "axis-tiled-local") -> dict[str, Any]:
    storage = DATA_COMPLEX_FP64 if dtype == torch.complex128 else DATA_COMPLEX_FP32
    compute = COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32
    spec = dict(op=OP_FFT, storage=storage, compute=compute, shape=shape,
                batch=batch, mode=EXEC_XD, direction=direction,
                mapping="xd:" + family)
    plan = make_plan(api, handle, torch, **spec)
    try:
        mapping = json.loads(plan.metadata["mapping_json"])
    finally:
        plan.close()
    if mapping.get("lowering") != family:
        raise AssertionError(
            f"uniform local seed resolved as {mapping.get('lowering')!r}, expected {family!r}")
    _assert_local_axis_mapping_contract(
        {"mapping_json": json.dumps(mapping)},
        f"uniform local seed {family!r}",
        16 if family.endswith("-cap16") else 0,
        128 if 128 in shape else 256 if 256 in shape else None,
        expected_twiddle_kind=("recurrence" if family.endswith("-recurrence")
                               else "constant"))
    physical = mapping.get("physical", {})
    if len(physical.get("axis_resources", [])) != len(shape):
        raise AssertionError("uniform local seed did not serialize every axis resource")
    return mapping


def _make_local_mixed_c2c_mapping(api: Api, handle: Handle, torch: Any,
                                  shape: Sequence[int], batch: int,
                                  dtype: Any, direction: int) -> dict[str, Any]:
    """Combine a local 128-axis donor with a Block donor for the other axis."""
    local = _uniform_local_seed(api, handle, torch, shape, batch, dtype, direction)
    block = _uniform_mixed_seed(api, handle, torch, shape, batch, dtype, direction,
                                "axis-tiled-cufftdx")
    mixed = json.loads(json.dumps(local))
    mixed["lowering"] = "axis-tiled-mixed"
    physical = mixed["physical"]
    local_physical = local["physical"]
    block_physical = block["physical"]
    local_axes = {int(axis["axis"]): axis for axis in local_physical["axes"]}
    block_axes = {int(axis["axis"]): axis for axis in block_physical["axes"]}
    if 0 not in local_axes or 1 not in block_axes:
        raise AssertionError("local mixed donors omitted one of the logical axes")
    physical["axes"] = [json.loads(json.dumps(
        local_axes[0] if int(axis["axis"]) == 0 else block_axes[int(axis["axis"])]))
                         for axis in physical["axes"]]
    local_resources = local_physical["axis_resources"]
    physical["axis_resources"] = [
        json.loads(json.dumps(local_resources[0])),
        _mixed_resource_from_physical(block_physical),
    ]
    return mixed


def _mapping_local_mixed_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (128, 16)
    batch = 2
    dtype = torch.complex64
    source = _make_random(torch, (batch, *shape), dtype)
    source_before = source.clone()
    forward_mapping = _make_local_mixed_c2c_mapping(
        api, handle, torch, shape, batch, dtype, FORWARD)
    spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD,
                mapping=json.dumps(forward_mapping))
    expected = torch.fft.fftn(source, dim=(1, 2), norm="backward")
    forward_out = torch.empty_like(source)
    forward, forward_meta = _run_plan(api, handle, torch, spec, source, forward_out)
    try:
        if json.loads(forward_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("local mixed forward replay changed its mapping")
        _assert_local_mixed_mapping_contract(forward_meta, "local mixed forward")
        forward_error = _assert_close(
            forward_out, expected, atol=5e-4, rtol=5e-4,
            label="local mixed forward")
        if not torch.equal(source, source_before):
            raise AssertionError("local mixed forward modified its input")
    finally:
        forward.close()

    replay_out = torch.empty_like(source)
    replay, replay_meta = _run_plan(
        api, handle, torch, dict(spec, mapping=json.dumps(forward_mapping)),
        source, replay_out)
    try:
        if json.loads(replay_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("local mixed replay changed its mapping")
        _assert_local_mixed_mapping_contract(replay_meta, "local mixed replay")
        replay_error = _assert_close(
            replay_out, forward_out, atol=5e-4, rtol=5e-4,
            label="local mixed replay")
        if not torch.equal(source, source_before):
            raise AssertionError("local mixed replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _make_local_mixed_c2c_mapping(
        api, handle, torch, shape, batch, dtype, INVERSE)
    recovered = torch.empty_like(source)
    spectrum_before = forward_out.clone()
    inverse, inverse_meta = _run_plan(
        api, handle, torch,
        dict(spec, direction=INVERSE, mapping=json.dumps(inverse_mapping)),
        forward_out, recovered)
    try:
        if json.loads(inverse_meta["mapping_json"]) != inverse_mapping:
            raise AssertionError("local mixed inverse replay changed its mapping")
        _assert_local_mixed_mapping_contract(inverse_meta, "local mixed inverse")
        inverse_error = _assert_close(
            recovered, source, atol=5e-4, rtol=5e-4,
            label="local mixed inverse")
        if not torch.equal(forward_out, spectrum_before):
            raise AssertionError("local mixed inverse modified its input")
    finally:
        inverse.close()
    return {"shape": list(shape), "batch": batch, "dtype": str(dtype),
            "mapping_exact": True, "input_unchanged": True,
            "forward": forward_error, "replay": replay_error,
            "inverse": inverse_error}


def _make_compact_local_mapping(api: Api, handle: Handle, torch: Any,
                                real_dtype: Any, direction: int,
                                fft_type: int,
                                family: str = "axis-tiled-local") -> dict[str, Any]:
    real_shape = (128, 8, 33)
    payload_shape = (128, 8, 17)
    storage = DATA_FP64 if real_dtype == torch.float64 else DATA_FP32
    compute = COMPUTE_FP64 if real_dtype == torch.float64 else COMPUTE_FP32
    template_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                         shape=real_shape, batch=2, mode=EXEC_COMPOSITE,
                         direction=direction, fft_type=fft_type)
    template_plan = make_plan(api, handle, torch, **template_spec)
    try:
        template = json.loads(template_plan.metadata["mapping_json"])
    finally:
        template_plan.close()
    physical = template.get("physical", {})
    local_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    local = _uniform_local_seed(
        api, handle, torch, payload_shape, 2, local_dtype, direction, family)
    local_physical = local["physical"]
    local_axes = {int(axis["axis"]): axis for axis in local_physical["axes"]}
    template_order = [int(axis) for axis in physical.get("axis_order", [])]
    if sorted(template_order) != [0, 1] or any(axis not in local_axes for axis in template_order):
        raise AssertionError("compact local template has an unexpected complex axis order")
    physical["axes"] = [json.loads(json.dumps(local_axes[axis]))
                         for axis in template_order]
    physical["fused_axes"] = True
    physical["packed_axes"] = True
    physical["paired_axes"] = False
    physical["pair_groups"] = 0
    physical["axis_core"] = "thread-shared"
    physical["axis_ept"] = 8
    physical["output_permutation"] = []
    physical.pop("axis_ffts_per_block_cap", None)
    physical.pop("axis_exchange_policy", None)
    # axis_resources is indexed by logical axis, whereas `axes` follows the
    # execution order retained by the compact real template.
    physical["axis_resources"] = [json.loads(json.dumps(
        local_physical["axis_resources"][axis])) for axis in (0, 1)]
    template["lowering"] = (
        f"xd-compact-c2r[{family}]" if fft_type == FFT_C2R
        else f"xd-compact-r2c[{family}]")
    return template


def _mapping_local_real_case(api: Api, handle: Handle, torch: Any,
                             real_dtype: Any | None = None,
                             family: str = "axis-tiled-local") -> dict[str, Any]:
    real_shape = (128, 8, 33)
    batch = 2
    real_dtype = torch.float32 if real_dtype is None else real_dtype
    is_fp64 = real_dtype == torch.float64
    storage = DATA_FP64 if is_fp64 else DATA_FP32
    compute = COMPUTE_FP64 if is_fp64 else COMPUTE_FP32
    complex_dtype = torch.complex128 if is_fp64 else torch.complex64
    twiddle_kind = "recurrence" if family.endswith("-recurrence") else "constant"
    source = torch.randn((batch, *real_shape), dtype=real_dtype, device="cuda")
    source_before = source.clone()
    expected = torch.fft.rfftn(source, dim=(1, 2, 3), norm="backward")
    forward_mapping = _make_compact_local_mapping(
        api, handle, torch, real_dtype, FORWARD, FFT_R2C, family)
    forward_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                        shape=real_shape, batch=batch, mode=EXEC_COMPOSITE,
                        direction=FORWARD, fft_type=FFT_R2C,
                        mapping=json.dumps(forward_mapping))
    forward_out = torch.empty((batch, 128, 8, 17), dtype=complex_dtype,
                              device="cuda")
    forward, forward_meta = _run_plan(
        api, handle, torch, forward_spec, source, forward_out)
    try:
        if json.loads(forward_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("compact local R2C replay changed its mapping")
        _assert_local_axis_mapping_contract(
            forward_meta, f"compact local R2C {family}", expected_cap=0,
            required_length=128, expected_twiddle_kind=twiddle_kind)
        forward_error = _assert_close(
            forward_out, expected,
            atol=5e-11 if is_fp64 else 5e-4,
            rtol=5e-11 if is_fp64 else 5e-4,
            label=f"compact local R2C {family}")
        if not torch.equal(source, source_before):
            raise AssertionError("compact local R2C modified its input")
    finally:
        forward.close()

    replay_out = torch.empty_like(forward_out)
    replay, replay_meta = _run_plan(
        api, handle, torch,
        dict(forward_spec, mapping=json.dumps(forward_mapping)),
        source, replay_out)
    try:
        if json.loads(replay_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("compact local R2C replay changed its mapping")
        _assert_local_axis_mapping_contract(
            replay_meta, f"compact local R2C {family} replay", expected_cap=0,
            required_length=128, expected_twiddle_kind=twiddle_kind)
        replay_error = _assert_close(
            replay_out, forward_out,
            atol=5e-11 if is_fp64 else 5e-4,
            rtol=5e-11 if is_fp64 else 5e-4,
            label=f"compact local R2C {family} replay")
        if not torch.equal(source, source_before):
            raise AssertionError("compact local R2C replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _make_compact_local_mapping(
        api, handle, torch, real_dtype, INVERSE, FFT_C2R, family)
    recovered = torch.empty_like(source)
    spectrum_before = forward_out.clone()
    inverse_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                        shape=real_shape, batch=batch, mode=EXEC_COMPOSITE,
                        direction=INVERSE, fft_type=FFT_C2R,
                        mapping=json.dumps(inverse_mapping))
    inverse, inverse_meta = _run_plan(
        api, handle, torch, inverse_spec, forward_out, recovered)
    try:
        if json.loads(inverse_meta["mapping_json"]) != inverse_mapping:
            raise AssertionError("compact local C2R replay changed its mapping")
        _assert_local_axis_mapping_contract(
            inverse_meta, f"compact local C2R {family}", expected_cap=0,
            required_length=128, expected_twiddle_kind=twiddle_kind)
        expected_inverse = torch.fft.irfftn(
            forward_out, s=real_shape, dim=(1, 2, 3), norm="backward")
        inverse_error = _assert_close(
            recovered, expected_inverse,
            atol=8e-11 if is_fp64 else 8e-4,
            rtol=8e-11 if is_fp64 else 8e-4,
            label=f"compact local C2R {family}")
        if not torch.equal(forward_out, spectrum_before):
            raise AssertionError("compact local C2R modified its input")
    finally:
        inverse.close()
    return {"shape": list(real_shape), "batch": batch, "dtype": str(real_dtype),
            "mapping_exact": True, "input_unchanged": True,
            "r2c": forward_error, "replay": replay_error,
            "c2r": inverse_error}


def _mapping_local_recurrence_c2c_case(api: Api, handle: Handle, torch: Any,
                                       dtype: Any) -> dict[str, Any]:
    """Exercise both recurrence roots, both directions, and exact replay."""
    shape = (128, 256)
    batch = 3
    storage = DATA_COMPLEX_FP64 if dtype == torch.complex128 else DATA_COMPLEX_FP32
    compute = COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32
    tolerance = 5e-11 if dtype == torch.complex128 else 5e-4
    source = _make_random(torch, (batch, *shape), dtype)
    source_before = source.clone()
    expected = torch.fft.fftn(source, dim=(1, 2), norm="backward")
    mapping = _uniform_local_seed(
        api, handle, torch, shape, batch, dtype, FORWARD,
        "axis-tiled-local-recurrence")
    spec = dict(op=OP_FFT, storage=storage, compute=compute, shape=shape,
                batch=batch, mode=EXEC_XD, direction=FORWARD,
                mapping=json.dumps(mapping))
    output = torch.empty_like(source)
    forward, forward_meta = _run_plan(api, handle, torch, spec, source, output)
    try:
        if json.loads(forward_meta["mapping_json"]) != mapping:
            raise AssertionError("local recurrence C2C forward replay changed its mapping")
        _assert_local_axis_mapping_contract(
            forward_meta, "local recurrence C2C forward", expected_cap=0,
            required_length=256, expected_twiddle_kind="recurrence")
        forward_error = _assert_close(
            output, expected, atol=tolerance, rtol=tolerance,
            label="local recurrence C2C forward")
        if not torch.equal(source, source_before):
            raise AssertionError("local recurrence C2C forward modified its input")
    finally:
        forward.close()

    replay_output = torch.empty_like(source)
    replay, replay_meta = _run_plan(
        api, handle, torch, dict(spec, mapping=json.dumps(mapping)),
        source, replay_output)
    try:
        if json.loads(replay_meta["mapping_json"]) != mapping:
            raise AssertionError("local recurrence C2C exact replay changed its mapping")
        _assert_local_axis_mapping_contract(
            replay_meta, "local recurrence C2C exact replay", expected_cap=0,
            required_length=256, expected_twiddle_kind="recurrence")
        replay_error = _assert_close(
            replay_output, output, atol=tolerance, rtol=tolerance,
            label="local recurrence C2C exact replay")
        if not torch.equal(source, source_before):
            raise AssertionError("local recurrence C2C replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _uniform_local_seed(
        api, handle, torch, shape, batch, dtype, INVERSE,
        "axis-tiled-local-recurrence")
    recovered = torch.empty_like(source)
    spectrum_before = output.clone()
    inverse, inverse_meta = _run_plan(
        api, handle, torch,
        dict(spec, direction=INVERSE, mapping=json.dumps(inverse_mapping)),
        output, recovered)
    try:
        if json.loads(inverse_meta["mapping_json"]) != inverse_mapping:
            raise AssertionError("local recurrence C2C inverse changed its mapping")
        _assert_local_axis_mapping_contract(
            inverse_meta, "local recurrence C2C inverse", expected_cap=0,
            required_length=256, expected_twiddle_kind="recurrence")
        inverse_error = _assert_close(
            recovered, source, atol=tolerance, rtol=tolerance,
            label="local recurrence C2C inverse")
        if not torch.equal(output, spectrum_before):
            raise AssertionError("local recurrence C2C inverse modified its input")
    finally:
        inverse.close()
    return {"shape": list(shape), "batch": batch, "dtype": str(dtype),
            "mapping_exact": True, "input_unchanged": True,
            "forward": forward_error, "replay": replay_error,
            "inverse": inverse_error, "twiddle_policy": "recurrence-root-64/128/256"}


def _local_recurrence_batch_mapping(api: Api, handle: Handle, torch: Any,
                                    dtype: Any, direction: int) -> dict[str, Any]:
    """Build a real batch-tail mapping from complete local donor plans."""
    shape = (128, 256)
    resolved: dict[int, dict[str, Any]] = {}
    for batch in (2, 1, 5):
        resolved[batch] = _uniform_local_seed(
            api, handle, torch, shape, batch, dtype, direction,
            "axis-tiled-local-recurrence")
    parent = json.loads(json.dumps(resolved[5]))
    body = json.loads(json.dumps(resolved[2]["physical"]))
    physical = json.loads(json.dumps(body))
    physical["tile_body"] = body
    physical["tile_tail"] = json.loads(json.dumps(resolved[1]["physical"]))
    physical["batch"] = 5
    physical["batch_tile"] = 2
    physical["batch_chunks"] = 3
    physical["axes_scope"] = "per-batch-tile"
    parent["physical"] = physical
    return parent


def _mapping_local_recurrence_batch_case(api: Api, handle: Handle, torch: Any,
                                         dtype: Any) -> dict[str, Any]:
    """Check recurrence resources through a nonempty batch tail and replay."""
    shape = (128, 256)
    batch = 5
    storage = DATA_COMPLEX_FP64 if dtype == torch.complex128 else DATA_COMPLEX_FP32
    compute = COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32
    tolerance = 5e-11 if dtype == torch.complex128 else 5e-4
    source = _make_random(torch, (batch, *shape), dtype)
    source_before = source.clone()
    expected = torch.fft.fftn(source, dim=(1, 2), norm="backward")
    mapping = _local_recurrence_batch_mapping(api, handle, torch, dtype, FORWARD)
    spec = dict(op=OP_FFT, storage=storage, compute=compute, shape=shape,
                batch=batch, mode=EXEC_XD, direction=FORWARD,
                mapping=json.dumps(mapping))
    output = torch.empty_like(source)
    forward, forward_meta = _run_plan(api, handle, torch, spec, source, output)
    try:
        resolved = json.loads(forward_meta["mapping_json"])
        if resolved != mapping:
            raise AssertionError("local recurrence batch-tail replay changed its mapping")
        physical = resolved.get("physical", {})
        if physical.get("batch_tile") != 2 or physical.get("batch_chunks") != 3:
            raise AssertionError("local recurrence batch-tail mapping lost its tile contract")
        _assert_local_axis_mapping_contract(
            forward_meta, "local recurrence batch-tail forward", expected_cap=0,
            required_length=256, expected_twiddle_kind="recurrence")
        forward_error = _assert_close(
            output, expected, atol=tolerance, rtol=tolerance,
            label="local recurrence batch-tail forward")
        if not torch.equal(source, source_before):
            raise AssertionError("local recurrence batch-tail forward modified its input")
    finally:
        forward.close()

    replay_output = torch.empty_like(source)
    replay, replay_meta = _run_plan(
        api, handle, torch, dict(spec, mapping=json.dumps(mapping)),
        source, replay_output)
    try:
        if json.loads(replay_meta["mapping_json"]) != mapping:
            raise AssertionError("local recurrence batch-tail exact replay changed its mapping")
        replay_error = _assert_close(
            replay_output, output, atol=tolerance, rtol=tolerance,
            label="local recurrence batch-tail exact replay")
        if not torch.equal(source, source_before):
            raise AssertionError("local recurrence batch-tail replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _local_recurrence_batch_mapping(api, handle, torch, dtype, INVERSE)
    recovered = torch.empty_like(source)
    spectrum_before = output.clone()
    inverse, inverse_meta = _run_plan(
        api, handle, torch,
        dict(spec, direction=INVERSE, mapping=json.dumps(inverse_mapping)),
        output, recovered)
    try:
        if json.loads(inverse_meta["mapping_json"]) != inverse_mapping:
            raise AssertionError("local recurrence batch-tail inverse changed its mapping")
        inverse_error = _assert_close(
            recovered, source, atol=tolerance, rtol=tolerance,
            label="local recurrence batch-tail inverse")
        if not torch.equal(output, spectrum_before):
            raise AssertionError("local recurrence batch-tail inverse modified its input")
    finally:
        inverse.close()
    return {"shape": list(shape), "batch": batch, "batch_tile": 2,
            "dtype": str(dtype), "mapping_exact": True, "input_unchanged": True,
            "forward": forward_error, "replay": replay_error,
            "inverse": inverse_error}


def _assert_forced_compact_local_recurrence(metadata: dict[str, Any],
                                            label: str) -> None:
    if "axis-tiled-local-recurrence" not in metadata.get("algorithm", ""):
        raise AssertionError(
            f"{label} did not expose the forced compact recurrence family")
    _assert_local_axis_mapping_contract(
        metadata, label, expected_cap=0, required_length=256,
        expected_twiddle_kind="recurrence")


def _mapping_local_recurrence_invalid_case(api: Api, handle: Handle, torch: Any,
                                           only: str | None = None) -> dict[str, Any]:
    """Reject out-of-range policy values and policy-1 nonlocal resources."""
    shape = (128, 256)
    batch = 2
    mapping = _uniform_local_seed(
        api, handle, torch, shape, batch, torch.complex64, FORWARD,
        "axis-tiled-local-recurrence")
    base_spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                     shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD)
    invalid_policy_values = {
        "policy2": 2,
        "policy_overflow": 4294967297,
        "policy_fraction": 1.5,
        "policy_negative": -1,
        "policy_string": "1",
    }

    nonlocal_mapping = _uniform_mixed_seed(
        api, handle, torch, (128, 16), batch, torch.complex64, FORWARD,
        "axis-tiled-cufftdx")
    nonlocal_physical = nonlocal_mapping["physical"]
    nonlocal_physical["axis_resources"] = [
        _mixed_resource_from_physical(nonlocal_physical)
        for _ in range(2)
    ]
    nonlocal_physical["axis_resources"][0]["local_twiddle_policy"] = 1
    variants: dict[str, dict[str, Any]] = {
        "policy1_nonlocal": nonlocal_mapping,
    }
    for label, value in invalid_policy_values.items():
        forged = json.loads(json.dumps(mapping))
        forged["physical"]["axis_resources"][0]["local_twiddle_policy"] = value
        variants[label] = forged
    if only is not None:
        if only == "policy2":
            variants = {label: forged for label, forged in variants.items()
                        if label.startswith("policy") and label != "policy1_nonlocal"}
        elif only in variants:
            variants = {only: variants[only]}
        else:
            raise ValueError(f"unknown local twiddle rejection case {only!r}")
    rejected: dict[str, dict[str, Any]] = {}
    for label, forged in variants.items():
        try:
            plan = make_plan(api, handle, torch,
                             **dict(base_spec, shape=forged["physical"]["extents"],
                                    batch=forged["physical"]["batch"],
                                    mapping=json.dumps(forged)))
        except CApiError as exc:
            rejected[label] = {"status_code": exc.status, "status_text": exc.message}
            continue
        plan.close()
        raise AssertionError(f"invalid local twiddle mapping was accepted: {label}")
    return {"family": "axis-tiled-local-recurrence", "rejections": rejected,
            "mapping_exact": True}


def _make_mixed_c2c_mapping(api: Api, handle: Handle, torch: Any,
                            shape: Sequence[int], batch: int, dtype: Any,
                            direction: int) -> dict[str, Any]:
    """Build a complete v15 replay by combining two real fused seed plans."""
    ept4 = _uniform_mixed_seed(api, handle, torch, shape, batch, dtype,
                               direction, _MIXED_EPT4_FAMILY)
    ept16 = _uniform_mixed_seed(api, handle, torch, shape, batch, dtype,
                                direction, _MIXED_EPT16_FAMILY)
    mixed = json.loads(json.dumps(ept4))
    mixed["lowering"] = "axis-tiled-mixed"
    physical = mixed["physical"]
    donor_axes = {
        int(axis["axis"]): axis for axis in ept16["physical"]["axes"]
    }
    main_axes = {
        int(axis["axis"]): axis for axis in ept4["physical"]["axes"]
    }
    # Use the EPT16 seed for odd logical axes and keep the EPT4 seed for even
    # axes.  Retaining the main plan's execution order avoids inventing a
    # layout; each copied record came from the same natural fused shape.
    selected_axes: dict[int, dict[str, Any]] = {}
    resources: list[dict[str, Any]] = []
    for logical_axis in range(len(shape)):
        source = ept16 if logical_axis & 1 else ept4
        source_physical = source["physical"]
        source_axes = donor_axes if source is ept16 else main_axes
        if logical_axis not in source_axes:
            raise AssertionError(f"uniform mixed seed omitted logical axis {logical_axis}")
        selected_axes[logical_axis] = json.loads(json.dumps(source_axes[logical_axis]))
        resources.append(_mixed_resource_from_physical(source_physical))
    physical["axes"] = [selected_axes[int(axis["axis"])]
                        for axis in ept4["physical"]["axes"]]
    physical["axis_resources"] = resources
    return mixed


def _assert_mixed_mapping_contract(mapping: dict[str, Any], rank: int,
                                   label: str,
                                   expected_lowering: str = "axis-tiled-mixed") -> None:
    if mapping.get("kind") != "xd" or mapping.get("lowering") != expected_lowering:
        raise AssertionError(f"{label} is not an axis-tiled-mixed xD mapping")
    physical = mapping.get("physical", {})
    resources = physical.get("axis_resources")
    if not isinstance(resources, list) or len(resources) != rank:
        raise AssertionError(f"{label} has an invalid axis_resources vector")
    axes = physical.get("axes", [])
    if len(axes) != rank:
        raise AssertionError(f"{label} has an unexpected physical axis count")
    for resource in resources:
        if set(("core", "ept", "register_lanes", "block_cap", "exchange")) - set(resource):
            raise AssertionError(f"{label} has an incomplete axis resource entry")
    if any(axis.get("core") != "cufftdx-axis-tiled" for axis in axes):
        raise AssertionError(f"{label} contains a non-fused axis record")


def _mapping_mixed_axis_case(api: Api, handle: Handle, torch: Any,
                             shape: Sequence[int], dtype: Any) -> dict[str, Any]:
    batch = 2
    storage = DATA_COMPLEX_FP64 if dtype == torch.complex128 else DATA_COMPLEX_FP32
    compute = COMPUTE_FP64 if dtype == torch.complex128 else COMPUTE_FP32
    source = _make_random(torch, (batch, *shape), dtype)
    source_before = source.clone()
    forward_mapping = _make_mixed_c2c_mapping(
        api, handle, torch, shape, batch, dtype, FORWARD)
    forward_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                        shape=shape, batch=batch, mode=EXEC_XD,
                        direction=FORWARD,
                        mapping=json.dumps(forward_mapping))
    expected_forward = torch.fft.fftn(
        source, dim=tuple(range(1, len(shape) + 1)), norm="backward")
    forward_out = torch.empty_like(source)
    forward, forward_meta = _run_plan(api, handle, torch, forward_spec,
                                      source, forward_out)
    try:
        resolved = json.loads(forward_meta["mapping_json"])
        _assert_mixed_mapping_contract(resolved, len(shape), "mixed forward")
        if resolved != forward_mapping:
            raise AssertionError("mixed forward replay changed its complete mapping")
        forward_error = _assert_close(
            forward_out, expected_forward,
            atol=5e-11 if dtype == torch.complex128 else 5e-4,
            rtol=5e-11 if dtype == torch.complex128 else 5e-4,
            label="mixed FFT forward")
        if not torch.equal(source, source_before):
            raise AssertionError("mixed forward modified its input")
    finally:
        forward.close()

    # Recreate the same plan from the exact JSON and require byte-for-byte
    # semantic equality, not merely equivalent numerical output.
    replay_out = torch.empty_like(source)
    replay, replay_meta = _run_plan(
        api, handle, torch,
        dict(forward_spec, mapping=json.dumps(forward_mapping)),
        source, replay_out)
    try:
        replay_mapping = json.loads(replay_meta["mapping_json"])
        if replay_mapping != forward_mapping:
            raise AssertionError("mixed C2C replay changed its complete mapping")
        replay_error = _assert_close(
            replay_out, forward_out,
            atol=5e-11 if dtype == torch.complex128 else 5e-4,
            rtol=5e-11 if dtype == torch.complex128 else 5e-4,
            label="mixed replay output")
        if not torch.equal(source, source_before):
            raise AssertionError("mixed replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _make_mixed_c2c_mapping(
        api, handle, torch, shape, batch, dtype, INVERSE)
    inverse_spec = dict(forward_spec, direction=INVERSE,
                        mapping=json.dumps(inverse_mapping))
    recovered = torch.empty_like(source)
    spectrum_before = forward_out.clone()
    inverse, inverse_meta = _run_plan(api, handle, torch, inverse_spec,
                                      forward_out, recovered)
    try:
        inverse_resolved = json.loads(inverse_meta["mapping_json"])
        _assert_mixed_mapping_contract(inverse_resolved, len(shape), "mixed inverse")
        if inverse_resolved != inverse_mapping:
            raise AssertionError("mixed inverse replay changed its complete mapping")
        inverse_error = _assert_close(
            recovered, source,
            atol=8e-11 if dtype == torch.complex128 else 5e-4,
            rtol=8e-11 if dtype == torch.complex128 else 5e-4,
            label="mixed FFT inverse")
        if not torch.equal(forward_out, spectrum_before):
            raise AssertionError("mixed inverse modified its input")
    finally:
        inverse.close()
    return {"shape": list(shape), "batch": batch, "dtype": str(dtype),
            "mapping_exact": True, "input_unchanged": True,
            "forward": forward_error, "replay": replay_error,
            "inverse": inverse_error}


def _make_compact_mixed_mapping(api: Api, handle: Handle, torch: Any,
                                real_dtype: Any, direction: int,
                                fft_type: int) -> dict[str, Any]:
    """Combine a real composite template with two real fused C2C seed axes."""
    real_shape = (64, 64, 33)
    payload_shape = (64, 64, 17)
    storage = DATA_FP64 if real_dtype == torch.float64 else DATA_FP32
    compute = COMPUTE_FP64 if real_dtype == torch.float64 else COMPUTE_FP32
    template_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                         shape=real_shape, batch=2, mode=EXEC_COMPOSITE,
                         direction=direction, fft_type=fft_type)
    template_plan = make_plan(api, handle, torch, **template_spec)
    try:
        template = json.loads(template_plan.metadata["mapping_json"])
    finally:
        template_plan.close()
    physical = template.get("physical", {})
    axis_order = [int(axis) for axis in physical.get("axis_order", [])]
    if sorted(axis_order) != [0, 1]:
        raise UnsupportedCase("compact real template did not retain the two logical complex axes")

    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    ept4 = _uniform_mixed_seed(
        api, handle, torch, payload_shape, 2, complex_dtype, direction,
        _MIXED_EPT4_FAMILY, required_axes=(0, 1))
    ept16 = _uniform_mixed_seed(
        api, handle, torch, payload_shape, 2, complex_dtype, direction,
        _MIXED_EPT16_FAMILY, required_axes=(0, 1))
    donor4 = {int(axis["axis"]): axis for axis in ept4["physical"]["axes"]}
    donor16 = {int(axis["axis"]): axis for axis in ept16["physical"]["axes"]}
    if any(axis not in donor4 or axis not in donor16 for axis in axis_order):
        raise AssertionError("compact mixed donor omitted a logical complex axis")

    mixed = json.loads(json.dumps(template))
    mixed["lowering"] = (
        "xd-compact-c2r[axis-tiled-mixed]" if fft_type == FFT_C2R
        else "xd-compact-r2c[axis-tiled-mixed]")
    physical = mixed["physical"]
    # The real plan's working complex extents are exactly the C2C payload
    # shape.  Axis 0 takes the EPT4 donor; axis 1 takes EPT16.
    physical["axes"] = [json.loads(json.dumps(
        donor16[axis] if axis == 1 else donor4[axis])) for axis in axis_order]
    physical["fused_axes"] = True
    physical["packed_axes"] = True
    physical["paired_axes"] = False
    physical["pair_groups"] = 0
    physical["axis_core"] = "cufftdx-block"
    physical["axis_ept"] = 4
    physical["axis_ffts_per_block_cap"] = 8
    physical["axis_exchange_policy"] = 2
    physical["output_permutation"] = []
    physical["complex_only"] = False
    physical["compact_last_extent"] = 17
    physical["extents"] = list(real_shape)
    physical["axis_resources"] = [
        _mixed_resource_from_physical(ept4["physical"]),
        _mixed_resource_from_physical(ept16["physical"]),
    ]
    return mixed


def _mapping_mixed_real_case(api: Api, handle: Handle, torch: Any,
                             real_dtype: Any) -> dict[str, Any]:
    real_shape = (64, 64, 33)
    batch = 2
    storage = DATA_FP64 if real_dtype == torch.float64 else DATA_FP32
    compute = COMPUTE_FP64 if real_dtype == torch.float64 else COMPUTE_FP32
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    source = torch.randn((batch, *real_shape), dtype=real_dtype, device="cuda")
    source_before = source.clone()
    compact_shape = (batch, 64, 64, 17)
    expected_forward = torch.fft.rfftn(
        source, dim=(1, 2, 3), norm="backward")
    forward_mapping = _make_compact_mixed_mapping(
        api, handle, torch, real_dtype, FORWARD, FFT_R2C)
    forward_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                        shape=real_shape, batch=batch, mode=EXEC_COMPOSITE,
                        direction=FORWARD, fft_type=FFT_R2C,
                        mapping=json.dumps(forward_mapping))
    forward_out = torch.empty(compact_shape, dtype=complex_dtype, device="cuda")
    forward, forward_meta = _run_plan(api, handle, torch, forward_spec,
                                      source, forward_out)
    try:
        resolved = json.loads(forward_meta["mapping_json"])
        _assert_mixed_mapping_contract(
            resolved, 2, "compact mixed R2C",
            "xd-compact-r2c[axis-tiled-mixed]")
        if resolved != forward_mapping:
            raise AssertionError("compact mixed R2C replay changed its complete mapping")
        if forward_meta["algorithm"] != "xd-compact-r2c[axis-tiled-mixed]":
            raise AssertionError("compact R2C did not expose the mixed algorithm family")
        forward_error = _assert_close(
            forward_out, expected_forward,
            atol=5e-11 if real_dtype == torch.float64 else 5e-4,
            rtol=5e-11 if real_dtype == torch.float64 else 5e-4,
            label="compact mixed R2C")
        if not torch.equal(source, source_before):
            raise AssertionError("compact mixed R2C modified its input")
    finally:
        forward.close()

    replay_out = torch.empty_like(forward_out)
    replay, replay_meta = _run_plan(
        api, handle, torch,
        dict(forward_spec, mapping=json.dumps(forward_mapping)),
        source, replay_out)
    try:
        if json.loads(replay_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("compact mixed R2C replay changed its mapping")
        replay_error = _assert_close(
            replay_out, forward_out,
            atol=5e-11 if real_dtype == torch.float64 else 5e-4,
            rtol=5e-11 if real_dtype == torch.float64 else 5e-4,
            label="compact mixed R2C replay")
        if not torch.equal(source, source_before):
            raise AssertionError("compact mixed R2C replay modified its input")
    finally:
        replay.close()

    inverse_mapping = _make_compact_mixed_mapping(
        api, handle, torch, real_dtype, INVERSE, FFT_C2R)
    inverse_spec = dict(op=OP_FFT, storage=storage, compute=compute,
                        shape=real_shape, batch=batch, mode=EXEC_COMPOSITE,
                        direction=INVERSE, fft_type=FFT_C2R,
                        mapping=json.dumps(inverse_mapping))
    spectrum_before = forward_out.clone()
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, inverse_spec,
                                      forward_out, recovered)
    try:
        resolved = json.loads(inverse_meta["mapping_json"])
        _assert_mixed_mapping_contract(
            resolved, 2, "compact mixed C2R",
            "xd-compact-c2r[axis-tiled-mixed]")
        if resolved != inverse_mapping:
            raise AssertionError("compact mixed C2R replay changed its complete mapping")
        if inverse_meta["algorithm"] != "xd-compact-c2r[axis-tiled-mixed]":
            raise AssertionError("compact C2R did not expose the mixed algorithm family")
        expected_inverse = torch.fft.irfftn(
            forward_out, s=real_shape, dim=(1, 2, 3), norm="backward")
        inverse_error = _assert_close(
            recovered, expected_inverse,
            atol=8e-11 if real_dtype == torch.float64 else 8e-4,
            rtol=8e-11 if real_dtype == torch.float64 else 8e-4,
            label="compact mixed C2R")
        if not torch.equal(forward_out, spectrum_before):
            raise AssertionError("compact mixed C2R modified its input")
    finally:
        inverse.close()
    return {"shape": list(real_shape), "batch": batch,
            "dtype": str(real_dtype), "mapping_exact": True,
            "input_unchanged": True, "r2c": forward_error,
            "replay": replay_error, "c2r": inverse_error}


def _mapping_mixed_invalid_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    shape = (64, 64)
    batch = 2
    dtype = torch.complex64
    mapping = _make_mixed_c2c_mapping(api, handle, torch, shape, batch, dtype, FORWARD)
    spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                shape=shape, batch=batch, mode=EXEC_XD, direction=FORWARD)
    variants: dict[str, dict[str, Any]] = {}

    forged_shape = json.loads(json.dumps(mapping))
    forged_shape["physical"]["extents"] = [64, 32]
    variants["physical_shape"] = forged_shape

    forged_batch = json.loads(json.dumps(mapping))
    forged_batch["physical"]["batch"] = batch + 1
    variants["physical_batch"] = forged_batch

    short_resources = json.loads(json.dumps(mapping))
    short_resources["physical"]["axis_resources"] = short_resources["physical"]["axis_resources"][:-1]
    variants["resource_length"] = short_resources

    invalid_core = json.loads(json.dumps(mapping))
    invalid_core["physical"]["axis_resources"][0]["core"] = "not-a-core"
    variants["invalid_core"] = invalid_core

    invalid_ept = json.loads(json.dumps(mapping))
    invalid_ept["physical"]["axis_resources"][0]["ept"] = 7
    variants["invalid_ept"] = invalid_ept

    tampered_axis = json.loads(json.dumps(mapping))
    tampered_axis["physical"]["axes"][0]["elements_per_thread"] += 1
    variants["tampered_axis"] = tampered_axis

    rejected: dict[str, dict[str, Any]] = {}
    for label, forged in variants.items():
        try:
            plan = make_plan(api, handle, torch, **dict(spec, mapping=json.dumps(forged)))
        except CApiError as exc:
            rejected[label] = {"status_code": exc.status, "status_text": exc.message}
            continue
        plan.close()
        raise AssertionError(f"forged mixed mapping was accepted: {label}")
    return {"family": "axis-tiled-mixed", "rejections": rejected,
            "mapping_exact": True}


def _mapping_batch_tile_replay_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    """Replay a complete B=5 cooperative pair plan assembled from B=2/B=1 tiles."""
    family = "pair-tiled-cooperative-layout1"
    mapping_name = "xd:" + family
    shape = (64, 64)

    def tile_mapping(direction: int) -> dict[str, Any]:
        base_spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32,
                         compute=COMPUTE_FP32, shape=shape, mode=EXEC_XD,
                         direction=direction)
        resolved: dict[int, dict[str, Any]] = {}
        for batch in (2, 1, 5):
            plan = make_plan(api, handle, torch,
                             **dict(base_spec, batch=batch, mapping=mapping_name))
            try:
                resolved[batch] = json.loads(plan.metadata["mapping_json"])
            finally:
                plan.close()

        # The outer contract must describe the real B=5 descriptor.  The
        # physical mapping is exactly the implementation's batch_body branch:
        # start with the B=2 body and add the nested body/tail records.
        parent = json.loads(json.dumps(resolved[5]))
        body = json.loads(json.dumps(resolved[2]["physical"]))
        physical = json.loads(json.dumps(body))
        physical["tile_body"] = body
        physical["tile_tail"] = json.loads(json.dumps(resolved[1]["physical"]))
        physical["batch"] = 5
        physical["batch_tile"] = 2
        physical["batch_chunks"] = 3
        physical["axes_scope"] = "per-batch-tile"
        parent["physical"] = physical
        return parent

    source = _make_random(torch, (5, *shape), torch.complex64)
    source_before = source.clone()
    expected = torch.fft.fftn(source, dim=(1, 2), norm="backward")
    forward_mapping = tile_mapping(FORWARD)
    forward_spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32,
                        compute=COMPUTE_FP32, shape=shape, batch=5,
                        mode=EXEC_XD, direction=FORWARD,
                        mapping=json.dumps(forward_mapping))
    forward_out = torch.empty_like(source)
    forward, forward_meta = _run_plan(api, handle, torch, forward_spec,
                                      source, forward_out)
    try:
        if json.loads(forward_meta["mapping_json"]) != forward_mapping:
            raise AssertionError("batch-tile forward replay changed its mapping")
        if not torch.equal(source, source_before):
            raise AssertionError("batch-tile forward replay modified its input")
        forward_error = _assert_close(forward_out, expected, atol=5e-4, rtol=5e-4,
                                      label="batch-tile forward")
    finally:
        forward.close()

    spectrum_before = expected.clone()
    inverse_mapping = tile_mapping(INVERSE)
    inverse_spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32,
                        compute=COMPUTE_FP32, shape=shape, batch=5,
                        mode=EXEC_XD, direction=INVERSE,
                        mapping=json.dumps(inverse_mapping))
    recovered = torch.empty_like(source)
    inverse, inverse_meta = _run_plan(api, handle, torch, inverse_spec,
                                      spectrum_before, recovered)
    try:
        if json.loads(inverse_meta["mapping_json"]) != inverse_mapping:
            raise AssertionError("batch-tile inverse replay changed its mapping")
        if not torch.equal(spectrum_before, expected):
            raise AssertionError("batch-tile inverse replay modified its input")
        inverse_error = _assert_close(recovered, source, atol=5e-4, rtol=5e-4,
                                      label="batch-tile inverse")
    finally:
        inverse.close()

    invalid = json.loads(json.dumps(forward_mapping))
    invalid["physical"]["batch_tile"] = 5
    try:
        rejected = make_plan(api, handle, torch,
                             **dict(forward_spec, mapping=json.dumps(invalid)))
    except CApiError as exc:
        rejection = {"status_code": exc.status, "status_text": exc.message}
    else:
        rejected.close()
        raise AssertionError("batch tile >= batch was accepted")
    return {"family": family, "batch": 5, "batch_tile": 2,
            "mapping_exact": True, "input_unchanged": True,
            "forward": forward_error, "inverse": inverse_error,
            "invalid_tile_rejected": rejection}


def _mapping_contract_mismatch_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    """A complete xD mapping must not replay across semantic contracts."""
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=(4, 2, 8), batch=2, mode=EXEC_XD, direction=FORWARD)
    first = make_plan(api, handle, torch, **spec)
    try:
        mapping = first.metadata["mapping_json"]
        mapping_object = json.loads(mapping)
    finally:
        first.close()
    if mapping_object.get("kind") != "xd":
        raise AssertionError("xD replay did not return a complete xD mapping")

    variants = {
        "batch": dict(spec, batch=3, mapping=mapping),
        "normalize_inverse": dict(spec, normalize_inverse=False, mapping=mapping),
        "shape": dict(spec, shape=(4, 2, 4), mapping=mapping),
    }
    rejections: dict[str, dict[str, Any]] = {}
    for label, variant in variants.items():
        try:
            replay = make_plan(api, handle, torch, **variant)
        except CApiError as exc:
            rejections[label] = {"status_code": exc.status, "status_text": exc.message}
            continue
        replay.close()
        raise AssertionError(f"mapping contract mismatch unexpectedly accepted: {label}")
    return {"mapping_kind": mapping_object["kind"], "rejections": rejections}


def _mapping_cache_replay_case(api: Api, handle: Handle, torch: Any) -> dict[str, Any]:
    """The same xD workload key must replay its cached source and mapping."""
    spec = dict(op=OP_FWHT, storage=DATA_FP32, compute=COMPUTE_FP32,
                shape=(4, 2, 8), batch=2, mode=EXEC_XD, direction=FORWARD)
    with tempfile.TemporaryDirectory(prefix="check_xd_cache_") as directory:
        cache_path = pathlib.Path(directory) / "selection.cache"
        api.check("cubutterflySetCachePath", api.cubutterflySetCachePath(
            handle.value, str(cache_path).encode("utf-8")))
        try:
            first = make_plan(api, handle, torch, **spec)
            first_mapping = first.metadata["mapping_json"]
            first_source = first.metadata["selection_source"]
            first.close()
            second = make_plan(api, handle, torch, **spec)
            second_mapping = second.metadata["mapping_json"]
            second_source = second.metadata["selection_source"]
            second.close()
            forward_flag_variant = make_plan(
                api, handle, torch, **dict(spec, normalize_inverse=False))
            forward_variant_mapping = json.loads(
                forward_flag_variant.metadata["mapping_json"])
            forward_flag_variant.close()
        finally:
            api.check("cubutterflySetCachePath", api.cubutterflySetCachePath(
                handle.value, None))
    if first_mapping != second_mapping:
        raise AssertionError("same workload key replayed a different xD mapping")
    if second_source != SELECTION_USER_CACHE:
        raise AssertionError(f"same workload key did not use the cache source: {second_source}")
    if forward_variant_mapping.get("contract", {}).get("normalize_inverse") is not False:
        raise AssertionError("forward normalize flag variant replayed the wrong xD contract")
    return {"mapping_exact": True, "first_selection_source": first_source,
            "replay_selection_source": second_source,
            "forward_variant_contract": forward_variant_mapping["contract"]}


def _invalid_api_cases(api: Api, handle: Handle, torch: Any) -> list[CaseResult]:
    results: list[CaseResult] = []

    def expect_failure(name: str, callback: Callable[[], int],
                       expected_status_text: str | None = None) -> CaseResult:
        def run() -> dict[str, Any]:
            status = int(callback())
            if status == STATUS_SUCCESS:
                raise AssertionError("call unexpectedly succeeded")
            status_text = api.status_text(status)
            if expected_status_text is not None and status_text != expected_status_text:
                raise AssertionError(
                    f"expected status {expected_status_text!r}, got {status_text!r}")
            return {"status_code": status, "status_text": status_text}
        return _result(name, run)

    def invalid_execution_enum() -> int:
        descriptor = api.create_descriptor()
        try:
            return int(api.cubutterflySetExecutionMode(descriptor, 99))
        finally:
            api.cubutterflyDestroyDescriptor(descriptor)

    def invalid_fft_enum() -> int:
        if api.fft_type_fn is None:
            raise UnsupportedCase("library has no cubutterflySetFftTransformType")
        descriptor = api.create_descriptor()
        try:
            return int(api.fft_type_fn(descriptor, 99))
        finally:
            api.cubutterflyDestroyDescriptor(descriptor)

    def rank1_xd_plan() -> int:
        descriptor = api.create_descriptor()
        plan = ct.c_void_p()
        try:
            api.check("cubutterflySetOperator", api.cubutterflySetOperator(descriptor, OP_FWHT))
            api.check("cubutterflySetDataType", api.cubutterflySetDataType(descriptor, DATA_FP32, COMPUTE_FP32))
            api.check("cubutterflySetShape", api.cubutterflySetShape(descriptor, 1, _ct_sizes((8,)), 1))
            api.check("cubutterflySetExecutionMode", api.cubutterflySetExecutionMode(descriptor, EXEC_XD))
            return int(api.cubutterflyCreatePlan(handle.value, descriptor, ct.byref(plan)))
        finally:
            if plan:
                api.cubutterflyDestroyPlan(plan)
            api.cubutterflyDestroyDescriptor(descriptor)

    def c2r_forward_plan() -> int:
        if api.fft_type_fn is None:
            raise UnsupportedCase("library has no cubutterflySetFftTransformType")
        descriptor = api.create_descriptor()
        plan = ct.c_void_p()
        try:
            api.check("cubutterflySetOperator", api.cubutterflySetOperator(descriptor, OP_FFT))
            api.check("cubutterflySetDataType", api.cubutterflySetDataType(descriptor, DATA_FP32, COMPUTE_FP32))
            api.check("cubutterflySetShape", api.cubutterflySetShape(
                descriptor, 2, _ct_sizes((4, 6)), 1))
            api.check("cubutterflySetFftTransformType", api.fft_type_fn(descriptor, FFT_C2R))
            api.check("cubutterflySetDirection", api.cubutterflySetDirection(
                descriptor, FORWARD, 1))
            return int(api.cubutterflyCreatePlan(handle.value, descriptor, ct.byref(plan)))
        finally:
            if plan:
                api.cubutterflyDestroyPlan(plan)
            api.cubutterflyDestroyDescriptor(descriptor)

    def c2c_xd_output_scale_plan() -> int:
        if api.fft_output_scale_fn is None:
            raise UnsupportedCase("library has no cubutterflySetFftOutputScale")
        descriptor = api.create_descriptor()
        plan = ct.c_void_p()
        try:
            calls = (
                api.cubutterflySetOperator(descriptor, OP_FFT),
                api.cubutterflySetDataType(
                    descriptor, DATA_COMPLEX_FP32, COMPUTE_FP32),
                api.cubutterflySetShape(
                    descriptor, 2, _ct_sizes((16, 32)), 1),
                api.cubutterflySetExecutionMode(descriptor, EXEC_XD),
                api.fft_output_scale_fn(descriptor, 0.5),
                api.cubutterflySetDirection(descriptor, FORWARD, 1),
                api.cubutterflySetPlacement(descriptor, OUT_OF_PLACE),
                api.cubutterflySetLengthMode(descriptor, LENGTH_STANDARD),
            )
            for status in calls:
                if int(status) != STATUS_SUCCESS:
                    return int(status)
            return int(api.cubutterflyCreatePlan(
                handle.value, descriptor, ct.byref(plan)))
        finally:
            if plan:
                api.cubutterflyDestroyPlan(plan)
            api.cubutterflyDestroyDescriptor(descriptor)

    def real_compute_mismatch_plan(storage: int, compute: int) -> int:
        if api.fft_type_fn is None:
            raise UnsupportedCase("library has no cubutterflySetFftTransformType")
        descriptor = api.create_descriptor()
        plan = ct.c_void_p()
        try:
            # Return the first setup/create status directly so the assertion
            # can inspect the public status text without assuming its enum id.
            calls = (
                api.cubutterflySetOperator(descriptor, OP_FFT),
                api.cubutterflySetDataType(descriptor, storage, compute),
                api.cubutterflySetShape(descriptor, 2, _ct_sizes((4, 6)), 1),
                api.cubutterflySetExecutionMode(descriptor, EXEC_XD),
                api.fft_type_fn(descriptor, FFT_R2C),
                api.cubutterflySetDirection(descriptor, FORWARD, 1),
                api.cubutterflySetPlacement(descriptor, OUT_OF_PLACE),
                api.cubutterflySetLengthMode(descriptor, LENGTH_STANDARD),
            )
            for status in calls:
                if int(status) != STATUS_SUCCESS:
                    return int(status)
            return int(api.cubutterflyCreatePlan(
                handle.value, descriptor, ct.byref(plan)))
        finally:
            if plan:
                api.cubutterflyDestroyPlan(plan)
            api.cubutterflyDestroyDescriptor(descriptor)

    def overflow_plan() -> int:
        descriptor = api.create_descriptor()
        plan = ct.c_void_p()
        try:
            api.check("cubutterflySetOperator", api.cubutterflySetOperator(descriptor, OP_FWHT))
            api.check("cubutterflySetDataType", api.cubutterflySetDataType(descriptor, DATA_FP32, COMPUTE_FP32))
            huge_power_of_two = 1 << (ct.sizeof(ct.c_size_t) * 8 - 1)
            # SetShape is allowed to reject the overflowing product itself;
            # that is already the expected contract failure and must not be
            # turned into a failed test by Api.check().
            status = int(api.cubutterflySetShape(
                descriptor, 2, _ct_sizes((huge_power_of_two, 4)), 1))
            if status != STATUS_SUCCESS:
                return status
            return int(api.cubutterflyCreatePlan(handle.value, descriptor, ct.byref(plan)))
        finally:
            if plan:
                api.cubutterflyDestroyPlan(plan)
            api.cubutterflyDestroyDescriptor(descriptor)

    results.append(expect_failure("invalid_execution_mode_enum", invalid_execution_enum))
    results.append(expect_failure("invalid_fft_transform_enum", invalid_fft_enum))
    results.append(expect_failure("rank1_xd_rejected", rank1_xd_plan))
    results.append(expect_failure("c2r_forward_rejected", c2r_forward_plan))
    results.append(expect_failure(
        "c2c_xd_output_scale_rejected", c2c_xd_output_scale_plan,
        expected_status_text="not supported"))
    results.append(expect_failure(
        "real_fp32_storage_fp64_compute_rejected",
        lambda: real_compute_mismatch_plan(DATA_FP32, COMPUTE_FP64),
        expected_status_text="not supported"))
    results.append(expect_failure(
        "real_fp64_storage_fp32_compute_rejected",
        lambda: real_compute_mismatch_plan(DATA_FP64, COMPUTE_FP32),
        expected_status_text="not supported"))
    results.append(expect_failure("shape_product_overflow_rejected", overflow_plan))
    return results


def _event_time(torch: Any, callback: Callable[[], None], warmup: int,
                repeat: int, stream: Any) -> float:
    for _ in range(warmup):
        callback()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(repeat):
        callback()
    stop.record(stream)
    stop.synchronize()
    return float(start.elapsed_time(stop)) / repeat


def _graph_event_time(torch: Any, callback: Callable[[], None], repeat: int,
                      stream: Any) -> tuple[float, bool, str | None]:
    """Time repeated launches through a CUDA graph when capture permits it."""
    if not hasattr(torch.cuda, "CUDAGraph"):
        return 0.0, False, "torch has no CUDAGraph"
    try:
        callback()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            callback()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(repeat):
            graph.replay()
        stop.record(stream)
        stop.synchronize()
        return float(start.elapsed_time(stop)) / repeat, True, None
    except Exception as exc:  # CUDA graph support varies by driver/cuFFT.
        torch.cuda.synchronize()
        return 0.0, False, str(exc)


def _benchmark(api: Api, handle: Handle, torch: Any, warmup: int,
               repeat: int, use_graph: bool) -> list[dict[str, Any]]:
    shape = (4, 8)
    batch = 2
    source = torch.randn((batch, *shape), dtype=torch.complex64, device="cuda")
    output = torch.empty_like(source)
    stream = torch.cuda.current_stream()
    rows: list[dict[str, Any]] = []
    for mode, name in ((EXEC_XD, "xd"), (EXEC_AUTO, "auto"), (EXEC_COMPOSITE, "composite")):
        spec = dict(op=OP_FFT, storage=DATA_COMPLEX_FP32, compute=COMPUTE_FP32,
                    shape=shape, batch=batch, mode=mode, direction=FORWARD)
        try:
            plan = make_plan(api, handle, torch, **spec)
            callback = lambda plan=plan: plan.execute(source, output)
            graph_ms = 0.0
            graph_used = False
            graph_error = None
            if use_graph:
                graph_ms, graph_used, graph_error = _graph_event_time(
                    torch, callback, repeat, stream)
            event_ms = graph_ms if graph_used else _event_time(
                torch, callback, warmup, repeat, stream)
            rows.append({"case": f"fft_{name}", "mode": name, "vendor": "cuButterfly",
                         "algorithm": plan.metadata["algorithm"], "selection_source": plan.metadata["selection_source"],
                         "kernel_ms": event_ms, "graph_used": graph_used,
                         "graph_error": graph_error})
            plan.close()
        except UnsupportedCase as exc:
            rows.append({"case": f"fft_{name}", "mode": name, "status": "skipped", "reason": str(exc)})
        except Exception as exc:
            rows.append({"case": f"fft_{name}", "mode": name, "status": "failed", "error": str(exc)})

    torch_out: Any = torch.empty_like(source)

    def torch_fft() -> None:
        nonlocal torch_out
        torch_out = torch.fft.fftn(source, dim=(1, 2), norm="backward")

    torch_ms = _event_time(torch, torch_fft, warmup, repeat, stream)
    rows.append({"case": "fft_torch", "mode": "torch.fft.fftn", "vendor": "PyTorch/cuFFT",
                 "kernel_ms": torch_ms, "graph_used": False})
    return rows


def _find_library(explicit: str | None) -> pathlib.Path:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    for key in ("CUBUTTERFLY_LIB", "LIBCUNTT"):
        if os.environ.get(key):
            candidates.append(os.environ[key])
    repo = pathlib.Path(__file__).resolve().parents[1]
    candidates.extend([
        str(repo / "build" / "libcuntt.so"),
        str(repo / "build-e13-shared" / "libcuntt.so"),
    ])
    candidates.extend(glob.glob(str(repo / "build-*" / "libcuntt.so")))
    seen: set[str] = set()
    for candidate in candidates:
        path = pathlib.Path(candidate).expanduser()
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("libcuntt.so not found; pass --library or CUBUTTERFLY_LIB")


def _run_cases(api: Api, handle: Handle, torch: Any,
               selected: str | None) -> list[CaseResult]:
    cases: list[tuple[str, Callable[[], dict[str, Any]]]] = []
    cases.append(("fft_rank2_4x8_fp32_xd", lambda: _fft_case(
        api, handle, torch, (4, 8), 2, torch.complex64, DATA_COMPLEX_FP32, EXEC_XD)))
    cases.append(("fft_rank3_4x2x8_fp32_xd", lambda: _fft_case(
        api, handle, torch, (4, 2, 8), 2, torch.complex64, DATA_COMPLEX_FP32, EXEC_XD)))
    cases.append(("fft_rank2_4x8_fp64_xd", lambda: _fft_case(
        api, handle, torch, (4, 8), 2, torch.complex128, DATA_COMPLEX_FP64, EXEC_XD)))
    cases.append(("fft_rank3_4x2x8_fp64_xd", lambda: _fft_case(
        api, handle, torch, (4, 2, 8), 2, torch.complex128, DATA_COMPLEX_FP64, EXEC_XD)))
    cases.append(("fft_rank3_2x3x5_fp32_xd", lambda: _fft_case(
        api, handle, torch, (2, 3, 5), 2, torch.complex64, DATA_COMPLEX_FP32, EXEC_XD)))
    cases.append(("fft_rank3_2x3x5_fp64_xd", lambda: _fft_case(
        api, handle, torch, (2, 3, 5), 2, torch.complex128, DATA_COMPLEX_FP64, EXEC_XD)))
    for lowering in ("strided-axis-cufft", "packed-axis-cufft", "axis-tiled-cufftdx",
                     "axis-tiled-cufftdx-ept16", "pair-tiled-cufftdx", "pair-tiled-cufftdx-ept16",
                     "axis-tiled-register", "pair-tiled-cooperative", "pair-tiled-cooperative-register"):
        mapping = f"xd:{lowering}"
        label = lowering.replace("-", "_")
        if "cooperative" in lowering:
            shapes = ((64, 64), (16, 64, 64))
        elif lowering.startswith("axis-tiled-register"):
            shapes = ((64, 128), (16, 64, 128))
        elif "pair" in lowering:
            shapes = ((16, 32), (16, 32, 32))
        elif "ept16" in lowering:
            shapes = ((16, 32), (8, 16, 32))
        else:
            shapes = ((4, 8), (4, 2, 8))
        for shape in shapes:
            shape_label = "x".join(map(str, shape))
            for dtype, storage, compute, precision in (
                    (torch.complex64, DATA_COMPLEX_FP32, COMPUTE_FP32, "fp32"),
                    (torch.complex128, DATA_COMPLEX_FP64, COMPUTE_FP64, "fp64")):
                cases.append((f"fft_{label}_{shape_label}_{precision}",
                              lambda mapping=mapping, shape=shape, dtype=dtype,
                              storage=storage, compute=compute: _fft_case(
                                  api, handle, torch, shape, 2, dtype, storage,
                                  EXEC_XD, mapping=mapping)))
    # The v10 families get one representative rectangular pair/axis shape in
    # each precision; replay below covers their complete mapping contracts.
    for lowering in ("axis-tiled-register-l8", "axis-tiled-register-l16",
                     "pair-tiled-cooperative-register-l8", "pair-tiled-cooperative-register-l16",
                     "pair-tiled-cooperative-layout1",
                     "pair-tiled-cooperative-register-layout1"):
        mapping = f"xd:{lowering}"
        shapes = ((64, 128), (256, 32), (8, 256, 32)) if lowering.startswith("axis-") else (
            (64, 64, 64) if "cooperative-register" in lowering else (64, 64),)
        label = lowering.replace("-", "_")
        for shape in shapes:
            shape_label = "x".join(map(str, shape))
            for dtype, storage, precision in (
                    (torch.complex64, DATA_COMPLEX_FP32, "fp32"),
                    (torch.complex128, DATA_COMPLEX_FP64, "fp64")):
                cases.append((f"fft_{label}_{shape_label}_{precision}",
                              lambda mapping=mapping, shape=shape, dtype=dtype,
                              storage=storage: _fft_case(
                                  api, handle, torch, shape, 2, dtype, storage,
                                  EXEC_XD, mapping=mapping)))
    cases.append(("register_axis_lane4_256_unavailable",
                  lambda: _register_axis_lane4_256_unavailable_case(
                      api, handle, torch)))
    # The axis block-cap families share the same rank-2 representative.  Keep
    # B=3 here so the forced C2C path covers both forward and inverse batches.
    for lowering in ("axis-tiled-cufftdx-cap8", "axis-tiled-cufftdx-cap16",
                     "axis-tiled-cufftdx-ept16-cap8", "axis-tiled-cufftdx-ept16-cap16"):
        mapping = f"xd:{lowering}"
        label = lowering.replace("-", "_")
        for dtype, storage, precision in (
                (torch.complex64, DATA_COMPLEX_FP32, "fp32"),
                (torch.complex128, DATA_COMPLEX_FP64, "fp64")):
            cases.append((f"fft_{label}_{precision}",
                          lambda mapping=mapping, dtype=dtype, storage=storage: _fft_case(
                          api, handle, torch, (64, 64), 3, dtype, storage,
                              EXEC_XD, mapping=mapping)))
    # The local family inventory is intentionally small: cover both caps,
    # all supported lengths, and both complex precisions. Each helper also
    # runs forward and normalized inverse while checking input preservation.
    for lowering in _LOCAL_FAMILIES:
        mapping = f"xd:{lowering}"
        for length in (64, 128, 256):
            for dtype, storage, precision in (
                    (torch.complex64, DATA_COMPLEX_FP32, "fp32"),
                    (torch.complex128, DATA_COMPLEX_FP64, "fp64")):
                cases.append((f"fft_{lowering.replace('-', '_')}_{length}_{precision}",
                              lambda mapping=mapping, length=length, dtype=dtype,
                              storage=storage: _fft_case(
                                  api, handle, torch, (length, 3), 3, dtype,
                                  storage, EXEC_XD, mapping=mapping)))
    # Exercise both exchange policies in both precisions. Cap/EPT suffixes are
    # intentionally mixed so forced metadata covers the new family contract.
    for lowering, dtype, storage, precision in (
            ("axis-tiled-cufftdx-direct", torch.complex64, DATA_COMPLEX_FP32, "fp32"),
            ("axis-tiled-cufftdx-cap8-direct", torch.complex128, DATA_COMPLEX_FP64, "fp64"),
            ("axis-tiled-cufftdx-unrolled", torch.complex64, DATA_COMPLEX_FP32, "fp32"),
            ("axis-tiled-cufftdx-ept16-cap16-unrolled", torch.complex128,
             DATA_COMPLEX_FP64, "fp64"),
            ("axis-tiled-cufftdx-ept4", torch.complex64, DATA_COMPLEX_FP32, "fp32"),
            ("axis-tiled-cufftdx-ept4-unrolled", torch.complex128,
             DATA_COMPLEX_FP64, "fp64"),
            ("axis-tiled-cufftdx-unrolled32", torch.complex64, DATA_COMPLEX_FP32, "fp32"),
            ("axis-tiled-cufftdx-ept16-unrolled32", torch.complex128,
             DATA_COMPLEX_FP64, "fp64")):
        mapping = f"xd:{lowering}"
        cases.append((f"fft_{lowering.replace('-', '_')}_{precision}",
                      lambda mapping=mapping, dtype=dtype, storage=storage: _fft_case(
                          api, handle, torch, (64, 64), 3, dtype, storage,
                          EXEC_XD, mapping=mapping)))
    cases.append(("fwht_rank3_4x2x8", lambda: _fwht_case(
        api, handle, torch, (4, 2, 8), 2)))
    cases.append(("subset_zeta_mod_2_32", lambda: _zeta_case(
        api, handle, torch, OP_SUBSET_ZETA)))
    cases.append(("superset_zeta_mod_2_32", lambda: _zeta_case(
        api, handle, torch, OP_SUPERSET_ZETA)))
    cases.append(("ntt_p97_uint32", lambda: _ntt_case(
        api, handle, torch, DATA_UINT32, COMPUTE_UINT32)))
    cases.append(("ntt_p97_uint64", lambda: _ntt_case(
        api, handle, torch, DATA_UINT64, COMPUTE_UINT64)))
    for mapping, label in (("xd:resident-t128", "resident"),
                           ("xd:composite", "composite"),
                           (None, "default")):
        cases.append((f"ntt_p97_uint32_normalize_flag_false_{label}",
                      lambda mapping=mapping: _ntt_case(
                          api, handle, torch, DATA_UINT32, COMPUTE_UINT32,
                          mapping=mapping, normalize_inverse=False)))
    cases.append(("structured_2x2_per_axis_matrices", lambda: _structured_case(
        api, handle, torch)))
    cases.append(("fwht_complex_strides", lambda: _strided_fwht_case(
        api, handle, torch)))
    cases.append(("fwht_rank3_xd_positive_strides", lambda: _strided_rank3_fwht_case(
        api, handle, torch)))
    cases.append(("fwht_in_place", lambda: _inplace_fwht_case(
        api, handle, torch)))
    cases.append(("fwht_rank2_xd_in_place", lambda: _inplace_rank2_fwht_case(
        api, handle, torch)))
    cases.append(("fft_rank2_xd_in_place", lambda: _inplace_fft_case(
        api, handle, torch)))
    cases.append(("fwht_zero_extended_embedding", lambda: _embedding_fwht_case(
        api, handle, torch)))
    cases.append(("fwht_rank3_zero_extended_embedding", lambda: _embedding_rank3_fwht_case(
        api, handle, torch)))
    cases.append(("mapping_replay_exact", lambda: _mapping_replay_case(
        api, handle, torch)))
    for fft_type, label in ((FFT_C2C, "c2c"), (FFT_R2C, "r2c"), (FFT_C2R, "c2r")):
        cases.append((f"mapping_replay_fft_{label}",
                      lambda fft_type=fft_type: _mapping_replay_case(api, handle, torch, fft_type)))
    cases.append(("mapping_replay_fft_axis_tiled", lambda: _mapping_replay_case(
        api, handle, torch, FFT_C2C, True)))
    for lowering in ("axis-tiled-cufftdx-ept16", "pair-tiled-cufftdx", "pair-tiled-cufftdx-ept16",
                     "axis-tiled-register", "pair-tiled-cooperative", "pair-tiled-cooperative-register"):
        cases.append((f"mapping_replay_fft_{lowering}", lambda lowering=lowering: _mapping_replay_case(
            api, handle, torch, FFT_C2C, True, lowering)))
    # Keep the new lane/layout replay probes small while covering every new
    # forced family in both precisions.  The forced FFT cases above exercise
    # both directions; these cases additionally require exact full-map replay.
    for lowering, dtype, precision in (
            ("axis-tiled-register-l8", torch.complex64, "fp32"),
            ("axis-tiled-register-l16", torch.complex128, "fp64"),
            ("pair-tiled-cooperative-register-l8", torch.complex64, "fp32"),
            ("pair-tiled-cooperative-register-l16", torch.complex128, "fp64"),
            ("pair-tiled-cooperative-layout1", torch.complex64, "fp32"),
            ("pair-tiled-cooperative-register-layout1", torch.complex128, "fp64")):
        cases.append((f"mapping_replay_fft_{lowering}_{precision}",
                      lambda lowering=lowering, dtype=dtype: _mapping_replay_case(
                          api, handle, torch, FFT_C2C, True, lowering, dtype)))
    for lowering in ("axis-tiled-register-l8", "axis-tiled-register-l16"):
        for dtype, precision in ((torch.complex64, "fp32"),
                                 (torch.complex128, "fp64")):
            cases.append((f"mapping_replay_fft_{lowering}_256x32_{precision}",
                          lambda lowering=lowering, dtype=dtype: _mapping_replay_case(
                              api, handle, torch, FFT_C2C, True, lowering, dtype,
                              batch=3, shape_override=(256, 32))))
    for lowering in ("axis-tiled-cufftdx-cap8", "axis-tiled-cufftdx-cap16",
                     "axis-tiled-cufftdx-ept16-cap8", "axis-tiled-cufftdx-ept16-cap16"):
        for dtype, precision in ((torch.complex64, "fp32"),
                                 (torch.complex128, "fp64")):
            cases.append((f"mapping_replay_fft_{lowering}_{precision}",
                          lambda lowering=lowering, dtype=dtype: _mapping_replay_case(
                              api, handle, torch, FFT_C2C, True, lowering, dtype,
                              batch=3)))
    for lowering, length, dtype, precision in (
            ("axis-tiled-local", 128, torch.complex64, "fp32"),
            ("axis-tiled-local-cap16", 256, torch.complex128, "fp64")):
        cases.append((f"mapping_replay_fft_{lowering}_{length}_{precision}",
                      lambda lowering=lowering, length=length, dtype=dtype: _mapping_replay_case(
                          api, handle, torch, FFT_C2C, True, lowering, dtype,
                          batch=3, shape_override=(length, 3))))
    cases.append(("mapping_replay_fft_local_mixed", lambda: _mapping_local_mixed_case(
        api, handle, torch)))
    cases.append(("mapping_replay_compact_local", lambda: _mapping_local_real_case(
        api, handle, torch)))
    for dtype, precision in ((torch.complex64, "fp32"),
                             (torch.complex128, "fp64")):
        cases.append((f"mapping_replay_fft_local_recurrence_{precision}",
                      lambda dtype=dtype: _mapping_local_recurrence_c2c_case(
                          api, handle, torch, dtype)))
    for real_dtype, precision in ((torch.float32, "fp32"),
                                  (torch.float64, "fp64")):
        cases.append((f"mapping_replay_compact_local_recurrence_{precision}",
                      lambda real_dtype=real_dtype: _mapping_local_real_case(
                          api, handle, torch, real_dtype,
                          "axis-tiled-local-recurrence")))
    cases.append(("mapping_forced_compact_local_recurrence_rank3_fp32",
                  lambda: _real_fft_case(
                      api, handle, torch, (128, 256, 33), 2, torch.float32,
                      DATA_FP32, mapping="xd:axis-tiled-local-recurrence",
                      mapping_validator=_assert_forced_compact_local_recurrence)))
    for dtype, precision in ((torch.complex64, "fp32"),
                             (torch.complex128, "fp64")):
        cases.append((f"mapping_replay_fft_local_recurrence_batch_tail_{precision}",
                      lambda dtype=dtype: _mapping_local_recurrence_batch_case(
                          api, handle, torch, dtype)))
    cases.append(("mapping_replay_fft_local_recurrence_invalid_policy2",
                  lambda: _mapping_local_recurrence_invalid_case(
                      api, handle, torch, "policy2")))
    cases.append(("mapping_replay_fft_local_recurrence_invalid_nonlocal",
                  lambda: _mapping_local_recurrence_invalid_case(
                      api, handle, torch, "policy1_nonlocal")))
    for lowering, dtype, precision in (
            ("axis-tiled-cufftdx-direct", torch.complex64, "fp32"),
            ("axis-tiled-cufftdx-cap8-direct", torch.complex128, "fp64"),
            ("axis-tiled-cufftdx-unrolled", torch.complex64, "fp32"),
            ("axis-tiled-cufftdx-ept16-cap16-unrolled", torch.complex128, "fp64"),
            ("axis-tiled-cufftdx-ept4", torch.complex64, "fp32"),
            ("axis-tiled-cufftdx-ept4-unrolled", torch.complex128, "fp64"),
            ("axis-tiled-cufftdx-unrolled32", torch.complex64, "fp32"),
            ("axis-tiled-cufftdx-ept16-unrolled32", torch.complex128, "fp64")):
        cases.append((f"mapping_replay_fft_{lowering}_{precision}",
                      lambda lowering=lowering, dtype=dtype: _mapping_replay_case(
                      api, handle, torch, FFT_C2C, True, lowering, dtype,
                              batch=3)))
    for shape, shape_label in (((64, 64), "64x64"),
                               ((16, 64, 64), "16x64x64")):
        for dtype, precision in ((torch.complex64, "fp32"),
                                 (torch.complex128, "fp64")):
            cases.append((f"mapping_replay_fft_axis_tiled_mixed_{shape_label}_{precision}",
                          lambda shape=shape, dtype=dtype: _mapping_mixed_axis_case(
                              api, handle, torch, shape, dtype)))
    cases.append(("mapping_replay_fft_axis_tiled_mixed_invalid",
                  lambda: _mapping_mixed_invalid_case(api, handle, torch)))
    for real_dtype, precision in ((torch.float32, "fp32"),
                                  (torch.float64, "fp64")):
        cases.append((f"mapping_replay_compact_mixed_{precision}",
                      lambda real_dtype=real_dtype: _mapping_mixed_real_case(
                          api, handle, torch, real_dtype)))
    cases.append(("mapping_replay_fft_batch_tile_layout1",
                  lambda: _mapping_batch_tile_replay_case(api, handle, torch)))
    cases.append(("mapping_contract_mismatch_rejected", lambda: _mapping_contract_mismatch_case(
        api, handle, torch)))
    cases.append(("mapping_cache_replay_exact", lambda: _mapping_cache_replay_case(
        api, handle, torch)))
    # Exercise every resident resource choice explicitly.  These plans bypass
    # the selector's winner and therefore validate both the 128- and
    # 256-thread contracts whenever the current device advertises them.
    for lowering in ("resident-t128", "resident-t256"):
        mapping = f"xd:{lowering}"
        label = lowering.replace("-", "_")
        cases.append((f"{label}_fft_fp32", lambda mapping=mapping: _fft_case(
            api, handle, torch, (4, 8), 2, torch.complex64, DATA_COMPLEX_FP32,
            EXEC_XD, mapping=mapping)))
        cases.append((f"{label}_fwht", lambda mapping=mapping: _fwht_case(
            api, handle, torch, (4, 8), 2, EXEC_XD, mapping=mapping)))
        cases.append((f"{label}_subset_zeta", lambda mapping=mapping: _zeta_case(
            api, handle, torch, OP_SUBSET_ZETA, mapping=mapping)))
        cases.append((f"{label}_superset_zeta", lambda mapping=mapping: _zeta_case(
            api, handle, torch, OP_SUPERSET_ZETA, mapping=mapping)))
        cases.append((f"{label}_ntt_uint32", lambda mapping=mapping: _ntt_case(
            api, handle, torch, DATA_UINT32, COMPUTE_UINT32, mapping=mapping)))
        cases.append((f"{label}_ntt_uint64", lambda mapping=mapping: _ntt_case(
            api, handle, torch, DATA_UINT64, COMPUTE_UINT64, mapping=mapping)))
        cases.append((f"{label}_structured", lambda mapping=mapping: _structured_case(
            api, handle, torch, mapping=mapping)))
    real_shapes = ((5,), (6,), (4, 6), (3, 5), (2, 3, 4), (2, 3, 5))
    for shape in real_shapes:
        rank = len(shape)
        cases.append((f"real_rank{rank}_{'x'.join(map(str, shape))}_fp32", lambda shape=shape: _real_fft_case(
            api, handle, torch, shape, 2, torch.float32, DATA_FP32)))
        cases.append((f"real_rank{rank}_{'x'.join(map(str, shape))}_fp64", lambda shape=shape: _real_fft_case(
            api, handle, torch, shape, 2, torch.float64, DATA_FP64)))
    for real_dtype, storage, precision in (
            (torch.float32, DATA_FP32, "fp32"),
            (torch.float64, DATA_FP64, "fp64")):
        cases.append((f"real_register_contract_64x1_{precision}",
                      lambda real_dtype=real_dtype, storage=storage: _real_fft_case(
                          api, handle, torch, (64, 1), 2, real_dtype, storage,
                          mapping_validator=_assert_real_register_mapping_contract)))
    cases.append(("real_fft_output_scale", lambda: _real_fft_case(
        api, handle, torch, (4, 6), 2, torch.float32, DATA_FP32, 0.5)))
    for shape in ((1, 8), (2, 1, 8)):
        shape_label = "x".join(map(str, shape))
        cases.append((f"fft_unitaxis_{shape_label}_default", lambda shape=shape: _fft_case(
            api, handle, torch, shape, 2, torch.complex64, DATA_COMPLEX_FP32,
            EXEC_AUTO)))
        cases.append((f"fft_unitaxis_{shape_label}_resident", lambda shape=shape: _fft_case(
            api, handle, torch, shape, 2, torch.complex64, DATA_COMPLEX_FP32,
            EXEC_XD, mapping="xd:resident-t128")))
    results: list[CaseResult] = []
    for name, callback in cases:
        if selected and selected not in name:
            continue
        results.append(_result(name, callback))
    if (not selected or "invalid" in selected or "overflow" in selected or
            "rank1" in selected or "real_fp" in selected):
        results.extend(_invalid_api_cases(api, handle, torch))
    return results


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="path to libcuntt.so")
    parser.add_argument("--device", type=int, default=None, help="CUDA device index")
    parser.add_argument("--case", dest="selected", help="run names containing this substring")
    parser.add_argument("--benchmark", action="store_true", help="run the xD/auto/composite benchmark")
    parser.add_argument("--benchmark-graph", action="store_true", help="try CUDA graph timing")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output", help="write the JSON report to this path")
    parser.add_argument("--indent", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    report: dict[str, Any] = {
        "script": str(pathlib.Path(__file__).resolve()),
        "timestamp": time.time(),
        "cases": [],
        "benchmarks": [],
    }
    try:
        import torch
    except Exception as exc:
        report["status"] = "skipped"
        report["reason"] = f"torch import failed: {exc}"
        print(json.dumps(report, indent=args.indent, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        report["status"] = "skipped"
        report["reason"] = "CUDA is unavailable"
        print(json.dumps(report, indent=args.indent, sort_keys=True))
        return 0
    if args.device is not None:
        torch.cuda.set_device(args.device)
    try:
        library = _find_library(args.library)
        api = Api(library)
        report["library"] = str(library)
        report["device"] = torch.cuda.get_device_name(torch.cuda.current_device())
        report["torch"] = torch.__version__
        stream = int(torch.cuda.current_stream().cuda_stream)
        with Handle(api, stream) as handle:
            results = _run_cases(api, handle, torch, args.selected)
            report["cases"] = [result.as_dict() for result in results]
            if args.benchmark:
                report["benchmarks"] = _benchmark(api, handle, torch, max(0, args.warmup),
                                                   max(1, args.repeat), args.benchmark_graph)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        print(json.dumps(report, indent=args.indent, sort_keys=True))
        return 2
    failed = sum(result.status == "failed" for result in [
        CaseResult(item["name"], item["status"], item)
        for item in report["cases"]
    ])
    report["status"] = "failed" if failed else "passed"
    report["summary"] = {
        "passed": sum(item["status"] == "passed" for item in report["cases"]),
        "failed": failed,
        "skipped": sum(item["status"] == "skipped" for item in report["cases"]),
        "benchmarks": len(report["benchmarks"]),
    }
    text = json.dumps(report, indent=args.indent, sort_keys=True)
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
