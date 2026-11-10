#!/usr/bin/env python3
"""Bounded QuaRot FWHT replacement backed by the cuButterfly C API.

The pinned QuaRot path calls ``matmul_hadU_cuda`` with a tensor whose last
dimension is either a power of two (the attention K=1 path) or is factored as
``K * (n // K)``.  This adapter replaces only the power-of-two transform.  It
does not pad a non-power-of-two axis and it leaves the optional ``hadK`` matrix
multiply to PyTorch, matching the upstream operation order.

The adapter is intentionally inference-oriented.  The C API execution is
asynchronous and follows the caller's current PyTorch CUDA stream.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import importlib
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIBRARY = _REPO_ROOT / "build-e13-shared" / "libcuntt.so"
PINNED_QUAROT_COMMIT = "5008669b08c1f11f9b64d52d16fddd47ca754c5a"

_FWHT = 2
_FP16 = 0
_FP32 = 2
_COMPUTE_NATIVE = 0
_COMPUTE_FP32 = 1
_FORWARD = 0
_OUT_OF_PLACE = 0
_STANDARD = 0
_DEFAULT_POLICY = 0
_MEASURE_POLICY = 1
_EXPLICIT_POLICY = 2


def _cache_path(value=None):
    value = os.environ.get("CUBUTTERFLY_PLAN_CACHE") if value is None else value
    if value is None:
        return None
    if not os.fspath(value):
        raise ValueError("plan cache path must be nonempty")
    path = Path(value).expanduser().resolve()
    if (path.exists() and not path.is_file()) or not path.parent.is_dir():
        raise ValueError("plan cache requires a regular file and an existing parent")
    if not os.access(path if path.exists() else path.parent, os.R_OK | os.W_OK):
        raise ValueError("plan cache must be readable and writable")
    return str(path)


def _file_identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def _registry_identity():
    explicit = os.environ.get("CUBUTTERFLY_REGISTRY")
    path = Path(explicit) if explicit is not None else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "cubutterfly/registry.json"
    path = path.expanduser().resolve()
    if explicit is not None and (not explicit or not path.is_file()):
        raise ValueError("explicit CUBUTTERFLY_REGISTRY must name an existing file")
    return _file_identity(path)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _as_stream_id(stream: Any) -> int:
    """Return the integer CUDA stream handle used by the C ABI."""

    value = getattr(stream, "cuda_stream", stream)
    return int(value)


class _Api:
    """Small ctypes declaration for the public cuButterfly C ABI."""

    def __init__(self, library: str | Path):
        self.path = Path(library).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"cuButterfly shared library not found: {self.path}")
        mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        self.lib = ctypes.CDLL(str(self.path), mode=mode)
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t

        self.lib.cubutterflyGetStatusString.restype = ctypes.c_char_p
        self.lib.cubutterflyGetStatusString.argtypes = [ctypes.c_int]

        self.lib.cubutterflyCreate.restype = ctypes.c_int
        self.lib.cubutterflyCreate.argtypes = [ctypes.POINTER(pointer)]
        self.lib.cubutterflyDestroy.restype = ctypes.c_int
        self.lib.cubutterflyDestroy.argtypes = [pointer]
        self.lib.cubutterflySetStream.restype = ctypes.c_int
        self.lib.cubutterflySetStream.argtypes = [pointer, ctypes.c_uint64]
        self.lib.cubutterflySetCachePath.restype = ctypes.c_int
        self.lib.cubutterflySetCachePath.argtypes = [pointer, ctypes.c_char_p]
        for name in ("MappingJson", "DescriptorJson", "SelectionReason", "AlgorithmName"):
            function = getattr(self.lib, "cubutterflyPlanGet" + name)
            function.restype = ctypes.c_int
            function.argtypes = [pointer, ctypes.c_void_p, ctypes.POINTER(size)]
        self.lib.cubutterflyPlanGetSelectionSource.restype = ctypes.c_int
        self.lib.cubutterflyPlanGetSelectionSource.argtypes = [pointer, ctypes.POINTER(ctypes.c_int)]

        self.lib.cubutterflyCreateDescriptor.restype = ctypes.c_int
        self.lib.cubutterflyCreateDescriptor.argtypes = [ctypes.POINTER(pointer)]
        self.lib.cubutterflyDestroyDescriptor.restype = ctypes.c_int
        self.lib.cubutterflyDestroyDescriptor.argtypes = [pointer]
        self.lib.cubutterflySetOperator.restype = ctypes.c_int
        self.lib.cubutterflySetOperator.argtypes = [pointer, ctypes.c_int]
        self.lib.cubutterflySetDataType.restype = ctypes.c_int
        self.lib.cubutterflySetDataType.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
        self.lib.cubutterflySetShape.restype = ctypes.c_int
        self.lib.cubutterflySetShape.argtypes = [pointer, ctypes.c_uint32, ctypes.POINTER(size), size]
        self.lib.cubutterflySetDirection.restype = ctypes.c_int
        self.lib.cubutterflySetDirection.argtypes = [pointer, ctypes.c_int, ctypes.c_int]
        self.lib.cubutterflySetPlacement.restype = ctypes.c_int
        self.lib.cubutterflySetPlacement.argtypes = [pointer, ctypes.c_int]
        self.lib.cubutterflySetLengthMode.restype = ctypes.c_int
        self.lib.cubutterflySetLengthMode.argtypes = [pointer, ctypes.c_int]
        for name, kind in (("OutputScale", ctypes.c_double), ("OutputRounding", ctypes.c_int)):
            function = getattr(self.lib, "cubutterflySet" + name, None)
            if function is not None:
                function.restype = ctypes.c_int
                function.argtypes = [pointer, kind]
        self.lib.cubutterflySetAlgorithmPolicy.restype = ctypes.c_int
        self.lib.cubutterflySetAlgorithmPolicy.argtypes = [pointer, ctypes.c_int, ctypes.c_char_p]

        self.lib.cubutterflyCreatePlan.restype = ctypes.c_int
        self.lib.cubutterflyCreatePlan.argtypes = [pointer, pointer, ctypes.POINTER(pointer)]
        self.lib.cubutterflyDestroyPlan.restype = ctypes.c_int
        self.lib.cubutterflyDestroyPlan.argtypes = [pointer]
        self.lib.cubutterflyPlanGetWorkspaceSize.restype = ctypes.c_int
        self.lib.cubutterflyPlanGetWorkspaceSize.argtypes = [pointer, ctypes.POINTER(size)]
        self.lib.cubutterflyPlanSetWorkspace.restype = ctypes.c_int
        self.lib.cubutterflyPlanSetWorkspace.argtypes = [pointer, pointer, size]
        self.lib.cubutterflyExecute.restype = ctypes.c_int
        self.lib.cubutterflyExecute.argtypes = [pointer, pointer, pointer, pointer]
        self.lib.cubutterflyExecuteSteady = getattr(self.lib, "cubutterflyExecuteSteady", None)
        if self.lib.cubutterflyExecuteSteady is not None:
            self.lib.cubutterflyExecuteSteady.restype = ctypes.c_int
            self.lib.cubutterflyExecuteSteady.argtypes = [pointer, pointer, pointer, pointer]

    def check(self, status: int, operation: str) -> None:
        status = int(status)
        if status == 0:
            return
        message = self.lib.cubutterflyGetStatusString(status)
        if message is None:
            detail = f"status {status}"
        else:
            detail = message.decode("utf-8", errors="replace")
        raise RuntimeError(f"{operation} failed: {detail} ({status})")


@dataclass
class _Plan:
    api: _Api
    context: ctypes.c_void_p
    plan: ctypes.c_void_p
    workspace: torch.Tensor | None
    device: torch.device
    extent: int
    batch: int
    stream: int
    metadata: dict[str, Any]
    reusable_output: torch.Tensor | None = None
    reusable_narrow: torch.Tensor | None = None

    def execute(self, values: torch.Tensor, *, reuse_output: bool = False) -> torch.Tensor:
        """Submit one out-of-place transform without synchronizing the host."""

        stream = _as_stream_id(torch.cuda.current_stream(values.device))
        if stream != self.stream:
            self.api.check(
                self.api.lib.cubutterflySetStream(self.context, stream),
                "set caller stream",
            )
            self.stream = stream
        if reuse_output:
            if (self.reusable_output is None or self.reusable_output.shape != values.shape
                    or self.reusable_output.dtype != values.dtype
                    or self.reusable_output.device != values.device):
                self.reusable_output = torch.empty_like(values)
            output = self.reusable_output
        else:
            output = torch.empty_like(values)
        execute = self.api.lib.cubutterflyExecuteSteady
        if execute is None:
            execute = self.api.lib.cubutterflyExecute
        self.api.check(
            execute(
                self.context,
                self.plan,
                ctypes.c_void_p(values.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
            ),
            "execute FWHT",
        )
        return output

    def close(self) -> None:
        if self.plan and self.plan.value:
            self.api.lib.cubutterflyDestroyPlan(self.plan)
            self.plan = ctypes.c_void_p()
        if self.context and self.context.value:
            self.api.lib.cubutterflyDestroy(self.context)
            self.context = ctypes.c_void_p()
        self.workspace = None
        self.reusable_output = None
        self.reusable_narrow = None


class QuaRotFWHTAdapter:
    """Replace QuaRot's power-of-two Hadamard subpath with cuButterfly.

    A plan is cached per dtype, sub-transform size, flattened batch, device and
    caller stream.  Caching by stream avoids sharing one mutable C handle and
    workspace between concurrently submitted PyTorch streams.
    """

    def __init__(self, library: str | Path = DEFAULT_LIBRARY, device: str | torch.device | None = None,
                 compute_contract: str = "native", cache_path: str | Path | None = None,
                 reuse_buffers: bool = False):
        if compute_contract not in ("native", "fp32_composed", "public_fp32_final"):
            raise ValueError("compute_contract must be native, fp32_composed or public_fp32_final")
        self.compute_contract = compute_contract
        # Applications with serialized inference streams can opt into output
        # storage reuse. The default preserves the standalone adapter's
        # independent-output ownership contract.
        self.reuse_buffers = bool(reuse_buffers)
        self.cache_path = _cache_path(cache_path)
        self.registry_identity = _registry_identity()
        self._scales: dict[int, float] = {}
        if not torch.cuda.is_available():
            raise RuntimeError("QuaRot FWHT adapter requires CUDA")
        self.api = _Api(library)
        self.library = self.api.path
        self.device = torch.device(device) if device is not None else None
        if self.device is not None and self.device.type != "cuda":
            raise ValueError("QuaRot FWHT adapter device must be CUDA")
        if self.device is not None:
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            if self.device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index is unavailable: {self.device}")
        self._plans: dict[tuple, _Plan] = {}
        self.plan_create_ms = 0.0
        self.plan_create_events: list[dict[str, Any]] = []
        self._closed = False

    def _check_input(self, values: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("QuaRot FWHT adapter is closed")
        if not isinstance(values, torch.Tensor):
            raise TypeError("FWHT input must be a torch.Tensor")
        if not values.is_cuda:
            raise ValueError("FWHT input must be a CUDA tensor")
        if values.dtype not in (torch.float16, torch.float32):
            raise TypeError("FWHT input dtype must be torch.float16 or torch.float32")
        if values.ndim == 0 or values.shape[-1] <= 0:
            raise ValueError("FWHT input must have a non-empty last dimension")
        if self.device is not None and values.device != self.device:
            raise ValueError(f"FWHT input device {values.device} does not match {self.device}")

    def _make_plan(self, values: torch.Tensor, extent: int, batch: int, stream: int, scale: float = 1.0) -> _Plan:
        context = ctypes.c_void_p()
        descriptor = ctypes.c_void_p()
        plan = ctypes.c_void_p()
        created_context = False
        created_plan = False
        try:
            self.api.check(self.api.lib.cubutterflyCreate(ctypes.byref(context)), "create context")
            created_context = True
            if self.cache_path is not None:
                self.api.check(self.api.lib.cubutterflySetCachePath(context, self.cache_path.encode()), "set plan cache path")
            self.api.check(self.api.lib.cubutterflySetStream(context, stream), "set plan stream")
            self.api.check(
                self.api.lib.cubutterflyCreateDescriptor(ctypes.byref(descriptor)),
                "create descriptor",
            )
            storage = _FP16 if values.dtype == torch.float16 else _FP32
            compute = _COMPUTE_NATIVE if values.dtype == torch.float16 else _COMPUTE_FP32
            if getattr(self, "compute_contract", "native") == "public_fp32_final":
                if values.dtype != torch.float16:
                    raise TypeError("public_fp32_final requires FP16 storage")
                compute = _COMPUTE_FP32
            self.api.check(self.api.lib.cubutterflySetOperator(descriptor, _FWHT), "set FWHT operator")
            self.api.check(
                self.api.lib.cubutterflySetDataType(descriptor, storage, compute),
                "set FWHT dtype",
            )
            extents = (ctypes.c_size_t * 1)(extent)
            self.api.check(
                self.api.lib.cubutterflySetShape(descriptor, 1, extents, batch),
                "set FWHT shape",
            )
            self.api.check(
                self.api.lib.cubutterflySetDirection(descriptor, _FORWARD, 0),
                "set FWHT direction",
            )
            self.api.check(
                self.api.lib.cubutterflySetPlacement(descriptor, _OUT_OF_PLACE),
                "set FWHT placement",
            )
            self.api.check(
                self.api.lib.cubutterflySetLengthMode(descriptor, _STANDARD),
                "set standard FWHT length",
            )
            if getattr(self, "compute_contract", "native") == "public_fp32_final":
                self.api.check(self.api.lib.cubutterflySetOutputRounding(descriptor, 1), "set final-storage rounding")
                self.api.check(self.api.lib.cubutterflySetOutputScale(descriptor, scale), "set output scale")
            policy_name = os.environ.get("CUBUTTERFLY_E13_ALGORITHM_POLICY", "default").lower()
            if policy_name == "measure":
                policy, explicit = _MEASURE_POLICY, None
            elif policy_name == "default":
                policy, explicit = _DEFAULT_POLICY, None
            else:
                policy, explicit = _EXPLICIT_POLICY, policy_name.encode()
            self.api.check(
                self.api.lib.cubutterflySetAlgorithmPolicy(descriptor, policy, explicit),
                "set FWHT algorithm policy",
            )
            self.api.check(
                self.api.lib.cubutterflyCreatePlan(context, descriptor, ctypes.byref(plan)),
                "create FWHT plan",
            )
            created_plan = True
        except Exception:
            if created_plan:
                self.api.lib.cubutterflyDestroyPlan(plan)
            if created_context:
                self.api.lib.cubutterflyDestroy(context)
            raise
        finally:
            if descriptor and descriptor.value:
                self.api.lib.cubutterflyDestroyDescriptor(descriptor)

        workspace_bytes = ctypes.c_size_t()
        try:
            self.api.check(
                self.api.lib.cubutterflyPlanGetWorkspaceSize(plan, ctypes.byref(workspace_bytes)),
                "query FWHT workspace",
            )
            workspace = (
                torch.empty((int(workspace_bytes.value),), dtype=torch.uint8, device=values.device)
                if workspace_bytes.value
                else None
            )
            if workspace is not None:
                self.api.check(
                    self.api.lib.cubutterflyPlanSetWorkspace(
                        plan,
                        ctypes.c_void_p(workspace.data_ptr()),
                        workspace.numel(),
                    ),
                    "bind FWHT workspace",
                )
            source = ctypes.c_int()
            self.api.check(self.api.lib.cubutterflyPlanGetSelectionSource(plan, ctypes.byref(source)), "query selection source")
            def query(name):
                function = getattr(self.api.lib, "cubutterflyPlanGet" + name)
                size = ctypes.c_size_t()
                self.api.check(function(plan, None, ctypes.byref(size)), "query " + name)
                buffer = ctypes.create_string_buffer(size.value)
                self.api.check(function(plan, buffer, ctypes.byref(size)), "read " + name)
                return buffer.value.decode()
            metadata = {
                "selection_source": {0: "measured_table", 1: "user_cache", 2: "runtime_measured", 3: "static_model", 4: "explicit"}.get(source.value, str(source.value)),
                "selection_reason": query("SelectionReason"), "algorithm": query("AlgorithmName"),
                "mapping": json.loads(query("MappingJson")), "descriptor": json.loads(query("DescriptorJson")),
                "cache": _file_identity(self.cache_path) if self.cache_path else None,
                "registry": _registry_identity(),
            }
            return _Plan(self.api, context, plan, workspace, values.device, extent, batch, stream, metadata)
        except Exception:
            self.api.lib.cubutterflyDestroyPlan(plan)
            self.api.lib.cubutterflyDestroy(context)
            raise

    def _plan_for(self, values: torch.Tensor, extent: int, batch: int, scale: float = 1.0) -> _Plan:
        stream = _as_stream_id(torch.cuda.current_stream(values.device))
        device_index = int(values.device.index if values.device.index is not None else torch.cuda.current_device())
        key = (device_index, values.dtype, int(extent), int(batch), stream)
        if self.compute_contract == "public_fp32_final":
            key = key + (float(scale),)
        plan = self._plans.get(key)
        if plan is None:
            begin = time.perf_counter()
            plan = (self._make_plan(values, extent, batch, stream, scale)
                    if self.compute_contract == "public_fp32_final"
                    else self._make_plan(values, extent, batch, stream))
            # Plan creation may compile or prepare a CUDA module.  Synchronize
            # before accounting it so no setup work can leak into a later
            # application timing interval.
            torch.cuda.synchronize(values.device)
            elapsed_ms = (time.perf_counter() - begin) * 1000.0
            self.plan_create_ms += elapsed_ms
            self.plan_create_events.append({
                "dtype": str(values.dtype).replace("torch.", ""),
                "extent": int(extent),
                "batch": int(batch),
                "stream": int(stream),
                "elapsed_ms": elapsed_ms,
            })
            self._plans[key] = plan
        return plan

    def plan_statistics(self) -> dict[str, Any]:
        """Return setup provenance for application reports."""
        return {
            "compute_contract": self.compute_contract,
            "reuse_buffers": self.reuse_buffers,
            "algorithm_policy": os.environ.get("CUBUTTERFLY_E13_ALGORITHM_POLICY", "default"),
            "plan_create_ms": self.plan_create_ms,
            "plan_create_events": list(self.plan_create_events),
            "plans_created": len(self._plans),
            "cache_path": self.cache_path,
            "registry": self.registry_identity,
            "plans": [plan.metadata for plan in self._plans.values()],
        }

    def _transform_vectors(self, values: torch.Tensor, extent: int, full_n: int) -> torch.Tensor:
        contiguous = values if values.is_contiguous() else values.contiguous()
        if self.compute_contract == "fp32_composed" and values.dtype == torch.float16:
            # Match Dao's float accumulation and scale-before-narrowing contract.
            # Explicit composition: both casts and scaling remain in the call.
            contiguous = contiguous.float()
        flat = contiguous.reshape(-1, extent)
        if full_n not in self._scales:
            self._scales[full_n] = (float(1.0 / torch.tensor(full_n).sqrt())
                                    if self.compute_contract in ("fp32_composed", "public_fp32_final")
                                    else 1.0 / math.sqrt(full_n))
        scale = self._scales[full_n]
        plan = (self._plan_for(flat, extent, flat.shape[0], scale)
                if self.compute_contract == "public_fp32_final"
                else self._plan_for(flat, extent, flat.shape[0]))
        output = plan.execute(flat, reuse_output=self.reuse_buffers)
        if self.compute_contract == "public_fp32_final":
            return output.reshape(values.shape)
        # cuButterfly exposes the unnormalized FWHT.  QuaRot scales by the
        # full original dimension, including the K factor, before hadK @ input.
        if full_n not in self._scales:
            self._scales[full_n] = (float(1.0 / torch.tensor(full_n).sqrt())
                                    if self.compute_contract == "fp32_composed" else 1.0 / math.sqrt(full_n))
        scale = self._scales[full_n]
        output.mul_(scale)
        if self.compute_contract == "fp32_composed" and values.dtype == torch.float16:
            if self.reuse_buffers:
                if (plan.reusable_narrow is None or plan.reusable_narrow.shape != output.shape
                        or plan.reusable_narrow.device != output.device
                        or plan.reusable_narrow.dtype != values.dtype):
                    plan.reusable_narrow = torch.empty(
                        output.shape, dtype=values.dtype, device=output.device)
                plan.reusable_narrow.copy_(output)
                output = plan.reusable_narrow
            else:
                output = output.to(values.dtype)
        return output.reshape(contiguous.shape)

    def matmul_hadU_cuda(
        self,
        values: torch.Tensor,
        hadK: torch.Tensor | None,
        K: int,
    ) -> torch.Tensor:
        """Drop-in replacement for pinned ``quarot.functional`` call site."""

        self._check_input(values)
        if values.ndim < 1:
            raise ValueError("FWHT input must have at least one dimension")
        n = int(values.shape[-1])
        K = int(K)
        if K <= 0 or n % K:
            raise ValueError(f"QuaRot FWHT requires K={K} to divide n={n}")
        extent = n // K
        if not _is_power_of_two(extent):
            raise ValueError(
                f"QuaRot FWHT refuses non-power-of-two subtransform n/K={extent}; no padding is allowed"
            )

        if K == 1:
            return self._transform_vectors(values, extent, n)

        if hadK is None:
            raise ValueError(f"QuaRot FWHT K={K} requires a hadK matrix")
        if tuple(hadK.shape) != (K, K):
            raise ValueError(f"hadK shape {tuple(hadK.shape)} does not match K={K}")

        # Match pinned QuaRot's view -> contiguous FWHT -> hadK @ input order.
        reshaped = values.view(*values.shape[:-1], K, extent)
        transformed = self._transform_vectors(reshaped, extent, n)
        hadK_cast = hadK.to(device=values.device).to(dtype=values.dtype)
        return torch.matmul(hadK_cast, transformed).reshape(values.shape)

    def transform(self, values: torch.Tensor, hadK: torch.Tensor | None = None, K: int = 1) -> torch.Tensor:
        """Alias with an explicit adapter-oriented name."""

        return self.matmul_hadU_cuda(values, hadK, K)

    def close(self) -> None:
        if self._closed:
            return
        for plan in self._plans.values():
            plan.close()
        self._plans.clear()
        self._closed = True

    def __enter__(self) -> "QuaRotFWHTAdapter":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask an interpreter shutdown exception.
            pass


# Keep a spelling alias for callers that use the repository's ``Quarot`` style.
QuarotFWHTAdapter = QuaRotFWHTAdapter


def _functional_targets(functional_module: Any) -> list[Any]:
    """Find package and implementation aliases without importing new code."""

    targets: list[Any] = [functional_module]
    name = getattr(functional_module, "__name__", "")
    if name.endswith(".hadamard"):
        package_name = name.rsplit(".", 1)[0]
        package = __import__(package_name, fromlist=["*"])
        targets.append(package)
    else:
        implementation = getattr(functional_module, "hadamard", None)
        if implementation is not None:
            targets.append(implementation)
    unique: list[Any] = []
    for target in targets:
        if target not in unique and hasattr(target, "matmul_hadU_cuda"):
            unique.append(target)
    if not unique:
        raise AttributeError("QuaRot functional module has no matmul_hadU_cuda call site")
    return unique


class _QuaRotPatch:
    def __init__(
        self,
        library: str | Path = DEFAULT_LIBRARY,
        functional_module: Any | str | None = None,
        adapter: QuaRotFWHTAdapter | None = None,
    ):
        if isinstance(functional_module, str):
            functional_module = importlib.import_module(functional_module)
        if functional_module is None:
            functional_module = importlib.import_module("quarot.functional")
        self.functional_module = functional_module
        self.adapter = adapter or QuaRotFWHTAdapter(library)
        self._owns_adapter = adapter is None
        self._old: list[tuple[Any, Any]] = []
        self._installed = False

    def install(self) -> QuaRotFWHTAdapter:
        if self._installed:
            return self.adapter
        for target in _functional_targets(self.functional_module):
            old = target.matmul_hadU_cuda
            self._old.append((target, old))
            target.matmul_hadU_cuda = self.adapter.matmul_hadU_cuda
        self._installed = True
        return self.adapter

    def close(self) -> None:
        if self._installed:
            for target, old in reversed(self._old):
                target.matmul_hadU_cuda = old
            self._old.clear()
            self._installed = False
        if self._owns_adapter:
            self.adapter.close()

    def __enter__(self) -> QuaRotFWHTAdapter:
        return self.install()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def patched_quarot_hadamard(
    library: str | Path = DEFAULT_LIBRARY,
    functional_module: Any | str | None = None,
    adapter: QuaRotFWHTAdapter | None = None,
) -> _QuaRotPatch:
    """Return a context manager patching only QuaRot's FWHT boundary."""

    return _QuaRotPatch(library, functional_module, adapter)


def install_quarot_fwht_adapter(
    library: str | Path = DEFAULT_LIBRARY,
    functional_module: Any | str | None = None,
    adapter: QuaRotFWHTAdapter | None = None,
) -> _QuaRotPatch:
    """Install the bounded replacement; call ``close`` to restore originals."""

    patch = _QuaRotPatch(library, functional_module, adapter)
    patch.install()
    return patch


@contextmanager
def patched_quarot(
    library: str | Path = DEFAULT_LIBRARY,
    functional_module: Any | str | None = None,
    adapter: QuaRotFWHTAdapter | None = None,
) -> Iterator[QuaRotFWHTAdapter]:
    """Convenience context manager for application runners."""

    patch = patched_quarot_hadamard(library, functional_module, adapter)
    try:
        yield patch.install()
    finally:
        patch.close()


__all__ = [
    "DEFAULT_LIBRARY",
    "PINNED_QUAROT_COMMIT",
    "QuaRotFWHTAdapter",
    "QuarotFWHTAdapter",
    "install_quarot_fwht_adapter",
    "patched_quarot",
    "patched_quarot_hadamard",
]
