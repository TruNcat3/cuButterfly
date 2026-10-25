"""Exact packed FWHT core adapter; semantic boundary conversions belong to the runner.

Support checks inspect distribution metadata and files without importing torch,
the Python backend, its CUDA extension, or initializing a device.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
from importlib import import_module, metadata
import json
import math
from pathlib import Path
from urllib.parse import unquote, urlparse


PACKAGE = "fast-hadamard-transform"
AUDITED_VERSION = "1.1.0"
# Reviewed installed sources: float register/stage arithmetic for all storage
# types, final storage conversion, internal 8-point padding, and 32768 limit.
AUDITED_SOURCES = {
    "csrc/fast_hadamard_transform.cpp": "ffda3fdc2ab5ad1f8330d691e8feb6b133303502ca3229b1c9eb1679f0058dc1",
    "csrc/fast_hadamard_transform_cuda.cu": "aed649e84fe379d1adfdb061fc6f622f55612ddcd103f51f08ee984ff4dff34f",
    "csrc/fast_hadamard_transform_common.h": "e51345eb6be7b43cb657d73b8db9b2debcb4060de2266c7b42fc45bb3b86b473",
}
MAX_POINTS = 32768
STORAGE_NAMES = {0: "float16", 1: "bfloat16", 2: "float32"}


class UnsupportedBaseline(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["reason"])


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


@lru_cache(maxsize=1)
def backend_inventory():
    """Read installed artifacts once; no module import or CUDA query."""
    try:
        distribution = metadata.distribution(PACKAGE)
    except metadata.PackageNotFoundError:
        return {"available": False, "reason": "Dao package fast-hadamard-transform is not installed"}
    result = {"available": True, "package": PACKAGE, "version": distribution.version,
              "contract_verified": False, "python_files": [], "native_files": [], "sources": []}
    try:
        for entry in distribution.files or []:
            name = str(entry)
            if name.startswith("fast_hadamard_transform/") and name.endswith(".py"):
                result["python_files"].append(_fingerprint(distribution.locate_file(entry)))
            elif name.startswith("fast_hadamard_transform_cuda") and name.endswith(".so"):
                result["native_files"].append(_fingerprint(distribution.locate_file(entry)))
        if not result["python_files"] or len(result["native_files"]) != 1:
            result.update(available=False, reason="installed Dao Python/native artifacts are missing or ambiguous")
            return result
        direct_url = distribution.read_text("direct_url.json")
        if not direct_url:
            result["reason"] = "installed Dao source provenance is unavailable; arithmetic contract is unverified"
            return result
        source_url = urlparse(json.loads(direct_url)["url"])
        if source_url.scheme != "file" or source_url.netloc not in ("", "localhost"):
            result["reason"] = "installed Dao source provenance is not a local auditable source tree"
            return result
        source_root = Path(unquote(source_url.path))
        for relative in AUDITED_SOURCES:
            item = _fingerprint(source_root / relative)
            item["relative_path"] = relative
            result["sources"].append(item)
        result["contract_verified"] = (distribution.version == AUDITED_VERSION and all(
            item["sha256"] == AUDITED_SOURCES[item["relative_path"]] for item in result["sources"]))
        result["reason"] = ("audited installed source uses FP32 arithmetic with original storage output"
                            if result["contract_verified"] else
                            "installed Dao version/source hashes differ from the reviewed arithmetic contract")
    except (OSError, ValueError, KeyError) as error:
        result["reason"] = f"cannot verify installed Dao artifacts/source contract: {error}"
    return result


def _report(supported, status, reason, **details):
    return {"supported": supported, "status": status, "reason": reason, "metadata": details}


def support(s, *, inventory=None):
    """Check canonical semantics before tensors or device initialization exist."""
    if not isinstance(s, dict) or s.get("schema") != "cubutterfly-semantic-v1":
        return _report(False, "unsupported-semantic", "canonical cubutterfly-semantic-v1 record required")
    if type(s.get("operator")) is not int or s["operator"] != 2:
        return _report(False, "unsupported-semantic", "Dao baseline supports the FWHT operator only")
    if type(s.get("length_mode")) is not int or s["length_mode"] != 0:
        return _report(False, "unsupported-semantic", "Dao baseline does not support zero-extended embedding")
    if type(s.get("storage")) is not int or s["storage"] not in STORAGE_NAMES:
        return _report(False, "unsupported-semantic", "Dao storage must be FP32, FP16, or BF16; no precision conversion")
    if type(s.get("compute")) is not int or s["compute"] != 1:
        return _report(False, "unsupported-semantic", "Dao arithmetic is FP32; native low-precision or other compute requests cannot be substituted")
    shape = s.get("shape")
    if not isinstance(shape, list) or not shape or any(type(n) is not int or n <= 0 or n & (n - 1) for n in shape):
        return _report(False, "unsupported-semantic", "every logical FWHT axis must have a positive power-of-two extent")
    points = math.prod(shape)
    if points > MAX_POINTS:
        return _report(False, "unsupported-semantic", f"flattened FWHT length {points} exceeds audited Dao limit {MAX_POINTS}; no batch/shape reduction")
    if type(s.get("batch")) is not int or not 1 <= s["batch"] <= (1 << 31) - 1:
        return _report(False, "unsupported-semantic", "full batch must fit Dao's positive signed 32-bit batch dimension")
    if type(s.get("inverse")) is not int or s["inverse"] not in (0, 1) or type(s.get("normalize_inverse")) is not bool:
        return _report(False, "unsupported-semantic", "canonical inverse enum and boolean normalize_inverse required")
    details = {"logical_shape": list(shape), "batch": s["batch"], "flattened_points": points,
               "storage": s["storage"], "compute": s["compute"],
               "scale": 1.0 / points if s["inverse"] and s["normalize_inverse"] else 1.0,
               "backend_internal_points": max(8, points)}
    inventory = backend_inventory() if inventory is None else inventory
    details["backend_identity"] = copy.deepcopy(inventory)
    if not inventory.get("available"):
        return _report(False, "missing-external-dependency", inventory.get("reason", "Dao backend unavailable"), **details)
    if not inventory.get("contract_verified"):
        return _report(False, "unverified-external-contract", inventory.get("reason", "Dao arithmetic contract unverified"), **details)
    return _report(True, "supported", "separable per-axis FWHT is the row-major flattened Hadamard tensor product", **details)


class _Core:
    def __init__(self, function, source, output, report):
        self._function = function
        self._source = source
        self._output = output
        self._details = copy.deepcopy(report["metadata"])
        operations = ["backend_output_allocation", "native_fp32_fwht_and_fused_output_scale",
                      "copy_backend_result_to_supplied_output"]
        if self._details["flattened_points"] < 8:
            operations.insert(0, "backend_zero_pad_to_8_and_output_crop")
        self._details.update(
            implementation="Dao-AILab fast-hadamard-transform",
            backend_kind="native-with-output-copy", logical_core_layout="packed-natural",
            included_operations=operations,
            timing_scope="Complete core call including backend allocations, internal padding/crop, native transform and output copy; common runner adds order/layout/placement boundaries",
            arithmetic_contract="FP32 stage arithmetic; final conversion to original requested storage",
            no_precision_conversion=True,
        )

    def __call__(self):
        if self._function is None:
            raise RuntimeError("Dao baseline core is closed")
        result = self._function(self._source, scale=self._details["scale"])
        if result.dtype != self._output.dtype or result.device != self._output.device or result.shape != self._output.shape:
            raise RuntimeError("Dao returned a different dtype, device, or full-batch shape")
        self._output.copy_(result)

    def metadata(self):
        return copy.deepcopy(self._details)

    def close(self):
        self._function = self._source = self._output = None


def build_core(torch, s, source, output):
    """Bind packed CUDA tensors without evaluating the backend on any input."""
    report = support(s)
    if not report["supported"]:
        raise UnsupportedBaseline(report)
    dtype = getattr(torch, STORAGE_NAMES[s["storage"]])
    elements = s["batch"] * report["metadata"]["flattened_points"]
    for name, tensor in (("source", source), ("output", output)):
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype differs from requested storage; no promotion allowed")
        if not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() != elements:
            raise ValueError(f"{name} must be a packed CUDA tensor containing the complete logical batch")
    if source.device != output.device:
        raise ValueError("source and output must be on the same device")
    backend = import_module("fast_hadamard_transform")
    if backend.__version__ != report["metadata"]["backend_identity"]["version"]:
        raise RuntimeError("imported Dao version differs from the inspected installation")
    inspected_python = {item["path"] for item in report["metadata"]["backend_identity"]["python_files"]}
    if str(Path(backend.__file__).resolve()) not in inspected_python:
        raise RuntimeError("imported Dao Python module differs from inspected installation")
    shape = (s["batch"], report["metadata"]["flattened_points"])
    return _Core(backend.hadamard_transform, source.view(*shape), output.view(*shape), report)
