"""Prepared, caller-stream GPU-NTT core; support checks are entirely CPU-only."""
from __future__ import annotations

import copy
import ctypes as ct
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIBRARY = ROOT / "build-gpuntt-semantic/libgpuntt_semantic_bridge_guarded_20261102.so"
SOURCE_REVISION = "d03c5eaeadaa780d153496afcb3b6a9b79a13a63"


class UnsupportedBaseline(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(report["reason"])


class BridgeError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


def _fingerprint(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


class Bridge:
    """C ABI setup binds functions only; validate calls the frozen CPU root helper."""
    def __init__(self, path):
        self.dll = ct.CDLL(str(Path(path).resolve(strict=True)))
        signatures = {
            "Validate": [ct.c_uint, ct.c_size_t, ct.c_uint, ct.c_uint64, ct.c_int, ct.POINTER(ct.c_uint64)],
            "Create": [ct.POINTER(ct.c_void_p), ct.c_uint, ct.c_size_t, ct.c_uint, ct.c_uint64,
                       ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_size_t)],
            "Execute": [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p],
            "Destroy": [ct.c_void_p],
        }
        for name, arguments in signatures.items():
            function = getattr(self.dll, "gpunttSemantic" + name)
            function.argtypes, function.restype = arguments, ct.c_int
        for name in ("SourceRevision", "LastError"):
            function = getattr(self.dll, "gpunttSemantic" + name)
            function.argtypes, function.restype = [], ct.c_char_p
        self.dll.gpunttSemanticAbiVersion.argtypes = []
        self.dll.gpunttSemanticAbiVersion.restype = ct.c_int
        if self.dll.gpunttSemanticAbiVersion() != 1:
            raise ValueError("GPU-NTT bridge ABI version must be 1")
        revision = self.dll.gpunttSemanticSourceRevision().decode("ascii")
        if revision != SOURCE_REVISION:
            raise ValueError("GPU-NTT bridge source revision differs from pinned vendor")

    def call(self, name, *args):
        status = getattr(self.dll, "gpunttSemantic" + name)(*args)
        if status:
            detail = self.dll.gpunttSemanticLastError().decode("utf-8", errors="replace")
            raise BridgeError(status, detail)

    def validate(self, s):
        root = ct.c_uint64()
        self.call("Validate", s["shape"][0].bit_length() - 1, s["batch"], s["word_bits"],
                  s["modulus"], s["inverse"], ct.byref(root))
        return root.value


def _identity(library_path):
    library = Path(library_path).resolve(strict=True)
    manifest_path = Path(str(library) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "cubutterfly-gpuntt-build-v1"
            or manifest.get("vendor_revision") != SOURCE_REVISION):
        raise ValueError("GPU-NTT build manifest schema or pinned revision differs")
    if Path(manifest["binary"]["path"]) != library:
        raise ValueError("GPU-NTT build manifest is bound to a different binary path")
    for record in [manifest["binary"], manifest["vendor_archive"],
                   manifest["root_helper_library"], *manifest["sources"]]:
        if _fingerprint(record["path"]) != record:
            raise ValueError(f"GPU-NTT build input/output fingerprint differs: {record['path']}")
    if not manifest["sources"]:
        raise ValueError("GPU-NTT build manifest has no source evidence")
    # Check on every inventory/support call, including after a file is replaced.
    return Bridge(library), {**manifest, "build_manifest": _fingerprint(manifest_path)}


def _report(supported, status, reason, **details):
    return {"supported": supported, "status": status, "reason": reason, "metadata": details}


def backend_inventory(librarypath=DEFAULT_LIBRARY):
    """Return the CPU-inspected native/library/source identity or a binding gap."""
    try:
        _, identity = _identity(str(Path(librarypath).resolve()))
        return {"available": True, **copy.deepcopy(identity)}
    except (OSError, AttributeError, ValueError, KeyError, TypeError) as error:
        return {"available": False, "status": "missing-external-binding",
                "reason": f"GPU-NTT native bridge unavailable or incompatible: {error}"}


def support(s, librarypath=DEFAULT_LIBRARY):
    if not isinstance(s, dict) or s.get("schema") != "cubutterfly-semantic-v1":
        return _report(False, "unsupported-semantic", "canonical cubutterfly-semantic-v1 record required")
    if type(s.get("operator")) is not int or s["operator"] != 1:
        return _report(False, "unsupported-semantic", "GPU-NTT bridge supports NTT only")
    if type(s.get("length_mode")) is not int or s["length_mode"] != 0:
        return _report(False, "unsupported-semantic", "GPU-NTT bridge supports standard cyclic NTT; embedding unsupported")
    shape = s.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        return _report(False, "unsupported-semantic", "GPU-NTT bridge supports rank one only; no flattened xD substitution")
    if type(shape[0]) is not int or shape[0] < 2 or shape[0] > (1 << 28) or shape[0] & (shape[0] - 1):
        return _report(False, "unsupported-semantic", "GPU-NTT extent must be power-of-two with logN in [1,28]")
    if type(s.get("batch")) is not int or not 1 <= s["batch"] <= 65535:
        return _report(False, "unsupported-semantic", "GPU-NTT complete batch must be in [1,65535]; no shrink")
    storage = s.get("storage")
    if type(storage) is not int or storage not in (8, 9):
        return _report(False, "unsupported-semantic", "GPU-NTT storage must be UINT32 or UINT64")
    bits, compute = (32, 3) if storage == 8 else (64, 4)
    if type(s.get("word_bits")) is not int or s["word_bits"] != bits or type(s.get("compute")) is not int or s["compute"] != compute:
        return _report(False, "unsupported-semantic", "GPU-NTT storage, word_bits and canonical compute must match exactly")
    q = s.get("modulus")
    # The pinned GPU-NTT Data64 Barrett path is exact only below 2^61.
    # Reject 62-bit moduli instead of publishing an invalid comparison.
    if type(q) is not int or not 2 < q < (1 << (30 if bits == 32 else 61)):
        return _report(False, "unsupported-semantic", "GPU-NTT modulus outside declared word arithmetic range")
    if type(s.get("inverse")) is not int or s["inverse"] not in (0, 1) or type(s.get("normalize_inverse")) is not bool:
        return _report(False, "unsupported-semantic", "canonical inverse enum and boolean normalize_inverse required")
    if s["normalize_inverse"] != bool(s["inverse"]):
        return _report(False, "unsupported-semantic", "public inverse NTT includes inverse-N; canonical normalization required")
    if any(s.get(key, 0) != 0 for key in ("ntt_input_layout", "ntt_output_layout")):
        return _report(False, "unsupported-semantic", "APPT static NTT layout is unsupported by the semantic bridge")
    try:
        bridge, identity = _identity(str(Path(librarypath).resolve()))
        root = bridge.validate(s)
    except BridgeError as error:
        return _report(False, "unsupported-semantic" if error.status == 1 else "external-binding-error", str(error))
    except (OSError, AttributeError, ValueError, KeyError, TypeError) as error:
        return _report(False, "missing-external-binding", f"GPU-NTT native bridge is unavailable or incompatible: {error}")
    return _report(True, "supported", "exact cyclic prime/root contract validated with the frozen cuButterfly helper",
                   logical_shape=list(shape), batch=s["batch"], modulus=q, word_bits=bits, root=root,
                   inverse=s["inverse"], normalize_inverse=s["normalize_inverse"],
                   backend_identity=copy.deepcopy(identity))


def _stream_pointer(stream):
    value = getattr(stream, "cuda_stream", stream)
    if type(value) is not int or value < 0:
        raise ValueError("stream must be a nonnegative CUDA stream pointer or torch stream")
    return ct.c_void_p(value)


class _Core:
    def __init__(self, bridge, s, source, output, stream, report):
        self.bridge, self.source, self.output = bridge, source, output
        self.stream = _stream_pointer(stream)
        self.handle = ct.c_void_p()
        self._details = copy.deepcopy(report["metadata"])
        root, workspace = ct.c_uint64(), ct.c_size_t()
        bridge.call("Create", ct.byref(self.handle), s["shape"][0].bit_length() - 1, s["batch"],
                    s["word_bits"], s["modulus"], s["inverse"], self.stream, ct.byref(root), ct.byref(workspace))
        if root.value != report["metadata"]["root"]:
            self.close()
            raise RuntimeError("prepared GPU-NTT root differs from CPU-validated public root")
        self._details.update(
            implementation="GPU-NTT Merge", backend_kind="native-with-order-adapter",
            logical_core_layout="packed-natural", workspace_bytes=workspace.value,
            included_operations=(["inverse_input_bit_reversal", "vendor_GPU_INTT_with_inverse_N"]
                                 if s["inverse"] else ["vendor_GPU_NTT", "forward_output_bit_reversal"]),
            timing_scope="Complete core call with order conversion and inverse normalization; common runner adds requested layout/order/placement boundaries",
            setup_excluded="root tables, modulus, inverse-N, scratch allocations and setup copies",
            execute_allocations=False, caller_stream=int(self.stream.value or 0),
        )

    def __call__(self):
        if not self.handle.value:
            raise RuntimeError("GPU-NTT baseline core is closed")
        self.bridge.call("Execute", self.handle, ct.c_void_p(self.source.data_ptr()),
                         ct.c_void_p(self.output.data_ptr()), self.stream)

    def close(self):
        if self.handle.value:
            self.bridge.call("Destroy", self.handle)
            self.handle = ct.c_void_p()
        self.source = self.output = None

    def metadata(self):
        return copy.deepcopy(self._details)


def build_core(torch, s, source, output, stream, librarypath=DEFAULT_LIBRARY):
    report = support(s, librarypath)
    if not report["supported"]:
        raise UnsupportedBaseline(report)
    dtype = getattr(torch, "uint32" if s["word_bits"] == 32 else "uint64", None)
    if dtype is None:
        raise ValueError("torch lacks the requested unsigned storage dtype; no signed/float substitution")
    elements = s["shape"][0] * s["batch"]
    for name, tensor in (("source", source), ("output", output)):
        if tensor.dtype != dtype or not tensor.is_cuda or not tensor.is_contiguous() or tensor.numel() != elements:
            raise ValueError(f"{name} must preserve unsigned dtype and contain the full packed CUDA batch")
    if source.device != output.device:
        raise ValueError("source and output must be on the same CUDA device")
    bridge, _ = _identity(str(Path(librarypath).resolve()))
    return _Core(bridge, s, source, output, stream, report)
