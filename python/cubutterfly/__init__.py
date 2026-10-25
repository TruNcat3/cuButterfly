"""Public ctypes frontend. Importing this package loads no CUDA or tensor library.

Load a built library with Library(path) or CUBUTTERFLY_LIBRARY. Plan workspace
is externally allocated: bind it with set_workspace(pointer, bytes). Inputs,
outputs, workspace, and the stream must remain alive until execution completes.
"""
import ctypes as ct
import json as _json
import os
import operator
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


def _plan_cache_path(value=None):
    value = os.environ.get("CUBUTTERFLY_PLAN_CACHE") if value is None else value
    if value is None:
        return None
    if not os.fspath(value):
        raise ValueError("plan cache path must be nonempty")
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_file():
        raise ValueError("plan cache path must be a regular file")
    if not path.parent.is_dir() or not os.access(path.parent, os.W_OK):
        raise ValueError("plan cache parent must exist and be writable")
    if path.exists() and not os.access(path, os.R_OK | os.W_OK):
        raise ValueError("plan cache must be readable and writable")
    return str(path)


class Operator(IntEnum):
    FFT = 0
    NTT = 1
    FWHT = 2
    SUBSET_ZETA = 3
    SUPERSET_ZETA = 4
    STRUCTURED_2X2 = 5


class DataType(IntEnum):
    FP16 = 0
    BF16 = 1
    FP32 = 2
    FP64 = 3
    COMPLEX_FP16 = 4
    COMPLEX_BF16 = 5
    COMPLEX_FP32 = 6
    COMPLEX_FP64 = 7
    UINT32 = 8
    UINT64 = 9


class ComputeType(IntEnum):
    NATIVE = 0
    FP32 = 1
    FP64 = 2
    UINT32 = 3
    UINT64 = 4


class LogicalOrder(IntEnum):
    NATURAL = 0
    BIT_REVERSED = 1


class NttLayout(IntEnum):
    NATURAL = 0
    APPT_STATIC = 1
    BIT_REVERSED = 2


class ExecutionMode(IntEnum):
    AUTO = 0
    XD = 1
    COMPOSITE = 2


class AlgorithmPolicy(IntEnum):
    DEFAULT = 0
    MEASURE = 1
    EXPLICIT = 2


class FftType(IntEnum):
    C2C = 0
    R2C = 1
    C2R = 2


class LengthMode(IntEnum):
    STANDARD = 0
    ZERO_EXTENDED_EMBEDDING = 1


class OutputRounding(IntEnum):
    STAGE_STORAGE = 0
    FINAL_STORAGE = 1


@dataclass
class Transform:
    op: int = Operator.FFT
    storage: int = DataType.COMPLEX_FP32
    compute: int = ComputeType.NATIVE
    extents: list = field(default_factory=lambda: [256])
    batch: int = 1
    inverse: bool = False
    normalize_inverse: bool = True
    in_place: bool = False
    fft_type: int = FftType.C2C
    fft_output_scale: float = 1.0
    output_scale: float = 1.0
    output_rounding: int = OutputRounding.STAGE_STORAGE
    length_mode: int = LengthMode.STANDARD
    input_strides: list = field(default_factory=list)
    output_strides: list = field(default_factory=list)
    input_batch_stride: int = 0
    output_batch_stride: int = 0
    modulus: int = 1152921504606584833
    word_bits: int = 64
    stage_matrices: list = field(default_factory=list)
    input_orders: list = field(default_factory=list)
    output_orders: list = field(default_factory=list)
    input_order: int = NttLayout.NATURAL
    output_order: int = NttLayout.NATURAL
    execution_mode: int = ExecutionMode.AUTO


@dataclass
class PlanOptions:
    policy: int = AlgorithmPolicy.DEFAULT
    mapping: str = ""
    execution_mode: int = ExecutionMode.AUTO


class Error(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class _Matrix(ct.Structure):
    _fields_ = [(name, ct.c_double) for name in ("m00", "m01", "m10", "m11")]


class Library:
    def __init__(self, path=None):
        path = path or os.environ.get("CUBUTTERFLY_LIBRARY")
        if not path:
            raise ValueError("provide a library path or set CUBUTTERFLY_LIBRARY")
        self._dll = ct.CDLL(os.fspath(path))
        p, n, i = ct.c_void_p, ct.c_size_t, ct.c_int
        pn, pi = ct.POINTER(n), ct.POINTER(i)
        signatures = {
            "GetStatusString": [i], "GetLastErrorString": [],
            "CreateDescriptor": [ct.POINTER(p)], "DestroyDescriptor": [p],
            "SetOperator": [p, i], "SetDataType": [p, i, i],
            "SetShape": [p, ct.c_uint32, pn, n], "SetExecutionMode": [p, i],
            "SetFftTransformType": [p, i], "SetFftOutputScale": [p, ct.c_double],
            "SetOutputScale": [p, ct.c_double],
            "SetOutputRounding": [p, i],
            "SetStrides": [p, pn, n, pn, n], "SetLogicalOrders": [p, pi, pi],
            "SetDirection": [p, i, i], "SetPlacement": [p, i], "SetLengthMode": [p, i],
            "SetAlgorithmPolicy": [p, i, ct.c_char_p],
            "SetModulus": [p, ct.c_uint64, ct.c_uint32], "SetNttLayouts": [p, i, i],
            "SetStageMatrices": [p, ct.c_uint32, ct.POINTER(_Matrix), n],
            "DescriptorValidate": [p], "DescriptorSetJson": [p, ct.c_char_p],
            "Create": [ct.POINTER(p)], "Destroy": [p], "SetStream": [p, p],
            "SetCachePath": [p, ct.c_char_p],
            "CreatePlan": [p, p, ct.POINTER(p)], "DestroyPlan": [p],
            "PlanSetWorkspace": [p, p, n], "Execute": [p, p, p, p],
            "ExecuteSteady": [p, p, p, p],
            "PlanGetSelectionSource": [p, pi],
        }
        for name in ("DescriptorGetJson", "DescriptorGetCapabilities", "PlanGetMappingJson",
                     "PlanGetDescriptorJson", "PlanGetAlgorithmName", "PlanGetSelectionReason"):
            signatures[name] = [p, ct.c_void_p, pn]
        for name in ("PlanGetWorkspaceSize", "PlanGetInputSize", "PlanGetOutputSize"):
            signatures[name] = [p, pn]
        self._functions = {}
        for name, args in signatures.items():
            function = getattr(self._dll, "cubutterfly" + name, None)
            if function is None and name in ("SetOutputScale", "SetOutputRounding", "ExecuteSteady"):
                continue
            if function is None:
                raise AttributeError("library is missing cubutterfly" + name)
            function.argtypes = args
            function.restype = ct.c_char_p if name in ("GetStatusString", "GetLastErrorString") else i
            # ctypes attribute lookup is surprisingly visible for small,
            # repeatedly launched kernels.  Keep the ABI-bound callable once
            # so the public execution path does not resolve the symbol for
            # every transform.
            self._functions[name] = function

    def _call(self, name, *args):
        status = self._functions[name](*args)
        if status:
            detail = self._dll.cubutterflyGetLastErrorString() or b""
            detail = detail.decode("utf-8", errors="replace")
            summary = self._dll.cubutterflyGetStatusString(status).decode("utf-8", errors="replace")
            raise Error(summary + (": " + detail if detail else ""), status)

    def _query(self, name, handle):
        size = ct.c_size_t()
        self._call(name, handle, None, ct.byref(size))
        buffer = ct.create_string_buffer(size.value)
        self._call(name, handle, buffer, ct.byref(size))
        return buffer.value.decode("utf-8")


def _array(kind, values):
    return (kind * len(values))(*values) if values else None


def _integer(value, name, kind):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    bits = ct.sizeof(kind) * 8
    maximum = (1 << (bits - 1)) - 1 if kind is ct.c_int else (1 << bits) - 1
    if result < 0 or result > maximum:
        raise ValueError(f"{name} is outside the {kind.__name__} ABI range")
    return result


def _pointer(value):
    if hasattr(value, "data_ptr"):
        value = value.data_ptr()
    return value if isinstance(value, ct.c_void_p) else ct.c_void_p(value)


class Descriptor:
    def __init__(self, library, transform, options=None):
        self.library = library
        self._handle = ct.c_void_p()
        library._call("CreateDescriptor", ct.byref(self._handle))
        try:
            self._configure(transform, options or PlanOptions())
        except BaseException:
            self.close()
            raise

    def _configure(self, t, options):
        for name in ("inverse", "normalize_inverse", "in_place"):
            if not isinstance(getattr(t, name), bool):
                raise TypeError(f"{name} must be bool")
        enum_names = ("op", "storage", "compute", "fft_type", "length_mode", "input_order",
                      "output_order", "execution_mode", "output_rounding")
        enums = {name: _integer(getattr(t, name), name, ct.c_int) for name in enum_names}
        policy = _integer(options.policy, "policy", ct.c_int)
        mode = _integer(options.execution_mode, "options.execution_mode", ct.c_int)
        extents = [_integer(value, "extents", ct.c_size_t) for value in t.extents]
        strides_in = [_integer(value, "input_strides", ct.c_size_t) for value in t.input_strides]
        strides_out = [_integer(value, "output_strides", ct.c_size_t) for value in t.output_strides]
        orders_in = [_integer(value, "input_orders", ct.c_int) for value in t.input_orders]
        orders_out = [_integer(value, "output_orders", ct.c_int) for value in t.output_orders]
        batch = _integer(t.batch, "batch", ct.c_size_t)
        batch_in = _integer(t.input_batch_stride, "input_batch_stride", ct.c_size_t)
        batch_out = _integer(t.output_batch_stride, "output_batch_stride", ct.c_size_t)
        modulus = _integer(t.modulus, "modulus", ct.c_uint64)
        word_bits = _integer(t.word_bits, "word_bits", ct.c_uint32)
        rank = len(t.extents)
        _integer(rank, "rank", ct.c_uint32)
        for values in (t.input_strides, t.output_strides, t.input_orders, t.output_orders):
            if values and len(values) != rank:
                raise ValueError("one entry is required per axis")
        legacy = enums["input_order"] != NttLayout.NATURAL or enums["output_order"] != NttLayout.NATURAL
        if legacy and enums["op"] != Operator.NTT:
            raise ValueError("legacy NTT layouts require an NTT operator")
        if enums["op"] != Operator.NTT and (modulus != 1152921504606584833 or word_bits != 64):
            raise ValueError("modulus and word_bits require an NTT operator")
        if legacy and (t.input_orders or t.output_orders):
            raise ValueError("legacy NTT layouts and logical orders are ambiguous")
        call = lambda name, *args: self.library._call(name, self._handle, *args)
        call("SetOperator", enums["op"])
        call("SetDataType", enums["storage"], enums["compute"])
        call("SetShape", rank, _array(ct.c_size_t, extents), batch)
        call("SetExecutionMode", mode or enums["execution_mode"])
        call("SetFftTransformType", enums["fft_type"])
        call("SetFftOutputScale", t.fft_output_scale)
        for name, value, default in (("SetOutputScale", t.output_scale, 1.0),
                                     ("SetOutputRounding", enums["output_rounding"], 0)):
            if getattr(self.library._dll, "cubutterfly" + name, None) is not None:
                call(name, value)
            elif value != default:
                raise Error("loaded library does not support explicit output epilogues", 3)
        call("SetStrides", _array(ct.c_size_t, strides_in), batch_in,
             _array(ct.c_size_t, strides_out), batch_out)
        call("SetDirection", int(t.inverse), int(t.normalize_inverse))
        call("SetPlacement", int(t.in_place))
        call("SetLengthMode", enums["length_mode"])
        call("SetAlgorithmPolicy", AlgorithmPolicy.EXPLICIT if options.mapping else policy,
             options.mapping.encode("utf-8"))
        if enums["op"] == Operator.NTT:
            call("SetModulus", modulus, word_bits)
            call("SetNttLayouts", enums["input_order"], enums["output_order"])
        if not legacy:
            call("SetLogicalOrders", _array(ct.c_int, orders_in), _array(ct.c_int, orders_out))
        for axis, matrices in enumerate(t.stage_matrices):
            _integer(axis, "matrix axis", ct.c_uint32)
            if not matrices:
                continue
            values = [_Matrix(*matrix) for matrix in matrices]
            call("SetStageMatrices", axis, _array(_Matrix, values), len(values))

    @classmethod
    def from_json(cls, library, value):
        result = cls.__new__(cls)
        result.library, result._handle = library, ct.c_void_p()
        library._call("CreateDescriptor", ct.byref(result._handle))
        try:
            text = value if isinstance(value, str) else _json.dumps(value)
            library._call("DescriptorSetJson", result._handle, text.encode("utf-8"))
        except BaseException:
            result.close()
            raise
        return result

    @property
    def native_handle(self):
        if not self._handle.value:
            raise Error("descriptor is closed")
        return self._handle

    def validate(self):
        self.library._call("DescriptorValidate", self.native_handle)

    def json(self):
        return self.library._query("DescriptorGetJson", self.native_handle)

    def capabilities_json(self):
        return self.library._query("DescriptorGetCapabilities", self.native_handle)

    def close(self):
        if self._handle.value:
            handle, self._handle = self._handle, ct.c_void_p()
            self.library._call("DestroyDescriptor", handle)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Plan:
    """GPU plan with an owned context and externally allocated workspace.

    set_workspace retains its supplied object, but raw pointer allocation
    lifetime remains the caller's responsibility. execute is asynchronous.
    """
    def __init__(self, library, transform, options=None, stream=None, cache_path=None):
        self.library = library
        self.cache_path = _plan_cache_path(cache_path)
        self._context, self._handle = ct.c_void_p(), ct.c_void_p()
        self._workspace = None
        try:
            library._call("Create", ct.byref(self._context))
            if self.cache_path is not None:
                library._call("SetCachePath", self._context, self.cache_path.encode("utf-8"))
            if stream is not None:
                self.set_stream(stream)
            if isinstance(transform, Descriptor):
                if options is not None:
                    raise ValueError("selection options must be set when constructing the Descriptor")
                if transform.library is not library:
                    raise ValueError("descriptor and plan must use the same Library")
                library._call("CreatePlan", self._context, transform.native_handle, ct.byref(self._handle))
            else:
                with Descriptor(library, transform, options) as descriptor:
                    library._call("CreatePlan", self._context, descriptor.native_handle, ct.byref(self._handle))
        except BaseException:
            self.close()
            raise

    @property
    def native_handle(self):
        if not self._handle.value:
            raise Error("plan is closed")
        return self._handle

    def set_stream(self, stream):
        if self._handle.value:
            raise Error("stream is fixed when the plan is created; pass stream= to Plan(...)")
        if not self._context.value:
            raise Error("plan context is closed")
        self.library._call("SetStream", self._context, _pointer(stream))

    def _size(self, name):
        size = ct.c_size_t()
        self.library._call(name, self.native_handle, ct.byref(size))
        return size.value

    def workspace_size(self):
        return self._size("PlanGetWorkspaceSize")

    def input_size(self):
        return self._size("PlanGetInputSize")

    def output_size(self):
        return self._size("PlanGetOutputSize")

    def set_workspace(self, pointer, bytes):
        self.library._call("PlanSetWorkspace", self.native_handle, _pointer(pointer), bytes)
        self._workspace = pointer

    def execute(self, input, output):
        self.library._call("Execute", self._context, self.native_handle, _pointer(input), _pointer(output))

    def execute_steady(self, input, output):
        """Submit a reused plan with the low-overhead steady-state ABI."""
        if "ExecuteSteady" not in self.library._functions:
            self.execute(input, output)
            return
        self.library._call("ExecuteSteady", self._context, self.native_handle,
                           _pointer(input), _pointer(output))

    def mapping_json(self):
        return self.library._query("PlanGetMappingJson", self.native_handle)

    def descriptor_json(self):
        return self.library._query("PlanGetDescriptorJson", self.native_handle)

    def algorithm_name(self):
        return self.library._query("PlanGetAlgorithmName", self.native_handle)

    def selection_reason(self):
        return self.library._query("PlanGetSelectionReason", self.native_handle)

    def close(self):
        try:
            if self._handle.value:
                handle, self._handle = self._handle, ct.c_void_p()
                self.library._call("DestroyPlan", handle)
        finally:
            if self._context.value:
                context, self._context = self._context, ct.c_void_p()
                self.library._call("Destroy", context)
            self._workspace = None

    __enter__ = Descriptor.__enter__
    __exit__ = Descriptor.__exit__
    __del__ = Descriptor.__del__
