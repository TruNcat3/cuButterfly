"""Native same-precision FFTW core for packed, natural-order CPU inputs."""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import math
from pathlib import Path
import threading
import time

import numpy as np


_PLANNER_LOCK = threading.Lock()
_THREAD_INITIALIZED = set()
_INT_MAX = (1 << 31) - 1


def _identity(lib, symbol, version_symbol):
    # Resolve the loaded object, rather than fingerprinting a guessed soname.
    dl = ctypes.CDLL(ctypes.util.find_library("dl") or None)
    class DlInfo(ctypes.Structure):
        _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]
    dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dl.dladdr.restype = ctypes.c_int
    info = DlInfo()
    if not dl.dladdr(ctypes.cast(getattr(lib, symbol), ctypes.c_void_p), ctypes.byref(info)):
        raise RuntimeError("cannot resolve loaded FFTW artifact")
    path = Path(info.filename.decode()).resolve()
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if version_symbol:
        first = ctypes.c_char.in_dll(lib, version_symbol)
        result["version"] = ctypes.string_at(ctypes.addressof(first)).decode()
    return result


def _validate(s, source, threads):
    if not isinstance(s, dict):
        raise ValueError("FFTW requires a semantic dictionary")
    for name, allowed in (("operator", (0,)), ("fft_type", (0,)),
                          ("storage", (6, 7)), ("length_mode", (0,))):
        if type(s.get(name)) is not int or s[name] not in allowed:
            raise ValueError(f"FFTW unsupported {name}: {s.get(name)!r}")
    compute = 1 if s["storage"] == 6 else 2
    if type(s.get("compute")) is not int or s["compute"] not in (0, compute):
        raise ValueError("FFTW compute must be native or match original storage precision")
    shape = s.get("shape")
    if (not isinstance(shape, (list, tuple)) or not 1 <= len(shape) <= 8
            or any(type(n) is not int or not 1 <= n <= _INT_MAX for n in shape)):
        raise ValueError("FFTW requires rank 1..8 and positive signed-int extents")
    if math.prod(shape) > _INT_MAX:
        raise ValueError("FFTW full transform distance exceeds plan_many signed-int limit")
    if type(s.get("batch")) is not int or not 1 <= s["batch"] <= _INT_MAX:
        raise ValueError("FFTW requires a positive signed-int full batch")
    packed = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
    if type(s.get("placement", 0)) is not int or s.get("placement", 0) != 0:
        raise ValueError("FFTW core requires out-of-place placement")
    for side in ("input", "output"):
        orders = s.get(side + "_orders", [0] * len(shape))
        strides = s.get(side + "_strides", packed)
        distance = s.get(side + "_batch_stride", math.prod(shape))
        if (not isinstance(orders, (list, tuple)) or len(orders) != len(shape)
                or any(type(order) is not int or order != 0 for order in orders)):
            raise ValueError("FFTW core requires natural input/output orders")
        if (not isinstance(strides, (list, tuple)) or list(strides) != packed
                or any(type(stride) is not int for stride in strides)
                or type(distance) is not int or distance != math.prod(shape)):
            raise ValueError("FFTW core requires packed input/output strides and batch distances")
    if type(s.get("inverse")) is not int or s["inverse"] not in (0, 1):
        raise ValueError("FFTW requires forward/inverse enum 0/1")
    if type(s.get("normalize_inverse")) is not bool:
        raise ValueError("FFTW requires boolean normalize_inverse")
    scale = s.get("fft_output_scale")
    if type(scale) not in (int, float) or not math.isfinite(scale):
        raise ValueError("FFTW requires finite fft_output_scale")
    if type(threads) is not int or threads not in (1, 8):
        raise ValueError("FFTW thread policy supports 1 or 8 threads")
    dtype = np.dtype(np.complex64 if s["storage"] == 6 else np.complex128)
    if (not isinstance(source, np.ndarray) or source.dtype != dtype
            or source.shape != (s["batch"], *shape) or not source.flags.c_contiguous):
        raise ValueError("FFTW source must be contiguous (batch, *shape) in original complex precision")
    return tuple(shape), dtype


class FFTWCore:
    """Plan once; execute full-batch C2C FFT plus same-precision scaling.

    The descriptor and core input/output must use packed natural order,
    out of place. The input is snapshotted at
    construction and is never overwritten by FFTW's measuring planner.
    """

    def __init__(self, s, source, threads=1):
        shape, dtype = _validate(s, source, threads)
        self._plan = None
        self._allocations = []
        self._shape = shape
        self._dtype = dtype
        self._normalization = 1.0 / math.prod(shape) if s["inverse"] and s["normalize_inverse"] else 1.0
        self._scale = s["fft_output_scale"]
        prefix = "fftwf" if dtype == np.dtype(np.complex64) else "fftw"
        stem = "fftw3f" if prefix == "fftwf" else "fftw3"
        try:
            self._lib = ctypes.CDLL(ctypes.util.find_library(stem) or f"lib{stem}.so.3", mode=ctypes.RTLD_GLOBAL)
            threadlib = ctypes.CDLL(ctypes.util.find_library(stem + "_threads") or f"lib{stem}_threads.so.3")
        except OSError as error:
            raise RuntimeError(f"FFTW runtime unavailable: {error}") from error
        self._threadlib = threadlib
        malloc = getattr(self._lib, prefix + "_malloc")
        malloc.argtypes, malloc.restype = [ctypes.c_size_t], ctypes.c_void_p
        self._free = getattr(self._lib, prefix + "_free")
        self._free.argtypes, self._free.restype = [ctypes.c_void_p], None
        self._destroy = getattr(self._lib, prefix + "_destroy_plan")
        self._destroy.argtypes, self._destroy.restype = [ctypes.c_void_p], None
        self._execute = getattr(self._lib, prefix + "_execute")
        self._execute.argtypes, self._execute.restype = [ctypes.c_void_p], None
        plan_many = getattr(self._lib, prefix + "_plan_many_dft")
        ip = ctypes.POINTER(ctypes.c_int)
        plan_many.argtypes = [ctypes.c_int, ip, ctypes.c_int, ctypes.c_void_p, ip,
                              ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ip,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        plan_many.restype = ctypes.c_void_p
        init = getattr(threadlib, prefix + "_init_threads")
        init.argtypes, init.restype = [], ctypes.c_int
        plan_threads = getattr(threadlib, prefix + "_plan_with_nthreads")
        plan_threads.argtypes, plan_threads.restype = [ctypes.c_int], None
        try:
            views = []
            scalar = ctypes.c_float if prefix == "fftwf" else ctypes.c_double
            for _ in range(2):
                address = malloc(source.nbytes)
                if not address:
                    raise MemoryError("FFTW aligned allocation failed")
                self._allocations.append(address)
                views.append(np.ctypeslib.as_array((scalar * (source.size * 2)).from_address(address)).view(dtype).reshape(source.shape))
            self._input, self.output = views
            self._input.fill(0)
            self.output.fill(0)
            dims = (ctypes.c_int * len(shape))(*shape)
            begin = time.perf_counter_ns()
            with _PLANNER_LOCK:
                if stem not in _THREAD_INITIALIZED:
                    if not init():
                        raise RuntimeError("FFTW thread initialization failed")
                    _THREAD_INITIALIZED.add(stem)
                plan_threads(threads)
                self._plan = plan_many(len(shape), dims, s["batch"], self._allocations[0],
                                       dims, 1, math.prod(shape), self._allocations[1],
                                       dims, 1, math.prod(shape), 1 if s["inverse"] else -1,
                                       0)  # FFTW_MEASURE; planner may overwrite both buffers.
            planning_ms = (time.perf_counter_ns() - begin) / 1e6
            if not self._plan:
                raise RuntimeError("FFTW MEASURE plan creation failed")
            np.copyto(self._input, source)
            self.output.fill(0)
            self._metadata = dict(
                library=_identity(self._lib, prefix + "_execute", prefix + "_version"),
                threads_library=_identity(threadlib, prefix + "_init_threads", None),
                threads=threads, planner="FFTW_MEASURE", planning_ms=planning_ms,
                planning_excluded=True, logical_shape=list(shape), batch=s["batch"],
                storage=s["storage"], compute=s["compute"],
                arithmetic="FP32" if prefix == "fftwf" else "FP64",
                logical_core_layout="packed-natural", placement="out-of-place",
                normalization_factor=self._normalization, fft_output_scale=self._scale,
                source_unchanged=True, input_snapshot=True,
                timing_scope="FFTW execute full batch plus same-precision inverse normalization and fft_output_scale; allocation, input snapshot, and MEASURE planning excluded",
                included_operations=["fftw_execute", "inverse_normalization_if_requested", "fft_output_scale_if_nonunit"],
            )
        except BaseException:
            self.close()
            raise

    def execute(self):
        if not self._plan:
            raise RuntimeError("FFTW core is closed")
        self._execute(self._plan)
        real_type = np.float32 if self._dtype == np.dtype(np.complex64) else np.float64
        if self._normalization != 1:
            np.multiply(self.output, real_type(self._normalization), out=self.output)
        if self._scale != 1:
            np.multiply(self.output, real_type(self._scale), out=self.output)

    def metadata(self):
        import copy
        return copy.deepcopy(self._metadata)

    def close(self):
        if self._plan:
            with _PLANNER_LOCK:
                self._destroy(self._plan)
            self._plan = None
        for address in self._allocations:
            self._free(address)
        self._allocations.clear()

    def __del__(self):
        if hasattr(self, "_allocations"):
            self.close()
