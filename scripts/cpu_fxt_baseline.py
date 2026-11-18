"""Auditable, independent FXT CPU adapter for packed FWHT and unsigned zeta."""
from __future__ import annotations

import ctypes
from functools import reduce
import hashlib
import json
import operator
from pathlib import Path
import shutil
import subprocess

import numpy as np

WRAPPER = Path(__file__).with_name("cpu_external_fxt.cpp")
SOURCE_URL = "https://www.jjj.de/fxt/fxt-2026.05.19.tar.gz"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_library(root, output):
    """Compile the standalone wrapper and record source/compiler provenance."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not (root / "src/walsh/arithtransform.h").is_file():
        raise ValueError("FXT source root must contain src/walsh/arithtransform.h")
    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("g++ is required to build the standalone FXT adapter")
    compiler = str(Path(compiler).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-O3", "-fPIC", "-shared", "-std=c++17", "-I",
               str(root / "src"), str(WRAPPER), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    hashes = {str(path.relative_to(root)): _hash(path)
              for path in sorted((root / "src").rglob("*.h"))}
    license_path = root / "COPYING.txt"
    if license_path.is_file():
        hashes["COPYING.txt"] = _hash(license_path)
    manifest = dict(backend="fxt-cpu", declared_version="2026.05.19",
                    declared_source_url=SOURCE_URL, source_root=str(root),
                    source_file_sha256=hashes,
                    source_tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
                    wrapper=str(WRAPPER.resolve()), wrapper_sha256=_hash(WRAPPER),
                    compiler=compiler, compiler_sha256=_hash(compiler),
                    compiler_version=subprocess.check_output([compiler, "--version"], text=True),
                    compile_command=command, library=str(output), library_sha256=_hash(output),
                    license="GPL-3.0-or-later; independent measurement tool",
                    threads=1, parallelism="serial FXT and serial adapter batch loop")
    Path(str(output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


class FXTCore:
    """One native call includes input copy, all batches and inverse FWHT scale."""

    def __init__(self, s, source, library, threads=1):
        if threads != 1:
            raise ValueError("FXT adapter supports threads=1 only")
        self.s = dict(s)
        shape = tuple(int(x) for x in s["shape"])
        if not shape or any(x <= 0 or x & (x - 1) for x in shape):
            raise ValueError("FXT requires every extent to be a power of two")
        self.n, self.batch = reduce(operator.mul, shape, 1), int(s["batch"])
        if self.batch <= 0 or self.n.bit_length() > ctypes.sizeof(ctypes.c_ulong) * 8:
            raise ValueError("invalid FXT batch or transform length")
        if s.get("length_mode", 0) != 0 or s.get("placement", 0) != 0:
            raise ValueError("FXT supports dyadic out-of-place semantics only")
        strides, stride = [0] * len(shape), 1
        for axis in reversed(range(len(shape))):
            strides[axis] = stride
            stride *= shape[axis]
        for side in ("input", "output"):
            if list(s.get(side + "_orders", [0] * len(shape))) != [0] * len(shape):
                raise ValueError("FXT supports natural order only")
            if list(s.get(side + "_strides", strides)) != strides or s.get(side + "_batch_stride", self.n) != self.n:
                raise ValueError("FXT supports packed layout only")
        op, storage, compute = s["operator"], s["storage"], s["compute"]
        valid = {(2, 2, 1): np.float32, (2, 3, 2): np.float64,
                 (3, 8, 3): np.uint32, (3, 9, 4): np.uint64,
                 (4, 8, 3): np.uint32, (4, 9, 4): np.uint64}
        dtype = valid.get((op, storage, compute))
        if dtype is None:
            raise ValueError("FXT has no exact operator/storage/compute implementation")
        if not isinstance(source, np.ndarray) or source.dtype != np.dtype(dtype):
            raise ValueError("FXT source dtype must exactly match storage; casting is not allowed")
        if not source.flags.c_contiguous or source.size != self.batch * self.n:
            raise ValueError("FXT source must be contiguous and match packed batch size")
        self.source, self.output = source, np.empty_like(source)
        path = Path(library).resolve()
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest["library_sha256"] != _hash(path):
            raise ValueError("FXT library hash does not match build manifest")
        self.lib = ctypes.CDLL(str(path))
        self.fn = self.lib.fxt_execute_batch
        self.fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                            ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
        self.fn.restype = ctypes.c_int
        self.args = (source.ctypes.data, self.output.ctypes.data, self.n.bit_length() - 1,
                     self.batch, storage, op, int(bool(s.get("inverse", False))),
                     int(bool(s.get("normalize_inverse", False))))
        self.closed = False
        self._metadata = dict(library=str(path), build_manifest=manifest,
                              build_manifest_path=str(manifest_path), threads=1,
                              planning_excluded=True,
                              timing_scope="One ctypes call: native input-to-output copy, all batch FXT transforms, and required inverse FWHT scaling",
                              native_algorithm="walsh_wak_loc_dif2" if op == 2 else
                              ("arith_transform_minus" if s.get("inverse") else "arith_transform_plus") if op == 3 else
                              ("rev_arith_transform_minus" if s.get("inverse") else "rev_arith_transform_plus"))

    def metadata(self):
        return dict(self._metadata)

    def execute(self):
        if self.closed:
            raise RuntimeError("FXTCore is closed")
        status = self.fn(*self.args)
        if status:
            raise RuntimeError(f"FXT native adapter rejected execution: {status}")
        return self.output

    def close(self):
        self.closed = True


class FP16FWHTAdapter:
    """Explicit FP16 I/O / FP32 FXT arithmetic comparator, not native FP16.

    Each execute includes both conversions, the complete transform, and any
    requested inverse normalization. Only setup buffers are allocated in advance.
    """

    def __init__(self, s, source, library, threads=1):
        if (s.get('operator'), s.get('storage'), s.get('compute')) != (2, 0, 0):
            raise ValueError('adapter requires FWHT with FP16 storage/native requested compute')
        if (not isinstance(source, np.ndarray) or source.dtype != np.float16 or
                source.shape != (s['batch'], *s['shape']) or not source.flags.c_contiguous):
            raise ValueError('adapter requires packed original FP16 input without substitution')
        native = dict(s, storage=2, compute=1)
        self.source = source
        self.output = np.empty_like(source)
        self._work = np.empty(source.shape, dtype=np.float32)
        self._core = FXTCore(native, self._work, library, threads=threads)
        self._metadata = dict(self._core.metadata(),
            backend='fxt-fp16-io-fp32-cpu', native_fp16=False,
            requested_storage='float16', external_core_storage='float32',
            precision_status='adapted-FP16-IO-FP32-arithmetic',
            conversion='FP16 input -> FP32 FXT -> FP16 output',
            rounding='FP32 CPU transform and inverse scaling, then final FP16 narrowing',
            all_conversion_and_scaling_timed=True,
            timing_scope='input widening, complete FP32 FXT including inverse scale, output narrowing; no initial host/device transfer',
            adapter_source_sha256=_hash(Path(__file__)))

    def metadata(self):
        return dict(self._metadata)

    def execute(self):
        np.copyto(self._work, self.source, casting='safe')
        self._core.execute()
        np.copyto(self.output, self._core.output, casting='unsafe')
        return self.output

    def close(self):
        self._core.close()
