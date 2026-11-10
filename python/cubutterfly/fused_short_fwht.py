"""Standalone candidate provider, not yet a C++ planner/search processing unit.

FP16 storage -> FP32 warp arithmetic -> explicit float scale -> FP16 store.
Import is CPU-safe; preparation compiles only this module, without loading CUDA.
"""
import ctypes as ct
import hashlib
import math
from pathlib import Path
import subprocess

SOURCE = Path(__file__).with_suffix(".cu")


def prepare(output, nvcc="/usr/local/cuda-12.4/bin/nvcc", sm=80):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc, "-std=c++17", "-O3", "--shared", "--cudart=shared", "-Xcompiler=-fPIC",
               f"-arch=sm_{int(sm)}", str(SOURCE), "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return dict(provider="standalone-fused-short-fwht-v1", public_search_integrated=False,
                candidate_descriptor=descriptor(),
                source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                module=str(output), module_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                compile_command=command, supported_n=[32, 64], storage="fp16", accumulation="fp32",
                scale="explicit caller float; applied before single output narrowing", kernel_launches=1)


def descriptor():
    """Return the semantic contract consumed by candidate inventory tooling.

    The standalone provider is intentionally explicit about its current scope.
    Selectors may use this descriptor to attempt registration, while the
    public C++ path must fall back until a matching compiled provider is linked.
    """
    return {
        "operator": "fwht",
        "storage_precision": "fp16",
        "accumulation": "fp32",
        "direction": "forward",
        "normalization": "caller-scale",
        "placement": "out-of-place",
        "layout": "contiguous-last-axis",
        "logN": [5, 6],
        "exchange": "warp-register",
        "launches": 1,
        "provider_kind": "standalone-cuda-provider",
        "requires_linked_provider": True,
        "fallback": "portable-low-precision-fwht",
    }


class FusedShortFWHT:
    def __init__(self, module):
        self.module = Path(module).resolve()
        self.dll = ct.CDLL(str(self.module))
        self.launch = self.dll.cubutterfly_fused_short_fwht
        self.launch.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_uint64, ct.c_int, ct.c_float, ct.c_void_p]
        self.launch.restype = ct.c_int

    def __call__(self, values, scale, out=None):
        import torch
        if not isinstance(values, torch.Tensor) or not values.is_cuda or values.dtype != torch.float16:
            raise ValueError("candidate requires a CUDA FP16 tensor")
        if values.ndim < 1 or values.shape[-1] not in (32, 64) or not values.is_contiguous():
            raise ValueError("candidate requires contiguous last-axis N32 or N64")
        if not math.isfinite(float(scale)) or not values.numel():
            raise ValueError("scale must be finite and batch nonempty")
        out = torch.empty_like(values) if out is None else out
        if out.shape != values.shape or out.dtype != values.dtype or out.device != values.device or not out.is_contiguous():
            raise ValueError("output must match contiguous input shape, dtype and device")
        if out.data_ptr() == values.data_ptr():
            raise ValueError("candidate is out-of-place")
        with torch.cuda.device(values.device):
            stream = torch.cuda.current_stream(values.device)
            status = self.launch(values.data_ptr(), out.data_ptr(), values.numel() // values.shape[-1],
                                 values.shape[-1], float(scale), stream.cuda_stream)
        if status:
            raise RuntimeError(f"fused short FWHT launch CUDA status {status}")
        return out
