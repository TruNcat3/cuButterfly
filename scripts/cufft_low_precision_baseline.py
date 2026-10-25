"""Independent low-storage cuFFT Xt baseline; import/support/build never initialize CUDA."""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/cufft_low_precision"
DOCS = "https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html"
MAX_ELEMENTS = (1 << 32) - 1
PRECISIONS = {"fp16": 1, "bf16": 2}


def support(shape, batch, *, precision="fp16", compute="vendor-native", accumulation="vendor-native",
            direction="forward", normalization="none", placement="out-of-place",
            element_stride=1, batch_stride=None, input_order="natural", output_order="natural", sm=None):
    reasons = []
    shape = [shape] if type(shape) is int else list(shape) if isinstance(shape, (tuple, list)) else []
    dimensions_ok = len(shape) in (1, 2) and all(type(n) is int and 0 < n <= MAX_ELEMENTS and
                                               n & (n - 1) == 0 for n in shape)
    if not dimensions_ok:
        reasons.append("adapter requires one or two positive power-of-two dimensions below 2^32")
    volume = 1
    if dimensions_ok:
        for n in shape:
            volume *= n
    if type(batch) is not int or batch < 1 or not dimensions_ok or volume * batch > MAX_ELEMENTS:
        reasons.append("batch*transform-volume must be positive and below 2^32 complex elements")
    if precision not in PRECISIONS:
        reasons.append("only native FP16 or BF16 storage is supported")
    if compute != "vendor-native" or accumulation != "vendor-native":
        reasons.append("cuFFT Xt requires matching input/output/execution precision; arbitrary compute or accumulation is unavailable")
    if direction not in ("forward", "inverse"):
        reasons.append("direction must be forward or inverse")
    if normalization != "none":
        reasons.append("normalization is unsupported; inverse is unscaled")
    if (placement != "out-of-place" or type(element_stride) is not int or element_stride != 1 or
            batch_stride is not None and (type(batch_stride) is not int or batch_stride != volume) or
            input_order != "natural" or output_order != "natural"):
        reasons.append("adapter requires contiguous natural-order out-of-place C2C with no padding")
    minimum = 53 if precision == "fp16" else 80
    if sm is not None and (type(sm) is not int or sm < minimum):
        reasons.append(f"vendor requires SM{minimum} or newer for this precision")
    return dict(supported=not reasons, reasons=reasons, provider="cuFFT Xt low-storage", compute_family="fft-l",
        shape=shape, batch=batch, storage_precision=precision, compute_precision="vendor-native",
        accumulation="vendor-native; internal accumulator precision not independently selectable",
        arithmetic_contract=("vendor-native FP16; internal accumulator details unspecified" if precision == "fp16" else
                             "vendor-native mixture of BF16 and single-precision operations"),
        input_type="CUDA_C_16F" if precision == "fp16" else "CUDA_C_16BF" if precision == "bf16" else None,
        output_type="CUDA_C_16F" if precision == "fp16" else "CUDA_C_16BF" if precision == "bf16" else None,
        execution_type="CUDA_C_16F" if precision == "fp16" else "CUDA_C_16BF" if precision == "bf16" else None,
        direction=direction, normalization=normalization, placement=placement, input_order=input_order,
        output_order=output_order, element_stride=element_stride, batch_stride=volume if dimensions_ok else None,
        minimum_sm=minimum, buffer_bytes=volume * batch * 4 if not reasons else None,
        graph_status="capture-ready execution structure; GPU validation pending",
        capability_status="documented static domain; actual cuFFT planning/resource acceptance pending",
        timed_operations=["complete cufftXtExec"], normalization_supported=False,
        documentation=DOCS, gpu_execution_verified=False)


class Descriptor(ct.Structure):
    _fields_ = [("abi_version", ct.c_uint32), ("precision", ct.c_int32), ("rank", ct.c_int32),
                ("direction", ct.c_int32), ("dimensions", ct.c_uint64 * 2), ("batch", ct.c_uint64),
                ("normalization", ct.c_int32), ("reserved", ct.c_uint32)]


class Info(ct.Structure):
    _fields_ = [("total_elements", ct.c_uint64), ("workspace_bytes", ct.c_uint64),
                ("device", ct.c_int32), ("input_type", ct.c_int32), ("output_type", ct.c_int32),
                ("execution_type", ct.c_int32)]


def descriptor(shape, batch, **kwargs):
    capability = support(shape, batch, **kwargs)
    if not capability["supported"]:
        raise ValueError("; ".join(capability["reasons"]))
    dims = capability["shape"]
    return Descriptor(1, PRECISIONS[capability["storage_precision"]], len(dims),
                      -1 if capability["direction"] == "forward" else 1,
                      (ct.c_uint64 * 2)(*(dims + [0] * (2 - len(dims)))), batch, 0, 0)


def load(library):
    """Bind the ABI without calling any CUDA or cuFFT entry point."""
    lib = ct.CDLL(str(library))
    signatures = dict(cufft_low_validate=[ct.POINTER(Descriptor)],
        cufft_low_create=[ct.POINTER(Descriptor), ct.c_size_t, ct.POINTER(ct.c_void_p)],
        cufft_low_get_info=[ct.c_void_p, ct.POINTER(Info)],
        cufft_low_execute=[ct.c_void_p, ct.c_size_t, ct.c_size_t], cufft_low_destroy=[ct.c_void_p],
        cufft_low_abi_version=[], cufft_low_header_version=[], cufft_low_cuda_header_version=[])
    for name, signature in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = signature, ct.c_int
    lib.cufft_low_last_error.argtypes, lib.cufft_low_last_error.restype = [], ct.c_char_p
    if lib.cufft_low_abi_version() != 1:
        raise RuntimeError("unsupported low cuFFT bridge ABI")
    return lib


def address(value, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1) or value > (1 << (ct.sizeof(ct.c_void_p) * 8)) - 1:
        raise ValueError("expected an explicit CUDA pointer/stream address fitting uintptr_t")
    return value


class Plan:
    """Bind one explicit stream at creation; caller owns buffers and active device.

    Create/warm up before timing or capture. Keep this plan and its stream alive
    for captured graph lifetime. Synchronize before close; serialize plan uses.
    """
    def __init__(self, library, shape, batch, *, stream, **kwargs):
        request = descriptor(shape, batch, **kwargs)
        self.capability, self.stream = support(shape, batch, **kwargs), address(stream, zero=True)
        self.lib, self.handle = load(library), ct.c_void_p()
        self._check(self.lib.cufft_low_create(ct.byref(request), self.stream, ct.byref(self.handle)))
        info = Info()
        try:
            self._check(self.lib.cufft_low_get_info(self.handle, ct.byref(info)))
        except Exception:
            self.close()
            raise
        self.info = {name: getattr(info, name) for name, _ in Info._fields_}
        self.info.update(cufft_header_version=self.lib.cufft_low_header_version(),
                         cuda_header_version=self.lib.cufft_low_cuda_header_version())

    def _check(self, code):
        if code:
            message = self.lib.cufft_low_last_error().decode("utf-8", errors="replace")
            raise RuntimeError(f"low cuFFT bridge status {code}: {message}")

    def execute(self, source, destination, *, stream):
        if not self.handle.value:
            raise RuntimeError("plan is closed")
        if address(stream, zero=True) != self.stream:
            raise ValueError("execution stream must match the stream bound at plan creation")
        self._check(self.lib.cufft_low_execute(self.handle, address(source), address(destination)))

    def close(self):
        if self.handle.value:
            handle, self.handle = self.handle, ct.c_void_p()
            self._check(self.lib.cufft_low_destroy(handle))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output, nvcc="/usr/local/cuda-12.4/bin/nvcc"):
    output, nvcc = Path(output).resolve(), Path(nvcc).resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    cuda = nvcc.parent.parent
    library = output / "libcufft_low_precision_baseline.so"
    command = [str(nvcc), "-std=c++17", "-O3", "-arch=sm_80", "--shared", "--cudart=shared",
               "-Xcompiler=-fPIC", str(SOURCE / "bridge.cu"), "-lcufft", "-L" + str(cuda / "lib64"),
               "-Xlinker=-rpath", "-Xlinker=" + str(cuda / "lib64"), "-o", str(library)]
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    subprocess.run(command, check=True)
    manifest = dict(schema="cufft-low-precision-bridge-v1", command=command, compiler=version,
        source_sha256={name: sha256(SOURCE / name) for name in ("bridge.cu", "bridge.h")},
        binding_sha256=sha256(__file__),
        vendor_header_sha256={name: sha256(cuda / "include" / name) for name in
                              ("cufft.h", "cufftXt.h", "library_types.h", "cuda_runtime_api.h")},
        library_sha256=sha256(library), nvcc=str(nvcc), architecture="sm_80",
        documented_cuda_version="12.4.0", documentation=DOCS,
        input_output_execution_types=["CUDA_C_16F", "CUDA_C_16BF"],
        gpu_initialization=False, gpu_execution=False, graph_validation="pending",
        normalization="none only; inverse unscaled", license="bridge: repository BSD-3-Clause; cuFFT: NVIDIA CUDA Toolkit EULA")
    vendor = cuda / "lib64/libcufft.so"
    if vendor.exists():
        manifest["vendor_library"] = dict(path=str(vendor.resolve()), sha256=sha256(vendor))
    (output / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return library


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--nvcc", default="/usr/local/cuda-12.4/bin/nvcc")
    parser.add_argument("--shape", type=int, nargs="+", default=[65536])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--precision", choices=tuple(PRECISIONS), default="fp16")
    parser.add_argument("--compute", default="vendor-native")
    parser.add_argument("--direction", choices=("forward", "inverse"), default="forward")
    parser.add_argument("--normalization", default="none")
    args = parser.parse_args(argv)
    if args.build:
        print(build(args.build, args.nvcc))
        return 0
    capability = support(args.shape, args.batch, precision=args.precision, compute=args.compute,
                         direction=args.direction, normalization=args.normalization)
    print(json.dumps(capability, indent=2))
    return 0 if capability["supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
