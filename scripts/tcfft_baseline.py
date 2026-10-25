"""Independent, restricted tcFFT baseline; importing this module never loads CUDA."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3e87a7083577741815e21b2a3c2bed22520786ce"
SOURCE_SHA256 = "441e6d959a8bcde190896bb77138a248e8897ed0994930a454e8f3112c7eb7d7"
HEADER_SHA256 = "0a92c8f478dbc854ed964614e8047531a8ecbf36f76b58abfdb00bd37b571240"
MERGES = {
    256: (256,), 512: (512,), 1024: (1024,),
    131072: (512, 256), 262144: (512, 512), 524288: (1024, 512),
    16777216: (256, 256, 256), 33554432: (512, 256, 256),
    67108864: (512, 512, 256), 134217728: (512, 512, 512),
}


def support(n: int, batch: int, *, precision="fp16", accumulation="fp16",
            direction="forward", normalization="none", rank=1) -> dict:
    reasons = []
    if (precision, accumulation) != ("fp16", "fp16"):
        reasons.append("upstream uses FP16 storage, multiplication and WMMA accumulators")
    if (direction, normalization, rank) != ("forward", "none", 1):
        reasons.append("adapter supports only rank-1 forward unnormalized C2C")
    if n not in MERGES:
        reasons.append("length has no initialized upstream 1D plan")
    if not isinstance(batch, int) or batch < 1:
        reasons.append("batch must be a positive integer")
    elif n in MERGES:
        total = n * batch
        if total > 2147483647:
            reasons.append("upstream signed-int N*batch would overflow")
        if total % 8192:
            reasons.append("first layer requires N*batch divisible by 8192; no padding")
        step = MERGES[n][0]
        for radix in MERGES[n][1:]:
            if total // step // radix > 65535:
                reasons.append("upstream grid.y exceeds 65535")
            step *= radix
    return {"supported": not reasons, "reasons": reasons, "n": n, "batch": batch,
            "provider": "tcFFT", "upstream_commit": COMMIT,
            "storage_precision": "fp16", "multiply_precision": "fp16",
            "accumulator_type": "fp16", "input_order": "natural",
            "output_order": "natural", "placement": "contiguous C2C",
            "timed_operations": ["device input mixed-radix reorder", "upstream FFT", "device output copy"],
            "graph_status": "prepared, GPU validation pending"}


def transformed_source(source: str) -> str:
    """Change only host dispatch; the hash pins all device arithmetic."""
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("tcFFT source differs from pinned upstream")
    source = source.replace("void tcfftExec(tcfftHandle plan, half *data)",
                            "void tcfftExecStream(tcfftHandle plan, half *data, cudaStream_t stream)")
    lines = source.splitlines(keepends=True)
    attributes = [line for line in lines if "cudaFuncSetAttribute(" in line]
    if len(attributes) != 2:
        raise ValueError("unexpected upstream attribute count")
    source = "".join(line for line in lines if "cudaFuncSetAttribute(" not in line)
    old = "RADIX * sizeof(half2) * n_cont[plan.mergings["
    # Two launch expressions have the same trailing shared-memory expression.
    for index in ("0", "i"):
        before = old + index + "]]>>>"
        if source.count(before) != 1:
            raise ValueError("unexpected upstream launch expression")
        source = source.replace(before, old + index + "]], stream>>>")
    return source


def build(output: Path, nvcc="/usr/local/cuda-12.4/bin/nvcc", upstream=ROOT / "external/tcFFT") -> Path:
    upstream = Path(upstream).resolve()
    if subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip() != COMMIT:
        raise ValueError("tcFFT checkout is not the pinned commit")
    if hashlib.sha256((upstream / "tcfft_half.h").read_bytes()).hexdigest() != HEADER_SHA256:
        raise ValueError("tcFFT header differs from pinned upstream")
    generated = transformed_source((upstream / "tcfft_half.cu").read_text())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    translation_unit = output / "tcfft_stream.cu"
    translation_unit.write_text(generated + '\n#include "' + str(ROOT / "benchmarks/tcfft/bridge.cuh") + '"\n')
    shutil.copyfile(upstream / "LICENSE", output / "LICENSE.tcFFT")
    library = output / "libtcfft_baseline.so"
    command = [nvcc, "-std=c++14", "-O3", "-gencode", "arch=compute_80,code=sm_80",
               "-shared", "-Xcompiler", "-fPIC", "-I" + str(upstream), str(translation_unit),
               "-o", str(library)]
    subprocess.run(command, check=True)
    (output / "build.json").write_text(json.dumps({"upstream_commit": COMMIT, "source_sha256": SOURCE_SHA256,
        "header_sha256": HEADER_SHA256, "command": command, "gpu_execution": False,
        "changes": ["explicit stream launch argument", "move function attributes to plan creation",
                    "append independent C ABI bridge"]}, indent=2) + "\n")
    return library


class Plan:
    """Owns scratch and coefficients. Caller owns device pointers and CUDA stream.

    Create and close outside capture/timing. Synchronize outstanding work before
    close. One plan cannot execute concurrently on multiple streams.
    """
    def __init__(self, library, n, batch):
        info = support(n, batch)
        if not info["supported"]:
            raise ValueError("; ".join(info["reasons"]))
        self.lib = ctypes.CDLL(str(library))
        self.lib.tcfft_baseline_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.tcfft_baseline_execute.argtypes = [ctypes.c_void_p] * 4
        self.lib.tcfft_baseline_destroy.argtypes = [ctypes.c_void_p]
        self.handle = ctypes.c_void_p()
        self._check(self.lib.tcfft_baseline_create(n, batch, ctypes.byref(self.handle)))

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"tcFFT adapter CUDA/status code {code}")

    def execute(self, source: int, destination: int, *, stream: int):
        if not self.handle.value:
            raise RuntimeError("plan is closed")
        if not isinstance(stream, int) or stream < 0:
            raise ValueError("stream must be an explicit nonnegative CUDA stream address")
        self._check(self.lib.tcfft_baseline_execute(self.handle, source, destination, stream))

    def close(self):
        if self.handle.value:
            self._check(self.lib.tcfft_baseline_destroy(self.handle))
            self.handle = ctypes.c_void_p()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--n", type=int, default=131072)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.build:
        print(build(args.build))
    else:
        print(json.dumps(support(args.n, args.batch), indent=2))


if __name__ == "__main__":
    main()
