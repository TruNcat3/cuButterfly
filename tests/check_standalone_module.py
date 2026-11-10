#!/usr/bin/env python3
"""Validate a compiled module through its C ABI without loading libcuntt."""
import argparse
import ctypes as c
import json
import pathlib
import numpy as np


class Invocation(c.Structure):
    _fields_ = [("input", c.c_void_p), ("output", c.c_void_p), ("workspace", c.c_void_p),
                ("batch", c.c_uint64), ("distance", c.c_uint64), ("stride", c.c_uint64),
                ("inverse", c.c_int), ("normalize", c.c_int), ("stream", c.c_void_p)]


class Resources(c.Structure):
    _fields_ = [("registers", c.c_uint32), ("threads", c.c_uint32),
                ("static_shared", c.c_uint32), ("dynamic_shared", c.c_uint32)]


ResourceQuery = c.CFUNCTYPE(c.c_int, c.c_int, c.c_uint32, c.POINTER(Resources))
Launch = c.CFUNCTYPE(c.c_int, c.POINTER(Invocation))


class Module(c.Structure):
    _fields_ = [("version", c.c_uint32), ("size", c.c_uint32), ("identity", c.c_char_p),
                ("sm", c.c_uint32), ("groups", c.c_uint32), ("resources", ResourceQuery), ("launch", Launch)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    metadata = json.loads(args.module.with_name("manifest.json").read_text())
    library = c.CDLL(str(args.module.resolve()))
    library.cubutterfly_module_v1.restype = c.POINTER(Module)
    module = library.cubutterfly_module_v1().contents
    group_launch=library.cubutterfly_module_launch_group_v1
    group_launch.argtypes=[c.c_uint32,c.POINTER(Invocation)]
    assert group_launch(module.groups,None)!=0
    assert module.version == 1 and module.size == c.sizeof(Module) and module.groups >= 1
    cuda = c.CDLL("/usr/local/cuda/lib64/libcudart.so")
    cuda.cudaMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
    cuda.cudaMemcpy.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int]
    cuda.cudaFree.argtypes = [c.c_void_p]
    def check(status):
        if status != 0:
            raise RuntimeError(f"CUDA/module status {status}")
    n = 1 << metadata["mapping"]["logN"]
    fp64 = metadata["mapping"].get("precision") == "fp64"
    dtype = np.complex128 if fp64 else np.complex64
    allocations = []
    rows = []
    try:
        for inverse, stride, grouped in ((0,1,False),(1,2,False),(0,2,True),(1,1,True)):
            resources = []
            for group in range(module.groups):
                resource = Resources()
                check(module.resources(inverse, group, c.byref(resource)))
                resources.append({name: getattr(resource,name) for name,_ in resource._fields_})
            distance, batch = n*stride+16, 3
            data = np.zeros((batch, distance), dtype=dtype)
            rng = np.random.default_rng(7)
            signal = (rng.uniform(-1,1,(batch,n))+1j*rng.uniform(-1,1,(batch,n))).astype(dtype)
            data[:, :n*stride:stride] = signal
            result = np.zeros_like(data)
            pointers = []
            for allocation in range(3):
                pointer = c.c_void_p()
                size=data.nbytes*(min(2,module.groups-1) if allocation==2 and module.groups>1 else 1)
                check(cuda.cudaMalloc(c.byref(pointer), size))
                pointers.append(pointer); allocations.append(pointer)
            check(cuda.cudaMemcpy(pointers[0], data.ctypes.data, data.nbytes, 1))
            invocation = Invocation(*pointers, batch, distance, stride, inverse, 1, None)
            if grouped:
                source=pointers[0]
                for group in range(module.groups):
                    destination=pointers[1] if group+1==module.groups else c.c_void_p(pointers[2].value+(group%2)*data.nbytes)
                    invocation_group=Invocation(source,destination,None,batch,distance,stride,inverse,1,None)
                    check(group_launch(group,c.byref(invocation_group)))
                    source=destination
            else:
                check(module.launch(c.byref(invocation)))
            check(cuda.cudaDeviceSynchronize())
            check(cuda.cudaMemcpy(result.ctypes.data, pointers[1], result.nbytes, 2))
            expected = np.fft.ifft(signal) if inverse else np.fft.fft(signal)
            actual = result[:, :n*stride:stride]
            error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
            rows.append(dict(inverse=inverse,stride=stride,grouped=grouped,relative_l2=error,correct=error<(1e-11 if fp64 else 1e-5),resources=resources))
            print(rows[-1], flush=True)
    finally:
        for pointer in allocations:
            cuda.cudaFree(pointer)
    args.output.write_text(json.dumps(dict(module=str(args.module),cases=rows),indent=2)+"\n")
    return 0 if all(row["correct"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
