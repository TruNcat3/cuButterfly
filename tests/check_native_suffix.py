#!/usr/bin/env python3
"""Prepare bounded native suffix modules on CPU; run separately under a GPU guard."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys

from check_standalone_module import Invocation, Module, Resources

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_cases():
    controls = json.loads((ROOT / "tests/data/native_suffix_provider_gaps.json").read_text())
    seeds = {row["suffix_logN"]: dict(row["mapping"]) for row in controls}
    cases = []
    for suffix in (10, 12):
        for precision in ("fp32", "fp64"):
            point = dict(seeds[suffix], suffix_core="native", suffix_ept=32,
                         suffix_threads=(1 << suffix) // 32 * 2, suffix_units_per_cta=2,
                         precision=precision, execution_group_mappings=[])
            point.pop("mapping_json", None)
            cases.append(dict(name=f"main-s{suffix}-{precision}", mapping=point, main=True))
        point = dict(seeds[suffix], suffix_core="cufftdx-block")
        point.pop("mapping_json", None)
        cases.append(dict(name=f"dx16-s{suffix}-fp64", mapping=point, main=True))
    # Small exact geometries include EPT1, Rows1, sub-warp Rows4, an odd log,
    # Columns1/4 and whole/chunk2/chunkEPT exchange. Both precisions participate.
    for suffix, ept, columns, chunk in ((5, 1, 1, 0), (5, 32, 4, 2), (7, 32, 4, 32)):
        for precision in ("fp32", "fp64"):
            point = dict(backend="online-reorder", fft_core="register-tile", suffix_core="native",
                         operator="fft", logN=4+suffix, batch=3, local_stages=4,
                         stage_partition=[4, suffix], precision=precision, prefix_codelet="native",
                         prefix_shared_layout="linear", prefix_codelet_lanes=1,
                         prefix_ept=4, prefix_threads=16, suffix_ept=ept,
                         suffix_threads=(1 << suffix)//ept*columns, exchange_chunks=[0, chunk],
                         cross_twiddle="recurrence", shared_layout="writer-aligned")
            cases.append(dict(name=f"edge-s{suffix}-e{ept}-c{columns}-{precision}", mapping=point, main=False))
    return cases


def prepare(out, cases_path=None):
    import numpy as np
    from scipy import fft
    if not hasattr(fft, "fft"):
        from scipy import fftpack as fft
    sys.path.insert(0, str(ROOT / "scripts"))
    from compile_module import compile_mapping
    cases = json.loads(cases_path.read_text()) if cases_path is not None else default_cases()
    if (not isinstance(cases, list) or not cases or any(
            not isinstance(case, dict) or not isinstance(case.get("name"), str) or
            not isinstance(case.get("mapping"), dict) or type(case.get("main")) is not bool
            for case in cases)):
        raise ValueError("cases must be a nonempty list of {name, mapping, main} objects")
    if len({case["name"] for case in cases}) != len(cases):
        raise ValueError("case names must be unique")
    out.mkdir(parents=True, exist_ok=True)
    references = {}
    for case in cases:
        point = case["mapping"]
        n, batch, precision = 1 << point["logN"], point["batch"], point["precision"]
        key = f"n{n}-b{batch}-{precision}"
        case["reference_key"] = key
        if key in references:
            continue
        dtype = np.complex128 if precision == "fp64" else np.complex64
        rng = np.random.RandomState(42)
        signal = (rng.uniform(-1, 1, (batch, n))+1j*rng.uniform(-1, 1, (batch, n))).astype(dtype)
        directory = out / "references" / key
        directory.mkdir(parents=True, exist_ok=True)
        assets = {}
        for name, array in (("input", signal), ("forward", fft.fft(signal, axis=-1)),
                            ("inverse", fft.ifft(signal, axis=-1))):
            if array.dtype != np.dtype(dtype):
                raise RuntimeError(f"reference changed precision: {array.dtype}, expected {dtype}")
            path = directory / f"{name}.npy"
            np.save(path, array)
            assets[name] = dict(path=str(path.resolve()), sha256=sha(path), dtype=str(array.dtype),
                                backend="prepared-input" if name == "input" else fft.__name__)
        references[key] = assets
    def compile_case(case):
        module = compile_mapping(case["mapping"], root=ROOT,
                                 mathdx=ROOT / "external/mathdx/nvidia/mathdx",
                                 nvcc=Path("/usr/local/cuda/bin/nvcc"), sm=80,
                                 cache=out / "modules", timeout=600)
        case.update(module=str(module.resolve()), module_sha256=sha(module),
                    manifest_sha256=sha(module.with_name("manifest.json")))
        print("prepared", case["name"], flush=True)
        return case
    with ThreadPoolExecutor(max_workers=2) as executor:
        cases = list(executor.map(compile_case, cases))
    (out / "prepared.json").write_text(json.dumps(dict(schema="native-suffix-check-v1", cases=cases,
        references=references,
        source_hashes={name: sha(ROOT / name) for name in
                       ("src/fft_native_block.cuh", "src/fft_grouped_suffix.cuh", "scripts/compile_module.py")}), indent=2)+"\n")


def run(out):
    # CUDA-capable libraries are imported only after the explicit run boundary.
    import cupy as cp
    import numpy as np
    prepared = json.loads((out / "prepared.json").read_text())
    for name, digest in prepared["source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"prepared source changed: {name}")
    rows, timings = [], []
    references = {}
    for key, assets in prepared["references"].items():
        references[key] = {}
        for name, asset in assets.items():
            if sha(asset["path"]) != asset["sha256"]:
                raise RuntimeError(f"prepared reference changed: {asset['path']}")
            references[key][name] = np.load(asset["path"], mmap_mode="r")
    stream = cp.cuda.Stream(non_blocking=True)
    sentinel = -999 + 777j
    def checkpoint():
        (out / "run.json").write_text(json.dumps(dict(cases=rows, timings=timings,
            floating_policy="Report same-precision scipy and cuFFT errors; no invented numeric threshold",
            gpu_interference="Caller must hold its external exclusive-GPU guard during --run"), indent=2)+"\n")
    def check(status):
        if status:
            raise RuntimeError(f"CUDA module status {status}")
    def metrics(actual, reference):
        delta = actual-reference
        denominator = np.linalg.norm(reference.ravel())
        return dict(finite=bool(np.isfinite(actual).all()), max_abs=float(np.max(np.abs(delta))),
                    relative_l2=float(np.linalg.norm(delta.ravel())/denominator) if denominator else 0.0)
    def launch(module, group_launch, source, destination, scratch, batch, distance, stride,
               inverse, normalize, grouped=False):
        invocation = Invocation(source.data.ptr, destination.data.ptr, scratch.data.ptr,
                                batch, distance, stride, inverse, normalize, stream.ptr)
        if grouped:
            first = Invocation(source.data.ptr, scratch.data.ptr, None, batch, distance, stride,
                               inverse, normalize, stream.ptr)
            second = Invocation(scratch.data.ptr, destination.data.ptr, None, batch, distance,
                                stride, inverse, normalize, stream.ptr)
            check(group_launch(0, c.byref(first))); check(group_launch(1, c.byref(second)))
        else:
            check(module.launch(c.byref(invocation)))
    def graph_time(operation):
        with stream:
            for _ in range(5):
                operation()
        stream.synchronize()
        with stream:
            stream.begin_capture()
            operation()
            graph = stream.end_capture()
        samples = []
        for _ in range(5):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            for _ in range(20):
                graph.launch(stream)
            end.record(stream); end.synchronize()
            samples.append(float(cp.cuda.get_elapsed_time(start, end))/20)
        return dict(graph_ms_samples=samples, graph_ms_median=float(np.median(samples)))
    for case in prepared["cases"]:
        path = Path(case["module"])
        if sha(path) != case["module_sha256"] or sha(path.with_name("manifest.json")) != case["manifest_sha256"]:
            raise RuntimeError(f"prepared artifact changed: {path}")
        library = c.CDLL(str(path))
        library.cubutterfly_module_v1.restype = c.POINTER(Module)
        module = library.cubutterfly_module_v1().contents
        if module.version != 1 or module.size != c.sizeof(Module) or module.groups != 2:
            raise RuntimeError("unexpected standalone ABI")
        group_launch = library.cubutterfly_module_launch_group_v1
        group_launch.argtypes = [c.c_uint32, c.POINTER(Invocation)]
        group_launch.restype = c.c_int
        point = case["mapping"]
        n, batch = 1 << point["logN"], point["batch"]
        dtype = np.complex128 if point["precision"] == "fp64" else np.complex64
        assets = references[case["reference_key"]]
        resources = []
        for inverse in (0, 1):
            for group in (0, 1):
                resource = Resources(); check(module.resources(inverse, group, c.byref(resource)))
                resources.append(dict(inverse=inverse, group=group,
                                      **{name: getattr(resource, name) for name, _ in resource._fields_}))
        for inverse, normalize in ((0, 0), (1, 0), (1, 1)):
            for stride in (1, 2):
                distance = n*stride+16
                valid = np.zeros((batch, distance), dtype=bool); valid[:, :n*stride:stride] = True
                for grouped in (False, True):
                    for pattern in ("random", "zero", "impulse"):
                        if pattern == "random":
                            signal = assets["input"]
                            reference = assets["inverse" if inverse else "forward"]
                            if inverse and not normalize: reference = reference*n
                        else:
                            signal = np.zeros((batch, n), dtype=dtype)
                            if pattern == "impulse": signal[:, 0] = np.arange(1, batch+1)
                            reference = np.zeros_like(signal)
                            if pattern == "impulse":
                                reference[:] = np.arange(1, batch+1, dtype=np.float64)[:, None]
                                if inverse and normalize: reference /= n
                        host = np.full((batch, distance), sentinel, dtype=dtype); host[valid] = signal.ravel()
                        with stream:
                            source = cp.asarray(host); destination = cp.full(host.shape, sentinel, dtype=dtype)
                            scratch = cp.empty_like(source)
                        launch(module, group_launch, source, destination, scratch, batch, distance, stride,
                               inverse, normalize, grouped)
                        stream.synchronize()
                        actual_host = cp.asnumpy(destination); actual = actual_host[:, :n*stride:stride]
                        with stream:
                            gpu_reference = cp.fft.ifft(cp.asarray(signal)) if inverse else cp.fft.fft(cp.asarray(signal))
                            if inverse and not normalize: gpu_reference *= n
                        stream.synchronize()
                        baseline = cp.asnumpy(gpu_reference)
                        error = metrics(actual, reference)
                        guarded = bool(np.all(actual_host[~valid] == dtype(sentinel)))
                        # Exact zero and origin impulse expose structural mistakes
                        # without introducing a floating-point tolerance policy.
                        structural = True
                        if pattern != "random":
                            structural = bool(np.array_equal(actual, reference))
                        row = dict(case=case["name"], inverse=inverse, normalize=normalize, stride=stride,
                                   grouped=grouped, pattern=pattern, **error, sentinel_guard=guarded,
                                   structural_exact=structural, reference_dtype=str(reference.dtype),
                                   cupy_reference_error=metrics(baseline, reference), resources=resources)
                        rows.append(row)
                        if not error["finite"] or not guarded or not structural:
                            checkpoint()
                            raise RuntimeError(f"structural native suffix failure: {row}")
        checkpoint()
        if case["main"] and point["precision"] == "fp64":
            with stream:
                source = cp.ones((batch, n), dtype=cp.complex128)
                destination = cp.empty_like(source); scratch = cp.empty_like(source)
            def operation():
                launch(module, group_launch, source, destination, scratch, batch, n, 1, 0, 0)
            timing = dict(case=case["name"], core=point["suffix_core"], ept=point["suffix_ept"],
                          threads=point["suffix_threads"],
                          columns=point["suffix_threads"]*point["suffix_ept"]//
                                  (1 << (point["logN"]-point["local_stages"])), batch=batch, n=n,
                          **graph_time(operation))
            timings.append(timing)
            if point["suffix_core"] == "native":
                fft_outputs = []
                def cufft():
                    fft_outputs[:] = [cp.fft.fft(source, axis=-1)]
                timings.append(dict(case=case["name"]+"-cupy-cufft", core="cufft", batch=batch, n=n,
                                    **graph_time(cufft)))
            checkpoint()
        print("checked", case["name"], flush=True)
    checkpoint()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", type=Path)
    mode.add_argument("--run", type=Path)
    parser.add_argument("--cases", type=Path, help="Explicit case list, valid only with --prepare")
    args = parser.parse_args()
    if args.cases is not None and args.prepare is None:
        parser.error("--cases is valid only with --prepare")
    if args.prepare is not None: prepare(args.prepare.resolve(), args.cases)
    else: run(args.run.resolve())


if __name__ == "__main__":
    main()
