"""Contract-checked adapters for finite research acceptance.

These reuse the existing benchmark binaries/harnesses. Unsupported contracts
are reported before timing; a failed run is never silently counted as a win.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

from calibration_space import command_for
from hardware_registry import canonical_semantics
from run_external_baseline_suite import read_one_csv

ROOT = Path(__file__).resolve().parents[1]
GPU_NTT_MODULUS = "576460756061519873"


def sha256(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binary_identity(path):
    """Include dynamically linked libraries, not just a small executable."""
    path = Path(path).resolve()
    result = subprocess.run(["ldd", str(path)], text=True, capture_output=True)
    libraries = sorted(set(re.findall(r"(?:=>\s+|^\s*)(/\S+)", result.stdout, re.M)))
    return dict(path=str(path), sha256=sha256(path),
                libraries={p: sha256(p) for p in libraries if Path(p).is_file()})


def baseline_identity(paths):
    result = {"adapter_sha256": sha256(__file__),
              "cufft_harness": binary_identity(paths["build_dir"] / "cubutterfly_bench")}
    for name in ("vkfft_binary", "gpuntt_binary"):
        path = paths.get(name)
        result[name] = binary_identity(path) if path and Path(path).is_file() else None
    python = paths.get("fht_python")
    if python and Path(python).is_file():
        # Package inspection must not acquire a CUDA context on an occupied GPU.
        code = ("import importlib.util,json,torch; "
                "names=['fast_hadamard_transform','fast_hadamard_transform_cuda']; "
                "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
                "modules={n:importlib.util.find_spec(n).origin for n in names})))")
        probe = subprocess.run([str(python), "-c", code], text=True, capture_output=True,
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        if probe.returncode:
            raise ValueError("configured Dao environment cannot import required packages: " + probe.stderr[-2000:])
        packages = json.loads(probe.stdout)
        packages["module_hashes"] = {n: sha256(p) for n, p in packages["modules"].items()}
        interface = Path(packages["modules"]["fast_hadamard_transform"]).with_name("fast_hadamard_transform_interface.py")
        packages["interface_sha256"] = sha256(interface)
        result["dao"] = dict(python=str(Path(python).resolve()), python_sha256=sha256(python),
                             harness_sha256=sha256(ROOT / "scripts/benchmark_external_fht.py"), packages=packages)
    else:
        result["dao"] = None
    return result


def available_baselines(workload, paths):
    semantic = canonical_semantics(workload)
    operator, precision = semantic["operator"], semantic["precision"]
    n = 1 << int(semantic["logN"])
    contiguous = int(semantic["element_stride"]) == 1 and int(semantic["batch_stride"]) == n
    forward = semantic["direction"] == "forward"
    available, unavailable = [], []

    def external(name, runner, path, supported, reason, **fields):
        if not supported:
            unavailable.append(dict(name=name, reason=reason))
        elif not path or not Path(path).is_file():
            unavailable.append(dict(name=name, reason="executable-not-configured-or-missing"))
        else:
            available.append(dict(name=name, runner=runner, external=True, **fields))

    if operator == "fft":
        external("cuFFT", "cufft", paths["build_dir"] / "cubutterfly_bench",
                 precision in ("fp32", "fp64"), "cuFFT adapter requires fp32/fp64")
        external("VkFFT", "vkfft", paths.get("vkfft_binary"),
                 precision == "fp32" and forward and contiguous,
                 "VkFFT adapter requires fp32 forward contiguous FFT")
    elif operator == "fwht":
        external("Dao-FWHT", "dao-fht", paths.get("fht_python"),
                 precision in ("fp16", "bf16", "fp32") and 3 <= int(semantic["logN"]) <= 15
                 and contiguous and semantic.get("placement") == "out-of-place"
                 and semantic["accumulation"] == "native",
                 "Dao adapter requires native fp16/bf16/fp32, logN=3..15, contiguous out-of-place FWHT")
    elif operator == "ntt":
        natural = semantic["output_order"] == "natural"
        external("GPU-NTT-natural" if natural else "GPU-NTT-bit-reversed", "gpuntt",
                 paths.get("gpuntt_binary"), precision == "word64" and forward and contiguous
                 and semantic["modulus"] == GPU_NTT_MODULUS and semantic["input_order"] == "natural"
                 and semantic["output_order"] in ("natural", "bit-reversed")
                 and 12 <= int(semantic["logN"]) <= 24,
                 "GPU-NTT harness requires forward word64 cyclic NTT, its fixed prime, natural input and logN=12..24",
                 naturalize=natural)
    else:
        unavailable.append(dict(name="external-library", reason="no matching external adapter for this operator"))
    # An ablation reference is useful even where no matching external library
    # exists, but is never labelled an external-library result.
    if operator != "ntt":
        available.append(dict(name="in-tree-radix2", runner="radix2", external=False))
    return dict(available=available, unavailable=unavailable)


def run_baseline(workload, spec, protocol, paths):
    if spec not in available_baselines(workload, paths)["available"]:
        raise ValueError("baseline is not supported for the requested semantic contract")
    runner = spec["runner"]
    common = ["--warmup", str(protocol["warmup"]), "--repeat", str(protocol["repeat"])]
    coverage = dict(method="CPU-reference", batches=int(workload["batch"]))
    if runner in ("cufft", "radix2"):
        point = dict(workload)
        if runner == "cufft":
            point["backend"] = "cufft"
        else:
            log_n = int(point["logN"])
            if log_n > 20:
                full_groups, remainder = divmod(log_n, 8)
                stages = [8] * full_groups
                if remainder:
                    stages.append(remainder)
                point.update(
                    backend="shared-iterative",
                    stage_partition=",".join(str(stage) for stage in stages),
                    tile_threads=128,
                    fft_core="scalar",
                    compute_unit="radix2",
                    local_exchange="shared",
                    shared_layout="writer-aligned",
                )
            else:
                small = log_n <= 8
                point.update(backend="temporal-tile" if small else "hierarchical", compute_unit="radix2",
                             local_stages=0 if small else 8, tile_threads=128,
                             fft_core="scalar", local_exchange="shared")
        binary = paths["build_dir"] / "cubutterfly_bench"
        command = command_for(binary, point) + common + ["--verify", "--verify-batches", str(protocol["verify_batches"]), "--csv"]
        response = subprocess.run(command, text=True, capture_output=True, check=True)
        row = read_one_csv(response.stdout)
        correct = row.get("correct") == "1"
        limit = int(protocol["verify_batches"])
        coverage["batches"] = min(limit, int(workload["batch"])) if limit else int(workload["batch"])
    elif runner == "vkfft":
        binary = paths["vkfft_binary"]
        command = [str(binary), "--library", "vkfft", "--logN", str(workload["logN"]),
                   "--batch", str(workload["batch"]), "--placement", workload["placement"],
                   *common, "--verify", "--csv"]
        response = subprocess.run(command, text=True, capture_output=True, check=True)
        row = read_one_csv(response.stdout)
        correct = row.get("correct") == "1"
        coverage["method"] = "cuFFT-reference"
    elif runner == "dao-fht":
        binary = paths["fht_python"]
        with tempfile.TemporaryDirectory(prefix="cubutterfly-dao-") as directory:
            output = Path(directory) / "trial.csv"
            command = [str(binary), str(ROOT / "scripts/benchmark_external_fht.py"),
                       "--logNs", str(workload["logN"]), "--dtypes", workload["precision"],
                       "--batch", str(workload["batch"]), *common, "--trials", "1", "--normalization",
                       canonical_semantics(workload)["normalization"], "--output", str(output)]
            response = subprocess.run(command, text=True, capture_output=True, check=True)
            with output.open() as source:
                row = next(csv.DictReader(source))
        # FWHT is self-inverse up to scaling; the existing adapter reports its
        # mathematical forward primitive. Preserve that output in raw_sample.
        correct = row.get("correct") == "1"
        coverage = dict(method="normalized-roundtrip", batches=min(2, int(workload["batch"])),
                        timing="CUDA events around PyTorch calls including output allocation")
    elif runner == "gpuntt":
        binary = paths["gpuntt_binary"]
        command = [str(binary), str(workload["logN"]), str(workload["batch"]),
                   str(protocol["warmup"]), str(protocol["repeat"]), "1", "1" if spec["naturalize"] else "0"]
        response = subprocess.run(command, text=True, capture_output=True, check=True)
        row = read_one_csv(response.stdout)
        marker = "natural_mismatches" if spec["naturalize"] else "bit_reversed_mismatches"
        correct = re.search(r"(?:^|,)" + marker + r"=0(?:,|$)", response.stderr.strip(), re.M) is not None
        coverage["batches"] = 1
    else:
        raise ValueError("unknown baseline runner: " + runner)
    latency = float(row["kernel_ms"])
    if not correct or not math.isfinite(latency) or latency <= 0:
        raise ValueError("baseline correctness or timing failed: " + spec["name"])
    for field in ("logN", "batch", "warmup", "repeat"):
        expected = workload[field] if field in workload else protocol[field]
        # Dao records no warmup/repeat columns; its arguments are recorded.
        if field in row and int(row[field]) != int(expected):
            raise ValueError("baseline semantic/protocol mismatch: " + field)
    if runner == "gpuntt" and row.get("modulus") != str(workload["modulus"]):
        raise ValueError("GPU-NTT modulus mismatch")
    wanted = canonical_semantics(workload)
    for field in ("precision", "placement", "direction", "element_stride", "batch_stride", "normalization"):
        if field in row and not (runner == "dao-fht" and field == "direction"):
            if str(row[field]) != wanted[field]:
                raise ValueError("baseline semantic mismatch: " + field)
    # Explicit adapter contracts fill fields not exported by older harnesses;
    # retain their raw output and validation coverage for audit.
    sample = {**wanted, **row, "correct": "1", "direction": wanted["direction"]}
    return dict(sample=sample, raw_sample=row, correct=True, command=command,
                binary_sha256=sha256(binary), verification=coverage, external=spec["external"],
                stderr=response.stderr)
