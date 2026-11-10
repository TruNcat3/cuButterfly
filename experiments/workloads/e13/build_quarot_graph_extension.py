"""Build the isolated current-stream-only QuaRot CUDA extension variant."""
import hashlib
import json
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile
import time
import os

import torch

from torch.utils import cpp_extension


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/e13_assets/sources/QuaRot"
KERNELS = ROOT / "results/e13_assets/quarot_graph_stream_kernels"
BUILD = ROOT / "results/e13_assets/quarot_graph_stream_build"
PATCH = Path(__file__).with_name("quarot_current_stream.patch")
FILES = ("bindings.cpp", "gemm.cu", "quant.cu", "flashinfer.cu")


def source_tree_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_configuration():
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "environment": {name: os.environ.get(name) for name in
                            ("MAX_JOBS", "TORCH_CUDA_ARCH_LIST", "CUDA_HOME")},
            "extra_cuda_cflags": ["-O3"], "extra_cxx_cflags": [],
            "original_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip(),
            "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "original_kernel_tree_sha256": source_tree_hash(SOURCE / "quarot/kernels"),
            "ported_kernel_tree_sha256": source_tree_hash(KERNELS),
            "cutlass_include_tree_sha256": source_tree_hash(SOURCE / "third-party/cutlass/include"),
            "nvcc_version": subprocess.check_output([str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"), "--version"], text=True).strip(),
            "cxx_version": subprocess.check_output([os.environ.get("CXX", "c++"), "--version"], text=True).splitlines()[0]}


def prepare_sources():
    with tempfile.TemporaryDirectory(prefix="quarot-current-stream-") as directory:
        clean = Path(directory) / "kernels"
        shutil.copytree(SOURCE / "quarot/kernels", clean)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(PATCH)], cwd=clean, check=True)
        if KERNELS.exists():
            if source_tree_hash(KERNELS) != source_tree_hash(clean):
                raise RuntimeError("Existing isolated source tree differs from reproducible patch")
        else:
            shutil.copytree(clean, KERNELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--configuration-output", type=Path,
                        help="Write source verification and environment provenance without rebuilding")
    args = parser.parse_args()
    prepare_sources()
    configuration = build_configuration()
    if args.configuration_output:
        provenance = {"configuration": configuration, "source_replay_verified": True,
                      "verification_only": args.prepare_only}
        if (BUILD / "_CUDA_graph.so").is_file():
            provenance["existing_extension_sha256"] = hashlib.sha256((BUILD / "_CUDA_graph.so").read_bytes()).hexdigest()
        args.configuration_output.write_text(json.dumps(provenance, indent=2) + "\n")
    if args.prepare_only:
        print("Isolated sources match the clean pinned checkout plus current-stream patch")
        print(json.dumps(configuration, indent=2))
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    cpp_extension.COMMON_NVCC_FLAGS[:] = [flag for flag in cpp_extension.COMMON_NVCC_FLAGS
                                         if not flag.startswith("-D__CUDA_NO_")]
    started = time.perf_counter()
    extension = cpp_extension.load(
        name="_CUDA_graph",
        sources=[str(KERNELS / name) for name in FILES],
        extra_include_paths=[str(KERNELS / "include"),
                             str(SOURCE / "third-party/cutlass/include"),
                             str(SOURCE / "third-party/cutlass/tools/util/include")],
        extra_cuda_cflags=["-O3"], build_directory=str(BUILD), verbose=True,
    )
    report = {"protocol": "quarot_current_stream_only_v1", "extension_path": extension.__file__,
              "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
              "build_seconds": time.perf_counter() - started,
              "configuration": configuration,
              "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
              "original_source_sha256": {name: hashlib.sha256((SOURCE / "quarot/kernels" / name).read_bytes()).hexdigest()
                                          for name in FILES},
              "source_sha256": {name: hashlib.sha256((KERNELS / name).read_bytes()).hexdigest()
                                for name in FILES}}
    (BUILD / "build_receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
