#!/usr/bin/env python3
"""Build and run the pinned HEonGPU TFHE complete-call benchmark.

The runner builds the repo-owned CUDA benchmark against the pinned HEonGPU
source without editing that external checkout.  The timed region is the full
public gate/circuit call; setup, encryption, and decryption are reported on
stderr and correctness is checked exactly in the executable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HEONGPU = ROOT / "results/e13_assets/sources/HEonGPU"
DEFAULT_BUILD = ROOT / "results/e13_assets/heongpu-build-tfhe-runner"
BENCHMARK = Path(__file__).resolve().with_name("heongpu_tfhe_e2e_bench.cu")
_CONDA_CMAKE = Path("cmake")
CMAKE = _CONDA_CMAKE if _CONDA_CMAKE.exists() else Path("cmake")


def run(command: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heongpu-source", type=Path, default=DEFAULT_HEONGPU)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shapes", default="1,8,64,512")
    parser.add_argument("--gates", default="NAND,AND,NOR,OR,XNOR,XOR,NOT,MUX")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--build-only", action="store_true",
                        help="prepare and compile without accessing a GPU")
    parser.add_argument("--backend", choices=("native", "cubutterfly-shared"),
                        default="native")
    parser.add_argument("--device-test", action="store_true",
                        help="build/run the exact native/template/CPU device test")
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    parser.add_argument("--graph-replays", type=int, default=1)
    parser.add_argument("--circuit-depths", default="2,4,8")
    parser.add_argument("--arch", default="80", help="CUDA architecture, e.g. 80 or 90")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    heongpu_source = args.heongpu_source.resolve()
    build_dir = args.build_dir.resolve()
    if args.backend == "cubutterfly-shared" and build_dir == DEFAULT_BUILD.resolve():
        build_dir = build_dir.with_name("heongpu-build-tfhe-shared")
    output = args.output.resolve()
    if not heongpu_source.exists():
        raise SystemExit(f"HEonGPU source does not exist: {heongpu_source}")
    if not BENCHMARK.exists():
        raise SystemExit(f"benchmark source does not exist: {BENCHMARK}")

    build_dir.mkdir(parents=True, exist_ok=True)
    cached_build = ROOT / "results/e13_assets/heongpu-build-e13"
    cpm_cache = cached_build / "_deps"
    prefix = Path("/tmp/heongpu-e13-prefix-20261003")
    prebuilt = cached_build / "src/libheongpu.a"
    original_prebuilt = prebuilt
    if args.backend == "cubutterfly-shared" and not prebuilt.exists():
        raise SystemExit("cubutterfly-shared requires the pinned prebuilt archive for bootstrap object replacement")
    if args.backend == "cubutterfly-shared" and prebuilt.exists():
        # nvlink reads device definitions from every archive member, unlike
        # the host linker. Replace only this member in a private archive copy.
        prebuilt = build_dir / "libheongpu-tfhe-without-bootstrap.a"
        if not args.skip_build:
            shutil.copy2(original_prebuilt, prebuilt)
            run(["ar", "d", str(prebuilt), "bootstrapping.cu.o"])
    gf2x_lib = Path("/tmp/heongpu-e13-gf2x/root/usr/lib/x86_64-linux-gnu/libgf2x.so.3.0.0")
    wrapper_dir = Path(tempfile.mkdtemp(prefix="cubutterfly-heongpu-tfhe-"))
    try:
        # Prefer the already-built pinned HEonGPU archive. Reconfiguring the
        # external project would make CMake fetch RMM/spdlog again and can
        # silently turn a local benchmark into a network-dependent build.
        cmake_file = wrapper_dir / "CMakeLists.txt"
        if prebuilt.exists():
            thirdparty = cached_build / "thirdparty"
            thirdparty_source = heongpu_source / "thirdparty"
            include_root = heongpu_source / "src/include"
            rmm_include = cpm_cache / "rmm-src/cpp/include"
            rmm_build_include = cpm_cache / "rmm-build/include"
            cmake_lines = [
                "cmake_minimum_required(VERSION 3.26)\n",
                "project(cubutterfly_tfhe_runner LANGUAGES CXX CUDA)\n",
                "find_package(CUDAToolkit REQUIRED)\n",
                "find_package(Threads REQUIRED)\n",
                "find_package(ZLIB REQUIRED)\n",
                "find_package(OpenSSL REQUIRED COMPONENTS SSL Crypto)\n",
                f"set(HEONGPU_INCLUDE_ROOT \"{include_root}\")\n",
                f"set(HEONGPU_NTT_INCLUDE \"{thirdparty_source / 'GPU-NTT/src/include'}\")\n",
                f"set(HEONGPU_FFT_INCLUDE \"{thirdparty_source / 'GPU-FFT/src/include'}\")\n",
                f"set(HEONGPU_RNG_INCLUDE \"{thirdparty_source / 'RNGonGPU/src/include'}\")\n",
                f"set(HEONGPU_RMM_INCLUDE \"{cpm_cache / 'rmm-src/cpp/include'}\")\n",
                f"set(HEONGPU_RMM_BUILD_INCLUDE \"{cpm_cache / 'rmm-build/include'}\")\n",
                f"set(HEONGPU_NVTX_INCLUDE \"{cpm_cache / 'nvtx3-src/c/include'}\")\n",
                f"set(HEONGPU_RAPIDS_INCLUDE \"{cpm_cache / 'rapids_logger-src/include'}\")\n",
                f"add_library(heongpu_ntt STATIC IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu_ntt PROPERTIES IMPORTED_LOCATION \"{thirdparty / 'GPU-NTT/src/libntt-1.0.a'}\" INTERFACE_INCLUDE_DIRECTORIES \"{thirdparty_source / 'GPU-NTT/src/include'}\")\n",
                "target_link_libraries(heongpu_ntt INTERFACE CUDA::cudart)\n",
                "add_library(heongpu_fft STATIC IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu_fft PROPERTIES IMPORTED_LOCATION \"{thirdparty / 'GPU-FFT/src/libfft-1.0.a'}\" INTERFACE_INCLUDE_DIRECTORIES \"{thirdparty_source / 'GPU-FFT/src/include'}\")\n",
                "target_link_libraries(heongpu_fft INTERFACE CUDA::cudart)\n",
                "add_library(heongpu_rng STATIC IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu_rng PROPERTIES IMPORTED_LOCATION \"{thirdparty / 'RNGonGPU/src/librngongpu-1.0.a'}\" INTERFACE_INCLUDE_DIRECTORIES \"{thirdparty_source / 'RNGonGPU/src/include'}\")\n",
                "target_link_libraries(heongpu_rng INTERFACE CUDA::cudart)\n",
                "add_library(heongpu_rapids SHARED IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu_rapids PROPERTIES IMPORTED_LOCATION \"{cpm_cache / 'rapids_logger-build/librapids_logger.so'}\" INTERFACE_INCLUDE_DIRECTORIES \"{cpm_cache / 'rapids_logger-src/include'}\")\n",
                "target_link_libraries(heongpu_rapids INTERFACE Threads::Threads)\n",
                "add_library(heongpu_rmm SHARED IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu_rmm PROPERTIES IMPORTED_LOCATION \"{cpm_cache / 'rmm-build/librmm.so'}\" INTERFACE_INCLUDE_DIRECTORIES \"{rmm_include};{rmm_build_include}\")\n",
                "target_link_libraries(heongpu_rmm INTERFACE CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS} heongpu_rapids)\n",
                "find_library(HEONGPU_GMP gmp PATHS /tmp/heongpu-e13-prefix-20261003/usr/lib/x86_64-linux-gnu REQUIRED)\n",
                "find_library(HEONGPU_NTL ntl PATHS /tmp/heongpu-e13-prefix-20261003/usr/lib/x86_64-linux-gnu REQUIRED)\n",
                f"set(HEONGPU_GF2X \"{gf2x_lib}\")\n",
                "add_library(heongpu STATIC IMPORTED GLOBAL)\n",
                f"set_target_properties(heongpu PROPERTIES IMPORTED_LOCATION \"{prebuilt}\" INTERFACE_INCLUDE_DIRECTORIES \"{include_root};{thirdparty_source / 'GPU-NTT/src/include'};{thirdparty_source / 'GPU-FFT/src/include'};{thirdparty_source / 'RNGonGPU/src/include'};{cpm_cache / 'rmm-src/cpp/include'};{cpm_cache / 'rmm-build/include'};{cpm_cache / 'nvtx3-src/c/include'};{cpm_cache / 'rapids_logger-src/include'}\")\n",
                "target_link_libraries(heongpu INTERFACE heongpu_ntt heongpu_fft heongpu_rng heongpu_rmm ${HEONGPU_GMP} ${HEONGPU_NTL} ${HEONGPU_GF2X} OpenSSL::SSL OpenSSL::Crypto CUDA::curand ZLIB::ZLIB)\n",
                f"add_executable(cubutterfly_heongpu_tfhe \"{BENCHMARK}\")\n",
                "target_link_libraries(cubutterfly_heongpu_tfhe PRIVATE heongpu CUDA::cudart)\n",
                f"target_include_directories(cubutterfly_heongpu_tfhe PRIVATE \"{prefix / 'usr/include'}\" \"{prefix / 'usr/include/x86_64-linux-gnu'}\")\n",
                "target_compile_definitions(cubutterfly_heongpu_tfhe PRIVATE CCCL_DISABLE_PDL CUB_DISABLE_NAMESPACE_MAGIC CUB_IGNORE_NAMESPACE_MAGIC_ERROR LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA THRUST_DISABLE_ABI_NAMESPACE THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_CPP THRUST_IGNORE_ABI_NAMESPACE_ERROR)\n",
                "target_compile_features(cubutterfly_heongpu_tfhe PRIVATE cxx_std_17)\n",
                "set_target_properties(cubutterfly_heongpu_tfhe PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_RUNTIME_LIBRARY Static BUILD_RPATH \"{0};{1};{2}\")\n".format(cpm_cache / "rmm-build", cpm_cache / "rapids_logger-build", gf2x_lib.parent),
                f"set_target_properties(cubutterfly_heongpu_tfhe PROPERTIES CUDA_ARCHITECTURES {args.arch})\n",
            ]
        else:
            cmake_lines = [
            "cmake_minimum_required(VERSION 3.26)\n",
            "project(cubutterfly_tfhe_runner LANGUAGES CXX CUDA)\n",
            "set(HEonGPU_BUILD_TESTS OFF CACHE BOOL \"\" FORCE)\n",
            "set(HEonGPU_BUILD_EXAMPLES OFF CACHE BOOL \"\" FORCE)\n",
            "set(HEonGPU_BUILD_BENCHMARKS OFF CACHE BOOL \"\" FORCE)\n",
            ]
            if cpm_cache.exists():
                cmake_lines.append(f"set(CPM_SOURCE_CACHE \"{cpm_cache}\" CACHE PATH \"\" FORCE)\n")
            cmake_lines.extend([
                f"add_subdirectory(\"{heongpu_source}\" heongpu-build)\n",
                f"add_executable(cubutterfly_heongpu_tfhe \"{BENCHMARK}\")\n",
                "target_link_libraries(cubutterfly_heongpu_tfhe PRIVATE heongpu CUDA::cudart)\n",
                "set_target_properties(cubutterfly_heongpu_tfhe PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)\n",
                f"set_target_properties(cubutterfly_heongpu_tfhe PROPERTIES CUDA_ARCHITECTURES {args.arch})\n",
            ])
        cmake_lines.extend([
            f'target_sources(cubutterfly_heongpu_tfhe PRIVATE "{ROOT / "adapters/heongpu/graph_memory.cpp"}")\n',
            f'target_include_directories(cubutterfly_heongpu_tfhe PRIVATE "{ROOT / "adapters/heongpu"}")\n',
            'target_link_options(cubutterfly_heongpu_tfhe PRIVATE "-Wl,--wrap=_ZNK7heongpu10MemoryPool19get_device_resourceEv" "-Wl,--wrap=_ZN7heongpu10MemoryPool8instanceEv" "-Wl,--wrap=cudaMemcpyAsync")\n',
        ])
        if args.backend == "cubutterfly-shared":
            adapter = ROOT / "adapters/heongpu"
            bootstrap = heongpu_source / "src/lib/kernel/bootstrapping.cu"
            cmake_lines.extend([
                f'target_sources(cubutterfly_heongpu_tfhe PRIVATE "{adapter / "tfhe_bootstrap_overlay.cu"}")\n',
                f'target_compile_definitions(cubutterfly_heongpu_tfhe PRIVATE CUBUTTERFLY_HEONGPU_TFHE_BOOTSTRAP_SOURCE="{bootstrap}" CUBUTTERFLY_TFHE_SHARED=1)\n',
            ])
        if args.device_test:
            cmake_lines.extend([
                f'add_executable(cubutterfly_heongpu_tfhe_device_test "{ROOT / "tests/test_heongpu_tfhe_device.cu"}")\n',
                f'target_sources(cubutterfly_heongpu_tfhe_device_test PRIVATE "{heongpu_source / "src/lib/kernel/small_ntt.cu"}")\n',
                f'target_include_directories(cubutterfly_heongpu_tfhe_device_test PRIVATE "{ROOT / "adapters/heongpu"}" "{heongpu_source / "src/include"}" "{heongpu_source / "thirdparty/GPU-NTT/src/include"}")\n',
                'target_link_libraries(cubutterfly_heongpu_tfhe_device_test PRIVATE CUDA::cudart)\n',
                'target_compile_features(cubutterfly_heongpu_tfhe_device_test PRIVATE cxx_std_17 cuda_std_17)\n',
                f'set_target_properties(cubutterfly_heongpu_tfhe_device_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON CUDA_ARCHITECTURES {args.arch})\n',
            ])
        cmake_file.write_text("".join(cmake_lines))
        if not args.skip_build:
            run([
                str(CMAKE), "-S", str(wrapper_dir), "-B", str(build_dir),
                "--fresh",
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}",
                *([f"-DCMAKE_PREFIX_PATH={prefix}"] if prefix.exists() else []),
                *([f"-DCPM_SOURCE_CACHE={cpm_cache}"] if cpm_cache.exists() else []),
                *([f"-DFETCHCONTENT_SOURCE_DIR_RMM={cpm_cache / 'rmm-src'}"]
                  if (cpm_cache / "rmm-src").exists() else []),
                *([f"-DFETCHCONTENT_BINARY_DIR_RMM={cpm_cache / 'rmm-build'}"]
                  if (cpm_cache / "rmm-build").exists() else []),
                *([f"-DFETCHCONTENT_SOURCE_DIR_RAPIDS_LOGGER={cpm_cache / 'rapids_logger-src'}"]
                  if (cpm_cache / "rapids_logger-src").exists() else []),
                *([f"-DFETCHCONTENT_SOURCE_DIR_SPDLOG={cpm_cache / 'spdlog'}"]
                  if (cpm_cache / "spdlog").exists() else []),
                *([f"-DFETCHCONTENT_BINARY_DIR_RAPIDS_LOGGER={cpm_cache / 'rapids_logger-build'}"]
                  if (cpm_cache / "rapids_logger-build").exists() else []),
                *([f"-DFETCHCONTENT_BINARY_DIR_SPDLOG={cpm_cache / 'spdlog-build'}"]
                  if (cpm_cache / "spdlog-build").exists() else []),
            ])
            run([str(CMAKE), "--build", str(build_dir), "--target",
                 "cubutterfly_heongpu_tfhe",
                 *(["cubutterfly_heongpu_tfhe_device_test"] if args.device_test else []),
                 "-j", str(args.jobs)])
        binary_name = "cubutterfly_heongpu_tfhe_device_test" if args.device_test else "cubutterfly_heongpu_tfhe"
        binary = build_dir / binary_name
        if not binary.exists():
            # CMake may use a configuration subdirectory with a multi-config
            # generator. Keep the lookup explicit and deterministic.
            candidates = list(build_dir.glob(f"**/{binary_name}"))
            if not candidates:
                raise SystemExit(f"built TFHE runner not found under {build_dir}")
            binary = candidates[0]
        if args.build_only:
            print(f"Built {args.backend}: {binary}", flush=True)
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        runtime_env = os.environ.copy()
        runtime_paths = [
            str(cpm_cache / "rmm-build"),
            str(cpm_cache / "rapids_logger-build"),
            str(gf2x_lib.parent),
            str(prefix / "usr/lib/x86_64-linux-gnu"),
        ]
        old_ld = runtime_env.get("LD_LIBRARY_PATH")
        runtime_env["LD_LIBRARY_PATH"] = ":".join(runtime_paths + ([old_ld] if old_ld else []))
        if args.device_test:
            run([str(binary)], env=runtime_env)
            return 0
        run([
            str(binary), "--shapes", args.shapes, "--gates", args.gates,
            "--warmup", str(args.warmup), "--repetitions", str(args.repetitions),
            "--seed", str(args.seed), "--output", str(output),
            "--expected-backend", args.backend,
            "--execution", args.execution, "--graph-replays", str(args.graph_replays),
            "--circuit-depths", args.circuit_depths,
        ], env=runtime_env)
    finally:
        shutil.rmtree(wrapper_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
