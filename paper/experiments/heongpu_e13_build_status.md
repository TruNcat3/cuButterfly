# HEonGPU E13 Build Status

Date: 2026-10-03 (Asia/Shanghai)

## Scope and source

- Source: `results/e13_assets/sources/HEonGPU`
- HEonGPU commit: `1928a14bb002c5d7f1cb5534eca9fa19b555b4f2`
- Submodules: GPU-FFT `b743607c1173000718ca1f84016fd03093ebb6fc`, GPU-NTT `8a4daf11c7cdea8dde45fa0f8f7183c138d42b3b`, RNGonGPU `d9aaa6b5d7cc67a2ca307ef9d6203f25cc2458b4`, nested RNGonGPU/GPU-NTT `32332ce0f367b7e04612e66ae63da3812fb3b8e3`.
- RMM fetched at `29dd32302eb7c3e16fb837a1cfe4baac98071512` (`branch-25.08`).
- rapids-logger fetched at `46070bb255482f0782ca840ae45de9354380e298`.
- NVTX3 fetched at `69c9949150ac1c310758a304082228a36d5e4758` (3.2.0).
- spdlog pin is `v1.14.1` (upstream tag commit `27cb4c76708608465c413f6d0e6b8d99a4d84302`). Its source was populated from a pinned tarball because the original git clone was too slow; the source tree retains its MIT `LICENSE`.

## Toolchain and parameters

- CMake: `/home/wt/yes/envs/cubutterfly/bin/cmake` 3.30.9.
- CUDA: 12.4.99 (`/usr/local/cuda/bin/nvcc`); host compiler GCC/G++ 9.4.0.
- Build directory: `results/e13_assets/heongpu-build-e13`.
- Actual configuration: `Release`, `CMAKE_CUDA_ARCHITECTURES=80`, `HEonGPU_BUILD_TESTS=OFF`, `HEonGPU_BUILD_EXAMPLES=OFF`, `HEonGPU_BUILD_BENCHMARKS=OFF`.
- Build command: `cmake --build results/e13_assets/heongpu-build-e13 --config Release --parallel 2`.

The build used local, non-system dependency extraction for missing development headers/libraries. Ubuntu focal packages `libgmp-dev` 6.2.0, `libntl-dev` 11.4.3, and `libntl43` 11.4.3 were downloaded and unpacked under `/tmp/heongpu-e13-prefix-20261003`; no system package was installed. The final cache records the GMP include/library paths and the corresponding NTL include/link flags.

## Result

Configuration completed successfully (`configure-e13-gmp2.log`, exit 0). The Release build completed successfully (`build-e13-release.log`, exit 0), including spdlog, rapids_logger, RMM, GPU-FFT, GPU-NTT, RNGonGPU, and HEonGPU. The primary artifact is:

```
results/e13_assets/heongpu-build-e13/src/libheongpu.a
```

Size: 23,334,786 bytes. No tests, examples, benchmarks, or GPU execution were run. Build warnings were limited to CMake policy/deprecation notices and nvcc NTL `tools.h` unused-variable warnings (`#550-D`).

## Exact bootstrap and safe upstream-main suggestion

The pinned-source bootstrap used the following project parameters before local dependency overrides:

```bash
cmake -S results/e13_assets/sources/HEonGPU \
  -B results/e13_assets/heongpu-build-e13 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DHEonGPU_BUILD_TESTS=OFF \
  -DHEonGPU_BUILD_BENCHMARKS=OFF \
  -DHEonGPU_BUILD_EXAMPLES=OFF
```

For a future upstream `main` check on this A100/SM80 host, the safe parameter set is the same explicit Release/SM80/OFF set, with `-DHEONGPU_CUDA_ARCH_FORCE_MANUAL=ON` added to prevent auto-detection from changing the requested architecture. That suggestion is recorded only; no upstream-main build was started in this task.
