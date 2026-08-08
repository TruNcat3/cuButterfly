# Getting Started

## Requirements

- Linux with an NVIDIA GPU
- CUDA Toolkit with `nvcc` and cuFFT
- CMake 3.20 or newer
- a C++17 host compiler supported by the installed CUDA version
- Python 3 for generators, sweeps, and result summarization
- optional: Nsight Compute, Nsight Systems, PyTorch, and external baselines

The checked-in V100 profile was built with CUDA 11.8 and `sm_70`. Building for
another GPU is supported, but the V100-selected mappings are not portable
performance defaults.

On hosts with multiple CUDA installations, CMake may otherwise find an older
`/usr/bin/nvcc`. The published clean-clone validation uses CUDA 11.8 and GCC 11.
CUDA 11.5 with GCC 11 fails in the standard library before compiling project
code; select a compatible CUDA/host-compiler pair explicitly.

## Configure And Build

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j
```

Select a different generated processing-unit specification with:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCUNTT_DESIGN_SPEC="$PWD/config/v100_design_points.json"
```

For a new GPU, copy the JSON specification and GPU profile instead of editing
the V100 records in place.

### Optional FFT Processing Units

cuFFTDx and TurboFFT are optional and the default build remains dependency
free beyond the CUDA Toolkit. Install the pinned versions and enable both
adapters with:

```bash
./scripts/install_cufftdx.sh
./scripts/install_turbofft.sh

cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCUBUTTERFLY_ENABLE_CUFFTDX=ON \
  -DCUBUTTERFLY_ENABLE_TURBOFFT=ON
cmake --build build -j
```

The cuFFTDx adapter supports FP32 forward/inverse block FFTs at `logN=3..10`
and direct-strided online dimensions at `logN=11..12`, including strided and
in-place plan semantics. The current TurboFFT artifact
adapter supports the generated FP32 forward, contiguous, out-of-place kernels
actually present upstream at `logN=7..10`.

## Test

```bash
cmake --build build --target test
```

This runs `cuntt_tests` and `cubutterfly_tests`. The tests compare CUDA results
against CPU references across lengths, directions, radices, layouts, tails,
generated units, and supported numeric types.

Optional memory checking:

```bash
compute-sanitizer --tool memcheck ./build/cuntt_tests
compute-sanitizer --tool memcheck ./build/cubutterfly_tests
compute-sanitizer --tool racecheck ./build/cubutterfly_tests
```

## NTT CLI

```bash
./build/cuntt_bench --help
```

Examples:

```bash
# Generic local tile
./build/cuntt_bench --logN 16 --batch 1 --backend tile256 --verify

# Hardware-parameterized two-pass mapping
./build/cuntt_bench --logN 20 --batch 4 --backend hybrid2d \
  --n1-log 10 --rows-per-block 2 --threads-per-block 512 \
  --compute-unit radix4 --cross-twiddle fused \
  --mod-multiply shoup --verify --csv

# Native bit-reversed compact path
./build/cuntt_bench --logN 20 --batch 4 --backend compact-stage \
  --modulus 576460756061519873 \
  --output-order bit-reversed --verify --csv
```

Kernel timing excludes host/device transfers. The CSV also reports transfer and
end-to-end rates.

## Common Butterfly CLI

```bash
./build/cubutterfly_bench --help
./build/cubutterfly_bench --list-capabilities
```

Examples:

```bash
# FP64 inverse FFT with padded, strided, in-place storage
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp64 --compute-unit radix8 --complex-multiply gauss3 \
  --logN 10 --inverse --normalization inverse --placement in-place \
  --element-stride 2 --batch-stride 2061 --tile-threads 256 \
  --batch 17 --verify

# BF16-storage FWHT with FP32 accumulation. On V100, BF16 native-width
# arithmetic is emulated and labeled in CSV output.
./build/cubutterfly_bench --operator fwht --backend temporal-tile \
  --precision bf16 --accumulation fp32 --compute-unit radix4 \
  --logN 8 --batch 16384 --verify --csv

# Generated CTA DFT8 design point
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp32 --fft-core cta-dft8 --compute-unit radix8 \
  --logN 8 --tile-threads 64 --batch 16384 --verify --csv

# Imported cuFFTDx block processing unit
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp32 --fft-core cufftdx-block --logN 9 \
  --tile-threads 32 --batch 8192 --verify --csv

# Imported TurboFFT generated processing unit
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp32 --fft-core turbofft-generated --logN 9 \
  --tile-threads 32 --batch 8192 --placement out-of-place \
  --normalization none --verify --csv

# Long two-pass cuFFTDx composition with register cross-twiddle recurrence
./build/cubutterfly_bench --operator fft --backend online-reorder \
  --precision fp32 --fft-core cufftdx-block --logN 20 \
  --local-stages 10 --cross-twiddle recurrence \
  --prefix-threads 512 --prefix-ept 8 \
  --suffix-threads 512 --suffix-ept 8 \
  --tile-threads 32 --reorder-columns 1 --batch 4 --verify --csv

# Independently search both FFT dimensions on the current GPU
./scripts/explore_fft_architecture.py --logNs 16 18 20 \
  --target-points 4194304 \
  --output-prefix results/fft_architecture_explore_v100

# Fully resident 64x64 schedule: one CTA, one launch, no global scratch traffic
./build/cubutterfly_bench --operator fft --backend online-reorder \
  --precision fp32 --fft-core cufftdx-resident --logN 12 \
  --local-stages 6 --cross-twiddle recurrence --tile-threads 1024 \
  --reorder-columns 1 --batch 1024 --verify --csv

# Whole-transform equivalent processing unit used by the same runtime
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp32 --fft-core cufftdx-direct --logN 12 \
  --tile-threads 512 --batch 1024 --verify --csv

# Long online-reorder FFT
./build/cubutterfly_bench --operator fft --backend online-reorder \
  --precision fp32 --compute-unit radix4 --logN 20 \
  --local-stages 10 --reorder-columns 1 --tile-threads 256 \
  --batch 4 --verify --csv

# Register FWHT
./build/cubutterfly_bench --operator fwht --backend temporal-tile \
  --precision fp32 --local-exchange warp-register \
  --logN 15 --batch 128 --verify --csv

# Vendor FFT baseline
./build/cubutterfly_bench --operator fft --backend cufft \
  --precision fp32 --logN 20 --batch 4 --csv

# Subset zeta and Mobius inverse over uint32 mask-indexed data
./build/cubutterfly_bench --operator subset-zeta --backend online-reorder \
  --compute-unit radix4 --logN 20 --local-stages 10 --batch 4 --verify --csv
./build/cubutterfly_bench --operator superset-zeta --backend hierarchical \
  --compute-unit radix4 --logN 20 --local-stages 10 --batch 4 --inverse --verify --csv

# A stage-parameterized real 2x2 butterfly; one matrix is broadcast.
./build/cubutterfly_bench --operator structured-2x2 \
  --stage-matrix 1,0.25,-0.5,1 --precision fp32 \
  --backend temporal-tile --local-exchange warp-register \
  --compute-unit radix2 --logN 12 --batch 1024 --verify --csv
```

## C++ API

The public headers are `include/cuntt/ntt.hpp` and
`include/cuntt/butterfly.hpp`. Applications can install and consume the CMake
package without depending on the source tree:

```bash
cmake --install build --prefix "$HOME/.local"
```

```cmake
find_package(cuButterfly CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cuButterfly::cuButterfly)
```

The primary integration path accepts caller-owned device pointers and a CUDA
stream. Query and optionally bind the exact workspace before submitting work:

```cpp
#include "cuntt/butterfly.hpp"

cuntt::ButterflyConfig config;
config.op = cuntt::ButterflyOperator::Fwht;
config.backend = cuntt::ButterflyBackend::OnlineReorder;
config.precision = cuntt::ButterflyPrecision::Fp32;
config.log_n = 15;
config.batch = 16;
config.auto_select = true;
config.auto_allocate_workspace = false;

cuntt::ButterflyPlan plan(config);
plan.set_stream(stream);
cudaMalloc(&workspace, plan.workspace_size());
plan.set_workspace(workspace, plan.workspace_size());
plan.execute_async(device_input, device_output);
```

`execute_async` performs no allocation, transfer, or synchronization. Input,
output, workspace, and stream lifetimes remain the caller's responsibility.
The complete contract and NTT equivalent are in [Device API](device_api.md).
The calibrated coverage and decision metadata are described in [Runtime
Mapping Selector](runtime_selector.md).

The [Programming Guide](programming_guide.md) defines device association,
layout, concurrency, CUDA Graph status, determinism, and resource lifetime.
Use the [C++ API Reference](api_reference.md) for individual configuration
fields and methods, the [Capability and Compatibility
Matrix](support_matrix.md) for supported ranges, and [Examples](examples.md)
for build-checked programs.

Typed host vectors remain convenient for reference execution and benchmark
timing:

```cpp
#include "cuntt/butterfly.hpp"

cuntt::ButterflyConfig config;
config.op = cuntt::ButterflyOperator::Fwht;
config.backend = cuntt::ButterflyBackend::TemporalTile;
config.precision = cuntt::ButterflyPrecision::Fp32;
config.log_n = 10;
config.batch = 32;
config.compute_unit = cuntt::ComputeUnit::Radix4;

cuntt::ButterflyPlan plan(config);
std::vector<float> input(config.batch * (1U << config.log_n));
std::vector<float> output;
const auto stats = plan.execute(input, output, 20, 100);
```

The host overload owns its staging allocations and reports transfer and kernel
timings; it should not be used to measure application submission overhead.

## Design-Space Sweeps

```bash
python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft subset-zeta superset-zeta structured-2x2 \
  --logNs 8 10 12 --target-points 4194304 \
  --trials 5 --output results/my_sweep_raw.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/my_sweep_raw.csv --output results/my_sweep_summary.csv
```

Use `--verify` for individual development points. Full sweeps rely on the test
suite and should be followed by targeted verification of selected mappings.

With both optional FFT units enabled, reproduce the V100 local-unit comparison
and derive the per-core medians with:

```bash
./scripts/benchmark_fft_processing_units.sh
./scripts/benchmark_fft_cufftdx_long.sh
```
