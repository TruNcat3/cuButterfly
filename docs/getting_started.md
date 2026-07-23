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

# Generated CTA DFT8 design point
./build/cubutterfly_bench --operator fft --backend temporal-tile \
  --precision fp32 --fft-core cta-dft8 --compute-unit radix8 \
  --logN 8 --tile-threads 64 --batch 16384 --verify --csv

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
```

## C++ API

The public headers are `include/cuntt/ntt.hpp` and
`include/cuntt/butterfly.hpp`. Both APIs use immutable plan configuration and
typed host vectors:

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

The current API owns allocation and transfers for trustworthy reference and
benchmark execution. A device-pointer API is future work.

## Design-Space Sweeps

```bash
python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft xor-zeta \
  --logNs 8 10 12 --target-points 4194304 \
  --trials 5 --output results/my_sweep_raw.csv

python3 scripts/summarize_cubutterfly_designs.py \
  results/my_sweep_raw.csv --output results/my_sweep_summary.csv
```

Use `--verify` for individual development points. Full sweeps rely on the test
suite and should be followed by targeted verification of selected mappings.
