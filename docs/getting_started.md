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

The experimental resident graph-streaming backend is selected explicitly:

```bash
./build/cuntt_bench --backend hybrid-dataflow --logN 12 --batch 16 \
  --compute-unit radix4 \
  --flow-tile-log 6 --stage-space 6 --role-stages 6 \
  --data-space 16 --data-time 12 --token-interleave 1 \
  --pipeline-buffers 2 --dataflow-layout hermes-xor \
  --dataflow-state inplace --stage-handoff named-barrier --verify
```

It has zero global workspace and rejects transforms that do not fit in one
CTA's shared memory. It is not selected by `--auto-select`.

The preserved v0.6 two-phase implementation is explicit:

```bash
./build/cuntt_bench --backend hierarchical-barrier \
  --logN 20 --batch 4 --word-bits 32 --modulus 998244353 \
  --n1-log 10 --rows-per-block 4 --data-time 1 \
  --threads-per-block 256 --verify
```

This backend requires cooperative launch support and one full-size workspace
for its online-transposed inter-layer boundary. `n1_log` selects the first
resident graph layer; the second contains `logN-n1_log` stages. Both local
lengths must currently be in `6..10`.

At the generated `10+10` point, add `--hierarchical-core hybrid2d-radix4` to
select the mature Hybrid2D local schedule as an explicit physical-unit
ablation. The default is `dataflow-radix4`.

The v0.7 backend accepts a variable subgraph decomposition and records direct
evidence of overlap:

```bash
./build/cuntt_bench --backend hierarchical-dataflow \
  --logN 15 --batch 2 --stage-partition 5,5,5 \
  --segment-threads 256 --segment-cta-weights 5,5,5 \
  --boundary-storage full-scratch --trace-pipeline --verify
```

For a large batch, `--boundary-storage ring --boundary-buffers 2` bounds each
edge to two transform slots. Epoch tokens prevent a slot from being overwritten
until every downstream task has loaded its prior transform.

Adjacent segments overlap when `pipeline_segment_(i+1).start` is smaller than
`pipeline_segment_i.end`. The streaming ABI provides generic radix-2/4/8
subgraph cores with warp-level data-space unfolding. On V100, `logN=20`
defaults to a generated `10+10` point that imports the mature four-row resident
radix-4 core; use an explicit `7,7,6` partition to study the three-segment
single-transform wavefront. The latter accepts `--segment-units 8|16|32`; its
measured V100 default is 32 units per CTA with `6,6,8` role weights.

`--compute-unit radix4` selects the primary resident physical unit. It combines
two NTT stages before each CTA synchronization while retaining the full state
on chip. `--compute-unit radix2` selects the original persistent-role pipeline
as an architectural ablation; `radix8` is not generated for this backend.

`--role-stages 1` selects the original one-stage-per-role stream.
`--role-stages K` fuses `K` consecutive stages in registers and is available
only for points emitted by the target manifest.
`--token-interleave 2` selects the two-token register software-pipeline
ablation. It is distinct from `--data-time`, which sets packet depth.
`--target-ctas-per-sm K` selects a generated launch-bounds candidate. The V100
default is `1`; higher targets are explicit physical-unit ablations, not runtime
promises.

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

For the unified C plan surface, arbitrary lengths, and rank-two shapes, use:

```bash
# Exact 3x5 FP32 FFT; default selects direct cuFFT when it wins
./build/cubutterfly_plan_bench --operator fft --shape 3x5 --repeat 100
./build/cubutterfly_plan_bench --operator fft --shape 3x5 --compare-cufft --repeat 100

# Retain Bluestein as an explicit research composition
./build/cubutterfly_plan_bench --operator fft --shape 3x5 \
  --algorithm bluestein-cufft-power2-core --repeat 100

# Exact length-3 uint64 NTT; the modulus/root domain is validated
./build/cubutterfly_plan_bench --operator ntt --shape 3 --precision uint64

# One-time search followed by exact-key cache reuse
./build/cubutterfly_plan_bench --operator fwht --shape 4096 --batch 64 \
  --policy measure --cache results/local_selection.cache
./build/cubutterfly_plan_bench --operator fwht --shape 4096 --batch 64 \
  --cache results/local_selection.cache

# Embedded Structured 2x2 with one broadcast matrix per axis
./build/cubutterfly_plan_bench --operator structured-2x2 --shape 32767 \
  --length-mode embedding --matrix 1,0.25,-0.5,1 --policy measure \
  --cache results/local_selection.cache
```

Profile misses are printed with the hardware/workload key and selected
fallback. See [C Plan API](c_api.md) and [General Shape Mapping](general_shapes.md).

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
