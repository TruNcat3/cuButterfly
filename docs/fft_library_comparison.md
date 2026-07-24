# FFT Library Comparison

This report isolates the FP32 complex-to-complex FFT gap on the V100. It does
not treat all projects containing an FFT as equivalent baselines.

## Baseline Scope

| Implementation | Relevant abstraction | V100 status | Comparison role |
|:--|:--|:--|:--|
| [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/) | complete general-purpose GPU FFT | measured | vendor reference |
| [VkFFT 1.3.4](https://github.com/DTolm/VkFFT) | complete cross-platform GPU FFT | measured at commit `066a17c` | direct external baseline |
| [NVIDIA cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/) | device-side thread/block FFT building blocks | MathDx 24.4 adapter measured on SM70 | imported local processing unit |
| [TurboFFT](https://github.com/shixun404/TurboFFT) | generated complete FFT research artifact | commit `918179c`, T4 points ported and measured on SM70 | imported generated processing unit |
| [FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv) | fused long convolution using FFT | Ampere/Hopper workload | not a raw V100 C2C FFT baseline |

cuFFTDx belongs in the processing-unit design space: importing one of its local
FFTs would test the mapping paradigm with a stronger arithmetic core. It should
not be ranked as a complete long-transform library until the surrounding
composition is specified. FlashFFTConv similarly answers a fused-operator
question rather than the raw FFT question measured here.

## Protocol

- Tesla V100-SXM2-16GB, CUDA 11.8 build, FP32 forward C2C.
- Contiguous 1D batches with `2^22` total complex points.
- In-place placement for all three implementations.
- Plan construction excluded; 20 warmups and 100 executions per trial.
- Three trials, median execution time; every row is checked against cuFFT.
- cuButterfly points are selected from the existing design exploration rather
  than tuned against these measurements.

The initial direct measurements are:

| logN | Batch | cuButterfly ms | cuFFT ms | VkFFT ms | cuButterfly / cuFFT throughput | cuButterfly / VkFFT throughput |
|--:|--:|--:|--:|--:|--:|--:|
| 3 | 524288 | 0.091310 | 0.088003 | 0.083610 | 96.4% | 91.6% |
| 8 | 16384 | 0.096082 | 0.084132 | 0.084152 | 87.6% | 87.6% |
| 12 | 1024 | 0.268278 | 0.089702 | 0.088515 | 33.4% | 33.0% |
| 16 | 64 | 0.418406 | 0.180634 | 0.195727 | 43.2% | 46.8% |
| 20 | 4 | 0.639795 | 0.223519 | 0.217252 | 34.9% | 34.0% |

VkFFT is within 8.3% of cuFFT on all five shapes and is faster on three of
them. Therefore the long-transform deficit is not specific to the choice of
cuFFT as the original baseline. Two independent complete FFT libraries define
approximately the same performance envelope on this V100.

This is a steady-state result, not a one-shot latency result. VkFFT generates
CUDA kernels through NVRTC: its median plan time rises from 172 ms at `logN=3`
to 803 ms at `logN=20`, while the measured cuFFT plans take 0.37-0.49 ms.
Applications that cannot cache and reuse plans should include this difference.

The small-transform result also localizes the issue. The generated CTA DFT8
and the radix-4 `N=256` point retain 88%-96% of cuFFT throughput. The loss grows
only when the transform crosses the local-unit boundary. The next optimization
target is consequently long-transform composition: prefix/suffix transaction
count, permutation layout, coefficient traffic, and occupancy. Replacing only
the DFT8 arithmetic is unlikely to recover the missing 2.1x-3.0x. This is now
measured directly: cuFFTDx reaches 99.3%-101.3% of cuFFT throughput at local
`logN=5..10`. Embedding that core into a two-pass tiled-transpose composition
raises long-transform throughput to 41.9%-88.8% of cuFFT at `logN=12..20`.
The remaining loss is now concentrated in the two physical layout edges,
cross-twiddle policy, and hardware-specific FFTs-per-CTA mapping rather than
the local arithmetic core.

A subsequent resident-granularity experiment removes the intermediate global
edge at `logN=12`. The 64x64 composed unit reaches 0.120412 ms, or 73.7% of
cuFFT throughput. Extending the direct whole-transform units through
`logN=11..14` gives stable same-run wins at `logN=11,12,14` (1.006x, 1.012x,
and 1.029x), with `logN=13` at 0.965x. Thus launch fusion alone is insufficient:
the hardware-specific equivalent-unit size and CTA shape are first-class
mapping choices. These are local-unit results; the current `logN=16..20`
two-pass composition remains below cuFFT.

## Reproduction

```bash
./scripts/install_vkfft.sh
cmake -S . -B build -DCUBUTTERFLY_ENABLE_VKFFT=ON
cmake --build build -j

./scripts/benchmark_fft_libraries.sh
./scripts/benchmark_fft_cubutterfly_selected.sh
python3 scripts/summarize_fft_libraries.py

# Optional local processing-unit comparison
./scripts/install_cufftdx.sh
./scripts/install_turbofft.sh
cmake -S . -B build -DCUBUTTERFLY_ENABLE_CUFFTDX=ON \
  -DCUBUTTERFLY_ENABLE_TURBOFFT=ON
cmake --build build -j
./scripts/benchmark_fft_processing_units.sh
./scripts/benchmark_fft_cufftdx_long.sh
```

The raw files are `results/fft_libraries_v100_raw.csv` and
`results/fft_cubutterfly_selected_v100_raw.csv`. The combined derived table is
`results/fft_library_comparison_v100_summary.csv`.
