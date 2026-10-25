# FFT Library Comparison

The historical sections isolate the FP32 complex-to-complex FFT gap on the V100.
The separate xD update below uses a matched A100 80GB protocol. Results from these
different populations and hardware targets are not pooled.

## A100 xD independent-axis experiment, v15, October 5

The [v15 report](../results/paper_profile_rc2/end_to_end/xd_axes_v15/README.md)
adds per-axis resource mappings and one bounded coordinate sweep over existing
cores. All forty rows of the same ten forward weak probes pass. No new mixed
mapping beats the promotion threshold; selected throughput changes by -0.78%
to +1.44% relative to old physical mappings executed in the same new binary.
This is a completed coverage improvement and a negative performance result on
these probes. It does not update the full64 aggregate below or establish that
all independent-axis combinations have been exhausted.

## A100 xD bounded core experiment, v14.1, October 5

The [v14/v14.1 report](../results/paper_profile_rc2/end_to_end/xd_register256_v14/README.md)
extends register-axis support to length 256 and replaces small-radix dynamic
root lookups with equivalent rotations. All 80 rows of the ten predeclared
forward probes pass. At 256²/B16 the forced register plans improve by
8.2–26.5% over the initial v14 implementation, but remain slower than existing
selected plans. No new core wins automatic selection in these probes.
NCU confirms fewer constant loads, while FP64 instruction expansion and low
active-warps utilization remain. This bounded experiment does not replace the
v13.1 full-population statistics below, and no aggregate improvement is claimed.

## A100 xD update, v13.1, October 5

The [v13.1 resource report](../results/paper_profile_rc2/end_to_end/xd_resources_v13/README.md)
adds EPT4 and range-checked 32-bit element indexing to the existing measured
search. It retains the same 64 cases, GPU, independent libraries and timing
protocol as v12. All 192 comparison rows pass, and compilation/search stay
outside timing.

| Population | Cases | cuButterfly / cuFFT throughput | cuButterfly / VkFFT throughput |
| --- | ---: | ---: | ---: |
| Original forward control | 16 | 1.0627× | 0.9565× |
| Complete fixed population | 64 | 1.1616× | 0.9255× |
| Forward | 36 | 1.0382× | 0.9250× |
| Normalized inverse | 28 | 1.3420× | 0.9261× |
| FP32 | 32 | 1.1173× | 0.8800× |
| FP64 | 32 | 1.2076× | 0.9733× |

These paired geometric means show another small improvement over v12, with
23/64 selections using new resource options. Seventeen points improve by more
than 3% over v12 and none regress by more than 3%. cuButterfly wins 48/64
against cuFFT, 13/64 against VkFFT, and is fastest of all three in 11/64.
Inverse normalization fusion contributes to the cuFFT comparison. These are
single-campaign results, not repeated-campaign significance or a universal lead.

Matched NCU explains why simply reducing registers is insufficient: EPT4 can
increase CTA count, global-load sectors and shared conflicts at larger sizes.
Narrow indices reduce address instructions, while the remaining imported-core
exchange cost persists. All profiled local-load/store sectors are zero. The
report retains both improvements and negative candidates; independent VkFFT
still exposes a gap, particularly in FP32.

## A100 xD update, v12, October 5

The [v12 exchange report](../results/paper_profile_rc2/end_to_end/xd_exchange_v12/README.md)
retains the exact v11 population, A100 80GB UUID, independent libraries and
20-call/7-replay graph protocol. New direct and unrolled exchange candidates
reuse the imported arithmetic core. All 192 comparison rows are correct;
compilation and search stay outside timing.

| Population | Cases | cuButterfly / cuFFT throughput | cuButterfly / VkFFT throughput |
| --- | ---: | ---: | ---: |
| Original forward control | 16 | 1.0560× | 0.9474× |
| Complete fixed population | 64 | 1.1417× | 0.9097× |
| Forward | 36 | 1.0267× | 0.9121× |
| Normalized inverse | 28 | 1.3087× | 0.9067× |

These paired geometric means improve on v11 below. cuButterfly wins 48/64
against cuFFT and 10/64 against VkFFT, and is fastest of all three in 8/64.
Fused normalization contributes to the inverse advantage over native cuFFT's
separate scaling launch. This is limited evidence for an overall gain against
cuFFT on this fixed population; it does not imply general superiority over FFT
libraries or repeated-campaign statistical significance.

Frozen-mapping NCU shows that unrolling reduces strided-stage instructions
while increasing registers; direct exchange can instead double global sectors.
The report preserves negative candidates and the validated 61+3 checkpoint
recovery after temporary external GPU use. Only safe completed data are reused.

## A100 xD update, v11, October 4

The [fixed 64-case xD report](../results/paper_profile_rc2/end_to_end/xd_block_v11/README.md)
adds independent VkFFT 1.3.4 to native cuFFT 11.2.1. The new optional
`vkfft_xd_bridge` accepts the same out-of-place C2C buffers, stream, precision,
rank and direction as the public C-ABI benchmark. Graph correctness, input
preservation and normalized inverse are checked; compilation/planning is excluded.

| Population | Cases | cuButterfly / cuFFT throughput | cuButterfly / VkFFT throughput |
| --- | ---: | ---: | ---: |
| Original forward control | 16 | 1.0073× | 0.9054× |
| Fixed expansion | 64 | 1.0546× | 0.8396× |
| Expanded forward | 36 | 0.9599× | 0.8540× |
| Expanded normalized inverse | 28 | 1.1901× | 0.8215× |

These are paired geometric means, not a mixture of library-specific ratios.
All 192 timed rows passed correctness. Across all 64 configurations, cuButterfly
beats both baselines in six. Independent VkFFT exposes additional xD gaps that
cannot be inferred from the historical 1D ordering. The inverse gain over cuFFT
also includes fused normalization versus native cuFFT's separate scaling call.
One campaign does not establish a statistically significant overall lead.

The complete comparison registry is maintained in
[`config/external_comparison_catalog.json`](../config/external_comparison_catalog.json)
and can be checked with `python3 scripts/validate_comparison_catalog.py`. It
contains direct baselines, imported processing units, end-to-end contexts, and
method references. Only entries marked `measured` or `measured-local-core` are
eligible for local aggregates.

## Baseline Scope

| Implementation | Relevant abstraction | V100 status | Comparison role |
|:--|:--|:--|:--|
| [NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/) | complete general-purpose GPU FFT | measured | vendor reference |
| [VkFFT 1.3.4](https://github.com/DTolm/VkFFT) | complete cross-platform GPU FFT | measured at commit `066a17c` | direct external baseline |
| [NVIDIA cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/) | device-side thread/block FFT building blocks | MathDx 24.4 adapter measured on SM70 | imported local processing unit |
| [TurboFFT](https://github.com/shixun404/TurboFFT) | generated complete FFT research artifact | commit `918179c`, T4 points ported and measured on SM70 | imported generated processing unit |
| [FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv) | fused long convolution using FFT | Ampere/Hopper workload | not a raw V100 C2C FFT baseline |
| [tcFFT](https://arxiv.org/abs/2104.11471) | Tensor-Core half-precision FFT and changing layouts | source-only | processing-unit context; no matching local adapter |
| [FFTX/SpectralPack](https://www.spiral.net/doc/papers/hipc_2018.pdf) | symbolic transform composition and code generation | source-only | method context; no local timing |
| [FFTW](https://www.fftw.org/fftw-paper-ieee.pdf) | adaptive planning and codelets | source-only | CPU planning context; not a GPU baseline |
| [SPIRAL](https://doi.org/10.1109/JPROC.2004.840306) | symbolic transform search and generation | source-only | generator context; no matching local GPU contract |
| [HadaCore](https://arxiv.org/abs/2412.08832) | Tensor-Core Hadamard processing unit | source-only | processing-unit context; no local adapter |
| [GPU-NTT Merge/Four-Step](https://eprint.iacr.org/2023/1410) | NTT factorization and shared-memory reuse | source-only | NTT method context; runnable GPU-NTT is listed above |
| [TensorIR](https://doi.org/10.1145/3575693.3576933) | block signature and tensorization separation | source-only | generic compiler method context |
| [TVM](https://www.usenix.org/conference/osdi18/presentation/chen) | tensor compiler and schedule separation | source-only | generic compiler method context |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | hierarchical sketches and learned search | source-only | generic search method context |
| [ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu) | hardware-aware tile construction and micro-modeling | source-only | generic mapping method context |
| [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) | tile graph and storage-level connecting | source-only | generic layout/mapping method context |
| [Timeloop](https://doi.org/10.1109/ISPASS.2019.00042) | workload, architecture and mapping separation | source-only | generic dataflow method context |
| [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html) | randomized Hadamard transform in quantization | source-only | end-to-end application context |

cuFFTDx belongs in the processing-unit design space: importing one of its local
FFTs would test the mapping paradigm with a stronger arithmetic core. It should
not be ranked as a complete long-transform library until the surrounding
composition is specified. FlashFFTConv similarly answers a fused-operator
question rather than the raw FFT question measured here.

The same rule applies to tcFFT, HadaCore, and TurboFFT when they are used as
local cores. FFTX, TensorIR, Ansor, and Welder explain generation or mapping
ideas and are useful comparison points for the method section, but they do not
become throughput baselines without a matched executable contract.

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
