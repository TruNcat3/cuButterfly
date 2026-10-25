# Open-source FFT template exploration, 2026-10-23

This note records the two bounded routes explored after the FFT layout audit.
The purpose is to identify reusable template mechanisms and to measure them
under the same cuButterfly plan protocol. It is not a claim that an external
library can be copied into the runtime without adapting its layout contract.

## External implementations reviewed

The source-level review used the public implementations and documentation of
[VkFFT](https://github.com/DTolm/VkFFT),
[rocFFT](https://github.com/ROCm/rocFFT),
[NVIDIA cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/), and the
[NVIDIA cuFFTDx samples](https://github.com/NVIDIA/CUDALibrarySamples).

The reusable mechanisms are:

| Source | Mechanism | What can be imported into cuButterfly |
| --- | --- | --- |
| VkFFT | Generated radix stages, register-resident codelets, radix shuffle, and four-step axis decomposition | Stage-local codelet choices, twiddle policy, and an explicit layout state at each handoff |
| rocFFT | Stockham generators and separate transpose/twiddle kernels | A stage-aware generator interface and a searchable continuity/tile parameter |
| cuFFTDx | Device-side block/thread FFT building blocks | A high-throughput local arithmetic core inside a cuButterfly resident tile |
| CUDALibrarySamples | Examples that place the local FFT inside application-owned shared memory and synchronization | A reference for the core/provider boundary and resource validation |

These systems do not establish a universal winner for the complete xD plan.
VkFFT and rocFFT own the whole generated schedule, while cuFFTDx exposes only a
local transform. The cuButterfly abstraction therefore remains useful: the
operator contract, stage graph, core provider, and global layout boundary stay
separate, and the complete plan is measured before selection.

## Route A: dependency-closed shared axis pair

The existing `xd_pair_fft` candidate keeps two physically trailing complex axes
inside one CTA. The first axis writes a padded shared tile, the second axis
consumes the transposed view, and the final write restores the natural layout.
Both axes use the cuFFTDx Thread codelet. The candidate is exposed through
`xd:pair-tiled-cufftdx`; cooperative register variants remain separate choices.

All timings below are CUDA-event steady-state medians on GPU 2, NVIDIA A100
80GB PCIe, SM80, with verification enabled. They exclude plan construction and
compilation. Each row used three trials, eight warmups, and thirty measured
repetitions.

| Shape | Precision | Batch | Route A ms | cuFFT ms | cuButterfly/cuFFT | Correct |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 32x32 | FP32 | 64 | 0.006007 | 0.007919 | **1.318x** | yes |
| 32x32 | FP64 | 64 | 0.007748 | 0.009455 | **1.220x** | yes |
| 64x64 | FP32 | 64 | 0.014882 | 0.010377 | 0.697x | yes |

The result isolates the intended mechanism. A dependency-closed pair removes a
global intermediate and one launch, so a small tile can beat the complete
vendor plan. At 64x64, the same implementation becomes resource limited by
register and shared-memory live state. This is a valid searchable candidate,
not a universal replacement. The existing layout and correctness regressions
pass (`26` cooperative-pair cases, zero threshold failures; three CTest pair
tests pass).

## Route B: generated stage-local core and twiddle policy

The local ThreadShared core uses the same mathematical factorization as a
generated Stockham/radix template: a register-local codelet, an explicit
twiddle policy, and a shared handoff. The current public choices are
`axis-tiled-local` and `axis-tiled-local-recurrence`.

| Shape | Precision | Batch | Constant-root ms | Recurrence-root ms | cuFFT ms | Best ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 256x256 | FP32 | 16 | 0.027341 | **0.021606** | 0.020019 | 0.926x |
| 256x256 | FP64 | 16 | — | **0.039424** | 0.036591 | 0.928x |
| 256x256 | FP32 | 64 | **0.114381** | 0.161894 | 0.101171 | 0.880x |

The recurrence policy improves the short batch point, but loses at the larger
batch in this run. The reason is not arithmetic correctness: all rows pass the
same output check. The recurrence choice changes register/liveness and the
cross-axis access pattern, so it must remain a searchable parameter together
with batch tile, writer layout, and CTA occupancy. A fixed “recurrence is
better” bonus would be wrong.

The long 1D weak point also demonstrates why online tuning is necessary. A
fresh `policy=measure` run for FP32 `N=65536, B=16` did not finish within the
180-second bounded window, so it produced no cache entry and no performance
claim. This is a search-cost issue, not evidence that the candidate is
unsupported. The installation path should use a bounded seed set and defer the
full coordinate search to post-install tuning.

## Deep integration prototype: stage-local generated pair

To test whether an open-source-style stage generator can be combined with the
cuButterfly resident-subgraph contract, `stockham_subgraph_bench` adds an
experimental FP32 32x32 pair. It performs a shared-memory bit-reversal,
five explicit radix-2 ping-pong stages for each axis, and captures the whole
CTA kernel in a CUDA Graph. The current prototype uses a constant-memory
root table and warp synchronization for the independent row stages; the
strided column stages retain CTA synchronization. This is a **stage-aware
prototype inspired by the radix/Stockham generators**, not a copy of VkFFT or
rocFFT source, and it is intentionally outside the public selector until its
resource model is calibrated.

The same buffers, stream, Graph replay protocol, and random input were used for
the prototype, the existing cuFFTDx pair, and cuFFT:

| Planes | Stage-local prototype | cuFFTDx pair | cuFFT | Prototype/cuFFT | Pair/cuFFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.00967 ms | 0.00526 ms | 0.00617 ms | 1.57x | 0.85x |
| 64 | 0.00992 ms | 0.00552 ms | 0.00673 ms | 1.47x | 0.82x |
| 256 | 0.02141 ms | 0.00605 ms | 0.00942 ms | 2.27x | 0.64x |

The prototype's relative errors are `2.04e-7`, `2.99e-7`, and `2.83e-7` for
1, 64, and 256 planes respectively. Constant roots and row warp
synchronization reduce the 64-plane time from the original `0.01219 ms` to
`0.00992 ms`, but the remaining column barriers and scalar radix-2 codelet
still leave it behind cuFFT. The result separates the two issues:
**stage-aware layout alone is insufficient without a competitive local
processing unit**. The next optimization should keep this stage boundary
contract while replacing only the radix-2 codelet with an imported or
generated high-throughput codelet.

The same 64-plane Graph protocol was also run in inverse direction. It
reported `0.009912 ms` for the prototype and `0.006666 ms` for cuFFT, with
relative error `2.96e-7`; the direction switch therefore does not introduce a
separate layout or correctness gap.

## What is incorporated into the template design

The following principles are now treated as implementation requirements:

1. A core provider may be imported from cuFFTDx, VkFFT, rocFFT-derived code,
   or a native template. Provider identity is recorded in the physical mapping.
2. Every stage group carries its input and output layout. A permutation is a
   measured boundary candidate, never an invisible fixed cost.
3. Twiddle generation (`table` or `recurrence`) is searched with the core,
   precision, batch tile, and shared layout. It is not a global preference.
4. A fused pair or resident group is eligible only when its complete plan fits
   the actual shared-memory and register contract. Larger shapes fall back to
   a legal multi-stage plan.
5. External complete-library rows remain independent baselines. Reusing a
   cuFFTDx or VkFFT local core inside a cuButterfly mapping is reported as an
   imported processing unit, not as an external-library speedup.

The next implementation target is a genuinely stage-aware FFT boundary: keep
the imported local codelet, but let its writer emit the next axis' physical
layout directly. This is the missing piece that separates the present
two-kernel axis composition from the NTT writer-aligned path. The selector must
measure the fused boundary against the existing whole-plan and permutation
fallbacks before promoting it.

## Route C: actual VkFFT provider and mixed subgraph

The repository already contains an optional VkFFT bridge. To test a real
open-source generated kernel rather than an approximation, the new
`open_source_subgraph_bench` creates a VkFFT rank-2 plan over the same 32x32
planes, captures its execution in a CUDA Graph, and compares it with cuFFT.
The plan is created before timing, so compile and plan construction are not
charged to the steady-state kernel result. On the same A100 GPU2 protocol:

| Planes | VkFFT Graph | cuFFT Graph | VkFFT/cuFFT | Relative error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004700 ms | 0.006267 ms | **0.75x** | 2.46e-7 |
| 64 | 0.005294 ms | 0.006799 ms | **0.78x** | 2.74e-7 |
| 256 | 0.007025 ms | 0.009523 ms | **0.74x** | 2.68e-7 |

This is the expected kind of evidence for the provider boundary: the
generated open-source core is competitive on this shape, and it can be
captured as a graph without changing the numerical contract. It is reported
as a VkFFT baseline, not as a cuButterfly speedup.

The new `vkfft_hybrid_subgraph_bench` then uses VkFFT for the contiguous row
axis and the cuButterfly staged axis provider for the strided column axis,
capturing both launches as one Graph. This is the direct mixed-provider test
of the cuButterfly organization contract:

| Planes | VkFFT row + cuButterfly column | cuFFT Graph | Hybrid/cuFFT | Relative error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.007690 ms | 0.006216 ms | 1.24x | 2.16e-7 |
| 64 | 0.008243 ms | 0.006758 ms | 1.22x | 2.71e-7 |
| 256 | 0.011080 ms | 0.009482 ms | 1.17x | 2.62e-7 |

The mixed graph is correct but slower because the provider boundary still
materializes a global row intermediate and launches a separate strided-axis
kernel. CUDA Graph removes host launch overhead; it cannot remove that data
movement. The next concrete optimization is therefore to make the imported
provider write the physical layout consumed by the next axis directly, or to
fuse the two providers behind one resident tile. Until that is measured, the
full VkFFT plan remains the stronger external baseline and the mixed route
stays experimental.

## Reproduction

```bash
# Route A
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-cufftdx \
  build-e13-cufftdx/cubutterfly_plan_bench --operator fft \
  --shape 32x32 --batch 64 --precision fp32 \
  --algorithm xd:pair-tiled-cufftdx --compare-cufft --verify \
  --warmup 8 --trials 3 --repeat 30

# Route B
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-cufftdx \
  build-e13-cufftdx/cubutterfly_plan_bench --operator fft \
  --shape 256x256 --batch 16 --precision fp32 \
  --algorithm xd:axis-tiled-local-recurrence --compare-cufft --verify \
  --warmup 8 --trials 3 --repeat 30

# Deep open-source-inspired stage-local prototype and CUDA Graph comparison
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-cufftdx \
  build-e13-cufftdx/stockham_subgraph_bench 64 20 100
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-cufftdx \
  build-e13-cufftdx/stockham_subgraph_bench 64 20 100 1  # inverse

# Actual VkFFT provider and mixed-provider graph (configure with
# -DCUBUTTERFLY_ENABLE_VKFFT=ON -DCUBUTTERFLY_ENABLE_XD_VKFFT=ON)
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-vkfft:$LD_LIBRARY_PATH \
  build-e13-vkfft/open_source_subgraph_bench 64 20 100
CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=build-e13-vkfft:$LD_LIBRARY_PATH \
  build-e13-vkfft/vkfft_hybrid_subgraph_bench 64 20 100
```
