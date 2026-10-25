# Weak-region quick replay (2026-10-23)

This is a bounded A100 80GB replay after the APPT descriptor and coefficient
contract audit. It is a diagnostic result, not a replacement for the full
campaign. The run used GPU2 (`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`), the
`build-e13-cufftdx` build, CUDA-event steady-state timing, warmup 2 or 5, and
10 or 20 repetitions. Compilation and plan construction were outside the
reported kernel interval.

## Automatic xD plan replay

The configured `plan_bench` matrix completed with finite outputs and matching
verification. Its FFT FP32 rows selected `xd[cufft-whole]`, so their near-parity
numbers are a cuFFT whole-transform route and do not measure the local
hybrid-stage kernel. The FP64 row selected the axis-tiled mixed lowering.

| Case | Selected lowering | cuButterfly ms | Reference ms | Ratio | Correct |
| --- | --- | ---: | ---: | ---: | --- |
| FFT FP32, 512×512, B=16 | `xd[cufft-whole]` | 0.097915 | 0.098284 cuFFT | 1.0038× | yes |
| FFT FP32, 1024×1024, B=4 | `xd[cufft-whole]` | 0.100024 | 0.100762 cuFFT | 1.0074× | yes |
| FFT FP64, 256×256, B=64 | `xd[axis-tiled-mixed]` | 0.179036 | 0.182354 cuFFT | 1.0185× | yes |
| NTT word64, N=65,536, B=64 | composite | 0.226877 | 0.291512 GPU-NTT | 1.2849× | yes |
| NTT word64, N=1,048,576, B=4 | composite | 0.354386 | 0.378388 GPU-NTT | 1.0677× | yes |

The FWHT FP32 128×256, B=4 and NTT 256×256, B=4 controls also completed;
these rows have no external baseline in this manifest. Raw records are under
`results/weak_regions_20261023_quick/results.json/`.

## Forced local FFT paths

To expose the local hybrid path hidden by the automatic whole-FFT selection, I
replayed two larger cases with the configured online-reorder and cuFFTDx block
cores, then ran the matching cuFFT backend. These are the current weak areas.

| Case | Local path | Local ms | cuFFT ms | cuButterfly speedup vs cuFFT | Verified |
| --- | --- | ---: | ---: | ---: | --- |
| FFT FP32, N=1,048,576, B=4 | online-reorder, 10+10, cuFFTDx block | 0.138445 | 0.115968 | 0.8377× | one batch |
| FFT FP64, N=65,536, B=64 | online-reorder, 8+8, cuFFTDx block | 0.254157 | 0.191283 | 0.7521× | one batch |

The public `--auto-select` path chose `register-tile` for the FP32 case at
0.164762 ms and cuFFTDx block for FP64 at 0.237158 ms. Matching ten-repeat
cuFFT runs were 0.114688 ms and 0.185242 ms, respectively, corresponding to
cuButterfly speedups of 0.6958× and 0.7817×. The selection log
said “first executable candidate ... no local timing estimate”, so these two
rows identify a selector/calibration gap: the current auto path is not choosing
from measured local candidates for these weak shapes.

For FWHT FP32, N=1,048,576, B=4, hierarchical radix4 measured 0.205722 ms
with one-batch verification, versus 0.212685 ms for online-reorder under the
same short replay. This is an internal organization control, not an external
library comparison.

The forced FFT paths were numerically accepted under the existing same-precision
verification. These measurements do not claim that the 13 omitted APPT NTT
variants are portable candidates; their status remains documented in the APPT
variant audit.

## Rank-one FFT runner repair

The original weak-region manifest sent every plan through `--execution-mode xd`.
That is invalid for rank one because public `xd` is the multidimensional
lowering policy. The runner now records the execution mode per case, defaults
rank one to `auto`, and forwards the manifest's bounded search budget. A
budgeted `policy=measure` replay on GPU2 completed both previously excluded
rank-one cases and verified all batches:

| Case | Search budget | Selected measured core | cuButterfly ms | cuFFT ms | Speedup vs cuFFT | Correct |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| FFT FP64, N=65,536, B=64 | 4 | online-reorder / cuFFTDx block | 0.235179 | 0.183979 | 0.7823× | yes |
| FFT FP32, N=1,048,576, B=4 | 4 | online-reorder / register-tile | 0.165205 | 0.113664 | 0.6880× | yes |

These are measured candidates under a deliberately small budget, not claims of
full-space optimality. They show that the rank-one coverage gap was a runner
configuration problem, while the remaining performance gap is a real local
kernel/selection issue. The manifest and runner changes are shared by all
butterfly operators; only the candidate inventory and semantic execution mode
vary by operator and rank.

A separate 4K-point smoke test also confirmed the persistence path: the first
`policy=measure` run wrote a 1,666-byte exact mapping cache, and a subsequent
`policy=default` run reused the same mapping without emitting a new
`measure-complete` event. This is the path used by the benchmark runner's
output-local `tuning/cache.json`.

## Manual split sweep and profile evidence

The fact that large NTT instances often benefit from splitting does not imply
that the current FFT split has the same access pattern. We therefore forced the
FFT candidates instead of relying on the selector and measured the same
1024×1024 FP32 transform at several batch sizes. The whole-transform route
remained faster at every batch size:

| Batch | `cufft-whole` ms | `axis-tiled-cufftdx` ms | Split / whole |
| ---: | ---: | ---: | ---: |
| 1 | 0.02703 | 0.03543 | 1.31× |
| 4 | 0.10045 | 0.13548 | 1.35× |
| 16 | 0.36434 | 0.48343 | 1.33× |
| 64 | 1.39069 | 1.85457 | 1.33× |

The exchange inventory does contain a better local point than the default
`axis-tiled-cufftdx`: the EPT=16 unrolled variant reached 0.12370 ms for B=4,
or 0.793× of cuFFT. Its batch sweep was 0.794×, 0.800×, 0.848×, and 0.844×
for B=1, 4, 16, and 64. It is a useful near-term kernel baseline, but it still
does not reach parity and does not change the complete-plan winner.

The result is not caused by a candidate-order shortcut. A bounded
`policy=measure` search for 1024×1024, B=4 timed the complete plans and chose
`xd[cufft-whole]` at about 0.0969 ms, against about 0.0983 ms for cuFFT. The
forced split path took about 0.135 ms. The current local-candidate inventory
also explains why the best small case does not generalize: the recurrence local
core is compiled only for short axes (64, 128, and 256). On 256×256, B=16 it
reached 0.02181 ms versus 0.02058 ms for the whole route, while the standard
local and capped variants were slower. For a 1024-point axis that candidate is
not executable, so the selector cannot choose it without adding a new compiled
variant.

Nsight Systems and Nsight Compute separate the remaining issue from occupancy.
For 1024×1024, B=4, a warm-up plus three-repeat trace reported the following
steady-state kernel summaries. The profiler's absolute timings include replay
overhead and are excluded; the normal CUDA-event timings above are the timing
source.

| Route | Kernel structure | Registers/thread | Active warps | DRAM throughput | Long-scoreboard stall |
| --- | --- | ---: | ---: | ---: | ---: |
| Whole | `regular_fft` + `vector_fft` | 56 | ≈45% | ≈52–71% | ≈13–22% |
| Split, contiguous axis | `xd_axis_kernel`, `bool0` | 40 | ≈66–68% | ≈66–71% | ≈22–24% |
| Split, strided axis | `xd_axis_kernel`, `bool1` | 40 | ≈66–68% | ≈29% | ≈37–38% |

The split kernels use fewer registers and expose more active warps, yet the
strided axis has both low global-memory throughput and the highest
long-scoreboard stall. This is the concrete reason that higher occupancy does
not turn the split into a faster FFT. The whole route spends more registers and
has lower occupancy, but its specialized regular/vector kernels preserve the
layout and schedule the twiddle and complex multiply path more effectively.

This also identifies the boundary of the current implementation. The mixed
dataflow idea remains applicable, but the present FFT lowering has not yet
materialized the next-axis-aligned write/read layout for the strided axis. The
next engineering experiment is therefore a layout-aligned suffix or a fused
transpose/twiddle boundary, followed by the same fixed-shape sweep and trace.
Simply forcing `axis-tiled-cufftdx` is not a valid optimization, and the current
measure-first selector is behaving correctly by retaining `cufft-whole` until a
new axis kernel wins a complete-plan measurement.

## NTT layout mechanism cross-check and follow-up candidates

The NTT implementation already contains the mechanism that was suspected to be
missing. `ntt_hybrid2d` stages a subgraph in padded shared memory and writes the
next layout with a transposed index. The APPT pipeline additionally uses a
32×32 shared transpose at a fold boundary and computes the next token layout
from the stage offsets. These writes make the next NTT stage consume a
coalesced address pattern.

The FFT path only reuses the local part of that idea. `xd_axis_kernel` stages
one strided axis in shared memory, but exports it in the same axis-major global
layout. The `axis-packed-cufftdx` family then inserts an independent tiled
permutation before the next axis and another permutation to restore the public
layout. Its 1024×1024 FP32, B=4 measurement was 0.1890 ms versus 0.1000 ms for
cuFFT. The no-permutation `axis-tiled-cufftdx` path was 0.1373 ms, while the
best currently compiled no-permutation point
(`axis-tiled-cufftdx-ept16-unrolled`) was 0.1237 ms. Moving the permutation into
a separate packed path does not yet reproduce the NTT benefit.

The existing pair subgraph is a useful small-shape control, but it is not a
large-axis solution. `pair-tiled-cufftdx` is compiled for FP32 axes up to 64
(FP64 up to 32); at 64×64, B=64 it measured 0.01536 ms versus 0.01116 ms for
cuFFT. At 1024×1024 it is correctly rejected because a full pair does not fit
the compiled shared/register resource contract. A trial 32×16 CTA variant of
the shared transpose was also measured and reverted after reaching 0.1890 ms,
slower than the existing 32×8 launch.

These controls narrow the implementation work into four explicit candidates:

1. **Fused axis-pair boundary.** Let the first axis kernel write the next-axis
   tile layout, then let the next kernel consume it without a separate global
   permutation. This is the direct FFT analogue of the NTT fold boundary.
2. **Hierarchical large-axis subgraph.** A 1024-point axis cannot be fixed by
   simply compiling a 1024-point local core. The local codelet would need
   additional 512/1024 twiddle tables and a multi-tile stage schedule. The
   existing pair template is therefore retained as the small-axis oracle,
   while large axes require a staged subgraph template.
3. **Twiddle/layout fusion.** FFT twiddles are already precomputed or generated
   by recurrence in the local core. That coefficient choice is orthogonal to
   the global address problem; the useful experiment must combine the twiddle
   multiply with the boundary write, then compare against the same layout with
   a separate multiply.
4. **Measured candidate registration.** Any new fused or hierarchical core
   must enter the complete-plan search with its layout, boundary launches, and
   workspace in the mapping. It must win the fixed 1024×1024 B=4/B=16 sweep
   before becoming eligible for the default cache.

The current result therefore confirms the user's observation: the NTT
cross-step layout method exists, but it has not yet been promoted into the FFT
axis boundary. The remaining work is a real fused subgraph implementation,
not another selector rule or a larger transpose launch.
