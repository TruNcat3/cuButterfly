# General-Shape V100 Selection Results

## Composition Selector Follow-up

The second experiment adds direct cuFFT as an exact FFT processing-unit
candidate, propagates physical-axis selection into embedded/rank-two plans,
and lets modular Bluestein use the selected NTT core. Each row below is the
median of five independent warmed processes with 50 warmups and 100 timed
executions. Plan creation, tuning, cache I/O, and host transfers are excluded.

| Operator | Logical shape | Batch | Selected composition | Initial ms | Selected-core ms | Direct-output ms | Fused-input ms | Total gain |
|:--|:--|--:|:--|--:|--:|--:|--:|--:|
| FFT FP32 | 1000 | 64 | direct cuFFT | 0.02622 | 0.00427 | 0.00427 | 0.00427 | 6.14x |
| FFT FP32 | 30x50 | 64 | direct cuFFT | 0.07993 | 0.00981 | 0.00981 | 0.00981 | 8.15x |
| NTT uint64 | 5419 | 1 | Bluestein + Hybrid2D radix-4 | 0.08608 | 0.03147 | 0.03147 | 0.03147 | 2.74x |
| NTT uint64 | 5419 | 64 | Bluestein + Hybrid2D radix-4 | 0.79822 | 0.21323 | 0.21323 | 0.21323 | 3.74x |
| FWHT FP32 embedding | 32767 -> 32768 | 256 | warp-register radix-2, fused load | 0.73772 | 0.32409 | 0.24271 | **0.11759** | **6.27x** |
| FWHT FP32 embedding | 255 -> 256 | 65536 | warp-register radix-2, fused load | 0.60309 | 0.58422 | 0.41231 | **0.16511** | **3.65x** |
| subset zeta embedding | 32767 -> 32768 | 256 | hierarchical radix-4 | 0.73606 | 0.73606 | 0.64882 | 0.64882 | 1.13x |
| Structured FP32 embedding | 32767 -> 32768 | 256 | measured warp-register radix-2 | 0.74460 | 0.34425 | 0.25772 | not selected | 2.89x |

Direct FFT is at timing parity with the matching cuFFT invocation: 0.998x at
`N=1000,batch=64` and 1.000x at `30x50,batch=64`. This is intentional
processing-unit reuse, not a claim that a wrapper around cuFFT outperforms
cuFFT. Runtime `MEASURE` still compares the complete direct and Bluestein
compositions and records the winner, while explicit Bluestein remains
available for architecture experiments.

There is no matching exact arbitrary-length GPU-NTT baseline in the archived
external suite. Relative to this repository's direct power-of-two core at the
same physical `M=16384`, modular Bluestein costs 2.97x at batch 1 and 2.72x at
batch 64. That is now close to its two transforms plus clear/pack/multiply/output
boundaries; the previous excess came from hard-coding baseline radix-2 instead
of reusing Hybrid2D radix-4.

FWHT, subset/superset zeta, and Structured 2x2 use zero-extended embedding,
not an exact `N`-point transform. For a contiguous physical output, the second
implementation removes the final scatter: rank-one floating/integer plans
write their last core directly to the caller output, while rank-two plans make
the final transpose land there. Strided outputs retain the materialized
scatter path. Algorithm names expose this decision as `+direct-output`.

For FP32 warp-register FWHT, the generated first processing unit now accepts
the logical length and independent input/output strides. It predicates the
tail to zero while loading the useful values, so no pack kernel or composition
buffer is required. The algorithm name exposes this as `+fused-input`. At
`P=32768,batch=256`, the complete logical-shape wrapper is 0.11759 ms, within
1.01x of the archived 0.11643 ms bare Dao core despite accepting `N=32767`
input batches rather than pre-padded input. At `P=256,batch=65536`, removing
the pack boundary gives a further 2.50x over the direct-output version.

This is a selected lowering, not a universal rule. Misaligned logical batch
distances force scalar loads in some batches. A two-aligned-vector stitching
candidate was measured and rejected because redundant memory transactions
increased latency. Structured 2x2 also retains the pack path: its tested fused
candidate was slower than the direct-output composition. The generator keeps
the specialized entry as a candidate for future aligned/padded input contracts,
but the default C plan only emits `+fused-input` for the measured-beneficial
FWHT family.

Subset and superset zeta remain symmetric at 0.64882/0.64900 ms and have no
semantics-matched external GPU baseline.

The repeated raw rows are
[`results/general_shape_selection_v100_raw.csv`](../results/general_shape_selection_v100_raw.csv)
and
[`results/general_shape_direct_boundary_v100_raw.csv`](../results/general_shape_direct_boundary_v100_raw.csv).
The fused-boundary trials are
[`results/general_shape_fused_input_v100_raw.csv`](../results/general_shape_fused_input_v100_raw.csv).

## Initial Bluestein-only Diagnostic

The first v0.6 diagnostic forced exact arbitrary FFT through Bluestein. It used
one aggregated CUDA-event run per row with 20 warmups and 100 timed executions.

| Shape | Batch | Bluestein physical shape | cuButterfly ms | cuFFT ms | cuButterfly/cuFFT |
|:--|--:|:--|--:|--:|--:|
| 5 | 1 | 16 | 0.01534 | 0.00274 | 0.179x |
| 5 | 4096 | 16 | 0.01878 | 0.00271 | 0.144x |
| 1000 | 1 | 2048 | 0.02048 | 0.00337 | 0.165x |
| 1000 | 64 | 2048 | 0.02622 | 0.00428 | 0.163x |
| 3x5 | 4096 | 8x16 | 0.06553 | 0.00690 | 0.105x |
| 30x50 | 1 | 64x128 | 0.04690 | 0.00788 | 0.168x |
| 30x50 | 64 | 64x128 | 0.07993 | 0.00979 | 0.122x |

Those rows are in
[`results/general_shapes_v100_smoke.csv`](../results/general_shapes_v100_smoke.csv).
Rank one performed chirp/pack, forward power-of-two cuFFT, pointwise multiply,
inverse cuFFT, and chirp/output. Rank two repeated that sequence per axis and
added transpose boundaries. The result established that a fast physical core
does not by itself make an efficient composition.

The follow-up implements processing-unit abstention, physical-core selection,
direct physical output, and a selected fused input boundary for warp-register
FWHT. Remaining work is to make boundary residence a searched property for
more processing units and to retain padded layouts across adjacent library
calls so applications can avoid repeated embedding work.
