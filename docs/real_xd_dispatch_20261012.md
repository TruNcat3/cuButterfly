# Real xD dispatch diagnosis (A100 80 GB)

This record separates the cost of the cached cuButterfly operator call from
device throughput. It is not an application end-to-end result: the measured
requests contain only the FFT adapter and its GPU work, with no surrounding
model, preprocessing, or postprocessing.

## Protocol

The calibration record is
`results/current_recalibration_20261012_xd_gpu2/public_calibration.json`.
The before/after test uses the same shared-library hash, plan cache, GPU UUID,
three shapes, seeded GPU-resident inputs, and cached mappings. The old adapter
is loaded from the pre-change snapshot
`results/real_xd_dispatch_20261012/cubutterfly_real_fft_before.py`.
Each operation is measured in six interleaved trials, with 20 warmups and 100
repetitions per CUDA-event sample. Eager timing includes the adapter's input
checks and fresh output allocation but excludes plan creation, calibration,
compilation, and reference checking. CUDA Graph timing is reported separately
as per-captured-call device throughput; it reuses graph-captured allocations
and therefore is not eager latency.

The A/B artifact is
`results/real_xd_dispatch_20261012/ab_retest/timings.csv` and its full
checkpoint is `report.json`. All cache gates passed. Both versions passed the
existing FP64-reference error report, input byte-preservation checks, fresh
output ownership, non-contiguous input, and nested-leading-batch checks.

## Result

Values are microseconds per operation. `before` and `after` are the old and
optimized Python adapters, respectively. `torch` is the matched PyTorch
reference call. The `after/before` column is an engineering dispatch
improvement, while `torch/after` is the matched operator-call ratio, not an
application speedup.

| shape | batch | operation | before | after | torch | after/before | torch/after |
|---|---:|---|---:|---:|---:|---:|---:|
| 64x64 | 8 | forward | 38.00 | 22.11 | 27.39 | 1.72x | 1.24x |
| 64x64 | 8 | inverse | 37.39 | 22.25 | 29.41 | 1.68x | 1.32x |
| 64x64 | 8 | pair | 74.96 | 43.23 | 59.74 | 1.73x | 1.38x |
| 32x32 | 64 | forward | 37.30 | 21.89 | 27.59 | 1.70x | 1.26x |
| 32x32 | 64 | inverse | 37.67 | 22.17 | 29.54 | 1.70x | 1.33x |
| 32x32 | 64 | pair | 75.17 | 43.85 | 60.24 | 1.71x | 1.37x |
| 8x8x8 | 16 | forward | 42.71 | 24.84 | 30.97 | 1.72x | 1.25x |
| 8x8x8 | 16 | inverse | 40.76 | 24.99 | 32.45 | 1.63x | 1.30x |
| 8x8x8 | 16 | pair | 82.51 | 50.36 | 66.24 | 1.64x | 1.32x |

Across these nine requests, the median eager call fell from 40.76 to 24.84
microseconds, a 1.64x reduction in the median latency (the row-wise median
ratio is 1.70x). The optimized adapter is faster than the matched PyTorch
call in every row under this protocol. This is a dispatch result for the
selected cached requests, not a claim about all FFT sizes.

Graph replay makes the remaining device work visible. The median per-call
times are 6.55 microseconds before, 6.54 microseconds after, and 8.79
microseconds for PyTorch. The nearly unchanged before/after graph time is
expected: the code change removes host-side shape/reshape work and does not
change the selected GPU kernels.

## Device evidence

NCU was run on the same A100 80 GB UUID with the public adapter and a single
warmed operation. For the three forward cases, cuButterfly's grids contain
20--68 CTAs while the GPU has 108 SMs. Reported SM issue throughput is
0.32--2.35% and DRAM throughput is 0.07--0.14%; the working sets are small
enough to be largely served by L2. The corresponding inverse cases show the
same pattern. PyTorch also remains below 13% active-warps occupancy in these
small requests. These numbers are diagnostic, not roofline claims: they show
that neither arithmetic throughput nor HBM bandwidth is saturated.

The dominant residual is therefore fixed launch and host submission latency,
plus the small number of blocks available for each request. A larger batch or
captured stream can amortize this fixed cost, but graph replay must be reported
as a separate throughput mode. The current A/B change does not alter the
kernel mapping or the calibrated cost model.

## Scope and remaining boundary

The adapter still creates a fresh output for ordinary calls, so these numbers
preserve the public ownership contract. Serialized callers may explicitly use
`reuse_buffers=True`; callers switching CUDA streams must synchronize before
reusing one plan's workspace. The test does not claim concurrent use of one
plan across streams. A full application benchmark must add the surrounding
lowering, tensor movement, and application work and should report those costs
separately from this operator-call diagnostic.
