# v0.6, v0.8, and Library Reconciliation

This note resolves a recurring comparison ambiguity: a v0.8 mapping label does
not imply that it uses the v0.6 physical core. `grid-compat` is the strict
v0.6-compatible candidate; `online`, `resident-queue`, and generic
`hybrid-dataflow` are different physical realizations and must be measured as
such.

## What The Earlier Claim Means

The statement that v0.6 was "全面超过 cuFFT" was too broad. The controlled
V100 data supports a narrower statement: the searched v0.6/cuButterfly path
reaches parity or wins on selected FP32 FFT shapes, while other precision and
length regimes remain slower. The table below uses the full-protocol rows
(`2^22` total points, 1000 warmups, 100 timed iterations, five trials).

### FFT Versus cuFFT

| precision | logN | batch | cuButterfly ms | cuFFT ms | cuButterfly/cuFFT | interpretation |
|:--|--:|--:|--:|--:|--:|:--|
| FP32 | 8 | 16,384 | 0.084234 | 0.083825 | 0.995x | parity |
| FP32 inverse+normalization | 12 | 1,024 | 0.088064 | 0.172892 | 1.963x | fused normalization; not raw FFT-core speedup |
| FP32 | 14 | 256 | 0.121129 | 0.121610 | 1.004x | parity |
| FP64 | 16 | 64 | 0.534313 | 0.342804 | 0.642x | scalar physical core gap |
| FP32 | 18 | 16 | 0.205087 | 0.211343 | 1.031x | selected cuFFTDx point wins |
| FP32 | 20 | 4 | 0.235438 | 0.210063 | 0.892x | long-path gap |
| FP32 stride-2 | 8 | 16,384 | 0.172636 | 0.171428 | 0.993x | parity |

Thus v0.6 is not universally faster than cuFFT. It is competitive when its
compiled cuFFTDx/direct or long-vector physical unit matches the workload.

### Other Mature Applications

The matching-protocol v0.6/search results are:

| operator | precision | logN | batch | v0.6/search ms | external library ms | v0.6/library | external |
|:--|:--|--:|--:|--:|--:|--:|:--|
| FWHT | FP32 | 8 | 65,536 | 0.165151 | 0.165396 | 1.001x | Dao-AILab FHT |
| FWHT | FP32 | 15 | 256 | 0.108636 | 0.117432 | 1.081x | Dao-AILab FHT |
| FWHT | FP32 | 15 | 512 | 0.195328 | 0.210289 | 1.077x | Dao-AILab FHT |
| NTT | uint64 | 16 | 4 | 0.029420 | 0.032881 | 1.118x | GPU-NTT-Merge |
| NTT | uint64 | 16 | 64 | 0.274883 | 0.386632 | 1.407x | GPU-NTT-Merge |
| NTT | uint64 | 16 | 256 | 1.054403 | 1.522080 | 1.444x | GPU-NTT-Merge |

These are the results behind the historical v0.6 library claim; they are not
evidence that every v0.6 backend or every shape beats its library baseline.

## Why A v0.8 Compatibility Row Can Be Slower

There are two different cases in the repository.

1. **Strict compatibility (`grid-compat`).** In the NTT matrix, this path
   imports the mature v0.6 Hybrid2D launch and physical core. Its measured
   ratio is essentially one: 0.999--1.002x v0.6 across the confirmed
   uint64 `logN=16` batch-4/64 points. It should not be slower except for
   normal clock/timing noise.
2. **Compatibility-lowered online/resident paths.** The cross-operator
   butterfly matrix labels these points
   `compatibility-lowered-two-segment`, but they use the online two-segment
   implementation, not the v0.6 kernel. They add a stage-boundary handoff,
   a global or shared intermediate representation, and an additional launch.
   At batch 16 this fixed cost is visible; at batch 64 it is mostly amortized.

The new native generic path is a third case. It keeps a complete transform
state in dynamic shared memory and uses a fixed generated point
(`flow_tile=8`, `stage_space=4`, `data_space=32`, `data_time=1`, radix-4 in
the current envelope). It is an architecture proof point, not the result of a
per-shape oracle search.

### Native Generic v0.8 Versus Matched v0.6 (batch=64)

| operator | precision | logN | v0.6 ms | native v0.8 ms | v0.8/v0.6 |
|:--|:--|--:|--:|--:|--:|
| FFT | FP32 | 10 | 0.012595 | 0.008679 | 1.451x |
| FFT | FP32 | 12 | 0.030618 | 0.021223 | 1.443x |
| FWHT | FP32 | 10 | 0.008397 | 0.004813 | 1.745x |
| FWHT | FP32 | 12 | 0.018663 | 0.010829 | 1.723x |
| subset-zeta | uint32 | 10 | 0.008320 | 0.004583 | 1.816x |
| subset-zeta | uint32 | 12 | 0.018279 | 0.010240 | 1.785x |
| superset-zeta | uint32 | 10 | 0.008269 | 0.004762 | 1.736x |
| superset-zeta | uint32 | 12 | 0.018458 | 0.010727 | 1.721x |
| xor-zeta | uint32 | 10 | 0.008269 | 0.004634 | 1.785x |
| xor-zeta | uint32 | 12 | 0.018381 | 0.010240 | 1.795x |
| structured-2x2 | FP32 | 10 | 0.008960 | 0.005402 | 1.659x |
| structured-2x2 | FP32 | 12 | 0.019559 | 0.011725 | 1.668x |

At `logN=14`, the generic resident state shows the expected occupancy
crossover: for FWHT and zeta, v0.8/v0.6 is about `0.46--0.76x` at batch 1/16,
but `1.35--1.43x` at batch 64. This is a resident-state amortization effect,
not a contradiction in the mapping principle.

FFT also has a direct vendor comparison in the same native manifest:

| logN | batch | native v0.8 ms | cuFFT ms | native v0.8/cuFFT |
|--:|--:|--:|--:|--:|
| 10 | 1 | 0.006580 | 0.004608 | 0.700x |
| 10 | 64 | 0.008679 | 0.005095 | 0.587x |
| 12 | 1 | 0.017741 | 0.006887 | 0.388x |
| 12 | 64 | 0.021223 | 0.008141 | 0.384x |

These native generic FFT numbers must not be compared to the older v0.6
cuFFTDx rows as if they used the same processing unit. The current build has
`CUBUTTERFLY_ENABLE_CUFFTDX=OFF`, so the optional cuFFTDx physical units are
not available in this run.

## Was The Missing Performance Model The Cause?

Partly, but it is not the only cause.

- The NTT physical-chain model already parameterizes logical segment count,
  resident group count, CTA allocation, worker service, and readiness terms.
  It explains why `grid-compat` is the safe fallback and why resident queues
  can lose at some batch sizes.
- The generic butterfly runtime selector now evaluates a parameterized model
  over the generated resident radix-4 point and the mature hierarchical
  radix-4 point.  Its score separates arithmetic service, global-memory
  traffic, startup, shared-memory capacity, and batch-amortization terms, and
  records the selected point and predicted latency in `SelectionInfo`.
  The first dispatch envelope is intentionally restricted to measured
  `logN=10/12/14` cells; broader searches over `stage_space`, `data_space`,
  `data_time`, role stages, buffers, radix, exchange mode, and layout remain a
  planned calibration expansion rather than an unvalidated extrapolation.
- A model cannot remove an unavailable physical core. FP64 scalar FFT and
  FP32 complex resident state are limited by arithmetic throughput and shared
  memory, respectively. At V100, FP32 complex generic state fails the dynamic
  shared-memory attribute at `logN=14` with the current layout, while real or
  integer state reaches that length.
- The online compatibility path has real boundary and launch costs. Those are
  not search noise and should be represented as explicit model terms.

The correct deployment policy is therefore:

```text
strict v0.6-compatible point when no validated v0.8 candidate wins
model-guided search over physical core + mapping for each supported cell
resident generic path only inside the device capacity envelope
```

Raw evidence is in
`results/v08_comprehensive_*`, `results/cross_operator_v08_gpu_native/`, and
`results/cross_operator_v08_gpu_native14/`. The native profile definition is
[`config/v100_native_operator_profiles.json`](../config/v100_native_operator_profiles.json).
