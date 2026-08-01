# FFT Pipeline Generator

The FFT generator separates four decisions that must not be collapsed into one
kernel implementation.

## Layers

1. **Factorization topology** selects a decomposition count `D` and an ordered
   stage vector `S=[s0,...,s(D-1)]`, constrained by `sum(S)=logN`. In the
   machine-readable schema these are `decomposition_count` and
   `stages_per_decomposition`. Neither `D`
   nor an individual `si` is fixed by the architecture paradigm.
2. **Processing unit mapping** describes one decomposition segment: supported stage count,
   thread and element-per-thread choices, exchange method, twiddle method, and compiled
   availability. `cufftdx-online-shared` covers local `logN=3..10`, while
   `cufftdx-large-online` adapts the direct unit for `logN=11..12`. A suffix
   direct unit independently selects either a strided natural-order store or a
   contiguous store followed by a padded tiled transpose. A prefix direct unit
   can instead transpose input into a workspace and consume contiguous rows.
   Each segment independently selects threads and elements per thread. The model derives
   units/CTA, grid blocks, shared storage, estimated resident CTAs, waves/SM,
   issued thread slots, and work imbalance.
3. **Boundary mapping** describes each of the `D-1` connections: twiddle
   generation, permutation, residency, synchronization, and physical handoff.
   A `fused` boundary remains inside one processing group; a `global-scratch`
   boundary terminates the group and materializes its output.
4. **Dispatch** is generated only from measured winners. It selects a mapping by
   `(GPU, semantics, logN, batch regime)` and then reuses the normal runtime
   validation path.

`config/v100_fft_pipeline.json` is the search specification.
`scripts/generate_fft_pipeline.py` emits the bounded candidate manifest. The
specification supplies ranges for `D` and `si`, not a list of splits; the
generator enumerates ordered integer compositions before lowering supported
topologies. Known measured incumbents are mandatory candidates so that static
pruning cannot exclude an existing strong point. Each selected row records the legal pool size
and shortlist limit so that pruning remains auditable. Prefix and suffix units
are selected independently. Prefix/suffix names appear only in the current
`D=2` lowering and are not part of the general architecture abstraction. The
current runtime accepts the same topology as `--stage-partition s0,s1,...`,
plus `--segment-threads`, `--segment-ept`, `--boundary-twiddle`, and
`--boundary-layout`. `--boundary-residency` controls logical-boundary fusion,
while `--group-threads` and `--group-ept` map the resulting physical groups.

The lowering derives an ordered physical partition `G=[g0,...,g(P-1)]` by
coalescing logical segments across fused boundaries. Thus `P <= D`,
`sum(G)=sum(S)=logN`, and the global handoff count is `P-1`, not `D-1`.
For example, logical `7+3+10` with boundary residency
`fused,global-scratch` lowers to physical `10+10`. It executes the same
optimized two-group path as native `D=2`, while preserving the D3 topology in
the architecture and experiment metadata. This separation is essential: `D`
expresses the algorithmic space/time decomposition, whereas `P` expresses a
hardware-dependent residency decision.

For `logN=18/20`, `D=2..4`, and `si=3..12`, the current manifest contains 385
factorization topologies:

| logN | `D=2` | `D=3` | `D=4` |
|--:|--:|--:|--:|
| 18 | 7 | 55 | 84 |
| 20 | 5 | 69 | 165 |

All 385 topologies have a compiled runtime lowering. The bounded performance
manifest selects 288 internal candidates from this topology/mapping product;
it preserves every `D=2` split and directional boundary realization, then
retains multiple `D=3/4` segment-mapping and physical-lowering classes under a
per-shape limit. Measured D2 incumbents are also injected as physical-group
mappings for compatible D3/D4 lowerings, including their boundary twiddle mode.

## V100 Search Result

The current scan contains 882 samples: 294 cases, three randomized trials per
case, 100 warmups, and 50 timed repetitions. Times are resident kernel times on
the repository V100. All 294 cases pass verification. A ratio above 1 means
cuButterfly is faster.

The complete `D=2` scan confirms that the earlier balanced points were winners
rather than assumptions. Best recurrence time per `logN=18` partition at batch
16 is:

| Stages | 6x12 | 7x11 | 8x10 | 9x9 | 10x8 | 11x7 | 12x6 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ms | 0.302141 | 0.305316 | 0.236872 | **0.205722** | 0.243384 | 0.324792 | 0.336261 |

| logN | Batch | Selected mapping | cuButterfly ms | cuFFT ms | Ratio |
|--:|--:|:--|--:|--:|--:|
| 18 | 2 | `9+9`, `256/256`, EPT `8/8` | 0.026051 | 0.028099 | 1.079x |
| 18 | 16 | `9+9`, `256/256`, EPT `8/8` | 0.205722 | 0.216146 | 1.051x |
| 18 | 64 | `9+9`, `256/256`, EPT `8/8` | 0.842875 | 0.717599 | 0.851x |
| 20 | 2 | `10+10`, `512/512`, EPT `8/8` | 0.128799 | 0.123556 | 0.959x |
| 20 | 8 | `10+10`, `256/128`, EPT `16/16` | 0.450109 | 0.382157 | 0.849x |
| 20 | 16 | `10+10`, `256/128`, EPT `16/16` | 0.883835 | 0.720855 | 0.816x |

The original generic multi-segment baseline is correct but not competitive when
each logical segment is a separate global-scratch pass. Per-segment mapping still
matters: the second scan replaced the initially selected uniform
`512-thread/EPT16` points with smaller codelets and improved `D=3` by roughly
15--28%. Best measured points by decomposition count are:

| logN | Batch | `D=2` ms / cuFFT | `D=3` stages, mapping, ms / cuFFT | `D=4` ms / cuFFT |
|--:|--:|:--|:--|:--|
| 18 | 2 | 0.026051 / 1.079x | `3+3+12`, `128/128/512`, 0.068301 / 0.411x | 0.114934 / 0.244x |
| 18 | 16 | 0.205722 / 1.051x | `3+3+12`, `128/128/512`, 0.485929 / 0.445x | 0.868803 / 0.249x |
| 18 | 64 | 0.842875 / 0.851x | `3+5+10`, `128/128/128`, 1.725727 / 0.416x | 3.172168 / 0.226x |
| 20 | 2 | 0.128799 / 0.959x | `7+3+10`, `128/128/128`, 0.306196 / 0.404x | 0.545915 / 0.226x |
| 20 | 8 | 0.450109 / 0.849x | `7+3+10`, `128/128/128`, 1.046466 / 0.365x | 1.800069 / 0.212x |
| 20 | 16 | 0.883835 / 0.816x | `7+3+10`, `128/128/128`, 2.050929 / 0.351x | 3.581092 / 0.201x |

These rows isolate the cost of setting `P=D`. Merely increasing `D` while
materializing every boundary in global memory is predictably bandwidth-bound.

The controlled fused-boundary scan then holds the physical group partition,
threads/EPT, recurrence twiddle, and direct-strided global boundary constant.
Only the logical decomposition metadata and fused internal boundaries change:

| logN | Batch | D2/P2 ms / cuFFT | D3/P2 ms / cuFFT | D4/P2 ms / cuFFT | D3/P3 ms / cuFFT | D4/P4 ms / cuFFT |
|--:|--:|:--|:--|:--|:--|:--|
| 18 | 2 | 0.026051 / 1.079x | 0.026030 / 1.079x | 0.026010 / 1.080x | 0.081039 / 0.347x | 0.114913 / 0.245x |
| 18 | 16 | 0.205722 / 1.051x | 0.205681 / 1.051x | 0.205496 / 1.052x | 0.716513 / 0.302x | 0.868434 / 0.249x |
| 18 | 64 | 0.842875 / 0.851x | 0.836690 / 0.858x | 0.839721 / 0.855x | 2.613678 / 0.275x | 3.176305 / 0.226x |
| 20 | 2 | 0.128799 / 0.959x | 0.128655 / 0.960x | 0.128573 / 0.961x | 0.371220 / 0.333x | 0.545833 / 0.226x |
| 20 | 8 | 0.450109 / 0.849x | 0.445338 / 0.858x | 0.446628 / 0.856x | 1.411707 / 0.271x | 1.799455 / 0.212x |
| 20 | 16 | 0.883835 / 0.816x | 0.868844 / 0.830x | 0.867799 / 0.831x | 2.792632 / 0.258x | 3.578204 / 0.201x |

The D2/P2, D3/P2, and D4/P2 columns agree within normal run-to-run variation
(at most 1.8% here). This is the central result of the lowering experiment:
logical decomposition count has no intrinsic execution penalty when it maps to
the same physical schedule. The large D3/P3 and D4/P4 loss is attributable to
extra global handoffs. Fusion does not itself improve the selected arithmetic
core; it prevents the architecture abstraction from imposing avoidable passes.

The boundary experiment compares the same `9+11`, 128/256-thread, EPT 16/8,
recurrence point. Both rows include every launched kernel.

| Batch | Direct-strided ms | Tiled-transpose ms | Tiled change | Tiled/cuFFT |
|--:|--:|--:|--:|--:|
| 2 | 0.169226 | 0.161690 | 4.66% faster | 0.764x |
| 8 | 0.579461 | 0.583332 | 0.67% slower | 0.655x |
| 16 | 1.133466 | 1.150116 | 1.47% slower | 0.627x |

The tiled realization makes the suffix FFT write its contiguous scratch row in
place, then launches a 32x32 shared-memory transpose with a 33-element pitch.
It improves the low-batch point but does not change the selected `10+10`
mapping. At saturated batch, the saved strided stores do not repay an extra
full global read/write and third launch. The result isolates the next target:
fuse the tiled permutation into an adjacent pass or retain a transposed layout
across more work, instead of adding another standalone arithmetic core.

`8+12` tiled recurrence reaches 0.170271, 0.611820, and 1.201418 ms at batch
2, 8, and 16.

The prefix-side experiment uses the same padded transpose before the direct
prefix, a dedicated workspace, and the existing twiddle/scratch epilogue. It
therefore measures the cost of converting strided input into a contiguous
processing-unit contract without changing arithmetic:

| Split | Batch | Direct-strided ms | Prefix transpose ms | Prefix change |
|:--|--:|--:|--:|--:|
| `11+9` | 2 | 0.173691 | 0.197345 | 13.62% slower |
| `11+9` | 8 | 0.675103 | 0.733348 | 8.63% slower |
| `11+9` | 16 | 1.370194 | 1.480540 | 8.05% slower |
| `12+8` | 2 | 0.181473 | 0.213893 | 17.86% slower |
| `12+8` | 8 | 0.655667 | 0.856842 | 30.68% slower |
| `12+8` | 16 | 1.314959 | 1.944105 | 47.84% slower |

The negative result is useful: coalescing one direct-unit boundary does not
justify a standalone full-array pass. A profitable realization must fuse the
permutation with a producer/consumer or preserve the transposed layout across
additional work. Neither directional transpose changes the current measured
dispatch winners.

## Commands

```bash
python3 scripts/generate_fft_pipeline.py \
  --output config/v100_fft_pipeline_candidates.json

python3 scripts/run_fft_pipeline.py --resume \
  --manifest config/v100_fft_pipeline_candidates.json \
  --binary build/cubutterfly_bench \
  --output results/v100_fft_pipeline_raw.csv

python3 scripts/summarize_fft_pipeline.py \
  results/v100_fft_pipeline_raw.csv \
  --output results/v100_fft_pipeline_summary.csv \
  --markdown results/v100_fft_pipeline_report.md

python3 scripts/generate_fft_dispatch.py \
  --summary results/v100_fft_pipeline_summary.csv \
  --output config/v100_fft_dispatch.json
```

At runtime, `--auto-select` currently accepts only the calibrated V100 SM70
semantics: FP32 forward, in-place, contiguous, no normalization, and
`logN=18/20`. The build must enable cuFFTDx. Unsupported hardware, build
capability, or semantics fail explicitly.
