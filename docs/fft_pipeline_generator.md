# FFT Pipeline Generator

The FFT generator separates three decisions that must not be collapsed into one
kernel implementation.

## Layers

1. **Processing unit** describes one local FFT: supported dimension, thread and
   element-per-thread choices, exchange method, twiddle method, and compiled
   availability. `cufftdx-online-shared` covers local `logN=3..10`, while
   `cufftdx-large-online` adapts the direct unit for `logN=11..12`. A suffix
   direct unit independently selects either a strided natural-order store or a
   contiguous store followed by a padded tiled transpose. A prefix direct unit
   can instead transpose input into a workspace and consume contiguous rows.
2. **Mapping** factors a transform into prefix and suffix dimensions. Each pass
   independently selects threads and elements per thread. The model derives
   units/CTA, grid blocks, shared storage, estimated resident CTAs, waves/SM,
   issued thread slots, and work imbalance.
3. **Dispatch** is generated only from measured winners. It selects a mapping by
   `(GPU, semantics, logN, batch regime)` and then reuses the normal runtime
   validation path.

`config/v100_fft_pipeline.json` is the search specification.
`scripts/generate_fft_pipeline.py` emits the bounded candidate manifest. Known
measured incumbents are mandatory candidates so that static pruning cannot
exclude an existing strong point. Each selected row records the legal pool size
and shortlist limit so that pruning remains auditable. Prefix and suffix units
are selected independently, so `8+12`, `9+11`, `11+9`, and `12+8` are real
mixed-unit candidates rather than aliases for `10+10`.

## V100 Search Result

The current scan contains 306 samples: 102 cases, three randomized trials per
case, 100 warmups, and 50 timed repetitions. Times are resident kernel times on
the repository V100. A ratio above 1 means cuButterfly is faster. The boundary
column is `direct-strided` for all current winners.

| logN | Batch | Selected mapping | cuButterfly ms | cuFFT ms | Ratio |
|--:|--:|:--|--:|--:|--:|
| 18 | 2 | `9+9`, `256/256`, EPT `8/8` | 0.026051 | 0.028099 | 1.079x |
| 18 | 16 | `9+9`, `256/256`, EPT `8/8` | 0.205722 | 0.216146 | 1.051x |
| 18 | 64 | `9+9`, `256/256`, EPT `8/8` | 0.842875 | 0.717599 | 0.851x |
| 20 | 2 | `10+10`, `512/512`, EPT `8/8` | 0.128799 | 0.123556 | 0.959x |
| 20 | 8 | `10+10`, `256/128`, EPT `16/16` | 0.450109 | 0.382157 | 0.849x |
| 20 | 16 | `10+10`, `256/128`, EPT `16/16` | 0.883835 | 0.720855 | 0.816x |

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
