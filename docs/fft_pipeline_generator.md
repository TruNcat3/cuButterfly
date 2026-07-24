# FFT Pipeline Generator

The FFT generator separates three decisions that must not be collapsed into one
kernel implementation.

## Layers

1. **Processing unit** describes one local FFT: supported dimension, thread and
   element-per-thread choices, exchange method, twiddle method, and compiled
   availability. `cufftdx-online-shared` covers local `logN=3..10`, while
   `cufftdx-large-online` adapts the direct unit for `logN=11..12`.
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

The current scan contains three randomized trials per point, 100 warmups and 50
timed repetitions. Times are resident kernel times on the repository V100. A
ratio above 1 means cuButterfly is faster.

| logN | Batch | Selected mapping | cuButterfly ms | cuFFT ms | Ratio |
|--:|--:|:--|--:|--:|--:|
| 18 | 2 | `9+9`, `256/256`, EPT `8/8` | 0.026051 | 0.028099 | 1.079x |
| 18 | 16 | `9+9`, `256/256`, EPT `8/8` | 0.205722 | 0.216146 | 1.051x |
| 18 | 64 | `9+9`, `256/256`, EPT `8/8` | 0.842875 | 0.717599 | 0.851x |
| 20 | 2 | `10+10`, `512/512`, EPT `8/8` | 0.128799 | 0.123556 | 0.959x |
| 20 | 8 | `10+10`, `256/128`, EPT `16/16` | 0.450109 | 0.382157 | 0.849x |
| 20 | 16 | `10+10`, `256/128`, EPT `16/16` | 0.883835 | 0.720855 | 0.816x |

The best asymmetric mixed-unit point is `9+11`: it reaches 0.730x, 0.660x, and
0.636x cuFFT throughput at batch 2, 8, and 16. The direct unit is competitive
when applied to a contiguous whole transform, so this regression attributes the
remaining problem to its cross-dimension strided boundary rather than its FFT
arithmetic. The next target is therefore a tiled transpose or swizzled boundary
for the large unit, not another arithmetic core.

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
