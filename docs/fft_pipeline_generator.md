# FFT Pipeline Generator

The FFT generator separates three decisions that must not be collapsed into one
kernel implementation.

## Layers

1. **Processing unit** describes one local FFT: supported dimension, thread and
   element-per-thread choices, exchange method, twiddle method, and compiled
   availability. The current `cufftdx-online-shared` unit covers local
   `logN=3..10`.
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
and shortlist limit so that pruning remains auditable. Unsupported `8+12`,
`9+11`, `11+9`, and `12+8` factorizations for `logN=20` are emitted as explicit
backlog rather than silently mapped to a different kernel.

## V100 Search Result

The current scan contains three randomized trials per point, 100 warmups and 50
timed repetitions. Times are resident kernel times on the repository V100. A
ratio above 1 means cuButterfly is faster.

| logN | Batch | Selected mapping | cuButterfly ms | cuFFT ms | Ratio |
|--:|--:|:--|--:|--:|--:|
| 18 | 2 | `9+9`, `256/256`, EPT `8/8` | 0.025969 | 0.028078 | 1.081x |
| 18 | 16 | `9+9`, `256/256`, EPT `8/8` | 0.205455 | 0.211681 | 1.030x |
| 18 | 64 | `9+9`, `256/256`, EPT `8/8` | 0.839352 | 0.706314 | 0.841x |
| 20 | 2 | `10+10`, `512/512`, EPT `8/8` | 0.128573 | 0.123269 | 0.959x |
| 20 | 8 | `10+10`, `256/128`, EPT `16/16` | 0.448369 | 0.381727 | 0.851x |
| 20 | 16 | `10+10`, `256/128`, EPT `16/16` | 0.869458 | 0.720978 | 0.829x |

This result validates batch-dependent mapping selection but does not establish
general cuFFT superiority. The next performance target is the large-batch
composition boundary: add compiled `logN=11..12` local units, asymmetric
factorizations, and shared-memory padding/swizzle choices, then rerun the same
selection protocol.

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
