# V100 Library/Base/Search Comparison

The external-library context and source links are collected in [Research
Positioning](cubutterfly_positioning.md), [FFT Library Comparison](fft_library_comparison.md),
and [V100 External Baselines](v100_external_baselines.md). This page focuses
on the matched V100 base/search matrix.

## Question

This experiment separates two performance questions:

1. how much the cuButterfly mapping and processing-unit search improves a fixed
   radix-2 configuration; and
2. how the selected implementation compares with a specialized library.

The base row is a valid fixed radix-2 cuButterfly/cuNTT mapping, not a CPU or
correctness reference. The searched row may change the space-time mapping,
radix, exchange path, stage decomposition, and physical processing unit. The
FFT search includes cuFFTDx units because physical-unit selection is an
explicit dimension of the framework.

The fixed mappings are temporal radix-2 at FFT/FWHT `logN=8`, hierarchical
radix-2 at `logN=14/15`, online-reorder radix-2 at FFT `logN=18/20`, and
Hybrid2D radix-2 for NTT. Exact arguments and searched configurations are
versioned in `config/v100_three_way_comparison.json`.

## Protocol

The matrix contains four FP32 FFT shapes, three FP32 FWHT shapes, and three
60-bit natural-order NTT shapes. Local rows use 1000 warmups, 100 timed
iterations, five process trials in randomized order, CUDA-event kernel time,
and correctness checks on a Tesla V100-SXM2-16GB.

FFT/cuFFT and FWHT/Dao-AILab FHT are measured in the current randomized run.
GPU-NTT is imported from the archived matching-protocol run on the same V100
because its comparator binary is not currently installed. Those NTT rows have
the same warmup, repeat, trial, modulus, and output-order contract, but were not
interleaved with this refresh.

## Result

| Operator | Shapes | Search/base geomean | Search/library geomean | Faster | Parity | Slower |
|:--|--:|--:|--:|--:|--:|--:|
| FFT | 4 | 2.931x | 1.015x | 1 | 3 | 0 |
| FWHT | 3 | 3.768x | 1.054x | 2 | 1 | 0 |
| NTT | 3 | 1.089x | 1.313x | 3 | 0 | 0 |
| Overall selected matrix | 10 | 2.349x | 1.109x | 6 | 4 | 0 |

`Faster`, `parity`, and `slower` use a +/-3% band. The complete per-shape table
is [V100 Three-Way Library Comparison](../results/v100_three_way_comparison.md).

The interpretation is narrower than the aggregate number:

- FFT search removes a 1.33x-5.95x fixed-mapping deficit and reaches cuFFT
  parity overall. `logN=18,batch=16` is 1.082x faster, while
  `logN=20,batch=4` remains 0.980x and is classified as parity.
- FWHT search is essential: the fixed hierarchical radix-2 mapping is poorly
  matched to these shapes, while the selected register-resident unit reaches
  parity or a 1.076x-1.085x advantage over Dao FHT.
- NTT radix-4 provides a smaller 1.072x-1.099x gain over fixed radix-2, but the
  selected implementation is 1.118x-1.438x faster than matching natural-order
  GPU-NTT rows.

The overall geomean summarizes this selected representative matrix, not all
lengths, batches, precisions, or semantic modes. FP16, BF16, FP64, Structured
2x2, and subset-zeta full-protocol rows are included in the report as internal
base/search evidence only when no exact external-library row is joined.

## Reproduction

```bash
python3 scripts/run_external_baseline_suite.py \
  --manifest config/v100_three_way_comparison.json \
  --fht-python /home/wt/.conda/envs/cubutterfly-baselines/bin/python \
  --output results/v100_three_way_raw.csv

python3 scripts/summarize_external_baseline_suite.py \
  results/v100_three_way_raw.csv \
  --output results/v100_three_way_summary.csv

python3 scripts/summarize_three_way_comparison.py \
  results/v100_three_way_summary.csv \
  --archived-baselines results/v100_external_baselines_summary.csv \
  --internal-raw results/v100_numeric_coverage_raw.csv \
  --internal-manifest results/v100_numeric_coverage_suite.json \
  --output results/v100_three_way_comparison.csv \
  --internal-output results/v100_three_way_internal.csv \
  --metrics results/v100_three_way_metrics.json \
  --report results/v100_three_way_comparison.md
```
