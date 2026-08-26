# V100 v0.6/v0.7 Comprehensive Comparison

## Scope

This study answers two separate questions. The matched NTT matrix compares the
v0.6 and v0.7 architecture paths at common lengths. The library matrix compares
the release-performance paths with cuFFT, Dao-AILab FHT, and archived GPU-NTT
under matching operator contracts. It does not compare different lengths or
output orders through a derived throughput ratio.

The four NTT roles are:

| Role | `logN=10` | `logN=12` | Meaning |
|:--|:--|:--|:--|
| v0.6 | generic baseline | Hybrid2D radix-2 | fixed pre-search mapping |
| v0.6 search | Tile256 | Hybrid2D radix-4 | selected mature physical path |
| v0.7 | HybridDataflow radix-2 `Ur=1,Td=4` | HybridDataflow radix-2 `Ur=1,Td=4` | initial generated role-pipeline point |
| v0.7 search | resident radix-4 | resident radix-4 | generated table selects the fused physical unit and mapping |

All 18 common shapes use forward natural-order NTTs, 30 warmups, 500 timed
iterations, three randomized process trials, CUDA-event kernel timing, and
correctness checks. They cover uint64 `logN=10/12`, uint32 `logN=12`, and batch
`1,16,80,160,256,512`. uint32 `logN=10` is excluded because v0.6 has no legal
backend for that contract.

Several latency rows at batch 1/16 are below the experiment's 0.020 ms timing
floor. They are retained for coverage, while conclusions about individual
low-latency rows should use the batch 80-512 trend as supporting evidence.

## Version Result

| Numeric/length | Shapes | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search throughput |
|:--|--:|--:|--:|--:|
| uint64 `logN=10` | 6 | 1.974x | 5.445x | 2.127x |
| uint32 `logN=12` | 6 | 1.128x | 4.026x | 0.785x |
| uint64 `logN=12` | 6 | 1.024x | 4.918x | 0.651x |
| Overall | 18 | 1.316x | 4.760x | 1.028x |

The new physical unit changes the conclusion. v0.7 wins 8 shapes, is at parity
on 1, and v0.6 wins 9. It wins every uint64 `logN=10` point by
1.805x--2.503x and crosses over for uint32 `logN=12` at batch 80/160. It remains
behind on most `logN=12` points, especially uint64 at saturated batch. The
complete per-shape table is in
[`results/v100_v06_v07_r4_refresh.md`](../results/v100_v06_v07_r4_refresh.md).

This isolates the architectural and physical-unit effects. The original
role-pipeline kernel materialized a shared channel and synchronized at each
role edge. The resident radix-4 unit preserves the same one-CTA, on-chip graph
contract but combines two stages per four-point unit and synchronizes once per
stage pair. The 3.769x direct physical-core geomean and 1.028x generation-level
result show that the scheduling paradigm is competitive once supplied with a
mature local unit. The remaining `logN=12` deficit comes from assigning a
larger resident state to one CTA: unlike Hybrid2D, it cannot expose multiple
CTAs within one transform while retaining strict shared-memory residency.

## Library Result

The current full-protocol refresh uses 1000 warmups, 100 iterations, five
randomized process trials, and correctness checks. GPU-NTT is the existing
same-V100 matching-protocol archive because its comparator is not installed on
this host.

| Operator | Shapes | Searched/base | Searched/library | Faster | Parity | Slower |
|:--|--:|--:|--:|--:|--:|--:|
| FP32 FFT / cuFFT | 4 | 2.914x | 1.019x | 1 | 3 | 0 |
| FP32 FWHT / Dao FHT | 3 | 3.741x | 1.052x | 2 | 1 | 0 |
| uint64 NTT / GPU-NTT | 3 | 1.091x | 1.314x | 3 | 0 | 0 |
| Combined representative matrix | 10 | 2.339x | 1.111x | 6 | 4 | 0 |

The combined geomean is not a universal superiority claim. FFT is at cuFFT
parity on all four selected shapes. The mature v0.6 NTT path is 1.118x-1.438x
faster than the archived natural-order GPU-NTT rows at `logN=16`. Exact-shape
specialization removes general-shape bounds checks from the FWHT register hot
path; the `logN=15` batch 256/512 rows are now 1.076x/1.077x faster than Dao.
The detailed refresh is
[`results/v100_three_way_v07_r4_final.md`](../results/v100_three_way_v07_r4_final.md).

HybridDataflow cannot be assigned a GPU-NTT ratio yet: its calibrated whole-
transform resident range ends at `logN=12`, while the matching GPU-NTT archive
starts at `logN=16`. The next performance milestone is therefore to compose
the v0.7 streaming schedule with v0.6 processing units and multi-CTA subgraphs,
then repeat a common `logN=16` external comparison.

## Reproduction

```bash
python3 scripts/run_external_baseline_suite.py \
  --manifest config/v100_v06_v07_comparison.json \
  --output results/v100_v06_v07_r4_refresh_raw.csv
python3 scripts/summarize_external_baseline_suite.py \
  results/v100_v06_v07_r4_refresh_raw.csv \
  --output results/v100_v06_v07_r4_refresh_summary.csv
python3 scripts/summarize_v06_v07_comparison.py \
  results/v100_v06_v07_r4_refresh_summary.csv \
  --output results/v100_v06_v07_r4_refresh_comparison.csv \
  --metrics results/v100_v06_v07_r4_refresh_metrics.json \
  --markdown results/v100_v06_v07_r4_refresh.md
```

The library refresh uses the commands in
[`v100_three_way_comparison.md`](v100_three_way_comparison.md), with output
names prefixed by `v100_three_way_v07_r4_final`.
