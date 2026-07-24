# V100 Matching-Protocol External Baselines

## Protocol

This refresh runs each local/external pair with the same workload semantics,
1000 warmups, 100 repetitions, five independently launched process trials,
randomized case order, and correctness verification.

- Dao-AILab `fast-hadamard-transform`: commit
  `e7706faf8d1c3b9f241e36860640ad1dac644ede`, PyTorch 2.1.2 with CUDA 11.8,
  plus the recorded `sm_70` build patch.
- GPU-NTT: commit `d03c5eaeadaa780d153496afcb3b6a9b79a13a63`, built with CUDA 11.8 for
  `sm_70`.
- GPU-NTT and cuNTT both use modulus `576460756061519873`.
- Natural-order `logN=16` includes GPU-NTT's required output permutation;
  native bit-reversed `logN=20` excludes it for both implementations.

Rows below 0.020 ms are retained but are not used for stable latency claims.
The complete table is
[`v100_external_baselines_report.md`](../results/v100_external_baselines_report.md).

## Qualified Results

Ratios are cuButterfly/cuNTT throughput divided by external throughput.

| Operator | Contract | Batch | Ratio | Interpretation |
|:--|:--|--:|--:|:--|
| FWHT FP32 | `logN=8`, natural | 65,536 | 1.001x | parity with Dao FHT |
| FWHT FP32 | `logN=15`, natural | 256 | 1.082x | cuButterfly ahead |
| FWHT FP32 | `logN=15`, natural | 512 | 1.076x | cuButterfly ahead; one timing outlier |
| NTT 60-bit | `logN=16`, natural | 4 | 1.109x | cuNTT ahead; one timing outlier in each row |
| NTT 60-bit | `logN=16`, natural | 64 | 1.393x | cuNTT ahead |
| NTT 60-bit | `logN=16`, natural | 256 | 1.425x | cuNTT ahead |
| NTT 60-bit | `logN=20`, bit-reversed | 2 | 1.139x | compact stage ahead |
| NTT 60-bit | `logN=20`, bit-reversed | 4 | 1.168x | compact stage ahead |
| NTT 60-bit | `logN=20`, bit-reversed | 16 | 1.192x | compact stage ahead; one local timing outlier |

The low-batch FWHT rows are below the timing floor and do not support a latency
claim. This refresh supersedes the historical Dao/GPU-NTT performance rows
when the listed semantics match.

## Reproduction

```bash
python3 scripts/run_external_baseline_suite.py --resume \
  --fht-python /home/wt/.conda/envs/cubutterfly-baselines/bin/python \
  --gpuntt-binary /tmp/gpuntt_merge_gap_bench \
  --output results/v100_external_baselines_raw.csv

python3 scripts/summarize_external_baseline_suite.py \
  results/v100_external_baselines_raw.csv \
  --output results/v100_external_baselines_summary.csv \
  --markdown results/v100_external_baselines_report.md
```
