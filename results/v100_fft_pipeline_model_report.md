# V100 FFT Static-Model Evaluation

Candidates are ordered only by the generator's static resource score; measured
latency is then used to compute selection regret. Lower regret is better.

| logN | Batch | Candidates | Winner static rank | Top-1 regret | Top-3 regret | Top-10 regret | Spearman |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 18 | 16 | 48 | 32 | 1.0253x | 1.0253x | 1.0253x | 0.342 |
| 18 | 2 | 48 | 31 | 1.4220x | 1.4220x | 1.2969x | 0.342 |
| 18 | 64 | 48 | 1 | 1.0000x | 1.0000x | 1.0000x | 0.365 |
| 20 | 16 | 48 | 7 | 1.0005x | 1.0005x | 1.0000x | 0.340 |
| 20 | 2 | 48 | 32 | 1.0642x | 1.0642x | 1.0642x | 0.321 |
| 20 | 8 | 48 | 7 | 1.0007x | 1.0007x | 1.0000x | 0.366 |

## Aggregate

| Budget | Exact recall | Geomean regret | Worst regret | Candidate fraction |
|:--|--:|--:|--:|--:|
| Top-1 | 16.7% | 1.0762x | 1.4220x | 2.1% |
| Top-3 | 16.7% | 1.0762x | 1.4220x | 6.2% |
| Top-10 | 50.0% | 1.0596x | 1.2969x | 20.8% |

Mean per-shape Spearman correlation: 0.346.

## Leave-One-Batch-Out Calibration

For each held-out shape, the calibrated selector sees only the other batches
at the same length and interpolates or extrapolates log-latency for matching
physical mappings. It never consumes timing from the target shape.

| logN | Batch | Covered | Winner rank | Top-1 regret | Top-3 regret | Top-10 regret |
|--:|--:|--:|--:|--:|--:|--:|
| 18 | 16 | 48/48 | 5 | 1.0043x | 1.0020x | 1.0000x |
| 18 | 2 | 44/48 | 5 | 1.0024x | 1.0016x | 1.0000x |
| 18 | 64 | 48/48 | 2 | 1.0800x | 1.0000x | 1.0000x |
| 20 | 16 | 48/48 | 2 | 1.0017x | 1.0000x | 1.0000x |
| 20 | 2 | 48/48 | 3 | 1.0029x | 1.0000x | 1.0000x |
| 20 | 8 | 48/48 | 5 | 1.0007x | 1.0007x | 1.0000x |

| Budget | Exact recall | Geomean regret | Worst regret | Candidate fraction |
|:--|--:|--:|--:|--:|
| Top-1 | 0.0% | 1.0149x | 1.0800x | 2.1% |
| Top-3 | 50.0% | 1.0007x | 1.0020x | 6.2% |
| Top-10 | 100.0% | 1.0000x | 1.0000x | 20.8% |

Mean mapping coverage: 98.6%.
