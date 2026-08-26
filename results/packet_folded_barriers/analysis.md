# Packet Wave-Barrier Folding

CUDA-event medians from independent processes.

| batch | mode | kernel (ms) |
|---:|:--|---:|
| 16 | aggregate | 1.084969 |
| 16 | interleaved | 1.144689 |
| 16 | interleaved_folded | 1.151406 |
| 16 | warp_rows | 1.125540 |
| 16 | warp_rows_folded | 1.121423 |
| 32 | aggregate | 2.420122 |
| 32 | interleaved | 2.461184 |
| 32 | interleaved_folded | 2.453934 |
| 32 | warp_rows | 2.447933 |
| 32 | warp_rows_folded | 2.449080 |
| 64 | aggregate | 4.965192 |
| 64 | interleaved | 5.016351 |
| 64 | interleaved_folded | 5.013852 |
| 64 | warp_rows | 5.006582 |
| 64 | warp_rows_folded | 5.013299 |

## Deltas

Batch 16: interleaved folding=0.994x; warp-row folding=1.004x; best=warp_rows_folded; best/aggregate throughput=0.967x.
Batch 32: interleaved folding=1.003x; warp-row folding=1.000x; best=warp_rows; best/aggregate throughput=0.989x.
Batch 64: interleaved folding=1.000x; warp-row folding=0.999x; best=warp_rows; best/aggregate throughput=0.992x.
