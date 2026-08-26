# Resident 7+7+6 NCU Attribution

All batch-4 rows are matched single-kernel captures on Tesla V100. Speedup above one is better than the v0.6 barrier kernel.

| bits | implementation | time us | speedup vs barrier | read ratio | write ratio | warp-inst ratio | integer-inst ratio | active warps | barrier stall | wait stall | long scoreboard | regs | shared B | waves/SM |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | barrier | 536.67 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 24.26% | 6.07% | 15.01% | 44.14% | 48 | 32800 | 1.00 |
| 64 | generic-7+7+6 | 1606.24 | 0.334x | 3.181x | 1.736x | 2.182x | 1.534x | 23.16% | 0.04% | 12.51% | 49.36% | 82 | 8192 | 1.00 |
| 64 | resident-7+7+6 | 3490.11 | 0.154x | 6.282x | 4.150x | 1.218x | 1.131x | 23.38% | 37.02% | 3.23% | 44.53% | 48 | 33024 | 1.00 |
| 64 | resident-10+10 | 609.15 | 0.881x | 0.995x | 1.001x | 1.096x | 1.017x | 24.30% | 3.53% | 15.83% | 32.20% | 48 | 32816 | 1.00 |
| 32 | barrier | 270.43 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 55.91% | 16.08% | 8.76% | 35.64% | 38 | 16400 | 1.00 |
| 32 | resident-10+10 | 347.84 | 0.777x | 1.064x | 1.009x | 1.136x | 1.020x | 47.21% | 5.08% | 10.21% | 33.31% | 40 | 16416 | 0.80 |

## Resident 7+7+6 Batch Scaling

For uint64, batch 4 takes `5.077x` the batch-1 time. The per-transform time is therefore `1.269x` worse, not better. DRAM read/write scale by `5.541x/6.491x`, while warp/integer instructions scale only by `3.829x/3.852x`. Barrier stall remains dominant (`45.00%` at batch 1, `37.02%` at batch 4), while wait stall is only `4.83%/3.23%`.

## Interpretation

The resident physical core reduces instructions relative to the generic three-layer kernel, but fixed role ownership leaves runnable upstream work behind synchronization-stalled roles. Repeated readiness loads amplify measured DRAM traffic as batch grows. The next kernel must make scheduling work-conserving without introducing a single contended global queue; changing the butterfly arithmetic alone cannot close this gap.
