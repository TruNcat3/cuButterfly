# Hierarchical local-core NCU attribution

Both kernels use the same persistent graph, cooperative boundary, grid/block shape, and shared allocation. Only the resident radix-4 instruction schedule changes.

| bits | native us | mature us | native speedup | warp inst change | integer inst change | long scoreboard change | regs native/mature | effective CTA/SM limit |
|---:|---:|---:|---:|---:|---:|---:|:--|:--|
| 32 | 269.25 | 308.93 | 1.147x | -5.7% | -8.3% | +12.45 pp | 38/32 | 5/5 |
| 64 | 506.72 | 521.31 | 1.029x | -4.6% | -6.0% | +3.43 pp | 48/44 | 2/2 |

## Interpretation

The mature Hybrid2D schedule reduces instruction count and registers, but does not cross an effective residency boundary. For uint32, shared memory fixes both kernels at five CTAs/SM even though the register limit improves from six to eight. For uint64, allocated register granularity leaves both at two CTAs/SM.

The mature schedule instead raises long-scoreboard stalls, especially for uint32. With near-equal traffic and unchanged active-warps percentage, this attributes the regression to load/dependency placement and reduced latency-hiding ILP inside the physical unit, not to the outer dataflow graph, arithmetic count, or DRAM traffic class.
