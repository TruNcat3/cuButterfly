# HierarchicalDataflow NCU attribution

Hybrid2D totals include captured sequential pass kernels; a valid plan comparison requires two. Percentage metrics are time-weighted.

| bits | backend | kernels | time us | read MiB | write MiB | integer inst | active warps | barrier stall | long scoreboard | regs/thread | shared B |
|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | hierarchical-dataflow | 1 | 268.96 | 61.81 | 34.17 | 1027577244 | 56.07% | 13.31% | 35.93% | 38 | 16400 |
| 32 | hybrid2d | 2 | 326.85 | 64.00 | 33.01 | 1043339264 | 93.41% | 13.01% | 16.06% | 32 | 8200 |
| 64 | hierarchical-dataflow | 1 | 515.52 | 128.37 | 70.23 | 2060173583 | 49.55% | 12.78% | 24.85% | 48 | 16400 |
| 64 | hybrid2d | 2 | 480.29 | 128.39 | 68.58 | 1923094528 | 94.55% | 9.35% | 32.41% | 32 | 16400 |

## Plan-level comparison

- uint32: hierarchical/Hybrid2D speedup `1.215x`; read ratio `0.966x`; write ratio `1.035x`; warp-instruction ratio `0.934x`; integer-instruction ratio `0.985x`; registers/thread `38` versus `32`.
- uint64: hierarchical/Hybrid2D speedup `0.932x`; read ratio `1.000x`; write ratio `1.024x`; warp-instruction ratio `1.054x`; integer-instruction ratio `1.071x`; registers/thread `48` versus `32`.

## Interpretation

Near-unit DRAM ratios mean the one-launch graph preserves, rather than removes, the single inter-layer global boundary. Performance differences are therefore attributed to CTA scheduling, instruction work, latency hiding, and synchronization instead of a different traffic-complexity class.
