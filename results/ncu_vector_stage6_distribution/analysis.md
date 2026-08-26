# Vector Radix-4 Stage-6 Distribution

| metric | v0.6 | vector d6 | vector d7 | d7/d6 |
|:--|--:|--:|--:|--:|
| time (us) | 1279.616 | 1356.576 | 1367.136 | 1.008x |
| DRAM read (MiB) | 284.414 | 294.352 | 274.797 | 0.934x |
| global-load sectors | 29,764,946 | 39,999,693 | 31,432,704 | 0.786x |
| global-store sectors | 8,388,611 | 8,297,389 | 8,447,972 | 1.018x |
| warp instructions | 182,077,639 | 234,664,087 | 237,284,451 | 1.011x |
| registers/thread | 40.000 | 106.000 | 105.000 | 0.991x |
| active warps (%) | 47.820 | 24.690 | 24.720 | 1.001x |
| long scoreboard stall (%) | 42.740 | 19.910 | 17.470 | 0.877x |
| MIO throttle stall (%) | 5.460 | 3.000 | 3.410 | 1.137x |

## Stage-6 Model Closure

- measured d6 -> d7 reduction: 8,566,989 sectors
- even/odd dual-bank prediction: 8,388,608 sectors
- measured/predicted: 1.021x
- d6 excess over v0.6 removed by d7: 83.7%

## Decision

d7 reduces global-load sectors but does not reduce kernel time. Keep d6 as the performance default. The next physical-core candidate should use an aligned, prepacked stage-6 coefficient table so each active lane can issue LDG.128 without the d7 shuffle path.
The measured d7/d6 warp-instruction ratio is 1.011x; use it with the stall deltas to distinguish instruction overhead from memory latency.
