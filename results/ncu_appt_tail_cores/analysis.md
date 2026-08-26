# APPT Tail-Core NCU Attribution

Matched V100 base-clock replay; throughput is relative to v0.6 10+10.

| bits | batch | core | time us | throughput/v0.6 | DRAM R/W MiB | global sectors R/W | local sectors R/W | active warps | barrier | scoreboard | MIO throttle | registers | shared B |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | `warp` | 312.6 | 0.478x | 20.2/13.5 | 8765540/2228250 | 0/0 | 25.2% | 52.4% | 20.5% | 0.4% | 72 | 16512 |
| 32 | 1 | `cta-radix4` | 305.8 | 0.488x | 20.3/13.4 | 5828527/2228146 | 0/0 | 22.8% | 43.0% | 25.7% | 1.0% | 32 | 16512 |
| 32 | 1 | `fused-tail` | 283.7 | 0.526x | 16.5/8.4 | 5671301/1179658 | 0/0 | 17.3% | 25.2% | 37.8% | 1.3% | 32 | 33024 |
| 32 | 1 | `split-tail` | 308.9 | 0.483x | 23.6/11.0 | 7216133/1689828 | 0/0 | 22.9% | 40.2% | 33.9% | 0.5% | 32 | 16512 |
| 32 | 1 | `register-tail` | 247.9 | 0.602x | 18.5/8.9 | 6689086/1179658 | 0/0 | 23.3% | 29.3% | 37.6% | 2.1% | 47 | 32768 |
| 32 | 4 | `warp` | 898.3 | 0.388x | 123.7/65.0 | 34996300/8912994 | 0/0 | 26.6% | 28.4% | 29.9% | 1.3% | 72 | 16512 |
| 32 | 4 | `cta-radix4` | 937.1 | 0.372x | 112.7/58.0 | 23401551/8913030 | 0/0 | 26.9% | 25.7% | 33.7% | 2.7% | 32 | 16512 |
| 32 | 4 | `fused-tail` | 855.7 | 0.408x | 83.3/42.1 | 22649090/4718626 | 0/0 | 24.2% | 28.8% | 29.0% | 3.0% | 32 | 33024 |
| 32 | 4 | `split-tail` | 996.3 | 0.350x | 186.7/60.4 | 29021058/6793259 | 0/0 | 27.1% | 17.3% | 40.5% | 2.8% | 32 | 16512 |
| 32 | 4 | `register-tail` | 846.2 | 0.412x | 170.8/54.4 | 26785553/4718626 | 0/0 | 29.3% | 17.1% | 40.4% | 5.1% | 47 | 32768 |
| 64 | 1 | `warp` | 666.8 | 0.342x | 42.6/25.9 | 9466961/2359322 | 0/0 | 18.4% | 30.6% | 33.8% | 1.1% | 80 | 33024 |
| 64 | 1 | `cta-radix4` | 444.7 | 0.512x | 50.2/25.5 | 6368564/2359322 | 0/0 | 19.8% | 46.4% | 25.6% | 0.5% | 40 | 33024 |
| 64 | 1 | `fused-tail` | 482.2 | 0.472x | 42.5/16.9 | 6064201/1310730 | 0/0 | 12.5% | 26.6% | 31.8% | 0.6% | 42 | 66048 |
| 64 | 1 | `split-tail` | 411.6 | 0.553x | 58.6/22.4 | 7612774/1828866 | 0/0 | 20.7% | 41.9% | 30.2% | 0.3% | 48 | 33024 |
| 64 | 1 | `register-tail` | 349.8 | 0.651x | 39.0/16.1 | 7086250/1310730 | 0/0 | 15.6% | 25.9% | 42.1% | 0.2% | 128 | 49152 |
| 64 | 4 | `warp` | 1977.6 | 0.307x | 269.8/117.6 | 37882284/9437282 | 0/0 | 22.2% | 20.6% | 39.9% | 2.2% | 80 | 33024 |
| 64 | 4 | `cta-radix4` | 1375.3 | 0.442x | 270.7/103.8 | 25456296/9437282 | 0/0 | 22.2% | 28.2% | 34.7% | 1.5% | 40 | 33024 |
| 64 | 4 | `fused-tail` | 1317.2 | 0.462x | 203.3/70.3 | 24256575/5242914 | 0/0 | 12.5% | 14.7% | 38.8% | 0.7% | 42 | 66048 |
| 64 | 4 | `split-tail` | 1352.2 | 0.450x | 265.8/83.2 | 30414249/7333798 | 0/0 | 23.1% | 38.0% | 31.1% | 0.4% | 48 | 33024 |
| 64 | 4 | `register-tail` | 1134.5 | 0.536x | 200.8/69.8 | 28323206/5242914 | 0/0 | 18.7% | 11.4% | 49.9% | 1.0% | 128 | 49152 |

## Findings

1. Register-tail removes the second N-sized state without adding arithmetic work. Its warp-instruction ratio is 0.67x--0.85x v0.6 and its integer-thread ratio is 0.81x--0.89x.
2. Zero local load/store sectors confirm that the retained low half does not spill. The 47/128-register uint32/uint64 kernels remain limited by the requested 3/2 CTA shared-memory residency.
3. The remaining request traffic is much larger than the materialized data volume. Register-tail global-load sectors are 2.33x--3.58x v0.6 and store sectors are 2.25x--2.50x. DRAM writes remain close to the one-state floor at batch 1, so this is primarily L1/L2 request and transaction overhead rather than an extra global state.
4. Register-tail L1 hit rate is only 27.5%--31.0%, while its L2 hit rate is 83.1%--93.5%. Together with 37.6%--49.9% long-scoreboard stall, this identifies repeated L2-served dependency/coefficient/state loads as the critical path.
5. Active-warps samples reach only 0.57x--0.77x v0.6. Barrier stall remains 11.4%--29.3%, so the fused tail is not work-conserving even though its logical graph is dependency closed.
6. Achieved DRAM throughput is 13.0%--31.1% of peak for uint32 and 18.9%--27.8% for uint64. The kernel is latency/request limited, not bandwidth saturated. Readiness polling is a source-level inference consistent with these counters; the aggregate-counter ablation must be recollected before assigning an exact fraction to it.

Stall percentages are scheduler-state samples and must not be summed as time fractions.
