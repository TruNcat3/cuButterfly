# APPT Tail-Core NCU Attribution

Matched V100 base-clock replay; throughput is relative to v0.6 10+10.

| bits | batch | core | time us | throughput/v0.6 | DRAM R/W MiB | global sectors R/W | local sectors R/W | active warps | barrier | scoreboard | MIO throttle | registers | shared B |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | `warp` | 312.5 | 0.464x | 20.2/13.5 | 8766726/2228250 | 0/0 | 25.2% | 52.7% | 20.3% | 0.4% | 72 | 16512 |
| 32 | 1 | `cta-radix4` | 306.6 | 0.473x | 20.2/13.5 | 5830364/2228250 | 0/0 | 22.9% | 43.3% | 26.0% | 1.1% | 32 | 16512 |
| 32 | 1 | `fused-tail` | 288.8 | 0.502x | 18.6/9.0 | 5674807/1179658 | 0/0 | 17.3% | 25.4% | 37.8% | 1.2% | 32 | 33024 |
| 32 | 1 | `split-tail` | 313.0 | 0.463x | 23.5/11.1 | 7220510/1699868 | 0/0 | 22.7% | 40.6% | 34.3% | 0.5% | 32 | 16512 |
| 32 | 1 | `register-tail` | 244.5 | 0.593x | 18.8/8.6 | 6690151/1179651 | 0/0 | 23.8% | 28.6% | 35.9% | 2.1% | 47 | 32768 |
| 32 | 4 | `warp` | 882.4 | 0.383x | 130.9/63.7 | 34993031/8912994 | 0/0 | 26.7% | 27.1% | 30.2% | 1.3% | 72 | 16512 |
| 32 | 4 | `cta-radix4` | 960.0 | 0.352x | 122.0/59.5 | 23315504/8913050 | 0/0 | 26.8% | 26.8% | 33.5% | 2.9% | 32 | 16512 |
| 32 | 4 | `fused-tail` | 874.0 | 0.386x | 85.4/36.7 | 22595653/4718626 | 0/0 | 24.0% | 29.8% | 28.0% | 2.8% | 32 | 33024 |
| 32 | 4 | `split-tail` | 975.0 | 0.346x | 175.6/60.3 | 28754586/6817854 | 0/0 | 27.0% | 17.2% | 41.0% | 2.8% | 32 | 16512 |
| 32 | 4 | `register-tail` | 792.1 | 0.426x | 147.5/49.1 | 26777553/4718595 | 0/0 | 29.6% | 22.8% | 37.6% | 4.4% | 47 | 32768 |
| 64 | 1 | `warp` | 669.8 | 0.342x | 42.7/25.9 | 9464099/2359322 | 0/0 | 18.4% | 30.6% | 33.6% | 1.2% | 80 | 33024 |
| 64 | 1 | `cta-radix4` | 440.7 | 0.519x | 50.2/25.4 | 6368310/2359322 | 0/0 | 19.8% | 46.0% | 25.9% | 0.5% | 40 | 33024 |
| 64 | 1 | `fused-tail` | 483.0 | 0.474x | 42.5/17.0 | 6064209/1310730 | 0/0 | 12.5% | 26.7% | 31.9% | 0.6% | 42 | 66048 |
| 64 | 1 | `split-tail` | 408.1 | 0.561x | 58.1/21.4 | 7616343/1821519 | 0/0 | 20.8% | 41.0% | 29.7% | 0.3% | 48 | 33024 |
| 64 | 1 | `register-tail` | 349.5 | 0.655x | 39.0/16.1 | 7086238/1310730 | 0/0 | 15.6% | 25.9% | 42.1% | 0.2% | 128 | 49152 |
| 64 | 4 | `warp` | 1970.2 | 0.310x | 268.0/124.4 | 37840328/9437282 | 0/0 | 22.1% | 20.0% | 40.0% | 2.1% | 80 | 33024 |
| 64 | 4 | `cta-radix4` | 1364.7 | 0.447x | 272.3/106.5 | 25451261/9437282 | 0/0 | 22.1% | 27.2% | 35.1% | 1.4% | 40 | 33024 |
| 64 | 4 | `fused-tail` | 1314.0 | 0.464x | 204.2/68.7 | 24256598/5242914 | 0/0 | 12.5% | 14.7% | 38.6% | 0.7% | 42 | 66048 |
| 64 | 4 | `split-tail` | 1365.3 | 0.447x | 270.4/83.7 | 30425302/7310721 | 0/0 | 23.1% | 38.7% | 31.5% | 0.5% | 48 | 33024 |
| 64 | 4 | `register-tail` | 1113.1 | 0.548x | 201.1/68.8 | 28390085/5242914 | 0/0 | 18.8% | 11.3% | 50.0% | 1.1% | 128 | 49152 |

## Findings

1. Register-tail removes the second N-sized state without adding arithmetic work. Its warp-instruction ratio is 0.69x--0.85x v0.6 and its integer-thread ratio is 0.80x--0.89x.
2. Zero local load/store sectors confirm that the retained low half does not spill. The 47/128-register uint32/uint64 kernels remain limited by the requested 3/2 CTA shared-memory residency.
3. The remaining request traffic is much larger than the materialized data volume. Register-tail global-load sectors are 2.35x--3.59x v0.6 and store sectors are 2.25x--2.50x. DRAM writes remain close to the one-state floor at batch 1, so this is primarily L1/L2 request and transaction overhead rather than an extra global state.
4. Register-tail L1 hit rate is only 27.6%--30.9%, while its L2 hit rate is 81.9%--93.7%. Together with 35.9%--50.0% long-scoreboard stall, this identifies repeated L2-served dependency/coefficient/state loads as the critical path.
5. Active-warps samples reach only 0.57x--0.77x v0.6. Barrier stall remains 11.3%--28.6%, so the fused tail is not work-conserving even though its logical graph is dependency closed.
6. The uint32 aggregate-readiness ablation improves NCU replay time by 1.4%--6.8% over the earlier distributed-flag capture, while global load/store sectors change by at most 0.03%. Readiness polling is therefore a latency contributor, but it does not explain the excess request traffic. Uint64 retains distributed flags because aggregate-counter contention was slower in the timing scan.
7. Achieved DRAM throughput is 13.2%--30.0% of peak for uint32 and 19.0%--27.1% for uint64. The kernel is latency/request limited, not bandwidth saturated.

Stall percentages are scheduler-state samples and must not be summed as time fractions.
