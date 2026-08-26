# APPT Tail-Core NCU Attribution

Matched V100 base-clock replay; throughput is relative to v0.6 10+10.

| bits | batch | core | time us | throughput/v0.6 | DRAM R/W MiB | global sectors R/W | local sectors R/W | active warps | barrier | scoreboard | MIO throttle | registers | shared B |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | `warp` | 312.6 | 0.481x | 20.3/13.6 | 8767341/2228250 | 0/0 | 25.3% | 52.4% | 20.2% | 0.4% | 72 | 16512 |
| 32 | 1 | `cta-radix4` | 310.0 | 0.485x | 20.2/13.5 | 5850325/2228090 | 0/0 | 22.7% | 43.1% | 25.9% | 0.9% | 32 | 16512 |
| 32 | 1 | `fused-tail` | 302.0 | 0.498x | 17.9/8.8 | 5674709/1179658 | 0/0 | 17.3% | 25.3% | 38.2% | 1.1% | 32 | 33024 |
| 32 | 1 | `split-tail` | 306.9 | 0.490x | 23.8/11.6 | 7213820/1732723 | 0/0 | 22.7% | 39.8% | 34.2% | 0.5% | 32 | 16512 |
| 32 | 1 | `register-tail` | 235.3 | 0.639x | 17.5/8.3 | 5316345/1179651 | 0/0 | 23.8% | 29.1% | 34.8% | 2.5% | 48 | 32768 |
| 32 | 4 | `warp` | 875.6 | 0.390x | 122.5/57.8 | 34964190/8912994 | 0/0 | 27.1% | 27.1% | 30.4% | 1.3% | 72 | 16512 |
| 32 | 4 | `cta-radix4` | 941.4 | 0.363x | 114.9/58.9 | 23327335/8913053 | 0/0 | 26.6% | 26.2% | 33.4% | 2.7% | 32 | 16512 |
| 32 | 4 | `fused-tail` | 872.2 | 0.392x | 82.4/38.7 | 22753235/4718626 | 0/0 | 24.3% | 30.2% | 29.1% | 2.9% | 32 | 33024 |
| 32 | 4 | `split-tail` | 1024.6 | 0.333x | 220.4/70.3 | 28790163/6841900 | 0/0 | 27.1% | 17.6% | 40.7% | 2.8% | 32 | 16512 |
| 32 | 4 | `register-tail` | 730.0 | 0.468x | 107.6/46.7 | 21274364/4718595 | 0/0 | 28.3% | 16.4% | 39.5% | 7.4% | 48 | 32768 |
| 64 | 1 | `warp` | 669.9 | 0.339x | 42.6/26.0 | 9463055/2359322 | 0/0 | 18.3% | 30.6% | 34.3% | 1.1% | 80 | 33024 |
| 64 | 1 | `cta-radix4` | 445.5 | 0.509x | 50.1/25.3 | 6358226/2359322 | 0/0 | 19.8% | 46.2% | 26.0% | 0.5% | 40 | 33024 |
| 64 | 1 | `fused-tail` | 479.9 | 0.473x | 42.5/16.9 | 6064209/1310730 | 0/0 | 12.5% | 26.8% | 31.9% | 0.5% | 42 | 66048 |
| 64 | 1 | `split-tail` | 410.8 | 0.552x | 58.4/21.5 | 7607668/1833199 | 0/0 | 20.7% | 42.3% | 30.0% | 0.3% | 48 | 33024 |
| 64 | 1 | `register-tail` | 336.2 | 0.675x | 40.4/16.8 | 5452400/1310730 | 0/0 | 15.3% | 25.8% | 41.3% | 0.4% | 126 | 49152 |
| 64 | 4 | `warp` | 1880.4 | 0.325x | 242.3/108.1 | 37851434/9437282 | 0/0 | 22.2% | 20.3% | 39.7% | 2.1% | 80 | 33024 |
| 64 | 4 | `cta-radix4` | 1399.2 | 0.437x | 280.9/116.0 | 25434847/9437282 | 0/0 | 22.4% | 28.4% | 35.5% | 1.5% | 40 | 33024 |
| 64 | 4 | `fused-tail` | 1306.2 | 0.469x | 202.7/69.8 | 24256594/5242914 | 0/0 | 12.5% | 14.8% | 38.6% | 0.7% | 42 | 66048 |
| 64 | 4 | `split-tail` | 1367.4 | 0.448x | 266.4/83.3 | 30437713/7370964 | 0/0 | 23.1% | 39.0% | 31.1% | 0.5% | 48 | 33024 |
| 64 | 4 | `register-tail` | 1050.0 | 0.583x | 193.2/68.8 | 21801264/5242914 | 0/0 | 17.9% | 12.5% | 47.6% | 2.0% | 126 | 49152 |

## Findings

1. Register-tail removes the second N-sized state without adding arithmetic work. Its warp-instruction ratio is 0.67x--0.84x v0.6 and its integer-thread ratio is 0.77x--0.87x.
2. Zero local load/store sectors confirm that the retained low half does not spill. The 48/126-register uint32/uint64 kernels remain limited by the requested 3/2 CTA shared-memory residency.
3. The remaining request traffic is much larger than the materialized data volume. Register-tail global-load sectors are 1.81x--2.85x v0.6 and store sectors are 2.25x--2.50x. DRAM writes remain close to the one-state floor at batch 1, so this is primarily L1/L2 request and transaction overhead rather than an extra global state.
4. Register-tail L1 hit rate is only 14.2%--14.6%, while its L2 hit rate is 83.9%--94.3%. Together with 34.8%--47.6% long-scoreboard stall, this identifies repeated L2-served dependency/coefficient/state loads as the critical path.
5. Active-warps samples reach only 0.57x--0.73x v0.6. Barrier stall remains 12.5%--29.1%, so the fused tail is not work-conserving even though its logical graph is dependency closed.
6. Readiness polling is consistent with the latency counters, but this capture alone does not isolate its contribution. Use --ablation-baseline with a matched distributed-flag capture to quantify it.
7. Achieved DRAM throughput is 12.9%--24.3% of peak for uint32 and 20.5%--29.2% for uint64. The kernel is latency/request limited, not bandwidth saturated.

Stall percentages are scheduler-state samples and must not be summed as time fractions.
