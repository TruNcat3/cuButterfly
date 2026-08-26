# APPT Physical-Core NCU Attribution

Matched V100 base-clock NCU replay. Ratios below use the v0.6 resident 10+10 radix-4 kernel at the same precision and batch.

| bits | batch | core | time us | time/v0.6 | DRAM R/W | load/store sectors | warp/int inst | active warps | barrier stall | DRAM peak | L1 hit |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | warp-radix2 | 312.6 | 2.14x | 1.50/1.55x | 4.64/4.25x | 1.02/0.76x | 25.2% | 52.7% | 12.7% | 44.2% |
| 32 | 1 | cta-radix4 | 307.3 | 2.11x | 1.50/1.56x | 3.10/4.25x | 0.80/0.93x | 22.9% | 42.8% | 12.9% | 25.7% |
| 32 | 4 | warp-radix2 | 900.7 | 2.59x | 1.94/1.88x | 4.68/4.25x | 1.14/0.77x | 26.9% | 28.8% | 24.2% | 43.3% |
| 32 | 4 | cta-radix4 | 944.7 | 2.72x | 1.87/1.87x | 3.13/4.25x | 0.90/0.95x | 26.7% | 25.7% | 22.4% | 24.9% |
| 64 | 1 | warp-radix2 | 669.4 | 2.93x | 1.38/1.59x | 3.13/4.50x | 1.38/0.88x | 18.4% | 30.6% | 12.0% | 37.5% |
| 64 | 1 | cta-radix4 | 442.9 | 1.94x | 1.62/1.56x | 2.10/4.50x | 0.83/0.95x | 19.8% | 46.3% | 20.0% | 24.0% |
| 64 | 4 | warp-radix2 | 1972.6 | 3.27x | 1.97/1.68x | 3.15/4.50x | 1.57/0.90x | 22.2% | 19.7% | 21.4% | 36.3% |
| 64 | 4 | cta-radix4 | 1383.5 | 2.29x | 2.12/1.64x | 2.12/4.50x | 0.93/0.97x | 22.1% | 28.7% | 31.9% | 23.1% |

## Findings

1. The uint64 CTA radix-4 core closes the processing-unit instruction gap: it executes 0.83x/0.93x the v0.6 warp instructions and 0.95x/0.97x the integer thread instructions at batch 1/4. Arithmetic instruction count is no longer the reason for its 1.94x/2.29x runtime gap.
2. The 7+7+6 online graph materializes two intermediate states, while v0.6 10+10 materializes one. The minimum state traffic therefore grows from two to three read/write passes. Measured DRAM writes grow 1.56x--1.87x, close to or above that 1.5x structural floor.
3. Online layouts amplify request traffic beyond the extra pass: CTA radix-4 load sectors are 2.10x--3.13x and store sectors 4.25x--4.50x v0.6. High L2 hit rates hide much of this from DRAM, but not from L1/L2 request handling and dependency latency.
4. CTA radix-4 barrier stalls are 25.7%--46.3%, versus 2.8%--5.1% for v0.6. This combines shared-core synchronization with the current one-thread readiness wait followed by a CTA-wide barrier. Static roles leave the other warps unable to execute ready work.
5. APPT reaches only 13.0%--31.9% of sustained DRAM throughput, below the matched v0.6 points. The kernel is latency, synchronization, and issue limited; it is not saturating memory bandwidth.
6. Shared bank conflicts are not the primary regression: CTA radix-4 shared-load conflicts are only about 1.2x--1.3x v0.6 and shared-store conflicts are lower. Removing bank conflicts alone cannot explain a roughly 2x runtime gap.

The stall percentages are scheduler-state samples and are not additive time fractions. They identify bottlenecks but must not be summed into a percentage attribution.
