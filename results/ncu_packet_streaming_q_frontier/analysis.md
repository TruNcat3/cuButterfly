# Packet Streaming Fixed-Clock Attribution

> **Rejected protocol:** clean rebuilds exposed a cross-producer publication
> race in the q-frontier aggregation scheme. These counters are retained only
> for synchronization attribution and must not be used for dispatch or
> headline performance claims.

| metric | aggregate b16 | online b16 | aggregate b32 | online b32 |
|:--|--:|--:|--:|--:|
| time (us) | 1211.520 | 1283.680 | 2755.328 | 2859.392 |
| DRAM read (MiB) | 274.560 | 285.983 | 499.113 | 529.683 |
| warp instructions | 175,780,991 | 188,968,860 | 357,989,224 | 384,678,837 |
| CBU pipe instructions | 4,468,647 | 4,402,395 | 11,790,908 | 11,215,425 |
| LSU pipe instructions | 22,822,518 | 22,585,780 | 46,870,701 | 46,078,795 |
| global-memory instructions | 11,254,026 | 11,214,592 | 24,578,876 | 24,540,248 |
| shared-memory instructions | 12,582,912 | 12,582,912 | 25,165,824 | 25,165,824 |
| global-load sectors | 29,115,076 | 31,476,080 | 58,817,423 | 63,414,402 |
| global-store sectors | 8,388,611 | 8,388,737 | 16,777,221 | 16,777,473 |
| shared-load bank conflicts | 6,709,499 | 8,582,292 | 13,241,985 | 16,534,402 |
| shared-store bank conflicts | 4,390,965 | 4,346,907 | 8,725,832 | 8,669,144 |
| active warps (%) | 24.190 | 23.750 | 24.760 | 24.790 |
| barrier stall (%) | 21.340 | 10.530 | 31.310 | 23.150 |
| long scoreboard stall (%) | 29.520 | 43.880 | 26.350 | 32.210 |
| MIO throttle stall (%) | 1.650 | 1.480 | 0.990 | 0.990 |
| short scoreboard stall (%) | 1.730 | 1.890 | 1.330 | 1.480 |
| registers/thread | 59.000 | 59.000 | 59.000 | 59.000 |
| shared bytes/CTA | 20592.000 | 20592.000 | 20592.000 | 20592.000 |

## Deltas

Batch 16: online/aggregate time=1.060x; instructions=1.075x.
Batch 32: online/aggregate time=1.038x; instructions=1.075x.
Interpret time together with barrier/CBU/LSU deltas: online publication is useful only when producer-consumer overlap repays its repeated readiness and wave synchronization cost.
