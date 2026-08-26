# Packet Readiness Fixed-Clock Attribution

## Batch 16

| metric | aggregate | per-packet | wave-bitmap |
|:--|--:|--:|--:|
| time (us) | 1235.904 | 1275.904 | 1220.512 |
| DRAM read (MiB) | 266.421 | 290.530 | 290.035 |
| warp instructions | 176,205,495 | 193,747,720 | 192,217,649 |
| CBU instructions | 4,505,306 | 5,160,431 | 4,832,204 |
| LSU instructions | 22,821,894 | 22,917,936 | 22,731,118 |
| global-load sectors | 29,095,894 | 31,639,874 | 31,431,418 |
| global-store sectors | 8,388,611 | 8,393,222 | 8,388,625 |
| shared-load conflicts | 6,664,758 | 8,296,043 | 8,353,854 |
| barrier stall (%) | 22.260 | 17.240 | 17.120 |
| long scoreboard (%) | 29.860 | 29.830 | 29.550 |
| registers/thread | 62.000 | 62.000 | 62.000 |

Bitmap/per-packet: time=0.957x; instructions=0.992x; load sectors=0.993x; shared-load conflicts=1.007x; CBU instructions=0.936x; bitmap/aggregate time=0.988x.

## Batch 32

| metric | aggregate | per-packet | wave-bitmap |
|:--|--:|--:|--:|
| time (us) | 2772.320 | 3002.624 | 2868.640 |
| DRAM read (MiB) | 497.674 | 569.737 | 529.350 |
| warp instructions | 358,439,921 | 399,051,077 | 391,436,105 |
| CBU instructions | 11,977,570 | 13,951,149 | 12,420,933 |
| LSU instructions | 46,917,946 | 47,377,530 | 46,649,864 |
| global-load sectors | 58,677,782 | 63,714,045 | 63,260,013 |
| global-store sectors | 16,777,221 | 16,786,430 | 16,777,249 |
| shared-load conflicts | 13,110,238 | 16,178,709 | 16,305,627 |
| barrier stall (%) | 31.360 | 25.660 | 28.790 |
| long scoreboard (%) | 26.870 | 26.000 | 24.460 |
| registers/thread | 62.000 | 62.000 | 62.000 |

Bitmap/per-packet: time=0.955x; instructions=0.981x; load sectors=0.993x; shared-load conflicts=1.008x; CBU instructions=0.890x; bitmap/aggregate time=1.035x.

## Batch 64

| metric | aggregate | per-packet | wave-bitmap |
|:--|--:|--:|--:|
| time (us) | 5813.024 | 5470.880 | 5339.872 |
| DRAM read (MiB) | 1095.021 | 1003.413 | 960.365 |
| warp instructions | 718,000,257 | 775,557,103 | 780,229,114 |
| CBU instructions | 23,542,658 | 22,971,137 | 24,626,737 |
| LSU instructions | 93,660,250 | 92,640,844 | 93,198,599 |
| global-load sectors | 117,596,227 | 126,922,715 | 126,157,000 |
| global-store sectors | 33,554,441 | 33,572,874 | 33,554,497 |
| shared-load conflicts | 26,401,726 | 32,717,936 | 32,670,300 |
| barrier stall (%) | 32.720 | 25.620 | 26.820 |
| long scoreboard (%) | 26.570 | 27.780 | 28.040 |
| registers/thread | 62.000 | 62.000 | 62.000 |

Bitmap/per-packet: time=0.976x; instructions=1.006x; load sectors=0.994x; shared-load conflicts=0.999x; CBU instructions=1.072x; bitmap/aggregate time=0.919x.

CUDA-event timing remains the ranking authority. This replay capture tests whether identity-preserving bitmap publication removes polling sectors without transferring the cost to CBU serialization.
