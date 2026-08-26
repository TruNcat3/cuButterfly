# Packet Streaming Fixed-Clock Attribution

| metric | aggregate b16 | online b16 | aggregate b32 | online b32 | aggregate b64 | online b64 |
|:--|--:|--:|--:|--:|--:|--:|
| time (us) | 1208.224 | 1257.344 | 2791.296 | 2996.384 | 5750.816 | 5810.016 |
| DRAM read (MiB) | 258.358 | 298.503 | 507.949 | 557.466 | 1083.053 | 1170.086 |
| warp instructions | 176,722,764 | 194,712,155 | 359,972,163 | 401,294,975 | 721,781,089 | 783,664,462 |
| CBU pipe instructions | 4,628,115 | 5,230,793 | 11,870,458 | 15,410,623 | 23,419,596 | 22,173,183 |
| LSU pipe instructions | 22,874,509 | 22,983,443 | 46,872,036 | 48,059,920 | 93,607,501 | 92,382,828 |
| global-memory instructions | 11,393,810 | 11,740,561 | 24,942,255 | 27,699,743 | 49,943,691 | 50,133,772 |
| shared-memory instructions | 12,582,912 | 12,582,912 | 25,165,824 | 25,165,824 | 50,331,648 | 50,331,648 |
| global-load sectors | 29,251,089 | 31,795,687 | 58,718,870 | 64,214,245 | 117,398,172 | 126,897,128 |
| global-store sectors | 8,388,611 | 8,393,218 | 16,777,221 | 16,786,423 | 33,554,441 | 33,572,858 |
| shared-load bank conflicts | 6,660,079 | 8,326,848 | 13,245,628 | 16,223,097 | 26,439,960 | 32,856,374 |
| shared-store bank conflicts | 4,390,126 | 4,339,291 | 8,707,456 | 8,633,472 | 17,396,885 | 17,262,598 |
| active warps (%) | 23.900 | 24.070 | 24.730 | 24.490 | 24.830 | 24.720 |
| barrier stall (%) | 21.580 | 17.710 | 31.520 | 26.680 | 32.510 | 25.660 |
| long scoreboard stall (%) | 30.020 | 31.070 | 26.470 | 28.350 | 25.510 | 25.050 |
| MIO throttle stall (%) | 1.600 | 1.590 | 0.950 | 1.010 | 0.910 | 1.070 |
| short scoreboard stall (%) | 1.570 | 1.740 | 1.190 | 1.340 | 1.160 | 1.430 |
| registers/thread | 64.000 | 64.000 | 64.000 | 64.000 | 64.000 | 64.000 |
| shared bytes/CTA | 20592.000 | 20592.000 | 20592.000 | 20592.000 | 20592.000 | 20592.000 |

## Deltas

Batch 16: online/aggregate time=1.041x; instructions=1.102x; load sectors=1.087x; shared-load conflicts=1.250x; CBU instructions=1.130x; long-scoreboard=1.035x.
Batch 32: online/aggregate time=1.073x; instructions=1.115x; load sectors=1.094x; shared-load conflicts=1.225x; CBU instructions=1.298x; long-scoreboard=1.071x.
Batch 64: online/aggregate time=1.010x; instructions=1.086x; load sectors=1.081x; shared-load conflicts=1.243x; CBU instructions=0.947x; long-scoreboard=0.982x.
Interpret time together with barrier/CBU/LSU deltas: online publication is useful only when producer-consumer overlap repays its repeated readiness and wave synchronization cost.
