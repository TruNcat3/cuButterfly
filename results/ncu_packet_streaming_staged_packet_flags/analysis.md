# Packet Streaming Fixed-Clock Attribution

| metric | aggregate b16 | online b16 | aggregate b32 | online b32 |
|:--|--:|--:|--:|--:|
| time (us) | 1226.208 | 1209.440 | 2779.168 | 3000.768 |
| DRAM read (MiB) | 264.541 | 271.504 | 497.060 | 540.686 |
| warp instructions | 176,797,829 | 194,908,356 | 359,882,985 | 401,645,735 |
| CBU pipe instructions | 4,444,527 | 5,555,687 | 11,794,001 | 15,332,157 |
| LSU pipe instructions | 22,795,851 | 23,116,047 | 46,839,259 | 48,029,637 |
| global-memory instructions | 11,367,870 | 11,914,513 | 25,022,849 | 27,746,276 |
| shared-memory instructions | 12,582,912 | 12,582,912 | 25,165,824 | 25,165,824 |
| global-load sectors | 29,143,390 | 31,929,018 | 58,798,042 | 63,974,001 |
| global-store sectors | 8,388,611 | 8,393,218 | 16,777,221 | 16,786,429 |
| shared-load bank conflicts | 6,695,234 | 8,311,476 | 13,018,908 | 16,182,319 |
| shared-store bank conflicts | 4,385,696 | 4,351,307 | 8,723,172 | 8,636,821 |
| active warps (%) | 24.260 | 24.040 | 24.780 | 24.470 |
| barrier stall (%) | 21.690 | 16.070 | 31.480 | 27.140 |
| long scoreboard stall (%) | 28.970 | 34.650 | 26.500 | 27.450 |
| MIO throttle stall (%) | 1.580 | 1.740 | 0.980 | 1.000 |
| short scoreboard stall (%) | 1.550 | 1.790 | 1.190 | 1.340 |
| registers/thread | 64.000 | 64.000 | 64.000 | 64.000 |
| shared bytes/CTA | 20592.000 | 20592.000 | 20592.000 | 20592.000 |

## Deltas

Batch 16: online/aggregate time=0.986x; instructions=1.102x; load sectors=1.096x; shared-load conflicts=1.241x; CBU instructions=1.250x; long-scoreboard=1.196x.
Batch 32: online/aggregate time=1.080x; instructions=1.116x; load sectors=1.088x; shared-load conflicts=1.243x; CBU instructions=1.300x; long-scoreboard=1.036x.
Interpret time together with barrier/CBU/LSU deltas: online publication is useful only when producer-consumer overlap repays its repeated readiness and wave synchronization cost.
