# Packet Streaming Fixed-Clock Attribution

| metric | aggregate b16 | online b16 | aggregate b32 | online b32 |
|:--|--:|--:|--:|--:|
| time (us) | 1212.224 | 1238.528 | 2819.680 | 3013.312 |
| DRAM read (MiB) | 267.623 | 303.481 | 500.761 | 576.791 |
| warp instructions | 176,544,026 | 196,350,514 | 359,955,145 | 404,149,301 |
| CBU pipe instructions | 4,509,993 | 5,387,860 | 11,497,967 | 15,541,667 |
| LSU pipe instructions | 23,009,484 | 23,045,407 | 47,443,220 | 48,109,022 |
| global-memory instructions | 11,332,088 | 11,953,833 | 24,889,591 | 26,943,470 |
| shared-memory instructions | 12,582,912 | 12,582,912 | 25,165,824 | 25,165,824 |
| global-load sectors | 29,225,868 | 31,873,442 | 58,709,999 | 64,210,757 |
| global-store sectors | 8,388,611 | 8,393,233 | 16,777,221 | 16,786,433 |
| shared-load bank conflicts | 6,713,368 | 8,371,750 | 13,235,041 | 16,273,835 |
| shared-store bank conflicts | 4,392,607 | 4,345,784 | 8,721,226 | 8,629,589 |
| active warps (%) | 23.990 | 23.920 | 24.760 | 24.480 |
| barrier stall (%) | 21.620 | 16.210 | 31.880 | 26.820 |
| long scoreboard stall (%) | 31.930 | 30.800 | 26.940 | 28.380 |
| MIO throttle stall (%) | 1.620 | 1.650 | 0.980 | 1.000 |
| short scoreboard stall (%) | 1.700 | 1.690 | 1.260 | 1.270 |
| registers/thread | 71.000 | 71.000 | 71.000 | 71.000 |
| shared bytes/CTA | 20608.000 | 20608.000 | 20608.000 | 20608.000 |

## Deltas

Batch 16: online/aggregate time=1.022x; instructions=1.112x; load sectors=1.091x; shared-load conflicts=1.247x; CBU instructions=1.195x; long-scoreboard=0.965x.
Batch 32: online/aggregate time=1.069x; instructions=1.123x; load sectors=1.094x; shared-load conflicts=1.230x; CBU instructions=1.352x; long-scoreboard=1.053x.
Interpret time together with barrier/CBU/LSU deltas: online publication is useful only when producer-consumer overlap repays its repeated readiness and wave synchronization cost.
