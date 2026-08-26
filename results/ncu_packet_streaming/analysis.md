# Packet Streaming Fixed-Clock Attribution

| metric | aggregate b16 | online b16 | aggregate b32 | online b32 |
|:--|--:|--:|--:|--:|
| time (us) | 1180.864 | 1246.432 | 2770.144 | 2835.520 |
| warp instructions | 176,259,975 | 183,308,264 | 359,414,277 | 383,330,204 |
| CBU pipe instructions | 4,538,765 | 5,709,393 | 11,766,698 | 18,542,418 |
| LSU pipe instructions | 22,869,000 | 22,917,693 | 46,893,097 | 48,009,025 |
| global-memory instructions | 11,381,766 | 11,832,461 | 24,969,467 | 24,567,800 |
| shared-memory instructions | 12,582,912 | 12,582,912 | 25,165,824 | 25,165,824 |
| global-load sectors | 29,169,820 | 33,739,814 | 58,661,624 | 67,973,263 |
| global-store sectors | 8,388,611 | 8,393,210 | 16,777,221 | 16,786,436 |
| active warps (%) | 24.030 | 24.150 | 24.780 | 24.420 |
| barrier stall (%) | 20.860 | 11.730 | 31.380 | 28.850 |
| long scoreboard stall (%) | 31.000 | 45.310 | 26.690 | 27.500 |
| MIO throttle stall (%) | 1.590 | 1.760 | 0.930 | 1.040 |
| short scoreboard stall (%) | 1.800 | 2.120 | 1.330 | 1.450 |
| registers/thread | 53.000 | 53.000 | 53.000 | 53.000 |
| shared bytes/CTA | 16496.000 | 16496.000 | 16496.000 | 16496.000 |

## Deltas

Batch 16: online/aggregate time=1.056x; instructions=1.040x.
Batch 32: online/aggregate time=1.024x; instructions=1.067x.
Interpret time together with barrier/CBU/LSU deltas: online publication is useful only when producer-consumer overlap repays its repeated readiness and wave synchronization cost.
