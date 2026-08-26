# Packed Stage-6 Vector Load

| metric | v0.6 | vector d6 | vector d7 | packed vector16 | packed lane32 | lane32/vector16 | lane32/d7 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| time (us) | 1252.960 | 1358.464 | 1343.360 | 1446.944 | 1336.288 | 0.924x | 0.995x |
| DRAM read (MiB) | 274.507 | 280.845 | 274.677 | 287.272 | 273.782 | 0.953x | 0.997x |
| DRAM write (MiB) | 163.390 | 141.541 | 138.300 | 149.863 | 136.938 | 0.914x | 0.990x |
| global-load sectors | 29,806,966 | 39,686,138 | 31,391,182 | 26,369,861 | 26,360,967 | 1.000x | 0.840x |
| global-store sectors | 8,388,611 | 8,459,205 | 8,368,865 | 8,440,812 | 8,332,930 | 0.987x | 0.996x |
| L2 hit (%) | 65.990 | 62.660 | 61.320 | 60.210 | 61.870 | 1.028x | 1.009x |
| L1/TEX hit (%) | 47.350 | 63.500 | 56.560 | 50.850 | 50.860 | 1.000x | 0.899x |
| warp instructions | 181,160,397 | 234,658,637 | 237,178,866 | 237,027,761 | 237,692,531 | 1.003x | 1.002x |
| local-load sectors | 0.000 | 105,772 | 100,384 | 88364.000 | 109,098 | 1.235x | 1.087x |
| local-store sectors | 16384.000 | 16384.000 | 16384.000 | 16384.000 | 16384.000 | 1.000x | 1.000x |
| registers/thread | 40.000 | 106.000 | 105.000 | 98.000 | 109.000 | 1.112x | 1.038x |
| active warps (%) | 47.880 | 24.710 | 24.690 | 24.640 | 24.720 | 1.003x | 1.001x |
| long scoreboard stall (%) | 43.660 | 19.400 | 17.390 | 22.050 | 16.470 | 0.747x | 0.947x |
| MIO throttle stall (%) | 5.490 | 3.000 | 3.400 | 2.720 | 3.050 | 1.121x | 0.897x |

## Decision

The 32-lane packed core combines aligned sectors with recovered memory-level parallelism. Promote it to the next v0.7 schedule search while retaining the other physical cores as ablations.
Lane32/vector16 time is 0.924x, lane32/d7 time is 0.995x, and lane32/d7 warp instructions are 1.002x.
