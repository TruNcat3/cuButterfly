# Packed Stage-6 Vector Load

| metric | v0.6 | vector d6 | vector d7 | packed stage6 | packed/d6 | packed/d7 |
|:--|--:|--:|--:|--:|--:|--:|
| time (us) | 1274.688 | 1377.248 | 1339.584 | 1434.240 | 1.041x | 1.071x |
| DRAM read (MiB) | 275.755 | 277.516 | 272.631 | 292.572 | 1.054x | 1.073x |
| global-load sectors | 29,758,260 | 39,934,164 | 31,343,547 | 26,726,090 | 0.669x | 0.853x |
| global-store sectors | 8,388,611 | 8,337,410 | 8,428,211 | 8,336,737 | 1.000x | 0.989x |
| warp instructions | 182,383,521 | 234,710,495 | 237,191,754 | 237,070,007 | 1.010x | 0.999x |
| local-load sectors | 0.000 | 108,967 | 101,333 | 88387.000 | 0.811x | 0.872x |
| local-store sectors | 16384.000 | 16384.000 | 16384.000 | 16384.000 | 1.000x | 1.000x |
| registers/thread | 40.000 | 106.000 | 105.000 | 98.000 | 0.925x | 0.933x |
| active warps (%) | 47.750 | 24.720 | 24.730 | 24.640 | 0.997x | 0.996x |
| long scoreboard stall (%) | 43.310 | 19.610 | 17.230 | 21.940 | 1.119x | 1.273x |
| MIO throttle stall (%) | 5.500 | 2.940 | 3.520 | 2.780 | 0.946x | 0.790x |

## Decision

The packed table improves one control but not both. Retain it as an experimental physical core and use the counter deltas to choose the next codelet change.
Packed/d6 time is 1.041x, packed/d7 time is 1.071x, and packed/d7 warp instructions are 0.999x.
