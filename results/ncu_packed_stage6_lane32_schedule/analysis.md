# Packed Lane32 Fixed-Clock Schedule Confirmation

| metric | v0.6 | d7 85:75 | lane32 84:76 | lane32 85:75 | lane32 86:74 |
|:--|--:|--:|--:|--:|--:|
| time (us) | 1255.040 | 1330.016 | 1333.984 | 1340.640 | 1357.088 |
| global-load sectors | 29,843,718 | 31,436,848 | 26,840,271 | 26,451,012 | 26,642,936 |
| DRAM read (MiB) | 274.286 | 271.605 | 273.191 | 276.438 | 276.511 |
| L2 hit (%) | 66.240 | 62.200 | 62.890 | 61.140 | 61.840 |
| warp instructions | 182,194,690 | 237,146,497 | 237,765,855 | 237,688,735 | 237,601,161 |
| long scoreboard stall (%) | 42.510 | 17.410 | 16.580 | 16.660 | 16.860 |
| active warps (%) | 47.620 | 24.750 | 24.630 | 24.720 | 24.650 |
| registers/thread | 40.000 | 105.000 | 109.000 | 109.000 | 109.000 |

## Decision

The tuned lane32 point does not beat d7 under fixed clock; do not promote it.
Best lane32 weights=84:76, time=1333.984 us, throughput/v0.6=0.941x, throughput/d7=0.997x.
