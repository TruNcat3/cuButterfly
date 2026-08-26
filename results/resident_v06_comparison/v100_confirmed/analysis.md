# Resident M/G Candidate Ablation versus v0.6

CUDA-event medians from order-rotated independent processes. v0.6-best is the faster of the mature Hybrid2D radix-4 path and, at logN=20, the established 10+10 resident dataflow path. M/G-best is selected only among these three new resident-lowering ablations; it is not the best implementation in the complete v0.7 repository.

| bits | logN | batch | v0.6-best | ms | M/G full | lowered generic | lowered core | M/G-best | vs v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|:--|---:|
| 32 | 12 | 1 | `v06_hybrid2d` | 0.007127 | 0.036516 | 0.021545 | 0.031519 | `v07_resident_generic` | 0.331x |
| 32 | 12 | 4 | `v06_hybrid2d` | 0.007455 | 0.053719 | 0.021750 | 0.031601 | `v07_resident_generic` | 0.343x |
| 32 | 12 | 16 | `v06_hybrid2d` | 0.009093 | 0.120709 | 0.033792 | 0.041738 | `v07_resident_generic` | 0.269x |
| 32 | 14 | 1 | `v06_hybrid2d` | 0.008458 | 0.075121 | 0.023634 | 0.031560 | `v07_resident_generic` | 0.358x |
| 32 | 14 | 4 | `v06_hybrid2d` | 0.010342 | 0.155013 | 0.033464 | 0.035226 | `v07_resident_generic` | 0.309x |
| 32 | 14 | 16 | `v06_hybrid2d` | 0.016589 | 0.467948 | 0.064901 | 0.070246 | `v07_resident_generic` | 0.256x |
| 32 | 16 | 1 | `v06_hybrid2d` | 0.012022 | 0.122143 | 0.041185 | 0.042537 | `v07_resident_generic` | 0.292x |
| 32 | 16 | 4 | `v06_hybrid2d` | 0.020255 | 0.279224 | 0.079544 | 0.083149 | `v07_resident_generic` | 0.255x |
| 32 | 16 | 16 | `v06_hybrid2d` | 0.053105 | 0.895447 | 0.204083 | 0.222822 | `v07_resident_generic` | 0.260x |
| 32 | 18 | 1 | `v06_hybrid2d` | 0.024433 | 0.485294 | 0.112108 | 0.115282 | `v07_resident_generic` | 0.218x |
| 32 | 18 | 4 | `v06_hybrid2d` | 0.061215 | 1.196093 | 0.227533 | 0.245412 | `v07_resident_generic` | 0.269x |
| 32 | 18 | 16 | `v06_hybrid2d` | 0.186061 | 3.512689 | 0.668160 | 0.716964 | `v07_resident_generic` | 0.278x |
| 32 | 20 | 1 | `v06_hybrid2d` | 0.085914 | 0.957604 | 0.370340 | 0.128737 | `v07_resident_core` | 0.667x |
| 32 | 20 | 4 | `v06_hybrid2d` | 0.280760 | 2.768507 | 0.981606 | 0.309105 | `v07_resident_core` | 0.908x |
| 32 | 20 | 16 | `v06_resident` | 1.016299 | 9.507962 | 3.740549 | 1.009623 | `v07_resident_core` | 1.007x |
| 64 | 12 | 1 | `v06_hybrid2d` | 0.007864 | 0.032748 | 0.016712 | 0.016712 | `v07_resident_generic` | 0.471x |
| 64 | 12 | 4 | `v06_hybrid2d` | 0.008028 | 0.050299 | 0.017920 | 0.017940 | `v07_resident_generic` | 0.448x |
| 64 | 12 | 16 | `v06_hybrid2d` | 0.009216 | 0.119173 | 0.029635 | 0.029594 | `v07_resident_core` | 0.311x |
| 64 | 14 | 1 | `v06_hybrid2d` | 0.009155 | 0.072991 | 0.020521 | 0.020521 | `v07_resident_generic` | 0.446x |
| 64 | 14 | 4 | `v06_hybrid2d` | 0.010445 | 0.158761 | 0.029082 | 0.029061 | `v07_resident_core` | 0.359x |
| 64 | 14 | 16 | `v06_hybrid2d` | 0.020480 | 0.489329 | 0.064225 | 0.064143 | `v07_resident_core` | 0.319x |
| 64 | 16 | 1 | `v06_hybrid2d` | 0.015524 | 0.123699 | 0.038973 | 0.038932 | `v07_resident_core` | 0.399x |
| 64 | 16 | 4 | `v06_hybrid2d` | 0.029532 | 0.289321 | 0.076739 | 0.076800 | `v07_resident_generic` | 0.385x |
| 64 | 16 | 16 | `v06_hybrid2d` | 0.082063 | 0.938783 | 0.200827 | 0.200970 | `v07_resident_generic` | 0.409x |
| 64 | 18 | 1 | `v06_hybrid2d` | 0.039158 | 0.511980 | 0.119112 | 0.119132 | `v07_resident_generic` | 0.329x |
| 64 | 18 | 4 | `v06_hybrid2d` | 0.103096 | 1.341194 | 0.288727 | 0.287785 | `v07_resident_core` | 0.358x |
| 64 | 18 | 16 | `v06_hybrid2d` | 0.341443 | 4.299305 | 0.862740 | 0.865014 | `v07_resident_generic` | 0.396x |
| 64 | 20 | 1 | `v06_hybrid2d` | 0.131994 | 1.179853 | 0.512778 | 0.203407 | `v07_resident_core` | 0.649x |
| 64 | 20 | 4 | `v06_hybrid2d` | 0.428339 | 3.225723 | 1.240187 | 0.553718 | `v07_resident_core` | 0.774x |
| 64 | 20 | 16 | `v06_hybrid2d` | 1.609196 | 11.788821 | 4.303053 | 1.840476 | `v07_resident_core` | 0.874x |

## Aggregate

Across 30 workload points, M/G-best wins at 1/30 points and reaches 0.394x v0.6-best geometric-mean throughput. Resident boundary lowering contributes 3.598x and the selected physical core contributes 1.157x relative to their immediately preceding layers.

| grouping | value | points | geomean vs v0.6 | wins | min | max |
|:--|---:|---:|---:|---:|---:|---:|
| bits | 32 | 15 | 0.353x | 1/15 | 0.218x | 1.007x |
| bits | 64 | 15 | 0.439x | 0/15 | 0.311x | 0.874x |
| logN | 12 | 6 | 0.355x | 0/6 | 0.269x | 0.471x |
| logN | 14 | 6 | 0.336x | 0/6 | 0.256x | 0.446x |
| logN | 16 | 6 | 0.327x | 0/6 | 0.255x | 0.409x |
| logN | 18 | 6 | 0.302x | 0/6 | 0.218x | 0.396x |
| logN | 20 | 6 | 0.803x | 1/6 | 0.649x | 1.007x |
| batch | 1 | 10 | 0.394x | 0/10 | 0.218x | 0.667x |
| batch | 4 | 10 | 0.403x | 0/10 | 0.255x | 0.908x |
| batch | 16 | 10 | 0.384x | 1/10 | 0.256x | 1.007x |

## Interpretation

This table is a controlled M/G-lowering ablation, not a release-level v0.6/v0.7 comparison. `v07_full` isolates the cost of materializing every logical boundary. `v07_resident_generic` changes only M-to-G lowering, and `v07_resident_core` additionally changes the physical execution-group core. The generated dataflow core is currently specialized only for 10+10; shorter execution groups use the descriptor radix-4 fallback. At logN=20 the dataflow path is physically equivalent to the v0.6 10+10 dataflow schedule, so parity there is expected and is an explicit equivalence check rather than an independent speedup claim.
