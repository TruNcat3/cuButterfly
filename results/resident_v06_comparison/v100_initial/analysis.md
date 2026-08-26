# Resident M/G Candidate Ablation versus v0.6

CUDA-event medians from order-rotated independent processes. v0.6-best is the faster of the mature Hybrid2D radix-4 path and, at logN=20, the established 10+10 resident dataflow path. M/G-best is selected only among these three new resident-lowering ablations; it is not the best implementation in the complete v0.7 repository.

| bits | logN | batch | v0.6-best | ms | M/G full | lowered generic | lowered core | M/G-best | vs v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|:--|---:|
| 32 | 12 | 1 | `v06_hybrid2d` | 0.007127 | 0.036557 | 0.021484 | 0.031478 | `v07_resident_generic` | 0.332x |
| 32 | 12 | 4 | `v06_hybrid2d` | 0.007434 | 0.053555 | 0.021688 | 0.031539 | `v07_resident_generic` | 0.343x |
| 32 | 12 | 16 | `v06_hybrid2d` | 0.009073 | 0.120668 | 0.033731 | 0.041677 | `v07_resident_generic` | 0.269x |
| 32 | 14 | 1 | `v06_hybrid2d` | 0.008520 | 0.074977 | 0.023654 | 0.031539 | `v07_resident_generic` | 0.360x |
| 32 | 14 | 4 | `v06_hybrid2d` | 0.010322 | 0.155034 | 0.033423 | 0.035185 | `v07_resident_generic` | 0.309x |
| 32 | 14 | 16 | `v06_hybrid2d` | 0.016589 | 0.468029 | 0.064983 | 0.070410 | `v07_resident_generic` | 0.255x |
| 32 | 16 | 1 | `v06_hybrid2d` | 0.012083 | 0.123290 | 0.041247 | 0.042639 | `v07_resident_generic` | 0.293x |
| 32 | 16 | 4 | `v06_hybrid2d` | 0.020214 | 0.278385 | 0.079258 | 0.083128 | `v07_resident_generic` | 0.255x |
| 32 | 16 | 16 | `v06_hybrid2d` | 0.053166 | 0.894280 | 0.204001 | 0.222290 | `v07_resident_generic` | 0.261x |
| 32 | 18 | 1 | `v06_hybrid2d` | 0.024433 | 0.485233 | 0.112087 | 0.115302 | `v07_resident_generic` | 0.218x |
| 32 | 18 | 4 | `v06_hybrid2d` | 0.058040 | 1.022648 | 0.214282 | 0.229540 | `v07_resident_generic` | 0.271x |
| 32 | 18 | 16 | `v06_hybrid2d` | 0.186061 | 3.510047 | 0.667546 | 0.716390 | `v07_resident_generic` | 0.279x |
| 32 | 20 | 1 | `v06_hybrid2d` | 0.085729 | 0.958300 | 0.370483 | 0.129823 | `v07_resident_core` | 0.660x |
| 32 | 20 | 4 | `v06_hybrid2d` | 0.280699 | 2.607739 | 0.986849 | 0.325775 | `v07_resident_core` | 0.862x |
| 32 | 20 | 16 | `v06_resident` | 1.007534 | 9.489613 | 3.744625 | 1.034404 | `v07_resident_core` | 0.974x |
| 64 | 12 | 1 | `v06_hybrid2d` | 0.007864 | 0.032748 | 0.016712 | 0.016712 | `v07_resident_generic` | 0.471x |
| 64 | 12 | 4 | `v06_hybrid2d` | 0.008028 | 0.050217 | 0.017818 | 0.017940 | `v07_resident_generic` | 0.451x |
| 64 | 12 | 16 | `v06_hybrid2d` | 0.009216 | 0.119255 | 0.029635 | 0.029655 | `v07_resident_generic` | 0.311x |
| 64 | 14 | 1 | `v06_hybrid2d` | 0.009175 | 0.072950 | 0.020500 | 0.020500 | `v07_resident_generic` | 0.448x |
| 64 | 14 | 4 | `v06_hybrid2d` | 0.010465 | 0.158986 | 0.029041 | 0.029082 | `v07_resident_generic` | 0.360x |
| 64 | 14 | 16 | `v06_hybrid2d` | 0.020460 | 0.488899 | 0.064184 | 0.064225 | `v07_resident_generic` | 0.319x |
| 64 | 16 | 1 | `v06_hybrid2d` | 0.015524 | 0.123720 | 0.038953 | 0.038994 | `v07_resident_generic` | 0.399x |
| 64 | 16 | 4 | `v06_hybrid2d` | 0.029450 | 0.289034 | 0.076718 | 0.076698 | `v07_resident_core` | 0.384x |
| 64 | 16 | 16 | `v06_hybrid2d` | 0.082022 | 0.937963 | 0.201011 | 0.200806 | `v07_resident_core` | 0.408x |
| 64 | 18 | 1 | `v06_hybrid2d` | 0.039137 | 0.511365 | 0.119009 | 0.119214 | `v07_resident_generic` | 0.329x |
| 64 | 18 | 4 | `v06_hybrid2d` | 0.102605 | 1.340621 | 0.287130 | 0.287662 | `v07_resident_generic` | 0.357x |
| 64 | 18 | 16 | `v06_hybrid2d` | 0.341443 | 4.297810 | 0.865608 | 0.858665 | `v07_resident_core` | 0.398x |
| 64 | 20 | 1 | `v06_hybrid2d` | 0.132076 | 1.183375 | 0.512614 | 0.225915 | `v07_resident_core` | 0.585x |
| 64 | 20 | 4 | `v06_hybrid2d` | 0.428442 | 3.226604 | 1.241027 | 0.560026 | `v07_resident_core` | 0.765x |
| 64 | 20 | 16 | `v06_hybrid2d` | 1.609114 | 11.778806 | 4.301599 | 1.933763 | `v07_resident_core` | 0.832x |

## Aggregate

Across 30 workload points, M/G-best wins at 0/30 points and reaches 0.391x v0.6-best geometric-mean throughput. Resident boundary lowering contributes 3.581x and the selected physical core contributes 1.148x relative to their immediately preceding layers.

| grouping | value | points | geomean vs v0.6 | wins | min | max |
|:--|---:|---:|---:|---:|---:|---:|
| bits | 32 | 15 | 0.351x | 0/15 | 0.218x | 0.974x |
| bits | 64 | 15 | 0.434x | 0/15 | 0.311x | 0.832x |
| logN | 12 | 6 | 0.355x | 0/6 | 0.269x | 0.471x |
| logN | 14 | 6 | 0.337x | 0/6 | 0.255x | 0.448x |
| logN | 16 | 6 | 0.327x | 0/6 | 0.255x | 0.408x |
| logN | 18 | 6 | 0.303x | 0/6 | 0.218x | 0.398x |
| logN | 20 | 6 | 0.769x | 0/6 | 0.585x | 0.974x |
| batch | 1 | 10 | 0.390x | 0/10 | 0.218x | 0.660x |
| batch | 4 | 10 | 0.401x | 0/10 | 0.255x | 0.862x |
| batch | 16 | 10 | 0.381x | 0/10 | 0.255x | 0.974x |

## Interpretation

This table is a controlled M/G-lowering ablation, not a release-level v0.6/v0.7 comparison. `v07_full` isolates the cost of materializing every logical boundary. `v07_resident_generic` changes only M-to-G lowering, and `v07_resident_core` additionally changes the physical execution-group core. The generated dataflow core is currently specialized only for 10+10; shorter execution groups use the descriptor radix-4 fallback. At logN=20 the dataflow path is physically equivalent to the v0.6 10+10 dataflow schedule, so parity there is expected and is an explicit equivalence check rather than an independent speedup claim.
