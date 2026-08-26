# Resident M/G Candidate Ablation versus v0.6

CUDA-event medians from order-rotated independent processes. v0.6-best is the faster of the mature Hybrid2D radix-4 path and, at logN=20, the established 10+10 resident dataflow path. M/G-best is selected only among these three new resident-lowering ablations; it is not the best implementation in the complete v0.7 repository.

| bits | logN | batch | v0.6-best | ms | M/G full | lowered generic | lowered core | M/G-best | vs v0.6 |
|---:|---:|---:|:--|---:|---:|---:|---:|:--|---:|
| 32 | 12 | 1 | `v06_hybrid2d` | 0.009216 | 0.040960 | 0.024576 | 0.034816 | `v07_resident_generic` | 0.375x |
| 64 | 12 | 1 | `v06_hybrid2d` | 0.011776 | 0.041984 | 0.023552 | 0.023552 | `v07_resident_generic` | 0.500x |

## Aggregate

Across 2 workload points, M/G-best wins at 0/2 points and reaches 0.433x v0.6-best geometric-mean throughput. Resident boundary lowering contributes 1.724x and the selected physical core contributes 0.840x relative to their immediately preceding layers.

| grouping | value | points | geomean vs v0.6 | wins | min | max |
|:--|---:|---:|---:|---:|---:|---:|
| bits | 32 | 1 | 0.375x | 0/1 | 0.375x | 0.375x |
| bits | 64 | 1 | 0.500x | 0/1 | 0.500x | 0.500x |
| logN | 12 | 2 | 0.433x | 0/2 | 0.375x | 0.500x |
| batch | 1 | 2 | 0.433x | 0/2 | 0.375x | 0.500x |

## Interpretation

This table is a controlled M/G-lowering ablation, not a release-level v0.6/v0.7 comparison. `v07_full` isolates the cost of materializing every logical boundary. `v07_resident_generic` changes only M-to-G lowering, and `v07_resident_core` additionally changes the physical execution-group core. The generated dataflow core is currently specialized only for 10+10; shorter execution groups use the descriptor radix-4 fallback. At logN=20 the dataflow path is physically equivalent to the v0.6 10+10 dataflow schedule, so parity there is expected and is an explicit equivalence check rather than an independent speedup claim.
