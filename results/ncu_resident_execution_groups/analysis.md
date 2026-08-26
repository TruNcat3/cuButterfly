# Resident Execution-Group NCU Attribution

NCU replay time is diagnostic only; CUDA-event measurements remain the performance authority. Ratios below are base/candidate, so values above 1 mean the candidate executes less work.

## Controlled Mechanisms

| batch | change | replay speedup | load sectors | store sectors | warp inst | integer inst | active warps candidate/base | registers | shared KiB | long scoreboard |
|---:|:--|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 1 | boundary-lowering | 2.515x | 2.728x | 2.030x | 5.011x | 2.701x | 0.907x | 72->72 | 1.0->32.0 | +18.80 pp |
| 1 | physical-core | 3.045x | 2.638x | 3.935x | 1.443x | 1.257x | 1.931x | 72->40 | 32.0->16.0 | -24.25 pp |
| 16 | boundary-lowering | 2.527x | 2.560x | 1.787x | 4.906x | 2.965x | 0.993x | 72->72 | 1.0->32.0 | +6.98 pp |
| 16 | physical-core | 3.855x | 2.624x | 4.201x | 1.425x | 1.100x | 1.994x | 72->40 | 32.0->16.0 | -3.43 pp |

Boundary lowering removes two global materializations and also coarsens four logical descriptors into two resident execution groups. The gain therefore appears in sectors, control/LSU work, and total warp instructions even though the larger resident tile uses more shared memory.

The mature dataflow core is a second independent gain. It combines lower sector and instruction demand with 40 rather than 72 registers per thread, about twice the active-warp percentage, and a lower long-scoreboard fraction. Its nonzero barrier cost is more than recovered by occupancy and dependency hiding.

## Logical-M Equivalence

| batch | replay delta | load-sector delta | store-sector delta | warp-inst delta | integer-inst delta | active-warp delta |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -1.01% | +0.16% | +0.00% | -1.27% | -0.00% | +0.74% |
| 16 | +1.39% | -0.14% | +0.00% | +0.40% | +0.00% | +0.47% |

M=2 and M=4 lower to the same G=2 `10+10` dataflow core. Their maximum sector/instruction delta is 1.27%, confirming that logical M no longer introduces hidden runtime work once the physical schedule is identical.
