# Cooperative Continuation Screen

V100 CUDA-event medians, uint32 Shoup NTT, `logN=20`, 10+10 factorization,
four-row static IO, three warmups, and twenty repetitions.

| batch | v0.6 ms | synchronous warp128 ms | fixed 3+1 ms | cooperative occ-4 ms | cooperative occ-3 ms | occ-3 / fixed 3+1 throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.148 | 0.190 | 0.498 | 0.244 | 0.234 | 2.13x |
| 4 | 0.350 | 0.483 | 1.356 | 0.665 | 0.590 | 2.30x |
| 16 | 1.124 | 1.567 | 4.376 | 2.164 | 2.009 | 2.18x |

The `4 -> 2x2 -> 4` schedule removes the permanently assigned merge warp.
Each warp owns two 128-point prefixes and their stage-7 continuation, two
64-thread groups perform stage 8, and all four warps perform stage 9. It is
numerically correct and improves the fixed 3+1 pipeline by 2.13x--2.30x.
This validates continuation parallelism as the dominant defect in that
prototype.

The occupancy-4 binary compiles to 128 registers/thread and a 104-byte stack
frame. The occupancy-3 binary compiles to 157 registers/thread and a 16-byte
stack frame, and is 1.04x--1.13x faster. Register pressure is therefore a
secondary but measurable limit.

The cooperative core still trails synchronous warp128 by 1.23x--1.29x. The
synchronous core crosses one 128-thread barrier, loads eight continuation
values, and completes stages 7--9 in registers. The explicit hierarchy
materializes intermediate results through shared memory at stage 7 and stage
8 and introduces pair and group barriers. A useful cross-row pipeline must
retain the synchronous register continuation while exposing independent next
row work; adding more intra-row hierarchy alone is not selected.

Matched NCU confirms the resource crossover. Occupancy-4 spills 5.694M local
load sectors and reaches 48.17% long-scoreboard stall. Occupancy-3 eliminates
almost all of that traffic and reduces barrier stall to 4.58%, but active
warps fall to 17.95% and wait stall rises to 23.96%. The selected successor is
therefore two independent synchronous warp128 groups per CTA, not another
cooperative-tail or register-cap point.
