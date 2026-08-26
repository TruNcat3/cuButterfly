# HybridDataflow fused-role NCU attribution

This V100 capture profiles one uint64 batch-16 launch per generated `Ur/Td`
point. The source table is [summary.csv](summary.csv). NCU replay time is used
for controlled attribution; the CUDA-event batch sweep remains the performance
reference. This capture predates the generated `Rb` launch-bounds dimension;
its instruction/channel conclusions remain valid, while current register counts
are reported in the role study rather than inferred from this table.

## Counter comparison

| point | time (us) | warp inst. | integer inst. | shared load conflicts | registers/thread | shared/CTA | barrier stall | long scoreboard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| n10 `Ur=1,Td=4` | 58.752 | 0.839 M | 11.027 M | 8879 | 63 | 16416 B | 35.13% | 3.10% |
| n10 `Ur=5,Td=4` | 52.832 | 0.594 M | 8.262 M | 652 | 78 | 8192 B | 17.26% | 24.50% |
| n12 `Ur=1,Td=4` | 151.776 | 3.281 M | 39.140 M | 35458 | 63 | 53288 B | 25.87% | 3.52% |
| n12 `Ur=1,Td=8` | 172.544 | 2.941 M | 34.730 M | 35318 | 64 | 73768 B | 40.40% | 4.37% |
| n12 `Ur=6,Td=8` | 160.448 | 2.049 M | 26.203 M | 3652 | 142 | 32768 B | 25.80% | 15.63% |

At `logN=10`, full fusion improves replay time by 1.11x. It removes 29.2% of
warp instructions, 25.1% of integer instructions, 92.7% of shared-load bank
conflicts, all measured shared-store conflicts, and half of the dynamic shared
allocation. This directly validates that the implementation is computing a
register-resident subgraph rather than replaying single-stage shared kernels.

At `logN=12`, comparing equal `Td=8`, full fusion is 1.08x faster and removes
30.3% of warp instructions, 24.6% of integer instructions, 89.7% of shared-load
conflicts, and all measured shared-store conflicts. Against the best low-batch
base point (`Ur=1,Td=4`), however, fusion is 5.7% slower under replay despite
37.5%/33.1% lower warp/integer instruction counts.

## Batch crossover mechanism

The remaining cost moves from shared materialization to a register dependency
chain. `Ur=6,Td=8` raises registers from 63 to 142 per thread and long-scoreboard
stall from 3.52% to 15.63%. Static cubin resource data reports 16 B of stack and
`LOCAL:0`, so this capture does not support a spill explanation. A fused warp
instead executes six dependent twiddle/modular stages before another role can
take over, exposing arithmetic and lookup latency.

For the implementation captured here, the resource transition explains the CUDA-event batch crossover. The base
point's 53288 B shared allocation permits only one CTA per 98304 B V100 SM.
The fused point uses 32768 B; its 142-register footprint becomes the limit at
two CTAs per SM. Batch 1/16 cannot use that extra residency and remains neutral,
while batch 256 supplies enough CTAs for latency hiding and reaches the measured
1.66x fused-role speedup. The subsequent launch-bounds implementation lowers
the normal fused point to 103 registers and three CTAs/SM; its refreshed
batch-256 speedup is 1.46x because the base point also benefits from the new
code generation.

The resulting method is piecewise: increase `Ur` while saved channel work and
CTA residency dominate, increase `Td` until role replicas are filled, and stop
when register dependency/occupancy cost dominates. Word width, subgraph length,
and batch all move this boundary.
