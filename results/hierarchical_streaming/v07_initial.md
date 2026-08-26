# v0.7 readiness-streaming initial checkpoint

Device: Tesla V100-SXM2-16GB (`sm_70`). All rows use exact modular
verification. Timings are CUDA-event kernel medians represented by the current
benchmark average after warmup; they are an implementation checkpoint, not a
release performance claim.

## Correctness and overlap

The implementation passes forward/inverse uint32/uint64 NTTs, two- and
three-segment decompositions, radix-2/4/8 physical subgraph cores, full-scratch
edges, and epoch-protected ring edges. `%globaltimer` traces prove computation
overlap for `logN=20,batch=1,partition=7+7+6`:

| segment | start | end |
|---:|---:|---:|
| 0 | 1786970045536564192 | 1786970045537395680 |
| 1 | 1786970045537339360 | 1786970045538238432 |
| 2 | 1786970045537401824 | 1786970045538693088 |

Segment 1 starts before segment 0 ends, and segment 2 starts before segment 1
ends. In contrast, `10+10,batch=1` does not overlap: each second-segment task
depends on all first-segment tasks. It remains a useful low-boundary negative
control and becomes pipeline-fillable through independent batch transforms.

## Current performance gap

| word bits | logN | batch | implementation | partition | kernel ms | relative to barrier |
|---:|---:|---:|:--|:--|---:|---:|
| 64 | 12 | 2 | HierarchicalBarrier radix-4 | 6+6 | 0.01075 | 1.000x |
| 64 | 12 | 2 | v0.7 full-scratch generic radix-4 | 6+6 | 0.02703 | 0.398x |
| 64 | 15 | 2 | HierarchicalBarrier radix-4 | 8+7 | 0.01618 | 1.000x |
| 64 | 15 | 2 | v0.7 full-scratch generic radix-4 | 5+5+5 | 0.21688 | 0.075x |
| 64 | 20 | 1 | HierarchicalBarrier radix-4 | 10+10 | 0.15591 | 1.000x |
| 64 | 20 | 1 | v0.7 full-scratch generic radix-4 | 7+7+6 | 2.13572 | 0.073x |

The result separates architectural correctness from physical-core maturity.
The true wavefront exists, but the current generic core pays scattered atomic
readiness polling, extra materialized edges, and runtime index work. The
release selector must therefore retain v0.6/Hybrid2D fallbacks until generated
specialized cores remove this gap.

## Ring tradeoff

For `logN=12,batch=5`, full scratch uses about 165,152 bytes and runs in
0.04139 ms. A two-slot ring uses about 66,104 bytes (40% of full scratch) and
runs in 0.33231 ms. The epoch protocol is correct but currently an explicit
memory-capacity option, not a speed choice.
