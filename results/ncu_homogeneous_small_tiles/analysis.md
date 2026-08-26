# Homogeneous Small-Tile NCU Attribution

V100, uint32, `logN=20`, batch 16. All points use the same 10+10
factorization, fused Shoup coefficient tree, four-row skewed static IO, and
natural output. Only the physical warp prefix and occupancy target change.

| point | NCU us | grid | regs | active warps | barrier | wait | long scoreboard | warp inst. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | 1261.216 | 320 | 40 | 47.85% | 5.97% | 10.54% | 42.86% | 181.686M |
| warp256 | 1479.712 | 240 | 164 | 18.42% | 7.98% | 18.17% | 26.08% | 211.339M |
| warp128 occupancy-4 | 1684.384 | 320 | 101 | 24.32% | 5.52% | 18.77% | 35.63% | 237.491M |
| warp64 occupancy-4 | 1680.704 | 320 | 113 | 24.26% | 5.69% | 19.42% | 36.70% | 238.237M |

The occupancy transition is real: both small tiles launch 320 CTAs and raise
active warps by about 32% relative to warp256. Barrier stall also falls by
about 30%. Global load sectors remain within 1% of warp256, so neither the
static boundary layout nor DRAM transaction count explains the regression.

The loss is in the synchronous completion codelet. Small tiles execute about
12.5% more warp instructions and 10--11% more integer thread instructions,
while long-scoreboard stall rises by 37--41%. More resident warps therefore
hide less latency per instruction stream than expected because stages moved
from a register prefix into dependent shared-memory completion. This rejects
"smaller warp tile" as a sufficient optimization, not the architecture-level
space/time mapping.

The first double-buffered row pipeline is retained as an explicit negative
control. A corrected 3-prefix/1-merge assignment with register radix-8 merge
is numerically correct, but reaches only 0.438/1.154/4.257 ms at batch
1/4/16. A subsequent single-slot implementation publishes each 128-point tile
independently and starts stage 7 after a tile pair arrives. It compiles to 128
registers/thread, no static local allocation, about 20.6 KiB shared memory,
and four CTAs/SM, but reaches only 0.497/1.354/4.376 ms. Thus neither excess
buffer storage nor waiting for all eight prefix tiles is the dominant loss.

The stable batch-16 ratio localizes the problem to fixed warp roles. Three
prefix warps eventually become idle while one warp owns every stage-7/8/9
continuation; the row-reuse barrier then prevents them from starting the next
row. The resulting runnable-warp collapse is continuous, so increasing batch
cannot amortize it. The next implementation must make continuation work
stealable or split it across multiple warps. Buffer-depth tuning alone is not
selected.
