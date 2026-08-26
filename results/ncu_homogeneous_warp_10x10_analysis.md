# Warp-Granular 10+10 NCU Analysis

V100, uint32, `logN=20`, batch 16. The first capture predates the static
writer and isolates the original warp-core bottleneck; the subsequent matched
captures add static writer and static IO points under the same workload.

| point | time (us) | DRAM read/write (MiB) | DRAM peak | active warps | barrier | wait | registers | shared (B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | 1075.520 | 261.698 / 143.136 | 43.94% | 48.87% | 5.64% | 42.72% | 40 | 16416 |
| 1024/warp | 4560.672 | 299.889 / 485.172 | 20.06% | 12.49% | 0.01% | 11.20% | 159 | 0 |
| 256/warp, 128 threads | 4615.104 | 803.743 / 590.644 | 35.72% | 18.56% | 27.09% | 28.02% | 161 | 4096 |
| 256/warp, 256 threads | 3439.552 | 335.646 / 566.175 | 30.51% | 12.49% | 24.14% | 21.16% | 161 | 8192 |

The original warp cores do not merely execute more radix-2 instructions. Their
one-row-at-a-time edge stores inflate DRAM writes by 3.4--4.1x relative to
v0.6, while the 128-thread point also inflates reads by 3.1x. The complete-warp
core is register-limited to one CTA/SM. The 256-point core exposes more groups,
but its row-strided global edges turn that parallelism into transaction
amplification. These counters motivated the word-size-dependent static writer.

## Static-Writer Attribution

The matched rerun adds the first static-writer core:

| point | time (us) | load/store sectors | DRAM read/write (MiB) | warp inst. | integer thread inst. | active warps | barrier | shared (B) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v0.6 | 1062.240 | 29.801M / 8.389M | 258.518 / 138.395 | 187.692M | 4.154B | 48.69% | 5.46% | 16416 |
| plain warp256, 128T | 4351.296 | 53.160M / 33.554M | 794.644 / 576.774 | 198.687M | 2.514B | 18.56% | 26.88% | 4096 |
| static writer, 128T | 2451.552 | 52.882M / 4.194M | 357.388 / 128.668 | 195.085M | 2.563B | 12.35% | 26.46% | 36864 |

Static output mapping reduces store sectors by exactly 8x relative to the
plain warp256 core and to half of v0.6, while DRAM writes fall below v0.6. Warp
and integer instruction counts barely change relative to plain warp256. The
remaining gap is therefore not radix instruction volume: static-writer warp
instructions are only 1.039x v0.6 and integer instructions are 0.617x. The
unresolved costs are 1.774x load sectors, one quarter of v0.6 active warps, and
13.5% versus 5.46% barrier stalls. This selects coalesced input and coefficient
reuse as the next experiments.

## Static-IO Attribution

Before the packet-level shared swizzle, full static IO reduces load sectors
from 52.883M to 38.074M and DRAM reads from 358.368 MiB to 253.720 MiB. Store
sectors remain optimal at 4.194M. However, per-row bit-reverse staging creates
17.980M/19.416M shared load/store bank conflicts; barrier stalls remain 16.40%
versus v0.6's 5.52%, long-scoreboard stalls reach 18.28%, and MIO throttle
rises from 0.37% to 5.09%. This identifies the
remaining read-side loss as an on-chip layout problem, not DRAM or coefficient
traffic.

The integrated implementation now uses `P(i)=i xor (i>>5)` for the entire
input packet, register subgraph, and output packet. It replaces one barrier per
row with one barrier per packet and keeps unconsumed rows in disjoint swizzled
slots. The same template exposes full and half row packets through
`NttSubgraphMapping.data_space`; the latter trades half-sector stores for a
third resident 128-thread CTA when registers permit it.

The matched rerun confirms that packet-wide XOR reduces shared load/store bank
conflicts from 17.980M/19.416M to 4.813M/6.359M and lowers the full-packet NCU
time to 2.073 ms. It also exposes the next limit: 172 registers per thread cap
both full and half static-IO kernels at two CTAs/SM. The half point therefore
launches 160 rather than the requested 240 blocks and takes 2.184 ms.

Implemented after this capture, the uint32 four-row specialization uses a
row-major tile with a seven-word bank skew. It compiles to 164 registers with
no local memory, launches three CTAs/SM, and reaches 0.837x/0.831x/0.822x v0.6
throughput at batch 1/4/16. The updated profiling script applies the measured
11:9, 8:7, and 17:13 role weights for those batches so the next capture profiles
the selected physical point rather than the obsolete 9:11 mapping.

## Skewed Half-Packet Confirmation

The selected uint32 batch-16 rerun confirms the mechanism:

| metric | previous XOR half | skewed half | v0.6 |
|---|---:|---:|---:|
| NCU time (us) | 2184.384 | 1455.616 | 1061.120 |
| launched blocks | 160 | 240 | 320 |
| registers/thread | 172 | 164 | 40 |
| active warps | 12.45% | 18.44% | 48.88% |
| shared load conflicts | 4.317M | 2.482M | 11.968M |
| shared store conflicts | 6.291M | 0 | 27.292M |
| barrier stall | 18.06% | 7.44% | 5.76% |
| wait stall | 18.54% | 18.46% | 10.44% |
| warp instructions | 218.509M | 211.083M | 185.759M |
| integer thread instructions | 3.257B | 3.076B | 4.154B |

The new point is 1.50x faster than the previous half-packet kernel. Shared
bank pressure, MIO throttle, and long-scoreboard pressure are all below or near
the control and are no longer the primary limit. The remaining NCU gap is a
physical-core latency-hiding problem: three 128-thread CTAs provide only 12
resident warps versus v0.6's 32, while fixed-latency `wait` stall is 1.77x the
control and warp instruction volume is 1.136x. Global load sectors are also
1.344x v0.6, but total DRAM bytes increase by only about 7%, so traffic alone
cannot explain the 37% NCU time gap.

The next core screen should therefore reduce the register live set rather than
add another boundary layout: 128- and 64-point warp subgraphs halve or quarter
the lane-owned value array, then finish the remaining stages through the same
packet tile. The success criterion is at least four 128-thread CTAs/SM without
local-memory traffic; radix-4/8 codelets remain alternatives inside each warp
subgraph.

## Small-Tile Occupancy Screen

The generated 128- and 64-point static-IO instances pass random verification
against v0.6. Their first integration also exposed a generator contract bug:
the new physical-core enums were not registered as specialized 10+10
consumers, so the ordinary root-power array was interpreted as a per-row fused
coefficient tree. Registering coefficient layout alongside launch selection
fixes both cores; future generated cores must declare this layout contract.

An occupancy-4 wrapper produces the intended V100 resource transition:

| point | registers/thread | requested CTAs/SM | event ms | throughput/v0.6 |
|---|---:|---:|---:|---:|
| v0.6 | 40 | 4 | 1.094196 | 1.000x |
| warp256 skewed half | 164 | 3 | 1.361562 | 0.804x |
| warp128 skewed half | 101 | 4 | 1.609574 | 0.680x |
| warp64 skewed half | 113 | 4 | 1.659750 | 0.659x |

These are two-trial CUDA-event medians at uint32 `logN=20`, batch 16; the raw
screen is in `results/homogeneous_warp_small_tiles_occupancy4/`. The occupancy
goal is met without local memory, but performance falls because all four warps
still rendezvous for each row and the smaller prefixes move more stages into
shared completion. NCU should next compare only the selected warp256 control
and the two occupancy-4 points. The architectural follow-up is a prefix/merge
warp pipeline with independent role counts and a shared ring, so prefix work
on row `r+1` overlaps merge work on row `r`.

The focused counter command is:

```bash
sudo -E ./scripts/profile_homogeneous_small_tiles_ncu.sh
```
