# Dual-Warp128 Vector Radix-4 Screen

## Controlled Change

The new `homogeneous-warp128-vector-radix4-static-io` physical core preserves
the `10+10` decomposition, dual independent row groups, static IO mapping,
full-scratch boundary, and producer/consumer service schedule. Each lane owns
four consecutive values. Stages 0 and 1 form one register-local radix-4
prefix; stages 2 through 6 use warp shuffle before the existing shared-memory
continuation. This changes the physical codelet without changing the logical
space-time mapping.

## V100 Event Screen

The screen uses uint32 `logN=20`, batch 16, modulus 998244353, fused Shoup
twiddles, `10+10`, and three execution-order-interleaved processes with ten
warmups and fifty timed repetitions per process.

| physical point | median kernel ms | throughput/vector |
|:--|--:|--:|
| v0.6 resident radix-4 | 1.514537 | 0.818x |
| dual warp128, coefficient reuse depth 4 | 1.275003 | 0.971x |
| dual warp128, vector radix-4 | 1.238446 | 1.000x |

The vector core improves median throughput by 2.95% over the selected
coefficient-reuse codelet and by 22.29% over the current-build v0.6 event
control. CTA-weight confirmation gives 1.237586/1.237975/1.239061 ms medians
for 8:7, 9:8, and 11:9, so the existing 8:7 mapping remains selected without
introducing a narrower workload-specific default.

Cubin resource inspection reports 106 registers/thread, a 16-byte stack frame,
and no local memory for vector radix-4. The depth-4 coefficient-reuse control
uses 112 registers/thread with the same stack and no local allocation. The
register-local radix-4 substitution therefore reduces both dynamic codelet
work and live state; it does not buy event time through spilling or lower CTA
residency.

The subsequent fixed-clock NCU pass does not reproduce the event lead. Vector
radix-4 takes 1448.032 us versus 1409.856 us for reuse d4 and 1270.944 us for
v0.6. It issues 68.8% more global-load sectors than reuse d4 despite unchanged
DRAM bytes and active warps, and barrier stall rises from 6.38% to 9.72%.
Consequently vector radix-4 is not promoted into the V100 runtime default.
The counter table and follow-up decision are in
[`results/ncu_homogeneous_vector_radix4/analysis.md`](../ncu_homogeneous_vector_radix4/analysis.md).

## Coefficient-Broadcast Follow-Up

The combined codelet exposes depths 3--6. Depth `d` broadcasts coefficient
groups for vector stages below `d`; stage 6 remains direct. All depths pass the
full NTT reference suite and compile without local allocation.

| point | registers/thread | event median ms |
|:--|--:|--:|
| radix-2 reuse d4 | 112 | 1.273815 |
| vector d0 | 106 | 1.239183 |
| vector d3 | 114 | 1.327022 |
| vector d4 | 112 | 1.329726 |
| vector d5 | 128 | 1.283973 |
| vector d6 | 128 | 1.292083 |

Broadcasting only the earliest repeated groups loses more shuffle/control work
than it saves. Depth 5 recovers most of that cost and is within 0.8% of the
radix-2 reuse control, while depth 6 regresses because stage-5 broadcast adds
no useful event-time overlap at the 128-register threshold. Since unlocked
events and fixed-clock replay rank vector d0 differently, all depths remain
ablation points until the expanded NCU script measures their load-sector and
instruction curves.

The first NCU pass over those depths reveals that lane masking does not reduce
load sectors: all depths remain near 53.2M. The corrected distributed-load
form assigns one consecutive coefficient to each lane and shuffles it to the
four register slots. It reduces the compiled register counts for d3/d4/d5/d6
to 103/105/103/106, with no local allocation, and produces the following
five-process medians:

| point | median ms | throughput/reuse-d4 |
|:--|--:|--:|
| radix-2 reuse d4 | 1.272627 | 1.000x |
| vector d0 | 1.239081 | 1.027x |
| distributed d3 | 1.316659 | 0.967x |
| distributed d4 | 1.329910 | 0.957x |
| distributed d5 | 1.238753 | 1.027x |
| distributed d6 | **1.201336** | **1.059x** |

The non-monotonic response is architectural: one or two distributed stages do
not amortize coefficient shuffles, while covering stages 2--5 removes enough
load instructions to expose the vector arithmetic gain. Unlike the first
broadcast attempt, d6 keeps the plain core's 106-register footprint and two
CTA/SM residency. A refreshed fixed-clock capture is still required before
runtime promotion because the earlier d0 event/NCU rankings disagreed.

The refreshed NCU pass confirms d6 at 1345.696 us, versus 1414.944 us for
radix-2 reuse d4 and 1424.224 us for vector d0. Its 39.989M load sectors are
25.1% below d0, while registers and active warps remain unchanged. Distributed
d6 is therefore promoted as the selected dual-warp128 physical unit. The
mature v0.6 control still leads fixed-clock replay at 1268.832 us, so global
dispatch retains that boundary.
