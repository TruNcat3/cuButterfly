# Dual-Warp128 Vector Radix-4 NCU Attribution

The matched V100 capture compares the current v0.6 resident radix-4 control,
dual-warp128 coefficient-reuse depth 4, and the first vector-radix4 codelet at
uint32 `logN=20`, batch 16. NCU uses base clocks, one replayed cooperative
kernel, and identical fused-Shoup/full-scratch semantics.

| metric | v0.6 | reuse d4 | vector radix-4 |
|:--|--:|--:|--:|
| replay time (us) | 1270.944 | 1409.856 | 1448.032 |
| global-load sectors | 29.782M | 31.575M | 53.291M |
| DRAM read (MiB) | 279.802 | 275.599 | 275.745 |
| warp instructions | 181.752M | 226.770M | 230.054M |
| integer thread instructions | 4.143B | 3.316B | 3.410B |
| active warps | 47.88% | 24.67% | 24.83% |
| barrier stall | 4.93% | 6.38% | 9.72% |
| long-scoreboard stall | 53.04% | 27.33% | 27.66% |
| registers/thread | 40 | 112 | 106 |

The vector codelet is 2.7% slower than reuse d4 under fixed clocks and remains
13.9% slower than v0.6. Occupancy and DRAM bytes do not explain the regression:
active warps are unchanged from reuse d4 and DRAM reads differ by only 0.05%.
Instead, lane-major four-value ownership issues 68.8% more global-load sectors,
mostly cache-resident coefficient requests, while adding 2.8% integer
instructions and raising barrier stall by 3.34 percentage points. The event
lead observed at unlocked clocks is therefore not sufficient evidence for
runtime promotion.

The selected follow-up retains register-local radix-4 but makes coefficient
broadcast depth a compile-time axis. For stages 2--5, only the lanes owning a
unique four-coefficient group load it; other groups receive it through warp
shuffle. Stage 6 remains direct because every active lane owns a distinct
group. The existing reuse-d4 core remains the V100 default until the combined
codelet passes event and matched fixed-clock screens.

The first broadcast implementation is also a negative result. Depths 3--6
remain at 53.17M--53.40M global-load sectors and take
1506.592/1503.648/1486.976/1476.800 us. Reducing the active lanes of each
slot-wise load does not reduce transactions because the coalescer already
combines equal-address requests; the added shuffles instead raise warp
instructions to 234.4M--259.7M. The corrected implementation transposes the
operation: each lane loads one consecutive coefficient once per stage, then
the four value slots shuffle from the lane that owns their offset. This is a
genuine four-load-to-one-load transformation rather than lane masking.

## Distributed-Load Confirmation

The refreshed current-build capture confirms the intended mechanism.

| metric | v0.6 | radix-2 reuse d4 | vector d0 | distributed d6 |
|:--|--:|--:|--:|--:|
| replay time (us) | 1268.832 | 1414.944 | 1424.224 | **1345.696** |
| global-load sectors | 29.793M | 31.726M | 53.403M | **39.989M** |
| warp instructions | 182.865M | 226.803M | 230.085M | 234.654M |
| integer thread instructions | 4.143B | 3.316B | 3.410B | **3.261B** |
| active warps | 47.78% | 24.65% | 24.80% | 24.73% |
| barrier stall | 4.56% | 6.33% | 9.44% | 7.59% |
| long-scoreboard stall | 54.86% | 26.63% | 27.98% | **19.38%** |
| registers/thread | 40 | 112 | 106 | 106 |

Against vector d0, distributed d6 cuts load sectors by 25.1%, integer
instructions by 4.4%, long-scoreboard stall by 8.60 percentage points, and
replay time by 5.5%. Warp instructions rise only 2.0%, active warps are
unchanged, and no local allocation is introduced. The physical transformation
is therefore validated rather than merely correlated with event timing.

Distributed d6 is 5.1% faster than the previous dual-warp128 radix-2 reuse-d4
selection under fixed clocks and becomes the selected dual-warp128 physical
unit. It remains 6.1% slower than the mature v0.6 control: it executes 28.3%
more warp instructions, requests 34.2% more global-load sectors, and exposes
roughly half the active-warp percentage. Consequently this is a v0.7
physical-unit promotion, not a claim that the specialized path replaces v0.6
for every `logN=20`, batch-16 dispatch.
