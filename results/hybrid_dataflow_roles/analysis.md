# Register-resident subgraph role study

This V100 screen isolates `Ur`, the number of consecutive butterfly stages
fused inside one persistent warp role. A role keeps its token fragment in
registers and uses warp shuffle for local exchange. Shared channels exist only
between roles. The source measurements are in [summary.csv](summary.csv).

## Best controlled results

| bits | logN | batch | base (`Ur=1`) ms | best fused point | fused ms | speedup |
|---:|---:|---:|---:|---|---:|---:|
| 32 | 10 | 1 | 0.040940 | `Ur=5,Td=4` | 0.029921 | 1.37x |
| 32 | 10 | 16 | 0.041103 | `Ur=5,Td=4` | 0.030167 | 1.36x |
| 32 | 10 | 256 | 0.060846 | `Ur=5,Td=4` | 0.039322 | 1.55x |
| 64 | 10 | 1 | 0.049193 | `Ur=5,Td=4` | 0.047043 | 1.05x |
| 64 | 10 | 16 | 0.049357 | `Ur=5,Td=4` | 0.047391 | 1.04x |
| 64 | 10 | 256 | 0.072786 | `Ur=5,Td=4` | 0.056545 | 1.29x |
| 64 | 12 | 1 | 0.133857 | `Ur=1,Td=4` | 0.133857 | 1.00x |
| 64 | 12 | 16 | 0.133509 | `Ur=1,Td=4` | 0.133509 | 1.00x |
| 64 | 12 | 256 | 0.528773 | `Ur=6,Td=8` | 0.362025 | 1.46x |

The result validates subgraph fusion, but not a universal fusion depth. At
`logN=10`, eliminating all four shared role boundaries wins for both integer
widths. At uint64 `logN=12`, the fused point needs `Td=8`: `Ur=6,Td=4` is
slower because only four of six data-replicated warps receive a token. Once
the packet is filled, fusion still loses at low batch but wins at saturated
batch, where its smaller channel state and reduced handoff work improve
resident throughput. A focused V100 sweep places the crossover between batch
80 and 160, matching twice the 80-SM device count.

The current uint32 `logN=12` register mapping does not beat `Ur=1`; it is a
real numeric/length boundary, not a reason to select one global policy. It
also indicates that the register/shuffle physical unit needs a word-width
specific ownership map.

## High-performance baseline gap

Fusion materially narrows but does not close the physical-unit gap. At uint64
batch 256, the best dataflow point is 1.58x slower than Tile256 at `logN=10`
and 2.00x slower at `logN=12`. At uint32 `logN=12`, batch 256, the best
dataflow point is 3.51x slower than Hybrid2D radix2. Low-batch gaps are larger
because strict residency assigns one transform to one CTA and cannot amortize
underfilled GPU occupancy.

All 78 cases in this screen report exact reference equality. NCU
counter attribution for the role boundary is collected separately with
`scripts/profile_hybrid_dataflow_roles_ncu.sh`. The completed attribution in
[`../ncu_hybrid_dataflow_roles/analysis.md`](../ncu_hybrid_dataflow_roles/analysis.md)
shows that fusion removes 25%--33% of integer instructions and about 90% of
shared-load conflicts. That capture predates the launch-bounds candidate: the
current `Rb=1` cubin reduces the fused point from 142 to 103 registers without
local allocation, allowing three CTAs/SM at 32768 B shared per CTA. Forcing
`Rb=3` lowers this further to 89 registers but creates no additional residency
and is 2%--26% slower at the three primary batches, so it remains an ablation.
