# Design Overview

## 1. Why A Mapping Paradigm Is Needed

FFT, NTT, FWHT, and XOR-zeta use different arithmetic, but their regular
power-of-two forms share the same dependency graph. For `N=2^m`, each stage
contains independent butterfly updates; an element produced at stage `s` is
consumed by a prescribed partner at stage `s+1`. A GPU implementation must
therefore answer two different questions:

1. how much independent data work should execute concurrently; and
2. how many dependent stage groups should remain on chip before ownership or
   layout changes.

Specialized libraries often answer both questions inside one highly tuned
kernel. That is effective for one operator and device, but it entangles the
arithmetic codelet with scheduling, storage, synchronization, and layout.
cuButterfly separates them. Its contribution is the architecture-level
organization of the dependency graph; a radix codelet, modular reduction,
cuFFTDx block, register FWHT, or register Structured hierarchy remains a replaceable physical
processing unit.

The separation is represented by seven objects:

```text
G: mathematical graph and exact transform semantics
A: architecture-level space-time mapping
P: replaceable processing unit
L: global and intermediate layout
F: concrete GPU kernel realization
Q: generation, selection state, and objective
H: target hardware capacities and service rates
```

A complete external library is a baseline, not a processing unit, unless its
local codelet can be isolated from its scheduler and measured under the same
contract.

## 2. Two Logical Dimensions, Four Unfolding Factors

Let `D` denote independent butterfly data groups and `S` denote the ordered
stage positions. The core logical iteration domain is the product `D x S`.
cuButterfly factorizes each dimension into a spatial part and a temporal part:

```text
D = Ud * Td
S = Us * Ts
```

`Ud` and `Us` are physical replication: they spend parallel hardware to expose
work at the same time. `Td` and `Ts` are physical reuse: they let the same
hardware serve multiple logical positions over time. These are architecture
factors, not fixed CUDA tile sizes.

```mermaid
flowchart LR
    subgraph Logical["Logical butterfly domain D x S"]
        D["Data dimension D<br/>independent groups"]
        S["Stage dimension S<br/>ordered dependencies"]
    end

    subgraph Factors["Two dimensions x two unfoldings"]
        Ud["Data-space Ud<br/>parallel groups"]
        Td["Data-time Td<br/>worker reuse"]
        Us["Stage-space Us<br/>parallel stage service"]
        Ts["Stage-time Ts<br/>resident stage reuse"]
    end

    subgraph Hardware["GPU realization selected per device"]
        Grid["grid / CTA / warp placement"]
        Resident["register + shared residence"]
        Transport["shuffle / shared / barrier transport"]
        Boundary["online reorder or global handoff"]
    end

    Unit["Replaceable processing unit<br/>radix, cuFFTDx, modular, FWHT, WMMA"]

    D --> Ud
    D --> Td
    S --> Us
    S --> Ts
    Ud --> Grid
    Td --> Resident
    Us --> Transport
    Ts --> Resident
    Grid --> Unit
    Resident --> Unit
    Transport --> Unit
    Unit --> Boundary
    Boundary -. "next stage group" .-> Resident
```

The four factors place different demands on the GPU:

| Factor | Architectural question | Typical GPU realization | Main limiting services |
|:--|:--|:--|:--|
| `Ud` | How many independent data groups execute now? | lanes, warps, CTAs, grid waves | thread slots, memory concurrency, transaction shape |
| `Td` | How many data groups reuse one physical worker? | per-thread items, CTA loops, repeated batch tiles | register lifetime, shared capacity, instruction count |
| `Us` | How much stage service is physically replicated? | lane roles, warp roles, local pipelines, grouped codelets | shuffles, barriers, shared ports, code size |
| `Ts` | How many dependent stages reuse that service? | register-resident or CTA-resident stage loops | state capacity, synchronization, coefficient supply |

Batch does not add a new dependency dimension. It contributes more independent
points to `D`. The implementation may still expose `Ub` and `Tb` to distinguish
batch-space and batch-time scheduling, but both refine the data-dimension
mapping rather than changing the two-dimensional graph model.

## 3. What Spatial And Temporal Mean On A GPU

Data-space unfolding is dependency-free: different transforms or independent
groups can be assigned to different lanes, warps, or CTAs without synchronizing
with one another. Stage-space unfolding is different. Adjacent stage work may
be placed in different physical roles, but dependency edges still require an
explicit transport and, when producer and consumer are not warp-synchronous, a
synchronization mechanism.

Temporal unfolding means reuse, not a promise that every value stays in a
register. Thread-private state can remain in registers; values exchanged by a
CTA normally reside in shared memory; state crossing CTA ownership requires a
cooperative launch mechanism or a global-memory boundary. `Td` and `Ts`
describe which logical work reuses a physical resource, while the residence
fields describe where its live state is held.

This distinction prevents two common but incorrect conclusions:

- a larger temporal factor is not automatically better, because longer live
  ranges can reduce occupancy or exceed shared-memory capacity;
- a single source-level CUDA function does not remove grid-wide dependency
  boundaries, because ordinary CTAs cannot safely synchronize with each other.

## 4. Residence, Transport, And Layout

The four factors alone cannot distinguish mappings with different handoff
costs. The expanded mapping descriptor is:

```text
M = (Us, Ts, Ud, Td, Ub, Tb, Hs, Rs, Rd, Rb, L, F, Q)
```

`Hs` records how spatial stage edges are transported. `Rs`, `Rd`, and `Rb`
record residence across stage, data, and batch reuse. `L` records input,
intermediate, bank, and output layouts. `F` selects a kernel family, while `Q`
contains its concrete threads, rows, radix, segment lengths, and processing-unit
choices.

For a long transform, stage groups are chosen independently of physical kernel
groups. Several logical groups may be fused into one CTA-resident execution
group when capacity and dependency scope permit. A global pass is introduced
only when the next group needs a different ownership domain or cannot fit in
the current resident state.

## 5. Online Reordering

After a resident stage group, the next group often needs a different view of
the same values. Writing the current result in the consumer's order performs
the permutation at an already-required handoff:

```mermaid
flowchart LR
    I["input tile"] --> P["resident prefix<br/>Ts stage groups"]
    P --> R["fused boundary store<br/>consumer-oriented layout"]
    R --> X["resident suffix<br/>new ownership view"]
    X --> O["requested output order"]
```

This is an online layout transformation, not a free permutation. It can remove
a later transpose and expose contiguous suffix work, but its address arithmetic,
coalescing, and shared-bank behavior must be measured. The V100 FP64 path is a
concrete example: recurrence reduced coefficient traffic, XOR swizzling reduced
bank conflicts, and strength-reduced address generation recovered the extra
integer and warp instructions introduced by the swizzle.

## 6. Replaceable Processing Units

A local unit is described independently from the mapping:

```text
P = (stage_group, arithmetic_core, coefficient_form, local_exchange)
```

The unit may use radix-2/4/8 composition, modular arithmetic, complex
four-multiply or Gauss-3, cuFFTDx, warp shuffles, shared memory, or a validated
external register hierarchy. Replacing it changes local latency, register use,
coefficient demand, and legal CTA shapes, so it changes the selected mapping.
It does not change the meanings of `Ud`, `Td`, `Us`, and `Ts`.

This boundary is why importing an established codelet strengthens rather than
weakens the architecture experiment: the same mapper can test whether a better
local unit remains efficient once embedded in a long-transform dataflow.

## 7. Hardware-Driven Selection

Selection proceeds from semantics to hardware rather than from a universal
tile constant:

1. Fix operator, precision or modulus, length, batch, direction,
   normalization, placement, strides, and output order.
2. Enumerate legal graph factorizations, four-factor mappings, residence and
   layout choices, and processing units.
3. Derive per-candidate register/shared/thread/warp CTA limits; reject only
   hardware-infeasible, layout-infeasible, or numerically invalid points.
4. Retain temporal state/thread, resident CTAs and warps, occupancy upper bound,
   grid waves, limiting resource, and adjacent-point resource-cliff features.
   A cliff expands the nearby decomposition/core search rather than pruning the
   resident point automatically.
5. Measure a reduced calibration set and dispatch among the remaining
   candidates.
6. Use NCU/NSYS to test the predicted limiting service and update the hardware
   profile.

Static constraints provide legality and a shortlist, but do not capture every
launch-limited crossover. On V100, a small measured calibration set reduces
the FFT selector's top-3 geometric-mean regret to `1.0007x`; this is why the
current method is counter-calibrated rather than presented as a purely analytic
cost model.

The full resource equations are in [Hardware Mapping
Methodology](hardware_mapping_methodology.md), and the complete separation of
graph, mapping, unit, layout, realization, and selector is specified in
[Complete Butterfly Design Space](butterfly_design_space.md).

## 8. Current Claim Boundary

The experimental NTT [Hybrid Dataflow backend](hybrid_dataflow_ntt.md) is the
first strict realization of fixed-size subgraph streaming across both stage and
data dimensions. It is separated from the older StagePipeline baseline so that
the architectural contribution can be tested against subkernel time reuse.

The artifact demonstrates on V100 that one mapping vocabulary survives changes
in operator and processing unit, that the best factorization changes with
length, batch, and precision, and that counter-guided changes predict measured
performance improvements. It does not yet establish cross-generation selector
transfer, non-power-of-two coverage, or universal superiority over specialized
libraries. Those remain explicit evidence boundaries rather than assumptions.
