# Architecture Guardrails

This is the local design memo for cuButterfly/cuNTT. It is intentionally
prescriptive. Before adding a kernel, a selector rule, or a benchmark claim,
review the invariants in this document. The purpose is to keep an optimization
from silently replacing the architecture-level contribution with a vendor
library call or with an unrelated one-off kernel.

## 1. The Claim

The contribution is a reusable way to organize regular butterfly computation
on a GPU. The arithmetic cell is replaceable; the contribution is the mapping
and execution of the dependency graph:

```text
mathematical graph
  -> two-dimensional space/time decomposition
  -> uniform tiles and dependency-closed subgraphs
  -> physical ownership and residence
  -> packet/role dataflow and boundary layout
  -> replaceable butterfly processing unit
  -> device-specific generated kernel
```

The implementation may use a radix-2/radix-4 cell, a register cell, WMMA, a
cuFFTDx codelet, or another measured cell. It must not confuse the cell with
the graph schedule. cuFFT is an opaque vendor plan and is therefore an
external comparison baseline, not a processing unit in the mixed-dataflow
implementation.

## 2. The Logical Graph And Four Unfoldings

For a regular power-of-two transform, let `D` be independent data groups and
`S` be the ordered stage positions. The graph domain is `D x S`. Both
dimensions are unfolded in space and time:

```text
D = Ud * Td
S = Us * Ts
```

`Ud` and `Us` expose independent work on lanes, warps, CTAs, or resident
subgraphs. `Td` and `Ts` reuse those workers over data groups and dependent
stage groups. They are architecture parameters, not fixed CUDA tile constants.
Batch is additional independent data in `D`; it is not a fifth dependency
dimension.

The smallest reasoning unit is a uniform subgraph tile:

```text
a = (Us, Ts, Ud, Td, layout, residence, core)
```

The tile must describe both what is computed and where its live state resides.
Changing the equivalent butterfly cell is allowed, but changing `a` changes
the architecture mapping and must be represented in the search space.

### 2.1 Mapping Responsibilities

| Component | Meaning | GPU resources affected |
|:--|:--|:--|
| `Us` | stage-space replication | warp roles, shuffles, shared ports, synchronization |
| `Ts` | stage-time reuse | register/shared live range, coefficient lifetime, barriers |
| `Ud` | data-space replication | lanes, warps, CTAs, memory transaction concurrency |
| `Td` | data-time reuse | per-thread items, CTA loops, registers, instruction count |
| tile shape | uniform subgraph extent | launch shape and tail fill |
| residence | register, shared, or global state | occupancy and boundary traffic |
| layout | logical-to-physical address mapping | coalescing, bank conflicts, reorder cost |
| processing unit | arithmetic implementation | instructions, registers, code size, throughput |

Data-space work is dependency-free between independent transforms or groups.
Stage-space work is only independent after its producer values have been
transported. A mapping must therefore state the transport and synchronization
for every stage-space edge; it is not enough to say that the work is in one
CUDA function.

## 3. What Mixed Dataflow Means

`HybridDataflow` is the complete execution organization, not merely a packet
format. It has six responsibilities:

1. **Uniform subgraph formation.** Partition both dimensions into repeated
   `a` tiles, with explicit tails and a stable ownership rule.
2. **Data packing and unpacking.** Build the packet representation required by
   the selected local cell and preserve the logical layout contract.
3. **Role and stage expansion.** Replicate stage roles (`Us`) and reuse them
   over stage time (`Ts`), while replicating or reusing data workers (`Ud/Td`).
4. **Physical subgraph residence.** Keep one or more dependency-closed tiles
   resident in registers/shared memory, and keep multiple resident blocks or
   CTAs in flight when the hardware permits. This is the key v0.7 insight.
5. **Cross-subgraph scheduling.** Publish and consume ready packets or tiles,
   select static or dynamic ownership, and overlap producer/consumer work
   without violating dependencies.
6. **Boundary and coefficient handling.** Fuse online reorder, twiddle/root
   generation, and final output layout whenever the next owner needs that
   permutation anyway.

The physical processing unit executes the arithmetic for one tile inside this
organization. It does not own the global decomposition, readiness protocol,
or boundary policy.

### 3.1 Three Residence Levels

The term "resident subgraph" has three distinct meanings and they must not be
collapsed:

```text
cell-resident      values for one local butterfly cell remain in registers
block-resident     one or more uniform subgraphs remain in CTA shared state
grid-resident      a CTA/warp pool remains active while it consumes ready work
```

v0.7 primarily exposed block/grid-resident subgraphs and homogeneous producer
and consumer roles. v0.8 separates that ownership axis from the logical stage
partition and from the arithmetic core. A design can use a cuFFTDx cell and
still be block-resident only if the codelet is actually embedded in the
resident execution; calling a standalone cuFFTDx kernel is not that claim.

## 4. Ownership, Pipeline, And Synchronization

Stage partition and physical ownership are separate descriptors:

```text
logical partition:   S -> [stage group 0, stage group 1, ...]
execution ownership: [grid tiled | transform resident | resident queue]
local core:          [radix2 | radix4 | register | cuFFTDx | ...]
```

The same logical partition must be testable with different owners and local
cores. In particular, v0.6 Hybrid2D remains a valid compatibility candidate in
v0.8; a new version may not win by deleting the old physical point from the
candidate set.

For each edge, record:

```text
producer -> storage/layout -> readiness signal -> consumer
```

The signal may be warp-synchronous, a named barrier, a block flag, a cooperative
grid protocol, or a global-memory handoff. Ordinary independent CTAs cannot
perform an implicit grid barrier. Data-space replication does not require a
barrier between independent groups; stage-space transfer does require a valid
transport unless the values are kept in the same synchronized scope.

The desired steady state is a wave of homogeneous resident subgraphs:

```text
producer(a0) -> producer(a1) -> producer(a2) -> ...
       |             |             |
       v             v             v
consumer(a0) -> consumer(a1) -> consumer(a2) -> ...
```

The implementation must distinguish startup, steady-state overlap, and tail
drain. A graph or a loop that merely executes producer and consumer sections
serially is not evidence of a pipeline.

## 5. Layout, Reorder, And Twiddle Rules

Every stage-group boundary has an explicit layout contract. A boundary is
allowed to materialize when ownership changes or the state no longer fits, but
the store should use the consumer's required order whenever possible. Online
reorder is part of the output operation, not an unrelated cleanup kernel.

Twiddle/root handling is another replaceable service dimension:

```text
table | recurrence | fused coefficient reuse | per-lane generated root
```

Measure its register footprint, instruction cost, and memory traffic separately
from the butterfly arithmetic. Do not claim that a fused core is superior if
it only moved the same traffic into an unmeasured packing kernel.

## 6. Processing-Unit Integration Boundary

The preferred FFT layering is:

```text
cuButterfly dataflow scheduler
  -> uniform local tile / packet
  -> cuFFTDx or another device-callable local codelet
  -> cuButterfly reorder, handoff, and next tile
```

cuFFTDx is suitable because its execute operation can be compiled into a CUDA
kernel and measured with our tile contract. cuFFT is not suitable for this
layer because its plan owns the complete transform, launch schedule, storage,
and synchronization. It cannot expose our `Us/Ts/Ud/Td` or resident packet
edges.

Current status:

| Path | Local core | Dataflow ownership | Status |
|:--|:--|:--|:--|
| `TemporalTile` | cuFFTDx block/direct | cuButterfly tile launch | implemented |
| `OnlineReorder` | cuFFTDx block/resident | cuButterfly boundary and segment composition | implemented |
| generic `HybridDataflow` | native radix-2/radix-4 | resident packet/role kernel | implemented |
| bounded `HybridDataflow` + cuFFTDx block | cuFFTDx inside one block-resident local tile | our hybrid config owns the tile contract; linear local layout only; no inter-role overlap yet | experimental adapter implemented |
| subgroup role codelet probe | cuFFTDx `Size=8`, `EPT=8`, `FFTsPerBlock=32` | one warp services 32 independent register packets | physical-unit probe implemented; no stage handoff yet |
| multi-role `HybridDataflow` + cuFFTDx execute | generated `cufftdx-role-pipeline` (`PacketSize=4|8|16`) | two resident warp roles with shared double-buffered packet handoff; complete transform is `PacketSize^2` | template family implemented; 16/64/256-point envelopes pass V100 correctness, with 8-point packet retained as the measured legacy alias |
| `CuFft` | opaque cuFFT plan | vendor-owned | explicit/external baseline only |

The last row must never appear in the automatic architecture selector. The
multi-role row is the work required to complete the strongest version of the
claim; it should be implemented as a generated codelet adapter, not as a
special case that silently launches cuFFT. The bounded adapter is deliberately
smaller: it proves that a local cuFFTDx block can be owned by the HybridDataflow
configuration and remain block-resident, but it does not claim role overlap.

The bounded adapter can be exercised explicitly on a cuFFTDx-enabled build:

```bash
./build-cuda118-cufftdx2/cubutterfly_bench --operator fft \
  --backend hybrid-dataflow --fft-core cufftdx-block --logN 8 --batch 64 \
  --flow-tile-log 8 --stage-space 8 --data-space 4 --data-time 1 \
  --role-stages 1 --target-ctas-per-sm 1 --pipeline-buffers 1 \
  --dataflow-layout linear --dataflow-state inplace \
       --verify --csv
```

For a reproducible envelope comparison, use
`scripts/benchmark_hybrid_cufftdx_adapter.sh`. It compares four distinct
execution organizations under the same FP32, out-of-place, unnormalized FFT
contract:

```bash
BIN=$PWD/build-cuda118-cufftdx2/cubutterfly_bench \
LOG_NS="8 9 10" BATCHES="1 4 16 64" TRIALS=3 \
OUTPUT_DIR=$PWD/results/hybrid_cufftdx_adapter \
./scripts/benchmark_hybrid_cufftdx_adapter.sh
```

The smallest role-local cuFFTDx unit is also available as a standalone probe:

```bash
./build-cuda118-cufftdx2/cufftdx_role_codelet_bench \
  --batch 1024 --warmup 10 --repeat 100
```

To collect a small batch envelope in one file:

```bash
BIN=$PWD/build-cuda118-cufftdx2/cufftdx_role_codelet_bench \
BATCHES="1 8 32 128 1024 4096" \
./scripts/benchmark_cufftdx_role_codelet.sh
```

The complete two-role resident adapter is available as a generated family:

```bash
./build-cuda118-cufftdx2/cufftdx_role_pipeline64_bench \
  --batch 1024 --warmup 10 --repeat 100
```

It performs one kernel launch, keeps the intermediate `PacketSize`-point packets in a
shared double buffer, and uses a ready flag per buffer slot. The producer warp
and consumer warp can therefore work on adjacent transform groups without a
global scratch handoff. `PacketSize` is a compile-time processing-unit
parameter; the complete transform has `N=PacketSize^2` points. The current V100
family contains packet sizes 4, 8, and 16 (`N=16,64,256`). The legacy
`cufftdx-role-pipeline64` name selects the 8-point packet. This remains a
generated envelope rather than a claim that every legal point is optimal.

The batch sweep used for the current artifact is reproducible with:

```bash
BIN=$PWD/build-cuda118-cufftdx2/cufftdx_role_pipeline64_bench \
BATCHES="1 4 16 64 256 1024" \
PACKET_SIZES="4 8 16" \
./scripts/benchmark_cufftdx_role_pipeline.sh
```

The standalone codelet probe uses `block_dim=(1,32,1)`: one warp services 32
independent `PacketSize`-point packets, with one local transform per packet.
That probe
performs only packet load, local FFT, and packet store. The separate
`cufftdx-role-pipeline64` path below is the one that publishes ready tokens,
consumes another role's output, and demonstrates producer/consumer overlap.

The generated `raw.csv` retains the selected implementation and any rejected
configuration message. `summary.csv` reports median and minimum kernel time;
it now also retains the physical execution contract emitted by
`ButterflyPlan::execution_realization()`. In particular:

| Realization | Meaning |
|:--|:--|
| `materialized-pipeline` | logical groups are separate launches with global boundaries |
| `resident-single-role` | one CTA-local core remains resident, but role overlap is not implemented |
| `resident-multi-role` | native stage roles and packet channels execute inside one resident launch |
| `vendor-opaque` | the vendor library owns internal launches and storage |

Only the `resident-multi-role` row exercises the current complete role pipeline.
For the generated reference point, `stage_space=8`, `role_stages=2`, and
`data_time=4` mean four resident stage roles traverse four data tokens per
packet. The bounded cuFFTDx adapter remains a valuable core-lowering experiment,
but its `resident-single-role` label prevents it from being mistaken for the
complete architecture.
`analysis.md` reports `reference_time / adapter_time`, so values above `1.0x`
mean that the bounded adapter is faster. A failed row is not a zero or an
implicit fallback. The `cufft` row is an external comparison only and is never
eligible for runtime selection.

The first V100 pilot (`results/hybrid_cufftdx_adapter/`, FP32,
out-of-place, three trials, `logN=8..10`, batch `1..64`) verified every row.
The bounded adapter and standalone TemporalTile cuFFTDx agree within the
measurement noise because both currently launch the same block codelet. The
adapter is about `1.05x..1.19x` faster than cuFFT for `logN=9..10`, while
`logN=8` remains about `4%..13%` slower than cuFFT. This is a feasibility check,
not evidence that role overlap has been implemented: the next adapter must
make packet roles consume distinct resident tiles and expose overlap in a trace
or an independently measured service term.

### Current Processing-Unit Integration Boundary

The first naive multi-role composition that placed two cuFFTDx block `execute`
operations into one shared tile, while treating the tile as an arbitrary
packet workspace, compiled but failed with an illegal memory access on the
V100. The failure was a codelet contract violation, not evidence that the
mixed-dataflow schedule is invalid: cuFFTDx owns a generated block layout and
its `workspace_type`/shared-memory partition. The explicit
`cufftdx-role-pipeline` adapter resolves that boundary for a bounded case by
using the generated role-local `Size=PacketSize/EPT=PacketSize/FFTsPerBlock=64` shape, a separate
application-owned packet channel, and a verified two-warp ownership protocol.
It is now callable through `ButterflyPlan` and reports `resident-multi-role`.
The remaining work is to generate this same contract for arbitrary partitions
and hardware envelopes; the bounded single-role adapter and the
native radix-2/radix-4 pipeline remain separate candidates.

This command is an explicit experimental point. It is not in automatic
selection until a matched sweep proves that the resident adapter improves the
complete mapping rather than only the local FFT instruction count.

## 7. Evolution Of The Work

The versions describe increasing separation of concerns, not merely faster
kernel names:

```text
v0.1-v0.5  establish regular butterfly kernels and CUDA API surface
v0.6       mature hierarchical/tiled baselines and verified operator coverage
v0.7       homogeneous resident subgraphs, role pipelines, static ownership,
           and the observation that block residence can amortize core reloads
v0.8       separate logical unfolding, physical ownership, local core, and
           performance-model selection; retain v0.6 as a compatibility point
next       embed efficient local cells (cuFFTDx/TurboFFT/register variants)
           into the resident mixed-dataflow packet pipeline
```

The v0.7 lesson is not "always use a resident queue". It is that a repeated,
homogeneous block-level subgraph can keep its state and arithmetic service
resident across a wave of data tiles. Whether it wins depends on shared
capacity, register pressure, wave count, launch overhead, and the local core.
The performance model must expose these variables instead of baking one
three-segment configuration into the selector.

## 8. Required Search Dimensions

The generator and selector must be able to represent, at minimum:

```text
logical:       stage partition, decomposition count, Us, Ts, Ud, Td
ownership:     grid-tiled, transform-resident, resident-queue
residence:     register/shared/global, ping-pong/in-place, pipeline buffers
pipeline:      role count, stage handoff, ready window, token interleave
layout:        linear, swizzled, online reorder, natural/bit-reversed output
core:          radix-2, radix-4, radix-8, register, WMMA, cuFFTDx, TurboFFT
numeric:       operator, precision, word width, modulus, twiddle/reduction mode
hardware:      SM count, shared limit, registers, warp size, service rates
```

If a parameter affects resource demand or dependency transport but is absent
from the candidate descriptor, the search is incomplete. If a candidate cannot
be lowered to a reproducible explicit configuration, it is not a deployable
candidate yet.

The CUDA implementation now exposes the same boundary in the kernel source:
`HybridRoleCodelet<Operator>` owns only `apply()` and `finalize()`, while
`hybrid_dataflow_kernel` owns state placement, packet transport, readiness, and
stage traversal. This is the intended insertion point for a role-local
processing unit. A raw CTA-collective cuFFTDx launcher does not satisfy this
interface; it needs a separate subgroup codelet lowering before it can become
part of a multi-role graph.

## 9. Pre-Change Review Checklist

Every implementation or optimization should answer these questions before
being accepted:

### Semantics

- Does forward/inverse, placement, stride, precision, and output order remain
  exact?
- Is numerical error measured against the same formula and tolerance, with any
  difference attributed to accumulation order rather than a changed transform?

### Architecture

- Which of `Us/Ts/Ud/Td` changed?
- What is the uniform subgraph `a` and its tail rule?
- Is the work cell separate from ownership, residence, and boundary policy?
- Is block/grid subgraph residence real, or only a sequence of child kernels?

### Hardware

- Where do live values, coefficients, flags, and packets reside?
- What are the register/shared-memory/occupancy costs and CTA wave count?
- Which synchronization is required for each stage-space edge?
- Are independent data-space groups free of unnecessary synchronization?

### Performance

- Are startup, steady-state overlap, and tail drain measured separately?
- Are global transactions, reorder traffic, and twiddle traffic counted?
- Is the candidate compared with the identical v0.6 physical point and with
  the external library under matched semantics?
- Does the model explain why a candidate wins, rather than fitting one table?

### Baseline Boundary

- Is cuFFT used only for comparison or an explicitly named compatibility
  fallback?
- Does automatic selection ever resolve `ButterflyBackend::CuFft`? It must not.
- If cuFFTDx is used, is it a local codelet inside our mapping rather than a
  replacement for the mapping?

### Evidence And Documentation

- Is the explicit resolved configuration emitted in CSV/diagnostics?
- Is the NCU command and reduction path recorded?
- Are unsupported lengths/devices rejected rather than silently remapped?
- Are this memo, the relevant design document, and the experiment table
  updated in the same change?

## 10. Immediate Next Milestones

1. Extend the bounded adapter into a generated `HybridDataflow` codelet
   interface with an explicit packet load/store contract for multiple roles.
   The first prerequisite is now implemented by
   `query_butterfly_lowering()`: it rejects CTA-collective cores before code
   generation and reports whether a selected core is actually role-local,
   multi-role, and data-time capable. The standalone subgroup probe and the
   64-point two-role resident adapter now validate the packet and stage-handoff
   mechanics. The generic planner still keeps this lowering experimental until
   arbitrary partitions and device envelopes are generated.
2. Validate one small resident point against the native radix-4 cell and the
   standalone cuFFTDx temporal tile, including shared bytes, registers, and
   sector counts.
3. Add block-resident wave depth and subgraph multiplicity to the performance
   model; do not reduce the model to a fixed `10+10` recipe.
4. Search local core choices around the same architecture point, then compare
   the complete mapping against v0.6 and cuFFT as separate rows.
5. Only after the adapter has a measured overlap benefit promote it from
   experimental to automatic selection.

This order preserves the research claim while allowing the best proven
processing-unit experience to improve the arithmetic layer.
