# Design overview

cuButterfly studies how shared butterfly dependencies can support a common
execution method across arithmetic families, workloads and hardware. This page
connects the research argument to the engineering layers. The
[research principles](research_principles.md) define the contribution and
experiment boundaries; the [engineering map](development/repository-layout.md)
locates the implementation.

## Start from the organization problem

FFT, NTT, FWHT and structured butterfly computations expose staged pair
relations with different arithmetic labels. For power-of-two domains, pairing
bits describe this common dependency envelope. One-sided zeta transforms use
only a subset of its dependencies. Shared structure does not make coefficients,
finite-field operations, precision or inverse semantics interchangeable.

Specialized libraries already optimize both arithmetic and data movement.
A high-throughput core still needs suitable storage, data supply and exchange
under the target's constraints. Roofline gives a resource ceiling; actual
traffic, synchronization, residency and execution efficiency determine how
closely an organization approaches it. More independent work alone does not
resolve these choices.

The common research question is whether dependency structure can provide a
reusable way to characterize those choices and their costs. The benefit is
shared performance reasoning and organization construction, while each operator
retains its compatible core and each target can prefer a different realization.

## Derive regions, handoffs and readiness

A **stage band** is a consecutive set of dependency stages. If its pairing bits
are `Q`, fixing every index bit outside `Q` gives a closed region of `2^|Q|`
values. Its pair operations remain inside that region; its inputs and outputs
connect to other bands. A whole-transform band is legal when its core and
resource demands permit it.

At a boundary, producer and consumer regions intersect on the values they
exchange. These intersections determine both the data handoff and the required
predecessor set. A consumer starts only after its required inputs have been
published. Complete-region publication and finer-grained publication can expose
different schedules; grouping does not remove dependencies.

For a 16-point Hadamard example, write `i = 4r + c`. A band over bits 0 and 1
computes four row regions; a band over bits 2 and 3 computes four column
regions. Every column receives one value from each row. Each local region uses
four values, but the inter-band boundary covers sixteen. A four-value local
budget is therefore not a claim that all intermediate data fit in four words.

The [paper](../paper/main.pdf) develops this example and the correctness
conditions. [APPT layout analysis](appt_static_layout.md) records the earlier
NTT-specific formulation.

## Characterize space, time and resource demand

Both stage and data dimensions admit spatial expansion and temporal reuse.
The implementation vocabulary includes `Us/Ts` for stage space/time and
`Ud/Td` for data space/time. These factors describe a mapping at a specified
scope, not an operator-specific fixed tile size.

| Choice | Resource implications |
| --- | --- |
| Stage spatial expansion | Concurrent stage service, intermediate state, coefficient supply and transport |
| Stage temporal reuse | Repeated core service, retained state, feedback and synchronization |
| Data spatial expansion | Concurrent independent regions, storage ports and boundary bandwidth |
| Data temporal reuse | Successive tiles per compute group, buffer lifetimes and pipeline fill/drain |

The **resource characterization** describes live state, communication,
arithmetic work and ready work. A **service model** combines these demands
with compatible measured service rates to estimate completion time. They are
different layers; dependency counts alone do not predict all physical costs.

Finite capacity can require smaller local regions and exchanges through larger
storage scopes. Repeating the construction gives **hierarchical hybrid
dataflow**. It does not assume a fixed two-band partition or create extra
physical capacity. Unit spatial factors and single-band execution remain
candidates. Deeper residency and overlap are optional and may lose performance.

The detailed parameter definitions and resource constraints are in the
[design space](butterfly_design_space.md) and
[mapping methodology](hardware_mapping_methodology.md).

## Realize the method with tile subgraphs

A tile subgraph binds dependency-closed work to a core, value ownership, storage
and handoffs. A compute group can retain values across dependent operations and
process successive independent tiles within one kernel. Only the selected live
tiles and lookahead buffers occupy local storage.

Value residency, persistent workers and submission graphs are distinct. A CUDA
Graph reduces submission overhead but does not itself retain intermediate
values. Subgraph execution does not require a submission graph. Within-band
load/compute/store pipelining can also be useful when the next band waits for
the preceding band to complete; cross-band overlap is a separate choice.

### Treat boundary writes and reads together

Producer output placement must agree with consumer input interpretation.
**Successor-aligned writeback** incorporates the required placement into output
stores when ownership permits, potentially avoiding a separate permutation.
The total cost still includes producer stores, consumer reads and any staging.
Aligning reads alone can make writes more expensive.

Local bank mapping addresses a different cost. Concurrent accesses to distinct
addresses can serialize at a limited bank. A suitable bijective swizzle can
distribute a specified producer or consumer access group. The Hadamard example
uses `4c + (r XOR c)` to preserve the consumer region and distribute row/column
accesses across banks. This analytical example does not guarantee conflict
freedom for arbitrary vectors, value widths or simultaneous groups.

The [boundary control evidence](../paper/evidence/BOUNDARY_IO_CONTROL_20261109.md)
measures external layout and local shared staging separately. These are
independent experiments, not a cumulative speedup chain.

### Keep local arithmetic replaceable

A core declares arithmetic, support domain, value roles, local exchange,
resource requirements and numerical behavior. Compatible register codelets,
modular kernels or imported cores such as cuFFTDx can occupy the same
organization interface. An FFT band may require global-index-dependent twiddles;
a standalone smaller FFT is not automatically a compatible replacement.

A complete external library is a baseline unless its usable interface is
explicitly bound as a provider. Calling a whole-library FFT does not establish
that its internal arithmetic core was extracted. See
[core integration](core_integration_strategy.md) and
[processing-unit contracts](processing_unit_design_space.md).

## Connect the engineering layers

| Layer | Responsibility | Canonical guide |
| --- | --- | --- |
| Semantics and workload | Preserve arithmetic, coefficients, precision, direction, order, rank, shape, batch and layout | [Unified semantics](unified_semantics.md) |
| Organization and constraints | Construct legal regions, partitions, residency, handoffs and concurrency | [Design space](butterfly_design_space.md) |
| Cores and lowering | Bind compatible arithmetic and physical data paths; expose actual resources | [Core integration](core_integration_strategy.md), [xD lowering](xd_lowering.md) |
| Hardware and services | Supply static capacity facts, measured services and platform rules with provenance | [Hardware profiles](hardware_profile_install.md), [platform rules](platform_tuning.md) |
| Generation and selection | Enumerate, rank, compile, validate, optionally measure, cache and report the selected plan | [Public planner](unified_planner.md) |
| Evaluation | Compare matched complete calls, isolate organization effects and explain resource changes | [Benchmark contract](../benchmark/README.md), [reporting contract](reporting_contract.md) |

The static-profile policy reuses compatible local or bundled artifacts.
Missing performance coverage remains explicit. Installation and explicit tuning
can acquire target-local measurements; ordinary calls do not silently launch a
benchmark campaign. Compilation can happen during preparation under the selected
policy, outside warm execution timing. See the
[static-profile decision](adr/0001-static-profile-driven-planning.md).

## What the evidence establishes

A100 comparisons, fixed-core controls and profiles support useful organization
choices across the measured families. External-library competitiveness,
fixed-core attribution and application gains answer different questions. They
retain their own precision, workload, device and timing contracts, including
negative results. Follow the [current evidence route](README.md#read-the-a100-paper-evidence)
and [paper terminology](../paper/TERMINOLOGY.md) for the accepted claims.

Each hardware target has its own calibration and performance evidence.
[Ascend-FFT and Kunlunxin-FFT](platform_ports.md) reuse the method with
different kernels and interfaces. A common construction is not proof
that every lowering exists, that every mechanism helps, or that finite selection
finds a global optimum.
