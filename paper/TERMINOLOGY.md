# cuButterfly terminology

This is the canonical vocabulary for the paper, figures, captions, and
experiment reports. The terms describe different layers of the method and
should not be substituted for one another.

| Term | Meaning | Avoid using it for |
| --- | --- | --- |
| **Butterfly dependency graph** | The staged pair relation of an operator instance. Arithmetic, shape, direction, precision, and layout remain part of the request contract. | A particular kernel implementation |
| **Dependency envelope** | The pair-dependency structure supplied by the ordered stages. One-sided operators may use only a subset of these dependencies. | A claim that every operator uses identical dependencies or arithmetic |
| **Stage** | One pairing step in the dependency graph. Its pairing bit identifies the values that exchange data. | A CUDA launch or a full transform pass |
| **Stage band** | An ordered consecutive set of stages considered together. Band width controls the logical reuse region. | A physical memory level |
| **Dependency-closed region** | The values and operations reachable within a stage band while all other index bits are fixed. It is the logical unit derived from dependencies. | A worker block or a compiled kernel |
| **Tile subgraph** | The executable lowering of one or more dependency-closed regions, including its arithmetic core, ownership, storage, and handoffs. | Every arbitrary rectangular tile |
| **Resident tile subgraph** | A tile subgraph whose intermediate values remain at a selected storage scope during its local execution. | A persistent worker or a permanently resident kernel |
| **Subgraph execution** | Executing a tile's dependent operations with explicit local value lifetimes and boundary publication. A compute group may process several tiles in one kernel. | Merely capturing kernel launches in a submission graph, or claiming all transform intermediates reside locally |
| **Handoff** | The producer-to-consumer transfer at a stage-band boundary, including readiness and value representation. | An unrelated input/output copy |
| **Boundary packet** | A dependency domain whose producer and consumer relations do not cross its fixed outside bits. | A physical buffer or a scheduling queue |
| **Traversal group** | Independent transforms visited by the same scheduling group. In the NTT endpoint study, `data_time` controls this grouping. | A boundary packet, a larger dependency-closed region, or proof of simultaneous value residency |
| **Successor-aligned writeback** | Writing a producer value in the representation expected by the next consumer, avoiding a separate permutation when legal. | A claim that every layout is coalesced |
| **Bank conflict** | Contention when concurrent accesses map to the same limited storage bank. | General external storage noncoalescing |
| **Swizzle** | An address remapping used to distribute a specified access group across storage banks. | A universal conflict-free layout |
| **Spatial expansion** | Allocating independent stages or data regions to concurrent execution resources. | Simply increasing batch size |
| **Temporal reuse** | Reusing execution resources and retained values across stages or data regions over time. | Cross-band overlap by itself |
| **Hybrid dataflow** | The joint choice of spatial expansion and temporal reuse along both stage and data dimensions. | A particular kernel or search heuristic |
| **Execution organization** | The assignment of operations, value ownership, storage, and handoffs to execution resources over space and time. | An arithmetic algorithm alone |
| **Compute group** | Cooperating execution resources assigned to execute tile subgraphs at a selected scope. | A stage band or a sequence of launches |
| **Arithmetic core** | A compatible local arithmetic implementation with explicit numerical semantics, value roles, and resource requirements. | The complete execution organization |
| **Leaf / codelet** | A local arithmetic building block used by a core. In fixed-leaf experiments, identify the retained implementation and the organization fields allowed to change. | A fixed thread geometry, launch structure, or storage arrangement |
| **Hierarchy** | Repeating the same dependency, handoff, and resource construction across bounded storage scopes. | Assuming that a larger logical region creates more hardware capacity |
| **Performance characterization** | The demand record for a legal organization: live state, boundary traffic, arithmetic work, ready work, storage needs, and service assumptions. The introduction calls its dependency-derived construction **dependency-driven resource characterization**; this is the same demand description, not a second model. | The search algorithm or a measured latency by itself |
| **Operator contract** | Arithmetic semantics and observable numerical requirements: coefficients, direction, normalization, precision, and logical input/output order. | A storage layout or an implementation configuration |
| **Service model / cost estimate** | Composition of organization demands with compatible core and storage services to estimate completion time, including startup, repeated work, tails, and boundary costs. A cost estimate is the resulting estimate for a realization. | A synonym for dependency-driven resource characterization or a guarantee of measured performance |
| **Workload domain** | The instantiated sizes, batch, placement, strides, and storage layout for one request. | Arithmetic semantics or a hardware capability |
| **Hardware capability** | Target capacities, storage services, execution services, compatible cores, and measured rates. | A workload-specific plan |
| **Selected plan** | One checked and measured organization/core/layout realization for a contract and target. | The whole finite design space |
| **Search/selection** | Engineering that enumerates or explores legal plans and confirms a selected plan. It is not a paper contribution. | The common dependency characterization |

The paper uses **external storage** and **compute group** as generic terms. It
mentions HBM, CUDA blocks, shared memory, or a named vendor only when the
hardware or baseline is itself the subject of the statement. Experimental
labels such as `stagewise`, `resident`, and `writer-aligned` are retained only
when they name a controlled intervention or a measured plan.

The evidence vocabulary is also fixed. **External comparisons** use a named
library or application baseline. **Controlled organization experiments** hold
the arithmetic core or workload fixed to isolate an organization change.
Neither category is silently presented as the other.

For two disjoint stage bands covering all index bits, each transform is one
boundary packet. A traversal group may visit several such packets without
creating dependencies between them. Logical regions, physical compute groups,
and launches are separate counts. The legacy `logical_regions=2` field in the
November 7 endpoint evidence counts stage bands, not closed regions; frozen
records retain that field for provenance, while manuscript counts use the
dependency formula. See the [worked evidence chain](evidence/DECISION_WALKTHROUGH_20261108.md).

The method overview pairs each abstract construction step with the same
analytical Hadamard example. Its four-value local budget is **per worker**;
the separate sixteen-word boundary is another storage scope. The composed
map `4c + (r XOR c)` retains successor offset `c` and distributes each row or
column access group across four banks. Per-group bank coverage does not imply
conflict freedom between simultaneously active groups. A timeline of legal
execution order does not specify durations or measured overlap.

**External layout** specifies value placement across a materialized stage-band
boundary. **Local bank mapping** specifies placement for accesses at a banked
storage scope, including shared staging inside a band. Both use producer and
consumer access requirements, but they are independent choices. The November 9
external-layout controls vary global cross-band placement while retaining the
local shared layout; the prefix shared-staging control varies only its local
bank permutation while retaining the global boundary. These controls use
different cores and executables and do not form one cumulative ablation.

**Logical plan descriptors** record the planner's organization. Descriptor
fields such as `execution_groups` or generic `grid_ctas` do not establish a
physical launch configuration. Use NCU's observed grid and block dimensions
(`actual_launch` in the November 9 evidence) for physical launch claims.
A kernel launch, a launched compute group, a logical region, and a tile-service
step are distinct counts; any correspondence must be established by the lowering.
In the resource-decision table, `G=2/7/1` counts ordered stage bands from
`compiled_and_launch_resources`, not the per-band `grid_ctas`. In the FP64
factor control, `t` counts consecutive tiles per compute group; each tile
contains four 256-point local transforms. Neither count denotes a single
local transform.
