# cuButterfly terminology

This is the canonical vocabulary for the paper, figures, captions, and
experiment reports. The terms describe different layers of the method and
should not be substituted for one another.

| Term | Meaning | Avoid using it for |
| --- | --- | --- |
| **Butterfly dependency graph** | The staged pair relation of an operator instance. Arithmetic, shape, direction, precision, and layout remain part of the request contract. | A particular kernel implementation |
| **Stage** | One pairing step in the dependency graph. Its pairing bit identifies the values that exchange data. | A CUDA launch or a full transform pass |
| **Stage band** | An ordered consecutive set of stages considered together. Band width controls the logical reuse region. | A physical memory level |
| **Dependency-closed region** | The values and operations reachable within a stage band while all other index bits are fixed. It is the logical unit derived from dependencies. | A worker block or a compiled kernel |
| **Tile subgraph** | The executable lowering of one or more dependency-closed regions, including its arithmetic core, ownership, storage, and handoffs. | Every arbitrary rectangular tile |
| **Resident tile subgraph** | A tile subgraph whose intermediate values remain at a selected storage scope during its local execution. | A persistent worker or a permanently resident kernel |
| **Handoff** | The producer-to-consumer transfer at a stage-band boundary, including readiness and value representation. | An unrelated input/output copy |
| **Boundary packet** | A dependency domain whose producer and consumer relations do not cross its fixed outside bits. | A physical buffer or a scheduling queue |
| **Successor-aligned writeback** | Writing a producer value in the representation expected by the next consumer, avoiding a separate permutation when legal. | A claim that every layout is coalesced |
| **Bank conflict** | Contention when concurrent accesses map to the same limited storage bank. | General external storage noncoalescing |
| **Swizzle** | An address remapping used to distribute a specified access group across storage banks. | A universal conflict-free layout |
| **Spatial expansion** | Allocating independent stages or data regions to concurrent execution resources. | Simply increasing batch size |
| **Temporal reuse** | Reusing execution resources and retained values across stages or data regions over time. | Cross-band overlap by itself |
| **Hybrid dataflow** | The joint choice of spatial expansion and temporal reuse along both stage and data dimensions. | A particular kernel or search heuristic |
| **Execution organization** | The assignment of operations, value ownership, storage, and handoffs to execution resources over space and time. | An arithmetic algorithm alone |
| **Compute group** | Cooperating execution resources assigned to execute tile subgraphs at a selected scope. | A stage band or a sequence of launches |
| **Arithmetic core** | A compatible local arithmetic implementation with explicit numerical semantics, value roles, and resource requirements. | The complete execution organization |
| **Hierarchy** | Repeating the same dependency, handoff, and resource construction across bounded storage scopes. | Assuming that a larger logical region creates more hardware capacity |
| **Performance characterization** | The demand record for a legal organization: live state, boundary traffic, arithmetic work, ready work, storage needs, and service assumptions. | The search algorithm or a measured latency by itself |
| **Operator contract** | Arithmetic semantics and observable numerical requirements: coefficients, direction, normalization, precision, and logical input/output order. | A storage layout or an implementation configuration |
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
