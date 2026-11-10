# Method figures and boundary I/O — 2026-11-09

The revision adds execution-relevant structure to the existing method figures.
It keeps the same research argument, numerical evidence, and figure count.
The central visual question is how one region's output becomes the next
region's correctly ordered and efficiently addressable input.

## Visual evidence chain

- **Overview:** the five abstract construction steps still accompany one
  16-point Hadamard example. The first step now contains the full four-stage
  dependency network and highlights closed producer regions and a consumer
  column. Resource bars count local values against the four-value worker
  budget. Separate worker slots and a local two-stage graph show what is
  retained and what is published.
- **Structural transport:** two complete eight-point, three-stage graphs show
  relabeling of every layer. A closed four-value prefix maps to a region with
  noncontiguous index labels. Ordered arithmetic labels remain attached;
  this is not arbitrary stage reordering or a free physical permutation.
- **Boundary I/O:** four producer cores write directly through the agreed
  address map, into a sixteen-word shared boundary. Consumer C1 selects its
  lane inputs from that boundary and runs its local core before writing the
  requested output indices. This expands one consumer; the others follow
  the same rule.
- **Prepared plan:** the existing generation diagram now uses the same N=16
  stage bands and C1 input roles instead of a separate N=8 binding.

## Boundary semantics

The running example uses logical role `i=4r+c`. Its shared-boundary address is
`a=4c+(r XOR c)`, with offset `c` and bank `r XOR c`. Producer colors track
value identity through writes and reads. For C1, consumer lane `r` reads
bank `r XOR 1` at offset one, giving local inputs in role order `{1,5,9,13}`.
The two local pairing bits correspond to global bits `{2,3}`. Final results
are written to output indices `{1,5,9,13}` under this example's natural-order
contract. The drawn routing wires represent address/ownership relations,
not a new hardware crossbar or an extra permutation kernel.

The lower panels intentionally distinguish storage scopes:

- With an illustrative four-word aligned external segment, C1 addresses
  `{1,5,9,13}` occupy four segments. Successor-major addresses `{4,5,6,7}`
  occupy one. Unstaged P0 stores change in the opposite direction, from
  `{0,1,2,3}` to `{0,4,8,12}`. Both sides, and any staging needed to regroup
  writers, belong to the boundary cost.
- In the shared-bank example, successor-major P0 writes all reach bank zero.
  Adding the XOR bank map distributes its four writes. This is one access
  group's bank distribution; it does not prove conflict freedom between
  simultaneously served groups or imply a measured external-traffic reduction.

The complete intermediate array is not claimed to fit in the four-value local
budget. Two local worker slots and the separate sixteen-word boundary have
different capacities and lifetimes. Larger problems may materialize a boundary
at an outer scope while retaining bounded local tiles.

## Full-manuscript review

The dependency → resource characterization → hierarchical organization →
subgraph execution → fixed-core and external evidence chain remains coherent.
Two small textual gaps were corrected: the service formula now defines R and
separates an execution group's service from band completion; application timing
now explicitly preserves cohort-specific Graph and eager protocols.

No new GPU experiment, measured speedup, hardware peak, or profiler result was
introduced. The examples are exact analytical constructions. Existing floating
error reporting, negative results, and the pending second-platform evaluation
retain their scope.

## Validation

The shared walkthrough checks closure, producer/consumer intersections,
bijective placement, bank coverage, readiness, and complete Hadamard output.
The transport figure checks the full directed edge sets, ordered pair mapping,
and closed-region correspondence. Independent review confirmed C1's address,
bank, local-core, and output ordering, and requested the explicit P0 label in
the single-group bank comparison. Figures use native editable draw.io objects.

The main and evidence PDFs compile successfully. The main paper remains 15
pages, including references. Rendered manuscript pages 4–8 were reviewed at
their final column widths; two clipped overview labels were shortened without
reducing font size. The complete boundary diagram appears on page 7. No
undefined references or overfull boxes were reported; existing underfull
typesetting warnings remain. Measured-data files were unchanged in this revision.
