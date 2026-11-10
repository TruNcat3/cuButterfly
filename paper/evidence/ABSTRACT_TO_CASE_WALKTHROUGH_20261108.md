# Abstract construction and connected example

This revision follows the research narrative from shared butterfly dependencies,
through execution organization and resource constraints, to bounded subgraph
execution. Every abstract step has a visual case using the **same** indices.
The example is analytical; it neither replaces experimental evidence nor
asserts that its mapping is the fastest on an A100 or any other GPU.

| Abstract step | Output of the rule | Connected 16-point Hadamard example |
| --- | --- | --- |
| Close regions | Local value domains for a stage band | With `i=4r+c`, stages `{0,1}` define rows `P_r`; `{2,3}` define columns `C_c` |
| Derive handoffs | Exchanged values and required predecessor regions | Every row/column intersection is one value; `C_1` requires `{1,5,9,13}` from all four producers |
| Match resources | Feasible spatial/temporal assignments and service demands | Two workers each reuse a four-value local slot over two stages and two regions; a separate 16-word boundary stores the handoff |
| Place values | A common producer/consumer physical representation | Successor offset `c`, bank `r XOR c`, address `4c+(r XOR c)` |
| Execute subgraphs | Production, publication, consumption, and reuse order | Worker 0 visits `P0,P2,C0,C2`; worker 1 visits `P1,P3,C1,C3`; all rows publish before any column starts |

## Resource and mapping assumptions

- Local capacity is per worker after auxiliary reservations. Two workers need
  two such slots. The boundary is a distinct storage scope, not part of either
  four-value slot. This small example fits the complete boundary in sixteen
  words; larger cases can use outer storage without increasing local residency.
- The resource example selects a feasible organization. Capacity does not
  establish that it beats another feasible choice; service rates are required.
- Stage reuse and consecutive tile traversal are temporal; two independent
  regions are assigned spatially. Neither cross-band overlap nor double
  buffering is required for the illustrated schedule.
- The bank example assumes four banks and one word per value/lane. Each row
  store group and column load group reaches every bank once. Two concurrently
  active groups may still compete for the same bank service. Vector widths,
  broadcasts, and external-memory transactions require their own analysis.
- Producer and consumer both use the composed address map. There is no
  separate transpose between them. For `C1`, addresses are `{5,4,7,6}` and
  banks `{1,0,3,2}`. Values retain logical identity throughout.
- Consumer outputs go to the requested final representation. The intermediate
  boundary can be reused after all reads complete. Figure widths show legal
  order rather than cycle counts or measured overlap.

## Evidence and validation

`paper/tools/walkthrough_example.py` checks region closure, singleton
intersections, bijective placement, all row/column bank groups, predecessor
readiness, and equality of the complete scheduled transform to a directly
constructed Hadamard matrix on an integer input. Both figure generators use
this witness. Independent review checked the same assumptions and identified
the need to distinguish per-group bank coverage from cross-worker contention.

The main overview replaces disconnected illustrative panels with these five
paired rules and examples. The boundary figure expands one step rather than
introducing a second incompatible layout. Existing measured plots and source
data remain unchanged. The Roofline schematic no longer labels distance from
its symbolic roof as a measured execution gap. Motivation points to the
fixed-core exchange/traffic evidence for its resource interpretation.

The evaluation retains its four subsections. Protocol details shortened in
the prose are preserved in [evaluation protocol notes](EVALUATION_PROTOCOL_NOTES_20261108.md).
The external comparisons, controlled organization evidence, and applications
retain their separate populations and limitations. The related-work table
compares decomposition and organization at a common conceptual level rather
than contrasting a logical tile subgraph with a physical CUDA kernel.

## Final checks

- Main and evidence PDFs compile to 15 pages, with resolved citations and
  cross-references and no overfull boxes. Existing underfull-spacing warnings
  remain non-blocking.
- Figure 3 now accompanies the method opening on page 4. The boundary figure
  and single-column related-work table were inspected at manuscript scale.
- The three changed draw.io sources match the PDF/SVG/PNG export manifest;
  their native cells match the combined editable bundle.
- Existing measured plot inputs and statistics are unchanged. Added numerical
  ranges come from the existing numerical-scope audit, not new runs.
