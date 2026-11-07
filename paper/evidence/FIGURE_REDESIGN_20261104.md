# Figure redesign and evidence boundaries — 2026-11-04

Author: cuButterfly research workspace; figure implementation assisted by Codex.
Scope: paper figures and their captions; no kernel, calibration, GPU queue, or
benchmark population changes. This is a visual revision, not new performance.

## Sources reviewed

| Primary source | Inspected material | Adopted presentation technique |
| --- | --- | --- |
| [Roller, OSDI 2022](https://www.usenix.org/system/files/osdi22-zhu.pdf) | Local official PDF, access-pattern Figure 1 and figure descriptions 2–6 | Show tile boundaries and physical access patterns together instead of generic optimization boxes |
| [Welder, OSDI 2023](https://www.usenix.org/system/files/osdi23-shi.pdf) | Local official PDF, Figures 1–5 and accompanying explanation | Link operator dependency, storage hierarchy, and actual measured intervention; keep independent core and organization comparisons distinct |
| [FlashFFTConv, author v1](https://arxiv.org/pdf/2311.05908v1) | Local PDF, Figure 3 and Sections 3.1–3.2 | Show decomposition and storage/transpose changes with the same colored data elements |
| [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT) | Official README retrieved; prior method audit in RELATED_WORK_DIMENSIONS_20261029.md | Retain explicit stage decomposition and exact finite-field identity rather than implying arithmetic is interchangeable |
| APPT26 and UnionSparse | Existing local assets and literature manifest | Preserve the original graph/dataflow narrative and measured distributions; do not substitute their measurements for current A100 data |

The web tool failed with HTTP 404. Local primary PDFs were therefore inspected
and their metadata are in `paper/literature/manifest.json`. A direct GPU-NTT
ePrint PDF retrieval returned HTTP 403; this revision does **not** claim visual
inspection of that PDF. GPU-NTT's official README was reachable. Prior literature
claims keep their existing citation keys. No external figures were copied.

## Visual contract

- Overview: contract, workload, hardware -> dependency closure -> resource
  characterization -> bounded tile execution. Space/time applies to both stage
  and data dimensions. Factor one remains a legal choice.
- Organization example: same 32 Hadamard pair updates; only band partition
  changes. The counts are analytical, not GPU timings or guaranteed occupancy.
- Roofline: preserve seven frozen measured points and symbolic roof; add
  categorical transaction/synchronization/live-state explanations. No invented
  utilization bars and no inferred unique bottleneck.
- Handoff: preserve logical indices through writeback; distinguish the
  transpose witness from the separately verified XOR bank witness.
- Semantic transport: depict both directed dependencies in a pair and carry
  arithmetic/coefficient/direction labels. Relabeling is not arbitrary stage
  commutation or automatic equivalence of forward/inverse arithmetic.
- Lifetime: publication follows production; slot reuse follows the last read.
  Logical lanes do not assert that all implementations overlap these stages.
- Selected plan: static profile and confirmation remain outside warm execution;
  the concrete successor pair example replaces the repeated wide architecture.
- Measurements: preserve 66 main requests plus one public FP16 replay; BF16
  selected-plan coverage remains pending. Preserve 46 external application pairs,
  36 separate matched integration controls, and all six stage-control bars.

## Matched profiles and excluded captures

The main mechanism profile captures include preparation/teardown and have no
operator-only range aggregation. They cannot provide a truthful apples-to-apples
resource histogram without a separate extraction/measurement contract. The
stage-control figure therefore uses a labeled partition sketch above actual latency
ratios. Separately, the existing A100 40GB scaling figure adds a bottom row from
the audited M1 kernel captures: registers/thread, active-warps percentage, and
barrier-stall percentage. These 54 values retain 18 main kernel instances and
three operators, with source CSV hashes and ordinary-journal bindings. Spans
cover the two kernels, not trial uncertainty. All 162 original timing samples
remain in the upper six panels. No made-up numbers, synthetic expected gains, or missing PRO 6000 data
enter either manuscript build. Fine-grained design points remain in source
records; an unknown mapping is not assigned a representative template.

## Reproduction and review

- `python paper/tools/drawio_system_figures.py`
- `python paper/tools/drawio_organization_figures.py`
- `python paper/tools/drawio_example_figures.py`
- `python paper/tools/export_drawio_figures.py`
- `python paper/tools/render_current_refresh.py --figures-only`
- `make -C paper all evidence`

The Draw.io generators replace individual sources; export preserves direct
editor changes. Export manifests record source/output hashes. The measured
renderer binds the frozen snapshot and keeps pending and fixed-control evidence
separate. This revision retains the existing figure count and replaces duplicate
wide architecture content with a compact single-column realization.

## Validation outcome

- Seven included conceptual figures retain native editable Draw.io objects; all
  vertex bounds fit their declared canvases. The separate bank-layout audit
  source is retained. All eight source/export hash bindings were checked.
- The 25 current-refresh CPU tests pass; analytical generator assertions and
  Python compilation pass. The profile renderer checks the original manifest
  and CSV hashes and ordinary-timing identity before plotting.
- Main and evidence manuscripts both compile to 15 pages with 12 figures. No
  undefined citations/references, missing glyphs, or overfull boxes remain.
  Existing underfull layout and PDF-version notices remain nonfatal.
- Inspected changed exports and manuscript pages 1, 5, 8, 9, 10, 12, and 13 at
  rendered size. Revised close Roofline labels, cropped inset boxes, and the
  application timing-boundary language during visual review.
- No GPU measurement or background-queue state was changed.

## Scope expansion after visual review (2026-11-05)

The next visual pass expands the displayed evidence without adding synthetic
values. Figure 10 now exposes all 72 FP32 trial ratios and 63 FP64 Graph timing
rounds behind the fixed-core summaries. Figure 11 uses the audited
`motivation_organization_a100.csv` cohort directly: 18 workloads, 54 cells,
and 162 observations across FFT, FWHT, and NTT. Figure 9 remains a single
axis without a schematic inset and adds four separately shaded historical
FNO/QuaRot complete-call records. Historical records are not pooled with the
current snapshot because their runner and artifact identities are older.
