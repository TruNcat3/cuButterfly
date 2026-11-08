# Working-draft validation

## Evidence consolidation — 2026-11-08

Both manuscript entries compile to 14 pages, down from 15, with the original
IEEEtran font sizes and geometry. The evaluation source has 2,838 whitespace
words versus 3,323 before revision (including captions and LaTeX commands).
Two core contributions lead the argument; evaluation keeps setup, comparisons,
ablations, and analysis. Repeated application rows remain in their frozen table,
and the 18-workload organization cohort is shown once with its resource counters.

The separate FWHT table reports three-process native/supplied-output GM ranges
and the distinct public DEFAULT check. Each range was recomputed from the
frozen per-process ratios. No main66, cyclic, FP16, application or timing CSV
was modified, and no GPU experiment or benchmark queue was started.
The preparation-cost paragraph records the CPU optimization's actual limits.

The fixed-core figure's label formerly confused 72 FP32 timing rows with
36 paired ratios. Both the figure and metadata now distinguish those counts.
The FP64 panel shows 56 local-core timing points and summarizes seven cuFFT
rounds with a line/band. Re-rendering validates historical input hashes and
all original estimators; its three CSVs remain byte-identical. This is a
count-label correction, not additional measurements or changed speedups.

Both TeX logs have no undefined references/citations, missing characters, or
overfull boxes. Existing underfull-box and input-PDF-version advisories remain.
Rendered pages were reviewed at manuscript size, including the updated interface
table, fixed-core caption, resource analysis, related work, and references.
Repository checks and changed-document link checks passed. These checks establish
manuscript consistency, not new hardware validation or submission readiness.
See [the evidence record](evidence/PAPER_CONSOLIDATION_20261108.md).

## Public FP16 cohort update — 2026-11-06

The operator plot and table now show nine completed public DEFAULT FP16
requests separately from main66 and cyclic90. The paired geometric mean is
0.853583, with a range of 0.733471–1.066737 and one nominal win. MEASURE and
installed DEFAULT are separate evidence; no minimum across policies enters
the comparison. Further BF16 automatic comparisons are deferred by the user's scope decision. The loader verifies
parent completion, current-build calibration/installation receipts, all request
and output hashes, matched inputs and complete-call timing. Its eight CPU tests
pass, including refusal of incomplete evidence and selection among policies.

Both PDF variants compile to 14 pages. Updated operator comparisons on page 9
and analysis on pages 12–13 were rendered for visual review. No undefined
references/citations, missing characters or overfull boxes occur; existing
underfull-box and PDF-version advisories remain. Main66, cyclic NTT and
application headline denominators are unchanged. See the
[cohort evidence](evidence/FP16_PUBLIC_ACCEPTANCE_20261105.md).

## Unified precision comparison refinement — 2026-11-04

The standalone FFT-L mechanism float has been removed. FP16/BF16 now occupy
precision columns in the shared operator plot and rows in the comparison table.
Only the valid FP16 N8192 B32 public-winner replay is admitted (0.711864x against
cuFFT); further BF16 experiments are deferred; cuFFT Xt supports BF16 storage. The
22 fixed two-band controls remain separate evidence and do not enter these
figures, tables, or original headline macros. Single observations are points.
Boundary counters and the single-band endpoint remain in the common analysis.

The offline loader checks frozen CSV hashes, paired semantics, hardware,
prepared-input identity and timing medians. Changed CSV or preparation identity
is rejected. Counts, missing cells, figure hashes and unchanged headline macros
were checked. Both PDF variants compile to 15 pages, with no undefined
references/citations, missing characters or overfull boxes. The revised operator
figure and table on pages 9 and 10 were rendered and inspected without clipping
or overlap. This change runs no GPU experiments or implicit measurement refresh.
The earlier standalone figure and its renderer remain archived, outside the
manuscript. Precision/variant placement is recorded in the research principles.

## FFT-L evidence and argument update — 2026-11-04

The new `generated/fft_l_20261104/` export checks all 111 scale summary rows
against their raw result JSONs and all 78 boundary rows against recorded hashes
and medians. It retains 24 old single-band lowering gaps, the two excluded
stream-timing probes, all negative controls, and the full build/semantic identity.
No new benchmark ran during this writing update. The newest repair build's 212
clean stage samples remain incomplete calibration, not new comparison evidence.

The manuscript adds a compact two-panel figure from all six confirmed N4096
boundary cases and a separate matched A100 40GB NCU capture. Ordinary speedups
and store sectors/request use different panels and platforms; no mixed-unit
counter aggregation or profiler timing ratio is used. The renderer and frozen
inputs have a hash manifest. Existing main66 and cyclic statistics are unchanged.
Public-selection weak FFT/FWHT points and the low-precision cuFFT gap are explicit.

The introduction now distinguishes ideal intensity from actual resource ceilings.
The method separates dependency-derived demands, compiled resources, and measured
services; it includes the single-band endpoint and no longer says dependency
intersections alone determine conflict-free physical layouts. CLAIMS is synchronized
with the current cohorts while retaining historical identities. Application history
and duplicated motivation prose are compressed without removing negative evidence.

Both manuscript entries compile to **15 Letter pages including references**, the
same page count as the PDF present before this update. Bibliographies are nonempty;
there are no undefined citations/references, missing characters, or overfull boxes.
Existing underfull-box advisories remain. Pages 1, 10, 12, and 13 were rendered and
visually inspected, including the new single-column figure and weak-point discussion.
Labels, captions, and text are legible without clipping or overlap. This is a working
draft review, not certification against a current venue's submission rules.

## Comprehensive result refresh — 2026-11-02

The exporter now joins the completed cyclic NTT recovery with its baseline-only
supplement. It verifies report and row hashes, manifest identities, descriptors,
actual mappings, root-helper identity and timing contracts before accepting a
replacement. Only a missing external binding can be replaced. The 90 requests,
54 measured pairs and 36 arithmetic-domain exclusions remain a separate cohort;
the main 66-request study and its historical replays retain their denominators.
All 20 targeted evidence-export tests pass, including changed source identities,
invalid replacements, duplicate rows and stale timing guards.

Table II adds word32/word64 cyclic rows with geometric means 1.400x/1.265x.
The abstract and introduction use the generated combined cyclic value 1.308x,
with scope stated. Main FFT/FWHT and application figures were regenerated from
the verified frozen snapshot. The three cyclic losses remain in the text, and
the 138 default finite-field contracts are identified as execution coverage.
The historical QuaRot 1.8% loss is explicitly an eager diagnostic; failed Graph
validation produces no Graph performance claim. No benchmark or numerical
acceptance policy was changed in this manuscript update.

Both PDF variants build to 14 Letter pages. Pages 9, 10 and 12 were visually
inspected for the updated table, application figure, NTT discussion and profile
text; no clipping or overlapping labels were observed. The logs contain no
undefined citations/references, missing characters or overfull boxes. Existing
underfull-box and input-PDF version advisories remain. Source paths, statuses,
negative results and remaining gaps are summarized in
`evidence/CURRENT_EVIDENCE_20261102.md`.

## Contribution boundaries and terminology — 2026-10-29

The manuscript now separates the dependency/resource characterization, the
hierarchical hybrid dataflow organization policy, its tile-subgraph lowering,
and the engineering search used to instantiate a plan. Search is explicitly
not presented as a contribution. External library/application comparisons and
controlled organization experiments use separate subsection names and evidence
populations.

The CKKS native-loop application comparison and the matched stage-batch
integration control are now separated in presentation. The external application
figure contains 46 ratios; the other 36 matched-batch ratios remain in the
unchanged snapshot and are summarized in the controlled-experiment table.
The table distinguishes the external Loop reference from the Batch and
Faster-native integration controls. This is a change of evidence organization,
not a new measurement, a numerical exclusion, or a claim of fixed-core causality.
Negative application and mechanism results remain in the text.

The canonical vocabulary is maintained in `TERMINOLOGY.md`. In the manuscript,
a dependency-closed region is the logical unit derived from a stage band; a tile
subgraph is its executable lowering; a resident tile subgraph retains its
values at a storage scope; and a handoff includes producer/consumer readiness
and representation. This revision removes avoidable alternation between
component, resident region, and resident group in the prose without changing
the mathematical labels or measured controls.

Related Work now compares the algebraic-generation interface of SPIRAL/FFTX
and the workload/architecture mapping interface of Timeloop with the
butterfly-specific region, intersection, and resource-demand construction.
It does not claim that arithmetic/organization separation itself is new or
that prior systems cannot represent equivalent programs. Existing BibTeX keys
and the primary-source audit are retained.

Both PDF variants build to 14 pages. The latest logs contain no undefined
citations/references, missing characters, duplicate labels, or overfull boxes.
The revised application figure and pages 10 and 12 were visually inspected at
manuscript scale; the grouped CKKS table fits the column. Underfull boxes and
input-PDF version advisories remain. The figure manifest records the 46/36
presentation split and preserves the source-snapshot checksum.
All seven `tests/test_paper_current_refresh.py` checks pass.

## Discovery path and decision-chain revision — 2026-10-29

The introduction now states the historical discovery path explicitly: related
butterfly dependencies were optimized separately by domain; arithmetic-core
strength did not explain the observed changes across workloads and targets; a
Roofline/intensity check showed different rates at equal ideal intensity; and
the remaining variable was execution organization. This is a clarification of
the research argument, not a new contribution or measurement claim.

The motivation now connects that observation to the method and states that
mechanisms are choices, not universally beneficial features. A running 16-point
example in the method makes the decision chain explicit: dependency closure
defines legal regions, capacity removes infeasible mappings, service and ready
work select among feasible organizations, and producer/consumer intersections
bind the handoff layout. The arithmetic, workload, and target change the
inputs, while the construction remains shared.

Both PDF variants rebuild to 14 Letter pages. The changes introduce no new
experimental values and do not alter the frozen evaluation populations.

## Front-matter pacing and related-work matrix — 2026-10-29

This revision supersedes the figure placement described in the entries below.
Related Work is now a separate section after Evaluation and before Conclusion.
Background and Motivation proceeds directly from butterfly dependencies to the
resource/organization trade-off. Repeated explanations in the introduction and
motivation were condensed while retaining every measured example, the analytical
hardware/capacity cases, and their different evidence boundaries.

The compiled placement is now Figure 1 (Roofline) on page 1, Figure 2
(same-workload organization) on page 2, the method opening on page 3, and
Figures 3–4 (overview and worked boundary example) on page 4. The first-column
top-float suppression keeps the abstract first and places Figure 1 in the right
column. The complete-library plot now appears on page 9 with its discussion.
No font or margin reduction was used to achieve these placements.

The old related-work Table I is now Table IV on page 12. Its matrix separates
operators, scheduling unit, stage decomposition, independent-work partition,
and boundary placement/exchange. Entries are short mechanism names rather
than prose. Its font increased from scriptsize to footnotesize. NR denotes an
unresolved partition in this comparison, not an unsupported feature. The
source audit is maintained in `evidence/RELATED_WORK_DIMENSIONS_20261029.md`.

Both variants compile to 14 Letter pages including references. Final logs have
no undefined citations/references, missing characters, duplicate labels, or
overfull boxes. Underfull-box and input-PDF version advisories remain. Pages
were rasterized; the opening pages and final method, evaluation, related-work,
and conclusion placements were inspected. The numerical snapshot and all
figure measurements remain unchanged. This revision needed no GPU runs or
repeat numerical tests.

## Figure order, compact plots, and explicit mapping — 2026-10-29

The figure order now follows the argument: Figure 1 presents the Roofline gap,
Figure 2 compares organizations of one workload, and Figure 3 introduces the
common construction at the opening of the method. The overview is no longer
an introductory float. The remaining figures retain their numbers. Figure 5
now isolates a concrete eight-index bit-reversal correspondence; Figure 6's
canvas height drops from 366 to 274. Figure 7 adds a concrete producer/consumer
example and selected-plan bindings at the existing size.

Figure 9 is a single-column partitioned plot with a common ratio axis and all
82 previously plotted application pairs. Figure 12 is a single-column 2×3
grouped-bar plot retaining 54 configurations and all 162 timings. These edits
change presentation, not the frozen experimental population. Missing baselines,
provisional filters, numerical-policy limits, and negative application results
remain explicit. Captions are grammatical sentences; statistical details move
to the surrounding discussion.

Table I now compares actual operator families, scheduled units, and stage/data
organization, with primary-source evidence in
`evidence/RELATED_WORK_DIMENSIONS_20261029.md`. It does not equate kernels with
simple data parallelism or imply that kernel and subgraph abstractions exclude
each other.

Algorithm 1 gives the APPT bank/offset rule and its precise assumptions.
The CPU checker passed 381 parameter cases and 1,527,934 logical values,
including aligned contiguous and cross-stride groups. The report
`evidence/BOUNDARY_MAPPING_CHECK_20261029.md` distinguishes bank-major storage
from GPU interleaved bank addressing; these checks do not prove arbitrary
vector accesses conflict-free.

Both PDF variants build to 14 Letter pages. The final logs contain no undefined
references/citations, missing characters, duplicate labels, or overfull boxes.
The remaining advisories are underfull boxes and input-PDF version messages.
All pages were rasterized; pages 3, 4, 6–12 were visually inspected for table
widths, figure order, labels, algorithm placement, and single-column readability.
The seven current-refresh regression tests pass. No GPU measurements, queue
changes, or refresh of the live experimental snapshot occurred.

## Figure and boundary-layout revision — 2026-10-29

The method now states the missing implementation difficulty at a stage boundary:
successor-aligned writeback preserves value identity, while producer and
consumer access groups can still serialize on limited storage banks. The paper
uses **bank conflict** for the phenomenon and **swizzle** for an address
remapping that can avoid it. The illustrative XOR map is scoped to its stated
single-word access pattern; outer-storage layouts are evaluated by alignment,
transaction count, and coalescing. No universal conflict-free map or automatic
fallback is claimed.

The overview, semantic-transport, and system-architecture schematics were
redrawn on 1000 × 300, 480 × 274, and 1000 × 330 canvases. The architecture
example now shows two producer groups writing four successor pairs. The operator
box plot uses compact labels to remain legible at single-column height, and the
fixed-core figure is 3.45 × 1.85 inches. Both PDF builds remain 14 pages; the
latest build has no undefined references, missing characters, or overfull boxes.
Remaining underfull boxes and input-PDF-version advisories are layout warnings
already documented below.

## Parallel organization argument and core attribution — 2026-10-29

The motivation now concerns the gap between a resource roof and realized
execution, including its dependence on workload and target hardware. It no
longer identifies insufficient independent work as the general cause. Three
explicit analytical limiting cases distinguish batch scaling, cross-target
roof efficiency, and capacity refinement. A component-size/traffic example
connects the dependency construction to a concrete organization decision.

The existing two-core × two-organization FFT control leads the ablation section.
Its independently recomputed organization geometric means are 1.217015944 and
1.220723259. Stronger organization references test the size of residual gains,
not whether the fixed-core attribution already exists. All previously described
negative controls remain. Development milestones and pending collection details
move to `evidence/ORGANIZATION_ARGUMENT_20261028.md`; numerical policy and
measured-population limitations remain visible in the manuscript.

The editable Roofline is now 480 × 252 and contains seven measured A100 40GB M1
organization points at their derived ideal boundary intensity. The three FFT,
two FWHT, and two NTT points use medians from the frozen scaling CSV; trial
ranges and source provenance are retained in
`generated/roofline_cases_20261029.csv`. The blue roof remains symbolic and is
not fitted to the observations or presented as a hardware peak. Source, bundle,
and exported PDF/SVG/PNG hashes match the draw.io manifest. Existing tooling
attribution stays in the figure documentation, outside the paper's research
evidence.

The new FWHT scaling and NTT grouping examples were recomputed from the unchanged
M1 CSV. Their exact times, trial ranges, derived throughput, and evidence limits
are recorded in the revision evidence file. The analytical 16-/32-point component
and transfer counts were checked separately. No GPU execution, new performance
samples, threshold changes, or queue changes were performed.

Both PDF builds pass and remain **14 Letter pages** including references.
Reader copies equal their build outputs. Final logs contain no undefined
references/citations, missing characters, duplicate labels, or overfull boxes.
Existing underfull-box and input-PDF-version advisories remain. All main pages
were rendered; pages 1, 4, 6, and 11–13 were visually inspected for the argument,
Roofline, analytical example, reorganized ablations, scaling evidence, and
conclusion. No clipping or overlapping labels were found. New evidence links
resolve, and the manuscript contains no obsolete work-supply-limit label or
unresolved-reference marker. No submission-readiness claim is made.

## Current measured tables and figures — 2026-10-28

The current main comparison now reads the 66-request A100 80GB campaign,
not the earlier 36-request screen. Its snapshot preserves all twelve actual
operator/precision groups, 49 measured external pairs, missing comparisons,
actual mappings, build/device identities and 91 source-file hashes. NTT5 and
grouped-FFT2 remeasurements remain separate. The application snapshot contains
82 pairs: 72 CKKS control pairs for 36 requests, six Poisson pairs, two
provisional filters, and two bootstrap pairs. Neither tuning populations nor
selection/sample CSVs increase the workload count.

Three vector figures and two LaTeX tables are included by the manuscript:
the unified latency boxes, application ratios, same-core organization control,
matched operator summary and CKKS batch summary. PRO 6000 is explicitly
unmeasured. Missing external libraries never become wins; floating results
retain error-report-only status. CKKS includes both native controls and their
per-request faster alternative. Historical FNO, QuaRot, and recomputed-filter
losses remain described separately. No experimental result, GPU queue or
accuracy threshold was changed.

The exporter passes seven focused CPU tests, including failed-timing exclusion,
metadata/exactness gates, paired-ratio aggregation and cohort separation.
Snapshot and all nine figure output hashes were checked. Explicit
`make refresh-data` freezes live inputs before rendering; ordinary PDF builds
consume that snapshot. Generated operator and CKKS headline numbers also feed
the prose. Full-cohort additions still require a scope and prose review.

Both `make all evidence` builds pass and remain **14 Letter pages**, including
references. Reader copies match build outputs. Final logs have no unresolved
references/citations, missing glyphs, duplicate labels or overfull boxes.
Underfull-box and existing input-PDF-version advisories remain. Main pages
1 and 9–12 and evidence pages 10–11 were rendered and inspected at manuscript
scale. The twelve-group boxplot, both single-column tables, application panel
labels and the single-column ablation remain legible without clipping.
The three application panels explicitly label differing y limits; mechanism
whiskers are observed three-trial ranges, not confidence intervals.

Sources, regeneration commands and unresolved evidence are documented in
`generated/current_refresh/README.md` and `figures/README.md`.

## Hardware constraints, search scope and A100-first order — 2026-10-27, evening

The introduction and motivation now state explicitly that efficient cores remain
constrained by hardware memory access, caches, capacity and concurrency. The
contribution connects common dependencies and space--time organization to those
resource demands. Search is an implementation mechanism: the text distinguishes
static/registry planning, model-based installation, explicit neighborhood tuning,
and complete-plan xD selection. A ranking-policy competition is no longer required
evidence. Historical replay data remain archived; matched tuning, actual mappings,
independent confirmation and preparation costs remain relevant to organization
ablations. Experiment plans require completion of the defined finite A100 list
before PRO6000 migration, with internal intervention losses distinguished from
external library comparisons.

Both `make all evidence` targets pass and remain 14 pages including references.
Reader copies match their respective build outputs. Final logs have no undefined
references/citations, overfull boxes, missing characters or duplicate labels.
Extracted main text has no unresolved markers or V100 mentions. Existing underfull
box and input-PDF-version advisories remain. Pages 1, 4, 8 and 11 were rendered and
checked for the revised introduction, motivation, selection and evaluation text;
figures, captions and paragraphs are legible without clipping. No numerical result
or figure value was changed by this revision.

## Single-column Roofline without the duplicate ratio panel — 2026-10-27

Figure 3 is now a compact, single-column 480 × 184 schematic on page 4.
The former second panel tested whether local organization changes performance
under fixed scalar arithmetic and HBM stage boundaries. It did not directly
identify insufficient parallel work as the cause of that change and duplicated
the complete local-organization experiment. It is therefore retained as an
audit asset rather than included beside the symbolic roof.

The motivation retains the cross-arithmetic observation and points to the
full Figure 12 in the evaluation. All 18 requests, 54 configurations, and
162 timing rows remain there, including deeper-grouping regressions. The
derived-ratio CSV is unchanged (SHA-256
`55f90d495c54968a0df940e526e6d32c5ae2cc42e32c9df8c6c99854fb12b66e`).
No raw measurements or experiment processes were changed.

Both `make all evidence` targets pass and remain 14 pages including references.
Reader copies match their build outputs. Logs contain no undefined references
or citations, overfull boxes, missing characters, or duplicate labels; existing
underfull-box and PDF-version advisories remain. The final figure page in both
PDFs and the following method page were rendered and visually checked. Labels
and the shortened caption remain legible without clipping at column width.
The earlier two-panel validation below records the superseded layout.

## Roofline motivation and A100/PRO 6000 scope — 2026-10-27

Figure 3 now contrasts a symbolic bandwidth/core roof with an illustrative,
mapping-dependent work-supply limit. It identifies the unattained resource
opportunity and explains why batch can change available work at unchanged
ideal intensity. Neither the roof nor the dashed limit is a measured or
calibrated bound. The adjacent measured panel now covers FFT/FWHT FP64 and
word64 NTT on A100 40GB, replacing the earlier single-operator panel.

`tools/render_motivation_organization.py` validates the complete M1 collection
and all 18 original release hashes. All 162 trial values and 36 derived
stagewise/grouped ratios are retained in CSV/JSON. The six boxes contain six
workloads each, with quartiles and min/max describing workload spread rather
than trial uncertainty. Ratios use separate three-process medians; geometry
and jitter do not change their values. The six geometric means reproduce
the existing result report, including deeper-grouping losses. The original
scaling figure still shows ordinary timing ranges. A10040 is a diagnostic
cohort, distinct from A10080 library comparisons.

All V100 measurements and dependent claims were removed from the active
manuscript, including the indirect introduction/search references and Volta
bibliography entry. Active platform plans now name A10080 and the pending
RTX PRO 6000 Blackwell Server Edition 96GB comparison; A10040 remains separate.
Historical evidence, old generated figures, and dated validation records below
remain archived and are not current paper inputs or future test requirements.
No running experiment or raw measurement was changed.

Both `make -C paper all evidence` targets pass and remain 14 Letter pages.
Reader copies match build outputs. Recursive source references resolve;
extracted PDF text contains no V100, Volta, or unresolved-reference marker.
Final logs contain no undefined citations/references, overfull boxes, or
missing characters. Existing underfull-box and included-PDF-version advisories
remain, with PDF 1.7 output. The updated figure, surrounding motivation/method
pages, and evaluation platform scope were rendered and inspected. Figure
labels remain legible at their final half-page width, with no text clipping.
Draw.io source/export/bundle hashes, all new measured-panel artifact hashes,
and updated local Markdown links pass. Tool attribution remains in the figure
maintenance documentation; no tooling reference is presented as research evidence.

## Figure consolidation and prose compression — 2026-10-27

The two reader PDFs decrease from 16 to **14 Letter pages**, including references.
The manuscript has 12 figures instead of 14 and retains four evaluation
subsections. Template type sizes and margins are unchanged. Evaluation text
decreases from 3136 to 2519 words under `texcount -sum` (about 20%); shared
protocols move to setup while cohort-specific differences remain explicit.
This remains a working draft, not a claim of submission page-limit compliance.

Figure 2 now abstracts repeated butterfly components into stage-band strips
on a 480 × 328 single-column draw.io canvas. Its generator checks index
coverage, closure, 32 pair updates in every organization, local component
sizes/counts, and 128/64/32 analytical value transfers. These transfers are
not relabeled as measured HBM transactions. The merged 480 × 436 hierarchy
figure preserves row/column fan-in, capacity refinement, and an explicitly
independent four-bank XOR example; it does not compose the two address maps.

Two fixed-core cohorts share a 7 × 2.2 inch figure with separate panels and
7–8 pt labels. Its renderer checks all 12 FP32 points and paired trial ratios
against the original CSV and all nine FP64 cases and 63 timings against the
frozen snapshot/CSV. Sources, point/sample tables, and hashes are recorded in
`generated/core_organization_compact_metadata.json`. Negative outcomes and the
complete-cuFFT comparator remain. The Poisson plot is redrawn at 3.5 × 1.75
inches with 7–7.5 pt labels, keeping all 120 observations, quartiles, medians,
and min/max whiskers. Its sample CSV, summary CSV, and generated table are
byte-identical to their pre-layout versions. Both plots embed TrueType fonts.

`make -C paper all evidence` passes and reader copies match build outputs.
All active labels and citations resolve. Final logs have no overfull boxes,
missing characters, or unresolved references; existing underfull-box and
included-PDF-version advisories remain, with PDF 1.7 output. All manuscript
pages were rendered and reviewed, with a final pass over the resized Poisson
figure and subsequent evaluation pages. Figures, captions, and cross-references
remain legible without clipping. Editable draw.io sources, exports, bundle,
and provenance manifests are synchronized. No GPU task or measurement input
was changed by this revision.

## Hardware-constrained characterization and four-section evaluation — 2026-10-27

The abstract, introduction, motivation, method, execution, search, and conclusion
now connect dependency-derived space--time organization to resource demand and
target constraints. The method distinguishes stage/data spatial expansion and
temporal reuse, with a compact four-row demand table and an ideal fixed-compute
budget example. Retained feedback need not cross HBM; resource characterization
is distinct from measured service rates and timing prediction. Hardware adaptation
is scoped to characterized resources, compatible cores, and executable lowerings.

Evaluation has exactly four subsections: experimental setup, performance comparison,
ablation studies, and analysis of generated kernels. Applications remain within
performance comparison; component controls and core-by-organization crossings
remain in ablations. Existing internal profiles are explicitly distinguished from
the still-needed matched external-library profiles. The second platform is the
user-confirmed RTX PRO 6000 Blackwell Server Edition 96GB, pending measurement.
`EXPERIMENTS.md` and `experiments/hardware_characterization_closure_20261027.md`
specify bounded missing evidence and distinguish migration pilots from the formal
cross-platform comparison. No GPU experiment, running queue, or raw statistic was
changed by this paper revision.

`make -C paper all evidence` passes. Both reader copies match their build outputs
and remain 16 Letter pages including references. Active-source labels resolve
without duplicates; final logs contain no undefined references/citations, missing
characters, or overfull boxes. Extracted text has no unresolved reference markers.
Updated Markdown local links resolve. Existing underfull-box and included-PDF
version advisories remain; output PDFs are version 1.7.

Rendered pages were inspected for the revised introduction and related-work table,
the resource-demand table and method continuation, setup, and all reordered
evaluation pages. Operator comparison is on page 11, application figures on page
12, core controls on page 13, scaling/prefetch controls on page 14, and kernel
analysis on page 15. Labels and captions are legible with no clipping or overlap.
No template font or margin reduction was used. The 16-page length remains an
open submission-budget issue; this revision does not claim final page compliance.

## Completed bounded M1--M3 evidence — 2026-10-27, afternoon

The later revision adds the complete 54-cell M1 curve (162 independent process
trials), nine matched NCU/NSYS cells, M2 source reuse and exceptions, and the
16-cell M3 crossing (48 trials plus four matched NTT profiles). The source
ledgers and numerical limits are in `evidence/NARRATIVE_SCALING_RESULTS_20261027.md`,
`evidence/NARRATIVE_SCALING_PROFILE_20261027.md`,
`evidence/ORGANIZATION_REUSE_AUDIT_20261027.md`, and
`evidence/CORE_CROSSING_RESULTS_20261027.md`. These independent 40GB cohorts do
not change the frozen operator or application populations.

The M1 renderer validates every original release/hash before accepting the
complete collection, retains all trial values, and emits CSV plus a provenance
manifest. Six panels show point throughput versus batch with median/min/max
from three processes. These are neither FLOP/s comparisons across arithmetic
nor confidence intervals. Counter interpretations separate ordinary timing,
single cold profiler captures, source ownership, active warps, and unmeasured
shared transactions/spills. Negative M1/M3 outcomes remain in the text.

`make -C paper all evidence` passes for both 16-page Letter PDFs. The added
curve and revised discussion on pages 12--13 were rendered and visually
checked at final inclusion size; labels, captions and trial ranges remain
legible. No undefined reference/citation, missing character or overfull-box
diagnostic appears. Existing underfull boxes and draw.io PDF-version warnings
remain; the resulting PDFs are version 1.7. No font or margin reduction was
used. The page count exceeds the earlier 15-page draft and remains a writing
task, not a completed submission budget.

Supporting CPU checks passed for nested mapping/NTT input validation (51),
preparation (6), per-case queue/checkpoint handling (13), and compile-proxy
dispatch plus related ranking/resource contracts (45). These checks are not
full-repository or full-numerical acceptance. Full float distributions,
parameter holdouts, matched adaptation costs and adaptive-selection evidence
remain open. The original 66-request refresh was separately recovered and is
still active; its future results are not claimed by this manuscript build.

## Research narrative and historical motivation control — 2026-10-27

The abstract, introduction, motivation, section transitions, evaluation framing,
and conclusion now follow the agreed research question: whether shared butterfly
dependencies support reusable performance organization despite different local
arithmetic. The argument separates a Roofline resource opportunity from sustained
parallel work, motivates a common construction using fixed-core observations,
and distinguishes competitiveness from organization attribution and transfer.
It does not claim that prior libraries optimize only arithmetic, that organization
and core performance are independent, or that selection proves global optimality.

The added motivation panel reads all three rows of the first table in
`results/homogeneous_warp_dual_groups/analysis.md`. Its historical V100 NTT
dual/single throughput ratios are 0.999, 0.906, and 1.154 at B=1, 4, and 16.
They are preserved as reported, rather than recomputed from rounded latencies.
The caption identifies the historical platform, fixed factorization and register
continuation, and the lack of trial-level uncertainty in this summary. The CTA
organization changes launch and resource configuration; it is not an identical
whole-kernel control. The symbolic Roofline and measured batch curve occupy
separate panels. The three CSV ratios and all six source, script, and output
hashes in `generated/motivation_crossover_20261027.json` were verified.

Both `make -C paper all evidence` targets pass. Both reader copies match their
build outputs and remain 15 Letter pages, unchanged from the pre-revision PDFs.
Final logs have no undefined citation/reference, missing character, or overfull
box, and extracted text contains no unresolved reference markers. Underfull-box
advisories and the existing draw.io PDF-version warning remain. Rendered pages
were inspected, including the changed introduction and motivation, method
transitions, evaluation figures and tables, conclusion, and references. The
organization illustration now appears on page 4 beside its explanation; the
combined Roofline/crossover figure appears on page 5. Panel labels are aligned
and distinguish the analytical model from the historical measurement. Template
font sizes and margins are unchanged. This is still a working draft, not a
claim that the submission page budget has been met.

The frozen operator statistics and existing application figures retain their
measurement populations. No new GPU measurement was performed for this revision.
The M1–M3 controlled-measurement, implementation-reuse, and core-crossing work
in `NARRATIVE_EVIDENCE_PLAN.md` remains pending. This revision does not close
floating-point numerical acceptance or demonstrate adaptation-cost savings.
No kernel, calibration, or experiment queue was changed by the paper edits.

## Principle-aligned evaluation and evidence reuse — 2026-10-12

Evaluation now follows five questions: automatically selected library performance,
optional mechanisms and restricted re-search, organization benefits at fixed
arithmetic cores, selection/coverage/preparation costs, and complete applications.
The abstract, introduction and claim map use the same bounded evidence. Current
operator/application records and historical mechanism cohorts remain separate;
the wider RC2 results remain visible with their different populations. An inverted
A100 40GB core ratio was corrected from the original trial CSV, without changing
the measurements. Missing full cross-core experiments and restricted-search
interfaces remain explicit dependencies.

The three main experiment figures now use real data: unified absolute operator
latencies, fixed-core organization ratios, and independent-batch application
throughput with paired ratios. Trial ranges are not confidence intervals, and
workload boxes are not measurement uncertainty. No synthetic performance figure
is included in either current PDF. Floating error reporting is not relabeled as
numerical acceptance, and incomplete external comparisons are not wins.

Both `make -C paper all evidence` targets pass. Each reader copy matches its build
output, has 13 Letter pages, and has no undefined reference/citation, missing
character or overfull-box diagnostic. Extracted text contains no unresolved
reference marker. Underfull-box advisories and the existing draw.io PDF-version
warning remain. The affected pages were rendered and visually checked after
inclusion, including the three new plots, two experiment tables and transitions.
Neither template size nor margins were reduced.

The CPU evidence recomputation verifies all 170 source records. The figure
manifest's 23 source/output hashes also match. See
`evidence/EXPERIMENT_REUSE_20261012.json` and
`generated/principle_figure_metadata.json`. New finite GPU controls and their
preparation, mapping, numeric and timing gates are recorded separately in
`evidence/PRINCIPLE_GAP_RUN_20261012.md`; their completion is not inferred from
this paper build. No kernel or running application configuration was changed by
the manuscript revision. This remains an internal draft, not submission approval.

The finite control preparation/audit tests pass (11 tests). The repair audit
returns nonzero on rejected mappings. NTT controls pass exact full-batch checks;
FWHT internal benchmark sanity does not close the recorded full-error-report
dependency or the user's numerical acceptance.

## Native draw.io figure rebuild — 2026-10-11

The eight method schematics were rebuilt as native draw.io pages and exported
with draw.io Desktop v32.4.1. The sources contain only editable shapes, text,
and connectors; the export manifest records source/output hashes, cell counts,
and page dimensions. Every page passed XML parsing, unique-cell-ID and
no-embedded-image checks. The analytical construction checks remain separate
from rendering and use the existing CPU-only validation reports.

The exported PDF/SVG/PNG previews were inspected both at native size and after
inclusion in the IEEEtran manuscript. The double-column overview, organization
comparison, and system architecture remain readable at manuscript width; the
single-column roofline, hierarchy, semantic transport, bank, and pipeline
figures retain legible labels and complete captions. The overview footer and
Roofline subscripts were corrected after this inclusion check. The organization
figure was moved before its first explanatory paragraph so the figure, caption,
and argument are adjacent. No experiment inputs, GPU queues, or measured data
were changed.

Both `make -C paper all evidence` targets pass after the rebuild. The reader
copies are 13 Letter pages each, bibliography and cross-reference checks pass,
and the only remaining Tectonic diagnostics are underfull-box/layout advisories
and the expected PDF-version warning for the draw.io vector exports. `paper/main.pdf`
and `paper/main-evidence.pdf` are synchronized with their build outputs.

## Related work, figure layouts, and compression — 2026-10-09

Both build targets passed: `make -C paper all evidence`. The layout draft is
11 US Letter pages including references, down from the 14-page reader PDF
saved in `revisions/20261009_before/main.pdf`. The evidence-only PDF is 10
pages and excludes all three synthetic figure layouts. The IEEEtran class,
body size, and margins are unchanged. Section-source `wc -w` decreased from
8,978 to 5,499, excluding the newly generated related-work table in both
counts. The comparison is source tokens, not natural-language word count.

The final source has 33 unique labels, no missing cross-reference or citation
key, and 30 cited bibliography entries. Both logs have no undefined reference,
undefined citation, missing character, or overfull box. Underfull spacing
advisories remain. Both reader copies match their canonical PDFs byte for byte.
Extracted PDF text contains neither `[?]` nor `??`; the evidence PDF contains
no synthetic warning, and its excluded floats are guarded with the same flag
as their surrounding draft-only references. All layout-draft pages were
rendered and inspected, including the method figures, measured boxplot,
application table, synthetic panels, related-work table, and bibliography.

The three new chart layouts contain deterministic synthetic example data,
conspicuous watermarks, vector PDF/SVG, embedded fonts, separate source CSVs,
and provenance/schema metadata. They are neither observations nor forecasts.
Their values do not enter prose performance claims. Details and replacement
requirements are in `figures/DRAFT_LAYOUTS.md`.

Related work was grouped and checked against primary sources, with access
limits recorded in `literature/review_20261009/related_work_sources.md`.
The comparison table describes reported methods rather than unsupported
capability checkmarks. Existing software records retain access dates without
invented publication years. No bibliographic metadata was guessed.

The FNO/QuaRot application table is a separate historical-build snapshot.
Raw-file hashes and aggregation were checked, including the corrected QuaRot
short-workload median. `evidence/APPLICATION_RECORDS_20261009.md` documents
quality, fallback, identity, and unavailable application boundaries. Historical
raw results were not overwritten or pooled with the new semantic campaign.

The CPU-only theory checker passed its 3,586 bit-reversal edges, 24 floating
transforms, 20 exact NTT transforms, 12 zeta round trips, 96 local inverse
identities, and five bank examples. The construction checker passed 192
mappings, 480 closure/layout checks, 3,264 dense-reference comparisons, 1,364
mask pairs, 66,429 intersections, and 3,905 packets. Reports are in
`build/review_20261009/`. These checks do not establish CUDA coverage,
arbitrary multidimensional lowering, or performance optimality.

Only paper sources, paper build handling, CPU figure assets, and supporting
documentation changed. Running GPU campaign inputs were not modified.
See `REVISION_20261009.md` for the contribution wording and remaining evidence.

The records below describe their respective earlier manuscript revisions.

## RC2 evidence update — 2026-10-02

The manuscript now includes a separate latest-results table for the
`0.9.1-rc.2/replay-fix.1` A100 80GB snapshot. The machine-readable source is
`evidence/rc2_snapshot_20261002.json`; the accompanying interpretation is
`RC2_STATUS_20261002.md`. It records FFT36, weak FP64 FFT10, NTT completed
pairs with explicit unavailable/baseline-blocked rows, and FWHT36. The older
71/72 snapshot and its generated figures remain historical evidence for the
distribution and causal-control sections and are not numerically pooled with
RC2.

CPU theory and construction checks passed again. `make -C paper` succeeded
with no undefined citation or cross-reference, missing character, or overfull
box. The updated PDF is 14 US Letter pages including references. Underfull
spacing advisories remain, but they do not affect the new result table or
reference resolution. This documentation update changes manuscript sources and
evidence bookkeeping only; it does not alter GPU kernels or experiment inputs.

The primary operator comparison is now the unified absolute-throughput boxplot
`figures/rc2_unified_absolute_boxplot.pdf`, reporting effective butterfly-pairs/s
and transforms/s for FFT, NTT, and FWHT groups with every sampled cell overlaid.
The FFT-only boxplot and paired per-workload figure remain detailed artifacts,
while the former relative boxplot is historical audit material. The end-to-end
section records reproducible application contracts and timing boundaries; it
deliberately reports no application speedup until those integrations have been
executed on the pinned inputs.

## Current overview, architecture, and application revision

The paper now connects the organizing idea, automatic realization, operator
efficiency, controlled mechanism evidence, and planned application value.
`overview` replaces the main-text roles of `butterfly` and `hybrid_dataflow`.
`system_architecture` replaces `framework` and separates preparation from
execution. The worked N=16 construction, roofline, layout, and scheduling
figures remain. Earlier vector assets are retained for provenance.

E13 selects one open-source application each for FFT, FWHT, and NTT. The
evaluation includes a compact application-contract table in place of the
redundant status table. Application integration, quality validation, and actual
application timing remain pending. No kernel speedup is relabeled as an
application result. The experiment manifest and preparation instructions live
outside the existing GPU queues.

The revised PDF remains **14 US Letter pages including references**, with
eight figures, six tables, and seven displayed equations. The template,
body font, and margins are unchanged. The section-source count is 8,270
words including LaTeX commands and captions. The bibliography contains 32
cited entries, including the three application software sources, and the
active source has 31 unique labels. The source audit finds no missing
reference or citation key.

The final build has no undefined citation/reference, missing character, or
overfull box. Six underfull spacing advisories remain. The complete PDF was
rendered for layout inspection, and the overview, architecture, scaling, and
application pages were checked at readable resolution. The new dependency
diagram contains all pair edges and shows two four-value producers feeding
four two-value consumers, with value handoffs separate from local arithmetic.

A simultaneous editor pdfTeX build and Tectonic reader-copy update produced
an interleaved top-level PDF during validation. The canonical `build/main.pdf`
parsed correctly. Both supported copy paths now publish by atomic rename,
preventing the copy operation from interleaving with another writer on the
same inode. No editor configuration was changed. Prefer `build/main.pdf` for
the validated Tectonic output when an editor uses a different output recipe.

Preparing E13 exposed a stale check in `prepare_experiments.py` that rejected
the existing E11/E12 entries. The check now accepts these and E13 as optional
registered extensions while preserving required E1--E10 entries, duplicate
rejection, and independent execution paths. The six existing preparation
contract tests pass, including live-input preservation and identity checks.
This change does not launch any generated command or alter an acceptance job.

The E13 checker passed on the final manifest and wrote
`results/paper_preparation_20260921/end_to_end/readiness.json` relative to the
repository root. It reports zero contract errors, five unexecuted source or
reference recipes, and eleven integration/asset/validation blockers.
`application_ready` is false. The checker neither accessed the network nor
queried a GPU. Source inspection is a separate recorded research step. FNO
uses verified 64/128 data-source metadata for its planned main cases. The CKKS
demo remains smoke-only, and a secure, depth-complete main parameter set is
still unselected. Source selection is complete; application experiments are not.
The current paper therefore treats the end-to-end material as an executable
measurement contract rather than a result claim.

The frozen snapshot SHA-256 is unchanged at
`6ded5240df2cf909edb6119f49055ceac2c7dbd11b1c53bca4c340c01d9f57b9`.
Theory and construction checks from the preceding revision remain applicable
because their mathematical inputs were not changed. GPU experiments, build
identities, profiles, and timing records were not modified in this revision.

## Earlier narrative and evidence revision

The following checks describe the preceding 2026-09-21 revision, before the
new overview, architecture, and application section.

The current reader copy, `main.pdf`, and `build/main.pdf` are identical.
The revised draft has **13 US Letter pages including references**, compared
with the 16-page PDF at the start of this revision. IEEEtran, body font size,
and conference margins were not changed. The section-source word count
(`wc -w sections/*.tex`, including LaTeX commands and captions) decreased
from 10,076 to 7,959. Displayed equations decreased from 20 to 7.

The method now develops dependency-closed subgraphs, hierarchical hybrid
dataflow, and successor-aligned writeback through one 16-point Hadamard
example. Detailed bit-mask relations and local inverse matrices remain in
`METHOD_BLUEPRINT.md`; concrete hardware mappings and composition arithmetic
remain in their supporting documents. Narrative colons and semicolons were
removed from the section sources. Mathematical syntax, citation keys, and the
user-selected title are not prose punctuation violations.

The evaluation presents overall boxplot distributions, matched core and
organization controls, existing trace diagnostics, measured FFT scaling,
and selection/coverage evidence. The distribution panels keep external
contracts separate from in-tree controls and use logarithmic ratio axes.
The scaling facets share a single logarithmic latency color scale and include
27 contiguous out-of-place FFT cells. Two strided or in-place cells are
excluded only from that view, not from the overall matched comparison.
No missing result is filled or interpolated.

The author's UnionSparse arXiv v1 was downloaded and read as an evaluation
writing exemplar. Its boxplots, controlled ablation, and profiling presentation
inform the organization, not cuButterfly's performance claims. Its source,
hash, and reading scope are recorded in `literature/manifest.json`,
`literature/README.md`, and `REFERENCES.md`. A BibTeX record is available but
is not forced into the manuscript as butterfly-related work.

Checks completed for this revision:

- `make -C paper` succeeded. The final log contains no undefined citation or
  cross-reference, missing character, or overfull box. Five underfull spacing
  advisories remain. The bibliography has 29 cited entries. The only uncited
  BibTeX entry is the new writing exemplar.
- Active-source traversal found 32 unique labels and no duplicate label,
  unresolved reference, or missing citation key. Extracted PDF text contains
  neither `[?]` nor `??`. All 13 pages were rendered and visually reviewed,
  including the revised diagrams and experimental panels. Plot integration
  fixed a colorbar overlap and dense point annotations before the final build.
- The CPU-only experiment renderer completed in frozen mode with 71/72
  complete cells and 119 matched rows. The snapshot SHA-256 remains
  `6ded5240df2cf909edb6119f49055ceac2c7dbd11b1c53bca4c340c01d9f57b9`.
  Its four external aggregates remain 0.978930, 0.992782, 1.129174, and
  2.277925 for cuFFT, VkFFT, GPU-NTT, and the qualified Dao interface.
- The strict FFT four-cell protocol was checked against its archived
  `runtime_manifest.json` and `config.json`: 10,000 warmups, 100 repeats,
  three trials, 24 cells, and 72 accepted rows on A100 80GB. This differs
  from the main acceptance snapshot's 100 warmups and is stated separately.
- The existing CPU construction checker passed 192 mappings, 480 closure
  and layout checks, 3,264 dense-reference comparisons, 1,364 bit-set pairs,
  66,429 intersections, and 3,905 boundary packets. The theory checker passed
  3,586 bit-reversal edges, 24 floating cases, 20 exact NTT cases, 12 zeta
  round trips, 96 local inverse checks, and five bank-layout examples. The
  maximum small FFT absolute error remains approximately 9.64e-13.

This revision modifies paper sources, support documents, and CPU-generated
figures. It does not change kernels, calibration, running GPU jobs, or their
build identity. Resource-counter evidence, broader precision/cryptographic
timings, and same-version V100 results retain their recorded pending status.
The checks establish manuscript consistency, not completion of the research
campaign or readiness for submission.

## Historical validation through 2026-09-18

The remaining records describe earlier revisions. Their page counts, figure
counts, and references to a current draft refer to the named earlier snapshot.

## Public resident-promotion update — 2026-09-18

The public `local_stage_partitions` and `exchange_chunks` cohort is complete
on the pinned A100 40GB and 80GB targets: 26 candidates, three alternating
trials per candidate, and 78/78 accepted units per target. Identity,
correctness, warmup, exclusivity, and thermal checks pass. The cohort is a
controlled comparison against the old shared lowering; it contains no cuFFT
samples and is therefore not merged into the complete-library matrix below.
Its exact medians and hashes are in
`../results/paper_completion_20260918/public_resident_promotion/RESULTS.md`.

The strict core\(\times\)organization ablation is also complete on the
pinned A100 80GB: six FFT workloads, four matched cells per workload, 72
accepted rows, and no correctness failures. Native versus cuFFTDx Thread
cores are crossed with stagewise-HBM versus resident-prefix-2 organizations;
the machine-readable analysis is in
`../results/paper_completion_20260917/core_dataflow_ablation/REPORT.md`.
The same cohort is independently replicated on the A100 40GB with 72/72
accepted rows; its report is in
`../results/paper_completion_20260917/core_dataflow_ablation/a100-40gb/REPORT.md`.
Correction on 2026-10-12: the older 40GB summary inverted the core comparison.
Recomputed native/cuFFTDx geometric means are 1.0383 and 1.0430 for stagewise
and resident execution, respectively. The raw trials are unchanged. The
arithmetic leaf is fixed within each organization pair; launch geometry and
wrappers are part of the treatment, not identical compiled kernels.
The bounded E3 cross-arithmetic control adds 24 accepted FWHT/word64 NTT units
at $N=2^{12}$ and batch 1/1024, with results in
`../results/paper_completion_20260918/e3_cross_operator/RESULTS.md`. Neither
cohort is a hardware-migration result or an external-library comparison.

The method-figure revision also replaces the NTT-specific motivation artwork
with an aligned generic butterfly stage/tile diagram and redraws the roofline
with explicitly sub-roof schematic design points. The paper build remains
free of undefined citations/references; underfull spacing advisories are
retained as layout diagnostics.

## Figure, narrative, and reference revision

The current revision replaces the descriptive figure boxes with APPT-derived
vector artwork and seven reproducible analytical diagrams, refreshes the A100
evaluation from accepted records, and connects motivation, construction,
execution, selection, and evaluation through explicit transitions and the
N=16 example. Figure scope and inheritance are in `figures/README.md`.

The compressed output is 14 US Letter pages with eight figure environments:
analytical diagrams and measured-results panels. The Roofline now includes
seven measured A100 organization points under a symbolic resource roof. Final source
traversal finds no duplicate labels or unresolved cross-reference target. The
final TeX log has no undefined references, missing characters, or overfull
boxes. Underfull spacing advisories remain.

The user-visible question marks were reproduced from the older top-level
`main.blg`: system BibTeX could not find `IEEEtran.bst`, leaving `main.bbl`
empty. The unmodified style is now vendored. A separate temporary-directory
check using the same `/usr/bin/bibtex` succeeded with the local style and all
29 entries. The supported build completes bibliography/reference passes,
checks the final log and nonempty bibliography, and synchronizes
`build/main.pdf` with `main.pdf`.

The refreshed source has 29 cited bibliography keys, with no missing or
duplicate key. The compiled bibliography has 29 entries; PDF text contains
no `[?]` or `??` reference markers. The generated figures were reviewed as a
contact sheet; the aligned generic hybrid-dataflow, corrected roofline, and
pipeline/framework pages were rendered to inspect labels, captions, and layout.
The class SHA-256 still matches the supplied template. Body font size and
conference margins were preserved; only float spacing and figure/table
composition were tightened.

Current matrix tables use 71 complete A100 80GB cells and one unavailable
NTT output-order cell. The 119 matched comparison rows use medians of three
trials per implementation and geometric means of the resulting ratios.
External-library pairs and internal radix2 controls remain distinct. The
original historical evidence package below is retained rather than rewritten
as a current measurement. Plotting, bibliography checks, and PDF builds are
CPU-only and do not alter GPU queues, calibrations, or experiment records.

The result figures and their source medians were independently checked after
integration. All 108 matched factorial speedups equal baseline-median time
divided by treatment-median time. The final library aggregates reproduce
0.978930 (29 cuFFT cases), 0.992782 (18 VkFFT), 1.129174 (six GPU-NTT), and
2.277925 (eight Dao interface comparisons). Final-page inspection corrected
an overlapping axis label and ensured the FFT legend no longer obscures the
highest data point. The original historical table remains an artifact but is
no longer inserted ahead of the current matrix in the main text.

The 40GB clock-stability qualification references the independent repair
diagnostic at `../cuButterfly-ntt-order/results/ntt_output_order_fix/REPORT.md`
relative to the repository root (SHA256
`83e3b6aa283b2187cf7cf4361ce6e05e2e12c224531326cac1157558a470d0d7`).
That report records clocks ramping under short warmup and a longer-warmup
comparison. None of its new-build timings are merged into the paper's A100
matrix or controls.

## Earlier validation records retained by scope

The following paragraphs describe preceding draft revisions; page numbers,
placeholder counts, and 5/72 coverage refer to those earlier snapshots.

The draft compiles with `make -C paper` using user-cache Tectonic 0.17.0.
`build/main.pdf` has sixteen US Letter pages including references, with embedded
Times-compatible Nimbus Roman body fonts. IEEEtran's class file matches the
supplied template byte-for-byte. Its geometry and sizes are unchanged.

The 2026-09-17 revisions include literature, dependency construction, worked
cases, and the Roofline/scalability clarification. The latest revision was
rebuilt and its changed claims checked; prior evidence checks are retained.
Review covers:

- All ten section inputs, one constructed dependency/layout figure, five
  descriptive figure placeholders, twenty-nine bibliography entries, and every
  explicit cross-reference (43 unique labels, 15 referenced targets).
- All 29 entries are cited; no missing, duplicate, or unused BibTeX key. The
  bibliography contains 22 research papers/manuscripts (two labeled preprints)
  and seven software/documentation entries. Source-reading limits are recorded
  in `REFERENCES.md`; closest-work comparisons are in `RELATED_WORK.md`.
- No unresolved citations/references, missing glyphs, font substitutions, or
  overfull boxes in the final TeX log. Underfull paragraph/page warnings
  remain; no font or margin compression is used to suppress them.
- Visual inspection of the latest Roofline and scalability equations on
  pages 3 and 6. GPU mapping, scheduling recurrence, and the composition
  table were inspected on pages 8, 9, and 12 of the current build;
  the dependency proof/figure was also inspected in an intermediate build. Earlier
  motivation, layout, related-work and bibliography checks remain recorded
  from the preceding revisions.
- Six downloaded exemplar PDFs: verified source-manifest sizes, SHA-256 and
  page counts. Downloaded source versions and reading scope are in `literature/`.
- Seven historical-motivation source hashes checked against the separate
  `evidence/historical_motivation.json`; paired speedup calculations checked.
  These records are not merged into the frozen comprehensive table.
- New constructive CPU witness: all 24 stage-bit orders and eight ordered
  compositions at N=16, totaling 192 mappings, 480 band closure/layout checks,
  and 3,264 comparisons with an independent dense Hadamard definition.
  A negative control rejects releasing a whole consumer column after only
  one producer row. Results: `generated/construction_checks.json`.
- Boundary extension: all 1,364 bit-set pairs at N=2..32 were checked against
  explicit set intersections and a separately traversed incidence graph,
  covering 66,429 intersections and 3,905 connected packets.
- Two same-workload FWHT examples frozen in `evidence/method_cases.json` and
  replayed using the production CPU composition simulator. Resource estimates
  and max-with-floor predictions agree to relative tolerance 1e-12. This
  reproduces the model calculation, not the GPU observation or its accuracy.
- All 27 source-hash entries across the main, historical-motivation, and
  worked-case evidence manifests verified. Local links in the ten reviewed
  supporting documents resolved, including source line anchors.
- The example checker and composition replay work with the host Python
  without requiring a new environment.
- Latest clarification: APPT's Roofline motivation and the hardware-mapping
  methodology were consulted. The fixed-leaf examples L=16,17,24 with an
  eight-bit leaf limit were checked by arithmetic coverage, without allocating
  large transforms or asserting new GPU support. The FWHT first-group factors
  `(1,8,128,1)`, 128 threads and 128 CTAs were verified against the archived
  descriptor and its existing source hash. All 147 local links in the eight
  updated method/support documents resolved.

The unchanged evidence and mathematical checks below were completed on
2026-09-16 and are retained; this manuscript revision does not report them
as newly executed GPU or mathematical experiments:

- Twelve frozen paired rows, nineteen source hashes, separate protocol cohorts,
  the unstable A10040 FWHT marker, and the A10080 5/72 qualification boundary.
- CPU checks for 3,586 bit-reversal edge correspondences, 24 floating-transform
  checks, 20 exact NTT checks, 12 zeta round trips, 96 local inverse identities,
  and five single-word bank-layout examples. The largest small FFT absolute
  error against the direct reference is approximately 9.64e-13.

These are manuscript, provenance, and mathematical-example checks. In addition,
the A100 80GB E11 extension ran 46/46 correct CT points covering FP16/BF16
native and FP32 accumulation, FP32/FP64, forward/inverse floating transforms,
all three zeta directions, and word32/64 NTT directions. Its fixed-mapping
timing summary is separate from the three-trial acceptance evidence. The FFT
boxplot is regenerated from the frozen 29 cuFFT and 18 VkFFT paired rows and
its quartiles are stored in `generated/current_results_summary.json`. GS remains
reference-only, and E12 preparation-cost measurements plus cross-platform
replays remain pending as documented in `EXPERIMENTS.md`.

Reproduce mathematical and evidence checks using the commands in `README.md`.
Five figures remain explanatory draft boxes; the dependency/layout example
is now drawn using actual index tables. The PDF is an internal working draft, not a completed
experimental campaign or submission-ready manuscript.
## 2026-10-11 manuscript validation

- `make -C paper evidence` and `make -C paper` passed after the narrative and
  figure revision. Both PDFs have 13 letter pages; the evidence build excludes
  watermarked synthetic performance-layout figures.
- The final PDF was rendered and visually inspected on pages 2--7, covering
  the reorganized Background/Related Work/Motivation flow, the organization
  comparison, symbolic roofline, hierarchy witness, semantic transport, and
  bank-layout figures. No clipping, overlap, unresolved citation, or undefined
  cross-reference was found. The TeX engine reports only underfull box warnings.
- `organization_tradeoffs` and `semantic_transport` are analytic constructions,
  not measurements. Their exact assumptions and limitations are in
  `figures/README.md`; no experimental snapshot was changed.
- CPU construction and theory witnesses passed again after the text revision.
  They verify small mathematical examples only and do not establish CUDA
  lowering coverage, GPU transaction efficiency, or global optimality.
