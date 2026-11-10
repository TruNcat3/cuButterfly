# Figure sources and scope

These are editable vector drafts for the cuButterfly manuscript. PDF files
are included by LaTeX; SVGs support inspection and editing. Schematics and
qualitative curves are explicitly distinguished from measured results.

## Reading-order revision

See [the experiment reading map](../evidence/EVALUATION_READING_MAP_20261104.md).
Figure 2 now includes the full sixteen-point pairing graph before the three
partitions; Figure 6 separates cross-band readiness from within-band tile supply.
Figure 10 names stage-retention and tile-supply interventions and spells out
its group/prefetch coordinates. It is 3.45 × 2.50 inches; numerical inputs are
unchanged. Platform identities are recorded in setup and manifests rather than
repeated inside plots. The four evaluation sections are setup, performance,
ablations, and analysis; Figure 12 belongs to analysis.

## Current visual revision, 2026-11-04

The method overview now carries the common dependency/resource/execution story.
The selected-plan diagram has become a single-column worked realization. Native
Draw.io sources add actual directed pair connections, hierarchical band boxes,
bank-contention slots, and publication/buffer-release events. See
[the current source sizes and evidence rules](drawio/README.md) and
[the reference-driven design review](../evidence/FIGURE_REDESIGN_20261104.md).
Dimensions and layout descriptions dated earlier below are historical.

The measured operator plot has separate family and precision label rows.
Application and fixed-core plots add qualitative insets, preserving every
measured point and timing range. Insets do not encode duration or resource use.
Run `python paper/tools/render_current_refresh.py --figures-only` to redraw from
the frozen snapshot without refreshing a running experiment or rewriting tables.
Current measured figure sizes are 7 × 2.55, 3.45 × 2.85, and 3.45 × 2.05 inches.
The mixed-scope NCU captures remain outside the plots. The independent A100
40GB scaling figure now adds three profile panels from its audited 18 main
kernels, with register use, active warps, and barrier stalls. Its 3.45 × 4.0 inch
canvas retains all 162 timing observations. Profile spans describe two kernels,
not trial uncertainty. No synthetic counters or unverified aggregates appear.

## Inherited APPT artwork

The author's APPT26 source is `../APPT26/samplepaper.tex` relative to the
repository root. It is an internal manuscript, not a new published citation.
Original vector assets are copied unchanged into this directory so the paper
can build without the sibling APPT26 checkout:

| Asset | Original relative to APPT26 | SHA-256 |
| --- | --- | --- |
| `appt/two_NTT.drawio.pdf` | `figure/two_NTT.drawio.pdf` | `04e49d1783b2a213aabfdd1309bd80dc8455697ca026e3104d2ac5c3ab3daf5b` |
| `appt/hybrid_NTT.pdf` | `figure/hybrid_NTT.pdf` | `07774b91f0c5224bd277edb739a23934c5412d984c8af76b8014ad3e9bdc7056` |

The original files remain available for provenance and side-by-side checking.
They are not the generic cuButterfly figure: their BU labels are conceptual
pair-update resources, and their NTT ordering labels are inherited. Arithmetic
contracts remain operator-specific.

## Adapted and new analytical figures

The draw.io assets included by the manuscript use native draw.io sources in
[`drawio/`](drawio/README.md). Edit their named pages in
[`cubutterfly-methods.drawio`](drawio/cubutterfly-methods.drawio) or an individual
`.drawio` file, then export with `tools/export_drawio_figures.py`. LaTeX includes
`drawio/exports/*.pdf`. The earlier Matplotlib redraws remain for provenance;
rerunning their renderer does not replace the manuscript's draw.io exports.

To reproduce the earlier Matplotlib drafts only, run
`python paper/tools/render_method_figures.py` from the repository root.
The script needs NumPy and Matplotlib, already present in the shared conda
environment. It neither accesses a GPU nor reads performance measurements.

| Figure | Relationship to APPT | Meaning |
| --- | --- | --- |
| `overview` | Extend the stage/data organization to labeled butterfly families | Main figure connects dependency construction, closed subgraphs and explicit exchange, and physical hierarchy |
| `system_architecture` | New system architecture | Preparation produces a checked, selected plan; runtime follows its groups, buffers, and dependencies |
| `butterfly` | Earlier redraw of the butterfly-network vocabulary | Retained source asset, superseded in the manuscript by `overview` |
| `roofline` | Adapt APPT's roofline motivation | Symbolic bandwidth/core roof with seven measured A100 organization points at their derived ideal intensity; matched FFT, FWHT, and NTT cases expose different execution gaps |
| `organization_tradeoffs` | APPT same-workload stage/data comparison, generalized to butterfly bands | Single-column stage-group strips for the same 16-point FWHT; exact local-value, component, and analytical transfer counts without repeated network elements |
| `hybrid_dataflow` | Earlier redraw of APPT's aligned stage/data progression | Retained source asset, its main-text role is merged into `overview` |
| `hierarchy` | Extend partial-stage/data grouping and bank/offset reasoning | Single-column N=16 row/column handoff, capacity-limited 8+8+8 strip, and a separately labeled four-bank XOR example; the two address maps are not composed |
| `bank_layout` | Adapt `data_arrangement.pdf` and bank/offset reasoning | Retained standalone source; its exact four-bank example is now included in `hierarchy`, not a separate manuscript figure |
| `pipeline` | Extend APPT's block progression and UnionSparse event lanes | A/B/C readiness barriers and buffer lifetime witness; not a measured trace or guaranteed GPU overlap |
| `semantic_transport` | Explicit butterfly order correspondence | Eight-index bit reversal maps adjacent to strided pairs while preserving the attached labels; not arbitrary stage commutation |
| `framework` | Earlier GPU generation workflow | Retained source asset, superseded by `system_architecture` |

The analytical figures use a shared palette and consistent labels. The roofline
uses a symbolic resource ceiling and measured execution points. The points are
derived from `../generated/roofline_cases_20261029.csv`, whose medians and trial
ranges are frozen from the A100 40GB M1 scaling evidence. The symbolic roof is
not fitted to those observations or presented as a hardware peak.
`organization_tradeoffs` follows the Roofline gap with a same-workload
organization comparison. `semantic_transport` makes the structural
correspondence and arithmetic contract visible. `hybrid_dataflow` is the generic
replacement candidate for the inherited NTT-only two-panel artwork and keeps
both panels geometrically aligned. Their numerical
indices are analytical examples, not placeholders for measured performance.

The overview follows those two motivation figures at the opening of the method,
where it becomes Figure 3. It explains the common organization, and the
architecture figure explains its automatic realization. Neither is a performance measurement. The
N=16 `hierarchy` figure remains the worked construction example. The paper
uses the analytical draw.io assets plus the measured cohorts listed below,
without adding empty application bars. Pending application evidence retains
its explicit status. Retained earlier redraws are not included merely to
increase the figure count.

The later 2026-10-29 pacing revision places Roofline on the first page and the
same-workload organization example on the second. The method starts on page 3,
with its overview and boundary example on page 4. Related work and its compact
feature matrix follow the evaluation, so the earlier pages can develop the
problem and construction without a detour through the literature survey.

## Experimental figures

### Current refresh, 2026-10-28

The current operator and application figures are now
`current_refresh_operators`, `current_refresh_applications`, and
`current_refresh_organization`, generated from the explicit snapshot in
`../generated/current_refresh/snapshot.json`. The first shows all 66 requests
in twelve actual operator/precision groups. Boxes retain available workloads;
paired ratios use only the intersection with each competitor. Missing external
baselines are explicit. The application figure retains all 36 CKKS requests
against each native control, the three Poisson endpoints, two provisional
fixed-filter endpoints, and two small-parameter bootstrap endpoints.

The single-column organization figure fixes the radix-2 core and changes only
stage grouping. Whiskers show the full range of three paired trial ratios;
these are internal controls, not external-library speedups. The application
figure is a 3.45 × 2.65 inch single-column plot with Poisson/filter, CKKS, and
bootstrap partitions sharing a linear speedup axis and a parity line. All 82
pairs remain. Workload points and geometric
means are not confidence intervals. Source/output hashes, data units,
aggregation rules, and descriptions are recorded in the snapshot and
`../generated/current_refresh/figure_manifest.json`.

The complementary `cpu_normalized_performance` figure uses the same main66
semantic requests and divides each backend's complete-call time by an
operator-specific CPU reference. FFT uses eight-thread FFTW; FWHT and zeta use
FXT; NTT uses HEXL; structured 2x2 uses the author CPU implementation. The
shared-request panel contains only exact intersections across cuButterfly and
available GPU baselines. The second panel retains cuButterfly requests without
a complete external GPU pair. CPU timings are a shared-host provisional
baseline and are not pooled with GPU/GPU ratios. Regenerate it with
`python paper/tools/export_cpu_normalized.py --cpu-results <cpu-results.json>`
followed by `python scripts/render_cpu_normalized_performance.py`.

Run `make -C paper refresh-data PYTHON=/home/wt/yes/envs/cubutterfly/bin/python`
from the repository root. This is CPU-only. The earlier `principle_e1_latency`,
filter-batch, and Poisson endpoint figures remain historical assets and no
longer supply current headline statistics. Their descriptions below document
their original populations. The compact cross-core and A100 40GB scaling
figures remain independent evidence in the manuscript.

`motivation_organization_a100` replaces the historical single-operator motivation
panel. Run `python paper/tools/render_motivation_organization.py` to validate
all 18 original releases and summarize all 162 M1 timings. Each of the six
boxes shows six workload ratios, median(stagewise)/median(grouped), for two
local groupings across FFT/FWHT FP64 and word64 NTT. All 36 ratios are plotted;
boxes are workload quartiles, whiskers min/max, not timing uncertainty. The
three-trial raw times, source hashes, and transformations are preserved in
`../generated/motivation_organization_a100.{csv,json}`. The separate six-panel
scaling plot retains the timing ranges. Neither plot substitutes for external
library comparisons or floating-point acceptance.

The manuscript now uses the compact 480 × 252 draw.io `roofline` figure at
single-column width. The ratio panel remains an audit asset rather than
duplicating the complete `narrative_scaling_20261027` measurements in the
evaluation. Seven markers report measured A100 40GB M1 medians: three matched
FFT FP64 organizations, two FWHT batch points, and two NTT organizations.
Their x coordinates are the derived ideal boundary intensity and their y
coordinates are pair rate; all use two materialized stage groups. The symbolic
roof is intentionally not fitted to the observations or treated as a hardware
peak. Equal ideal intensity therefore makes the execution gap visible without
claiming equal measured traffic or a unique bottleneck. Trial ranges and source
provenance remain in `../generated/roofline_cases_20261029.csv`.
Current paper evidence uses A100; PRO 6000 Server 96GB remains pending.

The 2026-10-29 compression combines two independent fixed-core cohorts into
`core_organization_compact`, a 3.45 × 1.85 inch two-panel single-column figure.
Reproduce it with
`python paper/tools/render_core_organization_compact.py`. Panel (a) preserves
all 12 historical FP32 ratios and paired three-trial ranges; panel (b) preserves
all nine FP64 cases, including complete cuFFT, and all 63 timing samples.
The renderer verifies the original source hashes and compares every plotted
point and range with the earlier CSVs. It does not resolve live binaries.
Separate point/sample CSVs and source/output hashes are in
`../generated/core_organization_compact*`. The cohorts are neither pooled nor
relabeled as current automatic-library results. The original standalone figures
remain available for audit.

The 2026-10-29 analytical redraws use compact canvases: `overview` is 1000 ×
300, `semantic_transport` is 480 × 274, and `system_architecture` is 1000 ×
330. The `pipeline` canvas is 480 × 274, reduced from 480 × 366 without changing
its readiness or buffer-lifetime witness. The architecture example shows two producer groups writing four successor
pairs through a shared aligned representation. The execution text defines the
remaining bank-conflict check and distinguishes local bank swizzles from
outer-storage coalescing; these schematics remain analytical and do not claim
that one XOR map is valid for every core or target.

The Poisson endpoint remains a separate figure, now drawn at native single-column
width rather than reducing a double-column plot. Both eager and graph modes,
all three backends, and all twenty repeated calls per cell remain visible.

`narrative_scaling_20261027` is the complete independent A100 40GB local
organization study. Run `python paper/tools/render_narrative_scaling.py` to
validate and plot the 18 original per-case releases referenced by
`results/narrative_gap_20261027/m1_gpu40_cases/collection.json`. Its six panels
cross FFT/FWHT FP64 and word64 NTT with N=2^16/2^20; each panel retains all
three batches and all three local groupings. Physical HBM groups are fixed
within each panel. Grouped bars/whiskers represent medians/full ranges of three
independent process trials, not confidence intervals. The ordinate counts
complex FFT values and scalar FWHT/NTT values, not common FLOPs. CSV and source,
script, identity, and output hashes are in
`../generated/narrative_scaling_20261027.{csv,json}`. The 3.45 × 3.15 inch
single-column PDF uses two rows by three columns, with operator-matched vertical
limits across rows and embedded TrueType fonts. All 54 configurations and 162
timings remain; its actual manuscript rendering was checked.
See the [result report](../evidence/NARRATIVE_SCALING_RESULTS_20261027.md) for
all 162 timings, exact NTT checks, floating-report limitations and negative
results. Interrupted attempts and other hardware never fill its cells.

The earlier manuscript included `principle_e1_latency`,
`principle_application_throughput_ratio`, and
`principle_application_poisson_endpoint`. Regenerate them with
`python paper/tools/render_principle_evaluation.py`. Their separate operator,
historical-core, image-filter, and Poisson populations and source/output hashes
are recorded in `../generated/principle_figure_metadata.json`. The historical
`principle_core_organization` is retained as an audit source for the combined
figure above. The headline
operator statistics in those archived figures retain the 36-request screen and
independent 12-request NTT extension; they are not pooled into the current refresh.

`factor_organization_20261025` is a separate FP64, N=65536, B=16 diagnostic:
two fixed cores, four organizations per core, and complete cuFFT. Regenerate
it with `python paper/tools/render_factor_organization.py` from the frozen
`../evidence/factor_organization_20261025.json`; the corresponding latency
and paired CSVs and metadata are in `../generated/`. Whiskers describe seven
rotated rounds in one process. The figure retains deeper-prefetch regressions
and the faster cuFFT result. It is an explicit matched-organization control,
not AUTO performance, a full-method ablation, or native library organization.

Figure preparation used the procedural guidance in Scientific Agent Skills
(Kassis et al., 2026, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065)).
This is tooling attribution, not a butterfly-method citation. All plotted
performance values come from the frozen measurement evidence above.

```bibtex
@misc{kassis2026scientificagentskills,
  title={Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents},
  author={Kassis, Timothy and Agarwal, Vinayak and He, Yuhuan and Patel, Darshil and Brueckner, Aubrey M.},
  year={2026},
  eprint={2609.00065},
  archivePrefix={arXiv},
  doi={10.48550/arXiv.2609.00065}
}
```

The assets below are retained historical figures rather than the current
manuscript's performance-figure inputs.

`motivation_crossover_20261027` and `tools/render_motivation_crossover.py`
retain the historical V100 NTT diagnostic and its original provenance.
They are no longer included in the manuscript and do not supply its claims
or future platform test requirements. The raw data and old exports remain
unchanged for audit.

`results_*` files are generated by `tools/render_experiment_figures.py` from
the recorded snapshot and medians. Its generated JSON/CSV summaries retain
case-level ratios, comparison contracts and source hashes. Do not merge
historical V100, current A100, profiler timing, and independent repair builds
into a common average. Unavailable or unfinished cells have no invented bar
height. The older `results_current_fft_distribution` and
`results_current_factorial_controls` files remain as generated historical
artifacts and are not included by the current evaluation. See
`../sections/evaluation.tex` for the scope of each comparison.

An earlier manuscript used `rc2_unified_absolute_boxplot` as its primary
performance figure. It reads the complete RC2 FFT, NTT, and FWHT summaries and
shows effective radix-two butterfly-pairs/s and transforms/s distributions on
logarithmic axes, grouped by operator and precision. Every sampled cell is
overlaid as a dot, so the boxes do not hide the workload count. The boxplot is
descriptive within each group and is not a pooled cross-operator score.
Each upper-right label reports the matched-cell geometric-mean cuButterfly
speedup and its observed minimum--maximum range. The historical coverage audit
also contains structured-$2\times2$, subset-zeta, superset-zeta, and xor-zeta;
those older cells are kept out of the RC2 absolute plot because they do not
share the latest RC2 baseline cohort.
`rc2_fft_absolute_boxplot` and `rc2_fft_absolute_throughput` remain available
for FFT-only audit and paired per-workload detail. The unified CSV retains
direction, shape, baseline identity, and source cohort. Missing contracts remain
empty, including VkFFT for FP64.

`profile_comparison_historical` is a measured Nsight Compute comparison built
from the preserved V100 APPT pipeline capture and a separate V100 structured
2x2 capture. It reports profiler replay time, traffic, registers, active warps,
and stalls; the captures are shown in separate panels and are never pooled.
Profiler time is mechanism evidence rather than warm CUDA-event latency. It is
not the pending E14 RC2 profile campaign.

`application_historical_complete_call` plots the four complete-call FNO and
QuaRot records already reported in the manuscript. It preserves absolute
latency and baseline-normalized speedup in separate panels; HEonGPU is left
unmeasured. The records are historical and are not pooled with operator
measurements.

`end_to_end_contract` is a protocol schematic for the three application
measurements. The manuscript also reports those historical pinned-build
complete-call timings in the application figure. HEonGPU remains unmeasured;
and no empty performance bars are drawn.

Regenerate these assets with
`python paper/tools/render_rc2_throughput.py`,
`python paper/tools/render_rc2_unified_boxplot.py`,
`python paper/tools/render_profile_comparison.py`,
`python paper/tools/render_application_historical.py`, and
`python paper/tools/render_end_to_end_contract.py` from the repository root.

`results_current_baseline_summary` remains a historical ratio distribution
artifact for auditability. It is no longer the primary figure because a ratio
boxplot hides the absolute scale and can mix unlike transform sizes. Dao's
interface includes allocation and round-trip normalization, as qualified in
the manuscript.

`results_current_fft_scaling` is a measured FFT length-by-batch scatter map
with separate FP32/FP64 and forward/inverse facets. Marker labels are
cuButterfly median kernel microseconds. Facets share one logarithmic color
scale. The view includes 27 contiguous out-of-place FFT cells and excludes
two cells with strided or in-place contracts. Missing cells remain blank
and no surface interpolation is applied. Both figures are generated from the frozen
`paper/evidence/current_snapshot.json`; the renderer defaults to that snapshot
and only reads live acceptance input when `--refresh-snapshot` is explicit.

## Archived FFT-L boundary figure (2026-11-04)

This standalone figure is no longer included in the manuscript. FFT FP16/BF16
now appear as precision columns in `current_refresh_operators.pdf`, with the
one public FP16 winner replay and explicit pending BF16 coverage. Fixed controls
are not substituted for automatic results. Profile observations remain in the
common analysis prose. The source below is preserved for evidence auditing.

`fft_l_boundary_20261104.pdf` is a 3.45 × 1.88 inch two-panel measured figure,
rendered by `paper/tools/render_fft_l_boundary.py`. Panel (a) retains all six
final-confirmed N4096 FP16/BF16 cases, including the BF16 B4 regression; the
large-batch points combine ownership, grouping, and boundary implementation
changes around fixed scalar radix-4 arithmetic. It is not an AUTO-library ratio.
Panel (b) plots store sectors/request from separate A100 40GB kernel captures,
with unchanged packed arithmetic counts. It is neither HBM bytes nor timing.
The raw profile's increased shared-memory bank conflicts are retained in prose.
Full sources, intermediate CSVs, exclusions, and build hashes are recorded in
`generated/fft_l_20261104/`; `figure_manifest.json` binds the exact renderer,
inputs, output dimensions, and files. No anticipated values are plotted.
