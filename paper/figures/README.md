# Figure sources and scope

These are editable vector drafts for the cuButterfly manuscript. PDF files
are included by LaTeX; SVGs support inspection and editing. Schematics and
qualitative curves are explicitly distinguished from measured results.

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

The eight analytical figures included by the manuscript now use native
draw.io sources in [`drawio/`](drawio/README.md). Edit their named pages in
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
| `roofline` | Adapt APPT's roofline motivation | Two-panel analytical schematic: traffic/core ceilings plus the independent-ready-subgraph ceiling |
| `organization_tradeoffs` | APPT same-workload stage/data comparison, generalized to butterfly bands | Same 16-point FWHT shown as stage-wise, two-band, and whole-transform organizations; exact analytical transfer and component counts |
| `hybrid_dataflow` | Earlier redraw of APPT's aligned stage/data progression | Retained source asset, its main-text role is merged into `overview` |
| `hierarchy` | Extend partial-stage/data grouping | N=16 producer/consumer example and a hypothetical capacity-limited 8+8+8 decomposition |
| `bank_layout` | Adapt `data_arrangement.pdf` and bank/offset reasoning | Exact four-bank XOR example for single-word values; not a universal GPU transaction proof |
| `pipeline` | Extend APPT's block progression and UnionSparse event lanes | A/B/C readiness barriers and buffer lifetime witness; not a measured trace or guaranteed GPU overlap |
| `semantic_transport` | TensorIR-style explicit interface plus butterfly order correspondence | Bit reversal, axis labels, and CT/local-inverse GS contracts; not arbitrary stage commutation |
| `framework` | Earlier GPU generation workflow | Retained source asset, superseded by `system_architecture` |

The analytical figures use a shared palette and consistent labels. The roofline
uses symbolic ceilings and explicit attainment conditions; it contains no
measured points. `organization_tradeoffs` is the same-workload bridge from
the background to the roofline. `semantic_transport` makes the structural
correspondence and arithmetic contract visible. `hybrid_dataflow` is the generic
replacement candidate for the inherited NTT-only two-panel artwork and keeps
both panels geometrically aligned. Their numerical
indices are analytical examples, not placeholders for measured performance.

The main figure explains the organization, and the architecture figure explains
its automatic realization. Neither is a performance measurement. The
N=16 `hierarchy` figure remains the worked construction example. The paper
uses eight analytical figures plus the measured cohorts listed below, without
adding empty application bars. Pending application evidence retains its explicit
status. Retained earlier redraws are not included merely to
increase the figure count.

## Experimental figures

The current manuscript includes `principle_e1_latency`, the historical
`principle_core_organization`, `principle_application_throughput_ratio`, and
`principle_application_poisson_endpoint`. Regenerate them with
`python paper/tools/render_principle_evaluation.py`. Their separate operator,
historical-core, image-filter, and Poisson populations and source/output hashes
are recorded in `../generated/principle_figure_metadata.json`. The headline
operator statistics retain the 36-request screen and independent 12-request
NTT extension; later prose supplements do not enter those populations.

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
