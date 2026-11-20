# Evaluation evidence and reproduction

The evaluation follows the method's evidence chain: hold arithmetic kernels
fixed to study execution organization, compare complete transforms, measure
application propagation, and explain resource and boundary trade-offs. This
bundle connects each experimental figure to measured data and executable
analysis. It contains no manuscript source or manuscript PDF.

## Rebuild the figures without a GPU

Use Python 3.10 or later. Install the plotting dependencies once, then run:

```bash
python -m pip install -r experiments/paper/requirements.txt
python experiments/paper/reproduce.py --check-only
python experiments/paper/reproduce.py --output-dir results/paper-figures
```

The last two commands require no network, CUDA, or device. They verify the
bundled inputs and regenerate five figures and the boundary-control table.
Each figure is exported as PDF, SVG and PNG with numerical records and
provenance. `reproduction.json` records output hashes. PDF metadata and font
versions can affect image hashes; numerical equivalence is checked separately.
A subset can be rebuilt with `--figures 9 11`. Figures sharing inputs are
rendered together. Original source hashes and sanitized-bundle hashes remain
separate in the provenance records.

| Evidence | Question answered | Bundled measurement and analysis | New measurement route |
| --- | --- | --- | --- |
| Fig. 8, organization attribution | Can organization improve performance with arithmetic fixed? | [Organization/application package](organization_applications/README.md); 150 exported observations, independent cohorts preserved | Controlled-study runners and arithmetic-provider constraints; exact historical builds are not reconstructed by redrawing |
| Fig. 9, complete transforms | Does the generated implementation compete with specialized libraries? | [Operator comparison](operator_comparison/README.md); 268 A100 and 143 PRO 6000 plotted observations, paired CPU/GPU timings and full coverage tables | [Platform runner](../../scripts/research_platform.py), explicit manifests and installed comparator providers; `full` covers main66, not every supplemental cohort |
| Fig. 10, applications | Do transform gains reach the application? | [Application inputs and analysis](organization_applications/README.md); 36 CKKS requests use the faster matched native alternative, with other integration cohorts separate | Application-specific runners with pinned dependencies and complete-call timing; historical cohorts remain distinct |
| Fig. 11, NTT mechanisms | How do retention and boundary layout change cost and concurrency? | [NTT mechanisms](ntt_mechanisms/README.md); 82 chart rows, timing samples and per-kernel counters, including the negative layout control | Pinned arithmetic-provider harness for panels a–d; local-layout panels additionally require the matching library benchmark |
| Fig. 12, resource costs | Why do different operators prefer different group depths? | [Resource profiles](organization_applications/README.md); 18 measured profile records | Fixed-request/kernel controls and separate ordinary timing/NCU captures |
| Boundary-control table | Can joint boundary placement improve a fixed implementation? | [Boundary exporter](organization_applications/export_boundary_positive.py); six paired process records across two batches | Fixed cuFFTDx arithmetic, global layout and launch resources; only the local slot permutation changes |

The numerical schemas, inputs, executable entry points and output names are
listed in [manifest.json](manifest.json). Figure numbers identify this evidence
revision; the semantic names and data hashes are the durable identifiers.

## Measure again

Offline regeneration does not execute or validate kernels on a new device.
Start a new campaign with the [benchmark guide](../../benchmark/README.md)
and [platform workflow](../../docs/hardware_profile_install.md), and record
hardware, compiler, provider versions, request semantics and numerical checks.
Each experiment keeps its timing boundary: initialization and compilation are
outside warm execution, while required conversions and normalization remain
inside complete-call timing. NCU counters explain behavior; they do not replace
ordinary performance measurements.

The older E1–E14 preparation plan is a broader research protocol, not this
figure manifest. E14 preparation now preserves its five requested comparisons
and generates explicit bindings for measured journal cases and variants.
Unbound cases and unavailable counters are recorded as missing. The generic
collector's kernel filter and metrics do not automatically cover the newer
NTT experiment; use the Figure 11 package for that control.

No new measured speedup is produced by the offline command. Missing backends,
floating-point error-review states, nonmatching populations and performance
losses remain visible in their source tables and descriptions.
