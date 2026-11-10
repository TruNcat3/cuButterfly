# Frozen benchmark results and evidence

Read results by experimental question. Each linked report retains its own
request population, device, build, numerical reference and timing boundary.
The [benchmark contracts](../../benchmark/README.md) explain how to declare and
measure a new campaign; the [reporting contract](../reporting_contract.md) defines
provenance and evidence labels. This page indexes existing measurements and
introduces no new statistics.

## Operator comparisons

The [November 2 frozen evidence](../../paper/evidence/CURRENT_EVIDENCE_20261102.md)
reports the A100 80GB main 66-request comparison, independent 90-request cyclic
NTT study and 138-request internal finite-field composition study. Read each
competitor's matched denominator: unsupported and inapplicable baseline cells
are explicit gaps. Overlapping requests do not become additional unique requests;
internal composition controls are separate from external-library comparisons.
The [November 7 closeout](../../paper/evidence/A100_CLOSEOUT_20261107.md) states
the bounded completion scope and build/device identities.

Preserve the FFT losses and the FP32 FFT/VkFFT distribution, where fewer than
half the measured requests lead despite a positive geometric mean. The
November 2 FWHT comparison used the historical supplied-output interface;
later native-return and supplied-output measurements remain distinct:
[FWHT layout repair](../fwht_native_repair_20261107.md) and
[A100 interface follow-up](../a100_evidence_followup_20261107.md).
The native-return results retain individual losses and do not inherit the
historical supplied-output aggregate.

Low-storage-precision FFT evidence is separate from the main population.
The [November 5 FP16 public acceptance](../../paper/evidence/FP16_PUBLIC_ACCEPTANCE_20261105.md)
covers nine A100 80GB public requests. The [November 7 status](../research_status_20261107.md)
and [closeout](../../paper/evidence/A100_CLOSEOUT_20261107.md) also identify the
independent eleven-request A100 40GB automatic-selection/default-replay supplement,
including its aggregate performance deficit. The latter skipped independent
stage-service calibration and leaves model ranking on unseen requests unvalidated.
It is not pooled into the 80GB headline population. Existing BF16 records retain
their scope; FP16 acceptance does not establish full BF16 comparison.

### Reading performance distributions

The [README overview](../../figures/readme_performance.svg) uses a paired
speedup for each complete request: external-library latency divided by
cuButterfly latency. Length, batch, rank, modulus, direction and layout remain
matched before aggregation. Each dot is one request; boxes show workload IQR
and whiskers span min/max, not confidence intervals. A geometric mean above
one does not imply every request is faster. The [plotted data](../../figures/readme_performance.csv)
and [provenance](../../figures/readme_performance.json) retain the separate studies.
Regenerate this view without a GPU with `python scripts/render_readme_performance.py`.

The complementary [CPU-normalized figure](../../paper/figures/cpu_normalized_performance.svg)
uses the same 66 main requests and plots CPU time divided by each backend's
complete-call time. The CPU denominator is explicit per operator family:
eight-thread FFTW for FFT, FXT for FWHT/zeta, HEXL for NTT, and the author CPU
implementation for structured 2x2. The shared panel uses only exact request
intersections across cuButterfly and the available GPU baselines; the second
panel keeps cuButterfly-only requests whose external GPU comparator is missing.
The 83 CPU measurements come from a shared AMD EPYC host with seven samples of
twenty calls after ten warmups. CPU wall-clock timing and resident GPU timing
have different boundaries, so this view is a complementary, provisional
baseline and does not replace the GPU/GPU comparison.

Absolute latency distributions answer a different question: how long the
represented requests take. They cannot establish relative performance across
different populations. For example, the earlier coverage figure includes nine
cuButterfly word64 NTT requests but only five GPU-NTT matches. Its four unmatched
multidimensional requests extend the cuButterfly range to 61.696 microseconds;
the five matched requests reach only 23.501 microseconds, against 26.419 for
GPU-NTT. Those unmatched requests must not enter a paired performance distribution.

GPU-NTT comparisons cover both 32-bit and 64-bit storage. The
[pinned arithmetic contract](../../benchmarks/gpuntt_crypto/README.md) requires
`2 < q < 2^30` for Data32 and `2 < q < 2^61` for Data64, with a prime modulus
supporting the transform order. Storage width is distinct from modulus width.
The four main word32 requests use `q = 2013265921`, outside the pinned Data32
range. Their missing comparisons do not mean that GPU-NTT has no 32-bit path.
The independent cyclic study includes compatible moduli and supplies these results:

| Study | Storage | Matched requests | Geometric mean | Ratios above 1 | Min–max |
| --- | --- | ---: | ---: | ---: | ---: |
| Main comparison | word64 | 5 | 1.188× | 4/5 | 0.976–1.516× |
| Cyclic extension | word32 | 18 | 1.400× | 18/18 | 1.196–1.990× |
| Cyclic extension | word64 | 36 | 1.265× | 33/36 | 0.972–1.593× |

The cyclic study retains 36 requests outside the pinned baseline's modulus
range as missing comparisons. Its 54 valid pairs contain 18 original pairs
and 36 separately supplemented GPU-NTT measurements using the original
cuButterfly timings; this is not a fresh interleaved rerun. The main comparison
and cyclic extension keep independent denominators. Observed ratios close to
one are not claims of statistically significant wins or losses.

## Fixed-core organization and resource attribution

[November 8 subgraph execution](../../paper/evidence/SUBGRAPH_EXECUTION_20261108.md)
and [representative profiles](../../paper/evidence/REPRESENTATIVE_PROFILES_20261108.md)
connect fixed-core organization with ordinary timing, traffic, resources and
stalls. The representative profile supplement covers two requests and twelve
variants with three-process timings and complete-call captures on an
independently identified A100 80GB. Profiling supplies mechanism evidence;
unprofiled timing supplies speedups. These captures are separate from the
frozen external comparison population.

[Resource decisions](../../paper/evidence/RESOURCE_DECISIONS_20261107.md) and the
[November 8 decision walkthrough](../../paper/evidence/DECISION_WALKTHROUGH_20261108.md)
separate retrospective service-prediction checks from prospective measured
selection. FWHT endpoint controls and unchanged NTT endpoints remain explicit;
there is no independent NTT numerical service prediction. A fixed arithmetic
leaf does not fix ownership, storage, geometry or wrappers, so these controls
do not isolate residency alone.

The [November 9 boundary I/O control](../../paper/evidence/BOUNDARY_IO_CONTROL_20261109.md)
and [method/I/O figure review](../../paper/evidence/METHOD_IO_FIGURE_REVIEW_20261109.md)
provide a bounded diagnostic and interpretation of the mapping figures.
The earlier [October 25 factor-core organization experiment](../factor_core_organization_experiment_20261025.md)
retains its explicit-mapping FP64 diagnostic and faster cuFFT comparator.
These studies do not certify AUTO selection over all requests or global optimality.

## Complete applications

The [November 7 closeout](../../paper/evidence/A100_CLOSEOUT_20261107.md) identifies
nine frozen A100 80GB application processes and fourteen comparison rows from
the recorded `fac109` build. These measurements remain build-prior evidence;
a newer installed library does not relabel their timing. The
[application cohort ledger](../../paper/generated/current_refresh/APPLICATION_COHORTS.md)
separates included historical observations, excluded observations and independent
follow-ups. Historical and new repetitions remain separate.

The [QuaRot fused-path report](../quarot_fused_path_20261107.md) records a separate
complete-call follow-up and retains the older composed-path losses. Native return,
supplied output, device events and host-wall timing have different boundaries.
The [evaluation protocol notes](../../paper/evidence/EVALUATION_PROTOCOL_NOTES_20261108.md)
explain Graph replay versus independent processes, the bootstrap source/binary
correspondence limitation and application inclusion rules. Small-parameter
bootstrap and random-weight QuaRot observations do not establish deployment
security or model task quality. Internal NTT compositions are not complete
FHE application runs.

## Numerical and statistical interpretation

[November 8 numerical scope](../../paper/evidence/NUMERICAL_SCOPE_20261108.md)
identifies reference arithmetic, error distributions, missing comparators and
source hashes for the main population, independent FP16 cohort and FWHT audit.
FP32 main-population references use higher precision; same-precision FWHT and
explicitly rounded FP16 references have separate contracts. Floating measurements
remain error reports without newly introduced acceptance thresholds; integer
comparisons retain their exactness requirements.

The [protocol notes](../../paper/evidence/EVALUATION_PROTOCOL_NOTES_20261108.md)
distinguish repeated Graph calls from independent process trials. The frozen
operator report uses matched per-request latency ratios and equal-weight
geometric means within the declared cohort. Neither a nominal win nor repeated
calls in one process establish statistical significance. Keep negative results,
missing adapters and unsupported requests visible rather than converting them
to zero latency or dropping them from an asserted complete population.

The [November 8 consolidation](../../paper/evidence/PAPER_CONSOLIDATION_20261108.md)
connects these evidence types to manuscript claims and limitations.
[Static preparation repair](../static_planning_preparation_20261108.md) documents
CPU preparation and public DEFAULT checks separately from kernel performance;
shared-host preparation observations are not an installation benchmark.

## Provenance and reproduction

Start with the [paper evidence index](../../paper/evidence/README.md), the linked
reports' exact manifests and [reproducibility guide](../reproducibility.md).
Earlier dated summaries retain their original populations and boundaries.
[Archived document inventory](../archive/README.md) includes frozen v0.8/V100
reports and older diagnostics; cross-device results are not one pooled cohort.

Raw research artifacts may be absent from a source-only checkout. Their
repository-relative paths remain recorded by the evidence reports. For recovery,
the November 2 ledger is `results/a100_closure_20261102/README.md`; its
`winner_correction_20261102/REPORT.md` documents the CSV string-order correction
and numerical replay, while `negative_trace_20261102/README.md` preserves bounded
negative-result diagnostics. The application repetition ledger is
`results/a100_closure_20261102/applications/progress_summary.md`.
Use the [artifact index](../../results/README.md) for available raw records and
[scope validation instructions](../../benchmark/README.md#reproduce-archived-populations)
for population/hash checks. Scope validation, numerical checking and evidence
completion are separate operations.
