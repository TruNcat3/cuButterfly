# Current measured paper snapshot

This directory is the refresh boundary between the live A100 campaign and the
manuscript. No synthetic performance values are used. Raw experiment files are
read-only inputs, not rewritten by figure preparation.

From the repository root:

```bash
make -C paper refresh-data PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
make -C paper all evidence
```

Use `make -C paper refresh` with the same `PYTHON` to perform both steps.
`CAMPAIGN` can select a compatible campaign directory, resolved relative to
`paper/`. Ordinary PDF builds do not reread live experiments.
`FP16_ACCEPTANCE` selects the independently completed low-precision cohort;
its default is `../results/a100_fp16_acceptance_20261105`. An incomplete
acceptance directory is an error, not a reason to substitute fixed controls.

| Output | Meaning |
| --- | --- |
| `snapshot.json` | Source hashes, hardware/build identities, actual mapping summaries, measured pairs, missing statuses and separate cohorts |
| `operator_summary.csv` | Geometric means and nominal faster counts, separated by study, operator, precision and competitor |
| `operator_table.tex` | Main comparison, nine-request FP16 DEFAULT cohort, and separate cyclic NTT extension; further BF16 experiments deferred; PRO 6000 pending |
| `ckks_table.tex` | CKKS by batch, preserving both native controls and the faster native choice per request |
| `numbers.tex` | Generated headline operator and CKKS numbers used by the manuscript |
| `cpu_normalized.json` | Exact main66 joins to the provisional 83-measurement CPU campaign; no cyclic/FP16 imputation |
| `figure_manifest.json` | Snapshot/renderer hashes, physical sizes, data descriptions and vector/raster output hashes |

The paired speedup figure includes all `main66` requests plus nine separately
identified FP16 DEFAULT requests. Every plotted point and box uses only a
request with a complete cuButterfly/baseline pair; no unmatched latency is
plotted. Their geometric mean against cuFFT is
0.853583, with one ratio above one. MEASURE and installed DEFAULT are retained
separately, never used as per-request minima. Further BF16 experiments are deferred. The earlier
22 fixed controls and single public-winner replay stay archived under
`generated/fft_l_20261104/` and do not enter this new cohort.
`tools/load_fp16_acceptance.py` checks completed parent receipts, paired inputs,
runtime identities, Graph timing and output hashes. The verified export is
`generated/fp16_acceptance_20261105/comparison.json`; the figure manifest binds
its loader and raw evidence. The original main snapshot and all headline macros
retain their previous denominators. `ntt5` and `fft_grouped2` remeasurements remain
separately identifiable. CKKS uses only
the five named holdout CSVs and `measured_auto_loop_or_stage_batch` rows;
population, selection, and sample CSVs are not additional workloads. Bootstrap
is separate. The same-padding Torch filter diagnostic is not a main competitor.

The independent `cyclic` object contains all 90 restored NTT requests and 54
GPU-NTT pairs. The table adds separate word32/word64 rows. Eighteen pairs are
original; 36 use unchanged cuButterfly timings and supplementary GPU-NTT
measurements on the same GPU and protocol. Source report/row checksums,
descriptors, mappings and timing contracts are verified when joining them.
The 36 out-of-domain comparisons retain null speedups. No cyclic pair is mixed
into the 66-request means or the five-request NTT replay. Two unsupported
requests occur in both studies, so their request totals are not a unique union.

Ratios are baseline/own complete-call median, with one vote per request in a
geometric mean. The five-point NTT replay and mechanism controls use the median
of paired trial ratios. Mechanism whiskers are trial min/max; operator boxes
show workload IQR/min/max, not uncertainty. Both raw medians and ratios remain
available in the snapshot. Nominal faster counts do not establish significance.

Floats remain error-report-only. Hollow fixed-filter points denote one process
per endpoint, not independent-process uncertainty or numerical-threshold
acceptance. No acceptance threshold is inferred from missing tolerances.
The snapshot retains original `accepted` and `formal_performance_eligible`
fields under `source_numerical_reports`; these provenance fields are not rewritten
when applying the error-report-only policy. Integer
comparison failures never yield eligible ratios. CKKS additionally requires
exact ciphertext, zero mismatches and completed, performance-eligible metadata.
Missing performance remains null; it is never zero or a win.

The 138 default finite-field compositions in this snapshot have no external
speedup claim. The separate searched-composition evaluation is now complete;
its [audited internal comparison](../../../results/a100_closure_20261102/composition_summary.md)
is reported in the controlled-experiment prose, not this snapshot's external
ratios. PRO 6000 migration remains unfinished. Neither enters the main means.
Incorporating a new cohort requires declaring its
scope, adding its presentation, and reviewing prose, rather than silently
growing a previous denominator. Existing historical negative evidence stays
separate. Final refresh also requires checking numerical prose and PDF layout.

The application figure upper panel shows the 46 current complete-call pairs
and two numerically checked historical FNO observations in a separately shaded
pinned-build cohort. Two early QuaRot observations remain archived but are
excluded because random-state preparation was invalid and outputs were not
checked. Historical FNO records are coverage evidence only and are excluded
from current batch aggregates. The lower panel retains the independent
application queue, FNO Graph aggregates, and fused QuaRot follow-up as separate
cohorts; see [APPLICATION_COHORTS.md](APPLICATION_COHORTS.md).
The organization figure reads the audited
`experiments/snapshots/motivation_organization_a100.csv` table directly and displays
18 workloads, 54 organization cells, and 162 trial observations across FFT,
FWHT, and NTT.
