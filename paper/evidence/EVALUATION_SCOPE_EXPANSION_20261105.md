# Evaluation scope expansion — 2026-11-05

This revision expands the information shown by Figures 9–11 using persisted,
hash-bound observations. It does not invent measurements or combine hardware
cohorts.

## Fixed-core controls (Figure 10)

The FP32 panel retains the six workloads, two arithmetic cores, two
organizations, and three paired trials per cell. The plot now shows all 72
trial ratios as points behind the 12 cell summaries. The FP64 panel retains
eight fixed-core organization cases and one cuFFT reference. It now shows all
63 Graph timing rounds behind the nine cell summaries. The independent A100
80GB cohorts remain separate within the two panels.

Sources and validators are unchanged:

- `paper/generated/core_organization_compact_fp32_measurements.csv`
- `paper/generated/core_organization_compact_fp32_points.csv`
- `paper/generated/core_organization_compact_fp64_latency.csv`
- `paper/generated/core_organization_compact_metadata.json`

## Cross-operator organization controls (Figure 11)

The former six-bar view is replaced by six operator/length groups. Each box
contains the three measured batches (`B=1,4,16`) and three trials for one of
two grouped organizations, yielding 18 workloads, 54 organization cells, and
162 observations. The statistic is `local0 / variant`; values above one mean
that grouping is faster than the stagewise control. FFT, FWHT, and NTT remain
visible together while the A100 40GB M1 cohort stays distinct from the A100
80GB fixed-core experiment in Figure 10.

Source: `paper/generated/motivation_organization_a100.csv`.

## Application coverage (Figure 9)

The current complete-call snapshot still contains 46 eligible pairs across
Poisson/filter, CKKS, and bootstrap. Four already audited historical records
(two FNO and two QuaRot workloads) are now shown in a shaded pinned-build
cohort. They are labeled as historical and are excluded from current-batch
aggregates because their artifact and runner identities predate the current
refresh. This gives five application families in the visual without claiming
current-build performance for the historical records.

Source: `paper/generated/application_historical_complete_call.json`; the
current snapshot remains `paper/generated/current_refresh/snapshot.json`.

## Validation

`render_core_organization_compact.py` and `render_current_refresh.py
--figures-only` regenerated the derived PDFs, SVGs, PNGs, and manifests.
The current-refresh test suite passes 25/25 under the cuButterfly environment.
No GPU queue, `/sdb` path, or raw result was modified.
