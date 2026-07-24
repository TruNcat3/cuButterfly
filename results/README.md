# Result Inventory

This directory stores measured records and reduced reports. Raw measurements
are immutable evidence; summaries and Markdown tables are reproducible derived
artifacts.

## Canonical V100 Evidence

| Dataset | Purpose | Raw record | Reduced report |
|:--|:--|:--|:--|
| comprehensive suite | controlled cross-operator baseline for `v0.2.0` | `comprehensive_v100_full_raw.csv` | `comprehensive_v100_full_summary.csv`, `comprehensive_v100_full_report.md` |
| orthogonal scaling | independent length/batch saturation and crossover scan | `v100_scaling_full_raw.csv` | `v100_scaling_full_summary.csv`, `v100_scaling_full_report.md` |
| hardware capabilities | V100 service and resource profile | `hardware_capabilities_raw.csv` | `hardware_capabilities_v100.json` |

The `quick` files are protocol smoke tests. Files containing `confirm` retain
independent reruns used to investigate instability; they do not silently
replace rows in the canonical full raw data.

## Focused Evidence

Files prefixed with `fft_`, `cubutterfly_`, `processing_units_`, `ntt_`, or
`stage_pipeline_` are focused design-space experiments. Their protocol and
interpretation live in the matching document under `docs/`; they must not be
mixed directly with the comprehensive suite unless their contracts and clock
policy are explicitly aligned.

Profiler exports are grouped in `ncu*` and `nsys*` directories. Binary
`.ncu-rep`, `.nsys-rep`, and `.sqlite` files are machine artifacts and are not
versioned; reduced CSV and analysis documents may be versioned.

## Naming Policy

- `*_raw.csv`: append-only trial or counter records;
- `*_summary.csv`: deterministic reduction of raw records;
- `*_report.md`: human-readable table generated from the summary;
- `*_search.csv`: explored candidates, including rejected or nonselected rows;
- `*_confirm_*`: an explicitly separate rerun for a suspected unstable case.

New paper-facing datasets must include the GPU, workload family, and protocol
scope in the basename and be linked from `docs/experiments.md`.
