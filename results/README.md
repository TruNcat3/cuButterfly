# Result Inventory

This directory stores measured records and reduced reports. Raw measurements
are immutable evidence; summaries and Markdown tables are reproducible derived
artifacts.

## v0.8 Frozen V100 Evidence

The current release-facing reports are linked from
[`docs/release_v08.md`](../docs/release_v08.md). The main v0.8 artifacts are:

| Dataset | Purpose | Entry point |
|:--|:--|:--|
| `v08_search_current_v100_v2/` | Fresh current-code butterfly candidate search and screening | `raw.csv`, `summary.md` |
| `v08_ntt_search_current_u32/`, `v08_ntt_search_current_u64/` | Width-specific NTT search and confirmation | `summary.md` |
| `cross_operator_v08_wide_current/` | Broad current-build operator matrix | `comparison.md` |
| `cross_operator_v08_external_baselines_current_v2/` | Matched external-baseline coverage | `comparison.md` |
| `fft_three_way_v08_current/` | Clean v0.6/v0.8/cuFFT long-FFT protocol | `report.md`, `summary.csv` |
| `ncu_cross_operator_v08/`, `ncu_fft_fp32_swizzle/` | Counter attribution and shared-layout evidence | reduced CSV/Markdown reports |

These datasets are V100 evidence. A directory containing a scan or probe is
not automatically a release result: use the linked report to check whether the
rows are screening, confirmed, external, or placeholder evidence.

## Canonical V100 Evidence

| Dataset | Purpose | Raw record | Reduced report |
|:--|:--|:--|:--|
| comprehensive suite | controlled cross-operator baseline for `v0.2.0` | `comprehensive_v100_full_raw.csv` | `comprehensive_v100_full_summary.csv`, `comprehensive_v100_full_report.md` |
| orthogonal scaling | independent length/batch saturation and crossover scan | `v100_scaling_full_raw.csv` | `v100_scaling_full_summary.csv`, `v100_scaling_full_report.md` |
| calibrated selector | held-out-shape mapping prediction | `v100_mapping_selector_evaluation.csv` | `v100_mapping_selector_metrics.json` |
| external baseline refresh | matching-protocol Dao FHT and GPU-NTT pairs | `v100_external_baselines_raw.csv` | `v100_external_baselines_summary.csv`, `v100_external_baselines_report.md` |
| crossover counters | NCU mechanism attribution for FFT/FWHT scaling boundaries | `ncu_scaling_crossovers/*.csv` | `ncu_scaling_crossovers/summary.csv`, `ncu_scaling_crossovers/attribution.md` |
| hardware capabilities | V100 service and resource profile | `hardware_capabilities_raw.csv` | `hardware_capabilities_v100.json` |
| Boolean-zeta direction check | subset/superset/legacy pair-update mapping comparison | `butterfly_operators_expanded_v100_raw.csv` | `butterfly_operators_expanded_v100_summary.csv` |
| structured 2x2 mapping | dense local-matrix length/precision design-space scan and selected-point preflight | `structured_2x2_v100_raw.csv`, `structured_2x2_v100_best_verify.csv` | `structured_2x2_v100_summary.csv`, `docs/structured_2x2_v100_results.md` |
| matched FWHT control | same candidate space used to isolate generic matrix-unit cost | `fwht_matched_v100_raw.csv` | `fwht_matched_v100_summary.csv` |
| structured/FWHT order control | alternating-order confirmation of the FP32 `logN=20` same-mapping result | `structured_fwht_log20_interleaved_v100_raw.csv` | interpreted in `docs/structured_2x2_v100_results.md` |
| structured 2x2 counters | same-mapping arithmetic plus broadcast/per-stage register attribution | `ncu_structured_2x2_policy/*.csv` | `ncu_structured_2x2_attribution.csv`, `ncu_structured_2x2_attribution.md` |
| structured register core | generated register-resident matrix core across representative resident lengths | `structured_register_v100_raw.csv`, `structured_register_vs_fwht_v100_raw.csv` | `structured_register_v100_summary.csv`, interpreted in `docs/structured_2x2_v100_results.md` |
| structured coefficient policy | equivalent broadcast-register versus repeated per-stage-table matrices | `structured_broadcast_policy_v100_raw.csv` | `structured_broadcast_policy_v100_summary.csv`, `structured_register_resources_v100.csv` |
| structured broadcast/FWHT control | alternating-order same-transport arithmetic comparison after coefficient reuse | `structured_broadcast_vs_fwht_v100_raw.csv` | interpreted in `docs/structured_2x2_v100_results.md` |
| structured residency features | portable hardware-resource features for both register coefficient policies | `structured_register_resource_profile_v100.csv` | `structured_register_residency_features_v100.csv` |

The `quick` files are protocol smoke tests. Files containing `confirm` retain
independent reruns used to investigate instability; they do not silently
replace rows in the canonical full raw data.

## Focused Evidence

`general_shapes_v100_smoke.csv` is the first seven-row v0.6 exact-length FFT
diagnostic against direct cuFFT. It uses one aggregated CUDA-event run per row,
so it locates a large implementation gap but is not canonical repeated-trial
evidence. Its protocol and interpretation are in
`docs/general_shape_results.md`.

`general_shape_selection_v100_raw.csv` is the five-trial follow-up after
composition selection. It covers direct-versus-Bluestein FFT, selected
Bluestein NTT cores, embedded FWHT/zeta, and default versus selected Structured
2x2. Runs are separate warmed processes and are interpreted in the same report.

`general_shape_direct_boundary_v100_raw.csv` isolates the next lowering step:
contiguous physical output replaces the final scatter for embedded FWHT,
subset/superset zeta, and Structured 2x2. The noncontiguous fallback remains a
C API correctness test.

`general_shape_fused_input_v100_raw.csv` records the selected first-boundary
fusion for FP32 warp-register FWHT and the rejected Structured control. The
FWHT rows use five independent warmed processes with 100 timed launches; the
algorithm suffix and zero workspace verify that the pack kernel is absent.

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
