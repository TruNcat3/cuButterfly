# Numerical evidence scope audit — 2026-11-08

CPU-only audit of existing records; no new measurements, references, thresholds, fitting, or manuscript changes. Cohorts remain separate. Min/median/max below are unweighted distributions **across recorded per-request metrics**, not pooled elementwise errors. Repeated FWHT outputs retain side/interface/process identity and are not 240 independent requests.

The source paths below are repository-relative. Raw `results/` data are local research artifacts; a source-only checkout can read this summary and provenance manifest without possessing those arrays.

## Frozen main66

Source: `results/full_current_20261025/measurement/results.json`; all 66 row payload hashes match index/envelope. Summaries use `row.correctness.{max_abs,l2_relative}` and baseline equivalents, plus `error_artifact.array_dtypes.reference` and `numeric_diagnostics.actual_nonfinite_elements`. Full per-request distributions are `correctness.error_distributions.{absolute,relative_nonzero_reference,absolute_zero_reference}` (quantiles, zero/nonfinite counts and log bins); lossless NPZ path/hash/dtypes are in `error_artifact`. Missing or inapplicable competitors are not assigned zero error.

| Operator / storage | Requests | Saved reference dtype | max_abs: min / median / max | L2-relative: min / median / max | Nonfinite actual count (missing diagnostics) |
|---|---:|---|---|---|---|
| FFT / FP32 | 11 | complex128 (11) | 3.79816809e-08 / 1.8589469e-05 / 0.00445484752 [missing=0, nonfinite=0] | 7.90264529e-08 / 1.87823631e-07 / 2.97920085e-07 [missing=0, nonfinite=0] | 0 (0) |
| FFT / FP64 | 6 | complex128 (6) | 6.3024981e-17 / 1.27162968e-13 / 4.88086075e-11 [missing=0, nonfinite=0] | 2.4454441e-16 / 4.13040984e-16 / 4.95777612e-16 [missing=0, nonfinite=0] | 0 (0) |
| FWHT / FP32 | 11 | float64 (11) | 1.181339e-08 / 5.77326864e-06 / 0.00132930324 [missing=0, nonfinite=0] | 7.83279206e-08 / 9.61227695e-08 / 1.09582879e-07 [missing=0, nonfinite=0] | 0 (0) |
| FWHT / FP64 | 2 | float64 (2) | 0 / 0 / 0 [missing=0, nonfinite=0] | 0 / 0 / 0 [missing=0, nonfinite=0] | 0 (0) |
| NTT / uint32 | 4 | uint32 (4) | not applicable (integer exact) | not applicable (integer exact) | not applicable |
| NTT / uint64 | 9 | uint64 (9) | not applicable (integer exact) | not applicable (integer exact) | not applicable |
| structured-2x2 / FP32 | 2 | float64 (2) | 3.92690708e-07 / 7.1577907e-07 / 1.03886743e-06 [missing=0, nonfinite=0] | 6.81615556e-08 / 7.98550004e-08 / 9.15484451e-08 [missing=0, nonfinite=0] | 0 (0) |
| structured-2x2 / FP64 | 5 | float64 (5) | 0 / 1.33226763e-15 / 7.10542736e-15 [missing=0, nonfinite=0] | 0 / 2.14429606e-16 / 2.97364904e-16 [missing=0, nonfinite=0] | 0 (0) |
| subset-zeta / uint32 | 3 | uint32 (3) | not applicable (integer exact) | not applicable (integer exact) | not applicable |
| subset-zeta / uint64 | 5 | uint64 (5) | not applicable (integer exact) | not applicable (integer exact) | not applicable |
| superset-zeta / uint32 | 3 | uint32 (3) | not applicable (integer exact) | not applicable (integer exact) | not applicable |
| superset-zeta / uint64 | 5 | uint64 (5) | not applicable (integer exact) | not applicable (integer exact) | not applicable |

FP32 FFT references are complex128; FP32 FWHT/structured references are float64. These are higher precision diagnostics, not same-precision CPU validation. FP64 storage matches reference dtype, which alone does not establish operation-by-operation arithmetic equivalence. Integer exactness uses original `mismatches`/verdict fields; floating error metrics and finite-value diagnostics are not applicable to exact-integer records. Floating historical `atol`/`rtol` and mismatch counts are explicitly historical diagnostics, not newly applied acceptance criteria.

| Operator / storage | Comparator | Numeric records / requests | max_abs min / median / max | L2-relative min / median / max |
|---|---|---:|---|---|
| FFT / FP32 | cufft | 11 / 11 | 3.79816809e-08 / 1.19405848e-05 / 0.00581064137 [missing=0, nonfinite=0] | 7.90264529e-08 / 1.87398151e-07 / 2.93587812e-07 [missing=0, nonfinite=0] |
| FFT / FP32 | vkfft | 11 / 11 | 3.79816809e-08 / 1.475294e-05 / 0.0124992277 [missing=0, nonfinite=0] | 7.90264529e-08 / 2.8033934e-07 / 4.19388708e-07 [missing=0, nonfinite=0] |
| FFT / FP64 | cufft | 6 / 6 | 1.1443917e-16 / 4.68812229e-13 / 4.60171964e-11 [missing=0, nonfinite=0] | 4.39238216e-16 / 4.73653977e-16 / 5.00457279e-16 [missing=0, nonfinite=0] |
| FFT / FP64 | vkfft | 6 / 6 | 1.11022302e-16 / 2.27435739e-13 / 3.25390717e-11 [missing=0, nonfinite=0] | 2.90646133e-16 / 3.8686852e-16 / 4.30468228e-16 [missing=0, nonfinite=0] |
| FWHT / FP32 | daofht | 10 / 11 | 1.181339e-08 / 2.90394041e-06 / 0.000387101165 [missing=1, nonfinite=0] | 7.83279206e-08 / 9.57624376e-08 / 1.07904877e-07 [missing=1, nonfinite=0] |

Dao has 10 measured FP32 numerical records for 11 FWHT requests; the unmatched FP32 request and both FP64 requests are `missing-baseline` / `unsupported-semantic` and are not lost error records. Structured records have no matched measured external floating comparator in this audit. Integer NTT/zeta records use exact comparisons; floating error scalars and floating finite diagnostics are not applicable to them. Inapplicable comparator rows are excluded above, rather than treated as missing measurements.

## Independent FP16 nine-request cohort

A100 80GB PCIe, GPU `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`, library `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`. Nine forward C2C requests per path. DEFAULT, MEASURE and installed DEFAULT remain distinct; the cuFFT rows repeat the same nine comparator records across paths. Sources resolve below `results/a100_fp16_acceptance_20261105`; each exported record carries `source`, `source_sha256`, `baseline_source`, `baseline_source_sha256`, `prepared_sha256` and `reference_sha256`.

| Path | Backend | Requests | max_abs_finite min / median / max | l2_relative_finite min / median / max | Actual / reference nonfinite |
|---|---|---:|---|---|---|
| default | cuButterfly | 9 | 0.171219007 / 0.254356572 / 1.20586134 [missing=0, nonfinite=0] | 0.00192599044 / 0.00225081277 / 0.00259573531 [missing=0, nonfinite=0] | 0 / 0 |
| default | cuFFT | 9 | 0.103127959 / 0.288110764 / 0.841432001 [missing=0, nonfinite=0] | 0.00172909817 / 0.00182793095 / 0.00198122134 [missing=0, nonfinite=0] | 0 / 0 |
| measure | cuButterfly | 9 | 0.171219007 / 0.254356572 / 1.20586134 [missing=0, nonfinite=0] | 0.00192599044 / 0.00225081277 / 0.00259573531 [missing=0, nonfinite=0] | 0 / 0 |
| measure | cuFFT | 9 | 0.103127959 / 0.288110764 / 0.841432001 [missing=0, nonfinite=0] | 0.00172909817 / 0.00182793095 / 0.00198122134 [missing=0, nonfinite=0] | 0 / 0 |
| installed | cuButterfly | 9 | 0.171219007 / 0.254356572 / 1.20586134 [missing=0, nonfinite=0] | 0.00192599044 / 0.00225081277 / 0.00259573531 [missing=0, nonfinite=0] | 0 / 0 |
| installed | cuFFT | 9 | 0.103127959 / 0.288110764 / 0.841432001 [missing=0, nonfinite=0] | 0.00172909817 / 0.00182793095 / 0.00198122134 [missing=0, nonfinite=0] | 0 / 0 |

The declared CPU implementation is `scripts/low_fft_cpu_reference.py`: native mode rounds each primitive RNE to FP16/BF16; twiddles use host-double cos/sin → FP32 → storage. FP32 containers do not imply FP32 native arithmetic; the code explicitly disclaims instruction equivalence and vendor arithmetic parity. Exported error fields contain per-request absolute quantiles and relative quantiles on nonzero reference values, finite-pair counts and exact-component counts; `acceptance_threshold` is null and verdict is `error-report-only`. A small L2 error does not bound maximum relative error near zeros. Per-path raw output records retain `all_outputs_finite`, `direct_graph_byte_equal` and `input_output_guards_passed`; semantic controls and continuity checks are outside these nine requests.

## FWHT same-precision saved-output audit

Ten requests, 240 saved outputs, three process trials; four sides and two return interfaces. The primary reference is actual FP32 Sylvester radix-2 add/sub, axis order 1..rank, FP32 per-axis inverse normalization. Error subtraction/distributions use FP64, which does not change reference transform precision. Original FP64-reference metrics stay in `correctness_fp64_reference_diagnostic`; original reports and arrays were not relabeled. This audit covers these saved outputs, not all main66 FWHT/structured cases.

| Side | Interface | Outputs (unique requests) | max_abs min / median / max | l2_relative min / median / max | Nonfinite absolute errors |
|---|---|---:|---|---|---:|
| dao | native-return | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| dao | supplied-output | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| new_same_mapping | native-return | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| new_same_mapping | supplied-output | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| new_screen_choice | native-return | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| new_screen_choice | supplied-output | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| old_original | native-return | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |
| old_original | supplied-output | 30 (10) | 0 / 0 / 0.00048828125 [missing=0, nonfinite=0] | 0 / 0 / 1.50258295e-07 [missing=0, nonfinite=0] | 0 |

`reference_records` binds all ten reference NPZ paths, SHA256 and per-array dtype/hash; each output record binds original and audited error artifacts. Zero-reference relative distributions with no elements retain null quantiles; this audit does not impute zeros. Sixty old/new same-mapping output pairs were bytewise equal; equality is continuity evidence, not an accuracy improvement. Finite outputs are required by the original protocol, while full floating error distributions remain report-only.

## Exact metric definitions and tails

For actual a and reference r, max absolute error is max_i |a_i-r_i|; relative L2 is ||a-r||₂ / ||r||₂. These are distinct from the maximum pointwise relative error max_{|r_i|>0} |a_i-r_i|/|r_i|; zero-reference absolute error is reported separately. FFT absolute values are complex magnitudes, not independent real/imaginary component errors. Main66 `scripts/run_semantic_campaign.py` computes the L2 ratio using rescaling to avoid overflow, returning null if undefined (zero reference with nonzero error); FP16 saved metrics cast arrays to complex128 for subtraction and use finite pairs only, returning null when reference norm is zero. No missing or nonfinite scalar is imputed as zero.

The nine FP16 DEFAULT per-request maximum pointwise relative errors range **1.32438424–16.5437394** for cuButterfly and **1.35351395–7.43303437** for cuFFT, despite relative-L2 ranges **0.00192599044–0.00259573531** and **0.00172909817–0.00198122134**, respectively. Thus a small global norm error cannot justify a small worst-element relative error claim. These maxima use recorded `relative_error_quantiles_nonzero_reference["1"]`; no additional error calculation was performed.

## Suggested paper wording (five sentences)

Numerical results are reported separately from performance and do not impose a new floating-point acceptance threshold. The frozen 66-request cohort retains per-request error distributions and comparator errors where available, but its FP32 FFT, FWHT and structured records use higher precision CPU references and therefore do not establish same-precision validation. The independent nine-request native FP16 cohort records finite outputs for cuButterfly and cuFFT and compares both with the same explicitly rounded CPU reference; their internal arithmetic need not coincide. A separate ten-request FWHT audit compares 240 saved outputs with an actual FP32 CPU transform, finding maximum absolute error 4.8828125e-4 and maximum L2-relative error 1.502582946e-7, while retaining the original FP64-reference diagnostics. These observations apply to the recorded inputs, mappings and builds; near-zero aggregate errors and bytewise continuity are not general correctness guarantees.

## Provenance and reproducibility

Recompute each table by grouping the named JSON records exactly as specified, using Python `statistics.median` on finite recorded scalars; count null/absent and nonfinite scalars separately. No element arrays were pooled or recomputed. Source SHA256 follows. Row-envelope hashes below bind full per-request metrics, comparator records, finite flags, reference dtype and NPZ identities.

- `results/full_current_20261025/measurement/results.json` — `ada9a9de24a5c506fab5653ce1844ca90d6fc50def926ab2c3e82495949e4824`
- `paper/generated/current_refresh/fp16_comparison.json` — `d8afb30a01ff81099bbe4255ac0fa1ccfa8829d77663ff06b9439424b1768a05`
- `paper/generated/fwht_native_repair_20261107.json` — `9e24fd62ace07eee7aae7fce07ea6d289fa4d59dda600a88b5b7f88ea7cfd6b5`

The [provenance manifest](../generated/numerical_scope_20261108.json) records all 66 row-envelope hashes relative to the measurement root; no reference was recomputed.

Metric-definition source: `results/a100_fp16_acceptance_20261105/post_logs/reference_metrics_import_before.py` — SHA256 `1a3d57794a213e441787e973675488f21a094a241dedae823329a7daa01627ba`; see `metrics`. Main66 definition: `scripts/run_semantic_campaign.py`, `compare` implementation around lines 273–305.
