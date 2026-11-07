# Native FP16 public acceptance, 2026-11-05

The finite acceptance completed on November 6 on A100 80GB PCIe
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`. `status.json` is `complete`,
with zero remaining tasks; `summary.json` is `complete-error-report-only`.
This covers nine forward C2C native FP16 requests, four semantic controls and
three prior-installation continuity cells. It does not rerun the original
66-request main cohort or establish complete FFT-L/BF16 coverage.

## Identity and completed execution

The library SHA256 is
`775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`;
research runtime identity is
`3f24ebf748992e13153782f818b5503a1438510555609a1b5291af913edb3de5:research`.
Earlier explicit diagnostic timings retain their original identities.
The [build receipt](../../results/a100_fp16_acceptance_20261105/build/identity_receipt.json)
records reused kernel objects and the refreshed registry identity.

Stage calibration completed 16 described configurations, 334 records and 330
measured points, with coverage and validation complete and no adapter gaps,
omissions or pending descriptions. The usable `calibrated-local-serial` model
has 94.444% top-1 accuracy across 72 valid multi-candidate holdout workloads;
whole-plan relative error is 0.712% median / 6.993% P90, and mean latency regret
is 1.00683. Independent stage-service error is 0.938% median / 6.986% P90.
Applicability is limited to explicit current mappings and validated measured intervals.

Production seed-only confirmation completed 12 workloads and 32 mappings,
promoting 32 records. This is bounded confirmation, not full-inventory discovery.
Nine public DEFAULT and nine public MEASURE calls completed with identical
mappings in every cell. The shared `cubutterfly` conda installation is
`installed-validated`: nine installed DEFAULT calls, four installed semantic
controls and all three legacy-installed FP64/FWHT continuity cells completed.
The parent-confirmed [post_completed receipts](../../results/a100_fp16_acceptance_20261105/post_completed/)
retain individual result paths and SHA256 hashes; uncommitted attempts are excluded.

## Public performance

Ratios are cuFFT latency divided by complete cuButterfly call latency.
The nine DEFAULT ratios have geometric mean **0.853583×**, range
**0.733471–1.066737×**, with **1/9** nominal wins. MEASURE and installed DEFAULT
have separate geometric means **0.853388×** and **0.855909×**, respectively.
The same mapping can have different measured latency; no minimum is taken across paths.
These descriptive observations do not establish statistical significance.

| N | Batch | DEFAULT μs | cuFFT μs | cuFFT / DEFAULT | MEASURE μs | Installed DEFAULT μs |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 256 | 12.851 | 9.984 | 0.776892× | 12.544 | 12.595 |
| 4096 | 2048 | 59.238 | 47.923 | 0.808989× | 59.341 | 59.290 |
| 8192 | 256 | 27.750 | 21.760 | 0.784133× | 27.802 | 27.802 |
| 16384 | 64 | 23.757 | 18.688 | 0.786638× | 24.013 | 23.603 |
| 16384 | 512 | 114.842 | 103.731 | 0.903255× | 114.688 | 114.995 |
| 32768 | 32 | 24.781 | 18.176 | 0.733471× | 24.678 | 24.678 |
| 32768 | 256 | 118.221 | 106.906 | 0.904288× | 118.221 | 118.067 |
| 262144 | 4 | 26.419 | 25.651 | 0.970930× | 26.931 | 26.522 |
| 262144 | 32 | 146.534 | 156.314 | 1.066737× | 146.278 | 146.483 |

Complete-call CUDA Graph timing uses 20 calls per graph, seven timed replays and
ten warmups; planning, JIT, CPU reference work and validation copies are outside
timing. The direct-launch stage/seed timings are separate protocols and are not
pooled into these graph results. All outputs are finite, with direct/Graph
byte equality and input/output guard checks retained. Floating errors remain
report-only, without a new acceptance threshold. Matching FP16 storage and FFT
semantics does not establish matching internal cuFFT arithmetic.

## Selected designs

The table reports the resolved public DEFAULT mapping, identical to MEASURE and
installed DEFAULT. C is logical independent transforms per CTA for the single
local core, and logical factor columns for factor execution; EPT is the local
provider's elements per thread. T is data tiles per CTA and D is prefetch depth.
The factor mapping's `stage_partition` is a logical layer descriptor; its two
physical launches follow `factor_partition`, shown below as tile lengths.

| N / batch | Physical stages / factor tiles | Threads per stage | C / EPT | T / D | Core |
|---|---|---|---|---|---|
| 4096 / 256 | 1 / 4096 | 256 | 2 / 16 | 1 / 0 | cuFFTDx FP16 half2 |
| 4096 / 2048 | 1 / 4096 | 256 | 2 / 16 | 1 / 0 | cuFFTDx FP16 half2 |
| 8192 / 256 | 1 / 8192 | 512 | 2 / 16 | 1 / 0 | cuFFTDx FP16 half2 |
| 16384 / 64 | 2 / 128 → 128 | 128 → 128 | 16 / 8 | 1 / 0 | cuFFTDx FP16 half2 |
| 16384 / 512 | 2 / 128 → 128 | 128 → 128 | 16 / 8 | 4 / 1 | cuFFTDx FP16 half2 |
| 32768 / 32 | 2 / 128 → 256 | 128 → 256 | 16 / 8 | 1 / 0 | cuFFTDx FP16 half2 |
| 32768 / 256 | 2 / 128 → 256 | 128 → 256 | 16 / 8 | 4 / 1 | cuFFTDx FP16 half2 |
| 262144 / 4 | 2 / 512 → 512 | 256 → 256 | 8 / 8 | 1 / 0 | cuFFTDx FP16 half2 |
| 262144 / 32 | 2 / 512 → 512 | 256 → 256 | 8 / 8 | 4 / 1 | cuFFTDx FP16 half2 |

## Repairs and limits

The first repair prevents false seed rejection when the cuFFTDx provider resolves
an accepted `radix2` compute-unit label to `auto`. Canonicalization preserves real
factor/core/layout axes; failed attempts remain archived and candidates were
remeasured. See [calibration_seeds.py](../../scripts/calibration_seeds.py) and
[its regression tests](../../tests/test_calibration_seeds.py).

The second repair removes the eager source-Python-frontend import from CPU
error-metric review: frontend access is now lazy, so importing CPU metrics cannot
select the source frontend before installed validation. See
[CPU review](../../benchmark/review_fft_l_cpu_reference.py) and
[CPU reference tests](../../tests/test_low_fft_cpu_reference.py).
The latter suite passed **12 tests** in a CPU-only verification on November 6.

BF16 forward/normalized inverse, FP16 normalized inverse and FP16 with FP32
accumulation are four semantic controls through existing scalar paths. BF16 has
semantic smoke evidence here; its complete automatic external comparison remains
pending. The new FP16 provider's normalized inverse remains outside this scope.
The one large-batch win does not erase the eight losses or establish full-matrix
leadership. Historical A100 main-cohort means remain unchanged.

## Source hashes

Hashes below bind this report to the inspected completion snapshot. Individual
public/installed/legacy result hashes are in `post_completed/*.json`.

| Source | SHA256 |
|---|---|
| [summary.json](../../results/a100_fp16_acceptance_20261105/summary.json) | `de7455ff4499dd7fe45fe77205490026ee48792f9c6384a870f213ba7b73825a` |
| [status.json](../../results/a100_fp16_acceptance_20261105/status.json) | `7a2eab6c2d01977121488bf388cfbc6268119c9bc43d402a838eda9fa7d99f90` |
| [build/identity_receipt.json](../../results/a100_fp16_acceptance_20261105/build/identity_receipt.json) | `338adc46cddaf59604138fc2b4a2b9718ba4a08ff54629a0764efbaa7775be4b` |
| [stage/checkpoint.json](../../results/a100_fp16_acceptance_20261105/stage/checkpoint.json) | `47fc468b9e5f2ab4f8142dd6976ad86ffbe54a80a067bf6d26198f95cb85420e` |
| [stage/model-receipt.json](../../results/a100_fp16_acceptance_20261105/stage/model-receipt.json) | `ab33efdf1fd7806f40e20d5c51ba49d37fb02d9179f0610e0b5caf10b6d9a8a9` |
| [tuning/search_coverage.json](../../results/a100_fp16_acceptance_20261105/tuning/search_coverage.json) | `9a4491308c2c79c3e41d8cf04c1b40e61f56af9b5ba3d357fa6545f777233fb8` |
| [tuning/receipt.json](../../results/a100_fp16_acceptance_20261105/tuning/receipt.json) | `9e5fb3967a1fe5b3560fd4d2d4866593b89bb750702d2945372f03870f6cc7ca` |
| [installation.json](../../results/a100_fp16_acceptance_20261105/installation.json) | `00d6b77e9ab50231261a48aff9e14df543f7c5ee778fdd32e88914ae7070791d` |
