# Current Retest Campaign: 2026-10-06

This is a prepared campaign, not a result report. The preparation script did
not query CUDA, launch a benchmark, or import timing rows. Every performance
output must be written below the new
`results/recalibrated_campaign_20261006/` directory.

The executable plan is [manifest.json](../results/recalibrated_campaign_20261006/manifest.json).
It contains the copied finite gap matrix, a fresh rank-1 comparison manifest,
the catalog-backed external workload manifest, source/build fingerprints,
exact commands, and output gates. Regenerate it with:

```bash
python3 scripts/prepare_recalibrated_campaign.py
```

## Scope And Counts

The seven named families are the six public arithmetic operators plus the
`xor-zeta` compatibility spelling. `xor-zeta` inherits `subset-zeta` evidence;
it is not counted as an additional kernel.

| Population | Arithmetic rows | Alias rows | Semantic rows | Notes |
| --- | ---: | ---: | ---: | --- |
| Existing finite gap matrix | 113 | 0 | 113 | Exact snapshot of `operator_coverage_gap_20261006/manifest.json`; no timing fields |
| xD `coverage` profile | 480 | 48 | 528 | Four rank-2/3 shapes, batches 1/4/16, forward/inverse, selected + resident-t128 |
| xD `scale` profile | 240 | 24 | 264 | Four larger rank-2/3 shapes, batches 1/4/16, forward/inverse, selected |
| Rank-1 comparison suite | 135 cases | included | 135 | 98 standard cases plus all 37 historical semantic cases regenerated fresh |
| Rank-1 AUTO search input | 43 | 0 exact hits / 43 unmeasured-feasible | future per-cell input | Deduplicated current butterfly AUTO rows only; no search is started here |
| Exact anchor commands | 30 | 0 | 30 | 21 NTT modulus/order anchors plus 9 large-N/batch anchors |
| Catalog external comparison | 8 cases | 0 | 8 | A100 finite Dao-FHT/GPU-NTT workload matrix; fresh CSV only |

The 113-row gap snapshot is partitioned as follows. Rows marked
`missing-public-evidence` remain pending until the same-build runner produces a
public result; they are not silently promoted from low-level tests.

| Section | Rows | Initial state |
| --- | ---: | --- |
| low-precision public API | 48 | pending; public C-ABI evidence gap |
| real public API | 16 | pending; FP32/FP64 R2C/C2R |
| NTT semantic anchors | 21 | pending; fixed moduli, natural and legal bit-reversed output |
| rank/layout | 12 | pending; rank-2/3 strided OOP and packed in-place |
| large N/batch | 9 | pending; exact logN/batch anchors |
| layout boundaries | 3 | pending; packed-only contract |
| unsupported real low precision | 4 | `unsupported` negative cases |

The standard rank-1 cohort has 30 FFT, 24 FWHT, 24 structured-2x2, 8 NTT,
4 subset-zeta, 4 superset-zeta, and 4 xor-zeta cases. It includes FP32/FP64,
FP16/BF16 native and FP32 accumulation where the existing benchmark CLI
exposes those contracts, both directions, `N=2^8`/batch 1 and `N=2^12`/batch
16. The additional `historical-37-case` cohort re-expresses every case in
`config/a100_comprehensive_suite.json` with current binaries: 19 FFT, 7 FWHT,
5 xor-zeta, and 6 NTT cases. It retains `logN=3,14,15,16,18,20`, the inverse
rows, the FP16-FP32 row, and the `element_stride=2,batch_padding=7` stride
contract. Butterfly and NTT rows use their public AUTO selection contracts;
external reference rows retain their runner. NTT word width, direction, input
order, output order and exact anchor modulus remain explicit. The old
JSON is read for semantics only, and no old timing/result file is read. Exact
large-N/batch gap points remain separate anchor commands.

The catalog-backed external manifest is
`external_suite.json`. It expands the A100 finite workload matrix
into 8 fresh cases: Dao-FHT at FWHT `logN=8,batch=16384` and
`logN=15,batch=128`, plus GPU-NTT at `logN=16,batch=64` natural output and
`logN=20,batch=4` bit-reversed output. The adapter writes the existing raw CSV
schema (`scripts/run_external_baseline_suite.py`), and the existing summarizer
writes `comparison/external_summary.csv` and `.md`. Current
`cubutterfly_bench`/`cuntt_bench` are passed explicitly as the paired source
binaries; current NTT rows use `cuntt_bench --auto-select` with the workload's
word width and input/output order.

The generated `rank1_search_workloads.json` is a calibration input, not a
calibration result. The planner does not have a registry result for these
fresh rows, so every entry starts as `exact-hit=false,
status=unmeasured-feasible`. Registry matching must hit exact `logN`, `batch`,
and semantic fields; the 14-cell service calibration therefore must not be
described as coverage of all 98 standard comparison cases. External rows,
explicit low-precision mappings, alias rows, and NTT rows are intentionally
excluded from this butterfly AUTO search input; NTT AUTO rows and fixed-modulus
anchors remain separate semantic contracts.

## Identities And Prerequisites

The target build is `build-e13-cufftdx`: Release, CUDA architecture `80`,
`CUBUTTERFLY_ENABLE_CUFFTDX=ON`, and `CUBUTTERFLY_ENABLE_XD_CUFFTDX=ON`.
The legacy baseline is `build-a100-vkfft`: Release, architecture `80`, with
cuFFTDx disabled. SHA256 values for the source inputs and every binary are
recorded in the manifest; the current `libcuntt.so` hash is also injected by
the xD runner at execution time.

At preparation time these current-build binaries exist: `cubutterfly_bench`,
`cuntt_bench`, `libcuntt.so`, `cubutterfly_tests`,
`cubutterfly_xd_fft_layout_tests`, and `cubutterfly_rank3_permute_tests`.
The following targets are configured but not built in the checkout and must be
created by the prerequisite command: `cubutterfly_plan_bench`,
`cubutterfly_c_api_tests`, `cuntt_tests`, `cubutterfly_public_planner_tests`,
and `cubutterfly_mapping_tests`. The legacy `build-a100-vkfft/vkfft_bench`
exists and is the only legacy external binary required.

The catalog runners are also recorded instead of being labelled as absent:
the local Dao adapter uses
`/home/wt/yes/envs/cubutterfly/bin/python` and its installed
`fast_hadamard_transform` module; the local GPU-NTT adapter is
`/tmp/gpuntt_merge_gap_bench_a100`. The preparation step only records
executable/module metadata and hashes. It does not import CUDA or invoke either
external tool. If either path is missing on a later machine, the manifest marks
that task `unavailable` and does not invent a baseline.

Build the current targets before entering the GPU phases:

```bash
cmake --build build-e13-cufftdx --target \
  cubutterfly_c_api_tests cubutterfly_tests cuntt_tests \
  cubutterfly_xd_fft_layout_tests cubutterfly_rank3_permute_tests \
  cubutterfly_public_planner_tests cubutterfly_mapping_tests \
  cubutterfly_plan_bench -j2
```

Record a target-specific UUID and check exclusive use/free memory before any
GPU command. The runbook intentionally does not assume that a local index is a
stable identity:

```bash
export GPU_UUID=GPU-<target-uuid>
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
nvidia-smi --id="$GPU_UUID" \
  --query-gpu=uuid,name,memory.total,memory.used,compute_cap \
  --format=csv
nvidia-smi --id="$GPU_UUID" --query-compute-apps=pid,process_name,used_memory \
  --format=csv
```

## Ordered Execution

The manifest stores the exact command strings. They are intentionally ordered
as correctness, isolated large-memory, profile gate, and comparison. `CUBUTTERFLY_COMPILE_MODE` must
remain `research` for all current-build runs.

1. Correctness: run the current CTest operator/C-API/xD/rank/mapping checks,
   `tests/check_xd.py` against `build-e13-cufftdx/libcuntt.so`, the small
   `check_operator_coverage_gaps.py --execute` phase, the compact-real
   reference harness, and the four explicit unsupported FP16/BF16 real cases.
   The low-precision C++ contract remains low-level evidence; it does not
   certify a public C-ABI low-precision row.

2. Large memory: run the isolated `large_memory` manifest phase with
   `--section large_n_batch --include-large --memory-limit-bytes 68719476736`
   on the target A100 80GB (64 GiB IO/workspace allowance).
   Allocation failures are recorded as `memory-limit` and cannot block the
   small correctness phase.

3. Profile gate: run `scripts/benchmark_all_operators_xd.py` twice with the
   current plan bench and library. The coverage run requests selected and
   `resident-t128` with `--policy measure`; the scale run requests selected
   only. Then run `check_gpu_operator_matrix.py --input-run` for each `rows.json`
   (with `--include-resident` only for coverage) for independent public C-ABI
   value checks. Use the supplied strict summarizer for each output directory.
   `plan_bench --verify` is only a timing-side verification hook for supported
   FFT forward rows; it is not a proof of all operator/direction rows or of
   mapping identity.

4. Comparison: run the generated rank-1 suite with
   `scripts/run_comprehensive_suite.py`, summarize it with
   `scripts/summarize_comprehensive_suite.py`, run the catalog-backed external
   suite and `scripts/summarize_external_baseline_suite.py`, then run the fresh
   xD and compact-real comparison harnesses. The 21 fixed-modulus NTT anchors
   and the nine large-N/batch anchors are emitted as separate fresh CSV
   commands. The final comparison command is
   `python3 scripts/summarize_recalibrated_campaign.py --campaign
   results/recalibrated_campaign_20261006`.

The principal commands are:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" ctest --test-dir build-e13-cufftdx \
  --output-on-failure \
  -R 'cubutterfly_correctness|cuntt_correctness|cubutterfly_c_api|xd_fft_.*_correctness|rank3_permute_correctness|public_planner_consistency|mapping_record_roundtrip'

CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 tests/check_xd.py \
  --library build-e13-cufftdx/libcuntt.so \
  --output results/recalibrated_campaign_20261006/correctness/check_xd.json

CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 scripts/check_operator_coverage_gaps.py \
  --manifest results/recalibrated_campaign_20261006/gap_matrix.json \
  --library build-e13-cufftdx/libcuntt.so --execute \
  --output results/recalibrated_campaign_20261006/correctness/gap_public_cabi_small.json

CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 scripts/check_operator_coverage_gaps.py \
  --manifest results/recalibrated_campaign_20261006/gap_matrix.json \
  --library build-e13-cufftdx/libcuntt.so --section large_n_batch \
  --include-large --memory-limit-bytes 1073741824 --execute \
  --output results/recalibrated_campaign_20261006/correctness/gap_public_cabi_large.json

CUDA_VISIBLE_DEVICES="$GPU_UUID" CUBUTTERFLY_COMPILE_MODE=research \
  python3 scripts/benchmark_all_operators_xd.py \
  --bench build-e13-cufftdx/cubutterfly_plan_bench \
  --library build-e13-cufftdx/libcuntt.so \
  --output results/recalibrated_campaign_20261006/profile/xd_coverage \
  --profile coverage --policy measure --compare-cufft --verify \
  --gpu-uuid "$GPU_UUID" --compile-mode research \
  --warmup 10 --repeat 50 --trials 3 --modulus 257 \
  --matrix 1,0.25,-0.5,1 --keep-logs

CUDA_VISIBLE_DEVICES="$GPU_UUID" CUBUTTERFLY_COMPILE_MODE=research \
  python3 scripts/check_gpu_operator_matrix.py \
  --library build-e13-cufftdx/libcuntt.so --profile coverage \
  --include-resident \
  --input-run results/recalibrated_campaign_20261006/profile/xd_coverage/rows.json \
  --compile-mode research --modulus 257 --matrix 1,0.25,-0.5,1 \
  --output results/recalibrated_campaign_20261006/profile/xd_coverage_correctness.json

CUDA_VISIBLE_DEVICES="$GPU_UUID" CUBUTTERFLY_COMPILE_MODE=research \
  python3 scripts/check_gpu_operator_matrix.py \
  --library build-e13-cufftdx/libcuntt.so --profile scale \
  --input-run results/recalibrated_campaign_20261006/profile/xd_scale/rows.json \
  --compile-mode research --modulus 257 --matrix 1,0.25,-0.5,1 \
  --output results/recalibrated_campaign_20261006/profile/xd_scale_correctness.json

CUDA_VISIBLE_DEVICES="$GPU_UUID" CUBUTTERFLY_COMPILE_MODE=research \
  python3 scripts/run_comprehensive_suite.py \
  --manifest results/recalibrated_campaign_20261006/comparison_suite.json \
  --mode full \
  --output results/recalibrated_campaign_20261006/comparison/rank1_raw.csv \
  --resume --require-exclusive-gpu

CUDA_VISIBLE_DEVICES="$GPU_UUID" python3 scripts/run_external_baseline_suite.py \
  --manifest results/recalibrated_campaign_20261006/external_suite.json \
  --output results/recalibrated_campaign_20261006/comparison/external_raw.csv \
  --fht-python /home/wt/yes/envs/cubutterfly/bin/python \
  --gpuntt-binary /tmp/gpuntt_merge_gap_bench_a100 \
  --butterfly-binary build-e13-cufftdx/cubutterfly_bench \
  --ntt-binary build-e13-cufftdx/cuntt_bench --resume --require-exclusive-gpu

python3 scripts/summarize_external_baseline_suite.py \
  results/recalibrated_campaign_20261006/comparison/external_raw.csv \
  --output results/recalibrated_campaign_20261006/comparison/external_summary.csv \
  --markdown results/recalibrated_campaign_20261006/comparison/external_summary.md
```

The `scale` command, real-FFT commands, summarizers, and all fixed-anchor
commands are recorded in `manifest.json`; copy them from there rather than
substituting an older result directory. A fresh output directory is required
even when resuming an interrupted current campaign. The external raw CSV uses
the existing runner schema and is summarized before the final campaign
aggregate; no historical external CSV is an input.

## Baselines And Mapping Rules

| Workload | Comparator | Availability |
| --- | --- | --- |
| FFT C2C FP32 rank-1 | cuFFT and VkFFT | fresh runs through `build-a100-vkfft/vkfft_bench` |
| FFT C2C FP64 rank-1 | cuFFT | fresh current `cubutterfly_bench --backend cufft` rows |
| compact R2C/C2R FP32/FP64 | PyTorch/cuFFT | fresh `benchmark_xd_real.py` rows |
| xD six arithmetic families | resident-t128 control | fresh selected/resident rows from the current library |
| FWHT FP32 | Dao-AILab FHT | catalog runner `scripts/benchmark_external_fht.py`; fresh task uses the local conda module |
| NTT uint64 fixed prime | GPU-NTT Merge | catalog runner `scripts/run_external_baseline_suite.py`; natural and bit-reversed rows use the local executable |
| zeta, structured external baselines | no matching catalog runner | `missing-external-reference`; internal selected/resident controls still run |
| xor-zeta | subset-zeta alias | inherited semantic evidence; no second kernel claim |

Every selected xD row must contain the measured selected algorithm and preserve
the runner's selection metadata (including serialized `mapping_json` when that
runner emits it). Do not infer an unmeasured mapping from a selector default.
Every rank-1 `--auto-select` row must retain the observed
`selected_implementation`, `selection_confidence`, and `selection_reason`.
An unmeasured selector default is metadata, not a performance result. A
resident row counts as a control only when the reported algorithm actually
contains `resident-t128`; fallback to another algorithm is `unavailable`.

## Result Classification

Do not collapse all non-pass outcomes into one failure count.

| Result state | Rule |
| --- | --- |
| `passed` | Same-source correctness passed and the required timing/mapping fields exist |
| `failed` | Execution, parsing, semantic validation, or correctness failure |
| `unsupported` | Explicit implementation rejection or a passing negative contract test |
| `missing-external-reference` | The row can be checked internally, but the requested external comparator does not exist |
| `memory-limit` | Allocation/resource exhaustion, including `cudaErrorMemoryAllocation` and equivalent messages |
| `unverified` | Timing exists without correctness; never promote it to pass |

The large anchors are particularly important: an allocation failure at
`logN=23,batch=256` or another declared resource boundary is
`memory-limit`, not `unsupported`, `failed`, or `passed`. The final tables
should report counts and medians separately for these states, then show only
verified rows in speedup/winner tables.

## Recommendation

Build the five missing current targets, verify GPU identity/exclusivity, and
execute the manifest phases in order. Use the generated summaries as the only
source for current performance tables. The 37-case historical cohort is now a
fresh current-binary run and belongs in the rank-1 output; only the old
`comprehensive_current_20261006_cufftdx` timing files and old all-operator
`rows.json` remain forbidden inputs. The latter remain provenance for matrix
shape only, not current timing evidence.
