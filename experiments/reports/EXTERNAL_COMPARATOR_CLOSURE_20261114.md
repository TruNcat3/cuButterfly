# External comparator checkpoint, 2026-11-14

This checkpoint closes the bounded validation and repair confirmations below.
It does not complete the broader external-comparator performance matrix. The
runtime evidence is in `/home/wt/git/cuButterfly/results`; this source checkout
retains the report and adapter contracts. No default installation or release
is established by this checkpoint.

All three cohorts identify NVIDIA A100 80GB PCIe, UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` (84974239744 bytes reported).
The device index is only a local selector. A100 40GB fallback observations and
future PRO6000 measurements remain separate cohorts.

## HadaCore: local, native and public validation

The [HadaCore report](HADACORE_LOCAL_VALIDATION_20261113.md) records 204 local
geometry validations: 68 supported requests, each with groups-per-block 1/2/4,
from a frozen 90-request inventory. The 22 unsupported requests remain explicit.
All validated local outputs matched pinned upstream output bitwise; Graph,
input preservation and buffer checks passed. Native controls completed 68/68,
and public AUTO under explicit `ALLOW_FP16_MMA` completed 68/68 with
`validated-public-selection`. These are separate acceptance populations; 204
must not be reported as the number of public requests.

The local group=1/upstream complete-call comparison reports a 1.508× geometric
mean across 68 requests (range 1.117–1.996×). The upstream call includes a
separate required output scale; the local provider fuses that scale into final
stores. This measures complete-call organization/fusion benefits and changed
intermediate traffic, not a pure dataflow or temporal-reuse speedup. Group=1
also won the post-hoc geometry comparison; increasing sequential reuse did not
establish a benefit. The FP16 MMA normalized-coefficient/final-scale contract
remains distinct from native per-stage FP16 rounding. No floating acceptance
threshold was introduced. See the linked report for binary hashes, timing,
resources, native checkpoint, paired profile and public completion paths.

## tcFFT: two calibrated paired confirmations

The runtime `tcfft_setup_ready_gpu2_20261114/summary.json`
and `tcfft_setup_ready_gpu2_20261114/REPORT.md`
record two frozen forward, unnormalized FP16 C2C requests. Public `DEFAULT`
replayed exact hardware/semantic calibrated mappings, rather than performing
a new search. Both use factor partition `[9,9]`, cuFFTDx block cores, 256
threads, EPT=8, factor columns=8, writer-aligned shared exchange, one slice and
no factor overlap. Prefetch depth is 0 at B4 and 1 at B32; the serialized
summary retains the other mapping differences.

| N | B | cuButterfly, µs | cuFFT, µs | tcFFT, µs | cuFFT/cuButterfly | tcFFT/cuButterfly |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262144 | 4 | 26.368 | 26.061 | 48.589 | 0.988× | 1.843× |
| 262144 | 32 | 146.381 | 157.030 | 293.734 | 1.073× | 2.007× |

All three backends completed plan recreation, input and buffer checks, output
lifetime and Graph replay checks. Floating error distributions are retained as
`error-report-only`, with no new threshold; execution validation does not imply
numerical acceptance against a newly chosen tolerance. Timing covers complete
calls, including tcFFT input reorder and output copy, with 10 warmups, 20 calls
per Graph and median of seven replays. Compilation, allocation, planning and
CPU checking are excluded. Setup readiness is enforced in plan creation without
a diagnostic synchronization flag. These two points do not establish whole-suite
speedups or a pure fixed-core organization effect.

## GPU-FFT: independent adapter validation

The runtime `comparator_extension_resume_20261114/gpufft/validation_summary.json`
indexes the 17 raw rows in
`comparator_extension_20261111/gpufft_validation/rows/`.
Thirteen rank-one FP32/FP64 requests completed direct execution and Graph replay,
with finite outputs and before/after guards. Their numerical diagnostics and
lossless error artifacts are present, with status `accuracy-review-required`:
no floating acceptance threshold was imposed. The recorded L2 relative error
ranges are about 1.16e-7–1.90e-7 for FP32 and 4.48e-16–6.83e-16 for FP64.

The other four requests are multidimensional and remain `missing-baseline`
with reason `requires rank-one power-of-two length in [2,2^28]`. This is the
local adapter support boundary; flattening them would change the requested FFT.
The summary's `complete` status means that all 17 requests have a validation
outcome, including these four unsupported requests. It does not mean 17 numerical
acceptances or completed paired comparisons. Static adapter metadata in raw
rows may retain the earlier preparation description; measured row and Graph
statuses supply the validation outcome.

The raw rows use `purpose=validation-only`. Their complete-call timing is an
adapter diagnostic without a matching cuButterfly measurement, so it cannot
enter paired speedup aggregates. Fresh compatible public-selection pairs and
the remaining extension requests are still required for the broader comparison.
Resume GPU measurements only with exclusive use and matching hardware, build,
semantic and timing provenance; preserve completed evidence and explicit gaps.

## Prepared continuation: 26 requests awaiting paired measurements

The new runtime `comparator_closure_20261114/fft17/preparation.json` and
`comparator_closure_20261114/fwht_mma9/preparation.json` contain 17 and nine
CPU-prepared requests, respectively. All 26 preparation records are `prepared`;
neither campaign had measurement rows at this checkpoint. Preparation does not
establish numerical acceptance, Graph validation or paired performance.

The FFT campaign preserves all 17 frozen FFT descriptors and schedules cuFFT,
VkFFT and GPU-FFT. Its main library, registry, hardware profile and module cache
come from the matched `a100_fp16_acceptance_20261105` runtime installation.
It does not reuse an older paired subset that excluded the four multidimensional
requests; GPU-FFT support gaps remain visible in the new population.

The nine FWHT requests explicitly set `arithmetic_policy=ALLOW_FP16_MMA` and
use `build/hadacore_public_20261113/libcuntt.so` with
`hadacore_tile_validation_full_20261113/profile-paired.json`. They schedule fresh
HadaCore complete-call pairs. Earlier STRICT FWHT diagnostics, including inverse
and in-place controls in the historical 11-request cohort, retain their original
identity. This preparation does not update those controls.

The [extension runner](../../scripts/run_comparator_extension.py) accepts optional
`runtime_bindings: {}` in a preparation config. Preparation fills this snapshot
with hashes of the selected libraries, registry/profile JSON inputs and provider
libraries referenced by those inputs. The two manifests bind eight and four
files. Loading prepared work checks the binding population and hashes before
CUDA initialization; publication rechecks them. A changed runtime requires a
compatible new preparation rather than silently consuming changed calibration.

The remaining work is to execute these 26 paired requests, retain unsupported
and failed outcomes, inspect numerical and selection diagnostics, then summarize
the resulting pairs. The [bounded watcher](../../scripts/watch_comparator_extension.py)
waits for exclusive GPU use and requires a timezone-qualified `--until` deadline.
It retries busy outcomes only within that bound and stops on execution or row
errors instead of retrying errors indefinitely. It writes resumable status and
logs, and stops its owned runner at the deadline. Existing validated rows remain
subject to the runner's identity checks. For example, from the source root:

```sh
# Set an authorized future deadline, including timezone, before running.
python scripts/watch_comparator_extension.py \
  --campaign /home/wt/git/cuButterfly/results/comparator_closure_20261114/fft17 \
  --campaign /home/wt/git/cuButterfly/results/comparator_closure_20261114/fwht_mma9 \
  --until "$COMPARATOR_UNTIL" \
  --state-dir /home/wt/git/cuButterfly/results/comparator_closure_20261114/watch
```

`COMPARATOR_UNTIL` must be an ISO-8601 timestamp with explicit timezone; choose
a fresh finite execution window for each invocation. GPU availability is checked
at execution time and is not inferred from this dated report.

## Verification of this checkpoint

This document was checked against the two tcFFT summary rows, all 17 GPU-FFT
raw rows and their derived summary, and the HadaCore report. Documentation
checks cover links and repository consistency; no GPU experiment was rerun for
this documentation checkpoint.
