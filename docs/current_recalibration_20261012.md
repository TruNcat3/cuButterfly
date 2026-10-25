# Current finite calibration and comparison

The [runner](../scripts/run_current_recalibration.py) calibrates and confirms the
[23 requested 1D cells](../config/current_recalibration_workloads.json), then
measures public AUTO calls against matching adapters. This is a finite current-build
experiment. It does not qualify the complete semantic inventory or all design points.

## Run

The build must contain a current shared `libcuntt.so`, all calibration probes and
SM80 targets, with `CUBUTTERFLY_BUILD_SHARED`, `CUBUTTERFLY_ENABLE_CUFFTDX`, and
`CUBUTTERFLY_ENABLE_XD_CUFFTDX` enabled. Select one exclusively available GPU UUID.
The runner fails on contention through the existing engines; it does not wait indefinitely.

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_current_recalibration.py \
  --build-dir build-e13-cufftdx \
  --output-dir results/current_recalibration_20261012_gpu1 \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --vkfft-binary /home/wt/git/cuButterfly/build-a100-vkfft/vkfft_bench \
  --fht-python /home/wt/yes/envs/cubutterfly/bin/python \
  --gpuntt-binary /tmp/gpuntt_merge_gap_bench_a100 \
  --dry-run
```

Remove `--dry-run` and add `--resume` to execute the prepared plan. A first actual
run can instead use a fresh directory without `--resume`. The optional `--include-xd`
runs the separate three-cell public xD calibration and strict DEFAULT user-cache replay
after the 1D comparison. Leave it disabled when xD is measured separately on another GPU.

## Protocol and gates

| Setting | Value |
| --- | ---: |
| Search workload rows | 23 |
| Screened candidates per row | 16 |
| Finalists per row | 2 |
| Initial stage template budget | 8 |
| Trials / warmup / repeat | 3 / 20 / 50 |
| Verified transforms per timed batch | 2 |
| Additional specialization budget | 1200 seconds |

The 23 rows retain the 19 installation cells, including the xor-zeta compatibility
alias, and add FP16 FWHT with FP32 accumulation at N32/N64 and B128/B4096.
`search-seconds=0` leaves time unrestricted while finite candidate and compilation
budgets bound the declared experiment. Initial stage calibration uses the full
acquisition protocol on an explicitly sampled eight-template population. Finalization
independently remeasures confirmed mappings and missing physical services, schedule,
pipeline and composition data using freshly measured hardware rates.

All search rows must complete before finalization. Finalization must release a
nonempty registry and pass public selector replay before comparison. Composition
readiness is checked; model warnings, missing services and limited overlap coverage
remain in `status.json`. Neither a model warning nor sampled services are relabeled
as complete inventory calibration.

Comparison reuses `run_research_acceptance.compare_cell`: each public `--auto-select`
call must match the confirmed winner's implementation, full mapping and runtime
fingerprint. Three paired trials alternate backend order. Compilation, plan creation
and reference checks remain outside kernel timing. Baselines use the existing
contract checks; unmatched precision, accumulation, direction, modulus or order
remain explicit gaps. In particular, the DAO adapter does not match FP16 FWHT with
FP32 accumulation; zeta and structured have no external adapter here. In-tree radix2
controls are labeled internal, not external libraries.

## Outputs and recovery

`manifest.json` records commands and build/script/workload/GPU identities.
`status.json` is updated atomically after each phase; `calibrate.log`, `finalize.log`
and `compare.log` preserve stdout and stderr. `initial/` and `finalized/` preserve
the engines' raw checkpoints and registries. `comparison.json` keeps raw trial
samples and mapping evidence; `comparison.csv` contains absolute median times,
reference/cuButterfly ratios, and explicit unavailable rows with blank timings.

Resume preserves completed compatible phases and comparison trials. A changed
build, shared library, source input, workload, GPU UUID or configured baseline
requires a new output directory. A zero process exit is insufficient: output
readiness gates must also pass. Failed phases preserve their logs and checkpoints.
Existing identity-keyed JIT compile modules remain reusable; only measurement
registries are isolated. The effective JIT cache environment is recorded.
The optional xD engine has no partial-run resume; preserve its failed artifacts
and run that finite experiment in a new output rather than importing timings.

The final status is `complete-finite-1d-with-gaps` or, with the optional xD phase,
`complete-finite-with-gaps`. Application attribution, broad xD coverage, and full
semantic inventory acceptance remain separate declared tasks. The wrapper does
not consume legacy fixed mappings from `a100_comprehensive_suite.json`.

## Execution on October 12

The shared SM80 build was rebuilt with cuFFTDx and xD cuFFTDx enabled;
`public_planner_consistency` passed. The 23-cell GPU1 campaign was started at
`results/current_recalibration_20261012_gpu1`. Its actual phase status is in
`status.json`; this launch record is not a completion claim.

The independent GPU2 xD run completed at
`results/current_recalibration_20261012_xd_gpu2/public_calibration.json`.
All six R2C/C2R plans reported `runtime_measured` during calibration and
`user_cache` during DEFAULT replay, with identical mappings. Its three
FP32 shape/batch pairs are 64x64/B8, 32x32/B64 and 8x8x8/B16. Complete
Python adapter timings, including allocation and exposed host submission
gaps, reached 0.732–0.854x PyTorch/cuFFT speed for these nine direction/pair
measurements. This is a measured adapter deficit, not a missing-cache result
or an isolated kernel-throughput comparison. Both implementations' CPU FP64
reference errors and all trials are retained; no new float threshold was set.
