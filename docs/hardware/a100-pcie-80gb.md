# NVIDIA A100 PCIe 80GB

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

This page is the target-specific runbook for the A100 PCIe 80GB card. Current
and historical bounded experiments are separated below. Neither qualifies the
full application or FFT matrix.

On 2026-09-17 the original GPU 2 remains occupied, but GPU 1 of the same model
and capacity is available. Its independent calibration (local artifact: `../../results/staged_migration_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
uses UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` and fresh target-local
measurements. The stage basis and scheduling probes have completed; its
72-cell acceptance (local artifact: `../../results/comprehensive_20260917/a100-80gb-gpu1/report.md`; not included in this source release)
is running. The older GPU 2 acceptance below is preserved separately.

The current [72-cell comprehensive acceptance](../research_acceptance.md) is
interrupted at the GPU exclusivity guard. It uses a single resumable
search/calibration/replay/baseline entry.
The `acceptance.json`/`summary.json` snapshot records 5/72 complete cells.
Four FFT cases have geometric-mean throughput ratios of 0.97355 against cuFFT
and 0.88684 against VkFFT; the large N=262144/batch=16 case exceeds both,
while small-batch N=256 remains slower. One FWHT case is also complete; see
the run report (local artifact: `../../results/comprehensive_20260916/a100-80gb/report.md`; not included in this source release)
for its interface timing caveat and per-case evidence.

The probe-point type repair added one service request and grew the cumulative
curve inventory from 78 to 79 with `service_gaps=[]`; ranking improved from
top-1 false / pairwise `1/3` / regret `1.2115735x` to top-1 true / pairwise
`3/3` / regret `1.0x`. The old ranking and gaps remain in the journal's
`*_before_probe_type_fix` fields. The sixth cell retains 8/19 screening attempts;
the full 72-cell matrix and full migration are not complete. Resume when the
target is exclusive; use the linked
continuation command; do not reinstall or discard compatible stage records.

The offline module preparation (local artifact: `../../results/comprehensive_20260916/a100-80gb-precompile/report.md`; not included in this source release)
exports the same finite library inventory and compiles with CUDA devices hidden.
The complete 72-cell export has 33,690 deduplicated requests. The CPU queue has
finished with 33,572 successes and 118 unsupported FP64 register-tile cuFFTDx
configurations (`suffix_ept=32`); it is no longer running. Eleven CPU checks and one actual prepared-module GPU reuse
check pass. A CPU-only audit additionally validates 32 prepared module
manifests/hashes and 101 existing screening request projections without a
mismatch. Compilation completion remains separate from GPU calibration and
measured acceptance. The export/compile commands are in the acceptance runbook.

The shared conda helper also includes exact-value schedule-result caching for
large-batch CPU prediction. The bounded 12-point repeated diagnostic is 1.94x
faster with identical predictions, and all 42 affected CPU tests pass. It does
not change calibration, model equations or runtime kernels and requires no
library rebuild; it does not establish a GPU performance improvement.

## Current physical-stage implementation

The latest composition/reuse report (local artifact: `../../results/staged_migration_20260916/a100-80gb-composition/report.md`; not included in this source release)
adds measured BatchPipeline scheduling, actual full/tail service projections
and two-group OnlineReorder curve reuse. All 27 pipeline plans have measured
component services and pass public correctness, but FWHT composition error
still reaches 40.287%; concurrency and full migration remain unqualified.
Four new FFT combinations reuse curves without new stage timings; their
six-plan public diagnostic has median/max prediction error 0.196%/0.948%.
The cumulative `a100-80gb-online-reuse/stage_calibration.json` has 695 records,
78 curves and 342/342 covered stage holdouts. The shared conda environment
includes the new probe and helper modules.

The 2026-09-16 reuse report (local artifact: `../../results/staged_migration_20260916/a100-80gb-stage-reuse/report.md`; not included in this source release)
adds 72 correct source measurements to the unchanged 585 earlier records.
FFT and NTT four-stage recombinations reused measured physical curves;
six public plans passed with median/P90 prediction error 1.234%/1.900%.
This is a serial reuse diagnostic; full migration and external baselines remain
unfinished. Installation now accepts repeated `--stage-import-checkpoint`
arguments for strictly compatible earlier measurements. Keep both composition
and acquisition-inventory sidecars with a moved checkpoint.

The current target is `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`, model
`NVIDIA A100 80GB PCIe`, sm80, CUDA memory `84974239744` bytes. The shared
environment is `/home/wt/yes/envs/cubutterfly`; the research build is
`build-a100-cufftdx` with `CUBUTTERFLY_COMPILE_MODE=research`.

The current GPU report (local artifact: `../../results/staged_migration_20260915/a100-80gb-v3-stage/report.md`; not included in this source release)
records passing stage correctness checks and a six-design serial calibration:
17/17 service holdouts are covered (median/P90 error 1.710%/8.723%), and public
plan predictions have median/P90 error 0.166%/0.358%. No external library
baseline or full migration qualification follows from this diagnostic.
The default14 shared-stage basis also passed: 549 correct stage records,
268/268 covered holdouts, and all 28 original public-plan requests covered.
Whole-plan median/P90 error is 1.098%/4.508%, with 13/14 within-cell winner
matches. The report distinguishes this basis from full inventory search.
Files under `results/staged_migration_20260915/a100-80gb-v3/` remain prior-build
evidence.

The first full inventory attempt under `a100-80gb-v3-stage` was stopped during
description and predates the corrected probe binary. Preserve that directory;
start a fresh full inventory run in a new target-local directory (or inspect it
first with `--mode dry-run`). The command below imports the compatible current
695-record checkpoint; it preserves all earlier diagnostic measurements.
The source remains a serial subset, not a full migration certificate.

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=research \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/calibrate_hardware.py \
  --build-dir build-a100-cufftdx \
  --profile-dir results/staged_migration_20260915/a100-80gb-stage-full \
  --output-dir results/staged_migration_20260915/a100-80gb-stage-full \
  --search-workloads config/staged_model_validation_workloads.json \
  --stage-import-checkpoint results/staged_migration_20260916/a100-80gb-online-reuse/stage_calibration.json \
  --stage-calibration full --search-seconds 0 \
  --cost-model staged --mode dry-run
```

The dry run above only writes a plan. Execute it only after an exclusive GPU
check succeeds. A full stage run is complete only when its
`stage_service_profile.json` contains `validated: true` and
`calibration_status: "complete"`, and the migration manifest reports
`complete`. A bounded run may intentionally leave `incomplete-*` stage status
and is not a performance pass.

The stage adapter currently exposes the original launches for supported NTT
`baseline`, `tile256`, `hybrid2d`, `stage-pipeline`, and `shared-iterative`
physical groups. It does not substitute an alternate kernel. Low-precision
paths and selected multi-subgraph composite/overlap lowerings remain
unsupported pending a real group contract and exclusive-GPU validation.

## Historical auto-policy recovery

The remaining sections preserve the earlier recovery's exact identities and
commands. Its old binary was archived under
`results/staged_migration_20260915/v2_binaries/`. The current build no longer
matches that historical manifest; do not execute the old resume commands
against the rebuilt directory or combine their timing rows with the v3 run.

## Identity

| Field | Value |
| --- | --- |
| Device model | NVIDIA A100 80GB PCIe |
| Compute capability | sm80 (`8.0`) |
| CUDA memory reported | `84,974,239,744` bytes |
| Target UUID | `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` |
| Build | `build-a100-cufftdx` |
| Compile mode | `auto` |
| Recovery result directory | `results/unified_migration_20260914/seed_recovery/model_feedback_recovery/` |

Use the UUID when selecting this device on the current host. A local CUDA index
is only a host-specific selector:

```bash
export CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6
export CUBUTTERFLY_COMPILE_MODE=auto
```

The 80GB profile, workload file and checkpoints are not interchangeable with
the A100 40GB page. The cumulative registry may contain both targets, but its
records must be filtered by target identity; a 40GB mapping or timing is never
an 80GB qualification without fresh target-side checks.

## Dependencies and compile policy

Use the shared environment and current research build:

```bash
PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
BUILD_DIR=build-a100-cufftdx
GPU_UUID=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6
COMPILE_MODE=auto
RESULT_DIR=results/unified_migration_20260914/seed_recovery/model_feedback_recovery
PROFILE_DIR=results/unified_migration_20260914/NVIDIA-A100-80GB-PCIe-sm80-84974239744B
MANIFEST="$RESULT_DIR/migration_manifest.json"
```

The recovery manifest records the workload file and protocol. Its benchmark
SHA256 is
`9b0fc930a849b5e82e04b3e85ce5ee7578db18c20273c15e3b17969cbd99b8e9`; the
installed library hash is
`00908a56615ec151d50a8cbc471d32ec9a213dacaded76f6dae4488dc16c1e49`.
Keep these identities and the `auto` compile mode fixed for a direct resume.

## Unified migration and resume

The exact historical recovery checkpoint can be resumed through the unified
entry. Omitting strategy/model overrides preserves its recorded measurement
protocol and restores its build, profile, output, workload, seed snapshot,
registry and compile policy:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --resume-from "$MANIFEST"
```

Adding `--search-strategy evolutionary --cost-model staged` intentionally
starts an upgraded staged/evolutionary continuation. It may reuse compatible
measured trials, but it changes the search/model protocol and must not be
described as an exact reproduction of this historical run. Do not replace
`--resume-from` with a fresh 40GB profile or manually copied paths. A dry run
of a new target is optional and must use that target's own profile and output
directory. The unified entry remains the only benchmark, correctness,
promotion, staged-model and selector-replay owner.

## Bounded tuner

To add resumable evolutionary rounds to this manifest, run:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/tune_hardware.py \
  --resume-from "$MANIFEST" --gpu "$GPU_UUID" \
  --time-budget 1800 --round-budget 16 --max-rounds 4
```

The tuner calls the same unified calibrator with
`--search-strategy evolutionary --cost-model staged`, increases the cumulative
search budget per round, and stores progress in
`$RESULT_DIR/tuner_state.json`. A busy GPU pauses by default. Add
`--watch --poll-seconds 15` only when waiting is intended; polling is bounded by
the remaining wall budget and does not terminate other processes. Inspect the
next command without launching it with:

```bash
"$PYTHON" scripts/tune_hardware.py --resume-from "$MANIFEST" --dry-run
```

## Measured scope

The linked recovery report covers exactly three semantic cells:

| Operator | Precision | Direction | Placement | Size | Batch |
| --- | --- | --- | --- | ---: | ---: |
| FFT | FP32 | forward | in-place | `2^18` | 16 |
| FFT | FP32 | forward | in-place | `2^20` | 4 |
| FFT | FP32 | forward | in-place | `2^20` | 16 |

The run recorded 44 search attempts, 38 correct candidate records, six runtime
rejections and nine confirmed records, with all three requested search
protocols complete. Those counts describe this bounded recovery only. The
selected mappings match the earlier paired experiment, whose reported
cuFFT/cuButterfly throughput ratios for the three cells were `1.06220`,
`1.00106` and `0.985993`; they are prior paired measurements, not new timings
from this model-feedback run.

The migration status remains `incomplete` because the local model validation
gate is warning. The report gives corrected holdout top-1 accuracy `0.5` over
two eligible multi-design workload groups and mean latency regret `1.241687x`.
The broader historical 70-row refit remains warning as well. A valid measured
mapping is not evidence that the model ranks every unmeasured mapping.

## Uncovered scope

This result does not cover other FFT sizes, batches, directions, placements or
precisions; other operators such as NTT, FWHT, structured and zeta transforms;
or a full application matrix. It is not an exhaustive search of the exported
candidate inventories. The model warning, budget omissions and runtime
rejections must remain visible in any result update.

No mapping parameters are prescribed in this page. Candidate mappings must be
correctness-checked and repeatedly timed on this exact target/build/workload
identity before promotion.

## Evidence

- bounded model-feedback report (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/report.md`; not included in this source release)
- migration manifest (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/migration_manifest.json`; not included in this source release)
- search coverage (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/search_coverage.json`; not included in this source release)
- search measurements (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/search_measurements.json`; not included in this source release)
- cost model and validation (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/cost_model.json`; not included in this source release)
- selector replay (local artifact: `../../results/unified_migration_20260914/seed_recovery/model_feedback_recovery/selector_replay.json`; not included in this source release)
- paired FFT summary (local artifact: `../../results/unified_migration_20260914/seed_recovery/paired_fft/summary.json`; not included in this source release)
- historical mapping replay (local artifact: `../../results/unified_migration_20260914/historical_mapping_replay/report.md`; not included in this source release)

The [general hardware installation guide](../hardware_profile_install.md)
defines the manifest status and checkpoint contracts. The report also links
the frozen seed snapshot and target-local measurement identity.

## Result update process

For a new 80GB run, preserve this evidence and use a new result directory when
changing the workload matrix, seed source, candidate projection, model policy,
binary or compile policy. Update the manifest-derived identity and protocol
table first, then coverage, rejection/omission counts, model validation and
selector replay. Record exact command lines and artifact hashes in the report.

Only claim `complete` when the migration manifest and required gates report
complete. A warning model with verified measured mappings is a valid
`incomplete` result, not a reason to copy old timing rows into a new model or
to mix this profile with the A100 40GB workflow.
