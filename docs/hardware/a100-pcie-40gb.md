# NVIDIA A100 PCIe 40GB

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

This page is the target-specific runbook for the A100 PCIe 40GB card. The
saved staged migration is a historical partial checkpoint. This page makes no
current performance, winner or completion claim.

## Current build continuation

The shared build contains the v3 physical-stage implementation on 2026-09-15.
Its benchmark SHA256 is now
`27ea742ffcfb6b087d11af1626b17ed0eba15ab986b95b917c93742639f98279`.
The old 40GB checkpoint's timings cannot resume against that changed binary.
Its bandwidth/compute capability profile remains reusable. Once this exact
40GB GPU is idle, start a fresh output directory through the same unified entry:

```bash
CUDA_VISIBLE_DEVICES=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e \
CUBUTTERFLY_COMPILE_MODE=research \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/calibrate_hardware.py \
  --build-dir build-a100-cufftdx \
  --profile-dir results/staged_migration_20260915/a100-40gb \
  --output-dir results/staged_migration_20260915/a100-40gb-v3 \
  --search-workloads config/staged_model_validation_workloads.json \
  --search-only --search-budget 8 --seed-budget 2 --search-finalists 2 \
  --stage-calibration full --search-seconds 0 --compile-seconds 180 \
  --operator-trials 3 --operator-warmup 10 --operator-repeat 20 \
  --cost-model staged --search-strategy evolutionary
```

This command is prepared, not executed: the 40GB card remained occupied during
the 80GB experiment. It measures this card's scheduling curves independently;
80GB curves are not copied. After creating the new manifest, resume that new
manifest. The sections below preserve the old checkpoint's identities and
commands; those historical commands require its original matching binaries.

The current update has no 40GB GPU calibration or performance acceptance
result. A full stage run must produce `stage_service_profile.json` with
`validated: true` and `calibration_status: "complete"` before the migration
manifest can be called `complete`.

The stage adapter currently exposes the original launches for supported NTT
`baseline`, `tile256`, `hybrid2d`, `stage-pipeline`, and `shared-iterative`
physical groups. It does not substitute an alternate kernel. Low-precision
paths and selected multi-subgraph composite/overlap lowerings remain
unsupported pending a real group contract and exclusive-GPU validation.

## Identity

| Field | Value |
| --- | --- |
| Device model | NVIDIA A100-PCIE-40GB |
| Compute capability | sm80 (`8.0`) |
| CUDA memory reported | `42,285,268,992` bytes |
| Target UUID | `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e` |
| Result directory | `results/staged_migration_20260915/a100-40gb/` |
| Existing build | `build-a100-cufftdx` |
| Compile mode in checkpoint | `research` |

Use the UUID as the target selector on the current host:

```bash
export CUDA_VISIBLE_DEVICES=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e
```

The local GPU index is not an identity and must not be used to transfer this
page to another machine. Do not use an A100 80GB profile, checkpoint, timing row
or model-qualified result for this target. The cumulative registry may be
shared, but records must be filtered by target identity; an 80GB mapping can
only be a seed proposal for fresh 40GB correctness and timing checks.

## Dependencies and compile policy

The shared Python environment and current build are:

```bash
PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
BUILD_DIR=build-a100-cufftdx
RESULT_DIR=results/staged_migration_20260915/a100-40gb
PROFILE_DIR="$RESULT_DIR"
WORKLOADS=config/staged_model_validation_workloads.json
MANIFEST="$RESULT_DIR/migration_manifest.json"
COMPILE_MODE=research
```

The current checkpoint records `CUBUTTERFLY_COMPILE_MODE=research`; preserve
that value when resuming. It must not be inferred from the 80GB page or from an
older 40GB experiment.

Its migration protocol records `cost_model=staged`,
`search_strategy=evolutionary`, `stage_model_version=physical-stage-composition-v1`,
`cost_feature_version=physical-execution-groups-v2`,
`cost_projection_version=resolved-fft-core-axes-v2` and
`model_update_policy_version=correct-training-change-v1` in the search artifact.
Those were the checkpoint's versions. Source and build have since changed;
use the fresh-output command above for current-build work. These versions do
not certify the model or its ranking quality.

The old prepared 40GB matrix is available as
historical preparation evidence (local artifact: `../../results/framework_refactor_20260912/fft_acceptance_a100_40gb/prepared_matrix.json`; not included in this source release).
It is not current-build performance evidence and must not be promoted into the
new staged result directory. The earlier readiness record also states that
current-build 40GB calibration remained pending
(readiness note (local artifact: `../../results/framework_refactor_20260912/factor_partial_readiness_20260914.md`; not included in this source release)).

## Unified migration

The current checkpoint already has a target-specific migration manifest. Its
three configured cells are in `config/staged_model_validation_workloads.json`.
The following optional dry run restores the recorded paths and protocol and
writes a plan beside the existing checkpoint. For a new plan, choose a new
result directory:

```bash
GPU_UUID=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e
COMPILE_MODE=research
RESULT_DIR=results/staged_migration_20260915/a100-40gb
PROFILE_DIR="$RESULT_DIR"
WORKLOADS=config/staged_model_validation_workloads.json
MANIFEST="$RESULT_DIR/migration_manifest.json"

CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --resume-from "$MANIFEST" \
  --mode dry-run
```

For execution, rerun with `--mode execute`; dry-run is an optional preview. The
unified entry point owns the existing benchmark, correctness, promotion, staged
model and selector logic; this profile does not define another benchmark or a
fixed mapping choice.

Resume the exact partial migration checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --resume-from "$MANIFEST" \
  --search-strategy evolutionary --cost-model staged
```

## Bounded tuner

Once the migration manifest exists, use the tuner to add resumable search
rounds. It persists `tuner_state.json` beside the manifest and leaves a busy
GPU paused unless `--watch` is requested:

```bash
GPU_UUID=GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/tune_hardware.py \
  --resume-from "$MANIFEST" --gpu "$GPU_UUID" \
  --time-budget 600 --round-budget 4 --max-rounds 2
```

The existing `tuner_state.json` was created with this `600 / 4 / 2` protocol
and has next cumulative search budget 12. Keep those values on resume; the
tuner rejects changed round, max-round or total-time settings. For an
explicitly requested wait, add `--watch --poll-seconds 15`; the tuner never
kills another user's process. To inspect the next command without launching
calibration:

```bash
"$PYTHON" scripts/tune_hardware.py --resume-from "$MANIFEST" --dry-run
```

## Coverage and limits

The historical partial checkpoint contains a hardware-profile artifact but no
completed migration gate. The migration manifest is `failed` after partial
calibration/protocol interruption and remains resumable; it is not a
performance or completion result. It contains three configured cells:

| Cell | Partial state |
| --- | --- |
| FFT FP32, `logN=12`, batch 32 | 7 candidates screened; two search finalists confirmed; stopped by fair-share search time. |
| FFT FP32, `logN=18`, batch 16 | 3 candidates screened before a projection error; resume after the fix. |
| FWHT FP32, `logN=12`, batch 32 | No new search candidates in this checkpoint. |

The aggregate checkpoint contains 10 attempted rows, including valid and
unavailable outcomes. Exact counts and stop reasons are authoritative in
`search_coverage.json`; no latency, winner or model-quality conclusion should
be inferred from these partial rows.

When the checkpoint is resumed, update this section from the target's
`search_coverage.json` and manifest, separating:

- semantic cells that were searched and correctness-checked;
- confirmed finalists and promoted records;
- candidates rejected or unavailable at runtime;
- workload cells omitted by budget or deadline; and
- model validation and selector replay status.

Until that update, do not infer current 40GB FFT, NTT, FWHT, structured,
subset/superset, FP64, placement or full size/batch coverage from historical
files. The 40GB matrix preparation and prior NTT correctness notes establish
workflow context only, not a current staged performance gate.

## Evidence and result updates

The current evidence is recorded in the hardware profile (local artifact: `../../results/staged_migration_20260915/a100-40gb/hardware_profile.json`),
migration manifest (local artifact: `../../results/staged_migration_20260915/a100-40gb/migration_manifest.json`),
search coverage (local artifact: `../../results/staged_migration_20260915/a100-40gb/search_coverage.json`),
search measurements (local artifact: `../../results/staged_migration_20260915/a100-40gb/search_measurements.json`),
operator calibration (local artifact: `../../results/staged_migration_20260915/a100-40gb/operator_calibration.json`),
mapping seed snapshot (local artifact: `../../results/staged_migration_20260915/a100-40gb/mapping_seed_snapshot.json`)
and saved tuner state (local artifact: `../../results/staged_migration_20260915/a100-40gb/tuner_state.json`).
The historical prepared matrix (local artifact: `../../results/framework_refactor_20260912/fft_acceptance_a100_40gb/prepared_matrix.json`)
and pending-readiness note (local artifact: `../../results/framework_refactor_20260912/factor_partial_readiness_20260914.md`)
remain context only; neither is current staged performance evidence.

No current `cost_model.json`, selector replay or completion report exists for
this partial checkpoint. Link those artifacts after a successful resume.

For each update, preserve a fresh report and record the exact UUID, model,
memory identity, build/binary hashes, compile mode, workload hash, command,
protocol versions, coverage, warnings and replay status. Keep all 40GB evidence
in this target directory and never combine it with the 80GB artifacts.
