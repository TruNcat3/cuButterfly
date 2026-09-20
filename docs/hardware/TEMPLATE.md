# <GPU model and form factor> <memory capacity>

This page describes one hardware model and one memory capacity. It is a
runbook, not a repository of fixed mapping choices. Replace every angle-
bracket placeholder before publishing a profile.

## Identity

| Field | Value |
| --- | --- |
| Device model | `<nvidia-smi name>` |
| Memory capacity | `<nominal capacity>` |
| Compute capability | `<sm version>` |
| Target UUID | `<GPU UUID>` |
| Current result directory | `<relative results path>` |
| Profile directory | `<relative or absolute profile path>` |
| Compile mode | `<recorded compile mode>` |

Use the UUID in `CUDA_VISIBLE_DEVICES` for this host. A CUDA index is not a
portable hardware identity. Do not copy capability profiles, checkpoints or
timing qualifications from another memory capacity. A shared registry keeps
records for multiple targets and filters by identity; portable mappings may be
imported as seeds and remeasured locally.

## Dependencies and build

```bash
PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
BUILD_DIR=<build directory>
GPU_UUID=<target UUID>
COMPILE_MODE=<recorded compile mode>
RESULT_DIR=<result directory>
PROFILE_DIR=<profile directory>
WORKLOADS=<workload file>
MANIFEST="$RESULT_DIR/migration_manifest.json"
```

Confirm that the build contains the benchmark binaries used by
`scripts/calibrate_hardware.py`. Keep the build inputs stable while compiling.
The selected compile mode, build identity, workload hash and target identity
must be retained in the generated manifest.

## Unified migration

Use `--mode dry-run` for an optional fresh-plan preview:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --build-dir "$BUILD_DIR" \
  --profile-dir "$PROFILE_DIR" \
  --output-dir "$RESULT_DIR" \
  --search-workloads "$WORKLOADS" \
  --search-strategy evolutionary --cost-model staged \
  --mode dry-run
```

Execute with `--mode execute` when ready. Use the same paths and workload file
for resume:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --resume-from "$MANIFEST" \
  --search-strategy evolutionary --cost-model staged
```

The unified entry restores the recorded protocol and compatible checkpoints.
It owns candidate enumeration, correctness checks, timing, promotion, model
fitting and selector replay. Do not add a profile-specific benchmark command.

## Resumable tuner

The tuner runs the same unified calibrator with an increasing cumulative search
budget. Its state file is `$RESULT_DIR/tuner_state.json`:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/tune_hardware.py \
  --resume-from "$MANIFEST" \
  --gpu "$GPU_UUID" \
  --time-budget 1800 --round-budget 16 --max-rounds 4
```

Use `--watch --poll-seconds 15` only when waiting for a busy GPU is
requested. Without `--watch`, a busy GPU leaves a resumable paused state. Use
`--dry-run` to record the next command without launching a child process:

```bash
"$PYTHON" scripts/tune_hardware.py --resume-from "$MANIFEST" --dry-run
```

## Workload coverage

Record semantic workload coverage from `search_coverage.json`, not just the
number of timing rows:

| Scope | Expected | Screened/confirmed | Status |
| --- | ---: | ---: | --- |
| `<operator/precision/placement/size/batch matrix>` | `<count>` | `<counts>` | `<complete/incomplete>` |

List workloads that remain unmeasured, budget-omitted, unavailable or rejected.
Distinguish an individually verified mapping from a model that is validated for
unmeasured ranking. A warning model or incomplete search is not evidence of a
complete profile.

## Evidence

Link the target-specific report and machine-readable artifacts:

- `<result report>`
- `<migration_manifest.json>`
- `<search_coverage.json>`
- `<search_measurements.json>`
- `<cost_model.json>` and its validation section
- `<selector_replay.json>` and status
- `<tuner_state.json>` when rounds were run
- `<paired comparison>` when an external baseline was measured under the same protocol

Include exact command lines, artifact hashes, target UUID, compile mode,
workload hash and protocol versions in the report. Historical mappings may be
proposals for target-side revalidation; historical latency is not a target
training label.

## Result update process

1. Keep each target and protocol in its own result directory; do not overwrite
   a completed run when changing workloads, seeds, model policy or projection.
2. Update the identity and protocol table from the new manifest, then update
   coverage and warning/failed statuses from the corresponding JSON artifacts.
3. Report valid measurements, rejected candidates, unavailable candidates and
   omitted cells separately. State whether selector replay passed.
4. Only describe migration as complete when the manifest and all required gates
   say complete. A zero exit code with `incomplete` is a valid finished round,
   not a completion claim.
5. Link the new report from the profile page and retain the previous report as
   historical evidence. Never merge timing or winner claims across hardware
   capacities or incompatible identities.
