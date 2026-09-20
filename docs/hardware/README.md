# Hardware Profiles

These pages are model- and capacity-specific runbooks for the cuButterfly
migration workflow. A profile is local evidence for a
device identity, build, compile policy and workload file. It is not a portable
set of winning mapping parameters.

## Profiles

| Target | Device identity used for this host | Current documentation state |
| --- | --- | --- |
| [V100 portable campaign](v100.md) | SM70; model, capacity and UUID detected on the target machine | CPU preparation and same-contract campaign entry; new V100 GPU evidence pending. |
| [A100 PCIe 40GB](a100-pcie-40gb.md) | A100-PCIE-40GB, sm80, UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e` | A resumable partial staged checkpoint exists; no current performance or completion claim is made. |
| [A100 PCIe 80GB](a100-pcie-80gb.md) | A100 80GB PCIe, sm80, UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` | Three-cell recovery evidence exists; the model gate remains warning and the migration is incomplete. |

The UUIDs identify the intended device on the current multi-GPU host. A local
CUDA index is only a selection mechanism and must not be copied between hosts.
Do not reuse a profile, checkpoint, timing row or model-qualified result between
the 40GB and 80GB pages. The cumulative registry may be shared: its records
must be filtered by target identity, and a mapping from another target is only
a seed proposal that requires correctness and timing on this target.

## Shared prerequisites

The commands below describe this A100 host. For a separate V100 host, use the
portable environment/build/calibration commands in [the V100 runbook](v100.md).
On this host, use the shared conda environment and current research build:

```bash
PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
BUILD_DIR=build-a100-cufftdx
```

The build must contain the benchmark executables expected by the unified
calibrator. Keep the source tree and public headers stable during a CUDA build
or calibration. The calibrator records device identity, compile mode, workload
hash, benchmark paths and search protocol in its migration manifest.

The general operational contract is in
[Hardware Profile Initialization](../hardware_profile_install.md). It covers
the manifest status values, checkpoint reuse, correctness requirements,
selector replay and the distinction between a model warning and a completed
migration.

## Common workflow

Start a fresh target-specific migration with the unified entry point. Use
`--mode dry-run` when you want to inspect the plan; replace the output and
profile paths with paths owned by that target card:

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

Run the same command with `--mode execute` when ready. The migration entry is
the only benchmark/promotion owner; the profile pages do not add a second
benchmark or promotion path.

When the command writes `migration_manifest.json`, continue bounded rounds with
the resumable tuner:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/tune_hardware.py \
  --resume-from "$MANIFEST" \
  --gpu "$GPU_UUID" \
  --time-budget 1800 --round-budget 16 --max-rounds 4
```

Add `--watch` when waiting for a busy GPU is explicitly requested. The tuner
pauses by default, polls only within the remaining time budget when watching,
and writes `tuner_state.json` beside the manifest. A dry run records the next
command and budget without launching the calibrator:

```bash
"$PYTHON" scripts/tune_hardware.py --resume-from "$MANIFEST" --dry-run
```

The direct resume path is useful for one bounded calibrator invocation:

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
CUBUTTERFLY_COMPILE_MODE="$COMPILE_MODE" \
  "$PYTHON" scripts/calibrate_hardware.py \
  --resume-from "$MANIFEST" \
  --search-strategy evolutionary --cost-model staged
```

The resume entry restores the build, profile, output, workload, compile policy
and compatible checkpoint identities from the manifest. Keep the same target
UUID and do not substitute a capacity-specific profile. A zero exit code with
an `incomplete` manifest can mean that measured mappings are valid while the
model or search coverage is still a warning; check the manifest before making
a completion claim.

## Updating a profile

Each target page must be updated from artifacts produced for that target only.
Record the exact device identity, benchmark/build hashes, compile mode, workload
hash, commands, search protocol, coverage counts, rejected/unavailable cells,
model status and selector replay status. Link the report and machine-readable
artifacts rather than copying timing tables into a generic profile.

Use [TEMPLATE.md](TEMPLATE.md) for a new model-plus-capacity page. Preserve
old reports and manifests in their result directory; use a new result directory
for a changed workload, seed policy, candidate projection or model-update
policy. Never turn historical timings from one A100 capacity into target
measurements for the other.
