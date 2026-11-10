# Hardware Profiles

These pages are model- and capacity-specific runbooks for the cuButterfly
migration workflow. A profile is local evidence for a
device identity, build, compile policy and workload file. It is not a portable
set of winning mapping parameters.

## Hardware identity and compatibility

A target is identified by its GPU model, compute capability and reported memory
capacity. Measurements additionally record the device UUID, build, compile policy
and timing protocol. A CUDA index is only a local device selector.

Capacity facts, hardware service rates, stage-service models and selected plans
are separate artifacts. Compatible local or bundled files can be reused;
measurements from another target may propose candidates but do not establish
performance on the new device. See the
[profile policy](../hardware_profile_install.md#static-profile-policy).

## Target references

| Target | Reference |
| --- | --- |
| NVIDIA A100 PCIe 40GB | [Device and profile reference](a100-pcie-40gb.md) |
| NVIDIA A100 PCIe 80GB | [Device and profile reference](a100-pcie-80gb.md) |
| NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB | [Toolchain and target setup](pro6000_server96.md) |
| NVIDIA V100 | [SM70 target setup](v100.md) |
| Another GPU | [Profile template](TEMPLATE.md) |

Target pages may also retain dated experiment records. Use their declared
hardware and build identities when reproducing those records. A documented
setup does not imply that every operator configuration has been measured.

## Prepare a target

Follow [installation and calibration](../hardware_profile_install.md) with a CUDA
compiler that supports the target. Use the Python environment and build directory
selected for that installation:

```bash
PYTHON=python
BUILD_DIR=build
```

The calibrator records workload hashes, benchmark paths, device identity,
compile mode and measurement policy. Keep the source, binaries and target
profile compatible when resuming a run.

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
