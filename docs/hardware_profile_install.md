# Hardware Profile Initialization

cuButterfly does not treat a mapping measured on one GPU as a portable
constant. Installation can therefore initialize a local hardware profile and
build a first model table from the current device before an application begins
using tuning data.

## Recommended Installation

Configure the CUDA architecture explicitly, then use the repository wrapper:

```bash
cmake -S . -B build-local \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70

./scripts/install_with_hardware_profile.sh \
  --build-dir build-local \
  --prefix "$HOME/.local" \
  --profile-dir "$HOME/.local/share/cuButterfly/hardware/local"
```

The wrapper performs four steps:

1. build the configured tree;
2. run `ctest` so an installation cannot silently publish a broken binary;
3. install the library and profile helper commands;
4. run the hardware microbenchmark and generate the local model table.

The default probe uses five trials. It measures global feedback, modular
butterfly service, shared-memory exchange, CTA barrier service, and `Us=1/2/4/8`
stage-pipeline rates. The result directory contains:

```text
hardware_capabilities_raw.csv   immutable trial records
hardware_profile.json           median capability profile
unfolding_model.csv             parameterized Us/Ud/Ts/Td model table
manifest.json                   device fingerprint and provenance
```

The initialization step does not run Nsight Compute and does not require
administrator privileges. NCU collection remains a separate diagnostic step
because it changes clocks/replay behavior and requires profiler permissions.

## Direct Calibration Command

The same operation can be run without installation:

```bash
python3 scripts/initialize_hardware_profile.py \
  --microbench build-local/cuntt_hardware_microbench \
  --output-dir results/local_hardware_profile \
  --trials 5 \
  --logNs 8 10 12 14 16 18 20 \
  --spatial-budgets 32 64 128 256 \
  --word-bytes 4 8 \
  --force
```

The generated `hardware_profile.json` uses the same capability fields consumed
by `scripts/rank_unfolding.py`:

```bash
python3 scripts/rank_unfolding.py \
  --capabilities results/local_hardware_profile/hardware_profile.json \
  --logN 16 --spatial-budget 128 --word-bytes 8 \
  --stage-space 1 2 4 8 16 \
  --output results/local_hardware_profile/rank_logN16.csv
```

The initialization-generated `unfolding_model.csv` is a broad model table;
the ranker can produce a narrower workload-specific table without rerunning the
probe.

## Preventing Parameter Drift

Before reusing a local model, validate the GPU fingerprint:

```bash
~/.local/bin/check_hardware_profile.py \
  ~/.local/share/cuButterfly/hardware/local/hardware_profile.json
```

The command compares the recorded device name with `nvidia-smi`. A mismatch is
an error, not a warning. Use `--skip-device-check` only to inspect a profile on
a machine without CUDA. A profile is also tagged `calibrated-local` and carries
the CUDA-visible-device setting, host, probe executable, trial count, and
conditions in its provenance fields.

The profile is an input to model ranking, not an automatic claim that every
mapping has been measured. Runtime promotion still requires a correctness
check and repeated timing confirmation for the selected semantic cells. This
distinction prevents a fast microbenchmark from turning an analytic estimate
into an unsupported default.

## Updating A Profile

Re-run initialization after a driver/toolkit update, GPU replacement, clock
policy change, or major processing-unit change:

```bash
TRIALS=7 ./scripts/install_with_hardware_profile.sh \
  --build-dir build-local \
  --prefix "$HOME/.local" \
  --profile-dir "$HOME/.local/share/cuButterfly/hardware/local"
```

Keep old profiles in versioned directories when comparing model drift. Do not
overwrite a checked-in V100 profile with data from another device; local
calibration belongs in a user-specific profile directory or a separately named
results directory.
