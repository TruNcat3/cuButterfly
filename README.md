# cuButterfly

cuButterfly is a CUDA research library for FFT, NTT, FWHT, structured 2×2 and
subset/superset zeta transforms. It maps butterfly dependencies to hierarchical
hybrid tile dataflow with replaceable arithmetic cores. Requests preserve
precision, shape, batch, direction, normalization, input/output order and layout.

## Install

Use CMake 3.20+, a C++17 compiler, Python 3.9+ and a compatible CUDA toolkit and
driver. Activate the environment used for building and running the library.
From the repository root:

```bash
export PREFIX="${CONDA_PREFIX:-$HOME/.local}"
export BUILD="$PWD/build"
cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCUBUTTERFLY_BUILD_SHARED=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
bash scripts/install_with_hardware_profile.sh \
  --build-dir "$BUILD" --prefix "$PREFIX" \
  --stage-calibration bounded --search-seconds 300 --compile-seconds 600
python -m pip install ./python
```

The installer builds the library, detects the target, reuses compatible static
profiles or initializes missing profiles, and runs configured correctness checks.
Calibration requires an available GPU. Optional core providers have separate setup
instructions in the [installation guide](docs/hardware_profile_install.md).

## Calibrate and run

The [getting-started guide](docs/getting_started.md) explains profile refresh,
resumable tuning and environment activation. Calibration and compilation are
preparation work and are excluded from steady-state execution timing.

Use the [public planner](docs/unified_planner.md) through the C, C++ or Python API.
The [support matrix](docs/support_matrix.md) distinguishes valid requests from
implemented, compiled and measured mappings. Required layout conversions remain
part of complete-call execution. Examples and API references are indexed in
[the documentation](docs/README.md).

## Benchmark

Generate a workload manifest without a GPU:

```bash
python -m benchmark.manifest config/benchmark_quick.json --mode quick \
  --output /tmp/cubutterfly-quick-manifest.json
python scripts/check_repository.py
```

The [benchmark guide](benchmark/README.md) describes configuration-driven operator,
application and profiling campaigns. [Experiment tools](experiments/README.md)
provide CPU preparation, GPU measurement, fixed-core controls and result analysis.
Preserve matched semantics, timing boundaries and hardware/build identities;
record unavailable baselines explicitly. Compilation and plan initialization stay
outside kernel timing, while required conversions and normalization remain inside.

## License

This anonymous artifact is provided under the [research-only non-commercial
license](LICENSE). [Third-party notices](THIRD_PARTY_NOTICES.md) retain upstream
attribution and licensing requirements.
