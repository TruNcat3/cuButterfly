# Getting started

cuButterfly exposes butterfly transforms through the public C, C++ and Python
interfaces. Start with a semantic request and let the planner select an executable
mapping. The [unified planner](unified_planner.md) describes selection, explicit
mapping replay and current lowering limits.

## Choose the environment

Build prerequisites are CMake 3.20+, a C++17 compiler, a CUDA toolkit compatible
with the target GPU and host compiler, and Python 3.9+. Python tooling and CMake
may download dependencies. GPU execution also needs a compatible driver.
There is no universal CUDA 11.8 or SM70 default: a new native build detects the
architecture; cross compilation must name it explicitly.

Activate a Python environment that meets these prerequisites and reuse a
compatible existing build when available. Keep the interpreter, compiler,
architecture and install prefix consistent. Hardware-specific setup and evidence
are indexed in [hardware profiles](hardware/README.md); profiles from another
model or memory capacity are not interchangeable.

## Build and install on a target GPU

Run from the repository root in the chosen environment. `PREFIX` is the install
destination; choose a writable prefix and keep the same value throughout.
For a new Python-enabled build:

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

The configure step compiles no kernels for measurement. The installer then builds,
selects a compatible static profile, initializes one if missing, runs its configured
correctness tests and installs. **Initialization uses the live GPU**; the command
above is not a CPU-only setup command. Its bounded budget limits work, not coverage
or the quality of the resulting optimum. Optional providers such as cuFFTDx need
explicit setup; see [hardware installation](hardware_profile_install.md).

When reusing an existing build, preserve its compiler, architecture and codegen
policy. Inspect its CMake cache before changing them. For cross compilation add
`-DCMAKE_CUDA_ARCHITECTURES=<target>` to configuration; this permits compilation
without a visible GPU but does not make the installer CPU-only.

CPU precompilation produces executable specializations and compiler resource
information. GPU calibration acquires capacity/rate/service observations and
selected workload measurements. Neither establishes full design-space coverage.
`precompiled` is the default FFT codegen policy; research workflows may use
`--fft-codegen-mode on-demand` and `CUBUTTERFLY_COMPILE_MODE=research` for additional
implemented specializations. See [compile policies](unified_planner.md).

## Check the public Python boundary

Set `CUBUTTERFLY_LIBRARY` to the actual shared library in the build or prefix.
The following CPU descriptor check loads the library but creates no GPU plan:

```bash
export CUBUTTERFLY_LIBRARY="$BUILD/libcuntt.so"
python - <<'PY'
import json
import cubutterfly as cb
library = cb.Library()
request = cb.Transform(op=cb.Operator.FWHT, storage=cb.DataType.FP32,
                       extents=[64], batch=1)
with cb.Descriptor(library, request) as descriptor:
    descriptor.validate()
    print(descriptor.json())
    print(json.loads(descriptor.capabilities_json()))
PY
ctest --test-dir "$BUILD" --output-on-failure \
  -R '^(semantic_contract_cpu|cpp_descriptor_cpu)$'
```

Descriptor validation checks semantics; capability output distinguishes eligibility
from missing lowerings. It does not prove a kernel's numerical result or speed.
For GPU planning and execution use `cb.Plan` or C++ `<cubutterfly/plan.hpp>` as
specified in the [planner guide](unified_planner.md), and run the affected
correctness cases listed by `ctest --test-dir "$BUILD" -N`. Python execution
accepts device pointers; allocations and stream lifetimes remain the caller's
responsibility. The [semantic contract](unified_semantics.md) defines precision,
normalization, layout and unsupported combinations.

## Reuse, refresh and tune

The installer prefers an explicit `CUBUTTERFLY_HARDWARE_MODEL`, then compatible
local profiles, then bundled seeds. Reuse does not run calibration/search and does
not certify missing performance layers. Source the printed `profile_environment.sh`
for a custom profile location; conda installations also install activation hooks.

Use `--refresh-profile` to request new measurements, or `--resume-search` to
continue compatible interrupted work; they are mutually exclusive. Public
`DEFAULT` selection can reuse compatible measured entries or use a static model,
rate prior or labelled unmeasured fallback. Explicit `MEASURE` tuning is a
separate GPU measurement request. None of these paths guarantees a global optimum
or automatically tunes every size, batch and operator. Budgets, identities and
selection provenance are detailed in [hardware profiles](hardware_profile_install.md).

Historical low-level NTT/backend examples remain in the
[programming guide](programming_guide.md); use the public planner contract for new
applications. For changes and matching validation, see [Contributing](../CONTRIBUTING.md).
