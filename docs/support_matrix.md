# Capability and compatibility matrix

Support is evaluated for a complete semantic request in a particular build.
The public descriptor separates operator and workload from execution policy;
its capability query reports lowering gaps before GPU plan construction.
A valid request, an eligible lowering, a compiled executable and a measured
mapping are different states. None alone certifies performance.

## Public semantic dimensions

The [unified semantic contract](unified_semantics.md) defines validation and
canonicalization. The table below describes request dimensions, not a promise
that every Cartesian combination has an executable core.

| Dimension | Public representation and boundary |
| --- | --- |
| Operator | FFT, NTT, FWHT, subset/superset zeta (inverse gives Mobius), structured 2x2 |
| Arithmetic | Separate storage and compute types; NTT modulus and 32/64-bit word width; structured matrices |
| Direction and scaling | Forward/inverse, operator-specific normalization and explicit output scaling/rounding where lowered |
| Shape and load | Extents vector and independent batch; descriptors can express ranks beyond the current execution limit |
| Ordering and layout | Independent input/output per-axis logical order, axis strides and batch strides; in-place or out-of-place |
| FFT interpretation | C2C/R2C/C2R; real and complex sides have distinct element units and compact-spectrum shapes |
| Length interpretation | Standard transform or explicit zero-extended embedding; exact arbitrary-length execution depends on arithmetic and lowering |
| Selection | DEFAULT, explicit MEASURE tuning or EXPLICIT mapping; separate AUTO/XD/COMPOSITE execution mode |

Generic axis composition currently executes through rank eight. Some requests
use layout, numeric or real-FFT adapters with additional workspace and conversion
cost. See [unified semantics](unified_semantics.md) for current adapter coverage,
including UINT64 zeta and low-precision real FFT restrictions. Per-provider length,
precision, direction, modulus/root and resource requirements still apply.

Low storage precision does not imply one universal intermediate arithmetic rule.
Storage, compute, output rounding and the selected provider's contract must be
recorded together. In particular, the final-storage FWHT provider has a distinct
contract from stage-storage rounding; it cannot satisfy a request for the latter.
See [C API rounding semantics](c_api.md) and the [planner](unified_planner.md).

## Inspect a request and its execution state

Use `Descriptor.validate()` and `Descriptor.capabilities_json()` through the
[public Python interface](getting_started.md#check-the-public-python-boundary),
or the corresponding C/C++ descriptor API. These operations run on the CPU and
do not create a GPU plan. The descriptor CLI accepts a serialized descriptor:

```bash
python -m cubutterfly descriptor.json --library /absolute/path/to/libcuntt.so
```

| State | What it establishes |
| --- | --- |
| Semantic validation passes | The request satisfies the normalizer's mathematical/layout rules |
| `missing-lowering` | The request is expressible but the current execution path has a reported gap |
| `eligible` | Static structural/lowering checks pass; roots, device resources and executable specialization may still need validation |
| Plan construction succeeds | A concrete executable plan was obtained under the selected policy and resources |
| Correctness checked | The named cases match a reference within their arithmetic contract |
| Measured/selected | A mapping has observations for a named workload, hardware, build and timing protocol |

Explicit mapping replay preserves the requested semantics and core configuration;
an unavailable explicit core must fail rather than silently change the request.
AUTO selection may choose among other legal implementations. Capability output
is the first inspection step, not a replacement for plan or numerical checks.

## Toolchain and packaging

| Item | Current build contract |
| --- | --- |
| Platform/toolchain | Linux research workflow; C++17, CMake 3.20+, compatible CUDA toolkit and host compiler |
| CUDA architecture | Detected for a new native build; set `CMAKE_CUDA_ARCHITECTURES` explicitly for cross compilation |
| CUDA dependencies | CUDA Runtime and cuFFT development components |
| Python frontend | Python 3.9+, thin ctypes package; shared library built with `CUBUTTERFLY_BUILD_SHARED=ON` |
| CMake consumer | `find_package(cuButterfly CONFIG REQUIRED)`, target `cuButterfly::cuButterfly` |
| Package version compatibility | Same-major CMake compatibility; no stable minor-release binary ABI promise |

CUDA 11.8/GCC11/SM70 identify older V100 artifact builds, not universal defaults.
Use a compiler and optional provider version that actually accept the target.
See [installation](getting_started.md) and
[hardware profiles](hardware/README.md) for target-specific requirements.

| CMake option | Default | Scope |
| --- | --- | --- |
| `CUBUTTERFLY_ENABLE_CUFFTDX` | OFF | Optional MathDx/cuFFTDx cores; actual provider capabilities still constrain candidates |
| `CUBUTTERFLY_ENABLE_TURBOFFT` | OFF | Available TurboFFT-generated kernels |
| `CUBUTTERFLY_ENABLE_VKFFT` | OFF | Optional VkFFT comparison benchmark |
| `CUBUTTERFLY_ENABLE_XD_VKFFT` | Follows `ENABLE_VKFFT` | Optional VkFFT xD axis core; distinct from the comparison benchmark |
| `CUBUTTERFLY_BUILD_EXAMPLES` | ON | Public API examples |
| `CUBUTTERFLY_FFT_CODEGEN_MODE` | `precompiled` | Precompiled inventory versus `on-demand` research generation; not the theoretical support space |

## Selection and measurement

Current public planning is not restricted to a V100 static winner table.
Compatible measured registries, static model/rate inputs and labelled unmeasured
fallbacks have distinct provenance. Rank-one and xD selection have different
bounded candidate paths. DEFAULT does not automatically run a performance
benchmark; MEASURE explicitly requests measurements. Installation reuses
compatible profiles and measures missing targets or explicitly requested updates.
See [static profile policy](hardware_profile_install.md#static-profile-policy)
and [planner contracts](unified_planner.md) for compatibility and budget limits.

A capacity profile, successful compilation or a finite calibration population
does not establish that every operator/size/batch is tuned. Performance claims
require matching numerical semantics, hardware, build and timing boundaries.

## Historical backend snapshots

The old V100 selector and low-level NTT tutorials describe named backend
implementations and generated inventories. Their ranges (for example Hybrid2D,
CompactStage, StagePipeline and resident dataflow variants) are backend-specific
snapshots, not limits on the entire current public descriptor/API. In particular,
the former statement that uint32 NTT requires only Hybrid2D or HybridDataflow
must not be used as a current universal restriction.

Consult the [historical programming guide](programming_guide.md),
feature coverage records and
[legacy runtime selector](runtime_selector.md) when reproducing those paths.
Check the current descriptor capability and explicit plan construction for the
actual build. This page does not claim a fresh exhaustive audit of every legacy
backend, optional provider or semantic combination.

Public source changes are recorded in the release history. Generated
design JSON, measurement CSV and CLI schemas remain research interfaces; they do
not carry the same compatibility expectation as installed public headers.
