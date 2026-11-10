# cuButterfly v0.9

v0.9 is the research release family for hierarchical hybrid dataflow across
butterfly computations. It brings operator semantics, workload description,
execution organization, local cores and hardware constraints into a common
planning framework.

## Method and implementation

The method derives dependency-closed regions and their handoffs from staged
pair relations. Spatial expansion and temporal reuse expose resource demands
across storage scopes. Tile subgraph execution realizes those choices through
bounded residency, compatible arithmetic and producer/consumer boundary layouts.

The framework separates the following responsibilities:

| Layer | What it provides |
| --- | --- |
| Semantics and workload | Operator, precision, coefficients, direction, normalization, order, shape, batch and layout |
| Organization | Stage bands, closed regions, residency, spatial/time reuse, handoffs and concurrency |
| Arithmetic cores and lowering | Compatible local implementations, value roles, storage and physical resource requirements |
| Hardware descriptions | Capacity facts, measured service rates and explicit platform rules |
| Generation and selection | Constraint checks, compilation, model or measurement-based selection, cache reuse and plan reporting |
| Evaluation | Matched operator, fixed-core organization and application comparisons |

See the [design overview](design_overview.md) for the argument and the
[engineering map](development/repository-layout.md) for actual code boundaries.
Search and calibration implement selection; they are not separate research
contributions or guarantees of global optimality.

## Library capabilities

The public C, C++ and Python interfaces describe FFT, NTT, FWHT, structured 2×2,
and subset/superset zeta transforms. Precision, forward/inverse direction,
input/output order, shape, rank, batch and strides are request dimensions.
Multidimensional and adapted paths preserve these semantics and include required
conversion costs.

The [support contract](support_matrix.md) distinguishes expressible requests,
implemented lowerings, compiled specializations and measured mappings. Optional
providers and hardware resources constrain the available combinations. Single-band
and unit-expansion designs remain valid candidates alongside segmented or
pipelined execution.

## Installation and reproducibility

Follow [Getting started](getting_started.md) to build and install for a compatible
CUDA target. The [hardware workflow](hardware_profile_install.md) reuses
compatible static profiles and supports explicit calibration and tuning.
Preparation and compilation are separated from warm execution timing.

The [benchmark guide](../benchmark/README.md) provides configuration-driven
workload manifests and matched comparison rules. [Experimental results](benchmarks/results.md)
separate library competitiveness, fixed-core organization effects and application
benefits. Performance and numerical claims apply to their recorded workloads,
precision, hardware and protocols, including the reported negative results.

## Version identity

`v0.9` names the release family. The native library and Python frontend retain
component versions:

| Component | Version |
| --- | --- |
| Native library, C API and CMake package | `0.9.3` |
| Python frontend | `0.9.3.11` |

Record the source revision and build fingerprint when reproducing results.
Component versions do not replace the module ABI or cache identity. Earlier
versioned measurements retain their original identities. The
[changelog](../CHANGELOG.md) records detailed changes; software citation metadata
is maintained in [CITATION.cff](../CITATION.cff).
