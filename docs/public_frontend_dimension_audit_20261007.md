# Public frontend and dimension audit, 2026-10-07

This is the pre-repair audit. The subsequent [0.9.2 semantic repair](unified_semantics.md)
addresses the frontend propagation defects and common ordering boundary. The
old comparison campaign is cancelled; the findings below retain their audit-time context.

The common C/C++ planner exists, but the requested semantic dimensions are not
uniformly expressible and preserved through every frontend. The earlier
engineering review repaired selection, calibration identity and reporting
within its finite campaign. It did not establish a complete public contract
for every combination. This follow-up is a source audit, not a runtime repair
or new GPU measurement.

## What is already shared

[`Transform` and `Plan`](../include/cubutterfly/plan.hpp) translate a user
description into the [public C descriptor](../include/cubutterfly/cubutterfly.h).
Shape, batch, storage/compute types, direction, normalization, placement,
strides, FFT type and domain parameters have public entry points. The
[`workload_key`](../src/c_api.cpp) includes shape, batch, strides, types,
direction, placement, execution mode and runtime/platform identities. NTT
orders and structured matrices also enter this key where applicable.
Selection and mapping replay are real library facilities, not only scripts.

The missing step is consistent expression, validation and propagation of the
full contract across entry points and executable lowerings. A field existing
in a struct, or a specialized benchmark succeeding, does not establish that.

## Dimension-oriented findings

| User dimension | Public expression today | Gap or execution boundary |
| --- | --- | --- |
| Input/output logical order | `SetNttLayouts`; C++ order fields use the NTT enum | No common permutation contract for other operators. NTT bit-reversed input is rejected; bit-reversed output is limited to rank-one power-of-two natural-input NTT. These are implementation limits. |
| Axis and batch strides | Separate input/output arrays and batch strides | C++ drops nonzero batch strides when both axis-stride vectors are empty. Rank-one common lowering requires matching input/output layouts; direct NTT and real FFT have packed-layout restrictions. |
| Storage and computation precision | Separate storage/compute fields | Low-precision C2C, FWHT and structured paths exist, including native/FP32 accumulation. Compact real FFT accepts FP32/FP64 with matching compute type. A missing combination must be reported individually. |
| Direction and normalization | Public direction and normalization setter | NTT inverse includes modular inverse-N scaling; the flag controls FFT/FWHT scaling. R2C-forward and C2R-inverse are the defined real-transform directions. Direction is distinct from DIT/DIF/GS. |
| Batch and point counts | Batch plus an extent per axis | Accepted shapes still depend on available lowerings and resources. Large-batch memory limits are not mathematical exclusions. |
| 1D/2D/xD | Public rank is explicitly 1–3 | Rank above three is an implementation gap. Vector-shaped input does not imply arbitrary-rank lowering. |
| Placement and length interpretation | In-place/out-of-place and standard/embedding | Real FFT currently requires standard, out-of-place data. Embedding is also out-of-place and is not an exact arbitrary-length transform. |
| Transform/domain parameters | FFT C2C/R2C/C2R, modulus/word width, per-axis stage matrices | Operator-specific mathematics remains necessary; these fields should live in the same workload contract, rather than define separate experiment naming schemes. |

Implementation evidence is in `plan.hpp`'s `Plan` constructor and
`src/c_api.cpp`'s descriptor setters, `validate_storage`, `resolve_default_strides`,
`workload_key`, `xd_contract` and `cubutterflyCreatePlan`.

Two concrete frontend losses are visible without running a GPU:

* `Plan` calls `SetStrides` only when an axis-stride vector is nonempty. Thus
  setting only `input_batch_stride`/`output_batch_stride` silently uses default
  packed batching instead. Correct forwarding must resolve the two default
  axis layouts, including different real/half-complex shapes. Merely adding a
  benchmark flag does not fix this. The test binding's `make_plan` has the same
  conditional forwarding issue for batch strides without axis strides.
* `Plan` forwards `input_order`/`output_order` only for NTT. A non-default order
  on another operator is silently ignored. Until a general order contract is
  implemented, such a request must fail explicitly. Rejecting it fixes silent
  interpretation but does not complete general order support.

The public API exposes plan metadata after successful creation, but no
standalone descriptor export/validation or capability query. Most unsupported
combinations are discovered during plan creation. Exception details are
collapsed to status codes by `protect`; C++ `check` exposes the status string.
A plan's mapping JSON is not a substitute for a queryable semantic descriptor.

## Frontend and search boundaries

The reviewed tree has installed C headers, C++ headers and a CMake library
target. Its installation rules do not provide a general Python operator
package. [`benchmark_xd.py`](../scripts/benchmark_xd.py) imports the ctypes
binding from [`tests/check_xd.py`](../tests/check_xd.py); application adapters
also exist. These exercise public C ABI calls, but do not establish a supported,
installed Python frontend with equivalent dimension coverage.

[`cubutterfly_plan_bench`](../apps/cubutterfly_plan_bench.cpp) uses the C ABI,
but exposes a subset: shape, batch, selected precisions, direction, placement
and mapping policy. It fixes NTT orders to natural and does not expose the
full stride, accumulation and real-transform contract. Other executables
expose different subsets. Therefore an internal benchmark option is not
evidence of a corresponding option in every user entry point.

There are also multiple workload representations. The rank-one
[`hardware_registry.py`](../scripts/hardware_registry.py) key uses `logN` and
scalar strides; its historical NTT `placement` denotes output order. The C
descriptor separates allocation placement from logical order and carries
multidimensional shapes/strides. This is a schema boundary requiring explicit
translation, not evidence by itself of a current cache collision. xD has its
own contract and selection path. Sharing kernel inventories does not mean all
frontends invoke identical searches or consume the same complete calibration.

DIT/DIF/GS, core choice, tile sizes, residency and exchange policy belong to
the execution mapping. Users specify the required input/output representation
and mathematical transform; the planner chooses an equivalent implementation.
Any required permutation/conversion, whether separate or fused, belongs in
correctness checks and complete-call cost. Forced mappings remain useful for
research controls but must not be required to obtain ordinary semantics.

## Required integration and acceptance

The integration target is one normalized workload description consumed by
public frontends, search/calibration keys, mapping replay and matrix generation.
Existing operator traits and working kernels should be retained. Legitimate
arithmetic differences need capability rules; they do not require separate
user-facing workflows for every case.

1. Preserve or explicitly reject every supplied field. Add descriptor
   validation/export and structured capability reasons at the library boundary.
   Define general logical order independently from physical strides, with
   explicit per-axis meaning for multidimensional data. Keep current NTT
   encoding compatibility explicit.
2. Connect the normalized description to search and lowering. Missing order,
   rank, precision and layout paths remain named implementation gaps. A
   conversion fallback can establish semantics, but its overhead must enter
   selection and measurement; it cannot be advertised as fused/high-performance
   support without evidence.
3. Generate tests from dimension values and constraints, and execute them
   through the installed public interface. Small correctness cases cross
   order, storage/compute precision, direction, placement/layout, rank/shape and
   batch. Scaling cases cross declared point-count and batch ranges. Every
   omitted intersection needs a recorded reason, not an operator-specific
   handwritten exception.

The status vocabulary must distinguish invalid mathematics/contradictory
semantics, frontend-unexpressible requests, missing lowerings, resource limits,
untested combinations and verified execution. Baseline support is a separate
attribute. A reference library lacking a feature does not remove our coverage
obligation; an equivalent adapter may be compared if its complete cost and
semantics are recorded. Successful API forwarding, correct execution, search
coverage and measured speed are separate acceptance results.

For example, a low-precision inverse multidimensional transform with padded
batches must appear as one ordinary combination. If no lowering exists, the
same public entry returns a precise gap. It must neither reset a dimension to
a default nor require a separate test-only implementation.

## Current campaign interpretation

The frozen `results/reviewed_campaign_20261007_r1/manifest.json` remains a
finite regression campaign. Its xD arithmetic coverage crosses ten
operator/precision pairs, four shapes, three batches, two directions and two
mapping controls. It uses packed, natural-order data; it is not a crossing of
all public semantic dimensions. The 113 gap probes add selected intersections,
not the missing full product. In particular, low precision with rank three,
non-default order with multidimensional layouts, and layout with multiple
precisions/directions remain insufficiently covered.

The prior CPU tests and 49-command preparation checks remain valid for their
declared contracts. They do not establish the broader frontend target. This
audit changes documentation only, preserves the frozen campaign and records
the integration work as unfinished. Runtime repairs will need affected public
contract tests and a newly identified build/campaign before GPU acceptance.
