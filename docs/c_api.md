# C Plan API

The installed `<cubutterfly/cubutterfly.h>` header provides a CUDA-library-style
C ABI. Applications create one handle per stream context, describe an operator
and workload, create an immutable plan, query and bind device workspace, then
submit allocation-free executions.

```c
cubutterflyHandle_t handle;
cubutterflyDescriptor_t descriptor;
cubutterflyPlan_t plan;
size_t shape[2] = {3, 5};
size_t workspace_bytes;

cubutterflyCreate(&handle);
cubutterflySetStream(handle, stream);
cubutterflyCreateDescriptor(&descriptor);
cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FFT);
cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_COMPLEX_FP32,
                      CUBUTTERFLY_COMPUTE_FP32);
cubutterflySetShape(descriptor, 2, shape, batch);
cubutterflyCreatePlan(handle, descriptor, &plan);
cubutterflyPlanGetWorkspaceSize(plan, &workspace_bytes);
cudaMalloc(&workspace, workspace_bytes);
cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes);
cubutterflyExecute(handle, plan, input, output);
```

Plan creation may initialize coefficient tables and vendor plans. Execution
uses the handle's current stream, performs no allocation, and does not
synchronize the host. One plan must not be submitted concurrently from
multiple host threads. Independent plans and workspaces may use independent
streams.

## Length Semantics

`CUBUTTERFLY_LENGTH_STANDARD` preserves the requested mathematical length.
Power-of-two axes use native butterfly plans. For a packed non-power-of-two
FP32/FP64 FFT, direct cuFFT and Bluestein with `M = next_pow2(2N-1)` are legal
processing-unit candidates; the default currently prefers direct cuFFT and
`MEASURE` times both complete compositions. Strided data uses Bluestein. A
uint64 NTT uses modular Bluestein only when the prime modulus admits both the
required `2N`-th root and the power-of-two convolution root. Plan creation
rejects an invalid root domain.

`CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING` is deliberately different: each
axis is padded to `next_pow2(N)` and the power-of-two transform is returned.
Forward input has the logical shape and forward output has the embedded shape;
inverse reverses those extents. The physical extents and byte requirements are
queryable, so callers never infer allocation sizes.

Rank two is lowered as row transform, tiled transpose, column transform, and
transpose back. This applies to FFT, NTT, FWHT, subset/superset zeta, and
Structured 2x2. The transpose is a mapping boundary, not a change to operator
arithmetic.

Contiguous physical outputs use a direct-output boundary: the final physical
core or transpose lands in the caller buffer and no scatter kernel is issued.
Strided outputs retain scatter semantics. `cubutterflyPlanGetAlgorithmName`
marks the former with `+direct-output`.
Workspace queries reflect the lowering: a rank-one direct-output butterfly may
need no composition buffer, a direct-output NTT keeps one packed input buffer,
and rank two keeps both transpose buffers.
Forward FP32 FWHT embedding can additionally fuse logical reads and zero fill
into a selected warp-register unit. Its algorithm name includes
`+fused-input`, it launches one kernel, and its workspace remains zero.
The suffix is deliberately absent when the selected core, operator, direction,
or layout uses the materialized pack fallback.

## Selection And Cache

The default policy uses this order:

1. exact workload key in the caller-selected cache;
2. exact device/semantics/length/batch entry in the generated application table;
3. legal static resource-model fallback.

`CUBUTTERFLY_ALGORITHM_MEASURE` times a bounded legal candidate set. Direct
power-of-two plans tune one plan, embedded/rank-two plans tune every physical
axis, modular Bluestein tunes its power-of-two NTT core, and exact arbitrary
FFT compares the complete direct and Bluestein paths. When a cache path is
configured, winners are appended and subsequent default plans reuse them.
Cache keys include compute
capability, SM count, operator, numeric contract, direction, placement, length
mode, rank, extents, batch, strides, and NTT modulus. Moving the file to a
different GPU therefore cannot cause an accidental match.

Register a log callback with `cubutterflySetLogCallback`. A missing device
profile emits `hardware-profile-miss`; a known device with an unseen workload
emits `workload-profile-miss`. Both messages include the selected fallback and
recommend measured calibration. `cubutterflyPlanGetSelectionSource`,
`cubutterflyPlanGetAlgorithmName`, and `cubutterflyPlanGetSelectionReason`
provide the same decision programmatically.

The generated V100 application table is sourced from
`config/v100_application_profiles.json`. It contains only archived exact
anchors and does not label interpolation or a static fallback as measured.
`cubutterfly_plan_bench --compare-cufft` records a matching forward standard
FFT baseline for rank-one or rank-two shapes and reports the resident-kernel
ratio; plan creation and host transfers remain outside both timings.

## Current Boundaries

- Standard arbitrary-length FFT: complex FP32/FP64, rank one or two.
- Standard arbitrary-length NTT: uint64, rank one or two, compatible prime/root domain.
- Embedded non-power-of-two shapes: all public operators and rank one or two.
- Runtime `MEASURE`: direct plans, per-axis composite plans, and exact FFT
  direct-versus-Bluestein selection. Composite tuning currently minimizes each
  axis independently because pack/transpose boundaries are common to its
  candidates.
- Structured 2x2 requires matrices for every executed axis.

All API calls return `cubutterflyStatus_t`. Invalid descriptors, unsupported
mathematical domains, allocation failures, and backend failures are returned
across the C boundary; C++ exceptions never escape.
