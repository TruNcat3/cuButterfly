# Programming Guide

This guide defines the application-facing execution model. It complements the
[API Reference](api_reference.md), which lists individual public types and
functions, and the [Capability and Compatibility Matrix](support_matrix.md),
which states the supported semantic ranges.

The plan types described below are the power-of-two C++ physical-plan APIs.
Applications requiring exact arbitrary lengths, zero-extended embedding,
rank-two composition, selection caches, or status-code error handling should
use the [C Plan API](c_api.md); both surfaces share the same stream and caller-
owned workspace principles.

## Execution Model

Both API families use an immutable-shape plan:

1. Fill `ButterflyConfig` or `PlanConfig` with transform semantics and an
   optional physical mapping.
2. Construct `ButterflyPlan` or `Plan`. Construction validates the contract,
   resolves automatic fields, creates coefficient tables, and may allocate
   workspace.
3. Inspect `config()`, `selection()`, `data_size()`, and `workspace_size()`.
4. Bind a stream and, when desired, caller-owned workspace.
5. Submit one or more allocation-free `execute_async` calls.
6. Establish completion with CUDA stream or event operations before releasing
   input, output, workspace, stream, or plan resources.

Plan construction is intentionally outside the steady-state path. Reuse a plan
for repeated transforms with the same shape and semantics.

## Device And Context Association

Construction and execution use the CUDA device current on the calling host
thread. Internal allocations, coefficient tables, optional cuFFT state, input,
output, workspace, and stream must belong to that device. Changing the current
device and then using an existing plan is not supported. Create one plan per
device after selecting that device with `cudaSetDevice`.

The current release is a single-device API. It does not partition one transform
across GPUs. Independent plans may run on independent devices under normal CUDA
device and host-thread rules.

## Streams And Asynchrony

New plans use the default stream until `set_stream` is called.
`execute_async` enqueues work on the bound stream and returns without copying
data or synchronizing the stream. Kernel launch and immediately observable CUDA
API failures can throw on submission; asynchronous execution failures normally
surface at a later CUDA synchronization call.

The host-vector `execute` overload is different: it allocates staging buffers
on first use, performs transfers, uses CUDA events, synchronizes internally, and
returns timings. It is intended for validation and standalone measurements,
not latency-sensitive application integration.

Bind a stream only when no prior submission from the same plan is outstanding.
For concurrent streams, use one plan per stream. This rule also avoids changing
the stream associated with an internal cuFFT plan during execution.

## Data Layout

A logical butterfly transform contains `2^log_n` elements. `element_stride`
is the distance, in elements, between adjacent logical values.
`batch_stride` is the distance between transform starts. A zero batch stride is
resolved to the packed extent required by the element stride.

`ButterflyPlan::data_size()` returns the byte extent that must be addressable
from the input or output pointer, including batch padding and element stride.
The library does not insert hidden pack or unpack operations.

NTT currently uses contiguous batches. `Plan::data_size()` is
`batch * 2^log_n * (word_bits / 8)` with overflow checks.

## Placement And Output Semantics

Butterfly in-place execution requires `input == output`; out-of-place execution
requires distinct pointers. NTT device execution is currently out-of-place.

FFT and FWHT inverse transforms apply `1/N` scaling when
`normalize_inverse` is true. XOR Mobius inversion uses unsigned 32-bit modular
arithmetic and has no scale. NTT follows its modular inverse convention.
`OutputOrder` is part of the NTT contract: natural order is the general API
default, while the native bit-reversed path is available only for the documented
CompactStage configuration.

For FP16/BF16 storage, `ButterflyConfig::accumulation` selects native-width or
FP32 local accumulation and the result is narrowed after each logical
butterfly. The input/output pointer type remains the storage type: `Fp16` or
`Bf16` for real transforms and `Complex16` or `ComplexBf16` for FFT. V100 BF16
native-width execution is emulated and labeled; applications must not infer a
native SM70 BF16 arithmetic path from that semantic option.

## Workspace

`workspace_size()` is the exact minimum for the resolved plan. Zero means the
mapping needs no separate scratch allocation.

By default, a plan allocates required workspace during construction. Set
`auto_allocate_workspace = false` to use an application memory pool:

```cpp
config.auto_allocate_workspace = false;
cuntt::ButterflyPlan plan(config);

void* workspace = nullptr;
if (plan.workspace_size() != 0) {
    cudaMalloc(&workspace, plan.workspace_size());
    plan.set_workspace(workspace, plan.workspace_size());
}
```

External workspace must be at least 16-byte aligned, remain valid until all
submitted work completes, and must not be used by overlapping executions.
Butterfly plans reject a non-null workspace when the resolved mapping requires
zero bytes. Passing `nullptr` removes an external binding and exposes any
plan-owned allocation that was created at construction.

## Automatic Mapping Selection

`auto_select = true` delegates physical mapping fields to the calibrated
selector. The selector is strict: it rejects unknown GPUs and semantic shapes
outside its evidence table rather than silently using an unmeasured fallback.
After construction:

- `config()` contains the resolved mapping;
- `selection().implementation` identifies it;
- `selection().reason` explains the decision;
- `selection().calibrated` distinguishes measurement-backed selection.

Do not combine automatic selection with explicit decomposition or segment
mappings. See [Runtime Mapping Selector](runtime_selector.md) for the current
V100 coverage.

## Thread Safety And Reentrancy

Different plans may be used concurrently when their buffers and workspaces do
not overlap. A single plan is not thread-safe for concurrent submission or
mutation. `set_stream`, `set_workspace`, destruction, and execution must be
externally serialized for that plan.

The const query methods may be called when no other thread is mutating or
destroying the plan. No process-wide mutable execution handle is exposed.

## CUDA Graph Capture

CUDA Graph capture is not a supported contract in this release. The steady-state
device path is allocation-free and is structurally suitable for future capture
support, but the complete backend matrix, including cuFFT, has not been
validated under stream capture. Applications should not rely on capture
behavior until a tested compatibility matrix is published.

## Determinism

For a fixed build, GPU, resolved mapping, input, stream order, and configuration,
the library launches a fixed operation schedule. No cross-run autotuning occurs
inside `execute_async`. This is an execution-property statement, not a promise
of bitwise equality across backends, GPUs, CUDA versions, or FFT processing
units: floating-point operation order and mixed-precision rounding differ.

Integer NTT and XOR operations are expected to be exact within their documented
modular domains. Correctness tests, rather than timing comparisons, define this
contract.

## Resource Lifetime Summary

| Resource | Owner | Required lifetime |
|:--|:--|:--|
| plan | application | through completion of its last submission |
| stream | application | through completion of submitted work |
| input/output | application | until the stream no longer accesses them |
| external workspace | application | until all users of the workspace complete |
| internal workspace/tables | plan | managed by plan construction/destruction |

Destroying a plan with outstanding work is unsupported. Synchronize the bound
stream or establish equivalent completion first.
