# Device API

cuButterfly exposes a CUDA-library-style execution path for applications that
already own device memory and streams. Plan construction resolves the mapping,
builds coefficient tables, and may allocate plan-owned workspace. Repeated
`execute_async` calls perform no allocation, host transfer, or synchronization.

Set `auto_select = true` to resolve the backend and physical mapping from the
calibrated runtime table. The resolved parameters remain available through
`plan.config()` and the decision record through `plan.selection()`. See
[Runtime Mapping Selector](runtime_selector.md).

The host-vector `execute` overloads remain available for correctness checks and
standalone measurement. They use the same kernel submission path but include
host/device transfers and return timing statistics.

## Install And Link

```bash
cmake --install build --prefix "$HOME/.local"
```

```cmake
find_package(cuButterfly CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cuButterfly::cuButterfly)
```

Set `CMAKE_PREFIX_PATH` to the chosen prefix when it is not on CMake's default
search path.

## Butterfly Plan

```cpp
cuntt::ButterflyConfig config;
config.op = cuntt::ButterflyOperator::Fft;
config.backend = cuntt::ButterflyBackend::OnlineReorder;
config.precision = cuntt::ButterflyPrecision::Fp32;
config.placement = cuntt::ButterflyPlacement::OutOfPlace;
config.log_n = 18;
config.batch = 4;
config.auto_allocate_workspace = false;

cuntt::ButterflyPlan plan(config);
plan.set_stream(stream);
cudaMalloc(&workspace, plan.workspace_size());
plan.set_workspace(workspace, plan.workspace_size());
plan.execute_async(device_input, device_output);
```

`data_size()` is the byte extent of one input or output allocation, including
configured batch and element strides. `workspace_size()` is the exact minimum
for the resolved mapping. A zero value means no workspace is required.

The pointer type is part of the semantic contract:

| Operation and precision | Pointer type |
|:--|:--|
| FP32 FWHT | `float*` |
| FP64 FWHT | `double*` |
| FP16 FFT | `cuntt::Complex16*` |
| BF16 FFT | `cuntt::ComplexBf16*` |
| FP32 or legacy FP16/FP32-WMMA FFT | `cuntt::Complex32*` |
| FP64 FFT | `cuntt::Complex64*` |
| Subset/superset zeta and legacy XOR-zeta name | `std::uint32_t*` |
| FP16 structured 2x2 or FWHT | `cuntt::Fp16*` |
| BF16 structured 2x2 or FWHT | `cuntt::Bf16*` |
| FP32 structured 2x2 | `float*` |
| FP64 structured 2x2 | `double*` |

An in-place plan requires identical input and output pointers. An out-of-place
plan requires distinct pointers.

## NTT Plan

```cpp
cuntt::PlanConfig config;
config.log_n = 20;
config.batch = 4;
config.backend = cuntt::Backend::Hybrid2D;
config.word_bits = 64;
config.auto_allocate_workspace = false;

cuntt::Plan plan(config);
plan.set_stream(stream);
cudaMalloc(&workspace, plan.workspace_size());
plan.set_workspace(workspace, plan.workspace_size());
plan.execute_async(device_input, device_output);
```

NTT device execution is currently out-of-place. Use `std::uint32_t*` for a
32-bit plan and `std::uint64_t*` for a 64-bit plan. Hybrid2D and natural-order
CompactStage require one `data_size()` workspace buffer. Baseline, Tile256,
StagePipeline, and bit-reversed CompactStage use the output as their working
buffer and report zero workspace.

## Lifetime And Ordering

- The application owns device input, output, workspace, and stream objects.
- Those objects must remain valid until the stream has completed submitted work.
- One plan submits to one bound stream. Call `set_stream` before submission and
  do not change it while prior work from that plan is outstanding.
- A workspace may be shared by plans only when their executions do not overlap.
- A plan is not intended for concurrent submission from multiple host threads.
- `set_workspace(nullptr, 0)` returns to plan-owned workspace when automatic
  workspace allocation was enabled.

CUDA events or ordinary stream dependencies can compose the transform with
upstream and downstream kernels without a device-wide barrier. The complete
working examples are indexed in [Examples](examples.md). Device association,
thread safety, graph-capture status, and determinism are defined in the
[Programming Guide](programming_guide.md); failure behavior is defined in
[Error Handling](error_handling.md).
