# API Reference

The library now also installs a stable-style C plan surface in
`<cubutterfly/cubutterfly.h>`. It follows the handle, descriptor, immutable
plan, workspace-query, and asynchronous execute sequence used by CUDA
professional libraries. See [C Plan API](c_api.md) for the complete contract,
general shapes, selection cache, and logging behavior.

## C++ API

The installed public headers are `<cuntt/butterfly.hpp>` and
`<cuntt/ntt.hpp>`. All symbols are in namespace `cuntt`. The library requires
C++17 and links as `cuButterfly::cuButterfly`.

## Common Conventions

Enum-to-name functions are `noexcept` and return static lowercase CLI names.
`parse_*` functions accept those names and throw `std::invalid_argument` for an
unknown value. Plan objects are movable but not copyable. Query methods return
the configuration after defaults and automatic choices have been resolved.

Byte-size queries describe device allocations, not element counts. See
[Programming Guide](programming_guide.md) for stream, device, and lifetime
rules and [Error Handling](error_handling.md) for failure categories.

## Butterfly Value Types

```cpp
struct alignas(8) Complex32 { float real; float imag; };
struct alignas(16) Complex64 { double real; double imag; };
```

The actual `Complex32` fields are two `float` values named `real` and `imag`,
with 8-byte alignment. `Complex64` contains two `double` values with 16-byte
alignment. They are the device and host storage types for FFT plans.

| `ButterflyPrecision` | Storage/meaning |
|:--|:--|
| `Fp16` | `Fp16`/`Complex16` storage; `Native` or `Fp32` accumulation |
| `Bf16` | `Bf16`/`ComplexBf16` storage; `Native` or `Fp32` accumulation |
| `Fp32` | FP32 FWHT or complex FP32 FFT |
| `Fp64` | FP64 FWHT or complex FP64 FFT |
| `Fp16Fp32` | `Complex32` storage with supported mixed-precision FFT core |
| `Uint32` | subset/superset zeta/Mobius unsigned arithmetic |

## Butterfly Semantics

`ButterflyOperator` selects `Fwht`, `Fft`, `SubsetZeta`, `SupersetZeta`, or the
stage-parameterized `Structured2x2`, or the legacy-compatible `XorZeta`.
`XorZeta` retains its established subset-zeta
semantics; new code should use `SubsetZeta`. CLI aliases `or-zeta` and
`and-zeta` parse as subset and superset zeta respectively. See [Operator
Catalog](operator_catalog.md).
`ButterflyPlacement` selects `InPlace` or `OutOfPlace` and is validated against
pointer identity. `inverse` selects the inverse transform and
`normalize_inverse` controls FFT/FWHT inverse scaling.

`ButterflyBackend` selects the schedule family: `TemporalTile`,
`Hierarchical`, `OnlineReorder`, `WarpHybrid`, `StagePipeline`, or vendor
`CuFft`. Availability is queried with:

```cpp
std::vector<ButterflyCapability> butterfly_capabilities();
```

Each record reports backend length limits, compiled radix/core/precision,
placement and layout flags, plus a textual constraint. Runtime validation is
authoritative because optional build features change the compiled matrix.

## ButterflyConfig

### Workload fields

| Field | Meaning |
|:--|:--|
| `op` | transform operator |
| `precision` | arithmetic and pointer-type contract |
| `accumulation` | `Native` or explicit `Fp32`; the latter is legal for FP16/BF16 storage |
| `placement` | in-place or out-of-place execution |
| `log_n` | base-2 transform length |
| `stage_matrices` | structured-2x2 coefficient list: one broadcast matrix or `log_n` matrices |
| `batch` | number of transforms; must be positive |
| `batch_stride` | element distance between batches; zero selects packed layout |
| `element_stride` | element distance inside one transform; must be positive |
| `inverse` | forward when false, inverse when true |
| `normalize_inverse` | apply `1/N` for inverse FFT/FWHT |

### Selection and ownership fields

| Field | Meaning |
|:--|:--|
| `backend` | explicit schedule/backend |
| `auto_select` | resolve a calibrated mapping and override physical fields |
| `auto_allocate_workspace` | allocate required scratch during construction |

### Decomposition fields

`stage_partition` is the ordered list of logical stage counts and must sum to
`log_n`. `segment_mappings` assigns an `FftSegmentMapping` to each segment;
`boundaries` assigns an `FftBoundaryMapping` between adjacent segments.
`execution_group_mappings` describes physical groups after fused logical
boundaries have been lowered. These vectors are advanced explicit-mapping
controls and are currently constrained to supported online-reorder FFT forms.

`FftSegmentMapping` contains `core`, `exchange`, `threads`, and elements per
thread (`ept`). `FftBoundaryMapping` contains cross-twiddle mode, boundary
layout, and residency.

### Physical mapping fields

The remaining fields select physical design points: `stage_space`,
`stage_handoff`, `tile_threads`, `prefix_threads`, `suffix_threads`,
`prefix_ept`, `suffix_ept`, `prefix_units_per_cta`,
`suffix_units_per_cta`, `local_stages`, `reorder_columns`, `warp_stages`,
`pipeline_warps`, `compute_unit`, `complex_multiply`, `cross_twiddle`,
`direct_boundary`, `local_exchange`, `shared_layout`, and `fft_core`.
Legal combinations depend on the backend and compiled generated points. Use
`butterfly_capabilities()` or `cubutterfly_bench --list-capabilities`, then let
plan construction validate the complete combination.

The associated enums expose:

- `ComputeUnit`: `Auto`, `Radix2`, `Radix4`, `Radix8`;
- `ComplexMultiply`: `FourMul`, `Gauss3`;
- `CrossTwiddleMode`: `Table`, `Recurrence`;
- `DirectBoundary`: `Strided`, `TiledTranspose`, `PrefixTiledTranspose`;
- `FftBoundaryResidency`: `GlobalScratch`, `Fused`;
- `LocalExchange`: `SharedMemory`, `WarpRegister`;
- `SharedLayout`: `Linear`, `XorSwizzle`;
- `FftCore`: `Scalar`, `ThreadDft8`, `CtaDft8`, `WmmaDft8`,
  `CufftDxBlock`, `CufftDxDirect`, `CufftDxResident`,
  `TurboFftGenerated`.

## ButterflyPlan

| Member | Contract |
|:--|:--|
| `ButterflyPlan(config)` | validates and resolves configuration; may allocate and initialize CUDA resources |
| `config()` | resolved immutable-shape configuration |
| `selection()` | automatic-selection decision, or a non-automatic record |
| `data_size()` | required byte extent of each input/output allocation |
| `workspace_size()` | minimum scratch bytes; zero means none |
| `set_stream(stream)` | binds subsequent work; may update an internal cuFFT plan |
| `stream()` | currently bound stream |
| `set_workspace(ptr, bytes)` | binds aligned caller-owned scratch or removes a binding with null |
| `workspace()` | active external or plan-owned workspace pointer |
| `execute_async(input, output)` | typed, allocation-free device submission without synchronization |
| `execute_zero_extended_async(input, output, logical_points, input_batch_stride, input_element_stride)` | advanced FP32 warp-register boundary submission; reads a logical FWHT/Structured input and writes the full physical transform |
| `execute(input, output, warmup, repeat)` | synchronous host-vector convenience and timing path |

The nine `execute_async` overloads accept `Fp16`, `Bf16`, `float`, `double`,
`Complex16`, `ComplexBf16`, `Complex32`, `Complex64`, or `std::uint32_t`.
The chosen overload must match `op` and `precision`. FP16/BF16 results are
narrowed at each logical butterfly output. On V100, BF16 `Native` is an
explicit FP32-compute-and-narrow emulation and benchmark CSV marks it with
`emulated_native=1`; it is not presented as native SM70 BF16 throughput.
`ButterflyStats` reports H2D, kernel and D2H milliseconds plus
transforms, butterflies, and points per second.

`execute_zero_extended_async` is a processing-unit interface used by the C
plan composition layer. It requires a forward FP32 warp-register FWHT or
Structured 2x2 plan, distinct device pointers, and a logical extent no larger
than `2^log_n`. Most applications should request embedding through the C plan
API so selection can keep the explicit pack path when fusion is slower.

CPU helpers `reference_fwht`, `reference_fft`, `reference_subset_zeta`,
`reference_superset_zeta`, `reference_structured_2x2`, and legacy
`reference_xor_zeta` provide small-scale
semantic references; they are not performance implementations.

## NTT Semantics And PlanConfig

| Field | Meaning |
|:--|:--|
| `log_n` | base-2 transform length |
| `batch` | number of contiguous transforms |
| `modulus` | prime modulus admitting the requested power-of-two root |
| `inverse` | forward or inverse transform |
| `input_order` | natural or a compatible APPT static input layout |
| `output_order` | natural, supported bit-reversed output, or APPT static output |
| `backend` | `Baseline`, `Tile256`, `Hybrid2D`, `CompactStage`, `StagePipeline`, `HybridDataflow`, `HierarchicalBarrier`, or `HierarchicalDataflow` |
| `appt_role_mapping` | independent producer/tail/writer service weights, fragment width, and writer tiles per CTA |
| `stage_space` | StagePipeline unfolding; zero selects its default |
| `stage_handoff` | `Atomic` or `NamedBarrier` |
| `n1_log` | Hybrid2D first factor; zero selects a device-derived default |
| `rows_per_block` | Hybrid2D row grouping; zero selects a device-derived default |
| `threads_per_block` | Hybrid2D CTA size; zero selects a device-derived default |
| `compute_unit` | `Auto`, `Radix2`, `Radix4`, or `Radix8` |
| `hierarchical_core` | `DataflowRadix4` default or generated `Hybrid2DRadix4` physical-unit ablation |
| `stage_partition` | ordered v0.7 subgraph stage counts; its length is the large-stage count `M` (`2..8`), each generic entry is `2..10`, and entries sum to `log_n` |
| `subgraph_mappings` | per-subgraph core, thread count, unfolding, and CTA service weight |
| `boundary_mappings` | per-logical-edge `FullScratch`, `Ring`, or `ResidentFused` realization and buffer count |
| `execution_group_mappings` | optional physical codelet mappings after resident-fused lowering; its count is physical `G`, not logical `M` |
| `ready_window` | readiness control; packet-streaming NTT uses `1/2/4/8/16` as 32-cycle polling-backoff quanta (default `2`), while other fixed-role kernels retain the scan-window value for future task stealing |
| `packet_readiness_mode` | experimental packet-streaming publication identity: `PerPacket` uses one word per packet; `WaveBitmap` retains packet identity as bits and polls four masked words per q-wave |
| `packet_compute_layout` | experimental online stage-0/1 mapping: `InterleavedRows` mixes four rows per warp; `WarpRows` assigns one row to each warp for a controlled bank/broadcast ablation |
| `packet_fold_wave_barriers` | experimental packet-core synchronization ablation; folds the end-of-wave barrier into the next wave's staging barrier and retains one final barrier (default `false`) |
| `word_bits` | physical word width, 32 or 64 |
| `cross_twiddle_placement` | `FirstPass`, `SecondPass`, `Fused`, or legacy `FusedBarrett` alias |
| `modular_multiply` | `Shoup` or `Barrett` |
| `output_order` | `Natural` or `BitReversed` |
| `auto_select` | use the calibrated runtime table |
| `auto_allocate_workspace` | create required plan-owned scratch |

`kDefaultModulus` is `1152921504606584833`. Input coefficients must be reduced
modulo the selected modulus.

`Plan::config()` preserves the resolved logical `stage_partition`, mappings,
and boundary policies. Resident-fused edges are lowered internally; the
resolved physical choices are exposed in `execution_group_mappings`. The CLI
prints `logical_subgraphs`, `execution_stage_partition`, `execution_groups`,
and `materialized_boundaries` so measurements cannot conflate `M` with `G`.

## NTT Plan

| Member | Contract |
|:--|:--|
| `Plan(config)` | validates modulus/shape/backend and initializes tables/resources |
| `config()` | resolved configuration |
| `selection()` | automatic-selection decision metadata |
| `points_per_transform()` | `2^log_n` |
| `data_size()` | bytes for all contiguous batches |
| `pipeline_trace()` | per-physical-execution-group `%globaltimer` start/end records after a v0.7 execution |
| `workspace_size()` | minimum scratch bytes |
| `set_stream`, `stream` | bind/query the CUDA stream |
| `set_workspace`, `workspace` | bind/query caller-owned or internal scratch |
| `execute_async(uint32_t*, uint32_t*)` | submit a 32-bit out-of-place transform |
| `execute_async(uint64_t*, uint64_t*)` | submit a 64-bit out-of-place transform |
| `execute(vector<uint64_t>, ...)` | synchronous host-vector convenience and timing path |

`RunStats` reports transfer/kernel milliseconds and kernel/end-to-end NTT and
point rates. `DeviceInfo current_device_info()` exposes the current device id,
name, compute capability, global memory, and SM count.

Number-theory helpers are `mod_pow`, `is_prime`,
`find_primitive_power_of_two_root`, and `reference_ntt`. Invalid mathematical
domains throw rather than returning sentinel values.

## SelectionInfo

Both plan families return `SelectionInfo`:

| Field | Meaning |
|:--|:--|
| `automatic` | automatic selection was requested and resolved |
| `calibrated` | decision is backed by the checked-in calibration table |
| `target` | hardware/selector target |
| `implementation` | selected backend and mapping identity |
| `confidence` | selector confidence label |
| `reason` | human-readable selection rationale |
| `predicted_kernel_ms` | table/model estimate when available |
