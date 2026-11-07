# cuFFT Xt strategy audit

This note records what the BF16 cuFFT Xt baseline exposes and what can be
carried into cuButterfly's common search space. It does not identify or reuse
cuFFT's private arithmetic kernel. The API reference used by the local bridge
is the [CUDA 12.4 cuFFT documentation](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html).

## What the current baseline actually does

`benchmarks/cufft_low_precision/bridge.cu` creates one cuFFT plan with
`cufftXtMakePlanMany`, binds a caller-owned stream and workspace, and times a
complete `cufftXtExec`. The BF16 plan uses `CUDA_C_16BF` for input, output and
execution types. The bridge intentionally leaves the internal accumulation
contract unspecified; its result is a vendor reference, not a fixed-arithmetic
cuButterfly oracle.

The current BF16 cohort therefore does not use a cuFFTDx core. Its selected
cuButterfly mappings all report `fft_core = scalar`. The repository's cuFFTDx
adapter accepts native FP16, FP32 and FP64 only; BF16 remains an explicit
provider gap in [low-precision template coverage](low_precision_templates.md).

## Transferable strategy

| Xt capability | Common cuButterfly interpretation | Current status |
| --- | --- | --- |
| rank, extents, strides, distances and batch in `MakePlanMany` | Preserve the complete logical layout contract through lowering and include it in candidate identity | Public descriptors already carry rank, strides and batch; every lowering must retain them |
| explicit plan workspace and stream | Treat workspace size and lifetime as resource constraints, separate from measured execution time | Workspace is already queried and attachable; a policy dimension is still useful for bounded search |
| plan-time selection and cached execution | Compile and select before the timed graph, then replay the serialized mapping | Implemented by the measured registry and cache-only runner |
| load/store callbacks and JIT callbacks | Model boundary conversion, swizzle and fusion as explicit boundary policies | Not used by the BF16 bridge; requires a separate correctness and attribution experiment |
| linear, distributed and shuffled subformats | Represent producer/consumer layout and exchange as a composable boundary edge | Shared-memory and direct/global boundaries exist; Xt's exact internal distribution is opaque |
| multi-GPU placement | Treat device placement as a separate execution topology | Outside the current A100 single-device scope |

The practical lesson is that Xt's observable advantage is **plan-time
specialization plus a provider-specific core**, while its public API does not
expose radix, register allocation, shared-memory banking, warp scheduling or
Tensor Core use. Those details cannot be copied from a timing result.

## BF16 follow-up boundary

The vendor accepts BF16 storage and may use a mixed internal arithmetic path,
but that does not define a cuButterfly numerical contract. The existing
`accumulation=fp32` low-storage template widens each complex product and narrows
it before the butterfly additions; it is therefore not automatically
equivalent to the vendor path. A future BF16 provider/core experiment must
keep these rows separate:

1. native BF16 scalar arithmetic;
2. BF16 storage with the repository's explicit FP32 accumulation semantics;
3. any new vectorized BF16 core, with its own rounding and output contract.

Only a measured, correctness-checked adapter can enter the public registry.
Changing `fft_core` to `cufftdx-block` for BF16 without such an adapter would
silently change the arithmetic contract and is prohibited.

The current bounded cohort remains valid evidence for the hybrid organization:
the independent-column intervention improves selected large cases while the
scalar BF16 core remains the dominant gap to cuFFT Xt. This is a provider/core
limitation, not evidence against the dataflow abstraction.
