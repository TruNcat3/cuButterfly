# xD lowering: lessons from related GPU FFT systems

This note records which lowering mechanisms we borrow from existing GPU FFT
systems and which parts remain cuButterfly-specific. The goal is to guide a
bounded implementation decision, not to claim that cuButterfly reproduces a
complete vendor FFT implementation.

## What the related systems do

| System | Lowering idea | Relevance to cuButterfly |
| --- | --- | --- |
| [cuFFT](https://docs.nvidia.com/cuda/cufft/) | A multidimensional plan owns the complete transform. `cufftPlanMany` expresses strides, batches, and embeddings; planning may choose several kernels and intermediate buffers. | The public descriptor must own the complete xD contract. Per-axis calls are an implementation choice, not the application API. |
| [VkFFT](https://github.com/DTolm/VkFFT) and its [white paper](https://ieeexplore.ieee.org/document/10036080) | Generates a kernel sequence per axis, chooses radix and axis splits, keeps transforms in place when possible, and minimizes extra transposition uploads. Large transforms use a four-step decomposition and only introduce a reshuffle when the tile cannot remain local. | The useful lesson is to make layout transitions a first-class cost and to keep an axis tile in the same physical order across adjacent stages. A generic scalar permutation is an intentionally conservative fallback. |
| [TurboFFT](https://arxiv.org/abs/2412.05824) | Tiles a large signal into a shared-memory cube, applies threadblock and warp level FFTs in stages, uses architecture-aware parameters, and generates a small family of shape-specific kernels. Its register-to-shared store uses a swizzled order to avoid padding and bank conflicts. | This directly motivates `tile_shape`, `warp_tile`, and `stage_group` as search variables. It also shows why a bank-conflict-free layout must be represented in the physical mapping rather than hidden inside a cost constant. |
| [tcFFT](https://arxiv.org/abs/2104.11471) | Fuses multiple merging steps, changes the in-place layout between iterations, and chooses a continuity size that balances coalescing against occupancy. It replaces a fixed padded shared layout with a changing layout when useful. | Our offset layout and resident subgraph can express the same tradeoff. The missing piece is to carry the layout state between xD axis groups instead of materializing a full permutation after every axis. |
| [cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/) | Exposes device-side FFT building blocks so an application can place a local FFT inside its own block and shared-memory schedule. | This is a processing-unit source for cuButterfly. It should be imported as a local core inside a cuButterfly tile, not treated as an xD lowering by itself. |
| [FFTX/SpectralPack](https://www.spiral.net/doc/papers/hipc_2018.pdf) | Uses a symbolic transform algebra and code generation to combine transform stages, data movement, and application operations. | This supports a declarative lowering IR for butterfly stages and layout transitions. It does not provide a ready-made CUDA xD kernel for our six operator families. |
| [FlashFFTConv](https://arxiv.org/html/2311.05908v1) | Uses a factorized long-FFT formulation and fuses layout movement with surrounding computation, with hardware-specific choices for the factorization. | It motivates treating factorization and materialized boundaries as search variables. Its tensor-core and convolution assumptions do not transfer directly to FP64 or NTT. |
| [Welder](https://www.usenix.org/conference/osdi23/presentation/shi) and [TensorIR](https://doi.org/10.1145/3575693.3576933) | Represent tiled operator graphs and separate local tensorization from outer loop/memory mapping. | A butterfly-specific IR should expose partner indices, stage order, and readiness edges explicitly; ordinary affine bounding boxes are insufficient for bit permutations. |

## Changes applied in 0.9.1

The old rank-N permutation kernel decoded every output coordinate in one thread.
For the common contiguous rank-2 axis exchange, the library now lowers the
boundary to a 32x32 shared-memory tile with an extra shared column. Loads and
stores are coalesced, and the extra column prevents the classic transpose bank
conflict. Higher ranks, non-contiguous strides, and unusual permutations retain
the general kernel and its correctness contract.

This is deliberately a local improvement. It does not claim that a padded tile
is always optimal: TurboFFT's swizzled stores can save shared capacity when the
device and element size make padding expensive. The next searchable choice is
therefore:

```text
boundary_layout = {scalar_general, tiled_padded, tiled_swizzled}
```

The choice must be made together with tile shape, element size, and resident
occupancy. It cannot be hard-coded as a universal winner.

## Lowering plan for the next bounded step

The evidence suggests a three-level lowering that is compatible with the
cuButterfly model:

1. **Local core.** Select a radix/codelet or an imported cuFFTDx core for one
   contiguous tile. Its register footprint and arithmetic precision stay in the
   processing-unit contract.
2. **Resident stage group.** Fuse consecutive stages and axes while their live
   tile fits shared memory. Keep the producer layout as the consumer layout, or
   apply a swizzled shared-to-register exchange at the boundary.
3. **Global tile stream.** For larger transforms, stream tiles through HBM with
   one fused read, stage group, and write transaction. A global permutation is
   emitted only when the next stage cannot consume the current layout.

The planner should enumerate a small finite set of tile shapes and axis orders,
then measure complete execution including the remaining boundaries. This is
the same separation used elsewhere in cuButterfly: operator semantics define
the graph, the local core defines arithmetic, and the mapping defines tile
residency and handoff. It is not exhaustive global optimization.

## Current boundary and evidence

The current implementation has the first local step and a resident stage-group
candidate for small packed power-of-two tensors. It now has the tiled rank-2
boundary described above. It does not yet have cross-axis streaming overlap,
swizzled shared layouts, or a multi-tile FFT kernel for transforms larger than
one resident CTA. Those are separate changes because they alter workspace,
occupancy, and replay contracts.

The resulting lowering sketch is therefore:

```text
operator graph
    -> stage-aware tile sketch
    -> local core + resident stage group
    -> {tiled shared exchange | fused writer layout | global boundary}
    -> next stage group
```

The search should rank complete sketches, including exchange traffic and
occupancy, instead of assigning a fixed bonus to a resident or fused label.
This is the specific lesson shared by VkFFT, TurboFFT, tcFFT, and the generic
tile-graph systems above.

The current selector now takes the first bounded step in that direction: for
two or three complex axes it measures the historical order and one alternate
complex-axis order, then records the winner. The R2C/C2R boundary remains
fixed, so compact real transforms only reorder the remaining complex axes.
This is intentionally a small search space; a full factorial permutation
search, tile shape, writer-fused layout, and persistent ring edges remain
future candidate families.

The accepted correctness and performance evidence is collected in
[xD lowering v1 validation](../results/paper_profile_rc2/end_to_end/xd_lowering_v1.md).
The xD API and candidate eligibility remain in the
[xD lowering guide](xd_lowering.md).

## References

The links above point to the official CUDA documentation, the VkFFT repository
and white paper, and the primary arXiv papers for TurboFFT and tcFFT. They are
method references. Performance claims in this repository continue to use the
versioned raw measurements, hardware UUID, and timing protocol in the acceptance
report rather than values copied from those works.
