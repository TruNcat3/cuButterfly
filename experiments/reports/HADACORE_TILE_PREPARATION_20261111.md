# HadaCore local-core preparation

The pinned HadaCore arithmetic was converted into an **optional local tile ABI**
without registering it in automatic selection. The generated provider accepts a
packed tile group, an explicit CUDA stream, caller-owned input and output
buffers, a tile count, and a bounded groups-per-block value. One block reuses its
shared scratch between successive groups. The parent cuButterfly schedule still
owns stage grouping, inter-group layout, boundary swizzle, and dependencies.

The implementation is in `scripts/hadacore_tile_provider.py` and
`benchmarks/hadacore/tile_bridge.cuh`. It is generated from upstream commit
`2391954b19988bd76cf3c2ea84c1ce74b68d568b`, whose CUDA source hash is pinned in
the existing comparator. The preparation build succeeded for SM80 with CUDA
12.4. Its generated build record is kept outside the repository at
`/home/wt/git/cuButterfly/build/hadacore_tile_20261111/build.json`.

The compiled population contains the upstream launch geometries for local
lengths `2^1` through `2^15`: short lengths use one warp and one or two chunks;
larger lengths use the seven upstream `(chunks, warps, blocks/SM)` tuples. PTXAS
reported 26–128 registers per thread for the instantiated entries. The largest
entry uses 128 registers; the build log reports no spill loads or stores. One
intermediate entry reports an 8-byte cumulative stack allocation, which remains
a resource field to verify on the target GPU.

The provider preserves HadaCore's FP16 MMA and final FP16 rounding contract. It
is not bitwise equivalent to cuButterfly's native per-stage FP16 rounding and is
therefore not a drop-in replacement. `local_tile_executable` remains false in
the public inventory until the bounded GPU validation completes. The CPU-only
validation entry point is `scripts/validate_hadacore_tile.py`; it prepares the
quick nine cases and the full 90-case population, writes deterministic FP16
inputs and same-contract radix-2 references, and records incomplete groups as
unsupported rather than padding them. Each quick case produces three explicit
GPU tasks (`groups_per_block` 1, 2, and 4), for 27 pending tasks in total; the
tasks retain the fixed target UUID and arithmetic contract.

The remaining acceptance steps are GPU-only: correctness and guard checks for
each launch geometry, repeated execution and graph replay, resource capture,
and a fixed-core organization comparison with the existing FWHT path. No timing
or performance claim is made by this preparation record.
