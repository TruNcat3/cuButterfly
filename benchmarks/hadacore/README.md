HadaCore external complete FWHT comparator
========================================

Official source: https://github.com/pytorch-labs/applied-ai/tree/2391954b19988bd76cf3c2ea84c1ce74b68d568b/kernels/cuda/inference/hadamard_transform

The checkout is pinned to `2391954b19988bd76cf3c2ea84c1ce74b68d568b`.
Upstream root LICENSE is BSD-3-Clause (Meta, 2024). The build copies its full
license into the output directory. CUDA source and original PyTorch entry point
hashes are pinned in `scripts/hadacore_baseline.py`. The generated translation
unit removes exactly two framework includes and prepends a minimal scalar enum
and CUDA error shim; every remaining upstream device/dispatch line is unchanged.
An independent C ABI and rescale kernel follow the original translation unit.
This is an external complete transform, not a cuButterfly core provider or an
exact-rounding reference.

### Optional local-core integration

The separate local tile ABI now exists in `scripts/hadacore_tile_provider.py`.
It reuses Tensor Core fragments, register-resident subtransforms and launch
parameters while the cuButterfly planner owns stage grouping, tile residency,
layout, boundary swizzles, dependencies and selection. The complete-transform
`hadacore_execute` ABI documented here remains an external comparator.

See the [local-core contract](../../docs/hadacore_local_core_contract.md) for
building the tile provider, paired native/MMA calibration and public AUTO
validation. Selection requires explicit `ALLOW_FP16_MMA` arithmetic and matching
target-local evidence. Preserve the upstream BSD-3-Clause notice and identify
local changes. Neither ABI silently replaces native per-stage FP16 arithmetic.

Upstream FP16 uses `mma.sync...f16.f16.f16.f16`, FP16 storage and coefficients,
and FP16 results/accumulators. Internal Tensor Core reduction order/precision is
hardware defined. The constants encode rounded `1/sqrt(2)`, `1/2`, `1/sqrt(8)`,
and `1/4`: native output is orthonormal FWHT. An additional timed kernel multiplies
its FP16 result by a FP32 `sqrt(N)` for unnormalized forward/inverse, or
`1/sqrt(N)` for normalized inverse, and rounds back to FP16. This matches the
mathematical transform and storage, with different grouping/coefficient rounding
from native radix-2 FP16 butterflies. Keep complete error distributions; do not
claim bitwise/stage-rounding equivalence. FP32 requests and BF16 are rejected.
Runner-owned generic output-scale/order/layout conversions remain runner costs;
this adapter rejects nonnatural orders and nonpacked layouts.

Upstream's Python API returns an allocated output (and may pad rows to a multiple
of 256 elements). This adapter directly calls its existing explicit-stream
`run_fht` into supplied output, with no allocation and no padding. Record it as
an external supplied-output integration, not native PyTorch API timing. For this
supported subset there is no necessary copy. In-place and out-of-place pointers
must agree with placement; partially overlapping pointers are rejected. Caller
must provide correct device/context, allocation sizes, 16-byte pointer alignment,
and keep buffers alive until stream completion. Set/reset in-place inputs outside
timing. Native dispatch still calls CUDA launch checks and (for large shared
memory kernels) `cudaFuncSetAttribute`; ordinary host call and CUDA event latency
are distinct metrics. CUDA Graph behavior remains unvalidated.

`build_core` projects placement from the actual packed scratch pointer aliasing:
the common runner can use separate scratch buffers for a user in-place request.
Its metadata preserves the user request in `semantic`, the projected request in
`core_semantic`, and both placement values in `placement_projection`. Direct
`Plan` callers still must match their supplied semantic placement exactly.

CPU-only build (one nvcc process; no GPU query, torch import or global install):
run these commands from the repository root with Python and CUDA 12.4 `nvcc`
available on `PATH`. For an existing official checkout, use its path with
`--upstream` and check out the same pinned commit before building.

```sh
mkdir -p external
git clone --no-checkout https://github.com/pytorch-labs/applied-ai.git external/hadacore
git -C external/hadacore checkout --detach 2391954b19988bd76cf3c2ea84c1ce74b68d568b
python scripts/hadacore_baseline.py \
  --upstream external/hadacore \
  --output build/hadacore \
  --nvcc "$(command -v nvcc)"
export CUBUTTERFLY_HADACORE_LIBRARY="$PWD/build/hadacore/libhadacore_baseline.so"
```

The default build emits SM80 native code plus compute80 PTX for forward JIT.
For PRO6000, pass the target CUDA 12.8 compiler explicitly with `--nvcc`; see the
[target extension runbook](../../docs/benchmarks/external-extensions.md#target-host-fft17-and-optional-mma9).
`build.json` records command, nvcc version, original hashes, generated files and
binary hash. First-load JIT and plan binding belong before warmup/capture/timing.
GPU correctness, runtime compatibility (including Blackwell PTX JIT), stream
behavior, graph capture and timing must still be measured on the selected GPU.

```python
from scripts.hadacore_baseline import support, build_core
s = dict(schema='cubutterfly-semantic-v1', operator=2, storage=0, compute=0,
         shape=[4096], batch=16, length_mode=0, placement=0, inverse=0,
         normalize_inverse=False, input_orders=[0], output_orders=[0])
assert support(s)['supported']
# Allocate source/output CUDA FP16 tensors and select the target device beforehand.
core = build_core(torch, s, source, output, stream=stream.cuda_stream)
core()  # Entire supplied-output transform and rescale are in this stream.
# Synchronize outstanding work before releasing the core and buffers.
core.close()
```

The quick GPU matrix retains N={256,4096,32768}, batch={1,16,256}, FP16/native,
forward unnormalized natural packed out-of-place (nine cases), plus the two
semantic controls. The broader CPU preparation expands the local-core space to
all N=2^k for k=1..15 and batch={1,2,4,16,64,256}; it also records the exact
upstream launch configurations for each size and strategy controls for native
radix2/radix4/radix8, FP16 register fragments and FP32 register shuffle. These
are preparation candidates, not claims that every external or uncompiled core
is executable. Preserve unsupported N=1 and fractional-256 batch requests as
gaps. Do not mix this low-precision cohort with FP32 aggregate speedups.

CPU check: `python -m pytest -q tests/test_hadacore_baseline.py`. These tests
verify semantic rejection, complete-batch limits, scale, pointer/stream contracts
and hash gating; they are not GPU correctness tests.
