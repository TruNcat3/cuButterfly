# Restricted external tcFFT baseline

This adapter preserves the external tcFFT FFT arithmetic. It is separate from
libcuntt and does not register a cuButterfly core or change its build identity.
The official checkout is `external/tcFFT`, remote
<https://github.com/rox906/tcFFT>, commit
`3e87a7083577741815e21b2a3c2bed22520786ce`. Keep its GPLv3 `LICENSE` with the source
and derived build artifacts.
The build verifies the commit and SHA-256 of the 1D source and header.

## Audited support

The upstream `tcfftCreate` switch initializes only these 1D lengths:
256, 512, 1024, 131072, 262144, 524288, 16777216, 33554432, 67108864,
134217728. Its default case leaves plan fields uninitialized. In particular,
4096, 65536 and arbitrary powers of two are unsupported by this adapter.
The repository also has a separate 2D implementation; this adapter does not
expose it, so it makes no claim about that implementation's support domain.

The first layer processes 8192 complex elements per CTA without bounds checks.
`N * batch` must be divisible by 8192 and fit signed int. Thus minimum batches
are 32, 16 and 8 for N=256, 512 and 1024. No extra transforms or padding stand in
for a requested batch. Later launch dimensions must have grid.y <= 65535.
For multi-stage plans with first two merged radices r0 and r1, this means
`N * batch / (r0 * r1) <= 65535`. Resource and device limits still require
runtime verification. Large supported lengths may need substantial memory:
device scratch costs `4*N*batch` bytes and the index table `4*N` bytes;
creation temporarily holds two host index arrays of `4*N` bytes each.

The exposed semantics are contiguous rank-1 complex-to-complex, interleaved
FP16 real/imaginary pairs, forward negative exponential, no normalization,
natural input and natural output. Upstream's own test wrapper applies a
mixed-radix permutation before the FFT and reads natural output. The adapter
reproduces that permutation on the device for every execution. Input and output
may alias because a private scratch buffer holds the transformed data.
Inverse, normalization, strided layouts, FP32 accumulation, BF16 and rank>1
requests are rejected by the CPU support checker rather than substituted.

`tcfft_half.cu` declares half matrix operands and half WMMA accumulator
fragments (`complex_mul`, `complex_mul_acc`). Twiddle multiplications and
ordinary sums also use half operands; some twiddle coefficients are computed in
float and converted to half before multiplication. This is an FP16 accumulator
type, not a claim of FP32 accumulation or a bit-exact description of tensor-core
internal partial sums. Numerical error and overflow must be checked on the GPU.

## Build Without GPU Initialization

From the repository root:

```sh
python scripts/tcfft_baseline.py --n 131072 --batch 1
python scripts/tcfft_baseline.py --build build/tcfft_sm80
python -m pytest -q tests/test_tcfft_baseline.py
```

The builder uses `/usr/local/cuda-12.4/bin/nvcc`, C++14 and SM80. It writes
`libtcfft_baseline.so`, `tcfft_stream.cu` and `build.json` in the specified build
directory. The CLI does not expose an nvcc, upstream-directory or architecture
option. Its default requires nvcc at that exact path and the pinned checkout at
`external/tcFFT`. Python callers may pass `nvcc` and `upstream` to `build()`;
the compiler target remains SM80. Compilation does not execute a CUDA binary or
initialize a GPU.
The generated translation unit has three host-only differences: rename the
execution entry point and accept a stream; add the stream to both upstream
launch sites; move both function-attribute settings from execution into bridge
plan creation. The appended bridge also replaces the upstream initializer's
V100 function pointers with the existing upstream `_A100` kernels for SM80.
These kernels use the Ampere WMMA fragment layout. Plan creation rejects devices
whose compute-capability major version is not 8. All upstream device-kernel
bodies remain unchanged; selecting the A100 variants does not establish their
numerical correctness. The untouched clone is the source of record.

## GPU Integration Contract

```python
from scripts.tcfft_baseline import Plan
plan = Plan("build/tcfft_sm80/libtcfft_baseline.so", 131072, 1)
plan.execute(input_device_pointer, output_device_pointer, stream=user_stream_pointer)
# Synchronize the caller's stream before closing the plan.
plan.close()
```

Device pointers and streams are integer addresses. The caller owns their
lifetimes and the active CUDA device. The stream argument is mandatory; passing
0 explicitly selects the default stream. There is no silent default-stream
fallback. A plan has mutable scratch and cannot run concurrently on different
streams. Create and destroy outside timing and graph capture. Plan creation
allocates and copies coefficients/index tables using the upstream initializer.
Execution enqueues device reorder, upstream FFT and device-to-device output
copy on the supplied stream, with no allocation, host synchronization or
function-attribute updates. All three operations belong inside the timed call
and any captured graph. Timing must use events on the same supplied stream.
The call returns enqueue errors; the caller must also check asynchronous errors
at synchronization. The finite comparison runner below supplies timing.

GPU validation remains pending: compare all exposed lengths/batch boundaries
against a matched FP16 numerical reference, record overflow and rounding error,
verify impulse and nontrivial inputs for permutation/sign, verify distinct data
on repeated calls and input/output aliasing, test a nondefault stream, and
capture/replay the complete call with changed inputs and restored inputs.
Keep plan, input and output addresses alive for the graph lifetime. Check graph
output ownership across replays. CPU compilation alone does not establish
correctness, graph support, performance or acceptance for a benchmark matrix.

## Finite comparison preparation and execution

`scripts/run_tcfft_comparison.py` compares complete public cuButterfly DEFAULT,
cuFFT Xt FP16 and tcFFT calls. Its configuration supplies absolute paths for
`public_manifest`, `tcfft_library`, `cufft_library` and a `gpu_uuid`. Optional
`library`, `registry`, `profile_root`, `module_cache`, `template_root` and
`mathdx_root` override the public manifest's runtime paths. Match the library,
calibration and device identities before interpreting performance.

```sh
python scripts/run_tcfft_comparison.py --prepare --config CONFIG.json --output NEW_OUTPUT
python scripts/run_tcfft_comparison.py --run --output NEW_OUTPUT --case n262144-b4
python scripts/run_tcfft_comparison.py --run --output NEW_OUTPUT --case n262144-b32
```

Preparation runs on the CPU. It reuses the frozen nine-request cohort's input
and reference files, preserving input seed, scale and file hashes. Only
`N=262144, batch=4/32` intersect tcFFT's supported lengths. Preparation computes
the explicit per-stage FP16 rounded reference for these two requests, saves
`.npy` files under `prepared/`, and binds their hashes in the manifest. Execution
loads those references without recomputing them while holding the GPU. The
other seven requests retain unsupported metadata and do not execute on the GPU.
Preparation refuses to overwrite an existing manifest; use a new output
directory when code, configuration or binaries change.

Public DEFAULT receives no explicit mapping. The runner records the selected
mapping and selection reason, backend identities, full-call timings and CPU
and external-backend error reports. It uses the shared complete-call Graph
timing helper. Before timing, it checks input immutability, output guards,
output ownership after plan destruction, and capture/replay agreement. Caller
output buffers are poisoned between replays; private kernel scratch and device
arithmetic are unchanged. Floating-point errors remain report-only with no
new or relaxed error threshold.

## Current validation limitation

An independent SM80 diagnostic at N=262144, batch=4 now produces byte-identical
outputs across repeated eager execution, plan recreation and two Graph replays.
Its relative L2 error against the prepared FP16 reference is about 0.005798.
This is repeatability and error evidence, not numerical acceptance or a timing
result. The raw record is
`results/comparator_extension_20261111/tcfft_ampere_diagnostic/diagnostic.json`.

The November 11, 2026 retest in
`results/comparator_extension_20261111/tcfft_ampere_prepared/n262144-b4/comparison.json`
uses the upstream A100 dispatch and still reports tcFFT
`ValueError('direct/graph mismatch')`. Output lifetime before timing passed.
The shared error text does not distinguish the recreated-plan eager check from
the later Graph check, so the cause remains unknown. This request has not passed
the measurement prerequisites and must not enter performance comparisons.
Selecting the A100 kernels is a host dispatch correction, not evidence that
numerical correctness or Graph replay has been repaired. Broader length/batch
validation remains open.
