# Native low-storage cuFFT baseline

This independent bridge calls NVIDIA cuFFT Xt with FP16 or BF16 complex input
and output. It adds no conversion, padding, normalization or replacement FFT.
The high-precision adapter and cuButterfly library build remain separate.
The bridge is ready for GPU validation; CPU compilation establishes no
correctness, numerical error, Graph replay or performance result.

## Precision and support

The contract follows the [CUDA 12.4 cuFFT documentation](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html),
sections [2.3.1](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html#half-precision-cufft-transforms),
[2.3.2](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html#bfloat16-precision-cufft-transforms),
and [cufftXtMakePlanMany](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html#c.cufftXtMakePlanMany).
Local audited headers are `/usr/local/cuda-12.4/include/cufftXt.h`,
`library_types.h` and `cufft.h`: CUDA 12.4, cuFFT headers 11.2.0.44.

| Storage | Input/output/execution type | Arithmetic claim | Vendor minimum |
| --- | --- | --- | --- |
| FP16 | `CUDA_C_16F` for all three | Vendor-native FP16 FFT; internal accumulators unspecified | SM53 |
| BF16 | `CUDA_C_16BF` for all three | Vendor-native mixture of single and BF16 operations | SM80 |

The Xt API requires matching precision across input, output and execution type.
`compute="vendor-native"` and `accumulation="vendor-native"` describe that
restriction. Requests for an independently chosen FP32 execution or accumulator
are rejected. BF16 mixed arithmetic does not establish an all-FP32 accumulator
contract. FP16 storage does not establish that every internal partial sum is FP16.
This baseline belongs to `fft-l` and must retain its vendor arithmetic label.

This adapter exposes single-GPU contiguous row-major natural-order C2C,
out-of-place, rank 1 or 2, positive power-of-two dimensions, positive batch,
and `batch*volume < 2^32` complex elements. The final bound is a conservative
adapter interpretation of the documented four-billion-element limitation and
also checks buffer-byte overflow. Source and destination ranges must be disjoint
and four-byte aligned. Each complex value contains two consecutive 16-bit
components in the selected format. Actual device memory and cuFFT plan
acceptance remain runtime requirements. No input range or error threshold is
silently imposed; FP16 overflow is possible and must be reported.

The vendor also supports other ranks/layouts; this bridge rejects rank 3,
strided/padded arrays, real transforms, in-place calls and non-natural order.
The vendor restriction on real-part strides for low-precision R2C/C2R does not
imply that all vendor C2C strides are unsupported. Our contiguous boundary is
an adapter scope choice.

Both forward (negative exponent) and inverse (positive exponent) are unscaled.
cuFFT leaves normalization to the caller. This version rejects every requested
normalization, including normalized inverse; it cannot substitute an unscaled
result for that request. A future normalization implementation must include its
complete scaling/rounding overhead in every execution and captured Graph.

## Build and CPU checks

```sh
python scripts/cufft_low_precision_baseline.py --shape 64 1024 --precision bf16
python scripts/cufft_low_precision_baseline.py --build build/cufft_low_precision_sm80
python -m pytest -q tests/test_cufft_low_precision_baseline.py
```

The optional builder invokes NVCC 12.4 with SM80, shared CUDA runtime and
dynamic cuFFT linking. It never executes a CUDA binary or creates a context.
`build.json` records compiler command/version, source hashes, bridge hash,
linked vendor library hash and pending GPU validation. The SM80 build specifies
the target experiment; the bridge contains host API calls, with FFT device code
supplied by cuFFT. Import, capability queries and ABI validation also use no GPU.
Native CPU tests interpose CUDA initialization/plan creation with aborting stubs.

## GPU integration

```python
from scripts.cufft_low_precision_baseline import Plan

plan = Plan("build/cufft_low_precision_sm80/libcufft_low_precision_baseline.so",
            [64, 1024], 16, precision="bf16", stream=user_stream_pointer)
# Warm up and synchronize this stream before timing/capture.
plan.execute(input_device_pointer, output_device_pointer, stream=user_stream_pointer)
# Synchronize all outstanding executions and destroy captured graphs first.
plan.close()
```

Device buffers and stream are caller-owned integer addresses; stream 0 is
permitted only when explicitly supplied. Creation selects no device: the caller
must already select the intended GPU. Creation checks its architecture, creates
the Xt plan with automatic allocation disabled, allocates/binds its workspace,
and binds the supplied stream. Create, warmup and destroy belong outside timing
and capture. A plan owns mutable scratch and must be used serially on its bound
stream. `execute` only enqueues `cufftXtExec`; it performs no allocation, copy,
host synchronization, stream reassignment or normalization. Timings and Graphs
must include the complete call. Caller synchronization must check asynchronous
errors before using results or releasing buffers/workspace.

[cuFFT documents CUDA Graph support for single-GPU plans](https://docs.nvidia.com/cuda/archive/12.4.0/cufft/index.html#cuda-graphs-support).
This bridge uses no callbacks. Keep plan, stream, workspace, input and output
alive for the Graph lifetime. Actual capture/replay testing is pending, including
nondefault streams, changed/restored inputs, input preservation, output lifetime,
sign/order, finite values and same-semantics numerical references. CPU support
success is a static capability result, not a measured GPU acceptance.

## Source and license provenance

`bridge.h`, `bridge.cu`, the test interposer and Python binding are original
project adapter code under the repository BSD-3-Clause license. No NVIDIA
implementation or sample device arithmetic is copied. NVIDIA headers are included
from the installed CUDA Toolkit; cuFFT remains a dynamically linked external
library under the [CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html).
The builder records the external binary path/hash rather than redistributing it.
Preserve the repository license with bridge redistribution and follow NVIDIA's
terms for any separate vendor-library distribution.
