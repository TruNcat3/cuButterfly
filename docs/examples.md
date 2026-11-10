# Examples

Examples are built by default with `CUBUTTERFLY_BUILD_EXAMPLES=ON`:

```bash
cmake --build build --target \
  cubutterfly_basic_fft_example \
  cubutterfly_device_api_example \
  cuntt_device_api_example
```

They deliberately use processing units available with the CUDA Toolkit-only
build. Optional cuFFTDx and TurboFFT examples remain benchmark commands because
their legal generated points depend on build configuration.

## Basic FFT

[`examples/basic_fft.cpp`](../examples/basic_fft.cpp) demonstrates the shortest
host-vector workflow: configure a batched FP32 FFT, construct a plan, call the
synchronous convenience API, validate one result, and read kernel timing.

```bash
./build/cubutterfly_basic_fft_example
```

Use this path for API evaluation and correctness checks. Production pipelines
should use device pointers to avoid staging and synchronization.

## Butterfly Device API

[`examples/device_api.cpp`](../examples/device_api.cpp) demonstrates a
production-style FWHT submission with:

- automatic V100 mapping selection;
- a nonblocking caller-owned CUDA stream;
- device input and output allocations;
- queried, caller-owned workspace;
- asynchronous copies and transform submission.

```bash
./build/cubutterfly_device_api_example
```

Because automatic selection is deliberately strict, this example requires a
covered V100 workload. For another GPU, replace `auto_select = true` with a
backend and mapping supported by `--list-capabilities`.

## NTT Device API

[`examples/ntt_device_api.cpp`](../examples/ntt_device_api.cpp) demonstrates the
64-bit NTT pointer contract and the same stream/workspace lifecycle using the
portable Tile256 backend.

```bash
./build/cuntt_device_api_example
```

## Common Integration Patterns

For multiple CUDA streams, create one plan per stream. Plans may share an
external memory pool, but the same workspace region cannot back overlapping
executions. Record an event after `execute_async` when downstream work is on a
different stream; use `cudaStreamWaitEvent` rather than a device-wide
synchronization.

For repeated shapes, retain the plan and allocations. Reconstruct only when
length, batch, precision, layout, direction, or physical mapping changes.

See [Programming Guide](programming_guide.md) for the complete lifecycle and
[Error Handling](error_handling.md) for exception and asynchronous-error rules.
