# Error Handling

cuButterfly is a C++ API and reports library failures with exceptions rather
than a `status_t` return value. CUDA work remains asynchronous, so application
code must also check its own CUDA API and synchronization results.

## Exception Categories

| Type | Meaning | Typical causes |
|:--|:--|:--|
| `std::invalid_argument` | caller contract or unsupported combination | invalid length/modulus, type mismatch, pointer aliasing, undersized workspace, unsupported backend/core/build/device |
| `std::overflow_error` | requested extent cannot be represented | batch, transform, or allocation byte overflow |
| `std::logic_error` | resolved/internal state cannot execute the request | required workspace missing or an unreachable dispatch point |
| `std::runtime_error` | CUDA, cuFFT, initialization, or generated-unit failure | allocation/copy/launch error, cuFFT status, missing optional processing unit |

The message contains the failed condition and, for wrapped CUDA/cuFFT calls,
the backend status. Exception text is diagnostic and is not a stable interface;
applications should branch on exception type, not message contents.

## Synchronous And Asynchronous Failures

Plan construction performs configuration validation and resource creation, so
most unsupported shapes and allocation failures are reported there.
`set_workspace` validates size and alignment. `execute_async` validates pointer
type, nullness, placement, and workspace presence and checks immediate launch
status.

A successful return from `execute_async` means work was enqueued, not that GPU
execution completed successfully. Check the CUDA operation that establishes
completion:

```cpp
try {
    plan.execute_async(input, output);
    const cudaError_t status = cudaStreamSynchronize(plan.stream());
    if (status != cudaSuccess) {
        // Handle asynchronous CUDA execution failure.
    }
} catch (const std::invalid_argument& error) {
    // Correct the plan or pointer contract.
} catch (const std::exception& error) {
    // Report an initialization/backend failure.
}
```

The host-vector `execute` path synchronizes internally and converts failures
from its CUDA operations to `std::runtime_error`.

## Recovery

Invalid configuration and workspace errors do not enqueue transform work. Build
a corrected plan or binding and retry. After an asynchronous CUDA failure,
follow CUDA Runtime recovery requirements; do not assume the plan or context is
reusable when the CUDA error invalidates the context.

There is currently no C ABI, no no-throw execution variant, and no library-wide
last-error state. Adding a status-returning facade is a future compatibility
feature, not part of the current ABI.
