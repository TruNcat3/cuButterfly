/* SPDX-License-Identifier: BSD-3-Clause */
#ifndef CUFFT_LOW_PRECISION_BRIDGE_H
#define CUFFT_LOW_PRECISION_BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { CUFFT_LOW_FP16 = 1, CUFFT_LOW_BF16 = 2, CUFFT_LOW_ABI = 1 };
/* C2C, contiguous row-major, natural order, disjoint input/output, one GPU.
 * Direction: -1 forward, +1 inverse. Normalization must be zero (unscaled).
 * API return: 0 success; -1 invalid request, -2 unsupported GPU, -3 host OOM;
 * positive cuFFT status, or 1000 + cudaError_t. Diagnostics are thread-local.
 */
typedef struct {
    uint32_t abi_version;
    int32_t precision;
    int32_t rank;
    int32_t direction;
    uint64_t dimensions[2];
    uint64_t batch;
    int32_t normalization;
    uint32_t reserved;
} cufft_low_descriptor;

typedef struct {
    uint64_t total_elements;
    uint64_t workspace_bytes;
    int32_t device;
    int32_t input_type;
    int32_t output_type;
    int32_t execution_type;
} cufft_low_info;

int cufft_low_abi_version(void);
int cufft_low_header_version(void);
int cufft_low_cuda_header_version(void);
const char* cufft_low_last_error(void);
/* Validation/version functions never call CUDA or cuFFT. */
int cufft_low_validate(const cufft_low_descriptor* descriptor);
/* Creation owns workspace, binds stream, and may initialize CUDA/JIT cuFFT.
 * Create/destroy/warmup outside timing/capture. Caller selects the active GPU.
 * A nonzero opaque handle represents a plan, not a device pointer.
 */
int cufft_low_create(const cufft_low_descriptor* descriptor, uintptr_t stream, void** output);
int cufft_low_get_info(void* plan, cufft_low_info* output);
/* Execute only enqueues cufftXtExec on the creation stream. No scaling/copies.
 * Keep plan/stream/buffers alive for graph lifetime; serialize all plan uses.
 * Caller must synchronize and check asynchronous CUDA errors before destroy.
 */
int cufft_low_execute(void* plan, uintptr_t input, uintptr_t output);
int cufft_low_destroy(void* plan);
#ifdef __cplusplus
}
#endif
#endif
