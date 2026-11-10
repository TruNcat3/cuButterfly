// SPDX-License-Identifier: BSD-3-Clause
#include "bridge.h"
#include <cuda_runtime_api.h>
#include <cufftXt.h>
#include <cstdio>
#include <limits>
#include <new>

#if CUDART_VERSION < 12040
#error "This adapter contract is audited against CUDA 12.4 or later headers"
#endif
static_assert(sizeof(cufft_low_descriptor) == 48, "descriptor ABI size changed");
static_assert(sizeof(cufft_low_info) == 32, "info ABI size changed");

namespace {
thread_local char diagnostic[256] = "";
struct Plan {
    cufftHandle handle = 0;
    void* workspace = nullptr;
    cufft_low_info info{};
    int direction = CUFFT_FORWARD;
};
int fail(int code, const char* operation) {
    std::snprintf(diagnostic, sizeof(diagnostic), "%s (status %d)", operation, code);
    return code;
}
void release(Plan* plan) {
    if (plan->handle) cufftDestroy(plan->handle);
    if (plan->workspace) cudaFree(plan->workspace);
    delete plan;
}
}

extern "C" int cufft_low_abi_version() { return CUFFT_LOW_ABI; }
extern "C" int cufft_low_header_version() { return CUFFT_VERSION; }
extern "C" int cufft_low_cuda_header_version() { return CUDART_VERSION; }
extern "C" const char* cufft_low_last_error() { return diagnostic; }

extern "C" int cufft_low_validate(const cufft_low_descriptor* d) {
    if (!d || d->abi_version != CUFFT_LOW_ABI || d->reserved || d->normalization ||
        (d->precision != CUFFT_LOW_FP16 && d->precision != CUFFT_LOW_BF16) ||
        (d->rank != 1 && d->rank != 2) || (d->direction != -1 && d->direction != 1) ||
        !d->batch || d->batch > UINT32_MAX)
        return fail(-1, "invalid descriptor, precision, rank, direction, normalization or batch");
    uint64_t total = d->batch;
    for (int axis = 0; axis < d->rank; ++axis) {
        const auto n = d->dimensions[axis];
        if (!n || (n & (n - 1)) || n > UINT32_MAX / total)
            return fail(-1, "dimensions must be powers of two; batch*volume must be below 2^32");
        total *= n;
    }
    if ((d->rank == 1 && d->dimensions[1] != 0) ||
        total > std::numeric_limits<size_t>::max() / 4)
        return fail(-1, "unused dimension or buffer byte size is invalid");
    diagnostic[0] = '\0';
    return 0;
}

extern "C" int cufft_low_create(const cufft_low_descriptor* d, uintptr_t stream, void** output) {
    if (!output) return fail(-1, "null plan output");
    *output = nullptr;
    if (const auto status = cufft_low_validate(d)) return status;
    auto plan = new (std::nothrow) Plan;
    if (!plan) return fail(-3, "host plan allocation failed");
    cudaDeviceProp props{};
    auto cuda_status = cudaGetDevice(&plan->info.device);
    if (cuda_status == cudaSuccess) cuda_status = cudaGetDeviceProperties(&props, plan->info.device);
    if (cuda_status != cudaSuccess) { release(plan); return fail(1000 + cuda_status, "query CUDA device"); }
    const int minimum_sm = d->precision == CUFFT_LOW_FP16 ? 53 : 80;
    if (props.major * 10 + props.minor < minimum_sm) {
        release(plan); return fail(-2, "GPU architecture is below vendor low-precision minimum");
    }
    auto status = cufftCreate(&plan->handle);
    if (status == CUFFT_SUCCESS) status = cufftSetAutoAllocation(plan->handle, 0);
    long long dimensions[2] = {static_cast<long long>(d->dimensions[0]), static_cast<long long>(d->dimensions[1])};
    uint64_t volume = d->dimensions[0] * (d->rank == 2 ? d->dimensions[1] : 1);
    const auto type = d->precision == CUFFT_LOW_FP16 ? CUDA_C_16F : CUDA_C_16BF;
    size_t workspace_bytes = 0;
    if (status == CUFFT_SUCCESS) status = cufftXtMakePlanMany(plan->handle, d->rank, dimensions,
        dimensions, 1, static_cast<long long>(volume), type,
        dimensions, 1, static_cast<long long>(volume), type,
        static_cast<long long>(d->batch), &workspace_bytes, type);
    if (status != CUFFT_SUCCESS) { release(plan); return fail(status, "cuFFT Xt plan creation"); }
    if (workspace_bytes) {
        cuda_status = cudaMalloc(&plan->workspace, workspace_bytes);
        if (cuda_status != cudaSuccess) { release(plan); return fail(1000 + cuda_status, "allocate cuFFT workspace"); }
        status = cufftSetWorkArea(plan->handle, plan->workspace);
    }
    if (status == CUFFT_SUCCESS) status = cufftSetStream(plan->handle, reinterpret_cast<cudaStream_t>(stream));
    if (status != CUFFT_SUCCESS) { release(plan); return fail(status, "bind cuFFT workspace/stream"); }
    plan->info.total_elements = volume * d->batch;
    plan->info.workspace_bytes = workspace_bytes;
    plan->info.input_type = plan->info.output_type = plan->info.execution_type = type;
    plan->direction = d->direction;
    *output = plan;
    return 0;
}

extern "C" int cufft_low_get_info(void* handle, cufft_low_info* output) {
    if (!handle || !output) return fail(-1, "null plan/info");
    *output = static_cast<Plan*>(handle)->info;
    return 0;
}

extern "C" int cufft_low_execute(void* handle, uintptr_t input, uintptr_t output) {
    if (!handle || !input || !output || input % 4 || output % 4)
        return fail(-1, "nonzero aligned plan/buffer addresses required");
    auto plan = static_cast<Plan*>(handle);
    const auto bytes = plan->info.total_elements * 4;
    const auto distance = input > output ? input - output : output - input;
    if (distance < bytes || input > UINTPTR_MAX - bytes || output > UINTPTR_MAX - bytes)
        return fail(-1, "input/output buffers overlap or address range overflows");
    const auto status = cufftXtExec(plan->handle, reinterpret_cast<void*>(input),
                                   reinterpret_cast<void*>(output), plan->direction);
    return status == CUFFT_SUCCESS ? 0 : fail(status, "cuFFT Xt execution");
}

extern "C" int cufft_low_destroy(void* handle) {
    if (!handle) return 0;
    auto plan = static_cast<Plan*>(handle);
    const auto status = cufftDestroy(plan->handle);
    const auto cuda_status = plan->workspace ? cudaFree(plan->workspace) : cudaSuccess;
    delete plan;
    if (status != CUFFT_SUCCESS) return fail(status, "destroy cuFFT plan");
    return cuda_status == cudaSuccess ? 0 : fail(1000 + cuda_status, "free cuFFT workspace");
}
