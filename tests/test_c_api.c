#include <cubutterfly/cubutterfly.h>

#include <cuda_runtime_api.h>

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUB(call) do { \
    cubutterflyStatus_t status_ = (call); \
    if (status_ != CUBUTTERFLY_STATUS_SUCCESS) { \
        fprintf(stderr, "%s failed: %s\n", #call, cubutterflyGetStatusString(status_)); \
        return 1; \
    } \
} while (0)

#define CHECK_CUDA(call) do { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(status_)); \
        return 1; \
    } \
} while (0)

static void log_message(cubutterflyLogLevel_t level, const char* event,
                        const char* message, void* user_data) {
    int* warnings = (int*)user_data;
    if (level == CUBUTTERFLY_LOG_WARNING)
        ++*warnings;
    (void)event;
    (void)message;
}

typedef struct test_complex32 {
    float real;
    float imag;
} test_complex32_t;

static int run_arbitrary_fft(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extent = 5;
    test_complex32_t original[5] = {{0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}};
    test_complex32_t spectrum[5];
    test_complex32_t recovered[5];
    test_complex32_t* input = NULL;
    test_complex32_t* output = NULL;
    test_complex32_t* result = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t physical_extent = 0;
    size_t workspace_bytes = 0;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FFT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_COMPLEX_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
                                            "bluestein-cufft-power2-core"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetPhysicalExtents(forward, &physical_extent, 1));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    if (physical_extent != 16 || workspace_bytes < physical_extent * sizeof(test_complex32_t))
        return 1;
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(spectrum)));
    CHECK_CUDA(cudaMalloc((void**)&result, sizeof(recovered)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, output));
    CHECK_CUDA(cudaMemcpyAsync(spectrum, output, sizeof(spectrum), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t k = 0; k < extent; ++k) {
        const double angle = -2.0 * 3.14159265358979323846 * (double)k / (double)extent;
        if (fabsf(spectrum[k].real - (float)cos(angle)) > 2.0e-5f ||
            fabsf(spectrum[k].imag - (float)sin(angle)) > 2.0e-5f)
            return 1;
    }
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, output, result));
    CHECK_CUDA(cudaMemcpyAsync(recovered, result, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < extent; ++i) {
        if (fabsf(recovered[i].real - original[i].real) > 2.0e-5f ||
            fabsf(recovered[i].imag - original[i].imag) > 2.0e-5f)
            return 1;
    }
    cudaFree(workspace);
    cudaFree(result);
    cudaFree(output);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_2d_arbitrary_fft(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {3, 5};
    test_complex32_t original[15] = {{0, 0}};
    test_complex32_t spectrum[15];
    test_complex32_t recovered[15];
    test_complex32_t* input = NULL;
    test_complex32_t* output = NULL;
    test_complex32_t* result = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t physical[2] = {0, 0};
    size_t workspace_bytes = 0;
    original[1 * extents[1] + 2].real = 1.0f;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FFT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_COMPLEX_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetPhysicalExtents(forward, physical, 2));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    if (physical[0] != extents[0] || physical[1] != extents[1])
        return 1;
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(spectrum)));
    CHECK_CUDA(cudaMalloc((void**)&result, sizeof(recovered)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, output));
    CHECK_CUDA(cudaMemcpyAsync(spectrum, output, sizeof(spectrum), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t row = 0; row < extents[0]; ++row) {
        for (size_t col = 0; col < extents[1]; ++col) {
            const size_t index = row * extents[1] + col;
            const double angle = -2.0 * 3.14159265358979323846 *
                ((double)row / (double)extents[0] + 2.0 * (double)col / (double)extents[1]);
            if (fabsf(spectrum[index].real - (float)cos(angle)) > 4.0e-5f ||
                fabsf(spectrum[index].imag - (float)sin(angle)) > 4.0e-5f)
                return 1;
        }
    }
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, output, result));
    CHECK_CUDA(cudaMemcpyAsync(recovered, result, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 15; ++i) {
        if (fabsf(recovered[i].real - original[i].real) > 4.0e-5f ||
            fabsf(recovered[i].imag - original[i].imag) > 4.0e-5f)
            return 1;
    }
    cudaFree(workspace);
    cudaFree(result);
    cudaFree(output);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_2d_power_fft(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {4, 8};
    test_complex32_t original[32];
    test_complex32_t recovered[32];
    test_complex32_t* input = NULL;
    test_complex32_t* spectrum = NULL;
    test_complex32_t* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    for (size_t i = 0; i < 32; ++i) {
        original[i].real = (float)((i * 7) % 13) / 13.0f;
        original[i].imag = (float)((i * 5) % 11) / 17.0f;
    }
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FFT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_COMPLEX_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&spectrum, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(original)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, spectrum));
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes)
        return 1;
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, spectrum, output));
    CHECK_CUDA(cudaMemcpyAsync(recovered, output, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 32; ++i) {
        if (fabsf(recovered[i].real - original[i].real) > 2.0e-4f ||
            fabsf(recovered[i].imag - original[i].imag) > 2.0e-4f)
            return 1;
    }
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(spectrum);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_2d_ntt(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {4, 8};
    uint64_t original[32];
    uint64_t recovered[32];
    uint64_t* input = NULL;
    uint64_t* spectrum = NULL;
    uint64_t* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    for (size_t i = 0; i < 32; ++i)
        original[i] = (uint64_t)(i * i + 3 * i + 1);
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_NTT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT64, CUBUTTERFLY_COMPUTE_UINT64));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&spectrum, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(original)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, spectrum));
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes)
        return 1;
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, spectrum, output));
    CHECK_CUDA(cudaMemcpyAsync(recovered, output, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 32; ++i)
        if (recovered[i] != original[i])
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(spectrum);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_arbitrary_ntt(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extent = 3;
    uint64_t original[3] = {7, 11, 19};
    uint64_t recovered[3];
    uint64_t* input = NULL;
    uint64_t* spectrum = NULL;
    uint64_t* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_NTT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT64, CUBUTTERFLY_COMPUTE_UINT64));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&spectrum, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(original)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, spectrum));
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes)
        return 1;
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, spectrum, output));
    CHECK_CUDA(cudaMemcpyAsync(recovered, output, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < extent; ++i)
        if (recovered[i] != original[i])
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(spectrum);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_hierarchical_ntt(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extent = 4096;
    uint32_t* host_output = (uint32_t*)malloc(extent * sizeof(uint32_t));
    uint32_t* input = NULL;
    uint32_t* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    size_t workspace_bytes = 0;
    char algorithm[64];
    size_t algorithm_bytes = sizeof(algorithm);
    if (host_output == NULL)
        return 1;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_NTT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT32, CUBUTTERFLY_COMPUTE_UINT32));
    CHECK_CUB(cubutterflySetModulus(descriptor, 998244353, 32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
                                            "ntt-hierarchical-dataflow"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetAlgorithmName(plan, algorithm, &algorithm_bytes));
    if (strcmp(algorithm, "ntt-hierarchical-dataflow") != 0)
        return 1;
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(plan, &workspace_bytes));
    if (workspace_bytes <= extent * sizeof(uint32_t))
        return 1;
    CHECK_CUDA(cudaMalloc((void**)&input, extent * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void**)&output, extent * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUDA(cudaMemsetAsync(input, 0, extent * sizeof(uint32_t), stream));
    {
        const uint32_t one = 1;
        CHECK_CUDA(cudaMemcpyAsync(input, &one, sizeof(one), cudaMemcpyHostToDevice, stream));
    }
    CHECK_CUB(cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, plan, input, output));
    CHECK_CUDA(cudaMemcpyAsync(host_output, output, extent * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < extent; ++i)
        if (host_output[i] != 1)
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(input);
    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    free(host_output);
    return 0;
}

static int run_hierarchical_core_selection(cubutterflyHandle_t handle) {
    const size_t extent = 1u << 20;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    char algorithm[64];
    size_t algorithm_bytes = sizeof(algorithm);
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_NTT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT32, CUBUTTERFLY_COMPUTE_UINT32));
    CHECK_CUB(cubutterflySetModulus(descriptor, 998244353, 32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
                                            "ntt-hierarchical-barrier-hybrid2d-radix4"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetAlgorithmName(plan, algorithm, &algorithm_bytes));
    if (strcmp(algorithm, "ntt-hierarchical-barrier-hybrid2d-radix4") != 0)
        return 1;
    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_measure_cache(cubutterflyHandle_t handle) {
    const char* cache_path = "/tmp/cubutterfly_c_api_test.cache";
    size_t extent = 16;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t measured = NULL;
    cubutterflyPlan_t cached = NULL;
    cubutterflySelectionSource_t source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
    remove(cache_path);
    CHECK_CUB(cubutterflySetCachePath(handle, cache_path));
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FWHT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 8));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_MEASURE, NULL));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &measured));
    CHECK_CUB(cubutterflyPlanGetSelectionSource(measured, &source));
    if (source != CUBUTTERFLY_SELECTION_RUNTIME_MEASURED)
        return 1;
    cubutterflyDestroyPlan(measured);
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_DEFAULT, NULL));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &cached));
    CHECK_CUB(cubutterflyPlanGetSelectionSource(cached, &source));
    if (source != CUBUTTERFLY_SELECTION_USER_CACHE)
        return 1;
    cubutterflyDestroyPlan(cached);

    extent = 15;
    measured = NULL;
    cached = NULL;
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 8));
    CHECK_CUB(cubutterflySetLengthMode(descriptor, CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_MEASURE, NULL));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &measured));
    CHECK_CUB(cubutterflyPlanGetSelectionSource(measured, &source));
    if (source != CUBUTTERFLY_SELECTION_RUNTIME_MEASURED)
        return 1;
    cubutterflyDestroyPlan(measured);
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_DEFAULT, NULL));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &cached));
    CHECK_CUB(cubutterflyPlanGetSelectionSource(cached, &source));
    if (source != CUBUTTERFLY_SELECTION_USER_CACHE)
        return 1;
    cubutterflyDestroyPlan(cached);

    cubutterflyDestroyDescriptor(descriptor);
    CHECK_CUB(cubutterflySetCachePath(handle, NULL));
    remove(cache_path);
    return 0;
}

static int run_2d_zeta(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {4, 8};
    uint32_t original[32];
    uint32_t recovered[32];
    uint32_t* input = NULL;
    uint32_t* transformed = NULL;
    uint32_t* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    for (size_t i = 0; i < 32; ++i)
        original[i] = (uint32_t)(i * 3 + 1);
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_SUBSET_ZETA));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT32, CUBUTTERFLY_COMPUTE_UINT32));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&transformed, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(original)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, transformed));
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 0));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes)
        return 1;
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, transformed, output));
    CHECK_CUDA(cudaMemcpyAsync(recovered, output, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 32; ++i)
        if (recovered[i] != original[i])
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(transformed);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_2d_structured(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {4, 8};
    const cubutterflyMatrix2x2_t matrix = {1.0, 0.25, -0.5, 1.0};
    float original[32];
    float recovered[32];
    float* input = NULL;
    float* transformed = NULL;
    float* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    for (size_t i = 0; i < 32; ++i)
        original[i] = (float)((i * 5) % 17) / 17.0f;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_STRUCTURED_2X2));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflySetStageMatrices(descriptor, 0, &matrix, 1));
    CHECK_CUB(cubutterflySetStageMatrices(descriptor, 1, &matrix, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&transformed, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(original)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, input, transformed));
    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 0));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes)
        return 1;
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, transformed, output));
    CHECK_CUDA(cudaMemcpyAsync(recovered, output, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 32; ++i)
        if (fabsf(recovered[i] - original[i]) > 3.0e-4f)
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(transformed);
    cudaFree(input);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_2d_fwht(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extents[2] = {4, 8};
    float host_input[32];
    float host_output[32];
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    float* input = NULL;
    float* output = NULL;
    void* workspace = NULL;
    size_t workspace_bytes = 0;
    for (size_t i = 0; i < 32; ++i)
        host_input[i] = 1.0f;
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FWHT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 2, extents, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(plan, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(host_input)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(host_output)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, host_input, sizeof(host_input), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, plan, input, output));
    CHECK_CUDA(cudaMemcpyAsync(host_output, output, sizeof(host_output), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (fabsf(host_output[0] - 32.0f) > 1.0e-5f)
        return 1;
    for (size_t i = 1; i < 32; ++i)
        if (fabsf(host_output[i]) > 1.0e-5f)
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(input);
    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_embedding_fwht(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extent = 6;
    float original[6] = {1, 2, 3, 4, 5, 6};
    float spectrum[8];
    float recovered[6];
    float* logical = NULL;
    float* physical = NULL;
    float* result = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t forward = NULL;
    cubutterflyPlan_t inverse = NULL;
    size_t workspace_bytes = 0;
    size_t inverse_workspace_bytes = 0;
    char algorithm[256];
    size_t algorithm_bytes = sizeof(algorithm);
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FWHT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetLengthMode(descriptor, CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
                                            "temporal-warp-register-radix2"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &forward));
    CHECK_CUB(cubutterflyPlanGetAlgorithmName(forward, algorithm, &algorithm_bytes));
    if (strstr(algorithm, "+fused-input") == NULL)
        return 1;
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(forward, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&logical, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&physical, sizeof(spectrum)));
    CHECK_CUDA(cudaMalloc((void**)&result, sizeof(recovered)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(forward, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(logical, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, forward, logical, physical));
    CHECK_CUDA(cudaMemcpyAsync(spectrum, physical, sizeof(spectrum), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUB(cubutterflySetDirection(descriptor, CUBUTTERFLY_DIRECTION_INVERSE, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &inverse));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(inverse, &inverse_workspace_bytes));
    if (inverse_workspace_bytes > workspace_bytes) {
        cudaFree(workspace);
        CHECK_CUDA(cudaMalloc(&workspace, inverse_workspace_bytes));
        workspace_bytes = inverse_workspace_bytes;
    }
    CHECK_CUB(cubutterflyPlanSetWorkspace(inverse, workspace, workspace_bytes));
    CHECK_CUB(cubutterflyExecute(handle, inverse, physical, result));
    CHECK_CUDA(cudaMemcpyAsync(recovered, result, sizeof(recovered), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < 6; ++i)
        if (fabsf(recovered[i] - original[i]) > 1.0e-5f)
            return 1;
    cudaFree(workspace);
    cudaFree(result);
    cudaFree(physical);
    cudaFree(logical);
    cubutterflyDestroyPlan(inverse);
    cubutterflyDestroyPlan(forward);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_strided_embedding_fwht(cubutterflyHandle_t handle, cudaStream_t stream) {
    const size_t extent = 6;
    const size_t input_stride[1] = {1};
    const size_t output_stride[1] = {2};
    float original[6] = {1, 2, 3, 4, 5, 6};
    float expected[8] = {1, 2, 3, 4, 5, 6, 0, 0};
    float strided[15];
    float* input = NULL;
    float* output = NULL;
    void* workspace = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    size_t workspace_bytes = 0;
    for (size_t width = 1; width < 8; width *= 2) {
        for (size_t base = 0; base < 8; base += 2 * width) {
            for (size_t offset = 0; offset < width; ++offset) {
                const float left = expected[base + offset];
                const float right = expected[base + offset + width];
                expected[base + offset] = left + right;
                expected[base + offset + width] = left - right;
            }
        }
    }
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FWHT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetLengthMode(descriptor, CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING));
    CHECK_CUB(cubutterflySetStrides(descriptor, input_stride, extent,
                                    output_stride, 15));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetWorkspaceSize(plan, &workspace_bytes));
    CHECK_CUDA(cudaMalloc((void**)&input, sizeof(original)));
    CHECK_CUDA(cudaMalloc((void**)&output, sizeof(strided)));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    CHECK_CUB(cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes));
    CHECK_CUDA(cudaMemcpyAsync(input, original, sizeof(original), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemsetAsync(output, 0, sizeof(strided), stream));
    CHECK_CUB(cubutterflyExecute(handle, plan, input, output));
    CHECK_CUDA(cudaMemcpyAsync(strided, output, sizeof(strided), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (size_t index = 0; index < 8; ++index)
        if (fabsf(strided[2 * index] - expected[index]) > 1.0e-5f)
            return 1;
    cudaFree(workspace);
    cudaFree(output);
    cudaFree(input);
    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

static int run_appt_layout_query(cubutterflyHandle_t handle) {
    const size_t extent = (size_t)1 << 20;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    cubutterflyNttLayoutInfo_t layout;
    char algorithm[128];
    size_t algorithm_bytes = sizeof(algorithm);
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_NTT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_UINT32,
                                     CUBUTTERFLY_COMPUTE_UINT32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflySetModulus(descriptor, 998244353ULL, 32));
    CHECK_CUB(cubutterflySetNttLayouts(
        descriptor, CUBUTTERFLY_NTT_LAYOUT_NATURAL,
        CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetNttLayoutInfo(plan, &layout));
    if (layout.log_n != 20 || layout.stage_count != 3 ||
        layout.stage_partition[0] != 7 || layout.stage_partition[1] != 7 ||
        layout.stage_partition[2] != 6 || layout.fragment_width != 32 ||
        layout.bank_bits != 5 || !layout.xor_permutation ||
        strstr(layout.compatibility_id, "appt-ntt-log20-7x7x6") == NULL)
        return 1;
    cubutterflyDestroyPlan(plan);

    CHECK_CUB(cubutterflySetNttLayouts(
        descriptor, CUBUTTERFLY_NTT_LAYOUT_NATURAL,
        CUBUTTERFLY_NTT_LAYOUT_NATURAL));
    CHECK_CUB(cubutterflySetAlgorithmPolicy(
        descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
        "ntt-appt-register-tail-grouped-writer-final-resident"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetAlgorithmName(
        plan, algorithm, &algorithm_bytes));
    if (strcmp(algorithm,
               "ntt-appt-register-tail-grouped-writer-final-resident") != 0)
        return 1;
    cubutterflyDestroyPlan(plan);

    CHECK_CUB(cubutterflySetAlgorithmPolicy(
        descriptor, CUBUTTERFLY_ALGORITHM_EXPLICIT,
        "ntt-appt-register-tail-grouped-writer-final-resident-quarter"));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    algorithm_bytes = sizeof(algorithm);
    CHECK_CUB(cubutterflyPlanGetAlgorithmName(
        plan, algorithm, &algorithm_bytes));
    if (strcmp(algorithm,
               "ntt-appt-register-tail-grouped-writer-final-resident-quarter") !=
        0)
        return 1;
    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    return 0;
}

int main(void) {
    cubutterflyHandle_t handle = NULL;
    cubutterflyDescriptor_t descriptor = NULL;
    cubutterflyPlan_t plan = NULL;
    cudaStream_t stream = NULL;
    float* device_input = NULL;
    float* device_output = NULL;
    float host_input[8];
    float host_output[8];
    size_t extent = 8;
    size_t input_bytes = 0;
    size_t output_bytes = 0;
    int warnings = 0;

    for (size_t i = 0; i < extent; ++i)
        host_input[i] = 1.0f;

    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUB(cubutterflyCreate(&handle));
    CHECK_CUB(cubutterflySetStream(handle, stream));
    CHECK_CUB(cubutterflySetLogCallback(handle, log_message, &warnings));
    CHECK_CUB(cubutterflyCreateDescriptor(&descriptor));
    CHECK_CUB(cubutterflySetOperator(descriptor, CUBUTTERFLY_OPERATOR_FWHT));
    CHECK_CUB(cubutterflySetDataType(descriptor, CUBUTTERFLY_DATA_FP32, CUBUTTERFLY_COMPUTE_FP32));
    CHECK_CUB(cubutterflySetShape(descriptor, 1, &extent, 1));
    CHECK_CUB(cubutterflyCreatePlan(handle, descriptor, &plan));
    CHECK_CUB(cubutterflyPlanGetInputSize(plan, &input_bytes));
    CHECK_CUB(cubutterflyPlanGetOutputSize(plan, &output_bytes));
    {
        cubutterflyNttLayoutInfo_t layout;
        if (cubutterflyPlanGetNttLayoutInfo(plan, &layout) !=
            CUBUTTERFLY_STATUS_NOT_SUPPORTED)
            return 1;
    }
    if (input_bytes != sizeof(host_input) || output_bytes != sizeof(host_output))
        return 1;

    CHECK_CUDA(cudaMalloc((void**)&device_input, input_bytes));
    CHECK_CUDA(cudaMalloc((void**)&device_output, output_bytes));
    CHECK_CUDA(cudaMemcpyAsync(device_input, host_input, input_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUB(cubutterflyExecute(handle, plan, device_input, device_output));
    CHECK_CUDA(cudaMemcpyAsync(host_output, device_output, output_bytes, cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if (fabsf(host_output[0] - 8.0f) > 1.0e-5f)
        return 1;

    if (run_2d_fwht(handle, stream) != 0 || run_embedding_fwht(handle, stream) != 0 ||
        run_strided_embedding_fwht(handle, stream) != 0 ||
        run_arbitrary_fft(handle, stream) != 0)
        return 1;
    if (run_2d_arbitrary_fft(handle, stream) != 0)
        return 1;
    if (run_2d_power_fft(handle, stream) != 0 || run_2d_ntt(handle, stream) != 0)
        return 1;
    if (run_arbitrary_ntt(handle, stream) != 0 || run_hierarchical_ntt(handle, stream) != 0 ||
        run_hierarchical_core_selection(handle) != 0 ||
        run_appt_layout_query(handle) != 0)
        return 1;
    if (run_measure_cache(handle) != 0)
        return 1;
    if (run_2d_zeta(handle, stream) != 0 || run_2d_structured(handle, stream) != 0)
        return 1;
    if (warnings == 0)
        return 1;

    cubutterflyDestroyPlan(plan);
    cubutterflyDestroyDescriptor(descriptor);
    cubutterflyDestroy(handle);
    cudaFree(device_output);
    cudaFree(device_input);
    cudaStreamDestroy(stream);
    return 0;
}
