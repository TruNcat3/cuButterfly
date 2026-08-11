#pragma once

#include <cuda_runtime_api.h>

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define CUBUTTERFLYAPI __declspec(dllimport)
#else
#define CUBUTTERFLYAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define CUBUTTERFLY_VERSION_MAJOR 0
#define CUBUTTERFLY_VERSION_MINOR 6
#define CUBUTTERFLY_VERSION_PATCH 0

typedef struct cubutterflyHandle* cubutterflyHandle_t;
typedef struct cubutterflyDescriptor* cubutterflyDescriptor_t;
typedef struct cubutterflyPlan* cubutterflyPlan_t;

typedef enum cubutterflyStatus {
    CUBUTTERFLY_STATUS_SUCCESS = 0,
    CUBUTTERFLY_STATUS_INVALID_VALUE,
    CUBUTTERFLY_STATUS_INVALID_TYPE,
    CUBUTTERFLY_STATUS_NOT_SUPPORTED,
    CUBUTTERFLY_STATUS_ALLOC_FAILED,
    CUBUTTERFLY_STATUS_CUDA_ERROR,
    CUBUTTERFLY_STATUS_BACKEND_ERROR,
    CUBUTTERFLY_STATUS_INTERNAL_ERROR
} cubutterflyStatus_t;

typedef enum cubutterflyOperator {
    CUBUTTERFLY_OPERATOR_FFT = 0,
    CUBUTTERFLY_OPERATOR_NTT,
    CUBUTTERFLY_OPERATOR_FWHT,
    CUBUTTERFLY_OPERATOR_SUBSET_ZETA,
    CUBUTTERFLY_OPERATOR_SUPERSET_ZETA,
    CUBUTTERFLY_OPERATOR_STRUCTURED_2X2
} cubutterflyOperator_t;

typedef enum cubutterflyDataType {
    CUBUTTERFLY_DATA_FP16 = 0,
    CUBUTTERFLY_DATA_BF16,
    CUBUTTERFLY_DATA_FP32,
    CUBUTTERFLY_DATA_FP64,
    CUBUTTERFLY_DATA_COMPLEX_FP16,
    CUBUTTERFLY_DATA_COMPLEX_BF16,
    CUBUTTERFLY_DATA_COMPLEX_FP32,
    CUBUTTERFLY_DATA_COMPLEX_FP64,
    CUBUTTERFLY_DATA_UINT32,
    CUBUTTERFLY_DATA_UINT64
} cubutterflyDataType_t;

typedef enum cubutterflyComputeType {
    CUBUTTERFLY_COMPUTE_NATIVE = 0,
    CUBUTTERFLY_COMPUTE_FP32,
    CUBUTTERFLY_COMPUTE_FP64,
    CUBUTTERFLY_COMPUTE_UINT32,
    CUBUTTERFLY_COMPUTE_UINT64
} cubutterflyComputeType_t;

typedef enum cubutterflyDirection {
    CUBUTTERFLY_DIRECTION_FORWARD = 0,
    CUBUTTERFLY_DIRECTION_INVERSE
} cubutterflyDirection_t;

typedef enum cubutterflyPlacement {
    CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE = 0,
    CUBUTTERFLY_PLACEMENT_IN_PLACE
} cubutterflyPlacement_t;

typedef enum cubutterflyLengthMode {
    CUBUTTERFLY_LENGTH_STANDARD = 0,
    CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING
} cubutterflyLengthMode_t;

typedef enum cubutterflyAlgorithmPolicy {
    CUBUTTERFLY_ALGORITHM_DEFAULT = 0,
    CUBUTTERFLY_ALGORITHM_MEASURE,
    CUBUTTERFLY_ALGORITHM_EXPLICIT
} cubutterflyAlgorithmPolicy_t;

typedef enum cubutterflySelectionSource {
    CUBUTTERFLY_SELECTION_MEASURED_TABLE = 0,
    CUBUTTERFLY_SELECTION_USER_CACHE,
    CUBUTTERFLY_SELECTION_RUNTIME_MEASURED,
    CUBUTTERFLY_SELECTION_STATIC_MODEL,
    CUBUTTERFLY_SELECTION_EXPLICIT
} cubutterflySelectionSource_t;

typedef enum cubutterflyLogLevel {
    CUBUTTERFLY_LOG_INFO = 0,
    CUBUTTERFLY_LOG_WARNING,
    CUBUTTERFLY_LOG_ERROR
} cubutterflyLogLevel_t;

typedef struct cubutterflyMatrix2x2 {
    double m00;
    double m01;
    double m10;
    double m11;
} cubutterflyMatrix2x2_t;

typedef void (*cubutterflyLogCallback_t)(cubutterflyLogLevel_t level,
                                         const char* event,
                                         const char* message,
                                         void* user_data);

CUBUTTERFLYAPI const char* cubutterflyGetStatusString(cubutterflyStatus_t status);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyGetVersion(int* major, int* minor, int* patch);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreate(cubutterflyHandle_t* handle);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroy(cubutterflyHandle_t handle);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStream(cubutterflyHandle_t handle, cudaStream_t stream);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyGetStream(cubutterflyHandle_t handle, cudaStream_t* stream);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetLogCallback(cubutterflyHandle_t handle,
                                                            cubutterflyLogCallback_t callback,
                                                            void* user_data);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetCachePath(cubutterflyHandle_t handle, const char* path);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreateDescriptor(cubutterflyDescriptor_t* descriptor);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroyDescriptor(cubutterflyDescriptor_t descriptor);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetOperator(cubutterflyDescriptor_t descriptor,
                                                         cubutterflyOperator_t op);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetDataType(cubutterflyDescriptor_t descriptor,
                                                         cubutterflyDataType_t storage,
                                                         cubutterflyComputeType_t compute);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetShape(cubutterflyDescriptor_t descriptor,
                                                      uint32_t rank,
                                                      const size_t* extents,
                                                      size_t batch);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStrides(cubutterflyDescriptor_t descriptor,
                                                        const size_t* input_strides,
                                                        size_t input_batch_stride,
                                                        const size_t* output_strides,
                                                        size_t output_batch_stride);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetDirection(cubutterflyDescriptor_t descriptor,
                                                          cubutterflyDirection_t direction,
                                                          int normalize_inverse);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetPlacement(cubutterflyDescriptor_t descriptor,
                                                          cubutterflyPlacement_t placement);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetLengthMode(cubutterflyDescriptor_t descriptor,
                                                           cubutterflyLengthMode_t mode);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetAlgorithmPolicy(cubutterflyDescriptor_t descriptor,
                                                                cubutterflyAlgorithmPolicy_t policy,
                                                                const char* explicit_algorithm);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetModulus(cubutterflyDescriptor_t descriptor,
                                                        uint64_t modulus,
                                                        uint32_t word_bits);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStageMatrices(cubutterflyDescriptor_t descriptor,
                                                              uint32_t axis,
                                                              const cubutterflyMatrix2x2_t* matrices,
                                                              size_t count);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreatePlan(cubutterflyHandle_t handle,
                                                        cubutterflyDescriptor_t descriptor,
                                                        cubutterflyPlan_t* plan);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroyPlan(cubutterflyPlan_t plan);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetWorkspaceSize(cubutterflyPlan_t plan, size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetInputSize(cubutterflyPlan_t plan, size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetOutputSize(cubutterflyPlan_t plan, size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetPhysicalExtents(cubutterflyPlan_t plan,
                                                                    size_t* extents,
                                                                    uint32_t capacity);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetSelectionSource(cubutterflyPlan_t plan,
                                                                    cubutterflySelectionSource_t* source);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetAlgorithmName(cubutterflyPlan_t plan,
                                                                  char* name,
                                                                  size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetSelectionReason(cubutterflyPlan_t plan,
                                                                    char* reason,
                                                                    size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanSetWorkspace(cubutterflyPlan_t plan,
                                                              void* workspace,
                                                              size_t bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyExecute(cubutterflyHandle_t handle,
                                                     cubutterflyPlan_t plan,
                                                     const void* input,
                                                     void* output);

#ifdef __cplusplus
}
#endif
