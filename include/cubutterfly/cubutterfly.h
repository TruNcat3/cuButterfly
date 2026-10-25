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
#define CUBUTTERFLY_VERSION_MINOR 9
#define CUBUTTERFLY_VERSION_PATCH 3
#define CUBUTTERFLY_MAX_EXECUTION_RANK 8

/* Per-axis logical indexing, independent of physical memory strides and the
 * DIT/DIF/GS implementation selected by the planner. For bit reversal, logical
 * coordinate i is stored at reverse_bits(i, log2(axis_extent)). */
typedef enum cubutterflyLogicalOrder {
    CUBUTTERFLY_ORDER_NATURAL = 0,
    CUBUTTERFLY_ORDER_BIT_REVERSED
} cubutterflyLogicalOrder_t;

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

/* STAGE_STORAGE preserves the legacy intermediate rounding. FINAL_STORAGE
 * retains compute precision until the output scale and final storage cast. */
typedef enum cubutterflyOutputRounding {
    CUBUTTERFLY_ROUNDING_STAGE_STORAGE = 0,
    CUBUTTERFLY_ROUNDING_FINAL_STORAGE
} cubutterflyOutputRounding_t;

typedef enum cubutterflyLengthMode {
    CUBUTTERFLY_LENGTH_STANDARD = 0,
    CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING
} cubutterflyLengthMode_t;

/* Multidimensional execution/lowering policy. AUTO retains rank-1 selection
 * and enables bounded complete-plan selection for rank 2/3. XD requires a
 * multidimensional descriptor; COMPOSITE keeps per-axis composition. */
typedef enum cubutterflyExecutionMode {
    CUBUTTERFLY_EXECUTION_AUTO = 0,
    CUBUTTERFLY_EXECUTION_XD,
    CUBUTTERFLY_EXECUTION_COMPOSITE
} cubutterflyExecutionMode_t;

/* R2C/C2R use real storage to specify scalar precision. Shape is the full
 * real-domain shape; only the last axis is shortened in complex storage.
 * Strides count elements of the respective input/output type. */
typedef enum cubutterflyFftTransformType {
    CUBUTTERFLY_FFT_C2C = 0,
    CUBUTTERFLY_FFT_R2C,
    CUBUTTERFLY_FFT_C2R
} cubutterflyFftTransformType_t;

typedef enum cubutterflyNttLayout {
    CUBUTTERFLY_NTT_LAYOUT_NATURAL = 0,
    CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC,
    /* Legacy scalar spelling for rank-one bit-reversed logical order. New
     * callers can use per-axis SetLogicalOrders for any operator. */
    CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED
} cubutterflyNttLayout_t;

typedef struct cubutterflyNttLayoutInfo {
    uint32_t log_n;
    uint32_t stage_count;
    uint32_t stage_partition[3];
    uint32_t fragment_width;
    uint32_t bank_bits;
    int xor_permutation;
    char compatibility_id[64];
} cubutterflyNttLayoutInfo_t;

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
    CUBUTTERFLY_SELECTION_EXPLICIT,
    /* No compatible performance estimate; preserved executable/prior choice. */
    CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK
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
/* Detailed error for the calling thread; valid until its next protected API
 * call. Reading this string does not clear it. */
CUBUTTERFLYAPI const char* cubutterflyGetLastErrorString(void);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyGetVersion(int* major, int* minor, int* patch);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreate(cubutterflyHandle_t* handle);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroy(cubutterflyHandle_t handle);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStream(cubutterflyHandle_t handle, cudaStream_t stream);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyGetStream(cubutterflyHandle_t handle, cudaStream_t* stream);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetLogCallback(cubutterflyHandle_t handle,
                                                            cubutterflyLogCallback_t callback,
                                                            void* user_data);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetCachePath(cubutterflyHandle_t handle, const char* path);

/* Legacy real-FFT adapter kernels. New callers can use public R2C/C2R plans. */
CUBUTTERFLYAPI int cubutterflyPackRealFp32(const float* input, void* output,
                                           size_t elements, float scale,
                                           uint64_t stream);
CUBUTTERFLYAPI int cubutterflyExpandHermitianFp32(const void* half, void* full,
                                                  uint32_t height, uint32_t width,
                                                  size_t batch, uint64_t stream);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreateDescriptor(cubutterflyDescriptor_t* descriptor);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroyDescriptor(cubutterflyDescriptor_t descriptor);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetOperator(cubutterflyDescriptor_t descriptor,
                                                         cubutterflyOperator_t op);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetDataType(cubutterflyDescriptor_t descriptor,
                                                         cubutterflyDataType_t storage,
                                                         cubutterflyComputeType_t compute);
/* Descriptors express any positive rank. Capability queries distinguish the
 * currently executable rank range from the mathematical descriptor. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetShape(cubutterflyDescriptor_t descriptor,
                                                      uint32_t rank,
                                                      const size_t* extents,
                                                      size_t batch);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetExecutionMode(
    cubutterflyDescriptor_t descriptor,
    cubutterflyExecutionMode_t mode);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetFftTransformType(
    cubutterflyDescriptor_t descriptor,
    cubutterflyFftTransformType_t type);
/* Additional finite output scale for compact R2C/C2R (default 1).  Inverse
 * normalization, when requested, is multiplied into this scale. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetFftOutputScale(
    cubutterflyDescriptor_t descriptor, double scale);
/* Explicit output scale (default 1), applied before the final output narrowing.
 * Non-default scales require an implemented floating-point lowering. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetOutputScale(
    cubutterflyDescriptor_t descriptor, double scale);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetOutputRounding(
    cubutterflyDescriptor_t descriptor, cubutterflyOutputRounding_t rounding);
/* A NULL side selects its packed axis strides, resolved from its own shape.
 * Nonzero batch strides are preserved independently of the axis pointers. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStrides(cubutterflyDescriptor_t descriptor,
                                                        const size_t* input_strides,
                                                        size_t input_batch_stride,
                                                        const size_t* output_strides,
                                                        size_t output_batch_stride);
/* Each nullable array has descriptor rank entries. NULL selects natural
 * logical order, independently for each side. This replaces legacy NTT order
 * settings; APPT_STATIC remains available through cubutterflySetNttLayouts. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetLogicalOrders(
    cubutterflyDescriptor_t descriptor,
    const cubutterflyLogicalOrder_t* input_orders,
    const cubutterflyLogicalOrder_t* output_orders);
/* CPU-only semantic validation and canonical serialization. A valid semantic
 * description does not guarantee an executable lowering on a target GPU.
 * SetJson replaces the descriptor atomically; unknown fields are rejected.
 * String queries accept NULL to query size, including the trailing NUL. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDescriptorValidate(cubutterflyDescriptor_t descriptor);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDescriptorGetJson(
    cubutterflyDescriptor_t descriptor, char* json, size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDescriptorSetJson(
    cubutterflyDescriptor_t descriptor, const char* json);
/* Reports static lowering eligibility/reasons without probing a device.
 * Actual resources, compiled candidates and performance are checked later. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDescriptorGetCapabilities(
    cubutterflyDescriptor_t descriptor, char* json, size_t* bytes);
/* normalize_inverse controls FFT/FWHT scaling. Modular inverse NTT always
 * uses inverse_n; structured/zeta inverse semantics are operator-defined. */
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
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetNttLayouts(
    cubutterflyDescriptor_t descriptor,
    cubutterflyNttLayout_t input_layout,
    cubutterflyNttLayout_t output_layout);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflySetStageMatrices(cubutterflyDescriptor_t descriptor,
                                                              uint32_t axis,
                                                              const cubutterflyMatrix2x2_t* matrices,
                                                              size_t count);

CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyCreatePlan(cubutterflyHandle_t handle,
                                                        cubutterflyDescriptor_t descriptor,
                                                        cubutterflyPlan_t* plan);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyDestroyPlan(cubutterflyPlan_t plan);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetDescriptorJson(
    cubutterflyPlan_t plan, char* json, size_t* bytes);
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
/* Full execution mapping, including mixed cores, layouts and every physical
 * group. A rank-1 mapping can be passed to EXPLICIT without losing parameters. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetMappingJson(cubutterflyPlan_t plan,
                                                              char* json, size_t* bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanGetNttLayoutInfo(
    cubutterflyPlan_t plan,
    cubutterflyNttLayoutInfo_t* info);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyPlanSetWorkspace(cubutterflyPlan_t plan,
                                                              void* workspace,
                                                              size_t bytes);
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyExecute(cubutterflyHandle_t handle,
                                                     cubutterflyPlan_t plan,
                                                     const void* input,
                                                     void* output);
/* Submit a reused plan through the steady-state path. The first call and any
 * call after a stream change transparently fall back to cubutterflyExecute so
 * child streams remain correctly bound. */
CUBUTTERFLYAPI cubutterflyStatus_t cubutterflyExecuteSteady(cubutterflyHandle_t handle,
                                                            cubutterflyPlan_t plan,
                                                            const void* input,
                                                            void* output);

#ifdef __cplusplus
}
#endif
