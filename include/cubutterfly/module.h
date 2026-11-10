#pragma once
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define CUBUTTERFLY_MODULE_ABI_V1 1u
typedef struct cubutterflyModuleInvocationV1 {
    const void* input;
    void* output;
    void* workspace;
    uint64_t batch, batch_stride, element_stride;
    int inverse, normalize_inverse;
    cudaStream_t stream;
} cubutterflyModuleInvocationV1;
typedef struct cubutterflyModuleResourcesV1 {
    uint32_t registers_per_thread, threads, static_shared_bytes, dynamic_shared_bytes;
} cubutterflyModuleResourcesV1;
typedef struct cubutterflyModuleV1 {
    uint32_t abi_version, struct_size;
    const char* identity;
    uint32_t sm, group_count;
    int (*resources)(int inverse, uint32_t group, cubutterflyModuleResourcesV1* result);
    int (*launch)(const cubutterflyModuleInvocationV1* invocation);
} cubutterflyModuleV1;
typedef const cubutterflyModuleV1* (*cubutterflyModuleEntryV1)(void);
// Optional cubutterfly_module_local_bytes_v1 symbol. This reports compiler
// thread-local allocation, not dynamic spill traffic or off-chip bytes.
// Absence means unknown; preserve the original V1/V2 resource ABI.
typedef int (*cubutterflyModuleLocalBytesV1)(int inverse, uint32_t group, uint64_t* bytes);
// Optional symbol cubutterfly_module_launch_group_v1. Input/output are the
// selected group's endpoints; workspace is unused. Dependencies belong to
// the caller. V1's original descriptor and whole-transform ABI are unchanged.
typedef int (*cubutterflyModuleGroupLaunchV1)(uint32_t group, const cubutterflyModuleInvocationV1* invocation);
// Optional cubutterfly_module_launch_range_v1 symbol. Compact tile i maps to
// tile_first + (i / tile_span) * tile_period + i % tile_span. Tile units are
// FFTsPerBlock independent local transforms, including the batch dimension.
// Endpoints retain their full-array addressing. The caller owns dependencies
// and prevents scratch reuse until every consumer has finished.
typedef struct cubutterflyModuleTileRangeV1 {
    cubutterflyModuleInvocationV1 io;
    uint64_t tile_first, tile_count, tile_period, tile_span;
} cubutterflyModuleTileRangeV1;
typedef int (*cubutterflyModuleRangeLaunchV1)(uint32_t group, const cubutterflyModuleTileRangeV1* invocation);
// V2 retains V1 for register FFT modules, and adds typed semantic data for
// generic butterfly traits. Coefficients are device pointers owned by the plan.
typedef struct cubutterflyModuleInvocationV2 {
    cubutterflyModuleInvocationV1 io;
    const void* coefficients;
    const void* coefficients_aux;
    uint64_t modulus, scale, scale_shoup;
} cubutterflyModuleInvocationV2;
typedef struct cubutterflyModuleV2 {
    uint32_t abi_version, struct_size;
    const char* identity;
    uint32_t sm, group_count;
    int (*resources)(int inverse, uint32_t group, cubutterflyModuleResourcesV1* result);
    int (*launch)(const cubutterflyModuleInvocationV2* invocation);
} cubutterflyModuleV2;
typedef const cubutterflyModuleV2* (*cubutterflyModuleEntryV2)(void);
// Optional V2 symbol cubutterfly_module_launch_group_v2. The caller provides
// this group's input/output and owns all inter-group dependencies and scratch.
// The original descriptor and whole-transform entry remain ABI compatible.
typedef int (*cubutterflyModuleGroupLaunchV2)(uint32_t group, const cubutterflyModuleInvocationV2* invocation);
#ifdef __cplusplus
}
#endif
