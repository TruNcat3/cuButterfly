// cuButterfly experimental local provider. Upstream device arithmetic is in
// generated hadacore_device.cuh; retain LICENSE.HadaCore beside distributions.
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>

static __device__ __forceinline__ uint32_t cubutterfly_hada_scale_pair(uint32_t x, float scale) {
    const __half lo = __float2half_rn(__fmul_rn(__half2float(__ushort_as_half(x & 65535)), scale));
    const __half hi = __float2half_rn(__fmul_rn(__half2float(__ushort_as_half(x >> 16)), scale));
    return uint32_t(__half_as_ushort(lo)) | (uint32_t(__half_as_ushort(hi)) << 16);
}
static __device__ __forceinline__ int4 cubutterfly_hada_scale_vector(int4 x, float scale) {
    x.x = cubutterfly_hada_scale_pair(x.x, scale);
    x.y = cubutterfly_hada_scale_pair(x.y, scale);
    x.z = cubutterfly_hada_scale_pair(x.z, scale);
    x.w = cubutterfly_hada_scale_pair(x.w, scale);
    return x;
}

#include "hadacore_device.cuh"

// One block repeatedly invokes the local device function. It owns exactly one
// group's shared scratch, independent of the number of temporal iterations.
template<int LogN, int Chunks, int Warps, int Blocks>
__global__ __launch_bounds__(32 * Warps, Blocks)
void cubutterfly_hadacore_tiles(b16* input, b16* output, uint64_t groups,
                              uint32_t groups_per_block, float scale) {
    extern __shared__ b32 scratch[];
    constexpr uint64_t elements = 256 * Chunks * Warps;
    const uint64_t first = uint64_t(blockIdx.x) * groups_per_block;
    for (uint32_t iteration = 0; iteration < groups_per_block; ++iteration) {
        const uint64_t group = first + iteration;
        if (group >= groups) break; // uniform across this entire block
        cubutterfly_hadacore_device<Chunks, Warps, LogN, Blocks, false, torch::ScalarType::Half>(
            input + group * elements, output + group * elements, Chunks * Warps, scratch, scale);
        __syncthreads(); // scratch is reused only after all warps finish a group
    }
}

struct CubutterflyHadaTileInfo {
    int log_n, chunks_per_warp, warps_per_block, blocks_per_sm;
    int threads, group_elements, shared_bytes, stages;
};

// The bounded compiled population retains both upstream short-transform
// geometries. Additional geometries require separate correctness validation.
#define CUB_HADA_CONFIGS(M) \
    M(1, 1, 1, 24) M(2, 1, 1, 24) M(3, 1, 1, 24) M(4, 1, 1, 24) \
    M(5, 1, 1, 24) M(6, 1, 1, 24) M(7, 1, 1, 24) M(8, 1, 1, 24) \
    M(1, 2, 1, 24) M(2, 2, 1, 24) M(3, 2, 1, 24) M(4, 2, 1, 24) \
    M(5, 2, 1, 24) M(6, 2, 1, 24) M(7, 2, 1, 24) M(8, 2, 1, 24) \
    M(9, 2, 1, 24) M(10, 2, 2, 16) M(11, 2, 4, 8) M(12, 2, 8, 4) \
    M(13, 2, 16, 3) M(14, 4, 16, 2) M(15, 8, 16, 1)

extern "C" int cubutterfly_hadacore_tile_info(int log_n, int chunks, CubutterflyHadaTileInfo* out) {
    if (!out) return -1;
#define CUB_INFO(L,C,W,B) if (log_n==L && chunks==C) { \
    *out = {L,C,W,B,32*W,256*C*W,512*C*W,L}; return 0; }
    CUB_HADA_CONFIGS(CUB_INFO)
#undef CUB_INFO
    return -1;
}

// Preparation belongs before warmup/capture. This function is the only API here
// that changes the kernel's dynamic shared-memory launch attribute.
extern "C" int cubutterfly_hadacore_tile_prepare(int log_n, int chunks) {
#define CUB_PREPARE(L,C,W,B) if (log_n==L && chunks==C) { \
    if (512*C*W > 48*1024) return int(cudaFuncSetAttribute( \
        cubutterfly_hadacore_tiles<L,C,W,B>, cudaFuncAttributeMaxDynamicSharedMemorySize, 512*C*W)); \
    return 0; }
    CUB_HADA_CONFIGS(CUB_PREPARE)
#undef CUB_PREPARE
    return -1;
}

struct CubutterflyHadaTileResources {
    int registers_per_thread, static_shared_bytes, local_bytes_per_thread;
    int max_threads_per_block, active_blocks_per_sm, binary_version, ptx_version;
};

// Query the exact compiled specialization before capture and timing.
extern "C" int cubutterfly_hadacore_tile_resources(
    int log_n, int chunks, CubutterflyHadaTileResources* out) {
    if (!out) return -1;
#define CUB_RESOURCES(L,C,W,B) if (log_n==L && chunks==C) { \
    cudaFuncAttributes a{}; int active = 0; \
    cudaError_t e = cudaFuncGetAttributes(&a, cubutterfly_hadacore_tiles<L,C,W,B>); \
    if (e != cudaSuccess) return int(e); \
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor( \
        &active, cubutterfly_hadacore_tiles<L,C,W,B>, 32*W, 512*C*W); \
    if (e != cudaSuccess) return int(e); \
    *out = {a.numRegs, int(a.sharedSizeBytes), int(a.localSizeBytes), \
            a.maxThreadsPerBlock, active, a.binaryVersion, a.ptxVersion}; return 0; }
    CUB_HADA_CONFIGS(CUB_RESOURCES)
#undef CUB_RESOURCES
    return -1;
}

extern "C" int cubutterfly_hadacore_tile_execute(
    void* input, void* output, uint64_t tile_count, int log_n, int chunks,
    uint32_t groups_per_block, float scale, cudaStream_t stream) {
    CubutterflyHadaTileInfo info;
    if (cubutterfly_hadacore_tile_info(log_n, chunks, &info) || !input || !output ||
        !tile_count || !groups_per_block || !std::isfinite(scale) ||
        uintptr_t(input)%16 || uintptr_t(output)%16) return -1;
    if (tile_count > UINT64_MAX / (uint64_t(1) << log_n)) return -1;
    const uint64_t elements = tile_count << log_n;
    if (elements % info.group_elements || elements > UINT64_MAX/2) return -1;
    const uintptr_t a = uintptr_t(input), b = uintptr_t(output);
    const uintptr_t distance = a > b ? a-b : b-a;
    if (a != b && distance < 2*elements) return -1;
    const uint64_t groups = elements / info.group_elements;
    const uint64_t blocks = (groups-1) / groups_per_block + 1;
    if (blocks > 2147483647ULL) return -1;
#define CUB_LAUNCH(L,C,W,B) if (log_n==L && chunks==C) { \
    cubutterfly_hadacore_tiles<L,C,W,B><<<unsigned(blocks),32*W,512*C*W,stream>>>( \
        static_cast<b16*>(input),static_cast<b16*>(output),groups,groups_per_block,scale); \
    return int(cudaGetLastError()); }
    CUB_HADA_CONFIGS(CUB_LAUNCH)
#undef CUB_LAUNCH
    return -1;
}
#undef CUB_HADA_CONFIGS
