#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// The register hierarchy and XOR-swizzled exchange follow the BSD-3-Clause
// fast-hadamard-transform design by Tri Dao (2023). This implementation is
// standalone and preserves cuButterfly's layout and execution contracts.

namespace cuntt::detail {

constexpr std::uint32_t register_fwht_threads(std::uint32_t log_n) {
    return log_n <= 3 ? 1 : log_n == 4 ? 2 : log_n == 5 ? 4 : log_n == 6 ? 8 : log_n == 7 ? 16 : log_n <= 9 ? 32 : log_n == 10 ? 128 : 256;
}

__host__ __device__ constexpr std::uint32_t static_log2(std::uint32_t value) {
    return value <= 1 ? 0 : 1 + static_log2(value >> 1);
}

template <std::uint32_t LogN>
struct RegisterFwhtTraits {
    static constexpr std::uint32_t kN             = 1U << LogN;
    static constexpr std::uint32_t kThreads       = register_fwht_threads(LogN);
    static constexpr std::uint32_t kItems         = 4;
    static constexpr std::uint32_t kChunks        = kN / (kThreads * kItems);
    static constexpr std::uint32_t kWarpSize      = kThreads < 32 ? kThreads : 32;
    static constexpr std::uint32_t kWarps         = kThreads / kWarpSize;
    static constexpr std::uint32_t kSharedBytes   = kN * sizeof(float) < 32 * 1024 ? kN * sizeof(float) : 32 * 1024;
    static constexpr std::uint32_t kChunksPerPass = kSharedBytes / (sizeof(float4) * kThreads);
    static_assert(LogN >= 3 && LogN <= 15);
    static_assert(kN == kThreads * kItems * kChunks);
    static_assert((kChunks & (kChunks - 1)) == 0);
};

template <std::uint32_t Length>
__device__ __forceinline__ void register_wht(float (&values)[Length]) {
#pragma unroll
    for (std::uint32_t stage = 0; stage < static_log2(Length); ++stage) {
        const std::uint32_t half = 1U << stage;
#pragma unroll
        for (std::uint32_t unit = 0; unit < Length / 2; ++unit) {
            const std::uint32_t offset = unit & (half - 1);
            const std::uint32_t left   = (unit - offset) * 2 + offset;
            const float         a      = values[left];
            const float         b      = values[left + half];
            values[left]               = a + b;
            values[left + half]        = a - b;
        }
    }
}

__device__ __forceinline__ float structured_pair(float self, float partner, bool right, float4 matrix) {
    return right ? matrix.z * partner + matrix.w * self
                 : matrix.x * self + matrix.y * partner;
}

template <bool Broadcast, std::uint32_t Length, std::uint32_t StageBase>
__device__ __forceinline__ void register_structured(float (&values)[Length], const float4* matrices,
                                                    float4 broadcast_matrix) {
#pragma unroll
    for (std::uint32_t local_stage = 0; local_stage < static_log2(Length); ++local_stage) {
        const std::uint32_t half   = 1U << local_stage;
        const float4        matrix = Broadcast ? broadcast_matrix : matrices[StageBase + local_stage];
#pragma unroll
        for (std::uint32_t unit = 0; unit < Length / 2; ++unit) {
            const std::uint32_t offset = unit & (half - 1);
            const std::uint32_t left   = (unit - offset) * 2 + offset;
            const float         a      = values[left];
            const float         b      = values[left + half];
            values[left]               = matrix.x * a + matrix.y * b;
            values[left + half]        = matrix.z * a + matrix.w * b;
        }
    }
}

__device__ __forceinline__ float4 pack_float4(const float (&values)[4]) {
    return make_float4(values[0], values[1], values[2], values[3]);
}

__device__ __forceinline__ void unpack_float4(float4 packed, float (&values)[4]) {
    values[0] = packed.x;
    values[1] = packed.y;
    values[2] = packed.z;
    values[3] = packed.w;
}

template <std::uint32_t Width, std::uint32_t Chunks>
__device__ __forceinline__ void warp_wht(float (&values)[Chunks][4]) {
    if constexpr (Width > 1) {
        const std::uint32_t lane = threadIdx.x & (Width - 1);
        const unsigned int  mask = __activemask();
#pragma unroll
        for (std::uint32_t stage = 0; stage < static_log2(Width); ++stage) {
            const std::uint32_t lane_mask = 1U << stage;
            const float         sign      = (lane & lane_mask) == 0 ? 1.0F : -1.0F;
#pragma unroll
            for (std::uint32_t chunk = 0; chunk < Chunks; ++chunk) {
#pragma unroll
                for (std::uint32_t item = 0; item < 4; ++item) {
                    const float partner = __shfl_xor_sync(mask, values[chunk][item], lane_mask, Width);
                    values[chunk][item] = sign * values[chunk][item] + partner;
                }
            }
        }
    }
}

template <bool Broadcast, std::uint32_t Width, std::uint32_t Chunks, std::uint32_t StageBase>
__device__ __forceinline__ void warp_structured(float (&values)[Chunks][4], const float4* matrices,
                                                float4 broadcast_matrix) {
    if constexpr (Width > 1) {
        const std::uint32_t lane = threadIdx.x & (Width - 1);
        const unsigned int  mask = __activemask();
#pragma unroll
        for (std::uint32_t local_stage = 0; local_stage < static_log2(Width); ++local_stage) {
            const std::uint32_t lane_mask = 1U << local_stage;
            const bool          right     = (lane & lane_mask) != 0;
            const float4        matrix    = Broadcast ? broadcast_matrix : matrices[StageBase + local_stage];
#pragma unroll
            for (std::uint32_t chunk = 0; chunk < Chunks; ++chunk) {
#pragma unroll
                for (std::uint32_t item = 0; item < 4; ++item) {
                    const float self    = values[chunk][item];
                    const float partner = __shfl_xor_sync(mask, self, lane_mask, Width);
                    values[chunk][item] = structured_pair(self, partner, right, matrix);
                }
            }
        }
    }
}

template <typename Traits, bool Pre>
__device__ __forceinline__ void exchange_warps(float (&values)[Traits::kChunks][4], float4* shared) {
    constexpr std::uint32_t kWarpSize = Traits::kWarpSize;
    constexpr std::uint32_t kWarps    = Traits::kWarps;
    const std::uint32_t     warp      = threadIdx.x / kWarpSize;
    const std::uint32_t     lane      = threadIdx.x % kWarpSize;
    const std::uint32_t     row       = threadIdx.x % kWarps;
    const std::uint32_t     column    = threadIdx.x / kWarps;

#pragma unroll
    for (std::uint32_t pass = 0; pass < Traits::kChunks / Traits::kChunksPerPass; ++pass) {
        __syncthreads();
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunksPerPass; ++chunk) {
            const std::uint32_t source              = pass * Traits::kChunksPerPass + chunk;
            const std::uint32_t slot                = Pre ? ((warp * kWarpSize + lane) ^ warp) : ((row * kWarpSize + column) ^ row);
            shared[chunk * Traits::kThreads + slot] = pack_float4(values[source]);
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunksPerPass; ++chunk) {
            const std::uint32_t target = pass * Traits::kChunksPerPass + chunk;
            const std::uint32_t slot   = Pre ? ((row * kWarpSize + column) ^ row) : ((warp * kWarpSize + lane) ^ warp);
            unpack_float4(shared[chunk * Traits::kThreads + slot], values[target]);
        }
    }
}

template <std::uint32_t LogN>
__global__ __launch_bounds__(RegisterFwhtTraits<LogN>::kThreads) void register_fwht_kernel(const float* input, float* output,
                                                                                           std::uint64_t transforms, std::uint64_t batch_distance,
                                                                                           std::uint64_t element_stride, float scale) {
    using Traits = RegisterFwhtTraits<LogN>;
    extern __shared__ __align__(16) unsigned char storage[];
    float4*                                       shared    = reinterpret_cast<float4*>(storage);
    const std::uint64_t                           transform = blockIdx.x;
    if (transform >= transforms)
        return;

    const std::uint64_t base = transform * batch_distance;
    float               values[Traits::kChunks][4];
#pragma unroll
    for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
        const std::uint32_t vector_index = chunk * Traits::kThreads + threadIdx.x;
        if (element_stride == 1 && (base & 3U) == 0) {
            unpack_float4(reinterpret_cast<const float4*>(input + base)[vector_index], values[chunk]);
        } else {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item) {
                const std::uint32_t index = vector_index * 4 + item;
                values[chunk][item]       = input[base + static_cast<std::uint64_t>(index) * element_stride];
            }
        }
        register_wht(values[chunk]);
    }

    warp_wht<Traits::kWarpSize>(values);
    if constexpr (Traits::kWarps > 1) {
        exchange_warps<Traits, true>(values, shared);
        warp_wht<Traits::kWarps>(values);
        exchange_warps<Traits, false>(values, shared);
    }

    if constexpr (Traits::kChunks > 1) {
        float transposed[4][Traits::kChunks];
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item)
                transposed[item][chunk] = values[chunk][item];
        }
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item)
            register_wht(transposed[item]);
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item)
                values[chunk][item] = transposed[item][chunk];
        }
    }

#pragma unroll
    for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
        const std::uint32_t vector_index = chunk * Traits::kThreads + threadIdx.x;
        if (element_stride == 1 && (base & 3U) == 0) {
            float4 result = pack_float4(values[chunk]);
            result.x *= scale;
            result.y *= scale;
            result.z *= scale;
            result.w *= scale;
            reinterpret_cast<float4*>(output + base)[vector_index] = result;
        } else {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item) {
                const std::uint32_t index                                         = vector_index * 4 + item;
                output[base + static_cast<std::uint64_t>(index) * element_stride] = values[chunk][item] * scale;
            }
        }
    }
}

template <std::uint32_t LogN>
void launch_register_fwht(const float* input, float* output, std::uint64_t transforms, std::uint64_t batch_distance, std::uint64_t element_stride,
                          bool normalize, cudaStream_t stream) {
    using Traits      = RegisterFwhtTraits<LogN>;
    const float scale = normalize ? 1.0F / static_cast<float>(Traits::kN) : 1.0F;
    register_fwht_kernel<LogN><<<static_cast<unsigned int>(transforms), Traits::kThreads, Traits::kSharedBytes, stream>>>(
        input, output, transforms, batch_distance, element_stride, scale);
}

template <std::uint32_t LogN, bool Broadcast>
__global__ __launch_bounds__(RegisterFwhtTraits<LogN>::kThreads) void register_structured_kernel(
    const float* input, float* output, const float* matrix_entries, std::uint64_t transforms,
    std::uint64_t batch_distance, std::uint64_t element_stride) {
    using Traits = RegisterFwhtTraits<LogN>;
    extern __shared__ __align__(16) unsigned char storage[];
    float4*                                       shared    = reinterpret_cast<float4*>(storage);
    const float4*                                 matrices  = reinterpret_cast<const float4*>(matrix_entries);
    const float4                                  broadcast_matrix = Broadcast ? matrices[0] : make_float4(0, 0, 0, 0);
    const std::uint64_t                           transform = blockIdx.x;
    if (transform >= transforms)
        return;

    const std::uint64_t base = transform * batch_distance;
    float               values[Traits::kChunks][4];
#pragma unroll
    for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
        const std::uint32_t vector_index = chunk * Traits::kThreads + threadIdx.x;
        if (element_stride == 1 && (base & 3U) == 0) {
            unpack_float4(reinterpret_cast<const float4*>(input + base)[vector_index], values[chunk]);
        } else {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item) {
                const std::uint32_t index = vector_index * 4 + item;
                values[chunk][item]       = input[base + static_cast<std::uint64_t>(index) * element_stride];
            }
        }
        register_structured<Broadcast, 4, 0>(values[chunk], matrices, broadcast_matrix);
    }

    constexpr std::uint32_t kWarpStageBase = 2;
    warp_structured<Broadcast, Traits::kWarpSize, Traits::kChunks, kWarpStageBase>(
        values, matrices, broadcast_matrix);
    if constexpr (Traits::kWarps > 1) {
        exchange_warps<Traits, true>(values, shared);
        constexpr std::uint32_t kCtaStageBase = kWarpStageBase + static_log2(Traits::kWarpSize);
        warp_structured<Broadcast, Traits::kWarps, Traits::kChunks, kCtaStageBase>(
            values, matrices, broadcast_matrix);
        exchange_warps<Traits, false>(values, shared);
    }

    if constexpr (Traits::kChunks > 1) {
        float transposed[4][Traits::kChunks];
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item)
                transposed[item][chunk] = values[chunk][item];
        }
        constexpr std::uint32_t kChunkStageBase = 2 + static_log2(Traits::kThreads);
#pragma unroll
        for (std::uint32_t item = 0; item < 4; ++item)
            register_structured<Broadcast, Traits::kChunks, kChunkStageBase>(
                transposed[item], matrices, broadcast_matrix);
#pragma unroll
        for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item)
                values[chunk][item] = transposed[item][chunk];
        }
    }

#pragma unroll
    for (std::uint32_t chunk = 0; chunk < Traits::kChunks; ++chunk) {
        const std::uint32_t vector_index = chunk * Traits::kThreads + threadIdx.x;
        if (element_stride == 1 && (base & 3U) == 0) {
            reinterpret_cast<float4*>(output + base)[vector_index] = pack_float4(values[chunk]);
        } else {
#pragma unroll
            for (std::uint32_t item = 0; item < 4; ++item) {
                const std::uint32_t index = vector_index * 4 + item;
                output[base + static_cast<std::uint64_t>(index) * element_stride] = values[chunk][item];
            }
        }
    }
}

template <std::uint32_t LogN>
void launch_register_structured(const float* input, float* output, const float* matrices,
                                std::uint64_t transforms, std::uint64_t batch_distance,
                                std::uint64_t element_stride, bool broadcast, cudaStream_t stream) {
    using Traits = RegisterFwhtTraits<LogN>;
    if (broadcast) {
        register_structured_kernel<LogN, true><<<static_cast<unsigned int>(transforms), Traits::kThreads,
                                                Traits::kSharedBytes, stream>>>(
            input, output, matrices, transforms, batch_distance, element_stride);
    } else {
        register_structured_kernel<LogN, false><<<static_cast<unsigned int>(transforms), Traits::kThreads,
                                                 Traits::kSharedBytes, stream>>>(
            input, output, matrices, transforms, batch_distance, element_stride);
    }
}

}  // namespace cuntt::detail
