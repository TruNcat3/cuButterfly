#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cuntt::detail {

template <typename Operator, std::uint32_t LocalLogN, bool Radix4, bool ReorderOutput = false, bool Radix8 = false, bool FinalizeOutput = false>
__global__ void hierarchical_prefix_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_transforms,
                                           std::uint32_t log_n, std::uint64_t batch_distance, std::uint64_t element_stride, Operator op) {
    using Value = typename Operator::Value;
    extern __shared__ __align__(16) unsigned char storage[];
    Value*                                        tile = reinterpret_cast<Value*>(storage);

    constexpr std::uint32_t local_n              = 1U << LocalLogN;
    const std::uint32_t     chunks_per_transform = 1U << (log_n - LocalLogN);
    const std::uint64_t     tile_id              = blockIdx.x;
    const std::uint64_t     transform            = tile_id / chunks_per_transform;
    if (transform >= total_transforms)
        return;
    const std::uint32_t chunk = static_cast<std::uint32_t>(tile_id % chunks_per_transform);
    const std::uint64_t base  = transform * batch_distance;

    for (std::uint32_t index = threadIdx.x; index < local_n; index += blockDim.x) {
        const std::uint32_t global_index = chunk * local_n + index;
        const std::uint32_t input_index  = Operator::kBitReverseInput ? __brev(global_index) >> (32 - log_n) : global_index;
        tile[index]                      = input[base + static_cast<std::uint64_t>(input_index) * element_stride];
    }
    __syncthreads();

    if constexpr (Radix8) {
        constexpr std::uint32_t fused_stages = (LocalLogN / 3) * 3;
        if constexpr (fused_stages > 0) {
#pragma unroll
            for (std::uint32_t stage = 0; stage < fused_stages; stage += 3) {
                const std::uint32_t eighth = 1U << stage;
                for (std::uint32_t unit = threadIdx.x; unit < local_n / 8; unit += blockDim.x) {
                    const std::uint32_t group  = unit / eighth;
                    const std::uint32_t offset = unit - group * eighth;
                    const std::uint32_t i0     = group * (8 * eighth) + offset;
                    const std::uint32_t i1     = i0 + eighth;
                    const std::uint32_t i2     = i1 + eighth;
                    const std::uint32_t i3     = i2 + eighth;
                    const std::uint32_t i4     = i3 + eighth;
                    const std::uint32_t i5     = i4 + eighth;
                    const std::uint32_t i6     = i5 + eighth;
                    const std::uint32_t i7     = i6 + eighth;
                    Value               x0     = tile[i0];
                    Value               x1     = tile[i1];
                    Value               x2     = tile[i2];
                    Value               x3     = tile[i3];
                    Value               x4     = tile[i4];
                    Value               x5     = tile[i5];
                    Value               x6     = tile[i6];
                    Value               x7     = tile[i7];
                    op.apply(stage, offset, x0, x1);
                    op.apply(stage, offset, x2, x3);
                    op.apply(stage, offset, x4, x5);
                    op.apply(stage, offset, x6, x7);
                    op.apply(stage + 1, offset, x0, x2);
                    op.apply(stage + 1, offset + eighth, x1, x3);
                    op.apply(stage + 1, offset, x4, x6);
                    op.apply(stage + 1, offset + eighth, x5, x7);
                    op.apply(stage + 2, offset, x0, x4);
                    op.apply(stage + 2, offset + eighth, x1, x5);
                    op.apply(stage + 2, offset + 2 * eighth, x2, x6);
                    op.apply(stage + 2, offset + 3 * eighth, x3, x7);
                    tile[i0] = x0;
                    tile[i1] = x1;
                    tile[i2] = x2;
                    tile[i3] = x3;
                    tile[i4] = x4;
                    tile[i5] = x5;
                    tile[i6] = x6;
                    tile[i7] = x7;
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (std::uint32_t stage = fused_stages; stage < LocalLogN; ++stage) {
            const std::uint32_t half = 1U << stage;
            for (std::uint32_t butterfly = threadIdx.x; butterfly < local_n / 2; butterfly += blockDim.x) {
                const std::uint32_t group       = butterfly / half;
                const std::uint32_t offset      = butterfly - group * half;
                const std::uint32_t left_index  = group * (2 * half) + offset;
                const std::uint32_t right_index = left_index + half;
                Value               left        = tile[left_index];
                Value               right       = tile[right_index];
                op.apply(stage, offset, left, right);
                tile[left_index]  = left;
                tile[right_index] = right;
            }
            __syncthreads();
        }
    } else if constexpr (Radix4) {
        constexpr std::uint32_t paired_stages = LocalLogN & ~1U;
#pragma unroll
        for (std::uint32_t stage = 0; stage < paired_stages; stage += 2) {
            const std::uint32_t quarter = 1U << stage;
            for (std::uint32_t unit = threadIdx.x; unit < local_n / 4; unit += blockDim.x) {
                const std::uint32_t group  = unit / quarter;
                const std::uint32_t offset = unit - group * quarter;
                const std::uint32_t i0     = group * (4 * quarter) + offset;
                const std::uint32_t i1     = i0 + quarter;
                const std::uint32_t i2     = i1 + quarter;
                const std::uint32_t i3     = i2 + quarter;
                Value               a      = tile[i0];
                Value               b      = tile[i1];
                Value               c      = tile[i2];
                Value               d      = tile[i3];
                op.apply(stage, offset, a, b);
                op.apply(stage, offset, c, d);
                op.apply(stage + 1, offset, a, c);
                op.apply(stage + 1, offset + quarter, b, d);
                tile[i0] = a;
                tile[i1] = b;
                tile[i2] = c;
                tile[i3] = d;
            }
            __syncthreads();
        }
        if constexpr (paired_stages != LocalLogN) {
            constexpr std::uint32_t half = 1U << paired_stages;
            for (std::uint32_t offset = threadIdx.x; offset < half; offset += blockDim.x) {
                Value left  = tile[offset];
                Value right = tile[offset + half];
                op.apply(paired_stages, offset, left, right);
                tile[offset]        = left;
                tile[offset + half] = right;
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (std::uint32_t stage = 0; stage < LocalLogN; ++stage) {
            const std::uint32_t half = 1U << stage;
            for (std::uint32_t butterfly = threadIdx.x; butterfly < local_n / 2; butterfly += blockDim.x) {
                const std::uint32_t group       = butterfly / half;
                const std::uint32_t offset      = butterfly - group * half;
                const std::uint32_t left_index  = group * (2 * half) + offset;
                const std::uint32_t right_index = left_index + half;
                Value               left        = tile[left_index];
                Value               right       = tile[right_index];
                op.apply(stage, offset, left, right);
                tile[left_index]  = left;
                tile[right_index] = right;
            }
            __syncthreads();
        }
    }

    for (std::uint32_t index = threadIdx.x; index < local_n; index += blockDim.x) {
        const std::uint32_t global_index = chunk * local_n + index;
        const std::uint32_t output_index = ReorderOutput ? index * chunks_per_transform + chunk : global_index;
        const Value         value        = FinalizeOutput ? op.finalize(tile[index], local_n) : tile[index];
        output[base + static_cast<std::uint64_t>(output_index) * element_stride] = value;
    }
}

template <typename Operator, std::uint32_t RemainingLogN, bool Radix4, bool Radix8 = false>
__global__ void online_reorder_suffix_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_transforms,
                                             std::uint32_t log_n, std::uint32_t local_log_n, std::uint64_t batch_distance,
                                             std::uint64_t element_stride, std::uint32_t columns_per_tile, Operator op) {
    using Value = typename Operator::Value;
    extern __shared__ __align__(16) unsigned char storage[];
    Value*                                        tile = reinterpret_cast<Value*>(storage);

    constexpr std::uint32_t remaining_n  = 1U << RemainingLogN;
    const std::uint32_t     local_n      = 1U << local_log_n;
    const std::uint32_t     column_tiles = (local_n + columns_per_tile - 1) / columns_per_tile;
    const std::uint64_t     tile_id      = blockIdx.x;
    const std::uint64_t     transform    = tile_id / column_tiles;
    if (transform >= total_transforms)
        return;
    const std::uint32_t column_base    = static_cast<std::uint32_t>(tile_id % column_tiles) * columns_per_tile;
    const std::uint32_t active_columns = min(columns_per_tile, local_n - column_base);
    const std::uint64_t base           = transform * batch_distance;

    for (std::uint32_t work = threadIdx.x; work < active_columns * remaining_n; work += blockDim.x) {
        const std::uint32_t column_slot     = work / remaining_n;
        const std::uint32_t index           = work - column_slot * remaining_n;
        const std::uint64_t reordered_index = static_cast<std::uint64_t>(column_base + column_slot) * remaining_n + index;
        tile[work]                          = input[base + reordered_index * element_stride];
    }
    __syncthreads();

    if constexpr (Radix8) {
        constexpr std::uint32_t fused_stages = (RemainingLogN / 3) * 3;
        if constexpr (fused_stages > 0) {
#pragma unroll
            for (std::uint32_t local_stage = 0; local_stage < fused_stages; local_stage += 3) {
                const std::uint32_t eighth = 1U << local_stage;
                const std::uint32_t stage  = local_log_n + local_stage;
                for (std::uint32_t work = threadIdx.x; work < active_columns * (remaining_n / 8); work += blockDim.x) {
                    const std::uint32_t column_slot   = work / (remaining_n / 8);
                    const std::uint32_t unit          = work - column_slot * (remaining_n / 8);
                    const std::uint32_t column        = column_base + column_slot;
                    const std::uint32_t tile_base     = column_slot * remaining_n;
                    const std::uint32_t group         = unit / eighth;
                    const std::uint32_t offset        = unit - group * eighth;
                    const std::uint32_t i0            = tile_base + group * (8 * eighth) + offset;
                    const std::uint32_t i1            = i0 + eighth;
                    const std::uint32_t i2            = i1 + eighth;
                    const std::uint32_t i3            = i2 + eighth;
                    const std::uint32_t i4            = i3 + eighth;
                    const std::uint32_t i5            = i4 + eighth;
                    const std::uint32_t i6            = i5 + eighth;
                    const std::uint32_t i7            = i6 + eighth;
                    Value               x0            = tile[i0];
                    Value               x1            = tile[i1];
                    Value               x2            = tile[i2];
                    Value               x3            = tile[i3];
                    Value               x4            = tile[i4];
                    Value               x5            = tile[i5];
                    Value               x6            = tile[i6];
                    Value               x7            = tile[i7];
                    const auto          global_offset = [&](std::uint32_t value) { return (value << local_log_n) + column; };
                    op.apply(stage, global_offset(offset), x0, x1);
                    op.apply(stage, global_offset(offset), x2, x3);
                    op.apply(stage, global_offset(offset), x4, x5);
                    op.apply(stage, global_offset(offset), x6, x7);
                    op.apply(stage + 1, global_offset(offset), x0, x2);
                    op.apply(stage + 1, global_offset(offset + eighth), x1, x3);
                    op.apply(stage + 1, global_offset(offset), x4, x6);
                    op.apply(stage + 1, global_offset(offset + eighth), x5, x7);
                    op.apply(stage + 2, global_offset(offset), x0, x4);
                    op.apply(stage + 2, global_offset(offset + eighth), x1, x5);
                    op.apply(stage + 2, global_offset(offset + 2 * eighth), x2, x6);
                    op.apply(stage + 2, global_offset(offset + 3 * eighth), x3, x7);
                    tile[i0] = x0;
                    tile[i1] = x1;
                    tile[i2] = x2;
                    tile[i3] = x3;
                    tile[i4] = x4;
                    tile[i5] = x5;
                    tile[i6] = x6;
                    tile[i7] = x7;
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (std::uint32_t local_stage = fused_stages; local_stage < RemainingLogN; ++local_stage) {
            const std::uint32_t half  = 1U << local_stage;
            const std::uint32_t stage = local_log_n + local_stage;
            for (std::uint32_t work = threadIdx.x; work < active_columns * (remaining_n / 2); work += blockDim.x) {
                const std::uint32_t column_slot = work / (remaining_n / 2);
                const std::uint32_t butterfly   = work - column_slot * (remaining_n / 2);
                const std::uint32_t column      = column_base + column_slot;
                const std::uint32_t tile_base   = column_slot * remaining_n;
                const std::uint32_t group       = butterfly / half;
                const std::uint32_t offset      = butterfly - group * half;
                const std::uint32_t left_index  = tile_base + group * (2 * half) + offset;
                const std::uint32_t right_index = left_index + half;
                Value               left        = tile[left_index];
                Value               right       = tile[right_index];
                op.apply(stage, (offset << local_log_n) + column, left, right);
                tile[left_index]  = left;
                tile[right_index] = right;
            }
            __syncthreads();
        }
    } else if constexpr (Radix4) {
        constexpr std::uint32_t paired_stages = RemainingLogN & ~1U;
        if constexpr (paired_stages > 0) {
#pragma unroll
            for (std::uint32_t local_stage = 0; local_stage < paired_stages; local_stage += 2) {
                const std::uint32_t quarter = 1U << local_stage;
                const std::uint32_t stage   = local_log_n + local_stage;
                for (std::uint32_t work = threadIdx.x; work < active_columns * (remaining_n / 4); work += blockDim.x) {
                    const std::uint32_t column_slot  = work / (remaining_n / 4);
                    const std::uint32_t unit         = work - column_slot * (remaining_n / 4);
                    const std::uint32_t column       = column_base + column_slot;
                    const std::uint32_t tile_base    = column_slot * remaining_n;
                    const std::uint32_t group        = unit / quarter;
                    const std::uint32_t offset       = unit - group * quarter;
                    const std::uint32_t i0           = tile_base + group * (4 * quarter) + offset;
                    const std::uint32_t i1           = i0 + quarter;
                    const std::uint32_t i2           = i1 + quarter;
                    const std::uint32_t i3           = i2 + quarter;
                    const std::uint32_t first_offset = (offset << local_log_n) + column;
                    Value               a            = tile[i0];
                    Value               b            = tile[i1];
                    Value               c            = tile[i2];
                    Value               d            = tile[i3];
                    op.apply(stage, first_offset, a, b);
                    op.apply(stage, first_offset, c, d);
                    op.apply(stage + 1, first_offset, a, c);
                    op.apply(stage + 1, ((offset + quarter) << local_log_n) + column, b, d);
                    tile[i0] = a;
                    tile[i1] = b;
                    tile[i2] = c;
                    tile[i3] = d;
                }
                __syncthreads();
            }
        }
        if constexpr (paired_stages != RemainingLogN) {
            constexpr std::uint32_t half  = 1U << paired_stages;
            const std::uint32_t     stage = local_log_n + paired_stages;
            for (std::uint32_t work = threadIdx.x; work < active_columns * half; work += blockDim.x) {
                const std::uint32_t column_slot = work / half;
                const std::uint32_t offset      = work - column_slot * half;
                const std::uint32_t column      = column_base + column_slot;
                const std::uint32_t tile_base   = column_slot * remaining_n;
                Value               left        = tile[tile_base + offset];
                Value               right       = tile[tile_base + offset + half];
                op.apply(stage, (offset << local_log_n) + column, left, right);
                tile[tile_base + offset]        = left;
                tile[tile_base + offset + half] = right;
            }
            __syncthreads();
        }
    } else {
#pragma unroll
        for (std::uint32_t local_stage = 0; local_stage < RemainingLogN; ++local_stage) {
            const std::uint32_t half  = 1U << local_stage;
            const std::uint32_t stage = local_log_n + local_stage;
            for (std::uint32_t work = threadIdx.x; work < active_columns * (remaining_n / 2); work += blockDim.x) {
                const std::uint32_t column_slot = work / (remaining_n / 2);
                const std::uint32_t butterfly   = work - column_slot * (remaining_n / 2);
                const std::uint32_t column      = column_base + column_slot;
                const std::uint32_t tile_base   = column_slot * remaining_n;
                const std::uint32_t group       = butterfly / half;
                const std::uint32_t offset      = butterfly - group * half;
                const std::uint32_t left_index  = tile_base + group * (2 * half) + offset;
                const std::uint32_t right_index = left_index + half;
                Value               left        = tile[left_index];
                Value               right       = tile[right_index];
                op.apply(stage, (offset << local_log_n) + column, left, right);
                tile[left_index]  = left;
                tile[right_index] = right;
            }
            __syncthreads();
        }
    }

    const std::uint32_t n = 1U << log_n;
    for (std::uint32_t work = threadIdx.x; work < active_columns * remaining_n; work += blockDim.x) {
        const std::uint32_t column_slot               = work / remaining_n;
        const std::uint32_t index                     = work - column_slot * remaining_n;
        const std::uint32_t column                    = column_base + column_slot;
        const std::uint64_t logical_index             = static_cast<std::uint64_t>(index) * local_n + column;
        output[base + logical_index * element_stride] = op.finalize(tile[work], n);
    }
}

template <typename Operator>
__global__ void hierarchical_stage_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_transforms,
                                          std::uint32_t log_n, std::uint32_t stage, std::uint64_t batch_distance, std::uint64_t element_stride,
                                          bool final_stage, Operator op) {
    const std::uint64_t butterflies_per_transform = std::uint64_t{1} << (log_n - 1);
    const std::uint64_t work                      = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total_work                = total_transforms * butterflies_per_transform;
    if (work >= total_work)
        return;
    const std::uint64_t      transform   = work / butterflies_per_transform;
    const std::uint64_t      butterfly   = work - transform * butterflies_per_transform;
    const std::uint32_t      half        = 1U << stage;
    const std::uint64_t      group       = butterfly / half;
    const std::uint32_t      offset      = static_cast<std::uint32_t>(butterfly - group * half);
    const std::uint64_t      left_index  = group * (2ULL * half) + offset;
    const std::uint64_t      right_index = left_index + half;
    const std::uint64_t      base        = transform * batch_distance;
    typename Operator::Value left        = input[base + left_index * element_stride];
    typename Operator::Value right       = input[base + right_index * element_stride];
    op.apply(stage, offset, left, right);
    if (final_stage) {
        const std::uint32_t n = 1U << log_n;
        left                  = op.finalize(left, n);
        right                 = op.finalize(right, n);
    }
    output[base + left_index * element_stride]  = left;
    output[base + right_index * element_stride] = right;
}

__device__ __forceinline__ void stage_named_barrier(std::uint32_t barrier) {
    asm volatile("bar.sync %0, 64;" : : "r"(barrier) : "memory");
}

template <typename Operator, std::uint32_t StageSpace, bool NamedBarrier, std::uint32_t PipelineWarps = 8>
__global__ void stage_pipeline_256_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_tiles,
                                          std::uint32_t stage_base, std::uint64_t input_distance, std::uint64_t output_distance,
                                          std::uint64_t element_stride, Operator op) {
    using Value = typename Operator::Value;
    constexpr std::uint32_t kTilePoints        = 256;
    constexpr std::uint32_t kWarpsPerBlock     = PipelineWarps;
    constexpr std::uint32_t kTokensPerPipeline = 2;
    constexpr std::uint32_t kPipelines         = kWarpsPerBlock / StageSpace;
    static_assert(StageSpace == 1 || StageSpace == 2 || StageSpace == 4 || StageSpace == 8);
    static_assert(PipelineWarps == 4 || PipelineWarps == 8);
    static_assert(StageSpace <= PipelineWarps && PipelineWarps % StageSpace == 0);

    __shared__ Value buffers[kWarpsPerBlock * kTokensPerPipeline * kTilePoints];
    __shared__ int   ready[kWarpsPerBlock * kTokensPerPipeline];

    const std::uint32_t warp     = threadIdx.x / warpSize;
    const std::uint32_t lane     = threadIdx.x % warpSize;
    const std::uint32_t pipeline = warp / StageSpace;
    const std::uint32_t role     = warp % StageSpace;
    if constexpr (!NamedBarrier) {
        if (threadIdx.x < kWarpsPerBlock * kTokensPerPipeline) {
            ready[threadIdx.x] = 0;
        }
        __syncthreads();
    }

    for (std::uint32_t token = 0; token < kTokensPerPipeline; ++token) {
        const std::uint64_t tile = static_cast<std::uint64_t>(blockIdx.x) * (kPipelines * kTokensPerPipeline) +
                                   pipeline * kTokensPerPipeline + token;
        const bool active = tile < total_tiles;
        if (role != 0) {
            if constexpr (NamedBarrier) {
                stage_named_barrier(pipeline * (StageSpace - 1) + role - 1);
            } else {
                const std::uint32_t previous_flag = ((pipeline * StageSpace + role - 1) * kTokensPerPipeline) + token;
                while (atomicAdd(&ready[previous_flag], 0) == 0) {
                }
                __threadfence_block();
            }
        }
        __syncwarp();

        const std::uint32_t stage       = stage_base + role;
        const std::uint32_t half        = 1U << stage;
        const std::uint32_t source      = ((pipeline * StageSpace + (role == 0 ? 0 : role - 1)) * kTokensPerPipeline + token) * kTilePoints;
        const std::uint32_t target      = ((pipeline * StageSpace + role) * kTokensPerPipeline + token) * kTilePoints;
        const std::uint64_t global_input_base  = tile * input_distance;
        const std::uint64_t global_output_base = tile * output_distance;

        for (std::uint32_t butterfly_index = lane; butterfly_index < kTilePoints / 2; butterfly_index += warpSize) {
            const std::uint32_t group       = butterfly_index / half;
            const std::uint32_t offset      = butterfly_index - group * half;
            const std::uint32_t left_index  = group * (2 * half) + offset;
            const std::uint32_t right_index = left_index + half;
            Value               left{};
            Value               right{};
            if (active) {
                if (role == 0) {
                    const bool          reverse     = Operator::kBitReverseInput && stage_base == 0;
                    const std::uint32_t left_input  = reverse ? __brev(left_index) >> 24 : left_index;
                    const std::uint32_t right_input = reverse ? __brev(right_index) >> 24 : right_index;
                    left                            = input[global_input_base + left_input * element_stride];
                    right                           = input[global_input_base + right_input * element_stride];
                } else {
                    left  = buffers[source + left_index];
                    right = buffers[source + right_index];
                }
            }
            op.apply(stage, offset, left, right);
            buffers[target + left_index]  = left;
            buffers[target + right_index] = right;
        }
        __syncwarp();
        __threadfence_block();

        if (role + 1 < StageSpace) {
            if constexpr (NamedBarrier) {
                stage_named_barrier(pipeline * (StageSpace - 1) + role);
            } else if (lane == 0) {
                const std::uint32_t flag = ((pipeline * StageSpace + role) * kTokensPerPipeline) + token;
                atomicExch(&ready[flag], 1);
            }
        } else if (active) {
            for (std::uint32_t index = lane; index < kTilePoints; index += warpSize) {
                const Value value = buffers[target + index];
                output[global_output_base + index * element_stride] = stage_base + StageSpace >= 8 ? op.finalize(value, kTilePoints) : value;
            }
        }
    }
}

template <typename Operator>
__global__ void warp_hybrid_256_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_tiles,
                                       std::uint32_t warp_stages, std::uint64_t batch_distance, std::uint64_t element_stride, Operator op) {
    using Value = typename Operator::Value;
    __shared__ Value tile[256];

    const std::uint64_t transform = blockIdx.x;
    if (transform >= total_tiles) {
        return;
    }
    const std::uint32_t index       = threadIdx.x;
    const std::uint64_t base        = transform * batch_distance;
    const std::uint32_t input_index = Operator::kBitReverseInput ? __brev(index) >> 24 : index;
    Value               value       = input[base + input_index * element_stride];

    for (std::uint32_t stage = 0; stage < warp_stages; ++stage) {
        const std::uint32_t half    = 1U << stage;
        const std::uint32_t offset  = index & (half - 1);
        const Value         partner = op.shuffle(value, half);
        value                       = op.apply_lane(stage, offset, value, partner, (index & half) != 0);
    }

    tile[index] = value;
    __syncthreads();
    for (std::uint32_t stage = warp_stages; stage < 8; ++stage) {
        const std::uint32_t half = 1U << stage;
        if (index < 128) {
            const std::uint32_t group       = index / half;
            const std::uint32_t offset      = index - group * half;
            const std::uint32_t left_index  = group * (2 * half) + offset;
            const std::uint32_t right_index = left_index + half;
            Value               left        = tile[left_index];
            Value               right       = tile[right_index];
            op.apply(stage, offset, left, right);
            tile[left_index]  = left;
            tile[right_index] = right;
        }
        __syncthreads();
    }
    output[base + index * element_stride] = op.finalize(tile[index], 256);
}

template <typename Operator, std::uint32_t LogN>
__global__ void temporal_tile_kernel(const typename Operator::Value* input, typename Operator::Value* output, std::uint64_t total_tiles,
                                     std::uint64_t batch_distance, std::uint64_t element_stride, Operator op) {
    using Value = typename Operator::Value;
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    const std::uint64_t transform = blockIdx.x;
    if (transform >= total_tiles) {
        return;
    }
    constexpr std::uint32_t n = 1U << LogN;
    const std::uint64_t base = transform * batch_distance;
    for (std::uint32_t index = threadIdx.x; index < n; index += blockDim.x) {
        const std::uint32_t input_index = Operator::kBitReverseInput ? __brev(index) >> (32 - LogN) : index;
        tile[index]                    = input[base + input_index * element_stride];
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t stage = 0; stage < LogN; ++stage) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t butterfly_index = threadIdx.x; butterfly_index < n / 2; butterfly_index += blockDim.x) {
            const std::uint32_t group       = butterfly_index / half;
            const std::uint32_t offset      = butterfly_index - group * half;
            const std::uint32_t left_index  = group * (2 * half) + offset;
            const std::uint32_t right_index = left_index + half;
            Value               left        = tile[left_index];
            Value               right       = tile[right_index];
            op.apply(stage, offset, left, right);
            tile[left_index]  = left;
            tile[right_index] = right;
        }
        __syncthreads();
    }

    for (std::uint32_t index = threadIdx.x; index < n; index += blockDim.x) {
        output[base + index * element_stride] = op.finalize(tile[index], n);
    }
}

template <typename Operator, std::uint32_t LogN>
__global__ void temporal_tile_radix4_kernel(const typename Operator::Value* input, typename Operator::Value* output,
                                            std::uint64_t total_tiles, std::uint64_t batch_distance, std::uint64_t element_stride, Operator op) {
    using Value = typename Operator::Value;
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    const std::uint64_t transform = blockIdx.x;
    if (transform >= total_tiles) {
        return;
    }
    constexpr std::uint32_t n = 1U << LogN;
    const std::uint64_t base = transform * batch_distance;
    for (std::uint32_t index = threadIdx.x; index < n; index += blockDim.x) {
        const std::uint32_t input_index = Operator::kBitReverseInput ? __brev(index) >> (32 - LogN) : index;
        tile[index]                    = input[base + input_index * element_stride];
    }
    __syncthreads();

    constexpr std::uint32_t paired_stages = LogN & ~1U;
    if constexpr (paired_stages > 0) {
#pragma unroll
        for (int stage_index = 0; stage_index < static_cast<int>(paired_stages); stage_index += 2) {
            const std::uint32_t stage = static_cast<std::uint32_t>(stage_index);
            const std::uint32_t quarter = 1U << stage;
            for (std::uint32_t unit = threadIdx.x; unit < n / 4; unit += blockDim.x) {
            const std::uint32_t group  = unit / quarter;
            const std::uint32_t offset = unit - group * quarter;
            const std::uint32_t i0     = group * (4 * quarter) + offset;
            const std::uint32_t i1     = i0 + quarter;
            const std::uint32_t i2     = i1 + quarter;
            const std::uint32_t i3     = i2 + quarter;
            Value               a      = tile[i0];
            Value               b      = tile[i1];
            Value               c      = tile[i2];
            Value               d      = tile[i3];
            op.apply(stage, offset, a, b);
            op.apply(stage, offset, c, d);
            op.apply(stage + 1, offset, a, c);
            op.apply(stage + 1, offset + quarter, b, d);
            tile[i0] = a;
            tile[i1] = b;
            tile[i2] = c;
            tile[i3] = d;
            }
            __syncthreads();
        }
    }

    if constexpr (paired_stages != LogN) {
        const std::uint32_t half = 1U << paired_stages;
        for (std::uint32_t offset = threadIdx.x; offset < half; offset += blockDim.x) {
            Value left  = tile[offset];
            Value right = tile[offset + half];
            op.apply(paired_stages, offset, left, right);
            tile[offset]        = left;
            tile[offset + half] = right;
        }
        __syncthreads();
    }

    for (std::uint32_t index = threadIdx.x; index < n; index += blockDim.x) {
        output[base + index * element_stride] = op.finalize(tile[index], n);
    }
}

}  // namespace cuntt::detail
