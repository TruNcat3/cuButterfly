#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>

#include "cuntt/ntt.hpp"

namespace cuntt::detail {

struct HybridDataflowRuntime {
    std::uint32_t log_n;
    std::uint32_t data_space;
    std::uint32_t data_time;
    std::uint32_t pipeline_buffers;
    DataflowLayout layout;
    DataflowStateMode state_mode;
    StageHandoff handoff;
};

template <typename Word>
__device__ __forceinline__ std::uint32_t hybrid_state_index(
    std::uint32_t logical, std::uint32_t flow_tile_log_n, std::uint32_t data_space,
    DataflowLayout layout) {
    if (layout == DataflowLayout::Linear) {
        return logical;
    }
    const std::uint32_t bank_count = data_space << 1;
    const std::uint32_t bank_mask  = bank_count - 1;
    const std::uint32_t bank       = (logical ^ (logical >> flow_tile_log_n)) & bank_mask;
    return (logical & ~bank_mask) | bank;
}

__device__ __forceinline__ void hybrid_named_barrier(std::uint32_t barrier) {
    asm volatile("bar.sync %0, 64;" : : "r"(barrier) : "memory");
}

__device__ __forceinline__ void hybrid_wait_flag(int* flag, int expected, std::uint32_t lane) {
    int value = 0;
    do {
        if (lane == 0) {
            value = atomicAdd(flag, 0);
        }
        value = __shfl_sync(0xffffffffU, value, 0);
    } while (value != expected);
    __syncwarp();
    __threadfence_block();
}

template <typename Word>
__device__ __forceinline__ Word hybrid_shuffle_xor(Word value, int delta) {
    if constexpr (sizeof(Word) == sizeof(std::uint32_t)) {
        return static_cast<Word>(__shfl_xor_sync(0xffffffffU, static_cast<std::uint32_t>(value), delta));
    } else {
        const std::uint64_t bits = static_cast<std::uint64_t>(value);
        const std::uint32_t low = __shfl_xor_sync(0xffffffffU, static_cast<std::uint32_t>(bits), delta);
        const std::uint32_t high = __shfl_xor_sync(0xffffffffU, static_cast<std::uint32_t>(bits >> 32), delta);
        return static_cast<Word>((static_cast<std::uint64_t>(high) << 32) | low);
    }
}

template <typename Word, std::uint32_t ValuesPerLane, typename Operator>
__device__ __forceinline__ void hybrid_apply_register_stage(
    Word (&values)[ValuesPerLane], std::uint32_t token_points, std::uint32_t local_stage,
    std::uint32_t logical_stage_base, std::uint32_t subgraph_offset,
    std::uint32_t lane, Operator op) {
    const std::uint32_t half = 1U << local_stage;
    if (local_stage < 5) {
        const int lane_delta = 1 << local_stage;
        #pragma unroll
        for (std::uint32_t slot_index = 0; slot_index < ValuesPerLane; ++slot_index) {
            Word left = values[slot_index];
            Word right = hybrid_shuffle_xor(values[slot_index], lane_delta);
            Word right_result = Word{0};
            const std::uint32_t left_index = (slot_index << 5) + lane;
            if (left_index < token_points && (lane & lane_delta) == 0) {
                const std::uint32_t offset = left_index & (half - 1);
                const std::uint32_t logical_offset = subgraph_offset + (offset << logical_stage_base);
                op.apply(logical_stage_base + local_stage, logical_offset, left, right);
                right_result = right;
            }
            const Word received_right = hybrid_shuffle_xor(right_result, lane_delta);
            if ((lane & lane_delta) == 0) {
                values[slot_index] = left;
            } else if (left_index < token_points) {
                values[slot_index] = received_right;
            }
        }
    } else {
        const std::uint32_t slot_delta = 1U << (local_stage - 5);
        #pragma unroll
        for (std::uint32_t slot_index = 0; slot_index < ValuesPerLane; ++slot_index) {
            if ((slot_index & slot_delta) == 0 && slot_index + slot_delta < ValuesPerLane) {
                const std::uint32_t left_index = (slot_index << 5) + lane;
                if (left_index < token_points) {
                    Word left = values[slot_index];
                    Word right = values[slot_index + slot_delta];
                    const std::uint32_t offset = left_index & (half - 1);
                    const std::uint32_t logical_offset = subgraph_offset + (offset << logical_stage_base);
                    op.apply(logical_stage_base + local_stage, logical_offset, left, right);
                    values[slot_index] = left;
                    values[slot_index + slot_delta] = right;
                }
            }
        }
    }
}

template <typename Word, std::uint32_t FlowTileLogN, std::uint32_t StageSpace,
          std::uint32_t DataTime, std::uint32_t RoleStages,
          std::uint32_t TargetCtasPerSm, std::uint32_t TokenInterleave, typename Operator>
__global__ __launch_bounds__(((StageSpace + RoleStages - 1) / RoleStages) * RoleStages * 32,
                             TargetCtasPerSm)
void hybrid_dataflow_kernel(const Word* input, Word* output, std::uint64_t transforms,
                            HybridDataflowRuntime runtime, Operator op) {
    static_assert(StageSpace >= 2 && StageSpace <= 8);
    static_assert(StageSpace <= FlowTileLogN);
    static_assert(DataTime >= 1 && DataTime <= 16);
    static_assert(RoleStages >= 1 && RoleStages <= StageSpace);
    static_assert(TargetCtasPerSm >= 1 && TargetCtasPerSm <= 3);
    static_assert(TokenInterleave == 1 || TokenInterleave == 2);
    static_assert(TokenInterleave == 1 || (RoleStages > 1 && DataTime >= RoleStages * TokenInterleave));
    constexpr std::uint32_t kMaxTokenPoints = 1U << StageSpace;
    constexpr std::uint32_t kMaxRoles       = (StageSpace + RoleStages - 1) / RoleStages;
    constexpr std::uint32_t kRoleReplicas   = RoleStages;
    constexpr std::uint32_t kEdges          = kMaxRoles - 1;

    extern __shared__ __align__(16) unsigned char storage[];
    const std::uint32_t n = 1U << runtime.log_n;
    const bool ping_pong = runtime.state_mode == DataflowStateMode::PingPong;
    Word* state0 = reinterpret_cast<Word*>(storage);
    Word* state1 = ping_pong ? state0 + n : state0;
    Word* channels = state0 + n * (ping_pong ? 2U : 1U);
    constexpr std::size_t packet_words = static_cast<std::size_t>(kMaxTokenPoints) * DataTime;
    const std::size_t channel_words = static_cast<std::size_t>(kEdges) * runtime.pipeline_buffers * packet_words;
    int* flags = reinterpret_cast<int*>(channels + channel_words);

    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t role = warp / kRoleReplicas;
    const std::uint32_t replica = warp - role * kRoleReplicas;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint64_t transform = blockIdx.x;
    if (transform >= transforms) {
        return;
    }

    for (std::uint32_t index = threadIdx.x;
         index < kEdges * runtime.pipeline_buffers * kRoleReplicas; index += blockDim.x) {
        flags[index] = 0;
    }
    const std::uint64_t base = transform * n;
    for (std::uint32_t logical = threadIdx.x; logical < n; logical += blockDim.x) {
        const std::uint32_t reversed = __brev(logical) >> (32 - runtime.log_n);
        const std::uint32_t physical = hybrid_state_index<Word>(logical, FlowTileLogN, runtime.data_space, runtime.layout);
        state0[physical] = input[base + reversed];
    }
    __syncthreads();

    bool current_is_state0 = true;
    for (std::uint32_t round_base = 0; round_base < runtime.log_n; round_base += FlowTileLogN) {
        const std::uint32_t round_stages = min(FlowTileLogN, runtime.log_n - round_base);
        for (std::uint32_t fold = 0; fold < round_stages; fold += StageSpace) {
            const std::uint32_t active_stages = min(StageSpace, round_stages - fold);
            const std::uint32_t active_roles = (active_stages + RoleStages - 1) / RoleStages;
            const std::uint32_t logical_stage_base = round_base + fold;
            const std::uint32_t token_points = 1U << active_stages;
            const std::uint32_t token_count  = n / token_points;
            const std::uint32_t packet_count = (token_count + DataTime - 1) / DataTime;
            Word* input_state  = current_is_state0 ? state0 : state1;
            Word* output_state = ping_pong ? (current_is_state0 ? state1 : state0) : input_state;

            if (role < active_roles) {
              for (std::uint32_t packet = 0; packet < packet_count; ++packet) {
                const std::uint32_t slot = packet % runtime.pipeline_buffers;
                const std::uint32_t token_begin = packet * DataTime;
                const std::uint32_t packet_tokens = min(DataTime, token_count - token_begin);
                Word* source = nullptr;
                Word* target = nullptr;
                const std::uint32_t previous_edge = role == 0 ? 0 : role - 1;
                const std::size_t previous_flag =
                    (static_cast<std::size_t>(previous_edge) * runtime.pipeline_buffers + slot) * kRoleReplicas + replica;
                if (role != 0) {
                    source = channels + (static_cast<std::size_t>(previous_edge) * runtime.pipeline_buffers + slot) * packet_words;
                    if (runtime.handoff == StageHandoff::NamedBarrier) {
                        hybrid_named_barrier(previous_edge * kRoleReplicas + replica + 1);
                    } else {
                        hybrid_wait_flag(&flags[previous_flag], 1, lane);
                    }
                }
                if (role + 1 < active_roles) {
                    const std::uint32_t edge = role;
                    const std::size_t edge_flag =
                        (static_cast<std::size_t>(edge) * runtime.pipeline_buffers + slot) * kRoleReplicas + replica;
                    target = channels + (static_cast<std::size_t>(edge) * runtime.pipeline_buffers + slot) * packet_words;
                    // Ready uses the selected warp handoff. Slot release stays asynchronous so a
                    // consumer never blocks the producer iteration needed to reach the same slot.
                    if (packet >= runtime.pipeline_buffers) {
                        hybrid_wait_flag(&flags[edge_flag], 0, lane);
                    }
                }

                if constexpr (RoleStages == 1) {
                    #pragma unroll
                    for (std::uint32_t packet_token = replica; packet_token < DataTime;
                         packet_token += kRoleReplicas) {
                        if (packet_token >= packet_tokens) {
                            continue;
                        }
                        const std::uint32_t token = token_begin + packet_token;
                        Word* source_token = source == nullptr ? nullptr : source + packet_token * kMaxTokenPoints;
                        Word* target_token = target == nullptr ? nullptr : target + packet_token * kMaxTokenPoints;
                        const std::uint32_t stage_stride = 1U << logical_stage_base;
                        const std::uint32_t subgraph_block = token / stage_stride;
                        const std::uint32_t subgraph_offset = token - subgraph_block * stage_stride;
                        const std::uint32_t subgraph_base = subgraph_block << (logical_stage_base + active_stages);
                        auto load_value = [&](std::uint32_t local) {
                            if (role != 0) {
                                return source_token[local];
                            }
                            const std::uint32_t logical = subgraph_base + subgraph_offset + (local << logical_stage_base);
                            return input_state[hybrid_state_index<Word>(logical, FlowTileLogN, runtime.data_space, runtime.layout)];
                        };
                        auto store_value = [&](std::uint32_t local, Word value) {
                            if (role + 1 < active_roles) {
                                target_token[local] = value;
                            } else {
                                const std::uint32_t logical = subgraph_base + subgraph_offset + (local << logical_stage_base);
                                output_state[hybrid_state_index<Word>(logical, FlowTileLogN, runtime.data_space, runtime.layout)] = value;
                            }
                        };
                        const std::uint32_t local_stage = role;
                        const std::uint32_t half = 1U << local_stage;
                        for (std::uint32_t wave = 0; wave < token_points / 2; wave += runtime.data_space) {
                            const std::uint32_t butterfly = wave + lane;
                            if (lane < runtime.data_space && butterfly < token_points / 2) {
                                const std::uint32_t group = butterfly / half;
                                const std::uint32_t offset = butterfly - group * half;
                                const std::uint32_t left_index = group * (half << 1) + offset;
                                const std::uint32_t right_index = left_index + half;
                                Word left = load_value(left_index);
                                Word right = load_value(right_index);
                                const std::uint32_t logical_offset = subgraph_offset + (offset << logical_stage_base);
                                op.apply(logical_stage_base + local_stage, logical_offset, left, right);
                                store_value(left_index, left);
                                store_value(right_index, right);
                            }
                        }
                    }
                } else {
                    constexpr std::uint32_t kValuesPerLane = (kMaxTokenPoints + 31) / 32;
                    #pragma unroll
                    for (std::uint32_t group_token = replica * TokenInterleave; group_token < DataTime;
                         group_token += kRoleReplicas * TokenInterleave) {
                        Word values[TokenInterleave][kValuesPerLane];
                        bool valid[TokenInterleave];
                        std::uint32_t subgraph_offsets[TokenInterleave];
                        std::uint32_t subgraph_bases[TokenInterleave];

                        #pragma unroll
                        for (std::uint32_t interleave = 0; interleave < TokenInterleave; ++interleave) {
                            const std::uint32_t packet_token = group_token + interleave;
                            valid[interleave] = packet_token < packet_tokens;
                            const std::uint32_t token = token_begin + packet_token;
                            const std::uint32_t stage_stride = 1U << logical_stage_base;
                            const std::uint32_t subgraph_block = token / stage_stride;
                            subgraph_offsets[interleave] = token - subgraph_block * stage_stride;
                            subgraph_bases[interleave] = subgraph_block << (logical_stage_base + active_stages);
                            Word* source_token = source == nullptr ? nullptr : source + packet_token * kMaxTokenPoints;
                            #pragma unroll
                            for (std::uint32_t slot_index = 0; slot_index < kValuesPerLane; ++slot_index) {
                                const std::uint32_t local = (slot_index << 5) + lane;
                                Word value = Word{0};
                                if (valid[interleave] && local < token_points) {
                                    if (role != 0) {
                                        value = source_token[local];
                                    } else {
                                        const std::uint32_t logical = subgraph_bases[interleave] +
                                            subgraph_offsets[interleave] + (local << logical_stage_base);
                                        value = input_state[hybrid_state_index<Word>(logical, FlowTileLogN,
                                                                                    runtime.data_space, runtime.layout)];
                                    }
                                }
                                values[interleave][slot_index] = value;
                            }
                        }

                        const std::uint32_t role_stage_begin = role * RoleStages;
                        #pragma unroll
                        for (std::uint32_t fused_stage = 0; fused_stage < RoleStages; ++fused_stage) {
                            const std::uint32_t local_stage = role_stage_begin + fused_stage;
                            if (local_stage >= active_stages) {
                                continue;
                            }
                            #pragma unroll
                            for (std::uint32_t interleave = 0; interleave < TokenInterleave; ++interleave) {
                                if (valid[interleave]) {
                                    hybrid_apply_register_stage<Word, kValuesPerLane>(
                                        values[interleave], token_points, local_stage, logical_stage_base,
                                        subgraph_offsets[interleave], lane, op);
                                }
                            }
                        }

                        #pragma unroll
                        for (std::uint32_t interleave = 0; interleave < TokenInterleave; ++interleave) {
                            const std::uint32_t packet_token = group_token + interleave;
                            Word* target_token = target == nullptr ? nullptr : target + packet_token * kMaxTokenPoints;
                            #pragma unroll
                            for (std::uint32_t slot_index = 0; slot_index < kValuesPerLane; ++slot_index) {
                                const std::uint32_t local = (slot_index << 5) + lane;
                                if (valid[interleave] && local < token_points) {
                                    if (role + 1 < active_roles) {
                                        target_token[local] = values[interleave][slot_index];
                                    } else {
                                        const std::uint32_t logical = subgraph_bases[interleave] +
                                            subgraph_offsets[interleave] + (local << logical_stage_base);
                                        output_state[hybrid_state_index<Word>(logical, FlowTileLogN,
                                                                             runtime.data_space, runtime.layout)] =
                                            values[interleave][slot_index];
                                    }
                                }
                            }
                        }
                    }
                }
                __syncwarp();
                __threadfence_block();

                if (role + 1 < active_roles) {
                    const std::uint32_t edge = role;
                    const std::size_t edge_flag =
                        (static_cast<std::size_t>(edge) * runtime.pipeline_buffers + slot) * kRoleReplicas + replica;
                    if (runtime.handoff == StageHandoff::NamedBarrier) {
                        if (lane == 0) {
                            atomicExch(&flags[edge_flag], 1);
                        }
                        hybrid_named_barrier(edge * kRoleReplicas + replica + 1);
                    } else if (lane == 0) {
                        atomicExch(&flags[edge_flag], 1);
                    }
                }
                if (role != 0) {
                    if (lane == 0) {
                        atomicExch(&flags[previous_flag], 0);
                    }
                }
              }
            }
            __syncthreads();
            if (ping_pong) {
                current_is_state0 = !current_is_state0;
            }
        }
    }

    Word* final_state = current_is_state0 ? state0 : state1;
    for (std::uint32_t logical = threadIdx.x; logical < n; logical += blockDim.x) {
        const std::uint32_t physical = hybrid_state_index<Word>(logical, FlowTileLogN, runtime.data_space, runtime.layout);
        output[base + logical] = op.finalize(final_state[physical], n);
    }
}

template <typename Word, std::uint32_t FlowTileLogN, std::uint32_t StageSpace,
          std::uint32_t DataTime, std::uint32_t RoleStages>
constexpr std::size_t hybrid_dataflow_shared_bytes(std::uint32_t log_n, std::uint32_t data_space,
                                                   std::uint32_t buffers, DataflowStateMode state_mode) {
    const std::size_t n = std::size_t{1} << log_n;
    const std::size_t states = state_mode == DataflowStateMode::PingPong ? 2 : 1;
    constexpr std::size_t roles = (StageSpace + RoleStages - 1) / RoleStages;
    const std::size_t channels = static_cast<std::size_t>(roles - 1) * buffers *
                                 (std::size_t{1} << StageSpace) * DataTime;
    const std::size_t flags = static_cast<std::size_t>(roles - 1) * buffers * RoleStages * sizeof(int);
    (void)data_space;
    return (states * n + channels) * sizeof(Word) + flags;
}

template <typename Word, std::uint32_t FlowTileLogN, std::uint32_t TargetCtasPerSm,
          typename Operator>
__global__ __launch_bounds__(256, TargetCtasPerSm)
void hybrid_resident_radix4_kernel(const Word* input, Word* output, std::uint64_t transforms,
                                   HybridDataflowRuntime runtime, Operator op) {
    static_assert(TargetCtasPerSm >= 1 && TargetCtasPerSm <= 3);
    extern __shared__ __align__(16) unsigned char storage[];
    Word* state = reinterpret_cast<Word*>(storage);
    const std::uint32_t n = 1U << runtime.log_n;
    const std::uint64_t transform = blockIdx.x;
    if (transform >= transforms) {
        return;
    }

    const std::uint64_t base = transform * n;
    for (std::uint32_t logical = threadIdx.x; logical < n; logical += blockDim.x) {
        const std::uint32_t reversed = __brev(logical) >> (32 - runtime.log_n);
        const std::uint32_t physical = hybrid_state_index<Word>(
            logical, FlowTileLogN, runtime.data_space, runtime.layout);
        state[physical] = input[base + reversed];
    }
    __syncthreads();

    std::uint32_t stage = 0;
    for (; stage + 1 < runtime.log_n; stage += 2) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t units = n >> 2;
        for (std::uint32_t unit = threadIdx.x; unit < units; unit += blockDim.x) {
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            const std::uint32_t p0 = hybrid_state_index<Word>(i0, FlowTileLogN, runtime.data_space, runtime.layout);
            const std::uint32_t p1 = hybrid_state_index<Word>(i1, FlowTileLogN, runtime.data_space, runtime.layout);
            const std::uint32_t p2 = hybrid_state_index<Word>(i2, FlowTileLogN, runtime.data_space, runtime.layout);
            const std::uint32_t p3 = hybrid_state_index<Word>(i3, FlowTileLogN, runtime.data_space, runtime.layout);
            Word a = state[p0];
            Word b = state[p1];
            Word c = state[p2];
            Word d = state[p3];
            op.apply(stage, offset, a, b);
            op.apply(stage, offset, c, d);
            op.apply(stage + 1, offset, a, c);
            op.apply(stage + 1, half + offset, b, d);
            state[p0] = a;
            state[p1] = b;
            state[p2] = c;
            state[p3] = d;
        }
        __syncthreads();
    }
    if (stage < runtime.log_n) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t offset = threadIdx.x; offset < half; offset += blockDim.x) {
            const std::uint32_t left_physical = hybrid_state_index<Word>(
                offset, FlowTileLogN, runtime.data_space, runtime.layout);
            const std::uint32_t right_physical = hybrid_state_index<Word>(
                offset + half, FlowTileLogN, runtime.data_space, runtime.layout);
            Word left = state[left_physical];
            Word right = state[right_physical];
            op.apply(stage, offset, left, right);
            state[left_physical] = left;
            state[right_physical] = right;
        }
        __syncthreads();
    }

    for (std::uint32_t logical = threadIdx.x; logical < n; logical += blockDim.x) {
        const std::uint32_t physical = hybrid_state_index<Word>(
            logical, FlowTileLogN, runtime.data_space, runtime.layout);
        output[base + logical] = op.finalize(state[physical], n);
    }
}

template <bool FusedRows, typename Word, typename Operator>
__device__ __forceinline__ void hierarchical_apply(
    std::uint32_t stage, std::uint32_t offset, std::uint32_t row,
    std::uint32_t row_base, std::uint32_t local_n, const Word* fused_twiddles,
    const Word* fused_twiddles_shoup, Operator op, Word& left, Word& right) {
    if constexpr (FusedRows) {
        const std::uint64_t index = static_cast<std::uint64_t>(row_base + row) * (local_n - 1) +
                                    (1U << stage) - 1 + offset;
        op.apply_coefficient(fused_twiddles[index], fused_twiddles_shoup[index], left, right);
    } else {
        op.apply(stage, offset, left, right);
    }
}

// Executes a local radix-4 transform down the columns of a row-major resident
// tile. The APPT fused tail leaves [a][d] in place after its 7-stage row
// transform, then performs the final 6 stages over a without state2.
template <typename Word, std::uint32_t LocalLogN,
          std::uint32_t Columns, std::uint32_t RowStride, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix4_columns_offset(
    Word* state, Operator op, std::uint32_t logical_stage_base,
    std::uint32_t subgraph_base, std::uint32_t subgraph_stride) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kUnitsPerColumn = kLocalN >> 2;

    std::uint32_t stage = 0;
    for (; stage + 1 < LocalLogN; stage += 2) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < Columns * kUnitsPerColumn; linear += blockDim.x) {
            const std::uint32_t column = linear / kUnitsPerColumn;
            const std::uint32_t unit = linear - column * kUnitsPerColumn;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            const std::uint32_t subgraph =
                subgraph_base + column * subgraph_stride;
            Word a = state[i0 * RowStride + column];
            Word b = state[i1 * RowStride + column];
            Word c = state[i2 * RowStride + column];
            Word d = state[i3 * RowStride + column];
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), a, b);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), c, d);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + (offset << logical_stage_base), a, c);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + ((half + offset) << logical_stage_base), b, d);
            state[i0 * RowStride + column] = a;
            state[i1 * RowStride + column] = b;
            state[i2 * RowStride + column] = c;
            state[i3 * RowStride + column] = d;
        }
        __syncthreads();
    }
    if (stage < LocalLogN) {
        constexpr std::uint32_t kButterfliesPerColumn = kLocalN >> 1;
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < Columns * kButterfliesPerColumn; linear += blockDim.x) {
            const std::uint32_t column = linear / kButterfliesPerColumn;
            const std::uint32_t local = linear - column * kButterfliesPerColumn;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            const std::uint32_t right_index = left_index + half;
            Word left = state[left_index * RowStride + column];
            Word right = state[right_index * RowStride + column];
            op.apply(logical_stage_base + stage,
                     subgraph_base + column * subgraph_stride +
                         (offset << logical_stage_base),
                     left, right);
            state[left_index * RowStride + column] = left;
            state[right_index * RowStride + column] = right;
        }
        __syncthreads();
    }
}

template <bool FusedRows, typename Word, std::uint32_t LocalLogN,
          std::uint32_t RowsPerBlock, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix4(
    Word* state, Operator op, const Word* fused_twiddles = nullptr,
    const Word* fused_twiddles_shoup = nullptr, std::uint32_t row_base = 0) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    constexpr std::uint32_t kUnitsPerRow = kLocalN >> 2;

    std::uint32_t stage = 0;
    for (; stage + 1 < LocalLogN; stage += 2) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kUnitsPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kUnitsPerRow;
            const std::uint32_t unit = linear - row * kUnitsPerRow;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            Word a = state[row * kStride + i0];
            Word b = state[row * kStride + i1];
            Word c = state[row * kStride + i2];
            Word d = state[row * kStride + i3];
            hierarchical_apply<FusedRows>(stage, offset, row, row_base, kLocalN,
                                          fused_twiddles, fused_twiddles_shoup, op, a, b);
            hierarchical_apply<FusedRows>(stage, offset, row, row_base, kLocalN,
                                          fused_twiddles, fused_twiddles_shoup, op, c, d);
            hierarchical_apply<FusedRows>(stage + 1, offset, row, row_base, kLocalN,
                                          fused_twiddles, fused_twiddles_shoup, op, a, c);
            hierarchical_apply<FusedRows>(stage + 1, half + offset, row, row_base,
                                          kLocalN, fused_twiddles,
                                          fused_twiddles_shoup, op, b, d);
            state[row * kStride + i0] = a;
            state[row * kStride + i1] = b;
            state[row * kStride + i2] = c;
            state[row * kStride + i3] = d;
        }
        __syncthreads();
    }
    if (stage < LocalLogN) {
        constexpr std::uint32_t kButterfliesPerRow = kLocalN >> 1;
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kButterfliesPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kButterfliesPerRow;
            const std::uint32_t local = linear - row * kButterfliesPerRow;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            const std::uint32_t right_index = left_index + half;
            Word left = state[row * kStride + left_index];
            Word right = state[row * kStride + right_index];
            hierarchical_apply<FusedRows>(stage, offset, row, row_base, kLocalN,
                                          fused_twiddles, fused_twiddles_shoup,
                                          op, left, right);
            state[row * kStride + left_index] = left;
            state[row * kStride + right_index] = right;
        }
        __syncthreads();
    }
}

// CTA-resident radix-4 core for an APPT fold. Unlike the historical fused-row
// core, this computes each row's global-stage twiddle address directly and
// therefore does not stream an N-sized expanded coefficient table.
template <typename Word, std::uint32_t LocalLogN,
          std::uint32_t RowsPerBlock, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix4_offset(
    Word* state, Operator op, std::uint32_t logical_stage_base,
    std::uint32_t subgraph_base, std::uint32_t subgraph_stride) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    constexpr std::uint32_t kUnitsPerRow = kLocalN >> 2;

    std::uint32_t stage = 0;
    for (; stage + 1 < LocalLogN; stage += 2) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kUnitsPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kUnitsPerRow;
            const std::uint32_t unit = linear - row * kUnitsPerRow;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            const std::uint32_t subgraph = subgraph_base + row * subgraph_stride;
            Word a = state[row * kStride + i0];
            Word b = state[row * kStride + i1];
            Word c = state[row * kStride + i2];
            Word d = state[row * kStride + i3];
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), a, b);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), c, d);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + (offset << logical_stage_base), a, c);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + ((half + offset) << logical_stage_base), b, d);
            state[row * kStride + i0] = a;
            state[row * kStride + i1] = b;
            state[row * kStride + i2] = c;
            state[row * kStride + i3] = d;
        }
        __syncthreads();
    }
    if (stage < LocalLogN) {
        constexpr std::uint32_t kButterfliesPerRow = kLocalN >> 1;
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kButterfliesPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kButterfliesPerRow;
            const std::uint32_t local = linear - row * kButterfliesPerRow;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            const std::uint32_t right_index = left_index + half;
            Word left = state[row * kStride + left_index];
            Word right = state[row * kStride + right_index];
            op.apply(logical_stage_base + stage,
                     subgraph_base + row * subgraph_stride +
                         (offset << logical_stage_base),
                     left, right);
            state[row * kStride + left_index] = left;
            state[row * kStride + right_index] = right;
        }
        __syncthreads();
    }
}

// Cache the coefficient tree once for a row family. APPT row tiles execute
// the same local transform for every row, so loading coefficients per row
// creates request traffic without exposing additional parallelism.
template <typename Word, std::uint32_t LocalLogN, typename Operator>
__device__ __forceinline__ void hierarchical_load_coefficient_tree(
    Word* coefficients, Word* coefficients_shoup, Operator op,
    std::uint32_t logical_stage_base, std::uint32_t subgraph_base) {
    for (std::uint32_t local_stage = 0; local_stage < LocalLogN;
         ++local_stage) {
        const std::uint32_t count = 1U << local_stage;
        const std::uint32_t local_base = count - 1;
        const std::uint32_t global_base =
            (1U << (logical_stage_base + local_stage)) - 1;
        for (std::uint32_t offset = threadIdx.x; offset < count;
             offset += blockDim.x) {
            const std::uint32_t global_index =
                global_base + subgraph_base +
                (offset << logical_stage_base);
            coefficients[local_base + offset] = op.twiddles[global_index];
            coefficients_shoup[local_base + offset] =
                op.twiddles_shoup[global_index];
        }
    }
    __syncthreads();
}

template <typename Word, std::uint32_t LocalLogN,
          std::uint32_t RowsPerBlock, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix4_cached(
    Word* state, const Word* coefficients, const Word* coefficients_shoup,
    Operator op) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    constexpr std::uint32_t kUnitsPerRow = kLocalN >> 2;

    std::uint32_t stage = 0;
    for (; stage + 1 < LocalLogN; stage += 2) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t stage0_base = (1U << stage) - 1;
        const std::uint32_t stage1_base = (1U << (stage + 1)) - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kUnitsPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kUnitsPerRow;
            const std::uint32_t unit = linear - row * kUnitsPerRow;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            Word a = state[row * kStride + i0];
            Word b = state[row * kStride + i1];
            Word c = state[row * kStride + i2];
            Word d = state[row * kStride + i3];
            op.apply_coefficient(coefficients[stage0_base + offset],
                                 coefficients_shoup[stage0_base + offset],
                                 a, b);
            op.apply_coefficient(coefficients[stage0_base + offset],
                                 coefficients_shoup[stage0_base + offset],
                                 c, d);
            op.apply_coefficient(coefficients[stage1_base + offset],
                                 coefficients_shoup[stage1_base + offset],
                                 a, c);
            op.apply_coefficient(coefficients[stage1_base + half + offset],
                                 coefficients_shoup[stage1_base + half + offset],
                                 b, d);
            state[row * kStride + i0] = a;
            state[row * kStride + i1] = b;
            state[row * kStride + i2] = c;
            state[row * kStride + i3] = d;
        }
        __syncthreads();
    }
    if (stage < LocalLogN) {
        constexpr std::uint32_t kButterfliesPerRow = kLocalN >> 1;
        const std::uint32_t half = 1U << stage;
        const std::uint32_t coefficient_base = (1U << stage) - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kButterfliesPerRow;
             linear += blockDim.x) {
            const std::uint32_t row = linear / kButterfliesPerRow;
            const std::uint32_t local = linear - row * kButterfliesPerRow;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            const std::uint32_t right_index = left_index + half;
            Word left = state[row * kStride + left_index];
            Word right = state[row * kStride + right_index];
            op.apply_coefficient(coefficients[coefficient_base + offset],
                                 coefficients_shoup[coefficient_base + offset],
                                 left, right);
            state[row * kStride + left_index] = left;
            state[row * kStride + right_index] = right;
        }
        __syncthreads();
    }
}

// Three stages are fused per shared-memory round. This keeps independent
// modular multiplies visible to the scheduler while reducing CTA barriers
// compared with the radix-4 resident core.
template <typename Word, std::uint32_t LocalLogN,
          std::uint32_t RowsPerBlock, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix8_cached(
    Word* state, const Word* coefficients, const Word* coefficients_shoup,
    Operator op) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    std::uint32_t stage = 0;
    for (; stage + 2 < LocalLogN; stage += 3) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t units_per_row = kLocalN >> 3;
        const std::uint32_t base0 = half - 1;
        const std::uint32_t base1 = (half << 1) - 1;
        const std::uint32_t base2 = (half << 2) - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * units_per_row;
             linear += blockDim.x) {
            const std::uint32_t row = linear / units_per_row;
            const std::uint32_t unit = linear - row * units_per_row;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 3) + offset;
            Word v0 = state[row * kStride + i0];
            Word v1 = state[row * kStride + i0 + half];
            Word v2 = state[row * kStride + i0 + 2 * half];
            Word v3 = state[row * kStride + i0 + 3 * half];
            Word v4 = state[row * kStride + i0 + 4 * half];
            Word v5 = state[row * kStride + i0 + 5 * half];
            Word v6 = state[row * kStride + i0 + 6 * half];
            Word v7 = state[row * kStride + i0 + 7 * half];
            const Word c0 = coefficients[base0 + offset];
            const Word s0 = coefficients_shoup[base0 + offset];
            op.apply_coefficient(c0, s0, v0, v1);
            op.apply_coefficient(c0, s0, v2, v3);
            op.apply_coefficient(c0, s0, v4, v5);
            op.apply_coefficient(c0, s0, v6, v7);
            op.apply_coefficient(coefficients[base1 + offset],
                                 coefficients_shoup[base1 + offset], v0, v2);
            op.apply_coefficient(coefficients[base1 + half + offset],
                                 coefficients_shoup[base1 + half + offset], v1, v3);
            op.apply_coefficient(coefficients[base1 + offset],
                                 coefficients_shoup[base1 + offset], v4, v6);
            op.apply_coefficient(coefficients[base1 + half + offset],
                                 coefficients_shoup[base1 + half + offset], v5, v7);
            op.apply_coefficient(coefficients[base2 + offset],
                                 coefficients_shoup[base2 + offset], v0, v4);
            op.apply_coefficient(coefficients[base2 + half + offset],
                                 coefficients_shoup[base2 + half + offset], v1, v5);
            op.apply_coefficient(coefficients[base2 + 2 * half + offset],
                                 coefficients_shoup[base2 + 2 * half + offset], v2, v6);
            op.apply_coefficient(coefficients[base2 + 3 * half + offset],
                                 coefficients_shoup[base2 + 3 * half + offset], v3, v7);
            state[row * kStride + i0] = v0;
            state[row * kStride + i0 + half] = v1;
            state[row * kStride + i0 + 2 * half] = v2;
            state[row * kStride + i0 + 3 * half] = v3;
            state[row * kStride + i0 + 4 * half] = v4;
            state[row * kStride + i0 + 5 * half] = v5;
            state[row * kStride + i0 + 6 * half] = v6;
            state[row * kStride + i0 + 7 * half] = v7;
        }
        __syncthreads();
    }
    if (stage + 1 < LocalLogN) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t units_per_row = kLocalN >> 2;
        const std::uint32_t base0 = half - 1;
        const std::uint32_t base1 = (half << 1) - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * units_per_row;
             linear += blockDim.x) {
            const std::uint32_t row = linear / units_per_row;
            const std::uint32_t unit = linear - row * units_per_row;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            Word v0 = state[row * kStride + i0];
            Word v1 = state[row * kStride + i0 + half];
            Word v2 = state[row * kStride + i0 + 2 * half];
            Word v3 = state[row * kStride + i0 + 3 * half];
            op.apply_coefficient(coefficients[base0 + offset],
                                 coefficients_shoup[base0 + offset], v0, v1);
            op.apply_coefficient(coefficients[base0 + offset],
                                 coefficients_shoup[base0 + offset], v2, v3);
            op.apply_coefficient(coefficients[base1 + offset],
                                 coefficients_shoup[base1 + offset], v0, v2);
            op.apply_coefficient(coefficients[base1 + half + offset],
                                 coefficients_shoup[base1 + half + offset], v1, v3);
            state[row * kStride + i0] = v0;
            state[row * kStride + i0 + half] = v1;
            state[row * kStride + i0 + 2 * half] = v2;
            state[row * kStride + i0 + 3 * half] = v3;
        }
        __syncthreads();
        stage += 2;
    }
    if (stage < LocalLogN) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t butterflies_per_row = kLocalN >> 1;
        const std::uint32_t base = half - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * butterflies_per_row;
             linear += blockDim.x) {
            const std::uint32_t row = linear / butterflies_per_row;
            const std::uint32_t local = linear - row * butterflies_per_row;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            Word left = state[row * kStride + left_index];
            Word right = state[row * kStride + left_index + half];
            op.apply_coefficient(coefficients[base + offset],
                                 coefficients_shoup[base + offset], left, right);
            state[row * kStride + left_index] = left;
            state[row * kStride + left_index + half] = right;
        }
        __syncthreads();
    }
}

template <typename Word, std::uint32_t LocalLogN,
          std::uint32_t Columns, std::uint32_t RowStride, typename Operator>
__device__ __forceinline__ void hierarchical_resident_radix8_columns_offset(
    Word* state, Operator op, std::uint32_t logical_stage_base,
    std::uint32_t subgraph_base, std::uint32_t subgraph_stride) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    std::uint32_t stage = 0;
    for (; stage + 2 < LocalLogN; stage += 3) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t units_per_column = kLocalN >> 3;
        for (std::uint32_t linear = threadIdx.x;
             linear < Columns * units_per_column; linear += blockDim.x) {
            const std::uint32_t column = linear / units_per_column;
            const std::uint32_t unit = linear - column * units_per_column;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 3) + offset;
            const std::uint32_t subgraph =
                subgraph_base + column * subgraph_stride;
            Word v0 = state[i0 * RowStride + column];
            Word v1 = state[(i0 + half) * RowStride + column];
            Word v2 = state[(i0 + 2 * half) * RowStride + column];
            Word v3 = state[(i0 + 3 * half) * RowStride + column];
            Word v4 = state[(i0 + 4 * half) * RowStride + column];
            Word v5 = state[(i0 + 5 * half) * RowStride + column];
            Word v6 = state[(i0 + 6 * half) * RowStride + column];
            Word v7 = state[(i0 + 7 * half) * RowStride + column];
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v0, v1);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v2, v3);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v4, v5);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v6, v7);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + (offset << logical_stage_base), v0, v2);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + ((half + offset) << logical_stage_base), v1, v3);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + (offset << logical_stage_base), v4, v6);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + ((half + offset) << logical_stage_base), v5, v7);
            op.apply(logical_stage_base + stage + 2,
                     subgraph + (offset << logical_stage_base), v0, v4);
            op.apply(logical_stage_base + stage + 2,
                     subgraph + ((half + offset) << logical_stage_base), v1, v5);
            op.apply(logical_stage_base + stage + 2,
                     subgraph + ((2 * half + offset) << logical_stage_base), v2, v6);
            op.apply(logical_stage_base + stage + 2,
                     subgraph + ((3 * half + offset) << logical_stage_base), v3, v7);
            state[i0 * RowStride + column] = v0;
            state[(i0 + half) * RowStride + column] = v1;
            state[(i0 + 2 * half) * RowStride + column] = v2;
            state[(i0 + 3 * half) * RowStride + column] = v3;
            state[(i0 + 4 * half) * RowStride + column] = v4;
            state[(i0 + 5 * half) * RowStride + column] = v5;
            state[(i0 + 6 * half) * RowStride + column] = v6;
            state[(i0 + 7 * half) * RowStride + column] = v7;
        }
        __syncthreads();
    }
    if (stage + 1 < LocalLogN) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t units_per_column = kLocalN >> 2;
        for (std::uint32_t linear = threadIdx.x;
             linear < Columns * units_per_column; linear += blockDim.x) {
            const std::uint32_t column = linear / units_per_column;
            const std::uint32_t unit = linear - column * units_per_column;
            const std::uint32_t group = unit / half;
            const std::uint32_t offset = unit - group * half;
            const std::uint32_t i0 = group * (half << 2) + offset;
            const std::uint32_t subgraph =
                subgraph_base + column * subgraph_stride;
            Word v0 = state[i0 * RowStride + column];
            Word v1 = state[(i0 + half) * RowStride + column];
            Word v2 = state[(i0 + 2 * half) * RowStride + column];
            Word v3 = state[(i0 + 3 * half) * RowStride + column];
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v0, v1);
            op.apply(logical_stage_base + stage,
                     subgraph + (offset << logical_stage_base), v2, v3);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + (offset << logical_stage_base), v0, v2);
            op.apply(logical_stage_base + stage + 1,
                     subgraph + ((half + offset) << logical_stage_base), v1, v3);
            state[i0 * RowStride + column] = v0;
            state[(i0 + half) * RowStride + column] = v1;
            state[(i0 + 2 * half) * RowStride + column] = v2;
            state[(i0 + 3 * half) * RowStride + column] = v3;
        }
        __syncthreads();
        stage += 2;
    }
    if (stage < LocalLogN) {
        const std::uint32_t half = 1U << stage;
        const std::uint32_t butterflies_per_column = kLocalN >> 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < Columns * butterflies_per_column;
             linear += blockDim.x) {
            const std::uint32_t column = linear / butterflies_per_column;
            const std::uint32_t local = linear - column * butterflies_per_column;
            const std::uint32_t group = local / half;
            const std::uint32_t offset = local - group * half;
            const std::uint32_t left_index = group * (half << 1) + offset;
            Word left = state[left_index * RowStride + column];
            Word right = state[(left_index + half) * RowStride + column];
            op.apply(logical_stage_base + stage,
                     subgraph_base + column * subgraph_stride +
                         (offset << logical_stage_base),
                     left, right);
            state[left_index * RowStride + column] = left;
            state[(left_index + half) * RowStride + column] = right;
        }
        __syncthreads();
    }
}

template <bool FusedRows, typename Word, std::uint32_t LocalLogN,
          std::uint32_t RowsPerBlock, typename Operator>
__device__ __forceinline__ void hierarchical_hybrid2d_radix4(
    Word* state, Operator op, const Word* fused_twiddles = nullptr,
    const Word* fused_twiddles_shoup = nullptr, std::uint32_t row_base = 0) {
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    constexpr std::uint32_t kUnitsPerRow = kLocalN >> 2;
    std::uint32_t stage = 0;
    for (; stage + 1 < LocalLogN; stage += 2) {
        const std::uint32_t half0 = 1U << stage;
        const std::uint32_t twiddle_offset0 = half0 - 1;
        const std::uint32_t twiddle_offset1 = (half0 << 1) - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kUnitsPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kUnitsPerRow;
            const std::uint32_t local = linear - row * kUnitsPerRow;
            const std::uint32_t group = local / half0;
            const std::uint32_t offset = local - group * half0;
            const std::uint32_t i0 = row * kStride + group * (half0 << 2) + offset;
            const std::uint32_t i1 = i0 + half0;
            const std::uint32_t i2 = i1 + half0;
            const std::uint32_t i3 = i2 + half0;
            Word a = state[i0];
            Word b = state[i1];
            Word c = state[i2];
            Word d = state[i3];
            const std::uint32_t row_offset = FusedRows ? (row_base + row) * (kLocalN - 1) : 0;
            const Word* coefficients = FusedRows ? fused_twiddles : op.twiddles;
            const Word* coefficients_shoup = FusedRows ? fused_twiddles_shoup : op.twiddles_shoup;
            const Word coefficient0 = coefficients[row_offset + twiddle_offset0 + offset];
            const Word coefficient0_shoup = coefficients_shoup[row_offset + twiddle_offset0 + offset];
            op.apply_coefficient(coefficient0, coefficient0_shoup, a, b);
            op.apply_coefficient(coefficient0, coefficient0_shoup, c, d);
            std::uint32_t index = row_offset + twiddle_offset1 + offset;
            op.apply_coefficient(coefficients[index], coefficients_shoup[index], a, c);
            index += half0;
            op.apply_coefficient(coefficients[index], coefficients_shoup[index], b, d);
            state[i0] = a;
            state[i1] = b;
            state[i2] = c;
            state[i3] = d;
        }
        __syncthreads();
    }
    if (stage < LocalLogN) {
        constexpr std::uint32_t kButterfliesPerRow = kLocalN >> 1;
        constexpr std::uint32_t half = 1U << (LocalLogN - 1);
        constexpr std::uint32_t twiddle_offset = half - 1;
        for (std::uint32_t linear = threadIdx.x;
             linear < RowsPerBlock * kButterfliesPerRow; linear += blockDim.x) {
            const std::uint32_t row = linear / kButterfliesPerRow;
            const std::uint32_t offset = linear - row * kButterfliesPerRow;
            const std::uint32_t left_index = row * kStride + offset;
            const std::uint32_t right_index = left_index + half;
            const std::uint32_t row_offset = FusedRows ? (row_base + row) * (kLocalN - 1) : 0;
            const Word* coefficients = FusedRows ? fused_twiddles : op.twiddles;
            const Word* coefficients_shoup = FusedRows ? fused_twiddles_shoup : op.twiddles_shoup;
            Word left = state[left_index];
            Word right = state[right_index];
            const std::uint32_t index = row_offset + twiddle_offset + offset;
            op.apply_coefficient(coefficients[index], coefficients_shoup[index], left, right);
            state[left_index] = left;
            state[right_index] = right;
        }
        __syncthreads();
    }
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t RowsPerBlock, typename Operator, bool Hybrid2DCore = false>
__global__ void hierarchical_dataflow_kernel(
    const Word* input, Word* output, Word* boundary, std::uint64_t transforms,
    std::uint32_t data_time, const Word* root_powers,
    const Word* root_powers_shoup, Operator op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kFirstN = 1U << FirstLogN;
    constexpr std::uint32_t kSecondN = 1U << SecondLogN;
    constexpr std::uint32_t kN = kFirstN * kSecondN;
    constexpr std::uint32_t kFirstStride = kFirstN + 1;
    constexpr std::uint32_t kSecondStride = kSecondN + 1;
    extern __shared__ __align__(16) unsigned char storage[];
    Word* state = reinterpret_cast<Word*>(storage);

    const std::uint64_t first_groups = transforms * kSecondN / RowsPerBlock;
    const std::uint64_t first_packet = static_cast<std::uint64_t>(blockIdx.x) * data_time;
    const std::uint64_t packet_stride = static_cast<std::uint64_t>(gridDim.x) * data_time;
    for (std::uint64_t packet = first_packet; packet < first_groups; packet += packet_stride) {
        for (std::uint32_t token = 0; token < data_time; ++token) {
            const std::uint64_t group_index = packet + token;
            if (group_index >= first_groups) {
                break;
            }
            const std::uint64_t first_row = group_index * RowsPerBlock;
            const std::uint64_t transform = first_row / kSecondN;
            const std::uint32_t row_base = static_cast<std::uint32_t>(first_row - transform * kSecondN);
            const std::uint64_t transform_base = transform * kN;
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                const std::uint32_t first_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - first_index * RowsPerBlock;
                const std::uint32_t reversed = __brev(first_index) >> (32 - FirstLogN);
                state[row * kFirstStride + reversed] =
                    input[transform_base + static_cast<std::uint64_t>(first_index) * kSecondN + row_base + row];
            }
            __syncthreads();
            if constexpr (Hybrid2DCore) {
                hierarchical_hybrid2d_radix4<false, Word, FirstLogN, RowsPerBlock>(state, op);
            } else {
                hierarchical_resident_radix4<false, Word, FirstLogN, RowsPerBlock>(state, op);
            }
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                const std::uint32_t first_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - first_index * RowsPerBlock;
                const std::uint32_t second_index = row_base + row;
                boundary[transform_base + static_cast<std::uint64_t>(first_index) * kSecondN + second_index] =
                    state[row * kFirstStride + first_index];
            }
            __syncthreads();
        }
    }

    cg::this_grid().sync();

    const std::uint64_t second_groups = transforms * kFirstN / RowsPerBlock;
    const std::uint64_t second_packet = static_cast<std::uint64_t>(blockIdx.x) * data_time;
    for (std::uint64_t packet = second_packet; packet < second_groups; packet += packet_stride) {
        for (std::uint32_t token = 0; token < data_time; ++token) {
            const std::uint64_t group_index = packet + token;
            if (group_index >= second_groups) {
                break;
            }
            const std::uint64_t first_row = group_index * RowsPerBlock;
            const std::uint64_t transform = first_row / kFirstN;
            const std::uint32_t row_base = static_cast<std::uint32_t>(first_row - transform * kFirstN);
            const std::uint64_t transform_base = transform * kN;
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                const std::uint32_t row = linear / kSecondN;
                const std::uint32_t second_index = linear - row * kSecondN;
                const std::uint32_t first_index = row_base + row;
                const std::uint32_t reversed = __brev(second_index) >> (32 - SecondLogN);
                state[row * kSecondStride + reversed] =
                    boundary[transform_base + static_cast<std::uint64_t>(first_index) * kSecondN + second_index];
            }
            __syncthreads();
            if constexpr (Hybrid2DCore) {
                hierarchical_hybrid2d_radix4<true, Word, SecondLogN, RowsPerBlock>(
                    state, op, root_powers, root_powers_shoup, row_base);
            } else {
                hierarchical_resident_radix4<true, Word, SecondLogN, RowsPerBlock>(
                    state, op, root_powers, root_powers_shoup, row_base);
            }
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                const std::uint32_t second_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - second_index * RowsPerBlock;
                const std::uint32_t first_index = row_base + row;
                output[transform_base + static_cast<std::uint64_t>(second_index) * kFirstN + first_index] =
                    op.finalize(state[row * kSecondStride + second_index], kN);
            }
            __syncthreads();
        }
    }
}

}  // namespace cuntt::detail
