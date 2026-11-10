#pragma once

#include <cstddef>
#include <cstdint>

#include "cuntt/iterative_layout.hpp"

namespace cuntt::detail {

namespace shared_resident_detail {

template <unsigned... Parts>
struct partition_traits {
    static constexpr unsigned count = sizeof...(Parts);
    static constexpr unsigned stages = (0U + ... + Parts);
    // A leaf with R stages keeps 2^R values in registers.  Keep the
    // specialization bounded at 32 values per thread for this first kernel.
    static constexpr bool leaves_valid = ((Parts >= 1U && Parts <= 5U) && ...);
};

template <unsigned StageOffset, unsigned LeafStages, typename Operator>
__device__ __forceinline__ void run_leaf(
    typename Operator::Value*& current, typename Operator::Value*& next,
    std::uint32_t first_stage, std::uint32_t stages,
    bool writer_aligned, std::uint32_t tile, Operator op) {
    using Value = typename Operator::Value;
    constexpr unsigned value_count = 1U << LeafStages;

    Value values[value_count];
    const unsigned groups = (1U << stages) >> LeafStages;
    const unsigned low_mask = (1U << StageOffset) - 1U;

    for (unsigned group = threadIdx.x; group < groups; group += blockDim.x) {
        // A leaf owns all indices that differ only in the R stage bits being
        // fused.  Preserve lower bits and shift the remaining group bits over
        // that interval, matching the radix-2 dependency graph.
        const unsigned low = group & low_mask;
        const unsigned high = group >> StageOffset;
        const unsigned base = (high << (StageOffset + LeafStages)) | low;

#pragma unroll
        for (unsigned lane = 0; lane < value_count; ++lane) {
            const unsigned logical = base | (lane << StageOffset);
            const unsigned source = writer_aligned
                                        ? iterative_layout_index(logical, stages, StageOffset)
                                        : logical;
            values[lane] = current[source];
        }

#pragma unroll
        for (unsigned local_stage = 0; local_stage < LeafStages; ++local_stage) {
            const unsigned half = 1U << local_stage;
            const unsigned global_stage = first_stage + StageOffset + local_stage;
#pragma unroll
            for (unsigned pair = 0; pair < value_count / 2U; ++pair) {
                const unsigned left = ((pair >> local_stage) << (local_stage + 1U)) |
                                      (pair & (half - 1U));
                const unsigned right = left + half;
                const unsigned global_left = iterative_global_index(
                    tile, base | (left << StageOffset), first_stage, stages);
                op.apply(global_stage,
                         global_left & ((1U << global_stage) - 1U),
                         values[left], values[right]);
            }
        }

        const unsigned consumer_stage = StageOffset + LeafStages < stages
                                            ? StageOffset + LeafStages
                                            : stages - 1U;
#pragma unroll
        for (unsigned lane = 0; lane < value_count; ++lane) {
            const unsigned logical = base | (lane << StageOffset);
            const unsigned destination = writer_aligned
                                             ? iterative_layout_index(logical, stages, consumer_stage)
                                             : logical;
            next[destination] = values[lane];
        }
    }

    // Every leaf is a dependency-closed register subgraph.  The next leaf
    // reads only after all threads have published its shared state.
    __syncthreads();
    auto* swap = current;
    current = next;
    next = swap;
}

template <unsigned StageOffset, typename Operator, unsigned Head, unsigned... Tail>
__device__ __forceinline__ void run_leaves(
    typename Operator::Value*& current, typename Operator::Value*& next,
    std::uint32_t first_stage, std::uint32_t stages,
    bool writer_aligned, std::uint32_t tile, Operator op) {
    run_leaf<StageOffset, Head>(current, next, first_stage, stages,
                                writer_aligned, tile, op);
    if constexpr (sizeof...(Tail) != 0U) {
        run_leaves<StageOffset + Head, Operator, Tail...>(
            current, next, first_stage, stages, writer_aligned, tile, op);
    }
}

}  // namespace shared_resident_detail

// A shared-memory iterative tile whose radix-2 stages are grouped into
// dependency-closed register leaves.  The runtime argument list intentionally
// matches shared_iterative_kernel; compile-time Parts... describes only the
// resident subgraph inside this physical tile.
template <typename Operator, int StaticLogN = -1, int StaticFirst = -1,
          int StaticStages = -1, int StaticLayout = -1,
          int StaticOutputOrder = 0, unsigned... Parts>
__global__ void shared_resident_kernel(
    const typename Operator::Value* input, typename Operator::Value* output,
    std::uint32_t log_n, std::uint32_t first_stage, std::uint32_t stages,
    std::size_t batch_stride, std::size_t element_stride, bool writer_aligned,
    std::uint32_t batch_count, std::uint32_t batch_tile_count, Operator op,
    bool bit_reversed_output = false) {
    using Value = typename Operator::Value;
    using Partition = shared_resident_detail::partition_traits<Parts...>;

    static_assert(Partition::count != 0U,
                  "shared_resident_kernel requires at least one resident leaf");
    static_assert(Partition::leaves_valid,
                  "resident leaf stage counts must be in [1, 5]");
    static_assert(Partition::stages <= 30U,
                  "resident tile stage count must be at most 30");
    static_assert(StaticLogN < 0 || (StaticLogN >= 1 && StaticLogN <= 30),
                  "StaticLogN must be in [1, 30]");
    static_assert(StaticFirst < 0 || StaticFirst <= 30,
                  "StaticFirst is outside the supported logN range");
    static_assert(StaticStages < 0 || StaticStages > 0,
                  "StaticStages must be positive");
    static_assert(StaticStages < 0 || Partition::stages == static_cast<unsigned>(StaticStages),
                  "resident leaves must cover StaticStages exactly");
    static_assert(StaticLayout < 0 || StaticLayout == 0 || StaticLayout == 1,
                  "StaticLayout must be 0 or 1");
    static_assert(StaticOutputOrder < 0 || StaticOutputOrder == 0 || StaticOutputOrder == 1,
                  "StaticOutputOrder must be 0 or 1");
    static_assert(StaticFirst < 0 || StaticStages < 0 || StaticLogN < 0 ||
                      StaticFirst + StaticStages <= StaticLogN,
                  "resident stage interval exceeds StaticLogN");

    if constexpr (StaticLogN >= 0) log_n = StaticLogN;
    if constexpr (StaticFirst >= 0) first_stage = StaticFirst;
    if constexpr (StaticStages >= 0) stages = StaticStages;
    if constexpr (StaticLayout >= 0) writer_aligned = StaticLayout != 0;
    if constexpr (StaticOutputOrder >= 0) bit_reversed_output = StaticOutputOrder != 0;

    // Runtime dimensions are retained for compatibility with the generic
    // launcher. Invalid dimensions are rejected uniformly before any barrier.
    if (log_n == 0U || log_n > 30U || stages == 0U || stages > log_n ||
        first_stage > log_n - stages || stages != Partition::stages ||
        batch_tile_count == 0U)
        return;

    extern __shared__ __align__(16) unsigned char storage[];
    const unsigned n = 1U << log_n;
    const unsigned size = 1U << stages;
    const unsigned tiles_per_batch = n / size;
    const unsigned tile = blockIdx.x % tiles_per_batch;
    const unsigned batch_group = blockIdx.x / tiles_per_batch;
    const unsigned first_batch = batch_group * batch_tile_count;
    auto* current = reinterpret_cast<Value*>(storage);
    auto* next = current + size;

    for (std::uint32_t batch_item = 0; batch_item < batch_tile_count; ++batch_item) {
        const std::uint32_t batch_index = first_batch + batch_item;
        if (batch_index >= batch_count) break;
        const std::size_t batch_base = static_cast<std::size_t>(batch_index) * batch_stride;

        for (std::uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
            auto global = iterative_global_index(tile, i, first_stage, stages);
            if constexpr (Operator::kBitReverseInput) {
                if (first_stage == 0U)
                    global = __brev(global) >> (32U - log_n);
            }
            const auto destination = writer_aligned
                                         ? iterative_layout_index(i, stages, 0U)
                                         : i;
            current[destination] = input[batch_base + static_cast<std::size_t>(global) * element_stride];
        }
        __syncthreads();

        shared_resident_detail::run_leaves<0U, Operator, Parts...>(
            current, next, first_stage, stages, writer_aligned, tile, op);

        for (std::uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
            const bool reverse = bit_reversed_output && first_stage + stages == log_n;
            const auto logical = reverse ? __brev(i) >> (32U - stages) : i;
            const auto source = writer_aligned
                                    ? iterative_layout_index(logical, stages, stages - 1U)
                                    : logical;
            Value value = current[source];
            if (first_stage + stages == log_n) value = op.finalize(value, n);
            auto global = iterative_global_index(tile, logical, first_stage, stages);
            if (reverse) global = __brev(global) >> (32U - log_n);
            output[batch_base + static_cast<std::size_t>(global) * element_stride] = value;
        }
        __syncthreads();
    }
}

}  // namespace cuntt::detail
