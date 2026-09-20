#pragma once
#include "cuntt/iterative_layout.hpp"
namespace cuntt::detail {
template <typename Operator, int StaticLogN=-1, int StaticFirst=-1, int StaticStages=-1, int StaticLayout=-1,
          int StaticOutputOrder=0>
__global__ void shared_iterative_kernel(
    const typename Operator::Value* input, typename Operator::Value* output,
    std::uint32_t log_n, std::uint32_t first_stage, std::uint32_t stages,
    std::size_t batch_stride, std::size_t element_stride, bool writer_aligned,
    std::uint32_t batch_count, std::uint32_t batch_tile_count, Operator op,
    bool bit_reversed_output=false) {
    using Value = typename Operator::Value;
    if constexpr (StaticLogN>=0) log_n=StaticLogN;
    if constexpr (StaticFirst>=0) first_stage=StaticFirst;
    if constexpr (StaticStages>=0) stages=StaticStages;
    if constexpr (StaticLayout>=0) writer_aligned=StaticLayout;
    if constexpr (StaticOutputOrder>=0) bit_reversed_output=StaticOutputOrder!=0;
    extern __shared__ __align__(16) unsigned char storage[];
    const auto n = 1U << log_n;
    const auto size = 1U << stages;
    const auto tiles_per_batch = n / size;
    const auto tile = blockIdx.x % tiles_per_batch;
    const auto batch_group = blockIdx.x / tiles_per_batch;
    const auto first_batch = batch_group * batch_tile_count;
    auto* current = reinterpret_cast<Value*>(storage);
    auto* next = current + size;
    for (std::uint32_t batch_item = 0; batch_item < batch_tile_count; ++batch_item) {
      const auto batch_index = first_batch + batch_item;
      if (batch_index >= batch_count) break;
      const std::size_t batch_base = static_cast<std::size_t>(batch_index) * batch_stride;
      for (std::uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        auto global = iterative_global_index(tile, i, first_stage, stages);
        if constexpr (Operator::kBitReverseInput)
            if (first_stage == 0) global = __brev(global) >> (32 - log_n);
        const auto destination = writer_aligned ? iterative_layout_index(i, stages, 0) : i;
        current[destination] = input[batch_base + global * element_stride];
      }
    __syncthreads();
      for (std::uint32_t stage = 0; stage < stages; ++stage) {
        for (std::uint32_t pair = threadIdx.x; pair < size / 2; pair += blockDim.x) {
            const auto half = 1U << stage;
            const auto left = ((pair >> stage) << (stage + 1)) | (pair & (half - 1));
            const auto right = left + half;
            const auto load_left = writer_aligned ? iterative_layout_index(left, stages, stage) : left;
            const auto load_right = writer_aligned ? iterative_layout_index(right, stages, stage) : right;
            Value a = current[load_left], b = current[load_right];
            const auto global_left = iterative_global_index(tile, left, first_stage, stages);
            op.apply(first_stage + stage, global_left & ((1U << (first_stage + stage)) - 1), a, b);
            const auto consumer = stage + 1 < stages ? stage + 1 : stage;
            const auto store_left = writer_aligned ? iterative_layout_index(left, stages, consumer) : left;
            const auto store_right = writer_aligned ? iterative_layout_index(right, stages, consumer) : right;
            next[store_left] = a;
            next[store_right] = b;
        }
        // Separate buffers prevent a writer overwriting another warp's unread
        // input, including later data-time iterations. Publish each full stage.
        __syncthreads();
        Value* swap = current; current = next; next = swap;
      }
      for (std::uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        // Only the terminal boundary changes order. Enumerate its logical tile
        // in reverse-bit order so adjacent writer lanes still issue contiguous
        // global stores: rev(tile | (rev(i) << first)) = rev(tile) << stages | i.
        const bool reverse = bit_reversed_output && first_stage + stages == log_n;
        const auto logical = reverse ? __brev(i) >> (32 - stages) : i;
        const auto source = writer_aligned ? iterative_layout_index(logical, stages, stages - 1) : logical;
        Value value = current[source];
        if (first_stage + stages == log_n) value = op.finalize(value, n);
        auto global = iterative_global_index(tile, logical, first_stage, stages);
        if (reverse) global = __brev(global) >> (32 - log_n);
        output[batch_base + global * element_stride] = value;
      }
      __syncthreads();
    }
}
} // namespace cuntt::detail
