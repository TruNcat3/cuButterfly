#pragma once
#include "cuntt/iterative_layout.hpp"
namespace cuntt::detail {

// Transpose one materialized radix boundary with a conventional shared-memory
// tile.  The stage kernel owns a producer tile, whereas the consumer wants the
// producer's low tile bits and local stage bits exchanged.  Keeping this as a
// separate node makes both the data movement and its cost visible to the
// graph/search layer.
template <typename Value>
__global__ void iterative_boundary_transpose_kernel(
    const Value* input, Value* output, std::uint32_t log_n,
    std::uint32_t first_stage, std::uint32_t stages,
    std::size_t batch_stride, std::size_t element_stride,
    std::uint32_t batch_count) {
    __shared__ Value tile[32][33];
    const auto rows = 1U << first_stage;
    const auto cols = 1U << stages;
    const auto outer_count = 1U << (log_n - first_stage - stages);
    const auto matrix = static_cast<std::uint32_t>(blockIdx.z);
    const auto batch_index = matrix / outer_count;
    const auto outer = matrix % outer_count;
    if (batch_index >= batch_count) return;
    const auto batch_base = static_cast<std::size_t>(batch_index) * batch_stride;
    const auto outer_base = outer << (first_stage + stages);
    const auto x0 = blockIdx.x * 32U + threadIdx.x;
    const auto y0 = blockIdx.y * 32U + threadIdx.y;
    for (std::uint32_t j = 0; j < 32; j += blockDim.y) {
        const auto x = x0;
        const auto y = y0 + j;
        if (x < rows && y < cols) {
            const auto global = outer_base | x | (y << first_stage);
            tile[threadIdx.y + j][threadIdx.x] =
                input[batch_base + static_cast<std::size_t>(global) * element_stride];
        }
    }
    __syncthreads();
    for (std::uint32_t j = 0; j < 32; j += blockDim.y) {
        const auto x = blockIdx.x * 32U + threadIdx.y + j;
        const auto y = blockIdx.y * 32U + threadIdx.x;
        if (x < rows && y < cols) {
            const auto physical = outer_base | (x << stages) | y;
            output[batch_base + static_cast<std::size_t>(physical) * element_stride] =
                tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <typename Operator, int StaticLogN=-1, int StaticFirst=-1, int StaticStages=-1, int StaticLayout=-1,
          int StaticOutputOrder=0, unsigned TilesPerCta=1>
__global__ void shared_iterative_kernel(
    const typename Operator::Value* input, typename Operator::Value* output,
    std::uint32_t log_n, std::uint32_t first_stage, std::uint32_t stages,
    std::size_t batch_stride, std::size_t element_stride, bool writer_aligned,
    std::uint32_t batch_count, std::uint32_t batch_tile_count, Operator op,
    bool bit_reversed_output=false, bool input_transposed=false,
    bool output_transposed=false, std::uint32_t boundary_first_stage=0,
    std::uint32_t boundary_stages=0) {
    using Value = typename Operator::Value;
    static_assert(TilesPerCta && (TilesPerCta & (TilesPerCta - 1)) == 0,
                  "shared tile group must be a power of two");
    if constexpr (StaticLogN>=0) log_n=StaticLogN;
    if constexpr (StaticFirst>=0) first_stage=StaticFirst;
    if constexpr (StaticStages>=0) stages=StaticStages;
    if constexpr (StaticLayout>=0) writer_aligned=StaticLayout;
    if constexpr (StaticOutputOrder>=0) bit_reversed_output=StaticOutputOrder!=0;
    extern __shared__ __align__(16) unsigned char storage[];
    const auto n = 1U << log_n;
    const auto size = 1U << stages;
    const auto tiles_per_batch = n / size;
    const auto tile_groups = (tiles_per_batch + TilesPerCta - 1) / TilesPerCta;
    const auto first_tile = (blockIdx.x % tile_groups) * TilesPerCta;
    const auto batch_group = blockIdx.x / tile_groups;
    const auto first_batch = batch_group * batch_tile_count;
    auto* current = reinterpret_cast<Value*>(storage);
    auto* next = current + size * TilesPerCta;
    for (std::uint32_t batch_item = 0; batch_item < batch_tile_count; ++batch_item) {
      const auto batch_index = first_batch + batch_item;
      if (batch_index >= batch_count) break;
      const std::size_t batch_base = static_cast<std::size_t>(batch_index) * batch_stride;
      // This axis groups independent graph tiles, not external batches.  The
      // original one-tile instantiation retains its indexing and footprint.
      for (std::uint32_t item = threadIdx.x; item < size * TilesPerCta; item += blockDim.x) {
        const auto column = item % TilesPerCta;
        const auto tile = first_tile + column;
        if (tile >= tiles_per_batch) continue;
        const auto i = item / TilesPerCta;
        auto global = iterative_global_index(tile, i, first_stage, stages);
        if constexpr (Operator::kBitReverseInput)
            if (first_stage == 0) global = __brev(global) >> (32 - log_n);
        if (input_transposed)
            global = iterative_boundary_index(global, first_stage, stages);
        const auto destination = writer_aligned ? iterative_layout_index(i, stages, 0) : i;
        current[destination * TilesPerCta + column] = input[batch_base + global * element_stride];
      }
    __syncthreads();
      for (std::uint32_t stage = 0; stage < stages; ++stage) {
        for (std::uint32_t item = threadIdx.x; item < (size / 2) * TilesPerCta; item += blockDim.x) {
            const auto column = item % TilesPerCta;
            const auto tile = first_tile + column;
            if (tile >= tiles_per_batch) continue;
            const auto pair = item / TilesPerCta;
            const auto half = 1U << stage;
            const auto left = ((pair >> stage) << (stage + 1)) | (pair & (half - 1));
            const auto right = left + half;
            const auto load_left = writer_aligned ? iterative_layout_index(left, stages, stage) : left;
            const auto load_right = writer_aligned ? iterative_layout_index(right, stages, stage) : right;
            Value a = current[load_left * TilesPerCta + column];
            Value b = current[load_right * TilesPerCta + column];
            const auto global_left = iterative_global_index(tile, left, first_stage, stages);
            op.apply(first_stage + stage, global_left & ((1U << (first_stage + stage)) - 1), a, b);
            const auto consumer = stage + 1 < stages ? stage + 1 : stage;
            const auto store_left = writer_aligned ? iterative_layout_index(left, stages, consumer) : left;
            const auto store_right = writer_aligned ? iterative_layout_index(right, stages, consumer) : right;
            next[store_left * TilesPerCta + column] = a;
            next[store_right * TilesPerCta + column] = b;
        }
        // Separate buffers prevent a writer overwriting another warp's unread
        // input, including later data-time iterations. Publish each full stage.
        __syncthreads();
        Value* swap = current; current = next; next = swap;
      }
      for (std::uint32_t item = threadIdx.x; item < size * TilesPerCta; item += blockDim.x) {
        const auto column = item % TilesPerCta;
        const auto tile = first_tile + column;
        if (tile >= tiles_per_batch) continue;
        const auto i = item / TilesPerCta;
        // Only the terminal boundary changes order. Enumerate its logical tile
        // in reverse-bit order so adjacent writer lanes still issue contiguous
        // global stores: rev(tile | (rev(i) << first)) = rev(tile) << stages | i.
        const bool reverse = bit_reversed_output && first_stage + stages == log_n;
        const auto logical = reverse ? __brev(i) >> (32 - stages) : i;
        const auto source = writer_aligned ? iterative_layout_index(logical, stages, stages - 1) : logical;
        Value value = current[source * TilesPerCta + column];
        if (first_stage + stages == log_n) value = op.finalize(value, n);
        auto global = iterative_global_index(tile, logical, first_stage, stages);
        if (reverse) global = __brev(global) >> (32 - log_n);
        if (output_transposed)
            global = iterative_boundary_index(global, boundary_first_stage, boundary_stages);
        output[batch_base + global * element_stride] = value;
      }
      __syncthreads();
    }
}
} // namespace cuntt::detail
