#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace cuntt::detail {

constexpr std::uint32_t kGeneralMaxRank = 8;

struct GeneralNdLayout {
    std::uint32_t rank = 0;
    std::size_t extents[kGeneralMaxRank]{};
    std::size_t strides[kGeneralMaxRank]{};
    std::size_t points = 0;
    std::size_t batch = 0;
    std::size_t batch_stride = 0;
};

void launch_pack_1d(const void* input, void* packed,
                    std::size_t logical_n, std::size_t physical_n,
                    std::size_t batch, std::size_t input_stride,
                    std::size_t input_batch_stride, std::size_t element_bytes,
                    cudaStream_t stream);

void launch_scatter_1d(const void* packed, void* output,
                       std::size_t output_n, std::size_t physical_n,
                       std::size_t batch, std::size_t output_stride,
                       std::size_t output_batch_stride, std::size_t element_bytes,
                       cudaStream_t stream);

void launch_pack_2d(const void* input, void* packed,
                    std::size_t logical_rows, std::size_t logical_cols,
                    std::size_t physical_rows, std::size_t physical_cols,
                    std::size_t batch, std::size_t input_row_stride,
                    std::size_t input_col_stride, std::size_t input_batch_stride,
                    std::size_t element_bytes, cudaStream_t stream);

void launch_scatter_2d(const void* packed, void* output,
                       std::size_t output_rows, std::size_t output_cols,
                       std::size_t physical_rows, std::size_t physical_cols,
                       std::size_t batch, std::size_t output_row_stride,
                       std::size_t output_col_stride, std::size_t output_batch_stride,
                       std::size_t element_bytes, cudaStream_t stream);

void launch_transpose_2d(const void* input, void* output,
                         std::size_t rows, std::size_t cols, std::size_t batch,
                         std::size_t element_bytes, cudaStream_t stream);

// Generic packed rank-N boundary operations used by the xD lowering.  The
// layout is passed by value so the launch has no per-execution device metadata
// allocation.  `permutation[j]` names the input axis that becomes output axis
// j; both tensors are packed in their respective axis order.
void launch_pack_nd(const void* input, void* packed,
                    const GeneralNdLayout& logical,
                    const GeneralNdLayout& physical,
                    std::size_t element_bytes, cudaStream_t stream);

void launch_scatter_nd(const void* packed, void* output,
                       const GeneralNdLayout& physical,
                       const GeneralNdLayout& logical,
                       std::size_t element_bytes, cudaStream_t stream);

void launch_permute_nd(const void* input, void* output,
                       const GeneralNdLayout& input_layout,
                       const std::uint32_t* permutation,
                       std::size_t element_bytes, cudaStream_t stream);

// Boundary copies only: packed is natural-order, contiguous logical.extents.
// Host bit_reversed_axes contains rank flags (0/1); flagged axes require
// power-of-two extents, including 1. User strides and batch_stride are in
// elements, and the caller guarantees nonoverlapping logical elements/batches.
// Input and output must be distinct buffers; no embedding or padding is added.
void launch_gather_ordered_nd(const void* input, void* packed,
                              const GeneralNdLayout& logical,
                              const std::uint32_t* bit_reversed_axes,
                              std::size_t element_bytes, cudaStream_t stream);

void launch_scatter_ordered_nd(const void* packed, void* output,
                               const GeneralNdLayout& logical,
                               const std::uint32_t* bit_reversed_axes,
                               std::size_t element_bytes, cudaStream_t stream);

}  // namespace cuntt::detail
