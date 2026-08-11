#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

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

}  // namespace cuntt::detail
