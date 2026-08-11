#include "general_kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace cuntt::detail {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct alignas(16) Bytes16 { std::uint64_t lo; std::uint64_t hi; };

template <typename T>
__global__ void pack_1d_kernel(const T* input, T* packed, std::size_t logical_n,
                               std::size_t physical_n, std::size_t batch,
                               std::size_t input_stride, std::size_t input_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = logical_n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / logical_n;
    const std::size_t index = linear - transform * logical_n;
    packed[transform * physical_n + index] = input[transform * input_batch_stride + index * input_stride];
}

template <typename T>
__global__ void scatter_1d_kernel(const T* packed, T* output, std::size_t output_n,
                                  std::size_t physical_n, std::size_t batch,
                                  std::size_t output_stride, std::size_t output_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = output_n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / output_n;
    const std::size_t index = linear - transform * output_n;
    output[transform * output_batch_stride + index * output_stride] = packed[transform * physical_n + index];
}

template <typename T>
__global__ void pack_2d_kernel(const T* input, T* packed,
                               std::size_t logical_rows, std::size_t logical_cols,
                               std::size_t physical_rows, std::size_t physical_cols,
                               std::size_t batch, std::size_t input_row_stride,
                               std::size_t input_col_stride, std::size_t input_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = logical_rows * logical_cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / logical_cols;
    const std::size_t col = within - row * logical_cols;
    packed[(transform * physical_rows + row) * physical_cols + col] =
        input[transform * input_batch_stride + row * input_row_stride + col * input_col_stride];
}

template <typename T>
__global__ void scatter_2d_kernel(const T* packed, T* output,
                                  std::size_t output_rows, std::size_t output_cols,
                                  std::size_t physical_rows, std::size_t physical_cols,
                                  std::size_t batch, std::size_t output_row_stride,
                                  std::size_t output_col_stride, std::size_t output_batch_stride) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = output_rows * output_cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / output_cols;
    const std::size_t col = within - row * output_cols;
    output[transform * output_batch_stride + row * output_row_stride + col * output_col_stride] =
        packed[(transform * physical_rows + row) * physical_cols + col];
}

template <typename T>
__global__ void transpose_2d_kernel(const T* input, T* output,
                                    std::size_t rows, std::size_t cols, std::size_t batch) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t matrix_points = rows * cols;
    const std::size_t count = matrix_points * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / matrix_points;
    const std::size_t within = linear - transform * matrix_points;
    const std::size_t row = within / cols;
    const std::size_t col = within - row * cols;
    output[(transform * cols + col) * rows + row] = input[linear];
}

template <typename Function>
void dispatch_size(std::size_t bytes, Function&& function) {
    switch (bytes) {
        case 2: function(static_cast<std::uint16_t*>(nullptr)); return;
        case 4: function(static_cast<std::uint32_t*>(nullptr)); return;
        case 8: function(static_cast<std::uint64_t*>(nullptr)); return;
        case 16: function(static_cast<Bytes16*>(nullptr)); return;
        default: throw std::invalid_argument("general transform element size must be 2, 4, 8, or 16 bytes");
    }
}

unsigned int blocks_for(std::size_t count) {
    constexpr std::size_t threads = 256;
    const std::size_t blocks = (count + threads - 1) / threads;
    if (blocks > static_cast<std::size_t>(UINT32_MAX))
        throw std::overflow_error("general transform copy grid exceeds CUDA grid.x");
    return static_cast<unsigned int>(blocks);
}

}  // namespace

void launch_pack_1d(const void* input, void* packed, std::size_t logical_n, std::size_t physical_n,
                    std::size_t batch, std::size_t input_stride, std::size_t input_batch_stride,
                    std::size_t bytes, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(packed, 0, physical_n * batch * bytes, stream), "clear packed rank-1 buffer");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        pack_1d_kernel<<<blocks_for(logical_n * batch), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(packed), logical_n, physical_n,
            batch, input_stride, input_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-1 pack");
}

void launch_scatter_1d(const void* packed, void* output, std::size_t output_n, std::size_t physical_n,
                       std::size_t batch, std::size_t output_stride, std::size_t output_batch_stride,
                       std::size_t bytes, cudaStream_t stream) {
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        scatter_1d_kernel<<<blocks_for(output_n * batch), 256, 0, stream>>>(
            static_cast<const T*>(packed), static_cast<T*>(output), output_n, physical_n,
            batch, output_stride, output_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-1 scatter");
}

void launch_pack_2d(const void* input, void* packed, std::size_t logical_rows, std::size_t logical_cols,
                    std::size_t physical_rows, std::size_t physical_cols, std::size_t batch,
                    std::size_t input_row_stride, std::size_t input_col_stride,
                    std::size_t input_batch_stride, std::size_t bytes, cudaStream_t stream) {
    check_cuda(cudaMemsetAsync(packed, 0, physical_rows * physical_cols * batch * bytes, stream),
               "clear packed rank-2 buffer");
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        pack_2d_kernel<<<blocks_for(logical_rows * logical_cols * batch), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(packed), logical_rows, logical_cols,
            physical_rows, physical_cols, batch, input_row_stride, input_col_stride, input_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-2 pack");
}

void launch_scatter_2d(const void* packed, void* output, std::size_t output_rows, std::size_t output_cols,
                       std::size_t physical_rows, std::size_t physical_cols, std::size_t batch,
                       std::size_t output_row_stride, std::size_t output_col_stride,
                       std::size_t output_batch_stride, std::size_t bytes, cudaStream_t stream) {
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        scatter_2d_kernel<<<blocks_for(output_rows * output_cols * batch), 256, 0, stream>>>(
            static_cast<const T*>(packed), static_cast<T*>(output), output_rows, output_cols,
            physical_rows, physical_cols, batch, output_row_stride, output_col_stride, output_batch_stride);
    });
    check_cuda(cudaGetLastError(), "launch rank-2 scatter");
}

void launch_transpose_2d(const void* input, void* output, std::size_t rows, std::size_t cols,
                         std::size_t batch, std::size_t bytes, cudaStream_t stream) {
    dispatch_size(bytes, [&](auto type) {
        using T = std::remove_pointer_t<decltype(type)>;
        transpose_2d_kernel<<<blocks_for(rows * cols * batch), 256, 0, stream>>>(
            static_cast<const T*>(input), static_cast<T*>(output), rows, cols, batch);
    });
    check_cuda(cudaGetLastError(), "launch rank-2 transpose");
}

}  // namespace cuntt::detail
