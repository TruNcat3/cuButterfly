#include <cuda_runtime.h>
#include <cufftdx.hpp>

#include <stdexcept>

#include "external_fft_units.cuh"
#include "generated_cufftdx_online_config.cuh"

namespace cuntt::detail {
namespace {

template <unsigned int Size, cufftdx::fft_direction Direction>
using BlockFft = decltype(cufftdx::Block() + cufftdx::Size<Size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                          cufftdx::Direction<Direction>() + cufftdx::Precision<float>() + cufftdx::ElementsPerThread<8>() +
                          cufftdx::FFTsPerBlock<1024 / Size>() + cufftdx::SM<700>());

template <unsigned int Size, unsigned int Threads, unsigned int Ept, cufftdx::fft_direction Direction>
using OnlineBlockFft = decltype(cufftdx::Block() + cufftdx::Size<Size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                                cufftdx::Direction<Direction>() + cufftdx::Precision<float>() + cufftdx::ElementsPerThread<Ept>() +
                                cufftdx::FFTsPerBlock<(Threads * Ept) / Size>() + cufftdx::SM<700>());

template <unsigned int Elements, cufftdx::fft_direction Direction>
using Resident4096Fft = decltype(cufftdx::Block() + cufftdx::Size<64>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                                 cufftdx::Direction<Direction>() + cufftdx::Precision<float>() +
                                 cufftdx::ElementsPerThread<Elements>() + cufftdx::FFTsPerBlock<64>() + cufftdx::SM<700>());

template <unsigned int Size, unsigned int Elements, cufftdx::fft_direction Direction>
using DirectFft = decltype(cufftdx::Block() + cufftdx::Size<Size>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                           cufftdx::Direction<Direction>() + cufftdx::Precision<float>() +
                           cufftdx::ElementsPerThread<Elements>() + cufftdx::FFTsPerBlock<1>() + cufftdx::SM<700>());

template <unsigned int Elements, cufftdx::fft_direction Direction>
using Persistent16384Fft = decltype(cufftdx::Block() + cufftdx::Size<128>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                                    cufftdx::Direction<Direction>() + cufftdx::Precision<float>() +
                                    cufftdx::ElementsPerThread<Elements>() + cufftdx::FFTsPerBlock<32>() + cufftdx::SM<700>());

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_block_kernel(
    const Complex32* input, Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
    std::uint64_t element_stride, bool normalize) {
    using Value = typename FFT::value_type;
    const std::uint64_t transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const bool active = transform < transforms;
    Value thread_data[FFT::storage_size];
    const std::uint64_t base = transform * batch_distance;
    unsigned int index = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && index < FFT::input_length) {
            const Complex32 value = input[base + static_cast<std::uint64_t>(index) * element_stride];
            thread_data[item] = Value{value.real, value.imag};
        } else {
            thread_data[item] = Value{0.0F, 0.0F};
        }
        index += FFT::stride;
    }
    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    const float scale = normalize ? 1.0F / static_cast<float>(FFT::input_length) : 1.0F;
    index = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && index < FFT::output_length) {
            const Value value = thread_data[item];
            output[base + static_cast<std::uint64_t>(index) * element_stride] = {value.x * scale, value.y * scale};
        }
        index += FFT::stride;
    }
}

template <class FFT, bool Normalize>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_contiguous_block_kernel(
    const Complex32* input, Complex32* output) {
    using Value = typename FFT::value_type;
    const std::uint64_t base = static_cast<std::uint64_t>(blockIdx.x) * FFT::input_length;
    Value thread_data[FFT::storage_size];
    unsigned int index = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Complex32 value = input[base + index];
        thread_data[item] = Value{value.real, value.imag};
        index += FFT::stride;
    }
    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    constexpr float scale = Normalize ? 1.0F / static_cast<float>(FFT::input_length) : 1.0F;
    index = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Value value = thread_data[item];
        output[base + index] = {value.x * scale, value.y * scale};
        index += FFT::stride;
    }
}

__device__ __forceinline__ Complex32 cross_root(const Complex32* twiddles, std::uint32_t log_n,
                                                 std::uint64_t exponent) {
    const std::uint32_t n    = 1U << log_n;
    const std::uint32_t half = n >> 1;
    const std::uint32_t power = static_cast<std::uint32_t>(exponent) & (n - 1);
    const bool negate = power >= half;
    const Complex32 root = twiddles[half - 1 + (negate ? power - half : power)];
    return negate ? Complex32{-root.real, -root.imag} : root;
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_online_first_kernel(
    const Complex32* input, Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
    std::uint32_t log_n, std::uint32_t local_log_n, std::uint64_t batch_distance,
    std::uint64_t element_stride, bool recurrence_twiddle) {
    using Value = typename FFT::value_type;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n     = 1U << remaining_log_n;
    const std::uint64_t local_transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const std::uint64_t total_local     = transforms * remaining_n;
    const bool          active          = local_transform < total_local;
    const std::uint32_t n2              = static_cast<std::uint32_t>(local_transform) & (remaining_n - 1);
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    const std::uint32_t flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t flat_threads = blockDim.x * blockDim.y;
    constexpr std::uint32_t tile_values = FFT::input_length * FFT::ffts_per_block;
    for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
        const std::uint32_t element = work / FFT::ffts_per_block;
        const std::uint32_t slot    = work - element * FFT::ffts_per_block;
        const std::uint64_t staged_transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + slot;
        const bool staged_active = staged_transform < total_local;
        if (staged_active) {
            const std::uint64_t staged_batch = staged_transform >> remaining_log_n;
            const std::uint32_t staged_n2 = static_cast<std::uint32_t>(staged_transform) & (remaining_n - 1);
            const std::uint64_t logical = static_cast<std::uint64_t>(element) * remaining_n + staged_n2;
            const Complex32 value = input[staged_batch * batch_distance + logical * element_stride];
            tile[work] = Value{value.real, value.imag};
        } else {
            tile[work] = Value{0.0F, 0.0F};
        }
    }
    __syncthreads();

    Value thread_data[FFT::storage_size];
    unsigned int n1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && n1 < FFT::input_length) {
            thread_data[item] = tile[n1 * FFT::ffts_per_block + threadIdx.y];
        } else {
            thread_data[item] = Value{0.0F, 0.0F};
        }
        n1 += FFT::stride;
    }
    __syncthreads();
    FFT().execute(thread_data, storage);
    __syncthreads();

    Complex32 running_root = {1.0F, 0.0F};
    Complex32 root_step    = {1.0F, 0.0F};
    if (active && recurrence_twiddle) {
        running_root = cross_root(twiddles, log_n, static_cast<std::uint64_t>(threadIdx.x) * n2);
        root_step    = cross_root(twiddles, log_n, static_cast<std::uint64_t>(FFT::stride) * n2);
    }
    std::uint32_t k1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && k1 < FFT::output_length) {
            const Value value = thread_data[item];
            const Complex32 root = recurrence_twiddle
                                       ? running_root
                                       : cross_root(twiddles, log_n, static_cast<std::uint64_t>(k1) * n2);
            const Complex32 crossed = {value.x * root.real - value.y * root.imag,
                                       value.x * root.imag + value.y * root.real};
            tile[k1 * FFT::ffts_per_block + threadIdx.y] = Value{crossed.real, crossed.imag};
        }
        if (recurrence_twiddle) {
            running_root = {running_root.real * root_step.real - running_root.imag * root_step.imag,
                            running_root.real * root_step.imag + running_root.imag * root_step.real};
        }
        k1 += FFT::stride;
    }
    __syncthreads();
    for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
        const std::uint32_t element = work / FFT::ffts_per_block;
        const std::uint32_t slot    = work - element * FFT::ffts_per_block;
        const std::uint64_t staged_transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + slot;
        if (staged_transform < total_local) {
            const std::uint64_t staged_batch = staged_transform >> remaining_log_n;
            const std::uint32_t staged_n2 = static_cast<std::uint32_t>(staged_transform) & (remaining_n - 1);
            const std::uint64_t reordered = static_cast<std::uint64_t>(element) * remaining_n + staged_n2;
            const Value value = tile[work];
            scratch[staged_batch * batch_distance + reordered * element_stride] = {value.x, value.y};
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_online_second_kernel(
    const Complex32* scratch, Complex32* output, std::uint64_t transforms, std::uint32_t log_n,
    std::uint32_t local_log_n, std::uint64_t batch_distance, std::uint64_t element_stride,
    bool normalize) {
    using Value = typename FFT::value_type;
    const std::uint32_t local_n         = 1U << local_log_n;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n     = 1U << remaining_log_n;
    const std::uint64_t local_transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const std::uint64_t total_local     = transforms * local_n;
    const bool          active          = local_transform < total_local;
    const std::uint64_t transform       = local_transform >> local_log_n;
    const std::uint32_t k1              = static_cast<std::uint32_t>(local_transform) & (local_n - 1);
    const std::uint64_t base            = transform * batch_distance;

    Value thread_data[FFT::storage_size];
    unsigned int n2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && n2 < FFT::input_length) {
            const std::uint64_t reordered = static_cast<std::uint64_t>(k1) * remaining_n + n2;
            const Complex32 value = scratch[base + reordered * element_stride];
            thread_data[item] = Value{value.real, value.imag};
        } else {
            thread_data[item] = Value{0.0F, 0.0F};
        }
        n2 += FFT::stride;
    }

    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    __syncthreads();
    const float scale = normalize ? 1.0F / static_cast<float>(1U << log_n) : 1.0F;
    Value* tile = reinterpret_cast<Value*>(storage);

    std::uint32_t k2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        if (active && k2 < FFT::output_length) {
            const Value value = thread_data[item];
            tile[k2 * FFT::ffts_per_block + threadIdx.y] = Value{value.x * scale, value.y * scale};
        }
        k2 += FFT::stride;
    }
    __syncthreads();

    const std::uint32_t flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t flat_threads = blockDim.x * blockDim.y;
    constexpr std::uint32_t tile_values = FFT::output_length * FFT::ffts_per_block;
    for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
        const std::uint32_t element = work / FFT::ffts_per_block;
        const std::uint32_t slot    = work - element * FFT::ffts_per_block;
        const std::uint64_t staged_transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + slot;
        if (staged_transform < total_local) {
            const std::uint64_t staged_batch = staged_transform >> local_log_n;
            const std::uint32_t staged_k1 = static_cast<std::uint32_t>(staged_transform) & (local_n - 1);
            const std::uint64_t logical = static_cast<std::uint64_t>(element) * local_n + staged_k1;
            const Value value = tile[work];
            output[staged_batch * batch_distance + logical * element_stride] = {value.x, value.y};
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_online_direct_first_kernel(
    const Complex32* input, Complex32* scratch, const Complex32* twiddles, std::uint32_t log_n,
    std::uint32_t local_log_n, std::uint64_t batch_distance,
    std::uint64_t element_stride, bool recurrence_twiddle) {
    using Value = typename FFT::value_type;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n     = 1U << remaining_log_n;
    const std::uint64_t local_transform = blockIdx.x;
    const std::uint64_t transform       = local_transform >> remaining_log_n;
    const std::uint32_t n2              = static_cast<std::uint32_t>(local_transform) & (remaining_n - 1);
    const std::uint64_t base            = transform * batch_distance;

    Value thread_data[FFT::storage_size];
    std::uint32_t n1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const std::uint64_t logical = static_cast<std::uint64_t>(n1) * remaining_n + n2;
        const Complex32 value = input[base + logical * element_stride];
        thread_data[item] = Value{value.real, value.imag};
        n1 += FFT::stride;
    }

    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);

    Complex32 running_root = {1.0F, 0.0F};
    Complex32 root_step    = {1.0F, 0.0F};
    if (recurrence_twiddle) {
        running_root = cross_root(twiddles, log_n, static_cast<std::uint64_t>(threadIdx.x) * n2);
        root_step    = cross_root(twiddles, log_n, static_cast<std::uint64_t>(FFT::stride) * n2);
    }
    std::uint32_t k1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Value value = thread_data[item];
        const Complex32 root = recurrence_twiddle
                                   ? running_root
                                   : cross_root(twiddles, log_n, static_cast<std::uint64_t>(k1) * n2);
        const Complex32 crossed = {value.x * root.real - value.y * root.imag,
                                   value.x * root.imag + value.y * root.real};
        const std::uint64_t reordered = static_cast<std::uint64_t>(k1) * remaining_n + n2;
        scratch[base + reordered * element_stride] = crossed;
        if (recurrence_twiddle) {
            running_root = {running_root.real * root_step.real - running_root.imag * root_step.imag,
                            running_root.real * root_step.imag + running_root.imag * root_step.real};
        }
        k1 += FFT::stride;
    }
}

template <class FFT, bool ContiguousOutput>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_online_direct_second_kernel(
    const Complex32* scratch, Complex32* output, std::uint32_t log_n, std::uint32_t local_log_n,
    std::uint64_t batch_distance, std::uint64_t element_stride,
    bool normalize) {
    using Value = typename FFT::value_type;
    const std::uint32_t local_n         = 1U << local_log_n;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n     = 1U << remaining_log_n;
    const std::uint64_t local_transform = blockIdx.x;
    const std::uint64_t transform       = local_transform >> local_log_n;
    const std::uint32_t k1              = static_cast<std::uint32_t>(local_transform) & (local_n - 1);
    const std::uint64_t base            = transform * batch_distance;

    Value thread_data[FFT::storage_size];
    std::uint32_t n2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const std::uint64_t reordered = static_cast<std::uint64_t>(k1) * remaining_n + n2;
        const Complex32 value = scratch[base + reordered * element_stride];
        thread_data[item] = Value{value.real, value.imag};
        n2 += FFT::stride;
    }

    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    const float scale = normalize ? 1.0F / static_cast<float>(1U << log_n) : 1.0F;
    std::uint32_t k2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Value value = thread_data[item];
        const std::uint64_t logical = ContiguousOutput
                                          ? static_cast<std::uint64_t>(k1) * remaining_n + k2
                                          : static_cast<std::uint64_t>(k2) * local_n + k1;
        output[base + logical * element_stride] = {value.x * scale, value.y * scale};
        k2 += FFT::stride;
    }
}

__global__ void transpose_complex32_kernel(const Complex32* input, Complex32* output,
                                           std::uint32_t rows, std::uint32_t columns,
                                           std::uint64_t tiles_per_transform,
                                           std::uint64_t batch_distance,
                                           std::uint64_t element_stride) {
    __shared__ Complex32 tile[32][33];
    const std::uint64_t flat_block = blockIdx.x;
    const std::uint64_t transform = flat_block / tiles_per_transform;
    const std::uint64_t tile_index = flat_block - transform * tiles_per_transform;
    const std::uint32_t tiles_x = (columns + 31U) / 32U;
    const std::uint32_t tile_y = static_cast<std::uint32_t>(tile_index / tiles_x);
    const std::uint32_t tile_x = static_cast<std::uint32_t>(tile_index - static_cast<std::uint64_t>(tile_y) * tiles_x);
    const std::uint64_t base = transform * batch_distance;

#pragma unroll
    for (std::uint32_t offset = 0; offset < 32; offset += 8) {
        const std::uint32_t row = tile_y * 32U + threadIdx.y + offset;
        const std::uint32_t column = tile_x * 32U + threadIdx.x;
        if (row < rows && column < columns)
            tile[threadIdx.y + offset][threadIdx.x] =
                input[base + (static_cast<std::uint64_t>(row) * columns + column) * element_stride];
    }
    __syncthreads();

#pragma unroll
    for (std::uint32_t offset = 0; offset < 32; offset += 8) {
        const std::uint32_t output_row = tile_x * 32U + threadIdx.y + offset;
        const std::uint32_t output_column = tile_y * 32U + threadIdx.x;
        if (output_row < columns && output_column < rows)
            output[base + (static_cast<std::uint64_t>(output_row) * rows + output_column) * element_stride] =
                tile[threadIdx.x][threadIdx.y + offset];
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_resident_4096_kernel(
    const Complex32* input, Complex32* output, const Complex32* twiddles, std::uint64_t transforms,
    std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize, bool recurrence_twiddle) {
    using Value = typename FFT::value_type;
    constexpr std::uint32_t dimension = 64;
    constexpr std::uint32_t tile_values = dimension * dimension;
    constexpr std::uint32_t tile_pitch = dimension + 1;
    static_assert(FFT::input_length == dimension);
    static_assert(FFT::output_length == dimension);
    static_assert(FFT::ffts_per_block == dimension);

    const std::uint64_t transform = blockIdx.x;
    if (transform >= transforms)
        return;
    const std::uint64_t base = transform * batch_distance;
    const std::uint32_t flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t flat_threads = blockDim.x * blockDim.y;
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
        const Complex32 value = input[base + static_cast<std::uint64_t>(work) * element_stride];
        const std::uint32_t row = work / dimension;
        const std::uint32_t column = work - row * dimension;
        tile[row * tile_pitch + column] = Value{value.real, value.imag};
    }
    __syncthreads();

    Value first_dimension[FFT::storage_size];
    std::uint32_t n1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        first_dimension[item] = tile[n1 * tile_pitch + threadIdx.y];
        n1 += FFT::stride;
    }
    __syncthreads();
    FFT().execute(first_dimension, storage);
    __syncthreads();

    Complex32 running_root = {1.0F, 0.0F};
    Complex32 root_step = {1.0F, 0.0F};
    const std::uint32_t n2 = threadIdx.y;
    if (recurrence_twiddle) {
        running_root = cross_root(twiddles, 12, static_cast<std::uint64_t>(threadIdx.x) * n2);
        root_step = cross_root(twiddles, 12, static_cast<std::uint64_t>(FFT::stride) * n2);
    }
    std::uint32_t k1 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Value value = first_dimension[item];
        const Complex32 root = recurrence_twiddle
                                   ? running_root
                                   : cross_root(twiddles, 12, static_cast<std::uint64_t>(k1) * n2);
        tile[k1 * tile_pitch + n2] = Value{value.x * root.real - value.y * root.imag,
                                           value.x * root.imag + value.y * root.real};
        if (recurrence_twiddle) {
            running_root = {running_root.real * root_step.real - running_root.imag * root_step.imag,
                            running_root.real * root_step.imag + running_root.imag * root_step.real};
        }
        k1 += FFT::stride;
    }
    __syncthreads();

    Value second_dimension[FFT::storage_size];
    std::uint32_t second_n2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        second_dimension[item] = tile[threadIdx.y * tile_pitch + second_n2];
        second_n2 += FFT::stride;
    }
    __syncthreads();
    FFT().execute(second_dimension, storage);
    __syncthreads();

    const float scale = normalize ? 1.0F / static_cast<float>(tile_values) : 1.0F;
    std::uint32_t k2 = threadIdx.x;
#pragma unroll
    for (unsigned int item = 0; item < FFT::storage_size; ++item) {
        const Value value = second_dimension[item];
        tile[k2 * tile_pitch + threadIdx.y] = Value{value.x * scale, value.y * scale};
        k2 += FFT::stride;
    }
    __syncthreads();

    for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
        const std::uint32_t row = work / dimension;
        const std::uint32_t column = work - row * dimension;
        const Value value = tile[row * tile_pitch + column];
        output[base + static_cast<std::uint64_t>(work) * element_stride] = {value.x, value.y};
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void cufftdx_persistent_16384_kernel(
    const Complex32* input, Complex32* output, Complex32* scratch, const Complex32* twiddles,
    std::uint64_t transforms, std::uint64_t batch_distance, std::uint64_t element_stride,
    bool normalize, bool recurrence_twiddle) {
    using Value = typename FFT::value_type;
    constexpr std::uint32_t dimension = 128;
    constexpr std::uint32_t wave_size = 32;
    constexpr std::uint32_t waves = dimension / wave_size;
    constexpr std::uint32_t tile_pitch = wave_size + 1;
    constexpr std::uint32_t tile_values = dimension * wave_size;
    static_assert(FFT::input_length == dimension);
    static_assert(FFT::output_length == dimension);
    static_assert(FFT::ffts_per_block == wave_size);

    const std::uint64_t transform = blockIdx.x;
    if (transform >= transforms)
        return;
    const std::uint64_t base = transform * batch_distance;
    const std::uint32_t flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t flat_threads = blockDim.x * blockDim.y;
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    for (std::uint32_t wave = 0; wave < waves; ++wave) {
        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const std::uint32_t n1 = work / wave_size;
            const std::uint32_t slot = work - n1 * wave_size;
            const std::uint32_t n2 = wave * wave_size + slot;
            const std::uint64_t logical = static_cast<std::uint64_t>(n1) * dimension + n2;
            const Complex32 value = input[base + logical * element_stride];
            tile[n1 * tile_pitch + slot] = Value{value.real, value.imag};
        }
        __syncthreads();

        Value values[FFT::storage_size];
        std::uint32_t n1 = threadIdx.x;
#pragma unroll
        for (unsigned int item = 0; item < FFT::storage_size; ++item) {
            values[item] = tile[n1 * tile_pitch + threadIdx.y];
            n1 += FFT::stride;
        }
        __syncthreads();
        FFT().execute(values, storage);
        __syncthreads();

        const std::uint32_t n2 = wave * wave_size + threadIdx.y;
        Complex32 running_root = {1.0F, 0.0F};
        Complex32 root_step = {1.0F, 0.0F};
        if (recurrence_twiddle) {
            running_root = cross_root(twiddles, 14, static_cast<std::uint64_t>(threadIdx.x) * n2);
            root_step = cross_root(twiddles, 14, static_cast<std::uint64_t>(FFT::stride) * n2);
        }
        std::uint32_t k1 = threadIdx.x;
#pragma unroll
        for (unsigned int item = 0; item < FFT::storage_size; ++item) {
            const Value value = values[item];
            const Complex32 root = recurrence_twiddle
                                       ? running_root
                                       : cross_root(twiddles, 14, static_cast<std::uint64_t>(k1) * n2);
            tile[k1 * tile_pitch + threadIdx.y] = Value{value.x * root.real - value.y * root.imag,
                                                        value.x * root.imag + value.y * root.real};
            if (recurrence_twiddle) {
                running_root = {running_root.real * root_step.real - running_root.imag * root_step.imag,
                                running_root.real * root_step.imag + running_root.imag * root_step.real};
            }
            k1 += FFT::stride;
        }
        __syncthreads();

        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const std::uint32_t crossed_k1 = work / wave_size;
            const std::uint32_t slot = work - crossed_k1 * wave_size;
            const std::uint32_t crossed_n2 = wave * wave_size + slot;
            const Value value = tile[crossed_k1 * tile_pitch + slot];
            const std::uint64_t logical = static_cast<std::uint64_t>(crossed_k1) * dimension + crossed_n2;
            scratch[base + logical * element_stride] = {value.x, value.y};
        }
        __syncthreads();
    }

    for (std::uint32_t wave = 0; wave < waves; ++wave) {
        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const std::uint32_t n2 = work / wave_size;
            const std::uint32_t slot = work - n2 * wave_size;
            const std::uint32_t k1 = wave * wave_size + slot;
            const std::uint64_t logical = static_cast<std::uint64_t>(k1) * dimension + n2;
            const Complex32 value = scratch[base + logical * element_stride];
            tile[n2 * tile_pitch + slot] = Value{value.real, value.imag};
        }
        __syncthreads();

        Value values[FFT::storage_size];
        std::uint32_t n2 = threadIdx.x;
#pragma unroll
        for (unsigned int item = 0; item < FFT::storage_size; ++item) {
            values[item] = tile[n2 * tile_pitch + threadIdx.y];
            n2 += FFT::stride;
        }
        __syncthreads();
        FFT().execute(values, storage);
        __syncthreads();

        const float scale = normalize ? 1.0F / static_cast<float>(dimension * dimension) : 1.0F;
        std::uint32_t k2 = threadIdx.x;
#pragma unroll
        for (unsigned int item = 0; item < FFT::storage_size; ++item) {
            const Value value = values[item];
            tile[k2 * tile_pitch + threadIdx.y] = Value{value.x * scale, value.y * scale};
            k2 += FFT::stride;
        }
        __syncthreads();

        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const std::uint32_t output_k2 = work / wave_size;
            const std::uint32_t slot = work - output_k2 * wave_size;
            const std::uint32_t output_k1 = wave * wave_size + slot;
            const Value value = tile[output_k2 * tile_pitch + slot];
            const std::uint64_t logical = static_cast<std::uint64_t>(output_k2) * dimension + output_k1;
            output[base + logical * element_stride] = {value.x, value.y};
        }
        __syncthreads();
    }
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void launch(const Complex32* input, Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
            std::uint64_t element_stride, bool normalize) {
    using FFT = BlockFft<Size, Direction>;
    if (FFT::shared_memory_size > 48U * 1024U) {
        cudaFuncSetAttribute(cufftdx_block_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, FFT::shared_memory_size);
    }
    const auto blocks = static_cast<unsigned int>((transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    cufftdx_block_kernel<FFT><<<blocks, FFT::block_dim, FFT::shared_memory_size>>>(
        input, output, transforms, batch_distance, element_stride, normalize);
}

template <cufftdx::fft_direction Direction>
void dispatch(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
              std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    switch (log_n) {
        case 3: launch<8, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 4: launch<16, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 5: launch<32, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 6: launch<64, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 7: launch<128, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 8: launch<256, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 9: launch<512, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 10: launch<1024, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        default: throw std::invalid_argument("cufftdx-block supports logN=3..10");
    }
}

template <unsigned int Size, unsigned int Threads, unsigned int Ept, cufftdx::fft_direction Direction>
void launch_online_first(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                         Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
                         std::uint64_t batch_distance, std::uint64_t element_stride, bool recurrence_twiddle) {
    if constexpr (Ept <= Size && Threads * Ept >= Size && (Threads * Ept) % Size == 0) {
        using FFT = OnlineBlockFft<Size, Threads, Ept, Direction>;
        const std::uint64_t local_transforms = transforms << (log_n - local_log_n);
        const auto blocks = static_cast<unsigned int>((local_transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
        constexpr std::size_t tile_bytes = Size * FFT::ffts_per_block * sizeof(typename FFT::value_type);
        constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
        if constexpr (shared_bytes > 48U * 1024U)
            cudaFuncSetAttribute(cufftdx_online_first_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        cufftdx_online_first_kernel<FFT><<<blocks, FFT::block_dim, shared_bytes>>>(
            input, scratch, twiddles, transforms, log_n, local_log_n, batch_distance, element_stride, recurrence_twiddle);
    } else {
        throw std::invalid_argument("cuFFTDx prefix thread/EPT point cannot cover an integer number of FFTs");
    }
}

template <unsigned int Size, unsigned int Threads, unsigned int Ept, cufftdx::fft_direction Direction>
void launch_online_second(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* scratch,
                          Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                          std::uint64_t element_stride, bool normalize) {
    if constexpr (Ept <= Size && Threads * Ept >= Size && (Threads * Ept) % Size == 0) {
        using FFT = OnlineBlockFft<Size, Threads, Ept, Direction>;
        const std::uint64_t local_transforms = transforms << local_log_n;
        const auto blocks = static_cast<unsigned int>((local_transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
        constexpr std::size_t tile_bytes = Size * FFT::ffts_per_block * sizeof(typename FFT::value_type);
        constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
        if constexpr (shared_bytes > 48U * 1024U)
            cudaFuncSetAttribute(cufftdx_online_second_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        cufftdx_online_second_kernel<FFT><<<blocks, FFT::block_dim, shared_bytes>>>(
            scratch, output, transforms, log_n, local_log_n, batch_distance, element_stride, normalize);
    } else {
        throw std::invalid_argument("cuFFTDx suffix thread/EPT point cannot cover an integer number of FFTs");
    }
}

template <unsigned int Size, unsigned int Threads, unsigned int Ept, cufftdx::fft_direction Direction>
void launch_online_direct_first(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                                Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
                                std::uint64_t batch_distance, std::uint64_t element_stride,
                                bool recurrence_twiddle) {
    if constexpr (Threads * Ept == Size) {
        using FFT = DirectFft<Size, Ept, Direction>;
        static_assert(FFT::block_dim.x == Threads);
        if constexpr (FFT::shared_memory_size > 48U * 1024U)
            cudaFuncSetAttribute(cufftdx_online_direct_first_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size);
        const auto blocks = static_cast<unsigned int>(transforms << (log_n - local_log_n));
        cufftdx_online_direct_first_kernel<FFT><<<blocks, FFT::block_dim, FFT::shared_memory_size>>>(
            input, scratch, twiddles, log_n, local_log_n, batch_distance, element_stride, recurrence_twiddle);
    } else {
        throw std::invalid_argument("cuFFTDx direct prefix requires exactly one FFT per CTA");
    }
}

template <unsigned int Size, unsigned int Threads, unsigned int Ept, cufftdx::fft_direction Direction,
          bool ContiguousOutput>
void launch_online_direct_second(std::uint32_t log_n, std::uint32_t local_log_n, Complex32* scratch,
                                 Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                                 std::uint64_t element_stride, bool normalize) {
    if constexpr (Threads * Ept == Size) {
        using FFT = DirectFft<Size, Ept, Direction>;
        static_assert(FFT::block_dim.x == Threads);
        if constexpr (FFT::shared_memory_size > 48U * 1024U)
            cudaFuncSetAttribute(cufftdx_online_direct_second_kernel<FFT, ContiguousOutput>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 FFT::shared_memory_size);
        const auto blocks = static_cast<unsigned int>(transforms << local_log_n);
        cufftdx_online_direct_second_kernel<FFT, ContiguousOutput><<<blocks, FFT::block_dim, FFT::shared_memory_size>>>(
            scratch, ContiguousOutput ? scratch : output, log_n, local_log_n,
            batch_distance, element_stride, normalize);
    } else {
        throw std::invalid_argument("cuFFTDx direct suffix requires exactly one FFT per CTA");
    }
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void dispatch_online_direct_first_threads(std::uint32_t threads, std::uint32_t ept, std::uint32_t log_n,
                                          std::uint32_t local_log_n, const Complex32* input, Complex32* scratch,
                                          const Complex32* twiddles, std::uint64_t transforms,
                                          std::uint64_t batch_distance, std::uint64_t element_stride,
                                          bool recurrence_twiddle) {
    switch ((ept << 16) | threads) {
        case (2U << 16) | 1024U: launch_online_direct_first<Size, 1024, 2, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case (4U << 16) | 512U: launch_online_direct_first<Size, 512, 4, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case (4U << 16) | 1024U: launch_online_direct_first<Size, 1024, 4, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case (8U << 16) | 256U: launch_online_direct_first<Size, 256, 8, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case (8U << 16) | 512U: launch_online_direct_first<Size, 512, 8, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case (16U << 16) | 256U: launch_online_direct_first<Size, 256, 16, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        default: throw std::invalid_argument("unsupported cuFFTDx direct prefix thread/EPT point");
    }
}

template <unsigned int Size, cufftdx::fft_direction Direction, bool ContiguousOutput>
void dispatch_online_direct_second_threads_impl(std::uint32_t threads, std::uint32_t ept, std::uint32_t log_n,
                                                std::uint32_t local_log_n, Complex32* scratch, Complex32* output,
                                                std::uint64_t transforms, std::uint64_t batch_distance,
                                                std::uint64_t element_stride, bool normalize) {
    switch ((ept << 16) | threads) {
        case (2U << 16) | 1024U: launch_online_direct_second<Size, 1024, 2, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case (4U << 16) | 512U: launch_online_direct_second<Size, 512, 4, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case (4U << 16) | 1024U: launch_online_direct_second<Size, 1024, 4, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case (8U << 16) | 256U: launch_online_direct_second<Size, 256, 8, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case (8U << 16) | 512U: launch_online_direct_second<Size, 512, 8, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case (16U << 16) | 256U: launch_online_direct_second<Size, 256, 16, Direction, ContiguousOutput>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        default: throw std::invalid_argument("unsupported cuFFTDx direct suffix thread/EPT point");
    }
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void dispatch_online_direct_second_threads(std::uint32_t threads, std::uint32_t ept, std::uint32_t log_n,
                                           std::uint32_t local_log_n, Complex32* scratch, Complex32* output,
                                           std::uint64_t transforms, std::uint64_t batch_distance,
                                           std::uint64_t element_stride, bool normalize, bool tiled_transpose) {
    if (tiled_transpose)
        dispatch_online_direct_second_threads_impl<Size, Direction, true>(threads, ept, log_n, local_log_n,
                                                                          scratch, output, transforms, batch_distance,
                                                                          element_stride, normalize);
    else
        dispatch_online_direct_second_threads_impl<Size, Direction, false>(threads, ept, log_n, local_log_n,
                                                                           scratch, output, transforms, batch_distance,
                                                                           element_stride, normalize);
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void dispatch_online_first_threads(std::uint32_t threads, std::uint32_t ept, std::uint32_t log_n, std::uint32_t local_log_n,
                                   const Complex32* input, Complex32* scratch, const Complex32* twiddles,
                                   std::uint64_t transforms, std::uint64_t batch_distance,
                                   std::uint64_t element_stride, bool recurrence_twiddle) {
    switch ((ept << 16) | threads) {
#define CUNTT_CUFFTDX_ONLINE_LAUNCH(Threads, Ept) \
        launch_online_first<Size, Threads, Ept, Direction>(log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle)
#include "generated_cufftdx_online_dispatch.inc"
#undef CUNTT_CUFFTDX_ONLINE_LAUNCH
        default: throw std::invalid_argument("requested cuFFTDx prefix thread/EPT point was not generated");
    }
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void dispatch_online_second_threads(std::uint32_t threads, std::uint32_t ept, std::uint32_t log_n, std::uint32_t local_log_n,
                                    const Complex32* scratch, Complex32* output, std::uint64_t transforms,
                                    std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    switch ((ept << 16) | threads) {
#define CUNTT_CUFFTDX_ONLINE_LAUNCH(Threads, Ept) \
        launch_online_second<Size, Threads, Ept, Direction>(log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize)
#include "generated_cufftdx_online_dispatch.inc"
#undef CUNTT_CUFFTDX_ONLINE_LAUNCH
        default: throw std::invalid_argument("requested cuFFTDx suffix thread/EPT point was not generated");
    }
}

template <cufftdx::fft_direction Direction>
void dispatch_online_first(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                           Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
                           std::uint64_t batch_distance, std::uint64_t element_stride, bool recurrence_twiddle,
                           std::uint32_t threads, std::uint32_t ept) {
    switch (local_log_n) {
        case 3: dispatch_online_first_threads<8, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 4: dispatch_online_first_threads<16, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 5: dispatch_online_first_threads<32, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 6: dispatch_online_first_threads<64, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 7: dispatch_online_first_threads<128, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 8: dispatch_online_first_threads<256, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 9: dispatch_online_first_threads<512, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 10: dispatch_online_first_threads<1024, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 11: dispatch_online_direct_first_threads<2048, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        case 12: dispatch_online_direct_first_threads<4096, Direction>(threads, ept, log_n, local_log_n, input, scratch, twiddles, transforms, batch_distance, element_stride, recurrence_twiddle); return;
        default: throw std::invalid_argument("cufftdx online first pass supports logN=3..12");
    }
}

template <cufftdx::fft_direction Direction>
void dispatch_online_second(std::uint32_t log_n, std::uint32_t local_log_n, Complex32* scratch,
                            Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                            std::uint64_t element_stride, bool normalize, std::uint32_t threads, std::uint32_t ept,
                            bool tiled_transpose) {
    const auto launch_transpose = [&] {
        const std::uint32_t rows = 1U << local_log_n;
        const std::uint32_t columns = 1U << (log_n - local_log_n);
        const std::uint64_t tiles_x = (columns + 31U) / 32U;
        const std::uint64_t tiles_y = (rows + 31U) / 32U;
        const std::uint64_t tiles_per_transform = tiles_x * tiles_y;
        const std::uint64_t blocks = transforms * tiles_per_transform;
        if (blocks > 0x7fffffffULL)
            throw std::invalid_argument("tiled transpose grid exceeds the CUDA grid.x limit");
        transpose_complex32_kernel<<<static_cast<unsigned int>(blocks), dim3(32, 8)>>>(
            scratch, output, rows, columns, tiles_per_transform, batch_distance, element_stride);
    };
    switch (log_n - local_log_n) {
        case 3: if (tiled_transpose) break; dispatch_online_second_threads<8, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 4: if (tiled_transpose) break; dispatch_online_second_threads<16, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 5: if (tiled_transpose) break; dispatch_online_second_threads<32, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 6: if (tiled_transpose) break; dispatch_online_second_threads<64, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 7: if (tiled_transpose) break; dispatch_online_second_threads<128, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 8: if (tiled_transpose) break; dispatch_online_second_threads<256, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 9: if (tiled_transpose) break; dispatch_online_second_threads<512, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 10: if (tiled_transpose) break; dispatch_online_second_threads<1024, Direction>(threads, ept, log_n, local_log_n, scratch, output, transforms, batch_distance, element_stride, normalize); return;
        case 11:
            dispatch_online_direct_second_threads<2048, Direction>(threads, ept, log_n, local_log_n, scratch, output,
                                                                   transforms, batch_distance, element_stride, normalize,
                                                                   tiled_transpose);
            if (tiled_transpose) launch_transpose();
            return;
        case 12:
            dispatch_online_direct_second_threads<4096, Direction>(threads, ept, log_n, local_log_n, scratch, output,
                                                                   transforms, batch_distance, element_stride, normalize,
                                                                   tiled_transpose);
            if (tiled_transpose) launch_transpose();
            return;
        default: throw std::invalid_argument("cufftdx online second pass supports logN=3..12");
    }
    throw std::invalid_argument("tiled transpose requires a direct suffix in logN=11..12");
}

template <unsigned int Elements, cufftdx::fft_direction Direction>
void launch_resident_4096(const Complex32* input, Complex32* output, const Complex32* twiddles,
                          std::uint64_t transforms, std::uint64_t batch_distance, std::uint64_t element_stride,
                          bool normalize, bool recurrence_twiddle) {
    using FFT = Resident4096Fft<Elements, Direction>;
    constexpr std::size_t tile_bytes = 64 * 65 * sizeof(Complex32);
    constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    cufftdx_resident_4096_kernel<FFT><<<static_cast<unsigned int>(transforms), FFT::block_dim, shared_bytes>>>(
        input, output, twiddles, transforms, batch_distance, element_stride, normalize, recurrence_twiddle);
}

template <unsigned int Size, unsigned int Elements, cufftdx::fft_direction Direction>
void launch_direct(const Complex32* input, Complex32* output, std::uint64_t transforms,
                   std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    using FFT = DirectFft<Size, Elements, Direction>;
    if (FFT::shared_memory_size > 48U * 1024U) {
        cudaFuncSetAttribute(cufftdx_block_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             FFT::shared_memory_size);
        cudaFuncSetAttribute(cufftdx_contiguous_block_kernel<FFT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             FFT::shared_memory_size);
        cudaFuncSetAttribute(cufftdx_contiguous_block_kernel<FFT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             FFT::shared_memory_size);
    }
    if (batch_distance == Size && element_stride == 1) {
        if (normalize) {
            cufftdx_contiguous_block_kernel<FFT, true>
                <<<static_cast<unsigned int>(transforms), FFT::block_dim, FFT::shared_memory_size>>>(input, output);
        } else {
            cufftdx_contiguous_block_kernel<FFT, false>
                <<<static_cast<unsigned int>(transforms), FFT::block_dim, FFT::shared_memory_size>>>(input, output);
        }
        return;
    }
    cufftdx_block_kernel<FFT><<<static_cast<unsigned int>(transforms), FFT::block_dim, FFT::shared_memory_size>>>(
        input, output, transforms, batch_distance, element_stride, normalize);
}

template <unsigned int Size, cufftdx::fft_direction Direction>
void dispatch_direct_threads(std::uint32_t tile_threads, const Complex32* input, Complex32* output,
                             std::uint64_t transforms, std::uint64_t batch_distance,
                             std::uint64_t element_stride, bool normalize) {
    switch (tile_threads) {
        case 256:
            launch_direct<Size, Size / 256, Direction>(input, output, transforms, batch_distance,
                                                       element_stride, normalize);
            return;
        case 512:
            launch_direct<Size, Size / 512, Direction>(input, output, transforms, batch_distance,
                                                       element_stride, normalize);
            return;
        case 1024:
            launch_direct<Size, Size / 1024, Direction>(input, output, transforms, batch_distance,
                                                        element_stride, normalize);
            return;
        default:
            throw std::invalid_argument("cufftdx-direct tile_threads must be 256, 512, or 1024");
    }
}

template <cufftdx::fft_direction Direction>
void dispatch_direct_16384(std::uint32_t tile_threads, const Complex32* input, Complex32* output,
                           std::uint64_t transforms, std::uint64_t batch_distance,
                           std::uint64_t element_stride, bool normalize) {
    switch (tile_threads) {
        case 512:
            launch_direct<16384, 32, Direction>(input, output, transforms, batch_distance, element_stride, normalize);
            return;
        case 1024:
            launch_direct<16384, 16, Direction>(input, output, transforms, batch_distance, element_stride, normalize);
            return;
        default:
            throw std::invalid_argument("cufftdx-direct logN=14 requires 512 or 1024 threads");
    }
}

template <cufftdx::fft_direction Direction>
void dispatch_direct(std::uint32_t log_n, std::uint32_t tile_threads, const Complex32* input,
                     Complex32* output, std::uint64_t transforms, std::uint64_t batch_distance,
                     std::uint64_t element_stride, bool normalize) {
    switch (log_n) {
        case 11:
            dispatch_direct_threads<2048, Direction>(tile_threads, input, output, transforms, batch_distance,
                                                      element_stride, normalize);
            return;
        case 12:
            dispatch_direct_threads<4096, Direction>(tile_threads, input, output, transforms, batch_distance,
                                                      element_stride, normalize);
            return;
        case 13:
            dispatch_direct_threads<8192, Direction>(tile_threads, input, output, transforms, batch_distance,
                                                      element_stride, normalize);
            return;
        case 14:
            dispatch_direct_16384<Direction>(tile_threads, input, output, transforms, batch_distance,
                                              element_stride, normalize);
            return;
        default:
            throw std::invalid_argument("cufftdx-direct supports logN=11..14");
    }
}

template <unsigned int Elements, cufftdx::fft_direction Direction>
void launch_persistent_16384(const Complex32* input, Complex32* output, Complex32* scratch,
                             const Complex32* twiddles, std::uint64_t transforms, std::uint64_t batch_distance,
                             std::uint64_t element_stride, bool normalize, bool recurrence_twiddle) {
    using FFT = Persistent16384Fft<Elements, Direction>;
    constexpr std::size_t tile_bytes = 128 * 33 * sizeof(Complex32);
    constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    cufftdx_persistent_16384_kernel<FFT><<<static_cast<unsigned int>(transforms), FFT::block_dim, shared_bytes>>>(
        input, output, scratch, twiddles, transforms, batch_distance, element_stride, normalize, recurrence_twiddle);
}

} // namespace

bool cufftdx_block_available(std::uint32_t log_n) noexcept { return log_n >= 3 && log_n <= 10; }

bool cufftdx_online_available(std::uint32_t log_n, std::uint32_t threads, std::uint32_t ept) noexcept {
    if (log_n == 11 || log_n == 12) {
        return (threads == 256 || threads == 512 || threads == 1024) &&
               static_cast<std::uint64_t>(threads) * ept == (1ULL << log_n);
    }
    return generated_cufftdx_online_available(log_n, threads, ept);
}

bool cufftdx_direct_available(std::uint32_t log_n) noexcept { return log_n >= 11 && log_n <= 14; }

bool cufftdx_resident_available(std::uint32_t log_n, std::uint32_t local_log_n) noexcept {
    return (log_n == 12 && local_log_n == 6) || (log_n == 14 && local_log_n == 7);
}

void launch_cufftdx_block(std::uint32_t log_n, const Complex32* input, Complex32* output, std::uint64_t transforms,
                          std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse, bool normalize) {
    if (inverse) {
        dispatch<cufftdx::fft_direction::inverse>(log_n, input, output, transforms, batch_distance, element_stride, normalize);
    } else {
        dispatch<cufftdx::fft_direction::forward>(log_n, input, output, transforms, batch_distance, element_stride, false);
    }
}

void launch_cufftdx_direct(std::uint32_t log_n, const Complex32* input, Complex32* output,
                           std::uint64_t transforms, std::uint64_t batch_distance, std::uint64_t element_stride,
                           bool inverse, bool normalize, std::uint32_t tile_threads) {
    if (!cufftdx_direct_available(log_n))
        throw std::invalid_argument("cufftdx-direct requires logN=11..14");
    if (inverse) {
        dispatch_direct<cufftdx::fft_direction::inverse>(log_n, tile_threads, input, output, transforms,
                                                         batch_distance, element_stride, normalize);
    } else {
        dispatch_direct<cufftdx::fft_direction::forward>(log_n, tile_threads, input, output, transforms,
                                                         batch_distance, element_stride, false);
    }
}

void launch_cufftdx_online_reorder(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                                   Complex32* output, Complex32* scratch, const Complex32* twiddles,
                                   std::uint64_t transforms, std::uint64_t batch_distance,
                                   std::uint64_t element_stride, bool inverse, bool normalize,
                                   CrossTwiddleMode cross_twiddle, std::uint32_t prefix_threads,
                                   std::uint32_t suffix_threads, std::uint32_t prefix_ept,
                                   std::uint32_t suffix_ept, DirectBoundary direct_boundary) {
    const bool recurrence_twiddle = cross_twiddle == CrossTwiddleMode::Recurrence;
    const bool tiled_transpose = direct_boundary == DirectBoundary::TiledTranspose;
    if (inverse) {
        dispatch_online_first<cufftdx::fft_direction::inverse>(log_n, local_log_n, input, scratch, twiddles,
                                                               transforms, batch_distance, element_stride, recurrence_twiddle,
                                                               prefix_threads, prefix_ept);
        dispatch_online_second<cufftdx::fft_direction::inverse>(log_n, local_log_n, scratch, output, transforms,
                                                                batch_distance, element_stride, normalize, suffix_threads, suffix_ept,
                                                                tiled_transpose);
    } else {
        dispatch_online_first<cufftdx::fft_direction::forward>(log_n, local_log_n, input, scratch, twiddles,
                                                               transforms, batch_distance, element_stride, recurrence_twiddle,
                                                               prefix_threads, prefix_ept);
        dispatch_online_second<cufftdx::fft_direction::forward>(log_n, local_log_n, scratch, output, transforms,
                                                                batch_distance, element_stride, false, suffix_threads, suffix_ept,
                                                                tiled_transpose);
    }
}

void launch_cufftdx_resident(std::uint32_t log_n, std::uint32_t local_log_n, const Complex32* input,
                             Complex32* output, Complex32* scratch, const Complex32* twiddles, std::uint64_t transforms,
                             std::uint64_t batch_distance, std::uint64_t element_stride, bool inverse,
                             bool normalize, CrossTwiddleMode cross_twiddle, std::uint32_t tile_threads) {
    if (!cufftdx_resident_available(log_n, local_log_n))
        throw std::invalid_argument("cufftdx-resident requires logN=12/6+6 or logN=14/7+7");
    const bool recurrence_twiddle = cross_twiddle == CrossTwiddleMode::Recurrence;
    if (log_n == 14) {
        if (inverse) {
            if (tile_threads == 256)
                launch_persistent_16384<16, cufftdx::fft_direction::inverse>(input, output, scratch, twiddles, transforms,
                                                                            batch_distance, element_stride, normalize,
                                                                            recurrence_twiddle);
            else if (tile_threads == 512)
                launch_persistent_16384<8, cufftdx::fft_direction::inverse>(input, output, scratch, twiddles, transforms,
                                                                           batch_distance, element_stride, normalize,
                                                                           recurrence_twiddle);
            else if (tile_threads == 1024)
                launch_persistent_16384<4, cufftdx::fft_direction::inverse>(input, output, scratch, twiddles, transforms,
                                                                           batch_distance, element_stride, normalize,
                                                                           recurrence_twiddle);
            else
                throw std::invalid_argument("cufftdx-resident tile_threads must be 256, 512, or 1024");
        } else {
            if (tile_threads == 256)
                launch_persistent_16384<16, cufftdx::fft_direction::forward>(input, output, scratch, twiddles, transforms,
                                                                            batch_distance, element_stride, false,
                                                                            recurrence_twiddle);
            else if (tile_threads == 512)
                launch_persistent_16384<8, cufftdx::fft_direction::forward>(input, output, scratch, twiddles, transforms,
                                                                           batch_distance, element_stride, false,
                                                                           recurrence_twiddle);
            else if (tile_threads == 1024)
                launch_persistent_16384<4, cufftdx::fft_direction::forward>(input, output, scratch, twiddles, transforms,
                                                                           batch_distance, element_stride, false,
                                                                           recurrence_twiddle);
            else
                throw std::invalid_argument("cufftdx-resident tile_threads must be 256, 512, or 1024");
        }
        return;
    }
    if (inverse) {
        if (tile_threads == 256)
            launch_resident_4096<16, cufftdx::fft_direction::inverse>(input, output, twiddles, transforms, batch_distance,
                                                                     element_stride, normalize, recurrence_twiddle);
        else if (tile_threads == 512)
            launch_resident_4096<8, cufftdx::fft_direction::inverse>(input, output, twiddles, transforms, batch_distance,
                                                                    element_stride, normalize, recurrence_twiddle);
        else if (tile_threads == 1024)
            launch_resident_4096<4, cufftdx::fft_direction::inverse>(input, output, twiddles, transforms, batch_distance,
                                                                    element_stride, normalize, recurrence_twiddle);
        else
            throw std::invalid_argument("cufftdx-resident tile_threads must be 256, 512, or 1024");
    } else {
        if (tile_threads == 256)
            launch_resident_4096<16, cufftdx::fft_direction::forward>(input, output, twiddles, transforms, batch_distance,
                                                                     element_stride, false, recurrence_twiddle);
        else if (tile_threads == 512)
            launch_resident_4096<8, cufftdx::fft_direction::forward>(input, output, twiddles, transforms, batch_distance,
                                                                    element_stride, false, recurrence_twiddle);
        else if (tile_threads == 1024)
            launch_resident_4096<4, cufftdx::fft_direction::forward>(input, output, twiddles, transforms, batch_distance,
                                                                    element_stride, false, recurrence_twiddle);
        else
            throw std::invalid_argument("cufftdx-resident tile_threads must be 256, 512, or 1024");
    }
}

} // namespace cuntt::detail
