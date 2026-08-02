#include <cuda_runtime.h>
#include <cufftdx.hpp>

#include <stdexcept>

#include "external_fft_units.cuh"

namespace cuntt::detail {
namespace {

template <unsigned Size, cufftdx::fft_direction Direction>
using Fp64BlockFft = decltype(cufftdx::Block() + cufftdx::Size<Size>() +
                              cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Direction>() +
                              cufftdx::Precision<double>() + cufftdx::ElementsPerThread<8>() +
                              cufftdx::FFTsPerBlock<1024 / Size>() + cufftdx::SM<700>());

template <unsigned Size, unsigned Threads, unsigned Ept, cufftdx::fft_direction Direction>
using Fp64OnlineFft = decltype(cufftdx::Block() + cufftdx::Size<Size>() +
                               cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Direction>() +
                               cufftdx::Precision<double>() + cufftdx::ElementsPerThread<Ept>() +
                               cufftdx::FFTsPerBlock<(Threads * Ept) / Size>() + cufftdx::SM<700>());

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void fp64_block_kernel(
    const Complex64* input, Complex64* output, std::uint64_t transforms,
    std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    using Value = typename FFT::value_type;
    const std::uint64_t transform = static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const bool active = transform < transforms;
    const std::uint64_t base = transform * batch_distance;
    Value thread_data[FFT::storage_size];
    unsigned index = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && index < FFT::input_length) {
            const Complex64 value = input[base + static_cast<std::uint64_t>(index) * element_stride];
            thread_data[item] = Value{value.real, value.imag};
        } else {
            thread_data[item] = Value{0.0, 0.0};
        }
        index += FFT::stride;
    }
    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    const double scale = normalize ? 1.0 / static_cast<double>(FFT::input_length) : 1.0;
    index = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && index < FFT::output_length) {
            const Value value = thread_data[item];
            output[base + static_cast<std::uint64_t>(index) * element_stride] =
                {value.x * scale, value.y * scale};
        }
        index += FFT::stride;
    }
}

__device__ __forceinline__ Complex64 fp64_cross_root(const Complex64* twiddles, std::uint32_t log_n,
                                                      std::uint64_t exponent) {
    const std::uint32_t n = 1U << log_n;
    const std::uint32_t half = n >> 1;
    const std::uint32_t power = static_cast<std::uint32_t>(exponent) & (n - 1);
    const bool negate = power >= half;
    const Complex64 root = twiddles[half - 1 + (negate ? power - half : power)];
    return negate ? Complex64{-root.real, -root.imag} : root;
}

template <class FFT, bool XorSwizzle>
__device__ __forceinline__ std::uint32_t fp64_tile_index(std::uint32_t element, std::uint32_t slot) {
    static_assert((FFT::ffts_per_block & (FFT::ffts_per_block - 1)) == 0);
    if constexpr (XorSwizzle)
        slot ^= (element >> 1) & (FFT::ffts_per_block - 1);
    return element * FFT::ffts_per_block + slot;
}

template <class FFT, bool XorSwizzle>
__launch_bounds__(FFT::max_threads_per_block) __global__ void fp64_online_first_kernel(
    const Complex64* input, Complex64* scratch, const Complex64* twiddles,
    std::uint64_t transforms, std::uint32_t log_n, std::uint32_t local_log_n,
    std::uint64_t batch_distance, std::uint64_t element_stride, bool recurrence_twiddle) {
    using Value = typename FFT::value_type;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n = 1U << remaining_log_n;
    const std::uint64_t local_transform =
        static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const std::uint64_t total_local = transforms * remaining_n;
    const bool active = local_transform < total_local;
    const std::uint32_t n2 = static_cast<std::uint32_t>(local_transform) & (remaining_n - 1);
    const std::uint32_t flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr std::uint32_t flat_threads = FFT::max_threads_per_block;
    constexpr std::uint32_t elements_per_block_step = flat_threads / FFT::ffts_per_block;
    constexpr std::uint32_t tile_item_step = FFT::stride * FFT::ffts_per_block;
    constexpr std::uint32_t tile_values = FFT::input_length * FFT::ffts_per_block;
    static_assert(flat_threads % FFT::ffts_per_block == 0);
    static_assert(!XorSwizzle || elements_per_block_step % (2 * FFT::ffts_per_block) == 0);
    static_assert(!XorSwizzle || FFT::stride % (2 * FFT::ffts_per_block) == 0);
    extern __shared__ __align__(16) unsigned char storage[];
    Value* tile = reinterpret_cast<Value*>(storage);

    const std::uint32_t first_element = flat_thread / FFT::ffts_per_block;
    const std::uint32_t slot = flat_thread - first_element * FFT::ffts_per_block;
    const std::uint64_t staged_transform =
        static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + slot;
    std::uint32_t stage_index = fp64_tile_index<FFT, XorSwizzle>(first_element, slot);
    if (staged_transform < total_local) {
        const std::uint64_t batch = staged_transform >> remaining_log_n;
        const std::uint32_t staged_n2 = static_cast<std::uint32_t>(staged_transform) & (remaining_n - 1);
        std::uint64_t global_index = batch * batch_distance +
                                     (static_cast<std::uint64_t>(first_element) * remaining_n + staged_n2) * element_stride;
        const std::uint64_t global_step =
            static_cast<std::uint64_t>(elements_per_block_step) * remaining_n * element_stride;
        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const Complex64 value = input[global_index];
            tile[stage_index] = Value{value.real, value.imag};
            stage_index += flat_threads;
            global_index += global_step;
        }
    } else {
        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            tile[stage_index] = Value{0.0, 0.0};
            stage_index += flat_threads;
        }
    }
    __syncthreads();

    Value thread_data[FFT::storage_size];
    unsigned n1 = threadIdx.x;
    std::uint32_t load_index = fp64_tile_index<FFT, XorSwizzle>(n1, threadIdx.y);
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        thread_data[item] = active && n1 < FFT::input_length ? tile[load_index] : Value{0.0, 0.0};
        n1 += FFT::stride;
        load_index += tile_item_step;
    }
    __syncthreads();
    FFT().execute(thread_data, storage);
    __syncthreads();

    Complex64 running_root{1.0, 0.0};
    Complex64 root_step{1.0, 0.0};
    if (active && recurrence_twiddle) {
        running_root = fp64_cross_root(twiddles, log_n, static_cast<std::uint64_t>(threadIdx.x) * n2);
        root_step = fp64_cross_root(twiddles, log_n, static_cast<std::uint64_t>(FFT::stride) * n2);
    }
    unsigned k1 = threadIdx.x;
    std::uint32_t store_index = fp64_tile_index<FFT, XorSwizzle>(k1, threadIdx.y);
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && k1 < FFT::output_length) {
            const Value value = thread_data[item];
            const Complex64 root = recurrence_twiddle
                                       ? running_root
                                       : fp64_cross_root(twiddles, log_n, static_cast<std::uint64_t>(k1) * n2);
            tile[store_index] =
                Value{value.x * root.real - value.y * root.imag,
                      value.x * root.imag + value.y * root.real};
        }
        if (recurrence_twiddle) {
            running_root = {running_root.real * root_step.real - running_root.imag * root_step.imag,
                            running_root.real * root_step.imag + running_root.imag * root_step.real};
        }
        k1 += FFT::stride;
        store_index += tile_item_step;
    }
    __syncthreads();

    if (staged_transform < total_local) {
        const std::uint64_t batch = staged_transform >> remaining_log_n;
        const std::uint32_t staged_n2 = static_cast<std::uint32_t>(staged_transform) & (remaining_n - 1);
        std::uint64_t global_index = batch * batch_distance +
                                     (static_cast<std::uint64_t>(first_element) * remaining_n + staged_n2) * element_stride;
        const std::uint64_t global_step =
            static_cast<std::uint64_t>(elements_per_block_step) * remaining_n * element_stride;
        std::uint32_t export_index = fp64_tile_index<FFT, XorSwizzle>(first_element, slot);
        for (std::uint32_t work = flat_thread; work < tile_values; work += flat_threads) {
            const Value value = tile[export_index];
            scratch[global_index] = {value.x, value.y};
            export_index += flat_threads;
            global_index += global_step;
        }
    }
}

template <class FFT>
__launch_bounds__(FFT::max_threads_per_block) __global__ void fp64_online_second_kernel(
    const Complex64* scratch, Complex64* output, std::uint64_t transforms,
    std::uint32_t log_n, std::uint32_t local_log_n, std::uint64_t batch_distance,
    std::uint64_t element_stride, bool normalize) {
    using Value = typename FFT::value_type;
    const std::uint32_t local_n = 1U << local_log_n;
    const std::uint32_t remaining_log_n = log_n - local_log_n;
    const std::uint32_t remaining_n = 1U << remaining_log_n;
    const std::uint64_t local_transform =
        static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + threadIdx.y;
    const std::uint64_t total_local = transforms * local_n;
    const bool active = local_transform < total_local;
    const std::uint64_t batch = local_transform >> local_log_n;
    const std::uint32_t k1 = static_cast<std::uint32_t>(local_transform) & (local_n - 1);

    Value thread_data[FFT::storage_size];
    unsigned n2 = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && n2 < FFT::input_length) {
            const std::uint64_t logical = static_cast<std::uint64_t>(k1) * remaining_n + n2;
            const Complex64 value = scratch[batch * batch_distance + logical * element_stride];
            thread_data[item] = Value{value.real, value.imag};
        } else {
            thread_data[item] = Value{0.0, 0.0};
        }
        n2 += FFT::stride;
    }
    extern __shared__ __align__(16) unsigned char storage[];
    FFT().execute(thread_data, storage);
    __syncthreads();
    Value* tile = reinterpret_cast<Value*>(storage);
    const double scale = normalize ? 1.0 / static_cast<double>(1U << log_n) : 1.0;
    unsigned k2 = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
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
        const std::uint32_t slot = work - element * FFT::ffts_per_block;
        const std::uint64_t staged_transform =
            static_cast<std::uint64_t>(blockIdx.x) * FFT::ffts_per_block + slot;
        if (staged_transform < total_local) {
            const std::uint64_t staged_batch = staged_transform >> local_log_n;
            const std::uint32_t staged_k1 = static_cast<std::uint32_t>(staged_transform) & (local_n - 1);
            const std::uint64_t logical = static_cast<std::uint64_t>(element) * local_n + staged_k1;
            const Value value = tile[work];
            output[staged_batch * batch_distance + logical * element_stride] = {value.x, value.y};
        }
    }
}

template <unsigned Size, cufftdx::fft_direction Direction>
void launch_fp64_block_impl(const Complex64* input, Complex64* output, std::uint64_t transforms,
                            std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    using FFT = Fp64BlockFft<Size, Direction>;
    if constexpr (FFT::shared_memory_size > 48U * 1024U)
        cudaFuncSetAttribute(fp64_block_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             FFT::shared_memory_size);
    const auto blocks = static_cast<unsigned>((transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    fp64_block_kernel<FFT><<<blocks, FFT::block_dim, FFT::shared_memory_size>>>(
        input, output, transforms, batch_distance, element_stride, normalize);
}

template <cufftdx::fft_direction Direction>
void dispatch_fp64_block(std::uint32_t log_n, const Complex64* input, Complex64* output,
                         std::uint64_t transforms, std::uint64_t batch_distance,
                         std::uint64_t element_stride, bool normalize) {
    switch (log_n) {
        case 3: launch_fp64_block_impl<8, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 4: launch_fp64_block_impl<16, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 5: launch_fp64_block_impl<32, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 6: launch_fp64_block_impl<64, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 7: launch_fp64_block_impl<128, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 8: launch_fp64_block_impl<256, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 9: launch_fp64_block_impl<512, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        case 10: launch_fp64_block_impl<1024, Direction>(input, output, transforms, batch_distance, element_stride, normalize); return;
        default: throw std::invalid_argument("FP64 cufftdx-block supports logN=3..10");
    }
}

template <unsigned Threads, unsigned Ept, cufftdx::fft_direction Direction>
void launch_fp64_online_first(const Complex64* input, Complex64* scratch, const Complex64* twiddles,
                              std::uint64_t transforms, std::uint64_t batch_distance,
                              std::uint64_t element_stride, bool recurrence_twiddle, bool xor_swizzle) {
    using FFT = Fp64OnlineFft<256, Threads, Ept, Direction>;
    constexpr std::size_t tile_bytes = 256 * FFT::ffts_per_block * sizeof(typename FFT::value_type);
    constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    if constexpr (shared_bytes > 48U * 1024U)
        if (xor_swizzle)
            cudaFuncSetAttribute(fp64_online_first_kernel<FFT, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
        else
            cudaFuncSetAttribute(fp64_online_first_kernel<FFT, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    const auto local_transforms = transforms << 8;
    const auto blocks = static_cast<unsigned>((local_transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    if (xor_swizzle)
        fp64_online_first_kernel<FFT, true><<<blocks, FFT::block_dim, shared_bytes>>>(
            input, scratch, twiddles, transforms, 16, 8, batch_distance, element_stride, recurrence_twiddle);
    else
        fp64_online_first_kernel<FFT, false><<<blocks, FFT::block_dim, shared_bytes>>>(
            input, scratch, twiddles, transforms, 16, 8, batch_distance, element_stride, recurrence_twiddle);
}

template <unsigned Threads, unsigned Ept, cufftdx::fft_direction Direction>
void launch_fp64_online_second(const Complex64* scratch, Complex64* output, std::uint64_t transforms,
                               std::uint64_t batch_distance, std::uint64_t element_stride, bool normalize) {
    using FFT = Fp64OnlineFft<256, Threads, Ept, Direction>;
    constexpr std::size_t tile_bytes = 256 * FFT::ffts_per_block * sizeof(typename FFT::value_type);
    constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    if constexpr (shared_bytes > 48U * 1024U)
        cudaFuncSetAttribute(fp64_online_second_kernel<FFT>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    const auto local_transforms = transforms << 8;
    const auto blocks = static_cast<unsigned>((local_transforms + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    fp64_online_second_kernel<FFT><<<blocks, FFT::block_dim, shared_bytes>>>(
        scratch, output, transforms, 16, 8, batch_distance, element_stride, normalize);
}

template <cufftdx::fft_direction Direction, bool First>
void dispatch_fp64_online_point(std::uint32_t threads, std::uint32_t ept,
                                const Complex64* input, Complex64* output, const Complex64* twiddles,
                                std::uint64_t transforms, std::uint64_t batch_distance,
                                std::uint64_t element_stride, bool option, bool xor_swizzle) {
#define CUNTT_FP64_POINT(T, E) \
    case ((E) << 16) | (T): \
        if constexpr (First) launch_fp64_online_first<T, E, Direction>(input, output, twiddles, transforms, batch_distance, element_stride, option, xor_swizzle); \
        else launch_fp64_online_second<T, E, Direction>(input, output, transforms, batch_distance, element_stride, option); \
        return
    switch ((ept << 16) | threads) {
        CUNTT_FP64_POINT(128, 4);
        CUNTT_FP64_POINT(256, 4);
        CUNTT_FP64_POINT(512, 4);
        CUNTT_FP64_POINT(128, 8);
        CUNTT_FP64_POINT(256, 8);
        CUNTT_FP64_POINT(512, 8);
        default: throw std::invalid_argument("unsupported FP64 cuFFTDx thread/EPT point");
    }
#undef CUNTT_FP64_POINT
}

} // namespace

bool cufftdx_fp64_block_available(std::uint32_t log_n) noexcept { return log_n >= 3 && log_n <= 10; }

bool cufftdx_fp64_online_available(std::uint32_t log_n, std::uint32_t threads, std::uint32_t ept) noexcept {
    if (log_n != 8) return false;
    return (threads == 128 || threads == 256 || threads == 512) && (ept == 4 || ept == 8);
}

void launch_cufftdx_fp64_block(std::uint32_t log_n, const Complex64* input, Complex64* output,
                               std::uint64_t transforms, std::uint64_t batch_distance,
                               std::uint64_t element_stride, bool inverse, bool normalize) {
    if (inverse)
        dispatch_fp64_block<cufftdx::fft_direction::inverse>(log_n, input, output, transforms,
                                                             batch_distance, element_stride, normalize);
    else
        dispatch_fp64_block<cufftdx::fft_direction::forward>(log_n, input, output, transforms,
                                                             batch_distance, element_stride, false);
}

void launch_cufftdx_fp64_online_reorder(std::uint32_t log_n, std::uint32_t local_log_n,
                                        const Complex64* input, Complex64* output, Complex64* scratch,
                                        const Complex64* twiddles, std::uint64_t transforms,
                                        std::uint64_t batch_distance, std::uint64_t element_stride,
                                        bool inverse, bool normalize, CrossTwiddleMode cross_twiddle,
                                        SharedLayout shared_layout, std::uint32_t prefix_threads,
                                        std::uint32_t suffix_threads, std::uint32_t prefix_ept,
                                        std::uint32_t suffix_ept) {
    if (log_n != 16 || local_log_n != 8)
        throw std::invalid_argument("FP64 online cuFFTDx currently compiles the logN=16, 8+8 decomposition");
    const bool recurrence_twiddle = cross_twiddle == CrossTwiddleMode::Recurrence;
    const bool xor_swizzle = shared_layout == SharedLayout::XorSwizzle;
    if (inverse) {
        dispatch_fp64_online_point<cufftdx::fft_direction::inverse, true>(
            prefix_threads, prefix_ept, input, scratch, twiddles, transforms,
            batch_distance, element_stride, recurrence_twiddle, xor_swizzle);
        dispatch_fp64_online_point<cufftdx::fft_direction::inverse, false>(
            suffix_threads, suffix_ept, scratch, output, nullptr, transforms,
            batch_distance, element_stride, normalize, false);
    } else {
        dispatch_fp64_online_point<cufftdx::fft_direction::forward, true>(
            prefix_threads, prefix_ept, input, scratch, twiddles, transforms,
            batch_distance, element_stride, recurrence_twiddle, xor_swizzle);
        dispatch_fp64_online_point<cufftdx::fft_direction::forward, false>(
            suffix_threads, suffix_ept, scratch, output, nullptr, transforms,
            batch_distance, element_stride, false, false);
    }
}

} // namespace cuntt::detail
