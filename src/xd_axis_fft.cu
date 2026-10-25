#include "xd_axis_fft.hpp"

#include "cuntt/butterfly.hpp"

#include <cuda_runtime.h>
#include <cufftdx.hpp>

#include <cstdint>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#ifndef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_CUFFTDX_SM 700
#endif

namespace cuntt::detail {
namespace {

template <typename Real>
struct global_complex;

template <>
struct global_complex<float> {
    using type = Complex32;
};

template <>
struct global_complex<double> {
    using type = Complex64;
};

template <typename Real>
using global_complex_t = typename global_complex<Real>::type;

template <typename Real, unsigned Length, unsigned RequestedEpt,
          unsigned FftsPerBlockCap = 0, unsigned ExactFftsPerBlock = 0>
struct axis_configuration {
    // Keep a bounded number of independent lines per CTA while retaining a
    // 256-thread packing target for short axes; long axes use one line.
    static constexpr unsigned elements_per_thread = xd_axis_fft_ept(Length, RequestedEpt);
    static constexpr unsigned ffts_per_block =
        xd_axis_fft_block_batch(Length, sizeof(Real) == sizeof(double), RequestedEpt,
                                FftsPerBlockCap, ExactFftsPerBlock);
    static_assert(FftsPerBlockCap == 0 || FftsPerBlockCap == 8 ||
                      FftsPerBlockCap == 16,
                  "xD axis FFTsPerBlock cap must be zero, 8 or 16");
    static_assert((Length % elements_per_thread) == 0,
                  "xD axis length must be divisible by the fixed EPT");
    static_assert(ffts_per_block != 0 && !(ffts_per_block & (ffts_per_block - 1)),
                  "xD FFTsPerBlock must be a power of two");
};

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap = 0,
          unsigned ExactFftsPerBlock = 0>
using AxisFft = decltype(
    cufftdx::Block() + cufftdx::Size<Length>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Direction>() +
    cufftdx::Precision<Real>() +
    cufftdx::ElementsPerThread<axis_configuration<Real, Length, RequestedEpt,
                                                   FftsPerBlockCap, ExactFftsPerBlock>::elements_per_thread>() +
    cufftdx::FFTsPerBlock<axis_configuration<Real, Length, RequestedEpt,
                                             FftsPerBlockCap, ExactFftsPerBlock>::ffts_per_block>() +
    cufftdx::SM<CUBUTTERFLY_XD_CUFFTDX_SM>());

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap, bool Staged,
          unsigned ExactFftsPerBlock = 0>
__launch_bounds__(AxisFft<Real, Length, Direction, RequestedEpt,
                           FftsPerBlockCap, ExactFftsPerBlock>::max_threads_per_block)
__global__ void xd_axis_kernel(const global_complex_t<Real>* input,
                               global_complex_t<Real>* output,
                               std::uint64_t suffix, std::uint64_t lines,
                               unsigned suffix_shift, Real scale) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>;
    using GlobalComplex = global_complex_t<Real>;
    using Value = typename FFT::value_type;

    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>::ffts_per_block;
    constexpr unsigned TilePitch = Length + 1;
    constexpr unsigned TileValues = FftsPerBlock * Length;
    const std::uint64_t first_line =
        static_cast<std::uint64_t>(blockIdx.x) * FftsPerBlock;
    const unsigned flat_thread = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned flat_threads = blockDim.x * blockDim.y;

    extern __shared__ __align__(16) unsigned char storage[];

    if constexpr (Staged) {
        // Write axis-major work into line-major shared rows.  Consecutive
        // slots are adjacent in the natural tensor for each axis element,
        // while the padded pitch keeps the FFT reads independent of bank
        // alignment.  Tail lines are zero-filled but still execute the core.
        auto* tile = reinterpret_cast<Value*>(storage);
        for (unsigned work = flat_thread; work < TileValues; work += flat_threads) {
            const unsigned axis = static_cast<unsigned>(work / FftsPerBlock);
            const unsigned slot = static_cast<unsigned>(work % FftsPerBlock);
            const std::uint64_t line = first_line + slot;
            Value value{0, 0};
            if (line < lines) {
                const std::uint64_t outer = suffix_shift < 64
                    ? line >> suffix_shift : line / suffix;
                const std::uint64_t base = line + outer * (Length - 1U) * suffix;
                const GlobalComplex source = input[base +
                    static_cast<std::uint64_t>(axis) * suffix];
                value = Value{source.real, source.imag};
            }
            tile[static_cast<std::size_t>(slot) * TilePitch + axis] = value;
        }
        __syncthreads();

        Value thread_data[FFT::storage_size];
        unsigned axis = threadIdx.x;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            thread_data[item] = axis < Length
                                    ? tile[static_cast<std::size_t>(threadIdx.y) * TilePitch + axis]
                                    : Value{0, 0};
            axis += FFT::stride;
        }
        // Release the staging tile before cuFFTDx aliases the same shared
        // allocation as its work area.  Every CTA thread reaches execute.
        __syncthreads();
        FFT().execute(thread_data, storage);
        __syncthreads();

        const std::uint64_t line = first_line + threadIdx.y;
        const bool active = line < lines;
        axis = threadIdx.x;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            if (active && axis < Length) {
                const Value value = thread_data[item];
                tile[static_cast<std::size_t>(threadIdx.y) * TilePitch + axis] =
                    Value{value.x * scale, value.y * scale};
            }
            axis += FFT::stride;
        }
        __syncthreads();

        // Export with the same axis-major/slot-minor traversal used by the
        // load.  This is the single natural-order store for every line.
        for (unsigned work = flat_thread; work < TileValues; work += flat_threads) {
            const unsigned axis_index = static_cast<unsigned>(work / FftsPerBlock);
            const unsigned slot = static_cast<unsigned>(work % FftsPerBlock);
            const std::uint64_t output_line = first_line + slot;
            if (output_line < lines) {
                const std::uint64_t outer = suffix_shift < 64
                    ? output_line >> suffix_shift : output_line / suffix;
                const std::uint64_t base = output_line + outer * (Length - 1U) * suffix;
                const Value value = tile[static_cast<std::size_t>(slot) * TilePitch + axis_index];
                output[base + static_cast<std::uint64_t>(axis_index) * suffix] =
                    GlobalComplex{value.x, value.y};
            }
        }
        return;
    }

    // For a contiguous axis the imported core can read and write registers
    // directly.  The inactive tail slots remain convergent and are discarded
    // only after the FFT has completed.
    const std::uint64_t line = first_line + threadIdx.y;
    const bool active = line < lines;
    const std::uint64_t base = active
                                   ? (line * static_cast<std::uint64_t>(Length))
                                   : 0;
    Value thread_data[FFT::storage_size];
    unsigned axis = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && axis < Length) {
            const GlobalComplex source = input[base + axis];
            thread_data[item] = Value{source.real, source.imag};
        } else {
            thread_data[item] = Value{0, 0};
        }
        axis += FFT::stride;
    }
    // Registers are private and the core owns its internal synchronization.
    // No external shared tile is published or reused on this path.
    FFT().execute(thread_data, storage);

    axis = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && axis < Length) {
            const Value value = thread_data[item];
            output[base + axis] = GlobalComplex{value.x * scale, value.y * scale};
        }
        axis += FFT::stride;
    }
}

// Experimental direct exchange for a non-contiguous axis.  The cuFFTDx block
// core still uses its private register/shared workspace, but no caller-owned
// staging tile is used.  Each CTA thread follows the same line/axis mapping as
// the contiguous kernel and only the global element stride changes.
template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap>
__launch_bounds__(AxisFft<Real, Length, Direction, RequestedEpt,
                           FftsPerBlockCap>::max_threads_per_block)
__global__ void xd_axis_direct_kernel(const global_complex_t<Real>* input,
                                      global_complex_t<Real>* output,
                                      std::uint64_t suffix, std::uint64_t lines,
                                      unsigned suffix_shift, Real scale) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap>;
    using GlobalComplex = global_complex_t<Real>;
    using Value = typename FFT::value_type;

    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap>::ffts_per_block;
    const std::uint64_t line =
        static_cast<std::uint64_t>(blockIdx.x) * FftsPerBlock + threadIdx.y;
    const bool active = line < lines;
    const std::uint64_t logical_line = active ? line : 0;
    const std::uint64_t outer = suffix_shift < 64
                                    ? logical_line >> suffix_shift
                                    : logical_line / suffix;
    const std::uint64_t base = logical_line +
                               outer * (Length - 1U) * suffix;

    extern __shared__ __align__(16) unsigned char storage[];
    Value thread_data[FFT::storage_size];
    unsigned axis = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && axis < Length) {
            const GlobalComplex source = input[
                base + static_cast<std::uint64_t>(axis) * suffix];
            thread_data[item] = Value{source.real, source.imag};
        } else {
            thread_data[item] = Value{0, 0};
        }
        axis += FFT::stride;
    }
    FFT().execute(thread_data, storage);

    axis = threadIdx.x;
#pragma unroll
    for (unsigned item = 0; item < FFT::storage_size; ++item) {
        if (active && axis < Length) {
            const Value value = thread_data[item];
            output[base + static_cast<std::uint64_t>(axis) * suffix] =
                GlobalComplex{value.x * scale, value.y * scale};
        }
        axis += FFT::stride;
    }
}

// Experimental compile-time-unrolled variant of the staged exchange.  The
// global tile traversal uses a flattened compile-time block size; because the
// flat thread id advances by the full block, each thread retains one tile slot
// while its axis advances by FFT::stride.  The compile-time contract below
// prevents a changed MathDx block layout from silently changing the mapping.
template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap,
          typename Index = std::uint64_t, unsigned ExactFftsPerBlock = 0>
__launch_bounds__(AxisFft<Real, Length, Direction, RequestedEpt,
                           FftsPerBlockCap, ExactFftsPerBlock>::max_threads_per_block)
__global__ void xd_axis_kernel_unrolled(const global_complex_t<Real>* input,
                                        global_complex_t<Real>* output,
                                        Index suffix, Index lines,
                                        unsigned suffix_shift, Real scale) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>;
    using GlobalComplex = global_complex_t<Real>;
    using Value = typename FFT::value_type;

    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>::ffts_per_block;
    constexpr unsigned BlockX = FFT::block_dim.x;
    constexpr unsigned BlockY = FFT::block_dim.y;
    constexpr unsigned BlockZ = FFT::block_dim.z;
    constexpr unsigned FlatThreads = BlockX * BlockY * BlockZ;
    constexpr unsigned TilePitch = Length + 1;
    constexpr unsigned IndexBits = sizeof(Index) * 8U;
    constexpr unsigned AxisLength = Length;
    constexpr unsigned Stride = FFT::stride;
    constexpr bool FlatMappingValid =
        BlockZ == 1 && BlockY == FftsPerBlock &&
        (FlatThreads % FftsPerBlock) == 0 &&
        (FlatThreads / FftsPerBlock) == FFT::stride &&
        (FFT::storage_size * FFT::stride) >= Length;
    static_assert(FlatMappingValid,
                  "xD unrolled axis mapping requires a flat FFTsPerBlock layout");

    extern __shared__ __align__(16) unsigned char storage[];
    if constexpr (FlatMappingValid) {
        auto* tile = reinterpret_cast<Value*>(storage);
        const unsigned flat_thread = threadIdx.y * BlockX + threadIdx.x;
        const unsigned slot = flat_thread % FftsPerBlock;
        const Index first_line = static_cast<Index>(blockIdx.x) *
                                 static_cast<Index>(FftsPerBlock);
        const Index line = first_line + static_cast<Index>(slot);
        const bool active = line < lines;
        const Index logical_line = active ? line : 0;
        const Index outer = suffix_shift < IndexBits
                                        ? logical_line >> suffix_shift
                                        : logical_line / suffix;
        const Index base = logical_line +
                           outer * static_cast<Index>(Length - 1U) * suffix;

        unsigned axis = flat_thread / FftsPerBlock;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            Value value{0, 0};
            if (active && axis < AxisLength) {
                const GlobalComplex source = input[
                    base + static_cast<Index>(axis) * suffix];
                value = Value{source.real, source.imag};
            }
            if (axis < AxisLength) {
                const std::size_t tile_offset = static_cast<std::size_t>(slot) *
                                                TilePitch;
                tile[tile_offset + axis] = value;
            }
            axis += Stride;
        }
        __syncthreads();

        Value thread_data[FFT::storage_size];
        axis = threadIdx.x;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            thread_data[item] = axis < AxisLength
                                    ? tile[static_cast<std::size_t>(threadIdx.y) *
                                               TilePitch + axis]
                                    : Value{0, 0};
            axis += Stride;
        }
        __syncthreads();
        FFT().execute(thread_data, storage);
        __syncthreads();

        axis = threadIdx.x;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            if (axis < AxisLength) {
                const Value value = thread_data[item];
                tile[static_cast<std::size_t>(threadIdx.y) * TilePitch + axis] =
                    Value{value.x * scale, value.y * scale};
            }
            axis += Stride;
        }
        __syncthreads();

        axis = flat_thread / FftsPerBlock;
#pragma unroll
        for (unsigned item = 0; item < FFT::storage_size; ++item) {
            if (active && axis < AxisLength) {
                const std::size_t tile_offset = static_cast<std::size_t>(slot) *
                                                TilePitch;
                const Value value = tile[tile_offset + axis];
                output[base + static_cast<Index>(axis) * suffix] =
                    GlobalComplex{value.x, value.y};
            }
            axis += Stride;
        }
    }
}

inline bool valid_length(std::size_t length) noexcept {
    return xd_axis_fft_length_supported(length);
}

inline bool compiled_sm_matches_device() noexcept {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return false;
    const int runtime_sm = properties.major * 100 + properties.minor * 10;
    return runtime_sm == CUBUTTERFLY_XD_CUFFTDX_SM;
}

template <typename Kernel>
void configure_axis_shared_memory(Kernel kernel, std::size_t shared_bytes) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        throw std::runtime_error("query xD axis cuFFTDx device");
    // Resource attributes and opt-in are initialization work. Cache by kernel
    // and device so repeated plan execution only checks the current device.
    static thread_local std::unordered_map<const void*, std::pair<int, std::size_t>> ready;
    const auto key = reinterpret_cast<const void*>(kernel);
    const auto found = ready.find(key);
    if (found != ready.end() && found->second.first == device &&
        found->second.second >= shared_bytes)
        return;
    cudaDeviceProp properties{};
    cudaFuncAttributes attributes{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
        throw std::runtime_error("query xD axis cuFFTDx shared-memory resources");
    if (shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlockOptin)
        throw std::invalid_argument("xD axis cuFFTDx mapping exceeds device shared-memory budget");
    if (shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlock) {
        const auto status = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("opt in xD axis cuFFTDx shared memory: ") +
                                     cudaGetErrorString(status));
    }
    ready[key] = {device, shared_bytes};
}

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap, unsigned ExactFftsPerBlock = 0>
void launch_length_impl(const global_complex_t<Real>* input,
                        global_complex_t<Real>* output, std::size_t suffix,
                        std::size_t lines, Real scale, cudaStream_t stream) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>;
    using Value = typename FFT::value_type;
    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>::ffts_per_block;
    constexpr std::size_t tile_bytes =
        static_cast<std::size_t>(FftsPerBlock) * (Length + 1) * sizeof(Value);
    constexpr std::size_t shared_bytes =
        FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    static_assert(FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z <= 1024,
                  "xD axis cuFFTDx mapping exceeds CUDA block thread limit");
    const auto blocks = (static_cast<std::uint64_t>(lines) + FftsPerBlock - 1) /
                        FftsPerBlock;
    if (blocks > 0x7fffffffULL)
        throw std::invalid_argument("xD axis cuFFTDx grid exceeds CUDA grid.x");
    // Power-of-two tensor strides need only an offset and shift. Keep the
    // general division path for mixed/odd shapes, including compact spectra.
    unsigned suffix_shift = 64;
    if ((suffix & (suffix - 1)) == 0) {
        suffix_shift = 0;
        for (auto remaining = suffix; remaining > 1; remaining >>= 1)
            ++suffix_shift;
    }
    if (suffix == 1) {
        configure_axis_shared_memory(
            xd_axis_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, false, ExactFftsPerBlock>,
            FFT::shared_memory_size);
        xd_axis_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, false, ExactFftsPerBlock>
            <<<static_cast<unsigned>(blocks), FFT::block_dim, FFT::shared_memory_size, stream>>>(
                input, output, suffix, static_cast<std::uint64_t>(lines), suffix_shift, scale);
    } else {
        configure_axis_shared_memory(
            xd_axis_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, true, ExactFftsPerBlock>,
            shared_bytes);
        xd_axis_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, true, ExactFftsPerBlock>
            <<<static_cast<unsigned>(blocks), FFT::block_dim, shared_bytes, stream>>>(
                input, output, suffix, static_cast<std::uint64_t>(lines), suffix_shift, scale);
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD axis cuFFTDx: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap>
void launch_length_direct_impl(const global_complex_t<Real>* input,
                               global_complex_t<Real>* output, std::size_t suffix,
                               std::size_t lines, Real scale, cudaStream_t stream) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap>;
    static_assert(FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z <= 1024,
                  "xD axis cuFFTDx mapping exceeds CUDA block thread limit");
    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap>::ffts_per_block;
    if (suffix == 1) {
        // Keep the established contiguous path byte-for-byte selected for the
        // unit-stride case; policy 1 only changes non-contiguous exchange.
        launch_length_impl<Real, Length, Direction, RequestedEpt, FftsPerBlockCap>(
            input, output, suffix, lines, scale, stream);
        return;
    }
    const auto blocks = (static_cast<std::uint64_t>(lines) + FftsPerBlock - 1) /
                        FftsPerBlock;
    if (blocks > 0x7fffffffULL)
        throw std::invalid_argument("xD axis cuFFTDx grid exceeds CUDA grid.x");
    unsigned suffix_shift = 64;
    if ((suffix & (suffix - 1)) == 0) {
        suffix_shift = 0;
        for (auto remaining = suffix; remaining > 1; remaining >>= 1)
            ++suffix_shift;
    }
    configure_axis_shared_memory(
        xd_axis_direct_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap>,
        FFT::shared_memory_size);
    xd_axis_direct_kernel<Real, Length, Direction, RequestedEpt, FftsPerBlockCap>
        <<<static_cast<unsigned>(blocks), FFT::block_dim, FFT::shared_memory_size, stream>>>(
            input, output, suffix, static_cast<std::uint64_t>(lines), suffix_shift, scale);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD direct axis cuFFTDx: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap,
          typename Index = std::uint64_t, unsigned ExactFftsPerBlock = 0>
void launch_length_unrolled_impl(const global_complex_t<Real>* input,
                                 global_complex_t<Real>* output, std::size_t suffix,
                                 std::size_t lines, Real scale, cudaStream_t stream) {
    using FFT = AxisFft<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>;
    using Value = typename FFT::value_type;
    constexpr unsigned FftsPerBlock =
        axis_configuration<Real, Length, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>::ffts_per_block;
    constexpr std::size_t tile_bytes =
        static_cast<std::size_t>(FftsPerBlock) * (Length + 1) * sizeof(Value);
    constexpr std::size_t shared_bytes =
        FFT::shared_memory_size > tile_bytes ? FFT::shared_memory_size : tile_bytes;
    constexpr unsigned flat_threads =
        FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z;
    constexpr bool flat_mapping_valid =
        FFT::block_dim.z == 1 && FFT::block_dim.y == FftsPerBlock &&
        (flat_threads % FftsPerBlock) == 0 &&
        (flat_threads / FftsPerBlock) == FFT::stride &&
        (FFT::storage_size * FFT::stride) >= Length;
    static_assert(flat_mapping_valid,
                  "xD unrolled axis mapping requires a flat FFTsPerBlock layout");
    static_assert(flat_threads <= 1024,
                  "xD axis cuFFTDx mapping exceeds CUDA block thread limit");
    if (suffix == 1) {
        launch_length_impl<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, ExactFftsPerBlock>(
            input, output, suffix, lines, scale, stream);
        return;
    }
    if constexpr (sizeof(Index) == sizeof(std::uint32_t)) {
        if (!xd_axis_fft_u32_index_fits(Length, suffix, lines))
            throw std::invalid_argument(
                "xD axis uint32 exchange requires length*lines and suffix <= UINT32_MAX");
    }
    const auto blocks = (static_cast<std::uint64_t>(lines) + FftsPerBlock - 1) /
                        FftsPerBlock;
    if (blocks > 0x7fffffffULL)
        throw std::invalid_argument("xD axis cuFFTDx grid exceeds CUDA grid.x");
    unsigned suffix_shift = 64;
    if ((suffix & (suffix - 1)) == 0) {
        suffix_shift = 0;
        for (auto remaining = suffix; remaining > 1; remaining >>= 1)
            ++suffix_shift;
    }
    configure_axis_shared_memory(
        xd_axis_kernel_unrolled<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, Index, ExactFftsPerBlock>,
        shared_bytes);
    xd_axis_kernel_unrolled<Real, Length, Direction, RequestedEpt, FftsPerBlockCap, Index, ExactFftsPerBlock>
        <<<static_cast<unsigned>(blocks), FFT::block_dim, shared_bytes, stream>>>(
            input, output, static_cast<Index>(suffix), static_cast<Index>(lines),
            suffix_shift, scale);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD unrolled axis cuFFTDx: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, cufftdx::fft_direction Direction, unsigned RequestedEpt,
          unsigned FftsPerBlockCap>
void dispatch_length(std::size_t length, const global_complex_t<Real>* input,
                     global_complex_t<Real>* output, std::size_t suffix,
                     std::size_t lines, Real scale, cudaStream_t stream) {
#define CUNTT_XD_AXIS_CASE(N) \
    case N: \
        if constexpr (xd_axis_fft_configuration_supported( \
                          N, sizeof(Real) == sizeof(double), RequestedEpt, FftsPerBlockCap)) { \
        launch_length_impl<Real, N, Direction, RequestedEpt, \
            ((FftsPerBlockCap != 0 && \
             FftsPerBlockCap < xd_axis_fft_block_batch( \
                 N, sizeof(Real) == sizeof(double), RequestedEpt)) \
                ? FftsPerBlockCap : 0)>(input, output, suffix, lines, scale, stream); \
        return; \
        } else { \
            throw std::invalid_argument("xD axis cuFFTDx configuration exceeds resource budget"); \
        }
    switch (length) {
        CUNTT_XD_AXIS_CASE(8);
        CUNTT_XD_AXIS_CASE(16);
        CUNTT_XD_AXIS_CASE(32);
        CUNTT_XD_AXIS_CASE(64);
        CUNTT_XD_AXIS_CASE(128);
        CUNTT_XD_AXIS_CASE(256);
        CUNTT_XD_AXIS_CASE(512);
        CUNTT_XD_AXIS_CASE(1024);
        CUNTT_XD_AXIS_CASE(2048);
        CUNTT_XD_AXIS_CASE(4096);
        default:
            throw std::invalid_argument("xD axis cuFFTDx supports lengths 8..4096 power-of-two");
    }
#undef CUNTT_XD_AXIS_CASE
}

template <typename Real, cufftdx::fft_direction Direction, unsigned RequestedEpt,
          unsigned FftsPerBlockCap>
void dispatch_length_direct(std::size_t length, const global_complex_t<Real>* input,
                            global_complex_t<Real>* output, std::size_t suffix,
                            std::size_t lines, Real scale, cudaStream_t stream) {
#define CUNTT_XD_AXIS_DIRECT_CASE(N) \
    case N: \
        if constexpr (xd_axis_fft_configuration_supported( \
                          N, sizeof(Real) == sizeof(double), RequestedEpt, FftsPerBlockCap)) { \
        launch_length_direct_impl<Real, N, Direction, RequestedEpt, \
            ((FftsPerBlockCap != 0 && \
             FftsPerBlockCap < xd_axis_fft_block_batch( \
                 N, sizeof(Real) == sizeof(double), RequestedEpt)) \
                ? FftsPerBlockCap : 0)>(input, output, suffix, lines, scale, stream); \
        return; \
        } else { \
            throw std::invalid_argument("xD axis cuFFTDx configuration exceeds resource budget"); \
        }
    switch (length) {
        CUNTT_XD_AXIS_DIRECT_CASE(8);
        CUNTT_XD_AXIS_DIRECT_CASE(16);
        CUNTT_XD_AXIS_DIRECT_CASE(32);
        CUNTT_XD_AXIS_DIRECT_CASE(64);
        CUNTT_XD_AXIS_DIRECT_CASE(128);
        CUNTT_XD_AXIS_DIRECT_CASE(256);
        CUNTT_XD_AXIS_DIRECT_CASE(512);
        CUNTT_XD_AXIS_DIRECT_CASE(1024);
        CUNTT_XD_AXIS_DIRECT_CASE(2048);
        CUNTT_XD_AXIS_DIRECT_CASE(4096);
        default:
            throw std::invalid_argument("xD axis cuFFTDx supports lengths 8..4096 power-of-two");
    }
#undef CUNTT_XD_AXIS_DIRECT_CASE
}

template <typename Real, cufftdx::fft_direction Direction, unsigned RequestedEpt,
          unsigned FftsPerBlockCap, typename Index = std::uint64_t>
void dispatch_length_unrolled(std::size_t length, const global_complex_t<Real>* input,
                              global_complex_t<Real>* output, std::size_t suffix,
                              std::size_t lines, Real scale, cudaStream_t stream) {
#define CUNTT_XD_AXIS_UNROLLED_CASE(N) \
    case N: \
        if constexpr (xd_axis_fft_configuration_supported( \
                          N, sizeof(Real) == sizeof(double), RequestedEpt, FftsPerBlockCap)) { \
        launch_length_unrolled_impl<Real, N, Direction, RequestedEpt, \
            ((FftsPerBlockCap != 0 && \
             FftsPerBlockCap < xd_axis_fft_block_batch( \
                 N, sizeof(Real) == sizeof(double), RequestedEpt)) \
                ? FftsPerBlockCap : 0), Index>(input, output, suffix, lines, scale, stream); \
        return; \
        } else { \
            throw std::invalid_argument("xD axis cuFFTDx configuration exceeds resource budget"); \
        }
    switch (length) {
        CUNTT_XD_AXIS_UNROLLED_CASE(8);
        CUNTT_XD_AXIS_UNROLLED_CASE(16);
        CUNTT_XD_AXIS_UNROLLED_CASE(32);
        CUNTT_XD_AXIS_UNROLLED_CASE(64);
        CUNTT_XD_AXIS_UNROLLED_CASE(128);
        CUNTT_XD_AXIS_UNROLLED_CASE(256);
        CUNTT_XD_AXIS_UNROLLED_CASE(512);
        CUNTT_XD_AXIS_UNROLLED_CASE(1024);
        CUNTT_XD_AXIS_UNROLLED_CASE(2048);
        CUNTT_XD_AXIS_UNROLLED_CASE(4096);
        default:
            throw std::invalid_argument("xD axis cuFFTDx supports lengths 8..4096 power-of-two");
    }
#undef CUNTT_XD_AXIS_UNROLLED_CASE
}

template <typename Real, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned FftsPerBlockCap,
          unsigned ExchangePolicy>
void dispatch_policy_length(std::size_t length, const global_complex_t<Real>* input,
                            global_complex_t<Real>* output, std::size_t suffix,
                            std::size_t lines, Real scale, cudaStream_t stream) {
    static_assert(ExchangePolicy <= 3, "invalid xD axis exchange policy");
    if constexpr (ExchangePolicy == 0) {
        dispatch_length<Real, Direction, RequestedEpt, FftsPerBlockCap>(
            length, input, output, suffix, lines, scale, stream);
    } else if constexpr (ExchangePolicy == 1) {
        dispatch_length_direct<Real, Direction, RequestedEpt, FftsPerBlockCap>(
            length, input, output, suffix, lines, scale, stream);
    } else if constexpr (ExchangePolicy == 2) {
        dispatch_length_unrolled<Real, Direction, RequestedEpt, FftsPerBlockCap>(
            length, input, output, suffix, lines, scale, stream);
    } else {
        dispatch_length_unrolled<Real, Direction, RequestedEpt, FftsPerBlockCap,
                                std::uint32_t>(length, input, output, suffix, lines,
                                               scale, stream);
    }
}

template <typename Real, cufftdx::fft_direction Direction,
          unsigned RequestedEpt, unsigned ExchangePolicy>
void dispatch_ept(std::size_t length, const global_complex_t<Real>* input,
                  global_complex_t<Real>* output, std::size_t suffix,
                  std::size_t lines, Real scale, cudaStream_t stream,
                  unsigned ffts_per_block_cap) {
    switch (ffts_per_block_cap) {
        case 0:
            dispatch_policy_length<Real, Direction, RequestedEpt, 0, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream);
            return;
        case 8:
            dispatch_policy_length<Real, Direction, RequestedEpt, 8, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream);
            return;
        case 16:
            dispatch_policy_length<Real, Direction, RequestedEpt, 16, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream);
            return;
        default:
            throw std::invalid_argument(
                "xD axis cuFFTDx FFTsPerBlock cap must be zero, 8 or 16");
    }
}

template <typename Real, cufftdx::fft_direction Direction,
          unsigned ExchangePolicy>
void dispatch_configuration_policy(std::size_t length,
                                   const global_complex_t<Real>* input,
                                   global_complex_t<Real>* output,
                                   std::size_t suffix, std::size_t lines,
                                   Real scale, cudaStream_t stream,
                                   unsigned elements_per_thread,
                                   unsigned ffts_per_block_cap) {
    switch (elements_per_thread) {
        case 4:
            dispatch_ept<Real, Direction, 4, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream,
                ffts_per_block_cap);
            return;
        case 8:
            dispatch_ept<Real, Direction, 8, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream,
                ffts_per_block_cap);
            return;
        case 16:
            dispatch_ept<Real, Direction, 16, ExchangePolicy>(
                length, input, output, suffix, lines, scale, stream,
                ffts_per_block_cap);
            return;
        default:
            throw std::invalid_argument("xD axis cuFFTDx supports EPT 4, 8 or 16");
    }
}

template <typename Real, cufftdx::fft_direction Direction>
void dispatch_configuration(std::size_t length, const global_complex_t<Real>* input,
                            global_complex_t<Real>* output, std::size_t suffix,
                            std::size_t lines, Real scale, cudaStream_t stream,
                            unsigned elements_per_thread,
                            unsigned ffts_per_block_cap,
                            unsigned exchange_policy) {
    if (ffts_per_block_cap != 0 && ffts_per_block_cap != 8 &&
        ffts_per_block_cap != 16)
        throw std::invalid_argument(
            "xD axis cuFFTDx FFTsPerBlock cap must be zero, 8 or 16");
    switch (exchange_policy) {
        case 0:
            dispatch_configuration_policy<Real, Direction, 0>(
                length, input, output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap);
            return;
        case 1:
            dispatch_configuration_policy<Real, Direction, 1>(
                length, input, output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap);
            return;
        case 2:
            dispatch_configuration_policy<Real, Direction, 2>(
                length, input, output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap);
            return;
        case 3:
            dispatch_configuration_policy<Real, Direction, 3>(
                length, input, output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap);
            return;
        default:
            throw std::invalid_argument("xD axis cuFFTDx exchange policy must be 0, 1, 2 or 3");
    }
}

template <typename Real, unsigned Length, cufftdx::fft_direction Direction,
          unsigned Ept, unsigned Group>
bool grouped_length(const global_complex_t<Real>* input, global_complex_t<Real>* output,
                    std::size_t suffix, std::size_t lines, Real scale,
                    cudaStream_t stream, unsigned exchange, bool query) {
    using FFT = AxisFft<Real, Length, Direction, Ept, 0, Group>;
    constexpr std::size_t tile_bytes = Group * (Length + 1ULL) * sizeof(typename FFT::value_type);
    constexpr std::size_t shared_bytes = FFT::shared_memory_size > tile_bytes
                                            ? FFT::shared_memory_size : tile_bytes;
    constexpr unsigned threads = FFT::block_dim.x * FFT::block_dim.y * FFT::block_dim.z;
    cudaFuncAttributes attributes{};
    cudaError_t status = cudaErrorInvalidValue;
    if (exchange == 0)
        status = cudaFuncGetAttributes(&attributes,
            xd_axis_kernel<Real, Length, Direction, Ept, 0, true, Group>);
    else if (exchange == 2)
        status = cudaFuncGetAttributes(&attributes,
            xd_axis_kernel_unrolled<Real, Length, Direction, Ept, 0, std::uint64_t, Group>);
    else if (exchange == 3)
        status = cudaFuncGetAttributes(&attributes,
            xd_axis_kernel_unrolled<Real, Length, Direction, Ept, 0, std::uint32_t, Group>);
    int device = 0;
    cudaDeviceProp properties{};
    if (status != cudaSuccess || cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        threads > static_cast<unsigned>(properties.maxThreadsPerBlock) ||
        threads > static_cast<unsigned>(attributes.maxThreadsPerBlock) ||
        shared_bytes + attributes.sharedSizeBytes > properties.sharedMemPerBlockOptin)
        return false;
    if (query) return true;
    if (exchange == 0)
        launch_length_impl<Real, Length, Direction, Ept, 0, Group>(
            input, output, suffix, lines, scale, stream);
    else if (exchange == 2)
        launch_length_unrolled_impl<Real, Length, Direction, Ept, 0, std::uint64_t, Group>(
            input, output, suffix, lines, scale, stream);
    else
        launch_length_unrolled_impl<Real, Length, Direction, Ept, 0, std::uint32_t, Group>(
            input, output, suffix, lines, scale, stream);
    return true;
}

template <typename Real, cufftdx::fft_direction Direction, unsigned Ept, unsigned Group>
bool dispatch_grouped_length(std::size_t length, const global_complex_t<Real>* input,
                            global_complex_t<Real>* output, std::size_t suffix,
                            std::size_t lines, Real scale, cudaStream_t stream,
                            unsigned exchange, bool query) {
    if (length == 2048)
        return grouped_length<Real, 2048, Direction, Ept, Group>(
            input, output, suffix, lines, scale, stream, exchange, query);
    if constexpr (xd_axis_fft_configuration_supported(4096, sizeof(Real) == sizeof(double), Ept, 0, Group)) {
        if (length == 4096)
            return grouped_length<Real, 4096, Direction, Ept, Group>(
                input, output, suffix, lines, scale, stream, exchange, query);
    }
    return false;
}

template <typename Real, cufftdx::fft_direction Direction>
bool dispatch_grouped_configuration(std::size_t length, const global_complex_t<Real>* input,
                                   global_complex_t<Real>* output, std::size_t suffix,
                                   std::size_t lines, Real scale, cudaStream_t stream,
                                   unsigned ept, unsigned group, unsigned exchange, bool query) {
    if (exchange != 0 && exchange != 2 && exchange != 3) return false;
#define CUNTT_XD_GROUP(E, G) \
    if (ept == E && group == G) \
        return dispatch_grouped_length<Real, Direction, E, G>( \
            length, input, output, suffix, lines, scale, stream, exchange, query)
    CUNTT_XD_GROUP(8, 2);
    CUNTT_XD_GROUP(8, 4);
    CUNTT_XD_GROUP(16, 2);
    CUNTT_XD_GROUP(16, 4);
#undef CUNTT_XD_GROUP
    return false;
}

template <typename Real>
bool grouped_available(std::size_t length, unsigned ept, unsigned group, unsigned exchange) {
    return dispatch_grouped_configuration<Real, cufftdx::fft_direction::forward>(
               length, nullptr, nullptr, 2, 0, Real{1}, nullptr, ept, group, exchange, true) &&
           dispatch_grouped_configuration<Real, cufftdx::fft_direction::inverse>(
               length, nullptr, nullptr, 2, 0, Real{1}, nullptr, ept, group, exchange, true);
}

}  // namespace

bool xd_axis_fft_available(std::size_t length) noexcept {
    return valid_length(length) && compiled_sm_matches_device();
}

bool xd_axis_fft_exact_configuration_available(std::size_t length, bool fp64,
    unsigned requested, unsigned exact_ffts_per_block, unsigned exchange_policy) noexcept {
    if (!xd_axis_fft_configuration_supported(length, fp64, requested, 0, exact_ffts_per_block) ||
        exact_ffts_per_block == 0 || !xd_axis_fft_available(length)) return false;
    return fp64 ? grouped_available<double>(length, requested, exact_ffts_per_block, exchange_policy)
                : grouped_available<float>(length, requested, exact_ffts_per_block, exchange_policy);
}

void launch_xd_axis_fft(const void* input, void* output, std::size_t length,
                        std::size_t suffix, std::size_t lines, bool fp64,
                        bool inverse, double scale, cudaStream_t stream,
                        unsigned elements_per_thread,
                        unsigned ffts_per_block_cap,
                        unsigned exchange_policy, unsigned exact_ffts_per_block) {
    if (!valid_length(length))
        throw std::invalid_argument("xD axis cuFFTDx supports lengths 8..4096 power-of-two");
    if (!xd_axis_fft_configuration_supported(length, fp64, elements_per_thread,
                                              ffts_per_block_cap, exact_ffts_per_block))
        throw std::invalid_argument("xD axis cuFFTDx configuration exceeds resource budget");
    if (suffix == 0)
        throw std::invalid_argument("xD axis cuFFTDx requires a positive suffix stride");
    if (ffts_per_block_cap != 0 && ffts_per_block_cap != 8 &&
        ffts_per_block_cap != 16)
        throw std::invalid_argument(
            "xD axis cuFFTDx FFTsPerBlock cap must be zero, 8 or 16");
    if (exchange_policy > 3)
        throw std::invalid_argument("xD axis cuFFTDx exchange policy must be 0, 1, 2 or 3");
    if (exact_ffts_per_block != 0 && exchange_policy == 1)
        throw std::invalid_argument("xD exact axis grouping supports exchange policies 0, 2 and 3");
    if (lines == 0)
        return;
    if (exchange_policy == 3 && suffix != 1 &&
        !xd_axis_fft_u32_index_fits(length, suffix, lines))
        throw std::invalid_argument(
            "xD axis uint32 exchange requires length*lines and suffix <= UINT32_MAX");
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("xD axis cuFFTDx requires non-null input and output");
    if (exact_ffts_per_block != 0) {
        bool launched = false;
#define CUNTT_XD_LAUNCH_GROUPED(Real, Direction) \
        dispatch_grouped_configuration<Real, cufftdx::fft_direction::Direction>( \
            length, static_cast<const global_complex_t<Real>*>(input), \
            static_cast<global_complex_t<Real>*>(output), suffix, lines, \
            static_cast<Real>(scale), stream, elements_per_thread, \
            exact_ffts_per_block, exchange_policy, false)
        if (fp64) launched = inverse ? CUNTT_XD_LAUNCH_GROUPED(double, inverse)
                                     : CUNTT_XD_LAUNCH_GROUPED(double, forward);
        else launched = inverse ? CUNTT_XD_LAUNCH_GROUPED(float, inverse)
                                : CUNTT_XD_LAUNCH_GROUPED(float, forward);
#undef CUNTT_XD_LAUNCH_GROUPED
        if (!launched)
            throw std::invalid_argument("xD exact axis grouping exceeds device resources or uses an unavailable exchange policy");
        return;
    }
    if (fp64) {
        const auto* typed_input = static_cast<const Complex64*>(input);
        auto* typed_output = static_cast<Complex64*>(output);
        if (inverse)
            dispatch_configuration<double, cufftdx::fft_direction::inverse>(
                length, typed_input, typed_output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap, exchange_policy);
        else
            dispatch_configuration<double, cufftdx::fft_direction::forward>(
                length, typed_input, typed_output, suffix, lines, scale, stream,
                elements_per_thread, ffts_per_block_cap, exchange_policy);
    } else {
        const auto* typed_input = static_cast<const Complex32*>(input);
        auto* typed_output = static_cast<Complex32*>(output);
        if (inverse)
            dispatch_configuration<float, cufftdx::fft_direction::inverse>(
                length, typed_input, typed_output, suffix, lines,
                static_cast<float>(scale), stream, elements_per_thread,
                ffts_per_block_cap, exchange_policy);
        else
            dispatch_configuration<float, cufftdx::fft_direction::forward>(
                length, typed_input, typed_output, suffix, lines,
                static_cast<float>(scale), stream, elements_per_thread,
                ffts_per_block_cap, exchange_policy);
    }
}

}  // namespace cuntt::detail
