#include "xd_pair_fft.hpp"

#include "cuntt/butterfly.hpp"

#include <cuda_runtime.h>
#include <cufftdx.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef CUBUTTERFLY_XD_PAIR_SM
#ifdef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_PAIR_SM CUBUTTERFLY_XD_CUFFTDX_SM
#else
#define CUBUTTERFLY_XD_PAIR_SM 700
#endif
#endif

namespace cuntt::detail {
namespace {

constexpr unsigned kMinAxis = 8;
constexpr unsigned kMaxAxis = 64;
constexpr unsigned kFp64MaxAxis = 32;
constexpr std::size_t kSharedLimit = 48U * 1024U;
constexpr unsigned kBlockThreads = 128;
constexpr unsigned kGridXLimit = 65535;
constexpr unsigned kGridYLimit = 65535;

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

template <typename Real, unsigned Length, cufftdx::fft_direction Direction>
using ThreadFft = decltype(
    cufftdx::Thread() + cufftdx::Size<Length>() +
    cufftdx::Type<cufftdx::fft_type::c2c>() + cufftdx::Direction<Direction>() +
    cufftdx::Precision<Real>() + cufftdx::SM<CUBUTTERFLY_XD_PAIR_SM>());

inline bool valid_axis(std::size_t value) noexcept {
    return value >= kMinAxis && value <= kMaxAxis && (value & (value - 1U)) == 0;
}

inline bool valid_shape(std::size_t rows, std::size_t cols, bool fp64) noexcept {
    if (!valid_axis(rows) || !valid_axis(cols))
        return false;
    // MathDx's Thread FFT database tops out at 40 points for FP64. The
    // nearest supported power-of-two inventory therefore stops at 32.
    return !fp64 || (rows <= kFp64MaxAxis && cols <= kFp64MaxAxis);
}

inline std::size_t shared_pitch(std::size_t cols) noexcept { return cols + 1U; }

inline bool shared_shape_fits(std::size_t rows, std::size_t cols,
                              bool fp64) noexcept {
    const std::size_t pitch = shared_pitch(cols);
    if (rows > std::numeric_limits<std::size_t>::max() / pitch)
        return false;
    const std::size_t shared_values = rows * pitch;
    const std::size_t element_bytes = fp64 ? sizeof(Complex64) : sizeof(Complex32);
    return shared_values <= std::numeric_limits<std::size_t>::max() / element_bytes &&
           shared_values * element_bytes <= kSharedLimit;
}

inline bool compiled_sm_matches_device() noexcept {
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        return false;
    const int runtime_sm = properties.major * 100 + properties.minor * 10;
    return runtime_sm == CUBUTTERFLY_XD_PAIR_SM;
}

template <typename Real, unsigned Rows, unsigned Cols,
          cufftdx::fft_direction Direction>
__launch_bounds__(kBlockThreads) __global__ void xd_pair_fft_kernel(
    const global_complex_t<Real>* input, global_complex_t<Real>* output,
    std::uint64_t planes, Real scale) {
    using GlobalComplex = global_complex_t<Real>;
    using RowFFT = ThreadFft<Real, Cols, Direction>;
    using ColFFT = ThreadFft<Real, Rows, Direction>;
    using Value = typename RowFFT::value_type;

    static_assert(RowFFT::storage_size == Cols,
                  "xD pair Thread FFT row storage must be the full row");
    static_assert(ColFFT::storage_size == Rows,
                  "xD pair Thread FFT column storage must be the full column");
    static_assert(sizeof(Value) == sizeof(GlobalComplex),
                  "xD pair Thread FFT and global complex layouts must match");
    static_assert(alignof(Value) == alignof(GlobalComplex),
                  "xD pair Thread FFT and global complex alignment must match");
    constexpr unsigned Points = Rows * Cols;
    constexpr unsigned Pitch = Cols + 1U;
    extern __shared__ __align__(16) unsigned char raw_storage[];
    auto* tile = reinterpret_cast<Value*>(raw_storage);

    const std::uint64_t plane = static_cast<std::uint64_t>(blockIdx.x) +
                                static_cast<std::uint64_t>(blockIdx.y) * gridDim.x;
    const bool active = plane < planes;
    const std::uint64_t base = plane * static_cast<std::uint64_t>(Points);

    // Stage one natural-order plane. The extra slot per row removes the
    // column-pass bank-alignment hazard while exact input/output aliasing stays
    // safe because no global store occurs before this CTA-wide handoff.
    for (unsigned index = threadIdx.x; index < Points; index += blockDim.x) {
        const unsigned row = index / Cols;
        const unsigned col = index - row * Cols;
        const unsigned shared_index = row * Pitch + col;
        const GlobalComplex source = active ? input[base + index] : GlobalComplex{0, 0};
        tile[shared_index] = Value{source.real, source.imag};
    }
    __syncthreads();

    // One thread owns one complete row. Thread FFT execute has no CTA barrier;
    // only the exchange into the column layout is synchronized below.
    {
        typename RowFFT::value_type row_data[RowFFT::storage_size];
        if (threadIdx.x < Rows) {
            const unsigned row = threadIdx.x;
#pragma unroll
            for (unsigned item = 0; item < RowFFT::storage_size; ++item)
                row_data[item] = tile[row * Pitch + item];
            RowFFT().execute(row_data);
#pragma unroll
            for (unsigned item = 0; item < RowFFT::storage_size; ++item)
                tile[row * Pitch + item] = row_data[item];
        }
    }
    __syncthreads();

    // The row-major shared tile is now the transposition boundary. One thread
    // owns one complete column and executes the second static codelet in
    // registers, then restores natural row-major placement in shared memory.
    {
        typename ColFFT::value_type column_data[ColFFT::storage_size];
        if (threadIdx.x < Cols) {
            const unsigned col = threadIdx.x;
#pragma unroll
            for (unsigned item = 0; item < ColFFT::storage_size; ++item)
                column_data[item] = tile[item * Pitch + col];
            ColFFT().execute(column_data);
#pragma unroll
            for (unsigned item = 0; item < ColFFT::storage_size; ++item)
                tile[item * Pitch + col] = column_data[item];
        }
    }
    __syncthreads();

    for (unsigned index = threadIdx.x; index < Points; index += blockDim.x) {
        if (active) {
            const unsigned row = index / Cols;
            const unsigned col = index - row * Cols;
            const Value value = tile[row * Pitch + col];
            output[base + index] = GlobalComplex{value.x * scale, value.y * scale};
        }
    }
}

template <typename Real, unsigned Rows, unsigned Cols,
          cufftdx::fft_direction Direction>
void launch_shape(const void* input, void* output, std::size_t planes,
                  double scale, cudaStream_t stream) {
    using FFT = ThreadFft<Real, Cols, Direction>;
    using GlobalComplex = global_complex_t<Real>;
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(Rows) * (Cols + 1U) * sizeof(typename FFT::value_type);
    static_assert(shared_bytes <= kSharedLimit,
                  "xD pair Thread FFT exceeds the 48 KiB shared-memory budget");

    constexpr std::uint64_t kMaxGridPlanes =
        static_cast<std::uint64_t>(kGridXLimit) * kGridYLimit;
    if (planes > kMaxGridPlanes)
        throw std::invalid_argument("xD pair FFT plane count exceeds the CUDA grid limit");
    const unsigned grid_x = static_cast<unsigned>(
        planes < kGridXLimit ? planes : kGridXLimit);
    const unsigned grid_y = static_cast<unsigned>((planes + grid_x - 1U) / grid_x);
    xd_pair_fft_kernel<Real, Rows, Cols, Direction>
        <<<dim3(grid_x, grid_y, 1U), dim3(kBlockThreads, 1U, 1U), shared_bytes, stream>>>(
            static_cast<const GlobalComplex*>(input), static_cast<GlobalComplex*>(output),
            static_cast<std::uint64_t>(planes), static_cast<Real>(scale));
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD pair Thread FFT: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, cufftdx::fft_direction Direction>
void dispatch_fp32_shape(std::size_t rows, std::size_t cols, const void* input,
                         void* output, std::size_t planes, double scale,
                         cudaStream_t stream) {
#define CUNTT_XD_PAIR_FP32_CASE(R, C) \
    if (rows == (R) && cols == (C)) { \
        launch_shape<Real, R, C, Direction>(input, output, planes, scale, stream); \
        return; \
    }
    CUNTT_XD_PAIR_FP32_CASE(8, 8)
    CUNTT_XD_PAIR_FP32_CASE(8, 16)
    CUNTT_XD_PAIR_FP32_CASE(8, 32)
    CUNTT_XD_PAIR_FP32_CASE(8, 64)
    CUNTT_XD_PAIR_FP32_CASE(16, 8)
    CUNTT_XD_PAIR_FP32_CASE(16, 16)
    CUNTT_XD_PAIR_FP32_CASE(16, 32)
    CUNTT_XD_PAIR_FP32_CASE(16, 64)
    CUNTT_XD_PAIR_FP32_CASE(32, 8)
    CUNTT_XD_PAIR_FP32_CASE(32, 16)
    CUNTT_XD_PAIR_FP32_CASE(32, 32)
    CUNTT_XD_PAIR_FP32_CASE(32, 64)
    CUNTT_XD_PAIR_FP32_CASE(64, 8)
    CUNTT_XD_PAIR_FP32_CASE(64, 16)
    CUNTT_XD_PAIR_FP32_CASE(64, 32)
    CUNTT_XD_PAIR_FP32_CASE(64, 64)
#undef CUNTT_XD_PAIR_FP32_CASE
    throw std::invalid_argument("xD pair Thread FFT received an unsupported FP32 shape");
}

template <typename Real, cufftdx::fft_direction Direction>
void dispatch_fp64_shape(std::size_t rows, std::size_t cols, const void* input,
                         void* output, std::size_t planes, double scale,
                         cudaStream_t stream) {
#define CUNTT_XD_PAIR_FP64_CASE(R, C) \
    if (rows == (R) && cols == (C)) { \
        launch_shape<Real, R, C, Direction>(input, output, planes, scale, stream); \
        return; \
    }
    CUNTT_XD_PAIR_FP64_CASE(8, 8)
    CUNTT_XD_PAIR_FP64_CASE(8, 16)
    CUNTT_XD_PAIR_FP64_CASE(8, 32)
    CUNTT_XD_PAIR_FP64_CASE(16, 8)
    CUNTT_XD_PAIR_FP64_CASE(16, 16)
    CUNTT_XD_PAIR_FP64_CASE(16, 32)
    CUNTT_XD_PAIR_FP64_CASE(32, 8)
    CUNTT_XD_PAIR_FP64_CASE(32, 16)
    CUNTT_XD_PAIR_FP64_CASE(32, 32)
#undef CUNTT_XD_PAIR_FP64_CASE
    throw std::invalid_argument("xD pair Thread FFT received an unsupported FP64 shape");
}

}  // namespace

bool xd_pair_fft_available(std::size_t rows, std::size_t cols,
                           bool fp64) noexcept {
    return valid_shape(rows, cols, fp64) && shared_shape_fits(rows, cols, fp64) &&
           compiled_sm_matches_device();
}

void launch_xd_pair_fft(const void* input, void* output, std::size_t rows,
                        std::size_t cols, std::size_t planes, bool fp64,
                        bool inverse, double scale, cudaStream_t stream) {
    if (!valid_shape(rows, cols, fp64)) {
        if (fp64)
            throw std::invalid_argument(
                "xD pair FP64 Thread FFT supports axes 8, 16, or 32");
        throw std::invalid_argument(
            "xD pair FP32 Thread FFT supports axes 8, 16, 32, or 64");
    }
    if (!shared_shape_fits(rows, cols, fp64))
        throw std::invalid_argument(
            "xD pair Thread FFT exceeds the 48 KiB shared-memory budget");
    if (planes == 0)
        return;
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument("xD pair Thread FFT requires non-null input and output");

    if (fp64) {
        if (inverse)
            dispatch_fp64_shape<double, cufftdx::fft_direction::inverse>(
                rows, cols, input, output, planes, scale, stream);
        else
            dispatch_fp64_shape<double, cufftdx::fft_direction::forward>(
                rows, cols, input, output, planes, scale, stream);
    } else {
        if (inverse)
            dispatch_fp32_shape<float, cufftdx::fft_direction::inverse>(
                rows, cols, input, output, planes, scale, stream);
        else
            dispatch_fp32_shape<float, cufftdx::fft_direction::forward>(
                rows, cols, input, output, planes, scale, stream);
    }
}

}  // namespace cuntt::detail
