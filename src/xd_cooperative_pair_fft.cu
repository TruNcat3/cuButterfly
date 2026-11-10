#include "xd_cooperative_pair_fft.hpp"

#include "cuntt/butterfly.hpp"
#include "xd_cooperative_codelet.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef CUBUTTERFLY_XD_COOPERATIVE_PAIR_SM
#ifdef CUBUTTERFLY_XD_CUFFTDX_SM
#define CUBUTTERFLY_XD_COOPERATIVE_PAIR_SM CUBUTTERFLY_XD_CUFFTDX_SM
#else
#define CUBUTTERFLY_XD_COOPERATIVE_PAIR_SM 800
#endif
#endif

namespace cuntt::detail {
namespace {

constexpr std::size_t kMinAxis = 32;
constexpr std::size_t kMaxAxis = 128;
constexpr unsigned kCodeletLanes = 4;
constexpr unsigned kColumnGroups = 8;
constexpr unsigned kSharedLayoutPaddedRowMajor = 0;
constexpr unsigned kSharedLayoutXorBank = 1;
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

constexpr bool is_power_of_two(std::size_t value) noexcept {
    return value != 0 && (value & (value - 1U)) == 0;
}

inline bool valid_axis(std::size_t value) noexcept {
    return value >= kMinAxis && value <= kMaxAxis && is_power_of_two(value);
}

inline bool valid_shape(std::size_t rows, std::size_t cols) noexcept {
    return valid_axis(rows) && valid_axis(cols);
}

template <typename Real, unsigned Rows, unsigned Cols>
__host__ __device__ constexpr unsigned block_threads_for() noexcept {
    if constexpr (sizeof(Real) == sizeof(double))
        return (Rows <= 32U && Cols <= 32U) ? 128U : 256U;
    else
        return (Rows > Cols ? Rows : Cols) * kCodeletLanes;
}

constexpr bool valid_shared_layout(unsigned shared_layout) noexcept {
    return shared_layout == kSharedLayoutPaddedRowMajor ||
           shared_layout == kSharedLayoutXorBank;
}

// Return a physical shared-memory slot for one logical row/column pair. The
// XOR layout is a row permutation independently for every column, so it is a
// bijection over the same padded allocation as layout 0. Shifting the column
// digit by two selects the low 32-bit-bank-relevant row bits for Complex32;
// Complex64 keeps the same map and remains resource-neutral.
template <unsigned Rows, unsigned Cols, unsigned SharedLayout>
__device__ __forceinline__ unsigned shared_index(unsigned row, unsigned column) {
    static_assert(SharedLayout == kSharedLayoutPaddedRowMajor ||
                      SharedLayout == kSharedLayoutXorBank,
                  "unknown xD cooperative pair shared layout");
    constexpr unsigned pitch = Cols + 1U;
    if constexpr (SharedLayout == kSharedLayoutPaddedRowMajor) {
        return row * pitch + column;
    } else {
        const unsigned bank_row = row ^ ((column << 2U) & (Rows - 1U));
        return bank_row * pitch + column;
    }
}

constexpr std::size_t shared_bytes_for(std::size_t rows, std::size_t cols,
                                       bool fp64,
                                       unsigned shared_layout) noexcept {
    if (!valid_shared_layout(shared_layout))
        return 0;
    // The extra column is a row-pitch pad. It avoids relying on one particular
    // shared-bank alignment while preserving a simple bijective row-major map.
    const std::size_t pitch = cols + 1U;
    if (rows > std::numeric_limits<std::size_t>::max() / pitch)
        return 0;
    const std::size_t values = rows * pitch;
    const std::size_t bytes_per_value = fp64 ? sizeof(Complex64) : sizeof(Complex32);
    if (values > std::numeric_limits<std::size_t>::max() / bytes_per_value)
        return 0;
    return values * bytes_per_value;
}

inline std::size_t device_shared_limit() noexcept {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return 0;
    int optin = 0;
    int regular = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&regular, cudaDevAttrMaxSharedMemoryPerBlock,
                               device) != cudaSuccess)
        return 0;
    const int limit = optin > regular ? optin : regular;
    return limit > 0 ? static_cast<std::size_t>(limit) : 0;
}

inline bool compiled_sm_matches_device() noexcept {
    int device = 0;
    int major = 0, minor = 0;
    // This check also runs on the direct-launch path. Full device properties
    // can take milliseconds on a shared host; a captured graph hides that
    // host cost and would give the selector a misleading advantage. Query
    // only the immutable SM attributes, preserving checks after device switches.
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
        return false;
    const int runtime_sm = major * 100 + minor * 10;
    return runtime_sm == CUBUTTERFLY_XD_COOPERATIVE_PAIR_SM;
}

template <typename Complex, unsigned Columns, bool Inverse>
__device__ __forceinline__ void cooperative_fft32(
    Complex (&values)[8], unsigned lane) {
    // Length 32 is kept local to this translation unit. Four lanes each own
    // an eight-point Thread FFT, then perform the same two radix-2 exchanges
    // as the established four-lane 64/128 path.
    xd_cooperative_detail::local_fft<3U, Inverse>(values);

    const unsigned mask = __activemask();
#pragma unroll
    for (unsigned k = 0; k < 8U; ++k) {
        auto value = values[k];
        if (k && lane) {
            const auto phase = xd_cooperative_detail::table_twiddle<Inverse, Complex>(
                lane * k * 4U);
            value = register_tile::multiply(value, phase);
        }

        const Complex peer_radix2{
            __shfl_xor_sync(mask, value.real, 2U * Columns),
            __shfl_xor_sync(mask, value.imag, 2U * Columns)};
        const bool upper_radix2 = (lane & 2U) != 0;
        value = {(upper_radix2 ? -value.real : value.real) + peer_radix2.real,
                 (upper_radix2 ? -value.imag : value.imag) + peer_radix2.imag};
        if (lane == 3U)
            value = Inverse ? Complex{-value.imag, value.real}
                            : Complex{value.imag, -value.real};

        const Complex peer_final{
            __shfl_xor_sync(mask, value.real, Columns),
            __shfl_xor_sync(mask, value.imag, Columns)};
        const bool upper_final = (lane & 1U) != 0;
        values[k] = {(upper_final ? -value.real : value.real) + peer_final.real,
                     (upper_final ? -value.imag : value.imag) + peer_final.imag};
    }
}

template <typename Complex, unsigned Length, unsigned Columns, bool Inverse>
__device__ __forceinline__ void cooperative_line_fft(
    Complex (&values)[Length / kCodeletLanes], unsigned lane) {
    if constexpr (Length == 32U)
        cooperative_fft32<Complex, Columns, Inverse>(values, lane);
    else
        xd_cooperative_fft4<register_tile::power_log2(Length), Columns, Inverse>(values, lane);
}

template <typename Real, unsigned Rows, unsigned Cols, bool Inverse,
          unsigned SharedLayout, unsigned BlockThreads>
__launch_bounds__(BlockThreads) __global__ void cooperative_pair_kernel(
    const global_complex_t<Real>* input, global_complex_t<Real>* output,
    std::uint64_t planes, Real scale) {
    using Complex = global_complex_t<Real>;
    constexpr unsigned RowThreads = Rows * kCodeletLanes;
    constexpr unsigned ColThreads = Cols * kCodeletLanes;
    constexpr unsigned Points = Rows * Cols;
    constexpr unsigned PhysicalRowLines = BlockThreads / kCodeletLanes;
    constexpr unsigned PhysicalColumnGroups =
        BlockThreads / (kColumnGroups * kCodeletLanes);
    constexpr unsigned RowBatches =
        (Rows + PhysicalRowLines - 1U) / PhysicalRowLines;
    constexpr unsigned ColumnBatches =
        (Cols / kColumnGroups + PhysicalColumnGroups - 1U) / PhysicalColumnGroups;

    static_assert(BlockThreads == block_threads_for<Real, Rows, Cols>(),
                  "cooperative pair CTA geometry must match precision resources");
    static_assert(RowThreads <= 512 && ColThreads <= 512,
                  "cooperative pair CTA geometry exceeds the 512-thread bound");
    static_assert((Cols % kCodeletLanes) == 0 && (Rows % kCodeletLanes) == 0,
                  "cooperative pair axes must be divisible by four lanes");

    extern __shared__ __align__(16) unsigned char raw_storage[];
    auto* tile = reinterpret_cast<Complex*>(raw_storage);

    const std::uint64_t plane = static_cast<std::uint64_t>(blockIdx.x) +
                                static_cast<std::uint64_t>(blockIdx.y) * gridDim.x;
    const bool active = plane < planes;
    const std::uint64_t base = plane * static_cast<std::uint64_t>(Points);
    const unsigned thread = threadIdx.x;

    // Row pass: each row is one four-lane cooperative FFT. Global values for a
    // row are read as j*4+lane, so each subgroup traverses adjacent elements.
    // FP64 uses 256 physical threads and folds 128 rows into two iterations;
    // this keeps the register allocation below the SM80 65536-register CTA
    // budget while preserving one complete row dependency per iteration.
#pragma unroll
    for (unsigned batch = 0; batch < RowBatches; ++batch) {
        const unsigned row = thread / kCodeletLanes + batch * PhysicalRowLines;
        if (row < Rows) {
            const unsigned lane = thread % kCodeletLanes;
            Complex values[Cols / kCodeletLanes];
#pragma unroll
            for (unsigned j = 0; j < Cols / kCodeletLanes; ++j) {
                const unsigned col = j * kCodeletLanes + lane;
                values[j] = active ? input[base + static_cast<std::uint64_t>(row) * Cols + col]
                                   : Complex{0, 0};
            }
            cooperative_line_fft<Complex, Cols, 1, Inverse>(values, lane);
#pragma unroll
            for (unsigned j = 0; j < Cols / kCodeletLanes; ++j) {
                const unsigned frequency =
                    j + (Cols / kCodeletLanes) * register_tile::codelet_output_lane<kCodeletLanes>(lane);
                tile[shared_index<Rows, Cols, SharedLayout>(row, frequency)] = values[j];
            }
        }
    }
    __syncthreads();

    // Column pass: groups of eight adjacent columns occupy the low thread bits
    // (8 columns x 4 lanes = one warp). This keeps each column's shuffle group
    // closed while rows are exchanged through the shared plane.
    // As above, a 256-thread FP64 CTA visits two sets of eight column groups
    // when Cols=128. Each iteration has complete four-lane shuffle groups.
#pragma unroll
    for (unsigned batch = 0; batch < ColumnBatches; ++batch) {
        const unsigned physical_group = thread / (kColumnGroups * kCodeletLanes);
        const unsigned group = physical_group + batch * PhysicalColumnGroups;
        if (group < Cols / kColumnGroups) {
            const unsigned column = group * kColumnGroups + thread % kColumnGroups;
            const unsigned lane = (thread / kColumnGroups) % kCodeletLanes;
            Complex values[Rows / kCodeletLanes];
#pragma unroll
            for (unsigned j = 0; j < Rows / kCodeletLanes; ++j)
                values[j] = tile[shared_index<Rows, Cols, SharedLayout>(
                    j * kCodeletLanes + lane, column)];
            // All readers of these eight columns belong to this warp. The
            // shuffled FFT exchanges registers, but does not order shared
            // memory before another lane reuses the same column for output.
            __syncwarp();
            cooperative_line_fft<Complex, Rows, kColumnGroups, Inverse>(values, lane);
#pragma unroll
            for (unsigned j = 0; j < Rows / kCodeletLanes; ++j) {
                const unsigned frequency =
                    j + (Rows / kCodeletLanes) * register_tile::codelet_output_lane<kCodeletLanes>(lane);
                tile[shared_index<Rows, Cols, SharedLayout>(frequency, column)] = values[j];
            }
        }
    }
    __syncthreads();

    // Export only after both passes have a complete dependency-closed plane.
    // No store occurs before this point, so input == output is safe.
    for (unsigned index = thread; index < Points; index += BlockThreads) {
        if (active) {
            const unsigned row = index / Cols;
            const unsigned column = index % Cols;
            const Complex value = tile[shared_index<Rows, Cols, SharedLayout>(row, column)];
            output[base + index] = Complex{value.real * scale, value.imag * scale};
        }
    }
}

template <typename Real, unsigned Rows, unsigned Cols, bool Inverse,
          unsigned SharedLayout>
void prepare_shape(std::size_t shared_bytes) {
    constexpr unsigned block_threads = block_threads_for<Real, Rows, Cols>();
    static std::mutex mutex;
    static int configured_device = -1;
    std::lock_guard<std::mutex> lock(mutex);

    int device = 0;
    const cudaError_t selected = cudaGetDevice(&device);
    if (selected != cudaSuccess)
        throw std::runtime_error(std::string("select xD cooperative pair CUDA device: ") +
                                 cudaGetErrorString(selected));
    // A process may switch devices between plans. Reapply the attribute on a
    // new device, while the common steady execute path never reaches here.
    if (configured_device == device)
        return;

    auto status = cudaFuncSetAttribute(
        cooperative_pair_kernel<Real, Rows, Cols, Inverse, SharedLayout, block_threads>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes));
    if (status == cudaSuccess) {
        int active_blocks = 0;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            cooperative_pair_kernel<Real, Rows, Cols, Inverse, SharedLayout, block_threads>,
            block_threads, shared_bytes);
        if (status == cudaSuccess && active_blocks < 1)
            status = cudaErrorInvalidConfiguration;
    }
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("prepare xD cooperative pair FFT: ") +
                                 cudaGetErrorString(status));
    configured_device = device;
}

template <typename Real, unsigned Rows, unsigned Cols, bool Inverse,
          unsigned SharedLayout>
void launch_shape(const void* input, void* output, std::size_t planes,
                  double scale, cudaStream_t stream) {
    using Complex = global_complex_t<Real>;
    constexpr unsigned block_threads = block_threads_for<Real, Rows, Cols>();
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(Rows) * (Cols + 1U) * sizeof(Complex);
    constexpr std::uint64_t kMaxGridPlanes =
        static_cast<std::uint64_t>(kGridXLimit) * kGridYLimit;
    if (planes > kMaxGridPlanes)
        throw std::invalid_argument("xD cooperative pair FFT plane count exceeds the CUDA grid limit");
    const unsigned grid_x = static_cast<unsigned>(planes < kGridXLimit ? planes : kGridXLimit);
    const unsigned grid_y = static_cast<unsigned>((planes + grid_x - 1U) / grid_x);
    cooperative_pair_kernel<Real, Rows, Cols, Inverse, SharedLayout, block_threads>
        <<<dim3(grid_x, grid_y, 1U), dim3(block_threads, 1U, 1U), shared_bytes, stream>>>(
            static_cast<const Complex*>(input), static_cast<Complex*>(output),
            static_cast<std::uint64_t>(planes), static_cast<Real>(scale));
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch xD cooperative pair FFT: ") +
                                 cudaGetErrorString(status));
}

template <typename Real, bool Inverse, unsigned SharedLayout>
void dispatch_shape(std::size_t rows, std::size_t cols, const void* input,
                    void* output, std::size_t planes, double scale,
                    cudaStream_t stream) {
#define CUNTT_XD_COOPERATIVE_PAIR_CASE(R, C) \
    if (rows == (R) && cols == (C)) { \
        launch_shape<Real, R, C, Inverse, SharedLayout>(input, output, planes, scale, stream); \
        return; \
    }
    CUNTT_XD_COOPERATIVE_PAIR_CASE(32, 32)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(32, 64)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(32, 128)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(64, 32)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(64, 64)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(64, 128)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(128, 64)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(128, 32)
    CUNTT_XD_COOPERATIVE_PAIR_CASE(128, 128)
#undef CUNTT_XD_COOPERATIVE_PAIR_CASE
    throw std::invalid_argument("xD cooperative pair FFT received an unsupported shape");
}

template <typename Real, bool Inverse, unsigned SharedLayout>
void dispatch_prepare(std::size_t rows, std::size_t cols, std::size_t shared_bytes) {
#define CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(R, C) \
    if (rows == (R) && cols == (C)) { \
        prepare_shape<Real, R, C, Inverse, SharedLayout>(shared_bytes); \
        return; \
    }
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(32, 32)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(32, 64)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(32, 128)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(64, 32)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(64, 64)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(64, 128)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(128, 64)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(128, 32)
    CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE(128, 128)
#undef CUNTT_XD_COOPERATIVE_PAIR_PREPARE_CASE
    throw std::invalid_argument("xD cooperative pair FFT received an unsupported shape");
}

}  // namespace

bool xd_cooperative_pair_fft_available(std::size_t rows, std::size_t cols,
                                       bool fp64,
                                       unsigned shared_layout) noexcept {
    if (!valid_shape(rows, cols) || !valid_shared_layout(shared_layout) ||
        !compiled_sm_matches_device())
        return false;
    const std::size_t shared_bytes = shared_bytes_for(rows, cols, fp64, shared_layout);
    const std::size_t limit = device_shared_limit();
    return shared_bytes != 0 && limit != 0 && shared_bytes <= limit;
}

void prepare_xd_cooperative_pair_fft(std::size_t rows, std::size_t cols,
                                     bool fp64, bool inverse,
                                     unsigned shared_layout) {
    if (!valid_shape(rows, cols) || !valid_shared_layout(shared_layout))
        throw std::invalid_argument(
            "xD cooperative pair FFT supports 32, 64 or 128 point axes and shared layouts 0 or 1");
    if (!compiled_sm_matches_device())
        throw std::invalid_argument(
            "xD cooperative pair FFT was compiled for a different CUDA SM");
    const std::size_t shared_bytes = shared_bytes_for(rows, cols, fp64, shared_layout);
    const std::size_t shared_limit = device_shared_limit();
    if (shared_bytes == 0 || shared_limit == 0 || shared_bytes > shared_limit)
        throw std::invalid_argument(
            "xD cooperative pair FFT shape exceeds the device opt-in shared-memory limit");
    if (fp64) {
        if (inverse)
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_prepare<double, true, kSharedLayoutXorBank>(rows, cols, shared_bytes);
            else
                dispatch_prepare<double, true, kSharedLayoutPaddedRowMajor>(rows, cols, shared_bytes);
        else
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_prepare<double, false, kSharedLayoutXorBank>(rows, cols, shared_bytes);
            else
                dispatch_prepare<double, false, kSharedLayoutPaddedRowMajor>(rows, cols, shared_bytes);
    } else {
        if (inverse)
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_prepare<float, true, kSharedLayoutXorBank>(rows, cols, shared_bytes);
            else
                dispatch_prepare<float, true, kSharedLayoutPaddedRowMajor>(rows, cols, shared_bytes);
        else
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_prepare<float, false, kSharedLayoutXorBank>(rows, cols, shared_bytes);
            else
                dispatch_prepare<float, false, kSharedLayoutPaddedRowMajor>(rows, cols, shared_bytes);
    }
}

void launch_xd_cooperative_pair_fft(const void* input, void* output,
                                    std::size_t rows, std::size_t cols,
                                    std::size_t planes, bool fp64, bool inverse,
                                    double scale, cudaStream_t stream,
                                    unsigned shared_layout) {
    if (!valid_shape(rows, cols) || !valid_shared_layout(shared_layout))
        throw std::invalid_argument(
            "xD cooperative pair FFT supports 32, 64 or 128 point axes and shared layouts 0 or 1");
    if (!compiled_sm_matches_device())
        throw std::invalid_argument(
            "xD cooperative pair FFT was compiled for a different CUDA SM");
    const std::size_t shared_bytes = shared_bytes_for(rows, cols, fp64, shared_layout);
    const std::size_t shared_limit = device_shared_limit();
    if (shared_bytes == 0 || shared_limit == 0 || shared_bytes > shared_limit)
        throw std::invalid_argument(
            "xD cooperative pair FFT shape exceeds the device opt-in shared-memory limit");
    if (planes == 0)
        return;
    if (input == nullptr || output == nullptr)
        throw std::invalid_argument(
            "xD cooperative pair FFT requires non-null input and output");

    if (fp64) {
        if (inverse)
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_shape<double, true, kSharedLayoutXorBank>(rows, cols, input, output, planes, scale, stream);
            else
                dispatch_shape<double, true, kSharedLayoutPaddedRowMajor>(rows, cols, input, output, planes, scale, stream);
        else
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_shape<double, false, kSharedLayoutXorBank>(rows, cols, input, output, planes, scale, stream);
            else
                dispatch_shape<double, false, kSharedLayoutPaddedRowMajor>(rows, cols, input, output, planes, scale, stream);
    } else {
        if (inverse)
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_shape<float, true, kSharedLayoutXorBank>(rows, cols, input, output, planes, scale, stream);
            else
                dispatch_shape<float, true, kSharedLayoutPaddedRowMajor>(rows, cols, input, output, planes, scale, stream);
        else
            if (shared_layout == kSharedLayoutXorBank)
                dispatch_shape<float, false, kSharedLayoutXorBank>(rows, cols, input, output, planes, scale, stream);
            else
                dispatch_shape<float, false, kSharedLayoutPaddedRowMajor>(rows, cols, input, output, planes, scale, stream);
    }
}

}  // namespace cuntt::detail
