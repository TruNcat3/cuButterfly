#include "xd_stockham_pair_fft.hpp"
#include "fft_radix4_codelet.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace cuntt::detail {
namespace {

template <typename T>
struct Cx;
template <>
// Preserve the complex element's vector alignment at the shared/global
// exchange boundary. Scalar-aligned pairs prevent 64-bit shared operations
// and retain bank conflicts even with a padded [32][33] tile.
struct alignas(8) Cx<float> { float x, y; };
template <>
struct alignas(16) Cx<double> { double x, y; };

// The experimental provider is fixed to 32-point axes. Keep
// the generated radix stages explicit, but supply their roots from constant
// memory instead of evaluating sin/cos for every lane on every replay.  The
// table is laid out by stage: offset = length/2 - 1, with length/2 entries.
__device__ __constant__ Cx<float> kRadix2Roots32[31] = {
    {1.0f, 0.0f},
    {1.0f, 0.0f}, {0.0f, -1.0f},
    {1.0f, 0.0f}, {0.7071067811865476f, -0.7071067811865476f},
    {0.0f, -1.0f}, {-0.7071067811865476f, -0.7071067811865476f},
    {1.0f, 0.0f}, {0.9238795325112867f, -0.3826834323650898f},
    {0.7071067811865476f, -0.7071067811865476f},
    {0.3826834323650898f, -0.9238795325112867f},
    {0.0f, -1.0f}, {-0.3826834323650898f, -0.9238795325112867f},
    {-0.7071067811865476f, -0.7071067811865476f},
    {-0.9238795325112867f, -0.3826834323650898f},
    {1.0f, 0.0f}, {0.9807852804032304f, -0.1950903220161283f},
    {0.9238795325112867f, -0.3826834323650898f},
    {0.8314696123025452f, -0.5555702330196022f},
    {0.7071067811865476f, -0.7071067811865476f},
    {0.5555702330196022f, -0.8314696123025452f},
    {0.3826834323650898f, -0.9238795325112867f},
    {0.1950903220161283f, -0.9807852804032304f},
    {0.0f, -1.0f}, {-0.1950903220161283f, -0.9807852804032304f},
    {-0.3826834323650898f, -0.9238795325112867f},
    {-0.5555702330196022f, -0.8314696123025452f},
    {-0.7071067811865476f, -0.7071067811865476f},
    {-0.8314696123025452f, -0.5555702330196022f},
    {-0.9238795325112867f, -0.3826834323650898f},
    {-0.9807852804032304f, -0.1950903220161283f},
};

// Supply FP64 roots directly; casting the FP32 table loses requested precision.
__device__ __constant__ Cx<double> kRadix2Roots32Double[31] = {
    {1.0, 0.0},
    {1.0, 0.0}, {0.0, -1.0},
    {1.0, 0.0}, {0.70710678118654752440, -0.70710678118654752440},
    {0.0, -1.0}, {-0.70710678118654752440, -0.70710678118654752440},
    {1.0, 0.0}, {0.92387953251128675613, -0.38268343236508977173},
    {0.70710678118654752440, -0.70710678118654752440},
    {0.38268343236508977173, -0.92387953251128675613},
    {0.0, -1.0}, {-0.38268343236508977173, -0.92387953251128675613},
    {-0.70710678118654752440, -0.70710678118654752440},
    {-0.92387953251128675613, -0.38268343236508977173},
    {1.0, 0.0}, {0.98078528040323044913, -0.19509032201612826785},
    {0.92387953251128675613, -0.38268343236508977173},
    {0.83146961230254523708, -0.55557023301960222474},
    {0.70710678118654752440, -0.70710678118654752440},
    {0.55557023301960222474, -0.83146961230254523708},
    {0.38268343236508977173, -0.92387953251128675613},
    {0.19509032201612826785, -0.98078528040323044913},
    {0.0, -1.0}, {-0.19509032201612826785, -0.98078528040323044913},
    {-0.38268343236508977173, -0.92387953251128675613},
    {-0.55557023301960222474, -0.83146961230254523708},
    {-0.70710678118654752440, -0.70710678118654752440},
    {-0.83146961230254523708, -0.55557023301960222474},
    {-0.92387953251128675613, -0.38268343236508977173},
    {-0.98078528040323044913, -0.19509032201612826785},
};

template <typename T>
__device__ __forceinline__ Cx<T> add(Cx<T> a, Cx<T> b) {
    return {a.x + b.x, a.y + b.y};
}
template <typename T>
__device__ __forceinline__ Cx<T> sub(Cx<T> a, Cx<T> b) {
    return {a.x - b.x, a.y - b.y};
}
template <typename T>
__device__ __forceinline__ Cx<T> mul(Cx<T> a, Cx<T> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
template <typename T>
__device__ __forceinline__ Cx<T> scale(Cx<T> a, T s) {
    return {a.x * s, a.y * s};
}

template <typename T>
__device__ __forceinline__ Cx<T> twiddle(unsigned j, unsigned span,
                                         bool inverse) {
    constexpr T Pi = static_cast<T>(3.1415926535897932384626433832795L);
    const T angle = (inverse ? T(2) : T(-2)) * Pi * static_cast<T>(j) /
                    static_cast<T>(span);
    return {cos(angle), sin(angle)};
}

template <>
__device__ __forceinline__ Cx<double> twiddle<double>(unsigned j, unsigned span,
                                                      bool inverse) {
    const unsigned phase = j & (span - 1U);
    const unsigned half = span >> 1U;
    const Cx<double> root = kRadix2Roots32Double[half - 1U + (phase & (half - 1U))];
    const double sign = phase >= half ? -1.0 : 1.0;
    return {sign * root.x, (inverse ? -sign : sign) * root.y};
}

template <>
__device__ __forceinline__ Cx<float> twiddle<float>(unsigned j, unsigned span,
                                                    bool inverse) {
    const unsigned phase = j & (span - 1U);
    const unsigned half = span >> 1U;
    const Cx<float> root = kRadix2Roots32[half - 1U + (phase & (half - 1U))];
    const float sign = phase >= half ? -1.0f : 1.0f;
    return {sign * root.x, (inverse ? -sign : sign) * root.y};
}

template <typename T, unsigned N>
__device__ __forceinline__ unsigned reverse_bits(unsigned value) {
    unsigned result = 0;
#pragma unroll
    for (unsigned bit = 0; bit < 5; ++bit)
        result = (result << 1U) | ((value >> bit) & 1U);
    return result;
}

// The generated codelet keeps every radix-2 boundary explicit. The initial
// bit-reversal is shared-memory only; each following stage ping-pongs between
// the two arrays and writes both outputs of one butterfly from one lane.
template <typename T, unsigned N, bool WarpSync>
__device__ __forceinline__ void radix2_1d(Cx<T>*& src, Cx<T>*& dst,
                                          unsigned lane, bool inverse) {
    dst[reverse_bits<T, N>(lane)] = src[lane];
    if constexpr (WarpSync) __syncwarp();
    else __syncthreads();
    { Cx<T>* tmp = src; src = dst; dst = tmp; }
#pragma unroll
    for (unsigned stage = 1; stage <= 5; ++stage) {
        const unsigned length = 1U << stage;
        const unsigned half = length >> 1U;
        if (lane < half) {
            const auto w = twiddle<T>(lane, length, inverse);
            for (unsigned group = 0; group < N / length; ++group) {
                const unsigned base = group * length + lane;
                const Cx<T> a = src[base];
                const Cx<T> b = mul(src[base + half], w);
                dst[base] = add(a, b);
                dst[base + half] = sub(a, b);
            }
        }
        if constexpr (WarpSync) __syncwarp();
        else __syncthreads();
        { Cx<T>* tmp = src; src = dst; dst = tmp; }
    }
}

template <typename T, unsigned N>
__launch_bounds__(N * N) __global__ void stockham_pair_kernel(
    const Cx<T>* input, Cx<T>* output, std::uint64_t planes, bool inverse, T factor) {
    const std::uint64_t plane = static_cast<std::uint64_t>(blockIdx.x);
    const unsigned lane = threadIdx.x;
    if (lane >= N * N) return;
    extern __shared__ unsigned char raw[];
    auto* a = reinterpret_cast<Cx<T>*>(raw);
    auto* b = a + N * N;
    const bool active = plane < planes;
    const std::uint64_t base = plane * static_cast<std::uint64_t>(N * N);
    a[lane] = active ? input[base + lane] : Cx<T>{0, 0};
    __syncthreads();

    Cx<T>* src = a;
    Cx<T>* dst = b;
    // Row-major stages. The stage boundaries remain in shared memory, while
    // the bit-reversal makes the radix-2 recurrence unambiguously correct.
    const unsigned row_owner = lane / N;
    Cx<T>* row_src = src + row_owner * N;
    Cx<T>* row_dst = dst + row_owner * N;
    radix2_1d<T, N, true>(row_src, row_dst, lane % N, inverse);
    // Six shared-memory swaps (bit reversal plus five stages) return the
    // block-wide source pointer to the original array.
    __syncthreads();

    // Column stages use the same recurrence while preserving row-major
    // placement in shared memory. Each lane owns one (row, column) element.
    const unsigned col = lane % N;
    const unsigned column_lane = lane / N;
    dst[reverse_bits<T, N>(column_lane) * N + col] =
        src[column_lane * N + col];
    __syncthreads();
    { Cx<T>* tmp = src; src = dst; dst = tmp; }
#pragma unroll
    for (unsigned stage = 1; stage <= 5; ++stage) {
        const unsigned length = 1U << stage;
        const unsigned half = length >> 1U;
        if (column_lane < half) {
            const auto w = twiddle<T>(column_lane, length, inverse);
            for (unsigned group = 0; group < N / length; ++group) {
                const unsigned stage_base = group * length + column_lane;
                const Cx<T> a = src[stage_base * N + col];
                const Cx<T> b = mul(src[(stage_base + half) * N + col], w);
                dst[stage_base * N + col] = add(a, b);
                dst[(stage_base + half) * N + col] = sub(a, b);
            }
        }
        __syncthreads();
        { Cx<T>* tmp = src; src = dst; dst = tmp; }
    }
    if (active) output[base + lane] = scale(src[lane], factor);
}

// Combine two binary DIT stages. Binary-bit-reversed input orders the four
// subtransforms as 0,2,1,3 relative to the natural radix-4 codelet. Twiddles
// must follow those logical indices, not the physical source slot.
template <bool Inverse, unsigned Length, typename T>
__device__ __forceinline__ void fused_radix4(
    Cx<T>& x0, Cx<T>& x1, Cx<T>& x2, Cx<T>& x3, unsigned j) {
    Cx<T> v0 = x0;
    Cx<T> v1 = mul(x2, twiddle<T>(j, Length, Inverse));
    Cx<T> v2 = mul(x1, twiddle<T>(2U * j, Length, Inverse));
    Cx<T> v3 = mul(x3, twiddle<T>(3U * j, Length, Inverse));
    fft_radix4<Inverse>(v0, v1, v2, v3);
    x0 = v0; x1 = v1; x2 = v2; x3 = v3;
}

// Preserve the original row/column ownership and ping-pong layout. Only the
// arithmetic stages change from five binary stages to two fused stages + one.
template <bool Inverse, unsigned Stride, bool WarpSync>
__device__ __forceinline__ void staged_radix4_32(
    Cx<float>*& src, Cx<float>*& dst, unsigned lane) {
    dst[reverse_bits<float, 32>(lane) * Stride] = src[lane * Stride];
    if constexpr (WarpSync) __syncwarp(); else __syncthreads();
    { auto* tmp = src; src = dst; dst = tmp; }
    if (lane == 0) {
#pragma unroll
        for (unsigned base = 0; base < 32; base += 4) {
            auto v0 = src[base * Stride];
            auto v1 = src[(base + 1) * Stride];
            auto v2 = src[(base + 2) * Stride];
            auto v3 = src[(base + 3) * Stride];
            fused_radix4<Inverse, 4>(v0, v1, v2, v3, 0);
            dst[base * Stride] = v0;
            dst[(base + 1) * Stride] = v1;
            dst[(base + 2) * Stride] = v2;
            dst[(base + 3) * Stride] = v3;
        }
    }
    if constexpr (WarpSync) __syncwarp(); else __syncthreads();
    { auto* tmp = src; src = dst; dst = tmp; }
    if (lane < 4) {
#pragma unroll
        for (unsigned base = lane; base < 32; base += 16) {
            auto v0 = src[base * Stride];
            auto v1 = src[(base + 4) * Stride];
            auto v2 = src[(base + 8) * Stride];
            auto v3 = src[(base + 12) * Stride];
            fused_radix4<Inverse, 16>(v0, v1, v2, v3, lane);
            dst[base * Stride] = v0;
            dst[(base + 4) * Stride] = v1;
            dst[(base + 8) * Stride] = v2;
            dst[(base + 12) * Stride] = v3;
        }
    }
    if constexpr (WarpSync) __syncwarp(); else __syncthreads();
    { auto* tmp = src; src = dst; dst = tmp; }
    if (lane < 16) {
        const auto a = src[lane * Stride];
        const auto b = mul(src[(lane + 16) * Stride], twiddle<float>(lane, 32, Inverse));
        dst[lane * Stride] = add(a, b);
        dst[(lane + 16) * Stride] = sub(a, b);
    }
    if constexpr (WarpSync) __syncwarp(); else __syncthreads();
    { auto* tmp = src; src = dst; dst = tmp; }
}

template <bool Inverse>
__launch_bounds__(1024) __global__ void radix4_stage_pair_kernel(
    const Cx<float>* input, Cx<float>* output, float factor) {
    __shared__ Cx<float> a[1024], b[1024];
    const unsigned tid = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 1024;
    a[tid] = input[base + tid];
    __syncthreads();
    Cx<float>* row_src = a + (tid / 32) * 32;
    Cx<float>* row_dst = b + (tid / 32) * 32;
    staged_radix4_32<Inverse, 1, true>(row_src, row_dst, tid % 32);
    // Four swaps return each row to a. Synchronize before cross-row reads.
    __syncthreads();
    Cx<float>* col_src = a + tid % 32;
    Cx<float>* col_dst = b + tid % 32;
    staged_radix4_32<Inverse, 32, false>(col_src, col_dst, tid / 32);
    output[base + tid] = scale(col_src[(tid / 32) * 32], factor);
}

// One register-local FFT. The same body is used by resident and split
// organizations, so changing the inter-axis boundary never changes the core.
template <bool Inverse, typename T>
__device__ __forceinline__ void register_radix4_32(Cx<T> (&v)[32]) {
#pragma unroll
    for (unsigned i = 0; i < 32; ++i) {
        const unsigned j = reverse_bits<T, 32>(i);
        if (i < j) { const auto tmp = v[i]; v[i] = v[j]; v[j] = tmp; }
    }
#pragma unroll
    for (unsigned base = 0; base < 32; base += 4)
        fused_radix4<Inverse, 4>(v[base], v[base+1], v[base+2], v[base+3], 0);
#pragma unroll
    for (unsigned group = 0; group < 2; ++group) {
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned base = group * 16 + j;
            fused_radix4<Inverse, 16>(v[base], v[base+4], v[base+8], v[base+12], j);
        }
    }
#pragma unroll
    for (unsigned j = 0; j < 16; ++j) {
        const auto a = v[j];
        const auto b = mul(v[j+16], twiddle<T>(j, 32, Inverse));
        v[j] = add(a,b); v[j+16] = sub(a,b);
    }
}

template <bool Inverse, bool Rows, bool Columns, typename T>
__launch_bounds__(128) __global__ void radix4_register_pair_kernel(
    const Cx<T>* input, Cx<T>* output, T factor) {
    // The same padded tile as the cuFFTDx Thread pair. The final natural-order
    // write is coalesced for both organizations; no hidden transpose kernel.
    __shared__ Cx<T> tile[32][33];
    const unsigned tid = threadIdx.x;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * 1024;
    for (unsigned i = tid; i < 1024; i += 128) tile[i/32][i%32] = input[base+i];
    __syncthreads();
    if constexpr (Rows) {
        if (tid < 32) {
            Cx<T> v[32];
#pragma unroll
            for (unsigned i = 0; i < 32; ++i) v[i] = tile[tid][i];
            register_radix4_32<Inverse>(v);
#pragma unroll
            for (unsigned i = 0; i < 32; ++i) tile[tid][i] = v[i];
        }
        __syncthreads();
    }
    if constexpr (Columns) {
        if (tid < 32) {
            Cx<T> v[32];
#pragma unroll
            for (unsigned i = 0; i < 32; ++i) v[i] = tile[i][tid];
            register_radix4_32<Inverse>(v);
#pragma unroll
            for (unsigned i = 0; i < 32; ++i) tile[i][tid] = v[i];
        }
        __syncthreads();
    }
    for (unsigned i = tid; i < 1024; i += 128) output[base+i] = scale(tile[i/32][i%32],factor);
}

template <bool Inverse>
void launch_radix4(const void* input, void* output, std::size_t planes,
                  float factor, cudaStream_t stream, XdStagePairVariant variant,
                  void* scratch) {
    const auto* in = static_cast<const Cx<float>*>(input);
    auto* out = static_cast<Cx<float>*>(output);
    const unsigned grid = static_cast<unsigned>(planes);
    switch (variant) {
    case XdStagePairVariant::Radix4Stage:
        radix4_stage_pair_kernel<Inverse><<<grid,1024,0,stream>>>(in,out,factor);
        break;
    case XdStagePairVariant::Radix4Register:
        radix4_register_pair_kernel<Inverse,true,true><<<grid,128,0,stream>>>(in,out,factor);
        break;
    case XdStagePairVariant::Radix4RegisterSplit: {
        if (!scratch || scratch == input || scratch == output)
            throw std::invalid_argument("split radix-4 needs distinct caller-owned scratch");
        auto* tmp = static_cast<Cx<float>*>(scratch);
        radix4_register_pair_kernel<Inverse,true,false><<<grid,128,0,stream>>>(in,tmp,1.0f);
        const auto first = cudaGetLastError();
        if (first != cudaSuccess) throw std::runtime_error(cudaGetErrorString(first));
        radix4_register_pair_kernel<Inverse,false,true><<<grid,128,0,stream>>>(tmp,out,factor);
        break;
    }
    default: throw std::invalid_argument("unknown experimental stage-pair variant");
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T>
void launch_impl(const void* input, void* output, std::size_t planes,
                 bool inverse, double factor, cudaStream_t stream) {
    constexpr unsigned N = 32;
    const std::size_t bytes = 2U * N * N * sizeof(Cx<T>);
    const unsigned grid = static_cast<unsigned>(planes);
    if (planes == 0 || planes > 2147483647ULL)
        throw std::invalid_argument("Stockham plane count is outside the CUDA grid range");
    stockham_pair_kernel<T, N><<<grid, N * N, bytes, stream>>>(
        static_cast<const Cx<T>*>(input), static_cast<Cx<T>*>(output),
        static_cast<std::uint64_t>(planes), inverse, static_cast<T>(factor));
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("launch Stockham pair FFT: ") +
                                 cudaGetErrorString(status));
}

}  // namespace

bool xd_stockham_pair_fft_available(std::size_t rows, std::size_t cols,
                                    bool fp64, XdStagePairVariant variant) noexcept {
    if (rows != 32 || cols != 32) return false;
    switch (variant) {
        case XdStagePairVariant::Radix4Register: return true;
        case XdStagePairVariant::Radix2:
        case XdStagePairVariant::Radix4Stage:
        case XdStagePairVariant::Radix4RegisterSplit: return !fp64;
    }
    return false;
}

void launch_xd_stockham_pair_fft(const void* input, void* output,
                                 std::size_t rows, std::size_t cols,
                                 std::size_t planes, bool fp64, bool inverse,
                                 double factor, cudaStream_t stream,
                                 XdStagePairVariant variant, void* scratch) {
    if (!xd_stockham_pair_fft_available(rows, cols, fp64, variant))
        throw std::invalid_argument("stage-pair supports 32x32 only; FP64 requires radix4-register");
    if (!input || !output || planes == 0 || planes > 2147483647ULL || !std::isfinite(factor))
        throw std::invalid_argument("invalid experimental stage-pair buffers, planes or scale");
    if (fp64) {
        const auto* in = static_cast<const Cx<double>*>(input);
        auto* out = static_cast<Cx<double>*>(output);
        const unsigned grid = static_cast<unsigned>(planes);
        if (inverse)
            radix4_register_pair_kernel<true,true,true><<<grid,128,0,stream>>>(in,out,factor);
        else
            radix4_register_pair_kernel<false,true,true><<<grid,128,0,stream>>>(in,out,factor);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    } else if (variant == XdStagePairVariant::Radix2)
        launch_impl<float>(input, output, planes, inverse, factor, stream);
    else if (inverse)
        launch_radix4<true>(input, output, planes, static_cast<float>(factor), stream, variant, scratch);
    else
        launch_radix4<false>(input, output, planes, static_cast<float>(factor), stream, variant, scratch);
}

}  // namespace cuntt::detail
