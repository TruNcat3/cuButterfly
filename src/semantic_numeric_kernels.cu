#include "semantic_numeric_kernels.hpp"

#include "operator_traits.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace cuntt::detail {
namespace {

constexpr unsigned kThreads = 256;

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("semantic numeric boundary size overflows size_t");
    return a * b;
}

void validate_count(std::size_t count, std::size_t element_bytes) {
    checked_product(count, element_bytes);
}

std::size_t validate_layout(const GeneralNdLayout& layout) {
    if (layout.rank == 0 || layout.rank > kGeneralMaxRank || layout.batch == 0)
        throw std::invalid_argument("semantic numeric boundary requires rank 1..8 and positive batch");
    std::size_t points = 1;
    for (std::uint32_t axis = 0; axis < layout.rank; ++axis) {
        if (layout.extents[axis] == 0)
            throw std::invalid_argument("semantic numeric boundary requires positive extents");
        points = checked_product(points, layout.extents[axis]);
    }
    if (points != layout.points)
        throw std::invalid_argument("semantic numeric boundary points differ from extent product");
    const auto count = checked_product(points, layout.batch);
    validate_count(count, sizeof(Complex16));
    return count;
}

unsigned blocks_for(std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>(65535, count / kThreads + (count % kThreads != 0)));
}

void check_launch(const char* operation) {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <typename From, typename To>
__global__ void cast_components_kernel(const From* input, To* output, std::size_t count) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride)
        output[index] = static_cast<To>(input[index]);
}

template <typename Storage>
__device__ Storage conjugate_imaginary(Storage value);

template <>
__device__ Fp16 conjugate_imaginary(Fp16 value) {
    return __ushort_as_half(static_cast<unsigned short>(__half_as_ushort(value) ^ 0x8000U));
}

template <>
__device__ Bf16 conjugate_imaginary(Bf16 value) {
    auto raw = static_cast<__nv_bfloat16_raw>(value);
    raw.x ^= 0x8000U;
    return Bf16(raw);
}

template <typename Storage, bool Pack>
__global__ void real_complex_kernel(const void* input, void* output, std::size_t count) {
    using Complex = typename traits::LowComplexType<Storage>::type;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        if constexpr (Pack) {
            auto& value = static_cast<Complex*>(output)[index];
            value.real = static_cast<const Storage*>(input)[index];
            value.imag = traits::LowStorage<Storage>::narrow(0.0f);
        } else {
            static_cast<Storage*>(output)[index] = static_cast<const Complex*>(input)[index].real;
        }
    }
}

template <typename Storage, bool Expand>
__global__ void half_spectrum_kernel(const void* input, void* output,
                                     GeneralNdLayout layout, std::size_t count) {
    using Complex = typename traits::LowComplexType<Storage>::type;
    const std::size_t last = layout.extents[layout.rank - 1];
    const std::size_t half = last / 2 + 1;
    const std::size_t compact_points = (layout.points / last) * half;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        if constexpr (!Expand) {
            const auto within = index % compact_points;
            const auto full_index = (index / compact_points) * layout.points
                                    + (within / half) * last + within % half;
            static_cast<Complex*>(output)[index] = static_cast<const Complex*>(input)[full_index];
        } else {
            const auto batch = index / layout.points;
            const auto within = index % layout.points;
            const bool mirror = within % last >= half;
            std::size_t source_full = within;
            if (mirror) {
                // Hermitian symmetry negates the complete frequency vector.
                std::size_t remaining = within, multiplier = 1;
                source_full = 0;
                for (std::uint32_t axis = layout.rank; axis-- > 0;) {
                    const auto extent = layout.extents[axis];
                    const auto coordinate = remaining % extent;
                    remaining /= extent;
                    source_full += (coordinate == 0 ? 0 : extent - coordinate) * multiplier;
                    multiplier *= extent;
                }
            }
            const auto compact_index = batch * compact_points
                                       + (source_full / last) * half + source_full % last;
            auto value = static_cast<const Complex*>(input)[compact_index];
            if (mirror)
                value.imag = conjugate_imaginary(value.imag);
            static_cast<Complex*>(output)[index] = value;
        }
    }
}

template <typename Storage>
__global__ void scale_components_kernel(Storage* values, std::size_t count, Storage scale) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride)
        values[index] = traits::LowStorage<Storage>::mul(values[index], scale);
}

template <typename Storage>
__global__ void scale_fp32_components_kernel(Storage* values, std::size_t count, float scale) {
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride)
        values[index] = traits::LowStorage<Storage>::narrow(traits::LowStorage<Storage>::widen(values[index]) * scale);
}

template <bool Pack>
void launch_real_complex(const void* input, void* output, const GeneralNdLayout& layout,
                         bool bf16, cudaStream_t stream) {
    const auto count = validate_layout(layout);
    if (bf16)
        real_complex_kernel<Bf16, Pack><<<blocks_for(count), kThreads, 0, stream>>>(input, output, count);
    else
        real_complex_kernel<Fp16, Pack><<<blocks_for(count), kThreads, 0, stream>>>(input, output, count);
    check_launch("low real/complex boundary");
}

template <bool Expand>
void launch_half_spectrum(const void* input, void* output, const GeneralNdLayout& layout,
                          bool bf16, cudaStream_t stream) {
    auto count = validate_layout(layout);
    if constexpr (!Expand) {
        const auto last = layout.extents[layout.rank - 1];
        count = checked_product((layout.points / last) * (last / 2 + 1), layout.batch);
    }
    if (bf16)
        half_spectrum_kernel<Bf16, Expand><<<blocks_for(count), kThreads, 0, stream>>>(input, output, layout, count);
    else
        half_spectrum_kernel<Fp16, Expand><<<blocks_for(count), kThreads, 0, stream>>>(input, output, layout, count);
    check_launch("low Hermitian spectrum boundary");
}

}  // namespace

void launch_fp32_to_fp64(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    validate_count(count, sizeof(double));
    if (count == 0) return;
    cast_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const float*>(input), static_cast<double*>(output), count);
    check_launch("FP32 to FP64 components");
}

void launch_fp64_to_fp32(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    validate_count(count, sizeof(double));
    if (count == 0) return;
    cast_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const double*>(input), static_cast<float*>(output), count);
    check_launch("FP64 to FP32 components");
}

void launch_low_real_to_complex(const void* input, void* output, const GeneralNdLayout& layout,
                                bool bf16, cudaStream_t stream) {
    launch_real_complex<true>(input, output, layout, bf16, stream);
}

void launch_low_complex_to_real(const void* input, void* output, const GeneralNdLayout& layout,
                                bool bf16, cudaStream_t stream) {
    launch_real_complex<false>(input, output, layout, bf16, stream);
}

void launch_low_extract_half_spectrum(const void* input, void* output, const GeneralNdLayout& layout,
                                      bool bf16, cudaStream_t stream) {
    launch_half_spectrum<false>(input, output, layout, bf16, stream);
}

void launch_low_expand_half_spectrum(const void* input, void* output, const GeneralNdLayout& layout,
                                     bool bf16, cudaStream_t stream) {
    launch_half_spectrum<true>(input, output, layout, bf16, stream);
}

void launch_low_output_scale(void* values, std::size_t count, double scale,
                             bool bf16, cudaStream_t stream, bool fp32_compute) {
    if (!std::isfinite(scale))
        throw std::invalid_argument("low output scale must be finite");
    validate_count(count, sizeof(Fp16));
    if (count == 0 || scale == 1.0) return;
    if (bf16 && fp32_compute)
        scale_fp32_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            static_cast<Bf16*>(values), count, static_cast<float>(scale));
    else if (fp32_compute)
        scale_fp32_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            static_cast<Fp16*>(values), count, static_cast<float>(scale));
    else if (bf16)
        scale_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            static_cast<Bf16*>(values), count, traits::LowStorage<Bf16>::narrow(static_cast<float>(scale)));
    else
        scale_components_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            static_cast<Fp16*>(values), count, traits::LowStorage<Fp16>::narrow(static_cast<float>(scale)));
    check_launch("low output scale");
}

}  // namespace cuntt::detail
