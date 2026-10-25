#include "xd_resident.hpp"

#include "ntt_traits.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cuntt {
namespace detail {
namespace {

constexpr std::uint32_t kMaxXdRank = kGeneralMaxRank;

template <typename Real>
struct XdDeviceMatrix {
    Real m00;
    Real m01;
    Real m10;
    Real m11;
};

template <typename Compute>
struct XdDeviceParams {
    std::uint32_t rank = 0;
    std::uint32_t extents[kMaxXdRank]{};
    std::uint32_t logs[kMaxXdRank]{};
    std::size_t strides[kMaxXdRank]{};
    std::size_t points = 0;
    std::uint64_t batch = 0;
    std::uint32_t axis_order[kMaxXdRank]{};
    std::uint32_t matrix_stride = 0;
    bool inverse = false;
    bool normalize_inverse = false;
    std::uint64_t modulus = 0;
    double floating_scale = 1.0;
    Compute modular_scale{};
    Compute modular_scale_shoup{};
    const Compute* twiddles[kMaxXdRank]{};
    const Compute* twiddles_shoup[kMaxXdRank]{};
    const XdDeviceMatrix<Compute>* matrices = nullptr;
};

__device__ __forceinline__ std::uint32_t xd_reverse_bits(std::uint32_t value,
                                                          std::uint32_t bits) {
    std::uint32_t result = 0;
#pragma unroll
    for (std::uint32_t bit = 0; bit < 32; ++bit) {
        if (bit < bits) {
            result = (result << 1U) | ((value >> bit) & 1U);
        }
    }
    return result;
}

template <typename Storage, typename Compute>
__device__ __forceinline__ Compute xd_load_value(const Storage* pointer) {
    if constexpr (std::is_same_v<Storage, Fp16>) {
        return static_cast<Compute>(__half2float(*pointer));
    } else if constexpr (std::is_same_v<Storage, Bf16>) {
        return static_cast<Compute>(__bfloat162float(*pointer));
    } else if constexpr (std::is_same_v<Storage, Complex16>) {
        Compute result;
        result.real = static_cast<decltype(result.real)>(__half2float(pointer->real));
        result.imag = static_cast<decltype(result.imag)>(__half2float(pointer->imag));
        return result;
    } else if constexpr (std::is_same_v<Storage, ComplexBf16>) {
        Compute result;
        result.real = static_cast<decltype(result.real)>(__bfloat162float(pointer->real));
        result.imag = static_cast<decltype(result.imag)>(__bfloat162float(pointer->imag));
        return result;
    } else {
        return static_cast<Compute>(*pointer);
    }
}

template <typename Storage, typename Compute>
__device__ __forceinline__ void xd_store_value(Storage* pointer, Compute value) {
    if constexpr (std::is_same_v<Storage, Fp16>) {
        *pointer = __float2half_rn(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Storage, Bf16>) {
        *pointer = __float2bfloat16_rn(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Storage, Complex16>) {
        pointer->real = __float2half_rn(static_cast<float>(value.real));
        pointer->imag = __float2half_rn(static_cast<float>(value.imag));
    } else if constexpr (std::is_same_v<Storage, ComplexBf16>) {
        pointer->real = __float2bfloat16_rn(static_cast<float>(value.real));
        pointer->imag = __float2bfloat16_rn(static_cast<float>(value.imag));
    } else {
        *pointer = static_cast<Storage>(value);
    }
}

template <typename Real>
__device__ __forceinline__ void xd_fft_butterfly(Real&, Real&, const Real&) {
    // The FFT specialization below is selected for the two complex storage
    // types.  Keeping this declaration out of the generic arithmetic paths
    // gives nvcc a concrete complex type at each instantiation.
}

__device__ __forceinline__ void xd_fft_butterfly(Complex32& left, Complex32& right,
                                                 const Complex32& twiddle) {
    const Complex32 value{right.real * twiddle.real - right.imag * twiddle.imag,
                          right.real * twiddle.imag + right.imag * twiddle.real};
    const Complex32 old = left;
    left = {old.real + value.real, old.imag + value.imag};
    right = {old.real - value.real, old.imag - value.imag};
}

__device__ __forceinline__ void xd_fft_butterfly(Complex64& left, Complex64& right,
                                                 const Complex64& twiddle) {
    const Complex64 value{right.real * twiddle.real - right.imag * twiddle.imag,
                          right.real * twiddle.imag + right.imag * twiddle.real};
    const Complex64 old = left;
    left = {old.real + value.real, old.imag + value.imag};
    right = {old.real - value.real, old.imag - value.imag};
}

template <typename Word>
__device__ __forceinline__ Word xd_ntt_product(Word value, Word twiddle, Word shoup,
                                               Word modulus) {
    return traits::mul_shoup(value, twiddle, shoup, modulus);
}

template <typename Word>
__device__ __forceinline__ void xd_ntt_butterfly(Word& left, Word& right, Word twiddle,
                                                 Word twiddle_shoup, Word modulus) {
    const Word u = left;
    const Word v = xd_ntt_product(right, twiddle, twiddle_shoup, modulus);
    if constexpr (sizeof(Word) == sizeof(std::uint32_t)) {
        const std::uint64_t sum = static_cast<std::uint64_t>(u) + static_cast<std::uint64_t>(v);
        left = static_cast<Word>(sum >= modulus ? sum - modulus : sum);
        right = u >= v ? static_cast<Word>(u - v)
                       : static_cast<Word>(static_cast<std::uint64_t>(modulus) + u - v);
    } else {
        const Word sum = static_cast<Word>(u + v);
        left = sum >= modulus ? static_cast<Word>(sum - modulus) : sum;
        right = u >= v ? static_cast<Word>(u - v) : static_cast<Word>(modulus + u - v);
    }
}

template <typename Compute, XdOperator Op>
__device__ __forceinline__ void xd_apply_butterfly(
    Compute& left, Compute& right, const XdDeviceParams<Compute>& params,
    std::uint32_t axis, std::uint32_t stage, std::uint32_t offset) {
    if constexpr (Op == XdOperator::Fft) {
        const std::uint32_t length = params.extents[axis];
        const std::uint32_t half = 1U << stage;
        const std::uint32_t table_index = offset * (length / (half << 1U));
        xd_fft_butterfly(left, right, params.twiddles[axis][table_index]);
    } else if constexpr (Op == XdOperator::Ntt) {
        const std::uint32_t length = params.extents[axis];
        const std::uint32_t half = 1U << stage;
        const std::uint32_t table_index = offset * (length / (half << 1U));
        xd_ntt_butterfly(left, right, params.twiddles[axis][table_index],
                         params.twiddles_shoup[axis][table_index],
                         static_cast<Compute>(params.modulus));
    } else if constexpr (Op == XdOperator::Fwht) {
        const Compute old = left;
        left = old + right;
        right = old - right;
    } else if constexpr (Op == XdOperator::SubsetZeta) {
        if (params.inverse) {
            right = right - left;
        } else {
            right = right + left;
        }
    } else if constexpr (Op == XdOperator::SupersetZeta) {
        if (params.inverse) {
            left = left - right;
        } else {
            left = left + right;
        }
    } else if constexpr (Op == XdOperator::Structured2x2) {
        const auto matrix = params.matrices[axis * params.matrix_stride + stage];
        Compute m00 = matrix.m00;
        Compute m01 = matrix.m01;
        Compute m10 = matrix.m10;
        Compute m11 = matrix.m11;
        if (params.inverse) {
            const Compute determinant = m00 * m11 - m01 * m10;
            const Compute old_m00 = m00;
            const Compute old_m01 = m01;
            m00 = m11 / determinant;
            m01 = -old_m01 / determinant;
            m10 = -m10 / determinant;
            m11 = old_m00 / determinant;
        }
        const Compute old_left = left;
        const Compute old_right = right;
        left = m00 * old_left + m01 * old_right;
        right = m10 * old_left + m11 * old_right;
    }
}

template <typename Compute, XdOperator Op>
__device__ __forceinline__ Compute xd_finalize_value(
    Compute value, const XdDeviceParams<Compute>& params) {
    if constexpr (Op == XdOperator::Fft) {
        if (params.inverse && params.normalize_inverse) {
            value.real = static_cast<decltype(value.real)>(value.real * params.floating_scale);
            value.imag = static_cast<decltype(value.imag)>(value.imag * params.floating_scale);
        }
    } else if constexpr (Op == XdOperator::Fwht) {
        if (params.inverse && params.normalize_inverse) {
            value = static_cast<Compute>(value * params.floating_scale);
        }
    } else if constexpr (Op == XdOperator::Ntt) {
        if (params.inverse && params.normalize_inverse) {
            value = xd_ntt_product(value, params.modular_scale,
                                   params.modular_scale_shoup,
                                   static_cast<Compute>(params.modulus));
        }
    }
    return value;
}

template <typename Storage, typename Compute, XdOperator Op>
__global__ void xd_resident_kernel(const Storage* input, Storage* output,
                                   XdDeviceParams<Compute> params) {
    extern __shared__ __align__(16) unsigned char shared_storage[];
    Compute* values = reinterpret_cast<Compute*>(shared_storage);
    const std::uint64_t batch_index = blockIdx.x;
    if (batch_index >= params.batch) {
        return;
    }

    const std::size_t batch_base = static_cast<std::size_t>(batch_index) * params.points;
    constexpr bool bit_reverse_input = Op == XdOperator::Fft || Op == XdOperator::Ntt;

    for (std::size_t linear = threadIdx.x; linear < params.points; linear += blockDim.x) {
        std::size_t source = 0;
        for (std::uint32_t axis = 0; axis < params.rank; ++axis) {
            const std::size_t coordinate =
                (linear / params.strides[axis]) % params.extents[axis];
            const auto source_coordinate = bit_reverse_input
                                               ? xd_reverse_bits(static_cast<std::uint32_t>(coordinate),
                                                                 params.logs[axis])
                                               : static_cast<std::uint32_t>(coordinate);
            source += static_cast<std::size_t>(source_coordinate) * params.strides[axis];
        }
        values[linear] = xd_load_value<Storage, Compute>(&input[batch_base + source]);
        if constexpr (Op == XdOperator::Ntt) {
            if (values[linear] >= static_cast<Compute>(params.modulus)) {
                values[linear] = static_cast<Compute>(values[linear] %
                                                       static_cast<Compute>(params.modulus));
            }
        }
    }
    __syncthreads();

    const bool reverse_stage_order = params.inverse &&
                                    (Op == XdOperator::Fwht ||
                                     Op == XdOperator::SubsetZeta ||
                                     Op == XdOperator::SupersetZeta ||
                                     Op == XdOperator::Structured2x2);
    for (std::uint32_t axis_position = 0; axis_position < params.rank; ++axis_position) {
        const std::uint32_t axis = params.axis_order[axis_position];
        const std::uint32_t log_extent = params.logs[axis];
        for (std::uint32_t stage_position = 0; stage_position < log_extent; ++stage_position) {
            const std::uint32_t stage = reverse_stage_order
                                             ? log_extent - 1U - stage_position
                                             : stage_position;
            const std::size_t half = std::size_t{1} << stage;
            const std::size_t stride = params.strides[axis];
            const std::size_t pair_count = params.points / 2U;
            for (std::size_t pair = threadIdx.x; pair < pair_count; pair += blockDim.x) {
                const std::size_t bit = half * stride;
                const std::size_t left_index = (pair / bit) * (bit << 1U) + (pair % bit);
                const std::size_t right_index = left_index + bit;
                const std::size_t offset = (left_index / stride) % half;
                Compute left = values[left_index];
                Compute right = values[right_index];
                xd_apply_butterfly<Compute, Op>(left, right, params, axis, stage,
                                                 static_cast<std::uint32_t>(offset));
                values[left_index] = left;
                values[right_index] = right;
            }
            __syncthreads();
        }
    }

    for (std::size_t linear = threadIdx.x; linear < params.points; linear += blockDim.x) {
        xd_store_value<Storage, Compute>(&output[batch_base + linear],
                                          xd_finalize_value<Compute, Op>(values[linear], params));
    }
}

std::string cuda_error(cudaError_t status, const char* operation) {
    std::ostringstream stream;
    stream << operation << ": " << cudaGetErrorString(status);
    return stream.str();
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cuda_error(status, operation));
    }
}

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1U)) == 0;
}

std::uint32_t log_two(std::size_t value) {
    std::uint32_t result = 0;
    while (value > 1) {
        value >>= 1U;
        ++result;
    }
    return result;
}

std::uint64_t mod_pow_host(std::uint64_t base, std::uint64_t exponent,
                           std::uint64_t modulus) {
    std::uint64_t result = 1 % modulus;
    base %= modulus;
    while (exponent != 0) {
        if (exponent & 1U) {
            result = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(result) * base) % modulus);
        }
        exponent >>= 1U;
        if (exponent != 0) {
            base = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(base) * base) % modulus);
        }
    }
    return result;
}

bool is_prime_host(std::uint64_t value) {
    if (value < 2) return false;
    constexpr std::uint64_t small_primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (const auto prime : small_primes) {
        if (value % prime == 0) return value == prime;
    }
    std::uint64_t odd_part = value - 1;
    std::uint32_t powers = 0;
    while ((odd_part & 1U) == 0) {
        odd_part >>= 1U;
        ++powers;
    }
    constexpr std::uint64_t witnesses[] = {
        2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (const auto witness : witnesses) {
        if (witness % value == 0) continue;
        auto residue = mod_pow_host(witness, odd_part, value);
        if (residue == 1 || residue == value - 1) continue;
        bool passed = false;
        for (std::uint32_t round = 1; round < powers; ++round) {
            residue = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(residue) * residue) % value);
            if (residue == value - 1) {
                passed = true;
                break;
            }
        }
        if (!passed) return false;
    }
    return true;
}

std::uint64_t primitive_power_two_root(std::uint32_t log_extent,
                                       std::uint64_t modulus) {
    if (log_extent == 0) {
        return 1;
    }
    const std::uint64_t order = std::uint64_t{1} << log_extent;
    if ((modulus - 1U) % order != 0) {
        return 0;
    }
    for (std::uint64_t candidate = 2; candidate < (std::uint64_t{1} << 20); ++candidate) {
        const auto root = mod_pow_host(candidate, (modulus - 1U) / order, modulus);
        if (mod_pow_host(root, order, modulus) == 1 &&
            mod_pow_host(root, order >> 1U, modulus) != 1) {
            return root;
        }
    }
    return 0;
}

std::uint64_t inverse_mod_host(std::uint64_t value, std::uint64_t modulus) {
    return mod_pow_host(value, modulus - 2U, modulus);
}

template <typename Word>
Word shoup_host(Word value, std::uint64_t modulus) {
    if constexpr (sizeof(Word) == sizeof(std::uint32_t)) {
        return static_cast<Word>((static_cast<unsigned __int128>(value) << 32U) / modulus);
    }
    return static_cast<Word>((static_cast<unsigned __int128>(value) << 64U) / modulus);
}

const char* operator_name(XdOperator op) {
    switch (op) {
        case XdOperator::Fft: return "fft";
        case XdOperator::Fwht: return "fwht";
        case XdOperator::SubsetZeta: return "subset-zeta";
        case XdOperator::SupersetZeta: return "superset-zeta";
        case XdOperator::Structured2x2: return "structured-2x2";
        case XdOperator::Ntt: return "ntt";
    }
    return "unknown";
}

const char* storage_name(XdStorageType type) {
    switch (type) {
        case XdStorageType::Fp16: return "fp16";
        case XdStorageType::Bf16: return "bf16";
        case XdStorageType::Fp32: return "fp32";
        case XdStorageType::Fp64: return "fp64";
        case XdStorageType::ComplexFp32: return "complex-fp32";
        case XdStorageType::ComplexFp64: return "complex-fp64";
        case XdStorageType::Uint32: return "uint32";
        case XdStorageType::Uint64: return "uint64";
    }
    return "unknown";
}

const char* compute_name(XdComputeType type) {
    switch (type) {
        case XdComputeType::Native: return "native";
        case XdComputeType::Fp32: return "fp32";
        case XdComputeType::Fp64: return "fp64";
        case XdComputeType::Uint32: return "uint32";
        case XdComputeType::Uint64: return "uint64";
    }
    return "unknown";
}

std::size_t storage_bytes(XdStorageType type) {
    switch (type) {
        case XdStorageType::Fp16: return sizeof(Fp16);
        case XdStorageType::Bf16: return sizeof(Bf16);
        case XdStorageType::Fp32: return sizeof(float);
        case XdStorageType::Fp64: return sizeof(double);
        case XdStorageType::ComplexFp32: return sizeof(Complex32);
        case XdStorageType::ComplexFp64: return sizeof(Complex64);
        case XdStorageType::Uint32: return sizeof(std::uint32_t);
        case XdStorageType::Uint64: return sizeof(std::uint64_t);
    }
    return 0;
}

std::size_t compute_bytes(const XdResidentConfig& config) {
    switch (config.op) {
        case XdOperator::Fft:
            return config.storage == XdStorageType::ComplexFp64 ? sizeof(Complex64)
                                                                  : sizeof(Complex32);
        case XdOperator::Fwht:
            return config.storage == XdStorageType::Fp64 ? sizeof(double) : sizeof(float);
        case XdOperator::Structured2x2:
            return config.storage == XdStorageType::Fp64 ? sizeof(double) : sizeof(float);
        case XdOperator::SubsetZeta:
        case XdOperator::SupersetZeta:
            return storage_bytes(config.storage);
        case XdOperator::Ntt:
            return config.storage == XdStorageType::Uint64 ? sizeof(std::uint64_t)
                                                            : sizeof(std::uint32_t);
    }
    return 0;
}

std::size_t resident_shared_bytes(const XdResidentConfig& config,
                                  std::size_t points) {
    return points * compute_bytes(config);
}

std::size_t twiddle_count_for_config(const XdResidentConfig& config) {
    if (config.op != XdOperator::Fft && config.op != XdOperator::Ntt) return 0;
    std::size_t count = 0;
    for (const auto extent : config.extents) count += extent / 2U;
    return count;
}

XdResidentAvailability semantic_check(const XdResidentConfig& config) {
    XdResidentAvailability result;
    if (config.extents.size() < 2 || config.extents.size() > kMaxXdRank) {
        result.reason = "xD resident rank exceeds the common multidimensional layout contract";
        return result;
    }
    if (config.batch == 0) {
        result.reason = "xD resident batch must be positive";
        return result;
    }
    if (config.threads == 0 || config.threads % 32 != 0) {
        result.reason = "xD resident threads must be a positive warp multiple";
        return result;
    }
    if (!config.axis_order.empty()) {
        auto order = config.axis_order;
        std::sort(order.begin(), order.end());
        if (order.size() != config.extents.size()) {
            result.reason = "xD resident axis order must cover every axis";
            return result;
        }
        for (std::size_t axis = 0; axis < order.size(); ++axis) {
            if (order[axis] != axis) {
                result.reason = "xD resident axis order must be a permutation";
                return result;
            }
        }
    }
    std::size_t points = 1;
    for (const auto extent : config.extents) {
        if (!is_power_of_two(extent) || extent > (std::size_t{1} << 30)) {
            result.reason = "xD resident extents must be nonzero powers of two up to 2^30";
            return result;
        }
        if (points > std::numeric_limits<std::size_t>::max() / extent) {
            result.reason = "xD resident point count overflows size_t";
            return result;
        }
        points *= extent;
    }
    if (points == 0 || points > std::numeric_limits<std::uint32_t>::max()) {
        result.reason = "xD resident point count exceeds the CTA index contract";
        return result;
    }
    if (config.batch > std::numeric_limits<std::size_t>::max() / points) {
        result.reason = "xD resident batch data extent overflows size_t";
        return result;
    }

    const bool complex_fft = config.op == XdOperator::Fft;
    if (complex_fft && config.storage != XdStorageType::ComplexFp32 &&
        config.storage != XdStorageType::ComplexFp64) {
        result.reason = "xD FFT requires ComplexFp32 or ComplexFp64 storage";
        return result;
    }
    if (complex_fft && config.storage == XdStorageType::ComplexFp32 &&
        config.compute != XdComputeType::Native && config.compute != XdComputeType::Fp32) {
        result.reason = "ComplexFp32 FFT requires native or FP32 compute";
        return result;
    }
    if (complex_fft && config.storage == XdStorageType::ComplexFp64 &&
        config.compute != XdComputeType::Native && config.compute != XdComputeType::Fp64) {
        result.reason = "ComplexFp64 FFT requires native or FP64 compute";
        return result;
    }
    if (config.op == XdOperator::Fwht) {
        const bool real_storage = config.storage == XdStorageType::Fp16 ||
                                  config.storage == XdStorageType::Bf16 ||
                                  config.storage == XdStorageType::Fp32 ||
                                  config.storage == XdStorageType::Fp64;
        if (!real_storage) {
            result.reason = "xD FWHT requires FP16, BF16, FP32, or FP64 storage";
            return result;
        }
        if ((config.storage == XdStorageType::Fp16 || config.storage == XdStorageType::Bf16) &&
            config.compute != XdComputeType::Fp32) {
            result.reason = "low precision xD FWHT requires explicit FP32 shared accumulation";
            return result;
        }
        if (config.storage == XdStorageType::Fp32 && config.compute != XdComputeType::Native &&
            config.compute != XdComputeType::Fp32) {
            result.reason = "FP32 xD FWHT requires native or FP32 compute";
            return result;
        }
        if (config.storage == XdStorageType::Fp64 && config.compute != XdComputeType::Native &&
            config.compute != XdComputeType::Fp64) {
            result.reason = "FP64 xD FWHT requires native or FP64 compute";
            return result;
        }
    }
    if (config.op == XdOperator::Structured2x2 &&
        config.storage != XdStorageType::Fp32 && config.storage != XdStorageType::Fp64) {
        result.reason = "xD structured 2x2 supports FP32 and FP64 storage";
        return result;
    }
    if (config.op == XdOperator::Structured2x2 &&
        ((config.storage == XdStorageType::Fp32 && config.compute != XdComputeType::Native &&
          config.compute != XdComputeType::Fp32) ||
         (config.storage == XdStorageType::Fp64 && config.compute != XdComputeType::Native &&
          config.compute != XdComputeType::Fp64))) {
        result.reason = "structured xD compute type must match storage";
        return result;
    }
    if ((config.op == XdOperator::SubsetZeta || config.op == XdOperator::SupersetZeta) &&
        config.storage != XdStorageType::Uint32 && config.storage != XdStorageType::Uint64) {
        result.reason = "xD zeta transforms require uint32 or uint64 storage";
        return result;
    }
    if (config.op == XdOperator::SubsetZeta || config.op == XdOperator::SupersetZeta) {
        const auto expected = config.storage == XdStorageType::Uint64 ? XdComputeType::Uint64 : XdComputeType::Uint32;
        if (config.compute != XdComputeType::Native && config.compute != expected) {
            result.reason = "xD zeta compute type must match storage";
            return result;
        }
    }
    if (config.op == XdOperator::Ntt) {
        if (config.storage != XdStorageType::Uint32 && config.storage != XdStorageType::Uint64) {
            result.reason = "xD NTT requires uint32 or uint64 storage";
            return result;
        }
        if ((config.storage == XdStorageType::Uint32 && config.compute != XdComputeType::Native &&
             config.compute != XdComputeType::Uint32) ||
            (config.storage == XdStorageType::Uint64 && config.compute != XdComputeType::Native &&
             config.compute != XdComputeType::Uint64)) {
            result.reason = "xD NTT compute type must match storage";
            return result;
        }
        if (config.modulus <= 2 || config.modulus >= (std::uint64_t{1} << 63) ||
            (config.storage == XdStorageType::Uint32 && config.modulus > std::numeric_limits<std::uint32_t>::max())) {
            result.reason = "xD NTT requires a modulus in the supported word range";
            return result;
        }
        std::uint32_t max_log = 0;
        for (const auto extent : config.extents) {
            max_log = std::max(max_log, log_two(extent));
        }
        if (!is_prime_host(config.modulus)) {
            result.reason = "xD NTT modulus must be prime";
            return result;
        }
        if (max_log != 0 && primitive_power_two_root(max_log, config.modulus) == 0) {
            result.reason = "xD NTT modulus has no primitive root for the largest axis";
            return result;
        }
        if (config.inverse && config.normalize_inverse && config.modulus != 0 &&
            points % config.modulus == 0) {
            result.reason = "xD NTT inverse point count is not invertible modulo modulus";
            return result;
        }
    }
    if (config.op == XdOperator::Structured2x2) {
        std::uint32_t max_log = 0;
        std::size_t stage_count = 0;
        for (const auto extent : config.extents) {
            const auto stages = log_two(extent);
            max_log = std::max(max_log, stages);
            stage_count += stages;
        }
        const auto matrix_size = config.stage_matrices.size();
        if (matrix_size != 0 && matrix_size != 1 && matrix_size != stage_count &&
            matrix_size != max_log && matrix_size != config.extents.size() * max_log) {
            result.reason = "structured xD matrices must broadcast or cover every axis stage";
            return result;
        }
        for (const auto& matrix : config.stage_matrices) {
            if (!std::isfinite(matrix.m00) || !std::isfinite(matrix.m01) ||
                !std::isfinite(matrix.m10) || !std::isfinite(matrix.m11)) {
                result.reason = "structured xD matrix entries must be finite";
                return result;
            }
            if (config.inverse && matrix.m00 * matrix.m11 - matrix.m01 * matrix.m10 == 0.0) {
                result.reason = "structured xD inverse requires nonsingular matrices";
                return result;
            }
        }
    }
    return result;
}

}  // namespace

}  // namespace detail

using namespace detail;

class XdResidentPlan::Impl {
  public:
    explicit Impl(XdResidentConfig config) : config_(std::move(config)) {
        const auto semantic = semantic_check(config_);
        if (!semantic.reason.empty()) {
            reason_ = semantic.reason;
            return;
        }
        if (config_.axis_order.empty()) {
            const bool reverse = config_.inverse && config_.op != XdOperator::Fft &&
                                 config_.op != XdOperator::Ntt;
            for (std::size_t axis = 0; axis < config_.extents.size(); ++axis)
                config_.axis_order.push_back(reverse ? config_.extents.size() - 1 - axis : axis);
        }
        points_ = 1;
        for (const auto extent : config_.extents) {
            points_ *= extent;
        }
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            logs_[axis] = log_two(config_.extents[axis]);
            std::size_t stride = 1;
            for (std::size_t tail = axis + 1; tail < config_.extents.size(); ++tail) {
                stride *= config_.extents[tail];
            }
            strides_[axis] = stride;
            max_log_ = std::max(max_log_, logs_[axis]);
        }
        shared_bytes_ = resident_shared_bytes(config_, points_);
        matrix_stride_ = std::max(1U, max_log_);
        threads_ = config_.threads;
        std::size_t coefficient_bytes = 0;
        if (config_.op == XdOperator::Fft || config_.op == XdOperator::Ntt) {
            coefficient_bytes = twiddle_count_for_config(config_) *
                                storage_bytes(config_.storage) *
                                (config_.op == XdOperator::Ntt ? 2U : 1U);
        } else if (config_.op == XdOperator::Structured2x2) {
            coefficient_bytes = config_.extents.size() * matrix_stride_ *
                                compute_bytes(config_);
        }
        resource_bytes_ = shared_bytes_ + coefficient_bytes;

        int device = 0;
        cudaError_t status = cudaGetDevice(&device);
        if (status != cudaSuccess) {
            reason_ = cuda_error(status, "query CUDA device");
            return;
        }
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) {
            reason_ = cuda_error(status, "query CUDA device properties");
            return;
        }
        if (config_.threads > static_cast<std::uint32_t>(properties.maxThreadsPerBlock)) {
            std::ostringstream stream;
            stream << "xD resident threads " << config_.threads
                   << " exceed device maxThreadsPerBlock " << properties.maxThreadsPerBlock;
            reason_ = stream.str();
            return;
        }
        if (shared_bytes_ > static_cast<std::size_t>(properties.sharedMemPerBlock)) {
            std::ostringstream stream;
            stream << "xD resident shared bytes " << shared_bytes_
                   << " exceed device sharedMemPerBlock " << properties.sharedMemPerBlock;
            reason_ = stream.str();
            return;
        }
        if (config_.batch > static_cast<std::size_t>(properties.maxGridSize[0])) {
            reason_ = "xD resident batch exceeds device grid.x";
            return;
        }
        try {
            build_coefficients();
        } catch (const std::exception& error) {
            reason_ = error.what();
            return;
        }
        available_ = true;
        reason_ = "shared resident multidimensional radix-2 kernel is available";
    }

    ~Impl() {
        if (device_twiddles_ != nullptr) cudaFree(device_twiddles_);
        if (device_twiddles_shoup_ != nullptr) cudaFree(device_twiddles_shoup_);
        if (device_matrices_ != nullptr) cudaFree(device_matrices_);
    }

    template <typename T>
    void upload_twiddles(const std::vector<T>& values, const std::vector<T>* shoup = nullptr) {
        if (!values.empty()) {
            check_cuda(cudaMalloc(&device_twiddles_, values.size() * sizeof(T)),
                       "allocate xD resident twiddles");
            check_cuda(cudaMemcpy(device_twiddles_, values.data(), values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       "upload xD resident twiddles");
        }
        if (shoup != nullptr && !shoup->empty()) {
            check_cuda(cudaMalloc(&device_twiddles_shoup_, shoup->size() * sizeof(T)),
                       "allocate xD resident Shoup twiddles");
            check_cuda(cudaMemcpy(device_twiddles_shoup_, shoup->data(),
                                  shoup->size() * sizeof(T), cudaMemcpyHostToDevice),
                       "upload xD resident Shoup twiddles");
        }
        twiddle_bytes_ = values.size() * sizeof(T) +
                         (shoup == nullptr ? 0 : shoup->size() * sizeof(T));
    }

    template <typename Real>
    void upload_matrices(const std::vector<XdDeviceMatrix<Real>>& values) {
        if (values.empty()) return;
        check_cuda(cudaMalloc(&device_matrices_, values.size() * sizeof(XdDeviceMatrix<Real>)),
                   "allocate xD resident matrices");
        check_cuda(cudaMemcpy(device_matrices_, values.data(),
                              values.size() * sizeof(XdDeviceMatrix<Real>),
                              cudaMemcpyHostToDevice),
                   "upload xD resident matrices");
        matrix_bytes_ = values.size() * sizeof(XdDeviceMatrix<Real>);
    }

    template <typename Complex>
    void build_fft_coefficients() {
        std::vector<Complex> values;
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            axis_offsets_[axis] = values.size();
            const std::size_t extent = config_.extents[axis];
            const double sign = config_.inverse ? 1.0 : -1.0;
            for (std::size_t index = 0; index < extent / 2U; ++index) {
                const double angle = sign * 2.0 * 3.141592653589793238462643383279502884 *
                                     static_cast<double>(index) / static_cast<double>(extent);
                Complex value{};
                using Real = decltype(value.real);
                value.real = static_cast<Real>(std::cos(angle));
                value.imag = static_cast<Real>(std::sin(angle));
                values.push_back(value);
            }
        }
        upload_twiddles(values);
    }

    template <typename Word>
    void build_ntt_coefficients() {
        std::vector<Word> values;
        std::vector<Word> shoup;
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            axis_offsets_[axis] = values.size();
            const std::size_t extent = config_.extents[axis];
            std::uint64_t axis_root = primitive_power_two_root(logs_[axis], config_.modulus);
            if (config_.inverse && logs_[axis] != 0) {
                axis_root = inverse_mod_host(axis_root, config_.modulus);
            }
            for (std::size_t index = 0; index < extent / 2U; ++index) {
                const auto coefficient = mod_pow_host(axis_root, index, config_.modulus);
                values.push_back(static_cast<Word>(coefficient));
                shoup.push_back(shoup_host(static_cast<Word>(coefficient), config_.modulus));
            }
        }
        upload_twiddles(values, &shoup);
        if (config_.inverse && config_.normalize_inverse) {
            const auto inverse_points = inverse_mod_host(points_ % config_.modulus, config_.modulus);
            modular_scale_ = static_cast<Word>(inverse_points);
            modular_scale_shoup_ = shoup_host(static_cast<Word>(inverse_points), config_.modulus);
        } else {
            modular_scale_ = static_cast<Word>(1);
            modular_scale_shoup_ = shoup_host(static_cast<Word>(1), config_.modulus);
        }
    }

    template <typename Real>
    void build_matrix_coefficients() {
        std::vector<XdDeviceMatrix<Real>> values(
            config_.extents.size() * matrix_stride_, XdDeviceMatrix<Real>{1, 0, 0, 1});
        std::size_t stage_count = 0;
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            stage_count += logs_[axis];
        }
        const bool flattened = config_.stage_matrices.size() == stage_count;
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            const std::size_t inner_axis = config_.extents.size() - 1U - axis;
            std::size_t inner_prefix = 0;
            for (std::size_t prefix_axis = config_.extents.size() - 1U;
                 prefix_axis > axis; --prefix_axis) {
                inner_prefix += logs_[prefix_axis];
            }
            for (std::uint32_t stage = 0; stage < logs_[axis]; ++stage) {
                ButterflyMatrix2x2 source{};
                if (config_.stage_matrices.size() == 1) {
                    source = config_.stage_matrices.front();
                } else if (flattened) {
                    source = config_.stage_matrices[inner_prefix + stage];
                } else if (config_.stage_matrices.size() == max_log_) {
                    source = config_.stage_matrices[stage];
                } else if (config_.stage_matrices.size() == config_.extents.size() * max_log_) {
                    source = config_.stage_matrices[inner_axis * max_log_ + stage];
                }
                values[axis * matrix_stride_ + stage] = {
                    static_cast<Real>(source.m00), static_cast<Real>(source.m01),
                    static_cast<Real>(source.m10), static_cast<Real>(source.m11)};
            }
        }
        upload_matrices(values);
    }

    void build_coefficients() {
        if (config_.op == XdOperator::Fft) {
            if (config_.storage == XdStorageType::ComplexFp32) {
                build_fft_coefficients<Complex32>();
            } else {
                build_fft_coefficients<Complex64>();
            }
        } else if (config_.op == XdOperator::Ntt) {
            if (config_.storage == XdStorageType::Uint32) {
                build_ntt_coefficients<std::uint32_t>();
            } else {
                build_ntt_coefficients<std::uint64_t>();
            }
        } else if (config_.op == XdOperator::Structured2x2) {
            if (config_.storage == XdStorageType::Fp32) {
                build_matrix_coefficients<float>();
            } else {
                build_matrix_coefficients<double>();
            }
        }
        const std::size_t coefficient_bytes = twiddle_bytes_ + matrix_bytes_;
        resource_bytes_ = shared_bytes_ + coefficient_bytes;
    }

    template <typename Compute>
    XdDeviceParams<Compute> make_params() const {
        XdDeviceParams<Compute> params;
        params.rank = static_cast<std::uint32_t>(config_.extents.size());
        params.points = points_;
        params.batch = config_.batch;
        params.inverse = config_.inverse;
        params.normalize_inverse = config_.normalize_inverse;
        params.matrix_stride = matrix_stride_;
        params.modulus = config_.modulus;
        params.floating_scale = (config_.inverse && config_.normalize_inverse &&
                                 (config_.op == XdOperator::Fft || config_.op == XdOperator::Fwht))
                                    ? 1.0 / static_cast<double>(points_)
                                    : 1.0;
        if constexpr (std::is_integral_v<Compute>) {
            params.modular_scale = static_cast<Compute>(modular_scale_);
            params.modular_scale_shoup = static_cast<Compute>(modular_scale_shoup_);
        }
        for (std::size_t axis = 0; axis < config_.extents.size(); ++axis) {
            params.extents[axis] = static_cast<std::uint32_t>(config_.extents[axis]);
            params.logs[axis] = logs_[axis];
            params.strides[axis] = strides_[axis];
            params.axis_order[axis] = static_cast<std::uint32_t>(config_.axis_order[axis]);
            if (device_twiddles_ != nullptr) {
                params.twiddles[axis] = static_cast<const Compute*>(device_twiddles_) + axis_offsets_[axis];
            }
            if (device_twiddles_shoup_ != nullptr) {
                params.twiddles_shoup[axis] = static_cast<const Compute*>(device_twiddles_shoup_) + axis_offsets_[axis];
            }
        }
        if (device_matrices_ != nullptr) {
            params.matrices = static_cast<const XdDeviceMatrix<Compute>*>(device_matrices_);
        }
        return params;
    }

    template <typename Storage, typename Compute, XdOperator Op>
    void launch(const void* input, void* output, cudaStream_t stream) const {
        const auto params = make_params<Compute>();
        xd_resident_kernel<Storage, Compute, Op>
            <<<static_cast<unsigned int>(config_.batch), threads_, shared_bytes_, stream>>>(
                static_cast<const Storage*>(input), static_cast<Storage*>(output), params);
        check_cuda(cudaGetLastError(), "launch xD resident kernel");
    }

    void execute(const void* input, void* output, cudaStream_t stream) const {
        if (!available_) {
            throw std::runtime_error("xD resident plan unavailable: " + reason_);
        }
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("xD resident input and output must be non-null");
        }
        switch (config_.op) {
            case XdOperator::Fft:
                if (config_.storage == XdStorageType::ComplexFp32) {
                    launch<Complex32, Complex32, XdOperator::Fft>(input, output, stream);
                } else {
                    launch<Complex64, Complex64, XdOperator::Fft>(input, output, stream);
                }
                return;
            case XdOperator::Fwht:
                if (config_.storage == XdStorageType::Fp16) {
                    launch<Fp16, float, XdOperator::Fwht>(input, output, stream);
                } else if (config_.storage == XdStorageType::Bf16) {
                    launch<Bf16, float, XdOperator::Fwht>(input, output, stream);
                } else if (config_.storage == XdStorageType::Fp32) {
                    launch<float, float, XdOperator::Fwht>(input, output, stream);
                } else {
                    launch<double, double, XdOperator::Fwht>(input, output, stream);
                }
                return;
            case XdOperator::SubsetZeta:
                if (config_.storage == XdStorageType::Uint64)
                    launch<std::uint64_t, std::uint64_t, XdOperator::SubsetZeta>(input, output, stream);
                else
                    launch<std::uint32_t, std::uint32_t, XdOperator::SubsetZeta>(input, output, stream);
                return;
            case XdOperator::SupersetZeta:
                if (config_.storage == XdStorageType::Uint64)
                    launch<std::uint64_t, std::uint64_t, XdOperator::SupersetZeta>(input, output, stream);
                else
                    launch<std::uint32_t, std::uint32_t, XdOperator::SupersetZeta>(input, output, stream);
                return;
            case XdOperator::Structured2x2:
                if (config_.storage == XdStorageType::Fp32) {
                    launch<float, float, XdOperator::Structured2x2>(input, output, stream);
                } else {
                    launch<double, double, XdOperator::Structured2x2>(input, output, stream);
                }
                return;
            case XdOperator::Ntt:
                if (config_.storage == XdStorageType::Uint32) {
                    launch<std::uint32_t, std::uint32_t, XdOperator::Ntt>(input, output, stream);
                } else {
                    launch<std::uint64_t, std::uint64_t, XdOperator::Ntt>(input, output, stream);
                }
                return;
        }
    }

    XdResidentConfig config_;
    bool available_ = false;
    std::string reason_;
    std::size_t points_ = 0;
    std::uint32_t logs_[kMaxXdRank]{};
    std::size_t strides_[kMaxXdRank]{};
    std::size_t axis_offsets_[kMaxXdRank]{};
    std::uint32_t max_log_ = 0;
    std::uint32_t matrix_stride_ = 1;
    std::uint32_t threads_ = 0;
    std::size_t shared_bytes_ = 0;
    std::size_t twiddle_bytes_ = 0;
    std::size_t matrix_bytes_ = 0;
    std::size_t resource_bytes_ = 0;
    std::uint64_t modular_scale_ = 1;
    std::uint64_t modular_scale_shoup_ = 0;
    void* device_twiddles_ = nullptr;
    void* device_twiddles_shoup_ = nullptr;
    void* device_matrices_ = nullptr;
};

XdResidentAvailability XdResidentPlan::check(const XdResidentConfig& config) {
    auto result = semantic_check(config);
    if (!result.reason.empty()) {
        return result;
    }
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        result.reason = cuda_error(status, "query CUDA device");
        return result;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        result.reason = cuda_error(status, "query CUDA device properties");
        return result;
    }
    std::size_t points = 1;
    for (const auto extent : config.extents) points *= extent;
    const auto shared = resident_shared_bytes(config, points);
    if (config.threads > static_cast<std::uint32_t>(properties.maxThreadsPerBlock)) {
        std::ostringstream stream;
        stream << "xD resident threads " << config.threads
               << " exceed device maxThreadsPerBlock " << properties.maxThreadsPerBlock;
        result.reason = stream.str();
        return result;
    }
    if (shared > static_cast<std::size_t>(properties.sharedMemPerBlock)) {
        std::ostringstream stream;
        stream << "xD resident shared bytes " << shared
               << " exceed device sharedMemPerBlock " << properties.sharedMemPerBlock;
        result.reason = stream.str();
        return result;
    }
    if (config.batch > static_cast<std::size_t>(properties.maxGridSize[0])) {
        result.reason = "xD resident batch exceeds device grid.x";
        return result;
    }
    result.available = true;
    result.reason = "shared resident multidimensional radix-2 kernel is available";
    return result;
}

bool XdResidentPlan::supported(const XdResidentConfig& config, std::string* reason) {
    const auto result = check(config);
    if (reason != nullptr) *reason = result.reason;
    return result.available;
}

XdResidentPlan::XdResidentPlan(XdResidentConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

XdResidentPlan::~XdResidentPlan() = default;
XdResidentPlan::XdResidentPlan(XdResidentPlan&&) noexcept = default;
XdResidentPlan& XdResidentPlan::operator=(XdResidentPlan&&) noexcept = default;

bool XdResidentPlan::available() const noexcept { return impl_->available_; }
const std::string& XdResidentPlan::availability_reason() const noexcept { return impl_->reason_; }
const XdResidentConfig& XdResidentPlan::config() const noexcept { return impl_->config_; }
std::size_t XdResidentPlan::points_per_transform() const noexcept { return impl_->points_; }
std::size_t XdResidentPlan::data_size() const noexcept {
    return impl_->points_ * impl_->config_.batch * storage_bytes(impl_->config_.storage);
}
std::size_t XdResidentPlan::workspace_size() const noexcept { return 0; }
std::uint32_t XdResidentPlan::threads() const noexcept { return impl_->threads_; }

std::string XdResidentPlan::mapping_json() const {
    std::ostringstream json;
    const auto& config = impl_->config_;
    const bool reverse_order = config.inverse &&
                               (config.op == XdOperator::Fwht ||
                                config.op == XdOperator::SubsetZeta ||
                                config.op == XdOperator::SupersetZeta ||
                                config.op == XdOperator::Structured2x2);
    bool legacy_order = config.axis_order.size() == config.extents.size();
    for (std::size_t index = 0; legacy_order && index < config.extents.size(); ++index)
        legacy_order = legacy_order && config.axis_order[index] ==
            (reverse_order ? config.extents.size() - 1 - index : index);
    json << "{\"backend\":\"xd-resident\",\"core\":\"shared-radix2\""
         << ",\"operator\":\"" << operator_name(config.op) << "\""
         << ",\"storage\":\"" << storage_name(config.storage) << "\""
         << ",\"compute\":\"" << compute_name(config.compute) << "\""
         << ",\"rank\":" << config.extents.size() << ",\"extents\":[";
    for (std::size_t axis = 0; axis < config.extents.size(); ++axis) {
        if (axis != 0) json << ',';
        json << config.extents[axis];
    }
    json << "]"
         << ",\"batch\":" << config.batch
         << ",\"inverse\":" << (config.inverse ? "true" : "false")
         << ",\"normalize_inverse\":" << (config.normalize_inverse ? "true" : "false")
         << ",\"threads\":" << impl_->threads_
         << ",\"resource_bytes\":" << impl_->resource_bytes_
         << ",\"shared_bytes\":" << impl_->shared_bytes_
         << ",\"coefficient_bytes\":" << impl_->twiddle_bytes_ + impl_->matrix_bytes_
         << ",\"workspace_size\":0"
         << ",\"order\":\""
         << ((config.op == XdOperator::Fft || config.op == XdOperator::Ntt)
                 ? "per-axis-bit-reversed-input-natural-output"
                 : (!legacy_order ? (reverse_order ? "explicit-axis-reverse-stage-inverse" :
                                                    "explicit-axis-natural-stage") :
                    reverse_order ? "reverse-axis-reverse-stage-inverse" : "natural-axis-natural-stage"))
         << "\",\"axis_order\":[";
    for (std::size_t index = 0; index < config.axis_order.size(); ++index) {
        if (index != 0) json << ',';
        json << config.axis_order[index];
    }
    json << "]"
         << ",\"available\":" << (impl_->available_ ? "true" : "false")
         << ",\"reason\":\"";
    for (const char character : impl_->reason_) {
        if (character == '\\' || character == '"') json << '\\';
        json << character;
    }
    json << "\",\"axes\":[";
    for (std::size_t axis = 0; impl_->available_ && axis < config.extents.size(); ++axis) {
        if (axis != 0) json << ',';
        json << "{\"axis\":" << axis << ",\"extent\":" << config.extents[axis]
             << ",\"stages\":" << impl_->logs_[axis]
             << ",\"stride\":" << impl_->strides_[axis] << '}';
    }
    json << "]}";
    return json.str();
}

void XdResidentPlan::execute(const void* input, void* output, cudaStream_t stream) const {
    impl_->execute(input, output, stream);
}

}  // namespace cuntt
