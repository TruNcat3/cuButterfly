#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace cuntt::traits {
__device__ __forceinline__ std::uint64_t mul_shoup(std::uint64_t value, std::uint64_t twiddle, std::uint64_t twiddle_shoup, std::uint64_t modulus) {
    const std::uint64_t quotient = __umul64hi(value, twiddle_shoup);
    std::uint64_t       reduced  = value * twiddle - quotient * modulus;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

__device__ __forceinline__ std::uint32_t mul_shoup(std::uint32_t value, std::uint32_t twiddle, std::uint32_t twiddle_shoup, std::uint32_t modulus) {
    const std::uint32_t quotient = __umulhi(value, twiddle_shoup);
    std::uint32_t       reduced  = value * twiddle - quotient * modulus;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

template <typename Word>
struct NttStageOperatorT {
    using Value = Word;
    static constexpr bool kBitReverseInput = true;

    const Word* twiddles;
    const Word* twiddles_shoup;
    Word        modulus;
    Word        scale;
    Word        scale_shoup;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const std::uint32_t twiddle_base = (1U << stage) - 1;
        const Value u = left;
        const Value v = mul_shoup(right, twiddles[twiddle_base + offset], twiddles_shoup[twiddle_base + offset], modulus);
        const Value sum = u + v;
        left  = sum >= modulus ? sum - modulus : sum;
        right = u >= v ? u - v : modulus + u - v;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const {
        return scale == 1 ? value : mul_shoup(value, scale, scale_shoup, modulus);
    }

    __device__ __forceinline__ Value multiply(Value value, Value coefficient,
                                               Value coefficient_shoup) const {
        return mul_shoup(value, coefficient, coefficient_shoup, modulus);
    }

    __device__ __forceinline__ void apply_coefficient(
        Value coefficient, Value coefficient_shoup, Value& left, Value& right) const {
        const Value u = left;
        const Value v = mul_shoup(right, coefficient, coefficient_shoup, modulus);
        const Value sum = u + v;
        left = sum >= modulus ? sum - modulus : sum;
        right = u >= v ? u - v : modulus + u - v;
    }
};

using NttStageOperator = NttStageOperatorT<std::uint64_t>;
} // namespace cuntt::traits
