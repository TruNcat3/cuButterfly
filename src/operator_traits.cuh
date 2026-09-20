#pragma once
#include <cuda_runtime.h>
#include <cuntt/butterfly.hpp>

namespace cuntt::traits {
template <typename Storage>
struct LowStorage;

template <>
struct LowStorage<Fp16> {
    __host__ __device__ static float widen(Fp16 value) { return __half2float(value); }
    __host__ __device__ static Fp16 narrow(float value) { return __float2half_rn(value); }
    __device__ static Fp16 add(Fp16 left, Fp16 right) { return __hadd(left, right); }
    __device__ static Fp16 sub(Fp16 left, Fp16 right) { return __hsub(left, right); }
    __device__ static Fp16 mul(Fp16 left, Fp16 right) { return __hmul(left, right); }
    __device__ static Fp16 shuffle(Fp16 value, std::uint32_t mask) {
        const auto bits = __shfl_xor_sync(0xffffffffU, static_cast<unsigned int>(__half_as_ushort(value)), mask);
        return __ushort_as_half(static_cast<unsigned short>(bits));
    }
};

template <>
struct LowStorage<Bf16> {
    __host__ __device__ static float widen(Bf16 value) { return __bfloat162float(value); }
    __host__ __device__ static Bf16 narrow(float value) { return __float2bfloat16_rn(value); }
    // V100 has no native BF16 arithmetic. Explicit narrowing makes the
    // emulated-native contract deterministic and visible to the experiment.
    __device__ static Bf16 add(Bf16 left, Bf16 right) { return narrow(widen(left) + widen(right)); }
    __device__ static Bf16 sub(Bf16 left, Bf16 right) { return narrow(widen(left) - widen(right)); }
    __device__ static Bf16 mul(Bf16 left, Bf16 right) { return narrow(widen(left) * widen(right)); }
    __device__ static Bf16 shuffle(Bf16 value, std::uint32_t mask) {
        __nv_bfloat16_raw raw = static_cast<__nv_bfloat16_raw>(value);
        raw.x = static_cast<unsigned short>(
            __shfl_xor_sync(0xffffffffU, static_cast<unsigned int>(raw.x), mask));
        return Bf16(raw);
    }
};

template <typename Storage, bool Native>
struct LowMath {
    static __device__ Storage add(Storage left, Storage right) {
        if constexpr (Native)
            return LowStorage<Storage>::add(left, right);
        return LowStorage<Storage>::narrow(LowStorage<Storage>::widen(left) + LowStorage<Storage>::widen(right));
    }
    static __device__ Storage sub(Storage left, Storage right) {
        if constexpr (Native)
            return LowStorage<Storage>::sub(left, right);
        return LowStorage<Storage>::narrow(LowStorage<Storage>::widen(left) - LowStorage<Storage>::widen(right));
    }
    static __device__ Storage mul(Storage left, Storage right) {
        if constexpr (Native)
            return LowStorage<Storage>::mul(left, right);
        return LowStorage<Storage>::narrow(LowStorage<Storage>::widen(left) * LowStorage<Storage>::widen(right));
    }
    static __device__ Storage linear(Storage c0, Storage x0, Storage c1, Storage x1) {
        if constexpr (Native)
            return add(mul(c0, x0), mul(c1, x1));
        return LowStorage<Storage>::narrow(LowStorage<Storage>::widen(c0) * LowStorage<Storage>::widen(x0) +
                                           LowStorage<Storage>::widen(c1) * LowStorage<Storage>::widen(x1));
    }
};

template <typename Storage>
struct LowComplexType;

template <>
struct LowComplexType<Fp16> { using type = Complex16; };
template <>
struct LowComplexType<Bf16> { using type = ComplexBf16; };

template <typename Real>
struct FwhtOperator {
    using Value                            = Real;
    static constexpr bool kBitReverseInput = false;

    bool inverse;
    bool normalize_inverse;

    __device__ __forceinline__ void apply(std::uint32_t, std::uint32_t, Value& left, Value& right) const {
        const Real a = left;
        const Real b = right;
        left         = a + b;
        right        = a - b;
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const { return __shfl_xor_sync(0xffffffffU, value, mask); }

    __device__ __forceinline__ Value apply_lane(std::uint32_t, std::uint32_t, Value self, Value partner, bool right_lane) const {
        return right_lane ? partner - self : self + partner;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        return inverse && normalize_inverse ? value / static_cast<Real>(n) : value;
    }
};

template <typename Storage, bool Native>
struct LowFwhtOperator {
    using Value                            = Storage;
    static constexpr bool kBitReverseInput = false;

    bool inverse;
    bool normalize_inverse;

    __device__ __forceinline__ void apply(std::uint32_t, std::uint32_t, Value& left, Value& right) const {
        const Storage a = left;
        const Storage b = right;
        left            = LowMath<Storage, Native>::add(a, b);
        right           = LowMath<Storage, Native>::sub(a, b);
    }
    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return LowStorage<Storage>::shuffle(value, mask);
    }
    __device__ __forceinline__ Value apply_lane(std::uint32_t, std::uint32_t, Value self, Value partner, bool right_lane) const {
        return right_lane ? LowMath<Storage, Native>::sub(partner, self) : LowMath<Storage, Native>::add(self, partner);
    }
    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        if (!inverse || !normalize_inverse)
            return value;
        return LowStorage<Storage>::narrow(LowStorage<Storage>::widen(value) / static_cast<float>(n));
    }
};

template <typename Real>
struct DeviceMatrix2x2 {
    Real m00;
    Real m01;
    Real m10;
    Real m11;
};

template <typename Real>
struct Structured2x2Operator {
    using Value                            = Real;
    static constexpr bool kBitReverseInput = false;

    const DeviceMatrix2x2<Real>* matrices;
    bool inverse = false;
    bool normalize_inverse = false;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t, Value& left, Value& right) const {
        const auto matrix = matrices[stage];
        const Real a      = left;
        const Real b      = right;
        left              = matrix.m00 * a + matrix.m01 * b;
        right             = matrix.m10 * a + matrix.m11 * b;
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return __shfl_xor_sync(0xffffffffU, value, mask);
    }

    __device__ __forceinline__ Value apply_lane(std::uint32_t stage, std::uint32_t, Value self, Value partner,
                                                 bool right_lane) const {
        const auto matrix = matrices[stage];
        return right_lane ? matrix.m10 * partner + matrix.m11 * self
                          : matrix.m00 * self + matrix.m01 * partner;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const { return value; }
};

template <typename Storage, bool Native>
struct LowStructured2x2Operator {
    using Value                            = Storage;
    static constexpr bool kBitReverseInput = false;

    const DeviceMatrix2x2<Storage>* matrices;
    bool inverse = false;
    bool normalize_inverse = false;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t, Value& left, Value& right) const {
        const auto matrix = matrices[stage];
        const Storage a = left;
        const Storage b = right;
        left  = LowMath<Storage, Native>::linear(matrix.m00, a, matrix.m01, b);
        right = LowMath<Storage, Native>::linear(matrix.m10, a, matrix.m11, b);
    }
    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return LowStorage<Storage>::shuffle(value, mask);
    }
    __device__ __forceinline__ Value apply_lane(std::uint32_t stage, std::uint32_t, Value self, Value partner,
                                                 bool right_lane) const {
        const auto matrix = matrices[stage];
        return right_lane ? LowMath<Storage, Native>::linear(matrix.m10, partner, matrix.m11, self)
                          : LowMath<Storage, Native>::linear(matrix.m00, self, matrix.m01, partner);
    }
    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const { return value; }
};

template <typename Complex, typename Real>
struct FftOperator {
    using Value                            = Complex;
    static constexpr bool kBitReverseInput = true;

    const Complex*  twiddles;
    bool            inverse;
    bool            normalize_inverse;
    ComplexMultiply multiply;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const Complex root = twiddles[(1U << stage) - 1 + offset];
        Real          vr;
        Real          vi;
        if (multiply == ComplexMultiply::Gauss3) {
            const Real p0 = right.real * root.real;
            const Real p1 = right.imag * root.imag;
            const Real p2 = (right.real + right.imag) * (root.real + root.imag);
            vr            = p0 - p1;
            vi            = p2 - p0 - p1;
        } else {
            vr = right.real * root.real - right.imag * root.imag;
            vi = right.real * root.imag + right.imag * root.real;
        }
        const Real lr = left.real;
        const Real li = left.imag;
        left          = {lr + vr, li + vi};
        right         = {lr - vr, li - vi};
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return {__shfl_xor_sync(0xffffffffU, value.real, mask), __shfl_xor_sync(0xffffffffU, value.imag, mask)};
    }

    __device__ __forceinline__ Value apply_lane(std::uint32_t stage, std::uint32_t offset, Value self, Value partner, bool right_lane) const {
        const Complex left  = right_lane ? partner : self;
        const Complex right = right_lane ? self : partner;
        const Complex root  = twiddles[(1U << stage) - 1 + offset];
        Real          vr;
        Real          vi;
        if (multiply == ComplexMultiply::Gauss3) {
            const Real p0 = right.real * root.real;
            const Real p1 = right.imag * root.imag;
            const Real p2 = (right.real + right.imag) * (root.real + root.imag);
            vr            = p0 - p1;
            vi            = p2 - p0 - p1;
        } else {
            vr = right.real * root.real - right.imag * root.imag;
            vi = right.real * root.imag + right.imag * root.real;
        }
        return right_lane ? Value{left.real - vr, left.imag - vi} : Value{left.real + vr, left.imag + vi};
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        if (!inverse || !normalize_inverse) {
            return value;
        }
        const Real scale = Real{1} / static_cast<Real>(n);
        return {value.real * scale, value.imag * scale};
    }
};

template <typename Storage, bool Native>
struct LowFftOperator {
    using Value                            = typename LowComplexType<Storage>::type;
    static constexpr bool kBitReverseInput = true;

    const Value* twiddles;
    bool inverse;
    bool normalize_inverse;
    ComplexMultiply multiply;

    __device__ __forceinline__ Value product(Value right, Value root) const {
        if constexpr (Native) {
            const Storage rr = LowMath<Storage, true>::mul(right.real, root.real);
            const Storage ii = LowMath<Storage, true>::mul(right.imag, root.imag);
            const Storage ri = LowMath<Storage, true>::mul(right.real, root.imag);
            const Storage ir = LowMath<Storage, true>::mul(right.imag, root.real);
            return {LowMath<Storage, true>::sub(rr, ii), LowMath<Storage, true>::add(ri, ir)};
        }
        const float rr = LowStorage<Storage>::widen(right.real);
        const float ri = LowStorage<Storage>::widen(right.imag);
        const float wr = LowStorage<Storage>::widen(root.real);
        const float wi = LowStorage<Storage>::widen(root.imag);
        return {LowStorage<Storage>::narrow(rr * wr - ri * wi), LowStorage<Storage>::narrow(rr * wi + ri * wr)};
    }
    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const Value weighted = product(right, twiddles[(1U << stage) - 1 + offset]);
        const Value original = left;
        left  = {LowMath<Storage, Native>::add(original.real, weighted.real),
                 LowMath<Storage, Native>::add(original.imag, weighted.imag)};
        right = {LowMath<Storage, Native>::sub(original.real, weighted.real),
                 LowMath<Storage, Native>::sub(original.imag, weighted.imag)};
    }
    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return {LowStorage<Storage>::shuffle(value.real, mask), LowStorage<Storage>::shuffle(value.imag, mask)};
    }
    __device__ __forceinline__ Value apply_lane(std::uint32_t stage, std::uint32_t offset, Value self, Value partner, bool right_lane) const {
        Value left = right_lane ? partner : self;
        Value right = right_lane ? self : partner;
        apply(stage, offset, left, right);
        return right_lane ? right : left;
    }
    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        if (!inverse || !normalize_inverse)
            return value;
        const float scale = 1.0F / static_cast<float>(n);
        return {LowStorage<Storage>::narrow(LowStorage<Storage>::widen(value.real) * scale),
                LowStorage<Storage>::narrow(LowStorage<Storage>::widen(value.imag) * scale)};
    }
};

template <bool UpdateLeft>
struct BooleanZetaOperator {
    using Value                            = std::uint32_t;
    static constexpr bool kBitReverseInput = false;

    bool inverse;

    __device__ __forceinline__ void apply(std::uint32_t, std::uint32_t, Value& left, Value& right) const {
        if constexpr (UpdateLeft) {
            left = inverse ? left - right : left + right;
        } else {
            right = inverse ? right - left : right + left;
        }
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const { return __shfl_xor_sync(0xffffffffU, value, mask); }

    __device__ __forceinline__ Value apply_lane(std::uint32_t, std::uint32_t, Value self, Value partner, bool right_lane) const {
        if constexpr (UpdateLeft) {
            return right_lane ? self : (inverse ? self - partner : self + partner);
        } else {
            return right_lane ? (inverse ? self - partner : partner + self) : self;
        }
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const { return value; }
};

} // namespace cuntt::traits
