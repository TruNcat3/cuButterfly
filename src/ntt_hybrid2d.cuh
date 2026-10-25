#pragma once

#include "cuntt/ntt.hpp"
#include "ntt_traits.cuh"

// Shared by the single-domain and RNS plans. Arithmetic/tiles stay identical.
namespace cuntt::hybrid2d_detail {
using traits::mul_shoup;

struct Wide128 {
    std::uint64_t low;
    std::uint64_t high;
};

__device__ __forceinline__ Wide128 wide_multiply(std::uint64_t left, std::uint64_t right) {
    Wide128 result{};
    // Match the Data64 Barrett provider used by the pinned GPU-NTT core.
    // Keeping the carry-producing pair explicit avoids lowering the 128-bit
    // intermediate through a generic aggregate operation.
    asm("{\n\t"
        "mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;\n\t"
        "}"
        : "=l"(result.low), "=l"(result.high)
        : "l"(left), "l"(right));
    return result;
}

__device__ __forceinline__ Wide128 wide_shift_right(Wide128 value, std::uint32_t shift) {
    if (shift == 0) {
        return value;
    }
    if (shift < 64) {
        return {(value.low >> shift) | (value.high << (64 - shift)), value.high >> shift};
    }
    if (shift < 128) {
        return {value.high >> (shift - 64), 0};
    }
    return {0, 0};
}

__device__ __forceinline__ Wide128 wide_subtract(Wide128 left, Wide128 right) {
    Wide128 result{};
    asm("{\n\t"
        "sub.cc.u64 %0, %2, %4;\n\t"
        "subc.u64 %1, %3, %5;\n\t"
        "}"
        : "=l"(result.low), "=l"(result.high)
        : "l"(left.low), "l"(left.high), "l"(right.low), "l"(right.high));
    return result;
}

__device__ __forceinline__ std::uint64_t mul_barrett(std::uint64_t left, std::uint64_t right, std::uint64_t modulus, std::uint32_t modulus_bits,
                                                     std::uint64_t mu) {
    Wide128 product  = wide_multiply(left, right);
    Wide128 quotient = wide_shift_right(product, modulus_bits - 2);
    quotient         = wide_multiply(quotient.low, mu);
    quotient         = wide_shift_right(quotient, modulus_bits + 3);
    product          = wide_subtract(product, wide_multiply(quotient.low, modulus));
    return product.low >= modulus ? product.low - modulus : product.low;
}

__device__ __forceinline__ std::uint32_t mul_barrett(std::uint32_t left, std::uint32_t right, std::uint32_t modulus, std::uint32_t modulus_bits,
                                                     std::uint32_t mu) {
    std::uint64_t product  = static_cast<std::uint64_t>(left) * right;
    std::uint64_t quotient = product >> (modulus_bits - 2);
    quotient               = static_cast<std::uint64_t>(static_cast<std::uint32_t>(quotient)) * mu;
    quotient >>= modulus_bits + 3;
    product -= static_cast<std::uint64_t>(static_cast<std::uint32_t>(quotient)) * modulus;
    return static_cast<std::uint32_t>(product >= modulus ? product - modulus : product);
}

constexpr std::uint32_t kHybridMinLocalLog = 6;
constexpr std::uint32_t kHybridMaxLocalLog = 10;

template <typename Word> struct DomainView {
    Word modulus;
    const Word *twiddles, *twiddles_shoup, *roots, *roots_shoup;
    // nullptr for cyclic. Inverse factors already include 1/N.
    const Word *boundary, *boundary_shoup;
    Word scale, scale_shoup;
};
struct PolynomialView {
    std::uint32_t domain;
    std::uint64_t input_offset, output_offset;
};
template <typename Word> struct BatchView {
    const DomainView<Word>* domains = nullptr;
    const PolynomialView* polynomials = nullptr;
    bool inverse = false;
    bool input_bit_reversed = false;
    bool output_bit_reversed = false;
    bool physical_boundary_traversal = false;
};

template <bool Fused, typename Word>
__device__ __forceinline__ Word local_twiddle(const Word* standard, const Word* fused, std::uint32_t row_base, std::uint32_t row,
                                              std::uint32_t local_n, std::uint32_t index) {
    if constexpr (Fused) {
        return fused[(row_base + row) * (local_n - 1) + index];
    }
    return standard[index];
}

template <bool Barrett, bool Fused, typename Word>
__device__ __forceinline__ Word local_twiddle_shoup(const Word* standard, const Word* fused, std::uint32_t row_base, std::uint32_t row,
                                                    std::uint32_t local_n, std::uint32_t index) {
    if constexpr (Barrett) {
        return 0;
    }
    return local_twiddle<Fused>(standard, fused, row_base, row, local_n, index);
}

template <bool Barrett, typename Word>
__device__ __forceinline__ void local_butterfly(Word& left, Word& right, Word twiddle, Word twiddle_shoup, Word modulus, std::uint32_t modulus_bits,
                                                Word barrett_mu) {
    const Word u = left;
    Word       v;
    if constexpr (Barrett) {
        v = mul_barrett(right, twiddle, modulus, modulus_bits, barrett_mu);
    } else {
        v = mul_shoup(right, twiddle, twiddle_shoup, modulus);
    }
    const Word sum = u + v;
    left           = sum >= modulus ? sum - modulus : sum;
    right          = u >= v ? u - v : modulus + u - v;
}

template <typename Word, std::uint32_t LocalLog, std::uint32_t RowsPerBlock, ComputeUnit Unit, bool Fused = false, bool Barrett = false>
__device__ void local_ntt(Word* tiles, const Word* twiddles, const Word* twiddles_shoup, Word modulus, const Word* fused_twiddles = nullptr,
                          const Word* fused_twiddles_shoup = nullptr, std::uint32_t row_base = 0, std::uint32_t modulus_bits = 0,
                          Word barrett_mu = 0) {
    constexpr std::uint32_t local_n = 1U << LocalLog;
    constexpr std::uint32_t stride  = local_n + 1;
    if constexpr (Unit == ComputeUnit::Radix2) {
        constexpr std::uint32_t butterflies_per_row = local_n / 2;
        for (std::uint32_t stage = 0; stage < LocalLog; ++stage) {
            const std::uint32_t half           = 1U << stage;
            const std::uint32_t twiddle_offset = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t local       = linear - row * butterflies_per_row;
                const std::uint32_t group       = local / half;
                const std::uint32_t offset      = local - group * half;
                const std::uint32_t left_index  = row * stride + group * (half << 1) + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    } else if constexpr (Unit == ComputeUnit::Radix4) {
        constexpr std::uint32_t radix4_units_per_row = local_n / 4;
        std::uint32_t           stage                = 0;
        for (; stage + 1 < LocalLog; stage += 2) {
            const std::uint32_t half0           = 1U << stage;
            const std::uint32_t twiddle_offset0 = half0 - 1;
            const std::uint32_t twiddle_offset1 = (half0 << 1) - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * radix4_units_per_row; linear += blockDim.x) {
                const std::uint32_t row      = linear / radix4_units_per_row;
                const std::uint32_t local    = linear - row * radix4_units_per_row;
                const std::uint32_t group    = local / half0;
                const std::uint32_t offset   = local - group * half0;
                const std::uint32_t i0       = row * stride + group * (half0 << 2) + offset;
                const std::uint32_t i1       = i0 + half0;
                const std::uint32_t i2       = i1 + half0;
                const std::uint32_t i3       = i2 + half0;
                Word                a        = tiles[i0];
                Word                b        = tiles[i1];
                Word                c        = tiles[i2];
                Word                d        = tiles[i3];
                const Word          twiddle0 = local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset0 + offset);
                const Word          twiddle0_shoup =
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset0 + offset);
                local_butterfly<Barrett>(a, b, twiddle0, twiddle0_shoup, modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(c, d, twiddle0, twiddle0_shoup, modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(
                    a, c, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset1 + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset1 + offset),
                    modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(b, d,
                                         local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset1 + half0 + offset),
                                         local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n,
                                                                             twiddle_offset1 + half0 + offset),
                                         modulus, modulus_bits, barrett_mu);
                tiles[i0] = a;
                tiles[i1] = b;
                tiles[i2] = c;
                tiles[i3] = d;
            }
            __syncthreads();
        }
        if (stage < LocalLog) {
            constexpr std::uint32_t butterflies_per_row = local_n / 2;
            constexpr std::uint32_t half                = 1U << (LocalLog - 1);
            constexpr std::uint32_t twiddle_offset      = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t offset      = linear - row * butterflies_per_row;
                const std::uint32_t left_index  = row * stride + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    } else {
        constexpr std::uint32_t radix8_units_per_row = local_n / 8;
        std::uint32_t           stage                = 0;
        for (; stage + 2 < LocalLog; stage += 3) {
            const std::uint32_t eighth          = 1U << stage;
            const std::uint32_t twiddle_offset0 = eighth - 1;
            const std::uint32_t twiddle_offset1 = (eighth << 1) - 1;
            const std::uint32_t twiddle_offset2 = (eighth << 2) - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * radix8_units_per_row; linear += blockDim.x) {
                const std::uint32_t row       = linear / radix8_units_per_row;
                const std::uint32_t local     = linear - row * radix8_units_per_row;
                const std::uint32_t group     = local / eighth;
                const std::uint32_t offset    = local - group * eighth;
                const std::uint32_t i0        = row * stride + group * (eighth << 3) + offset;
                const std::uint32_t i1        = i0 + eighth;
                const std::uint32_t i2        = i1 + eighth;
                const std::uint32_t i3        = i2 + eighth;
                const std::uint32_t i4        = i3 + eighth;
                const std::uint32_t i5        = i4 + eighth;
                const std::uint32_t i6        = i5 + eighth;
                const std::uint32_t i7        = i6 + eighth;
                Word                x0        = tiles[i0];
                Word                x1        = tiles[i1];
                Word                x2        = tiles[i2];
                Word                x3        = tiles[i3];
                Word                x4        = tiles[i4];
                Word                x5        = tiles[i5];
                Word                x6        = tiles[i6];
                Word                x7        = tiles[i7];
                const auto          butterfly = [&](Word& left, Word& right, std::uint32_t twiddle_index) {
                    local_butterfly<Barrett>(
                        left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_index),
                        local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_index), modulus,
                        modulus_bits, barrett_mu);
                };
                butterfly(x0, x1, twiddle_offset0 + offset);
                butterfly(x2, x3, twiddle_offset0 + offset);
                butterfly(x4, x5, twiddle_offset0 + offset);
                butterfly(x6, x7, twiddle_offset0 + offset);
                butterfly(x0, x2, twiddle_offset1 + offset);
                butterfly(x1, x3, twiddle_offset1 + eighth + offset);
                butterfly(x4, x6, twiddle_offset1 + offset);
                butterfly(x5, x7, twiddle_offset1 + eighth + offset);
                butterfly(x0, x4, twiddle_offset2 + offset);
                butterfly(x1, x5, twiddle_offset2 + eighth + offset);
                butterfly(x2, x6, twiddle_offset2 + 2 * eighth + offset);
                butterfly(x3, x7, twiddle_offset2 + 3 * eighth + offset);
                tiles[i0] = x0;
                tiles[i1] = x1;
                tiles[i2] = x2;
                tiles[i3] = x3;
                tiles[i4] = x4;
                tiles[i5] = x5;
                tiles[i6] = x6;
                tiles[i7] = x7;
            }
            __syncthreads();
        }
        for (; stage < LocalLog; ++stage) {
            constexpr std::uint32_t butterflies_per_row = local_n / 2;
            const std::uint32_t     half                = 1U << stage;
            const std::uint32_t     twiddle_offset      = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t local       = linear - row * butterflies_per_row;
                const std::uint32_t group       = local / half;
                const std::uint32_t offset      = local - group * half;
                const std::uint32_t left_index  = row * stride + group * (half << 1) + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    }
}

template <typename Word, std::uint32_t N2Log, std::uint32_t RowsPerBlock, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett, bool Batched = false>
__global__ void hybrid2d_first_pass_kernel(const Word* input, Word* scratch, const Word* twiddles, const Word* twiddles_shoup,
                                           const Word* root_powers, const Word* root_powers_shoup, Word modulus, std::uint32_t n1_log,
                                           std::uint32_t modulus_bits, Word barrett_mu, std::uint32_t log_n, BatchView<Word> batch = {}) {
    constexpr std::uint32_t         N2      = 1U << N2Log;
    const std::uint32_t             n1_size = 1U << n1_log;
    extern __shared__ unsigned char shared_storage[];
    Word*                           tiles = reinterpret_cast<Word*>(shared_storage);

    const std::uint32_t blocks_per_transform = n1_size / RowsPerBlock;
    const std::uint64_t batch_index          = blockIdx.x / blocks_per_transform;
    const std::uint32_t row_base             = (blockIdx.x % blocks_per_transform) * RowsPerBlock;
    const std::uint64_t transform_base       = batch_index << log_n;

    std::uint64_t input_base = transform_base;
    const Word *boundary = nullptr, *boundary_shoup = nullptr;
    if constexpr (Batched) {
        static_assert(!Barrett, "RNS domain view currently uses Shoup");
        const auto polynomial = batch.polynomials[batch_index];
        const auto domain = batch.domains[polynomial.domain];
        input_base = polynomial.input_offset;
        modulus = domain.modulus;
        twiddles = domain.twiddles;
        twiddles_shoup = domain.twiddles_shoup;
        root_powers = domain.roots;
        root_powers_shoup = domain.roots_shoup;
        if (!batch.inverse) {
            boundary = domain.boundary;
            boundary_shoup = domain.boundary_shoup;
        }
    }

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N2; linear += blockDim.x) {
        std::uint32_t n2                 = linear / RowsPerBlock;
        std::uint32_t row                = linear - n2 * RowsPerBlock;
        if constexpr (Batched) {
            if (batch.physical_boundary_traversal && batch.input_bit_reversed) {
                row = linear / N2;
                const std::uint32_t physical_n2 = linear % N2;
                n2 = __brev(physical_n2) >> (32 - N2Log);
            }
        }
        const std::uint32_t n1           = row_base + row;
        const std::uint32_t reversed     = __brev(n2) >> (32 - N2Log);
        const std::uint32_t logical = n2 * n1_size + n1;
        std::uint32_t address = logical;
        if constexpr (Batched) {
            if (batch.input_bit_reversed) address = __brev(logical) >> (32 - log_n);
        }
        Word value = input[input_base + address];
        if constexpr (Batched) {
            if (boundary) value = mul_shoup(value, boundary[logical], boundary_shoup[logical], modulus);
        }
        tiles[row * (N2 + 1) + reversed] = value;
    }
    __syncthreads();

    local_ntt<Word, N2Log, RowsPerBlock, Unit, false, Barrett>(tiles, twiddles, twiddles_shoup, modulus, nullptr, nullptr, 0, modulus_bits,
                                                               barrett_mu);

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N2; linear += blockDim.x) {
        const std::uint32_t k2    = linear / RowsPerBlock;
        const std::uint32_t row   = linear - k2 * RowsPerBlock;
        const std::uint32_t n1    = row_base + row;
        Word                value = tiles[row * (N2 + 1) + k2];
        if constexpr (Placement == CrossTwiddlePlacement::FirstPass) {
            const std::uint32_t exponent = n1 * k2;
            if constexpr (Barrett) {
                value = mul_barrett(value, root_powers[exponent], modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, root_powers[exponent], root_powers_shoup[exponent], modulus);
            }
        }
        scratch[transform_base + static_cast<std::uint64_t>(k2) * n1_size + n1] = value;
    }
}

template <typename Word, std::uint32_t N1Log, std::uint32_t RowsPerBlock, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett, bool Batched = false>
__global__ void hybrid2d_second_pass_kernel(const Word* scratch, Word* output, const Word* twiddles, const Word* twiddles_shoup, Word modulus,
                                            const Word* root_powers, const Word* root_powers_shoup, bool apply_scale, Word scale, Word scale_shoup,
                                            std::uint32_t modulus_bits, Word barrett_mu, std::uint32_t n2_log, std::uint32_t log_n, BatchView<Word> batch = {}) {
    constexpr std::uint32_t         N1 = 1U << N1Log;
    const std::uint32_t             n2 = 1U << n2_log;
    extern __shared__ unsigned char shared_storage[];
    Word*                           tiles = reinterpret_cast<Word*>(shared_storage);

    const std::uint32_t blocks_per_transform = n2 / RowsPerBlock;
    const std::uint64_t batch_index          = blockIdx.x / blocks_per_transform;
    const std::uint32_t row_base             = (blockIdx.x % blocks_per_transform) * RowsPerBlock;
    const std::uint64_t transform_base       = batch_index << log_n;

    std::uint64_t output_base = transform_base;
    const Word *boundary = nullptr, *boundary_shoup = nullptr;
    if constexpr (Batched) {
        static_assert(!Barrett && Placement != CrossTwiddlePlacement::Fused,
                      "RNS currently uses Shoup with first/second-pass cross twiddles");
        const auto polynomial = batch.polynomials[batch_index];
        const auto domain = batch.domains[polynomial.domain];
        output_base = polynomial.output_offset;
        modulus = domain.modulus;
        twiddles = domain.twiddles;
        twiddles_shoup = domain.twiddles_shoup;
        root_powers = domain.roots;
        root_powers_shoup = domain.roots_shoup;
        scale = domain.scale;
        scale_shoup = domain.scale_shoup;
        if (batch.inverse) {
            boundary = domain.boundary;
            boundary_shoup = domain.boundary_shoup;
        }
        apply_scale = batch.inverse && !boundary;
    }

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N1; linear += blockDim.x) {
        const std::uint32_t row      = linear / N1;
        const std::uint32_t n1       = linear - row * N1;
        const std::uint32_t k2       = row_base + row;
        const std::uint32_t reversed = __brev(n1) >> (32 - N1Log);
        Word                value    = scratch[transform_base + static_cast<std::uint64_t>(k2) * N1 + n1];
        if constexpr (Placement == CrossTwiddlePlacement::SecondPass) {
            const std::uint32_t exponent = n1 * k2;
            if constexpr (Barrett) {
                value = mul_barrett(value, root_powers[exponent], modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, root_powers[exponent], root_powers_shoup[exponent], modulus);
            }
        }
        tiles[row * (N1 + 1) + reversed] = value;
    }
    __syncthreads();

    if constexpr (Placement == CrossTwiddlePlacement::Fused) {
        local_ntt<Word, N1Log, RowsPerBlock, Unit, true, Barrett>(tiles, twiddles, twiddles_shoup, modulus, root_powers, root_powers_shoup, row_base,
                                                                  modulus_bits, barrett_mu);
    } else {
        local_ntt<Word, N1Log, RowsPerBlock, Unit, false, Barrett>(tiles, twiddles, twiddles_shoup, modulus, nullptr, nullptr, 0, modulus_bits,
                                                                   barrett_mu);
    }

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N1; linear += blockDim.x) {
        std::uint32_t k1          = linear / RowsPerBlock;
        std::uint32_t row         = linear - k1 * RowsPerBlock;
        if constexpr (Batched) {
            if (batch.physical_boundary_traversal && batch.output_bit_reversed) {
                row = linear / N1;
                const std::uint32_t physical_k1 = linear % N1;
                k1 = __brev(physical_k1) >> (32 - N1Log);
            }
        }
        const std::uint32_t k2    = row_base + row;
        Word                value = tiles[row * (N1 + 1) + k1];
        if (apply_scale) {
            if constexpr (Barrett) {
                value = mul_barrett(value, scale, modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, scale, scale_shoup, modulus);
            }
        }
        const std::uint32_t logical = k1 * n2 + k2;
        std::uint32_t address = logical;
        if constexpr (Batched) {
            if (boundary) value = mul_shoup(value, boundary[logical], boundary_shoup[logical], modulus);
            if (batch.output_bit_reversed) address = __brev(logical) >> (32 - log_n);
        }
        output[output_base + address] = value;
    }
}


} // namespace cuntt::hybrid2d_detail
