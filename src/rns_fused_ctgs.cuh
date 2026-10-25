#pragma once

#include "ntt_hybrid2d.cuh"
#include "ntt_traits.cuh"

// Fused negacyclic root strategy: pinned GPU-NTT merge CT/GS implementation in
// results/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT (Apache-2.0).
// The tile indexing below is independently derived for stage intervals.
namespace cuntt::rns_fused_detail {
using Word = std::uint64_t;
using hybrid2d_detail::PolynomialView;

struct Domain {
    Word modulus;
    const Word* roots;
    const Word* shoup;
    Word scale;
    Word scale_shoup;
    std::uint32_t modulus_bits;
    Word barrett_mu;
};

__device__ __forceinline__ void exchange_barrier(unsigned warp_mask, unsigned boundary) {
    // The host proves same-warp ownership for every element at this boundary.
    // Execute outside unit loops: inactive workers must participate as well.
    if (warp_mask & (1u << boundary)) __syncwarp();
    else __syncthreads();
}

template<bool Inverse, bool Barrett>
__device__ __forceinline__ void butterfly(Word& left, Word& right, Word root, Word shoup,
                                          Word modulus, std::uint32_t modulus_bits,
                                          Word barrett_mu) {
    const Word a = left;
    Word b = right;
    if constexpr (Inverse) {
        const Word sum = a + b;
        left = sum >= modulus ? sum - modulus : sum;
        const Word difference = a >= b ? a - b : modulus + a - b;
        if constexpr (Barrett)
            right = hybrid2d_detail::mul_barrett(difference, root, modulus, modulus_bits, barrett_mu);
        else
            right = traits::mul_shoup(difference, root, shoup, modulus);
    } else {
        if constexpr (Barrett)
            b = hybrid2d_detail::mul_barrett(b, root, modulus, modulus_bits, barrett_mu);
        else
            b = traits::mul_shoup(b, root, shoup, modulus);
        const Word sum = a + b;
        left = sum >= modulus ? sum - modulus : sum;
        right = a >= b ? a - b : modulus + a - b;
    }
}

template<unsigned LocalLog, unsigned Rows, bool Inverse, unsigned RadixLog = 1,
         bool Barrett = false, bool Compact = false, bool First = true>
__global__ void stage_kernel(const Word* input, Word* output,
                             const Domain* domains, const PolynomialView* polynomials,
                             unsigned logN, bool reverse_input, bool reverse_output,
                             bool in_place, unsigned warp_barrier_mask) {
    static_assert(LocalLog > 0 && LocalLog < 32, "invalid fused local transform size");
    static_assert(Rows > 0 && (Rows & (Rows - 1)) == 0, "rows must be a power of two");
    static_assert(RadixLog == 1 || RadixLog == 2, "fused core supports radix2 or radix4");
    // This lowering covers two complementary stage intervals. Their execution
    // roles are known at dispatch, including reversed execution for GS.
    constexpr bool first = First;
    constexpr bool last = !First;
    constexpr bool prefix = Inverse ? !First : First;
    const unsigned stage_start = prefix ? 0 : logN - LocalLog;
    constexpr unsigned local_n = 1U << LocalLog;
    constexpr unsigned butterflies_per_tile = local_n / 2;
    constexpr unsigned rows_log = [] {
        unsigned bits = 0;
        for (unsigned value = Rows; value > 1; value >>= 1) ++bits;
        return bits;
    }();
    const unsigned blocks_log = logN - LocalLog - rows_log;
    const unsigned poly_index = blockIdx.x >> blocks_log;
    const unsigned tile_base = (blockIdx.x & ((1U << blocks_log) - 1)) << rows_log;
    const unsigned d_log = prefix ? logN - LocalLog : 0;
    const std::uint64_t dense_base = static_cast<std::uint64_t>(poly_index) << logN;
    unsigned domain_index = 0;
    std::uint64_t input_base = dense_base;
    std::uint64_t output_base = dense_base;
    if constexpr (Compact) {
        // The compact provider consumes the public per-polynomial domain
        // sequence while keeping each row's local coefficients contiguous.
        // The vector order may be a modulus-grouped or sparse permutation, so
        // physical offsets—not poly_index—identify the row. Intermediate
        // storage remains dense by logical polynomial order for the
        // separate-buffer contract.
        const auto polynomial = polynomials[poly_index];
        domain_index = polynomial.domain;
        input_base = first ? polynomial.input_offset
                           : (in_place ? polynomial.output_offset : dense_base);
        output_base = last ? polynomial.output_offset
                           : (in_place ? polynomial.output_offset : dense_base);
    } else {
        const auto polynomial = polynomials[poly_index];
        domain_index = polynomial.domain;
        input_base = first ? polynomial.input_offset : dense_base;
        output_base = last ? polynomial.output_offset : dense_base;
        // A fused plan may execute in place even when the caller's physical
        // rows are a permutation or contain holes.  The old dense-only
        // intermediate path silently ignored `in_place` here, forcing the
        // adapter to allocate a private scratch layout whenever the logical
        // polynomial order differed from the physical row order.  Preserve
        // the explicit row offset for both passes in the in-place case; the
        // plan validator already guarantees disjoint output intervals and
        // the public adapter uses input_offset == output_offset for this
        // contract.
        if (in_place) {
            if (!first) input_base = polynomial.output_offset;
            if (!last) output_base = polynomial.output_offset;
        }
    }
    const auto domain = domains[domain_index];

    extern __shared__ unsigned char shared_storage[];
    Word* tiles = reinterpret_cast<Word*>(shared_storage);

    // Columns r are neighboring global addresses until the last CT interval.
    for (unsigned linear = threadIdx.x; linear < Rows * local_n; linear += blockDim.x) {
        const unsigned row = prefix ? linear & (Rows - 1) : linear >> LocalLog;
        const unsigned u = prefix ? linear >> rows_log : linear & (local_n - 1);
        const unsigned tile = tile_base + row;
        // Prefix tiles enumerate the low coordinates, so tile < 2^d_log;
        // suffix tiles enumerate the high coordinates and d_log is zero.
        const unsigned logical = prefix ? (u << d_log) + tile : (tile << LocalLog) + u;
        const unsigned address = first && reverse_input ? __brev(logical) >> (32 - logN) : logical;
        tiles[linear] = input[input_base + address];
    }
    exchange_barrier(warp_barrier_mask, 0);

#pragma unroll
    for (unsigned iteration = 0; iteration < LocalLog; iteration += RadixLog) {
        if constexpr (RadixLog == 2) {
            if (iteration + 1 < LocalLog) {
                const unsigned ell = Inverse ? LocalLog - iteration - 2 : iteration;
                const unsigned quarter_log = LocalLog - ell - 2;
                const unsigned quarter = 1U << quarter_log;
                constexpr unsigned units_per_tile = local_n / 4;
                for (unsigned linear = threadIdx.x; linear < Rows * units_per_tile; linear += blockDim.x) {
                    const unsigned row = prefix ? linear & (Rows - 1) : linear >> (LocalLog - 2);
                    const unsigned unit = prefix ? linear >> rows_log : linear & (units_per_tile - 1);
                    const unsigned group = unit >> quarter_log;
                    const unsigned offset = unit & (quarter - 1);
                    const unsigned u0 = (group << (quarter_log + 2)) + offset;
                    const unsigned u1 = u0 + quarter;
                    const unsigned u2 = u1 + quarter;
                    const unsigned u3 = u2 + quarter;
                    const unsigned i0 = prefix ? (u0 << rows_log) + row : (row << LocalLog) + u0;
                    const unsigned i1 = prefix ? (u1 << rows_log) + row : (row << LocalLog) + u1;
                    const unsigned i2 = prefix ? (u2 << rows_log) + row : (row << LocalLog) + u2;
                    const unsigned i3 = prefix ? (u3 << rows_log) + row : (row << LocalLog) + u3;
                    const unsigned outer_group = prefix ? 0 : tile_base + row;
                    const unsigned root0 = (1U << (stage_start + ell)) + (outer_group << ell) + group;
                    const unsigned root1 = (1U << (stage_start + ell + 1)) + (outer_group << (ell + 1)) + (group << 1);
                    Word a = tiles[i0], b = tiles[i1], c = tiles[i2], e = tiles[i3];
                    const Word w0 = domain.roots[root0], s0 = domain.shoup[root0];
                    const Word w1 = domain.roots[root1], s1 = domain.shoup[root1];
                    const Word w2 = domain.roots[root1 + 1], s2 = domain.shoup[root1 + 1];
                    if constexpr (Inverse) {
                        butterfly<true, Barrett>(a, b, w1, s1, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<true, Barrett>(c, e, w2, s2, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<true, Barrett>(a, c, w0, s0, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<true, Barrett>(b, e, w0, s0, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                    } else {
                        butterfly<false, Barrett>(a, c, w0, s0, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<false, Barrett>(b, e, w0, s0, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<false, Barrett>(a, b, w1, s1, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                        butterfly<false, Barrett>(c, e, w2, s2, domain.modulus, domain.modulus_bits, domain.barrett_mu);
                    }
                    tiles[i0] = a;
                    tiles[i1] = b;
                    tiles[i2] = c;
                    tiles[i3] = e;
                }
                exchange_barrier(warp_barrier_mask, iteration + 1);
                continue;
            }
        }
        const unsigned ell = Inverse ? LocalLog - 1 - iteration : iteration;
        const unsigned half_log = LocalLog - ell - 1;
        const unsigned half = 1U << half_log;
        for (unsigned linear = threadIdx.x; linear < Rows * butterflies_per_tile; linear += blockDim.x) {
            const unsigned row = prefix ? linear & (Rows - 1) : linear >> (LocalLog - 1);
            const unsigned butterfly_index = prefix ? linear >> rows_log : linear & (butterflies_per_tile - 1);
            const unsigned group = butterfly_index >> half_log;
            const unsigned offset = butterfly_index & (half - 1);
            const unsigned left_u = (group << (half_log + 1)) + offset;
            const unsigned right_u = left_u + half;
            const unsigned left_index = prefix ? (left_u << rows_log) + row : (row << LocalLog) + left_u;
            const unsigned right_index = prefix ? (right_u << rows_log) + row : (row << LocalLog) + right_u;
            const unsigned outer_group = prefix ? 0 : tile_base + row;
            const unsigned root_index = (1U << (stage_start + ell)) + (outer_group << ell) + group;
            Word left = tiles[left_index], right = tiles[right_index];
            butterfly<Inverse, Barrett>(left, right, domain.roots[root_index], domain.shoup[root_index],
                                        domain.modulus, domain.modulus_bits, domain.barrett_mu);
            tiles[left_index] = left;
            tiles[right_index] = right;
        }
        exchange_barrier(warp_barrier_mask, iteration + 1);
    }

    for (unsigned linear = threadIdx.x; linear < Rows * local_n; linear += blockDim.x) {
        const unsigned row = prefix ? linear & (Rows - 1) : linear >> LocalLog;
        const unsigned u = prefix ? linear >> rows_log : linear & (local_n - 1);
        const unsigned tile = tile_base + row;
        const unsigned logical = prefix ? (u << d_log) + tile : (tile << LocalLog) + u;
        const unsigned address = last && reverse_output ? __brev(logical) >> (32 - logN) : logical;
        Word value = tiles[linear];
        if constexpr (Inverse) {
            if (last) {
                if constexpr (Barrett)
                    value = hybrid2d_detail::mul_barrett(value, domain.scale, domain.modulus,
                                                         domain.modulus_bits, domain.barrett_mu);
                else
                    value = traits::mul_shoup(value, domain.scale, domain.scale_shoup, domain.modulus);
            }
        }
        output[output_base + address] = value;
    }
}

} // namespace cuntt::rns_fused_detail
