#pragma once

#include "fft_register_tile.cuh"

namespace cuntt::detail {

// Explicit native C2C block core. Registers use cyclic natural ownership on
// both sides; the full tile is core storage regardless of suffix exchange chunk.
template<unsigned Log, unsigned Ept, unsigned Columns, bool Inverse, class Complex>
struct NativeBlock {
    static_assert(Log < 31, "native block FFT size must fit unsigned indexing");
    static constexpr unsigned N = 1U << Log;
    static_assert(Ept && !(Ept & (Ept - 1)) && Ept <= N,
                  "native block EPT must be a power of two dividing size");
    static_assert(Columns && !(Columns & (Columns - 1)),
                  "native block columns must be a positive power of two");
    static constexpr unsigned Rows = N / Ept;
    static_assert(Rows * Columns <= 1024, "native block exceeds CUDA thread limit");
    using Real = decltype(Complex{}.real);
    static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                  "native block supports FP32 and FP64 complex storage");
    using value_type = Complex;
    static constexpr unsigned storage_size = Ept;
    static constexpr unsigned elements_per_thread = Ept;
    static constexpr unsigned input_length = N, output_length = N;
    static constexpr unsigned max_threads_per_block = Rows * Columns;
    static constexpr unsigned shared_memory_size = N * Columns * sizeof(Complex);
    static constexpr dim3 block_dim{Rows, Columns, 1};
    static constexpr unsigned EptLog = register_tile::power_log2(Ept);
    static constexpr unsigned RowLog = Log - EptLog;

    // Row-dependent XOR permutes each row's EPT slots bijectively and spreads
    // warp-varying rows across banks without changing column ownership.
    __host__ __device__ static constexpr unsigned index(unsigned row, unsigned i, unsigned column) {
        return (row * Ept + (i ^ (row & (Ept - 1)))) * Columns + column;
    }

    __device__ __forceinline__ void execute(value_type (&v)[Ept], void* shared) const {
        const unsigned row = threadIdx.x, column = threadIdx.y;
        auto* tile = static_cast<Complex*>(shared);
        // Consume the caller's register array directly. A second private array
        // kept both representations live through an inlined register FFT.
        if constexpr (Ept > 1) register_tile::fft<EptLog, Inverse>(v);

        // n=row+Rows*j, k=i+Ept*q: the first EPT FFT is followed by
        // W_N^(row*i), then a Rows FFT. Bit-reverse its row input for DIT.
        unsigned reversed = 0;
        if constexpr (RowLog > 0) reversed = __brev(row) >> (32 - RowLog);
        const auto step = register_tile::root<Inverse, Complex>(Real(row) / Real(N));
        Complex cross{Real(1), Real(0)};
        #pragma unroll
        for (unsigned i = 0; i < Ept; ++i) {
            if (i && i % 8 == 0)
                cross = register_tile::root<Inverse, Complex>(Real(i * row) / Real(N));
            tile[index(reversed, i, column)] =
                register_tile::multiply(v[i], cross);
            cross = register_tile::multiply(cross, step);
        }
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < Ept; ++i) v[i] = tile[index(row, i, column)];
        __syncthreads();

        if constexpr (RowLog > 0) {
          #pragma unroll
          for (unsigned stage = 0; stage < RowLog; ++stage) {
            const unsigned half = 1U << stage;
            const bool upper = (row & half) != 0;
            const auto phase = register_tile::root<Inverse, Complex>(Real(row & (half - 1)) / Real(2 * half));
            if (half < 32) {
                // Power-of-two Rows aligns each column's subgroup; explicit
                // shuffle width isolates columns even when Rows is sub-warp.
                constexpr unsigned Width = Rows < 32 ? Rows : 32;
                const unsigned mask = __activemask();
                #pragma unroll
                for (unsigned i = 0; i < Ept; ++i) {
                    const Complex peer{__shfl_xor_sync(mask, v[i].real, half, Width),
                                       __shfl_xor_sync(mask, v[i].imag, half, Width)};
                    const auto a = upper ? peer : v[i];
                    const auto b = register_tile::multiply(upper ? v[i] : peer, phase);
                    v[i] = upper ? Complex{a.real - b.real, a.imag - b.imag} :
                                   Complex{a.real + b.real, a.imag + b.imag};
                }
            } else {
                #pragma unroll
                for (unsigned i = 0; i < Ept; ++i)
                    tile[index(row, i, column)] = v[i];
                __syncthreads();
                #pragma unroll
                for (unsigned i = 0; i < Ept; ++i) {
                    const auto peer = tile[index(row ^ half, i, column)];
                    const auto a = upper ? peer : v[i];
                    const auto b = register_tile::multiply(upper ? v[i] : peer, phase);
                    v[i] = upper ? Complex{a.real - b.real, a.imag - b.imag} :
                                   Complex{a.real + b.real, a.imag + b.imag};
                }
                __syncthreads();
            }
          }
        }
        // The row FFT owns frequency i+Ept*row. Restore the caller's cyclic
        // frequency row+i*Rows before its unchanged suffix exchange/store.
        #pragma unroll
        for (unsigned i = 0; i < Ept; ++i) tile[index(row, i, column)] = v[i];
        __syncthreads();
        #pragma unroll
        for (unsigned i = 0; i < Ept; ++i) {
            const unsigned k = row + i * Rows;
            const auto result = tile[index(k / Ept, k % Ept, column)];
            v[i] = result;
        }
        // execute may be composed with a shared-memory consumer immediately.
        __syncthreads();
    }
};

} // namespace cuntt::detail
