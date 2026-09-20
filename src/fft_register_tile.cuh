#pragma once

#include "cuntt/butterfly.hpp"
#include <cuda_runtime.h>
#include <type_traits>

namespace cuntt::detail::register_tile {
template<class Complex>
__device__ __forceinline__ Complex multiply(Complex a, Complex b) {
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

template <bool Inverse, class Complex = Complex32>
__device__ __forceinline__ Complex root(decltype(Complex{}.real) turns) {
    using Real = decltype(Complex{}.real);
    Real s, c;
    // The phase is a binary rational for these power-of-two FFTs. Preserve it
    // through pi-scaled range reduction instead of first rounding 2*pi*x.
    if constexpr (std::is_same_v<Real, double>) sincospi((Inverse ? 2.0 : -2.0) * turns, &s, &c);
    else sincospif((Inverse ? 2.0f : -2.0f) * turns, &s, &c);
    return {c, s};
}

// Small FFT with compile-time butterfly indices and immediate roots.  No
// shared/global loads and no cross-thread synchronization inside this unit.
template <unsigned Log, bool Inverse, class Complex>
__device__ __forceinline__ void fft(Complex (&v)[1U << Log]) {
    using Real = decltype(Complex{}.real);
    constexpr unsigned N = 1U << Log;
#pragma unroll
    for (unsigned i = 0; i < N; ++i) {
        const unsigned j = __brev(i) >> (32 - Log);
        if (i < j) { const auto t = v[i]; v[i] = v[j]; v[j] = t; }
    }
#pragma unroll
    for (unsigned stage = 0; stage < Log; ++stage) {
        const unsigned half = 1U << stage;
#pragma unroll
        for (unsigned p = 0; p < N / 2; ++p) {
            const unsigned j = p & (half - 1);
            const unsigned a = 2 * (p - j) + j;
            const auto x = v[a];
            const auto y = multiply(v[a + half], root<Inverse, Complex>(Real(j) / (2 * half)));
            v[a] = {x.real + y.real, x.imag + y.imag};
            v[a + half] = {x.real - y.real, x.imag - y.imag};
        }
    }
}

template <unsigned Log, bool Inverse>
struct NativeCodelet {
    __device__ __forceinline__ void operator()(Complex32 (&v)[1U << Log]) const {
        fft<Log, Inverse>(v);
    }
};

__host__ __device__ constexpr unsigned power_log2(unsigned value) {
    unsigned result = 0;
    while (value > 1) { value >>= 1; ++result; }
    return result;
}

// A radix-4 DIF exchange leaves lane frequencies in bit-reversed order.
// Carry that order into the existing write addresses instead of shuffling
// registers solely to restore natural lane order between local transforms.
template <unsigned Lanes>
__device__ __forceinline__ unsigned codelet_output_lane(unsigned lane) {
    if constexpr (Lanes == 4) return ((lane & 1U) << 1) | (lane >> 1);
    else return lane;
}

// x[G*j+lane] forms G interleaved length-R/G transforms. The diagonal
// W_R^(lane*k), followed by a G-point lane FFT, gives
// X[k+(R/G)*codelet_output_lane(lane)].
// Independent columns still occupy the low thread-index bits, preserving
// the original coalesced global transactions as more lanes share a codelet.
template <unsigned Log, unsigned Columns, unsigned Lanes, bool Inverse,
          template <unsigned, bool> class Codelet, class Complex>
__device__ __forceinline__ void cooperative_fft(Complex (&v)[(1U << Log) / Lanes], unsigned lane) {
    static_assert(Lanes && !(Lanes & (Lanes-1)) && Lanes <= (1U << Log));
    static_assert(Lanes == 1 || Lanes * Columns <= 32, "codelet shuffle group must fit a warp");
    constexpr unsigned LaneLog = power_log2(Lanes);
    constexpr unsigned K = (1U << Log) / Lanes;
    using Real = decltype(Complex{}.real);
    if constexpr (Log > LaneLog) {
        if constexpr (std::is_same_v<Complex, Complex32>) Codelet<Log-LaneLog, Inverse>{}(v);
        else fft<Log-LaneLog, Inverse>(v);
    }
    if constexpr (Lanes == 2 || Lanes == 4) {
        const unsigned mask = __activemask();
#pragma unroll
        for (unsigned k=0; k<K; ++k) {
            auto value = v[k];
            if (k && lane) {
                // Select a compile-time root, then issue one complex multiply.
                // Multiplying separately under each lane predicate makes all
                // three radix-4 phase branches consume warp issue bandwidth.
                Complex phase{Real(1), Real(0)};
#pragma unroll
                for (unsigned p=1; p<Lanes; ++p)
                    if (lane == p) phase = root<Inverse, Complex>(Real(p*k) / (1U << Log));
                value = multiply(value, phase);
            }
            if constexpr (Lanes == 4) {
                const Complex peer{__shfl_xor_sync(mask, value.real, 2*Columns),
                                   __shfl_xor_sync(mask, value.imag, 2*Columns)};
                const bool upper = (lane & 2) != 0;
                value = {(upper ? -value.real : value.real) + peer.real,
                         (upper ? -value.imag : value.imag) + peer.imag};
                if (lane == 3)
                    value = Inverse ? Complex{-value.imag, value.real} :
                                      Complex{value.imag, -value.real};
            }
            const Complex peer{__shfl_xor_sync(mask, value.real, Columns),
                               __shfl_xor_sync(mask, value.imag, Columns)};
            const bool upper = (lane & 1) != 0;
            v[k] = {(upper ? -value.real : value.real) + peer.real,
                    (upper ? -value.imag : value.imag) + peer.imag};
        }
    } else if constexpr (Lanes > 1) {
        const unsigned mask = __activemask();
        const unsigned reversed = __brev(lane) >> (32-LaneLog);
        const unsigned source = (threadIdx.x & 31U) ^ ((lane ^ reversed) * Columns);
#pragma unroll
        for (unsigned k=0; k<K; ++k) {
            auto value = v[k];
            // Both loop bounds are compile-time constants. Keep each phase
            // constant so an expensive runtime sincos is not introduced.
#pragma unroll
            for (unsigned p=1; p<Lanes; ++p)
                if (lane == p) value = multiply(value, root<Inverse, Complex>(Real(p*k) / (1U << Log)));
            value = {__shfl_sync(mask, value.real, source), __shfl_sync(mask, value.imag, source)};
#pragma unroll
            for (unsigned stage=0; stage<LaneLog; ++stage) {
                const unsigned half=1U << stage;
                const Complex peer{__shfl_xor_sync(mask, value.real, half*Columns),
                                   __shfl_xor_sync(mask, value.imag, half*Columns)};
                const bool upper = (lane & half) != 0;
                const auto x = upper ? peer : value;
                auto y = upper ? value : peer;
#pragma unroll
                for (unsigned j=1; j<half; ++j)
                    if ((lane & (half-1)) == j)
                        y = multiply(y, root<Inverse, Complex>(Real(j) / (2*half)));
                value = upper ? Complex{x.real-y.real, x.imag-y.imag} :
                                Complex{x.real+y.real, x.imag+y.imag};
            }
            v[k] = value;
        }
    }
}

template <unsigned Bits, unsigned Rotation>
__device__ __forceinline__ unsigned rotate_shared_rows(unsigned value) {
    if constexpr (!Bits || Rotation % Bits == 0) return value;
    else {
        constexpr unsigned mask=(1U << Bits)-1, shift=Rotation % Bits;
        const unsigned low=value & mask;
        return (value & ~mask) | ((low >> shift) | ((low << (Bits-shift)) & mask));
    }
}

// A bijective write offset for each row. The lane-frequency digit contributes
// to bank selection; rotate the low row bits so both sides of the transpose
// distribute the cooperative lanes across banks. Lanes=1 is the old XOR map.
template <unsigned R, unsigned Columns, unsigned Lanes, class Complex, bool Xor>
__device__ __forceinline__ unsigned shared_row(unsigned a, unsigned b) {
    if constexpr (!Xor) return b;
    else if constexpr (Lanes == 1) return b ^ a;
    else {
        constexpr unsigned rows=128U / (sizeof(Complex)*Columns);
        constexpr unsigned Bits=power_log2(rows < R ? rows : R);
        constexpr unsigned G=power_log2(Lanes);
        constexpr unsigned H=G < Bits ? G : Bits;
        constexpr unsigned mask=(1U << Bits)-1;
        const unsigned lane_digit=(a >> power_log2(R/Lanes)) << (Bits-H);
        return rotate_shared_rows<Bits,H>(b) ^ a ^
               (rotate_shared_rows<Bits,H>(lane_digit) & mask);
    }
}

// A dependency-closed local FFT is split between register codelets and one
// shared-memory handoff. Lanes run across independent columns, so global
// input and writer-aligned output do not need a shared staging conversion.
// The full long-FFT boundary remains global: this tile is not a whole-N CTA.
template <unsigned LocalLog, unsigned Columns, bool Inverse,
          template <unsigned, bool> class Codelet = NativeCodelet, class Complex = Complex32,
          bool XorSharedLayout = false, unsigned CodeletLanes = 1>
__launch_bounds__((1U << (LocalLog / 2)) * Columns * CodeletLanes)
__global__ void prefix(const Complex* input, Complex* scratch,
                       unsigned log_n, std::uint64_t batch_distance,
                       std::uint64_t element_stride) {
    static_assert(LocalLog % 2 == 0);
    using Real = decltype(Complex{}.real);
    constexpr unsigned R = 1U << (LocalLog / 2);
    constexpr unsigned LocalN = R * R;
    constexpr unsigned K = R / CodeletLanes;
    static_assert(CodeletLanes && !(CodeletLanes & (CodeletLanes-1)) && CodeletLanes <= R);
    static_assert(CodeletLanes == 1 || CodeletLanes*Columns <= 32);
    const unsigned remaining_log = log_n - LocalLog;
    const unsigned remaining = 1U << remaining_log;
    const unsigned row = threadIdx.x / (Columns*CodeletLanes);
    const unsigned lane = (threadIdx.x / Columns) % CodeletLanes;
    const unsigned frequency_lane = codelet_output_lane<CodeletLanes>(lane);
    const unsigned col = threadIdx.x % Columns;
    const std::uint64_t transform = std::uint64_t(blockIdx.x) * Columns + col;
    const unsigned n2 = transform & (remaining - 1);
    const std::uint64_t base = (transform >> remaining_log) * batch_distance;
    Complex v[K];
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        v[j] = input[base + (std::uint64_t((j*CodeletLanes+lane) * R + row) * remaining + n2) * element_stride];
    cooperative_fft<LocalLog/2,Columns,CodeletLanes,Inverse,Codelet>(v,lane);
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* tile = reinterpret_cast<Complex*>(shared_storage);
#pragma unroll
    for (unsigned j = 0; j < K; ++j) {
        const unsigned k=j+K*frequency_lane;
        tile[(k * R + shared_row<R,Columns,CodeletLanes,Complex,XorSharedLayout>(k,row)) * Columns + col] =
            multiply(v[j], root<Inverse, Complex>(Real(k * row) / LocalN));
    }
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < K; ++j)
        v[j] = tile[(row * R + shared_row<R,Columns,CodeletLanes,Complex,XorSharedLayout>(row,j*CodeletLanes+lane)) * Columns + col];
    cooperative_fft<LocalLog/2,Columns,CodeletLanes,Inverse,Codelet>(v,lane);
    const auto step = root<Inverse, Complex>(Real(R * n2) / Real(1U << log_n));
    auto cross = root<Inverse, Complex>(Real((row+R*K*frequency_lane) * n2) / Real(1U << log_n));
#pragma unroll
    for (unsigned j = 0; j < K; ++j) {
        const unsigned k=j+K*frequency_lane;
        if constexpr (R > 8)
            if (j && (j % 8) == 0)
                cross = root<Inverse, Complex>(Real((row + R*k) * n2) / Real(1U << log_n));
        const auto value = multiply(v[j], cross);
        scratch[base + (std::uint64_t(row + R * k) * remaining + n2) * element_stride] = value;
        cross = multiply(cross, step);
    }
}

// Two independent local FFT factors share one resident tile. Equal EPT K
// permits distinct cooperative widths GA=A/K and GB=B/K without idle roles.
template<unsigned LogA,unsigned LogB,unsigned K,unsigned Columns,class Complex,bool Xor>
__device__ __forceinline__ unsigned rectangular_row(unsigned a,unsigned b) {
    if constexpr (!Xor) return b;
    else {
        constexpr unsigned A=1U<<LogA,B=1U<<LogB,GA=A/K,GB=B/K;
        constexpr unsigned rows=128U/(sizeof(Complex)*Columns);
        constexpr unsigned Bits=power_log2(rows<B?rows:B);
        constexpr unsigned LA=power_log2(GA),LB=power_log2(GB);
        static_assert(LA<=Bits && LB<=Bits,"lane digits must fit a shared transaction");
        constexpr unsigned mask=(1U<<Bits)-1;
        const unsigned column_digit=(a<<LB)&mask;
        const unsigned lane_digit=((a/K)<<(Bits-LA))&mask;
        return rotate_shared_rows<Bits,LA>(b) ^
               rotate_shared_rows<Bits,LA>(column_digit) ^
               rotate_shared_rows<Bits,LA>(lane_digit);
    }
}
template<unsigned LogA,unsigned LogB,unsigned K,unsigned Columns,bool Inverse,
         template<unsigned,bool> class Codelet=NativeCodelet,class Complex=Complex32,bool Xor=false>
__launch_bounds__((1U<<(LogA+LogB))*Columns/K)
__global__ void rectangular_prefix(const Complex* input,Complex* scratch,
                                  unsigned log_n,std::uint64_t distance,std::uint64_t stride) {
    constexpr unsigned A=1U<<LogA,B=1U<<LogB,LocalN=A*B;
    static_assert(K && !(K&(K-1)) && K<=A && K<=B);
    constexpr unsigned GA=A/K,GB=B/K;
    static_assert((GA==1 || GA*Columns<=32) && (GB==1 || GB*Columns<=32));
    using Real=decltype(Complex{}.real);
    const unsigned col=threadIdx.x%Columns;
    const unsigned row_a=threadIdx.x/(Columns*GA),lane_a=(threadIdx.x/Columns)%GA;
    const unsigned row_b=threadIdx.x/(Columns*GB),lane_b=(threadIdx.x/Columns)%GB;
    const unsigned remaining_log=log_n-LogA-LogB,remaining=1U<<remaining_log;
    const std::uint64_t transform=std::uint64_t(blockIdx.x)*Columns+col;
    const unsigned n2=transform&(remaining-1);
    const std::uint64_t base=(transform>>remaining_log)*distance;
    Complex v[K];
#pragma unroll
    for(unsigned j=0;j<K;++j)
        v[j]=input[base+(std::uint64_t((j*GA+lane_a)*B+row_a)*remaining+n2)*stride];
    cooperative_fft<LogA,Columns,GA,Inverse,Codelet>(v,lane_a);
    extern __shared__ __align__(16) unsigned char storage[];
    auto* tile=reinterpret_cast<Complex*>(storage);
#pragma unroll
    for(unsigned j=0;j<K;++j) {
        const unsigned k_a=j+K*codelet_output_lane<GA>(lane_a);
        tile[(k_a*B+rectangular_row<LogA,LogB,K,Columns,Complex,Xor>(k_a,row_a))*Columns+col]=
            multiply(v[j],root<Inverse,Complex>(Real(k_a*row_a)/LocalN));
    }
    __syncthreads();
#pragma unroll
    for(unsigned j=0;j<K;++j)
        v[j]=tile[(row_b*B+rectangular_row<LogA,LogB,K,Columns,Complex,Xor>(row_b,j*GB+lane_b))*Columns+col];
    cooperative_fft<LogB,Columns,GB,Inverse,Codelet>(v,lane_b);
    const unsigned frequency_lane=codelet_output_lane<GB>(lane_b);
    const auto step=root<Inverse,Complex>(Real(A*n2)/Real(1U<<log_n));
    auto cross=root<Inverse,Complex>(Real((row_b+A*K*frequency_lane)*n2)/Real(1U<<log_n));
#pragma unroll
    for(unsigned j=0;j<K;++j) {
        const unsigned k_b=j+K*frequency_lane;
        if constexpr (K>8)
            if(j && j%8==0)cross=root<Inverse,Complex>(Real((row_b+A*k_b)*n2)/Real(1U<<log_n));
        scratch[base+(std::uint64_t(row_b+A*k_b)*remaining+n2)*stride]=multiply(v[j],cross);
        cross=multiply(cross,step);
    }
}
template<unsigned LogA,unsigned LogB,unsigned K,unsigned Columns,bool Inverse=false,
         template<unsigned,bool> class Codelet=NativeCodelet,class Complex=Complex32,bool Xor=false>
void launch_rectangular_prefix(const Complex* input,Complex* scratch,unsigned log_n,
                              std::uint64_t batch,std::uint64_t distance,
                              std::uint64_t stride=1,cudaStream_t stream=nullptr) {
    constexpr unsigned bytes=(1U<<(LogA+LogB))*Columns*sizeof(Complex);
    constexpr unsigned threads=(1U<<(LogA+LogB))*Columns/K;
    if constexpr(bytes>48U*1024U)
        cudaFuncSetAttribute(rectangular_prefix<LogA,LogB,K,Columns,Inverse,Codelet,Complex,Xor>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
    rectangular_prefix<LogA,LogB,K,Columns,Inverse,Codelet,Complex,Xor>
        <<<static_cast<unsigned>((batch<<(log_n-LogA-LogB))/Columns),threads,bytes,stream>>>
        (input,scratch,log_n,distance,stride);
}

template <unsigned LocalLog, unsigned Columns, bool Inverse = false,
          template <unsigned, bool> class Codelet = NativeCodelet, class Complex = Complex32,
          bool XorSharedLayout = false, unsigned CodeletLanes = 1>
void launch_prefix(const Complex* input, Complex* scratch, unsigned log_n,
                   std::uint64_t batch, std::uint64_t batch_distance,
                   std::uint64_t element_stride = 1, cudaStream_t stream = nullptr) {
    constexpr unsigned threads = (1U << (LocalLog / 2)) * Columns * CodeletLanes;
    constexpr unsigned bytes = (1U << LocalLog) * Columns * sizeof(Complex);
    if constexpr (bytes > 48U * 1024U)
        cudaFuncSetAttribute(prefix<LocalLog, Columns, Inverse, Codelet, Complex, XorSharedLayout,CodeletLanes>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    const auto blocks = static_cast<unsigned>((batch << (log_n - LocalLog)) / Columns);
    prefix<LocalLog, Columns, Inverse, Codelet, Complex, XorSharedLayout,CodeletLanes><<<blocks, threads, bytes, stream>>>(
        input, scratch, log_n, batch_distance, element_stride);
}
} // namespace cuntt::detail::register_tile
