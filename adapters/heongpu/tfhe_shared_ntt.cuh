#pragma once

#include <heongpu/kernel/small_ntt.cuh>

namespace cubutterfly::heongpu_tfhe {

// CT/GS stage indexing matches src/rns_fused_ctgs.cuh. Arithmetic is the
// pinned GPU-NTT provider (Apache-2.0), also used by HEonGPU Small*NTT.
// One dependency-closed polynomial stays in its caller-owned linear shared
// allocation. Every stage has a CTA barrier, including the warp-sized tail.
template <unsigned LogN, bool Inverse, typename T>
__device__ __forceinline__ void shared_ntt(
    T* data, const Root<T>* roots, Modulus<T> modulus,
    Ninverse<T> scale, bool cyclic) {
    static_assert(LogN > 0 && LogN <= 10, "TFHE shared tile supports logN 1..10");
    constexpr unsigned n = 1U << LogN;
#pragma unroll
    for (unsigned iteration = 0; iteration < LogN; ++iteration) {
        const unsigned ell = Inverse ? LogN - 1 - iteration : iteration;
        const unsigned half_log = LogN - ell - 1;
        const unsigned half = 1U << half_log;
        for (unsigned unit = threadIdx.x; unit < n / 2; unit += blockDim.x) {
            const unsigned group = unit >> half_log;
            const unsigned left = (group << (half_log + 1)) + (unit & (half - 1));
            const unsigned root = (cyclic ? 0U : (1U << ell)) + group;
            if constexpr (Inverse)
                gpuntt::GentlemanSandeUnit(data[left], data[left + half], roots[root], modulus);
            else
                gpuntt::CooleyTukeyUnit(data[left], data[left + half], roots[root], modulus);
        }
        __syncthreads();
    }
    if constexpr (Inverse) {
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
            data[i] = OPERATOR_GPU<T>::mult(data[i], scale, modulus);
        __syncthreads();
    }
}

}  // namespace cubutterfly::heongpu_tfhe

namespace heongpu {

std::size_t cubutterfly_tfhe_fix_graph(cudaGraph_t graph, bool shared,
                                     std::size_t bootstraps);

__device__ __forceinline__ void cubutterfly_tfhe_forward(
    Data64* data, const Root64* roots, Modulus64 modulus, bool cyclic) {
    cubutterfly::heongpu_tfhe::shared_ntt<10, false>(data, roots, modulus, Ninverse64{}, cyclic);
}

__device__ __forceinline__ void cubutterfly_tfhe_inverse(
    Data64* data, const Root64* roots, Modulus64 modulus, Ninverse64 scale, bool cyclic) {
    cubutterfly::heongpu_tfhe::shared_ntt<10, true>(data, roots, modulus, scale, cyclic);
}

}  // namespace heongpu
