#pragma once

// Design reference (not a source-code transplant): VkFFT, commit
// 066a17c17068c0f11c9298d848c2976c71fad1c1, inlineRadixKernelVkFFT,
// case 4, stageSize == 1:
// https://github.com/DTolm/VkFFT/blob/066a17c17068c0f11c9298d848c2976c71fad1c1/vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h
// VkFFT is MIT licensed, copyright Dmitrii Tolmachev (2021-present).
// Its two sum/difference pairs, direction-dependent +/-i rotation and final
// {0,2,1,3} permutation correspond to the formulas below. This independently
// written implementation directly assigns natural-order outputs and omits
// VkFFT's stage twiddles, code generator, and explicit final permutation.

namespace cuntt::detail {

// Complex must be a copyable complex-like type with float or double members x and y.
// The four references must name distinct objects. Input: v_n = x[n], n=0..3.
// Output: v_k = sum_n x[n] exp(s * 2*pi*i*n*k/4), k=0..3 (natural order).
// s=-1 for Inverse=false (forward); s=+1 for Inverse=true (inverse).
// Neither direction scales its output: inverse(forward(x)) = 4*x.
// This is a register-only four-point DFT; external stage twiddles belong to
// the caller. It uses 16 scalar adds/subtracts and sign changes, no multiplies.
template <bool Inverse, typename Complex>
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
void fft_radix4(Complex& v0, Complex& v1, Complex& v2, Complex& v3) {
    Complex even_sum = v0;
    Complex even_diff = v0;
    Complex odd_sum = v1;
    Complex odd_diff = v1;
    even_sum.x = v0.x + v2.x;
    even_sum.y = v0.y + v2.y;
    even_diff.x = v0.x - v2.x;
    even_diff.y = v0.y - v2.y;
    odd_sum.x = v1.x + v3.x;
    odd_sum.y = v1.y + v3.y;
    odd_diff.x = v1.x - v3.x;
    odd_diff.y = v1.y - v3.y;

    // rotated = (Inverse ? +i : -i) * odd_diff.
    Complex rotated = odd_diff;
    if constexpr (Inverse) {
        rotated.x = -odd_diff.y;
        rotated.y = odd_diff.x;
    } else {
        rotated.x = odd_diff.y;
        rotated.y = -odd_diff.x;
    }
    v0.x = even_sum.x + odd_sum.x;
    v0.y = even_sum.y + odd_sum.y;
    v2.x = even_sum.x - odd_sum.x;
    v2.y = even_sum.y - odd_sum.y;
    v1.x = even_diff.x + rotated.x;
    v1.y = even_diff.y + rotated.y;
    v3.x = even_diff.x - rotated.x;
    v3.y = even_diff.y - rotated.y;
}

template <bool Inverse, typename Complex>
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
void fft_radix4(Complex (&v)[4]) {
    fft_radix4<Inverse>(v[0], v[1], v[2], v[3]);
}

} // namespace cuntt::detail
