#pragma once
#include "cuntt/butterfly.hpp"
#include <cufftdx.hpp>
#ifndef CUBUTTERFLY_CUFFTDX_SM
#error "Define CUBUTTERFLY_CUFFTDX_SM from the selected CUDA architecture"
#endif

namespace cuntt::detail::register_tile {
template <unsigned Log, bool Inverse>
struct DxCodelet {
    __device__ __forceinline__ void operator()(Complex32 (&v)[1U << Log]) const {
        using FFT = decltype(cufftdx::Thread() + cufftdx::Size<(1U<<Log)>() +
            cufftdx::Type<cufftdx::fft_type::c2c>() +
            cufftdx::Direction<Inverse ? cufftdx::fft_direction::inverse : cufftdx::fft_direction::forward>() +
            cufftdx::Precision<float>() + cufftdx::SM<CUBUTTERFLY_CUFFTDX_SM>());
        typename FFT::value_type data[FFT::storage_size];
#pragma unroll
        for (unsigned i=0; i<(1U<<Log); ++i) data[i] = typename FFT::value_type{v[i].real, v[i].imag};
        FFT().execute(data);
#pragma unroll
        for (unsigned i=0; i<(1U<<Log); ++i) v[i] = {data[i].x, data[i].y};
    }
};
} // namespace cuntt::detail::register_tile
