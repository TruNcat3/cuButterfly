#pragma once

#include "xd_axis_resources.hpp"
#include <cuda_runtime_api.h>
#include <cstddef>

namespace cuntt::detail {

// A dependency-closed pair of physically trailing complex axes. Each provider
// computes last axis then penultimate axis, natural order in/out, both signs,
// with one final scale. This contract has no 1D cross-factor twiddle and must
// not be used to flatten a 2D FFT into a 1D FFT of the same element count.
bool xd_pair_core_available(XdPairCore core, std::size_t rows,
                            std::size_t cols, bool fp64,
                            unsigned layout = 0) noexcept;
void prepare_xd_pair_core(XdPairCore core, std::size_t rows,
                         std::size_t cols, bool fp64, bool inverse,
                         unsigned layout = 0);
void launch_xd_pair_core(XdPairCore core, const void* input, void* output,
                        std::size_t rows, std::size_t cols, std::size_t planes,
                        bool fp64, bool inverse, double scale,
                        cudaStream_t stream, unsigned layout = 0);
const char* xd_pair_execution_name(XdPairCore core);
const char* xd_pair_twiddle_policy(XdPairCore core);
unsigned xd_pair_cooperative_lanes(XdPairCore core);

}  // namespace cuntt::detail
