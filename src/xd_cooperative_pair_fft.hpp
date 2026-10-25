#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cuntt::detail {

// Launch one CTA-local two-dimensional C2C transform for every contiguous
// rows x cols plane.  Each one-dimensional pass assigns four warp lanes to a
// line and keeps the line's values in registers; the only cross-axis handoff
// is the natural row-major plane in dynamic shared memory.  The supported
// axes are each 32, 64 or 128.  FP64 128x128 is rejected when the device opt-in
// dynamic shared-memory limit cannot hold its padded plane (and the same
// resource check applies to every shape and precision).
// Exact in-place operation is supported because no global store occurs until
// the whole input plane has crossed the CTA staging barrier.
// `shared_layout=0` selects the established padded row-major tile. `1`
// selects the equal-size XOR-bank row permutation; both layouts preserve
// natural output and use the same bijection for row write, column exchange,
// and export.
bool xd_cooperative_pair_fft_available(std::size_t rows, std::size_t cols,
                                       bool fp64,
                                       unsigned shared_layout = 0) noexcept;

// Prepare one precision/direction/shape specialization. Callers building a
// persistent plan should invoke this before the first launch; it sets the
// opt-in dynamic-shared-memory attribute and verifies occupancy once per
// device/specialization.
void prepare_xd_cooperative_pair_fft(std::size_t rows, std::size_t cols,
                                     bool fp64, bool inverse,
                                     unsigned shared_layout = 0);

void launch_xd_cooperative_pair_fft(const void* input, void* output,
                                    std::size_t rows, std::size_t cols,
                                    std::size_t planes, bool fp64, bool inverse,
                                    double scale, cudaStream_t stream,
                                    unsigned shared_layout = 0);

}  // namespace cuntt::detail
