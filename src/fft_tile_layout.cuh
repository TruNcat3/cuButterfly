#pragma once
#include <cstdint>

namespace cuntt::detail {

// Placement for a complex-FP32 tile [element][FFT slot]. CUDA shared memory
// has 32 four-byte banks, hence 16 complex values per bank cycle. Keep each
// row intact and XOR only its slot bits, so vector global I/O stays unchanged.
// All tile writers and readers use this contract; the cuFFTDx work area is
// reused only after a CTA barrier and does not see the permutation.
template <unsigned Slots, bool Swizzled>
struct FftTileLayout {
    static_assert(Slots && !(Slots & (Slots - 1)), "FFT slots must be a power of two");
    static constexpr unsigned element_shift() {
        unsigned shift = 0;
        for (unsigned width = Slots; width < 16; width *= 2) ++shift;
        return shift;
    }
    static constexpr unsigned shift = element_shift();
#ifdef __CUDACC__
    __host__ __device__
#endif
    static constexpr std::uint32_t index(std::uint32_t element, std::uint32_t slot) {
        if constexpr (Swizzled) slot ^= (element >> shift) & (Slots - 1);
        return element * Slots + slot;
    }
};

} // namespace cuntt::detail
