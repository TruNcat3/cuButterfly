#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define CUB_ITERATIVE_HD __host__ __device__
#else
#define CUB_ITERATIVE_HD
#endif
namespace cuntt {
// Move the consumer's butterfly pair bit to the highest tile bit. Producers
// store using the next consumer stage. A linearized URAM bank formula is not
// assumed to describe CUDA shared-memory banking.
CUB_ITERATIVE_HD constexpr std::uint32_t iterative_layout_index(
    std::uint32_t logical, std::uint32_t stages, std::uint32_t consumer_stage) {
    const auto low_mask = (1U << consumer_stage) - 1;
    return (logical & low_mask) | ((logical >> (consumer_stage + 1)) << consumer_stage) |
           (((logical >> consumer_stage) & 1U) << (stages - 1));
}
// One tile owns all indices differing in [first_stage, first_stage+stages).
CUB_ITERATIVE_HD constexpr std::uint32_t iterative_global_index(
    std::uint32_t tile, std::uint32_t local, std::uint32_t first_stage, std::uint32_t stages) {
    return ((tile >> first_stage) << (first_stage + stages)) |
           (tile & ((1U << first_stage) - 1)) | (local << first_stage);
}

// Materialize a boundary in the layout consumed by the following group.  The
// consumer's tile id occupies the high bits of the physical address and its
// local stage bits occupy the low bits, so each consumer tile reads contiguous
// global elements while preserving the logical index represented by `global`.
CUB_ITERATIVE_HD constexpr std::uint32_t iterative_boundary_index(
    std::uint32_t global, std::uint32_t first_stage, std::uint32_t stages) {
    const auto tile = ((global >> (first_stage + stages)) << first_stage) |
                      (global & ((1U << first_stage) - 1));
    const auto local = (global >> first_stage) & ((1U << stages) - 1);
    return (tile << stages) | local;
}
} // namespace cuntt
#undef CUB_ITERATIVE_HD
