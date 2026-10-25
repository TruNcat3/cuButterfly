#pragma once

#include <stdexcept>

namespace cuntt::detail {

// Shared butterfly tiles: a group owns 2^width coefficients whose local
// coordinate bits [low, low+width) vary. Work units are assigned round-robin
// to a power-of-two number of threads. This geometry is arithmetic-independent.
inline unsigned butterfly_warp_barrier_mask(unsigned local_log, unsigned rows,
                                            unsigned threads, bool inverse,
                                            unsigned radix_log, bool interleaved) {
    if (!local_log || local_log >= 31 || !rows || (rows & (rows - 1)) ||
        threads < 32 || threads > 1024 || (threads & (threads - 1)) ||
        (radix_log != 1 && radix_log != 2))
        throw std::invalid_argument("invalid shared butterfly exchange geometry");
    unsigned rows_log = 0;
    for (unsigned r = rows; r > 1; r >>= 1) ++rows_log;
    if (local_log + rows_log >= 32)
        throw std::invalid_argument("shared butterfly exchange index overflows");
    const auto owner_warp = [&](unsigned s, unsigned low, unsigned width) {
        if (!width) return (s & (threads - 1)) >> 5; // load/store owner
        const unsigned row = interleaved ? s & (rows - 1) : s >> local_log;
        const unsigned u = interleaved ? s >> rows_log : s & ((1u << local_log) - 1);
        const unsigned unit = (u & ((1u << low) - 1)) | ((u >> (low + width)) << low);
        const unsigned linear = interleaved ? (unit << rows_log) | row
                                           : (row << (local_log - width)) | unit;
        return (linear & (threads - 1)) >> 5;
    };
    const auto same_warp = [&](unsigned previous_low, unsigned previous_width,
                               unsigned next_low, unsigned next_width) {
        // Both owner maps are bit projections/permutations without carries.
        // Equality on the complete shared-index basis proves equality on all
        // indices, including row bits and units beyond the first thread wave.
        for (unsigned bit = 0; bit < local_log + rows_log; ++bit)
            if (owner_warp(1u << bit, previous_low, previous_width) !=
                owner_warp(1u << bit, next_low, next_width)) return false;
        return true;
    };
    unsigned mask = 0, boundary = 0, previous_low = 0, previous_width = 0;
    for (unsigned iteration = 0; iteration < local_log; iteration += radix_log) {
        const unsigned width = radix_log == 2 && iteration + 1 < local_log ? 2 : 1;
        const unsigned low = inverse ? iteration : local_log - iteration - width;
        if (same_warp(previous_low, previous_width, low, width)) mask |= 1u << boundary;
        previous_low = low;
        previous_width = width;
        boundary = iteration + 1;
    }
    if (same_warp(previous_low, previous_width, 0, 0)) mask |= 1u << boundary;
    return mask;
}

} // namespace cuntt::detail
