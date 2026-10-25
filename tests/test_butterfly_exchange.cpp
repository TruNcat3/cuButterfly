#include "butterfly_exchange.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Owners = std::vector<unsigned>;
constexpr unsigned missing = std::numeric_limits<unsigned>::max();

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Owners boundary_owners(unsigned cells, unsigned threads) {
    Owners owners(cells, missing);
    for (unsigned thread = 0; thread < threads; ++thread)
        for (unsigned linear = thread; linear < cells; linear += threads)
            owners[linear] = thread;
    return owners;
}

// Reproduce the kernel's thread/unit loops and shared writes, without using
// the host helper's inverse-index or basis-vector construction.
Owners group_owners(unsigned local_log, unsigned rows, unsigned threads,
                    bool inverse, unsigned radix_log, bool interleaved,
                    unsigned iteration) {
    const unsigned local_n = 1U << local_log;
    const bool radix4 = radix_log == 2 && iteration + 1 < local_log;
    const unsigned width_log = radix4 ? 2 : 1;
    const unsigned units_per_tile = local_n >> width_log;
    const unsigned ell = inverse ? local_log - iteration - width_log : iteration;
    const unsigned stride_log = local_log - ell - width_log;
    const unsigned stride = 1U << stride_log;
    Owners owners(rows * local_n, missing);
    for (unsigned thread = 0; thread < threads; ++thread) {
        for (unsigned linear = thread; linear < rows * units_per_tile; linear += threads) {
            const unsigned row = interleaved ? linear % rows : linear / units_per_tile;
            const unsigned unit = interleaved ? linear / rows : linear % units_per_tile;
            const unsigned group = unit >> stride_log;
            const unsigned offset = unit & (stride - 1);
            const unsigned u0 = (group << (stride_log + width_log)) + offset;
            for (unsigned position = 0; position < (1U << width_log); ++position) {
                const unsigned u = u0 + position * stride;
                const unsigned index = interleaved ? u * rows + row : row * local_n + u;
                require(index < owners.size(), "oracle shared index out of range");
                require(owners[index] == missing, "oracle assigned a shared cell twice");
                owners[index] = thread;
            }
        }
    }
    require(std::find(owners.begin(), owners.end(), missing) == owners.end(),
            "oracle left a shared cell without an owner");
    return owners;
}

bool same_warps(const Owners& before, const Owners& after) {
    require(before.size() == after.size(), "oracle owner map sizes differ");
    for (unsigned index = 0; index < before.size(); ++index)
        if (before[index] / 32 != after[index] / 32) return false;
    return true;
}

unsigned oracle_mask(unsigned local_log, unsigned rows, unsigned threads,
                     bool inverse, unsigned radix_log, bool interleaved,
                     unsigned& warp_boundaries, unsigned& cta_boundaries) {
    const Owners boundary = boundary_owners(rows * (1U << local_log), threads);
    Owners previous = boundary;
    unsigned mask = 0;
    unsigned boundary_bit = 0;
    const auto record = [&](const Owners& next, unsigned bit) {
        if (same_warps(previous, next)) {
            mask |= 1U << bit;
            ++warp_boundaries;
        } else {
            ++cta_boundaries;
        }
    };
    for (unsigned iteration = 0; iteration < local_log; iteration += radix_log) {
        Owners next = group_owners(local_log, rows, threads, inverse, radix_log,
                                   interleaved, iteration);
        record(next, boundary_bit);
        previous = std::move(next);
        boundary_bit = iteration + 1;
    }
    record(boundary, boundary_bit);
    return mask;
}

void expect_invalid(unsigned local_log, unsigned rows, unsigned threads, unsigned radix_log) {
    try {
        (void)cuntt::detail::butterfly_warp_barrier_mask(
            local_log, rows, threads, false, radix_log, false);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("invalid butterfly exchange geometry was accepted");
}
} // namespace

int main() try {
    unsigned geometries = 0;
    unsigned warp_boundaries = 0;
    unsigned cta_boundaries = 0;
    unsigned mixed_geometries = 0;
    unsigned single_warp_geometries = 0;
    for (unsigned local_log = 2; local_log <= 10; ++local_log)
        for (unsigned rows = 1; rows <= 128; rows <<= 1)
            for (unsigned threads : {32U, 64U, 128U, 256U, 512U})
                for (bool inverse : {false, true})
                    for (unsigned radix_log : {1U, 2U})
                        for (bool interleaved : {false, true}) {
                            const unsigned previous_warp = warp_boundaries;
                            const unsigned previous_cta = cta_boundaries;
                            const unsigned expected = oracle_mask(
                                local_log, rows, threads, inverse, radix_log, interleaved,
                                warp_boundaries, cta_boundaries);
                            const unsigned actual = cuntt::detail::butterfly_warp_barrier_mask(
                                local_log, rows, threads, inverse, radix_log, interleaved);
                            if (actual != expected) {
                                std::ostringstream error;
                                error << "barrier mask mismatch: L=" << local_log
                                      << " rows=" << rows << " threads=" << threads
                                      << " inverse=" << inverse << " radix_log=" << radix_log
                                      << " interleaved=" << interleaved << " expected=" << expected
                                      << " actual=" << actual;
                                throw std::runtime_error(error.str());
                            }
                            if (threads == 32) {
                                require(cta_boundaries == previous_cta,
                                        "single-warp geometry requires a CTA boundary");
                                require(warp_boundaries > previous_warp,
                                        "single-warp geometry has no warp boundary");
                                ++single_warp_geometries;
                            }
                            mixed_geometries += warp_boundaries > previous_warp &&
                                                cta_boundaries > previous_cta;
                            ++geometries;
                        }
    require(warp_boundaries > 0, "no warp-local boundary was covered");
    require(cta_boundaries > 0, "no cross-warp boundary was covered");
    require(mixed_geometries > 0, "no geometry mixed warp and CTA boundaries");
    require(single_warp_geometries == 576, "single-warp geometry coverage changed");
    expect_invalid(0, 1, 64, 1);
    expect_invalid(32, 1, 64, 1);
    expect_invalid(4, 0, 64, 1);
    expect_invalid(4, 3, 64, 1);
    expect_invalid(4, 1, 0, 1);
    expect_invalid(4, 1, 16, 1);
    expect_invalid(4, 1, 48, 1);
    expect_invalid(4, 1, 64, 0);
    expect_invalid(4, 1, 64, 3);
    std::cout << "Butterfly exchange oracle passed: " << geometries << " geometries, "
              << warp_boundaries << " warp boundaries, " << cta_boundaries
              << " CTA boundaries, " << mixed_geometries << " mixed geometries\n";
    std::cout << "Single-warp oracle passed: " << single_warp_geometries << " geometries\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
