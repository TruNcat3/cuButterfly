#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace cuntt::detail {

// Mirror executor precision normalization for integer zeta operators before
// computing storage. Keep this templated so the capacity contract is CPU-only.
template <typename Config>
inline unsigned online_reorder_value_bytes(const Config& config) {
    using Op = decltype(config.op);
    using Precision = decltype(config.precision);
    const bool integer_zeta = config.op == Op::SubsetZeta || config.op == Op::SupersetZeta || config.op == Op::XorZeta;
    const unsigned scalar_bytes = integer_zeta ? (config.precision == Precision::Uint64 ? 8 : 4) :
        (config.precision == Precision::Fp64 || config.precision == Precision::Uint64 ? 8 :
         (config.precision == Precision::Fp16 || config.precision == Precision::Bf16 ? 2 : 4));
    return scalar_bytes * (config.op == Op::Fft ? 2 : 1);
}

struct OnlineReorderShape {
    bool legal = false;
    std::uint64_t prefix_values = 0, suffix_values = 0;
    std::uint64_t prefix_shared_bytes = 0, suffix_shared_bytes = 0, max_shared_bytes = 0;
};

// XOR shared-memory permutations are bijections and need no padding storage.
inline OnlineReorderShape online_reorder_shape(unsigned log_n, unsigned local_stages,
        unsigned prefix_tiles, unsigned suffix_columns, unsigned value_bytes,
        std::uint64_t shared_limit_bytes = 48 * 1024) {
    OnlineReorderShape shape;
    if (local_stages < 5 || local_stages > 10 || log_n <= local_stages ||
        log_n - local_stages > 10 || !value_bytes) return shape;
    const auto power2 = [](unsigned n) { return n && !(n & (n - 1)); };
    const auto local_n = std::uint64_t{1} << local_stages;
    const auto remaining_n = std::uint64_t{1} << (log_n - local_stages);
    if (!power2(prefix_tiles) || !power2(suffix_columns) ||
        prefix_tiles > remaining_n || suffix_columns > local_n) return shape;
    shape.prefix_values = local_n * prefix_tiles;
    shape.suffix_values = remaining_n * suffix_columns;
    shape.prefix_shared_bytes = shape.prefix_values * value_bytes;
    shape.suffix_shared_bytes = shape.suffix_values * value_bytes;
    shape.max_shared_bytes = std::max(shape.prefix_shared_bytes, shape.suffix_shared_bytes);
    // Compiled exponents bound each tile, while grouped storage is limited by
    // the hardware byte capacity and the executor's non-opt-in shared limit.
    shape.legal = shape.max_shared_bytes <=
        std::min<std::uint64_t>(shared_limit_bytes, 48 * 1024);
    return shape;
}

inline std::vector<std::pair<unsigned, unsigned>> online_reorder_group_candidates(
        unsigned log_n, unsigned local_stages, unsigned value_bytes,
        std::uint64_t shared_limit_bytes = 48 * 1024) {
    std::vector<std::pair<unsigned, unsigned>> result;
    const auto append = [&](unsigned g, unsigned c) {
        const auto pair = std::make_pair(g, c);
        if (online_reorder_shape(log_n, local_stages, g, c, value_bytes, shared_limit_bytes).legal &&
            std::find(result.begin(), result.end(), pair) == result.end()) result.push_back(pair);
    };
    append(1, 1);
    for (unsigned group = 2; group <= 1024; group *= 2) {
        append(group, 1);
        append(1, group);
        append(group, group);
    }
    return result;
}

} // namespace cuntt::detail
