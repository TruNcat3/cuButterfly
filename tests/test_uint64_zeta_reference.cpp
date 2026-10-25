#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cuntt/butterfly.hpp"

int main() {
    for (const bool superset : {false, true}) {
        for (const bool inverse : {false, true}) {
            std::vector<std::uint64_t> input(32);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = UINT64_MAX - i * (std::uint64_t{1} << 40) + i;
            auto actual = input;
            if (superset) cuntt::reference_superset_zeta(actual, inverse);
            else cuntt::reference_subset_zeta(actual, inverse);
            for (std::size_t target = 0; target < input.size(); ++target) {
                std::uint64_t expected = 0;
                for (std::size_t source = 0; source < input.size(); ++source) {
                    if (superset ? ((target & source) != target) : ((target & source) != source)) continue;
                    const unsigned distance = __builtin_popcount(static_cast<unsigned>(target ^ source));
                    if (inverse && (distance & 1)) expected -= input[source];
                    else expected += input[source];
                }
                if (actual[target] != expected) throw std::runtime_error("UINT64 combinatorial oracle mismatch");
            }
            if (superset) cuntt::reference_superset_zeta(actual, !inverse);
            else cuntt::reference_subset_zeta(actual, !inverse);
            if (actual != input) throw std::runtime_error("UINT64 roundtrip mismatch");
        }
    }
    if (cuntt::parse_butterfly_precision("uint64") != cuntt::ButterflyPrecision::Uint64)
        throw std::runtime_error("UINT64 precision parser mismatch");
    std::cout << "PASS UINT64 zeta direct combinatorial oracle and roundtrip\n";
}
