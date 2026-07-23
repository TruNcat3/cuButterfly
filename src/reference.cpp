#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "cuntt/ntt.hpp"

namespace cuntt {
namespace {

std::uint64_t mod_mul(std::uint64_t a, std::uint64_t b, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % modulus);
}

void bit_reverse(std::vector<std::uint64_t>& values) {
    std::size_t j = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        std::size_t bit = values.size() >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
}

}  // namespace

const char* backend_name(Backend backend) noexcept {
    switch (backend) {
        case Backend::Baseline:
            return "baseline";
        case Backend::Tile256:
            return "tile256";
        case Backend::Hybrid2D:
            return "hybrid2d";
        case Backend::CompactStage:
            return "compact-stage";
        case Backend::StagePipeline:
            return "stage-pipeline";
    }
    return "unknown";
}

Backend parse_backend(const std::string& name) {
    if (name == "baseline") {
        return Backend::Baseline;
    }
    if (name == "tile256") {
        return Backend::Tile256;
    }
    if (name == "hybrid2d" || name == "merge256") {
        return Backend::Hybrid2D;
    }
    if (name == "compact-stage") {
        return Backend::CompactStage;
    }
    if (name == "stage-pipeline") {
        return Backend::StagePipeline;
    }
    throw std::invalid_argument("unknown backend: " + name);
}

const char* stage_handoff_name(StageHandoff handoff) noexcept {
    switch (handoff) {
        case StageHandoff::Atomic:
            return "atomic";
        case StageHandoff::NamedBarrier:
            return "named-barrier";
    }
    return "unknown";
}

StageHandoff parse_stage_handoff(const std::string& name) {
    if (name == "atomic") {
        return StageHandoff::Atomic;
    }
    if (name == "named-barrier") {
        return StageHandoff::NamedBarrier;
    }
    throw std::invalid_argument("unknown stage handoff: " + name);
}

const char* output_order_name(OutputOrder order) noexcept {
    switch (order) {
        case OutputOrder::Natural:
            return "natural";
        case OutputOrder::BitReversed:
            return "bit-reversed";
    }
    return "unknown";
}

OutputOrder parse_output_order(const std::string& name) {
    if (name == "natural") {
        return OutputOrder::Natural;
    }
    if (name == "bit-reversed") {
        return OutputOrder::BitReversed;
    }
    throw std::invalid_argument("unknown output order: " + name);
}

const char* compute_unit_name(ComputeUnit unit) noexcept {
    switch (unit) {
        case ComputeUnit::Auto:
            return "auto";
        case ComputeUnit::Radix2:
            return "radix2";
        case ComputeUnit::Radix4:
            return "radix4";
        case ComputeUnit::Radix8:
            return "radix8";
    }
    return "unknown";
}

ComputeUnit parse_compute_unit(const std::string& name) {
    if (name == "auto") {
        return ComputeUnit::Auto;
    }
    if (name == "radix2") {
        return ComputeUnit::Radix2;
    }
    if (name == "radix4") {
        return ComputeUnit::Radix4;
    }
    if (name == "radix8") {
        return ComputeUnit::Radix8;
    }
    throw std::invalid_argument("unknown compute unit: " + name);
}

const char* cross_twiddle_placement_name(CrossTwiddlePlacement placement) noexcept {
    switch (placement) {
        case CrossTwiddlePlacement::FirstPass:
            return "first";
        case CrossTwiddlePlacement::SecondPass:
            return "second";
        case CrossTwiddlePlacement::Fused:
            return "fused";
        case CrossTwiddlePlacement::FusedBarrett:
            return "fused-barrett";
    }
    return "unknown";
}

CrossTwiddlePlacement parse_cross_twiddle_placement(const std::string& name) {
    if (name == "first") {
        return CrossTwiddlePlacement::FirstPass;
    }
    if (name == "second") {
        return CrossTwiddlePlacement::SecondPass;
    }
    if (name == "fused") {
        return CrossTwiddlePlacement::Fused;
    }
    if (name == "fused-barrett") {
        return CrossTwiddlePlacement::FusedBarrett;
    }
    throw std::invalid_argument("unknown cross-twiddle placement: " + name);
}

const char* modular_multiply_name(ModularMultiply multiply) noexcept {
    switch (multiply) {
        case ModularMultiply::Shoup:
            return "shoup";
        case ModularMultiply::Barrett:
            return "barrett";
    }
    return "unknown";
}

ModularMultiply parse_modular_multiply(const std::string& name) {
    if (name == "shoup") {
        return ModularMultiply::Shoup;
    }
    if (name == "barrett") {
        return ModularMultiply::Barrett;
    }
    throw std::invalid_argument("unknown modular multiply: " + name);
}

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    if (modulus < 2) {
        throw std::invalid_argument("modulus must be at least 2");
    }
    std::uint64_t result = 1;
    base %= modulus;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exponent >>= 1;
    }
    return result;
}

bool is_prime(std::uint64_t value) {
    if (value < 2) {
        return false;
    }
    constexpr std::array<std::uint64_t, 12> kSmallPrimes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (const auto prime : kSmallPrimes) {
        if (value % prime == 0) {
            return value == prime;
        }
    }

    std::uint64_t odd_part     = value - 1;
    std::uint32_t power_of_two = 0;
    while ((odd_part & 1U) == 0) {
        odd_part >>= 1;
        ++power_of_two;
    }

    constexpr std::array<std::uint64_t, 7> kWitnesses = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (const auto witness : kWitnesses) {
        if (witness % value == 0) {
            continue;
        }
        std::uint64_t result = mod_pow(witness, odd_part, value);
        if (result == 1 || result == value - 1) {
            continue;
        }
        bool probably_prime = false;
        for (std::uint32_t round = 1; round < power_of_two; ++round) {
            result = mod_mul(result, result, value);
            if (result == value - 1) {
                probably_prime = true;
                break;
            }
        }
        if (!probably_prime) {
            return false;
        }
    }
    return true;
}

std::uint64_t find_primitive_power_of_two_root(std::uint32_t log_n, std::uint64_t modulus) {
    if (log_n == 0 || log_n >= 63) {
        throw std::invalid_argument("log_n must be in [1, 62]");
    }
    if (!is_prime(modulus)) {
        throw std::invalid_argument("modulus must be prime");
    }
    const std::uint64_t n = 1ULL << log_n;
    if ((modulus - 1) % n != 0) {
        throw std::invalid_argument("modulus - 1 is not divisible by the requested NTT length");
    }
    for (std::uint64_t candidate = 2; candidate < (1ULL << 20); ++candidate) {
        const std::uint64_t root = mod_pow(candidate, (modulus - 1) / n, modulus);
        if (mod_pow(root, n, modulus) == 1 && mod_pow(root, n >> 1, modulus) != 1) {
            return root;
        }
    }
    throw std::runtime_error("could not find a primitive power-of-two root");
}

void reference_ntt(std::vector<std::uint64_t>& values, std::uint64_t modulus, bool inverse) {
    const std::size_t n = values.size();
    if (n < 2 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("NTT length must be a power of two of at least two");
    }
    if (modulus >= (1ULL << 63)) {
        throw std::invalid_argument("cuNTT currently requires modulus < 2^63");
    }

    std::uint32_t log_n = 0;
    for (std::size_t size = n; size > 1; size >>= 1) {
        ++log_n;
    }
    std::uint64_t root = find_primitive_power_of_two_root(log_n, modulus);
    if (inverse) {
        root = mod_pow(root, modulus - 2, modulus);
    }

    for (auto& value : values) {
        value %= modulus;
    }
    bit_reverse(values);

    for (std::size_t length = 2; length <= n; length <<= 1) {
        const std::size_t   half      = length >> 1;
        const std::uint64_t step_root = mod_pow(root, n / length, modulus);
        for (std::size_t start = 0; start < n; start += length) {
            std::uint64_t omega = 1;
            for (std::size_t offset = 0; offset < half; ++offset) {
                const std::size_t   left = start + offset;
                const std::uint64_t u    = values[left];
                const std::uint64_t v    = mod_mul(values[left + half], omega, modulus);
                const std::uint64_t sum  = u + v;
                values[left]             = sum >= modulus ? sum - modulus : sum;
                values[left + half]      = u >= v ? u - v : modulus + u - v;
                omega                    = mod_mul(omega, step_root, modulus);
            }
        }
    }

    if (inverse) {
        const std::uint64_t inverse_n = mod_pow(static_cast<std::uint64_t>(n), modulus - 2, modulus);
        for (auto& value : values) {
            value = mod_mul(value, inverse_n, modulus);
        }
    }
}

}  // namespace cuntt
