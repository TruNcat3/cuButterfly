#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace {

template <typename Real>
void check_roundtrip(std::uint32_t log_n, const std::vector<cuntt::ButterflyMatrix2x2>& matrices, double tolerance) {
    std::vector<Real> values(std::size_t{1} << log_n);
    std::iota(values.begin(), values.end(), static_cast<Real>(-11));
    const auto original = values;
    cuntt::reference_structured_2x2(values, matrices, false);
    cuntt::reference_structured_2x2(values, matrices, true);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double error = std::abs(static_cast<double>(values[index] - original[index]));
        if (error > tolerance * std::max(1.0, std::abs(static_cast<double>(original[index]))))
            throw std::runtime_error("structured reference roundtrip mismatch");
    }
}

}  // namespace

int main() {
    try {
        check_roundtrip<double>(6, {{1.0, 0.25, -0.5, 1.0}}, 1.0e-12);
        check_roundtrip<float>(6, {{1.0, 0.25, -0.5, 1.0}}, 2.0e-5);

        std::vector<cuntt::ButterflyMatrix2x2> per_stage;
        for (std::uint32_t stage = 0; stage < 6; ++stage) {
            const double delta = 0.01 * stage;
            per_stage.push_back({1.0 + delta, 0.125, -0.0625, 0.875 - 0.5 * delta});
        }
        check_roundtrip<double>(6, per_stage, 1.0e-11);
        check_roundtrip<float>(6, per_stage, 5.0e-5);

        try {
            std::vector<double> values(8, 1.0);
            cuntt::reference_structured_2x2(values, {{1.0, 2.0, 2.0, 4.0}}, true);
            throw std::runtime_error("singular structured matrix was accepted");
        } catch (const std::invalid_argument&) {
        }
        std::cout << "All structured reference tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
