#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include "cuntt/butterfly.hpp"

int main() {
    try {
        cuntt::ButterflyConfig config;
        config.op                = cuntt::ButterflyOperator::Fft;
        config.backend           = cuntt::ButterflyBackend::CuFft;
        config.precision         = cuntt::ButterflyPrecision::Fp32;
        config.placement         = cuntt::ButterflyPlacement::OutOfPlace;
        config.log_n             = 10;
        config.batch             = 4;
        config.normalize_inverse = true;

        const std::size_t points = std::size_t{1} << config.log_n;
        std::vector<cuntt::Complex32> input(config.batch * points, {0.0F, 0.0F});
        for (std::size_t batch = 0; batch < config.batch; ++batch) {
            input[batch * points] = {1.0F, 0.0F};
        }

        cuntt::ButterflyPlan plan(config);
        std::vector<cuntt::Complex32> output;
        const auto stats = plan.execute(input, output);

        const bool valid = !output.empty() && std::abs(output.front().real - 1.0F) < 1.0e-5F &&
                           std::abs(output.front().imag) < 1.0e-5F;
        std::cout << "FFT kernel: " << stats.kernel_ms << " ms\n";
        return valid ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "basic FFT example failed: " << error.what() << '\n';
        return 1;
    }
}
