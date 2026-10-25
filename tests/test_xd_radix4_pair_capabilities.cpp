#include "../src/xd_stockham_pair_fft.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
}  // namespace

int main() {
    using namespace cuntt::detail;
    for (const bool fp64 : {false, true}) {
        require(xd_stockham_pair_fft_available(32, 32, fp64),
                "register pair must support FP32 and FP64");
        require(!xd_stockham_pair_fft_available(16, 32, fp64) &&
                    !xd_stockham_pair_fft_available(32, 64, fp64),
                "register pair must reject uncompiled shapes");
        require(!xd_stockham_pair_fft_available(
                    32, 32, fp64, static_cast<XdStagePairVariant>(-1)),
                "pair must reject unknown variants");
    }
    for (const auto variant : {XdStagePairVariant::Radix2,
                               XdStagePairVariant::Radix4Stage,
                               XdStagePairVariant::Radix4RegisterSplit}) {
        require(xd_stockham_pair_fft_available(32, 32, false, variant),
                "existing FP32 diagnostic must remain available");
        require(!xd_stockham_pair_fft_available(32, 32, true, variant),
                "unimplemented FP64 diagnostic must remain unavailable");
        // Unsupported precision is rejected before CUDA submission, including
        // when otherwise valid host-side argument values are supplied.
        double input[2048] = {}, output[2048] = {};
        bool rejected = false;
        try {
            launch_xd_stockham_pair_fft(input, output, 32, 32, 1, true,
                                        false, 1.0, nullptr, variant);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, "FP64 diagnostic must throw before submission");
    }
    std::cout << "radix4 pair capabilities and FP64 diagnostic rejection passed\n";
}
