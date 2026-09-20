#include "fft_register_dispatch.hpp"
#include <stdexcept>
namespace cuntt::detail {
std::vector<ButterflyConfig> register_tile_mappings() { return {}; }
void validate_register_tile(const ButterflyConfig&) {
    throw std::invalid_argument("register-tile requires a build with cuFFTDx enabled for its grouped suffix");
}
void launch_register_tile(const ButterflyConfig&,const void*,void*,void*,cudaStream_t) {
    throw std::invalid_argument("register-tile requires a build with cuFFTDx enabled for its grouped suffix");
}
void launch_register_tile_group(const ButterflyConfig&,unsigned,const void*,void*,std::uint64_t,cudaStream_t) {
    throw std::invalid_argument("register-tile requires a build with cuFFTDx enabled for its grouped suffix");
}
}
