#include "xd_vkfft_axis.hpp"

#include <stdexcept>

namespace cuntt::detail {

class XdVkfftAxisPlan::Impl {};

bool xd_vkfft_axis_available() noexcept { return false; }
XdVkfftAxisPlan::XdVkfftAxisPlan(std::size_t, std::size_t, std::size_t, bool, bool) {
    throw std::invalid_argument("xD VkFFT axis provider is not enabled in this build");
}
XdVkfftAxisPlan::~XdVkfftAxisPlan() = default;
std::size_t XdVkfftAxisPlan::owned_workspace_size() const noexcept { return 0; }
void XdVkfftAxisPlan::execute(const void*, void*, cudaStream_t) {
    throw std::invalid_argument("xD VkFFT axis provider is not enabled in this build");
}
std::string XdVkfftAxisPlan::metadata_json() const { return "{\"provider\":\"unavailable\"}"; }

}  // namespace cuntt::detail
