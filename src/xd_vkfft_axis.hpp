#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>

namespace cuntt::detail {

bool xd_vkfft_axis_available() noexcept;

// An unnormalized C2C transform over [lines / suffix, length, suffix].
// VkFFT owns its scratch and lookup tables; no caller workspace is required.
class XdVkfftAxisPlan {
  public:
    XdVkfftAxisPlan(std::size_t length, std::size_t suffix, std::size_t lines,
                    bool fp64, bool inverse);
    ~XdVkfftAxisPlan();
    XdVkfftAxisPlan(const XdVkfftAxisPlan&) = delete;
    XdVkfftAxisPlan& operator=(const XdVkfftAxisPlan&) = delete;

    std::size_t owned_workspace_size() const noexcept;
    void execute(const void* input, void* output, cudaStream_t stream);
    std::string metadata_json() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
