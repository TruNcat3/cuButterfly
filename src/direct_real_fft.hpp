#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cuntt::detail {

// A complete packed R2C/C2R transform backed by one cuFFT plan.  This is a
// search candidate for public xD plans, not a separate execution mode: the
// selector compares it with the hierarchical axis candidates using the same
// full-plan timing protocol.
class DirectRealFftPlan {
  public:
    DirectRealFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                      bool inverse, bool normalize_inverse, bool fp64,
                      double output_scale);
    ~DirectRealFftPlan();

    DirectRealFftPlan(const DirectRealFftPlan&) = delete;
    DirectRealFftPlan& operator=(const DirectRealFftPlan&) = delete;

    void execute(const void* input, void* output, cudaStream_t stream);

    std::size_t workspace_size() const noexcept;
    void set_workspace(void* workspace, std::size_t bytes);
    std::string mapping_json() const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
