#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace cuntt::detail {

class DirectFftPlan {
  public:
    DirectFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                  bool inverse, bool normalize_inverse, bool fp64);
    ~DirectFftPlan();

    DirectFftPlan(const DirectFftPlan&) = delete;
    DirectFftPlan& operator=(const DirectFftPlan&) = delete;

    void execute(const void* input, void* output, cudaStream_t stream);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
