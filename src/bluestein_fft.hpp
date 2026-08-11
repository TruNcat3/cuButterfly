#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuntt::detail {

class BluesteinFftPlan {
  public:
    BluesteinFftPlan(std::size_t length, std::size_t convolution_length,
                     std::size_t batch, bool inverse, bool normalize_inverse,
                     bool fp64);
    ~BluesteinFftPlan();

    BluesteinFftPlan(const BluesteinFftPlan&) = delete;
    BluesteinFftPlan& operator=(const BluesteinFftPlan&) = delete;

    std::size_t workspace_size() const noexcept;
    void execute(const void* input, void* output, void* workspace,
                 std::size_t input_stride, std::size_t input_batch_stride,
                 std::size_t output_stride, std::size_t output_batch_stride,
                 cudaStream_t stream);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
