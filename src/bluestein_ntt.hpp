#pragma once

#include "cuntt/ntt.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cuntt::detail {

class BluesteinNttPlan {
  public:
    BluesteinNttPlan(std::size_t length, std::size_t convolution_length,
                     std::size_t batch, std::uint64_t modulus,
                     bool inverse, cuntt::Backend core_backend,
                     cuntt::ComputeUnit core_compute_unit, const cuntt::PlanConfig* mapping = nullptr,
                     const std::string& replay = {});
    ~BluesteinNttPlan();

    BluesteinNttPlan(const BluesteinNttPlan&) = delete;
    BluesteinNttPlan& operator=(const BluesteinNttPlan&) = delete;

    std::size_t workspace_size() const noexcept;
    std::string mapping_json() const;
    void execute(const std::uint64_t* input, std::uint64_t* output,
                 void* workspace, std::size_t input_stride,
                 std::size_t input_batch_stride, std::size_t output_stride,
                 std::size_t output_batch_stride, cudaStream_t stream);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
