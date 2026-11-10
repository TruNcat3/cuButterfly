#pragma once

#include <cuntt/rns_ntt.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cubutterfly_heongpu {

struct BoundRnsRequest {
    cuntt::RnsPlanConfig config;
    const std::uint64_t* input = nullptr;
    std::uint64_t* output = nullptr;
};

enum class StageBackend { Native, CuButterfly };

// Device workspace shared by stages that execute serially in one graph. The
// arena is sized for the largest stage; each stage borrows it for its own
// gather/transform/scatter sequence.
struct DeviceWordArena;
using SharedDeviceWordArena = std::shared_ptr<DeviceWordArena>;

SharedDeviceWordArena make_device_word_arena(std::size_t words);

// A bounded stage of independent, canonical word64 negacyclic transforms.
// Requests share log_n/direction/orders; offsets and domain sequences are free.
// Construction prepares/tunes outside capture. Execution allocates no device
// memory; pinned native dispatch may create temporary host launch descriptors.
// Keep this object and bound allocations alive until executions/graphs finish;
// execute on only one stream at a time. Inputs must be canonical residues.
class RnsStageBatch {
  public:
    RnsStageBatch(const std::vector<BoundRnsRequest>& requests,
                  StageBackend backend, cudaStream_t stream);
    RnsStageBatch(const std::vector<BoundRnsRequest>& requests,
                  StageBackend backend, cudaStream_t stream,
                  SharedDeviceWordArena arena);
    ~RnsStageBatch();
    RnsStageBatch(const RnsStageBatch&) = delete;
    RnsStageBatch& operator=(const RnsStageBatch&) = delete;
    void execute_async(cudaStream_t stream);
    std::size_t polynomial_count() const noexcept;
    // Bytes moved by the stage boundary kernels per execution. This is two
    // word64 transfers per packed coefficient: external requests -> arena and
    // arena -> external requests. It is exposed so stage cost models do not
    // mistake the provider-only kernel probe for full stage service.
    std::size_t boundary_copy_bytes() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cubutterfly_heongpu
