#pragma once
#include "cuntt/butterfly.hpp"
namespace cuntt::detail {
// Validate the compiled unit product and active-device resource limits. This
// family uses prebuilt native/radix4 register prefix codelets (or an explicit
// research JIT codelet) and a grouped cuFFTDx suffix.
void validate_register_tile(const ButterflyConfig& config);
std::vector<ButterflyConfig> register_tile_mappings();
void launch_register_tile(const ButterflyConfig& config, const void* input,
                          void* output, void* scratch, cudaStream_t stream);
void launch_register_tile_group(const ButterflyConfig& config, unsigned group,
                                const void* input, void* output, std::uint64_t batch, cudaStream_t stream);
}
