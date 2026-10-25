#pragma once

#include "general_kernels.hpp"

namespace cuntt::detail {

// Packed scalar components; complex arrays have two components per element.
// These boundaries implement explicitly requested FP64 compute with FP32 storage.
void launch_fp32_to_fp64(const void* input, void* output,
                         std::size_t component_count, cudaStream_t stream);
void launch_fp64_to_fp32(const void* input, void* output,
                         std::size_t component_count, cudaStream_t stream);

// All buffers are packed natural order, with contiguous batches. full_layout
// supplies rank, full real-transform extents, points and batch; its strides are
// unused. The compact spectrum has floor(last_extent/2)+1 elements on its last
// axis. Storage and arithmetic remain FP16 (bf16=false) or BF16 (bf16=true).
// Caller gathers/scatters user layouts and invokes the matching complex core.
void launch_low_real_to_complex(const void* input, void* output,
                                const GeneralNdLayout& full_layout,
                                bool bf16, cudaStream_t stream);
void launch_low_complex_to_real(const void* input, void* output,
                                const GeneralNdLayout& full_layout,
                                bool bf16, cudaStream_t stream);
void launch_low_extract_half_spectrum(const void* full, void* compact,
                                      const GeneralNdLayout& full_layout,
                                      bool bf16, cudaStream_t stream);
// C2R input must represent a valid Hermitian spectrum. Missing last-axis
// frequencies are conjugates at negated coordinates on every axis.
void launch_low_expand_half_spectrum(const void* compact, void* full,
                                     const GeneralNdLayout& full_layout,
                                     bool bf16, cudaStream_t stream);
// Apply after the complex core. Coefficient rounds to the requested low
// storage, followed by low arithmetic. Explicit FP32 compute instead multiplies
// in FP32 before rounding to low storage. Exact scale=1 leaves all bits intact.
void launch_low_output_scale(void* values, std::size_t component_count,
                             double scale, bool bf16, cudaStream_t stream,
                             bool fp32_compute = false);

}  // namespace cuntt::detail
