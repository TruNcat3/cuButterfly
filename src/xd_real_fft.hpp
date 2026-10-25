#pragma once

#include "xd_axis_resources.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace cuntt::detail {

// A compact multidimensional real FFT lowering, up to the public ND layout
// rank limit. The last axis is lowered
// with a rank-1 R2C/C2R plan and every preceding axis with a rank-1 C2C plan.
// packed_axes selects a permutation-boundary variant for non-contiguous axes,
// while false preserves the strided rank-1 lowering.  complex_only is used by
// the xD scheduler for a full-complex axis chain in either variant.
// fused_axes imports optional tiled device cores on supported axes, preserving
// the current tensor layout; unsupported axes retain the selected fallback.
// axis_exchange_policy=4 combines packed_axes with a contiguous cuFFTDx core
// and retains each axis-last layout until the final output restoration.
class XdRealFftPlan {
  public:
    XdRealFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                  bool inverse, bool normalize_inverse, bool fp64,
                  bool complex_only = false, double output_scale = 1.0,
                  bool packed_axes = false,
                  const std::vector<std::size_t>& axis_order = {},
                  bool fused_axes = false, unsigned axis_ept = 8,
                  bool paired_axes = false,
                  XdAxisCore axis_core = XdAxisCore::CufftdxBlock,
                  XdPairCore pair_core = XdPairCore::Thread,
                  unsigned register_lanes = 4, unsigned pair_layout = 0,
                  std::size_t batch_tile = 0, unsigned axis_ffts_per_block_cap = 0,
                  unsigned axis_exchange_policy = 0,
                  const std::vector<XdAxisResources>& axis_resources = {},
                  bool c2r_prefix_scale = false);
    ~XdRealFftPlan();

    XdRealFftPlan(const XdRealFftPlan&) = delete;
    XdRealFftPlan& operator=(const XdRealFftPlan&) = delete;
    XdRealFftPlan(XdRealFftPlan&&) noexcept;
    XdRealFftPlan& operator=(XdRealFftPlan&&) noexcept;

    std::size_t workspace_size() const noexcept;
    void set_workspace(void* workspace, std::size_t bytes);
    void execute(const void* input, void* output, cudaStream_t stream);
    std::string mapping_json() const;
    bool can_fuse_c2r_scale() const noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt::detail
