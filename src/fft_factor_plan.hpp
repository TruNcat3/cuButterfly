#pragma once
#include <cuntt/butterfly.hpp>
namespace cuntt::detail {
void normalize_factor_mapping(ButterflyConfig& config);
MixedDataflowPlan make_factor_dataflow_plan(const ButterflyConfig& config);
struct FactorTileRange {
    std::uint64_t first, count, period, span;
};
// Ranges use full-array addressing. Prefix factors keep the low unprocessed
// digit fixed; the final factor joins every slice of that digit.
FactorTileRange factor_tile_range(const ButterflyConfig& config,unsigned group,unsigned slice);
}
