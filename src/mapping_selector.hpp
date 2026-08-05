#pragma once

#include "cuntt/butterfly.hpp"

namespace cuntt::detail {

struct ButterflySelectionResult {
    ButterflyConfig config;
    SelectionInfo   info;
};

struct NttSelectionResult {
    PlanConfig    config;
    SelectionInfo info;
};

ButterflySelectionResult select_butterfly_mapping(ButterflyConfig config);
NttSelectionResult       select_ntt_mapping(PlanConfig config);

}  // namespace cuntt::detail
