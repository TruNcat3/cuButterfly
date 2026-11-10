#pragma once

#include "rns_stage_batch.hpp"
#include <cuda_runtime_api.h>
#include <vector>

namespace cubutterfly_heongpu {

struct CapturedNttStage {
    cudaGraphNode_t marker = nullptr;
    BoundRnsRequest request;
};

// Explicit lowering-only capture. During stream capture transform() emits an
// empty dependency node instead of executing NTT, recording its full semantics
// and buffer bindings. Such a graph is NOT executable until every marker has
// been replaced. Preparation calls outside capture still execute real NTTs.
// Only prepared NativeRns plans are supported; no allocations or metadata reads
// may occur while recording markers. The caller retains graph/buffer ownership.
void record_ntt_stages(std::vector<CapturedNttStage>* stages);

struct MarkedNttGraph {
    cudaGraph_t graph = nullptr;
    std::vector<CapturedNttStage> stages;
};

} // namespace cubutterfly_heongpu
