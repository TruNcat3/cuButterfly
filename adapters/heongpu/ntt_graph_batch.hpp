#pragma once

#include "ntt_capture.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace cubutterfly_heongpu {

// Input graphs and their buffer allocations must outlive this object. Each
// input graph must use independent scratch storage. Launches must not overlap
// because each stage owns reusable packing and transform workspace.
class BatchedNttGraph {
public:
    BatchedNttGraph(const std::vector<MarkedNttGraph>& graphs,
                    StageBackend backend, cudaStream_t stream);
    ~BatchedNttGraph();
    BatchedNttGraph(const BatchedNttGraph&) = delete;
    BatchedNttGraph& operator=(const BatchedNttGraph&) = delete;

    void launch(cudaStream_t stream);
    cudaGraph_t graph() const;
    std::size_t kernel_nodes() const;
    std::size_t stage_count() const;
    std::size_t polynomial_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace cubutterfly_heongpu
