#pragma once
#include <cuda_runtime.h>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace cuntt::detail {
// Plan-owned scheduler with one stream per physical group and two slots per
// boundary. Events are reused in submission order:
// cudaStreamWaitEvent captures the most recent record at the time of the call.
// A slot is never overwritten until its consumer (including transpose) finishes.
class BatchPipeline {
    std::vector<cudaStream_t> streams_;
    cudaEvent_t begin_ = nullptr, done_ = nullptr;
    std::vector<std::array<cudaEvent_t, 2>> ready_, consumed_;
    static void check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void release() noexcept {
        for (auto stream : streams_) if (stream) cudaStreamSynchronize(stream);
        for (auto& edge : ready_) for (auto event : edge) if (event) cudaEventDestroy(event);
        for (auto& edge : consumed_) for (auto event : edge) if (event) cudaEventDestroy(event);
        if (begin_) cudaEventDestroy(begin_);
        if (done_) cudaEventDestroy(done_);
        for (auto stream : streams_) if (stream) cudaStreamDestroy(stream);
    }
public:
    explicit BatchPipeline(std::size_t groups = 2) {
        if (groups < 2) throw std::invalid_argument("a batch pipeline needs at least two physical groups");
        streams_.resize(groups);
        ready_.resize(groups - 1); consumed_.resize(groups - 1);
        try {
            for (auto& stream : streams_) check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            check(cudaEventCreateWithFlags(&begin_, cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
            for (auto& edge : ready_) for (auto& event : edge) check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            for (auto& edge : consumed_) for (auto& event : edge) check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        } catch (...) { release(); throw; }
    }
    ~BatchPipeline() { release(); }
    BatchPipeline(const BatchPipeline&) = delete;
    BatchPipeline& operator=(const BatchPipeline&) = delete;

    template<class Launch>
    void enqueue(std::size_t batch, std::size_t tile_batch, cudaStream_t caller,
                 Launch launch) {
        if (!batch || !tile_batch) throw std::invalid_argument("pipeline batch and tile must be positive");
        // Fence input uploads and earlier invocations, even on a new caller stream.
        check(cudaStreamWaitEvent(caller, done_, 0));
        check(cudaEventRecord(begin_, caller));
        for (auto stream : streams_) check(cudaStreamWaitEvent(stream, begin_, 0));
        std::size_t tile = 0;
        for (std::size_t offset = 0; offset < batch; offset += tile_batch, ++tile) {
            const auto slot = tile % 2;
            const auto count = std::min(tile_batch, batch - offset);
            for (std::size_t group = 0; group < streams_.size(); ++group) {
                auto stream = streams_[group];
                // Reuse an output slot only after its immediate consumer has
                // finished reading it, independently for every boundary.
                if (group + 1 < streams_.size() && tile >= 2)
                    check(cudaStreamWaitEvent(stream, consumed_[group][slot], 0));
                if (group) check(cudaStreamWaitEvent(stream, ready_[group - 1][slot], 0));
                launch(group, offset, count, slot, stream);
                check(cudaGetLastError());
                if (group) check(cudaEventRecord(consumed_[group - 1][slot], stream));
                if (group + 1 < streams_.size()) check(cudaEventRecord(ready_[group][slot], stream));
            }
        }
        check(cudaEventRecord(done_, streams_.back()));
        // Keep the public async API and CUDA-event timing on the caller stream.
        check(cudaStreamWaitEvent(caller, done_, 0));
    }
    template<class First, class Second>
    void enqueue(std::size_t batch, std::size_t tile_batch, cudaStream_t caller,
                 First first, Second second) {
        if (streams_.size() != 2) throw std::logic_error("two-callback pipeline needs two groups");
        enqueue(batch, tile_batch, caller, [&](auto group, auto offset, auto count, auto slot, auto stream) {
            if (group == 0) first(offset, count, slot, stream);
            else second(offset, count, slot, stream);
        });
    }
};
} // namespace cuntt::detail
