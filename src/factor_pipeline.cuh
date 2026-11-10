#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace cuntt::detail {
// Dependency-closed slices within one transform, unlike BatchPipeline's batch
// slices. Each edge has its own full-array scratch: slices on an edge are
// disjoint, but their address sets change at digit rotation. Reusing ping-pong
// storage across concurrent, nonadjacent edges would overwrite live values.
class FactorPipeline {
    std::vector<cudaStream_t> streams_;
    std::vector<std::vector<cudaEvent_t>> ready_;
    cudaEvent_t begin_=nullptr,done_=nullptr;
    unsigned slices_;
    static void check(cudaError_t s) {
        if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
    }
    void release() noexcept {
        for(auto s:streams_) if(s) cudaStreamSynchronize(s);
        for(auto& edge:ready_) for(auto e:edge) if(e) cudaEventDestroy(e);
        if(begin_) cudaEventDestroy(begin_);
        if(done_) cudaEventDestroy(done_);
        for(auto s:streams_) if(s) cudaStreamDestroy(s);
    }
public:
    FactorPipeline(unsigned groups,unsigned slices):slices_(slices) {
        if(groups<2 || slices<2) throw std::invalid_argument("factor pipeline needs multiple groups and slices");
        streams_.resize(groups); ready_.resize(groups-1,std::vector<cudaEvent_t>(slices));
        try {
            for(auto& s:streams_) check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            for(auto& edge:ready_) for(auto& e:edge) check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&begin_,cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&done_,cudaEventDisableTiming));
        } catch(...) { release(); throw; }
    }
    ~FactorPipeline() { release(); }
    FactorPipeline(const FactorPipeline&)=delete;
    FactorPipeline& operator=(const FactorPipeline&)=delete;
    template<class Launch> void enqueue(cudaStream_t caller,Launch launch) {
        check(cudaStreamWaitEvent(caller,done_,0));
        check(cudaEventRecord(begin_,caller));
        for(auto s:streams_) check(cudaStreamWaitEvent(s,begin_,0));
        for(unsigned slice=0;slice<slices_;++slice) {
            for(unsigned group=0;group+1<streams_.size();++group) {
                auto s=streams_[group];
                if(group) check(cudaStreamWaitEvent(s,ready_[group-1][slice],0));
                launch(group,slice,s);
                check(cudaEventRecord(ready_[group][slice],s));
            }
        }
        // The last producer stream is ordered, so its last event covers every
        // slice. The last local FFT mixes this digit and needs the complete fan-in.
        check(cudaStreamWaitEvent(streams_.back(),ready_.back().back(),0));
        launch(static_cast<unsigned>(streams_.size()-1),0,streams_.back());
        check(cudaEventRecord(done_,streams_.back()));
        check(cudaStreamWaitEvent(caller,done_,0));
    }
};
}
