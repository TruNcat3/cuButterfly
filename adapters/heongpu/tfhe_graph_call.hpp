#pragma once
#include "graph_memory.hpp"
#include <heongpu/heongpu.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cubutterfly::heongpu_tfhe {
using Cipher = heongpu::Ciphertext<heongpu::Scheme::TFHE>;
using Outputs = std::vector<std::unique_ptr<Cipher>>;
inline void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};

struct GraphCall {
    cubutterfly_heongpu::GraphMemory memory;
    Outputs outputs;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t stream;
    std::size_t nodes = 0, bootstrap_nodes = 0;
    explicit GraphCall(cudaStream_t value) : stream(value) {}
    ~GraphCall() {
        cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        outputs.clear();
    }
    template<class Run, class Configure>
    void prepare(Run&& run, Configure&& configure) {
        memory.recording();
        cubutterfly_heongpu::GraphMemory::activate(&memory);
        bool capturing = false;
        try {
            run(outputs);
            check(cudaStreamSynchronize(stream));
            outputs.clear();
            memory.replaying();
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            capturing = true;
            run(outputs);
            check(cudaStreamEndCapture(stream, &graph));
            capturing = false;
            memory.verify_replay_complete();
            cubutterfly_heongpu::GraphMemory::activate(nullptr);
            bootstrap_nodes = configure(graph);
            check(cudaGraphGetNodes(graph, nullptr, &nodes));
            check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        } catch (...) {
            cubutterfly_heongpu::GraphMemory::activate(nullptr);
            if (capturing) cudaStreamEndCapture(stream, &graph);
            throw;
        }
    }
    void launch() { check(cudaGraphLaunch(executable, stream)); }
    double measure(unsigned replays) {
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
        check(cudaEventRecord(start, stream));
        for (unsigned i = 0; i < replays; ++i) launch();
        check(cudaEventRecord(stop, stream)); check(cudaEventSynchronize(stop));
        float milliseconds;
        check(cudaEventElapsedTime(&milliseconds, start, stop));
        check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
        return milliseconds / replays;
    }
};
}  // namespace cubutterfly::heongpu_tfhe
