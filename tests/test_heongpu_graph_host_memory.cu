// Build with graph_memory.cpp and the HEonGPU dependency includes/libraries;
// link --wrap=cudaMemcpyAsync and both MemoryPool wraps documented in its header.
#include "graph_memory.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using cubutterfly_heongpu::GraphMemory;
using Word = std::uint64_t;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function function, const char* message) {
    try { function(); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct Device {
    Word* data = nullptr;
    explicit Device(std::size_t words) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), words * sizeof(Word)));
    }
    ~Device() { if (data) cudaFree(data); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
struct Activation {
    explicit Activation(GraphMemory& arena) { GraphMemory::activate(&arena); }
    ~Activation() { GraphMemory::activate(nullptr); }
};

void assert_values(const Device& output, const std::vector<Word>& expected, cudaStream_t stream) {
    std::vector<Word> actual(expected.size());
    check(cudaMemcpyAsync(actual.data(), output.data, actual.size() * sizeof(Word),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    require(actual == expected, "device output differs from host metadata");
}

void test_retained_snapshot() {
    constexpr std::size_t words = 4096, bytes = words * sizeof(Word);
    const std::vector<Word> expected(words, 0x12345678abcdef01ULL);
    Stream stream;
    Device input(words), output(words);
    GraphMemory arena;
    Graph graph; // Graph must be destroyed before its source-owning arena.
    {
        Activation active(arena);
        arena.recording();
        {
            auto temporary = expected;
            check(cudaMemcpyAsync(input.data, temporary.data(), bytes, cudaMemcpyHostToDevice, stream.value));
            check(cudaMemcpyAsync(output.data, input.data, bytes, cudaMemcpyDeviceToDevice, stream.value));
            check(cudaStreamSynchronize(stream.value));
            std::fill(temporary.begin(), temporary.end(), 0);
        }
        require(arena.host_snapshots() == 1, "D2D must not create a host snapshot");
        assert_values(output, expected, stream.value);
        arena.replaying();
        {
            auto temporary = expected;
            check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
            check(cudaMemcpyAsync(input.data, temporary.data(), bytes, cudaMemcpyHostToDevice, stream.value));
            check(cudaMemcpyAsync(output.data, input.data, bytes, cudaMemcpyDeviceToDevice, stream.value));
            check(cudaStreamEndCapture(stream.value, &graph.graph));
            arena.verify_replay_complete();
            std::fill(temporary.begin(), temporary.end(), ~Word{0});
        }
        require(arena.host_snapshots() == 1, "capture allocated a new host snapshot");
    }
    std::size_t node_count = 0;
    check(cudaGraphGetNodes(graph.graph, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    check(cudaGraphGetNodes(graph.graph, nodes.data(), &node_count));
    unsigned h2d = 0, d2d = 0;
    for (auto node : nodes) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node, &type));
        if (type != cudaGraphNodeTypeMemcpy) continue;
        cudaMemcpy3DParms parameters{};
        check(cudaGraphMemcpyNodeGetParams(node, &parameters));
        if (parameters.kind == cudaMemcpyHostToDevice) {
            ++h2d;
            cudaPointerAttributes attributes{};
            check(cudaPointerGetAttributes(&attributes, parameters.srcPtr.ptr));
            require(attributes.type == cudaMemoryTypeHost, "captured H2D source is not pinned host memory");
        }
        if (parameters.kind == cudaMemcpyDeviceToDevice) ++d2d;
    }
    require(h2d == 1 && d2d == 1, "real H2D and D2D Graph nodes were not preserved");
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < 3; ++replay) {
        check(cudaMemsetAsync(input.data, 0, bytes, stream.value));
        check(cudaMemsetAsync(output.data, 0, bytes, stream.value));
        check(cudaGraphLaunch(graph.executable, stream.value));
        assert_values(output, expected, stream.value);
    }
}

void test_metadata_rejections() {
    Stream stream, other;
    Device output(4);
    GraphMemory arena;
    Activation active(arena);
    std::vector<Word> metadata{1, 2, 3, 4};
    const auto bytes = metadata.size() * sizeof(Word);
    arena.recording();
    check(cudaMemcpyAsync(output.data, metadata.data(), bytes, cudaMemcpyHostToDevice, stream.value));
    check(cudaStreamSynchronize(stream.value));
    arena.replaying();
    rejects([&] { arena.verify_replay_complete(); }, "missing H2D sequence was accepted");
    rejects([&] {
        check(cudaMemcpyAsync(output.data, metadata.data(), bytes - sizeof(Word),
                              cudaMemcpyHostToDevice, stream.value));
    }, "changed H2D size was accepted");
    rejects([&] {
        check(cudaMemcpyAsync(output.data, metadata.data(), bytes, cudaMemcpyHostToDevice, other.value));
    }, "changed H2D stream was accepted");
    metadata[2] = 999;
    Graph graph;
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    rejects([&] {
        check(cudaMemcpyAsync(output.data, metadata.data(), bytes, cudaMemcpyHostToDevice, stream.value));
    }, "changed captured metadata was silently frozen");
    check(cudaStreamEndCapture(stream.value, &graph.graph));
    metadata[2] = 3;
    check(cudaMemcpyAsync(output.data, metadata.data(), bytes, cudaMemcpyHostToDevice, stream.value));
    check(cudaStreamSynchronize(stream.value));
    arena.verify_replay_complete();
    rejects([&] {
        check(cudaMemcpyAsync(output.data, metadata.data(), bytes, cudaMemcpyHostToDevice, stream.value));
    }, "unrecorded H2D copy was accepted");
}

void test_recording_cannot_allocate_during_capture() {
    Stream stream;
    Device output(1);
    GraphMemory arena;
    Graph graph;
    Activation active(arena);
    arena.recording();
    Word source = 42;
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    rejects([&] {
        check(cudaMemcpyAsync(output.data, &source, sizeof(source), cudaMemcpyHostToDevice, stream.value));
    }, "recording allocated a pinned snapshot during capture");
    check(cudaStreamEndCapture(stream.value, &graph.graph));
    require(arena.host_snapshots() == 0, "rejected capture created a snapshot");
}

void test_no_arena_passthrough() {
    Stream stream;
    Device input(4), output(4);
    for (Word value : {Word{7}, Word{19}, Word{7}}) {
        const std::vector<Word> host(4, value);
        check(cudaMemcpyAsync(input.data, host.data(), host.size() * sizeof(Word),
                              cudaMemcpyHostToDevice, stream.value));
        check(cudaMemcpyAsync(output.data, input.data, host.size() * sizeof(Word),
                              cudaMemcpyDeviceToDevice, stream.value));
        assert_values(output, host, stream.value);
    }
}
} // namespace

int main() {
    try {
        test_no_arena_passthrough();
        test_retained_snapshot();
        test_metadata_rejections();
        test_recording_cannot_allocate_during_capture();
        std::cout << "PASS GraphMemory pinned H2D lifetime and metadata validation\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
