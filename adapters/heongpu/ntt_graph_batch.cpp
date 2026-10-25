#include "ntt_graph_batch.hpp"

#include <algorithm>
#include <set>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cubutterfly_heongpu {
namespace {

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

struct GraphOwner {
    cudaGraph_t value = nullptr;
    ~GraphOwner() { if (value) cudaGraphDestroy(value); }
    GraphOwner() = default;
    GraphOwner(const GraphOwner&) = delete;
    GraphOwner& operator=(const GraphOwner&) = delete;
};

std::vector<cudaGraphNode_t> nodes(cudaGraph_t graph) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "get graph node count");
    std::vector<cudaGraphNode_t> result(count);
    if (count)
        check(cudaGraphGetNodes(graph, result.data(), &count), "get graph nodes");
    result.resize(count);
    return result;
}

std::size_t validate_and_count(cudaGraph_t graph) {
    std::size_t kernels = 0;
    for (auto node : nodes(graph)) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node, &type), "get graph node type");
        switch (type) {
        case cudaGraphNodeTypeKernel: ++kernels; break;
        case cudaGraphNodeTypeMemcpy:
        case cudaGraphNodeTypeMemset:
        case cudaGraphNodeTypeEmpty: break;
        case cudaGraphNodeTypeGraph: {
            cudaGraph_t child = nullptr;
            check(cudaGraphChildGraphNodeGetGraph(node, &child), "get child graph");
            kernels += validate_and_count(child);
            break;
        }
        default:
            throw std::invalid_argument("NTT graph batching does not support node type " +
                                        std::to_string(static_cast<int>(type)));
        }
    }
    return kernels;
}

cudaGraphNode_t clone_node(cudaGraph_t destination, cudaGraphNode_t source) {
    cudaGraphNode_t result = nullptr;
    cudaGraphNodeType type;
    check(cudaGraphNodeGetType(source, &type), "get source node type");
    switch (type) {
    case cudaGraphNodeTypeKernel: {
        cudaKernelNodeParams params{};
        check(cudaGraphKernelNodeGetParams(source, &params), "get kernel parameters");
        check(cudaGraphAddKernelNode(&result, destination, nullptr, 0, &params),
              "clone kernel node");
        check(cudaGraphKernelNodeCopyAttributes(source, result), "copy kernel attributes");
        break;
    }
    case cudaGraphNodeTypeMemcpy: {
        cudaMemcpy3DParms params{};
        check(cudaGraphMemcpyNodeGetParams(source, &params), "get memcpy parameters");
        check(cudaGraphAddMemcpyNode(&result, destination, nullptr, 0, &params),
              "clone memcpy node");
        break;
    }
    case cudaGraphNodeTypeMemset: {
        cudaMemsetParams params{};
        check(cudaGraphMemsetNodeGetParams(source, &params), "get memset parameters");
        check(cudaGraphAddMemsetNode(&result, destination, nullptr, 0, &params),
              "clone memset node");
        break;
    }
    case cudaGraphNodeTypeEmpty:
        check(cudaGraphAddEmptyNode(&result, destination, nullptr, 0), "clone empty node");
        break;
    case cudaGraphNodeTypeGraph: {
        cudaGraph_t child = nullptr;
        check(cudaGraphChildGraphNodeGetGraph(source, &child), "get source child graph");
        check(cudaGraphAddChildGraphNode(&result, destination, nullptr, 0, child),
              "clone child graph node");
        break;
    }
    default:
        throw std::invalid_argument("unsupported source graph node");
    }
    return result;
}

void clone_edges(cudaGraph_t source, cudaGraph_t destination,
                 const std::unordered_map<cudaGraphNode_t, cudaGraphNode_t>& mapping,
                 std::set<std::pair<cudaGraphNode_t, cudaGraphNode_t>>& seen) {
    std::size_t count = 0;
    check(cudaGraphGetEdges(source, nullptr, nullptr, &count), "get graph edge count");
    std::vector<cudaGraphNode_t> from(count), to(count);
    if (count)
        check(cudaGraphGetEdges(source, from.data(), to.data(), &count), "get graph edges");
    for (std::size_t i = 0; i < count; ++i) {
        auto a = mapping.at(from[i]);
        auto b = mapping.at(to[i]);
        if (seen.emplace(a, b).second)
            check(cudaGraphAddDependencies(destination, &a, &b, 1), "clone graph edge");
    }
}

bool same_signature(const cuntt::RnsPlanConfig& a, const cuntt::RnsPlanConfig& b) {
    if (a.log_n != b.log_n || a.inverse != b.inverse || a.ring != b.ring ||
        a.input_order != b.input_order || a.output_order != b.output_order ||
        a.domains.size() != b.domains.size() ||
        a.polynomials.size() != b.polynomials.size())
        return false;
    for (std::size_t i = 0; i < a.domains.size(); ++i)
        if (a.domains[i].modulus != b.domains[i].modulus ||
            a.domains[i].root != b.domains[i].root)
            return false;
    for (std::size_t i = 0; i < a.polynomials.size(); ++i) {
        const auto& x = a.polynomials[i];
        const auto& y = b.polynomials[i];
        if (x.domain != y.domain || x.input_offset != y.input_offset ||
            x.output_offset != y.output_offset)
            return false;
    }
    return true;
}

void capture_stage(RnsStageBatch& stage, cudaStream_t stream, GraphOwner& child) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin stage graph capture");
    try {
        stage.execute_async(stream);
    } catch (...) {
        // End even an invalidated capture so constructor failure leaves the
        // caller's stream usable. Any successfully produced graph is owned.
        cudaStreamEndCapture(stream, &child.value);
        throw;
    }
    check(cudaStreamEndCapture(stream, &child.value), "end stage graph capture");
}

} // namespace

struct BatchedNttGraph::Impl {
    GraphOwner graph;
    cudaGraphExec_t executable = nullptr;
    SharedDeviceWordArena arena;
    std::vector<std::unique_ptr<RnsStageBatch>> stages;
    std::size_t kernels = 0;
    std::size_t polynomials = 0;

    ~Impl() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph.value) cudaGraphDestroy(graph.value);
        graph.value = nullptr;
    }
};

BatchedNttGraph::BatchedNttGraph(const std::vector<MarkedNttGraph>& inputs,
                               StageBackend backend, cudaStream_t stream)
    : impl_(std::make_unique<Impl>()) {
    if (inputs.empty())
        throw std::invalid_argument("NTT graph batching needs at least one input graph");
    cudaStreamCaptureStatus capture_status;
    check(cudaStreamIsCapturing(stream, &capture_status), "get stream capture status");
    if (capture_status != cudaStreamCaptureStatusNone)
        throw std::invalid_argument("prepare batched NTT graphs outside stream capture");

    const auto stage_count = inputs.front().stages.size();
    std::vector<std::vector<cudaGraphNode_t>> source_nodes;
    source_nodes.reserve(inputs.size());
    for (std::size_t g = 0; g < inputs.size(); ++g) {
        const auto& input = inputs[g];
        if (!input.graph || input.stages.size() != stage_count)
            throw std::invalid_argument("NTT input graphs have null graphs or different stage counts");
        validate_and_count(input.graph);
        source_nodes.push_back(nodes(input.graph));
        std::unordered_set<cudaGraphNode_t> present(source_nodes.back().begin(),
                                                 source_nodes.back().end());
        std::unordered_set<cudaGraphNode_t> markers;
        for (std::size_t s = 0; s < stage_count; ++s) {
            const auto& stage = input.stages[s];
            if (!stage.marker || !present.count(stage.marker) ||
                !markers.insert(stage.marker).second)
                throw std::invalid_argument("NTT stage marker is missing, foreign, or duplicated");
            cudaGraphNodeType type;
            check(cudaGraphNodeGetType(stage.marker, &type), "get marker type");
            if (type != cudaGraphNodeTypeEmpty)
                throw std::invalid_argument("NTT stage markers must be empty nodes");
            if (!same_signature(inputs.front().stages[s].request.config,
                                stage.request.config))
                throw std::invalid_argument("NTT stage signature mismatch at graph " +
                                            std::to_string(g) + ", stage " + std::to_string(s));
        }
    }

    // Prepare one workspace for the largest stage. Stages execute in graph
    // order, so the same arena can be reused without changing captured
    // pointers or allowing concurrent writes.
    std::size_t max_words = 0;
    for (std::size_t s = 0; s < stage_count; ++s) {
        const auto& config = inputs.front().stages[s].request.config;
        if (config.log_n >= sizeof(std::size_t) * 8)
            throw std::invalid_argument("NTT stage size exceeds host address range");
        const std::size_t n = std::size_t{1} << config.log_n;
        std::size_t stage_words = 0;
        for (const auto& input : inputs) {
            const auto count = input.stages[s].request.config.polynomials.size();
            if (count > (std::numeric_limits<std::size_t>::max() - stage_words) / n)
                throw std::invalid_argument("NTT shared stage arena size overflow");
            stage_words += count * n;
        }
        max_words = std::max(max_words, stage_words);
    }
    impl_->arena = make_device_word_arena(max_words);

    // Prepare every stage before starting capture. All stages borrow the
    // shared arena while retaining independent plans and metadata.
    for (std::size_t s = 0; s < stage_count; ++s) {
        std::vector<BoundRnsRequest> requests;
        requests.reserve(inputs.size());
        for (const auto& input : inputs) requests.push_back(input.stages[s].request);
        auto stage = std::make_unique<RnsStageBatch>(requests, backend, stream, impl_->arena);
        impl_->polynomials += stage->polynomial_count();
        impl_->stages.push_back(std::move(stage));
    }

    check(cudaGraphCreate(&impl_->graph.value, 0), "create batched NTT graph");
    std::vector<cudaGraphNode_t> replacements(stage_count);
    for (std::size_t s = 0; s < stage_count; ++s) {
        GraphOwner child;
        capture_stage(*impl_->stages[s], stream, child);
        check(cudaGraphAddChildGraphNode(&replacements[s], impl_->graph.value,
                                        nullptr, 0, child.value),
              "add batched NTT stage");
    }

    std::set<std::pair<cudaGraphNode_t, cudaGraphNode_t>> edges;
    for (std::size_t g = 0; g < inputs.size(); ++g) {
        std::unordered_map<cudaGraphNode_t, cudaGraphNode_t> mapping;
        for (std::size_t s = 0; s < stage_count; ++s)
            mapping.emplace(inputs[g].stages[s].marker, replacements[s]);
        for (auto node : source_nodes[g])
            if (!mapping.count(node))
                mapping.emplace(node, clone_node(impl_->graph.value, node));
        clone_edges(inputs[g].graph, impl_->graph.value, mapping, edges);
    }
    impl_->kernels = validate_and_count(impl_->graph.value);
    check(cudaGraphInstantiateWithFlags(&impl_->executable, impl_->graph.value, 0),
          "instantiate batched NTT graph (stage order may introduce a cycle)");
}

BatchedNttGraph::~BatchedNttGraph() = default;

void BatchedNttGraph::launch(cudaStream_t stream) {
    check(cudaGraphLaunch(impl_->executable, stream), "launch batched NTT graph");
}

cudaGraph_t BatchedNttGraph::graph() const { return impl_->graph.value; }
std::size_t BatchedNttGraph::kernel_nodes() const { return impl_->kernels; }
std::size_t BatchedNttGraph::stage_count() const { return impl_->stages.size(); }
std::size_t BatchedNttGraph::polynomial_count() const { return impl_->polynomials; }

} // namespace cubutterfly_heongpu
