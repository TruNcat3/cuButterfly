#include "xd_axis_fft.hpp"
#include "xd_real_fft.hpp"

#include <cuda.h>

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct ValidationPlan {
    std::unique_ptr<cuntt::detail::XdRealFftPlan> plan;
    std::string mapping;
    std::string resources;
};
}

extern "C" {
const char* xd_grouped_error() { return last_error.c_str(); }

void* xd_grouped_create(const std::size_t* shape, unsigned rank, std::size_t batch,
                       int inverse, int normalize, int fp64, int complex_only,
                       double output_scale, unsigned ept, unsigned group,
                       unsigned exchange) {
    last_error.clear();
    try {
        using namespace cuntt::detail;
        if (!xd_axis_fft_configuration_available(shape[0], fp64 != 0, ept, 0,
                                                  group, exchange)) {
            last_error = "unsupported grouped axis descriptor or device resources";
            return nullptr;
        }
        std::vector<std::size_t> extents(shape, shape + rank);
        const unsigned axis_count = complex_only ? rank : rank - 1;
        std::vector<XdAxisResources> resources(axis_count);
        resources.at(0).ept = ept;
        resources.at(0).exchange = exchange;
        resources.at(0).ffts_per_block = group;
        auto result = std::make_unique<ValidationPlan>();
        result->plan = std::make_unique<XdRealFftPlan>(
            extents, batch, inverse != 0, normalize != 0, fp64 != 0,
            complex_only != 0, output_scale, true, std::vector<std::size_t>{},
            true, 8, false, XdAxisCore::CufftdxBlock, XdPairCore::Thread,
            4, 0, 0, 0, 0, resources, false);
        result->mapping = result->plan->mapping_json();
        return result.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

const char* xd_grouped_mapping(void* handle) {
    return static_cast<ValidationPlan*>(handle)->mapping.c_str();
}

std::size_t xd_grouped_workspace_size(void* handle) {
    return static_cast<ValidationPlan*>(handle)->plan->workspace_size();
}

int xd_grouped_set_workspace(void* handle, void* workspace, std::size_t bytes) {
    try {
        static_cast<ValidationPlan*>(handle)->plan->set_workspace(workspace, bytes);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

int xd_grouped_execute(void* handle, const void* input, void* output, void* stream) {
    try {
        static_cast<ValidationPlan*>(handle)->plan->execute(
            input, output, static_cast<cudaStream_t>(stream));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}

const char* xd_grouped_capture_resources(void* handle, const void* input,
                                       void* output, void* raw_stream) {
    auto* value = static_cast<ValidationPlan*>(handle);
    auto stream = static_cast<cudaStream_t>(raw_stream);
    CUgraph graph = nullptr;
    try {
        auto check = [](CUresult status, const char* step) {
            if (status == CUDA_SUCCESS) return;
            const char* name = nullptr;
            const char* message = nullptr;
            cuGetErrorName(status, &name);
            cuGetErrorString(status, &message);
            throw std::runtime_error(std::string(step) + ": " +
                (name ? name : "unknown driver error") + ": " + (message ? message : ""));
        };
        auto driver_stream = reinterpret_cast<CUstream>(stream);
        check(cuStreamBeginCapture(driver_stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
              "cuStreamBeginCapture");
        const int execute_status = xd_grouped_execute(handle, input, output, raw_stream);
        const auto capture_status = cuStreamEndCapture(driver_stream, &graph);
        if (execute_status) throw std::runtime_error(last_error);
        check(capture_status, "cuStreamEndCapture");
        std::size_t count = 0;
        check(cuGraphGetNodes(graph, nullptr, &count), "cuGraphGetNodes(count)");
        std::vector<CUgraphNode> nodes(count);
        check(cuGraphGetNodes(graph, nodes.data(), &count), "cuGraphGetNodes(nodes)");
        CUdevice device = 0;
        int regs_per_block = 0;
        int regs_per_sm = 0;
        check(cuCtxGetDevice(&device), "cuCtxGetDevice");
        check(cuDeviceGetAttribute(&regs_per_block, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                                   device), "cuDeviceGetAttribute(regs-per-block)");
        check(cuDeviceGetAttribute(&regs_per_sm, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                   device), "cuDeviceGetAttribute(regs-per-sm)");
        std::ostringstream json;
        json << "{\"probe_api\":\"cuda-driver\",\"regs_per_block_limit\":" << regs_per_block
             << ",\"regs_per_sm\":" << regs_per_sm
             << ",\"kernels\":[";
        unsigned kernels = 0;
        for (const auto node : nodes) {
            CUgraphNodeType type;
            check(cuGraphNodeGetType(node, &type), "cuGraphNodeGetType");
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
            CUDA_KERNEL_NODE_PARAMS parameters{};
            check(cuGraphKernelNodeGetParams(node, &parameters), "cuGraphKernelNodeGetParams");
            const unsigned threads = parameters.blockDimX * parameters.blockDimY * parameters.blockDimZ;
            int num_regs = 0;
            int shared_bytes = 0;
            int max_threads = 0;
            check(cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, parameters.func),
                  "cuFuncGetAttribute(num-regs)");
            check(cuFuncGetAttribute(&shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, parameters.func),
                  "cuFuncGetAttribute(static-shared)");
            check(cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, parameters.func),
                  "cuFuncGetAttribute(max-threads)");
            int active_blocks = 0;
            check(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, parameters.func, threads, parameters.sharedMemBytes),
                "cuOccupancyMaxActiveBlocksPerMultiprocessor");
            if (kernels++) json << ',';
            json << "{\"threads\":" << threads << ",\"num_regs\":" << num_regs
                 << ",\"nominal_block_regs\":" << threads * num_regs
                 << ",\"static_shared_bytes\":" << shared_bytes
                 << ",\"dynamic_shared_bytes\":" << parameters.sharedMemBytes
                 << ",\"max_threads_per_block\":" << max_threads
                 << ",\"active_blocks_per_sm\":" << active_blocks << '}';
        }
        json << "]}";
        check(cuGraphDestroy(graph), "cuGraphDestroy");
        graph = nullptr;
        value->resources = json.str();
        return value->resources.c_str();
    } catch (const std::exception& error) {
        if (graph) cuGraphDestroy(graph);
        last_error = error.what();
        return nullptr;
    }
}

void xd_grouped_destroy(void* handle) { delete static_cast<ValidationPlan*>(handle); }
}
