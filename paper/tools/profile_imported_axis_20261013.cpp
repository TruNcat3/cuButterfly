#include "xd_axis_fft.hpp"
#include "xd_vkfft_axis.hpp"

#include <cuda.h>

#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct CorePlan {
    std::unique_ptr<cuntt::detail::XdVkfftAxisPlan> vkfft;
    bool inverse = false;
    unsigned ept = 0;
    unsigned group = 0;
    unsigned exchange = 0;
    std::string resources;
    std::string metadata;
};
void driver_check(CUresult status, const char* step) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string(step) + ": " + (name ? name : "unknown") +
                             ": " + (message ? message : ""));
}
}

extern "C" {
const char* imported_axis_error() { return last_error.c_str(); }
void* imported_axis_create(int vkfft, int inverse) {
    try {
        using namespace cuntt::detail;
        auto plan = std::make_unique<CorePlan>();
        plan->inverse = inverse != 0;
        plan->ept = inverse ? 8 : 16;
        plan->group = inverse ? 4 : 2;
        plan->exchange = inverse ? 0 : 2;
        if (vkfft) {
            plan->vkfft = std::make_unique<XdVkfftAxisPlan>(2048, 1025, 16400, false, inverse != 0);
            plan->metadata = plan->vkfft->metadata_json();
        } else {
            if (!xd_axis_fft_configuration_available(2048, false, plan->ept, 0,
                                                      plan->group, plan->exchange))
                throw std::runtime_error("specified cuFFTDx axis resource is unavailable");
            std::ostringstream json;
            json << "{\"provider\":\"cufftdx-axis\",\"length\":2048,\"suffix\":1025,"
                 << "\"lines\":16400,\"ept\":" << plan->ept << ",\"group\":" << plan->group
                 << ",\"exchange\":" << plan->exchange << '}';
            plan->metadata = json.str();
        }
        return plan.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}
const char* imported_axis_metadata(void* opaque) {
    return static_cast<CorePlan*>(opaque)->metadata.c_str();
}
int imported_axis_execute(void* opaque, const void* input, void* output, void* raw_stream) {
    try {
        auto* plan = static_cast<CorePlan*>(opaque);
        auto stream = static_cast<cudaStream_t>(raw_stream);
        if (plan->vkfft)
            plan->vkfft->execute(input, output, stream);
        else
            cuntt::detail::launch_xd_axis_fft(input, output, 2048, 1025, 16400, false,
                plan->inverse, 1.0, stream, plan->ept, 0, plan->exchange, plan->group);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
const char* imported_axis_capture_resources(void* opaque, const void* input,
                                           void* output, void* raw_stream) {
    auto* plan = static_cast<CorePlan*>(opaque);
    CUgraph graph = nullptr;
    try {
        auto stream = reinterpret_cast<CUstream>(raw_stream);
        driver_check(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL), "begin capture");
        const auto execute_status = imported_axis_execute(opaque, input, output, raw_stream);
        const auto capture_status = cuStreamEndCapture(stream, &graph);
        if (execute_status) throw std::runtime_error(last_error);
        driver_check(capture_status, "end capture");
        std::size_t count = 0;
        driver_check(cuGraphGetNodes(graph, nullptr, &count), "get node count");
        std::vector<CUgraphNode> nodes(count);
        driver_check(cuGraphGetNodes(graph, nodes.data(), &count), "get nodes");
        CUdevice device = 0;
        int regs_per_block = 0;
        int regs_per_sm = 0;
        driver_check(cuCtxGetDevice(&device), "get context device");
        driver_check(cuDeviceGetAttribute(&regs_per_block, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
                                         device), "get register block limit");
        driver_check(cuDeviceGetAttribute(&regs_per_sm, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                         device), "get register SM limit");
        std::ostringstream json;
        json << "{\"probe_api\":\"cuda-driver\",\"graph_nodes\":" << count
             << ",\"regs_per_block_limit\":" << regs_per_block
             << ",\"regs_per_sm\":" << regs_per_sm << ",\"kernels\":[";
        unsigned kernels = 0;
        for (const auto node : nodes) {
            CUgraphNodeType type;
            driver_check(cuGraphNodeGetType(node, &type), "get node type");
            if (type != CU_GRAPH_NODE_TYPE_KERNEL) continue;
            CUDA_KERNEL_NODE_PARAMS parameters{};
            driver_check(cuGraphKernelNodeGetParams(node, &parameters), "get kernel node params");
            const unsigned threads = parameters.blockDimX * parameters.blockDimY * parameters.blockDimZ;
            int regs = 0, shared = 0, max_threads = 0, active_blocks = 0;
            driver_check(cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, parameters.func), "get registers");
            driver_check(cuFuncGetAttribute(&shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, parameters.func), "get shared bytes");
            driver_check(cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                           parameters.func), "get max threads");
            driver_check(cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks, parameters.func, threads, parameters.sharedMemBytes), "get active blocks");
            if (kernels++) json << ',';
            json << "{\"threads\":" << threads << ",\"num_regs\":" << regs
                 << ",\"nominal_block_regs\":" << threads * regs
                 << ",\"static_shared_bytes\":" << shared
                 << ",\"dynamic_shared_bytes\":" << parameters.sharedMemBytes
                 << ",\"max_threads_per_block\":" << max_threads
                 << ",\"active_blocks_per_sm\":" << active_blocks
                 << ",\"grid\":[" << parameters.gridDimX << ',' << parameters.gridDimY
                 << ',' << parameters.gridDimZ << "],\"block\":["
                 << parameters.blockDimX << ',' << parameters.blockDimY << ','
                 << parameters.blockDimZ << "]}";
        }
        json << "]}";
        driver_check(cuGraphDestroy(graph), "destroy graph");
        graph = nullptr;
        plan->resources = json.str();
        return plan->resources.c_str();
    } catch (const std::exception& error) {
        if (graph) cuGraphDestroy(graph);
        last_error = error.what();
        return nullptr;
    }
}
void imported_axis_destroy(void* opaque) { delete static_cast<CorePlan*>(opaque); }
}
