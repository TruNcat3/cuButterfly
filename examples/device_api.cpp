#include <cuda_runtime_api.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main() {
    try {
        cuntt::ButterflyConfig config;
        config.op                          = cuntt::ButterflyOperator::Fwht;
        config.precision                   = cuntt::ButterflyPrecision::Fp32;
        config.placement                   = cuntt::ButterflyPlacement::OutOfPlace;
        config.log_n                       = 15;
        config.batch                       = 16;
        config.auto_select                 = true;
        config.auto_allocate_workspace     = false;

        cuntt::ButterflyPlan plan(config);
        std::cout << "selected " << plan.selection().implementation << " (" << plan.selection().confidence << ")\n";
        cudaStream_t         stream = nullptr;
        void*                input = nullptr;
        void*                output = nullptr;
        void*                workspace = nullptr;

        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        check_cuda(cudaMalloc(&input, plan.data_size()), "allocate input");
        check_cuda(cudaMalloc(&output, plan.data_size()), "allocate output");
        if (plan.workspace_size() != 0) {
            check_cuda(cudaMalloc(&workspace, plan.workspace_size()), "allocate workspace");
            plan.set_workspace(workspace, plan.workspace_size());
        }
        plan.set_stream(stream);

        std::vector<float> host_input(plan.data_size() / sizeof(float), 1.0F);
        std::vector<float> host_output(host_input.size());
        check_cuda(cudaMemcpyAsync(input, host_input.data(), plan.data_size(), cudaMemcpyHostToDevice, stream), "copy input");
        plan.execute_async(static_cast<const float*>(input), static_cast<float*>(output));
        check_cuda(cudaMemcpyAsync(host_output.data(), output, plan.data_size(), cudaMemcpyDeviceToHost, stream), "copy output");
        check_cuda(cudaStreamSynchronize(stream), "wait for transform");

        std::cout << "FWHT[0] = " << host_output.front() << '\n';
        cudaFree(workspace);
        cudaFree(output);
        cudaFree(input);
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::cerr << "device API example failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
