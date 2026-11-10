#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    void* input = nullptr;
    void* output = nullptr;
    void* workspace = nullptr;

    try {
        cuntt::PlanConfig config;
        config.log_n                   = 16;
        config.batch                   = 2;
        config.backend                 = cuntt::Backend::Tile256;
        config.word_bits               = 64;
        config.auto_allocate_workspace = false;

        cuntt::Plan plan(config);
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        check_cuda(cudaMalloc(&input, plan.data_size()), "allocate input");
        check_cuda(cudaMalloc(&output, plan.data_size()), "allocate output");
        if (plan.workspace_size() != 0) {
            check_cuda(cudaMalloc(&workspace, plan.workspace_size()), "allocate workspace");
            plan.set_workspace(workspace, plan.workspace_size());
        }
        plan.set_stream(stream);

        std::vector<std::uint64_t> host_input(plan.data_size() / sizeof(std::uint64_t), 0);
        std::vector<std::uint64_t> host_output(host_input.size());
        host_input.front() = 1;
        check_cuda(cudaMemcpyAsync(input, host_input.data(), plan.data_size(), cudaMemcpyHostToDevice, stream), "copy input");
        plan.execute_async(static_cast<const std::uint64_t*>(input), static_cast<std::uint64_t*>(output));
        check_cuda(cudaMemcpyAsync(host_output.data(), output, plan.data_size(), cudaMemcpyDeviceToHost, stream), "copy output");
        check_cuda(cudaStreamSynchronize(stream), "wait for transform");

        std::cout << "NTT[0] = " << host_output.front() << '\n';
        cudaFree(workspace);
        cudaFree(output);
        cudaFree(input);
        cudaStreamDestroy(stream);
        return 0;
    } catch (const std::exception& error) {
        cudaFree(workspace);
        cudaFree(output);
        cudaFree(input);
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        std::cerr << "NTT device API example failed: " << error.what() << '\n';
        return 1;
    }
}
