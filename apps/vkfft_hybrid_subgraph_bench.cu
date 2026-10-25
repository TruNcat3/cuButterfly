#include "xd_axis_fft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int vkfft_xd_create(void** plan, std::uint32_t rank,
                    const std::uint64_t* shape, std::uint64_t batch,
                    int fp64, int inverse, std::uintptr_t stream,
                    std::uintptr_t input, std::uintptr_t output) noexcept;
int vkfft_xd_execute(void* plan) noexcept;
int vkfft_xd_destroy(void* plan) noexcept;
const char* vkfft_xd_last_error() noexcept;
}

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_fft(cufftResult status) {
    if (status != CUFFT_SUCCESS) throw std::runtime_error("cuFFT failure");
}
void check_vk(int status, const char* operation) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 (vkfft_xd_last_error() ? vkfft_xd_last_error() : "VkFFT error"));
    }
}
struct Buf {
    cufftComplex* ptr{};
    explicit Buf(std::size_t count) { check(cudaMalloc(&ptr, count * sizeof(*ptr))); }
    ~Buf() { cudaFree(ptr); }
};
float replay(cudaGraphExec_t instance, cudaStream_t stream,
             cudaEvent_t start, cudaEvent_t stop, unsigned warmup,
             unsigned repeat) {
    for (unsigned i = 0; i < warmup; ++i) check(cudaGraphLaunch(instance, stream));
    check(cudaStreamSynchronize(stream));
    check(cudaEventRecord(start, stream));
    for (unsigned i = 0; i < repeat; ++i) check(cudaGraphLaunch(instance, stream));
    check(cudaEventRecord(stop, stream));
    check(cudaEventSynchronize(stop));
    float total = 0.0f;
    check(cudaEventElapsedTime(&total, start, stop));
    return total / static_cast<float>(repeat);
}
}

int main(int argc, char** argv) {
    try {
        const std::size_t planes = argc > 1 ? std::stoull(argv[1]) : 64;
        const unsigned warmup = argc > 2 ? std::stoul(argv[2]) : 20;
        const unsigned repeat = argc > 3 ? std::stoul(argv[3]) : 100;
        constexpr std::size_t axis = 32;
        constexpr std::size_t points = axis * axis;
        if (planes == 0 || repeat == 0)
            throw std::invalid_argument("usage: vkfft_hybrid_subgraph_bench [planes] [warmup] [repeat]");
        if (!cuntt::detail::xd_axis_fft_configuration_available(
                axis, false, 8, 0, 0, 0))
            throw std::runtime_error("cuButterfly axis provider is unavailable for the hybrid route");

        const std::size_t count = points * planes;
        const std::size_t lines = planes * axis;
        Buf input(count), row_output(count), hybrid_output(count), reference(count);
        std::vector<cufftComplex> host(count);
        std::mt19937 rng(7721);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& value : host) value = {dist(rng), dist(rng)};
        check(cudaMemcpy(input.ptr, host.data(), count * sizeof(host[0]),
                         cudaMemcpyHostToDevice));

        cufftHandle cufft_plan{};
        int dims[2] = {static_cast<int>(axis), static_cast<int>(axis)};
        check_fft(cufftPlanMany(&cufft_plan, 2, dims, nullptr, 1,
                                static_cast<int>(points), nullptr, 1,
                                static_cast<int>(points), CUFFT_C2C,
                                static_cast<int>(planes)));
        check_fft(cufftExecC2C(cufft_plan, input.ptr, reference.ptr, CUFFT_FORWARD));
        check(cudaDeviceSynchronize());

        cudaStream_t stream{};
        check(cudaStreamCreate(&stream));
        void* vk_plan = nullptr;
        const std::uint64_t shape[1] = {axis};
        check_vk(vkfft_xd_create(&vk_plan, 1, shape, lines, 0, 0,
                                 reinterpret_cast<std::uintptr_t>(stream),
                                 reinterpret_cast<std::uintptr_t>(input.ptr),
                                 reinterpret_cast<std::uintptr_t>(row_output.ptr)),
                  "vkfft_xd_create");

        check_vk(vkfft_xd_execute(vk_plan), "vkfft_xd_execute");
        cuntt::detail::launch_xd_axis_fft(
            row_output.ptr, hybrid_output.ptr, axis, axis, lines, false,
            false, 1.0, stream, 8, 0, 0, 0);
        check(cudaStreamSynchronize(stream));

        std::vector<cufftComplex> got(count), expected(count);
        check(cudaMemcpy(got.data(), hybrid_output.ptr, count * sizeof(got[0]),
                         cudaMemcpyDeviceToHost));
        check(cudaMemcpy(expected.data(), reference.ptr, count * sizeof(expected[0]),
                         cudaMemcpyDeviceToHost));
        double max_error = 0.0;
        double max_reference = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            max_error = std::max(max_error,
                                 std::hypot(double(got[i].x) - expected[i].x,
                                            double(got[i].y) - expected[i].y));
            max_reference = std::max(max_reference,
                                     std::hypot(double(expected[i].x),
                                                double(expected[i].y)));
        }
        std::cout << "planes=" << planes << " hybrid_max_abs_error=" << max_error
                  << " hybrid_relative_max_error="
                  << max_error / std::max(1.0, max_reference) << '\n';

        cudaGraph_t graph{};
        cudaGraphExec_t instance{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        check_vk(vkfft_xd_execute(vk_plan), "vkfft_xd_execute(capture)");
        cuntt::detail::launch_xd_axis_fft(
            row_output.ptr, hybrid_output.ptr, axis, axis, lines, false,
            false, 1.0, stream, 8, 0, 0, 0);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

        cudaEvent_t start{}, stop{};
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&stop));
        const float hybrid_ms = replay(instance, stream, start, stop, warmup, repeat);
        std::cout << "vkfft_then_cubutterfly_graph_ms=" << hybrid_ms << '\n';

        check_fft(cufftSetStream(cufft_plan, stream));
        cudaGraph_t cufft_graph{};
        cudaGraphExec_t cufft_instance{};
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        check_fft(cufftExecC2C(cufft_plan, input.ptr, reference.ptr, CUFFT_FORWARD));
        check(cudaStreamEndCapture(stream, &cufft_graph));
        check(cudaGraphInstantiate(&cufft_instance, cufft_graph, nullptr, nullptr, 0));
        const float cufft_ms = replay(cufft_instance, stream, start, stop, warmup, repeat);
        std::cout << "cufft_graph_ms=" << cufft_ms
                  << " hybrid_over_cufft=" << (hybrid_ms / cufft_ms) << '\n';

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGraphExecDestroy(instance);
        cudaGraphExecDestroy(cufft_instance);
        cudaGraphDestroy(graph);
        cudaGraphDestroy(cufft_graph);
        check_vk(vkfft_xd_destroy(vk_plan), "vkfft_xd_destroy");
        cufftDestroy(cufft_plan);
        cudaStreamDestroy(stream);
        return max_error <= 2e-3 * std::max(1.0, max_reference) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
