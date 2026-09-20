#include "cuntt/butterfly.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    cuntt::Complex32* pointer = nullptr;
    explicit Buffer(std::size_t bytes) { check(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes)); }
    ~Buffer() { cudaFree(pointer); }
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};
}

// Positional arguments: logN batch rounds repeat [--layout-sweep]. All plan construction,
// allocation, preheating and warmups are outside CUDA-event timing. A single
// process alternates identical local-unit mappings and a cuFFT reference.
int main(int argc, char** argv) {
    try {
        const unsigned log_n = argc > 1 ? std::stoul(argv[1]) : 20;
        const std::size_t batch = argc > 2 ? std::stoull(argv[2]) : 16;
        const unsigned rounds = argc > 3 ? std::stoul(argv[3]) : 7;
        const unsigned repeat = argc > 4 ? std::stoul(argv[4]) : 100;
        const bool layout_sweep = argc > 5 && std::string(argv[5]) == "--layout-sweep";
        if (argc > 6 || (argc > 5 && !layout_sweep))
            throw std::invalid_argument("optional fifth argument must be --layout-sweep");
        if ((log_n != 18 && log_n != 20) || batch == 0 || batch > 256 || rounds == 0 || repeat == 0)
            throw std::invalid_argument("usage: fft_stage_overlap_bench {18|20} batch[1..256] rounds repeat");
        Buffer input((std::size_t{1} << log_n) * batch * sizeof(cuntt::Complex32));
        Buffer output((std::size_t{1} << log_n) * batch * sizeof(cuntt::Complex32));
        // The same random input and device allocation are shared by all mappings.
        {
            std::vector<cuntt::Complex32> values((std::size_t{1} << log_n) * batch);
            std::mt19937 generator(127);
            std::uniform_real_distribution<float> distribution(-1, 1);
            for (auto& value : values) value = {distribution(generator), distribution(generator)};
            check(cudaMemcpy(input.pointer, values.data(), values.size() * sizeof(values[0]), cudaMemcpyHostToDevice));
        }
        std::vector<std::unique_ptr<cuntt::ButterflyPlan>> plans;
        std::vector<std::string> names;
        std::vector<unsigned> tiles{0, 1, 4, 8, 16};
        if (layout_sweep) tiles = {0};
        for (auto tile : tiles) {
            if (tile > batch) continue;
            cuntt::ButterflyConfig config;
            config.op = cuntt::ButterflyOperator::Fft;
            config.backend = cuntt::ButterflyBackend::OnlineReorder;
            config.fft_core = cuntt::FftCore::CufftDxBlock;
            config.log_n = log_n;
            config.batch = batch;
            config.stage_partition = {log_n - 12, 12};
            config.prefix_threads = config.suffix_threads = 256;
            config.prefix_ept = config.suffix_ept = 16;
            config.direct_boundary = cuntt::DirectBoundary::TiledTranspose;
            config.stage_overlap = tile != 0;
            config.batch_tile_count = std::max(1U, tile);
            if (layout_sweep) {
                for (unsigned threads : {128U, 256U, 512U})
                    for (auto twiddle : {cuntt::CrossTwiddleMode::Table, cuntt::CrossTwiddleMode::Recurrence})
                        for (auto layout : {cuntt::SharedLayout::Linear, cuntt::SharedLayout::XorSwizzle}) {
                            config.prefix_threads = threads;
                            config.cross_twiddle = twiddle;
                            config.shared_layout = layout;
                            plans.push_back(std::make_unique<cuntt::ButterflyPlan>(config));
                            names.push_back(std::string(layout == cuntt::SharedLayout::Linear ? "linear" : "xor") +
                                "-p" + std::to_string(threads) +
                                (twiddle == cuntt::CrossTwiddleMode::Table ? "-table" : "-recurrence"));
                        }
            } else {
                plans.push_back(std::make_unique<cuntt::ButterflyPlan>(config));
                names.push_back(tile ? "overlap-" + std::to_string(tile) : "bulk");
            }
        }
        cuntt::ButterflyConfig reference;
        reference.op = cuntt::ButterflyOperator::Fft;
        reference.backend = cuntt::ButterflyBackend::CuFft;
        reference.log_n = log_n;
        reference.batch = batch;
        plans.push_back(std::make_unique<cuntt::ButterflyPlan>(reference));
        names.push_back("cufft");
        // Validate every measured mapping against cuFFT on the same input.
        // Neither this check nor plan construction is part of event timing.
        const auto values_count = (std::size_t{1} << log_n) * batch;
        std::vector<cuntt::Complex32> expected(values_count), actual(values_count);
        plans.back()->execute_async(input.pointer, output.pointer);
        check(cudaMemcpy(expected.data(), output.pointer, values_count * sizeof(expected[0]), cudaMemcpyDeviceToHost));
        double max_reference = 0;
        for (const auto& value : expected)
            max_reference = std::max(max_reference, std::hypot(double(value.real), double(value.imag)));
        for (std::size_t i = 0; i + 1 < plans.size(); ++i) {
            plans[i]->execute_async(input.pointer, output.pointer);
            check(cudaMemcpy(actual.data(), output.pointer, values_count * sizeof(actual[0]), cudaMemcpyDeviceToHost));
            double max_error = 0;
            for (std::size_t j = 0; j < values_count; ++j) {
                const double error = std::hypot(double(actual[j].real) - expected[j].real,
                                                double(actual[j].imag) - expected[j].imag);
                if (!std::isfinite(error)) throw std::runtime_error("non-finite result: " + names[i]);
                max_error = std::max(max_error, error);
            }
            if (max_error > 2e-4 * max_reference)
                throw std::runtime_error("cuFFT comparison failed: " + names[i]);
            std::cerr << "verified " << names[i] << " relative_max_error=" << max_error / max_reference << '\n';
        }
        Event start, stop;
        const auto preheat_start = std::chrono::steady_clock::now();
        do {
            for (int i = 0; i < 10; ++i) plans[0]->execute_async(input.pointer, output.pointer);
            check(cudaStreamSynchronize(nullptr));
        } while (std::chrono::steady_clock::now() - preheat_start < std::chrono::milliseconds(400));
        std::vector<unsigned> order(plans.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(127);
        std::cout << "logN,batch,round,mode,kernel_ms,workspace_bytes\n";
        for (unsigned round = 0; round < rounds; ++round) {
            std::shuffle(order.begin(), order.end(), random);
            for (auto index : order) {
                auto& plan = *plans[index];
                for (int i = 0; i < 10; ++i) plan.execute_async(input.pointer, output.pointer);
                check(cudaStreamSynchronize(nullptr));
                check(cudaEventRecord(start.value));
                for (unsigned i = 0; i < repeat; ++i) plan.execute_async(input.pointer, output.pointer);
                check(cudaEventRecord(stop.value));
                check(cudaEventSynchronize(stop.value));
                float elapsed = 0;
                check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
                std::cout << log_n << ',' << batch << ',' << round << ',' << names[index] << ','
                          << elapsed / repeat << ',' << plan.workspace_size() << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
