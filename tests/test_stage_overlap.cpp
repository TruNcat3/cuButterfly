#include "cuntt/butterfly.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Buffer {
    void* data = nullptr;
    explicit Buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    cuntt::Complex32* values() { return static_cast<cuntt::Complex32*>(data); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
template <typename Value> struct PinnedValues {
    Value* pointer = nullptr;
    explicit PinnedValues(std::size_t size) {
        check(cudaMallocHost(reinterpret_cast<void**>(&pointer), size * sizeof(*pointer)));
    }
    ~PinnedValues() { cudaFreeHost(pointer); }
    Value* data() { return pointer; }
    const Value& operator[](std::size_t index) const { return pointer[index]; }
};
template <typename Value=cuntt::Complex32>
void run(bool imported, bool inplace, bool inverse, std::size_t batch, unsigned tile,
         bool large_shared = false, bool swizzled = false, bool vectorized = false,
         const std::vector<std::uint32_t>& partition = {}, bool register_core=false) {
    cuntt::ButterflyConfig config;
    config.op = cuntt::ButterflyOperator::Fft;
    config.backend = imported ? cuntt::ButterflyBackend::OnlineReorder : cuntt::ButterflyBackend::SharedIterative;
    config.fft_core = imported ? cuntt::FftCore::CufftDxBlock : cuntt::FftCore::Scalar;
    config.log_n = imported ? 16 : (large_shared ? 14 : 12);
    config.local_stages = large_shared ? 12 : config.log_n / 2;
    config.stage_partition = {config.local_stages, config.log_n - config.local_stages};
    if (!partition.empty()) config.stage_partition = partition;
    if (large_shared) config.shared_layout = cuntt::SharedLayout::WriterAligned;
    if (swizzled) config.shared_layout = cuntt::SharedLayout::XorSwizzle;
    config.prefix_threads = config.suffix_threads = 256;
    config.prefix_ept = config.suffix_ept = 8;
    config.tile_threads = 128;
    config.precision=std::is_same_v<Value,cuntt::Complex64> ? cuntt::ButterflyPrecision::Fp64 : cuntt::ButterflyPrecision::Fp32;
    if (register_core) {
        config.backend=cuntt::ButterflyBackend::OnlineReorder;
        config.fft_core=cuntt::FftCore::RegisterTile;
        config.log_n=16; config.local_stages=6; config.stage_partition={6,10};
        config.prefix_threads=64; config.prefix_ept=8;
        config.suffix_threads=256; config.suffix_ept=16;
        config.shared_layout=cuntt::SharedLayout::WriterAligned;
        config.cross_twiddle=cuntt::CrossTwiddleMode::Recurrence;
    }
    config.stage_overlap = true;
    config.batch_tile_count = tile;
    config.batch = batch;
    config.inverse = inverse;
    config.placement = inplace ? cuntt::ButterflyPlacement::InPlace : cuntt::ButterflyPlacement::OutOfPlace;
    const std::size_t stride = vectorized ? 1 : 2;
    config.element_stride = stride;
    const std::size_t n = std::size_t{1} << config.log_n;
    config.batch_stride = stride * n + (vectorized ? 0 : 13);
    config.auto_allocate_workspace = false;
    const auto extent = config.batch_stride * batch;
    const auto bytes = extent * sizeof(Value);
    Buffer input0(bytes), input1(bytes), output0(bytes), output1(bytes);
    Stream stream0, stream1;
    cuntt::ButterflyPlan plan(config);
    if (plan.workspace_size() != (config.stage_partition.size()-1) * 2 * tile * config.batch_stride * sizeof(Value))
        throw std::runtime_error("workspace does not match two bounded tile slots per edge");
    Buffer workspace(plan.workspace_size() + 256);
    check(cudaMemset(workspace.data, 0x5a, plan.workspace_size() + 256));
    plan.set_workspace(static_cast<char*>(workspace.data) + 128, plan.workspace_size());
    const Value sentinel{1234.0f, -4321.0f};
    std::vector<Value> input(extent, sentinel), expected(extent, sentinel);
    PinnedValues<Value> upload(extent), result0(extent), result1(extent);
    std::mt19937 generator(127);
    std::uniform_real_distribution<float> distribution(-1, 1);
    for (std::size_t b = 0; b < batch; ++b) {
        std::vector<Value> transform(n);
        for (std::size_t j = 0; j < n; ++j)
            input[b * config.batch_stride + stride * j] = transform[j] = {distribution(generator), distribution(generator)};
        cuntt::reference_fft(transform, inverse, true);
        for (std::size_t j = 0; j < n; ++j) expected[b * config.batch_stride + stride * j] = transform[j];
    }
    std::copy(input.begin(), input.end(), upload.data());
    for (auto* buffer : {&input1, &output0, &output1})
        check(cudaMemcpy(buffer->data, upload.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaMemset(input0.data, 0, bytes));
    // Test setup runs on the default stream; the two caller streams are
    // nonblocking. Finish the memset before the first caller uploads input0,
    // otherwise the setup itself can zero the input after that upload.
    // This fence precedes both submissions and cannot mask their handoff.
    check(cudaStreamSynchronize(nullptr));
    // Pinned host memory prevents a pageable D2H copy from implicitly waiting
    // on the host and accidentally masking a missing cross-invocation fence.
    // Two submissions share the ring but use different caller streams/buffers.
    // Each download relies exclusively on execute_async's caller-stream join.
    auto submit = [&](Stream& stream, Buffer& source, Buffer& destination, auto& host_result, bool upload_input) {
        plan.set_stream(stream.value);
        if (upload_input)
            check(cudaMemcpyAsync(source.data, upload.data(), bytes, cudaMemcpyHostToDevice, stream.value));
        auto* output = static_cast<Value*>(inplace ? source.data : destination.data);
        plan.execute_async(static_cast<Value*>(source.data), output);
        check(cudaMemcpyAsync(host_result.data(), output, bytes, cudaMemcpyDeviceToHost, stream.value));
    };
    submit(stream0, input0, output0, result0, true);
    // The second input is already ready: its upload must not incidentally
    // delay the second execution until the first consumer finishes.
    submit(stream1, input1, output1, result1, false);
    check(cudaStreamSynchronize(stream0.value));
    check(cudaStreamSynchronize(stream1.value));
    double max_reference = 0, max_error = 0;
    for (std::size_t i = 0; i < extent; ++i) {
        const auto local = i % config.batch_stride;
        const bool padding = local >= stride * n || local % stride;
        for (const auto* result : {&result0, &result1}) {
            if (padding && ((*result)[i].real != sentinel.real || (*result)[i].imag != sentinel.imag))
                throw std::runtime_error("strided padding was overwritten");
            if (!padding) {
                const double error = std::hypot((*result)[i].real - expected[i].real, (*result)[i].imag - expected[i].imag);
                if (!std::isfinite(error)) throw std::runtime_error("non-finite output");
                max_error = std::max(max_error, error);
                max_reference = std::max(max_reference, std::hypot(double(expected[i].real), double(expected[i].imag)));
            }
        }
    }
    const double tolerance=std::is_same_v<Value,cuntt::Complex64> ? 2e-11 : 2e-4;
    if (max_error > tolerance * max_reference) {
        std::cerr << "FAIL imported=" << imported << " inplace=" << inplace << " inverse=" << inverse
                  << " batch=" << batch << " tile=" << tile << " swizzled=" << swizzled
                  << " stride=" << stride
                  << " max_error=" << max_error << " max_reference=" << max_reference << '\n';
        throw std::runtime_error("FFT output mismatch");
    }
    unsigned char guards[256];
    check(cudaMemcpy(guards, workspace.data, 128, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(guards + 128, static_cast<char*>(workspace.data) + 128 + plan.workspace_size(), 128, cudaMemcpyDeviceToHost));
    for (auto byte : guards) if (byte != 0x5a) throw std::runtime_error("workspace canary changed");
    std::cout << "PASS overlap logN=" << config.log_n << " imported=" << imported << " inplace=" << inplace << " inverse=" << inverse
              << " register=" << register_core << " value_bytes=" << sizeof(Value)
              << " batch=" << batch << " tile=" << tile << " swizzled=" << swizzled
              << " stride=" << stride
              << " relative_max_error=" << max_error / max_reference << '\n';
}
}
int main() {
    try {
        for (bool imported : {false, true}) {
#ifndef CUBUTTERFLY_TEST_CUFFTDX
            if (imported) continue;
#endif
            for (bool inplace : {false, true})
                for (bool inverse : {false, true})
                    for (auto batch : {1U, 5U, 17U}) run(imported, inplace, inverse, batch, 2);
        }
        run(false, false, false, 5, 3, true);
        run(false, true, true, 5, 3, true);
        for (const auto& partition : {std::vector<std::uint32_t>{3,4,5},
                                     std::vector<std::uint32_t>{2,3,3,4}, std::vector<std::uint32_t>(12,1)})
            for (bool inplace : {false,true}) for (bool inverse : {false,true})
                run(false,inplace,inverse,17,3,false,false,false,partition);
#ifdef CUBUTTERFLY_TEST_CUFFTDX
        const char* policy=std::getenv("CUBUTTERFLY_COMPILE_MODE");
        for (bool inplace : {false,true}) for (bool inverse : {false,true}) for (auto batch : {1U,5U,17U}) {
            run(false,inplace,inverse,batch,2,false,false,false,{},true);
            if (policy && std::string(policy)=="research")
                run<cuntt::Complex64>(false,inplace,inverse,batch,2,false,false,false,{},true);
        }
        for (bool inplace : {false, true})
            for (bool inverse : {false, true})
                run(true, inplace, inverse, 5, 2, false, true);
        run(true, false, false, 5, 2, false, true, true);
        run(true, true, true, 5, 2, false, true, true);
#endif
        for (int invalid = 0; invalid < 3; ++invalid) {
            cuntt::ButterflyConfig config;
            config.op = cuntt::ButterflyOperator::Fft;
            config.backend = invalid == 0 ? cuntt::ButterflyBackend::CuFft : cuntt::ButterflyBackend::SharedIterative;
            config.log_n = 12;
            config.stage_partition = invalid == 2 ? std::vector<std::uint32_t>{12}
                                                  : std::vector<std::uint32_t>{6, 6};
            config.stage_overlap = true;
            config.batch_tile_count = invalid == 1 ? 0 : 1;
            bool rejected = false;
            try { cuntt::ButterflyPlan plan(config); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("unsupported stage-overlap configuration accepted");
        }
        std::cout << "PASS unsupported overlap configurations rejected\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
