// Optional HEonGPU component regression. CPU arithmetic is independent of both
// stage backends; capture/replay exercises the actual bound stage primitive.
#include "rns_stage_batch.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using U64 = std::uint64_t;
using U128 = unsigned __int128;
using cubutterfly_heongpu::BoundRnsRequest;
using cubutterfly_heongpu::RnsStageBatch;
using cubutterfly_heongpu::StageBackend;
constexpr unsigned kBits = 12;
constexpr std::size_t kN = std::size_t{1} << kBits;
constexpr std::size_t kWords = 9 * kN + 64;
constexpr U64 kSentinel = 0xfedcba9876543210ULL;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
U64 mul(U64 a, U64 b, U64 q) { return U128(a) * b % q; }
U64 powmod(U64 a, U64 e, U64 q) {
    U64 result = 1;
    for (; e; e >>= 1, a = mul(a, a, q)) if (e & 1) result = mul(result, a, q);
    return result;
}
unsigned rev(unsigned x) {
    unsigned result = 0;
    for (unsigned i = 0; i < kBits; ++i, x >>= 1) result = (result << 1) | (x & 1);
    return result;
}
U64 primitive_root(U64 q) {
    for (U64 g = 2; g < q; ++g) {
        const U64 root = powmod(g, (q - 1) / (2 * kN), q);
        if (powmod(root, kN, q) == q - 1) return root;
    }
    throw std::runtime_error("test primitive root search failed");
}
std::vector<U64> cpu(std::vector<U64> values, const cuntt::RnsDomain& domain, bool inverse) {
    const U64 q = domain.modulus, psi = domain.root;
    if (inverse) {
        const auto ordered = values;
        for (unsigned j = 0; j < kN; ++j) values[j] = ordered[rev(j)];
    } else {
        U64 twist = 1;
        for (unsigned j = 0; j < kN; ++j, twist = mul(twist, psi, q))
            values[j] = mul(values[j], twist, q);
    }
    for (unsigned j = 0; j < kN; ++j)
        if (j < rev(j)) std::swap(values[j], values[rev(j)]);
    U64 omega = mul(psi, psi, q);
    if (inverse) omega = powmod(omega, q - 2, q);
    for (unsigned length = 2; length <= kN; length <<= 1) {
        const U64 step = powmod(omega, kN / length, q);
        for (unsigned base = 0; base < kN; base += length) {
            U64 w = 1;
            for (unsigned j = 0; j < length / 2; ++j, w = mul(w, step, q)) {
                const U64 a = values[base + j], b = mul(values[base + j + length / 2], w, q);
                values[base + j] = (a + b) % q;
                values[base + j + length / 2] = (a + q - b) % q;
            }
        }
    }
    if (inverse) {
        U64 scale = powmod(kN, q - 2, q), step = powmod(psi, q - 2, q);
        for (unsigned j = 0; j < kN; ++j, scale = mul(scale, step, q))
            values[j] = mul(values[j], scale, q);
    } else {
        const auto ordered = values;
        for (unsigned j = 0; j < kN; ++j) values[rev(j)] = ordered[j];
    }
    return values;
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Device {
    U64* data = nullptr;
    Device() { check(cudaMalloc(reinterpret_cast<void**>(&data), kWords * sizeof(U64))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};
void upload(U64* destination, const std::vector<U64>& source, cudaStream_t stream) {
    check(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(U64), cudaMemcpyHostToDevice, stream));
}
void equal(const std::vector<U64>& actual, const std::vector<U64>& expected, const std::string& context) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end())
        throw std::runtime_error(context + " word=" + std::to_string(mismatch.first - actual.begin()));
}

std::vector<cuntt::RnsPlanConfig> configs(bool inverse) {
    const U64 q60 = 1152921504606584833ULL, q59 = 576460756061519873ULL;
    const cuntt::RnsDomain a{q60, primitive_root(q60)}, b{q59, primitive_root(q59)};
    // Equal moduli with distinct primitive roots must retain distinct tables.
    const cuntt::RnsDomain c{q60, powmod(a.root, 3, q60)};
    std::vector<cuntt::RnsPlanConfig> result(2);
    for (auto& config : result) {
        config.log_n = kBits;
        config.inverse = inverse;
        config.input_order = inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
        config.output_order = inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed;
    }
    result[0].domains = {a, b, a, c};
    result[1].domains = {b, c, a, b};
    // Both the physical rows and the domain sequence are nonmonotonic. Input
    // and output rows overlap within each aliased request, requiring a snapshot.
    result[0].polynomials = {{2, 5 * kN + 17, kN + 9},
                             {1, kN + 9, 7 * kN + 3},
                             {0, 7 * kN + 3, 3 * kN + 31},
                             {3, 3 * kN + 31, 5 * kN + 17}};
    result[1].polynomials = {{0, 5 * kN + 17, 7 * kN + 3},
                             {1, kN + 9, kN + 9},
                             {2, 7 * kN + 3, 5 * kN + 17}};
    return result;
}

int run(StageBackend backend, bool inverse, bool alias, cudaStream_t stream) {
    auto configuration = configs(inverse);
    Device input0, input1, output0, output1;
    std::vector<U64*> inputs{input0.data, input1.data};
    std::vector<U64*> outputs{alias ? input0.data : output0.data, alias ? input1.data : output1.data};
    std::vector<BoundRnsRequest> requests;
    for (std::size_t r = 0; r < configuration.size(); ++r)
        requests.push_back({configuration[r], inputs[r], outputs[r]});
    RnsStageBatch stage(requests, backend, stream);
    if (stage.polynomial_count() != 7) throw std::runtime_error("stage polynomial count mismatch");
    if (stage.boundary_copy_bytes() != 7 * kN * 2 * sizeof(U64))
        throw std::runtime_error("stage boundary byte count mismatch");
    Graph graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    stage.execute_async(stream);
    check(cudaStreamEndCapture(stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    const std::string label = "backend=" + std::to_string(int(backend)) + " inverse=" +
                              std::to_string(inverse) + " alias=" + std::to_string(alias);
    const std::vector<U64> sentinel(kWords, kSentinel);
    for (unsigned trial = 0; trial < 3; ++trial) {
        std::mt19937_64 random(trial == 1 ? 20261020 : 20261019);
        std::vector<std::vector<U64>> values(2, std::vector<U64>(kWords, kSentinel));
        for (std::size_t r = 0; r < configuration.size(); ++r) {
            for (const auto& p : configuration[r].polynomials) {
                const U64 q = configuration[r].domains[p.domain].modulus;
                for (std::size_t j = 0; j < kN; ++j) values[r][p.input_offset + j] = random() % q;
                values[r][p.input_offset] = 0;
                values[r][p.input_offset + 1] = q - 1;
            }
        }
        auto expected = alias ? values : std::vector<std::vector<U64>>(2, std::vector<U64>(kWords, kSentinel));
        for (std::size_t r = 0; r < configuration.size(); ++r) {
            for (const auto& p : configuration[r].polynomials) {
                const auto begin = values[r].begin() + p.input_offset;
                const auto transformed = cpu(std::vector<U64>(begin, begin + kN),
                                               configuration[r].domains[p.domain], inverse);
                std::copy(transformed.begin(), transformed.end(), expected[r].begin() + p.output_offset);
            }
            upload(inputs[r], values[r], stream);
            if (!alias) upload(outputs[r], sentinel, stream);
        }
        check(cudaGraphLaunch(graph.executable, stream));
        for (std::size_t r = 0; r < configuration.size(); ++r) {
            std::vector<U64> actual(kWords), preserved(kWords);
            check(cudaMemcpyAsync(actual.data(), outputs[r], kWords * sizeof(U64), cudaMemcpyDeviceToHost, stream));
            if (!alias)
                check(cudaMemcpyAsync(preserved.data(), inputs[r], kWords * sizeof(U64), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            const auto context = label + " trial=" + std::to_string(trial) + " request=" + std::to_string(r);
            equal(actual, expected[r], "stage CPU/output/gap mismatch " + context);
            if (!alias) equal(preserved, values[r], "stage modified separate input " + context);
        }
    }
    return 3;
}

void rejects(const std::vector<BoundRnsRequest>& requests, StageBackend backend,
             cudaStream_t stream, const char* label) {
    try {
        RnsStageBatch stage(requests, backend, stream);
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string("stage accepted ") + label);
}
int rejection_cases(StageBackend backend, cudaStream_t stream) {
    Device input0, input1, output0, output1;
    const auto configuration = configs(false);
    const std::vector<BoundRnsRequest> valid{{configuration[0], input0.data, output0.data},
                                           {configuration[1], input1.data, output1.data}};
    auto requests = valid;
    requests[1].output = requests[0].output;
    rejects(requests, backend, stream, "cross-request output overlap");
    requests = valid;
    requests[1].input = requests[0].output;
    rejects(requests, backend, stream, "cross-request input/output dependency");
    requests = valid;
    requests[1].config = configs(true)[1];
    rejects(requests, backend, stream, "incompatible direction");
    requests = valid;
    requests[1].config.output_order = cuntt::RnsOrder::Natural;
    rejects(requests, backend, stream, "incompatible output order");
    requests = valid;
    requests[0].config.input_order = cuntt::RnsOrder::BitReversed;
    requests[1].config.input_order = cuntt::RnsOrder::BitReversed;
    rejects(requests, backend, stream, "unsupported common input order");
    return 5;
}

int aligned_vectorized_case(cudaStream_t stream) {
    auto configuration = configs(false);
    std::size_t row = 0;
    for (auto& config : configuration)
        for (auto& polynomial : config.polynomials) {
            polynomial.input_offset = row * kN;
            polynomial.output_offset = row * kN;
            ++row;
        }
    Device input0, input1, output0, output1;
    std::vector<BoundRnsRequest> requests{{configuration[0], input0.data, output0.data},
                                           {configuration[1], input1.data, output1.data}};
    RnsStageBatch stage(requests, StageBackend::CuButterfly, stream);
    if (stage.boundary_copy_bytes() != row * kN * 2 * sizeof(U64))
        throw std::runtime_error("aligned stage boundary byte count mismatch");
    Graph graph;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    stage.execute_async(stream);
    check(cudaStreamEndCapture(stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
    std::mt19937_64 random(20261020);
    std::vector<std::vector<U64>> values(2, std::vector<U64>(kWords, kSentinel));
    std::vector<std::vector<U64>> expected(2, std::vector<U64>(kWords, kSentinel));
    const std::vector<U64> sentinel(kWords, kSentinel);
    for (std::size_t request = 0; request < configuration.size(); ++request) {
        for (const auto& polynomial : configuration[request].polynomials) {
            const U64 q = configuration[request].domains[polynomial.domain].modulus;
            for (std::size_t j = 0; j < kN; ++j) values[request][polynomial.input_offset + j] = random() % q;
            const auto transformed = cpu(std::vector<U64>(values[request].begin() + polynomial.input_offset,
                                                           values[request].begin() + polynomial.input_offset + kN),
                                         configuration[request].domains[polynomial.domain], false);
            std::copy(transformed.begin(), transformed.end(), expected[request].begin() + polynomial.output_offset);
        }
        upload(input0.data, values[0], stream);
        upload(input1.data, values[1], stream);
    }
    upload(output0.data, sentinel, stream);
    upload(output1.data, sentinel, stream);
    check(cudaGraphLaunch(graph.executable, stream));
    for (std::size_t request = 0; request < configuration.size(); ++request) {
        std::vector<U64> actual(kWords);
        U64* output = request == 0 ? output0.data : output1.data;
        check(cudaMemcpyAsync(actual.data(), output, kWords * sizeof(U64), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        ::equal(actual, expected[request], "aligned vectorized boundary mismatch");
    }
    return 1;
}
} // namespace

int main() {
    try {
        Stream stream;
        int cases = 0, replays = 0, rejected = 0;
        for (auto backend : {StageBackend::Native, StageBackend::CuButterfly}) {
            for (bool inverse : {false, true})
                for (bool alias : {false, true}) {
                    replays += run(backend, inverse, alias, stream.value);
                    ++cases;
                }
            rejected += rejection_cases(backend, stream.value);
        }
        const int vectorized = aligned_vectorized_case(stream.value);
        std::cout << "PASS " << cases << " stage cases, " << replays
                  << " A/B/A Graph replays exact against CPU including all gaps and preserved inputs, "
                  << rejected << " invalid stages rejected, " << vectorized
                  << " aligned vectorized boundary case exact\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
