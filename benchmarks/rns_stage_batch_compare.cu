// Standalone stage diagnostic: explicit independent requests versus packed rows.
// This is not a CKKS end-to-end benchmark and assumes an already packed arena.
// Build with -arch=sm_80 -rdc=true --cudart shared: link pinned device code and
// share libcuntt's runtime so Graph kernel queries resolve its kernel symbols.
#include <cuntt/rns_ntt.hpp>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Word = std::uint64_t;
using Wide = unsigned __int128;
std::array<Word, 2> kModuli = {1152921504606584833ULL, 576460756061519873ULL};
constexpr const char* kModes[] = {"cu-loop", "cu-packed", "native-loop", "native-packed",
                                "cu-pack-transform-unpack", "native-pack-transform-unpack"};
template <bool Pack>
__global__ void transfer_requests(Word* const* requests, Word* arena,
                                  std::size_t request_words, std::size_t words) {
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < words) {
        Word* separate = requests[index / request_words] + index % request_words;
        if constexpr (Pack) arena[index] = *separate;
        else *separate = arena[index];
    }
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cudaError_t status, const std::string& context) {
    if (status != cudaSuccess) throw std::runtime_error(context + ": " + cudaGetErrorString(status));
}
Word multiply(Word a, Word b, Word q) { return static_cast<Word>(Wide(a) * b % q); }
Word power(Word a, Word e, Word q) {
    Word result = 1;
    for (; e; e >>= 1, a = multiply(a, a, q)) if (e & 1) result = multiply(result, a, q);
    return result;
}
std::size_t reverse_bits(std::size_t a, unsigned bits) {
    std::size_t result = 0;
    for (unsigned i = 0; i < bits; ++i, a >>= 1) result = (result << 1) | (a & 1);
    return result;
}
struct Options {
    unsigned log_n = 14, rows = 2, requests = 8, inverse = 0, repetitions = 20;
    bool include_packing = false;
    std::array<Word, 2> roots{};
};
std::array<Word, 2> pair(const std::string& value) {
    const auto comma = value.find(',');
    if (comma == std::string::npos || value.find(',', comma + 1) != std::string::npos)
        throw std::invalid_argument("expected two comma-separated unsigned integers");
    const auto a = value.substr(0, comma), b = value.substr(comma + 1);
    if (a.empty() || b.empty() || a.find_first_not_of("0123456789") != std::string::npos ||
        b.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("invalid modulus/root pair");
    return {std::stoull(a), std::stoull(b)};
}
unsigned number(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected unsigned integer");
    const auto parsed = std::stoull(value);
    if (parsed > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("integer overflow");
    return static_cast<unsigned>(parsed);
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        if (key == "--moduli") { kModuli = pair(argv[i + 1]); continue; }
        if (key == "--roots") { o.roots = pair(argv[i + 1]); continue; }
        const unsigned v = number(argv[i + 1]);
        if (key == "--log-n") o.log_n = v;
        else if (key == "--rows-per-request") o.rows = v;
        else if (key == "--request-count") o.requests = v;
        else if (key == "--inverse") o.inverse = v;
        else if (key == "--repetitions") o.repetitions = v;
        else if (key == "--include-packing") {
            if (v > 1) throw std::invalid_argument("include-packing must be 0 or 1");
            o.include_packing = v != 0;
        }
        else throw std::invalid_argument("unknown option " + key);
    }
    if (o.log_n < 12 || o.log_n > 20 || (o.rows != 2 && o.rows != 12) ||
        (o.requests != 1 && o.requests != 8 && o.requests != 32) || o.inverse > 1 || !o.repetitions)
        throw std::invalid_argument("support: log-n 12..20 (subject to 2N root), rows 2/12, requests 1/8/32, inverse 0/1, repetitions > 0");
    for (Word q : kModuli)
        if (q < 3 || q >= (Word{1} << 62) || (q - 1) % (Word{1} << (o.log_n + 1)))
            throw std::invalid_argument("fixed native modulus does not admit a primitive 2N-th root");
    return o;
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
template <class T> struct Device {
    T* data = nullptr;
    explicit Device(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
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
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

// CPU DIT with explicit boundary permutation and negacyclic twist. No GPU tables.
std::vector<Word> reference(const std::vector<Word>& input, const Options& o,
                            const std::array<Word, 2>& roots) {
    const std::size_t n = std::size_t{1} << o.log_n;
    std::vector<Word> result(input.size()), values(n);
    for (unsigned row = 0; row < o.rows * o.requests; ++row) {
        const Word q = kModuli[row % 2], psi = roots[row % 2];
        Word twist = 1;
        for (std::size_t j = 0; j < n; ++j) {
            values[j] = input[row * n + (o.inverse ? reverse_bits(j, o.log_n) : j)];
            if (!o.inverse) values[j] = multiply(values[j], twist, q);
            twist = multiply(twist, psi, q);
        }
        Word omega = multiply(psi, psi, q);
        if (o.inverse) omega = power(omega, q - 2, q);
        for (std::size_t j = 0; j < n; ++j) {
            const auto r = reverse_bits(j, o.log_n);
            if (j < r) std::swap(values[j], values[r]);
        }
        for (std::size_t length = 2; length <= n; length *= 2) {
            const Word step = power(omega, n / length, q);
            for (std::size_t base = 0; base < n; base += length) {
                Word w = 1;
                for (std::size_t j = 0; j < length / 2; ++j) {
                    const Word a = values[base + j], b = multiply(values[base + j + length / 2], w, q);
                    values[base + j] = a + b >= q ? a + b - q : a + b;
                    values[base + j + length / 2] = a >= b ? a - b : a + q - b;
                    w = multiply(w, step, q);
                }
            }
        }
        Word factor = o.inverse ? power(n, q - 2, q) : 1;
        const Word step = o.inverse ? power(psi, q - 2, q) : 1;
        for (std::size_t j = 0; j < n; ++j) {
            result[row * n + (o.inverse ? j : reverse_bits(j, o.log_n))] = multiply(values[j], factor, q);
            factor = multiply(factor, step, q);
        }
    }
    return result;
}
std::vector<Word> input_for(std::size_t words, std::size_t n, Word seed) {
    std::mt19937_64 random(seed);
    std::vector<Word> input(words);
    for (std::size_t i = 0; i < words; ++i) input[i] = random() % kModuli[(i / n) % 2];
    for (std::size_t base = 0; base < words; base += n) {
        input[base] = 0;
        input[base + 1] = kModuli[(base / n) % 2] - 1;
    }
    return input;
}
std::string mapping(const cuntt::RnsMapping& m) {
    std::ostringstream s;
    s << "family=" << static_cast<int>(m.core_family) << ";n1=" << m.n1_log
      << ";rows=" << m.rows_per_block << ";threads=" << m.threads_per_block
      << ";core=" << static_cast<int>(m.compute_unit) << ";cross=" << static_cast<int>(m.cross_twiddle)
      << ";physical=" << m.physical_boundary_traversal << ";second_rows=" << m.second_rows_per_block
      << ";second_threads=" << m.second_threads_per_block << ";arithmetic=" << static_cast<int>(m.arithmetic)
      << ";exchange=" << (m.exchange == cuntt::RnsExchange::DependencyScoped ? "dependency_scoped" : "block");
    return s.str();
}
std::size_t describe_graph(const Graph& g, const char* mode, unsigned repeats) {
    std::size_t count = 0, kernels = 0;
    check(cudaGraphGetNodes(g.graph, nullptr, &count), std::string(mode) + " Graph node count");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(g.graph, nodes.data(), &count), std::string(mode) + " Graph nodes");
    for (auto node : nodes) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node, &type), std::string(mode) + " Graph node type");
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams p{};
        check(cudaGraphKernelNodeGetParams(node, &p), std::string(mode) + " Graph kernel parameters");
        std::cout << "kernel,mode," << mode << ",index," << kernels++ << ",capture_repeats," << repeats
                  << ",function," << p.func << ",grid," << p.gridDim.x << ':' << p.gridDim.y << ':' << p.gridDim.z
                  << ",block," << p.blockDim.x << ':' << p.blockDim.y << ':' << p.blockDim.z
                  << ",shared_bytes," << p.sharedMemBytes << '\n';
    }
    return kernels;
}
void run(const Options& o) {
    const std::size_t n = std::size_t{1} << o.log_n, rows = o.rows * o.requests, words = n * rows;
    const std::size_t request_words = o.rows * n;
    const unsigned mode_count = o.include_packing ? 6 : 4;
    std::array<Word, 2> roots;
    for (unsigned d = 0; d < 2; ++d) {
        roots[d] = o.roots[d] ? o.roots[d] : cuntt::find_primitive_power_of_two_root(o.log_n + 1, kModuli[d]);
        if (roots[d] >= kModuli[d] || power(roots[d], 2 * n, kModuli[d]) != 1 ||
            power(roots[d], n, kModuli[d]) != kModuli[d] - 1)
            throw std::invalid_argument("root must be a canonical primitive forward 2N-th root");
    }
    std::cout << "fixture,moduli," << kModuli[0] << ':' << kModuli[1]
              << ",forward_roots," << roots[0] << ':' << roots[1]
              << ",boundary," << (o.inverse ? "BR-to-natural" : "natural-to-BR")
              << ",scope," << (o.include_packing ? "same-stage-layout-cost-control" : "same-stage-prepacked-control") << '\n';
    auto config_for = [&](unsigned count) {
        cuntt::RnsPlanConfig c;
        c.log_n = o.log_n;
        c.inverse = o.inverse;
        c.input_order = o.inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
        c.output_order = o.inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed;
        std::array<std::uint32_t, 2> domain_indices;
        for (unsigned d = 0; d < 2; ++d) {
            const auto found = std::find_if(c.domains.begin(), c.domains.end(), [&](const cuntt::RnsDomain& domain) {
                return domain.modulus == kModuli[d] && domain.root == roots[d];
            });
            domain_indices[d] = static_cast<std::uint32_t>(found - c.domains.begin());
            if (found == c.domains.end()) c.domains.push_back({kModuli[d], roots[d]});
        }
        for (unsigned row = 0; row < count; ++row)
            c.polynomials.push_back({domain_indices[row % 2], row * n, row * n});
        return c;
    };
    Stream stream;
    Device<Word> arena(words);
    std::vector<std::unique_ptr<Device<Word>>> separate;
    std::vector<Word*> request_pointers;
    std::unique_ptr<Device<Word*>> device_requests;
    if (o.include_packing) {
        for (unsigned request = 0; request < o.requests; ++request) {
            separate.emplace_back(std::make_unique<Device<Word>>(request_words));
            request_pointers.push_back(separate.back()->data);
        }
        device_requests = std::make_unique<Device<Word*>>(o.requests);
        check(cudaMemcpy(device_requests->data, request_pointers.data(), o.requests * sizeof(Word*), cudaMemcpyHostToDevice));
        std::cout << "packing,request_allocations," << o.requests << ",separate_bytes," << words * sizeof(Word)
                  << ",arena_bytes," << words * sizeof(Word) << ",pointer_array_bytes," << o.requests * sizeof(Word*)
                  << ",transfer_threads,256,loop_boundary,separate-requests,packed_boundary,arena"
                  << ",adapted_boundary,separate-requests,scope,stage-only-not-full-CKKS\n";
    }
    cuntt::RnsPlan small(config_for(o.rows)), packed(config_for(static_cast<unsigned>(rows)));
    const auto& native_domains = small.config().domains;
    const unsigned native_domain_count = static_cast<unsigned>(native_domains.size());
    std::cout << "domains,cu_unique," << native_domain_count << ",native_tables," << native_domain_count << '\n';
    small.tune(cuntt::RnsPlacement::InPlace, stream.value, 5);
    packed.tune(cuntt::RnsPlacement::InPlace, stream.value, 5);
    std::cout << "mapping,mode,cu-loop,value," << mapping(small.mapping()) << '\n'
              << "mapping,mode,cu-packed,value," << mapping(packed.mapping()) << '\n';
    if (o.include_packing)
        std::cout << "mapping,mode,cu-pack-transform-unpack,value," << mapping(packed.mapping()) << '\n';
    std::vector<Root<Word>> host_roots(native_domain_count * n);
    std::vector<Modulus<Word>> host_moduli;
    std::vector<Word> host_inverse;
    for (unsigned d = 0; d < native_domain_count; ++d) {
        const Word q = native_domains[d].modulus, psi = native_domains[d].root;
        const Word root = o.inverse ? power(psi, q - 2, q) : psi;
        Word w = 1;
        for (std::size_t j = 0; j < n; ++j, w = multiply(w, root, q))
            host_roots[d * n + reverse_bits(j, o.log_n)] = w;
        host_moduli.emplace_back(q);
        host_inverse.push_back(power(n, q - 2, q));
    }
    Device<Root<Word>> root_table(host_roots.size());
    Device<Modulus<Word>> moduli(native_domain_count);
    Device<Word> inverses(native_domain_count);
    check(cudaMemcpy(root_table.data, host_roots.data(), host_roots.size() * sizeof(Root<Word>), cudaMemcpyHostToDevice));
    check(cudaMemcpy(moduli.data, host_moduli.data(), host_moduli.size() * sizeof(Modulus<Word>), cudaMemcpyHostToDevice));
    check(cudaMemcpy(inverses.data, host_inverse.data(), host_inverse.size() * sizeof(Word), cudaMemcpyHostToDevice));
    gpuntt::ntt_rns_configuration<Word> native{};
    native.n_power = o.log_n;
    native.ntt_type = o.inverse ? gpuntt::INVERSE : gpuntt::FORWARD;
    native.ntt_layout = gpuntt::PerPolynomial;
    native.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
    native.mod_inverse = inverses.data;
    native.stream = stream.value;
    auto native_launch = [&](Word* data, unsigned count) {
        if (o.inverse) gpuntt::GPU_INTT_Inplace(data, root_table.data, moduli.data, native, count, native_domain_count);
        else gpuntt::GPU_NTT_Inplace(data, root_table.data, moduli.data, native, count, native_domain_count);
    };
    auto launch = [&](unsigned mode) {
        if (mode >= 4) {
            transfer_requests<true><<<(words + 255) / 256, 256, 0, stream.value>>>(device_requests->data, arena.data, request_words, words);
            if (mode == 4) packed.execute_async(arena.data, arena.data, stream.value);
            else native_launch(arena.data, rows);
            transfer_requests<false><<<(words + 255) / 256, 256, 0, stream.value>>>(device_requests->data, arena.data, request_words, words);
        } else if (mode == 1) packed.execute_async(arena.data, arena.data, stream.value);
        else if (mode == 3) native_launch(arena.data, rows);
        else for (unsigned request = 0; request < o.requests; ++request) {
            Word* base = o.include_packing ? request_pointers[request] : arena.data + request * request_words;
            if (mode == 0) small.execute_async(base, base, stream.value);
            else native_launch(base, o.rows);
        }
    };
    std::array<Graph, 6> single, repeated;
    std::array<std::size_t, 6> kernel_counts, repeat_nodes;
    auto capture = [&](Graph& g, unsigned mode, unsigned repeats) {
        const std::string label = std::string(kModes[mode]) + " repeats=" + std::to_string(repeats);
        check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal), label + " capture begin");
        for (unsigned i = 0; i < repeats; ++i) launch(mode);
        check(cudaStreamEndCapture(stream.value, &g.graph), label + " capture end");
        check(cudaGraphInstantiate(&g.executable, g.graph, nullptr, nullptr, 0), label + " Graph instantiate");
    };
    for (unsigned mode = 0; mode < mode_count; ++mode) {
        capture(single[mode], mode, 1);
        kernel_counts[mode] = describe_graph(single[mode], kModes[mode], 1);
        capture(repeated[mode], mode, o.repetitions);
        repeat_nodes[mode] = 0;
        check(cudaGraphGetNodes(repeated[mode].graph, nullptr, &repeat_nodes[mode]));
    }
    const auto original = input_for(words, n, 0x20261019ULL + o.inverse);
    auto reset = [&](const std::vector<Word>& input) {
        check(cudaMemcpyAsync(arena.data, input.data(), words * sizeof(Word), cudaMemcpyHostToDevice, stream.value));
        for (unsigned request = 0; request < separate.size(); ++request)
            check(cudaMemcpyAsync(request_pointers[request], input.data() + request * request_words,
                                  request_words * sizeof(Word), cudaMemcpyHostToDevice, stream.value));
        check(cudaStreamSynchronize(stream.value));
    };
    // A/B/A verifies changing inputs and reset/replay on the same static address.
    for (unsigned trial = 0; trial < 3; ++trial) {
        const auto input = trial == 1 ? input_for(words, n, 0x20261020ULL + o.inverse) : original;
        const auto expected = reference(input, o, roots);
        std::vector<Word> actual(words);
        for (unsigned mode = 0; mode < mode_count; ++mode) {
            reset(input);
            check(cudaGraphLaunch(single[mode].executable, stream.value));
            if (o.include_packing && mode != 1 && mode != 3) {
                for (unsigned request = 0; request < o.requests; ++request)
                    check(cudaMemcpyAsync(actual.data() + request * request_words, request_pointers[request],
                                          request_words * sizeof(Word), cudaMemcpyDeviceToHost, stream.value));
            } else check(cudaMemcpyAsync(actual.data(), arena.data, words * sizeof(Word), cudaMemcpyDeviceToHost, stream.value));
            check(cudaStreamSynchronize(stream.value));
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
            if (mismatch.first != actual.end())
                throw std::runtime_error(std::string(kModes[mode]) + " CPU mismatch trial=" + std::to_string(trial) +
                                         " word=" + std::to_string(mismatch.first - actual.begin()));
            std::cout << "validation,mode," << kModes[mode] << ",trial," << trial << ",exact,1\n";
        }
    }
    Event begin, end;
    auto measure = [&](unsigned mode) {
        reset(original);
        check(cudaEventRecord(begin.value, stream.value));
        check(cudaGraphLaunch(repeated[mode].executable, stream.value));
        check(cudaEventRecord(end.value, stream.value));
        check(cudaEventSynchronize(end.value));
        float ms = 0;
        check(cudaEventElapsedTime(&ms, begin.value, end.value));
        return static_cast<double>(ms);
    };
    std::vector<double> warm(mode_count, 0);
    unsigned warm_round = 0;
    while (*std::min_element(warm.begin(), warm.end()) < 100.0) {
        for (unsigned slot = 0; slot < mode_count; ++slot) {
            const unsigned mode = (slot + warm_round) % mode_count;
            warm[mode] += measure(mode);
        }
        ++warm_round;
    }
    std::array<std::vector<double>, 6> samples;
    for (unsigned round = 0; round < 10; ++round) {
        for (unsigned slot = 0; slot < mode_count; ++slot) {
            const unsigned mode = round % 2 ? (round + mode_count - slot) % mode_count : (round + slot) % mode_count;
            const double ms = measure(mode) / o.repetitions;
            samples[mode].push_back(ms);
            std::cout << "sample,mode," << kModes[mode] << ",round," << round << ",order," << slot
                      << ",full_stage_ms," << ms << ",perpoly_ms," << ms / rows << '\n';
        }
    }
    for (unsigned mode = 0; mode < mode_count; ++mode) {
        auto values = samples[mode];
        std::sort(values.begin(), values.end());
        const double median = (values[4] + values[5]) / 2;
        std::cout << "result,mode," << kModes[mode] << ",log_n," << o.log_n << ",rows_per_request," << o.rows
                  << ",request_count," << o.requests << ",polynomials," << rows << ",inverse," << o.inverse
                  << ",repetitions," << o.repetitions << ",rounds,10,kernel_nodes," << kernel_counts[mode]
                  << ",repeated_graph_nodes," << repeat_nodes[mode]
                  << ",arena_bytes," << words * sizeof(Word) << ",warm_ms," << warm[mode]
                  << ",median_ms," << median << ",perpoly_ms," << median / rows << ",exact,1\n";
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        std::cout << std::setprecision(10);
        run(options);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
