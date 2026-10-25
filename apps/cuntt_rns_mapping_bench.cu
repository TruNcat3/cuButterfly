// Standalone same-request mapping diagnostic; link against libcuntt and cudart.
#include <cuntt/rns_ntt.hpp>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
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
constexpr Word kPrimes[] = {1152921504577486849ULL, 576460756061519873ULL};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Device {
    Word* data = nullptr;
    bool owns = true;
    explicit Device(std::size_t words) {
        if (words) check(cudaMalloc(reinterpret_cast<void**>(&data), words * sizeof(Word)));
    }
    ~Device() { if (owns && data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void alias(Device& owner) {
        if (owns && data) cudaFree(data);
        data = owner.data;
        owns = false;
    }
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
struct Options {
    unsigned log_n = 14;
    unsigned batch = 6;
    unsigned inverse = 0;
    unsigned repetitions = 30;
    bool compare_radix = false;
    std::string profile_case;
    bool has_fixed_mapping = false;
    cuntt::RnsMapping fixed_mapping;
    bool has_compare_mapping = false;
    cuntt::RnsMapping compare_mapping;
    bool inplace = false;
};
unsigned number(const std::string& value) {
    std::size_t consumed = 0;
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected unsigned integer");
    const auto parsed = std::stoull(value, &consumed);
    if (consumed != value.size() || parsed > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("invalid unsigned integer");
    return static_cast<unsigned>(parsed);
}
cuntt::RnsMapping parse_mapping(const std::string& value) {
    std::istringstream stream(value);
    std::vector<std::string> fields;
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    if ((fields.size() != 6 && fields.size() != 7 && fields.size() != 9 && fields.size() != 10 && fields.size() != 11) || value.back() == ',')
        throw std::invalid_argument("fixed-mapping requires n1,rows,threads,radix,cross,physical[,hybrid|fused|gpuntt[,second_rows,second_threads[,shoup|barrett[,block|dependency_scoped]]]]");
    cuntt::RnsMapping mapping;
    mapping.n1_log = number(fields[0]);
    mapping.rows_per_block = number(fields[1]);
    mapping.threads_per_block = number(fields[2]);
    const unsigned radix = number(fields[3]);
    if (radix == 2) mapping.compute_unit = cuntt::ComputeUnit::Radix2;
    else if (radix == 4) mapping.compute_unit = cuntt::ComputeUnit::Radix4;
    else if (radix == 8) mapping.compute_unit = cuntt::ComputeUnit::Radix8;
    else throw std::invalid_argument("fixed-mapping radix must be 2, 4 or 8");
    if (fields[4] == "first") mapping.cross_twiddle = cuntt::CrossTwiddlePlacement::FirstPass;
    else if (fields[4] == "second") mapping.cross_twiddle = cuntt::CrossTwiddlePlacement::SecondPass;
    else throw std::invalid_argument("fixed-mapping cross must be first or second");
    const unsigned physical = number(fields[5]);
    if (physical > 1) throw std::invalid_argument("fixed-mapping physical must be 0 or 1");
    mapping.physical_boundary_traversal = physical != 0;
    mapping.core_family = cuntt::RnsCoreFamily::Hybrid2D;
    if (fields.size() >= 7) {
        if (fields[6] == "fused") mapping.core_family = cuntt::RnsCoreFamily::FusedCTGS;
        else if (fields[6] == "gpuntt") mapping.core_family = cuntt::RnsCoreFamily::GpuNttMerge;
        else if (fields[6] != "hybrid")
            throw std::invalid_argument("fixed-mapping family must be hybrid, fused or gpuntt");
    }
    if (fields.size() >= 9) {
        mapping.second_rows_per_block = number(fields[7]);
        mapping.second_threads_per_block = number(fields[8]);
    }
    if (fields.size() >= 10) {
        if (fields[9] == "barrett") mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
        else if (fields[9] != "shoup")
            throw std::invalid_argument("fixed-mapping arithmetic must be shoup or barrett");
    }
    if (fields.size() == 11) {
        if (fields[10] == "dependency_scoped") mapping.exchange = cuntt::RnsExchange::DependencyScoped;
        else if (fields[10] == "block") mapping.exchange = cuntt::RnsExchange::Block;
        else throw std::invalid_argument("fixed-mapping exchange must be block or dependency_scoped");
    }
    return mapping;
}
Options parse(int argc, char** argv) {
    Options options;
    for (int j = 1; j < argc; j += 2) {
        const std::string name = argv[j];
        if (j + 1 == argc) throw std::invalid_argument("missing value for " + name);
        if (name == "--profile-case") {
            options.profile_case = argv[j + 1];
            if (options.profile_case != "original" && options.profile_case != "physical" &&
                options.profile_case != "wide" && options.profile_case != "fused" && options.profile_case != "tuned")
                throw std::invalid_argument("profile-case must be original, physical, wide, fused or tuned");
            continue;
        }
        if (name == "--fixed-mapping") {
            options.fixed_mapping = parse_mapping(argv[j + 1]);
            options.has_fixed_mapping = true;
            continue;
        }
        if (name == "--compare-mapping") {
            options.compare_mapping = parse_mapping(argv[j + 1]);
            options.has_compare_mapping = true;
            continue;
        }
        if (name == "--inplace") {
            if (number(argv[j + 1]) != 1) throw std::invalid_argument("inplace must be 1");
            options.inplace = true;
            continue;
        }
        const unsigned value = number(argv[j + 1]);
        if (name == "--log-n") options.log_n = value;
        else if (name == "--batch") options.batch = value;
        else if (name == "--inverse") options.inverse = value;
        else if (name == "--repetitions") options.repetitions = value;
        else if (name == "--compare-radix") {
            if (value != 1) throw std::invalid_argument("compare-radix must be 1");
            options.compare_radix = true;
        }
        else throw std::invalid_argument("unknown option " + name);
    }
    if (options.log_n < 12 || options.log_n > 20 || !options.batch ||
        options.inverse > 1 || !options.repetitions)
        throw std::invalid_argument("require log-n 12..20, batch/repetitions > 0, inverse 0 or 1");
    if (options.compare_radix && (!options.has_fixed_mapping ||
        options.fixed_mapping.core_family != cuntt::RnsCoreFamily::FusedCTGS ||
        !options.profile_case.empty()))
        throw std::invalid_argument("compare-radix requires a fused fixed-mapping and no profile-case");
    if (options.has_compare_mapping && (!options.has_fixed_mapping ||
        options.compare_radix || !options.profile_case.empty()))
        throw std::invalid_argument("compare-mapping requires fixed-mapping, no compare-radix and no profile-case");
    return options;
}
Word multiply(Word a, Word b, Word q) { return Wide(a) * b % q; }
Word power(Word a, Word exponent, Word q) {
    Word result = 1;
    for (; exponent; exponent >>= 1, a = multiply(a, a, q))
        if (exponent & 1) result = multiply(result, a, q);
    return result;
}
std::size_t reverse_bits(std::size_t index, unsigned bits) {
    std::size_t result = 0;
    for (unsigned j = 0; j < bits; ++j, index >>= 1) result = (result << 1) | (index & 1);
    return result;
}

// CPU radix-2 DIT with explicit order, twist and normalization, independent of
// the GPU kernels and their mapping. Input and result are in physical order.
std::vector<Word> reference(const std::vector<Word>& input,
                            const cuntt::RnsPlanConfig& config) {
    const std::size_t n = std::size_t{1} << config.log_n;
    std::vector<Word> result(input.size());
    for (const auto& poly : config.polynomials) {
        const auto& domain = config.domains[poly.domain];
        const Word q = domain.modulus;
        std::vector<Word> values(n);
        for (std::size_t j = 0; j < n; ++j)
            values[j] = input[poly.input_offset + (config.inverse ? reverse_bits(j, config.log_n) : j)];
        if (!config.inverse) {
            Word twist = 1;
            for (auto& value : values) {
                value = multiply(value, twist, q);
                twist = multiply(twist, domain.root, q);
            }
        }
        Word omega = multiply(domain.root, domain.root, q);
        if (config.inverse) omega = power(omega, q - 2, q);
        for (std::size_t j = 0; j < n; ++j) {
            const auto reversed = reverse_bits(j, config.log_n);
            if (j < reversed) std::swap(values[j], values[reversed]);
        }
        for (std::size_t length = 2; length <= n; length *= 2) {
            const Word step = power(omega, n / length, q);
            for (std::size_t base = 0; base < n; base += length) {
                Word twiddle = 1;
                for (std::size_t j = 0; j < length / 2; ++j) {
                    const Word a = values[base + j];
                    const Word b = multiply(values[base + j + length / 2], twiddle, q);
                    values[base + j] = Word((Wide(a) + b) % q);
                    values[base + j + length / 2] = Word((Wide(a) + q - b) % q);
                    twiddle = multiply(twiddle, step, q);
                }
            }
        }
        if (config.inverse) {
            Word factor = power(n, q - 2, q);
            const Word step = power(domain.root, q - 2, q);
            for (auto& value : values) {
                value = multiply(value, factor, q);
                factor = multiply(factor, step, q);
            }
        }
        for (std::size_t j = 0; j < n; ++j)
            result[poly.output_offset + (config.inverse ? j : reverse_bits(j, config.log_n))] = values[j];
    }
    return result;
}
bool equal_device(const Word* device, const std::vector<Word>& expected, cudaStream_t stream) {
    std::vector<Word> actual(expected.size());
    check(cudaMemcpyAsync(actual.data(), device, actual.size() * sizeof(Word), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return actual == expected;
}
void capture(Graph& graph, cuntt::RnsPlan& plan, const Device& input, Device& output,
             cudaStream_t stream, unsigned repetitions) {
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        for (unsigned j = 0; j < repetitions; ++j) plan.execute_async(input.data, output.data, stream);
    } catch (...) {
        cudaGraph_t abandoned = nullptr;
        cudaStreamEndCapture(stream, &abandoned);
        if (abandoned) cudaGraphDestroy(abandoned);
        throw;
    }
    check(cudaStreamEndCapture(stream, &graph.graph));
    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
}
double measure(cuntt::RnsPlan& plan, const Device& input, Device& output,
               cudaStream_t stream, unsigned repetitions) {
    Graph graph;
    capture(graph, plan, input, output, stream, repetitions);
    for (unsigned j = 0; j < 5; ++j) check(cudaGraphLaunch(graph.executable, stream));
    check(cudaStreamSynchronize(stream));
    Event start, stop;
    std::array<double, 5> samples{};
    for (auto& sample : samples) {
        check(cudaEventRecord(start.value, stream));
        check(cudaGraphLaunch(graph.executable, stream));
        check(cudaEventRecord(stop.value, stream));
        check(cudaEventSynchronize(stop.value));
        float elapsed = 0;
        check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
        sample = elapsed / repetitions;
    }
    std::sort(samples.begin(), samples.end());
    return samples[2];
}
std::array<double, 2> compare_mappings(cuntt::RnsPlan& plan,
        const cuntt::RnsMapping* mappings, Device& input, Device& output,
        const std::vector<Word>& expected, const std::vector<Word>& host_input,
        cudaStream_t stream, unsigned repetitions, bool mapping_control, bool inplace) {
    std::array<Graph, 2> graphs;
    const auto validate = [&](unsigned index) {
        check(cudaMemcpyAsync(input.data, host_input.data(), host_input.size() * sizeof(Word),
                              cudaMemcpyHostToDevice, stream));
        check(cudaGraphLaunch(graphs[index].executable, stream));
        if (!equal_device(output.data, expected, stream) ||
            (!inplace && !equal_device(input.data, host_input, stream)))
            throw std::runtime_error("mapping comparison Graph failed CPU-exact validation");
    };
    for (unsigned index = 0; index < graphs.size(); ++index) {
        plan.set_mapping(mappings[index]);
        check(cudaMemcpyAsync(input.data, host_input.data(), host_input.size() * sizeof(Word),
                              cudaMemcpyHostToDevice, stream));
        check(cudaStreamSynchronize(stream));
        plan.execute_async(input.data, output.data, stream);
        check(cudaStreamSynchronize(stream));
        if (!equal_device(output.data, expected, stream) ||
            (!inplace && !equal_device(input.data, host_input, stream)))
            throw std::runtime_error("mapping comparison single-execution exact validation failed");
        capture(graphs[index], plan, input, output, stream, repetitions);
    }
    Event start, stop;
    float warmup_ms = 0;
    unsigned warmup_rounds = 0;
    check(cudaEventRecord(start.value, stream));
    do {
        for (unsigned j = 0; j < 8; ++j) {
            const unsigned first = warmup_rounds % 2;
            check(cudaGraphLaunch(graphs[first].executable, stream));
            check(cudaGraphLaunch(graphs[1 - first].executable, stream));
            ++warmup_rounds;
        }
        check(cudaEventRecord(stop.value, stream));
        check(cudaEventSynchronize(stop.value));
        check(cudaEventElapsedTime(&warmup_ms, start.value, stop.value));
    } while (warmup_ms < 100.0f);
    std::cerr << (mapping_control ? "interleaved_mapping_control_protocol" : "compare_radix_protocol")
              << ",warmup_ms=" << warmup_ms
              << (mapping_control ? ",warmup_graph_launches_per_mapping=" : ",warmup_graph_launches_per_core=")
              << warmup_rounds
              << ",rounds=10,repetitions=" << repetitions << '\n';
    std::array<std::array<double, 10>, 2> samples{};
    for (unsigned round = 0; round < 10; ++round) {
        for (unsigned position = 0; position < 2; ++position) {
            const unsigned index = (round + position) % 2;
            check(cudaEventRecord(start.value, stream));
            check(cudaGraphLaunch(graphs[index].executable, stream));
            check(cudaEventRecord(stop.value, stream));
            check(cudaEventSynchronize(stop.value));
            float elapsed = 0;
            check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
            samples[index][round] = elapsed / repetitions;
        }
        if (mapping_control)
            std::cerr << "interleaved_mapping_control_sample,round=" << round
                      << ",first_mapping=" << round % 2
                      << ",mapping0_ms=" << samples[0][round]
                      << ",mapping1_ms=" << samples[1][round] << '\n';
        else
            std::cerr << "compare_radix_sample,round=" << round << ",first_radix=" << (round % 2 ? 4 : 2)
                      << ",radix2_ms=" << samples[0][round] << ",radix4_ms=" << samples[1][round] << '\n';
    }
    std::array<double, 2> medians{};
    for (unsigned index = 0; index < graphs.size(); ++index) {
        if (!inplace) validate(index);
        std::sort(samples[index].begin(), samples[index].end());
        medians[index] = (samples[index][4] + samples[index][5]) / 2;
    }
    return medians;
}
const char* core(cuntt::ComputeUnit unit) {
    if (unit == cuntt::ComputeUnit::Radix2) return "radix2";
    if (unit == cuntt::ComputeUnit::Radix4) return "radix4";
    if (unit == cuntt::ComputeUnit::Radix8) return "radix8";
    throw std::runtime_error("unexpected RNS compute unit");
}
const char* family(cuntt::RnsCoreFamily value) {
    if (value == cuntt::RnsCoreFamily::Hybrid2D) return "hybrid";
    if (value == cuntt::RnsCoreFamily::FusedCTGS) return "fused";
    if (value == cuntt::RnsCoreFamily::GpuNttMerge) return "gpuntt";
    throw std::runtime_error("unexpected RNS core family");
}
const char* arithmetic(cuntt::RnsArithmetic value) {
    if (value == cuntt::RnsArithmetic::Shoup) return "shoup";
    if (value == cuntt::RnsArithmetic::Barrett) return "barrett";
    throw std::runtime_error("unexpected RNS arithmetic provider");
}
const char* exchange(cuntt::RnsExchange value) {
    if (value == cuntt::RnsExchange::Block) return "block";
    if (value == cuntt::RnsExchange::DependencyScoped) return "dependency_scoped";
    throw std::runtime_error("unexpected RNS exchange strategy");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        const std::size_t n = std::size_t{1} << options.log_n;
        if (options.batch > std::numeric_limits<std::size_t>::max() / n / sizeof(Word))
            throw std::invalid_argument("batch storage overflows size_t");
        cuntt::RnsPlanConfig config;
        config.log_n = options.log_n;
        config.inverse = options.inverse != 0;
        config.ring = cuntt::RnsRing::Negacyclic;
        config.input_order = config.inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
        config.output_order = config.inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed;
        for (Word q : kPrimes)
            config.domains.push_back({q, cuntt::find_primitive_power_of_two_root(options.log_n + 1, q)});
        std::mt19937_64 random(20261018);
        std::vector<Word> host_input(n * options.batch);
        for (unsigned p = 0; p < options.batch; ++p) {
            const unsigned domain = p % 2;
            config.polynomials.push_back({domain, p * n, p * n});
            const Word q = config.domains[domain].modulus;
            for (std::size_t j = 0; j < n; ++j) host_input[p * n + j] = random() % q;
            host_input[p * n] = 0;
            host_input[p * n + 1] = q - 1;
        }
        const auto expected = reference(host_input, config);
        Stream stream;
        Device input(host_input.size()), output(host_input.size());
        if (options.inplace) output.alias(input);
        check(cudaMemcpyAsync(input.data, host_input.data(), host_input.size() * sizeof(Word),
                              cudaMemcpyHostToDevice, stream.value));
        check(cudaStreamSynchronize(stream.value));
        cuntt::RnsPlan plan(config);
        std::array<cuntt::RnsMapping, 5> mappings;
        bool fused_available = options.has_fixed_mapping;
        if (options.has_fixed_mapping) {
            plan.set_mapping(options.fixed_mapping);
            mappings.fill(options.fixed_mapping);
            if (options.compare_radix) {
                mappings[0].compute_unit = cuntt::ComputeUnit::Radix2;
                mappings[1].compute_unit = cuntt::ComputeUnit::Radix4;
            }
            if (options.has_compare_mapping) mappings[1] = options.compare_mapping;
        } else {
            plan.tune(options.inplace ? cuntt::RnsPlacement::InPlace : cuntt::RnsPlacement::Separate,
                      stream.value);
            const auto full = plan.mapping();
            const auto measurements = plan.measurements();
            const cuntt::RnsMeasurement* original = nullptr;
            const cuntt::RnsMeasurement* expanded = nullptr;
            const cuntt::RnsMeasurement* fused = nullptr;
            for (const auto& candidate : measurements) {
                if (candidate.mapping.core_family == cuntt::RnsCoreFamily::FusedCTGS) {
                    if (!fused || candidate.kernel_ms < fused->kernel_ms) fused = &candidate;
                    continue;
                }
                if (candidate.mapping.core_family != cuntt::RnsCoreFamily::Hybrid2D) continue;
                if (candidate.mapping.physical_boundary_traversal) continue;
                if (!expanded || candidate.kernel_ms < expanded->kernel_ms) expanded = &candidate;
                if (candidate.mapping.rows_per_block <= 4 &&
                    (!original || candidate.kernel_ms < original->kernel_ms)) original = &candidate;
            }
            if (!original || !expanded) throw std::runtime_error("tune omitted required comparison subsets");
            auto physical = original->mapping;
            physical.physical_boundary_traversal = true;
            fused_available = fused != nullptr;
            mappings = {original->mapping, physical, expanded->mapping,
                         fused ? fused->mapping : cuntt::RnsMapping{}, full};
        }
        const char* names[] = {"original_subset_winner", "original_winner_physical_only",
                               "expanded_rows_subset_winner", "fused_core_subset_winner", "full_tuned_winner"};
        const char* comparisons[] = {"best_rows_le4_physical_false", "sole_field_physical_boundary",
                                     "best_hybrid_physical_false_not_isolated_ablation",
                                     "best_fused_core_not_isolated_ablation", "best_full_not_isolated_ablation"};
        if (!options.profile_case.empty()) {
            const std::string aliases[] = {"original", "physical", "wide", "fused", "tuned"};
            const auto selected = std::find(std::begin(aliases), std::end(aliases), options.profile_case);
            const auto index = static_cast<std::size_t>(selected - std::begin(aliases));
            if (index == 3 && !fused_available)
                throw std::runtime_error("tune produced no fused core candidate");
            plan.set_mapping(mappings[index]);
            plan.execute_async(input.data, output.data, stream.value);
            if (!equal_device(output.data, expected, stream.value) ||
                (!options.inplace && !equal_device(input.data, host_input, stream.value)))
                throw std::runtime_error("profile mapping failed exact validation");
            check(cudaMemcpyAsync(input.data, host_input.data(), host_input.size() * sizeof(Word),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaStreamSynchronize(stream.value));
            check(cudaProfilerStart());
            plan.execute_async(input.data, output.data, stream.value);
            check(cudaStreamSynchronize(stream.value));
            check(cudaProfilerStop());
            const auto& mapping = mappings[index];
            std::cout << "profile_case,mapping_source,n1_log,rows_per_block,threads_per_block,compute_unit,cross_twiddle,physical_boundary_traversal,core_family,second_rows_per_block,second_threads_per_block,arithmetic,status,exchange\n"
                      << options.profile_case << ',' << (options.has_fixed_mapping ? "fixed" : "tuned_subset")
                      << ',' << mapping.n1_log << ',' << mapping.rows_per_block << ',' << mapping.threads_per_block
                      << ',' << core(mapping.compute_unit)
                      << ',' << (mapping.cross_twiddle == cuntt::CrossTwiddlePlacement::FirstPass ? "first" : "second")
                      << ',' << mapping.physical_boundary_traversal << ',' << family(mapping.core_family)
                      << ',' << mapping.second_rows_per_block << ',' << mapping.second_threads_per_block
                      << ',' << arithmetic(mapping.arithmetic) << ",exact," << exchange(mapping.exchange) << '\n';
            return 0;
        }
        std::cout.precision(10);
        std::array<double, 2> paired_milliseconds{};
        const bool paired_comparison = options.compare_radix || options.has_compare_mapping;
        if (paired_comparison)
            paired_milliseconds = compare_mappings(plan, mappings.data(), input, output, expected,
                host_input, stream.value, options.repetitions, options.has_compare_mapping, options.inplace);
        std::cout << "case,comparison,log_n,batch,inverse,ring,input_order,output_order,q0,q1,n1_log,rows_per_block,threads_per_block,compute_unit,cross_twiddle,physical_boundary_traversal,core_family,second_rows_per_block,second_threads_per_block,arithmetic,repetitions,samples,median_ms,input_preserved,status,exchange\n";
        bool passed = true;
        const std::size_t count = paired_comparison ? 2 : options.has_fixed_mapping ? 1 : mappings.size();
        for (std::size_t j = 0; j < count; ++j) {
            if (j == 3 && !fused_available) continue;
            const auto& mapping = mappings[j];
            plan.set_mapping(mapping);
            check(cudaMemcpyAsync(input.data, host_input.data(), host_input.size() * sizeof(Word),
                                  cudaMemcpyHostToDevice, stream.value));
            check(cudaStreamSynchronize(stream.value));
            plan.execute_async(input.data, output.data, stream.value);
            const bool before = equal_device(output.data, expected, stream.value);
            const double milliseconds = paired_comparison ? paired_milliseconds[j]
                : measure(plan, input, output, stream.value, options.repetitions);
            const bool exact = options.inplace ? before :
                (equal_device(output.data, expected, stream.value) && before);
            const bool preserved = options.inplace || equal_device(input.data, host_input, stream.value);
            passed = passed && exact && preserved;
            std::cout << (options.has_compare_mapping && j == 1 ? "compare_mapping" : options.has_fixed_mapping ? "fixed_mapping" : names[j]) << ','
                      << (options.has_compare_mapping ? "interleaved_mapping_control" : options.compare_radix ? "interleaved_radix_control" : options.has_fixed_mapping ? "explicit_fixed_configuration" : comparisons[j])
                      << ',' << options.log_n << ',' << options.batch
                      << ',' << options.inverse << ",negacyclic," << (config.inverse ? "bit_reversed,natural" : "natural,bit_reversed")
                      << ',' << kPrimes[0] << ',' << kPrimes[1] << ',' << mapping.n1_log << ',' << mapping.rows_per_block
                      << ',' << mapping.threads_per_block << ',' << core(mapping.compute_unit)
                      << ',' << (mapping.cross_twiddle == cuntt::CrossTwiddlePlacement::FirstPass ? "first" : "second")
                      << ',' << mapping.physical_boundary_traversal << ',' << family(mapping.core_family)
                      << ',' << mapping.second_rows_per_block << ',' << mapping.second_threads_per_block
                      << ',' << arithmetic(mapping.arithmetic)
                      << ',' << options.repetitions << ',' << (paired_comparison ? 10 : 5) << ','
                      << milliseconds << ',' << preserved << ',' << (exact && preserved ? "exact" : "validation_failed")
                      << ',' << exchange(mapping.exchange) << '\n';
        }
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "cuntt_rns_mapping_bench: " << error.what() << '\n';
        return 1;
    }
}
