// Standalone exact regression for the public RNS boundary contract.
// Build against a libcuntt.so containing RnsPlan; see the command in main's report.
#include <cuntt/rns_ntt.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Word = std::uint64_t;
using Wide = unsigned __int128;
constexpr Word kSentinel = 0xfedcba9876543210ULL;
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
    explicit Device(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Word)));
    }
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

Word multiply(Word a, Word b, Word q) { return Wide(a) * b % q; }
Word power(Word a, Word exponent, Word q) {
    Word value = 1;
    for (; exponent; exponent >>= 1, a = multiply(a, a, q))
        if (exponent & 1) value = multiply(value, a, q);
    return value;
}
std::size_t reverse_bits(std::size_t index, unsigned bits) {
    std::size_t result = 0;
    for (unsigned j = 0; j < bits; ++j, index >>= 1)
        result = (result << 1) | (index & 1);
    return result;
}

// Ordinary CPU radix-2 DIT, independent of all GPU arithmetic and mappings.
std::vector<Word> reference(const std::vector<Word>& physical,
                            const cuntt::RnsPlanConfig& config,
                            const cuntt::RnsDomain& domain) {
    const std::size_t n = physical.size();
    const Word q = domain.modulus;
    std::vector<Word> values(n);
    for (std::size_t j = 0; j < n; ++j)
        values[j] = physical[config.input_order == cuntt::RnsOrder::Natural
                                  ? j : reverse_bits(j, config.log_n)];
    const bool negacyclic = config.ring == cuntt::RnsRing::Negacyclic;
    if (negacyclic && !config.inverse) {
        Word twist = 1;
        for (auto& value : values) {
            value = multiply(value, twist, q);
            twist = multiply(twist, domain.root, q);
        }
    }
    Word omega = negacyclic ? multiply(domain.root, domain.root, q) : domain.root;
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
        const Word step = negacyclic ? power(domain.root, q - 2, q) : 1;
        for (auto& value : values) {
            value = multiply(value, factor, q);
            factor = multiply(factor, step, q);
        }
    }
    std::vector<Word> result(n);
    for (std::size_t j = 0; j < n; ++j)
        result[config.output_order == cuntt::RnsOrder::Natural
                   ? j : reverse_bits(j, config.log_n)] = values[j];
    return result;
}

cuntt::RnsPlanConfig config_for(unsigned bits, bool negacyclic, bool inverse,
                               bool input_reversed, bool output_reversed) {
    cuntt::RnsPlanConfig config;
    config.log_n = bits;
    config.inverse = inverse;
    config.ring = negacyclic ? cuntt::RnsRing::Negacyclic : cuntt::RnsRing::Cyclic;
    config.input_order = input_reversed ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
    config.output_order = output_reversed ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
    const std::size_t n = std::size_t{1} << bits;
    for (unsigned d = 0; d < 2; ++d) {
        const unsigned root_bits = bits + (negacyclic ? 1 : 0);
        const Word primitive = cuntt::find_primitive_power_of_two_root(root_bits, kPrimes[d]);
        const Word root = power(primitive, d == 0 ? 3 : 5, kPrimes[d]);
        const Word order = Word{1} << root_bits;
        if (power(root, order, kPrimes[d]) != 1 ||
            power(root, order / 2, kPrimes[d]) != kPrimes[d] - 1)
            throw std::runtime_error("root fixture is not primitive");
        config.domains.push_back({kPrimes[d], root});
    }
    // Domain counts are 3:2 and neither boundary groups by domain. The same
    // physical slots are permuted, exercising aliasing between polynomials.
    const unsigned domains[] = {1, 0, 0, 1, 0};
    const unsigned input_slots[] = {3, 0, 6, 2, 5};
    const unsigned output_slots[] = {5, 2, 0, 6, 3};
    for (unsigned p = 0; p < 5; ++p)
        config.polynomials.push_back({domains[p], input_slots[p] * (n + 17) + 11,
                                      output_slots[p] * (n + 17) + 11});
    return config;
}

std::size_t storage_words(const cuntt::RnsPlanConfig& config) {
    return 7 * ((std::size_t{1} << config.log_n) + 17) + 29;
}
std::vector<Word> input_for(const cuntt::RnsPlanConfig& config, Word seed) {
    const std::size_t n = std::size_t{1} << config.log_n;
    std::vector<Word> result(storage_words(config), kSentinel);
    std::mt19937_64 random(seed);
    for (const auto& poly : config.polynomials) {
        const Word q = config.domains[poly.domain].modulus;
        for (std::size_t j = 0; j < n; ++j) result[poly.input_offset + j] = random() % q;
        result[poly.input_offset] = 0;
        result[poly.input_offset + 1] = q - 1;
        result[poly.input_offset + 2] = q / 2;
    }
    return result;
}
std::vector<Word> expected_for(const cuntt::RnsPlanConfig& config,
                               const std::vector<Word>& input, bool in_place) {
    const std::size_t n = std::size_t{1} << config.log_n;
    auto expected = in_place ? input : std::vector<Word>(input.size(), kSentinel);
    for (const auto& poly : config.polynomials) {
        const std::vector<Word> source(input.begin() + poly.input_offset,
                                       input.begin() + poly.input_offset + n);
        const auto transformed = reference(source, config, config.domains[poly.domain]);
        std::copy(transformed.begin(), transformed.end(), expected.begin() + poly.output_offset);
    }
    return expected;
}
void assert_equal(const std::vector<Word>& actual, const std::vector<Word>& expected,
                  const std::string& label) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + ": size mismatch");
    for (std::size_t j = 0; j < actual.size(); ++j)
        if (actual[j] != expected[j])
            throw std::runtime_error(label + ": word " + std::to_string(j) + " expected=" +
                                     std::to_string(expected[j]) + " actual=" + std::to_string(actual[j]));
}
void upload(Word* target, const std::vector<Word>& source, cudaStream_t stream) {
    check(cudaMemcpyAsync(target, source.data(), source.size() * sizeof(Word), cudaMemcpyHostToDevice, stream));
}
std::vector<Word> download(const Word* source, std::size_t words, cudaStream_t stream) {
    std::vector<Word> result(words);
    check(cudaMemcpyAsync(result.data(), source, words * sizeof(Word), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return result;
}

enum class Placement { Alternating, InPlace, OutOfPlace };

void run_case(const cuntt::RnsPlanConfig& config, unsigned index,
               cudaStream_t stream, bool tune, bool graph_test,
               const cuntt::RnsMapping* requested_mapping = nullptr,
               bool invert_placement = false,
               Placement placement = Placement::Alternating,
               bool check_exchange_search = false,
               bool check_single_warp_search = false,
               bool prepare_native_in_place = false) {
    const std::string label = "case " + std::to_string(index) + " log_n=" + std::to_string(config.log_n);
    const bool in_place = placement == Placement::Alternating
        ? (index % 2 == 0) != invert_placement : placement == Placement::InPlace;
    cuntt::RnsPlan plan(config);
    cuntt::RnsMapping mapping;
    const unsigned minimum_split = std::max(6u, config.log_n - 10);
    const unsigned maximum_split = std::min(10u, config.log_n - 6);
    mapping.n1_log = minimum_split + index % (maximum_split - minimum_split + 1);
    mapping.rows_per_block = 1u << (index % 3);
    mapping.threads_per_block = 128;
    const cuntt::ComputeUnit cores[] = {cuntt::ComputeUnit::Radix2,
                                        cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8};
    mapping.compute_unit = cores[index % 3];
    mapping.cross_twiddle = index % 2 ? cuntt::CrossTwiddlePlacement::SecondPass
                                     : cuntt::CrossTwiddlePlacement::FirstPass;
    if (requested_mapping) mapping = *requested_mapping;
    if (prepare_native_in_place) {
        if (!in_place) throw std::runtime_error(label + ": native preparation needs in-place buffers");
        if (plan.workspace_bytes() == 0)
            throw std::runtime_error(label + ": native scratch-release fixture has no scratch");
        plan.set_mapping(mapping, cuntt::RnsPlacement::InPlace);
        if (plan.workspace_bytes() != 0)
            throw std::runtime_error(label + ": native in-place mapping retained scratch");
    } else {
        plan.set_mapping(mapping);
    }
    if (requested_mapping && (mapping.core_family == cuntt::RnsCoreFamily::FusedCTGS ||
                              mapping.core_family == cuntt::RnsCoreFamily::GpuNttMerge)) {
        const auto& actual = plan.mapping();
        if (actual.core_family != mapping.core_family || actual.n1_log != mapping.n1_log ||
            actual.rows_per_block != mapping.rows_per_block ||
            actual.threads_per_block != mapping.threads_per_block ||
            actual.second_rows_per_block != mapping.second_rows_per_block ||
            actual.second_threads_per_block != mapping.second_threads_per_block ||
            actual.compute_unit != mapping.compute_unit ||
            actual.cross_twiddle != mapping.cross_twiddle ||
            actual.physical_boundary_traversal != mapping.physical_boundary_traversal ||
            actual.arithmetic != mapping.arithmetic || actual.exchange != mapping.exchange)
            throw std::runtime_error(label + ": explicit fused mapping was changed");
    }
    if (tune) {
        plan.tune(in_place ? cuntt::RnsPlacement::InPlace : cuntt::RnsPlacement::Separate,
                  stream, 1);
        if (plan.measurements().empty()) throw std::runtime_error("tune produced no measurements");
        if (!plan.selection().automatic || plan.selection().calibrated)
            throw std::runtime_error("tune selection provenance is incorrect");
        const std::string placement_marker = in_place ? "; placement=in-place" : "; placement=separate";
        if (plan.selection().reason.find(placement_marker) == std::string::npos)
            throw std::runtime_error("tune selection placement provenance is incorrect");
        const bool check_fused_search = config.log_n == 12 && !config.inverse &&
            config.ring == cuntt::RnsRing::Negacyclic &&
            config.input_order == cuntt::RnsOrder::Natural &&
            config.output_order == cuntt::RnsOrder::BitReversed;
        bool native_geometry = false, independent_threads = false;
        bool block_exchange = false, dependency_exchange = false;
        bool coupled_single_warp = false, first_single_warp = false, second_single_warp = false;
        for (const auto& measurement : plan.measurements()) {
            if (!(measurement.kernel_ms > 0))
                throw std::runtime_error("tune recorded a nonpositive measurement");
            const auto& candidate = measurement.mapping;
            if (candidate.core_family != cuntt::RnsCoreFamily::Hybrid2D) {
                block_exchange |= candidate.exchange == cuntt::RnsExchange::Block;
                dependency_exchange |= candidate.exchange == cuntt::RnsExchange::DependencyScoped;
            }
            const unsigned second_rows = candidate.second_rows_per_block
                ? candidate.second_rows_per_block : candidate.rows_per_block;
            const unsigned second_threads = candidate.second_threads_per_block
                ? candidate.second_threads_per_block : candidate.threads_per_block;
            if (candidate.core_family == cuntt::RnsCoreFamily::FusedCTGS &&
                candidate.compute_unit == cuntt::ComputeUnit::Radix2 &&
                candidate.arithmetic == cuntt::RnsArithmetic::Shoup &&
                candidate.exchange == cuntt::RnsExchange::DependencyScoped) {
                coupled_single_warp |= candidate.n1_log == 6 && candidate.rows_per_block == 1 &&
                    second_rows == 1 && candidate.threads_per_block == 32 && second_threads == 32;
                if (candidate.n1_log == 9 && candidate.rows_per_block == 64 && second_rows == 1) {
                    first_single_warp |= candidate.threads_per_block == 32 && second_threads == 64;
                    second_single_warp |= candidate.threads_per_block == 64 && second_threads == 32;
                }
            }
            if (candidate.core_family != cuntt::RnsCoreFamily::FusedCTGS ||
                candidate.compute_unit != cuntt::ComputeUnit::Radix2 ||
                candidate.n1_log != 9 || candidate.rows_per_block != 64 ||
                second_rows != 1 || second_threads != 256 ||
                candidate.cross_twiddle != cuntt::CrossTwiddlePlacement::FirstPass ||
                candidate.physical_boundary_traversal) continue;
            native_geometry |= candidate.threads_per_block == 256;
            independent_threads |= candidate.threads_per_block == 64;
        }
        if (check_fused_search && (!native_geometry || !independent_threads))
            throw std::runtime_error("tune omitted native or independent-thread fused geometry");
        if (check_exchange_search && (!block_exchange || !dependency_exchange))
            throw std::runtime_error("tune omitted Block or DependencyScoped measurements");
        if ((check_fused_search || check_single_warp_search) &&
            (!coupled_single_warp || !first_single_warp || !second_single_warp))
            throw std::runtime_error("tune omitted coupled or asymmetric single-warp fused measurements");
        std::cout << "PASS tune measurements=" << plan.measurements().size()
                  << " automatic=" << plan.selection().automatic
                  << " calibrated=" << plan.selection().calibrated
                  << " fused_search_checked=" << check_fused_search
                  << " exchange_search_checked=" << check_exchange_search
                  << " single_warp_search_checked="
                  << (check_fused_search || check_single_warp_search) << '\n';
    }
    const auto original = input_for(config, 20261018 + index);
    const std::vector<Word> sentinel(original.size(), kSentinel);
    const auto expected = expected_for(config, original, in_place);
    Device input(original.size()), output(original.size());
    Word* destination = in_place ? input.data : output.data;
    upload(input.data, original, stream);
    upload(output.data, sentinel, stream);
    plan.execute_async(input.data, destination, stream);
    assert_equal(download(destination, original.size(), stream), expected, label);
    if (!in_place) assert_equal(download(input.data, original.size(), stream), original, label + " input changed");

    if (graph_test) {
        Graph graph;
        upload(input.data, original, stream);
        check(cudaStreamSynchronize(stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        try {
            plan.execute_async(input.data, destination, stream);
        } catch (...) {
            cudaGraph_t abandoned = nullptr;
            cudaStreamEndCapture(stream, &abandoned);
            if (abandoned) cudaGraphDestroy(abandoned);
            throw;
        }
        check(cudaStreamEndCapture(stream, &graph.graph));
        check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
        const auto changed = input_for(config, 0x9e3779b97f4a7c15ULL);
        std::vector<Word> saved_first;
        for (unsigned replay = 0; replay < 3; ++replay) {
            const auto& values = replay == 1 ? changed : original;
            upload(input.data, values, stream);
            if (!in_place) upload(output.data, sentinel, stream);
            check(cudaGraphLaunch(graph.executable, stream));
            const auto actual = download(destination, original.size(), stream);
            assert_equal(actual, expected_for(config, values, in_place), label + " Graph replay " + std::to_string(replay));
            if (replay == 0) saved_first = actual;
            if (replay == 1 && actual == saved_first)
                throw std::runtime_error("Graph did not consume changed input");
            if (replay == 2) assert_equal(actual, saved_first, label + " Graph restore");
            if (!in_place) assert_equal(download(input.data, original.size(), stream), values, label + " Graph input changed");
        }
        // Graph objects die before the buffers and plan that they reference.
        check(cudaStreamSynchronize(stream));
    }
    std::cout << "PASS " << label << " inplace=" << in_place << " tuned=" << tune
              << " graph=" << graph_test << " rows=" << plan.mapping().rows_per_block
              << " n1_log=" << plan.mapping().n1_log
              << " threads=" << plan.mapping().threads_per_block
              << " second_threads=" << (plan.mapping().second_threads_per_block
                    ? plan.mapping().second_threads_per_block : plan.mapping().threads_per_block)
              << " exchange=" << (plan.mapping().exchange == cuntt::RnsExchange::Block
                                       ? "block" : "dependency-scoped")
              << " physical_boundary=" << plan.mapping().physical_boundary_traversal << '\n';
}

void expanded_mappings(unsigned& cases, cudaStream_t stream) {
    const cuntt::ComputeUnit cores[] = {cuntt::ComputeUnit::Radix2,
                                        cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8};
    for (unsigned variant = 0; variant < 3; ++variant)
        for (bool negacyclic : {false, true})
            for (bool inverse : {false, true})
                for (bool input_reversed : {false, true})
                    for (bool output_reversed : {false, true}) {
                        // The largest tile stays below 48 KiB even with 32 rows.
                        const unsigned bits = variant == 2 ? 13 : 14;
                        const auto config = config_for(bits, negacyclic, inverse,
                                                       input_reversed, output_reversed);
                        cuntt::RnsMapping mapping;
                        mapping.n1_log = inverse ? 6 : bits - 6;
                        mapping.rows_per_block = 8u << variant;
                        mapping.threads_per_block = variant == 0 ? 64 : 256;
                        mapping.compute_unit = cores[variant];
                        mapping.cross_twiddle = inverse ? cuntt::CrossTwiddlePlacement::FirstPass
                                                        : cuntt::CrossTwiddlePlacement::SecondPass;
                        mapping.physical_boundary_traversal = input_reversed || output_reversed;
                        const bool graph_test = negacyclic && input_reversed && output_reversed;
                        run_case(config, cases++, stream, false, graph_test, &mapping, inverse);
                    }
}

cuntt::RnsMapping fused_mapping(unsigned split, unsigned rows, unsigned threads,
                                cuntt::ComputeUnit core = cuntt::ComputeUnit::Radix2) {
    cuntt::RnsMapping mapping;
    mapping.n1_log = split;
    mapping.rows_per_block = rows;
    mapping.threads_per_block = threads;
    mapping.compute_unit = core;
    mapping.cross_twiddle = cuntt::CrossTwiddlePlacement::FirstPass;
    mapping.physical_boundary_traversal = false;
    mapping.core_family = cuntt::RnsCoreFamily::FusedCTGS;
    return mapping;
}

void exchange_cases(unsigned& cases, cudaStream_t stream, bool sanitizer,
                    bool tune_search = false) {
    const unsigned first_case = cases;
    const unsigned bits[] = {12, 13, 14, 12, 12, 13, 14, 12};
    const unsigned splits[] = {9, 7, 9, 10, 9, 7, 9, 10};
    const unsigned short_rows[] = {64, 8, 16, 128, 64, 8, 16, 128};
    // Each variant selects a geometry, core, arithmetic and provider. The
    // log-3/5/7/9 axes exercise the radix4 single-stage remainder.
    for (unsigned variant = 0; variant < (sanitizer ? 2u : 8u); ++variant)
        for (bool inverse : {false, true}) {
            const unsigned order = (variant + unsigned(inverse)) % 4;
            auto config = config_for(sanitizer ? 12 : bits[variant], true, inverse,
                                      (order & 2) != 0, (order & 1) != 0);
            const bool merge = variant >= 4;
            if (merge) {
                const unsigned domains[] = {1, 0, 1, 0};
                const unsigned input_slots[] = {5, 1, 4, 2};
                const unsigned output_slots[] = {2, 4, 1, 5};
                config.polynomials.clear();
                for (unsigned p = 0; p < 4; ++p) {
                    const std::size_t input_offset = std::size_t(input_slots[p]) << config.log_n;
                    const std::size_t output_offset = std::size_t(
                        variant % 2 ? input_slots[p] : output_slots[p]) << config.log_n;
                    config.polynomials.push_back({domains[p], input_offset, output_offset});
                }
            }
            const auto core = !merge && variant % 2 ? cuntt::ComputeUnit::Radix4
                                                   : cuntt::ComputeUnit::Radix2;
            const unsigned rows = sanitizer ? 64 : short_rows[variant];
            auto mapping = fused_mapping(sanitizer ? 9 : splits[variant], inverse ? 1 : rows,
                                          inverse ? 256 : 64, core);
            mapping.second_rows_per_block = inverse ? rows : 1;
            mapping.second_threads_per_block = inverse ? 64 : 256;
            mapping.core_family = merge ? cuntt::RnsCoreFamily::GpuNttMerge
                                        : cuntt::RnsCoreFamily::FusedCTGS;
            mapping.arithmetic = (merge || (sanitizer ? variant % 2 : (variant / 2) % 2))
                ? cuntt::RnsArithmetic::Barrett : cuntt::RnsArithmetic::Shoup;
            const auto placement = (variant + unsigned(inverse)) % 2
                ? Placement::InPlace : Placement::OutOfPlace;
            for (auto exchange : {cuntt::RnsExchange::Block,
                                   cuntt::RnsExchange::DependencyScoped}) {
                if (sanitizer && exchange == cuntt::RnsExchange::Block) continue;
                mapping.exchange = exchange;
                const bool graph = exchange == cuntt::RnsExchange::DependencyScoped &&
                    (sanitizer || variant == 0 || variant == 7);
                run_case(config, cases++, stream, false, graph, &mapping, false, placement);
            }
        }
    const unsigned expected_cases = sanitizer ? 4 : 32;
    if (cases - first_case != expected_cases)
        throw std::runtime_error("exchange case coverage changed");
    if (tune_search) {
        const auto config = config_for(12, true, false, false, true);
        const auto mapping = fused_mapping(9, 64, 64);
        run_case(config, cases++, stream, true, false, &mapping, false,
                 Placement::OutOfPlace, true);
    }
    std::cout << "PASS " << expected_cases << " exact exchange cases; sanitizer="
              << sanitizer << " tune_cases=" << tune_search << '\n';
}

void native_single_warp_cases(unsigned& cases, cudaStream_t stream) {
    const unsigned first_case = cases;
    for (auto family : {cuntt::RnsCoreFamily::FusedCTGS, cuntt::RnsCoreFamily::GpuNttMerge})
        for (bool inverse : {false, true}) {
            auto config = config_for(12, true, inverse, inverse, !inverse);
            const unsigned slots[] = {5, 1, 4, 2};
            config.polynomials.clear();
            for (unsigned p = 0; p < 4; ++p) {
                const std::size_t offset = std::size_t(slots[p]) << config.log_n;
                config.polynomials.push_back({p % 2, offset, offset});
            }
            const bool merge = family == cuntt::RnsCoreFamily::GpuNttMerge;
            auto mapping = fused_mapping(9, inverse ? 1 : 64, 32,
                merge ? cuntt::ComputeUnit::Radix2 : cuntt::ComputeUnit::Radix4);
            mapping.second_rows_per_block = inverse ? 64 : 1;
            mapping.second_threads_per_block = 32;
            mapping.core_family = family;
            mapping.arithmetic = merge ? cuntt::RnsArithmetic::Barrett : cuntt::RnsArithmetic::Shoup;
            mapping.exchange = cuntt::RnsExchange::DependencyScoped;
            run_case(config, cases++, stream, false, true, &mapping, false,
                     Placement::InPlace, false, false, true);
        }
    if (cases - first_case != 4)
        throw std::runtime_error("native single-warp case coverage changed");
    std::cout << "PASS 4 exact native single-warp cases; scratch=0 and Graph A-B-A\n";
}

void single_warp_cases(unsigned& cases, cudaStream_t stream, bool sanitizer,
                       bool tune_search = false) {
    const unsigned first_case = cases;
    for (unsigned variant = 0; variant < (sanitizer ? 3u : 5u); ++variant)
        for (bool inverse : {false, true}) {
            const bool merge = variant == (sanitizer ? 2u : 4u);
            const auto core = variant % 2 && !merge ? cuntt::ComputeUnit::Radix4
                                                    : cuntt::ComputeUnit::Radix2;
            const unsigned order = (variant + unsigned(inverse)) % 4;
            auto config = config_for(12, true, inverse, (order & 2) != 0, (order & 1) != 0);
            if (merge) {
                const unsigned slots[] = {5, 1, 4, 2};
                config.polynomials.clear();
                for (unsigned p = 0; p < 4; ++p) {
                    const std::size_t offset = std::size_t(slots[p]) << config.log_n;
                    config.polynomials.push_back({p % 2, offset, offset});
                }
            }
            // Local logs 3 and 9 include radix4 remainder stages. Opposite
            // pass orientations exercise 32/64 and 64/32 thread ownership.
            auto mapping = fused_mapping(9, inverse ? 1 : 64, 32, core);
            mapping.second_rows_per_block = inverse ? 64 : 1;
            mapping.second_threads_per_block = variant == 0 ? 32 : 64;
            if (inverse && variant != 0) {
                mapping.threads_per_block = 64;
                mapping.second_threads_per_block = 32;
            }
            mapping.arithmetic = (merge || (sanitizer ? variant % 2 : variant / 2))
                ? cuntt::RnsArithmetic::Barrett : cuntt::RnsArithmetic::Shoup;
            mapping.core_family = merge ? cuntt::RnsCoreFamily::GpuNttMerge
                                        : cuntt::RnsCoreFamily::FusedCTGS;
            const auto placement = (variant + unsigned(inverse)) % 2
                ? Placement::InPlace : Placement::OutOfPlace;
            for (auto exchange : {cuntt::RnsExchange::Block,
                                   cuntt::RnsExchange::DependencyScoped}) {
                if (sanitizer && exchange == cuntt::RnsExchange::Block) continue;
                mapping.exchange = exchange;
                run_case(config, cases++, stream, false,
                         exchange == cuntt::RnsExchange::DependencyScoped,
                         &mapping, false, placement);
            }
        }
    native_single_warp_cases(cases, stream);
    const unsigned expected_cases = sanitizer ? 10 : 24;
    if (cases - first_case != expected_cases)
        throw std::runtime_error("single-warp case coverage changed");
    if (tune_search) {
        const auto config = config_for(12, true, false, false, true);
        const auto mapping = fused_mapping(9, 64, 32);
        run_case(config, cases++, stream, true, false, &mapping, false,
                 Placement::OutOfPlace, true, true);
    }
    std::cout << "PASS " << expected_cases << " exact single-warp cases; sanitizer="
              << sanitizer << " tune_cases=" << tune_search << '\n';
}

void equal_offset_cases(unsigned& cases, cudaStream_t stream) {
    const unsigned first_case = cases;
    const unsigned domains[] = {1, 0, 1, 0};
    const unsigned slots[] = {5, 1, 4, 2};
    for (auto family : {cuntt::RnsCoreFamily::FusedCTGS,
                        cuntt::RnsCoreFamily::GpuNttMerge})
        for (bool inverse : {false, true})
            for (bool input_reversed : {false, true})
                for (bool output_reversed : {false, true}) {
                    auto config = config_for(14, true, inverse,
                                             input_reversed, output_reversed);
                    config.polynomials.clear();
                    // Sparse physical rows, permuted logical order and mixed
                    // domains force explicit offsets instead of dense indexing.
                    for (unsigned p = 0; p < 4; ++p) {
                        const std::size_t offset = std::size_t(slots[p]) << 14;
                        config.polynomials.push_back({domains[p], offset, offset});
                    }
                    auto mapping = fused_mapping(9, inverse ? 1 : 16, 256);
                    mapping.second_rows_per_block = inverse ? 16 : 1;
                    mapping.second_threads_per_block = 256;
                    mapping.core_family = family;
                    if (family == cuntt::RnsCoreFamily::GpuNttMerge)
                        mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
                    for (auto placement : {Placement::InPlace, Placement::OutOfPlace})
                        run_case(config, cases++, stream, false, true, &mapping,
                                 false, placement);
                }

    // FusedCTGS also accepts unaligned rows. Both boundaries reverse the
    // native CT/GS order here, exercising the alias-safe fallback in each direction.
    for (bool inverse : {false, true}) {
        auto config = config_for(14, true, inverse, !inverse, inverse);
        for (auto& polynomial : config.polynomials)
            polynomial.output_offset = polynomial.input_offset;
        auto mapping = fused_mapping(9, inverse ? 1 : 16, 256);
        mapping.second_rows_per_block = inverse ? 16 : 1;
        mapping.second_threads_per_block = 256;
        run_case(config, cases++, stream, false, true, &mapping,
                 false, Placement::InPlace);
    }
    if (cases - first_case != 34)
        throw std::runtime_error("equal-offset case coverage changed");
    std::cout << "PASS " << cases - first_case
              << " exact equal-offset cases; both placements and all boundary orders\n";
}

void fused_cases(unsigned& cases, cudaStream_t stream,
                 const cuntt::RnsDomain& wide_domain, cuntt::ComputeUnit core) {
    const unsigned first_case = cases;
    unsigned variant = 0;
    for (bool inverse : {false, true})
        for (bool input_reversed : {false, true})
            for (bool output_reversed : {false, true}) {
                const auto config = config_for(12, true, inverse,
                                               input_reversed, output_reversed);
                const unsigned threads[] = {64, 128, 256};
                const auto mapping = fused_mapping(6, 1u << (variant % 4),
                                                   threads[variant % 3], core);
                run_case(config, cases++, stream, false,
                         input_reversed && output_reversed, &mapping, inverse);
                ++variant;
            }

    // Both orientations of uneven splits; large row tiles use short axes so
    // these tests do not require opt-in shared memory on the target device.
    const unsigned bits[] = {13, 13, 14, 14, 15, 15};
    const unsigned splits[] = {6, 7, 6, 8, 6, 9};
    const unsigned rows[] = {16, 32, 1, 2, 1, 2};
    const unsigned threads[] = {64, 256, 128, 64, 256, 128};
    for (unsigned v = 0; v < 6; ++v) {
        const auto config = config_for(bits[v], true, v % 2 != 0,
                                       v % 3 != 0, v % 3 != 1);
        const auto mapping = fused_mapping(splits[v], rows[v], threads[v], core);
        run_case(config, cases++, stream, false, v == 1, &mapping);
    }

    // Include both directions at each upper-size edge, including local log 10.
    for (unsigned log_n : {19u, 20u})
        for (bool inverse : {false, true}) {
            const auto config = config_for(log_n, true, inverse, inverse, !inverse);
            const auto mapping = fused_mapping(log_n == 19 && !inverse ? 9 : 10,
                                               inverse ? 2 : 1, inverse ? 256 : 128, core);
            run_case(config, cases++, stream, false, false, &mapping);
        }

    for (bool inverse : {false, true}) {
        auto config = config_for(12, true, inverse, !inverse, inverse);
        config.domains[0] = wide_domain;
        const auto mapping = fused_mapping(6, inverse ? 32 : 16, inverse ? 64 : 256, core);
        run_case(config, cases++, stream, false, true, &mapping);
    }
    if (cases - first_case != 20) throw std::runtime_error("fused case coverage changed");
    std::cout << "PASS " << cases - first_case << " exact FusedCTGS cases radix="
              << (core == cuntt::ComputeUnit::Radix4 ? 4 : 2) << '\n';
}

void asymmetric_fused_cases(unsigned& cases, cudaStream_t stream) {
    const unsigned first_case = cases;
    const unsigned bits[] = {12, 14, 12};
    const unsigned splits[] = {9, 9, 10};
    const unsigned short_axis_rows[] = {64, 16, 128};
    for (auto core : {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4})
        for (bool inverse : {false, true})
            for (unsigned shape = 0; shape < 3; ++shape) {
                // The log-3/log-9 shape covers all boundary orders. The other
                // shapes add log-5 and log-2 prefixes with independent passes.
                const unsigned orders = shape == 0 ? 4 : 1;
                for (unsigned order = 0; order < orders; ++order) {
                    const bool input_reversed = shape == 0 ? (order & 2) != 0 : inverse;
                    const bool output_reversed = shape == 0 ? (order & 1) != 0 : !inverse;
                    const auto config = config_for(bits[shape], true, inverse,
                                                   input_reversed, output_reversed);
                    auto mapping = fused_mapping(splits[shape],
                                                  inverse ? 1 : short_axis_rows[shape],
                                                  inverse ? 256 : 64, core);
                    mapping.second_rows_per_block = inverse ? short_axis_rows[shape] : 1;
                    mapping.second_threads_per_block = inverse ? 64 : 256;
                    run_case(config, cases++, stream, false,
                             (input_reversed && output_reversed) || shape == 2,
                             &mapping, inverse);
                }
            }
    if (cases - first_case != 24)
        throw std::runtime_error("asymmetric fused case coverage changed");
    std::cout << "PASS " << cases - first_case << " exact asymmetric FusedCTGS cases\n";
}

template<class Function> void rejects(Function&& function, const char* label) {
    try { function(); }
    catch (const std::invalid_argument&) { std::cout << "PASS reject " << label << '\n'; return; }
    throw std::runtime_error(std::string("invalid request accepted: ") + label);
}

void invalid_fused_requests() {
    const auto config = config_for(12, true, false, true, true);
    cuntt::RnsPlan plan(config);
    auto mapping = fused_mapping(6, 1, 128);
    mapping.compute_unit = cuntt::ComputeUnit::Radix8;
    rejects([&] { plan.set_mapping(mapping); }, "FusedCTGS radix8 core");
    mapping = fused_mapping(6, 1, 128);
    mapping.cross_twiddle = cuntt::CrossTwiddlePlacement::SecondPass;
    rejects([&] { plan.set_mapping(mapping); }, "FusedCTGS second-pass cross twiddle");
    mapping = fused_mapping(6, 1, 128);
    mapping.physical_boundary_traversal = true;
    rejects([&] { plan.set_mapping(mapping); }, "FusedCTGS physical boundary traversal");
    const auto cyclic = config_for(12, false, false, true, true);
    cuntt::RnsPlan cyclic_plan(cyclic);
    mapping = fused_mapping(6, 1, 128);
    rejects([&] { cyclic_plan.set_mapping(mapping); }, "FusedCTGS cyclic ring");

    mapping = fused_mapping(6, 1, 128);
    mapping.second_rows_per_block = 3;
    rejects([&] { plan.set_mapping(mapping); }, "invalid second-pass rows");
    mapping = fused_mapping(6, 1, 128);
    mapping.second_threads_per_block = 96;
    rejects([&] { plan.set_mapping(mapping); }, "invalid second-pass threads");

    // The log-9 pass has only eight independent rows at N=4096. Reject
    // oversized groups before integer grid division can produce zero blocks.
    mapping = fused_mapping(9, 64, 64);
    mapping.second_rows_per_block = 16;
    rejects([&] { plan.set_mapping(mapping); }, "second-pass rows exceed independent lines");
    const auto inverse_config = config_for(12, true, true, true, true);
    cuntt::RnsPlan inverse_plan(inverse_config);
    mapping = fused_mapping(9, 16, 128);
    mapping.second_rows_per_block = 64;
    rejects([&] { inverse_plan.set_mapping(mapping); }, "first-pass rows exceed independent lines");

    mapping = fused_mapping(9, 1, 128);
    mapping.core_family = cuntt::RnsCoreFamily::Hybrid2D;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D short local log");
    mapping = fused_mapping(6, 64, 128);
    mapping.core_family = cuntt::RnsCoreFamily::Hybrid2D;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D first-pass rows64");
    mapping.rows_per_block = 1;
    mapping.second_rows_per_block = 64;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D second-pass rows64");

    mapping = fused_mapping(6, 1, 128);
    mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
    plan.set_mapping(mapping);
    if (plan.mapping().arithmetic != cuntt::RnsArithmetic::Barrett)
        throw std::runtime_error("GPU-NTT-compatible Barrett mapping was not retained");
}

void invalid_exchange_requests() {
    auto config = config_for(12, true, false, false, true);
    for (unsigned p = 0; p < config.polynomials.size(); ++p) {
        config.polynomials[p].input_offset = std::size_t(p) << config.log_n;
        config.polynomials[p].output_offset = std::size_t(p) << config.log_n;
    }
    cuntt::RnsPlan plan(config);
    auto mapping = fused_mapping(6, 1, 128);
    mapping.core_family = cuntt::RnsCoreFamily::Hybrid2D;
    mapping.exchange = cuntt::RnsExchange::DependencyScoped;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D DependencyScoped exchange");
    for (auto family : {cuntt::RnsCoreFamily::Hybrid2D, cuntt::RnsCoreFamily::FusedCTGS,
                        cuntt::RnsCoreFamily::GpuNttMerge}) {
        mapping = fused_mapping(6, 1, 128);
        mapping.core_family = family;
        if (family == cuntt::RnsCoreFamily::GpuNttMerge)
            mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
        mapping.exchange = static_cast<cuntt::RnsExchange>(99);
        rejects([&] { plan.set_mapping(mapping); }, "invalid exchange enum");
    }
}

void invalid_single_warp_requests() {
    const auto config = config_for(12, true, false, false, true);
    cuntt::RnsPlan plan(config);
    auto mapping = fused_mapping(6, 1, 32);
    mapping.core_family = cuntt::RnsCoreFamily::Hybrid2D;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D first-pass threads32");
    mapping.threads_per_block = 64;
    mapping.second_threads_per_block = 32;
    rejects([&] { plan.set_mapping(mapping); }, "Hybrid2D second-pass threads32");
}

void invalid_requests() {
    const auto base = config_for(12, true, false, false, false);
    auto config = base;
    // A composite admitting an exact primitive 8192-th root: a root-order-only
    // validator must not accidentally accept this domain as prime.
    constexpr Word composite = 65537ULL * 65537ULL;
    Word composite_root = 0;
    for (Word g = 2; g < 1000 && !composite_root; ++g) {
        const Word candidate = power(g, (65537ULL * 65536ULL) / 8192, composite);
        if (power(candidate, 4096, composite) == composite - 1 && power(candidate, 8192, composite) == 1)
            composite_root = candidate;
    }
    if (!composite_root) throw std::runtime_error("composite root fixture failed");
    config.domains[0] = {composite, composite_root};
    rejects([&] { cuntt::RnsPlan plan(config); }, "composite modulus with valid root order");
    config = base;
    config.domains[0].root = 1;
    rejects([&] { cuntt::RnsPlan plan(config); }, "nonprimitive root");
    config = base;
    config.domains[0].root = 0;
    rejects([&] { cuntt::RnsPlan plan(config); }, "zero root");
    config = base;
    config.polynomials[1].output_offset = config.polynomials[0].output_offset + 2048;
    rejects([&] { cuntt::RnsPlan plan(config); }, "overlapping output intervals");
    config = base;
    config.input_order = static_cast<cuntt::RnsOrder>(99);
    rejects([&] { cuntt::RnsPlan plan(config); }, "invalid input order");
    config = base;
    config.output_order = static_cast<cuntt::RnsOrder>(99);
    rejects([&] { cuntt::RnsPlan plan(config); }, "invalid output order");

    cuntt::RnsPlan natural_plan(base);
    rejects([&] { natural_plan.tune(static_cast<cuntt::RnsPlacement>(99)); },
            "invalid tune placement");
    auto mapping = natural_plan.mapping();
    mapping.physical_boundary_traversal = true;
    rejects([&] { natural_plan.set_mapping(mapping); }, "physical traversal with natural boundaries");

    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    const std::size_t oversized_shared = 32 * 1025 * sizeof(Word);
    if (oversized_shared > properties.sharedMemPerBlock) {
        const auto large_config = config_for(20, true, false, false, true);
        cuntt::RnsPlan large_plan(large_config);
        mapping = large_plan.mapping();
        mapping.n1_log = 10;
        mapping.rows_per_block = 32;
        rejects([&] { large_plan.set_mapping(mapping); }, "rows exceed device shared memory");
    } else {
        std::cout << "SKIP oversized shared mapping: all supported tiles fit this device\n";
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool exchange_only = false, exchange_sanitizer = false;
        bool single_warp_only = false, single_warp_sanitizer = false;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string option(argv[argument]);
            if (option == "--exchange-only") exchange_only = true;
            else if (option == "--exchange-sanitizer") exchange_sanitizer = true;
            else if (option == "--single-warp-only") single_warp_only = true;
            else if (option == "--single-warp-sanitizer") single_warp_sanitizer = true;
            else throw std::invalid_argument("unknown test option: " + option);
        }
        Stream stream;
        unsigned cases = 0;
        if (single_warp_only || single_warp_sanitizer) {
            single_warp_cases(cases, stream.value, single_warp_sanitizer, !single_warp_sanitizer);
            invalid_single_warp_requests();
            std::cout << "PASS " << cases << " exact RNS single-warp cases and 2 rejection checks\n";
            return 0;
        }
        if (exchange_only || exchange_sanitizer) {
            exchange_cases(cases, stream.value, exchange_sanitizer, !exchange_sanitizer);
            invalid_exchange_requests();
            std::cout << "PASS " << cases << " exact RNS exchange cases and 4 rejection checks\n";
            return 0;
        }
        for (unsigned bits : {12u, 14u})
            for (bool negacyclic : {false, true})
                for (bool inverse : {false, true})
                    for (bool input_reversed : {false, true})
                        for (bool output_reversed : {false, true}) {
                            const auto config = config_for(bits, negacyclic, inverse,
                                                           input_reversed, output_reversed);
                            const bool tune = cases == 0 ||
                                (bits == 12 && negacyclic && !inverse &&
                                 !input_reversed && output_reversed);
                            run_case(config, cases, stream.value, tune, cases == 1);
                            ++cases;
                        }
        run_case(config_for(16, true, true, true, false), cases++, stream.value, false, false);
        run_case(config_for(18, false, false, false, true), cases++, stream.value, false, false);
        run_case(config_for(20, true, true, true, false), cases++, stream.value, false, false);
        auto wide_modulus = config_for(12, true, true, true, false);
        Word prime = (Word{1} << 62) + 1;
        unsigned attempts = 0;
        while (!cuntt::is_prime(prime)) {
            if (++attempts == 4096) throw std::runtime_error("large prime fixture search exhausted");
            prime += 8192;
        }
        const Word primitive = cuntt::find_primitive_power_of_two_root(13, prime);
        const Word root = power(primitive, 7, prime);
        if (power(root, 8192, prime) != 1 || power(root, 4096, prime) != prime - 1)
            throw std::runtime_error("large prime root fixture is not primitive");
        wide_modulus.domains[0] = {prime, root};
        std::cout << "large_modulus=" << prime << '\n';
        run_case(wide_modulus, cases++, stream.value, false, false);
        auto barrett_config = config_for(14, true, false, false, true);
        auto barrett_mapping = fused_mapping(9, 64, 256, cuntt::ComputeUnit::Radix2);
        barrett_mapping.second_rows_per_block = 1;
        barrett_mapping.second_threads_per_block = 256;
        barrett_mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
        run_case(barrett_config, cases++, stream.value, false, true, &barrett_mapping);
        auto barrett_inverse_config = config_for(14, true, true, true, false);
        auto barrett_inverse_mapping = fused_mapping(9, 1, 256, cuntt::ComputeUnit::Radix2);
        barrett_inverse_mapping.second_rows_per_block = 16;
        barrett_inverse_mapping.second_threads_per_block = 256;
        barrett_inverse_mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
        run_case(barrett_inverse_config, cases++, stream.value, false, true, &barrett_inverse_mapping);
        auto gpuntt_config = config_for(14, true, false, false, true);
        gpuntt_config.polynomials.clear();
        for (unsigned p = 0; p < 4; ++p)
            gpuntt_config.polynomials.push_back({p / 2, std::size_t(p) << 14, std::size_t(p) << 14});
        auto gpuntt_mapping = fused_mapping(9, 16, 256, cuntt::ComputeUnit::Radix2);
        gpuntt_mapping.second_rows_per_block = 1;
        gpuntt_mapping.second_threads_per_block = 256;
        gpuntt_mapping.core_family = cuntt::RnsCoreFamily::GpuNttMerge;
        gpuntt_mapping.arithmetic = cuntt::RnsArithmetic::Barrett;
        run_case(gpuntt_config, cases++, stream.value, true, true, &gpuntt_mapping,
                 false, Placement::InPlace, true);
        exchange_cases(cases, stream.value, false);
        single_warp_cases(cases, stream.value, false);
        equal_offset_cases(cases, stream.value);
        expanded_mappings(cases, stream.value);
        fused_cases(cases, stream.value, wide_modulus.domains[0], cuntt::ComputeUnit::Radix2);
        fused_cases(cases, stream.value, wide_modulus.domains[0], cuntt::ComputeUnit::Radix4);
        asymmetric_fused_cases(cases, stream.value);
        invalid_requests();
        invalid_fused_requests();
        invalid_exchange_requests();
        invalid_single_warp_requests();
        std::cout << "PASS " << cases << " exact RNS cases; sentinel, mixed-domain, mapping, tune, Graph and rejection checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
