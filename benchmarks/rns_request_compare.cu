// Same-request provider diagnostic, restricted to dense equal-offset arenas.
// Link cuntt, pinned GPU-NTT device code, shared cudart and dl; use -rdc=true.
#include <cuntt/rns_tuning.hpp>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"
#include "../paper/experiments/gpu_thermal_guard.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
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
    std::string request, cache_directory, output, exchange_override;
    std::string native_dispatch = "modulus-ordered";
    std::string cache_regime = "warm";
    std::string placement = "in-place";
    cuntt::RnsCachePolicy policy = cuntt::RnsCachePolicy::Prefer;
    unsigned repetitions = 20, tune_repetitions = 5, rounds = 10;
    bool thermal_guard = false;
};
unsigned number(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("expected unsigned integer: " + text);
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("integer overflow");
    return static_cast<unsigned>(value);
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--thermal-guard") { o.thermal_guard = true; continue; }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--request") o.request = value;
        else if (key == "--cache-dir") o.cache_directory = value;
        else if (key == "--output") o.output = value;
        else if (key == "--repetitions") o.repetitions = number(value);
        else if (key == "--tune-repetitions") o.tune_repetitions = number(value);
        else if (key == "--rounds") o.rounds = number(value);
        else if (key == "--placement") {
            if (value != "in-place" && value != "separate")
                throw std::invalid_argument("placement must be in-place or separate");
            o.placement = value;
        }
        else if (key == "--cache-regime") {
            if (value != "warm" && value != "l2-displaced")
                throw std::invalid_argument("cache-regime must be warm or l2-displaced");
            o.cache_regime = value;
        }
        else if (key == "--native-dispatch") {
            if (value != "modulus-ordered" && value != "contiguous")
                throw std::invalid_argument("native-dispatch must be modulus-ordered or contiguous");
            o.native_dispatch = value;
        }
        else if (key == "--exchange-override") {
            if (value != "block" && value != "dependency_scoped")
                throw std::invalid_argument("exchange-override must be block or dependency_scoped");
            o.exchange_override = value;
        }
        else if (key == "--cache-policy") {
            if (value == "prefer") o.policy = cuntt::RnsCachePolicy::Prefer;
            else if (value == "require") o.policy = cuntt::RnsCachePolicy::Require;
            else if (value == "refresh") o.policy = cuntt::RnsCachePolicy::Refresh;
            else throw std::invalid_argument("cache-policy must be prefer, require or refresh");
        } else throw std::invalid_argument("unknown option " + key);
    }
    if (o.request.empty() || o.cache_directory.empty() || o.output.empty() ||
        !o.repetitions || o.repetitions > 1000 ||
        !o.tune_repetitions || o.tune_repetitions > 1000 || !o.rounds)
        throw std::invalid_argument("required: --request JSON --cache-dir PATH --output CSV; repetitions and tune-repetitions 1..1000, rounds > 0");
    if (o.cache_regime == "l2-displaced" && o.repetitions != 1)
        throw std::invalid_argument("l2-displaced requires --repetitions 1");
    return o;
}
std::string read_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open request: " + path);
    std::ostringstream text;
    text << input.rdbuf();
    if (input.bad()) throw std::runtime_error("cannot read request: " + path);
    return text.str();
}
std::string csv(const std::string& value) {
    std::string result = "\"";
    for (char c : value) { if (c == '"') result += '"'; result += c; }
    return result + '"';
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(std::size_t count) {
        if (!count || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("invalid device extent");
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~Device() { if (data) cudaFree(data); }
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
    ~Event() { if (value) cudaEventDestroy(value); }
};

// Touch only the dedicated diagnostic allocation, never request data or roots.
__global__ void displace_l2(volatile Word* scratch, std::size_t words) {
    for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < words; index += std::size_t(gridDim.x) * blockDim.x)
        scratch[index] = scratch[index] + Word(index) + 1;
}

// Preserve logical polynomial ordering; only the native order table is indexed
// by physical row. No packing, reordering kernels or boundary copies are added.
std::size_t validate_layout(const cuntt::RnsPlanConfig& c) {
    if (c.log_n < 12 || c.log_n > 20 || c.ring != cuntt::RnsRing::Negacyclic ||
        c.input_order != (c.inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural) ||
        c.output_order != (c.inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed) ||
        c.domains.empty() || c.polynomials.empty())
        throw std::invalid_argument("diagnostic requires logN 12..20, negacyclic canonical u64 and native CT/GS boundaries");
    const std::size_t n = std::size_t{1} << c.log_n, count = c.polynomials.size();
    if (count > std::min<std::size_t>(65535, std::numeric_limits<int>::max() / n) ||
        c.domains.size() > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("request exceeds pinned native grid/address range");
    std::vector<bool> rows(count, false);
    for (const auto& p : c.polynomials) {
        if (p.domain >= c.domains.size() || p.input_offset != p.output_offset ||
            p.input_offset % n || p.input_offset / n >= count || rows[p.input_offset / n])
            throw std::invalid_argument("unsupported layout: requires equal input/output offsets and exactly the dense N-aligned physical rows 0..P-1; logical polynomial order may be permuted");
        rows[p.input_offset / n] = true;
    }
    for (const auto& d : c.domains)
        if (d.modulus < 3 || d.modulus >= (Word{1} << 62) || (d.modulus - 1) % (2 * n) ||
            d.root >= d.modulus || power(d.root, n, d.modulus) != d.modulus - 1 ||
            power(d.root, 2 * n, d.modulus) != 1)
            throw std::invalid_argument("native provider requires q < 2^62 and canonical primitive forward 2N root");
    return count * n;
}
std::vector<Word> input_for(const cuntt::RnsPlanConfig& c, std::size_t words, Word seed) {
    std::vector<Word> input(words);
    std::mt19937_64 random(seed);
    const std::size_t n = std::size_t{1} << c.log_n;
    for (const auto& p : c.polynomials) {
        const Word q = c.domains[p.domain].modulus;
        for (std::size_t j = 0; j < n; ++j) input[p.input_offset + j] = random() % q;
        input[p.input_offset] = 0;
        input[p.input_offset + 1] = q - 1;
    }
    return input;
}
// Independent CPU DIT reference: twist and boundary permutations are explicit.
std::vector<Word> reference(const std::vector<Word>& input, const cuntt::RnsPlanConfig& c) {
    const std::size_t n = std::size_t{1} << c.log_n;
    std::vector<Word> result(input.size()), values(n);
    for (const auto& p : c.polynomials) {
        const auto& d = c.domains[p.domain];
        const Word q = d.modulus, psi = d.root;
        Word twist = 1;
        for (std::size_t j = 0; j < n; ++j) {
            values[j] = input[p.input_offset + (c.inverse ? reverse_bits(j, c.log_n) : j)];
            if (!c.inverse) values[j] = multiply(values[j], twist, q);
            twist = multiply(twist, psi, q);
        }
        Word omega = multiply(psi, psi, q);
        if (c.inverse) omega = power(omega, q - 2, q);
        for (std::size_t j = 0; j < n; ++j) {
            const auto r = reverse_bits(j, c.log_n);
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
        Word factor = c.inverse ? power(n, q - 2, q) : 1;
        const Word step = c.inverse ? power(psi, q - 2, q) : 1;
        for (std::size_t j = 0; j < n; ++j) {
            result[p.output_offset + (c.inverse ? j : reverse_bits(j, c.log_n))] = multiply(values[j], factor, q);
            factor = multiply(factor, step, q);
        }
    }
    return result;
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const auto mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}
void run(const Options& o) {
    if (std::filesystem::exists(o.output))
        throw std::runtime_error("refusing to overwrite benchmark output: " + o.output);
    const auto config = cuntt::rns_config_from_json(read_file(o.request));
    const bool separate = o.placement == "separate";
    const auto placement = separate ? cuntt::RnsPlacement::Separate : cuntt::RnsPlacement::InPlace;
    const auto words = validate_layout(config);
    const std::size_t n = std::size_t{1} << config.log_n, count = config.polynomials.size();
    if (o.native_dispatch == "contiguous") {
        if (config.domains.size() > count)
            throw std::invalid_argument("contiguous native dispatch requires domains <= polynomials");
        for (const auto& p : config.polynomials)
            if (p.domain != (p.input_offset / n) % config.domains.size())
                throw std::invalid_argument("contiguous native dispatch requires physical row domain = row % domain_count");
    }
    Stream stream;
    int device = 0;
    check(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    std::size_t displacement_bytes = 0;
    std::unique_ptr<Device<Word>> displacement;
    const std::string cache_scope = o.cache_regime == "l2-displaced"
        ? "artificial_l2_displacement_not_application_cache_replay" : "warm_repeated_graph";
    if (o.cache_regime == "l2-displaced") {
        constexpr std::size_t maximum_displacement_bytes = std::size_t{1} << 30;
        if (properties.l2CacheSize <= 0 ||
            std::size_t(properties.l2CacheSize) > maximum_displacement_bytes / 2)
            throw std::invalid_argument("l2-displaced requires known positive L2 capacity and at most 1 GiB scratch");
        displacement_bytes = ((2 * std::size_t(properties.l2CacheSize) + sizeof(Word) - 1)
                              / sizeof(Word)) * sizeof(Word);
        if (displacement_bytes > properties.totalGlobalMem / 4)
            throw std::invalid_argument("l2-displaced scratch exceeds one quarter of device memory");
        displacement = std::make_unique<Device<Word>>(displacement_bytes / sizeof(Word));
        check(cudaMemsetAsync(displacement->data, 0, displacement_bytes, stream.value));
        check(cudaStreamSynchronize(stream.value));
    }
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    for (unsigned char byte : properties.uuid.bytes) uuid << std::setw(2) << unsigned(byte);
    std::cerr << "device,uuid=" << uuid.str() << ",name=" << properties.name
              << ",memory_bytes=" << properties.totalGlobalMem << '\n';
    // Monitor every sample even when cooling/throttle rejection is not requested.
    GpuThermalGuard thermal(true);
    cuntt::RnsPlan plan(config);
    cuntt::RnsPrepareOptions preparation_options;
    preparation_options.cache_directory = o.cache_directory;
    preparation_options.policy = o.policy;
    preparation_options.repetitions = o.tune_repetitions;
    if (o.thermal_guard) thermal.ready();
    const auto prepared = cuntt::prepare_rns_plan(plan, placement,
                                                 preparation_options, stream.value);
    std::cerr << "prepared,cache_hit=" << prepared.cache_hit << ",key=" << prepared.cache_key
              << ",source=" << prepared.source << ",kernel_ms=" << prepared.kernel_ms
              << ",mapping=" << prepared.mapping_json << '\n';
    // Apply the component control after preparation; never publish it to cache.
    if (!o.exchange_override.empty()) {
        auto controlled = plan.mapping();
        controlled.exchange = o.exchange_override == "block" ? cuntt::RnsExchange::Block
                                                              : cuntt::RnsExchange::DependencyScoped;
        plan.set_mapping(controlled, placement);
    }
    const auto actual_mapping = cuntt::rns_mapping_json(plan.mapping());
    const bool component_control = !o.exchange_override.empty() || o.cache_regime == "l2-displaced";
    const std::string selection_kind = component_control ? "component_control" : "automatic_library";
    std::cerr << "cache_regime,value=" << o.cache_regime << ",l2_cache_bytes=" << properties.l2CacheSize
              << ",cache_displacement_bytes=" << displacement_bytes << ",scope=" << cache_scope << '\n';
    std::cerr << "selection,kind=" << selection_kind << ",exchange_override=" << o.exchange_override
              << ",mapping=" << actual_mapping << ",prepared_mapping=" << prepared.mapping_json << '\n';

    std::vector<Root<Word>> host_roots(config.domains.size() * n);
    std::vector<Modulus<Word>> host_moduli;
    std::vector<Word> host_inverses;
    std::vector<int> host_order(count);
    for (std::size_t d = 0; d < config.domains.size(); ++d) {
        const Word q = config.domains[d].modulus;
        const Word root = config.inverse ? power(config.domains[d].root, q - 2, q) : config.domains[d].root;
        Word w = 1;
        for (std::size_t j = 0; j < n; ++j, w = multiply(w, root, q))
            host_roots[d * n + reverse_bits(j, config.log_n)] = w;
        host_moduli.emplace_back(q);
        host_inverses.push_back(power(n, q - 2, q));
    }
    for (const auto& p : config.polynomials) host_order[p.input_offset / n] = static_cast<int>(p.domain);
    Device<Root<Word>> roots(host_roots.size());
    Device<Modulus<Word>> moduli(host_moduli.size());
    Device<Word> inverses(host_inverses.size()), arena(words);
    std::unique_ptr<Device<Word>> separate_output;
    if (separate) separate_output = std::make_unique<Device<Word>>(words);
    Word* const output_data = separate ? separate_output->data : arena.data;
    Device<int> order(host_order.size());
    check(cudaMemcpy(roots.data, host_roots.data(), host_roots.size() * sizeof(Root<Word>), cudaMemcpyHostToDevice));
    check(cudaMemcpy(moduli.data, host_moduli.data(), host_moduli.size() * sizeof(Modulus<Word>), cudaMemcpyHostToDevice));
    check(cudaMemcpy(inverses.data, host_inverses.data(), host_inverses.size() * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemcpy(order.data, host_order.data(), host_order.size() * sizeof(int), cudaMemcpyHostToDevice));
    gpuntt::ntt_rns_configuration<Word> native{};
    native.n_power = config.log_n;
    native.ntt_type = config.inverse ? gpuntt::INVERSE : gpuntt::FORWARD;
    native.ntt_layout = gpuntt::PerPolynomial;
    native.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
    native.mod_inverse = inverses.data;
    native.stream = stream.value;
    const auto native_parameters = config.inverse ? gpuntt::CreateInverseNTTKernel<Word>()
                                                  : gpuntt::CreateForwardNTTKernel<Word>();
    std::ostringstream native_mapping;
    const std::string native_api = (o.native_dispatch == "modulus-ordered"
        ? "GPU_NTT_Modulus_Ordered" : config.inverse ? "GPU_INTT" : "GPU_NTT") +
        std::string(separate ? "" : "_Inplace");
    const auto native_mod_count = o.native_dispatch == "modulus-ordered" ? count : config.domains.size();
    const std::string native_description = native_api + ";PerPolynomial;X_N_plus;batch_size=" +
        std::to_string(count) + ";mod_count=" + std::to_string(native_mod_count);
    native_mapping << native_description;
    std::cerr << "native_dispatch,value=" << o.native_dispatch << ",description=" << native_description
              << ",scope=semantic_equivalent_not_original_application_dispatch\n";
    for (const auto& p : native_parameters.at(config.log_n))
        native_mapping << ";grid=" << p.griddim_x << ':' << p.griddim_y << ':' << count
                       << ";block=" << p.blockdim_x << ':' << p.blockdim_y
                       << ";shared=" << p.shared_memory << ";shared_index=" << p.shared_index
                       << ";logm=" << p.logm << ";k=" << p.k
                       << ";stages=" << p.outer_iteration_count << ";not_last=" << p.not_last_kernel;
    auto launch = [&](unsigned backend) {
        if (backend == 0) plan.execute_async(arena.data, output_data, stream.value);
        else if (separate && o.native_dispatch == "modulus-ordered")
            gpuntt::GPU_NTT_Modulus_Ordered(arena.data, output_data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(count), order.data);
        else if (separate && config.inverse)
            gpuntt::GPU_INTT<Word>(arena.data, output_data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(native_mod_count));
        else if (separate)
            gpuntt::GPU_NTT<Word>(arena.data, output_data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(native_mod_count));
        else if (o.native_dispatch == "modulus-ordered")
            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(arena.data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(count), order.data);
        else if (config.inverse)
            gpuntt::GPU_INTT_Inplace(arena.data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(native_mod_count));
        else gpuntt::GPU_NTT_Inplace(arena.data, roots.data, moduli.data,
                native, static_cast<int>(count), static_cast<int>(native_mod_count));
    };
    std::array<Graph, 2> single, repeated;
    auto capture = [&](Graph& g, unsigned backend, unsigned repeats) {
        check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        for (unsigned repeat = 0; repeat < repeats; ++repeat) launch(backend);
        check(cudaStreamEndCapture(stream.value, &g.graph));
        check(cudaGraphInstantiate(&g.executable, g.graph, nullptr, nullptr, 0));
    };
    for (unsigned backend = 0; backend < 2; ++backend) {
        capture(single[backend], backend, 1);
        capture(repeated[backend], backend, o.repetitions);
    }
    auto reset = [&](const std::vector<Word>& input) {
        check(cudaMemcpyAsync(arena.data, input.data(), words * sizeof(Word), cudaMemcpyHostToDevice, stream.value));
        check(cudaStreamSynchronize(stream.value));
    };
    const auto original = input_for(config, words, 0x20261021ULL);
    const auto second = input_for(config, words, 0x20261022ULL);
    std::array<std::vector<Word>, 2> expected{reference(original, config), reference(second, config)};
    constexpr const char* names[] = {"cubutterfly", "native-gpuntt"};
    for (unsigned trial = 0; trial < 3; ++trial) {
        const bool b = trial == 1;
        const auto& input = b ? second : original;
        for (unsigned backend = 0; backend < 2; ++backend) {
            reset(input);
            if (separate) check(cudaMemsetAsync(output_data, 0xff, words * sizeof(Word), stream.value));
            check(cudaGraphLaunch(single[backend].executable, stream.value));
            std::vector<Word> actual(words);
            check(cudaMemcpyAsync(actual.data(), output_data, words * sizeof(Word), cudaMemcpyDeviceToHost, stream.value));
            check(cudaStreamSynchronize(stream.value));
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected[b].begin());
            if (mismatch.first != actual.end()) {
                const auto index = static_cast<std::size_t>(mismatch.first - actual.begin());
                throw std::runtime_error(std::string(names[backend]) + " CPU exact mismatch trial=" +
                    std::to_string(trial) + " word=" + std::to_string(index) + " actual=" +
                    std::to_string(actual[index]) + " expected=" + std::to_string(expected[b][index]));
            }
            if (separate) {
                std::vector<Word> preserved(words);
                check(cudaMemcpyAsync(preserved.data(), arena.data, words * sizeof(Word), cudaMemcpyDeviceToHost, stream.value));
                check(cudaStreamSynchronize(stream.value));
                if (preserved != input)
                    throw std::runtime_error(std::string(names[backend]) + " separate input modified trial=" +
                                             std::to_string(trial));
            }
            std::cerr << "validation,backend=" << names[backend] << ",trial=" << trial
                      << ",exact=1,input_preserved=" << (separate ? "1" : "not_applicable") << '\n';
        }
    }
    std::ofstream output(o.output);
    if (!output) throw std::runtime_error("cannot open output: " + o.output);
    output << std::setprecision(12)
           << "record,backend,round,order,kernel_ms,repetitions,tune_repetitions,rounds,log_n,polynomials,domains,inverse,placement,cache_hit,key,source,mapping,exact,gpu_uuid,gpu_name,gpu_memory_bytes,request,selection_kind,exchange_override,prepared_mapping,prepared_source,native_dispatch,native_dispatch_description,cache_regime,l2_cache_bytes,cache_displacement_bytes,cache_scope,measurement_kind,input_preserved,validation_trials,repeated_input_semantics\n";
    auto emit = [&](const char* record, unsigned backend, int round, int slot, double ms) {
        output << record << ',' << names[backend] << ',' << round << ',' << slot << ',' << ms << ','
               << o.repetitions << ',' << o.tune_repetitions << ',' << o.rounds << ',' << config.log_n << ',' << count << ','
               << config.domains.size() << ',' << config.inverse << ',' << o.placement << ',' << (backend == 0 && prepared.cache_hit) << ','
               << csv(prepared.cache_key) << ',' << csv(backend == 0 ?
                   (component_control ? "component_control" : prepared.source) : "pinned-native") << ','
               << csv(backend == 0 ? actual_mapping : native_mapping.str()) << ",1," << csv(uuid.str()) << ','
               << csv(properties.name) << ',' << properties.totalGlobalMem << ',' << csv(o.request) << ','
               << csv(backend == 0 ? selection_kind : "pinned-native") << ',' << csv(o.exchange_override) << ','
               << csv(prepared.mapping_json) << ',' << csv(prepared.source) << ','
               << csv(o.native_dispatch) << ',' << csv(native_description) << ','
               << csv(o.cache_regime) << ',' << properties.l2CacheSize << ',' << displacement_bytes << ','
               << csv(cache_scope) << ','
               << csv(component_control || o.native_dispatch == "contiguous"
                      ? "component_control" : "real_request_operator_diagnostic") << ','
               << (separate ? "1" : "not_applicable") << ",3,"
               << (separate ? "same_preserved_input" : "chained_in_place_transforms") << '\n';
        output.flush();
        if (!output) throw std::runtime_error("cannot write output: " + o.output);
    };
    Event start, stop;
    std::array<std::vector<double>, 2> samples;
    for (unsigned round = 0; round < o.rounds; ++round) {
        for (unsigned slot = 0; slot < 2; ++slot) {
            const unsigned backend = (round + slot) % 2;
            if (o.thermal_guard) thermal.ready();
            // Upload/first Graph launch and input restoration are outside events.
            reset(original);
            check(cudaGraphLaunch(repeated[backend].executable, stream.value));
            check(cudaStreamSynchronize(stream.value));
            reset(original);
            const std::string label = std::string(names[backend]) + ":round=" + std::to_string(round);
            thermal.log((label + ":before").c_str());
            if (displacement) {
                const auto displacement_words = displacement_bytes / sizeof(Word);
                displace_l2<<<static_cast<unsigned>(std::min<std::size_t>(65535,
                    (displacement_words + 255) / 256)), 256, 0, stream.value>>>(
                        displacement->data, displacement_words);
                check(cudaGetLastError());
            }
            // Both regimes start timing from a drained stream. Otherwise the
            // long displacement could hide host Graph submission gaps only
            // for that regime and confound the cache comparison.
            check(cudaStreamSynchronize(stream.value));
            check(cudaEventRecord(start.value, stream.value));
            check(cudaGraphLaunch(repeated[backend].executable, stream.value));
            check(cudaEventRecord(stop.value, stream.value));
            check(cudaEventSynchronize(stop.value));
            float elapsed = 0;
            check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
            thermal.log((label + ":after").c_str());
            if (o.thermal_guard) thermal.check_sample();
            if (!(elapsed > 0)) throw std::runtime_error("invalid Graph timing");
            const double ms = static_cast<double>(elapsed) / o.repetitions;
            samples[backend].push_back(ms);
            emit("sample", backend, static_cast<int>(round), static_cast<int>(slot), ms);
        }
    }
    for (unsigned backend = 0; backend < 2; ++backend) {
        const double ms = median(samples[backend]);
        emit("median", backend, -1, -1, ms);
        std::cout << "median,backend=" << names[backend] << ",kernel_ms=" << ms << ",exact=1\n";
    }
}
} // namespace
int main(int argc, char** argv) {
    try { run(parse(argc, argv)); return 0; }
    catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}
