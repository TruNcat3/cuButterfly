#include "ntt_adapter.hpp"
#include "ntt_capture.hpp"
#include "rns_preparation.hpp"
#include <cuntt/ntt.hpp>
#include <cuntt/rns_ntt.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace cubutterfly_heongpu {
namespace {
using U64 = std::uint64_t;
using U128 = unsigned __int128;
thread_local bool active = false;
thread_local bool frozen = false;
thread_local Execution execution = Execution::NativeRns;
thread_local std::size_t calls = 0, polynomials = 0;
thread_local std::vector<CapturedNttStage>* stage_recorder = nullptr;

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
}
U64 multiply(U64 a, U64 b, U64 q) { return U128(a) * b % q; }
U64 power(U64 a, U64 e, U64 q) {
    U64 y = 1;
    for (; e; e >>= 1, a = multiply(a, a, q))
        if (e & 1) y = multiply(y, a, q);
    return y;
}
unsigned reverse(unsigned v, int bits) {
    unsigned r = 0;
    for (int i = 0; i < bits; ++i) { r = (r << 1) | (v & 1); v >>= 1; }
    return r;
}
U64 validate_table(U64 q, const std::vector<U64>& table, int log_n, bool inverse) {
    const unsigned n = 1u << log_n;
    if (q < 3 || q >= (U64(1) << 62) || (q - 1) % (2 * U64(n)))
        throw std::invalid_argument("adapter requires a word64 NTT prime below 2^62 with a 2N-th root");
    const U64 table_root = table[n / 2];
    const U64 psi = inverse ? power(table_root, q - 2, q) : table_root;
    if (power(psi, n, q) != q - 1 || power(psi, 2 * U64(n), q) != 1)
        throw std::invalid_argument("adapter table is not a primitive negacyclic root");
    U64 value = 1;
    for (unsigned j = 0; j < n; ++j) {
        if (table[reverse(j, log_n)] != value)
            throw std::invalid_argument("unsupported HEonGPU twiddle table representation");
        value = multiply(value, table_root, q);
    }
    return psi;
}
template<class T> struct Buffer {
    T* pointer = nullptr;
    explicit Buffer(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "adapter allocation");
    }
    ~Buffer() { if (pointer) cudaFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
template<class T> std::vector<T> read(const T* source, std::size_t count, cudaStream_t stream) {
    std::vector<T> host(count);
    check(cudaMemcpyAsync(host.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "adapter metadata read");
    check(cudaStreamSynchronize(stream), "adapter metadata preparation");
    return host;
}
template<class T> void upload(Buffer<T>& target, const std::vector<T>& host, cudaStream_t stream) {
    check(cudaMemcpyAsync(target.pointer, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "adapter metadata upload");
    check(cudaStreamSynchronize(stream), "adapter metadata upload completion");
}

__device__ U64 shoup(U64 a, U64 b, U64 b_shoup, U64 q) {
    U64 t = a * b - __umul64hi(a, b_shoup) * q;
    return t >= q ? t - q : t;
}
__global__ void pack(const U64* input, U64* work, const int* locations,
                     const U64* twist, const U64* twist_shoup, U64 q,
                     int log_n, unsigned multiplier, bool inverse, std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned n = 1u << log_n, j = i & (n - 1);
    const std::size_t base = std::size_t(locations[i >> log_n]) << log_n;
    if (inverse) {
        const unsigned k = (U64(j) * multiplier) & (n - 1);
        work[i] = input[base + (__brev(k) >> (32 - log_n))];
    } else {
        work[i] = shoup(input[base + j], twist[j], twist_shoup[j], q);
    }
}
__global__ void unpack(const U64* work, U64* output, const int* locations,
                       const U64* twist, const U64* twist_shoup, U64 q,
                       int log_n, unsigned multiplier_inverse, bool inverse,
                       std::size_t count) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const unsigned n = 1u << log_n, j = i & (n - 1);
    const std::size_t base = std::size_t(locations[i >> log_n]) << log_n;
    if (inverse) {
        output[base + j] = shoup(work[i], twist[j], twist_shoup[j], q);
    } else {
        const unsigned k = (U64(j) * multiplier_inverse) & (n - 1);
        output[base + (__brev(j) >> (32 - log_n))] = work[(i & ~(std::size_t(n) - 1)) + k];
    }
}

struct Group {
    U64 q;
    int log_n;
    bool inverse;
    unsigned multiplier = 0, multiplier_inverse = 0;
    std::size_t count;
    Buffer<int> locations;
    Buffer<U64> twist, twist_shoup, input, output;
    std::unique_ptr<cuntt::Plan> plan;

    Group(U64 modulus, const std::vector<U64>& table, const std::vector<int>& positions,
          gpuntt::ntt_rns_configuration<Data64> cfg)
        : q(modulus), log_n(cfg.n_power), inverse(cfg.ntt_type == gpuntt::INVERSE),
          count(positions.size() << log_n), locations(positions.size()),
          twist(1u << log_n), twist_shoup(1u << log_n), input(count), output(count) {
        const unsigned n = 1u << log_n;
        if (q < 3 || q >= (U64(1) << 62) || (q - 1) % (2 * U64(n)))
            throw std::invalid_argument("adapter requires a word64 NTT prime below 2^62 with a 2N-th root");
        const U64 table_root = table[n / 2];
        const U64 psi = inverse ? power(table_root, q - 2, q) : table_root;
        if (power(psi, n, q) != q - 1 || power(psi, 2 * U64(n), q) != 1)
            throw std::invalid_argument("adapter table is not a primitive negacyclic root");
        std::vector<U64> factors(n), quotients(n);
        U64 value = 1;
        for (unsigned j = 0; j < n; ++j) {
            if (table[reverse(j, log_n)] != value)
                throw std::invalid_argument("unsupported HEonGPU twiddle table representation");
            factors[j] = value;
            quotients[j] = (U128(value) << 64) / q;
            value = multiply(value, table_root, q);
        }
        const U64 omega = multiply(psi, psi, q);
        const U64 cu_root = cuntt::find_primitive_power_of_two_root(log_n, q);
        value = 1;
        for (unsigned j = 0; j < n; ++j) {
            if (value == cu_root) { multiplier = j; break; }
            value = multiply(value, omega, q);
        }
        if (!(multiplier & 1)) throw std::runtime_error("cyclic root equivalence not found");
        for (unsigned j = 1; j < n; j += 2)
            if ((U64(j) * multiplier & (n - 1)) == 1) { multiplier_inverse = j; break; }
        if (!multiplier_inverse) throw std::runtime_error("root permutation inverse not found");
        upload(locations, positions, cfg.stream);
        upload(twist, factors, cfg.stream);
        upload(twist_shoup, quotients, cfg.stream);
        cuntt::PlanConfig request;
        request.log_n = log_n;
        request.batch = positions.size();
        request.modulus = q;
        request.inverse = inverse;
        request.auto_select = true;
        request.word_bits = 64;
        plan = std::make_unique<cuntt::Plan>(request);
        plan->set_stream(cfg.stream);
        const auto& selected = plan->selection();
        std::cerr << "ntt_adapter_plan,N=" << n << ",B=" << positions.size()
                  << ",execution=grouped"
                  << ",q=" << q << ",inverse=" << inverse
                  << ",backend=" << cuntt::backend_name(plan->config().backend)
                  << ",selection_calibrated=" << selected.calibrated
                  << ",confidence=" << selected.confidence
                  << ",reason=" << selected.reason << '\n';
    }
    void run(Data64* source, Data64* destination, cudaStream_t stream) {
        pack<<<(count + 255) / 256, 256, 0, stream>>>(source, input.pointer,
            locations.pointer, twist.pointer, twist_shoup.pointer, q,
            log_n, multiplier, inverse, count);
        check(cudaGetLastError(), "adapter pack");
        plan->execute_async(input.pointer, output.pointer);
        unpack<<<(count + 255) / 256, 256, 0, stream>>>(output.pointer, destination,
            locations.pointer, twist.pointer, twist_shoup.pointer, q,
            log_n, multiplier_inverse, inverse, count);
        check(cudaGetLastError(), "adapter unpack");
    }
};

// This cache owns buffers and plans per stream. It is local to a thread, and
// device identity is included. Pointer-backed tables must remain immutable.
using Key = std::tuple<int, std::uintptr_t, std::uintptr_t, std::uintptr_t,
    std::uintptr_t, std::uintptr_t, int, int, int, int, int, int, bool>;
struct Prepared {
    std::vector<std::unique_ptr<Group>> groups;
    std::unique_ptr<cuntt::RnsPlan> rns;
};
thread_local std::map<Key, Prepared> cache;
}  // namespace

void set_enabled(bool value) { active = value; }
bool enabled() { return active; }
void set_execution(Execution value) {
    if (value != Execution::NativeRns && value != Execution::Grouped)
        throw std::invalid_argument("invalid HEonGPU NTT execution mode");
    execution = value;
}
Execution get_execution() { return execution; }
const char* execution_name(Execution value) {
    switch (value) {
        case Execution::NativeRns: return "native-rns";
        case Execution::Grouped: return "grouped";
    }
    throw std::invalid_argument("invalid HEonGPU NTT execution mode");
}
void freeze_plans(bool value) { frozen = value; }
void reset_counters() { calls = polynomials = 0; }
std::size_t replacement_calls() { return calls; }
std::size_t replacement_polynomials() { return polynomials; }
void clear_cache() { cache.clear(); frozen = false; }
void record_ntt_stages(std::vector<CapturedNttStage>* stages) {
    if (stages && (stage_recorder || execution != Execution::NativeRns || !active))
        throw std::logic_error("NTT stage recording requires enabled native-rns and no active recorder");
    stage_recorder = stages;
}

void transform(Data64* source, Data64* destination, Root64* roots, Modulus64* moduli,
               gpuntt::ntt_rns_configuration<Data64> cfg, int batch,
               int modulus_count, Ordering ordering, int* order) {
    if (cfg.ntt_layout != gpuntt::PerPolynomial || cfg.zero_padding ||
        (ordering != Ordering::Contiguous && ordering != Ordering::Modulus && ordering != Ordering::Polynomial) ||
        cfg.reduction_poly != gpuntt::ReductionPolynomial::X_N_plus ||
        (cfg.ntt_type != gpuntt::FORWARD && cfg.ntt_type != gpuntt::INVERSE) ||
        cfg.n_power < 1 || cfg.n_power > 26 || batch <= 0 || modulus_count <= 0 ||
        !source || !destination || !roots || !moduli ||
        (ordering != Ordering::Contiguous && !order) ||
        (cfg.ntt_type == gpuntt::INVERSE && !cfg.mod_inverse))
        throw std::invalid_argument("unsupported or invalid HEonGPU NTT adapter request");
    if (execution == Execution::NativeRns && (cfg.n_power < 12 || cfg.n_power > 20))
        throw std::invalid_argument("native-rns compiled candidate family supports logN 12..20; no implicit fallback");
    int device = 0;
    check(cudaGetDevice(&device), "adapter device identity");
    const Key key{device, reinterpret_cast<std::uintptr_t>(cfg.stream),
        reinterpret_cast<std::uintptr_t>(roots), reinterpret_cast<std::uintptr_t>(moduli),
        reinterpret_cast<std::uintptr_t>(order), reinterpret_cast<std::uintptr_t>(cfg.mod_inverse),
        cfg.n_power, int(cfg.ntt_type), batch, modulus_count, int(ordering), int(execution),
        source == destination};
    auto found = cache.find(key);
    if (found == cache.end()) {
        if (frozen) throw std::runtime_error("NTT adapter plan miss during frozen measurement/capture");
        const unsigned n = 1u << cfg.n_power;
        std::vector<int> indices;
        if (ordering != Ordering::Contiguous)
            indices = read(order, ordering == Ordering::Modulus ? modulus_count : batch, cfg.stream);
        std::map<int, std::vector<int>> groups;
        std::set<int> seen;
        for (int b = 0; b < batch; ++b) {
            const int m = ordering == Ordering::Modulus ? indices[b % modulus_count] : b % modulus_count;
            const int p = ordering == Ordering::Polynomial ? indices[b] : b;
            if (m < 0 || p < 0 || !seen.insert(p).second)
                throw std::invalid_argument("negative or overlapping ordered polynomial request");
            groups[m].push_back(p);
        }
        Prepared prepared;
        cuntt::RnsPlanConfig request;
        request.log_n = cfg.n_power;
        request.inverse = cfg.ntt_type == gpuntt::INVERSE;
        request.ring = cuntt::RnsRing::Negacyclic;
        request.input_order = request.inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural;
        request.output_order = request.inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed;
        std::map<std::pair<U64, U64>, std::uint32_t> domains;
        for (const auto& entry : groups) {
            const auto modulus = read(moduli + entry.first, 1, cfg.stream)[0].value;
            auto table = read(roots + std::size_t(entry.first) * n, n, cfg.stream);
            const U64 psi = validate_table(modulus, table, cfg.n_power, request.inverse);
            if (cfg.ntt_type == gpuntt::INVERSE &&
                read(cfg.mod_inverse + entry.first, 1, cfg.stream)[0] != power(n, modulus - 2, modulus))
                throw std::invalid_argument("inverse NTT normalization differs from 1/N");
            if (execution == Execution::Grouped) {
                prepared.groups.emplace_back(std::make_unique<Group>(modulus, table, entry.second, cfg));
            } else {
                const auto identity = std::make_pair(U64(modulus), psi);
                auto domain = domains.emplace(identity, request.domains.size());
                if (domain.second) request.domains.push_back({modulus, psi});
                for (const int position : entry.second) {
                    const std::size_t offset = std::size_t(position) * n;
                    request.polynomials.push_back({domain.first->second, offset, offset});
                }
            }
        }
        if (execution == Execution::NativeRns) {
            prepared.rns = std::make_unique<cuntt::RnsPlan>(request);
            const auto preparation = prepare_rns(*prepared.rns,
                source == destination ? cuntt::RnsPlacement::InPlace : cuntt::RnsPlacement::Separate,
                cfg.stream, "scalar");
            const auto& mapping = prepared.rns->mapping();
            const auto& selected = prepared.rns->selection();
            std::cerr << "ntt_adapter_plan,N=" << n << ",B=" << batch
                      << ",execution=native-rns,domains=" << request.domains.size()
                      << ",placement=" << (source == destination ? "in-place" : "separate")
                      << ",inverse=" << request.inverse
                      << ",n1_log=" << mapping.n1_log
                      << ",core_family=" << (mapping.core_family == cuntt::RnsCoreFamily::FusedCTGS ? "fused-ctgs" :
                                                   mapping.core_family == cuntt::RnsCoreFamily::GpuNttMerge ? "gpuntt-merge" : "hybrid2d")
                      << ",rows_per_block=" << mapping.rows_per_block
                      << ",threads_per_block=" << mapping.threads_per_block
                      << ",second_rows_per_block=" << mapping.second_rows_per_block
                      << ",second_threads_per_block=" << mapping.second_threads_per_block
                      << ",arithmetic=" << (mapping.arithmetic == cuntt::RnsArithmetic::Barrett ? "barrett" : "shoup")
                      << ",exchange=" << (mapping.exchange == cuntt::RnsExchange::DependencyScoped ? "dependency_scoped" : "block")
                      << ",compute_unit=" << cuntt::compute_unit_name(mapping.compute_unit)
                      << ",cross_twiddle=" << (mapping.core_family == cuntt::RnsCoreFamily::FusedCTGS ? "embedded" : cuntt::cross_twiddle_placement_name(mapping.cross_twiddle))
                      << ",physical_boundary_traversal=" << mapping.physical_boundary_traversal
                      << ",candidate_count=" << prepared.rns->measurements().size()
                      << ",confirmed_kernel_ms=" << preparation.kernel_ms
                      << ",prepared_cache_hit=" << preparation.cache_hit
                      << ",prepared_source=" << preparation.source
                      << ",selection_calibrated=" << selected.calibrated
                      << ",confidence=" << (preparation.cache_hit ? "prepared-rns-replay" : selected.confidence)
                      << ",reason=" << selected.reason << '\n';
            for (std::size_t d = 0; d < request.domains.size(); ++d)
                std::cerr << "ntt_adapter_domain,N=" << n << ",B=" << batch
                          << ",inverse=" << request.inverse << ",domain=" << d
                          << ",q=" << request.domains[d].modulus
                          << ",forward_psi=" << request.domains[d].root << '\n';
        }
        found = cache.emplace(key, std::move(prepared)).first;
    }
    ++calls;
    polynomials += batch;
    if (stage_recorder) {
        cudaStreamCaptureStatus status;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t* dependencies = nullptr;
        std::size_t dependency_count = 0;
        check(cudaStreamGetCaptureInfo_v2(cfg.stream, &status, nullptr, &graph,
              &dependencies, &dependency_count), "NTT stage capture info");
        if (status != cudaStreamCaptureStatusActive || !found->second.rns)
            throw std::logic_error("NTT stage markers require active capture and a prepared RNS plan");
        cudaGraphNode_t marker;
        check(cudaGraphAddEmptyNode(&marker, graph, dependencies, dependency_count), "NTT stage marker");
        check(cudaStreamUpdateCaptureDependencies(cfg.stream, &marker, 1,
              cudaStreamSetCaptureDependencies), "NTT marker stream dependency");
        stage_recorder->push_back({marker, {found->second.rns->config(), source, destination}});
        return;
    }
    if (found->second.rns) found->second.rns->execute_async(source, destination, cfg.stream);
    else for (auto& group : found->second.groups) group->run(source, destination, cfg.stream);
}
}  // namespace cubutterfly_heongpu
