#include "rns_stage_batch.hpp"
#include "rns_preparation.hpp"
#include "gpuntt/ntt_merge/ntt.cuh"

#include <algorithm>
#include <limits>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace cubutterfly_heongpu {
namespace {
using Word = std::uint64_t;
using Wide = unsigned __int128;

void check(cudaError_t status, const char* context) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
}
Word multiply(Word a, Word b, Word q) { return Word(Wide(a) * b % q); }
Word power(Word a, Word e, Word q) {
    Word result = 1;
    for (; e; e >>= 1, a = multiply(a, a, q))
        if (e & 1) result = multiply(result, a, q);
    return result;
}
bool prime(Word q) {
    if (q < 2) return false;
    for (Word p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL})
        if (q % p == 0) return q == p;
    Word odd = q - 1;
    unsigned shifts = 0;
    while (!(odd & 1)) { odd >>= 1; ++shifts; }
    for (Word a : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL}) {
        if (a % q == 0) continue;
        Word x = power(a % q, odd, q);
        if (x == 1 || x == q - 1) continue;
        unsigned j = 1;
        for (; j < shifts; ++j) {
            x = multiply(x, x, q);
            if (x == q - 1) break;
        }
        if (j == shifts) return false;
    }
    return true;
}
std::size_t reverse_bits(std::size_t value, unsigned bits) {
    std::size_t result = 0;
    for (unsigned i = 0; i < bits; ++i, value >>= 1)
        result = (result << 1) | (value & 1);
    return result;
}
template<class T> class Buffer {
  public:
    T* data = nullptr;
    ~Buffer() { if (data) cudaFree(data); }
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("stage allocation size overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)), "stage allocation");
    }
    void upload(const std::vector<T>& host, cudaStream_t stream) {
        allocate(host.size());
        check(cudaMemcpyAsync(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, stream),
              "stage metadata upload");
        // Host vectors may be destroyed immediately after preparation.
        check(cudaStreamSynchronize(stream), "stage metadata ready");
    }
};
struct Range {
    std::uintptr_t begin, end;
    std::size_t request;
};
Range range(const void* pointer, std::size_t offset, std::size_t n, std::size_t request) {
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto limit = std::numeric_limits<std::uintptr_t>::max();
    if (!pointer || address % alignof(Word) || n > limit / sizeof(Word) ||
        offset > (limit - address) / sizeof(Word))
        throw std::invalid_argument("invalid stage pointer or offset");
    const auto begin = address + offset * sizeof(Word);
    if (n * sizeof(Word) > limit - begin)
        throw std::invalid_argument("stage address range overflow");
    return {begin, begin + n * sizeof(Word), request};
}
bool overlaps(const Range& a, const Range& b) { return a.begin < b.end && b.begin < a.end; }

template<bool Gather>
__global__ void transfer(const Word* const* inputs, Word* const* outputs,
                         Word* arena, unsigned log_n, std::size_t words) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < words; i += std::size_t(blockDim.x) * gridDim.x) {
        const auto row = i >> log_n;
        const auto coefficient = i & ((std::size_t{1} << log_n) - 1);
        if constexpr (Gather) arena[i] = inputs[row][coefficient];
        else outputs[row][coefficient] = arena[i];
    }
}

template<bool Gather>
__global__ void transfer_vec2(const Word* const* inputs, Word* const* outputs,
                              Word* arena, unsigned log_n, std::size_t pairs) {
    const auto* source = inputs;
    auto* destination = outputs;
    const auto pair_mask = (std::size_t{1} << (log_n - 1)) - 1;
    for (std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pairs; pair += std::size_t(blockDim.x) * gridDim.x) {
        const auto row = pair >> (log_n - 1);
        const auto coefficient_pair = pair & pair_mask;
        if constexpr (Gather) {
            reinterpret_cast<ulonglong2*>(arena)[pair] =
                reinterpret_cast<const ulonglong2*>(source[row])[coefficient_pair];
        } else {
            reinterpret_cast<ulonglong2*>(destination[row])[coefficient_pair] =
                reinterpret_cast<const ulonglong2*>(arena)[pair];
        }
    }
}
} // namespace

struct DeviceWordArena {
    Word* data = nullptr;
    std::size_t words = 0;

    explicit DeviceWordArena(std::size_t count) : words(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Word))
            throw std::invalid_argument("shared stage arena size overflow");
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Word)),
              "shared stage arena allocation");
    }
    ~DeviceWordArena() { if (data) cudaFree(data); }
    DeviceWordArena(const DeviceWordArena&) = delete;
    DeviceWordArena& operator=(const DeviceWordArena&) = delete;
};

SharedDeviceWordArena make_device_word_arena(std::size_t words) {
    if (!words) throw std::invalid_argument("shared stage arena must be non-empty");
    return std::make_shared<DeviceWordArena>(words);
}

class RnsStageBatch::Impl {
  public:
    StageBackend backend;
    cuntt::RnsPlanConfig packed;
    std::size_t rows = 0, words = 0;
    Buffer<const Word*> inputs;
    Buffer<Word*> outputs;
    Buffer<Word> owned_arena;
    SharedDeviceWordArena shared_arena;
    Word* arena = nullptr;
    bool vectorized_boundary = false;
    Buffer<Root<Word>> roots;
    Buffer<Modulus<Word>> moduli;
    Buffer<Word> inverses;
    Buffer<int> order;
    std::unique_ptr<cuntt::RnsPlan> plan;

    Impl(const std::vector<BoundRnsRequest>& requests, StageBackend selected,
         cudaStream_t stream, SharedDeviceWordArena supplied_arena)
        : backend(selected), shared_arena(std::move(supplied_arena)) {
        if (requests.empty()) throw std::invalid_argument("stage requires at least one request");
        if (backend != StageBackend::Native && backend != StageBackend::CuButterfly)
            throw std::invalid_argument("invalid stage backend");
        packed = requests.front().config;
        packed.domains.clear();
        packed.polynomials.clear();
        if (packed.log_n < 12 || packed.log_n > 20 || packed.ring != cuntt::RnsRing::Negacyclic ||
            packed.input_order != (packed.inverse ? cuntt::RnsOrder::BitReversed : cuntt::RnsOrder::Natural) ||
            packed.output_order != (packed.inverse ? cuntt::RnsOrder::Natural : cuntt::RnsOrder::BitReversed))
            throw std::invalid_argument("stage requires log_n 12..20, negacyclic native boundary orders");
        const std::size_t n = std::size_t{1} << packed.log_n;
        std::vector<const Word*> host_inputs;
        std::vector<Word*> host_outputs;
        std::vector<int> host_order;
        std::vector<Range> input_ranges, output_ranges;
        std::map<std::pair<Word, Word>, std::uint32_t> domains;
        for (std::size_t r = 0; r < requests.size(); ++r) {
            const auto& request = requests[r];
            const auto& c = request.config;
            if (c.log_n != packed.log_n || c.inverse != packed.inverse || c.ring != packed.ring ||
                c.input_order != packed.input_order || c.output_order != packed.output_order || c.polynomials.empty())
                throw std::invalid_argument("incompatible or empty stage request");
            for (const auto& p : c.polynomials) {
                if (p.domain >= c.domains.size()) throw std::invalid_argument("stage domain index out of range");
                // Vendor kernels shift a signed int row index before widening;
                // also respect CUDA's grid.z bound on the native path.
                if (rows >= std::min<std::size_t>(65535, std::numeric_limits<int>::max() / n))
                    throw std::invalid_argument("stage exceeds bounded native row/address range");
                const auto& d = c.domains[p.domain];
                auto key = std::make_pair(d.modulus, d.root);
                auto found = domains.find(key);
                if (found == domains.end()) {
                    if (d.modulus < 3 || d.modulus >= (Word{1} << 62) || !prime(d.modulus) ||
                        (d.modulus - 1) % (2 * n) || d.root >= d.modulus ||
                        power(d.root, n, d.modulus) != d.modulus - 1 ||
                        power(d.root, 2 * n, d.modulus) != 1)
                        throw std::invalid_argument("stage requires prime q < 2^62 and canonical primitive forward 2N root");
                    const auto index = static_cast<std::uint32_t>(packed.domains.size());
                    found = domains.emplace(key, index).first;
                    packed.domains.push_back(d);
                }
                const auto in = range(request.input, p.input_offset, n, r);
                const auto out = range(request.output, p.output_offset, n, r);
                input_ranges.push_back(in);
                output_ranges.push_back(out);
                host_inputs.push_back(reinterpret_cast<const Word*>(in.begin));
                host_outputs.push_back(reinterpret_cast<Word*>(out.begin));
                host_order.push_back(static_cast<int>(found->second));
                packed.polynomials.push_back({found->second, rows * n, rows * n});
                ++rows;
            }
        }
        for (std::size_t i = 0; i < output_ranges.size(); ++i) {
            for (std::size_t j = 0; j < i; ++j)
                if (overlaps(output_ranges[i], output_ranges[j]))
                    throw std::invalid_argument("stage output polynomial ranges overlap");
            for (const auto& in : input_ranges)
                if (in.request != output_ranges[i].request && overlaps(in, output_ranges[i]))
                    throw std::invalid_argument("stage requests have input/output dependencies");
        }
        words = rows * n;
        vectorized_boundary = (words % 2 == 0);
        for (const auto& input_range : input_ranges)
            vectorized_boundary = vectorized_boundary && (input_range.begin % alignof(ulonglong2) == 0);
        for (const auto& output_range : output_ranges)
            vectorized_boundary = vectorized_boundary && (output_range.begin % alignof(ulonglong2) == 0);
        if (shared_arena) {
            if (!shared_arena->data || shared_arena->words < words)
                throw std::invalid_argument("shared stage arena is smaller than stage workspace");
            arena = shared_arena->data;
        } else {
            owned_arena.allocate(words);
            arena = owned_arena.data;
        }
        inputs.upload(host_inputs, stream);
        outputs.upload(host_outputs, stream);
        if (backend == StageBackend::CuButterfly) {
            plan = std::make_unique<cuntt::RnsPlan>(packed);
            const auto preparation = prepare_rns(*plan, cuntt::RnsPlacement::InPlace, stream, "stage");
            const auto& m = plan->mapping();
            const auto& selection = plan->selection();
            std::cerr << "rns_stage_batch_plan,N=" << n << ",requests=" << requests.size()
                      << ",rows=" << rows << ",domains=" << packed.domains.size()
                      << ",inverse=" << packed.inverse << ",placement=in-place"
                      << ",core_family=" << static_cast<int>(m.core_family) << ",n1_log=" << m.n1_log
                      << ",rows_per_block=" << m.rows_per_block << ",threads_per_block=" << m.threads_per_block
                      << ",second_rows_per_block=" << m.second_rows_per_block
                      << ",second_threads_per_block=" << m.second_threads_per_block
                      << ",compute_unit=" << cuntt::compute_unit_name(m.compute_unit)
                      << ",arithmetic=" << static_cast<int>(m.arithmetic)
                      << ",exchange=" << (m.exchange == cuntt::RnsExchange::DependencyScoped ? "dependency_scoped" : "block")
                      << ",cross_twiddle=" << static_cast<int>(m.cross_twiddle)
                      << ",physical_boundary_traversal=" << m.physical_boundary_traversal
                      << ",candidate_count=" << plan->measurements().size()
                      << ",boundary_copy_bytes=" << (words * sizeof(Word) * 2)
                      << ",boundary_vectorized=" << vectorized_boundary
                      << ",confirmed_kernel_ms=" << preparation.kernel_ms
                      << ",prepared_cache_hit=" << preparation.cache_hit
                      << ",prepared_source=" << preparation.source
                      << ",confidence=" << (preparation.cache_hit ? "prepared-rns-replay" : selection.confidence)
                      << ",reason=" << selection.reason << '\n';
        } else {
            std::vector<Root<Word>> host_roots(packed.domains.size() * n);
            std::vector<Modulus<Word>> host_moduli;
            std::vector<Word> host_inverses;
            for (std::size_t d = 0; d < packed.domains.size(); ++d) {
                const Word q = packed.domains[d].modulus;
                const Word root = packed.inverse ? power(packed.domains[d].root, q - 2, q) : packed.domains[d].root;
                Word value = 1;
                for (std::size_t j = 0; j < n; ++j, value = multiply(value, root, q))
                    host_roots[d * n + reverse_bits(j, packed.log_n)] = value;
                host_moduli.emplace_back(q);
                host_inverses.push_back(power(n, q - 2, q));
            }
            roots.upload(host_roots, stream);
            moduli.upload(host_moduli, stream);
            inverses.upload(host_inverses, stream);
            order.upload(host_order, stream);
        }
    }

    void execute(cudaStream_t stream) {
        const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((words + 255) / 256, 65535));
        if (vectorized_boundary) {
            const unsigned pair_blocks = static_cast<unsigned>(std::min<std::size_t>((words / 2 + 255) / 256, 65535));
            transfer_vec2<true><<<pair_blocks, 256, 0, stream>>>(inputs.data, outputs.data, arena, packed.log_n, words / 2);
        } else {
            transfer<true><<<blocks, 256, 0, stream>>>(inputs.data, outputs.data, arena, packed.log_n, words);
        }
        check(cudaGetLastError(), "stage gather");
        if (plan) plan->execute_async(arena, arena, stream);
        else {
            // mod_count cycles only the order index; tables stay deduplicated.
            // The pinned public dispatch prepares host launch descriptors here;
            // captured Graph replay contains only its native kernel launches.
            const int count = static_cast<int>(rows);
            gpuntt::ntt_rns_configuration<Word> config{};
            config.n_power = packed.log_n;
            config.ntt_type = packed.inverse ? gpuntt::INVERSE : gpuntt::FORWARD;
            config.ntt_layout = gpuntt::PerPolynomial;
            config.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
            config.mod_inverse = inverses.data;
            config.stream = stream;
            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(arena, roots.data, moduli.data,
                                                   config, count, count, order.data);
        }
        if (vectorized_boundary) {
            const unsigned pair_blocks = static_cast<unsigned>(std::min<std::size_t>((words / 2 + 255) / 256, 65535));
            transfer_vec2<false><<<pair_blocks, 256, 0, stream>>>(inputs.data, outputs.data, arena, packed.log_n, words / 2);
        } else {
            transfer<false><<<blocks, 256, 0, stream>>>(inputs.data, outputs.data, arena, packed.log_n, words);
        }
        check(cudaGetLastError(), "stage scatter");
    }
};

RnsStageBatch::RnsStageBatch(const std::vector<BoundRnsRequest>& requests,
                           StageBackend backend, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(requests, backend, stream, nullptr)) {}
RnsStageBatch::RnsStageBatch(const std::vector<BoundRnsRequest>& requests,
                           StageBackend backend, cudaStream_t stream,
                           SharedDeviceWordArena arena)
    : impl_(std::make_unique<Impl>(requests, backend, stream, std::move(arena))) {}
RnsStageBatch::~RnsStageBatch() = default;
void RnsStageBatch::execute_async(cudaStream_t stream) { impl_->execute(stream); }
std::size_t RnsStageBatch::polynomial_count() const noexcept { return impl_->rows; }
std::size_t RnsStageBatch::boundary_copy_bytes() const noexcept {
    return impl_->words * sizeof(Word) * 2;
}

} // namespace cubutterfly_heongpu
