#include "cuntt/rns_ntt.hpp"
#include "ntt_hybrid2d.cuh"
#include "rns_fused_ctgs.cuh"
#include "butterfly_exchange.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cuntt {
namespace {
using U64 = std::uint64_t;
using U128 = unsigned __int128;
using hybrid2d_detail::BatchView;
using hybrid2d_detail::DomainView;
using hybrid2d_detail::PolynomialView;
constexpr std::array<unsigned, 4> fused_thread_candidates{32, 64, 128, 256};

const char* family_name(RnsCoreFamily family, RnsArithmetic arithmetic = RnsArithmetic::Shoup) {
    if (family == RnsCoreFamily::FusedCTGS)
        return arithmetic == RnsArithmetic::Barrett
            ? "fused-ctgs-gpuntt-barrett-rns-native-boundary"
            : "fused-ctgs-rns-native-boundary";
    if (family == RnsCoreFamily::GpuNttMerge)
        return "gpuntt-merge-rns-native-boundary";
    return "hybrid2d-rns-native-boundary";
}

RnsMapping pass_mapping(const RnsMapping& mapping, bool first) {
    auto result = mapping;
    if (!first) {
        if (mapping.second_rows_per_block) result.rows_per_block = mapping.second_rows_per_block;
        if (mapping.second_threads_per_block) result.threads_per_block = mapping.second_threads_per_block;
    }
    return result;
}

std::array<unsigned, 2> exchange_masks(const RnsPlanConfig& config, const RnsMapping& mapping) {
    std::array<unsigned, 2> masks{};
    if (mapping.exchange == RnsExchange::Block) return masks;
    for (unsigned pass = 0; pass < 2; ++pass) {
        const auto stage = pass_mapping(mapping, pass == 0);
        const bool prefix = config.inverse ? pass != 0 : pass == 0;
        const unsigned log = prefix ? config.log_n - mapping.n1_log : mapping.n1_log;
        masks[pass] = detail::butterfly_warp_barrier_mask(log, stage.rows_per_block,
            stage.threads_per_block, config.inverse,
            mapping.compute_unit == ComputeUnit::Radix4 ? 2 : 1, prefix);
    }
    return masks;
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("RNS NTT: ") + cudaGetErrorString(status));
}
U64 multiply(U64 a, U64 b, U64 q) { return static_cast<U64>(U128(a) * b % q); }
U64 shoup(U64 value, U64 q) { return static_cast<U64>((U128(value) << 64) / q); }
template<class T> struct Array {
    T* data = nullptr;
    explicit Array(std::size_t size) {
        if (!size || size > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::invalid_argument("invalid RNS allocation extent");
        check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(T)));
    }
    explicit Array(const std::vector<T>& host) : Array(host.size()) {
        try { check(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice)); }
        catch (...) { cudaFree(data); data = nullptr; throw; }
    }
    ~Array() { if (data) cudaFree(data); }
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

void require_uncaptured(cudaStream_t stream) {
    cudaStreamCaptureStatus status;
    check(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone)
        throw std::logic_error("RNS preparation/tuning must precede Graph capture");
}

template<unsigned Log, unsigned Rows, ComputeUnit Core, CrossTwiddlePlacement Cross>
void launch(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
            const U64* input, U64* scratch, U64* output, BatchView<U64> view, cudaStream_t stream) {
    const unsigned other = config.log_n - Log;
    const std::size_t blocks = config.polynomials.size() * ((std::size_t(1) << other) / Rows);
    const std::size_t shared = Rows * ((std::size_t(1) << Log) + 1) * sizeof(U64);
    if (first) {
        hybrid2d_detail::hybrid2d_first_pass_kernel<U64, Log, Rows, Core, Cross, false, true>
            <<<static_cast<unsigned>(blocks), mapping.threads_per_block, shared, stream>>>(
                input, scratch, nullptr, nullptr, nullptr, nullptr, 0, other, 0, 0, config.log_n, view);
    } else {
        hybrid2d_detail::hybrid2d_second_pass_kernel<U64, Log, Rows, Core, Cross, false, true>
            <<<static_cast<unsigned>(blocks), mapping.threads_per_block, shared, stream>>>(
                scratch, output, nullptr, nullptr, 0, nullptr, nullptr, false, 1, 0,
                0, 0, other, config.log_n, view);
    }
    check(cudaGetLastError());
}
template<unsigned Log, ComputeUnit Core, CrossTwiddlePlacement Cross>
void dispatch_rows(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
                   const U64* input, U64* scratch, U64* output, BatchView<U64> view, cudaStream_t stream) {
#define RNS_ROWS(R) launch<Log, R, Core, Cross>(config, mapping, first, input, scratch, output, view, stream)
    switch (mapping.rows_per_block) {
        case 1: RNS_ROWS(1); break;
        case 2: RNS_ROWS(2); break;
        case 4: RNS_ROWS(4); break;
        case 8: RNS_ROWS(8); break;
        case 16: RNS_ROWS(16); break;
        case 32: RNS_ROWS(32); break;
        default: throw std::logic_error("unresolved RNS row mapping");
    }
#undef RNS_ROWS
}
template<ComputeUnit Core, CrossTwiddlePlacement Cross>
void dispatch_log(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
                  const U64* input, U64* scratch, U64* output, BatchView<U64> view, cudaStream_t stream) {
    const auto log = first ? config.log_n - mapping.n1_log : mapping.n1_log;
#define RNS_LOG(L) dispatch_rows<L, Core, Cross>(config, mapping, first, input, scratch, output, view, stream)
    switch (log) {
        case 6: RNS_LOG(6); break;
        case 7: RNS_LOG(7); break;
        case 8: RNS_LOG(8); break;
        case 9: RNS_LOG(9); break;
        case 10: RNS_LOG(10); break;
        default: throw std::logic_error("unresolved RNS split");
    }
#undef RNS_LOG
}
template<ComputeUnit Core>
void dispatch_cross(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
                    const U64* input, U64* scratch, U64* output, BatchView<U64> view, cudaStream_t stream) {
    if (mapping.cross_twiddle == CrossTwiddlePlacement::FirstPass)
        dispatch_log<Core, CrossTwiddlePlacement::FirstPass>(config, mapping, first, input, scratch, output, view, stream);
    else dispatch_log<Core, CrossTwiddlePlacement::SecondPass>(config, mapping, first, input, scratch, output, view, stream);
}

template<unsigned Log, unsigned Rows, bool Barrett, bool Compact>
void launch_fused(const RnsPlanConfig& config, const RnsMapping& mapping,
                  bool first, const U64* input, U64* output,
                  const rns_fused_detail::Domain* domains, const PolynomialView* polynomials,
                  bool in_place, cudaStream_t stream, unsigned warp_mask) {
    const std::size_t blocks = config.polynomials.size() * ((std::size_t(1) << (config.log_n - Log)) / Rows);
    const std::size_t shared = Rows * (std::size_t(1) << Log) * sizeof(U64);
    const bool reverse_input = config.inverse ? config.input_order == RnsOrder::Natural
                                              : config.input_order == RnsOrder::BitReversed;
    const bool reverse_output = config.inverse ? config.output_order == RnsOrder::BitReversed
                                               : config.output_order == RnsOrder::Natural;
    const auto dispatch = [&](auto inverse_tag, auto radix_tag, auto first_tag) {
        constexpr bool inverse = decltype(inverse_tag)::value;
        constexpr unsigned radix_log = decltype(radix_tag)::value;
        constexpr bool first_pass = decltype(first_tag)::value;
        rns_fused_detail::stage_kernel<Log, Rows, inverse, radix_log, Barrett, Compact, first_pass>
            <<<static_cast<unsigned>(blocks), mapping.threads_per_block, shared, stream>>>(
                input, output, domains, polynomials, config.log_n,
                reverse_input, reverse_output, in_place, warp_mask);
    };
    const auto pass = [&](auto inverse_tag, auto radix_tag) {
        if (first) dispatch(inverse_tag, radix_tag, std::true_type{});
        else dispatch(inverse_tag, radix_tag, std::false_type{});
    };
    const auto direction = [&](auto radix_tag) {
        if (config.inverse) pass(std::true_type{}, radix_tag);
        else pass(std::false_type{}, radix_tag);
    };
    if (mapping.compute_unit == ComputeUnit::Radix4)
        direction(std::integral_constant<unsigned, 2>{});
    else direction(std::integral_constant<unsigned, 1>{});
    check(cudaGetLastError());
}
template<unsigned Log, bool Barrett, bool Compact>
void dispatch_fused_rows(const RnsPlanConfig& config, const RnsMapping& mapping,
                         bool first, const U64* input, U64* output,
                         const rns_fused_detail::Domain* domains, const PolynomialView* polynomials,
                         bool in_place, cudaStream_t stream, unsigned warp_mask) {
#define FUSED_ROWS(R) launch_fused<Log, R, Barrett, Compact>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask)
    switch (mapping.rows_per_block) {
        case 1: FUSED_ROWS(1); break;
        case 2: FUSED_ROWS(2); break;
        case 4: FUSED_ROWS(4); break;
        case 8: FUSED_ROWS(8); break;
        case 16: FUSED_ROWS(16); break;
        case 32: FUSED_ROWS(32); break;
        case 64: FUSED_ROWS(64); break;
        case 128: FUSED_ROWS(128); break;
        default: throw std::logic_error("unresolved fused RNS row mapping");
    }
#undef FUSED_ROWS
}
template<bool Barrett, bool Compact>
void dispatch_fused_arithmetic(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
                    const U64* input, U64* output, const rns_fused_detail::Domain* domains,
                    const PolynomialView* polynomials, bool in_place, cudaStream_t stream, unsigned warp_mask) {
    const bool prefix = config.inverse ? !first : first;
    const unsigned log = prefix ? config.log_n - mapping.n1_log : mapping.n1_log;
#define FUSED_LOG(L) dispatch_fused_rows<L, Barrett, Compact>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask)
    switch (log) {
        case 2: FUSED_LOG(2); break;
        case 3: FUSED_LOG(3); break;
        case 4: FUSED_LOG(4); break;
        case 5: FUSED_LOG(5); break;
        case 6: FUSED_LOG(6); break;
        case 7: FUSED_LOG(7); break;
        case 8: FUSED_LOG(8); break;
        case 9: FUSED_LOG(9); break;
        case 10: FUSED_LOG(10); break;
        default: throw std::logic_error("unresolved fused RNS split");
    }
#undef FUSED_LOG
}
void dispatch_fused(const RnsPlanConfig& config, const RnsMapping& mapping, bool first,
                    const U64* input, U64* output, const rns_fused_detail::Domain* domains,
                    const PolynomialView* polynomials, bool in_place, bool compact, cudaStream_t stream, unsigned warp_mask) {
    if (mapping.core_family == RnsCoreFamily::GpuNttMerge) {
        // Keep the sparse/permuted-row contract in Compact mode, but remove
        // its physical-offset arithmetic for the native dense layout. This
        // is the same separation as the public provider contract: generic
        // addressing is available when requested, while the common contiguous
        // GPU-NTT shape does not pay for it on every stage.
        if (compact)
            dispatch_fused_arithmetic<true, true>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask);
        else
            dispatch_fused_arithmetic<true, false>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask);
    } else if (mapping.arithmetic == RnsArithmetic::Barrett)
        dispatch_fused_arithmetic<true, false>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask);
    else
        dispatch_fused_arithmetic<false, false>(config, mapping, first, input, output, domains, polynomials, in_place, stream, warp_mask);
}

__global__ void initialize_tuning_input(U64* values, std::size_t size, U64 modulus) {
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += std::size_t(gridDim.x) * blockDim.x) {
        U64 x = i + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        values[i] = (x ^ (x >> 31)) % modulus;
    }
}
} // namespace

class RnsPlan::Impl {
  public:
    explicit Impl(const RnsPlanConfig& requested) : config(requested) {
        require_uncaptured(nullptr);
        check(cudaGetDevice(&device));
        check(cudaGetDeviceProperties(&properties, device));
        if (config.log_n < 12 || config.log_n > 20 || config.domains.empty() || config.polynomials.empty())
            throw std::invalid_argument("RNS Hybrid2D requires log_n in [12,20], domains and polynomials");
        if ((config.ring != RnsRing::Cyclic && config.ring != RnsRing::Negacyclic) ||
            (config.input_order != RnsOrder::Natural && config.input_order != RnsOrder::BitReversed) ||
            (config.output_order != RnsOrder::Natural && config.output_order != RnsOrder::BitReversed))
            throw std::invalid_argument("invalid RNS ring or order");
        n = std::size_t(1) << config.log_n;
        if (config.polynomials.size() > std::numeric_limits<std::size_t>::max() / n / sizeof(U64))
            throw std::invalid_argument("RNS batch size overflows workspace extent");
        std::vector<std::pair<std::size_t, std::size_t>> outputs;
        std::vector<PolynomialView> host_polys;
        for (const auto& p : config.polynomials) {
            const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(U64) - n;
            if (p.domain >= config.domains.size() || p.input_offset > limit || p.output_offset > limit)
                throw std::invalid_argument("invalid RNS polynomial domain or extent");
            input_extent = std::max(input_extent, p.input_offset + n);
            output_extent = std::max(output_extent, p.output_offset + n);
            outputs.emplace_back(p.output_offset, p.output_offset + n);
            host_polys.push_back({p.domain, p.input_offset, p.output_offset});
        }
        std::sort(outputs.begin(), outputs.end());
        for (std::size_t i = 1; i < outputs.size(); ++i)
            if (outputs[i].first < outputs[i - 1].second)
                throw std::invalid_argument("overlapping RNS output intervals");
        std::vector<DomainView<U64>> host_domains;
        std::vector<rns_fused_detail::Domain> host_fused_domains;
        for (const auto& d : config.domains) {
            const U64 q = d.modulus;
            const U64 order = config.ring == RnsRing::Negacyclic ? 2 * n : n;
            if (q < 3 || q >= (U64(1) << 63) || !is_prime(q) || !d.root || d.root >= q ||
                (q - 1) % order || mod_pow(d.root, order, q) != 1 || mod_pow(d.root, order / 2, q) != q - 1)
                throw std::invalid_argument("RNS requires prime modulus <2^63 and an explicit primitive ring root");
            const U64 directed_root = config.inverse ? mod_pow(d.root, q - 2, q) : d.root;
            const U64 root = config.ring == RnsRing::Negacyclic ? multiply(directed_root, directed_root, q) : directed_root;
            const U64 scale = config.inverse ? mod_pow(n, q - 2, q) : 1;
            std::vector<U64> twiddles(n - 1), twiddle_shoup(n - 1), roots(n), root_shoup(n);
            for (unsigned stage = 0; stage < config.log_n; ++stage) {
                const std::size_t half = std::size_t(1) << stage;
                const U64 step = mod_pow(root, n / (2 * half), q);
                U64 omega = 1;
                for (std::size_t j = 0; j < half; ++j) {
                    twiddles[half - 1 + j] = omega;
                    twiddle_shoup[half - 1 + j] = shoup(omega, q);
                    omega = multiply(omega, step, q);
                }
            }
            U64 omega = 1;
            for (std::size_t j = 0; j < n; ++j) {
                roots[j] = omega;
                root_shoup[j] = shoup(omega, q);
                omega = multiply(omega, root, q);
            }
            DomainView<U64> view{};
            view.modulus = q;
            view.twiddles = upload(twiddles);
            view.twiddles_shoup = upload(twiddle_shoup);
            view.roots = upload(roots);
            view.roots_shoup = upload(root_shoup);
            view.scale = scale;
            view.scale_shoup = shoup(scale, q);
            if (config.ring == RnsRing::Negacyclic) {
                std::vector<U64> boundary(n), quotients(n);
                U64 factor = config.inverse ? scale : 1;
                for (std::size_t j = 0; j < n; ++j) {
                    boundary[j] = factor;
                    quotients[j] = shoup(factor, q);
                    factor = multiply(factor, directed_root, q);
                }
                view.boundary = upload(boundary);
                view.boundary_shoup = upload(quotients);
                // Mature CT/GS negacyclic form uses psi^bitreverse(i) directly
                // at each butterfly. Build directed roots once, outside timing.
                std::vector<U64> fused(n), fused_shoup(n);
                U64 power = 1;
                for (std::size_t j = 0; j < n; ++j) {
                    std::size_t reversed = 0, remaining = j;
                    for (unsigned bit = 0; bit < config.log_n; ++bit, remaining >>= 1)
                        reversed = (reversed << 1) | (remaining & 1);
                    fused[reversed] = power;
                    fused_shoup[reversed] = shoup(power, q);
                    power = multiply(power, directed_root, q);
                }
                const unsigned modulus_bits = 64u - static_cast<unsigned>(__builtin_clzll(q));
                U64 barrett_mu = 0;
                if (q < (U64(1) << 62)) {
                    barrett_mu = static_cast<U64>((static_cast<U128>(1) << (2 * modulus_bits + 1)) / q);
                }
                host_fused_domains.push_back({q, upload(fused), upload(fused_shoup), scale,
                                              shoup(scale, q), modulus_bits, barrett_mu});
            }
            host_domains.push_back(view);
        }
        domains = std::make_unique<Array<DomainView<U64>>>(host_domains);
        if (!host_fused_domains.empty())
            fused_domains = std::make_unique<Array<rns_fused_detail::Domain>>(host_fused_domains);
        polynomials = std::make_unique<Array<PolynomialView>>(host_polys);
        scratch = std::make_unique<Array<U64>>(config.polynomials.size() * n);
        view = {domains->data, polynomials->data, config.inverse,
                config.input_order == RnsOrder::BitReversed, config.output_order == RnsOrder::BitReversed};
        mapping.n1_log = config.log_n / 2;
        validate_mapping(mapping);
        selection.target = properties.name;
        selection.implementation = "hybrid2d-rns-native-boundary";
        selection.confidence = "unmeasured-feasible";
        selection.reason = "Explicit RNS candidate family; call tune outside timed execution for measured mapping selection";
    }
    const U64* upload(const std::vector<U64>& host) {
        storage.emplace_back(std::make_unique<Array<U64>>(host));
        return storage.back()->data;
    }
    bool alias_without_scratch() const {
        return (mapping.core_family == RnsCoreFamily::FusedCTGS ||
                mapping.core_family == RnsCoreFamily::GpuNttMerge) &&
               fused_alias_eligible();
    }
    void ensure_scratch() {
        if (!scratch)
            scratch = std::make_unique<Array<U64>>(config.polynomials.size() * n);
    }
    void release_unused_scratch(bool allow_in_place) {
        if (allow_in_place && alias_without_scratch()) scratch.reset();
        else ensure_scratch();
    }
    void check_device() const {
        int current;
        check(cudaGetDevice(&current));
        if (current != device) throw std::invalid_argument("RNS plan used on a different CUDA device");
    }
    bool gpuntt_merge_eligible() const {
        if (config.polynomials.empty() || config.domains.empty()) return false;
        std::vector<std::size_t> input_rows, output_rows;
        input_rows.reserve(config.polynomials.size());
        output_rows.reserve(config.polynomials.size());
        for (const auto& polynomial : config.polynomials) {
            if (polynomial.domain >= config.domains.size() ||
                polynomial.input_offset % n || polynomial.output_offset % n)
                return false;
            input_rows.push_back(polynomial.input_offset / n);
            output_rows.push_back(polynomial.output_offset / n);
        }
        std::sort(input_rows.begin(), input_rows.end());
        std::sort(output_rows.begin(), output_rows.end());
        return std::adjacent_find(input_rows.begin(), input_rows.end()) == input_rows.end() &&
               std::adjacent_find(output_rows.begin(), output_rows.end()) == output_rows.end();
    }
    bool gpuntt_alias_eligible() const {
        return gpuntt_merge_eligible() && fused_alias_eligible();
    }
    bool same_row_offsets() const {
        if (config.polynomials.empty()) return false;
        for (const auto& polynomial : config.polynomials)
            if (polynomial.input_offset != polynomial.output_offset) return false;
        return true;
    }
    bool fused_alias_eligible() const {
        // Equal physical rows suffice only at native CT/GS boundaries. Global
        // bit reversal can move values between CTAs, whose stores may overwrite
        // another CTA's unread input. Keep both passes on the private scratch
        // route whenever either boundary needs that permutation.
        const auto native_input = config.inverse ? RnsOrder::BitReversed : RnsOrder::Natural;
        const auto native_output = config.inverse ? RnsOrder::Natural : RnsOrder::BitReversed;
        return same_row_offsets() && config.input_order == native_input &&
               config.output_order == native_output;
    }
    void validate_mapping(const RnsMapping& m) const {
        if (m.core_family != RnsCoreFamily::Hybrid2D && m.core_family != RnsCoreFamily::FusedCTGS &&
            m.core_family != RnsCoreFamily::GpuNttMerge)
            throw std::invalid_argument("unsupported RNS core family");
        const bool fused_family = m.core_family == RnsCoreFamily::FusedCTGS ||
                                  m.core_family == RnsCoreFamily::GpuNttMerge;
        if (m.exchange != RnsExchange::Block && m.exchange != RnsExchange::DependencyScoped)
            throw std::invalid_argument("invalid RNS exchange strategy");
        if (m.exchange == RnsExchange::DependencyScoped && !fused_family)
            throw std::invalid_argument("dependency-scoped exchange currently requires a fused CT/GS core");
        const unsigned minimum_log = fused_family ? 2 : 6;
        if (m.n1_log < minimum_log || m.n1_log > 10 || m.n1_log > config.log_n ||
            config.log_n - m.n1_log < minimum_log || config.log_n - m.n1_log > 10 ||
            (m.compute_unit != ComputeUnit::Radix2 && m.compute_unit != ComputeUnit::Radix4 && m.compute_unit != ComputeUnit::Radix8) ||
            (m.cross_twiddle != CrossTwiddlePlacement::FirstPass && m.cross_twiddle != CrossTwiddlePlacement::SecondPass))
            throw std::invalid_argument("unsupported RNS mapping");
        if (fused_family &&
            (config.ring != RnsRing::Negacyclic ||
             (m.compute_unit != ComputeUnit::Radix2 && m.compute_unit != ComputeUnit::Radix4) ||
             m.cross_twiddle != CrossTwiddlePlacement::FirstPass || m.physical_boundary_traversal))
            throw std::invalid_argument("fused CT/GS requires negacyclic radix2/4 with embedded roots and native traversal");
        if (m.core_family == RnsCoreFamily::GpuNttMerge) {
            if (m.arithmetic != RnsArithmetic::Barrett || m.compute_unit != ComputeUnit::Radix2)
                throw std::invalid_argument("GPU-NTT Merge requires aligned independent RNS rows, radix2 and Barrett arithmetic");
            if (!gpuntt_merge_eligible()) {
                for (std::size_t index = 0; index < config.polynomials.size(); ++index) {
                    const auto& polynomial = config.polynomials[index];
                    const bool aligned = polynomial.input_offset % n == 0 &&
                        polynomial.output_offset % n == 0 &&
                        polynomial.domain < config.domains.size();
                    if (!aligned) {
                        std::ostringstream message;
                        message << "GPU-NTT Merge requires aligned independent rows; polynomial " << index
                                << " has input_offset=" << polynomial.input_offset
                                << ", output_offset=" << polynomial.output_offset
                                << ", domain=" << polynomial.domain << ", N=" << n;
                        throw std::invalid_argument(message.str());
                    }
                }
                throw std::invalid_argument("GPU-NTT Merge requires aligned independent RNS rows");
            }
        }
        if (m.arithmetic == RnsArithmetic::Barrett) {
            if (m.core_family != RnsCoreFamily::FusedCTGS && m.core_family != RnsCoreFamily::GpuNttMerge)
                throw std::invalid_argument("GPU-NTT-compatible Barrett currently requires fused CT/GS");
            for (const auto& domain : config.domains)
                if (domain.modulus >= (U64(1) << 62))
                    throw std::invalid_argument("GPU-NTT-compatible Barrett requires moduli below 2^62");
        }
        if (m.physical_boundary_traversal && config.input_order == RnsOrder::Natural &&
            config.output_order == RnsOrder::Natural)
            throw std::invalid_argument("physical boundary traversal requires a bit-reversed boundary");
        for (bool first : {true, false}) {
            const auto stage = pass_mapping(m, first);
            const unsigned maximum_rows = fused_family ? 128 : 32;
            if (!stage.rows_per_block || (stage.rows_per_block & (stage.rows_per_block - 1)) ||
                stage.rows_per_block > maximum_rows ||
                (stage.threads_per_block != 64 && stage.threads_per_block != 128 &&
                 stage.threads_per_block != 256 &&
                 !(fused_family && stage.threads_per_block == 32) &&
                 !(m.core_family == RnsCoreFamily::GpuNttMerge && stage.threads_per_block == 512)))
                throw std::invalid_argument("unsupported per-pass RNS rows or threads");
            const bool prefix = fused_family && config.inverse ? !first : first;
            const unsigned log = prefix ? config.log_n - m.n1_log : m.n1_log;
            const std::size_t tiles = std::size_t(1) << (config.log_n - log);
            if (stage.rows_per_block > tiles)
                throw std::invalid_argument("RNS rows exceed available independent tiles");
            const std::size_t padding = fused_family ? 0 : 1;
            const std::size_t shared = stage.rows_per_block * ((std::size_t(1) << log) + padding) * sizeof(U64);
            const std::size_t blocks_per_poly = tiles / stage.rows_per_block;
            if (shared > properties.sharedMemPerBlock || stage.threads_per_block > unsigned(properties.maxThreadsPerBlock) ||
                config.polynomials.size() > std::size_t(properties.maxGridSize[0]) / blocks_per_poly)
                throw std::invalid_argument("RNS mapping exceeds device shared memory, thread or grid limits");
        }
    }
    void execute(const U64* input, U64* output, cudaStream_t stream) {
        check_device();
        if (!input || !output) throw std::invalid_argument("null RNS input/output");
        if (mapping.core_family == RnsCoreFamily::FusedCTGS || mapping.core_family == RnsCoreFamily::GpuNttMerge) {
            // GPU-NTT's native Merge entry point is in-place. Preserve that
            // contract when the caller aliases input/output so the second
            // stage can consume the first stage's writes directly. Separate
            // buffers retain the public input-preserving behavior.
            const bool row_inplace = input == output &&
                ((mapping.core_family == RnsCoreFamily::GpuNttMerge && gpuntt_alias_eligible()) ||
                 (mapping.core_family == RnsCoreFamily::FusedCTGS && fused_alias_eligible()));
            const bool compact = mapping.core_family == RnsCoreFamily::GpuNttMerge && !same_row_offsets();
            if (row_inplace) {
                dispatch_fused(config, pass_mapping(mapping, true), true, input, output,
                               fused_domains->data, polynomials->data, true, compact, stream, warp_masks[0]);
                dispatch_fused(config, pass_mapping(mapping, false), false, output, output,
                               fused_domains->data, polynomials->data, true, compact, stream, warp_masks[1]);
            } else {
                if (!scratch)
                    throw std::invalid_argument("RNS plan was prepared for in-place execution; prepare separate placement before using distinct buffers");
                dispatch_fused(config, pass_mapping(mapping, true), true, input, scratch->data,
                               fused_domains->data, polynomials->data, false, compact, stream, warp_masks[0]);
                dispatch_fused(config, pass_mapping(mapping, false), false, scratch->data, output,
                               fused_domains->data, polynomials->data, false, compact, stream, warp_masks[1]);
            }
            return;
        }
        view.physical_boundary_traversal = mapping.physical_boundary_traversal;
        for (const bool first : {true, false}) {
            const auto stage = pass_mapping(mapping, first);
            switch (mapping.compute_unit) {
                case ComputeUnit::Radix2: dispatch_cross<ComputeUnit::Radix2>(config, stage, first, input, scratch->data, output, view, stream); break;
                case ComputeUnit::Radix4: dispatch_cross<ComputeUnit::Radix4>(config, stage, first, input, scratch->data, output, view, stream); break;
                case ComputeUnit::Radix8: dispatch_cross<ComputeUnit::Radix8>(config, stage, first, input, scratch->data, output, view, stream); break;
                default: throw std::logic_error("unresolved RNS compute unit");
            }
        }
    }
    void tune(RnsPlacement placement, cudaStream_t stream, unsigned repetitions) {
        check_device();
        require_uncaptured(stream);
        if (placement != RnsPlacement::Separate && placement != RnsPlacement::InPlace)
            throw std::invalid_argument("invalid RNS tuning placement");
        if (!repetitions || repetitions > 1000) throw std::invalid_argument("RNS tuning repetitions must be in [1,1000]");
        // Selection state commits only after all measurements succeed.
        const auto previous_mapping = mapping;
        const auto previous_warp_masks = warp_masks;
        try {
            // A previous in-place selection may have released this workspace.
            // Restore it before exploring candidates, outside execution/capture.
            ensure_scratch();
            const bool alias = placement == RnsPlacement::InPlace;
            const auto words = alias ? std::max(input_extent, output_extent) : input_extent;
            Array<U64> input(words);
            std::unique_ptr<Array<U64>> output;
            if (!alias) output = std::make_unique<Array<U64>>(output_extent);
            U64* destination = alias ? input.data : output->data;
            U64 min_q = config.domains.front().modulus;
            for (const auto& domain : config.domains) min_q = std::min(min_q, domain.modulus);
            auto reset = [&] {
                initialize_tuning_input<<<std::min<std::size_t>(65535, (words + 255) / 256), 256, 0, stream>>>(input.data, words, min_q);
                check(cudaGetLastError());
            };
            reset();
            Event start, stop;
            auto measure = [&](const RnsMapping& candidate, unsigned repeat) {
                mapping = candidate;
                warp_masks = exchange_masks(config, mapping);
                // Remapped aliased rows can change modulus ownership after a
                // transform. Replay each on fresh canonical data rather than
                // feeding one domain's residues into a different domain.
                const bool reset_each = alias && !same_row_offsets();
                const unsigned graph_repeat = reset_each ? 1 : repeat;
                if (alias) reset();
                execute(input.data, destination, stream);
                check(cudaStreamSynchronize(stream));
                Graph graph;
                // Use a private stream for capture even when caller supplied the
                // legacy default stream (which cannot be captured).
                cudaStream_t capture = nullptr;
                check(cudaStreamCreateWithFlags(&capture, cudaStreamNonBlocking));
                bool capturing = false;
                try {
                    check(cudaStreamBeginCapture(capture, cudaStreamCaptureModeThreadLocal));
                    capturing = true;
                    for (unsigned i = 0; i < graph_repeat; ++i) execute(input.data, destination, capture);
                    check(cudaStreamEndCapture(capture, &graph.graph));
                    capturing = false;
                    check(cudaGraphInstantiate(&graph.executable, graph.graph, nullptr, nullptr, 0));
                } catch (...) {
                    if (capturing) {
                        cudaGraph_t abandoned = nullptr;
                        cudaStreamEndCapture(capture, &abandoned);
                        if (abandoned) cudaGraphDestroy(abandoned);
                    }
                    cudaStreamDestroy(capture);
                    throw;
                }
                check(cudaStreamDestroy(capture));
                // First-launch graph upload belongs to preparation too.
                if (alias) reset();
                check(cudaGraphLaunch(graph.executable, stream));
                double total = 0;
                const unsigned samples = reset_each ? repeat : 1;
                for (unsigned sample = 0; sample < samples; ++sample) {
                    if (alias) reset();
                    check(cudaEventRecord(start.value, stream));
                    check(cudaGraphLaunch(graph.executable, stream));
                    check(cudaEventRecord(stop.value, stream));
                    check(cudaEventSynchronize(stop.value));
                    float elapsed = 0;
                    check(cudaEventElapsedTime(&elapsed, start.value, stop.value));
                    if (!(elapsed > 0)) throw std::runtime_error("invalid RNS candidate timing");
                    total += elapsed;
                }
                return total / (samples * graph_repeat);
            };
            std::vector<RnsMeasurement> tested;
            for (unsigned split = 6; split <= 10; ++split) {
                if (split > config.log_n || config.log_n - split < 6 || config.log_n - split > 10) continue;
                for (unsigned rows : {1u, 2u, 4u, 8u, 16u, 32u})
                    for (unsigned threads : {64u, 128u, 256u})
                        for (auto core : {ComputeUnit::Radix2, ComputeUnit::Radix4, ComputeUnit::Radix8})
                            for (auto cross : {CrossTwiddlePlacement::FirstPass, CrossTwiddlePlacement::SecondPass})
                              for (bool physical : {false, true}) {
                                RnsMapping candidate{split, rows, threads, core, cross, physical};
                                try { validate_mapping(candidate); }
                                catch (const std::invalid_argument&) { continue; }
                                tested.push_back({candidate, measure(candidate, repetitions)});
                            }
            }
            if (config.ring == RnsRing::Negacyclic)
              for (unsigned split = 2; split <= 10; ++split) {
                if (split > config.log_n || config.log_n - split < 2 || config.log_n - split > 10) continue;
                const unsigned prefix = config.log_n - split;
                auto test_fused = [&](unsigned first_rows, unsigned second_rows,
                                      unsigned first_threads, unsigned second_threads, ComputeUnit core,
                                      RnsArithmetic arithmetic,
                                      RnsCoreFamily family = RnsCoreFamily::FusedCTGS) {
                    RnsMapping candidate{split, first_rows, first_threads, core,
                        CrossTwiddlePlacement::FirstPass, false, family};
                    candidate.second_rows_per_block = second_rows == first_rows ? 0 : second_rows;
                    candidate.second_threads_per_block = second_threads == first_threads ? 0 : second_threads;
                    candidate.arithmetic = arithmetic;
                    try { validate_mapping(candidate); }
                    catch (const std::invalid_argument&) { return; }
                    for (auto exchange : {RnsExchange::Block, RnsExchange::DependencyScoped}) {
                        candidate.exchange = exchange;
                        const auto masks = exchange_masks(config, candidate);
                        // Identical barrier schedules are the same executable
                        // design point; do not spend timing budget twice.
                        if (exchange == RnsExchange::DependencyScoped && !masks[0] && !masks[1]) continue;
                        tested.push_back({candidate, measure(candidate, repetitions)});
                    }
                };
                // Preserve coupled mappings, including previous winners.
                for (unsigned rows : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u})
                        for (unsigned threads : fused_thread_candidates)
                          for (auto core : {ComputeUnit::Radix2, ComputeUnit::Radix4})
                            for (auto arithmetic : {RnsArithmetic::Shoup, RnsArithmetic::Barrett})
                                test_fused(rows, rows, threads, threads, core, arithmetic);
                // Equal tile capacity can require different rows per pass.
                // Derive these candidates from stage geometry, not GPU model
                // or application identity. This is a bounded subset of the
                // independent per-pass mapping space exposed by set_mapping.
                for (unsigned tile_log = std::max(prefix, split);
                     tile_log <= std::min(prefix + 7, split + 7); ++tile_log) {
                    const unsigned prefix_rows = 1u << (tile_log - prefix);
                    const unsigned suffix_rows = 1u << (tile_log - split);
                    const unsigned first_rows = config.inverse ? suffix_rows : prefix_rows;
                    const unsigned second_rows = config.inverse ? prefix_rows : suffix_rows;
                    for (unsigned first_threads : fused_thread_candidates)
                      for (unsigned second_threads : fused_thread_candidates) {
                        if (first_rows == second_rows && first_threads == second_threads) continue;
                        for (auto core : {ComputeUnit::Radix2, ComputeUnit::Radix4})
                            for (auto arithmetic : {RnsArithmetic::Shoup, RnsArithmetic::Barrett})
                                test_fused(first_rows, second_rows, first_threads, second_threads, core, arithmetic);
                      }
                }
                // GPU-NTT Merge provider: reproduce its 512-word tile and
                // 5/7/9-style split when each requested row is N-aligned and
                // independent. The compact kernel retains the caller's
                // physical row offsets while using a dense private scratch
                // envelope for the separate-buffer contract.
                if (gpuntt_merge_eligible() && config.log_n >= 12 && config.log_n <= 16 && split == 9) {
                    const unsigned prefix_log = config.log_n - split;
                    const unsigned first_rows = 512u >> prefix_log;
                    if (first_rows && !(first_rows & (first_rows - 1))) {
                        const unsigned rows_first = config.inverse ? 1u : first_rows;
                        const unsigned rows_second = config.inverse ? first_rows : 1u;
                        for (unsigned threads : fused_thread_candidates)
                            test_fused(rows_first, rows_second, threads, threads, ComputeUnit::Radix2,
                                       RnsArithmetic::Barrett, RnsCoreFamily::GpuNttMerge);
                        // GPU-NTT's 1024-word merge configurations use a
                        // 512-thread block for N >= 2^16. Keep this as a
                        // provider-specific candidate: the generic fused
                        // family intentionally remains capped at 256 threads.
                        if (config.log_n >= 16)
                            test_fused(rows_first, rows_second, 512, 512, ComputeUnit::Radix2,
                                       RnsArithmetic::Barrett, RnsCoreFamily::GpuNttMerge);
                    }
                }
              }
            if (tested.empty()) throw std::runtime_error("no legal RNS mapping on current device");
            std::sort(tested.begin(), tested.end(), [](const auto& a, const auto& b) { return a.kernel_ms < b.kernel_ms; });
            RnsMeasurement best{tested.front().mapping, std::numeric_limits<double>::infinity()};
            for (std::size_t i = 0; i < std::min<std::size_t>(3, tested.size()); ++i) {
                std::vector<double> confirmation;
                for (unsigned repeat = 0; repeat < 3; ++repeat)
                    confirmation.push_back(measure(tested[i].mapping, 3 * repetitions));
                std::sort(confirmation.begin(), confirmation.end());
                if (confirmation[1] < best.kernel_ms) best = {tested[i].mapping, confirmation[1]};
            }
            mapping = best.mapping;
            warp_masks = exchange_masks(config, mapping);
            selection.implementation = family_name(mapping.core_family, mapping.arithmetic);
            measurements = std::move(tested);
            selection.automatic = true;
            selection.calibrated = false;
            selection.confidence = "measured-rns-request";
            selection.predicted_kernel_ms = best.kernel_ms;
            selection.reason = "Prepared request CUDA-Graph measured across bounded RNS families/splits/radices/root placements/boundary traversals/block or dependency-scoped exchange and coupled or equal-tile-capacity per-pass rows/threads; three finalists confirmed, not exhaustive search or installation cost-model calibration";
            selection.reason += alias ? "; placement=in-place" : "; placement=separate";
            // Fused CT/GS and the GPU-NTT-compatible provider can execute the
            // native-boundary, equal-row contract in place. Their first pass
            // writes the same arena consumed by the second pass, so the
            // private two-pass scratch used by Hybrid2D is dead after tuning.
            // Keep it during candidate measurement, then release it before a
            // captured stage graph retains this plan.
            release_unused_scratch(alias);
        } catch (...) { mapping = previous_mapping; warp_masks = previous_warp_masks; throw; }
    }

    RnsPlanConfig config;
    RnsMapping mapping;
    std::array<unsigned, 2> warp_masks{};
    SelectionInfo selection;
    std::vector<RnsMeasurement> measurements;
    std::size_t n = 0, input_extent = 0, output_extent = 0;
    int device = 0;
    cudaDeviceProp properties{};
    BatchView<U64> view;
    std::vector<std::unique_ptr<Array<U64>>> storage;
    std::unique_ptr<Array<DomainView<U64>>> domains;
    std::unique_ptr<Array<rns_fused_detail::Domain>> fused_domains;
    std::unique_ptr<Array<PolynomialView>> polynomials;
    std::unique_ptr<Array<U64>> scratch;
};

RnsPlan::RnsPlan(const RnsPlanConfig& config) : impl_(std::make_unique<Impl>(config)) {}
RnsPlan::~RnsPlan() = default;
const RnsPlanConfig& RnsPlan::config() const noexcept { return impl_->config; }
const RnsMapping& RnsPlan::mapping() const noexcept { return impl_->mapping; }
const std::vector<RnsMeasurement>& RnsPlan::measurements() const noexcept { return impl_->measurements; }
const SelectionInfo& RnsPlan::selection() const noexcept { return impl_->selection; }
std::size_t RnsPlan::workspace_bytes() const noexcept {
    return impl_->scratch ? impl_->n * impl_->config.polynomials.size() * sizeof(U64) : 0;
}
void RnsPlan::set_mapping(const RnsMapping& mapping) {
    set_mapping(mapping, RnsPlacement::Separate);
}
void RnsPlan::set_mapping(const RnsMapping& mapping, RnsPlacement placement) {
    impl_->check_device();
    if (placement != RnsPlacement::Separate && placement != RnsPlacement::InPlace)
        throw std::invalid_argument("invalid RNS replay placement");
    impl_->validate_mapping(mapping);
    const auto masks = exchange_masks(impl_->config, mapping);
    auto selection = impl_->selection;
    selection.implementation = family_name(mapping.core_family, mapping.arithmetic);
    selection.automatic = false;
    selection.calibrated = false;
    selection.predicted_kernel_ms = 0;
    selection.confidence = "explicit-unmeasured";
    selection.reason = "Caller supplied a legal mapping";
    const bool discard_scratch = placement == RnsPlacement::InPlace &&
        (mapping.core_family == RnsCoreFamily::FusedCTGS || mapping.core_family == RnsCoreFamily::GpuNttMerge) &&
        impl_->fused_alias_eligible();
    // Complete allocations before publishing any new mapping or barrier schedule.
    // A failed switch must leave the previously executable plan intact.
    if (!discard_scratch) impl_->ensure_scratch();
    impl_->mapping = mapping;
    impl_->warp_masks = masks;
    impl_->selection = std::move(selection);
    impl_->measurements.clear();
    if (discard_scratch) impl_->scratch.reset();
}
void RnsPlan::tune(cudaStream_t stream, unsigned repetitions) { impl_->tune(RnsPlacement::Separate, stream, repetitions); }
void RnsPlan::tune(RnsPlacement placement, cudaStream_t stream, unsigned repetitions) { impl_->tune(placement, stream, repetitions); }
void RnsPlan::execute_async(const U64* input, U64* output, cudaStream_t stream) { impl_->execute(input, output, stream); }
} // namespace cuntt
