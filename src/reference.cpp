#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

#include "cuntt/ntt.hpp"

namespace cuntt {
namespace {

std::uint64_t mod_mul(std::uint64_t a, std::uint64_t b, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % modulus);
}

void bit_reverse(std::vector<std::uint64_t>& values) {
    std::size_t j = 0;
    for (std::size_t i = 1; i < values.size(); ++i) {
        std::size_t bit = values.size() >> 1;
        while ((j & bit) != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(values[i], values[j]);
        }
    }
}

}  // namespace

const char* backend_name(Backend backend) noexcept {
    switch (backend) {
        case Backend::Baseline:
            return "baseline";
        case Backend::Tile256:
            return "tile256";
        case Backend::Hybrid2D:
            return "hybrid2d";
        case Backend::CompactStage:
            return "compact-stage";
        case Backend::StagePipeline:
            return "stage-pipeline";
        case Backend::HybridDataflow:
            return "hybrid-dataflow";
        case Backend::HierarchicalBarrier:
            return "hierarchical-barrier";
        case Backend::HierarchicalDataflow:
            return "hierarchical-dataflow";
    }
    return "unknown";
}

Backend parse_backend(const std::string& name) {
    if (name == "baseline") {
        return Backend::Baseline;
    }
    if (name == "tile256") {
        return Backend::Tile256;
    }
    if (name == "hybrid2d" || name == "merge256") {
        return Backend::Hybrid2D;
    }
    if (name == "compact-stage") {
        return Backend::CompactStage;
    }
    if (name == "stage-pipeline") {
        return Backend::StagePipeline;
    }
    if (name == "hybrid-dataflow" || name == "graph-stream") {
        return Backend::HybridDataflow;
    }
    if (name == "hierarchical-barrier" || name == "hierarchical-v06") {
        return Backend::HierarchicalBarrier;
    }
    if (name == "hierarchical-dataflow" || name == "hierarchical-graph") {
        return Backend::HierarchicalDataflow;
    }
    throw std::invalid_argument("unknown backend: " + name);
}

const char* ntt_subgraph_core_name(NttSubgraphCore core) noexcept {
    switch (core) {
        case NttSubgraphCore::Radix2: return "radix2";
        case NttSubgraphCore::Radix4: return "radix4";
        case NttSubgraphCore::Radix8: return "radix8";
        case NttSubgraphCore::DataflowRadix4: return "dataflow-radix4";
        case NttSubgraphCore::HomogeneousRadix4: return "homogeneous-radix4";
        case NttSubgraphCore::HomogeneousWarpRadix2: return "homogeneous-warp-radix2";
        case NttSubgraphCore::HomogeneousWarp256Radix2: return "homogeneous-warp256-radix2";
        case NttSubgraphCore::HomogeneousWarp256StaticRadix2: return "homogeneous-warp256-static-radix2";
        case NttSubgraphCore::HomogeneousWarp256StaticIoRadix2: return "homogeneous-warp256-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp256CoefficientReuseStaticIoRadix2: return "homogeneous-warp256-coefficient-reuse-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp128StaticIoRadix2: return "homogeneous-warp128-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp128CoefficientReuseStaticIoRadix2: return "homogeneous-warp128-coefficient-reuse-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo: return "homogeneous-warp128-vector-radix4-static-io";
        case NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6StaticIo: return "homogeneous-warp128-vector-radix4-packed-stage6-static-io";
        case NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo: return "homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io";
        case NttSubgraphCore::HomogeneousWarp128PacketSharedRadix4StaticIo: return "homogeneous-warp128-packet-shared-radix4-static-io";
        case NttSubgraphCore::HomogeneousWarp64StaticIoRadix2: return "homogeneous-warp64-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2: return "homogeneous-warp128-pipeline-static-io-radix2";
        case NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2: return "homogeneous-warp128-cooperative-static-io-radix2";
        case NttSubgraphCore::Hybrid2DRadix4: return "hybrid2d-radix4";
        case NttSubgraphCore::ApptPipeline: return "appt-pipeline";
        case NttSubgraphCore::ApptOnline: return "appt-online";
        case NttSubgraphCore::ApptOnlineRadix4: return "appt-online-radix4";
        case NttSubgraphCore::ApptOnlineFusedTail: return "appt-online-fused-tail";
        case NttSubgraphCore::ApptOnlineSplitTail: return "appt-online-split-tail";
        case NttSubgraphCore::ApptOnlineRegisterTail: return "appt-online-register-tail";
        case NttSubgraphCore::ApptOnlineRegisterTailWarp: return "appt-online-register-tail-warp";
        case NttSubgraphCore::ApptOnlineRegisterTailColumnWarp: return "appt-online-register-tail-column-warp";
        case NttSubgraphCore::ApptOnlineRegisterTailRadix8: return "appt-online-register-tail-radix8";
        case NttSubgraphCore::ApptOnlineRegisterTailGrouped: return "appt-online-register-tail-grouped";
        case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal: return "appt-online-register-tail-grouped-writer-final";
        case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalDataTime: return "appt-online-register-tail-grouped-writer-final-data-time";
        case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResident: return "appt-online-register-tail-grouped-writer-final-resident";
        case NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter: return "appt-online-register-tail-grouped-writer-final-resident-quarter";
    }
    return "unknown";
}

NttSubgraphCore parse_ntt_subgraph_core(const std::string& name) {
    if (name == "radix2") return NttSubgraphCore::Radix2;
    if (name == "radix4") return NttSubgraphCore::Radix4;
    if (name == "radix8") return NttSubgraphCore::Radix8;
    if (name == "dataflow-radix4") return NttSubgraphCore::DataflowRadix4;
    if (name == "homogeneous-radix4") return NttSubgraphCore::HomogeneousRadix4;
    if (name == "homogeneous-warp-radix2")
        return NttSubgraphCore::HomogeneousWarpRadix2;
    if (name == "homogeneous-warp256-radix2")
        return NttSubgraphCore::HomogeneousWarp256Radix2;
    if (name == "homogeneous-warp256-static-radix2")
        return NttSubgraphCore::HomogeneousWarp256StaticRadix2;
    if (name == "homogeneous-warp256-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp256StaticIoRadix2;
    if (name == "homogeneous-warp256-coefficient-reuse-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp256CoefficientReuseStaticIoRadix2;
    if (name == "homogeneous-warp128-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp128StaticIoRadix2;
    if (name == "homogeneous-warp128-coefficient-reuse-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp128CoefficientReuseStaticIoRadix2;
    if (name == "homogeneous-warp128-vector-radix4-static-io")
        return NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo;
    if (name == "homogeneous-warp128-vector-radix4-packed-stage6-static-io")
        return NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6StaticIo;
    if (name == "homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io")
        return NttSubgraphCore::HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
    if (name == "homogeneous-warp128-packet-shared-radix4-static-io")
        return NttSubgraphCore::HomogeneousWarp128PacketSharedRadix4StaticIo;
    if (name == "homogeneous-warp64-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp64StaticIoRadix2;
    if (name == "homogeneous-warp128-pipeline-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2;
    if (name == "homogeneous-warp128-cooperative-static-io-radix2")
        return NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2;
    if (name == "hybrid2d-radix4") return NttSubgraphCore::Hybrid2DRadix4;
    if (name == "appt-pipeline" || name == "appt-register" || name == "macro-tile")
        return NttSubgraphCore::ApptPipeline;
    if (name == "appt-online" || name == "appt-stream")
        return NttSubgraphCore::ApptOnline;
    if (name == "appt-online-radix4" || name == "appt-stream-radix4")
        return NttSubgraphCore::ApptOnlineRadix4;
    if (name == "appt-online-fused-tail" || name == "appt-fused-tail")
        return NttSubgraphCore::ApptOnlineFusedTail;
    if (name == "appt-online-split-tail" || name == "appt-split-tail")
        return NttSubgraphCore::ApptOnlineSplitTail;
    if (name == "appt-online-register-tail" || name == "appt-register-tail")
        return NttSubgraphCore::ApptOnlineRegisterTail;
    if (name == "appt-online-register-tail-warp" ||
        name == "appt-register-tail-warp")
        return NttSubgraphCore::ApptOnlineRegisterTailWarp;
    if (name == "appt-online-register-tail-column-warp" ||
        name == "appt-register-tail-column-warp")
        return NttSubgraphCore::ApptOnlineRegisterTailColumnWarp;
    if (name == "appt-online-register-tail-radix8" ||
        name == "appt-register-tail-radix8")
        return NttSubgraphCore::ApptOnlineRegisterTailRadix8;
    if (name == "appt-online-register-tail-grouped" ||
        name == "appt-register-tail-grouped")
        return NttSubgraphCore::ApptOnlineRegisterTailGrouped;
    if (name == "appt-online-register-tail-grouped-writer-final" ||
        name == "appt-register-tail-grouped-writer-final")
        return NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal;
    if (name == "appt-online-register-tail-grouped-writer-final-data-time" ||
        name == "appt-register-tail-grouped-writer-final-data-time")
        return NttSubgraphCore::
            ApptOnlineRegisterTailGroupedWriterFinalDataTime;
    if (name == "appt-online-register-tail-grouped-writer-final-resident" ||
        name == "appt-register-tail-grouped-writer-final-resident")
        return NttSubgraphCore::
            ApptOnlineRegisterTailGroupedWriterFinalResident;
    if (name ==
            "appt-online-register-tail-grouped-writer-final-resident-quarter" ||
        name == "appt-register-tail-grouped-writer-final-resident-quarter")
        return NttSubgraphCore::
            ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
    throw std::invalid_argument("unknown NTT subgraph core: " + name);
}

const char* boundary_storage_name(BoundaryStorage storage) noexcept {
    switch (storage) {
        case BoundaryStorage::FullScratch: return "full-scratch";
        case BoundaryStorage::Ring: return "ring";
        case BoundaryStorage::ResidentFused: return "resident-fused";
    }
    return "unknown";
}

BoundaryStorage parse_boundary_storage(const std::string& name) {
    if (name == "full-scratch" || name == "full") return BoundaryStorage::FullScratch;
    if (name == "ring") return BoundaryStorage::Ring;
    if (name == "resident-fused" || name == "resident" || name == "fused")
        return BoundaryStorage::ResidentFused;
    throw std::invalid_argument("unknown boundary storage: " + name);
}

const char* packet_readiness_mode_name(PacketReadinessMode mode) noexcept {
    switch (mode) {
        case PacketReadinessMode::PerPacket: return "per-packet";
        case PacketReadinessMode::WaveBitmap: return "wave-bitmap";
    }
    return "unknown";
}

PacketReadinessMode parse_packet_readiness_mode(const std::string& name) {
    if (name == "per-packet" || name == "packet") {
        return PacketReadinessMode::PerPacket;
    }
    if (name == "wave-bitmap" || name == "bitmap") {
        return PacketReadinessMode::WaveBitmap;
    }
    throw std::invalid_argument("unknown packet readiness mode: " + name);
}

const char* packet_compute_layout_name(PacketComputeLayout layout) noexcept {
    switch (layout) {
        case PacketComputeLayout::InterleavedRows: return "interleaved-rows";
        case PacketComputeLayout::WarpRows: return "warp-rows";
    }
    return "unknown";
}

PacketComputeLayout parse_packet_compute_layout(const std::string& name) {
    if (name == "interleaved-rows" || name == "interleaved") {
        return PacketComputeLayout::InterleavedRows;
    }
    if (name == "warp-rows" || name == "warp-row") {
        return PacketComputeLayout::WarpRows;
    }
    throw std::invalid_argument("unknown packet compute layout: " + name);
}

const char* dataflow_layout_name(DataflowLayout layout) noexcept {
    switch (layout) {
        case DataflowLayout::HermesXor:
            return "hermes-xor";
        case DataflowLayout::Linear:
            return "linear";
    }
    return "unknown";
}

DataflowLayout parse_dataflow_layout(const std::string& name) {
    if (name == "hermes-xor" || name == "xor") {
        return DataflowLayout::HermesXor;
    }
    if (name == "linear") {
        return DataflowLayout::Linear;
    }
    throw std::invalid_argument("unknown dataflow layout: " + name);
}

const char* dataflow_state_mode_name(DataflowStateMode mode) noexcept {
    switch (mode) {
        case DataflowStateMode::InPlace:
            return "inplace";
        case DataflowStateMode::PingPong:
            return "ping-pong";
    }
    return "unknown";
}

DataflowStateMode parse_dataflow_state_mode(const std::string& name) {
    if (name == "inplace" || name == "in-place") {
        return DataflowStateMode::InPlace;
    }
    if (name == "ping-pong" || name == "pingpong") {
        return DataflowStateMode::PingPong;
    }
    throw std::invalid_argument("unknown dataflow state mode: " + name);
}

const char* stage_handoff_name(StageHandoff handoff) noexcept {
    switch (handoff) {
        case StageHandoff::Atomic:
            return "atomic";
        case StageHandoff::NamedBarrier:
            return "named-barrier";
    }
    return "unknown";
}

StageHandoff parse_stage_handoff(const std::string& name) {
    if (name == "atomic") {
        return StageHandoff::Atomic;
    }
    if (name == "named-barrier") {
        return StageHandoff::NamedBarrier;
    }
    throw std::invalid_argument("unknown stage handoff: " + name);
}

const char* output_order_name(OutputOrder order) noexcept {
    switch (order) {
        case OutputOrder::Natural:
            return "natural";
        case OutputOrder::BitReversed:
            return "bit-reversed";
        case OutputOrder::ApptStatic:
            return "appt-static";
    }
    return "unknown";
}

OutputOrder parse_output_order(const std::string& name) {
    if (name == "natural") {
        return OutputOrder::Natural;
    }
    if (name == "bit-reversed") {
        return OutputOrder::BitReversed;
    }
    if (name == "appt-static" || name == "appt_static") {
        return OutputOrder::ApptStatic;
    }
    throw std::invalid_argument("unknown output order: " + name);
}

const char* input_order_name(InputOrder order) noexcept {
    switch (order) {
        case InputOrder::Natural:
            return "natural";
        case InputOrder::ApptStatic:
            return "appt-static";
    }
    return "unknown";
}

InputOrder parse_input_order(const std::string& name) {
    if (name == "natural") {
        return InputOrder::Natural;
    }
    if (name == "appt-static" || name == "appt_static") {
        return InputOrder::ApptStatic;
    }
    throw std::invalid_argument("unknown input order: " + name);
}

namespace {

void validate_appt_layout(const ApptLayoutInfo& layout) {
    if (layout.log_n != 20 ||
        layout.stage_partition != std::vector<std::uint32_t>{7, 7, 6} ||
        (layout.fragment_width != 8 && layout.fragment_width != 16 &&
         layout.fragment_width != 32) ||
        !layout.xor_permutation) {
        throw std::invalid_argument("unsupported APPT static-layout descriptor");
    }
}

}  // namespace

std::uint64_t appt_static_index(std::uint64_t natural_index,
                                const ApptLayoutInfo& layout) {
    validate_appt_layout(layout);
    const std::uint64_t n = 1ULL << layout.log_n;
    if (natural_index >= n) {
        throw std::out_of_range("APPT natural index exceeds one transform");
    }
    const std::uint64_t a = natural_index >> 14;
    const std::uint64_t d = (natural_index >> 7) & 127U;
    const std::uint64_t c = natural_index & 127U;
    const std::uint64_t mask = layout.fragment_width - 1U;
    return (a << 14) | (c << 7) |
           ((d & ~mask) | ((d ^ c) & mask));
}

std::uint64_t appt_natural_index(std::uint64_t static_index,
                                 const ApptLayoutInfo& layout) {
    validate_appt_layout(layout);
    const std::uint64_t n = 1ULL << layout.log_n;
    if (static_index >= n) {
        throw std::out_of_range("APPT static index exceeds one transform");
    }
    const std::uint64_t a = static_index >> 14;
    const std::uint64_t c = (static_index >> 7) & 127U;
    const std::uint64_t stored_d = static_index & 127U;
    const std::uint64_t mask = layout.fragment_width - 1U;
    const std::uint64_t d = (stored_d & ~mask) |
                            ((stored_d ^ c) & mask);
    return (a << 14) | (d << 7) | c;
}

const char* compute_unit_name(ComputeUnit unit) noexcept {
    switch (unit) {
        case ComputeUnit::Auto:
            return "auto";
        case ComputeUnit::Radix2:
            return "radix2";
        case ComputeUnit::Radix4:
            return "radix4";
        case ComputeUnit::Radix8:
            return "radix8";
    }
    return "unknown";
}

ComputeUnit parse_compute_unit(const std::string& name) {
    if (name == "auto") {
        return ComputeUnit::Auto;
    }
    if (name == "radix2") {
        return ComputeUnit::Radix2;
    }
    if (name == "radix4") {
        return ComputeUnit::Radix4;
    }
    if (name == "radix8") {
        return ComputeUnit::Radix8;
    }
    throw std::invalid_argument("unknown compute unit: " + name);
}

const char* hierarchical_core_name(HierarchicalCore core) noexcept {
    switch (core) {
        case HierarchicalCore::DataflowRadix4:
            return "dataflow-radix4";
        case HierarchicalCore::Hybrid2DRadix4:
            return "hybrid2d-radix4";
    }
    return "unknown";
}

HierarchicalCore parse_hierarchical_core(const std::string& name) {
    if (name == "dataflow-radix4") {
        return HierarchicalCore::DataflowRadix4;
    }
    if (name == "hybrid2d-radix4") {
        return HierarchicalCore::Hybrid2DRadix4;
    }
    throw std::invalid_argument("unknown hierarchical core: " + name);
}

const char* cross_twiddle_placement_name(CrossTwiddlePlacement placement) noexcept {
    switch (placement) {
        case CrossTwiddlePlacement::FirstPass:
            return "first";
        case CrossTwiddlePlacement::SecondPass:
            return "second";
        case CrossTwiddlePlacement::Fused:
            return "fused";
        case CrossTwiddlePlacement::FusedBarrett:
            return "fused-barrett";
    }
    return "unknown";
}

CrossTwiddlePlacement parse_cross_twiddle_placement(const std::string& name) {
    if (name == "first") {
        return CrossTwiddlePlacement::FirstPass;
    }
    if (name == "second") {
        return CrossTwiddlePlacement::SecondPass;
    }
    if (name == "fused") {
        return CrossTwiddlePlacement::Fused;
    }
    if (name == "fused-barrett") {
        return CrossTwiddlePlacement::FusedBarrett;
    }
    throw std::invalid_argument("unknown cross-twiddle placement: " + name);
}

const char* modular_multiply_name(ModularMultiply multiply) noexcept {
    switch (multiply) {
        case ModularMultiply::Shoup:
            return "shoup";
        case ModularMultiply::Barrett:
            return "barrett";
    }
    return "unknown";
}

ModularMultiply parse_modular_multiply(const std::string& name) {
    if (name == "shoup") {
        return ModularMultiply::Shoup;
    }
    if (name == "barrett") {
        return ModularMultiply::Barrett;
    }
    throw std::invalid_argument("unknown modular multiply: " + name);
}

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    if (modulus < 2) {
        throw std::invalid_argument("modulus must be at least 2");
    }
    std::uint64_t result = 1;
    base %= modulus;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exponent >>= 1;
    }
    return result;
}

bool is_prime(std::uint64_t value) {
    if (value < 2) {
        return false;
    }
    constexpr std::array<std::uint64_t, 12> kSmallPrimes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (const auto prime : kSmallPrimes) {
        if (value % prime == 0) {
            return value == prime;
        }
    }

    std::uint64_t odd_part     = value - 1;
    std::uint32_t power_of_two = 0;
    while ((odd_part & 1U) == 0) {
        odd_part >>= 1;
        ++power_of_two;
    }

    constexpr std::array<std::uint64_t, 7> kWitnesses = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
    for (const auto witness : kWitnesses) {
        if (witness % value == 0) {
            continue;
        }
        std::uint64_t result = mod_pow(witness, odd_part, value);
        if (result == 1 || result == value - 1) {
            continue;
        }
        bool probably_prime = false;
        for (std::uint32_t round = 1; round < power_of_two; ++round) {
            result = mod_mul(result, result, value);
            if (result == value - 1) {
                probably_prime = true;
                break;
            }
        }
        if (!probably_prime) {
            return false;
        }
    }
    return true;
}

std::uint64_t find_primitive_power_of_two_root(std::uint32_t log_n, std::uint64_t modulus) {
    if (log_n == 0 || log_n >= 63) {
        throw std::invalid_argument("log_n must be in [1, 62]");
    }
    if (!is_prime(modulus)) {
        throw std::invalid_argument("modulus must be prime");
    }
    const std::uint64_t n = 1ULL << log_n;
    if ((modulus - 1) % n != 0) {
        throw std::invalid_argument("modulus - 1 is not divisible by the requested NTT length");
    }
    for (std::uint64_t candidate = 2; candidate < (1ULL << 20); ++candidate) {
        const std::uint64_t root = mod_pow(candidate, (modulus - 1) / n, modulus);
        if (mod_pow(root, n, modulus) == 1 && mod_pow(root, n >> 1, modulus) != 1) {
            return root;
        }
    }
    throw std::runtime_error("could not find a primitive power-of-two root");
}

void reference_ntt(std::vector<std::uint64_t>& values, std::uint64_t modulus, bool inverse) {
    const std::size_t n = values.size();
    if (n < 2 || (n & (n - 1)) != 0) {
        throw std::invalid_argument("NTT length must be a power of two of at least two");
    }
    if (modulus >= (1ULL << 63)) {
        throw std::invalid_argument("cuNTT currently requires modulus < 2^63");
    }

    std::uint32_t log_n = 0;
    for (std::size_t size = n; size > 1; size >>= 1) {
        ++log_n;
    }
    std::uint64_t root = find_primitive_power_of_two_root(log_n, modulus);
    if (inverse) {
        root = mod_pow(root, modulus - 2, modulus);
    }

    for (auto& value : values) {
        value %= modulus;
    }
    bit_reverse(values);

    for (std::size_t length = 2; length <= n; length <<= 1) {
        const std::size_t   half      = length >> 1;
        const std::uint64_t step_root = mod_pow(root, n / length, modulus);
        for (std::size_t start = 0; start < n; start += length) {
            std::uint64_t omega = 1;
            for (std::size_t offset = 0; offset < half; ++offset) {
                const std::size_t   left = start + offset;
                const std::uint64_t u    = values[left];
                const std::uint64_t v    = mod_mul(values[left + half], omega, modulus);
                const std::uint64_t sum  = u + v;
                values[left]             = sum >= modulus ? sum - modulus : sum;
                values[left + half]      = u >= v ? u - v : modulus + u - v;
                omega                    = mod_mul(omega, step_root, modulus);
            }
        }
    }

    if (inverse) {
        const std::uint64_t inverse_n = mod_pow(static_cast<std::uint64_t>(n), modulus - 2, modulus);
        for (auto& value : values) {
            value = mod_mul(value, inverse_n, modulus);
        }
    }
}

}  // namespace cuntt
