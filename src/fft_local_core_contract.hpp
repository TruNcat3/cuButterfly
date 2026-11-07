#pragma once

#include <cuntt/butterfly.hpp>
#include <optional>
#include <stdexcept>

namespace cuntt::detail {

// Precision of the arithmetic implemented by our cuFFTDx adapters, not the
// width of arbitrary user storage. In particular BF16 is never FP32 here.
// Size, packing, launch geometry and compiled resources are separate checks.
inline std::optional<unsigned> cufftdx_arithmetic_precision_bits(const ButterflyConfig& c) {
    if (c.op != ButterflyOperator::Fft || c.accumulation != ButterflyAccumulation::Native)
        return std::nullopt;
    switch (c.precision) {
    case ButterflyPrecision::Fp16:
        if (c.inverse && c.normalize_inverse) return std::nullopt;
        return 16U;
    case ButterflyPrecision::Fp32: return 32U;
    case ButterflyPrecision::Fp64: return 64U;
    default: return std::nullopt;
    }
}

// A cuFFTDx half register packs two independent transforms. EPT counts
// complex half2 values, whereas public N and batch count logical transforms.
inline bool fp16_local_block(const ButterflyConfig& c) {
    return c.op == ButterflyOperator::Fft && c.precision == ButterflyPrecision::Fp16 &&
           c.fft_core == FftCore::CufftDxBlock && c.backend == ButterflyBackend::TemporalTile;
}

struct Fp16LocalBlockShape {
    unsigned lanes, transforms_per_cta;
};

inline Fp16LocalBlockShape fp16_local_block_shape(const ButterflyConfig& c) {
    if (!fp16_local_block(c) || c.accumulation != ButterflyAccumulation::Native)
        throw std::invalid_argument("FP16 local cuFFTDx requires native FP16 FFT arithmetic");
    if (c.inverse && c.normalize_inverse)
        throw std::invalid_argument("FP16 local cuFFTDx currently preserves unnormalized inverse semantics; normalized inverse uses the scalar contract");
    if (c.log_n == 0 || c.log_n > 20 || !c.prefix_ept ||
        (c.prefix_ept & (c.prefix_ept - 1)) || c.prefix_ept < 2 || c.prefix_ept > 32 ||
        c.prefix_ept > (1U << c.log_n))
        throw std::invalid_argument("invalid FP16 local FFT size or EPT");
    const auto lanes = (1U << c.log_n) / c.prefix_ept;
    // This is CUDA launch geometry, not a resource-feasibility guarantee.
    // Device limits and compiled provider resources are checked separately.
    if (c.tile_threads < 32 || c.tile_threads > 1024 || c.tile_threads % 32 ||
        c.tile_threads % lanes)
        throw std::invalid_argument("FP16 local FFT threads must contain integral half2 transform pairs");
    if (c.local_exchange != LocalExchange::SharedMemory || c.shared_layout != SharedLayout::Linear ||
        (c.compute_unit != ComputeUnit::Auto && c.compute_unit != ComputeUnit::Radix2) ||
        c.complex_multiply != ComplexMultiply::FourMul || c.stage_overlap ||
        (!c.stage_partition.empty() && c.stage_partition != std::vector<std::uint32_t>{c.log_n}) ||
        !c.boundaries.empty() || !c.local_stage_partitions.empty() || !c.exchange_chunks.empty() ||
        !c.segment_mappings.empty() || !c.execution_group_mappings.empty())
        throw std::invalid_argument("FP16 local cuFFTDx requires one unsplit provider group with provider-owned arithmetic/layout");
    return {lanes, 2U * (c.tile_threads / lanes)};
}

} // namespace cuntt::detail
