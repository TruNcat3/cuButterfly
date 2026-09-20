#include "mapping_selector.hpp"
#include "plan_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "generated_fft_dispatch.cuh"
#include "generated_local_selector.hpp"
#include "generated_runtime_selector.hpp"
#include "cuntt/mixed_dataflow.hpp"

#ifndef CUBUTTERFLY_HAS_CUFFTDX
#define CUBUTTERFLY_HAS_CUFFTDX 0
#endif

namespace cuntt::detail {
namespace {

void validate_selected_plan(const ButterflyConfig& config);

template <std::size_t Size>
double predict_latency(const std::array<GeneratedLatencyAnchor, Size>& anchors, std::size_t batch, std::string& confidence) {
    static_assert(Size >= 2);
    auto right = std::lower_bound(anchors.begin(), anchors.end(), batch,
                                  [](const GeneratedLatencyAnchor& anchor, std::size_t value) { return anchor.batch < value; });
    if (right != anchors.end() && right->batch == batch) {
        confidence = "calibrated-anchor";
        return right->kernel_ms;
    }

    const GeneratedLatencyAnchor* left = nullptr;
    if (right == anchors.begin()) {
        left       = &anchors[0];
        right      = anchors.begin() + 1;
        confidence = "extrapolated-low";
    } else if (right == anchors.end()) {
        left       = &anchors[Size - 2];
        right      = anchors.end() - 1;
        confidence = "extrapolated-high";
    } else {
        left       = &*(right - 1);
        confidence = "interpolated";
    }

    const double x0    = std::log2(static_cast<double>(left->batch));
    const double x1    = std::log2(static_cast<double>(right->batch));
    const double y0    = std::log(left->kernel_ms);
    const double y1    = std::log(right->kernel_ms);
    const double slope = std::clamp((y1 - y0) / (x1 - x0), 0.0, 1.0);
    return std::exp(y0 + slope * (std::log2(static_cast<double>(batch)) - x0));
}

SelectionInfo make_info(const char* implementation, std::size_t batch, double predicted_ms, std::string confidence,
                        std::size_t working_set_bytes, bool online_boundary, bool register_resident) {
    const auto device = current_device_info();
    std::ostringstream reason;
    reason << "V100 calibrated " << confidence << "; ";
    if (batch < static_cast<std::size_t>(device.multiprocessors)) {
        reason << "batch concurrency below one transform per SM";
    } else {
        reason << "batch concurrency covers all SMs";
    }
    reason << "; estimated two-buffer working set " << working_set_bytes << " bytes";
    if (online_boundary) {
        reason << "; boundary permutation is fused with the required store";
    }
    if (register_resident) {
        reason << "; local exchange is register-resident";
    }
    return {true, true, "v100-sm70", implementation, std::move(confidence), reason.str(), predicted_ms};
}

const LocalSelectorPoint* find_local_point(const ButterflyConfig& config) {
    if (kLocalSelectorPoints.empty())
        return nullptr;
    const auto device = current_device_info();
    if (std::strcmp(device.name.c_str(), kLocalSelectorDeviceName) != 0)
        return nullptr;
    if (device.compute_major != kLocalSelectorComputeMajor || device.compute_minor != kLocalSelectorComputeMinor)
        return nullptr;
#ifdef CUBUTTERFLY_LOCAL_SELECTOR_DATA
    if (kLocalSelectorMemoryBytes && device.global_memory_bytes != kLocalSelectorMemoryBytes)
        return nullptr;
#else
    if (config.inverse || config.element_stride != 1 ||
        (config.batch_stride != 0 && config.batch_stride != (std::size_t{1} << config.log_n)))
        return nullptr;
#endif
    const char* operator_name  = butterfly_operator_name(config.op);
    const char* precision_name = butterfly_precision_name(config.precision);
    const char* placement_name = butterfly_placement_name(config.placement);
    for (const auto& point : kLocalSelectorPoints) {
        if (point.log_n == config.log_n && point.batch == config.batch &&
            std::strcmp(point.operator_name, operator_name) == 0 &&
            std::strcmp(point.precision, precision_name) == 0 &&
            std::strcmp(point.placement, placement_name) == 0
#ifdef CUBUTTERFLY_LOCAL_SELECTOR_DATA
            && local_semantics_match(point, config)
#endif
            )
            return &point;
    }
    return nullptr;
}

const LocalSelectorPoint* find_local_ntt_point(const PlanConfig& config) {
    if (kLocalSelectorPoints.empty())
        return nullptr;
    const auto device = current_device_info();
#ifdef CUBUTTERFLY_LOCAL_SELECTOR_DATA
    if (kLocalSelectorMemoryBytes && device.global_memory_bytes != kLocalSelectorMemoryBytes)
        return nullptr;
#endif
    if (std::strcmp(device.name.c_str(), kLocalSelectorDeviceName) != 0 ||
        device.compute_major != kLocalSelectorComputeMajor || device.compute_minor != kLocalSelectorComputeMinor ||
        config.inverse)
        return nullptr;
    const std::string precision = std::string("word") + std::to_string(config.word_bits);
    const char* order = output_order_name(config.output_order);
    for (const auto& point : kLocalSelectorPoints) {
        if (std::strcmp(point.operator_name, "ntt") == 0 && point.log_n == config.log_n && point.batch == config.batch &&
            point.modulus == config.modulus && std::strcmp(point.precision, precision.c_str()) == 0 &&
            std::strcmp(point.placement, order) == 0)
            return &point;
    }
    return nullptr;
}

NttSelectionResult select_local_ntt_mapping(PlanConfig config, const LocalSelectorPoint& point) {
    config.auto_select = false;
    config.backend = Backend::Hybrid2D;
    config.compute_unit = ComputeUnit::Radix4;
    config.cross_twiddle_placement = CrossTwiddlePlacement::Fused;
    SelectionInfo info{true, true, kLocalSelectorTarget, point.candidate, "calibrated-local",
                       std::string("local calibrated ") + kLocalSelectorTarget + "; exact measured NTT workload point",
                       point.kernel_ms};
    return {std::move(config), std::move(info)};
}

#ifndef CUBUTTERFLY_LOCAL_SELECTOR_DATA
// Compatibility for pre-data-schema generated headers only. New calibrations
// serialize their complete mappings and never extend this name table.
bool apply_local_candidate(const char* candidate, ButterflyConfig& config, bool& online_boundary, bool& register_resident) {
    if (std::strcmp(candidate, "fft-log8-cufft") == 0 || std::strcmp(candidate, "fft-log12-cufft") == 0 ||
        std::strcmp(candidate, "fft-log16-cufft") == 0) {
        config.backend = ButterflyBackend::CuFft;
        return true;
    }
    if (std::strcmp(candidate, "fft-log8-cta-dft8") == 0) {
        config.backend = ButterflyBackend::TemporalTile;
        config.fft_core = FftCore::CtaDft8;
        config.compute_unit = ComputeUnit::Radix8;
        config.local_exchange = LocalExchange::SharedMemory;
        config.tile_threads = 128;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-scalar-hierarchical") == 0) {
        config.backend = ButterflyBackend::Hierarchical;
        config.fft_core = FftCore::Scalar;
        config.compute_unit = ComputeUnit::Radix4;
        config.local_exchange = LocalExchange::SharedMemory;
        config.tile_threads = 256;
        config.local_stages = 8;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-shared-temporal-radix4") == 0) {
        config.backend = ButterflyBackend::TemporalTile;
        config.fft_core = FftCore::Scalar;
        config.compute_unit = ComputeUnit::Radix4;
        config.local_exchange = LocalExchange::SharedMemory;
        config.tile_threads = 256;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-cufftdx-direct") == 0) {
        config.backend = ButterflyBackend::TemporalTile;
        config.fft_core = FftCore::CufftDxDirect;
        config.tile_threads = 1024;
        register_resident = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-cufftdx-direct512") == 0) {
        config.backend = ButterflyBackend::TemporalTile;
        config.fft_core = FftCore::CufftDxDirect;
        config.tile_threads = 512;
        register_resident = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-cufftdx-online") == 0) {
        config.backend = ButterflyBackend::OnlineReorder;
        config.fft_core = FftCore::CufftDxBlock;
        config.local_stages = 6;
        config.reorder_columns = 1;
        config.prefix_threads = 512;
        config.suffix_threads = 512;
        config.prefix_ept = 8;
        config.suffix_ept = 8;
        online_boundary = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log16-cufftdx-online") == 0) {
        config.backend = ButterflyBackend::OnlineReorder;
        config.fft_core = FftCore::CufftDxBlock;
        config.local_stages = 8;
        config.reorder_columns = 1;
        config.prefix_threads = 128;
        config.suffix_threads = 128;
        config.prefix_ept = 8;
        config.suffix_ept = 8;
        config.cross_twiddle = CrossTwiddleMode::Recurrence;
        online_boundary = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log16-fp64-cufftdx-online") == 0) {
        config.backend = ButterflyBackend::OnlineReorder;
        config.fft_core = FftCore::CufftDxBlock;
        config.local_stages = 8;
        config.reorder_columns = 1;
        config.prefix_threads = 256;
        config.suffix_threads = 256;
        config.prefix_ept = 8;
        config.suffix_ept = 8;
        config.cross_twiddle = CrossTwiddleMode::Table;
        online_boundary = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log18-cufftdx-online-inplace") == 0 ||
        std::strcmp(candidate, "fft-log18-cufftdx-online-inplace-batch64") == 0 ||
        std::strcmp(candidate, "fft-log18-cufftdx-online-inplace-s10") == 0 ||
        std::strcmp(candidate, "fft-log20-cufftdx-online-inplace") == 0 ||
        std::strcmp(candidate, "fft-log20-cufftdx-online-inplace-batch16") == 0 ||
        std::strcmp(candidate, "fft-log20-cufftdx-online-inplace-s8") == 0) {
        config.backend = ButterflyBackend::OnlineReorder;
        config.fft_core = FftCore::CufftDxBlock;
        config.local_stages = std::strcmp(candidate, "fft-log18-cufftdx-online-inplace-batch64") == 0 ? 9 :
                              (std::strcmp(candidate, "fft-log18-cufftdx-online-inplace") == 0 ? 8 :
                               (std::strcmp(candidate, "fft-log20-cufftdx-online-inplace-s8") == 0 ? 8 : 10));
        config.reorder_columns = 1;
        config.prefix_threads = 256;
        config.suffix_threads = 256;
        config.prefix_ept = config.local_stages == 9 ? 8 : 16;
        config.suffix_ept = config.local_stages == 9 ? 8 : 16;
        config.cross_twiddle = CrossTwiddleMode::Recurrence;
        online_boundary = true;
        return true;
    }
    if (std::strcmp(candidate, "fft-log12-cufftdx-resident") == 0) {
        config.backend = ButterflyBackend::OnlineReorder;
        config.fft_core = FftCore::CufftDxResident;
        config.local_stages = 6;
        config.reorder_columns = 1;
        config.tile_threads = 512;
        online_boundary = true;
        register_resident = true;
        return true;
    }
    if (std::strcmp(candidate, "fwht-log15-warp-register") == 0) {
        config.backend = ButterflyBackend::TemporalTile;
        config.local_exchange = LocalExchange::WarpRegister;
        config.compute_unit = ComputeUnit::Radix2;
        config.tile_threads = 256;
        register_resident = true;
        return true;
    }
    if (std::strcmp(candidate, "subset-zeta-log12-hierarchical") == 0 ||
        std::strcmp(candidate, "superset-zeta-log12-hierarchical") == 0 ||
        std::strcmp(candidate, "xor-zeta-log12-hierarchical") == 0) {
        config.backend = ButterflyBackend::Hierarchical;
        config.compute_unit = ComputeUnit::Radix4;
        config.tile_threads = 256;
        config.local_stages = 8;
        return true;
    }
    if (std::strcmp(candidate, "structured-2x2-log12-hierarchical") == 0) {
        config.backend = ButterflyBackend::Hierarchical;
        config.compute_unit = ComputeUnit::Radix4;
        config.tile_threads = 256;
        config.local_stages = 8;
        config.stage_matrices = {{0.9238795, -0.3826834, 0.3826834, 0.9238795}};
        return true;
    }
    return false;
}

#endif

ButterflySelectionResult select_local_mapping(ButterflyConfig config, const LocalSelectorPoint& point) {
    config.auto_select = false;
    bool online_boundary = false;
    bool register_resident = false;
#ifdef CUBUTTERFLY_LOCAL_SELECTOR_DATA
    if (!apply_measured_local_candidate(point, config))
        throw std::invalid_argument("local calibration has no serialized mapping for this point");
    online_boundary = config.backend == ButterflyBackend::OnlineReorder;
    register_resident = config.local_exchange == LocalExchange::WarpRegister;
#else
    if (!apply_local_candidate(point.candidate, config, online_boundary, register_resident))
        throw std::invalid_argument(std::string("local calibration candidate is not understood by this library: ") + point.candidate);
#endif
    std::ostringstream reason;
    reason << "local calibrated " << kLocalSelectorTarget << "; exact measured workload point";
    if (online_boundary)
        reason << "; boundary permutation is fused with the selected local path";
    if (register_resident)
        reason << "; local exchange is register/resident";
    validate_selected_plan(config);
    SelectionInfo info{true, true, kLocalSelectorTarget, point.candidate, "calibrated-local", reason.str(), point.kernel_ms};
    return {std::move(config), std::move(info)};
}

void reject_explicit_butterfly_mapping(const ButterflyConfig& config) {
    if (!config.stage_partition.empty() || !config.segment_mappings.empty() || !config.boundaries.empty() ||
        !config.execution_group_mappings.empty()) {
        throw std::invalid_argument("auto-select owns butterfly decomposition and physical mappings");
    }
}

void validate_selected_plan(const ButterflyConfig& config) {
    const auto plan = make_dataflow_plan(config);
    const auto status = check_lowering(plan);
    if (status != LoweringStatus::Supported && status != LoweringStatus::RequiresBackendValidation) {
        throw std::invalid_argument(std::string("selected butterfly plan cannot be lowered: ") +
                                    lowering_status_name(status));
    }
    // Imported codelets have their own per-segment launch/resource validation.
    if (plan.dispatch == DataflowDispatch::ExternalFft)
        return;
    const auto hardware = query_hardware_resource_model();
    const auto estimate = estimate_resources(plan, hardware);
    if (!estimate.feasible) {
        throw std::invalid_argument("selected butterfly plan exceeds active GPU resources: " + estimate.reason);
    }
}

}  // namespace

ButterflySelectionResult select_legacy_butterfly_mapping(ButterflyConfig config) {
    reject_explicit_butterfly_mapping(config);
    const auto device = current_device_info();
    if (const auto* point = find_local_point(config))
        return select_local_mapping(std::move(config), *point);
    if (device.compute_major != 7 || device.compute_minor != 0) {
        throw std::invalid_argument("automatic mapping selection has no validated local point for this GPU/workload; rerun install-time calibration or choose an explicit mapping");
    }
    reject_explicit_butterfly_mapping(config);
    const std::size_t contiguous_stride = std::size_t{1} << config.log_n;
    if (config.inverse || config.element_stride != 1 ||
        (config.batch_stride != 0 && config.batch_stride != contiguous_stride)) {
        throw std::invalid_argument("automatic butterfly selection currently requires forward contiguous semantics");
    }

    config.auto_select       = false;
    config.normalize_inverse = false;
    std::string confidence;
    double predicted_ms = 0.0;
    const char* implementation = nullptr;
    bool online_boundary = false;
    bool register_resident = false;

    if (config.op == ButterflyOperator::Fft) {
        if (!CUBUTTERFLY_HAS_CUFFTDX) {
            throw std::invalid_argument("automatic FFT selection requires a build with CUBUTTERFLY_ENABLE_CUFFTDX=ON");
        }
        if (config.precision != ButterflyPrecision::Fp32 || config.placement != ButterflyPlacement::InPlace) {
            throw std::invalid_argument("automatic FFT selection requires FP32 forward, contiguous, in-place semantics");
        }
        config.local_exchange = LocalExchange::SharedMemory;
        config.fft_core       = FftCore::CufftDxBlock;
        if (config.log_n == 8) {
            config.backend      = ButterflyBackend::TemporalTile;
            config.tile_threads = 32;
            predicted_ms        = predict_latency(kFft8Anchors, config.batch, confidence);
            implementation     = "cuButterfly-cuFFTDx";
        } else if (config.log_n == 14) {
            config.backend      = ButterflyBackend::TemporalTile;
            config.fft_core     = FftCore::CufftDxDirect;
            config.tile_threads = 1024;
            predicted_ms        = predict_latency(kFft14Anchors, config.batch, confidence);
            implementation     = "cuButterfly-cuFFTDx-direct";
            register_resident  = true;
        } else if (config.log_n == 18 || config.log_n == 20) {
            GeneratedFftMapping mapping{};
            if (!select_generated_fft_mapping(config.log_n, config.batch, mapping)) {
                throw std::invalid_argument("no generated V100 FFT mapping exists for this length and batch");
            }
            config.backend         = ButterflyBackend::OnlineReorder;
            config.local_stages    = mapping.local_stages;
            config.reorder_columns = 1;
            config.prefix_threads  = mapping.prefix_threads;
            config.suffix_threads  = mapping.suffix_threads;
            config.prefix_ept      = mapping.prefix_ept;
            config.suffix_ept      = mapping.suffix_ept;
            config.cross_twiddle   = mapping.recurrence_twiddle ? CrossTwiddleMode::Recurrence : CrossTwiddleMode::Table;
            config.direct_boundary = mapping.prefix_tiled_transpose
                                         ? DirectBoundary::PrefixTiledTranspose
                                         : (mapping.suffix_tiled_transpose ? DirectBoundary::TiledTranspose : DirectBoundary::Strided);
            predicted_ms    = config.log_n == 18 ? predict_latency(kFft18Anchors, config.batch, confidence)
                                                 : predict_latency(kFft20Anchors, config.batch, confidence);
            implementation = "cuButterfly-cuFFTDx-online";
            online_boundary = true;
        } else {
            throw std::invalid_argument("automatic FFT selection is calibrated only for logN 8, 14, 18, and 20");
        }
    } else if (config.op == ButterflyOperator::Fwht) {
        if (config.precision != ButterflyPrecision::Fp32 || config.placement != ButterflyPlacement::OutOfPlace) {
            throw std::invalid_argument("automatic FWHT selection requires FP32 forward, contiguous, out-of-place semantics");
        }
        if (config.log_n == 8) {
            config.backend        = ButterflyBackend::TemporalTile;
            config.local_exchange = LocalExchange::WarpRegister;
            config.compute_unit   = ComputeUnit::Radix2;
            config.tile_threads   = 32;
            predicted_ms          = predict_latency(kFwht8WarpAnchors, config.batch, confidence);
            implementation       = "cuButterfly-warp-register";
            register_resident    = true;
        } else if (config.log_n == 15 && config.batch <= 2) {
            config.backend         = ButterflyBackend::OnlineReorder;
            config.compute_unit    = ComputeUnit::Radix4;
            config.tile_threads    = 256;
            config.local_stages    = 8;
            config.reorder_columns = 1;
            predicted_ms           = predict_latency(kFwht15OnlineAnchors, config.batch, confidence);
            implementation        = "cuButterfly-online-radix4";
            online_boundary       = true;
        } else if (config.log_n == 15) {
            config.backend        = ButterflyBackend::TemporalTile;
            config.local_exchange = LocalExchange::WarpRegister;
            config.compute_unit   = ComputeUnit::Radix2;
            config.tile_threads   = 256;
            predicted_ms          = predict_latency(kFwht15WarpAnchors, config.batch, confidence);
            implementation       = "cuButterfly-warp-register";
            register_resident    = true;
        } else if (config.log_n == 20) {
            config.backend         = ButterflyBackend::OnlineReorder;
            config.compute_unit    = ComputeUnit::Radix4;
            config.tile_threads    = 256;
            config.local_stages    = 10;
            config.reorder_columns = 1;
            predicted_ms           = predict_latency(kFwht20OnlineAnchors, config.batch, confidence);
            implementation        = "cuButterfly-online-radix4";
            online_boundary       = true;
        } else {
            throw std::invalid_argument("automatic FWHT selection is calibrated only for logN 8, 15, and 20");
        }
    } else {
        if (config.op != ButterflyOperator::XorZeta) {
            throw std::invalid_argument("automatic selection is not calibrated for this butterfly operator; select an explicit backend");
        }
        if (config.precision != ButterflyPrecision::Uint32 || config.placement != ButterflyPlacement::OutOfPlace) {
            throw std::invalid_argument("automatic XOR-zeta selection requires uint32 forward, contiguous, out-of-place semantics");
        }
        config.compute_unit = ComputeUnit::Radix4;
        if (config.log_n == 8) {
            config.backend        = ButterflyBackend::TemporalTile;
            config.local_exchange = LocalExchange::SharedMemory;
            config.tile_threads   = 128;
            predicted_ms          = predict_latency(kXor8Radix4Anchors, config.batch, confidence);
            implementation       = "cuButterfly-radix4";
        } else if (config.log_n == 20) {
            config.backend         = ButterflyBackend::OnlineReorder;
            config.tile_threads    = 256;
            config.local_stages    = 10;
            config.reorder_columns = 1;
            predicted_ms           = predict_latency(kXor20OnlineAnchors, config.batch, confidence);
            implementation        = "cuButterfly-online-radix4";
            online_boundary       = true;
        } else {
            throw std::invalid_argument("automatic XOR-zeta selection is calibrated only for logN 8 and 20");
        }
    }

    const std::size_t batch = config.batch;
    const std::size_t element_bytes = config.op == ButterflyOperator::Fft
                                          ? (config.precision == ButterflyPrecision::Fp64 ? sizeof(Complex64) : sizeof(Complex32))
                                          : (config.precision == ButterflyPrecision::Fp64 ? sizeof(double) : sizeof(std::uint32_t));
    const std::size_t working_set = batch * (std::size_t{1} << config.log_n) * element_bytes * 2;
    auto info = make_info(implementation, batch, predicted_ms, std::move(confidence), working_set,
                          online_boundary, register_resident);
    validate_selected_plan(config);
    return {std::move(config), std::move(info)};
}

NttSelectionResult select_legacy_ntt_mapping(PlanConfig config) {
    const auto device = current_device_info();
    if (device.compute_major != 7 || device.compute_minor != 0) {
        if (const auto* point = find_local_ntt_point(config))
            return select_local_ntt_mapping(std::move(config), *point);
        throw std::invalid_argument("automatic NTT selection has no validated local point for this GPU/workload; rerun install-time calibration or choose an explicit mapping");
    }
    if (config.inverse) {
        throw std::invalid_argument("automatic NTT selection currently requires a forward transform");
    }
    config.auto_select = false;
    std::string confidence;
    double predicted_ms = 0.0;
    const char* implementation = nullptr;
    bool online_boundary = false;

    if (config.log_n == 12 && config.word_bits == 32 && config.output_order == OutputOrder::Natural && config.modulus < (1ULL << 30)) {
        config.backend                  = Backend::Hybrid2D;
        config.compute_unit             = ComputeUnit::Radix4;
        config.cross_twiddle_placement  = CrossTwiddlePlacement::Fused;
        predicted_ms                    = predict_latency(kNtt12Radix4Anchors, config.batch, confidence);
        implementation                 = "cuNTT-Hybrid2D-radix4";
        online_boundary                = true;
    } else if (config.log_n == 16 && config.word_bits == 64 && config.output_order == OutputOrder::Natural) {
        config.backend                  = Backend::Hybrid2D;
        config.compute_unit             = ComputeUnit::Radix4;
        config.cross_twiddle_placement  = CrossTwiddlePlacement::Fused;
        predicted_ms                    = predict_latency(kNtt16Radix4Anchors, config.batch, confidence);
        implementation                 = "cuNTT-Hybrid2D-radix4";
        online_boundary                = true;
    } else if (config.log_n == 20 && config.word_bits == 64 && config.output_order == OutputOrder::BitReversed) {
        config.backend     = Backend::CompactStage;
        config.compute_unit = ComputeUnit::Radix2;
        predicted_ms       = predict_latency(kNtt20CompactAnchors, config.batch, confidence);
        implementation    = "cuNTT-compact-stage";
        online_boundary   = true;
    } else {
        throw std::invalid_argument("NTT workload is outside the calibrated V100 runtime selector table");
    }

    const std::size_t batch = config.batch;
    const std::size_t working_set = batch * (std::size_t{1} << config.log_n) * (config.word_bits / 8) * 2;
    auto info = make_info(implementation, batch, predicted_ms, std::move(confidence), working_set, online_boundary, false);
    return {std::move(config), std::move(info)};
}

ButterflySelectionResult select_butterfly_mapping(ButterflyConfig config) {
    reject_explicit_butterfly_mapping(config);
    if (auto selected = registry_butterfly_mapping(config)) return std::move(*selected);
    try { return select_legacy_butterfly_mapping(config); }
    catch (const std::invalid_argument&) { return select_portable_butterfly(config); }
}

NttSelectionResult select_ntt_mapping(PlanConfig config) {
    if (auto selected = registry_ntt_mapping(config)) return std::move(*selected);
    try { return select_legacy_ntt_mapping(config); }
    catch (const std::invalid_argument&) { return select_portable_ntt(config); }
}

}  // namespace cuntt::detail
