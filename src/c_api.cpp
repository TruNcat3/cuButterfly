#include "cubutterfly/cubutterfly.h"
#include "cubutterfly/mapping.hpp"

#include "cuntt/butterfly.hpp"
#include "cuntt/ntt.hpp"
#include "bounded_candidate_inventory.hpp"
#include "bluestein_fft.hpp"
#include "bluestein_ntt.hpp"
#include "direct_fft.hpp"
#include "direct_real_fft.hpp"
#include "general_kernels.hpp"
#include "plan_registry.hpp"
#include "plan_search_trace.hpp"
#include "platform_tuning.hpp"
#include "xd_real_fft.hpp"
#include "xd_axis_fft.hpp"
#include "xd_vkfft_axis.hpp"
#include "xd_register_fft.hpp"
#include "xd_local_fft.hpp"
#include "xd_pair_core.hpp"
#include "xd_resident.hpp"
#include "semantic_contract.hpp"
#include "semantic_numeric_kernels.hpp"
#include "cufftdx_provider.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <sstream>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#ifndef CUBUTTERFLY_HAS_CUFFTDX
#define CUBUTTERFLY_HAS_CUFFTDX 0
#endif

struct cubutterflyHandle {
    cudaStream_t stream = nullptr;
    cubutterflyLogCallback_t logger = nullptr;
    void* logger_data = nullptr;
    std::string cache_path;
    bool suppress_profile_miss = false;
};

struct cubutterflyPlan {
    enum class Kind { Butterfly, Ntt, DirectFft, BluesteinFft, BluesteinNtt } kind = Kind::Butterfly;
    std::unique_ptr<cuntt::ButterflyPlan> butterfly;
    std::unique_ptr<cuntt::Plan> ntt;
    std::vector<std::unique_ptr<cuntt::detail::BluesteinFftPlan>> bluestein_axes;
    std::vector<std::unique_ptr<cuntt::detail::BluesteinNttPlan>> bluestein_ntt_axes;
    std::unique_ptr<cuntt::detail::DirectFftPlan> direct_fft;
    std::unique_ptr<cuntt::detail::DirectRealFftPlan> direct_real_fft;
    std::vector<std::unique_ptr<cuntt::ButterflyPlan>> butterfly_axes;
    std::vector<std::unique_ptr<cuntt::Plan>> ntt_axes;
    cubutterflyDescriptor descriptor;
    bool composite = false;
    bool xd = false;
    std::unique_ptr<cuntt::detail::XdRealFftPlan> xd_fft;
    std::unique_ptr<cuntt::detail::XdResidentPlan> xd_resident;
    std::unique_ptr<cubutterflyPlan> xd_selected;
    std::unique_ptr<cubutterflyPlan> boundary_child;
    // Explicit arithmetic/real-spectrum adapters retain their own mapping and
    // full execution cost, independently of order/layout boundary handling.
    std::unique_ptr<cubutterflyPlan> numeric_child;
    bool full_spectrum_real = false;
    std::vector<std::unique_ptr<cubutterflyPlan>> arbitrary_children;
    std::size_t boundary_input_bytes = 0;
    std::size_t boundary_output_bytes = 0;
    std::string xd_mapping;
    bool direct_input_boundary = false;
    bool direct_output_boundary = false;
    bool fused_input_boundary = false;
    cubutterflySelectionSource_t selection_source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
    std::string algorithm;
    std::string reason;
    std::vector<std::size_t> physical_extents;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t element_bytes = 0;
    std::size_t packed_bytes = 0;
    std::size_t composite_buffers = 0;
    std::size_t workspace_bytes = 0;
    void* workspace = nullptr;
    // Generic rank-N execution has a deterministic axis permutation schedule.
    // Keep its host-side records with the immutable plan so each execute only
    // submits kernels and does not rebuild vectors or search axis positions.
    struct XdPermutationStep {
        bool required = false;
        cuntt::detail::GeneralNdLayout input_layout{};
        std::vector<std::uint32_t> permutation;
    };
    std::vector<XdPermutationStep> xd_permutation_steps;
    cuntt::detail::GeneralNdLayout xd_final_input_layout{};
    std::vector<std::uint32_t> xd_final_permutation;
    bool xd_final_permutation_required = false;
    // Child plans are bound to the handle stream when a plan is prepared.
    // Keep the last propagated stream so repeated asynchronous executions do
    // not rebind every axis plan on the host.
    cudaStream_t bound_stream = nullptr;
};

namespace {

thread_local std::string last_error;

class unsupported_error : public std::runtime_error {
  public:
    explicit unsupported_error(const char* message) : std::runtime_error(message) {}
    explicit unsupported_error(const std::string& message) : std::runtime_error(message) {}
};

template <typename Function>
cubutterflyStatus_t protect(Function&& function) noexcept {
    try {
        last_error.clear();
        function();
        return CUBUTTERFLY_STATUS_SUCCESS;
    } catch (const cuntt::detail::ProviderImplementationMissing& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_NOT_SUPPORTED;
    } catch (const std::invalid_argument& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    } catch (const std::overflow_error& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    } catch (const std::bad_alloc& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_ALLOC_FAILED;
    } catch (const unsupported_error& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_NOT_SUPPORTED;
    } catch (const nlohmann::json::exception& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    } catch (const std::logic_error& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_INTERNAL_ERROR;
    } catch (const std::runtime_error& e) {
        last_error = e.what();
        return CUBUTTERFLY_STATUS_BACKEND_ERROR;
    } catch (...) {
        last_error = "unrecognized exception";
        return CUBUTTERFLY_STATUS_INTERNAL_ERROR;
    }
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::invalid_argument(message);
}

// Runtime measurement can retain the complete legal inventory while probing a
// bounded, deterministic subset.  Zero keeps the historical exhaustive mode.
std::size_t search_budget() {
    const char* value = std::getenv("CUBUTTERFLY_SEARCH_BUDGET");
    if (!value || !*value) return 0;
    if (*value == '-') throw std::invalid_argument("CUBUTTERFLY_SEARCH_BUDGET must be nonnegative");
    std::size_t used = 0;
    const auto budget = std::stoull(value, &used);
    if (used != std::string(value).size())
        throw std::invalid_argument("CUBUTTERFLY_SEARCH_BUDGET must be an integer");
    return budget;
}

auto search_family_key(const cuntt::ButterflyConfig& candidate) {
    return std::make_tuple(candidate.backend, candidate.compute_unit,
                           candidate.local_exchange, candidate.stage_overlap, candidate.fft_core);
}

auto search_family_key(const cuntt::PlanConfig& candidate) {
    return std::make_tuple(candidate.backend, candidate.compute_unit,
                           candidate.hierarchical_core, candidate.stage_overlap);
}

template <typename Candidate>
auto search_organization_key(const Candidate& candidate) {
    return std::make_tuple(candidate.backend, candidate.stage_overlap);
}

template <typename Candidate>
std::vector<Candidate> bounded_search_inventory(const std::vector<Candidate>& inventory,
                                                std::size_t budget) {
    if (!budget || inventory.size() <= budget) return inventory;
    const auto indices = cuntt::detail::bounded_candidate_indices(
        inventory, budget, [](const Candidate& candidate) { return search_family_key(candidate); },
        [](const Candidate& candidate) { return search_organization_key(candidate); });
    std::vector<Candidate> selected;
    selected.reserve(indices.size());
    for (const auto index : indices) selected.push_back(inventory[index]);
    return selected;
}

std::size_t checked_product(std::size_t a, std::size_t b) {
    require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
            "shape or storage product overflows size_t");
    return a * b;
}

std::string workload_key(const cubutterflyDescriptor& descriptor);
bool is_power_of_two(std::size_t value);

struct Rank2FftPolicy {
    bool compact = false;
    bool measure = false;
    std::size_t max_length = 128;
    std::vector<std::string> candidates{"temporal-radix4", "cufft"};
    std::string mode = "auto";
    std::string token = "auto";
};

Rank2FftPolicy rank2_fft_policy(cubutterflyAlgorithmPolicy_t algorithm_policy) {
    Rank2FftPolicy policy;
    const char* mode_value = std::getenv("CUBUTTERFLY_RANK2_FFT_POLICY");
    const std::string mode = mode_value == nullptr || *mode_value == '\0' ? "auto" : mode_value;
    require(mode == "auto" || mode == "compact" || mode == "off" || mode == "measure",
            "CUBUTTERFLY_RANK2_FFT_POLICY must be auto, compact, measure, or off");
    policy.mode = mode;
    policy.measure = mode == "measure";
    policy.compact = mode == "compact" || mode == "measure" ||
                     (mode == "auto" && algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE);
    policy.token = mode;

    if (const char* limit_value = std::getenv("CUBUTTERFLY_RANK2_FFT_MAX_LENGTH")) {
        require(*limit_value != '-', "CUBUTTERFLY_RANK2_FFT_MAX_LENGTH must be nonnegative");
        std::size_t consumed = 0;
        unsigned long long parsed = 0;
        try {
            parsed = std::stoull(limit_value, &consumed);
        } catch (const std::exception&) {
            throw std::invalid_argument(
                "CUBUTTERFLY_RANK2_FFT_MAX_LENGTH must be a positive power of two");
        }
        require(consumed != 0 && limit_value[consumed] == '\0' &&
                    parsed <= std::numeric_limits<std::size_t>::max() && parsed != 0 &&
                    is_power_of_two(static_cast<std::size_t>(parsed)),
                "CUBUTTERFLY_RANK2_FFT_MAX_LENGTH must be a positive power of two");
        policy.max_length = static_cast<std::size_t>(parsed);
    }

    if (const char* candidates_value = std::getenv("CUBUTTERFLY_RANK2_FFT_CANDIDATES")) {
        std::vector<std::string> parsed;
        std::stringstream stream(candidates_value);
        std::string candidate;
        while (std::getline(stream, candidate, ',')) {
            if (!candidate.empty()) parsed.push_back(candidate);
        }
        require(!parsed.empty(), "CUBUTTERFLY_RANK2_FFT_CANDIDATES must not be empty");
        for (const auto& value : parsed)
            require(value == "temporal-radix2" || value == "temporal-radix4" || value == "cufft",
                    "unsupported rank-2 FFT candidate");
        policy.candidates = std::move(parsed);
    }
    policy.token += "-max" + std::to_string(policy.max_length);
    for (const auto& candidate : policy.candidates) policy.token += "-" + candidate;
    return policy;
}

bool is_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::uint32_t integer_log2(std::size_t value) {
    require(is_power_of_two(value), "extent must be a power of two");
    std::uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

std::size_t next_power_of_two(std::size_t value) {
    require(value != 0, "extent must be positive");
    if (is_power_of_two(value))
        return value;
    require(value <= (static_cast<std::size_t>(-1) >> 1) + 1, "power-of-two extent overflows size_t");
    --value;
    for (std::size_t shift = 1; shift < sizeof(value) * 8; shift <<= 1)
        value |= value >> shift;
    return value + 1;
}

void log_event(cubutterflyHandle_t handle, cubutterflyLogLevel_t level,
               const char* event, const std::string& message) {
    if (handle != nullptr && handle->logger != nullptr)
        handle->logger(level, event, message.c_str(), handle->logger_data);
}

std::size_t element_bytes(cubutterflyDataType_t type) {
    switch (type) {
        case CUBUTTERFLY_DATA_FP16:
        case CUBUTTERFLY_DATA_BF16:
            return 2;
        case CUBUTTERFLY_DATA_FP32:
        case CUBUTTERFLY_DATA_COMPLEX_FP16:
        case CUBUTTERFLY_DATA_COMPLEX_BF16:
        case CUBUTTERFLY_DATA_UINT32:
            return 4;
        case CUBUTTERFLY_DATA_FP64:
        case CUBUTTERFLY_DATA_COMPLEX_FP32:
        case CUBUTTERFLY_DATA_UINT64:
            return 8;
        case CUBUTTERFLY_DATA_COMPLEX_FP64:
            return 16;
    }
    throw std::invalid_argument("unknown data type");
}

std::size_t strided_elements(const cubutterflyDescriptor& descriptor, bool input,
                             const std::vector<std::size_t>& extents) {
    const auto& strides = input ? descriptor.input_strides : descriptor.output_strides;
    const auto batch_stride = input ? descriptor.input_batch_stride : descriptor.output_batch_stride;
    require(strides.size() == descriptor.rank, "stride rank does not match shape rank");
    std::size_t extent = 1;
    for (std::uint32_t axis = 0; axis < descriptor.rank; ++axis) {
        require(strides[axis] != 0, "strides must be positive");
        const std::size_t add = checked_product(extents[axis] - 1, strides[axis]);
        require(extent <= static_cast<std::size_t>(-1) - add, "strided extent overflows size_t");
        extent += add;
    }
    if (descriptor.batch > 1) {
        // Non-overlap is checked by the semantic normalizer, including
        // interleaved batches. A batch stride need not exceed the axis span.
        const std::size_t add = checked_product(descriptor.batch - 1, batch_stride);
        require(extent <= static_cast<std::size_t>(-1) - add, "batch extent overflows size_t");
        extent += add;
    }
    return extent;
}

void resolve_default_strides(cubutterflyDescriptor& descriptor,
                             const std::vector<std::size_t>& input_extents,
                             const std::vector<std::size_t>& output_extents) {
    auto resolve = [&](std::vector<std::size_t>& strides, std::size_t& batch_stride,
                       const std::vector<std::size_t>& extents) {
        if (strides.empty()) {
            strides.assign(descriptor.rank, 1);
            for (std::uint32_t axis = descriptor.rank; axis-- > 1;)
                strides[axis - 1] = checked_product(strides[axis], extents[axis]);
        }
        require(strides.size() == descriptor.rank, "stride rank does not match shape rank");
        if (batch_stride == 0) {
            batch_stride = 1;
            for (std::uint32_t axis = 0; axis < descriptor.rank; ++axis) {
                const auto add = checked_product(extents[axis] - 1, strides[axis]);
                require(batch_stride <= std::numeric_limits<std::size_t>::max() - add,
                        "batch stride overflows size_t");
                batch_stride += add;
            }
        }
    };
    resolve(descriptor.input_strides, descriptor.input_batch_stride, input_extents);
    resolve(descriptor.output_strides, descriptor.output_batch_stride, output_extents);
}

bool packed_layout(const cubutterflyDescriptor& descriptor) {
    std::size_t stride = 1;
    for (std::uint32_t axis = descriptor.rank; axis-- > 0;) {
        if (descriptor.input_strides[axis] != stride || descriptor.output_strides[axis] != stride)
            return false;
        stride = checked_product(stride, descriptor.extents[axis]);
    }
    return descriptor.input_batch_stride == stride && descriptor.output_batch_stride == stride;
}

bool packed_side(const cubutterflyDescriptor& descriptor, bool input,
                 const std::vector<std::size_t>& extents) {
    const auto& strides = input ? descriptor.input_strides : descriptor.output_strides;
    const auto batch_stride = input ? descriptor.input_batch_stride
                                    : descriptor.output_batch_stride;
    std::size_t stride = 1;
    for (std::uint32_t axis = descriptor.rank; axis-- > 0;) {
        if (strides[axis] != stride)
            return false;
        stride = checked_product(stride, extents[axis]);
    }
    return batch_stride == stride;
}

cuntt::ButterflyOperator butterfly_operator(cubutterflyOperator_t op) {
    switch (op) {
        case CUBUTTERFLY_OPERATOR_FFT: return cuntt::ButterflyOperator::Fft;
        case CUBUTTERFLY_OPERATOR_FWHT: return cuntt::ButterflyOperator::Fwht;
        case CUBUTTERFLY_OPERATOR_SUBSET_ZETA: return cuntt::ButterflyOperator::SubsetZeta;
        case CUBUTTERFLY_OPERATOR_SUPERSET_ZETA: return cuntt::ButterflyOperator::SupersetZeta;
        case CUBUTTERFLY_OPERATOR_STRUCTURED_2X2: return cuntt::ButterflyOperator::Structured2x2;
        case CUBUTTERFLY_OPERATOR_NTT: break;
    }
    throw std::invalid_argument("NTT is not a common butterfly plan operator");
}

cuntt::ButterflyPrecision butterfly_precision(const cubutterflyDescriptor& descriptor) {
    switch (descriptor.storage) {
        case CUBUTTERFLY_DATA_FP16:
        case CUBUTTERFLY_DATA_COMPLEX_FP16: return cuntt::ButterflyPrecision::Fp16;
        case CUBUTTERFLY_DATA_BF16:
        case CUBUTTERFLY_DATA_COMPLEX_BF16: return cuntt::ButterflyPrecision::Bf16;
        case CUBUTTERFLY_DATA_FP32:
        case CUBUTTERFLY_DATA_COMPLEX_FP32: return cuntt::ButterflyPrecision::Fp32;
        case CUBUTTERFLY_DATA_FP64:
        case CUBUTTERFLY_DATA_COMPLEX_FP64: return cuntt::ButterflyPrecision::Fp64;
        case CUBUTTERFLY_DATA_UINT32: return cuntt::ButterflyPrecision::Uint32;
        case CUBUTTERFLY_DATA_UINT64: return cuntt::ButterflyPrecision::Uint64;
    }
    throw std::invalid_argument("data type is not valid for a common butterfly plan");
}

void validate_storage(const cubutterflyDescriptor& descriptor) {
    const bool complex = descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP16 ||
                         descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_BF16 ||
                         descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 ||
                         descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP64;
    const bool real = descriptor.storage == CUBUTTERFLY_DATA_FP16 ||
                      descriptor.storage == CUBUTTERFLY_DATA_BF16 ||
                      descriptor.storage == CUBUTTERFLY_DATA_FP32 ||
                      descriptor.storage == CUBUTTERFLY_DATA_FP64;
    if (descriptor.fft_type != CUBUTTERFLY_FFT_C2C) {
        require(descriptor.op == CUBUTTERFLY_OPERATOR_FFT && real,
                "R2C/C2R require FFT and real storage precision");
        require((descriptor.fft_type == CUBUTTERFLY_FFT_R2C) ==
                    (descriptor.direction == CUBUTTERFLY_DIRECTION_FORWARD),
                "R2C is forward and C2R is inverse");
    } else if (descriptor.op == CUBUTTERFLY_OPERATOR_FFT)
        require(complex, "C2C FFT requires a complex storage type");
    else if (descriptor.op == CUBUTTERFLY_OPERATOR_NTT)
        require(descriptor.storage == CUBUTTERFLY_DATA_UINT32 || descriptor.storage == CUBUTTERFLY_DATA_UINT64,
                "NTT requires uint32 or uint64 storage");
    else if (descriptor.op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || descriptor.op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA)
        require(descriptor.storage == CUBUTTERFLY_DATA_UINT32 || descriptor.storage == CUBUTTERFLY_DATA_UINT64,
                "zeta transforms require uint32 or uint64 storage");
    else
        require(real, "real butterfly operator requires a real floating-point storage type");
}

void log_profile_miss(cubutterflyHandle_t handle, const cubutterflyDescriptor& descriptor,
                      const std::string& reason, const std::string& algorithm) {
    if (handle->suppress_profile_miss) return;
    const auto device = cuntt::current_device_info();
    std::ostringstream message;
    message << "no verified mapping for device='" << device.name << "' sm_"
            << device.compute_major << device.compute_minor << " memory_bytes=" << device.global_memory_bytes
            << "; " << workload_key(descriptor) << "; " << reason << "; using " << algorithm
            << "; set MEASURE with a cache path to calibrate this workload";
    log_event(handle, CUBUTTERFLY_LOG_WARNING, "workload-profile-miss", message.str());
}

class ScopedProfileMissSuppression {
  public:
    explicit ScopedProfileMissSuppression(cubutterflyHandle_t handle)
        : handle_(handle), previous_(handle->suppress_profile_miss) {
        handle_->suppress_profile_miss = true;
    }
    ScopedProfileMissSuppression(const ScopedProfileMissSuppression&) = delete;
    ScopedProfileMissSuppression& operator=(const ScopedProfileMissSuppression&) = delete;
    ~ScopedProfileMissSuppression() { handle_->suppress_profile_miss = previous_; }

  private:
    cubutterflyHandle_t handle_;
    bool previous_;
};

cuntt::SelectionInfo choose_butterfly_mapping(cuntt::ButterflyConfig& config) {
    auto selected = cuntt::detail::select_butterfly_mapping(config);
    config = std::move(selected.config);
    return selected.info;
}

cubutterflySelectionSource_t mapping_selection_source(const cuntt::SelectionInfo& info) {
    if (info.calibrated) return CUBUTTERFLY_SELECTION_MEASURED_TABLE;
    return info.predicted_kernel_ms > 0 ? CUBUTTERFLY_SELECTION_STATIC_MODEL
                                       : CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK;
}

void copy_text(const std::string& value, char* destination, std::size_t* bytes) {
    require(bytes != nullptr, "size query pointer must be non-null");
    const std::size_t required = value.size() + 1;
    if (destination == nullptr) {
        *bytes = required;
        return;
    }
    require(*bytes >= required, "destination string buffer is too small");
    std::memcpy(destination, value.c_str(), required);
    *bytes = required;
}

void execute_butterfly(cuntt::ButterflyPlan& plan, const void* input, void* output,
                       cubutterflyDataType_t storage) {
    switch (storage) {
        case CUBUTTERFLY_DATA_FP16:
            plan.execute_async(static_cast<const cuntt::Fp16*>(input), static_cast<cuntt::Fp16*>(output)); return;
        case CUBUTTERFLY_DATA_BF16:
            plan.execute_async(static_cast<const cuntt::Bf16*>(input), static_cast<cuntt::Bf16*>(output)); return;
        case CUBUTTERFLY_DATA_FP32:
            plan.execute_async(static_cast<const float*>(input), static_cast<float*>(output)); return;
        case CUBUTTERFLY_DATA_FP64:
            plan.execute_async(static_cast<const double*>(input), static_cast<double*>(output)); return;
        case CUBUTTERFLY_DATA_COMPLEX_FP16:
            plan.execute_async(static_cast<const cuntt::Complex16*>(input), static_cast<cuntt::Complex16*>(output)); return;
        case CUBUTTERFLY_DATA_COMPLEX_BF16:
            plan.execute_async(static_cast<const cuntt::ComplexBf16*>(input), static_cast<cuntt::ComplexBf16*>(output)); return;
        case CUBUTTERFLY_DATA_COMPLEX_FP32:
            plan.execute_async(static_cast<const cuntt::Complex32*>(input), static_cast<cuntt::Complex32*>(output)); return;
        case CUBUTTERFLY_DATA_COMPLEX_FP64:
            plan.execute_async(static_cast<const cuntt::Complex64*>(input), static_cast<cuntt::Complex64*>(output)); return;
        case CUBUTTERFLY_DATA_UINT32:
            plan.execute_async(static_cast<const std::uint32_t*>(input), static_cast<std::uint32_t*>(output)); return;
        case CUBUTTERFLY_DATA_UINT64:
            plan.execute_async(static_cast<const std::uint64_t*>(input), static_cast<std::uint64_t*>(output)); return;
    }
    throw std::invalid_argument("invalid butterfly execution type");
}

cuntt::ButterflyConfig make_axis_butterfly_config(const cubutterflyDescriptor& descriptor,
                                                   std::uint32_t axis, std::size_t length,
                                                   std::size_t batch) {
    cuntt::ButterflyConfig config;
    config.op = butterfly_operator(descriptor.op);
    config.precision = butterfly_precision(descriptor);
    const bool low_precision = config.precision == cuntt::ButterflyPrecision::Fp16 ||
                               config.precision == cuntt::ButterflyPrecision::Bf16;
    config.accumulation = low_precision && descriptor.compute == CUBUTTERFLY_COMPUTE_FP32
                              ? cuntt::ButterflyAccumulation::Fp32
                              : cuntt::ButterflyAccumulation::Native;
    config.placement = cuntt::ButterflyPlacement::InPlace;
    config.log_n = integer_log2(length);
    config.batch = batch;
    config.batch_stride = length;
    config.element_stride = 1;
    config.inverse = descriptor.direction == CUBUTTERFLY_DIRECTION_INVERSE;
    config.normalize_inverse = descriptor.normalize_inverse;
    config.auto_allocate_workspace = false;
    if (config.op == cuntt::ButterflyOperator::Structured2x2) {
        require(axis < descriptor.stage_matrices.size() && !descriptor.stage_matrices[axis].empty(),
                "structured-2x2 requires matrices for every executed axis");
        for (const auto& matrix : descriptor.stage_matrices[axis])
            config.stage_matrices.push_back({matrix.m00, matrix.m01, matrix.m10, matrix.m11});
    }
    return config;
}

cuntt::PlanConfig make_axis_ntt_config(const cubutterflyDescriptor& descriptor,
                                       std::size_t length, std::size_t batch) {
    cuntt::PlanConfig config;
    config.log_n = integer_log2(length);
    config.batch = batch;
    config.modulus = descriptor.modulus;
    config.inverse = descriptor.direction == CUBUTTERFLY_DIRECTION_INVERSE;
    config.word_bits = descriptor.word_bits;
    config.input_order = descriptor.input_ntt_layout ==
                                 CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC
                             ? cuntt::InputOrder::ApptStatic
                             : cuntt::InputOrder::Natural;
    config.output_order = descriptor.output_ntt_layout ==
                                  CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC
                              ? cuntt::OutputOrder::ApptStatic
                              : descriptor.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED
                              ? cuntt::OutputOrder::BitReversed
                              : cuntt::OutputOrder::Natural;
    config.auto_allocate_workspace = false;
    // Backend and compute-unit selection belongs to the shared registry and
    // portable candidate projection.  Do not seed the public C API with a
    // size-only implementation heuristic before that selector runs.
    return config;
}

std::size_t align_16(std::size_t value) {
    require(value <= static_cast<std::size_t>(-1) - 15, "workspace size overflows size_t");
    return (value + 15) & ~std::size_t{15};
}

cuntt::detail::GeneralNdLayout packed_nd_layout(const std::vector<std::size_t>& extents,
                                                std::size_t batch) {
    require(!extents.empty() && extents.size() <= cuntt::detail::kGeneralMaxRank,
            "xD lowering supports a bounded rank");
    cuntt::detail::GeneralNdLayout layout;
    layout.rank = static_cast<std::uint32_t>(extents.size());
    layout.batch = batch;
    layout.points = 1;
    for (std::size_t extent : extents) {
        require(extent != 0 && layout.points <= static_cast<std::size_t>(-1) / extent,
                "xD extent product overflows size_t");
        layout.points *= extent;
    }
    layout.batch_stride = layout.points;
    std::size_t stride = 1;
    for (std::size_t axis = extents.size(); axis-- > 0;) {
        layout.extents[axis] = extents[axis];
        layout.strides[axis] = stride;
        require(stride <= static_cast<std::size_t>(-1) / extents[axis],
                "xD stride product overflows size_t");
        stride *= extents[axis];
    }
    return layout;
}

cuntt::detail::GeneralNdLayout packed_nd_layout_in_order(
    const std::vector<std::size_t>& physical_extents,
    const std::vector<std::uint32_t>& order, std::size_t batch) {
    require(order.size() == physical_extents.size(), "xD axis order rank mismatch");
    std::vector<std::size_t> extents(order.size());
    std::vector<bool> seen(order.size(), false);
    for (std::size_t axis = 0; axis < order.size(); ++axis) {
        require(order[axis] < physical_extents.size() && !seen[order[axis]],
                "xD axis order is not a permutation");
        seen[order[axis]] = true;
        extents[axis] = physical_extents[order[axis]];
    }
    return packed_nd_layout(extents, batch);
}

void prepare_xd_permutation_schedule(cubutterflyPlan& plan) {
    if (!plan.xd || plan.descriptor.rank < 2)
        return;
    const auto rank = plan.descriptor.rank;
    plan.xd_permutation_steps.clear();
    plan.xd_permutation_steps.resize(rank);
    std::vector<std::uint32_t> current_order(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis)
        current_order[axis] = axis;
    for (std::size_t axis_plan_index = 0; axis_plan_index < rank; ++axis_plan_index) {
        const std::uint32_t source_axis = rank - 1 - static_cast<std::uint32_t>(axis_plan_index);
        auto target_order = current_order;
        const auto position = std::find(target_order.begin(), target_order.end(), source_axis);
        require(position != target_order.end(), "xD axis is missing from the initial order");
        target_order.erase(position);
        target_order.push_back(source_axis);
        auto& step = plan.xd_permutation_steps[axis_plan_index];
        if (target_order != current_order) {
            step.required = true;
            step.input_layout = packed_nd_layout_in_order(plan.physical_extents, current_order,
                                                          plan.descriptor.batch);
            step.permutation.resize(rank);
            for (std::uint32_t output_axis = 0; output_axis < rank; ++output_axis) {
                const auto wanted = target_order[output_axis];
                const auto input_position = std::find(current_order.begin(), current_order.end(), wanted);
                require(input_position != current_order.end(), "xD permutation axis is missing");
                step.permutation[output_axis] =
                    static_cast<std::uint32_t>(input_position - current_order.begin());
            }
            current_order = std::move(target_order);
        }
    }
    std::vector<std::uint32_t> expected_order(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis)
        expected_order[axis] = axis;
    if (current_order != expected_order) {
        plan.xd_final_permutation_required = true;
        plan.xd_final_input_layout = packed_nd_layout_in_order(plan.physical_extents, current_order,
                                                               plan.descriptor.batch);
        plan.xd_final_permutation.resize(rank);
        for (std::uint32_t output_axis = 0; output_axis < rank; ++output_axis) {
            const auto input_position = std::find(current_order.begin(), current_order.end(), output_axis);
            require(input_position != current_order.end(), "xD final axis is missing");
            plan.xd_final_permutation[output_axis] =
                static_cast<std::uint32_t>(input_position - current_order.begin());
        }
    }
}

std::string workload_key(const cubutterflyDescriptor& descriptor) {
    const auto device = cuntt::current_device_info();
    std::ostringstream key;
    // Platform tuning rules are part of the selection namespace.  Keep their
    // effective-content identity in the common key so editing a rule file at
    // the same path cannot replay a stale measured winner.
    key << "v4|platform-tuning-" << cuntt::detail::platform_tuning_identity()
        << '|' << cuntt::detail::runtime_fingerprint()
        << "|local-selector-" << cuntt::detail::local_selector_fingerprint()
        << '|' << device.name << "|memory" << device.global_memory_bytes
        << "|sm" << device.compute_major << device.compute_minor << "-mp" << device.multiprocessors
        << "|execution-mode" << static_cast<int>(descriptor.execution_mode)
        << "|semantics:" << cubutterfly::detail::semantic_json(descriptor).dump();
    return key.str();
}

cubutterflyDescriptor make_axis_descriptor(const cubutterflyDescriptor& descriptor,
                                            std::uint32_t source_axis, std::size_t length,
                                            std::size_t batch) {
    auto axis = descriptor;
    axis.rank = 1;
    axis.extents = {length};
    axis.batch = batch;
    axis.length_mode = CUBUTTERFLY_LENGTH_STANDARD;
    axis.placement = CUBUTTERFLY_PLACEMENT_IN_PLACE;
    axis.input_strides = {1};
    axis.output_strides = {1};
    axis.input_batch_stride = length;
    axis.output_batch_stride = length;
    axis.input_orders.assign(1, CUBUTTERFLY_ORDER_NATURAL);
    axis.output_orders.assign(1, CUBUTTERFLY_ORDER_NATURAL);
    if (descriptor.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
        require(source_axis < descriptor.stage_matrices.size(), "structured axis is out of range");
        for (auto& matrices : axis.stage_matrices) matrices.clear();
        axis.stage_matrices[0] = descriptor.stage_matrices[source_axis];
    }
    return axis;
}

class CacheLock {
    int descriptor_;
  public:
    CacheLock(const std::string& path, bool write)
        : descriptor_(open(path.c_str(),write ? O_RDWR|O_CREAT : O_RDONLY,0600)) {
        if(descriptor_<0) {
            if(write) throw std::runtime_error("cannot open selection cache");
            return;
        }
        if(flock(descriptor_,write ? LOCK_EX : LOCK_SH)) {
            close(descriptor_); descriptor_=-1; throw std::runtime_error("cannot lock selection cache");
        }
    }
    ~CacheLock() { if(descriptor_>=0) { flock(descriptor_,LOCK_UN); close(descriptor_); } }
};

std::string lookup_cache(const std::string& path, const std::string& key) {
    if (path.empty())
        return {};
    CacheLock lock(path,false);
    std::ifstream input(path);
    std::string line;
    std::string match;
    while (std::getline(input, line)) {
        if(input.eof()) break; // A terminated writer's incomplete last record is not replayable.
        const auto separator = line.find('\t');
        if (separator != std::string::npos && line.compare(0, separator, key) == 0 && separator == key.size())
            match = line.substr(separator + 1);
    }
    return match;
}

void store_cache(const std::string& path, const std::string& key, const std::string& algorithm) {
    if (path.empty())
        return;
    CacheLock lock(path,true);
    std::ofstream output(path, std::ios::app);
    if (!output)
        throw std::runtime_error("could not open the cuButterfly selection cache for writing");
    output << key << '\t' << algorithm << '\n';
    output.flush();
    if(!output) throw std::runtime_error("could not persist the selection cache record");
}

void apply_butterfly_algorithm(cuntt::ButterflyConfig& config, const std::string& algorithm) {
    if (!algorithm.empty() && algorithm.front() == '{') {
        cuntt::detail::apply_serialized_mapping(algorithm, config);
        return;
    }
    config.auto_select = false;
    config.fft_core = cuntt::FftCore::Scalar;
    config.local_exchange = cuntt::LocalExchange::SharedMemory;
    config.stage_partition.clear(); config.segment_mappings.clear();
    config.execution_group_mappings.clear(); config.boundaries.clear();
    config.stage_overlap = false;
    config.shared_layout = cuntt::SharedLayout::Linear;
    config.cross_twiddle = cuntt::CrossTwiddleMode::Table;
    if (algorithm == "temporal-radix2") {
        config.backend = cuntt::ButterflyBackend::TemporalTile;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
        config.tile_threads = 128;
    } else if (algorithm == "temporal-radix4") {
        config.backend = cuntt::ButterflyBackend::TemporalTile;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.tile_threads = 128;
    } else if (algorithm == "temporal-warp-register-radix2") {
        config.backend = cuntt::ButterflyBackend::TemporalTile;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
        config.local_exchange = cuntt::LocalExchange::WarpRegister;
        config.tile_threads = 256;
    } else if (algorithm == "hierarchical-radix4") {
        config.backend = cuntt::ButterflyBackend::Hierarchical;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.local_stages = std::min<std::uint32_t>(10, config.log_n - 1);
        config.tile_threads = 256;
    } else if (algorithm == "online-radix4") {
        config.backend = cuntt::ButterflyBackend::OnlineReorder;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.local_stages = std::min<std::uint32_t>(10, config.log_n - 1);
        config.reorder_columns = 1;
        config.tile_threads = 256;
    } else if (algorithm == "cufft") {
        require(config.op == cuntt::ButterflyOperator::Fft, "cuFFT algorithm is valid only for FFT");
        config.backend = cuntt::ButterflyBackend::CuFft;
    } else {
        throw std::invalid_argument("unknown explicit butterfly algorithm");
    }
}

std::string butterfly_algorithm_id(const cuntt::ButterflyConfig& config) {
    if (config.backend == cuntt::ButterflyBackend::TemporalTile &&
        config.local_exchange == cuntt::LocalExchange::WarpRegister)
        return "temporal-warp-register-radix2";
    if (config.backend == cuntt::ButterflyBackend::TemporalTile)
        return config.compute_unit == cuntt::ComputeUnit::Radix2 ? "temporal-radix2" : "temporal-radix4";
    if (config.backend == cuntt::ButterflyBackend::Hierarchical)
        return "hierarchical-radix4";
    if (config.backend == cuntt::ButterflyBackend::OnlineReorder)
        return "online-radix4";
    if (config.backend == cuntt::ButterflyBackend::CuFft)
        return "cufft";
    return std::string(cuntt::butterfly_backend_name(config.backend)) + "/" +
           cuntt::fft_core_name(config.fft_core);
}

std::string ntt_algorithm_id(const cuntt::PlanConfig& config) {
    if (config.backend == cuntt::Backend::HierarchicalDataflow &&
        !config.subgraph_mappings.empty() &&
        (config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTail ||
         config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp ||
         config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
         config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
         config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
         config.subgraph_mappings.front().core ==
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal ||
         config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
             ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
         config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
             ApptOnlineRegisterTailGroupedWriterFinalResident ||
         config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
             ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter)) {
        if (config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter) {
            return "ntt-appt-register-tail-grouped-writer-final-resident-quarter";
        }
        if (config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResident) {
            return "ntt-appt-register-tail-grouped-writer-final-resident";
        }
        if (config.subgraph_mappings.front().core == cuntt::NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalDataTime) {
            return "ntt-appt-register-tail-grouped-writer-final-data-time";
        }
        if (config.subgraph_mappings.front().core ==
            cuntt::NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal) {
            return "ntt-appt-register-tail-grouped-writer-final";
        }
        if (config.subgraph_mappings.front().core ==
            cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped) {
            return "ntt-appt-register-tail-grouped";
        }
        if (config.subgraph_mappings.front().core ==
            cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8) {
            return "ntt-appt-register-tail-radix8";
        }
        if (config.subgraph_mappings.front().core ==
            cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp) {
            return "ntt-appt-register-tail-column-warp";
        }
        return config.subgraph_mappings.front().core ==
                       cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp
                   ? "ntt-appt-register-tail-warp"
                   : "ntt-appt-register-tail";
    }
    if (config.backend == cuntt::Backend::Baseline)
        return "ntt-baseline-radix2";
    if (config.backend == cuntt::Backend::SharedIterative)
        return "ntt-shared-iterative";
    if (config.backend == cuntt::Backend::Tile256)
        return "ntt-tile256-radix2";
    if (config.backend == cuntt::Backend::Hybrid2D)
        return config.compute_unit == cuntt::ComputeUnit::Radix4 ? "ntt-hybrid-radix4"
                                                                 : "ntt-hybrid-radix2";
    if (config.backend == cuntt::Backend::HybridDataflow) {
        return cuntt::detail::serialize_mapping(config);
    }
    if (config.backend == cuntt::Backend::HierarchicalBarrier)
        return config.hierarchical_core == cuntt::HierarchicalCore::Hybrid2DRadix4
                   ? "ntt-hierarchical-barrier-hybrid2d-radix4"
                   : "ntt-hierarchical-barrier";
    if (config.backend == cuntt::Backend::HierarchicalDataflow)
        return "ntt-hierarchical-dataflow";
    // Every executable backend can use the same exact replay path, including
    // CompactStage and StagePipeline, which have no legacy public name.
    return cuntt::detail::serialize_mapping(config);
}

void apply_ntt_algorithm(cuntt::PlanConfig& config, const std::string& algorithm) {
    if (!algorithm.empty() && algorithm.front() == '{') {
        cuntt::detail::apply_serialized_mapping(algorithm, config);
        return;
    }
    config.auto_select = false;
    if (algorithm == "ntt-baseline-radix2") {
        config.backend = cuntt::Backend::Baseline;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-shared-iterative") {
        config.backend = cuntt::Backend::SharedIterative;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-tile256-radix2") {
        config.backend = cuntt::Backend::Tile256;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-hybrid-radix2") {
        config.backend = cuntt::Backend::Hybrid2D;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-hybrid-radix4") {
        config.backend = cuntt::Backend::Hybrid2D;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
    } else if (algorithm == "ntt-hybrid-dataflow") {
        config.backend = cuntt::Backend::HybridDataflow;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-appt-register-tail" ||
               algorithm == "ntt-appt-register-tail-warp" ||
               algorithm == "ntt-appt-register-tail-column-warp" ||
               algorithm == "ntt-appt-register-tail-radix8" ||
               algorithm == "ntt-appt-register-tail-grouped" ||
               algorithm == "ntt-appt-register-tail-grouped-writer-final" ||
               algorithm ==
                   "ntt-appt-register-tail-grouped-writer-final-data-time" ||
               algorithm ==
                   "ntt-appt-register-tail-grouped-writer-final-resident" ||
               algorithm ==
                   "ntt-appt-register-tail-grouped-writer-final-resident-quarter") {
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.stage_partition = {7, 7, 6};
        config.subgraph_mappings.resize(3);
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = algorithm == "ntt-appt-register-tail-warp"
                               ? cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp
                           : algorithm == "ntt-appt-register-tail-column-warp"
                               ? cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp
                           : algorithm == "ntt-appt-register-tail-radix8"
                               ? cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8
                           : algorithm == "ntt-appt-register-tail-grouped"
                               ? cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped
                           : algorithm ==
                                     "ntt-appt-register-tail-grouped-writer-final"
                               ? cuntt::NttSubgraphCore::
                                     ApptOnlineRegisterTailGroupedWriterFinal
                           : algorithm ==
                                     "ntt-appt-register-tail-grouped-writer-final-data-time"
                               ? cuntt::NttSubgraphCore::
                                     ApptOnlineRegisterTailGroupedWriterFinalDataTime
                           : algorithm ==
                                     "ntt-appt-register-tail-grouped-writer-final-resident"
                               ? cuntt::NttSubgraphCore::
                                     ApptOnlineRegisterTailGroupedWriterFinalResident
                           : algorithm ==
                                     "ntt-appt-register-tail-grouped-writer-final-resident-quarter"
                               ? cuntt::NttSubgraphCore::
                                     ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter
                               : cuntt::NttSubgraphCore::ApptOnlineRegisterTail;
            mapping.units_per_cta = 8;
            mapping.data_space = 16;
            mapping.data_time = 4;
            mapping.role_stages = 2;
            mapping.token_interleave = 1;
            mapping.cta_weight = 0;
        }
        if (algorithm ==
                "ntt-appt-register-tail-grouped-writer-final-resident" ||
            algorithm ==
                "ntt-appt-register-tail-grouped-writer-final-resident-quarter") {
            config.subgraph_mappings[0].data_space = 32;
            config.subgraph_mappings[1].data_space = 16;
            config.subgraph_mappings[2].data_space = 16;
            config.output_order = cuntt::OutputOrder::Natural;
        }
    } else if (algorithm == "ntt-hierarchical-dataflow") {
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    } else if (algorithm == "ntt-hierarchical-barrier") {
        config.backend = cuntt::Backend::HierarchicalBarrier;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.hierarchical_core = cuntt::HierarchicalCore::DataflowRadix4;
    } else if (algorithm == "ntt-hierarchical-barrier-hybrid2d-radix4" ||
               algorithm == "ntt-hierarchical-dataflow-hybrid2d-radix4") {
        config.backend = cuntt::Backend::HierarchicalBarrier;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        config.hierarchical_core = cuntt::HierarchicalCore::Hybrid2DRadix4;
    } else {
        throw std::invalid_argument("unknown explicit NTT algorithm");
    }
}

struct DeviceAllocation {
    void* pointer = nullptr;
    explicit DeviceAllocation(std::size_t bytes) {
        if (bytes != 0 && cudaMalloc(&pointer, bytes) != cudaSuccess)
            throw std::bad_alloc();
    }
    ~DeviceAllocation() { cudaFree(pointer); }
};

template <typename Execute>
double measure_cuda_candidate(cudaStream_t stream, Execute&& execute) {
    execute();
    execute();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    auto checked = [](cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("candidate timing ") + operation + ": " + cudaGetErrorString(status));
    };
    try {
        checked(cudaEventCreate(&start), "create start event");
        checked(cudaEventCreate(&stop), "create stop event");
        checked(cudaEventRecord(start, stream), "record start event");
        for (int repeat = 0; repeat < 5; ++repeat) execute();
        checked(cudaEventRecord(stop, stream), "record stop event");
        checked(cudaEventSynchronize(stop), "synchronize stop event");
        float milliseconds = 0.0F;
        checked(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
        if (!std::isfinite(milliseconds) || milliseconds <= 0.0F)
            throw std::runtime_error("candidate timing returned a nonpositive or non-finite duration");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return milliseconds / 5.0;
    } catch (...) {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        throw;
    }
}

// Some cuFFT plans submit auxiliary work outside the caller stream.  A
// stream event then measures only the visible enqueue interval and can rank a
// complete real transform below a slower mixed plan.  This protocol is used
// only while tuning compact-real candidates; it synchronizes before and
// after the measured sequence and therefore accounts for all device work.
template <typename Execute>
double measure_cuda_candidate_synchronized(Execute&& execute) {
    auto synchronize = [](const char* operation) {
        const auto status = cudaDeviceSynchronize();
        if (status != cudaSuccess)
            throw std::runtime_error(std::string("synchronized candidate timing ") + operation + ": " +
                                     cudaGetErrorString(status));
    };
    constexpr int warmup = 2;
    constexpr int repeat = 5;
    synchronize("before warmup");
    for (int index = 0; index < warmup; ++index) execute();
    synchronize("after warmup");
    const auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < repeat; ++index) execute();
    synchronize("after measured executions");
    const auto stop = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count() / repeat;
    if (!std::isfinite(milliseconds) || milliseconds <= 0.0)
        throw std::runtime_error("synchronized candidate timing returned a nonpositive or non-finite duration");
    return milliseconds;
}

// xD candidate selection measures the complete plan through a CUDA Graph when
// possible.  Capture and graph instantiation happen before the timed launches;
// the graph object owns both handles so every failure path releases them.
class XdCudaGraph {
  public:
    XdCudaGraph() = default;
    XdCudaGraph(const XdCudaGraph&) = delete;
    XdCudaGraph& operator=(const XdCudaGraph&) = delete;
    ~XdCudaGraph() {
        if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
        if (graph_ != nullptr) cudaGraphDestroy(graph_);
    }

    cudaGraphExec_t executable() const noexcept { return executable_; }
    cudaGraph_t* graph_address() noexcept { return &graph_; }
    cudaGraphExec_t* executable_address() noexcept { return &executable_; }

  private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
};

constexpr unsigned kXdGraphPlanExecutions = 20;
constexpr unsigned kXdGraphWarmupLaunches = 2;
constexpr unsigned kXdGraphTimedLaunches = 5;

std::runtime_error xd_cuda_error(const char* operation, cudaError_t status) {
    return std::runtime_error(std::string("xD CUDA Graph ") + operation + ": " +
                              cudaGetErrorString(status));
}

template <typename Execute>
std::unique_ptr<XdCudaGraph> capture_xd_cuda_graph(cudaStream_t stream, Execute&& execute) {
    auto graph = std::make_unique<XdCudaGraph>();
    bool capturing = false;
    try {
        // Initialize lazy backend state once outside the captured graph.  The
        // capture itself then contains a steady sequence of complete plans.
        execute();
        auto status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (status != cudaSuccess) throw xd_cuda_error("capture begin", status);
        capturing = true;
        for (unsigned repeat = 0; repeat < kXdGraphPlanExecutions; ++repeat)
            execute();
        status = cudaStreamEndCapture(stream, graph->graph_address());
        capturing = false;
        if (status != cudaSuccess) throw xd_cuda_error("capture end", status);
        status = cudaGraphInstantiate(graph->executable_address(), *graph->graph_address(), 0);
        if (status != cudaSuccess) throw xd_cuda_error("instantiate", status);
        return graph;
    } catch (...) {
        if (capturing) {
            cudaGraph_t abandoned = nullptr;
            (void)cudaStreamEndCapture(stream, &abandoned);
            if (abandoned != nullptr) cudaGraphDestroy(abandoned);
        }
        // A failed capture can leave the runtime's last-error slot set even
        // though EndCapture has restored the stream to its normal state.  The
        // caller records the original exception and may then use the uniform
        // event fallback protocol.
        (void)cudaGetLastError();
        throw;
    }
}

double measure_xd_cuda_graph(cudaStream_t stream, cudaGraphExec_t executable) {
    for (unsigned warmup = 0; warmup < kXdGraphWarmupLaunches; ++warmup) {
        const auto status = cudaGraphLaunch(executable, stream);
        if (status != cudaSuccess) throw xd_cuda_error("warmup launch", status);
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    try {
        auto status = cudaEventCreate(&start);
        if (status != cudaSuccess) throw xd_cuda_error("create start event", status);
        status = cudaEventCreate(&stop);
        if (status != cudaSuccess) throw xd_cuda_error("create stop event", status);
        status = cudaEventRecord(start, stream);
        if (status != cudaSuccess) throw xd_cuda_error("record start event", status);
        for (unsigned repeat = 0; repeat < kXdGraphTimedLaunches; ++repeat) {
            status = cudaGraphLaunch(executable, stream);
            if (status != cudaSuccess) throw xd_cuda_error("launch", status);
        }
        status = cudaEventRecord(stop, stream);
        if (status != cudaSuccess) throw xd_cuda_error("record stop event", status);
        status = cudaEventSynchronize(stop);
        if (status != cudaSuccess) throw xd_cuda_error("synchronize stop event", status);
        float milliseconds = 0.0F;
        status = cudaEventElapsedTime(&milliseconds, start, stop);
        if (status != cudaSuccess) throw xd_cuda_error("elapsed time", status);
        if (!std::isfinite(milliseconds) || milliseconds <= 0.0F)
            throw std::runtime_error("candidate graph timing returned a nonpositive or non-finite duration");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return milliseconds / (kXdGraphTimedLaunches * kXdGraphPlanExecutions);
    } catch (...) {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        throw;
    }
}

// Keep the legacy/default caller stream out of xD capture.  Every xD tuning
// invocation is synchronous before this scope exits, so restoring the handle
// before destroying the temporary stream cannot expose incomplete work.
class XdTuningStream {
  public:
    explicit XdTuningStream(cubutterflyHandle_t handle)
        : handle_(handle), original_(handle->stream) {
        const auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (status != cudaSuccess) throw xd_cuda_error("create tuning stream", status);
        handle_->stream = stream_;
    }
    XdTuningStream(const XdTuningStream&) = delete;
    XdTuningStream& operator=(const XdTuningStream&) = delete;
    ~XdTuningStream() {
        handle_->stream = original_;
        if (stream_ != nullptr) {
            (void)cudaStreamSynchronize(stream_);
            (void)cudaStreamDestroy(stream_);
        }
    }

    cudaStream_t get() const noexcept { return stream_; }
    cudaStream_t original() const noexcept { return original_; }

  private:
    cubutterflyHandle_t handle_;
    cudaStream_t original_ = nullptr;
    cudaStream_t stream_ = nullptr;
};

void restore_xd_plan_streams(cubutterflyPlan* plan, cudaStream_t stream) {
    if (plan == nullptr) return;
    if (plan->xd_selected) restore_xd_plan_streams(plan->xd_selected.get(), stream);
    if (plan->numeric_child) restore_xd_plan_streams(plan->numeric_child.get(), stream);
    if (plan->boundary_child) restore_xd_plan_streams(plan->boundary_child.get(), stream);
    for (auto& child : plan->arbitrary_children) restore_xd_plan_streams(child.get(), stream);
    for (auto& axis : plan->butterfly_axes) axis->set_stream(stream);
    for (auto& axis : plan->ntt_axes) axis->set_stream(stream);
    if (plan->butterfly) plan->butterfly->set_stream(stream);
    if (plan->ntt) plan->ntt->set_stream(stream);
    plan->bound_stream = stream;
    // XdRealFftPlan has no persistent public stream setter: execute() calls
    // update_stream() before every cuFFT submission and rebinds it lazily.
}

void bind_plan_stream_if_needed(cubutterflyPlan* plan, cudaStream_t stream) {
    if (plan == nullptr || plan->bound_stream == stream) return;
    restore_xd_plan_streams(plan, stream);
}

double measure_direct_fft_candidate(const cubutterflyDescriptor& descriptor,
                                    std::size_t bytes, cudaStream_t stream) {
    cuntt::detail::DirectFftPlan candidate(
        descriptor.extents, descriptor.batch,
        descriptor.direction == CUBUTTERFLY_DIRECTION_INVERSE,
        descriptor.normalize_inverse,
        descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP64);
    DeviceAllocation input(bytes);
    DeviceAllocation output(bytes);
    const auto initialized = cudaMemsetAsync(input.pointer, 0, bytes, stream);
    if (initialized != cudaSuccess) throw xd_cuda_error("initialize direct candidate input", initialized);
    return measure_cuda_candidate(stream, [&] { candidate.execute(input.pointer, output.pointer, stream); });
}

double measure_bluestein_fft_candidate(const cubutterflyDescriptor& descriptor,
                                       const std::vector<std::size_t>& physical_extents,
                                       std::size_t element_size, cudaStream_t stream) {
    const bool inverse = descriptor.direction == CUBUTTERFLY_DIRECTION_INVERSE;
    const bool fp64 = descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP64;
    std::size_t logical_points = descriptor.batch;
    for (const auto extent : descriptor.extents)
        logical_points *= extent;
    const std::size_t logical_bytes = logical_points * element_size;
    DeviceAllocation input(logical_bytes);
    DeviceAllocation output(logical_bytes);
    const auto initialized = cudaMemsetAsync(input.pointer, 0, logical_bytes, stream);
    if (initialized != cudaSuccess) throw xd_cuda_error("initialize Bluestein candidate input", initialized);
    if (descriptor.rank == 1) {
        cuntt::detail::BluesteinFftPlan candidate(
            descriptor.extents[0], physical_extents[0], descriptor.batch,
            inverse, descriptor.normalize_inverse, fp64);
        DeviceAllocation workspace(candidate.workspace_size());
        return measure_cuda_candidate(stream, [&] {
            candidate.execute(input.pointer, output.pointer, workspace.pointer, 1,
                              descriptor.extents[0], 1, descriptor.extents[0], stream);
        });
    }

    cuntt::detail::BluesteinFftPlan rows(
        descriptor.extents[1], physical_extents[1],
        descriptor.batch * descriptor.extents[0], inverse,
        descriptor.normalize_inverse, fp64);
    cuntt::detail::BluesteinFftPlan columns(
        descriptor.extents[0], physical_extents[0],
        descriptor.batch * descriptor.extents[1], inverse,
        descriptor.normalize_inverse, fp64);
    const std::size_t packed_bytes = logical_bytes;
    const std::size_t axis_workspace = std::max(rows.workspace_size(), columns.workspace_size());
    DeviceAllocation workspace(2 * align_16(packed_bytes) + align_16(axis_workspace));
    auto* first = static_cast<unsigned char*>(workspace.pointer);
    auto* second = first + align_16(packed_bytes);
    auto* core_workspace = second + align_16(packed_bytes);
    return measure_cuda_candidate(stream, [&] {
        cuntt::detail::launch_pack_2d(
            input.pointer, first, descriptor.extents[0], descriptor.extents[1],
            descriptor.extents[0], descriptor.extents[1], descriptor.batch,
            descriptor.input_strides[0], descriptor.input_strides[1],
            descriptor.input_batch_stride, element_size, stream);
        rows.execute(first, second, core_workspace, 1, descriptor.extents[1], 1,
                     descriptor.extents[1], stream);
        cuntt::detail::launch_transpose_2d(
            second, first, descriptor.extents[0], descriptor.extents[1],
            descriptor.batch, element_size, stream);
        columns.execute(first, second, core_workspace, 1, descriptor.extents[0], 1,
                        descriptor.extents[0], stream);
        cuntt::detail::launch_transpose_2d(
            second, first, descriptor.extents[1], descriptor.extents[0],
            descriptor.batch, element_size, stream);
        cuntt::detail::launch_scatter_2d(
            first, output.pointer, descriptor.extents[0], descriptor.extents[1],
            descriptor.extents[0], descriptor.extents[1], descriptor.batch,
            descriptor.output_strides[0], descriptor.output_strides[1],
            descriptor.output_batch_stride, element_size, stream);
    });
}

template<class T> struct TuningValue {
    using Reference = std::conditional_t<std::is_same_v<T, double>, double, float>;
    static T make(double a, double) { return T(a); }
    static Reference reference(T v) { return static_cast<Reference>(v); }
    static long double square(Reference a, Reference b) { const long double x = a - b; return x*x; }
};
template<class T, class R> struct ComplexTuningValue {
    using Reference = R;
    static T make(double a, double b) { return {decltype(T::real)(a), decltype(T::imag)(b)}; }
    static R reference(T v) { return {decltype(R::real)(v.real), decltype(R::imag)(v.imag)}; }
    static long double square(R a, R b) { const long double x = a.real - b.real, y = a.imag - b.imag; return x*x + y*y; }
};
template<> struct TuningValue<cuntt::Complex32> : ComplexTuningValue<cuntt::Complex32, cuntt::Complex32> {};
template<> struct TuningValue<cuntt::Complex64> : ComplexTuningValue<cuntt::Complex64, cuntt::Complex64> {};
template<> struct TuningValue<cuntt::Complex16> : ComplexTuningValue<cuntt::Complex16, cuntt::Complex32> {};
template<> struct TuningValue<cuntt::ComplexBf16> : ComplexTuningValue<cuntt::ComplexBf16, cuntt::Complex32> {};

template<class T>
double checked_butterfly_measure(cuntt::ButterflyPlan& plan,
                                 cuntt::detail::PlanSearchTrace* trace = nullptr) {
    using Values = TuningValue<T>;
    using R = typename Values::Reference;
    const auto& config = plan.config();
    const std::size_t n = std::size_t{1} << config.log_n;
    std::vector<T> input, output;
    {
        cuntt::detail::PlanSearchTrace::Phase phase(trace, "input_generation");
        input.resize(plan.data_size() / sizeof(T));
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = Values::make((int((i*17+3)%127)-63) / (64.0*std::sqrt(double(n))),
                                   (int((i*11+7)%113)-56) / (64.0*std::sqrt(double(n))));
    }
    const auto stats = [&] {
        cuntt::detail::PlanSearchTrace::Phase phase(trace, "gpu_execute");
        return plan.execute(input, output, 3, 20);
    }();
    cuntt::detail::PlanSearchTrace::Phase reference_phase(trace, "reference_check");
    long double error = 0, norm = 0;
    for (std::size_t batch = 0; batch < config.batch; ++batch) {
        std::vector<R> expected(n);
        for (std::size_t i = 0; i < n; ++i) expected[i] = Values::reference(input[batch*config.batch_stride+i*config.element_stride]);
        if constexpr (std::is_same_v<R, cuntt::Complex32> || std::is_same_v<R, cuntt::Complex64>)
            cuntt::reference_fft(expected, config.inverse, config.normalize_inverse);
        else if (config.op == cuntt::ButterflyOperator::Structured2x2)
            cuntt::reference_structured_2x2(expected, config.stage_matrices, config.inverse);
        else cuntt::reference_fwht(expected, config.inverse, config.normalize_inverse);
        if constexpr (std::is_same_v<R, float> || std::is_same_v<R, double>)
            for (auto& value : expected) {
                value *= static_cast<R>(config.output_scale);
                if (config.final_storage_rounding) value = Values::reference(Values::make(value, 0));
            }
        for (std::size_t i = 0; i < n; ++i) {
            const auto actual = Values::reference(output[batch*config.batch_stride+i*config.element_stride]);
            const auto delta = Values::square(actual, expected[i]);
            if (!std::isfinite(delta)) throw std::invalid_argument("candidate produced non-finite output");
            error += delta; norm += Values::square(expected[i], R{});
        }
    }
    const bool low = config.precision == cuntt::ButterflyPrecision::Fp16 || config.precision == cuntt::ButterflyPrecision::Bf16;
    const auto tolerance = low ? 0.25L : config.precision == cuntt::ButterflyPrecision::Fp64 ? 1.0e-10L : 1.0e-4L;
    require(std::sqrt(error / std::max(norm, 1.0e-30L)) <= tolerance, "candidate failed reference validation");
    return stats.kernel_ms;
}

template<class UInt>
double checked_zeta_measure(cuntt::ButterflyPlan& plan,
                            cuntt::detail::PlanSearchTrace* trace = nullptr) {
    const auto& config = plan.config();
    std::vector<UInt> input, output;
    {
        cuntt::detail::PlanSearchTrace::Phase phase(trace, "input_generation");
        input.resize(plan.data_size() / sizeof(UInt));
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = UInt(i * 17 + 3) * (std::numeric_limits<UInt>::max() / UInt(127));
    }
    const auto stats = [&] {
        cuntt::detail::PlanSearchTrace::Phase phase(trace, "gpu_execute");
        return plan.execute(input, output, 3, 20);
    }();
    cuntt::detail::PlanSearchTrace::Phase reference_phase(trace, "reference_check");
    const auto n = std::size_t{1} << config.log_n;
    for (std::size_t b = 0; b < config.batch; ++b) {
        std::vector<UInt> expected(n);
        for (std::size_t i = 0; i < n; ++i)
            expected[i] = input[b * config.batch_stride + i * config.element_stride];
        if (config.op == cuntt::ButterflyOperator::SupersetZeta)
            cuntt::reference_superset_zeta(expected, config.inverse);
        else cuntt::reference_subset_zeta(expected, config.inverse);
        for (std::size_t i = 0; i < n; ++i)
            require(expected[i] == output[b * config.batch_stride + i * config.element_stride],
                    "zeta candidate failed exact verification");
    }
    return stats.kernel_ms;
}

double measure_butterfly(cuntt::ButterflyConfig config, cubutterflyDataType_t storage,
                         cudaStream_t stream, cuntt::detail::PlanSearchTrace* trace = nullptr) {
    config.auto_allocate_workspace = true;
    std::optional<cuntt::ButterflyPlan> plan;
    auto destruction = cuntt::detail::PlanSearchTrace::on_exit(trace, "destruction", [&] { plan.reset(); });
    {
        cuntt::detail::PlanSearchTrace::Phase phase(trace, "construction");
        plan.emplace(config);
        plan->set_stream(stream);
    }
    switch (storage) {
        case CUBUTTERFLY_DATA_FP16: return checked_butterfly_measure<cuntt::Fp16>(*plan, trace);
        case CUBUTTERFLY_DATA_BF16: return checked_butterfly_measure<cuntt::Bf16>(*plan, trace);
        case CUBUTTERFLY_DATA_FP32: return checked_butterfly_measure<float>(*plan, trace);
        case CUBUTTERFLY_DATA_FP64: return checked_butterfly_measure<double>(*plan, trace);
        case CUBUTTERFLY_DATA_COMPLEX_FP16: return checked_butterfly_measure<cuntt::Complex16>(*plan, trace);
        case CUBUTTERFLY_DATA_COMPLEX_BF16: return checked_butterfly_measure<cuntt::ComplexBf16>(*plan, trace);
        case CUBUTTERFLY_DATA_COMPLEX_FP32: return checked_butterfly_measure<cuntt::Complex32>(*plan, trace);
        case CUBUTTERFLY_DATA_COMPLEX_FP64: return checked_butterfly_measure<cuntt::Complex64>(*plan, trace);
        case CUBUTTERFLY_DATA_UINT32: return checked_zeta_measure<std::uint32_t>(*plan, trace);
        case CUBUTTERFLY_DATA_UINT64: return checked_zeta_measure<std::uint64_t>(*plan, trace);
        default: throw std::invalid_argument("unsupported tuning storage");
    }
}

std::pair<std::string, double> tune_butterfly(const cuntt::ButterflyConfig& base,
                                               cubutterflyDataType_t storage,
                                               cudaStream_t stream) {
    cuntt::detail::PlanSearchTrace trace("butterfly");
    cuntt::detail::PlanSearchTrace::Phase search_phase(&trace, "search_total");
    std::vector<cuntt::ButterflyConfig> inventory;
    {
        cuntt::detail::PlanSearchTrace::Phase phase(&trace, "enumeration");
        inventory = cuntt::detail::portable_butterfly_candidates(base);
    }
    {
        cuntt::detail::PlanSearchTrace::Phase phase(&trace, "selection");
        const auto selected = cuntt::detail::select_butterfly_mapping(base);
        inventory.insert(inventory.begin(), selected.config);
    }
    const auto full_candidate_count = inventory.size();
    const auto disabled_mechanisms = cuntt::detail::search_disabled_mechanisms();
    const auto unrestricted_inventory_count = inventory.size();
    inventory.erase(std::remove_if(inventory.begin(), inventory.end(), [&](const auto& candidate) {
        return cuntt::detail::mapping_uses_disabled_mechanism(
            cuntt::detail::serialize_mapping(candidate), disabled_mechanisms);
    }), inventory.end());
    const auto disabled_candidate_count = unrestricted_inventory_count - inventory.size();
    require(!inventory.empty(), "search restrictions removed every butterfly candidate");
    const auto budget = search_budget();
    auto candidates = bounded_search_inventory(inventory, budget);
    trace.event("inventory", [&] {
        nlohmann::json fields = {
            {"candidate_count", full_candidate_count},
            {"restricted_candidate_count", inventory.size()},
            {"disabled_candidate_count", disabled_candidate_count},
            {"scheduled_candidate_count", candidates.size()}, {"search_budget", budget},
            {"inventory_truncated", candidates.size() < full_candidate_count},
            {"batch", base.batch}, {"log_n", base.log_n},
            {"overlap_candidate_count", std::count_if(inventory.begin(), inventory.end(),
                [](const auto& c) { return c.stage_overlap; })}, {"warmup", 3}, {"repeat", 20}};
        fields["disabled_mechanisms"] = disabled_mechanisms;
        return fields;
    });
    std::pair<std::string, double> best{"", std::numeric_limits<double>::infinity()};
    std::size_t candidate_index = 0, accepted = 0, rejected = 0;
    for (const auto& candidate : candidates) {
        trace.context([&] {
            return nlohmann::json{{"search", "butterfly"}, {"candidate_index", candidate_index},
                {"candidate_count", candidates.size()}, {"batch", candidate.batch}, {"log_n", candidate.log_n},
                {"element_stride", candidate.element_stride}, {"batch_stride", candidate.batch_stride},
                {"operator", cuntt::butterfly_operator_name(candidate.op)},
                {"precision", cuntt::butterfly_precision_name(candidate.precision)},
                {"inverse", candidate.inverse}, {"normalize_inverse", candidate.normalize_inverse},
                {"output_scale", candidate.output_scale}, {"final_storage_rounding", candidate.final_storage_rounding},
                {"storage", static_cast<int>(storage)},
                {"warmup", 3}, {"repeat", 20},
                {"mapping", nlohmann::json::parse(cuntt::detail::serialize_mapping(candidate))}};
        });
        ++candidate_index;
        trace.event("candidate_begin", [] { return nlohmann::json{{"outcome", "started"}}; });
        cuntt::detail::PlanSearchTrace::Phase candidate_phase(&trace, "candidate_total");
        try {
            const double milliseconds = measure_butterfly(candidate, storage, stream, &trace);
            if (milliseconds < best.second)
                best = {cuntt::detail::serialize_mapping(candidate), milliseconds};
            ++accepted;
            trace.event("candidate_result", [&] {
                return nlohmann::json{{"outcome", "accepted"}, {"kernel_ms", milliseconds}};
            });
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            ++rejected;
            trace.event("candidate_result", [&] {
                return nlohmann::json{{"outcome", "rejected"}, {"error", error.what()}};
            });
            // A compiler/resource failure belongs to this candidate. A sticky
            // execution failure belongs to the CUDA context and must abort.
            (void)cudaGetLastError();
            if (cudaStreamSynchronize(stream) != cudaSuccess) throw;
            std::clog << "cuButterfly: rejected measured butterfly candidate: " << error.what() << '\n';
        }
        if ((candidate_index % 64 == 0) || candidate_index == candidates.size())
            std::clog << "cuButterfly: butterfly search progress " << candidate_index << "/"
                      << candidates.size() << " accepted=" << accepted << " rejected=" << rejected << '\n';
    }
    trace.context([&] { return nlohmann::json{{"search", "butterfly"}, {"batch", base.batch}, {"log_n", base.log_n}}; });
    trace.event("summary", [&] {
        return nlohmann::json{{"candidate_count", full_candidate_count},
            {"scheduled_candidate_count", candidates.size()}, {"search_budget", budget},
            {"accepted", accepted},
            {"rejected", rejected}, {"best_kernel_ms", best.second}, {"best_mapping", best.first}};
    });
    require(!best.first.empty(), "no legal butterfly candidate survived measurement");
    return best;
}

// Rank-2 real-FFT adapters execute two physical one-dimensional axes.  For
// the small FP32 axes used by spectral models, a large physical batch makes
// the generic candidate space unnecessarily expensive to probe while the
// useful decision is usually between the compact temporal codelet and the
// vendor C2C core.  Keep this bounded measurement separate from the general
// tuner so online calibration remains predictable at application startup.
struct Rank2FftTuning {
    std::string algorithm;
    double milliseconds = std::numeric_limits<double>::infinity();
    std::vector<std::string> rejected;
};

std::string rank2_fft_candidate_list(const std::vector<std::string>& values) {
    std::ostringstream result;
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) result << ", ";
        result << values[index];
    }
    return result.str();
}

std::string rank2_fft_selection_reason(const Rank2FftPolicy& policy,
                                       const std::string& prefix,
                                       const std::string& selected,
                                       const std::vector<std::string>& rejected) {
    std::ostringstream reason;
    reason << prefix << " policy=" << policy.mode << "; candidates=["
           << rank2_fft_candidate_list(policy.candidates) << "]; selected=" << selected;
    if (!rejected.empty()) {
        reason << "; rejected=[" << rank2_fft_candidate_list(rejected) << "]";
    }
    if (!policy.candidates.empty() && selected != policy.candidates.front())
        reason << "; fallback=" << selected;
    return reason.str();
}

Rank2FftTuning tune_compact_rank2_fft(const cuntt::ButterflyConfig& base,
                                      cubutterflyDataType_t storage,
                                      cudaStream_t stream,
                                      const Rank2FftPolicy& policy) {
    Rank2FftTuning best;
    for (const auto& algorithm : policy.candidates) {
        auto candidate = base;
        try {
            apply_butterfly_algorithm(candidate, algorithm);
            const double milliseconds = measure_butterfly(candidate, storage, stream);
            if (milliseconds < best.milliseconds)
                best = {algorithm, milliseconds, best.rejected};
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            (void)cudaGetLastError();
            if (cudaStreamSynchronize(stream) != cudaSuccess) throw;
            best.rejected.push_back(algorithm + ": " + error.what());
        }
    }
    if (best.algorithm.empty()) {
        throw std::invalid_argument(
            "no legal compact rank-2 FFT candidate survived measurement; candidates=[" +
            rank2_fft_candidate_list(policy.candidates) + "]; rejected=[" +
            rank2_fft_candidate_list(best.rejected) + "]");
    }
    return best;
}

struct Rank2FftCandidateSelection {
    cuntt::ButterflyConfig config;
    std::string algorithm;
    std::vector<std::string> rejected;
};

Rank2FftCandidateSelection select_compact_rank2_fft_candidate(
    const cuntt::ButterflyConfig& base, const Rank2FftPolicy& policy) {
    std::vector<std::string> rejected;
    for (const auto& algorithm : policy.candidates) {
        auto candidate = base;
        try {
            apply_butterfly_algorithm(candidate, algorithm);
            cuntt::ButterflyPlan probe(candidate);
            return {probe.config(), algorithm, std::move(rejected)};
        } catch (const std::invalid_argument& error) {
            rejected.push_back(algorithm + ": " + error.what());
        }
    }
    throw std::invalid_argument(
        "no legal compact rank-2 FFT candidate is available; candidates=[" +
        rank2_fft_candidate_list(policy.candidates) + "]; rejected=[" +
        rank2_fft_candidate_list(rejected) + "]");
}

double measure_ntt(cuntt::PlanConfig config, cudaStream_t stream) {
    config.auto_allocate_workspace = true;
    cuntt::Plan plan(config);
    plan.set_stream(stream);
    config = plan.config();
    require(config.input_order == cuntt::InputOrder::Natural &&
                (config.output_order == cuntt::OutputOrder::Natural ||
                 config.output_order == cuntt::OutputOrder::BitReversed),
            "runtime NTT measurement requires natural input and natural or bit-reversed output");
    const auto n = std::size_t{1} << config.log_n;
    std::vector<std::uint64_t> input(n*config.batch), output;
    for (std::size_t i=0; i<input.size(); ++i) input[i] = (i*17+3)%config.modulus;
    const auto stats = plan.execute(input, output, 3, 20);
    for (std::size_t b=0; b<config.batch; ++b) {
        std::vector<std::uint64_t> expected(input.begin()+b*n, input.begin()+(b+1)*n);
        cuntt::reference_ntt(expected, config.modulus, config.inverse);
        for (std::size_t i = 0; i < n; ++i) {
            auto index = i;
            if (config.output_order == cuntt::OutputOrder::BitReversed) {
                index = 0;
                for (unsigned bit = 0; bit < config.log_n; ++bit)
                    index = (index << 1) | ((i >> bit) & 1U);
            }
            require(expected[i] == output[b*n+index], "NTT candidate failed exact reference verification");
        }
    }
    return stats.kernel_ms;
}

std::pair<std::string, double> tune_ntt(const cuntt::PlanConfig& base, cudaStream_t stream) {
    cuntt::detail::PlanSearchTrace trace("ntt");
    cuntt::detail::PlanSearchTrace::Phase search_phase(&trace, "search_total");
    auto inventory = cuntt::detail::portable_ntt_candidates(base);
    const auto selected = cuntt::detail::select_ntt_mapping(base);
    inventory.insert(inventory.begin(), selected.config);
    const auto full_candidate_count = inventory.size();
    const auto disabled_mechanisms = cuntt::detail::search_disabled_mechanisms();
    const auto unrestricted_inventory_count = inventory.size();
    inventory.erase(std::remove_if(inventory.begin(), inventory.end(), [&](const auto& candidate) {
        return cuntt::detail::mapping_uses_disabled_mechanism(
            cuntt::detail::serialize_mapping(candidate), disabled_mechanisms);
    }), inventory.end());
    const auto disabled_candidate_count = unrestricted_inventory_count - inventory.size();
    require(!inventory.empty(), "search restrictions removed every NTT candidate");
    const auto budget = search_budget();
    auto candidates = bounded_search_inventory(inventory, budget);
    trace.event("inventory", [&] {
        nlohmann::json fields = {
            {"search", "ntt"}, {"candidate_count", full_candidate_count},
            {"restricted_candidate_count", inventory.size()},
            {"disabled_candidate_count", disabled_candidate_count},
            {"scheduled_candidate_count", candidates.size()}, {"search_budget", budget},
            {"inventory_truncated", candidates.size() < full_candidate_count},
            {"batch", base.batch}, {"log_n", base.log_n}, {"word_bits", base.word_bits},
            {"timing_protocol", "host-event-loop"}, {"warmup", 3}, {"repeat", 20}};
        fields["disabled_mechanisms"] = disabled_mechanisms;
        return fields;
    });
    std::clog << "cuButterfly: NTT search inventory=" << full_candidate_count
              << " scheduled=" << candidates.size() << " budget=" << budget << '\n';
    std::pair<std::string, double> best{"", std::numeric_limits<double>::infinity()};
    std::size_t candidate_index = 0, accepted = 0, rejected = 0;
    for (const auto& candidate : candidates) {
        ++candidate_index;
        trace.context([&] {
            return nlohmann::json{{"search", "ntt"}, {"candidate_index", candidate_index},
                {"candidate_count", candidates.size()}, {"mapping", cuntt::detail::serialize_mapping(candidate)}};
        });
        try {
            const double milliseconds = measure_ntt(candidate, stream);
            if (milliseconds < best.second)
                best = {cuntt::detail::serialize_mapping(candidate), milliseconds};
            trace.event("result", [&] {
                return nlohmann::json{{"status", "accepted"}, {"kernel_ms", milliseconds}};
            });
            ++accepted;
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            ++rejected;
            (void)cudaGetLastError();
            if (cudaStreamSynchronize(stream) != cudaSuccess) throw;
            trace.event("result", [&] {
                return nlohmann::json{{"status", "rejected"}, {"reason", error.what()}};
            });
            std::clog << "cuButterfly: rejected measured NTT candidate: " << error.what() << '\n';
        }
        if ((candidate_index % 64 == 0) || candidate_index == candidates.size())
            std::clog << "cuButterfly: NTT search progress " << candidate_index << "/"
                      << candidates.size() << " accepted=" << accepted << " rejected=" << rejected << '\n';
    }
    require(!best.first.empty(), "no legal NTT candidate survived measurement");
    return best;
}

struct ButterflyAxisSelection {
    cuntt::ButterflyConfig config;
    std::string algorithm;
    cubutterflySelectionSource_t source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
    std::string reason;
};

std::string composition_axis(const cubutterflyDescriptor& descriptor, unsigned source_axis) {
    if(descriptor.algorithm_policy!=CUBUTTERFLY_ALGORITHM_EXPLICIT || descriptor.explicit_algorithm.empty() ||
        descriptor.explicit_algorithm.front()!='{') return {};
    const auto mapping=nlohmann::json::parse(descriptor.explicit_algorithm);
    if(mapping.value("kind","")!="composition") return {};
    require(mapping.value("schema_version",0)==1 && mapping.at("axes").size()==descriptor.rank,
            "composition replay requires one complete mapping per axis");
    return mapping.at("axes").at(descriptor.rank-1-source_axis).dump();
}

std::string axis_algorithm(const cubutterflyDescriptor& descriptor,unsigned source_axis) {
    const auto axis=composition_axis(descriptor,source_axis);
    if(axis.empty()) return descriptor.explicit_algorithm;
    const auto mapping=nlohmann::json::parse(axis);
    return mapping.contains("forward") ? mapping.at("forward").dump() : axis;
}

ButterflyAxisSelection select_butterfly_axis(cubutterflyHandle_t handle,
                                             const cubutterflyDescriptor& descriptor,
                                             std::uint32_t source_axis, std::size_t length,
                                             std::size_t batch, bool direct_input = false) {
    auto config = make_axis_butterfly_config(descriptor, source_axis, length, batch);
    auto axis_descriptor = make_axis_descriptor(descriptor, source_axis, length, batch);
    if (direct_input) {
        config.placement = cuntt::ButterflyPlacement::OutOfPlace;
        axis_descriptor.placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    }
    const bool rank2_fft_policy_applies =
        descriptor.rank == 2 && descriptor.op == CUBUTTERFLY_OPERATOR_FFT &&
        descriptor.algorithm_policy != CUBUTTERFLY_ALGORITHM_EXPLICIT;
    const auto rank2_policy = rank2_fft_policy_applies
                                ? rank2_fft_policy(descriptor.algorithm_policy)
                                : Rank2FftPolicy{};
    const bool compact_rank2_fft = rank2_policy.compact && descriptor.rank == 2 &&
                                   descriptor.op == CUBUTTERFLY_OPERATOR_FFT &&
                                   descriptor.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 &&
                                   length <= rank2_policy.max_length;
    const auto key = workload_key(axis_descriptor) +
                     (rank2_fft_policy_applies
                          ? "|rank2-fft-policy-" + rank2_policy.token : "");
    const auto cached = descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT
                            ? lookup_cache(handle->cache_path, key)
                            : std::string{};
    ButterflyAxisSelection selected;
    if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
        apply_butterfly_algorithm(config, axis_algorithm(descriptor,source_axis));
        selected.source = CUBUTTERFLY_SELECTION_EXPLICIT;
        selected.reason = "applied the explicit algorithm to the physical axis";
    } else if (!cached.empty()) {
        apply_butterfly_algorithm(config, cached);
        selected.source = CUBUTTERFLY_SELECTION_USER_CACHE;
        selected.reason = compact_rank2_fft
                               ? rank2_fft_selection_reason(rank2_policy,
                                                            "matched compact rank-2 FFT cache",
                                                            cached, {})
                               : "matched the physical axis in the user selection cache";
    } else if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE ||
               (compact_rank2_fft && rank2_policy.measure)) {
        std::string tuned_algorithm;
        double tuned_milliseconds = 0.0;
        std::vector<std::string> rejected;
        if (compact_rank2_fft) {
            const auto tuned = tune_compact_rank2_fft(config, descriptor.storage, handle->stream,
                                                      rank2_policy);
            tuned_algorithm = tuned.algorithm;
            tuned_milliseconds = tuned.milliseconds;
            rejected = tuned.rejected;
        } else {
            const auto tuned = tune_butterfly(config, descriptor.storage, handle->stream);
            tuned_algorithm = tuned.first;
            tuned_milliseconds = tuned.second;
        }
        apply_butterfly_algorithm(config, tuned_algorithm);
        store_cache(handle->cache_path, key, tuned_algorithm);
        selected.source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
        selected.reason = compact_rank2_fft
                               ? rank2_fft_selection_reason(rank2_policy,
                                                            "measured compact rank-2 FFT candidates",
                                                            tuned_algorithm, rejected)
                               : "measured physical-axis candidates";
        selected.reason += "; kernel_ms=" + std::to_string(tuned_milliseconds);
    } else {
        // The compact path is an explicit policy choice.  AUTO now consults
        // the hardware registry/model, so a targeted optimization cannot
        // silently shadow a calibrated mapping on another GPU or workload.
        if (compact_rank2_fft) {
            const auto candidate = select_compact_rank2_fft_candidate(config, rank2_policy);
            config = candidate.config;
            selected.source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
            selected.reason = rank2_fft_selection_reason(rank2_policy,
                                                         "configured compact rank-2 FP32 FFT",
                                                         candidate.algorithm, candidate.rejected);
        } else {
            const auto info = choose_butterfly_mapping(config);
            selected.source = mapping_selection_source(info);
            selected.reason = info.reason;
        }
    }
    selected.algorithm = butterfly_algorithm_id(config);
    selected.config = std::move(config);
    return selected;
}

struct NttAxisSelection {
    cuntt::PlanConfig config;
    std::string algorithm;
    cubutterflySelectionSource_t source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
    std::string reason;
};

NttAxisSelection select_ntt_axis(cubutterflyHandle_t handle,
                                 const cubutterflyDescriptor& descriptor,
                                 std::uint32_t source_axis, std::size_t length,
                                 std::size_t batch) {
    auto config = make_axis_ntt_config(descriptor, length, batch);
    const auto axis_descriptor = make_axis_descriptor(descriptor, source_axis, length, batch);
    const auto key = workload_key(axis_descriptor);
    const auto cached = descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT
                            ? lookup_cache(handle->cache_path, key)
                            : std::string{};
    NttAxisSelection selected;
    if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
        apply_ntt_algorithm(config, axis_algorithm(descriptor,source_axis));
        selected.source = CUBUTTERFLY_SELECTION_EXPLICIT;
        selected.reason = "applied the explicit algorithm to the physical NTT axis";
    } else if (!cached.empty()) {
        apply_ntt_algorithm(config, cached);
        selected.source = CUBUTTERFLY_SELECTION_USER_CACHE;
        selected.reason = "matched the physical NTT axis in the user selection cache";
    } else if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE) {
        const auto tuned = tune_ntt(config, handle->stream);
        apply_ntt_algorithm(config, tuned.first);
        store_cache(handle->cache_path, key, tuned.first);
        selected.source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
        selected.reason = "measured physical NTT-axis candidates; kernel_ms=" + std::to_string(tuned.second);
    } else {
        const auto choice = cuntt::detail::select_ntt_mapping(config);
        config = choice.config;
        selected.source = mapping_selection_source(choice.info);
        selected.reason = choice.info.reason;
    }
    selected.algorithm = ntt_algorithm_id(config);
    selected.config = std::move(config);
    return selected;
}

cubutterflySelectionSource_t combined_source(
    cubutterflySelectionSource_t current, cubutterflySelectionSource_t next) {
    if (current == CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK || next == CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK)
        return CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK;
    if (current == CUBUTTERFLY_SELECTION_RUNTIME_MEASURED || next == CUBUTTERFLY_SELECTION_RUNTIME_MEASURED)
        return CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
    if (current == CUBUTTERFLY_SELECTION_USER_CACHE || next == CUBUTTERFLY_SELECTION_USER_CACHE)
        return CUBUTTERFLY_SELECTION_USER_CACHE;
    if (current == CUBUTTERFLY_SELECTION_EXPLICIT || next == CUBUTTERFLY_SELECTION_EXPLICIT)
        return CUBUTTERFLY_SELECTION_EXPLICIT;
    if (current == CUBUTTERFLY_SELECTION_MEASURED_TABLE || next == CUBUTTERFLY_SELECTION_MEASURED_TABLE)
        return CUBUTTERFLY_SELECTION_MEASURED_TABLE;
    return CUBUTTERFLY_SELECTION_STATIC_MODEL;
}

std::string get_mapping(cubutterflyPlan_t plan) {
    std::size_t bytes = 0;
    require(cubutterflyPlanGetMappingJson(plan, nullptr, &bytes) == CUBUTTERFLY_STATUS_SUCCESS,
            "cannot query child mapping");
    std::vector<char> text(bytes);
    require(cubutterflyPlanGetMappingJson(plan, text.data(), &bytes) == CUBUTTERFLY_STATUS_SUCCESS,
            "cannot read child mapping");
    return text.data();
}

nlohmann::json xd_contract(const cubutterflyDescriptor& d) {
    return cubutterfly::detail::semantic_json(d);
}

nlohmann::json xd_target() {
    const auto device = cuntt::current_device_info();
    return {{"name",device.name}, {"memory_bytes",device.global_memory_bytes},
            {"sm_major",device.compute_major}, {"sm_minor",device.compute_minor},
            {"multiprocessors",device.multiprocessors},
            {"runtime_fingerprint",cuntt::detail::runtime_fingerprint()}};
}

// Complete-plan tuning must not multiply a production-size allocation without
// a bound. A plan remains executable when empirical promotion is omitted.
std::size_t xd_tuning_budget() {
    const char* value = std::getenv("CUBUTTERFLY_XD_TUNE_MAX_BYTES");
    if (!value || !*value) {
        // Keep installation-time tuning bounded while allowing complete plans
        // whose live buffers are larger than the historical fixed 512 MiB
        // guard.  The final xd_tuning_status() check still requires the
        // temporary allocation to fit in a fraction of currently free VRAM.
        constexpr std::size_t kFloor = 512ULL * 1024 * 1024;
        constexpr std::size_t kCap = 2ULL * 1024 * 1024 * 1024;
        std::size_t free_bytes = 0, total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess)
            return std::max(kFloor, std::min(kCap, free_bytes / 4));
        return kFloor;
    }
    std::size_t used = 0;
    require(*value != '-', "xD tuning byte budget must be nonnegative");
    const auto bytes = std::stoull(value, &used);
    require(value[used] == '\0' && bytes <= std::numeric_limits<std::size_t>::max(),
            "invalid xD tuning byte budget");
    return static_cast<std::size_t>(bytes);
}

// CUDA-graph timing is stable enough for a 1% promotion margin on the normal
// installation-time search.  Shared or noisy systems can request the more
// conservative historical value through the explicit environment knob.
double xd_promotion_factor() {
    constexpr double kDefault = 0.99;
    const char* value = std::getenv("CUBUTTERFLY_XD_PROMOTION_MARGIN");
    if (!value || !*value) return kDefault;
    char* end = nullptr;
    const double margin = std::strtod(value, &end);
    if (end == value || *end != '\0' || !std::isfinite(margin) || margin < 0.0 || margin >= 1.0)
        return kDefault;
    return 1.0 - margin;
}

// Keep the memory gate explainable at every xD lowering boundary.  A boolean
// alone caused large workloads to silently fall back to the first (usually
// packed) candidate, making a performance loss look like a kernel limitation.
// The diagnostic is deliberately separate from the semantic mapping so it is
// safe to retain and replay a mapping across processes.
struct XdTuningStatus {
    bool fits = false;
    std::size_t budget_bytes = 0;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    std::string reason = "not-checked";
};

XdTuningStatus xd_tuning_status(std::size_t io_bytes, std::size_t work_bytes) {
    XdTuningStatus status;
    status.budget_bytes = xd_tuning_budget();
    if (io_bytes > status.budget_bytes / 2) {
        status.reason = "io-exceeds-half-budget";
        return status;
    }
    const auto remaining = status.budget_bytes - 2 * io_bytes;
    if (work_bytes > remaining) {
        status.reason = "workspace-exceeds-budget";
        return status;
    }
    const auto query = cudaMemGetInfo(&status.free_bytes, &status.total_bytes);
    if (query != cudaSuccess) {
        status.reason = std::string("cudaMemGetInfo-failed:") + cudaGetErrorString(query);
        return status;
    }
    const auto required = 2 * io_bytes + work_bytes;
    if (required > status.free_bytes / 2) {
        status.reason = "temporary-allocation-exceeds-half-free-memory";
        return status;
    }
    status.fits = true;
    status.reason = "within-budget";
    return status;
}

std::size_t xd_prefix_launches(const cubutterflyDescriptor& d) {
    // Non-contiguous per-axis cuFFT batches cannot span gaps between prefixes.
    std::size_t sum = 0, prefix = d.batch;
    for (unsigned axis = 0; axis + 1 < d.rank; ++axis) {
        if (prefix > 256 || sum > 256 - prefix) return 257;
        sum += prefix;
        prefix = checked_product(prefix,d.extents[axis]);
    }
    return sum;
}

// A bounded xD transform has a small, meaningful axis-order space.  The
// default order preserves the historical lowering; all permutations (at most
// six) let the tuner account for retained layouts and axis access patterns. For
// compact real FFT the final real axis is fixed by the R2C/C2R boundary, so
// only the preceding complex axes are permuted.
std::vector<std::vector<std::size_t>> xd_axis_orders(
    const cubutterflyDescriptor& d, bool complex_only, const nlohmann::json& replay) {
    if (!replay.is_null() && replay.contains("physical") &&
        replay.at("physical").contains("axis_order")) {
        std::vector<std::size_t> order = replay.at("physical").at("axis_order")
                                             .get<std::vector<std::size_t>>();
        return {std::move(order)};
    }
    const std::size_t count = complex_only ? d.rank : d.rank - 1U;
    std::vector<std::size_t> default_order;
    default_order.reserve(count);
    for (std::size_t axis = count; axis-- > 0;)
        default_order.push_back(axis);
    std::vector<std::vector<std::size_t>> orders{default_order};
    auto alternate = default_order;
    std::sort(alternate.begin(), alternate.end());
    // All permutations for up to three complex axes. At higher rank, use
    // cyclic rotations of forward/reverse traversal (<=2*rank candidates).
    // This bounds planning resources; explicit replay accepts any legal order.
    if (count > 3) {
        for (auto seed : {default_order, alternate}) {
            for (std::size_t shift = 0; shift < count; ++shift) {
                if (std::find(orders.begin(), orders.end(), seed) == orders.end())
                    orders.push_back(seed);
                std::rotate(seed.begin(), seed.begin() + 1, seed.end());
            }
        }
        return orders;
    }
    do {
        if (alternate != default_order) orders.push_back(alternate);
    } while (std::next_permutation(alternate.begin(), alternate.end()));
    return orders;
}

std::unique_ptr<cubutterflyPlan> make_child_plan(cubutterflyHandle_t handle, cubutterflyDescriptor d) {
    cubutterflyPlan_t raw = nullptr;
    const auto status = cubutterflyCreatePlan(handle, &d, &raw);
    if (status == CUBUTTERFLY_STATUS_NOT_SUPPORTED) throw unsupported_error("child lowering not supported");
    if (status != CUBUTTERFLY_STATUS_SUCCESS)
        throw std::runtime_error(std::string("child lowering: ") + cubutterflyGetStatusString(status));
    return std::unique_ptr<cubutterflyPlan>(raw);
}

using XdAxisResources = cuntt::detail::XdAxisResources;

// Defined with the other semantic-order helpers below the xD selector.
bool nonnatural_order(const cubutterflyDescriptor& d);

std::vector<XdAxisResources> xd_uniform_local_resources(
    const std::vector<std::size_t>& extents, std::size_t complex_axis_count,
    unsigned block_cap, unsigned local_twiddle_policy = 0) {
    require(complex_axis_count <= extents.size(), "xD local resource rank is invalid");
    require(local_twiddle_policy <= 1,
            "xD local twiddle policy must be 0 or 1");
    std::vector<XdAxisResources> resources;
    resources.reserve(complex_axis_count);
    for (std::size_t axis = 0; axis < complex_axis_count; ++axis) {
        const auto length = extents[axis];
        const auto core = cuntt::detail::xd_local_fft_available(length)
                              ? cuntt::detail::XdAxisCore::ThreadShared
                              : cuntt::detail::XdAxisCore::CufftdxBlock;
        resources.push_back({core, 8U, 4U, block_cap, 0U,
            core == cuntt::detail::XdAxisCore::ThreadShared ? local_twiddle_policy : 0U});
    }
    return resources;
}

bool xd_mapping_has_axis_core(const nlohmann::json& physical, const char* core) {
    const auto has_core = [core](const nlohmann::json& axes) {
        return axes.is_array() && std::any_of(axes.begin(), axes.end(), [core](const auto& axis) {
            return axis.value("core", "") == core;
        });
    };
    if (physical.contains("axes") && has_core(physical.at("axes")))
        return true;
    return physical.contains("tile_tail") && physical.at("tile_tail").is_object() &&
           physical.at("tile_tail").contains("axes") &&
           has_core(physical.at("tile_tail").at("axes"));
}

std::vector<XdAxisResources> xd_axis_resources(const nlohmann::json& physical) {
    std::vector<XdAxisResources> resources;
    if (!physical.contains("axis_resources")) return resources;
    const auto& entries = physical.at("axis_resources");
    require(entries.is_array() && !entries.empty(), "xD axis resources must be a nonempty array");
    for (const auto& entry : entries) {
        // Check the encoded range before narrowing. JSON's get<unsigned>()
        // alone can wrap oversized/negative integers into a valid design point.
        const auto parameter = [&](const char* name, bool optional = false) {
            if (optional && !entry.contains(name)) return 0U;
            const auto& value = entry.at(name);
            require(value.is_number_unsigned(),
                    std::string("xD axis parameter must be an unsigned integer: ") + name);
            const auto encoded = value.get<std::uint64_t>();
            require(encoded <= std::numeric_limits<unsigned>::max(),
                    std::string("xD axis parameter exceeds unsigned range: ") + name);
            return static_cast<unsigned>(encoded);
        };
        XdAxisResources resource{
            cuntt::detail::xd_axis_core_from_name(entry.at("core").get<std::string>()),
            parameter("ept"), parameter("register_lanes"), parameter("block_cap"),
            parameter("exchange"), parameter("local_twiddle_policy", true),
            parameter("ffts_per_block", true)};
        cuntt::detail::validate_xd_axis_resource(resource);
        resources.push_back(resource);
    }
    return resources;
}

std::unique_ptr<cuntt::detail::XdRealFftPlan> xd_plan_with_resources(
    const nlohmann::json& physical, const std::vector<XdAxisResources>& resources) {
    return std::make_unique<cuntt::detail::XdRealFftPlan>(
        physical.at("extents").get<std::vector<std::size_t>>(), physical.at("batch").get<std::size_t>(),
        physical.at("inverse").get<bool>(), physical.at("normalize_inverse").get<bool>(),
        physical.at("fp64").get<bool>(), physical.at("complex_only").get<bool>(),
        physical.at("output_scale").get<double>(), physical.at("packed_axes").get<bool>(),
        physical.at("axis_order").get<std::vector<std::size_t>>(), physical.at("fused_axes").get<bool>(),
        physical.at("axis_ept").get<unsigned>(), physical.at("paired_axes").get<bool>(),
        cuntt::detail::xd_axis_core_from_name(physical.at("axis_core").get<std::string>()),
        cuntt::detail::xd_pair_core_from_name(physical.at("pair_core").get<std::string>()),
        physical.at("register_lanes").get<unsigned>(), physical.at("pair_layout").get<unsigned>(),
        physical.value("batch_tile", std::size_t{0}), physical.value("axis_ffts_per_block_cap", 0U),
        physical.value("axis_exchange_policy", 0U), resources,
        physical.value("c2r_prefix_scale", false));
}

bool xd_axis_refinement_seed(const nlohmann::json& physical) {
    return physical.value("kind", "") == "xd-real-fft" && physical.value("fused_axes", false) &&
           physical.value("packed_axes", false) && !physical.value("paired_axes", false);
}

struct XdAxisRefinement {
    std::unique_ptr<cuntt::detail::XdRealFftPlan> plan;
    nlohmann::json physical;
    double milliseconds;
};

// One coordinate sweep of the compiled resources. This adds independent-axis
// choices without enumerating their Cartesian product. Every proposal is timed
// as a complete transform, including batch tails and real/layout boundaries.
XdAxisRefinement refine_xd_axis_resources(const nlohmann::json& seed, double seed_ms,
    void* input, void* output, void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream, bool graph_timing, nlohmann::json& measurements,
    nlohmann::json& omitted) {
    XdAxisRefinement result{nullptr, seed, seed_ms};
    const double promotion_factor = xd_promotion_factor();
    const auto order = seed.at("axis_order").get<std::vector<std::size_t>>();
    auto resources = xd_axis_resources(seed);
    if (resources.empty()) resources.assign(order.size(), XdAxisResources{
        cuntt::detail::xd_axis_core_from_name(seed.at("axis_core").get<std::string>()),
        seed.at("axis_ept").get<unsigned>(), seed.at("register_lanes").get<unsigned>(),
        seed.value("axis_ffts_per_block_cap", 0U), seed.value("axis_exchange_policy", 0U)});
    // A legacy register family applies only to eligible strided axes. Resolve
    // its actual fallbacks before making the per-axis request explicit.
    for (const auto& axis : seed.at("axes")) {
        auto& resource = resources.at(axis.at("axis").get<std::size_t>());
        const auto actual_core = axis.at("core").get<std::string>();
        if (actual_core == "vkfft-axis") {
            resource = XdAxisResources{cuntt::detail::XdAxisCore::Vkfft,8,4,0,0};
        } else if (actual_core == "thread-shared-axis") {
            resource.core = cuntt::detail::XdAxisCore::ThreadShared;
            resource.register_lanes = 4;
        } else if (actual_core != "register-cooperative-axis") {
            resource.core = cuntt::detail::XdAxisCore::CufftdxBlock;
            resource.register_lanes = 4;
        }
    }
    const auto options = cuntt::detail::xd_axis_resource_candidates(
        cuntt::detail::xd_vkfft_axis_available());
    auto signature = [](const nlohmann::json& physical) {
        auto key = physical.at("axes");
        if (physical.contains("tile_tail")) key.push_back(physical.at("tile_tail").at("axes"));
        return key.dump();
    };
    std::vector<std::string> seen{signature(seed)};
    for (const auto axis : order) {
        const auto base = resources;
        for (const auto& option : options) {
            auto trial_resources = base;
            trial_resources.at(axis) = option;
            try {
                auto trial = xd_plan_with_resources(seed, trial_resources);
                auto physical = nlohmann::json::parse(trial->mapping_json());
                const auto key = signature(physical);
                if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;
                seen.push_back(key);
                if (trial->workspace_size() > workspace_bytes) {
                    omitted.push_back({{"lowering","axis-tiled-mixed"},{"axis",axis},
                        {"reason","axis refinement exceeds allocated tuning workspace"}});
                    continue;
                }
                if (trial->workspace_size()) trial->set_workspace(workspace,workspace_bytes);
                auto execute = [&] { trial->execute(input,output,stream); };
                // A failed graph proposal is omitted, never mixed with direct
                // timing from a different protocol in this refinement sweep.
                auto graph = graph_timing ? capture_xd_cuda_graph(stream,execute) : nullptr;
                std::vector<double> times;
                for (unsigned repeat = 0; repeat < 3; ++repeat)
                    times.push_back(graph_timing ? measure_xd_cuda_graph(stream,graph->executable())
                                                 : measure_cuda_candidate_synchronized(execute));
                std::sort(times.begin(),times.end());
                nlohmann::json measured{{"lowering","axis-tiled-mixed"},
                    {"search_phase","one-axis-coordinate-sweep"},{"changed_axis",axis},
                    {"axis_resources",physical.at("axis_resources")},{"axis_order",order},
                    {"batch_tile",physical.value("batch_tile",std::size_t{0})},
                    {"c2r_prefix_scale",physical.value("c2r_prefix_scale",false)},
                    {"median_ms",times[1]},{"trials_ms",times},
                    {"timing_protocol",graph_timing ? "cuda-graph-steady-state" : "cuda-device-synchronized"}};
                if (graph_timing) {
                    measured["warmup_plan_executions"] = 1;
                    measured["warmup_graph_launches"] = kXdGraphWarmupLaunches;
                    measured["warmup_total_plan_executions"] = 1 + kXdGraphWarmupLaunches*kXdGraphPlanExecutions;
                    measured["captured_plan_executions"] = kXdGraphPlanExecutions;
                    measured["timed_graph_launches"] = kXdGraphTimedLaunches;
                    measured["timed_plan_executions"] = kXdGraphPlanExecutions*kXdGraphTimedLaunches;
                } else {
                    measured["warmup_launches"] = 2;
                    measured["timed_launches"] = 5;
                }
                measurements.push_back(std::move(measured));
                trial->set_workspace(nullptr,0);
                if (times[1] < result.milliseconds * promotion_factor) {
                    resources = trial_resources;
                    result = {std::move(trial),std::move(physical),times[1]};
                }
            } catch (const std::exception& error) {
                (void)cudaStreamSynchronize(stream);
                omitted.push_back({{"lowering","axis-tiled-mixed"},{"axis",axis},
                    {"core",cuntt::detail::xd_axis_core_name(option.core)},
                    {"ept",option.ept},{"register_lanes",option.register_lanes},
                    {"block_cap",option.block_cap},{"exchange",option.exchange},
                    {"exact_ffts_per_block",option.ffts_per_block},
                    {"reason",error.what()}});
            }
        }
    }
    return result;
}

// xD selection wraps executable implementations instead of relabeling a
// vendor whole transform. Complete-plan timing includes every boundary.
std::unique_ptr<cubutterflyPlan> select_xd_plan(cubutterflyHandle_t handle,
    const cubutterflyDescriptor& d, const nlohmann::json& explicit_replay,
    const std::string& forced_lowering = {}) {
    require(explicit_replay.is_null() ||
                !explicit_replay.value("physical",nlohmann::json::object()).value("c2r_prefix_scale",false),
            "C2R prefix scaling is invalid for a complex or non-FFT xD request");
    // The promotion policy changes which measured mapping is valid. Include
    // its effective factor so a cache created with the historical 3% guard
    // cannot silently bypass the default 1% installation search.
    const auto key = workload_key(d) + "|xd-v26-packed-pair-fp64|vkfft-" +
                     (cuntt::detail::xd_vkfft_axis_available() ? "1" : "0") +
                     "|raw-normalized|promotion-factor-" +
                     std::to_string(xd_promotion_factor()) + "|norm-" +
                     (d.normalize_inverse ? "1" : "0");
    nlohmann::json replay = explicit_replay;
    bool cached = false;
    if (replay.is_null() && d.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT) {
        const auto text = lookup_cache(handle->cache_path, key);
        if (!text.empty()) { replay = nlohmann::json::parse(text); cached = true; }
    }
    if (!replay.is_null()) {
        require(replay.value("schema_version", 0) == 1 && replay.value("kind", "") == "xd" &&
                    replay.at("contract") == xd_contract(d) && replay.at("target") == xd_target(),
                "xD replay contract or target mismatch");
    }
    const std::string requested = replay.is_null() ? forced_lowering : replay.at("lowering").get<std::string>();
    struct Candidate {
        std::string name;
        std::unique_ptr<cubutterflyPlan> plan;
        nlohmann::json physical;
    };
    std::vector<Candidate> candidates;
    nlohmann::json omitted = nlohmann::json::array();
    auto shell = [&] {
        auto p = std::make_unique<cubutterflyPlan>();
        p->descriptor = d;
        p->physical_extents = d.extents;
        const bool embedding = d.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
        if (embedding) for (auto& extent : p->physical_extents) extent = next_power_of_two(extent);
        const bool inverse = d.direction == CUBUTTERFLY_DIRECTION_INVERSE;
        const auto& input_extents = embedding && inverse ? p->physical_extents : d.extents;
        const auto& output_extents = embedding && !inverse ? p->physical_extents : d.extents;
        p->element_bytes = element_bytes(d.storage);
        p->input_bytes = checked_product(strided_elements(d, true, input_extents), p->element_bytes);
        p->output_bytes = checked_product(strided_elements(d, false, output_extents), p->element_bytes);
        return p;
    };
    auto wanted = [&](const std::string& name) { return requested.empty() || requested == name; };
    // Keep the old executable as the first candidate: equal/noisy measurements
    // favour the known path, and a new lowering cannot displace it on a model
    // prediction alone.
    const bool power2 = std::all_of(d.extents.begin(), d.extents.end(), is_power_of_two);
    const bool has_unit_axis = std::any_of(d.extents.begin(),d.extents.end(),
                                          [](std::size_t extent) { return extent == 1; });
    const bool composite_available = !has_unit_axis && (d.rank <= 2 || power2 ||
                                    d.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING);
    if (wanted("composite") && composite_available) {
        try {
            auto base = d;
            base.execution_mode = CUBUTTERFLY_EXECUTION_COMPOSITE;
            base.algorithm_policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
            base.explicit_algorithm.clear();
            if (!replay.is_null()) {
                base.algorithm_policy = CUBUTTERFLY_ALGORITHM_EXPLICIT;
                base.explicit_algorithm = replay.at("physical").dump();
            }
            // The composite plan is an internal candidate of the outer xD
            // selector. Its portable per-axis fallback must not emit a
            // public profile-miss warning while the outer selector is about
            // to measure complete candidates.
            ScopedProfileMissSuppression suppress_profile_miss(handle);
            auto p = make_child_plan(handle, base);
            auto mapping = nlohmann::json::parse(get_mapping(p.get()));
            candidates.push_back({"composite", std::move(p), std::move(mapping)});
        } catch (const std::exception& error) {
            // Composite is one executable lowering among several xD
            // candidates.  A resource/backend failure for one axis must not
            // prevent resident or tiled candidates from being considered.
            // An explicitly requested composite remains strict.
            if (!requested.empty()) throw;
            omitted.push_back({{"lowering", "composite"},
                               {"reason", error.what()}});
        }
    } else if (wanted("composite")) {
        omitted.push_back({{"lowering","composite"}, {"reason",has_unit_axis ?
            "legacy axis core has no extent-one lowering" : "rank-three Bluestein composition is not implemented"}});
    }
    // A complete vendor FFT is a legal computation core for packed rank-2
    // C2C transforms. Keep it in the same complete-plan search as axis-level
    // cores so the framework can choose it when cuFFT's whole-transform plan
    // is faster for a particular shape. The candidate is deliberately narrow:
    // arbitrary strides, logical order conversion, real-spectrum boundaries,
    // and custom output epilogues must retain their semantic adapters.
    const bool whole_cufft_available =
        d.rank == 2 && d.op == CUBUTTERFLY_OPERATOR_FFT &&
        d.fft_type == CUBUTTERFLY_FFT_C2C &&
        d.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
        d.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE &&
        (d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 ||
         d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64) &&
        d.fft_output_scale == 1.0 && d.output_scale == 1.0 &&
        !nonnatural_order(d) && packed_layout(d);
    if (wanted("cufft-whole") && whole_cufft_available) {
        try {
            auto p = shell();
            p->kind = cubutterflyPlan::Kind::DirectFft;
            p->direct_fft = std::make_unique<cuntt::detail::DirectFftPlan>(
                d.extents, d.batch,
                d.direction == CUBUTTERFLY_DIRECTION_INVERSE,
                d.normalize_inverse,
                d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64);
            p->workspace_bytes = 0;
            p->algorithm = "cufft-whole";
            p->reason = "vendor whole-transform FFT core; complete plan has no layout boundary";
            nlohmann::json mapping{{"schema_version", 1}, {"kind", "xd-vendor-whole-fft"},
                {"contract", xd_contract(d)}, {"target", xd_target()},
                {"extents", d.extents}, {"batch", d.batch},
                {"inverse", d.direction == CUBUTTERFLY_DIRECTION_INVERSE},
                {"normalize_inverse", d.normalize_inverse},
                {"fp64", d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64}};
            if (!replay.is_null())
                require(mapping == replay.at("physical"), "whole cuFFT replay mapping mismatch");
            candidates.push_back({"cufft-whole", std::move(p), std::move(mapping)});
        } catch (const std::exception& error) {
            if (!requested.empty()) throw;
            omitted.push_back({{"lowering", "cufft-whole"}, {"reason", error.what()}});
        }
    } else if (wanted("cufft-whole") && !whole_cufft_available) {
        omitted.push_back({{"lowering", "cufft-whole"},
                           {"reason", "requires packed rank-2 standard C2C complex FFT with natural layout"}});
    }
    const bool packed = packed_layout(d) && d.length_mode == CUBUTTERFLY_LENGTH_STANDARD;
    if (packed && power2) {
        // Tensor products of these butterflies are precisely a concatenation
        // of bit stages. Unlike FFT/NTT, no cross-axis twiddle is introduced.
        if (wanted("flattened") && (d.op == CUBUTTERFLY_OPERATOR_FWHT ||
            d.op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || d.op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA ||
            d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2)) {
          try {
            auto flat = d;
            flat.rank = 1;
            flat.extents = {packed_nd_layout(d.extents, 1).points};
            flat.input_orders.assign(1, CUBUTTERFLY_ORDER_NATURAL);
            flat.output_orders.assign(1, CUBUTTERFLY_ORDER_NATURAL);
            for (auto& matrices : flat.stage_matrices) matrices.clear();
            flat.execution_mode = CUBUTTERFLY_EXECUTION_COMPOSITE;
            flat.input_strides = flat.output_strides = {1};
            flat.input_batch_stride = flat.output_batch_stride = flat.extents[0];
            flat.algorithm_policy = d.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE
                ? CUBUTTERFLY_ALGORITHM_MEASURE : CUBUTTERFLY_ALGORITHM_DEFAULT;
            flat.explicit_algorithm.clear();
            if (d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
                flat.stage_matrices[0].clear();
                for (std::size_t axis = d.rank; axis-- > 0;) {
                    const auto& matrices = d.stage_matrices[axis];
                    const auto stages = integer_log2(d.extents[axis]);
                    require(matrices.size() == 1 || matrices.size() == stages, "invalid structured axis coefficients");
                    for (unsigned stage = 0; stage < stages; ++stage)
                        flat.stage_matrices[0].push_back(matrices[matrices.size() == 1 ? 0 : stage]);
                }
            }
            if (!replay.is_null()) {
                flat.algorithm_policy = CUBUTTERFLY_ALGORITHM_EXPLICIT;
                flat.explicit_algorithm = replay.at("physical").dump();
            }
            auto p = make_child_plan(handle, flat);
            auto mapping = nlohmann::json::parse(get_mapping(p.get()));
            candidates.push_back({"flattened", std::move(p), std::move(mapping)});
          } catch (const std::exception& error) {
            if (!requested.empty()) throw;
            omitted.push_back({{"lowering","flattened"}, {"reason",error.what()}});
          }
        }
        // Common organization axes, independent of the arithmetic operator.
        // Keep the legacy order first at each width; vary order at the default
        // width and width at the legacy order instead of a Cartesian sweep.
        std::vector<std::size_t> legacy_order;
        const bool reverse_resident = d.direction == CUBUTTERFLY_DIRECTION_INVERSE &&
            d.op != CUBUTTERFLY_OPERATOR_FFT && d.op != CUBUTTERFLY_OPERATOR_NTT;
        for (std::size_t axis = 0; axis < d.rank; ++axis)
            legacy_order.push_back(reverse_resident ? d.rank - 1 - axis : axis);
        auto resident_orders = xd_axis_orders(d, true, replay);
        if (replay.is_null()) {
            resident_orders.erase(std::remove(resident_orders.begin(), resident_orders.end(), legacy_order),
                                  resident_orders.end());
            resident_orders.insert(resident_orders.begin(), legacy_order);
        }
        std::vector<unsigned> resident_threads{128U, 256U, 32U, 64U, 512U, 1024U};
        if (requested.rfind("resident-t", 0) == 0) {
            const auto text = requested.substr(std::string("resident-t").size());
            require(!text.empty() && std::all_of(text.begin(), text.end(),
                        [](char c) { return c >= '0' && c <= '9'; }),
                    "resident thread width must be an unsigned decimal integer");
            const auto width = std::stoull(text);
            require(width > 0 && width <= 1024 && width % 32 == 0,
                    "resident thread width must be a warp multiple up to 1024");
            resident_threads = {static_cast<unsigned>(width)};
        }
        for (unsigned threads : resident_threads) {
            const auto name = "resident-t" + std::to_string(threads);
            if (!wanted(name)) continue;
          for (const auto& order : resident_orders) {
            if (replay.is_null() && threads != 128U && order != legacy_order) continue;
            cuntt::XdResidentConfig config;
            using Op = cuntt::XdOperator;
            switch (d.op) {
                case CUBUTTERFLY_OPERATOR_FFT: config.op = Op::Fft; break;
                case CUBUTTERFLY_OPERATOR_NTT: config.op = Op::Ntt; break;
                case CUBUTTERFLY_OPERATOR_FWHT: config.op = Op::Fwht; break;
                case CUBUTTERFLY_OPERATOR_SUBSET_ZETA: config.op = Op::SubsetZeta; break;
                case CUBUTTERFLY_OPERATOR_SUPERSET_ZETA: config.op = Op::SupersetZeta; break;
                case CUBUTTERFLY_OPERATOR_STRUCTURED_2X2: config.op = Op::Structured2x2; break;
            }
            using Storage = cuntt::XdStorageType;
            bool supported_type = true;
            switch (d.storage) {
                case CUBUTTERFLY_DATA_FP16: config.storage = Storage::Fp16; break;
                case CUBUTTERFLY_DATA_BF16: config.storage = Storage::Bf16; break;
                case CUBUTTERFLY_DATA_FP32: config.storage = Storage::Fp32; break;
                case CUBUTTERFLY_DATA_FP64: config.storage = Storage::Fp64; break;
                case CUBUTTERFLY_DATA_COMPLEX_FP32: config.storage = Storage::ComplexFp32; break;
                case CUBUTTERFLY_DATA_COMPLEX_FP64: config.storage = Storage::ComplexFp64; break;
                case CUBUTTERFLY_DATA_UINT32: config.storage = Storage::Uint32; break;
                case CUBUTTERFLY_DATA_UINT64: config.storage = Storage::Uint64; break;
                default: supported_type = false;
            }
            config.compute = static_cast<cuntt::XdComputeType>(d.compute);
            config.extents = d.extents; config.batch = d.batch;
            config.inverse = d.direction == CUBUTTERFLY_DIRECTION_INVERSE;
            config.normalize_inverse = d.normalize_inverse;
            config.modulus = d.modulus; config.threads = threads;
            config.axis_order = order;
            if (d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
                for (std::size_t axis = d.rank; axis-- > 0;) {
                    const auto stages = integer_log2(d.extents[axis]);
                    const auto& m = d.stage_matrices[axis];
                    require(m.size() == 1 || m.size() == stages, "invalid structured axis matrices");
                    for (unsigned stage = 0; stage < stages; ++stage) {
                        const auto& v = m[m.size() == 1 ? 0 : stage];
                        config.stage_matrices.push_back({v.m00,v.m01,v.m10,v.m11});
                    }
                }
            }
            const auto available = cuntt::XdResidentPlan::check(config);
            if (!supported_type || !available.available) {
                omitted.push_back({{"lowering", name}, {"reason", available.reason}});
                continue;
            }
            try {
                auto p = shell();
                p->xd_resident = std::make_unique<cuntt::XdResidentPlan>(config);
                p->workspace_bytes = p->xd_resident->workspace_size();
                auto mapping = nlohmann::json::parse(p->xd_resident->mapping_json());
                if (!replay.is_null()) require(mapping == replay.at("physical"), "resident replay resource/mapping mismatch");
                candidates.push_back({name, std::move(p), std::move(mapping)});
            } catch (const std::exception& error) {
                if (!requested.empty()) throw;
                omitted.push_back({{"lowering",name}, {"reason",error.what()}});
            }
          }
        }
    }
    if (wanted("strided-axis-cufft") && packed && d.op == CUBUTTERFLY_OPERATOR_FFT &&
        d.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE &&
        (d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64)) {
      if (requested.empty() && xd_prefix_launches(d) > 256) {
        omitted.push_back({{"lowering","strided-axis-cufft"}, {"reason","tuning budget: more than 256 prefix launches"}});
      } else {
        const auto orders = xd_axis_orders(d, true, replay);
        for (const auto& order : orders) {
         try {
          auto p = shell();
          p->xd_fft = std::make_unique<cuntt::detail::XdRealFftPlan>(d.extents,d.batch,
              d.direction == CUBUTTERFLY_DIRECTION_INVERSE,d.normalize_inverse,
              d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64,true,1.0,false,order);
          p->workspace_bytes = p->xd_fft->workspace_size();
          auto mapping = nlohmann::json::parse(p->xd_fft->mapping_json());
          if (!replay.is_null()) require(mapping == replay.at("physical"), "strided FFT replay mismatch");
          candidates.push_back({"strided-axis-cufft", std::move(p), std::move(mapping)});
         } catch (const std::exception& error) {
           if (!requested.empty()) throw;
           omitted.push_back({{"lowering","strided-axis-cufft"}, {"reason",error.what()}});
         }
        }
      }
    }
    struct AxisCoreChoice {
        std::string name;
        bool fused;
        unsigned ept;
        bool paired;
        cuntt::detail::XdAxisCore axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
        cuntt::detail::XdPairCore pair_core = cuntt::detail::XdPairCore::Thread;
        unsigned register_lanes = 4;
        unsigned pair_layout = 0;
        unsigned block_cap = 0;
        unsigned exchange = 0;
        bool local = false;
        unsigned local_twiddle_policy = 0;
    };
    std::vector<AxisCoreChoice> axis_core_choices = {
        {"packed-axis-cufft", false, 8, false},
        {"axis-tiled-cufftdx", true, 8, false},
        {"axis-tiled-cufftdx-ept16", true, 16, false},
        {"axis-packed-cufftdx", true, 8, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 0, 4},
        {"axis-packed-cufftdx-ept16", true, 16, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 0, 4},
        {"axis-tiled-cufftdx-cap8", true, 8, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 8},
        {"axis-tiled-cufftdx-cap16", true, 8, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 16},
        {"axis-tiled-cufftdx-ept16-cap8", true, 16, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 8},
        {"axis-tiled-cufftdx-ept16-cap16", true, 16, false, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Thread, 4, 0, 16},
        {"pair-tiled-cufftdx", true, 8, true},
        {"pair-tiled-cufftdx-ept16", true, 16, true},
        {"pair-tiled-radix4", true, 8, true, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::Radix4Register},
        {"axis-tiled-register", true, 8, false, cuntt::detail::XdAxisCore::CooperativeRegister},
        {"pair-tiled-cooperative", true, 8, true, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::CooperativeRegister},
        {"pair-tiled-cooperative-register", true, 8, true, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::CooperativeRegister},
        {"axis-tiled-register-l8", true, 8, false, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::Thread, 8},
        {"axis-tiled-register-l16", true, 8, false, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::Thread, 16},
        {"pair-tiled-cooperative-register-l8", true, 8, true, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::CooperativeRegister, 8},
        {"pair-tiled-cooperative-register-l16", true, 8, true, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::CooperativeRegister, 16},
        {"pair-tiled-cooperative-layout1", true, 8, true, cuntt::detail::XdAxisCore::CufftdxBlock,
            cuntt::detail::XdPairCore::CooperativeRegister, 4, 1},
        {"pair-tiled-cooperative-register-layout1", true, 8, true, cuntt::detail::XdAxisCore::CooperativeRegister,
            cuntt::detail::XdPairCore::CooperativeRegister, 4, 1},
    };
    for (unsigned exchange : {1U,2U})
        for (unsigned ept : {8U,16U})
            for (unsigned cap : {0U,8U,16U}) {
                std::string name = "axis-tiled-cufftdx";
                if (ept == 16) name += "-ept16";
                if (cap) name += "-cap" + std::to_string(cap);
                name += exchange == 1 ? "-direct" : "-unrolled";
                axis_core_choices.push_back({name,true,ept,false,
                    cuntt::detail::XdAxisCore::CufftdxBlock,cuntt::detail::XdPairCore::Thread,4,0,cap,exchange});
            }
    // Append resource candidates after the existing inventory so noisy ties
    // retain earlier mappings. Address width is an explicit, range-checked
    // exchange variant; EPT changes the local arithmetic resource footprint.
    for (unsigned exchange : {0U,1U,2U,3U})
        for (unsigned ept : {4U,8U,16U}) {
            if (ept != 4 && exchange != 3) continue;
            for (unsigned cap : {0U,8U,16U}) {
                std::string name = "axis-tiled-cufftdx";
                if (ept != 8) name += "-ept" + std::to_string(ept);
                if (cap) name += "-cap" + std::to_string(cap);
                if (exchange) name += exchange == 1 ? "-direct" :
                                      exchange == 2 ? "-unrolled" : "-unrolled32";
                axis_core_choices.push_back({name,true,ept,false,
                    cuntt::detail::XdAxisCore::CufftdxBlock,cuntt::detail::XdPairCore::Thread,4,0,cap,exchange});
            }
        }
    // Keep local uniform families at the end of the inventory. Their
    // per-axis resource vector selects the local core for 64/128/256 and the
    // established Block fallback for every other axis.
    axis_core_choices.push_back({"axis-tiled-local",true,8,false,
        cuntt::detail::XdAxisCore::ThreadShared,cuntt::detail::XdPairCore::Thread,
        4,0,0,0,true});
    axis_core_choices.push_back({"axis-tiled-local-cap16",true,8,false,
        cuntt::detail::XdAxisCore::ThreadShared,cuntt::detail::XdPairCore::Thread,
        4,0,16,0,true});
    axis_core_choices.push_back({"axis-tiled-local-recurrence",true,8,false,
        cuntt::detail::XdAxisCore::ThreadShared,cuntt::detail::XdPairCore::Thread,
        4,0,0,0,true,1});
    axis_core_choices.push_back({"axis-vkfft",true,8,false,
        cuntt::detail::XdAxisCore::Vkfft,cuntt::detail::XdPairCore::Thread});
    for (const auto& choice : axis_core_choices) {
    const std::string family = choice.name;
    if (wanted(family) && packed && d.op == CUBUTTERFLY_OPERATOR_FFT &&
        d.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE &&
        (d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64)) {
      const bool local_available = choice.local && std::any_of(
          d.extents.begin(), d.extents.end(), [](std::size_t length) {
              return cuntt::detail::xd_local_fft_available(length);
          });
      if (choice.local && !local_available) {
        omitted.push_back({{"lowering",family},
                           {"reason","no compiled local core for lengths 64/128/256 and target SM"}});
        continue;
      }
      if (choice.axis_core == cuntt::detail::XdAxisCore::Vkfft &&
          !cuntt::detail::xd_vkfft_axis_available()) {
        omitted.push_back({{"lowering",family}, {"reason","VkFFT axis provider is not enabled in this build"}});
        continue;
      }
      // Paired cores have an independent capability contract. A complete pair
      // need not import any Block axis core (including a build without MathDx).
      if (!choice.local && !choice.paired && choice.fused && choice.axis_core != cuntt::detail::XdAxisCore::Vkfft &&
          !std::any_of(d.extents.begin(), d.extents.end(), [&](std::size_t length) {
              return cuntt::detail::xd_axis_fft_configuration_available(length,
                  d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64, choice.ept, choice.block_cap);
          })) {
        omitted.push_back({{"lowering",family}, {"reason","no compiled axis core for these lengths and target SM"}});
        continue;
      }
      const auto orders = xd_axis_orders(d, true, replay);
      if (requested.empty() && !choice.local && choice.block_cap &&
          std::none_of(d.extents.begin(), d.extents.end(), [&](std::size_t length) {
              return cuntt::detail::xd_axis_fft_available(length) &&
                  cuntt::detail::xd_axis_fft_block_batch(length,
                      d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64,choice.ept) > choice.block_cap;
          })) continue;  // The cap would generate exactly the existing kernels.
      for (const auto& order : orders) {
       std::vector<std::size_t> batch_tiles{0};
       if (!replay.is_null()) {
           batch_tiles = {replay.at("physical").value("batch_tile",std::size_t{0})};
       } else if (requested.empty() && choice.fused && d.rank == 3 && d.batch > 1) {
           // Propose a bounded neighborhood around the device's L2 capacity
           // for two live complex buffers. This is a search budget, not a
           // restriction on replayable batch tiles. Always time complete B.
           int device = 0, l2_bytes = 0;
           require(cudaGetDevice(&device) == cudaSuccess &&
                   cudaDeviceGetAttribute(&l2_bytes,cudaDevAttrL2CacheSize,device) == cudaSuccess,
                   "query xD batch tiling cache capacity");
           std::size_t live_bytes = d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64 ? 32 : 16;
           for (const auto extent : d.extents) live_bytes = checked_product(live_bytes,extent);
           const auto fit = static_cast<std::size_t>(std::max(l2_bytes,0)) / live_bytes;
           if (fit > 0 && fit < d.batch) {
               std::size_t power = 1;
               while (power <= fit / 2) power *= 2;
               for (const auto tile : {std::max(std::size_t{1},power/2),power,fit})
                   if (std::find(batch_tiles.begin(),batch_tiles.end(),tile) == batch_tiles.end())
                       batch_tiles.push_back(tile);
           }
       }
       for (const auto batch_tile : batch_tiles) {
       try {
        auto p = shell();
        std::vector<XdAxisResources> resources = replay.is_null()
            ? (choice.local ? xd_uniform_local_resources(d.extents, d.extents.size(),
                                                          choice.block_cap,
                                                          choice.local_twiddle_policy)
                            : std::vector<XdAxisResources>{})
            : xd_axis_resources(replay.at("physical"));
        if (choice.local && !replay.is_null())
            require(!resources.empty(), "local xD replay is missing axis_resources");
        const auto axis_core = choice.local ? cuntt::detail::XdAxisCore::ThreadShared : choice.axis_core;
        p->xd_fft = std::make_unique<cuntt::detail::XdRealFftPlan>(d.extents,d.batch,
            d.direction == CUBUTTERFLY_DIRECTION_INVERSE,d.normalize_inverse,
            d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64,true,1.0,true,order,
            choice.fused,choice.ept,choice.paired,axis_core,choice.pair_core,
            choice.register_lanes,choice.pair_layout,batch_tile,choice.block_cap,choice.exchange,
            resources);
        p->workspace_bytes = p->xd_fft->workspace_size();
        auto mapping = nlohmann::json::parse(p->xd_fft->mapping_json());
        if (choice.local && !xd_mapping_has_axis_core(mapping, "thread-shared-axis")) {
            omitted.push_back({{"lowering",family}, {"axis_order",order},
                               {"reason","local family produced no actual local axis"}});
            continue;
        }
        if (choice.paired && mapping.value("pair_groups", 0) == 0) {
            omitted.push_back({{"lowering",family}, {"axis_order",order},
                              {"reason","no compiled contiguous pair fits this axis order and precision"}});
            continue;
        }
        if (!choice.paired && choice.axis_core == cuntt::detail::XdAxisCore::CooperativeRegister &&
            std::none_of(mapping.at("axes").begin(), mapping.at("axes").end(), [](const auto& axis) {
                return axis.value("core", "") == "register-cooperative-axis";
            })) {
            omitted.push_back({{"lowering",family}, {"axis_order",order},
                              {"reason","no supported non-contiguous register axis"}});
            continue;
        }
        if (!replay.is_null()) require(mapping == replay.at("physical"), "packed FFT replay mismatch");
        candidates.push_back({family, std::move(p), std::move(mapping)});
       } catch (const std::exception& error) {
         if (!requested.empty()) throw;
         omitted.push_back({{"lowering",family}, {"reason",error.what()}});
       }
       }
      }
    }
    }
    if (requested == "axis-tiled-mixed") {
        require(!replay.is_null(), "mixed-axis FFT requires a complete mapping replay");
        require(packed && d.op == CUBUTTERFLY_OPERATOR_FFT &&
                d.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE &&
                (d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64),
                "mixed-axis FFT requires packed out-of-place complex storage");
        auto physical = replay.at("physical");
        require(xd_axis_refinement_seed(physical), "invalid mixed-axis FFT mode");
        auto resources = xd_axis_resources(physical);
        require(!resources.empty(), "mixed-axis FFT requires per-axis resources");
        // Reconstruct semantics from the descriptor, then check exact mapping.
        // A forged physical block cannot override the validated contract.
        physical["extents"] = d.extents;
        physical["batch"] = d.batch;
        physical["inverse"] = d.direction == CUBUTTERFLY_DIRECTION_INVERSE;
        physical["normalize_inverse"] = d.normalize_inverse;
        physical["fp64"] = d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64;
        physical["complex_only"] = true;
        physical["output_scale"] = 1.0;
        auto p = shell();
        p->xd_fft = xd_plan_with_resources(physical,resources);
        p->workspace_bytes = p->xd_fft->workspace_size();
        auto mapping = nlohmann::json::parse(p->xd_fft->mapping_json());
        require(mapping == replay.at("physical"), "mixed-axis FFT replay mismatch");
        candidates.push_back({requested,std::move(p),std::move(mapping)});
    }
    if (candidates.empty()) {
        log_event(handle, CUBUTTERFLY_LOG_WARNING, "xd-selection",
                  std::string("no executable xD candidate; rejected=") +
                  omitted.dump());
        throw unsupported_error(
            std::string("requested xD lowering is unavailable; rejected=") +
            omitted.dump());
    }
    std::size_t best = 0;
    auto measurements = nlohmann::json::array();
    std::string tuning_protocol = "not-measured";
    std::string graph_fallback_reason;
    nlohmann::json tuning_diagnostics{{"candidate_count", candidates.size()},
                                      {"rejected_count", omitted.size()},
                                      {"requested_lowering", requested.empty() ? "auto" : requested}};
    if (requested.empty() && candidates.size() > 1) {
        const double promotion_factor = xd_promotion_factor();
        std::size_t io_bytes = 0, work_bytes = 0;
        for (const auto& candidate : candidates) {
            io_bytes = std::max({io_bytes, candidate.plan->input_bytes, candidate.plan->output_bytes});
            work_bytes = std::max(work_bytes, candidate.plan->workspace_bytes);
        }
        const auto tuning_status = xd_tuning_status(io_bytes, work_bytes);
        tuning_diagnostics["io_bytes"] = io_bytes;
        tuning_diagnostics["workspace_bytes"] = work_bytes;
        tuning_diagnostics["tuning"] = {
            {"attempted", tuning_status.fits},
            {"budget_bytes", tuning_status.budget_bytes},
            {"free_bytes", tuning_status.free_bytes},
            {"total_bytes", tuning_status.total_bytes},
            {"reason", tuning_status.reason}};
      if (!tuning_status.fits) {
        omitted.push_back({{"lowering","empirical-promotion"},
                           {"reason", "tuning memory budget; retain composite: " + tuning_status.reason},
                           {"io_bytes",io_bytes}, {"workspace_bytes",work_bytes}});
      } else {
        XdTuningStream tuning_stream(handle);
        DeviceAllocation input(io_bytes), output(io_bytes), workspace(work_bytes);
        const auto stream = tuning_stream.get();
        require(cudaMemsetAsync(input.pointer,0,io_bytes,stream) == cudaSuccess, "initialize xD tuning input");
        auto execute_candidate = [&](std::size_t index) {
            auto* p = candidates[index].plan.get();
            void* dst = d.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE ? input.pointer : output.pointer;
            const auto status = cubutterflyExecute(handle,p,input.pointer,dst);
            if (status != CUBUTTERFLY_STATUS_SUCCESS) throw std::runtime_error("xD tuning execution failed");
        };
        std::vector<std::unique_ptr<XdCudaGraph>> graphs(candidates.size());
        bool graph_timing = true;
        try {
            // Probe every candidate before timing any candidate.  A partial
            // graph set is never mixed with direct launches.
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                auto* p = candidates[i].plan.get();
                if (p->workspace_bytes)
                    require(cubutterflyPlanSetWorkspace(p,workspace.pointer,p->workspace_bytes) == CUBUTTERFLY_STATUS_SUCCESS,
                            "bind xD tuning workspace");
                graphs[i] = capture_xd_cuda_graph(stream, [&] { execute_candidate(i); });
            }
        } catch (const std::exception& error) {
            graph_timing = false;
            graph_fallback_reason = error.what();
            graphs.clear();
            (void)cudaStreamSynchronize(stream);
        } catch (...) {
            graph_timing = false;
            graph_fallback_reason = "unknown CUDA Graph capture failure";
            graphs.clear();
            (void)cudaStreamSynchronize(stream);
        }
        tuning_protocol = graph_timing ? "cuda-graph-steady-state" : "cuda-event-launch-fallback";
        double best_ms = std::numeric_limits<double>::infinity();
        std::size_t axis_seed = candidates.size();
        double axis_seed_ms = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            auto* p = candidates[i].plan.get();
            if (p->workspace_bytes)
                require(cubutterflyPlanSetWorkspace(p,workspace.pointer,p->workspace_bytes) == CUBUTTERFLY_STATUS_SUCCESS,
                        "bind xD tuning workspace");
            std::vector<double> times;
            try {
                for (unsigned trial = 0; trial < 3; ++trial)
                    times.push_back(graph_timing
                        ? measure_xd_cuda_graph(stream, graphs[i]->executable())
                        : measure_cuda_candidate(stream, [&] { execute_candidate(i); }));
            } catch (const std::bad_alloc&) { throw; }
              catch (const std::exception& error) {
                (void)cudaGetLastError();
                if (cudaStreamSynchronize(stream) != cudaSuccess) throw;
                omitted.push_back({{"lowering", candidates[i].name}, {"reason", error.what()},
                                   {"phase", "measurement"}});
                continue;
            }
            std::sort(times.begin(),times.end());
            nlohmann::json measurement{{"lowering", candidates[i].name},
                                       {"median_ms", times[1]}, {"trials_ms", times},
                                       {"timing_protocol", tuning_protocol}};
            if (graph_timing) {
                measurement["warmup_plan_executions"] = 1;
                measurement["warmup_graph_launches"] = kXdGraphWarmupLaunches;
                measurement["warmup_total_plan_executions"] =
                    1 + kXdGraphWarmupLaunches * kXdGraphPlanExecutions;
                measurement["captured_plan_executions"] = kXdGraphPlanExecutions;
                measurement["timed_graph_launches"] = kXdGraphTimedLaunches;
                measurement["timed_plan_executions"] = kXdGraphPlanExecutions * kXdGraphTimedLaunches;
            } else {
                measurement["warmup_launches"] = 2;
                measurement["timed_launches"] = 5;
                measurement["graph_fallback_reason"] = graph_fallback_reason;
            }
            if (candidates[i].physical.contains("axis_order"))
                measurement["axis_order"] = candidates[i].physical.at("axis_order");
            for (const char* field : {"register_lanes", "pair_layout", "batch_tile", "axis_ffts_per_block_cap", "axis_exchange_policy"})
                if (candidates[i].physical.contains(field))
                    measurement[field] = candidates[i].physical.at(field);
            measurements.push_back(std::move(measurement));
            // The configurable promotion factor keeps a noisy tie on the
            // existing lowering while allowing installation-time calibration
            // to admit a genuinely faster complete plan.
            if (times[1] < best_ms * promotion_factor) { best_ms = times[1]; best = i; }
            if (xd_axis_refinement_seed(candidates[i].physical) && times[1] < axis_seed_ms) {
                axis_seed = i;
                axis_seed_ms = times[1];
            }
            p->workspace = nullptr;
            for (auto& axis : p->butterfly_axes) axis->set_workspace(nullptr,0);
            for (auto& axis : p->ntt_axes) axis->set_workspace(nullptr,0);
            if (p->butterfly) p->butterfly->set_workspace(nullptr,0);
            if (p->ntt) p->ntt->set_workspace(nullptr,0);
            if (p->xd_fft) p->xd_fft->set_workspace(nullptr,0);
        }
        require(std::isfinite(best_ms), "no xD candidate survived measurement");
        if (axis_seed < candidates.size()) {
            auto refined = refine_xd_axis_resources(candidates[axis_seed].physical,axis_seed_ms,
                input.pointer,output.pointer,workspace.pointer,work_bytes,stream,graph_timing,measurements,omitted);
            if (refined.plan && refined.milliseconds < best_ms * promotion_factor) {
                auto p = shell();
                p->workspace_bytes = refined.plan->workspace_size();
                p->xd_fft = std::move(refined.plan);
                best = candidates.size();
                candidates.push_back({"axis-tiled-mixed",std::move(p),std::move(refined.physical)});
            }
        }
        for (const auto& candidate : candidates)
            restore_xd_plan_streams(candidate.plan.get(), tuning_stream.original());
      }
    } else {
        tuning_diagnostics["tuning"] = {
            {"attempted", false},
            {"reason", candidates.size() <= 1 ? "single-executable-candidate" : "explicit-or-replayed-lowering"}};
    }
    tuning_diagnostics["candidate_count"] = candidates.size();
    tuning_diagnostics["measured_count"] = measurements.size();
    tuning_diagnostics["rejected_count"] = omitted.size();
    auto result = shell();
    result->xd_selected = std::move(candidates[best].plan);
    result->physical_extents = result->xd_selected->physical_extents;
    result->input_bytes = result->xd_selected->input_bytes;
    result->output_bytes = result->xd_selected->output_bytes;
    result->workspace_bytes = result->xd_selected->workspace_bytes;
    result->algorithm = "xd[" + candidates[best].name + "]";
    result->selection_source = cached ? CUBUTTERFLY_SELECTION_USER_CACHE :
        (!replay.is_null() || !forced_lowering.empty()) ? CUBUTTERFLY_SELECTION_EXPLICIT :
        measurements.empty() ? CUBUTTERFLY_SELECTION_STATIC_MODEL : CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
    const double promotion_margin = 1.0 - xd_promotion_factor();
    result->reason = (measurements.empty() ? "replayed or sole executable xD lowering" :
        "bounded complete-plan measurement; includes boundaries; promotion margin=" +
        std::to_string(100.0 * promotion_margin) + "%; timing=" + tuning_protocol) +
        ((!measurements.empty() && !graph_fallback_reason.empty())
             ? std::string("; graph_fallback_reason=") + graph_fallback_reason : std::string{}) +
        std::string("; diagnostics=") + tuning_diagnostics.dump() +
        "; candidates=" + measurements.dump() + "; rejected=" + omitted.dump();
    nlohmann::json mapping{{"schema_version",1}, {"kind","xd"}, {"contract",xd_contract(d)}, {"target",xd_target()},
                          {"lowering", candidates[best].name}, {"physical",candidates[best].physical}};
    result->xd_mapping = mapping.dump();
    if (replay.is_null() && forced_lowering.empty() &&
        (!measurements.empty() || candidates.size() == 1))
        store_cache(handle->cache_path,key,result->xd_mapping);
    return result;
}

bool nonnatural_order(const cubutterflyDescriptor& d) {
    const auto reversed = [](cubutterflyLogicalOrder_t order) {
        return order != CUBUTTERFLY_ORDER_NATURAL;
    };
    return std::any_of(d.input_orders.begin(), d.input_orders.end(), reversed) ||
           std::any_of(d.output_orders.begin(), d.output_orders.end(), reversed);
}

bool appt_order(const cubutterflyDescriptor& d) {
    return d.input_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC ||
           d.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC;
}

bool native_ntt_reversed_output(const cubutterflyDescriptor& d) {
    return d.op == CUBUTTERFLY_OPERATOR_NTT && d.rank == 1 &&
        d.length_mode == CUBUTTERFLY_LENGTH_STANDARD && is_power_of_two(d.extents[0]) &&
        d.input_orders[0] == CUBUTTERFLY_ORDER_NATURAL &&
        d.output_orders[0] == CUBUTTERFLY_ORDER_BIT_REVERSED && !appt_order(d);
}

bool requires_semantic_boundary(const cubutterflyDescriptor& d) {
    if (nonnatural_order(d) && !native_ntt_reversed_output(d)) return true;
    const auto in = cubutterfly::detail::semantic_input_extents(d);
    const auto out = cubutterfly::detail::semantic_output_extents(d);
    const bool packed = packed_side(d, true, in) && packed_side(d, false, out);
    if ((d.storage == CUBUTTERFLY_DATA_FP32 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32) &&
        d.compute == CUBUTTERFLY_COMPUTE_FP64 && !packed) return true;
    if (d.fft_type != CUBUTTERFLY_FFT_C2C)
        return !packed || d.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE;
    if (d.op == CUBUTTERFLY_OPERATOR_NTT && d.rank == 1 && !appt_order(d))
        return !packed || d.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE;
    const bool arbitrary = std::any_of(d.extents.begin(), d.extents.end(),
                                      [](std::size_t n) { return !is_power_of_two(n); });
    if (arbitrary && d.rank > 2 && !packed) return true;
    if (d.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE &&
        (arbitrary || d.length_mode != CUBUTTERFLY_LENGTH_STANDARD)) return true;
    if (d.rank == 1 && d.length_mode == CUBUTTERFLY_LENGTH_STANDARD && !arbitrary) {
        const auto span = (d.extents[0] - 1) * d.input_strides[0] + 1;
        return d.input_strides != d.output_strides ||
            d.input_batch_stride != d.output_batch_stride ||
            (d.batch > 1 && d.input_batch_stride < span);
    }
    return false;
}

nlohmann::json static_capabilities(const cubutterflyDescriptor& d) {
    auto gaps = nlohmann::json::array();
    auto gap = [&](const char* code, const char* reason) {
        gaps.push_back({{"code", code}, {"reason", reason}});
    };
    if (d.output_scale != 1.0 || d.output_rounding != CUBUTTERFLY_ROUNDING_STAGE_STORAGE) {
        if (!(d.op == CUBUTTERFLY_OPERATOR_FWHT && d.rank == 1 &&
              d.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
              d.storage == CUBUTTERFLY_DATA_FP16 && d.compute == CUBUTTERFLY_COMPUTE_FP32 &&
              d.output_rounding == CUBUTTERFLY_ROUNDING_FINAL_STORAGE))
            gap("output-epilogue-lowering", "output epilogue requires rank-one standard FP16 FWHT, FP32 compute and final-storage rounding");
        if (!std::isfinite(static_cast<float>(d.output_scale)))
            gap("output-scale-lowering", "output scale must be representable as finite FP32");
    }
    if (d.rank > CUBUTTERFLY_MAX_EXECUTION_RANK)
        gap("rank-lowering", "generic execution currently supports rank one through eight");
    if (d.execution_mode == CUBUTTERFLY_EXECUTION_XD && d.rank == 1)
        gap("execution-policy", "forced xD execution requires at least two axes");
    if (d.fft_type == CUBUTTERFLY_FFT_C2C && d.fft_output_scale != 1.0)
        gap("fft-scale-lowering", "additional FFT output scaling currently requires R2C/C2R");
    const bool real_storage = d.storage >= CUBUTTERFLY_DATA_FP16 && d.storage <= CUBUTTERFLY_DATA_FP64;
    const bool low = d.storage == CUBUTTERFLY_DATA_FP16 || d.storage == CUBUTTERFLY_DATA_BF16 ||
                     d.storage == CUBUTTERFLY_DATA_COMPLEX_FP16 || d.storage == CUBUTTERFLY_DATA_COMPLEX_BF16;
    const auto expected_compute = (d.storage == CUBUTTERFLY_DATA_FP64 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64)
        ? CUBUTTERFLY_COMPUTE_FP64 : d.storage == CUBUTTERFLY_DATA_UINT32 ? CUBUTTERFLY_COMPUTE_UINT32 :
          d.storage == CUBUTTERFLY_DATA_UINT64 ? CUBUTTERFLY_COMPUTE_UINT64 : CUBUTTERFLY_COMPUTE_FP32;
    const bool fp32_fp64 = (d.storage == CUBUTTERFLY_DATA_FP32 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP32) &&
        d.compute == CUBUTTERFLY_COMPUTE_FP64;
    if (d.compute != expected_compute && !(low && d.compute == CUBUTTERFLY_COMPUTE_NATIVE) && !fp32_fp64)
        gap("compute-lowering", "this storage/compute combination has no arithmetic kernel");
    if ((d.op == CUBUTTERFLY_OPERATOR_FWHT || d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) && !real_storage)
        gap("storage-lowering", "FWHT and structured kernels currently use real floating storage");
    if ((d.op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || d.op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA) &&
        d.storage != CUBUTTERFLY_DATA_UINT32 && d.storage != CUBUTTERFLY_DATA_UINT64)
        gap("storage-lowering", "zeta kernels require uint32 or uint64 storage");
    if (d.fft_type != CUBUTTERFLY_FFT_C2C) {
        if (low && std::any_of(d.extents.begin(), d.extents.end(),
                              [](std::size_t n) { return n < 2 || !is_power_of_two(n); }))
            gap("real-precision-lowering", "low-precision real FFT requires power-of-two axes of length at least two");
        if (d.length_mode != CUBUTTERFLY_LENGTH_STANDARD)
            gap("real-embedding-lowering", "compact real FFT embedding is not implemented");
    } else {
        const bool arbitrary = std::any_of(d.extents.begin(), d.extents.end(),
                                          [](std::size_t n) { return !is_power_of_two(n); });
        if (arbitrary && d.length_mode == CUBUTTERFLY_LENGTH_STANDARD) {
            if (d.op != CUBUTTERFLY_OPERATOR_FFT && d.op != CUBUTTERFLY_OPERATOR_NTT)
                gap("length-lowering", "this operator requires power-of-two extents or explicit embedding");
            if (d.op == CUBUTTERFLY_OPERATOR_FFT && d.storage != CUBUTTERFLY_DATA_COMPLEX_FP32 &&
                d.storage != CUBUTTERFLY_DATA_COMPLEX_FP64)
                gap("arbitrary-precision-lowering", "Bluestein FFT currently has FP32/FP64 kernels");
            if (d.op == CUBUTTERFLY_OPERATOR_FFT && d.rank > 2)
                gap("rank-bluestein-lowering", "rank-three and higher Bluestein composition is not implemented");
            if (d.op == CUBUTTERFLY_OPERATOR_NTT && d.storage != CUBUTTERFLY_DATA_UINT64)
                gap("arbitrary-word-lowering", "Bluestein NTT currently has uint64 kernels");
        }
        if (std::any_of(d.extents.begin(), d.extents.end(), [](std::size_t n) { return n == 1; }))
            gap("unit-axis-lowering", "identity axes require a dedicated lowering");
    }
    const auto in = cubutterfly::detail::semantic_input_extents(d);
    const auto out = cubutterfly::detail::semantic_output_extents(d);
    if (!cubutterfly::detail::layout_is_nonoverlapping(in, d.input_strides, d.batch, d.input_batch_stride) ||
        !cubutterfly::detail::layout_is_nonoverlapping(out, d.output_strides, d.batch, d.output_batch_stride))
        gap("layout-proof-unavailable", "non-overlap of this large sparse layout was not established by bounded validation");
    if (appt_order(d) && !(d.rank == 1 && d.extents[0] == (1ULL << 20) &&
        d.length_mode == CUBUTTERFLY_LENGTH_STANDARD && d.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE &&
        packed_side(d, true, in) && packed_side(d, false, out)))
        gap("appt-layout-lowering", "legacy APPT layout requires packed out-of-place rank-one N=2^20 NTT");
    return {{"schema", "cubutterfly-capabilities-v1"}, {"semantic_contract", cubutterfly::detail::semantic_json(d)},
            {"status", gaps.empty() ? "eligible" : "missing-lowering"}, {"gaps", gaps},
            {"validation_scope", "CPU structural semantics and static lowering constraints; target resources and arithmetic roots checked at plan creation"},
            {"boundary_adapter", requires_semantic_boundary(d)}, {"performance_verified", false}};
}

void check_child(cubutterflyStatus_t status) {
    if (status == CUBUTTERFLY_STATUS_SUCCESS) return;
    const auto reason = last_error.empty() ? std::string(cubutterflyGetStatusString(status)) : last_error;
    if (status == CUBUTTERFLY_STATUS_NOT_SUPPORTED) throw unsupported_error(reason);
    if (status == CUBUTTERFLY_STATUS_INVALID_VALUE) throw std::invalid_argument(reason);
    if (status == CUBUTTERFLY_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(reason);
}

std::size_t boundary_element_bytes(const cubutterflyDescriptor& d, bool input) {
    auto bytes = element_bytes(d.storage);
    if ((input && d.fft_type == CUBUTTERFLY_FFT_C2R) ||
        (!input && d.fft_type == CUBUTTERFLY_FFT_R2C)) bytes *= 2;
    return bytes;
}

std::unique_ptr<cubutterflyPlan> make_boundary_plan(cubutterflyHandle_t handle,
                                                  const cubutterflyDescriptor& d) {
    auto child = d;
    child.input_orders.assign(d.rank, CUBUTTERFLY_ORDER_NATURAL);
    child.output_orders.assign(d.rank, CUBUTTERFLY_ORDER_NATURAL);
    child.input_ntt_layout = child.output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    child.input_strides.clear(); child.output_strides.clear();
    child.input_batch_stride = child.output_batch_stride = 0;
    child.placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    nlohmann::json replay;
    if (d.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT && !d.explicit_algorithm.empty() &&
        d.explicit_algorithm.front() == '{') {
        replay = nlohmann::json::parse(d.explicit_algorithm);
        require(replay.value("kind", "") == "semantic-boundary" && replay.value("schema_version", 0) == 1,
                "ordered/layout-adapted replay requires a complete semantic-boundary mapping");
        require(replay.at("contract") == xd_contract(d) && replay.at("target") == xd_target(),
                "semantic boundary replay contract or target mismatch");
        child.explicit_algorithm = replay.at("child").dump();
    }
    auto result = std::make_unique<cubutterflyPlan>();
    cubutterflyPlan_t raw = nullptr;
    check_child(cubutterflyCreatePlan(handle, &child, &raw));
    result->boundary_child.reset(raw);
    result->descriptor = d;
    result->physical_extents = raw->physical_extents;
    const auto in = cubutterfly::detail::semantic_input_extents(d);
    const auto out = cubutterfly::detail::semantic_output_extents(d);
    result->input_bytes = checked_product(strided_elements(d, true, in), boundary_element_bytes(d, true));
    result->output_bytes = checked_product(strided_elements(d, false, out), boundary_element_bytes(d, false));
    result->boundary_input_bytes = raw->input_bytes;
    result->boundary_output_bytes = raw->output_bytes;
    const auto first = align_16(result->boundary_input_bytes);
    const auto second = align_16(result->boundary_output_bytes);
    require(first <= std::numeric_limits<std::size_t>::max() - second &&
            first + second <= std::numeric_limits<std::size_t>::max() - raw->workspace_bytes,
            "semantic boundary workspace overflows size_t");
    result->workspace_bytes = first + second + raw->workspace_bytes;
    result->selection_source = raw->selection_source;
    result->algorithm = "semantic-boundary[" + raw->algorithm + "]";
    result->reason = "per-axis order and independent layout adapter; child selection: " + raw->reason +
        "; fixed boundary cost is included in execution, not in child-only selection timings";
    nlohmann::json mapping{{"schema_version", 1}, {"kind", "semantic-boundary"},
        {"contract", xd_contract(d)}, {"target", xd_target()},
        {"boundary", "gather-natural-child-scatter"}, {"child", nlohmann::json::parse(get_mapping(raw))},
        {"workspace_bytes", result->workspace_bytes}};
    if (!replay.is_null()) require(replay == mapping, "semantic boundary mapping changed during replay");
    result->xd_mapping = mapping.dump();
    return result;
}

std::unique_ptr<cubutterflyPlan> make_numeric_plan(cubutterflyHandle_t handle,
                                                const cubutterflyDescriptor& d,
                                                bool full_spectrum_real) {
    auto child = d;
    child.input_strides.clear(); child.output_strides.clear();
    child.input_batch_stride = child.output_batch_stride = 0;
    child.placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    if (full_spectrum_real) {
        child.storage = d.storage == CUBUTTERFLY_DATA_FP16
            ? CUBUTTERFLY_DATA_COMPLEX_FP16 : CUBUTTERFLY_DATA_COMPLEX_BF16;
        child.fft_type = CUBUTTERFLY_FFT_C2C;
        child.fft_output_scale = 1.0;
    } else {
        child.storage = d.storage == CUBUTTERFLY_DATA_FP32
            ? CUBUTTERFLY_DATA_FP64 : CUBUTTERFLY_DATA_COMPLEX_FP64;
    }
    const char* adapter = full_spectrum_real ? "same-precision-full-spectrum-real" : "fp32-storage-fp64-compute";
    nlohmann::json replay;
    if (d.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT && !d.explicit_algorithm.empty() &&
        d.explicit_algorithm.front() == '{') {
        replay = nlohmann::json::parse(d.explicit_algorithm);
        require(replay.value("kind", "") == "numeric-adapter" &&
                replay.value("schema_version", 0) == 1 && replay.at("adapter") == adapter,
                "numeric replay requires a complete matching adapter mapping");
        require(replay.at("contract") == xd_contract(d) && replay.at("target") == xd_target(),
                "numeric adapter replay contract or target mismatch");
        child.explicit_algorithm = replay.at("child").dump();
    }
    auto result = std::make_unique<cubutterflyPlan>();
    cubutterflyPlan_t raw = nullptr;
    check_child(cubutterflyCreatePlan(handle, &child, &raw));
    result->numeric_child.reset(raw);
    result->full_spectrum_real = full_spectrum_real;
    result->descriptor = d;
    result->physical_extents = raw->physical_extents;
    const auto in = cubutterfly::detail::semantic_input_extents(d);
    const auto out = cubutterfly::detail::semantic_output_extents(d);
    result->input_bytes = checked_product(strided_elements(d, true, in), boundary_element_bytes(d, true));
    result->output_bytes = checked_product(strided_elements(d, false, out), boundary_element_bytes(d, false));
    result->boundary_input_bytes = raw->input_bytes;
    result->boundary_output_bytes = raw->output_bytes;
    const auto first = align_16(raw->input_bytes), second = align_16(raw->output_bytes);
    require(first <= std::numeric_limits<std::size_t>::max() - second &&
            first + second <= std::numeric_limits<std::size_t>::max() - raw->workspace_bytes,
            "numeric adapter workspace overflows size_t");
    result->workspace_bytes = first + second + raw->workspace_bytes;
    result->selection_source = raw->selection_source;
    result->algorithm = std::string(adapter) + "[" + raw->algorithm + "]";
    result->reason = std::string(adapter) + "; boundary kernels included in execution; child selection: " + raw->reason;
    nlohmann::json mapping{{"schema_version", 1}, {"kind", "numeric-adapter"},
        {"contract", xd_contract(d)}, {"target", xd_target()}, {"adapter", adapter},
        {"path_class", "adapted"}, {"child", nlohmann::json::parse(get_mapping(raw))},
        {"workspace_bytes", result->workspace_bytes},
        {"timing_scope", "includes input and output conversion kernels"}};
    if (!replay.is_null()) require(replay == mapping, "numeric adapter mapping changed during replay");
    result->xd_mapping = mapping.dump();
    return result;
}

// The same separable axis composition is valid for arbitrary axis lengths.
// Each child owns its exact-length algorithm (including Bluestein), and the
// intermediate tensor retains the requested logical shape, never zero padding
// the user's transform into a different mathematical operation.
std::unique_ptr<cubutterflyPlan> make_arbitrary_nd_plan(cubutterflyHandle_t handle,
                                                     const cubutterflyDescriptor& d) {
    auto result = std::make_unique<cubutterflyPlan>();
    result->descriptor = d;
    result->physical_extents = d.extents;
    result->element_bytes = element_bytes(d.storage);
    const auto layout = packed_nd_layout(d.extents, d.batch);
    result->input_bytes = result->output_bytes = result->packed_bytes =
        checked_product(checked_product(layout.points, d.batch), result->element_bytes);
    nlohmann::json replay;
    if (d.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT && !d.explicit_algorithm.empty() &&
        d.explicit_algorithm.front() == '{') {
        replay = nlohmann::json::parse(d.explicit_algorithm);
        require(replay.value("kind", "") == "arbitrary-axis-composition" &&
                replay.value("schema_version", 0) == 1 && replay.at("children").size() == d.rank,
                "rank-N arbitrary replay requires a complete axis composition");
        require(replay.at("contract") == xd_contract(d) && replay.at("target") == xd_target(),
                "arbitrary axis replay contract or target mismatch");
    }
    auto mappings = nlohmann::json::array();
    std::size_t max_work = 0;
    for (std::uint32_t axis = d.rank; axis-- > 0;) {
        auto child = d;
        child.rank = 1;
        child.extents = {d.extents[axis]};
        child.batch = checked_product(d.batch, layout.points / d.extents[axis]);
        child.input_orders = child.output_orders = {CUBUTTERFLY_ORDER_NATURAL};
        child.input_strides.clear(); child.output_strides.clear();
        child.input_batch_stride = child.output_batch_stride = 0;
        child.placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
        child.execution_mode = CUBUTTERFLY_EXECUTION_AUTO;
        child.stage_matrices = {d.stage_matrices[axis]};
        if (!replay.is_null()) child.explicit_algorithm = replay.at("children").at(mappings.size()).dump();
        cubutterflyPlan_t raw = nullptr;
        check_child(cubutterflyCreatePlan(handle, &child, &raw));
        result->arbitrary_children.emplace_back(raw);
        mappings.push_back(nlohmann::json::parse(get_mapping(raw)));
        max_work = std::max(max_work, raw->workspace_bytes);
        result->selection_source = combined_source(result->selection_source, raw->selection_source);
    }
    const auto buffers = checked_product(align_16(result->packed_bytes), 2);
    require(buffers <= std::numeric_limits<std::size_t>::max() - max_work,
            "arbitrary axis workspace overflows size_t");
    result->workspace_bytes = buffers + max_work;
    result->algorithm = "rankN-arbitrary-axis-composition";
    result->reason = "exact-length per-axis child selection; all copies and permutations included in execution";
    nlohmann::json mapping{{"schema_version", 1}, {"kind", "arbitrary-axis-composition"},
        {"contract", xd_contract(d)}, {"target", xd_target()}, {"children", mappings},
        {"axis_order", "last-to-first"}, {"workspace_bytes", result->workspace_bytes}};
    if (!replay.is_null()) require(replay == mapping, "arbitrary axis mapping changed during replay");
    result->xd_mapping = mapping.dump();
    return result;
}

}  // namespace

const char* cubutterflyGetLastErrorString(void) { return last_error.c_str(); }

const char* cubutterflyGetStatusString(cubutterflyStatus_t status) {
    switch (status) {
        case CUBUTTERFLY_STATUS_SUCCESS: return "success";
        case CUBUTTERFLY_STATUS_INVALID_VALUE: return "invalid value";
        case CUBUTTERFLY_STATUS_INVALID_TYPE: return "invalid type";
        case CUBUTTERFLY_STATUS_NOT_SUPPORTED: return "not supported";
        case CUBUTTERFLY_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBUTTERFLY_STATUS_CUDA_ERROR: return "CUDA error";
        case CUBUTTERFLY_STATUS_BACKEND_ERROR: return "backend error";
        case CUBUTTERFLY_STATUS_INTERNAL_ERROR: return "internal error";
    }
    return "unknown status";
}

cubutterflyStatus_t cubutterflyGetVersion(int* major, int* minor, int* patch) {
    return protect([&] {
        require(major != nullptr && minor != nullptr && patch != nullptr, "version pointers must be non-null");
        *major = CUBUTTERFLY_VERSION_MAJOR;
        *minor = CUBUTTERFLY_VERSION_MINOR;
        *patch = CUBUTTERFLY_VERSION_PATCH;
    });
}

cubutterflyStatus_t cubutterflyCreate(cubutterflyHandle_t* handle) {
    return protect([&] { require(handle != nullptr, "handle pointer must be non-null"); *handle = new cubutterflyHandle; });
}

cubutterflyStatus_t cubutterflyDestroy(cubutterflyHandle_t handle) { delete handle; return CUBUTTERFLY_STATUS_SUCCESS; }

cubutterflyStatus_t cubutterflySetStream(cubutterflyHandle_t handle, cudaStream_t stream) {
    return protect([&] { require(handle != nullptr, "handle must be non-null"); handle->stream = stream; });
}

cubutterflyStatus_t cubutterflyGetStream(cubutterflyHandle_t handle, cudaStream_t* stream) {
    return protect([&] { require(handle != nullptr && stream != nullptr, "handle and stream pointer must be non-null"); *stream = handle->stream; });
}

cubutterflyStatus_t cubutterflySetLogCallback(cubutterflyHandle_t handle, cubutterflyLogCallback_t callback, void* user_data) {
    return protect([&] { require(handle != nullptr, "handle must be non-null"); handle->logger = callback; handle->logger_data = user_data; });
}

cubutterflyStatus_t cubutterflySetCachePath(cubutterflyHandle_t handle, const char* path) {
    return protect([&] { require(handle != nullptr, "handle must be non-null"); handle->cache_path = path == nullptr ? "" : path; });
}

cubutterflyStatus_t cubutterflyCreateDescriptor(cubutterflyDescriptor_t* descriptor) {
    return protect([&] { require(descriptor != nullptr, "descriptor pointer must be non-null"); *descriptor = new cubutterflyDescriptor; });
}

cubutterflyStatus_t cubutterflyDestroyDescriptor(cubutterflyDescriptor_t descriptor) { delete descriptor; return CUBUTTERFLY_STATUS_SUCCESS; }

cubutterflyStatus_t cubutterflyDescriptorValidate(cubutterflyDescriptor_t descriptor) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        (void)cubutterfly::detail::normalize_descriptor(*descriptor);
    });
}

cubutterflyStatus_t cubutterflyDescriptorGetJson(cubutterflyDescriptor_t descriptor, char* json, size_t* bytes) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        copy_text(cubutterfly::detail::descriptor_json(*descriptor).dump(), json, bytes);
    });
}

cubutterflyStatus_t cubutterflyDescriptorSetJson(cubutterflyDescriptor_t descriptor, const char* json) {
    return protect([&] {
        require(descriptor != nullptr && json != nullptr, "descriptor and JSON must be non-null");
        auto replacement = cubutterfly::detail::descriptor_from_json(nlohmann::json::parse(json));
        *descriptor = std::move(replacement);
    });
}

cubutterflyStatus_t cubutterflyDescriptorGetCapabilities(cubutterflyDescriptor_t descriptor, char* json, size_t* bytes) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        copy_text(static_capabilities(cubutterfly::detail::normalize_descriptor(*descriptor)).dump(), json, bytes);
    });
}

cubutterflyStatus_t cubutterflySetOperator(cubutterflyDescriptor_t descriptor, cubutterflyOperator_t op) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->op = op; });
}

cubutterflyStatus_t cubutterflySetDataType(cubutterflyDescriptor_t descriptor, cubutterflyDataType_t storage, cubutterflyComputeType_t compute) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->storage = storage; descriptor->compute = compute; });
}

cubutterflyStatus_t cubutterflySetShape(cubutterflyDescriptor_t descriptor, uint32_t rank, const size_t* extents, size_t batch) {
    return protect([&] {
        require(descriptor != nullptr && extents != nullptr, "descriptor and extents must be non-null");
        require(rank >= 1, "rank must be positive");
        require(batch != 0, "batch must be positive");
        std::vector<std::size_t> shape(extents, extents + rank);
        require(std::all_of(shape.begin(), shape.end(), [](std::size_t n) { return n != 0; }),
                "extents must be positive");
        std::size_t count = batch;
        for (auto extent : shape) count = checked_product(count, extent);
        descriptor->rank = rank;
        descriptor->extents = std::move(shape);
        descriptor->batch = batch;
        // Preserve supplied fields. If a rank change invalidates an existing
        // per-axis setting, final validation reports the mismatch explicitly.
        descriptor->stage_matrices.resize(std::max<std::size_t>(rank, descriptor->stage_matrices.size()));
    });
}

cubutterflyStatus_t cubutterflySetExecutionMode(cubutterflyDescriptor_t descriptor,
                                                cubutterflyExecutionMode_t mode) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        require(mode == CUBUTTERFLY_EXECUTION_AUTO ||
                    mode == CUBUTTERFLY_EXECUTION_XD ||
                    mode == CUBUTTERFLY_EXECUTION_COMPOSITE,
                "unknown execution mode");
        descriptor->execution_mode = mode;
    });
}

cubutterflyStatus_t cubutterflySetFftTransformType(cubutterflyDescriptor_t descriptor,
                                                   cubutterflyFftTransformType_t type) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        require(type >= CUBUTTERFLY_FFT_C2C && type <= CUBUTTERFLY_FFT_C2R,
                "unknown FFT transform type");
        descriptor->fft_type = type;
    });
}

cubutterflyStatus_t cubutterflySetOutputScale(cubutterflyDescriptor_t descriptor, double scale) {
    return protect([&] {
        require(descriptor != nullptr && std::isfinite(scale), "output scale must be finite");
        descriptor->output_scale = scale;
    });
}

cubutterflyStatus_t cubutterflySetOutputRounding(cubutterflyDescriptor_t descriptor, cubutterflyOutputRounding_t rounding) {
    return protect([&] {
        require(descriptor != nullptr && (rounding == CUBUTTERFLY_ROUNDING_STAGE_STORAGE ||
                    rounding == CUBUTTERFLY_ROUNDING_FINAL_STORAGE), "invalid output rounding");
        descriptor->output_rounding = rounding;
    });
}

cubutterflyStatus_t cubutterflySetFftOutputScale(cubutterflyDescriptor_t descriptor, double scale) {
    return protect([&] {
        require(descriptor != nullptr && std::isfinite(scale), "FFT scale must be finite");
        descriptor->fft_output_scale = scale;
    });
}

cubutterflyStatus_t cubutterflySetStrides(cubutterflyDescriptor_t descriptor, const size_t* input_strides, size_t input_batch_stride,
                                          const size_t* output_strides, size_t output_batch_stride) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        std::vector<std::size_t> in, out;
        if (input_strides) in.assign(input_strides, input_strides + descriptor->rank);
        if (output_strides) out.assign(output_strides, output_strides + descriptor->rank);
        descriptor->input_strides = std::move(in);
        descriptor->output_strides = std::move(out);
        descriptor->input_batch_stride = input_batch_stride;
        descriptor->output_batch_stride = output_batch_stride;
    });
}

cubutterflyStatus_t cubutterflySetLogicalOrders(cubutterflyDescriptor_t descriptor,
    const cubutterflyLogicalOrder_t* input_orders, const cubutterflyLogicalOrder_t* output_orders) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        std::vector<cubutterflyLogicalOrder_t> in(descriptor->rank, CUBUTTERFLY_ORDER_NATURAL);
        auto out = in;
        if (input_orders) std::copy(input_orders, input_orders + descriptor->rank, in.begin());
        if (output_orders) std::copy(output_orders, output_orders + descriptor->rank, out.begin());
        for (const auto& side : {in, out})
            for (const auto order : side)
                require(order == CUBUTTERFLY_ORDER_NATURAL || order == CUBUTTERFLY_ORDER_BIT_REVERSED,
                        "unknown logical order");
        descriptor->input_orders = std::move(in);
        descriptor->output_orders = std::move(out);
        descriptor->input_ntt_layout = descriptor->output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    });
}

cubutterflyStatus_t cubutterflySetDirection(cubutterflyDescriptor_t descriptor, cubutterflyDirection_t direction, int normalize_inverse) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->direction = direction; descriptor->normalize_inverse = normalize_inverse != 0; });
}

cubutterflyStatus_t cubutterflySetPlacement(cubutterflyDescriptor_t descriptor, cubutterflyPlacement_t placement) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->placement = placement; });
}

cubutterflyStatus_t cubutterflySetLengthMode(cubutterflyDescriptor_t descriptor, cubutterflyLengthMode_t mode) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->length_mode = mode; });
}

cubutterflyStatus_t cubutterflySetAlgorithmPolicy(cubutterflyDescriptor_t descriptor, cubutterflyAlgorithmPolicy_t policy, const char* explicit_algorithm) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        require(policy != CUBUTTERFLY_ALGORITHM_EXPLICIT || (explicit_algorithm != nullptr && explicit_algorithm[0] != '\0'),
                "explicit policy requires an algorithm name");
        descriptor->algorithm_policy = policy;
        descriptor->explicit_algorithm = explicit_algorithm == nullptr ? "" : explicit_algorithm;
    });
}

cubutterflyStatus_t cubutterflySetModulus(cubutterflyDescriptor_t descriptor, uint64_t modulus, uint32_t word_bits) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); require(word_bits == 32 || word_bits == 64, "word width must be 32 or 64"); descriptor->modulus = modulus; descriptor->word_bits = word_bits; });
}

cubutterflyStatus_t cubutterflySetNttLayouts(
    cubutterflyDescriptor_t descriptor, cubutterflyNttLayout_t input_layout,
    cubutterflyNttLayout_t output_layout) {
    return protect([&] {
        require(descriptor != nullptr, "descriptor must be non-null");
        require((input_layout == CUBUTTERFLY_NTT_LAYOUT_NATURAL ||
                 input_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC ||
                 input_layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED) &&
                    (output_layout == CUBUTTERFLY_NTT_LAYOUT_NATURAL ||
                     output_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC ||
                     output_layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED),
                "unknown NTT layout");
        descriptor->input_ntt_layout = input_layout;
        descriptor->output_ntt_layout = output_layout;
        descriptor->input_orders.clear();
        descriptor->output_orders.clear();
    });
}

cubutterflyStatus_t cubutterflySetStageMatrices(cubutterflyDescriptor_t descriptor, uint32_t axis,
                                                const cubutterflyMatrix2x2_t* matrices, size_t count) {
    return protect([&] {
        require(descriptor != nullptr && matrices != nullptr && count != 0, "matrix arguments must be non-null and non-empty");
        require(axis < descriptor->rank, "matrix axis must be in the descriptor rank");
        descriptor->stage_matrices[axis].assign(matrices, matrices + count);
    });
}

cubutterflyStatus_t cubutterflyCreatePlan(cubutterflyHandle_t handle, cubutterflyDescriptor_t descriptor, cubutterflyPlan_t* plan) {
    return protect([&] {
        require(handle != nullptr && descriptor != nullptr && plan != nullptr, "handle, descriptor, and plan pointer must be non-null");
        *plan = nullptr;
        auto resolved = cubutterfly::detail::normalize_descriptor(*descriptor);
        const auto capabilities = static_capabilities(resolved);
        if (capabilities.at("status") != "eligible")
            throw unsupported_error(capabilities.at("gaps").dump());
        if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT &&
            !resolved.explicit_algorithm.empty() && resolved.explicit_algorithm.front() == '{') {
            const auto replay = nlohmann::json::parse(resolved.explicit_algorithm);
            if (replay.contains("contract"))
                require(replay.at("contract") == xd_contract(resolved), "mapping semantic contract mismatch");
        }
        if (requires_semantic_boundary(resolved)) {
            *plan = make_boundary_plan(handle, resolved).release();
            return;
        }
        const bool fp32_fp64 = (resolved.storage == CUBUTTERFLY_DATA_FP32 ||
            resolved.storage == CUBUTTERFLY_DATA_COMPLEX_FP32) && resolved.compute == CUBUTTERFLY_COMPUTE_FP64;
        const bool low_real = resolved.fft_type != CUBUTTERFLY_FFT_C2C &&
            (resolved.storage == CUBUTTERFLY_DATA_FP16 || resolved.storage == CUBUTTERFLY_DATA_BF16);
        if (fp32_fp64 || low_real) {
            *plan = make_numeric_plan(handle, resolved, low_real).release();
            return;
        }
        if (resolved.rank > 2 && resolved.fft_type == CUBUTTERFLY_FFT_C2C &&
            resolved.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
            std::any_of(resolved.extents.begin(), resolved.extents.end(),
                        [](std::size_t n) { return !is_power_of_two(n); })) {
            *plan = make_arbitrary_nd_plan(handle, resolved).release();
            return;
        }
        // Keep the native NTT output permutation eligible. Other per-axis
        // orders use the common boundary around an equivalent natural plan.
        if (native_ntt_reversed_output(resolved))
            resolved.output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED;
        // Preserve the established NTT contract: modular inverse transforms
        // always apply inverse_n. Alternative physical lowerings must not
        // interpret the floating-transform normalization flag differently.
        if (resolved.op == CUBUTTERFLY_OPERATOR_NTT) resolved.normalize_inverse = true;
        require(resolved.execution_mode != CUBUTTERFLY_EXECUTION_XD || resolved.rank >= 2,
                "xD execution requires rank at least two");
        validate_storage(resolved);
        if (resolved.fft_type != CUBUTTERFLY_FFT_C2C) {
            const auto compute = resolved.storage == CUBUTTERFLY_DATA_FP64
                ? CUBUTTERFLY_COMPUTE_FP64 : CUBUTTERFLY_COMPUTE_FP32;
            if (resolved.compute != CUBUTTERFLY_COMPUTE_NATIVE && resolved.compute != compute)
                throw unsupported_error("compact real FFT compute precision must match storage");
            if (resolved.length_mode != CUBUTTERFLY_LENGTH_STANDARD ||
                resolved.placement != CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE ||
                (resolved.storage != CUBUTTERFLY_DATA_FP32 && resolved.storage != CUBUTTERFLY_DATA_FP64))
                throw unsupported_error("compact R2C/C2R requires out-of-place standard FP32/FP64");
            auto half = resolved.extents;
            half.back() = half.back() / 2 + 1;
            const bool inverse = resolved.fft_type == CUBUTTERFLY_FFT_C2R;
            const auto& in = inverse ? half : resolved.extents;
            const auto& out = inverse ? resolved.extents : half;
            resolve_default_strides(resolved, in, out);
            if (!packed_side(resolved, true, in) || !packed_side(resolved, false, out))
                throw unsupported_error("compact R2C/C2R currently requires packed input and output");
            auto result = std::make_unique<cubutterflyPlan>();
            result->descriptor = resolved;
            result->physical_extents = resolved.extents;
            const auto scalar = element_bytes(resolved.storage);
            result->element_bytes = scalar;
            result->input_bytes = checked_product(strided_elements(resolved, true, in), inverse ? 2 * scalar : scalar);
            result->output_bytes = checked_product(strided_elements(resolved, false, out), inverse ? scalar : 2 * scalar);
            // Bump the cache generation when the complete-plan tuning gate or
            // promotion protocol changes.  A previous static packed fallback
            // must not survive an upgrade and hide the newly legal candidates.
            const auto key = workload_key(resolved) + "|xd-real-v26-packed-pair-fp64|vkfft-" +
                             (cuntt::detail::xd_vkfft_axis_available() ? "1" : "0") + "|raw-normalized" +
                             (resolved.normalize_inverse ? "1" : "0") + "|promotion-factor-" +
                             std::to_string(xd_promotion_factor());
            nlohmann::json replay;
            std::string forced_axis_family;
            bool cached = false;
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
                if (!resolved.explicit_algorithm.empty() &&
                    resolved.explicit_algorithm.front() == '{') {
                    replay = nlohmann::json::parse(resolved.explicit_algorithm);
                } else {
                    forced_axis_family = resolved.explicit_algorithm.rfind("xd:", 0) == 0
                        ? resolved.explicit_algorithm.substr(3) : resolved.explicit_algorithm;
                    require(forced_axis_family == "cufft-whole-real" ||
                                forced_axis_family == "pair-tiled-radix4" ||
                                forced_axis_family == "pair-tiled-cufftdx" ||
                                forced_axis_family == "axis-tiled-local" ||
                                forced_axis_family == "axis-tiled-local-cap16" ||
                                forced_axis_family == "axis-tiled-local-recurrence" ||
                                forced_axis_family == "axis-vkfft",
                            "compact real explicit lowering must be a supported axis family or mapping JSON");
                }
            } else if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT) {
                const auto text = lookup_cache(handle->cache_path,key);
                if (!text.empty()) { replay = nlohmann::json::parse(text); cached = true; }
            }
            if (!replay.is_null())
                require(replay.at("contract") == xd_contract(resolved) && replay.at("target") == xd_target(),
                        "real xD replay contract or target mismatch");
            const bool whole_real_available =
                resolved.rank >= 2 && resolved.rank <= 3 &&
                (resolved.fft_type == CUBUTTERFLY_FFT_R2C ||
                 resolved.fft_type == CUBUTTERFLY_FFT_C2R) &&
                !nonnatural_order(resolved) && packed_side(resolved, true, in) &&
                packed_side(resolved, false, out);
            const bool whole_real_replay = !replay.is_null() &&
                replay.value("lowering", "") == "cufft-whole-real";
            const bool whole_real_requested = forced_axis_family == "cufft-whole-real" ||
                                               whole_real_replay;
            if (whole_real_requested)
                require(whole_real_available,
                        "complete real cuFFT lowering requires packed natural R2C/C2R semantics");
            std::unique_ptr<cuntt::detail::DirectRealFftPlan> whole_real_plan;
            if (whole_real_available &&
                (whole_real_requested || (replay.is_null() && forced_axis_family.empty()))) {
                whole_real_plan = std::make_unique<cuntt::detail::DirectRealFftPlan>(
                    resolved.extents, resolved.batch, inverse, resolved.normalize_inverse,
                    resolved.storage == CUBUTTERFLY_DATA_FP64, resolved.fft_output_scale);
            }
            std::vector<std::unique_ptr<cuntt::detail::XdRealFftPlan>> candidates;
            std::size_t whole_real_workspace = whole_real_plan ? whole_real_plan->workspace_size() : 0;
            // Packed axis batches provide a bounded-launch fallback even when
            // empirical promotion is omitted for a very large allocation.
            // Each choice selects packed fallback and fused axis cores. Compact
            // real boundaries remain vendor R2C/C2R, with the same spectrum.
            struct RealAxisChoice {
                bool packed; bool fused; unsigned ept;
                cuntt::detail::XdAxisCore core = cuntt::detail::XdAxisCore::CufftdxBlock;
                unsigned register_lanes = 4;
                unsigned block_cap = 0;
                unsigned exchange = 0;
                bool local = false;
                unsigned local_twiddle_policy = 0;
                unsigned exact_ffts = 0;
                bool paired = false;
                cuntt::detail::XdPairCore pair_core = cuntt::detail::XdPairCore::Thread;
            };
            const auto replay_core = replay.is_null()
                ? cuntt::detail::XdAxisCore::CufftdxBlock
                : cuntt::detail::xd_axis_core_from_name(
                      replay.at("physical").value("axis_core", "cufftdx-block"));
            const bool replay_local_family = !replay.is_null() &&
                replay.value("lowering", "").find("[axis-tiled-local") != std::string::npos;
            unsigned replay_local_twiddle_policy = 0;
            if (replay_local_family && replay.at("physical").contains("axis_resources")) {
                for (const auto& resource : replay.at("physical").at("axis_resources")) {
                    if (resource.value("core", "") == "thread-shared" &&
                        resource.contains("local_twiddle_policy") &&
                        resource.at("local_twiddle_policy").is_number_unsigned() &&
                        resource.at("local_twiddle_policy").get<std::uint64_t>() == 1U) {
                        replay_local_twiddle_policy = 1;
                        break;
                    }
                }
            }
            std::vector<RealAxisChoice> axis_choices = replay.is_null()
                ? std::vector<RealAxisChoice>{{true,false,8}}
                : std::vector<RealAxisChoice>{{replay.at("physical").value("packed_axes",false),
                                               replay.at("physical").value("fused_axes",false),
                                               replay.at("physical").value("axis_ept",8U),
                                               replay_core,
                                               replay.at("physical").value("register_lanes",4U),
                                               replay.at("physical").value("axis_ffts_per_block_cap",0U),
                                               replay.at("physical").value("axis_exchange_policy",0U),
                                               replay_local_family,
                                               replay_local_twiddle_policy}};
            if (whole_real_requested)
                axis_choices.clear();
            if (!replay.is_null() && !axis_choices.empty()) {
                axis_choices.front().paired = replay.at("physical").value("paired_axes",false);
                axis_choices.front().pair_core = cuntt::detail::xd_pair_core_from_name(
                    replay.at("physical").value("pair_core",std::string("thread")));
            }
            auto resource_omissions = nlohmann::json::array();
            if (replay.is_null() && forced_axis_family.empty() && resolved.rank > 1 &&
                resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE) {
                if (cuntt::detail::xd_vkfft_axis_available())
                    axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::Vkfft});
                else
                    resource_omissions.push_back({{"axis_core","vkfft"},
                        {"reason","VkFFT axis provider is not enabled in this build"}});
            }
            if (replay.is_null() && resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE &&
                resolved.rank > 1 && xd_prefix_launches(resolved) <= 256)
                axis_choices.push_back({false,false,8});
            if (replay.is_null() && resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE &&
                std::any_of(resolved.extents.begin(), resolved.extents.end() - 1,
                            cuntt::detail::xd_axis_fft_available)) {
                axis_choices.push_back({true,true,8});
                axis_choices.push_back({true,true,16});
                for (unsigned ept : {8U,16U})
                    axis_choices.push_back({true,true,ept,
                        cuntt::detail::XdAxisCore::CufftdxBlock,4,0,4});
                for (unsigned ept : {8U,16U})
                    for (unsigned cap : {8U,16U})
                        if (std::any_of(resolved.extents.begin(),resolved.extents.end()-1,
                            [&](std::size_t length) {
                                return cuntt::detail::xd_axis_fft_available(length) &&
                                    cuntt::detail::xd_axis_fft_block_batch(length,
                                        resolved.storage == CUBUTTERFLY_DATA_FP64,ept) > cap;
                            }))
                            axis_choices.push_back({true,true,ept,cuntt::detail::XdAxisCore::CufftdxBlock,4,cap});
                for (unsigned exchange : {1U,2U})
                    for (unsigned ept : {8U,16U})
                        for (unsigned cap : {0U,8U,16U})
                            if (cap == 0 || std::any_of(resolved.extents.begin(),resolved.extents.end()-1,
                                [&](std::size_t length) {
                                    return cuntt::detail::xd_axis_fft_available(length) &&
                                        cuntt::detail::xd_axis_fft_block_batch(length,
                                            resolved.storage == CUBUTTERFLY_DATA_FP64,ept) > cap;
                                }))
                                axis_choices.push_back({true,true,ept,cuntt::detail::XdAxisCore::CufftdxBlock,4,cap,exchange});
                for (unsigned lanes : {4U,8U,16U})
                    if (std::any_of(resolved.extents.begin(), resolved.extents.end() - 1,
                        [&](std::size_t length) {
                            return cuntt::detail::xd_register_fft_configuration_available(length, lanes);
                        }))
                        axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::CooperativeRegister,lanes});
                std::size_t compact_points = resolved.batch;
                for (std::size_t axis = 0; axis < resolved.extents.size(); ++axis)
                    compact_points = checked_product(compact_points,
                        axis + 1 == resolved.extents.size() ? resolved.extents[axis] / 2 + 1 : resolved.extents[axis]);
                const bool narrow_indices = compact_points <= std::numeric_limits<std::uint32_t>::max();
                if (!narrow_indices)
                    resource_omissions.push_back({{"axis_exchange_policy",3},
                        {"reason","compact tensor exceeds 32-bit element index range"}});
                for (unsigned exchange : {0U,1U,2U,3U})
                    for (unsigned ept : {4U,8U,16U}) {
                        if ((ept != 4 && exchange != 3) || (exchange == 3 && !narrow_indices)) continue;
                        for (unsigned cap : {0U,8U,16U})
                            if (cap == 0 || std::any_of(resolved.extents.begin(),resolved.extents.end()-1,
                                [&](std::size_t length) {
                                    return cuntt::detail::xd_axis_fft_available(length) &&
                                        cuntt::detail::xd_axis_fft_block_batch(length,
                                            resolved.storage == CUBUTTERFLY_DATA_FP64,ept) > cap;
                                }))
                                axis_choices.push_back({true,true,ept,cuntt::detail::XdAxisCore::CufftdxBlock,4,cap,exchange});
                    }
            }
            if (replay.is_null()) {
                const bool local_available = std::any_of(
                    resolved.extents.begin(), resolved.extents.end() - 1,
                    [](std::size_t length) {
                        return cuntt::detail::xd_local_fft_available(length);
                    });
                if (forced_axis_family == "cufft-whole-real") {
                    axis_choices.clear();
                } else if (forced_axis_family == "axis-vkfft") {
                    if (!cuntt::detail::xd_vkfft_axis_available() || resolved.rank < 2)
                        throw unsupported_error("forced VkFFT axis provider is unavailable or has no complex axis");
                    axis_choices = {{true,true,8,cuntt::detail::XdAxisCore::Vkfft}};
                } else if (forced_axis_family == "pair-tiled-radix4" ||
                           forced_axis_family == "pair-tiled-cufftdx") {
                    axis_choices = {{true,true,8}};
                    axis_choices.back().paired = true;
                    axis_choices.back().pair_core = forced_axis_family == "pair-tiled-radix4"
                        ? cuntt::detail::XdPairCore::Radix4Register : cuntt::detail::XdPairCore::Thread;
                } else if (!forced_axis_family.empty()) {
                    if (!local_available)
                        throw unsupported_error(
                            "forced compact local family is unavailable for these axes or target SM");
                    const unsigned cap = forced_axis_family == "axis-tiled-local-cap16" ? 16U : 0U;
                    const unsigned twiddle = forced_axis_family == "axis-tiled-local-recurrence" ? 1U : 0U;
                    axis_choices.clear();
                    axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::ThreadShared,
                                            4,cap,0,true,twiddle});
                } else if (resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE &&
                           local_available) {
                    axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::ThreadShared,
                                            4,0,0,true});
                    axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::ThreadShared,
                                            4,16,0,true});
                    axis_choices.push_back({true,true,8,cuntt::detail::XdAxisCore::ThreadShared,
                                            4,0,0,true,1});
                }
            }
            if (replay.is_null() && forced_axis_family.empty() && resolved.rank >= 3 &&
                resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE) {
                for (const auto core : {cuntt::detail::XdPairCore::Thread,
                                        cuntt::detail::XdPairCore::Radix4Register}) {
                    RealAxisChoice pair{true,true,8};
                    pair.paired = true;
                    pair.pair_core = core;
                    axis_choices.push_back(pair);
                }
            }
            // Group independent long-axis lines in one CTA. The exact count is
            // independent of the historical short-axis cap and default policy.
            if (replay.is_null() && forced_axis_family.empty() &&
                resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE &&
                std::any_of(resolved.extents.begin(),resolved.extents.end()-1,
                    [](std::size_t n) { return n > 1024 && cuntt::detail::xd_axis_fft_length_supported(n); })) {
                for (unsigned ept : {8U,16U})
                    for (unsigned exact : {2U,4U})
                        for (unsigned exchange : {0U,2U,3U})
                            axis_choices.push_back({true,true,ept,
                                cuntt::detail::XdAxisCore::CufftdxBlock,4,0,exchange,false,0,exact});
            }
            const auto axis_orders = xd_axis_orders(resolved, false, replay);
            std::vector<bool> candidate_packed;
            std::vector<bool> candidate_fused;
            std::vector<unsigned> candidate_ept;
            std::vector<cuntt::detail::XdAxisCore> candidate_core;
            std::vector<unsigned> candidate_lanes;
            std::vector<unsigned> candidate_caps;
            std::vector<unsigned> candidate_exchange;
            std::vector<unsigned> candidate_local_twiddle;
            std::vector<std::vector<std::size_t>> candidate_orders;
            std::vector<bool> candidate_prefix_scale;
            std::vector<unsigned> candidate_exact_ffts;
            std::size_t max_work = whole_real_workspace;
            for (const auto& choice : axis_choices) {
                const bool packed_axes = choice.packed, fused_axes = choice.fused;
                if (replay.is_null() && fused_axes && !choice.local && !choice.paired &&
                    choice.core == cuntt::detail::XdAxisCore::CufftdxBlock &&
                    std::none_of(resolved.extents.begin(), resolved.extents.end() - 1,
                        [&](std::size_t length) {
                            return cuntt::detail::xd_axis_fft_configuration_available(length,
                                resolved.storage == CUBUTTERFLY_DATA_FP64,
                                choice.ept, choice.block_cap);
                        })) {
                    resource_omissions.push_back({{"axis_ept", choice.ept},
                        {"axis_ffts_per_block_cap", choice.block_cap},
                        {"axis_exchange_policy", choice.exchange},
                        {"reason", "no compiled axis configuration fits lengths and device resources"}});
                    continue;
                }
                for (const auto& order : axis_orders) {
                    std::vector<XdAxisResources> resources = replay.is_null()
                        ? (choice.local ? xd_uniform_local_resources(
                              resolved.extents, resolved.extents.size() - 1U,
                              choice.block_cap, choice.local_twiddle_policy)
                                        : std::vector<XdAxisResources>{})
                        : xd_axis_resources(replay.at("physical"));
                    if (choice.local && !replay.is_null())
                        require(!resources.empty(), "local real xD replay is missing axis_resources");
                    if (choice.exact_ffts != 0) {
                        bool available = true;
                        for (std::size_t axis = 0; axis + 1 < resolved.extents.size(); ++axis) {
                            const auto length = resolved.extents[axis];
                            const auto exact = length > 1024 ? choice.exact_ffts : 0U;
                            resources.push_back({choice.core,choice.ept,4,0,choice.exchange,0,exact});
                            if (exact != 0 && !cuntt::detail::xd_axis_fft_configuration_available(
                                    length,resolved.storage == CUBUTTERFLY_DATA_FP64,
                                    choice.ept,0,exact,choice.exchange))
                                available = false;
                        }
                        if (!available) {
                            resource_omissions.push_back({{"axis_ept",choice.ept},
                                {"exact_ffts_per_block",choice.exact_ffts},
                                {"axis_exchange_policy",choice.exchange},
                                {"reason","grouped axis exceeds compiled or device resource support"}});
                            continue;
                        }
                    }
                    std::unique_ptr<cuntt::detail::XdRealFftPlan> candidate;
                    try {
                        candidate = std::make_unique<cuntt::detail::XdRealFftPlan>(
                            resolved.extents,resolved.batch,inverse,resolved.normalize_inverse,
                            resolved.storage == CUBUTTERFLY_DATA_FP64,false,resolved.fft_output_scale,
                            packed_axes,order,fused_axes,choice.ept,choice.paired,choice.core,
                            choice.pair_core,choice.register_lanes,0,0,choice.block_cap,choice.exchange,
                            resources, !replay.is_null() && replay.at("physical").value("c2r_prefix_scale",false));
                    } catch (const std::exception& error) {
                        if (!replay.is_null() || !forced_axis_family.empty() ||
                            (!choice.paired && choice.core != cuntt::detail::XdAxisCore::Vkfft)) throw;
                        resource_omissions.push_back({{"axis_core",cuntt::detail::xd_axis_core_name(choice.core)},
                            {"axis_order",order},{"phase","construction"},{"reason",error.what()}});
                        continue;
                    }
                    const auto physical = nlohmann::json::parse(candidate->mapping_json());
                    if (choice.paired && physical.value("pair_groups",0U) == 0) {
                        if (!forced_axis_family.empty() || !replay.is_null())
                            throw unsupported_error("compact-real pair request has no eligible pair of complex axes");
                        resource_omissions.push_back({{"pair_core",cuntt::detail::xd_pair_core_name(choice.pair_core)},
                            {"axis_order",order},{"reason","no eligible pair of complex axes"}});
                        continue;
                    }
                    if (choice.local && !xd_mapping_has_axis_core(physical, "thread-shared-axis")) {
                        resource_omissions.push_back({{"lowering", choice.local_twiddle_policy == 1
                                                             ? "axis-tiled-local-recurrence"
                                                             : "axis-tiled-local"},
                                                      {"reason", "local family produced no actual local axis"}});
                        continue;
                    }
                    if (choice.core == cuntt::detail::XdAxisCore::CooperativeRegister &&
                        (replay.is_null() || !replay.at("physical").contains("axis_resources"))) {
                        const bool uses_register_axis = std::any_of(
                            physical.at("axes").begin(), physical.at("axes").end(), [](const auto& axis) {
                                return axis.value("core", "") == "register-cooperative-axis";
                            });
                        if (!uses_register_axis) {
                            require(replay.is_null(), "real FFT replay requests an unavailable register axis");
                            continue;
                        }
                    }
                    candidates.push_back(std::move(candidate));
                    candidate_packed.push_back(packed_axes);
                    candidate_fused.push_back(fused_axes);
                    candidate_ept.push_back(choice.ept);
                    candidate_core.push_back(choice.core);
                    candidate_lanes.push_back(choice.register_lanes);
                    candidate_caps.push_back(choice.block_cap);
                    candidate_exchange.push_back(choice.exchange);
                    candidate_local_twiddle.push_back(choice.local_twiddle_policy);
                    candidate_orders.push_back(order);
                    candidate_prefix_scale.push_back(!replay.is_null() &&
                        replay.at("physical").value("c2r_prefix_scale",false));
                    candidate_exact_ffts.push_back(choice.exact_ffts);
                    max_work = std::max(max_work,candidates.back()->workspace_size());
                }
            }
            // Scaling is a lowering choice, not an application-specific input
            // rewrite. Keep the established post-C2R path and compare complete
            // plans with scale absorbed by the final complex core's store.
            if (replay.is_null() && forced_axis_family.empty() && inverse &&
                (resolved.fft_output_scale != 1.0 || resolved.normalize_inverse)) {
                const auto original_count = candidates.size();
                for (std::size_t index = 0; index < original_count; ++index) {
                    if (!candidates[index]->can_fuse_c2r_scale()) continue;
                    auto physical = nlohmann::json::parse(candidates[index]->mapping_json());
                    physical["c2r_prefix_scale"] = true;
                    auto scaled = xd_plan_with_resources(physical, xd_axis_resources(physical));
                    max_work = std::max(max_work,scaled->workspace_size());
                    candidates.push_back(std::move(scaled));
                    candidate_packed.push_back(candidate_packed[index]);
                    candidate_fused.push_back(candidate_fused[index]);
                    candidate_ept.push_back(candidate_ept[index]);
                    candidate_core.push_back(candidate_core[index]);
                    candidate_lanes.push_back(candidate_lanes[index]);
                    candidate_caps.push_back(candidate_caps[index]);
                    candidate_exchange.push_back(candidate_exchange[index]);
                    candidate_local_twiddle.push_back(candidate_local_twiddle[index]);
                    candidate_orders.push_back(candidate_orders[index]);
                    candidate_prefix_scale.push_back(true);
                    candidate_exact_ffts.push_back(candidate_exact_ffts[index]);
                }
            }
            const auto io_bytes = std::max(result->input_bytes,result->output_bytes);
            auto measurements = nlohmann::json::array();
            nlohmann::json tuning_diagnostics{
                {"candidate_choice_count", axis_choices.size()},
                {"axis_order_count", axis_orders.size()},
                {"candidate_count", candidates.size() + (whole_real_plan ? 1U : 0U)},
                {"requested_lowering", replay.is_null() ? "auto" : "replay"}};
            std::size_t best = 0;
            std::string tuning_protocol = "not-measured";
            std::string graph_fallback_reason;
            std::unique_ptr<cuntt::detail::XdRealFftPlan> refined_plan;
            bool whole_real_selected = whole_real_requested;
            if (replay.is_null() && forced_axis_family.empty() &&
                (candidates.size() > 1 || whole_real_plan)) {
                const auto tuning_status = xd_tuning_status(io_bytes, max_work);
                tuning_diagnostics["io_bytes"] = io_bytes;
                tuning_diagnostics["workspace_bytes"] = max_work;
                tuning_diagnostics["tuning"] = {
                    {"attempted", tuning_status.fits},
                    {"budget_bytes", tuning_status.budget_bytes},
                    {"free_bytes", tuning_status.free_bytes},
                    {"total_bytes", tuning_status.total_bytes},
                    {"reason", tuning_status.reason}};
            if (tuning_status.fits) {
                XdTuningStream tuning_stream(handle);
                DeviceAllocation input(io_bytes), output(io_bytes), workspace(max_work);
                const auto stream = tuning_stream.get();
                require(cudaMemsetAsync(input.pointer,0,io_bytes,stream) == cudaSuccess,
                        "initialize real xD tuning input");
                if (whole_real_plan && whole_real_plan->workspace_size())
                    whole_real_plan->set_workspace(workspace.pointer, max_work);
                std::vector<std::unique_ptr<XdCudaGraph>> graphs(candidates.size());
                // Compact-real candidates mix cuFFT R2C/C2R boundaries with
                // optional custom C2C axis cores. On the A100 runtime the
                // CUDA-Graph replay of this mixed sequence can report a
                // materially lower event time than the same complete plan
                // launched normally. Ranking with that value selected a
                // slower R2C axis path in the public adapter. Use the launch
                // event protocol for this selector so every candidate is
                // ranked by the execution protocol used by the adapter.
                // Graph capture remains available to the application caller
                // and to the already validated complex xD selector.
                bool graph_timing = false;
                graph_fallback_reason =
                    "compact-real mixed cuFFT/custom-axis graph timing disabled for selection";
                if (graph_timing) {
                    try {
                        // Bind each candidate before capture so graph and fallback
                        // execute the same complete-plan workspace contract.
                        for (std::size_t index = 0; index < candidates.size(); ++index) {
                            auto& candidate = candidates[index];
                            if (candidate->workspace_size()) candidate->set_workspace(workspace.pointer,max_work);
                            graphs[index] = capture_xd_cuda_graph(stream, [&] {
                                candidates[index]->execute(input.pointer,output.pointer,stream);
                            });
                        }
                    } catch (const std::exception& error) {
                        graph_timing = false;
                        graph_fallback_reason = error.what();
                        graphs.clear();
                        (void)cudaStreamSynchronize(stream);
                    } catch (...) {
                        graph_timing = false;
                        graph_fallback_reason = "unknown CUDA Graph capture failure";
                        graphs.clear();
                        (void)cudaStreamSynchronize(stream);
                    }
                }
                tuning_protocol = graph_timing ? "cuda-graph-steady-state" : "cuda-device-synchronized";
                double best_ms = std::numeric_limits<double>::infinity();
                std::size_t axis_seed = candidates.size();
                double axis_seed_ms = std::numeric_limits<double>::infinity();
                for (std::size_t index = 0; index < candidates.size(); ++index) {
                    auto& candidate = candidates[index];
                    if (candidate->workspace_size()) candidate->set_workspace(workspace.pointer,max_work);
                    std::vector<double> times;
                    try {
                        for (unsigned trial = 0; trial < 3; ++trial)
                            times.push_back(graph_timing
                                ? measure_xd_cuda_graph(stream, graphs[index]->executable())
                                : measure_cuda_candidate_synchronized([&] {
                                    candidate->execute(input.pointer,output.pointer,stream);
                                }));
                    } catch (const std::bad_alloc&) { throw; }
                      catch (const std::exception& error) {
                        (void)cudaGetLastError();
                        if (cudaStreamSynchronize(stream) != cudaSuccess) throw;
                        candidate->set_workspace(nullptr,0);
                        resource_omissions.push_back({{"candidate", index}, {"reason", error.what()},
                                                      {"phase", "measurement"}});
                        continue;
                    }
                    std::sort(times.begin(),times.end());
                    measurements.push_back({{"packed_axes",candidate_packed[index]},
                                            {"paired_axes",nlohmann::json::parse(candidate->mapping_json()).value("paired_axes",false)},
                                            {"fused_axes",candidate_fused[index]},
                                            {"axis_ept",candidate_ept[index]},
                                            {"register_lanes",candidate_lanes[index]},
                                            {"axis_ffts_per_block_cap",candidate_caps[index]},
                                            {"axis_exchange_policy",candidate_exchange[index]},
                                            {"local_twiddle_policy",candidate_local_twiddle[index]},
                                            {"c2r_prefix_scale",candidate_prefix_scale[index]},
                                            {"exact_ffts_per_block",candidate_exact_ffts[index]},
                                            {"axis_core",cuntt::detail::xd_axis_core_name(candidate_core[index])},
                                            {"axis_order",candidate_orders[index]},
                                            {"median_ms",times[1]},
                                            {"trials_ms",times},
                                            {"timing_protocol",tuning_protocol}});
                    auto& measurement = measurements.back();
                    if (graph_timing) {
                        measurement["warmup_plan_executions"] = 1;
                        measurement["warmup_graph_launches"] = kXdGraphWarmupLaunches;
                        measurement["warmup_total_plan_executions"] =
                            1 + kXdGraphWarmupLaunches * kXdGraphPlanExecutions;
                        measurement["captured_plan_executions"] = kXdGraphPlanExecutions;
                        measurement["timed_graph_launches"] = kXdGraphTimedLaunches;
                        measurement["timed_plan_executions"] = kXdGraphPlanExecutions * kXdGraphTimedLaunches;
                    } else {
                        measurement["warmup_launches"] = 2;
                        measurement["timed_launches"] = 5;
                        measurement["graph_fallback_reason"] = graph_fallback_reason;
                    }
                    if (times[1] < best_ms * xd_promotion_factor()) {
                        best = index;
                        best_ms = times[1];
                    }
                    if (xd_axis_refinement_seed(nlohmann::json::parse(candidate->mapping_json())) && times[1] < axis_seed_ms) {
                        axis_seed = index;
                        axis_seed_ms = times[1];
                    }
                    candidate->set_workspace(nullptr,0);
                }
                // Measure the complete vendor real transform with the same
                // steady-state protocol.  It has no caller workspace, so it
                // can be compared directly with the axis candidates without
                // changing the allocation contract.
                if (whole_real_plan) {
                    std::unique_ptr<XdCudaGraph> whole_graph;
                    bool whole_graph_ok = graph_timing;
                    if (graph_timing) {
                        try {
                            whole_graph = capture_xd_cuda_graph(stream, [&] {
                                whole_real_plan->execute(input.pointer, output.pointer, stream);
                            });
                        } catch (const std::exception& error) {
                            whole_graph_ok = false;
                            resource_omissions.push_back({{"lowering", "cufft-whole-real"},
                                               {"phase", "graph-capture"},
                                               {"reason", error.what()}});
                        } catch (...) {
                            whole_graph_ok = false;
                            resource_omissions.push_back({{"lowering", "cufft-whole-real"},
                                               {"phase", "graph-capture"},
                                               {"reason", "unknown CUDA Graph capture failure"}});
                        }
                    }
                    if (whole_graph_ok || !graph_timing) {
                        std::vector<double> times;
                        try {
                            for (unsigned trial = 0; trial < 3; ++trial)
                                times.push_back(whole_graph_ok
                                    ? measure_xd_cuda_graph(stream, whole_graph->executable())
                                    : measure_cuda_candidate_synchronized([&] {
                                        whole_real_plan->execute(input.pointer, output.pointer, stream);
                                    }));
                        } catch (const std::exception& error) {
                            resource_omissions.push_back({{"lowering", "cufft-whole-real"},
                                               {"phase", "measurement"},
                                               {"reason", error.what()}});
                            times.clear();
                        }
                        if (!times.empty()) {
                            std::sort(times.begin(), times.end());
                            nlohmann::json measurement{{"lowering", "cufft-whole-real"},
                                {"median_ms", times[1]}, {"trials_ms", times},
                                {"timing_protocol", whole_graph_ok
                                    ? "cuda-graph-steady-state" : "cuda-device-synchronized"},
                                {"workspace_bytes", whole_real_plan->workspace_size()}};
                            if (whole_graph_ok) {
                                measurement["warmup_plan_executions"] = 1;
                                measurement["warmup_graph_launches"] = kXdGraphWarmupLaunches;
                                measurement["warmup_total_plan_executions"] =
                                    1 + kXdGraphWarmupLaunches * kXdGraphPlanExecutions;
                                measurement["captured_plan_executions"] = kXdGraphPlanExecutions;
                                measurement["timed_graph_launches"] = kXdGraphTimedLaunches;
                                measurement["timed_plan_executions"] = kXdGraphPlanExecutions * kXdGraphTimedLaunches;
                            } else {
                                measurement["warmup_launches"] = 2;
                                measurement["timed_launches"] = 5;
                            }
                            measurements.push_back(std::move(measurement));
                            if (candidates.empty() || times[1] < best_ms * xd_promotion_factor()) {
                                best_ms = times[1];
                                whole_real_selected = true;
                            }
                        }
                    }
                }
                // Validate the tentative winner on the caller's original
                // stream.  cuFFT may use auxiliary internal work queues, so
                // the private tuning stream can under-report a complete
                // transform even with device synchronization.  This small
                // two-plan check keeps the search generic while making the
                // final choice agree with the stream used by the application.
                if (whole_real_plan && !whole_real_selected && !candidates.empty() &&
                    best < candidates.size() && forced_axis_family.empty()) {
                    const auto validation_stream = tuning_stream.original();
                    nlohmann::json validation{{"protocol", "caller-stream-device-synchronized"}};
                    try {
                        auto& winner = candidates[best];
                        if (winner->workspace_size()) winner->set_workspace(workspace.pointer, max_work);
                        const auto winner_ms = measure_cuda_candidate_synchronized([&] {
                            winner->execute(input.pointer, output.pointer, validation_stream);
                        });
                        winner->set_workspace(nullptr, 0);
                        if (whole_real_plan->workspace_size())
                            whole_real_plan->set_workspace(workspace.pointer, max_work);
                        const auto whole_ms = measure_cuda_candidate_synchronized([&] {
                            whole_real_plan->execute(input.pointer, output.pointer, validation_stream);
                        });
                        validation["winner_ms"] = winner_ms;
                        validation["whole_real_ms"] = whole_ms;
                        validation["winner_index"] = best;
                        validation["winner_algorithm"] = winner->mapping_json();
                        validation["whole_real_selected"] = whole_ms < winner_ms * xd_promotion_factor();
                        if (validation["whole_real_selected"].get<bool>()) {
                            whole_real_selected = true;
                            best_ms = whole_ms;
                        }
                    } catch (const std::exception& error) {
                        validation["error"] = error.what();
                    }
                    tuning_diagnostics["caller_stream_validation"] = std::move(validation);
                }
                if (!std::isfinite(best_ms)) {
                    // A deployment-sized tensor may exceed the temporary
                    // tuning budget. Retain a legal deterministic fallback
                    // instead of reporting a false unsupported plan.
                    if (!candidates.empty()) {
                        best = 0;
                        best_ms = 0.0;
                        whole_real_selected = false;
                    } else if (whole_real_plan) {
                        best_ms = 0.0;
                        whole_real_selected = true;
                    }
                }
                require(std::isfinite(best_ms), "no compact-real candidate survived measurement");
                if (!whole_real_selected && forced_axis_family.empty() && axis_seed < candidates.size()) {
                    auto refined = refine_xd_axis_resources(
                        nlohmann::json::parse(candidates[axis_seed]->mapping_json()),axis_seed_ms,
                        input.pointer,output.pointer,workspace.pointer,max_work,stream,graph_timing,
                        measurements,resource_omissions);
                    if (refined.plan && refined.milliseconds < best_ms * xd_promotion_factor())
                        refined_plan = std::move(refined.plan);
                }
            } else {
                // A skipped complete-plan search is a deployment constraint,
                // not evidence that the first packed candidate is optimal.
                tuning_diagnostics["tuning"]["selected_without_measurement"] = true;
            }
            } else {
                tuning_diagnostics["tuning"] = {
                    {"attempted", false},
                    {"reason", "single-executable-candidate"}};
            }
            tuning_diagnostics["candidate_count"] = candidates.size() + (whole_real_plan ? 1U : 0U);
            tuning_diagnostics["measured_count"] = measurements.size();
            tuning_diagnostics["rejected_count"] = resource_omissions.size();
            if (whole_real_plan)
                whole_real_plan->set_workspace(nullptr, 0);
            const bool replay_mixed = !replay.is_null() &&
                replay.value("lowering", "").find("[axis-tiled-mixed]") != std::string::npos;
            const bool mixed_axes = refined_plan != nullptr || replay_mixed;
            if (whole_real_selected) {
                require(whole_real_plan != nullptr,
                        "complete real cuFFT lowering was selected without a plan");
                result->direct_real_fft = std::move(whole_real_plan);
                result->workspace_bytes = result->direct_real_fft->workspace_size();
                result->algorithm = "cufft-whole-real";
            } else {
                require(!candidates.empty(), "no compact-real candidate is available");
                result->xd_fft = refined_plan ? std::move(refined_plan) : std::move(candidates[best]);
                result->workspace_bytes = result->xd_fft->workspace_size();
                result->algorithm = std::string(inverse ? "xd-compact-c2r" : "xd-compact-r2c") +
                    (mixed_axes ? "[axis-tiled-mixed]" :
                     nlohmann::json::parse(result->xd_fft->mapping_json()).value("paired_axes",false) ?
                        (nlohmann::json::parse(result->xd_fft->mapping_json()).value("pair_core",std::string("thread")) == "radix4-register"
                            ? "[pair-tiled-radix4]" : "[pair-tiled-cufftdx]") :
                     candidate_core[best] == cuntt::detail::XdAxisCore::Vkfft ? "[axis-vkfft]" :
                     candidate_core[best] == cuntt::detail::XdAxisCore::ThreadShared ?
                        (candidate_local_twiddle[best] == 1U ? "[axis-tiled-local-recurrence]" :
                         candidate_caps[best] == 16U ? "[axis-tiled-local-cap16]" :
                                                       "[axis-tiled-local]") :
                     candidate_core[best] == cuntt::detail::XdAxisCore::CooperativeRegister ? "[axis-tiled-register]" :
                     candidate_fused[best] && candidate_exchange[best] == 4 ?
                        (candidate_ept[best] == 16 ? "[axis-packed-cufftdx-ept16]" : "[axis-packed-cufftdx]") :
                     candidate_fused[best] ? (candidate_ept[best] == 4 ? "[axis-tiled-cufftdx-ept4]" :
                        candidate_ept[best] == 16 ? "[axis-tiled-cufftdx-ept16]" : "[axis-tiled-cufftdx]") :
                     candidate_packed[best] ? "[packed-axes]" : "[strided-axes]");
            }
            result->reason = whole_real_selected
                ? "complete vendor R2C/C2R plan selected as a legal xD compute core"
                : "compact Hermitian spectrum; per-axis cores; no full-spectrum expansion";
            result->reason += (measurements.empty() ? std::string{} : std::string("; timing=") + tuning_protocol) +
                (!graph_fallback_reason.empty() ? std::string("; graph_fallback_reason=") + graph_fallback_reason : std::string{}) +
                "; diagnostics=" + tuning_diagnostics.dump() +
                "; rejected=" + resource_omissions.dump() + "; candidates=" + measurements.dump();
            result->selection_source = cached ? CUBUTTERFLY_SELECTION_USER_CACHE :
                (!replay.is_null() || !forced_axis_family.empty()) ? CUBUTTERFLY_SELECTION_EXPLICIT :
                measurements.empty() ? CUBUTTERFLY_SELECTION_STATIC_MODEL : CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
            const auto physical_mapping = whole_real_selected
                ? nlohmann::json::parse(result->direct_real_fft->mapping_json())
                : nlohmann::json::parse(result->xd_fft->mapping_json());
            nlohmann::json mapping{{"schema_version", 1}, {"kind", "xd"},
                                  {"contract", xd_contract(resolved)}, {"target",xd_target()}, {"lowering", result->algorithm},
                                  {"physical", physical_mapping}};
            if (!replay.is_null()) {
                require(replay == mapping,
                        "real xD replay does not match the semantic or physical contract");
            }
            result->xd_mapping = mapping.dump();
            if (replay.is_null() && (!measurements.empty() || candidates.size() == 1))
                store_cache(handle->cache_path,key,result->xd_mapping);
            *plan = result.release();
            return;
        }
        if (resolved.fft_output_scale != 1.0)
            throw unsupported_error("additional FFT output scaling requires R2C/C2R");
        if (resolved.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED &&
            !(resolved.op == CUBUTTERFLY_OPERATOR_NTT && resolved.rank == 1 &&
              is_power_of_two(resolved.extents[0]) &&
              resolved.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
              resolved.input_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_NATURAL)) {
            throw unsupported_error("bit-reversed output requires a rank-1 power-of-two NTT with natural input");
        }
        const bool appt_layout_requested =
            resolved.input_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC ||
            resolved.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC;
        if (appt_layout_requested &&
            !(resolved.op == CUBUTTERFLY_OPERATOR_NTT && resolved.rank == 1 &&
              resolved.extents[0] == (1ULL << 20) &&
              resolved.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
              resolved.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE)) {
            throw unsupported_error(
                "APPT static layout currently requires a rank-1, logN=20, out-of-place NTT");
        }
        std::vector<std::size_t> physical_extents = resolved.extents;
        bool standard_arbitrary = false;
        if (resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING) {
            require(resolved.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE,
                    "zero-extended embedding currently requires out-of-place execution");
            for (auto& extent : physical_extents)
                extent = next_power_of_two(extent);
        } else {
            standard_arbitrary = std::any_of(physical_extents.begin(), physical_extents.end(),
                                             [](std::size_t extent) { return !is_power_of_two(extent); });
            if (standard_arbitrary) {
                require(resolved.op == CUBUTTERFLY_OPERATOR_FFT || resolved.op == CUBUTTERFLY_OPERATOR_NTT,
                        "standard arbitrary length supports FFT and NTT; use embedding mode for other operators");
                if (resolved.op == CUBUTTERFLY_OPERATOR_FFT)
                    require(resolved.storage == CUBUTTERFLY_DATA_COMPLEX_FP32 ||
                                resolved.storage == CUBUTTERFLY_DATA_COMPLEX_FP64,
                            "Bluestein FFT currently supports complex FP32 and FP64 storage");
                else
                    require(resolved.storage == CUBUTTERFLY_DATA_UINT64 && resolved.word_bits == 64,
                            "Bluestein NTT currently supports the uint64 storage contract");
                for (std::uint32_t axis = 0; axis < resolved.rank; ++axis) {
                    require(resolved.extents[axis] <= static_cast<std::size_t>(-1) / 2 + 1,
                            "Bluestein convolution length overflows size_t");
                    physical_extents[axis] = next_power_of_two(2 * resolved.extents[axis] - 1);
                }
            }
        }
        const bool embedding_inverse = resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING &&
                                       resolved.direction == CUBUTTERFLY_DIRECTION_INVERSE;
        const auto& input_extents = embedding_inverse ? physical_extents : resolved.extents;
        const auto& output_extents = resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING && !embedding_inverse
                                         ? physical_extents : resolved.extents;
        resolve_default_strides(resolved, input_extents, output_extents);
        const std::uint32_t log_n = resolved.rank == 1 ? integer_log2(physical_extents[0]) : 0;
        auto created = std::make_unique<cubutterflyPlan>();
        created->descriptor = resolved;
        created->physical_extents = physical_extents;
        created->element_bytes = element_bytes(resolved.storage);
        created->input_bytes = checked_product(strided_elements(resolved, true, input_extents), created->element_bytes);
        created->output_bytes = checked_product(strided_elements(resolved, false, output_extents), created->element_bytes);
        created->selection_source = resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT
                                        ? CUBUTTERFLY_SELECTION_EXPLICIT
                                        : CUBUTTERFLY_SELECTION_STATIC_MODEL;
        const bool requested_xd = resolved.execution_mode != CUBUTTERFLY_EXECUTION_COMPOSITE &&
                                  resolved.rank >= 2 && resolved.rank <= cuntt::detail::kGeneralMaxRank;
        // Rank-three composite execution uses the same generic layout schedule.
        // A resident or compact-spectrum candidate is selected independently
        // below; using this schedule alone does not imply on-chip residency.
        created->xd = resolved.rank >= 3 && !standard_arbitrary;
        created->composite = resolved.rank >= 2 || resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
        prepare_xd_permutation_schedule(*created);
        created->direct_output_boundary = !created->xd && created->composite && output_extents == physical_extents &&
                                          packed_side(resolved, false, output_extents);
        const std::string cache_key = workload_key(resolved);
        const std::string cached_algorithm = resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT
                                                 ? lookup_cache(handle->cache_path, cache_key)
                                                 : std::string{};
        nlohmann::json xd_replay;
        if (resolved.rank >= 2 && resolved.rank <= cuntt::detail::kGeneralMaxRank && resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT &&
            resolved.explicit_algorithm.rfind("xd:", 0) == 0) {
            const auto lowering = resolved.explicit_algorithm.substr(3);
            require(!lowering.empty(), "explicit xD lowering must be nonempty");
            *plan = select_xd_plan(handle, resolved, xd_replay, lowering).release();
            return;
        }
        if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT &&
            !resolved.explicit_algorithm.empty() && resolved.explicit_algorithm.front() == '{') {
            auto mapping = nlohmann::json::parse(resolved.explicit_algorithm);
            if (mapping.value("kind", "") == "xd") xd_replay = std::move(mapping);
        }
        if ((requested_xd && resolved.algorithm_policy != CUBUTTERFLY_ALGORITHM_EXPLICIT) || !xd_replay.is_null()) {
            *plan = select_xd_plan(handle, resolved, xd_replay).release();
            return;
        }
        if (requested_xd && !created->xd) {
            created->reason = "xD lowering unavailable for this layout or length; using the compatible composite path";
        }
        if (standard_arbitrary) {
            if (resolved.rank > 2) throw unsupported_error("rank-three Bluestein composition is not implemented");
            require(resolved.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE,
                    "Bluestein FFT currently requires out-of-place execution");
            const bool ntt_bluestein = resolved.op == CUBUTTERFLY_OPERATOR_NTT;
            const bool external_bluestein = resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT &&
                resolved.explicit_algorithm == "bluestein-cufft-power2-core";
            const char* base_algorithm = ntt_bluestein ? "bluestein-ntt-power2-core" :
                (external_bluestein ? "bluestein-cufft-power2-core" : "bluestein-cubutterfly-power2-core");
            bool direct_fft = false;
            std::string arbitrary_selection_reason;
            cubutterflySelectionSource_t arbitrary_selection_source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
            if (!ntt_bluestein) {
                if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
                    direct_fft = resolved.explicit_algorithm == "cufft-direct";
                    require(!direct_fft || packed_layout(resolved),
                            "cufft-direct requires packed contiguous batches");
                    arbitrary_selection_source = CUBUTTERFLY_SELECTION_EXPLICIT;
                } else if (!cached_algorithm.empty()) {
                    require(cached_algorithm == "cufft-direct" || cached_algorithm == base_algorithm,
                            "cached arbitrary FFT algorithm is not recognized");
                    direct_fft = cached_algorithm == "cufft-direct";
                    require(!direct_fft || packed_layout(resolved),
                            "cached cufft-direct selection requires packed contiguous batches");
                    arbitrary_selection_source = CUBUTTERFLY_SELECTION_USER_CACHE;
                    arbitrary_selection_reason = "matched the exact arbitrary FFT workload in the user cache";
                } else if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE) {
                    const double bluestein_ms = measure_bluestein_fft_candidate(
                        resolved, physical_extents, created->element_bytes, handle->stream);
                    const std::string selected_id = base_algorithm;
                    store_cache(handle->cache_path, cache_key, selected_id);
                    arbitrary_selection_source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
                    arbitrary_selection_reason = "measured internal Bluestein composition; kernel_ms=" + std::to_string(bluestein_ms);
                    log_event(handle, CUBUTTERFLY_LOG_INFO, "measure-complete",
                              arbitrary_selection_reason + "; algorithm=" + selected_id);
                } else {
                    arbitrary_selection_reason = "Bluestein composition uses the common internal power-of-two planner";
                }
                if (direct_fft) {
                    created->kind = cubutterflyPlan::Kind::DirectFft;
                    created->direct_fft = std::make_unique<cuntt::detail::DirectFftPlan>(
                        resolved.extents, resolved.batch,
                        resolved.direction == CUBUTTERFLY_DIRECTION_INVERSE,
                        resolved.normalize_inverse,
                        resolved.storage == CUBUTTERFLY_DATA_COMPLEX_FP64);
                    created->physical_extents = resolved.extents;
                    created->workspace_bytes = 0;
                    created->algorithm = "cufft-direct";
                    created->reason = arbitrary_selection_reason.empty()
                                          ? "selected the vendor native exact-length FFT candidate; Bluestein remains available explicitly"
                                          : arbitrary_selection_reason;
                    created->selection_source = arbitrary_selection_source;
                    log_event(handle, CUBUTTERFLY_LOG_INFO, "algorithm-selection",
                              created->reason + "; algorithm=" + created->algorithm);
                    *plan = created.release();
                    return;
                }
            }
            auto core_descriptor = resolved;
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
                if (!composition_axis(resolved,0).empty()) {
                    // Inner forward/inverse mappings are replayed independently.
                } else if (resolved.explicit_algorithm == base_algorithm) {
                    core_descriptor.algorithm_policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
                    core_descriptor.explicit_algorithm.clear();
                } else if (ntt_bluestein && resolved.explicit_algorithm.rfind("bluestein-", 0) == 0) {
                    core_descriptor.explicit_algorithm = resolved.explicit_algorithm.substr(10);
                } else {
                    throw std::invalid_argument("unknown explicit arbitrary-length algorithm");
                }
            }
            std::vector<NttAxisSelection> ntt_core_selections;
            if (ntt_bluestein) {
                created->selection_source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
                if (resolved.rank == 1) {
                    ntt_core_selections.push_back(select_ntt_axis(
                        handle, core_descriptor, 0, physical_extents[0], resolved.batch));
                } else {
                    ntt_core_selections.push_back(select_ntt_axis(
                        handle, core_descriptor, 1, physical_extents[1],
                        resolved.batch * resolved.extents[0]));
                    ntt_core_selections.push_back(select_ntt_axis(
                        handle, core_descriptor, 0, physical_extents[0],
                        resolved.batch * resolved.extents[1]));
                }
                for (const auto& selection : ntt_core_selections)
                    created->selection_source = combined_source(created->selection_source, selection.source);
            }
            created->kind = ntt_bluestein ? cubutterflyPlan::Kind::BluesteinNtt
                                          : cubutterflyPlan::Kind::BluesteinFft;
            created->algorithm = base_algorithm;
            if (ntt_bluestein) {
                std::ostringstream algorithm;
                algorithm << "bluestein-ntt[";
                for (std::size_t axis = 0; axis < ntt_core_selections.size(); ++axis)
                    algorithm << (axis == 0 ? "" : ",") << ntt_core_selections[axis].algorithm;
                algorithm << ']';
                created->algorithm = algorithm.str();
            }
            created->reason = ntt_bluestein
                ? "non-power-of-two standard NTT selected Bluestein after validating target and convolution roots"
                : "non-power-of-two standard FFT selected Bluestein with a power-of-two convolution core";
            if (!ntt_bluestein && !arbitrary_selection_reason.empty())
                created->reason += "; " + arbitrary_selection_reason;
            if (!ntt_bluestein)
                created->selection_source = arbitrary_selection_source;
            const bool inverse = resolved.direction == CUBUTTERFLY_DIRECTION_INVERSE;
            const bool fp64 = resolved.storage == CUBUTTERFLY_DATA_COMPLEX_FP64;
            if (resolved.rank == 1) {
                if (ntt_bluestein) {
                    created->bluestein_ntt_axes.push_back(std::make_unique<cuntt::detail::BluesteinNttPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch, resolved.modulus, inverse,
                        ntt_core_selections[0].config.backend,
                        ntt_core_selections[0].config.compute_unit, &ntt_core_selections[0].config, composition_axis(resolved,0)));
                    created->workspace_bytes = created->bluestein_ntt_axes[0]->workspace_size();
                } else {
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch,
                        inverse, resolved.normalize_inverse, fp64, external_bluestein, composition_axis(resolved,0)));
                    created->workspace_bytes = created->bluestein_axes[0]->workspace_size();
                }
            } else {
                require(resolved.extents[0] <= static_cast<std::size_t>(-1) / resolved.extents[1],
                        "rank-2 point count overflows size_t");
                const std::size_t matrix_points = resolved.extents[0] * resolved.extents[1];
                require(resolved.batch <= static_cast<std::size_t>(-1) / matrix_points,
                        "rank-2 batch point count overflows size_t");
                const std::size_t points = matrix_points * resolved.batch;
                require(points <= static_cast<std::size_t>(-1) / created->element_bytes,
                        "rank-2 packed byte count overflows size_t");
                created->packed_bytes = points * created->element_bytes;
                if (ntt_bluestein) {
                    created->bluestein_ntt_axes.push_back(std::make_unique<cuntt::detail::BluesteinNttPlan>(
                        resolved.extents[1], physical_extents[1], resolved.batch * resolved.extents[0],
                        resolved.modulus, inverse, ntt_core_selections[0].config.backend,
                        ntt_core_selections[0].config.compute_unit, &ntt_core_selections[0].config, composition_axis(resolved,1)));
                    created->bluestein_ntt_axes.push_back(std::make_unique<cuntt::detail::BluesteinNttPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch * resolved.extents[1],
                        resolved.modulus, inverse, ntt_core_selections[1].config.backend,
                        ntt_core_selections[1].config.compute_unit, &ntt_core_selections[1].config, composition_axis(resolved,0)));
                } else {
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[1], physical_extents[1], resolved.batch * resolved.extents[0],
                        inverse, resolved.normalize_inverse, fp64, external_bluestein, composition_axis(resolved,1)));
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch * resolved.extents[1],
                        inverse, resolved.normalize_inverse, fp64, external_bluestein, composition_axis(resolved,0)));
                }
                const auto first_work = ntt_bluestein ? created->bluestein_ntt_axes[0]->workspace_size()
                                                       : created->bluestein_axes[0]->workspace_size();
                const auto second_work = ntt_bluestein ? created->bluestein_ntt_axes[1]->workspace_size()
                                                        : created->bluestein_axes[1]->workspace_size();
                created->workspace_bytes = 2 * align_16(created->packed_bytes) +
                    std::max(align_16(first_work), align_16(second_work));
                if (ntt_bluestein)
                    created->algorithm = "rank2-transpose-" + created->algorithm;
                else
                    created->algorithm = std::string("rank2-transpose-") + base_algorithm;
            }
            log_event(handle, CUBUTTERFLY_LOG_INFO, "algorithm-selection",
                      created->reason + "; algorithm=" + created->algorithm);
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT)
                log_profile_miss(handle, resolved, created->reason, created->algorithm);
            *plan = created.release();
            return;
        }
        if (created->composite) {
            // A packed rank-two input is already in the first consumer's
            // layout. Let that axis read it directly and write into scratch.
            // This is independent of the arithmetic core and preserves the
            // caller's input. Explicit legacy mappings retain their pack node.
            const bool direct_input_eligible = resolved.rank == 2 &&
                resolved.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
                packed_side(resolved, true, physical_extents);
            created->direct_input_boundary = direct_input_eligible;
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT &&
                !resolved.explicit_algorithm.empty() && resolved.explicit_algorithm.front() == '{') {
                const auto replay = nlohmann::json::parse(resolved.explicit_algorithm);
                if (replay.value("kind", "") == "composition") {
                    created->direct_input_boundary = replay.value("direct_input_boundary", false);
                    require(!created->direct_input_boundary || direct_input_eligible,
                            "direct input boundary requires a packed standard rank-two input");
                }
            }
            std::size_t packed_points = resolved.batch;
            for (const auto extent : physical_extents) {
                require(packed_points <= static_cast<std::size_t>(-1) / extent, "packed point count overflows size_t");
                packed_points *= extent;
            }
            require(packed_points <= static_cast<std::size_t>(-1) / created->element_bytes, "packed byte count overflows size_t");
            created->packed_bytes = packed_points * created->element_bytes;
            created->kind = resolved.op == CUBUTTERFLY_OPERATOR_NTT ? cubutterflyPlan::Kind::Ntt : cubutterflyPlan::Kind::Butterfly;
            std::vector<std::string> axis_algorithms;
            std::vector<std::string> axis_reasons;
            created->selection_source = CUBUTTERFLY_SELECTION_STATIC_MODEL;
            auto add_ntt_axis = [&](std::uint32_t source_axis, std::size_t length, std::size_t batch) {
                auto selected = select_ntt_axis(handle, resolved, source_axis, length, batch);
                created->selection_source = combined_source(created->selection_source, selected.source);
                axis_algorithms.push_back(selected.algorithm);
                axis_reasons.push_back(selected.reason);
                created->ntt_axes.push_back(std::make_unique<cuntt::Plan>(selected.config));
                created->ntt_axes.back()->set_stream(handle->stream);
            };
            auto add_butterfly_axis = [&](std::uint32_t source_axis, std::size_t length, std::size_t batch) {
                auto selected = select_butterfly_axis(handle, resolved, source_axis, length, batch,
                    created->direct_input_boundary && created->butterfly_axes.empty());
                created->selection_source = combined_source(created->selection_source, selected.source);
                axis_algorithms.push_back(selected.algorithm);
                axis_reasons.push_back(selected.reason);
                created->butterfly_axes.push_back(std::make_unique<cuntt::ButterflyPlan>(selected.config));
                created->butterfly_axes.back()->set_stream(handle->stream);
            };
            if (resolved.rank == 1) {
                if (created->kind == cubutterflyPlan::Kind::Ntt)
                    add_ntt_axis(0, physical_extents[0], resolved.batch);
                else
                    add_butterfly_axis(0, physical_extents[0], resolved.batch);
            } else {
                // xD keeps one axis plan per logical dimension.  Processing
                // the innermost axis first gives every plan a contiguous
                // batch and makes the permutation schedule explicit.
                for (std::uint32_t source_axis = resolved.rank; source_axis-- > 0;) {
                    std::size_t transforms = resolved.batch;
                    for (std::uint32_t axis = 0; axis < resolved.rank; ++axis) {
                        if (axis == source_axis)
                            continue;
                        require(transforms <= static_cast<std::size_t>(-1) / physical_extents[axis],
                                "xD transform count overflows size_t");
                        transforms *= physical_extents[axis];
                    }
                    if (created->kind == cubutterflyPlan::Kind::Ntt)
                        add_ntt_axis(source_axis, physical_extents[source_axis], transforms);
                    else
                        add_butterfly_axis(source_axis, physical_extents[source_axis], transforms);
                }
            }
            created->fused_input_boundary = resolved.rank == 1 &&
                                             created->direct_output_boundary &&
                                             resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING &&
                                             resolved.direction == CUBUTTERFLY_DIRECTION_FORWARD &&
                                             created->kind == cubutterflyPlan::Kind::Butterfly &&
                                             resolved.storage == CUBUTTERFLY_DATA_FP32 &&
                                             resolved.op == CUBUTTERFLY_OPERATOR_FWHT &&
                                             created->butterfly_axes[0]->config().local_exchange ==
                                                 cuntt::LocalExchange::WarpRegister;
            created->composite_buffers = created->xd || resolved.rank == 2
                                             ? 2
                                             : (created->kind == cubutterflyPlan::Kind::Ntt
                                                    ? (created->direct_output_boundary ? 1 : 2)
                                                    : (created->direct_output_boundary ? 0 : 1));
            created->workspace_bytes = align_16(created->packed_bytes) * created->composite_buffers;
            if (created->kind == cubutterflyPlan::Kind::Ntt) {
                for (const auto& axis_plan : created->ntt_axes)
                    created->workspace_bytes += align_16(axis_plan->workspace_size());
            } else {
                for (const auto& axis_plan : created->butterfly_axes)
                    created->workspace_bytes += align_16(axis_plan->workspace_size());
            }
            std::ostringstream algorithm;
            algorithm << (created->xd ? "rankN-composite" : (resolved.rank == 2 ? "rank2-transpose" : "zero-extended")) << '[';
            for (std::size_t axis = 0; axis < axis_algorithms.size(); ++axis)
                algorithm << (axis == 0 ? "" : ",") << axis_algorithms[axis];
            algorithm << ']';
            if (created->direct_input_boundary)
                algorithm << "+direct-input";
            if (created->direct_output_boundary)
                algorithm << "+direct-output";
            if (created->fused_input_boundary)
                algorithm << "+fused-input";
            created->algorithm = algorithm.str();
            std::ostringstream reason;
            for (std::size_t axis = 0; axis < axis_reasons.size(); ++axis)
                reason << (axis == 0 ? "" : "; ") << "axis" << axis << ": " << axis_reasons[axis];
            if (requested_xd && !created->xd)
                reason << (axis_reasons.empty() ? "" : "; ")
                       << "xD fallback: layout or length contract is not eligible";
            created->reason = reason.str();
            log_event(handle, CUBUTTERFLY_LOG_INFO, "composite-selection",
                      created->reason + "; algorithm=" + created->algorithm);
            if (created->selection_source == CUBUTTERFLY_SELECTION_STATIC_MODEL)
                log_profile_miss(handle, resolved, created->reason, created->algorithm);
            *plan = created.release();
            return;
        }
        if (resolved.op == CUBUTTERFLY_OPERATOR_NTT) {
            require(resolved.input_strides[0] == 1 && resolved.output_strides[0] == 1 &&
                        resolved.input_batch_stride == resolved.extents[0] &&
                        resolved.output_batch_stride == resolved.extents[0],
                    "rank-1 NTT direct execution requires packed contiguous batches");
            cuntt::PlanConfig config;
            config.log_n = log_n;
            config.batch = resolved.batch;
            config.modulus = resolved.modulus;
            config.inverse = resolved.direction == CUBUTTERFLY_DIRECTION_INVERSE;
            config.word_bits = resolved.word_bits;
            config.input_order = resolved.input_ntt_layout ==
                                         CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC
                                     ? cuntt::InputOrder::ApptStatic
                                     : cuntt::InputOrder::Natural;
            config.output_order = resolved.output_ntt_layout ==
                                          CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC
                                      ? cuntt::OutputOrder::ApptStatic
                                      : resolved.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED
                                      ? cuntt::OutputOrder::BitReversed
                                      : cuntt::OutputOrder::Natural;
            config.auto_allocate_workspace = false;
            config.auto_select = false;
            if (appt_layout_requested) {
                config.auto_select = false;
                apply_ntt_algorithm(
                    config,
                    resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT
                        ? resolved.explicit_algorithm
                        : "ntt-appt-register-tail");
                if (ntt_algorithm_id(config).rfind("ntt-appt-register-tail", 0) != 0)
                    throw unsupported_error("APPT static layout requires an APPT register-tail core");
            } else if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
                apply_ntt_algorithm(config, resolved.explicit_algorithm);
            } else if (!cached_algorithm.empty()) {
                apply_ntt_algorithm(config, cached_algorithm);
                created->selection_source = CUBUTTERFLY_SELECTION_USER_CACHE;
            } else if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE) {
                const auto tuned = tune_ntt(config, handle->stream);
                apply_ntt_algorithm(config, tuned.first);
                created->selection_source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
                created->reason = "measured legal NTT candidates on the current hardware; kernel_ms=" +
                                  std::to_string(tuned.second);
                store_cache(handle->cache_path, cache_key, tuned.first);
                log_event(handle, CUBUTTERFLY_LOG_INFO, "measure-complete",
                          created->reason + "; algorithm=" + tuned.first);
            } else {
                const auto choice = cuntt::detail::select_ntt_mapping(config);
                config = choice.config;
                created->selection_source = mapping_selection_source(choice.info);
                created->reason = choice.info.reason;
            }
            created->kind = cubutterflyPlan::Kind::Ntt;
            if (created->reason.empty())
                created->reason = created->selection_source == CUBUTTERFLY_SELECTION_USER_CACHE
                                      ? "matched an exact workload key in the user selection cache"
                                      : "explicit execution mapping";
            created->ntt = std::make_unique<cuntt::Plan>(config);
            created->ntt->set_stream(handle->stream);
            created->algorithm = ntt_algorithm_id(created->ntt->config());
            created->workspace_bytes = created->ntt->workspace_size();
        } else {
            cuntt::ButterflyConfig config;
            config.op = butterfly_operator(resolved.op);
            config.precision = butterfly_precision(resolved);
            const bool low_precision = config.precision == cuntt::ButterflyPrecision::Fp16 ||
                                       config.precision == cuntt::ButterflyPrecision::Bf16;
            config.accumulation = low_precision && resolved.compute == CUBUTTERFLY_COMPUTE_FP32
                                      ? cuntt::ButterflyAccumulation::Fp32
                                      : cuntt::ButterflyAccumulation::Native;
            config.placement = resolved.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE ? cuntt::ButterflyPlacement::InPlace : cuntt::ButterflyPlacement::OutOfPlace;
            config.log_n = log_n;
            config.batch = resolved.batch;
            config.element_stride = resolved.input_strides[0];
            config.batch_stride = resolved.input_batch_stride;
            require(resolved.input_strides == resolved.output_strides && resolved.input_batch_stride == resolved.output_batch_stride,
                    "rank-1 common plans currently require matching input and output layouts");
            config.inverse = resolved.direction == CUBUTTERFLY_DIRECTION_INVERSE;
            config.normalize_inverse = resolved.normalize_inverse;
            config.output_scale = resolved.output_scale;
            config.final_storage_rounding = resolved.output_rounding == CUBUTTERFLY_ROUNDING_FINAL_STORAGE;
            config.auto_allocate_workspace = false;
            config.auto_select = false;
            if (config.op == cuntt::ButterflyOperator::Structured2x2) {
                require(!resolved.stage_matrices[0].empty(), "structured-2x2 requires stage matrices");
                for (const auto& matrix : resolved.stage_matrices[0])
                    config.stage_matrices.push_back({matrix.m00, matrix.m01, matrix.m10, matrix.m11});
            }
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
                apply_butterfly_algorithm(config, resolved.explicit_algorithm);
            } else if (!cached_algorithm.empty()) {
                apply_butterfly_algorithm(config, cached_algorithm);
                created->selection_source = CUBUTTERFLY_SELECTION_USER_CACHE;
            } else if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE) {
                const auto tuned = tune_butterfly(config, resolved.storage, handle->stream);
                apply_butterfly_algorithm(config, tuned.first);
                created->selection_source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
                created->reason = "measured legal butterfly candidates on the current hardware; kernel_ms=" +
                                  std::to_string(tuned.second);
                store_cache(handle->cache_path, cache_key, tuned.first);
                log_event(handle, CUBUTTERFLY_LOG_INFO, "measure-complete",
                          created->reason + "; algorithm=" + tuned.first);
            } else {
                const auto info = choose_butterfly_mapping(config);
                created->selection_source = mapping_selection_source(info);
                created->reason = info.reason;
            }
            created->kind = cubutterflyPlan::Kind::Butterfly;
            if (created->reason.empty())
                created->reason = created->selection_source == CUBUTTERFLY_SELECTION_USER_CACHE
                                      ? "matched an exact workload key in the user selection cache"
                                      : "explicit execution mapping";
            created->butterfly = std::make_unique<cuntt::ButterflyPlan>(config);
            created->butterfly->set_stream(handle->stream);
            created->algorithm = std::string(cuntt::butterfly_backend_name(created->butterfly->config().backend)) +
                                 "-" + cuntt::compute_unit_name(created->butterfly->config().compute_unit);
            if (created->butterfly->config().fft_core != cuntt::FftCore::Scalar)
                created->algorithm += std::string("/") + cuntt::fft_core_name(created->butterfly->config().fft_core);
            created->workspace_bytes = created->butterfly->workspace_size();
            if (config.final_storage_rounding)
                created->reason += "; FP32 arithmetic through explicit output scale and final FP16 narrowing; "
                    "generic mappings include storage-conversion kernels and workspace";
        }
        if ((created->selection_source == CUBUTTERFLY_SELECTION_STATIC_MODEL ||
             created->selection_source == CUBUTTERFLY_SELECTION_UNMEASURED_FALLBACK) &&
            resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT && cached_algorithm.empty())
            log_profile_miss(handle, resolved, created->reason, created->algorithm);
        *plan = created.release();
    });
}

cubutterflyStatus_t cubutterflyDestroyPlan(cubutterflyPlan_t plan) { delete plan; return CUBUTTERFLY_STATUS_SUCCESS; }

cubutterflyStatus_t cubutterflyPlanGetDescriptorJson(cubutterflyPlan_t plan, char* json, size_t* bytes) {
    return protect([&] {
        require(plan != nullptr, "plan must be non-null");
        copy_text(cubutterfly::detail::descriptor_json(plan->descriptor).dump(), json, bytes);
    });
}

cubutterflyStatus_t cubutterflyPlanGetMappingJson(cubutterflyPlan_t plan, char* json, size_t* bytes) {
    return protect([&] {
        require(plan != nullptr, "plan must be non-null");
        if (!plan->xd_mapping.empty()) return copy_text(plan->xd_mapping, json, bytes);
        if (plan->butterfly || plan->ntt) {
            auto mapping = nlohmann::json::parse(plan->butterfly
                ? cuntt::detail::serialize_mapping(plan->butterfly->config())
                : cuntt::detail::serialize_mapping(plan->ntt->config()));
            mapping["contract"] = xd_contract(plan->descriptor);
            return copy_text(mapping.dump(), json, bytes);
        }
        nlohmann::json mapping{{"schema_version", 1}, {"kind", "composition"},
                              {"contract", xd_contract(plan->descriptor)},
                              {"algorithm", plan->algorithm}, {"physical_extents", plan->physical_extents},
                              {"workspace_bytes", plan->workspace_bytes},
                              {"execution_mode", static_cast<int>(plan->descriptor.execution_mode)},
                              {"nd_composite", plan->xd},
                              {"axes", nlohmann::json::array()}};
        if (plan->direct_input_boundary)
            mapping["direct_input_boundary"] = true;
        for (const auto& axis : plan->butterfly_axes)
            mapping["axes"].push_back(nlohmann::json::parse(cuntt::detail::serialize_mapping(axis->config())));
        for (const auto& axis : plan->ntt_axes)
            mapping["axes"].push_back(nlohmann::json::parse(cuntt::detail::serialize_mapping(axis->config())));
        for (const auto& axis : plan->bluestein_axes)
            mapping["axes"].push_back(nlohmann::json::parse(axis->mapping_json()));
        for (const auto& axis : plan->bluestein_ntt_axes)
            mapping["axes"].push_back(nlohmann::json::parse(axis->mapping_json()));
        copy_text(mapping.dump(), json, bytes);
    });
}

cubutterflyStatus_t cubutterflyPlanGetWorkspaceSize(cubutterflyPlan_t plan, size_t* bytes) {
    return protect([&] { require(plan != nullptr && bytes != nullptr, "plan and output must be non-null"); *bytes = plan->workspace_bytes; });
}

cubutterflyStatus_t cubutterflyPlanGetInputSize(cubutterflyPlan_t plan, size_t* bytes) { return protect([&] { require(plan != nullptr && bytes != nullptr, "plan and output must be non-null"); *bytes = plan->input_bytes; }); }
cubutterflyStatus_t cubutterflyPlanGetOutputSize(cubutterflyPlan_t plan, size_t* bytes) { return protect([&] { require(plan != nullptr && bytes != nullptr, "plan and output must be non-null"); *bytes = plan->output_bytes; }); }

cubutterflyStatus_t cubutterflyPlanGetPhysicalExtents(cubutterflyPlan_t plan, size_t* extents, uint32_t capacity) {
    return protect([&] { require(plan != nullptr && extents != nullptr, "plan and extents must be non-null"); require(capacity >= plan->physical_extents.size(), "extent capacity is too small"); std::copy(plan->physical_extents.begin(), plan->physical_extents.end(), extents); });
}

cubutterflyStatus_t cubutterflyPlanGetSelectionSource(cubutterflyPlan_t plan, cubutterflySelectionSource_t* source) { return protect([&] { require(plan != nullptr && source != nullptr, "plan and output must be non-null"); *source = plan->selection_source; }); }
cubutterflyStatus_t cubutterflyPlanGetAlgorithmName(cubutterflyPlan_t plan, char* name, size_t* bytes) { return protect([&] { require(plan != nullptr, "plan must be non-null"); copy_text(plan->algorithm, name, bytes); }); }
cubutterflyStatus_t cubutterflyPlanGetSelectionReason(cubutterflyPlan_t plan, char* reason, size_t* bytes) { return protect([&] { require(plan != nullptr, "plan must be non-null"); copy_text(plan->reason, reason, bytes); }); }

cubutterflyStatus_t cubutterflyPlanGetNttLayoutInfo(
    cubutterflyPlan_t plan, cubutterflyNttLayoutInfo_t* info) {
    if (plan == nullptr || info == nullptr) {
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    }
    if (plan->ntt == nullptr || plan->composite ||
        plan->ntt->config().subgraph_mappings.empty() ||
        (plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTail &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8 &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinalDataTime &&
         plan->ntt->config().subgraph_mappings.front().core !=
             cuntt::NttSubgraphCore::
                 ApptOnlineRegisterTailGroupedWriterFinalResident)) {
        return CUBUTTERFLY_STATUS_NOT_SUPPORTED;
    }
    return protect([&] {
        const auto layout = plan->ntt->appt_layout_info();
        require(layout.stage_partition.size() <= 3,
                "layout stage partition exceeds the C ABI capacity");
        std::memset(info, 0, sizeof(*info));
        info->log_n = layout.log_n;
        info->stage_count =
            static_cast<std::uint32_t>(layout.stage_partition.size());
        for (std::size_t index = 0; index < layout.stage_partition.size();
             ++index) {
            info->stage_partition[index] = layout.stage_partition[index];
        }
        info->fragment_width = layout.fragment_width;
        info->bank_bits = layout.bank_bits;
        info->xor_permutation = layout.xor_permutation ? 1 : 0;
        std::strncpy(info->compatibility_id, layout.compatibility_id.c_str(),
                     sizeof(info->compatibility_id) - 1);
    });
}

cubutterflyStatus_t cubutterflyPlanSetWorkspace(cubutterflyPlan_t plan, void* workspace, size_t bytes) {
    return protect([&] {
        require(plan != nullptr, "plan must be non-null");
        if (!plan->arbitrary_children.empty()) {
            require(workspace != nullptr && bytes >= plan->workspace_bytes, "arbitrary axis workspace is too small");
            require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0, "workspace must be at least 16-byte aligned");
            auto* core_work = static_cast<unsigned char*>(workspace) + 2 * align_16(plan->packed_bytes);
            for (auto& child : plan->arbitrary_children)
                if (child->workspace_bytes)
                    check_child(cubutterflyPlanSetWorkspace(child.get(), core_work, child->workspace_bytes));
            plan->workspace = workspace;
            return;
        }
        if (plan->numeric_child) {
            require(workspace != nullptr && bytes >= plan->workspace_bytes, "numeric adapter workspace is too small");
            require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0, "workspace must be at least 16-byte aligned");
            auto* base = static_cast<unsigned char*>(workspace);
            const auto offset = align_16(plan->boundary_input_bytes) + align_16(plan->boundary_output_bytes);
            check_child(cubutterflyPlanSetWorkspace(plan->numeric_child.get(),
                plan->numeric_child->workspace_bytes ? base + offset : nullptr, plan->numeric_child->workspace_bytes));
            plan->workspace = workspace;
            return;
        }
        if (plan->boundary_child) {
            require(workspace != nullptr && bytes >= plan->workspace_bytes, "semantic boundary workspace is too small");
            require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0, "workspace must be at least 16-byte aligned");
            auto* base = static_cast<unsigned char*>(workspace);
            const auto offset = align_16(plan->boundary_input_bytes) + align_16(plan->boundary_output_bytes);
            check_child(cubutterflyPlanSetWorkspace(plan->boundary_child.get(),
                plan->boundary_child->workspace_bytes ? base + offset : nullptr,
                plan->boundary_child->workspace_bytes));
            plan->workspace = workspace;
            return;
        }
        if (plan->xd_selected) {
            if (plan->workspace_bytes) {
                const auto status = cubutterflyPlanSetWorkspace(plan->xd_selected.get(), workspace, bytes);
                require(status == CUBUTTERFLY_STATUS_SUCCESS, "xD selected plan workspace is invalid");
            }
            plan->workspace = workspace;
            return;
        }
        if (plan->xd_fft) {
            plan->xd_fft->set_workspace(workspace, bytes);
            plan->workspace = workspace;
            return;
        }
        if (plan->xd_resident) {
            require(bytes >= plan->workspace_bytes, "resident workspace is too small");
            plan->workspace = workspace;
            return;
        }
        if (plan->kind == cubutterflyPlan::Kind::DirectFft) {
            require(bytes == 0, "direct FFT does not require caller workspace");
            plan->workspace = workspace;
            return;
        }
        if (plan->direct_real_fft) {
            plan->direct_real_fft->set_workspace(workspace, bytes);
            plan->workspace = workspace;
            return;
        }
        if (plan->kind == cubutterflyPlan::Kind::BluesteinFft ||
            plan->kind == cubutterflyPlan::Kind::BluesteinNtt) {
            require(workspace != nullptr, "Bluestein plans require a non-null workspace");
            require(bytes >= plan->workspace_bytes, "workspace is smaller than the Bluestein plan requirement");
            require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0,
                    "workspace must be at least 16-byte aligned");
            plan->workspace = workspace;
            return;
        }
        if (!plan->composite) {
            if (plan->kind == cubutterflyPlan::Kind::Ntt)
                plan->ntt->set_workspace(workspace, bytes);
            else
                plan->butterfly->set_workspace(workspace, bytes);
            plan->workspace = workspace;
            return;
        }
        if (plan->workspace_bytes == 0) {
            require(bytes == 0, "zero-workspace composite plan received nonzero workspace bytes");
            plan->workspace = workspace;
            return;
        }
        require(workspace != nullptr, "composite plans require a non-null workspace");
        require(bytes >= plan->workspace_bytes, "workspace is smaller than the composite plan requirement");
        require(reinterpret_cast<std::uintptr_t>(workspace) % 16 == 0, "workspace must be at least 16-byte aligned");
        plan->workspace = workspace;
        std::size_t offset = align_16(plan->packed_bytes) * plan->composite_buffers;
        auto* base = static_cast<unsigned char*>(workspace);
        if (plan->kind == cubutterflyPlan::Kind::Ntt) {
            for (auto& axis_plan : plan->ntt_axes) {
                const std::size_t required = axis_plan->workspace_size();
                if (required != 0)
                    axis_plan->set_workspace(base + offset, required);
                offset += align_16(required);
            }
        } else {
            for (auto& axis_plan : plan->butterfly_axes) {
                const std::size_t required = axis_plan->workspace_size();
                if (required != 0)
                    axis_plan->set_workspace(base + offset, required);
                offset += align_16(required);
            }
        }
    });
}

cubutterflyStatus_t cubutterflyExecute(cubutterflyHandle_t handle, cubutterflyPlan_t plan, const void* input, void* output) {
    return protect([&] {
        require(handle != nullptr && plan != nullptr && input != nullptr && output != nullptr, "execution arguments must be non-null");
        bind_plan_stream_if_needed(plan, handle->stream);
        const bool in_place = plan->descriptor.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE;
        require(in_place ? input == output : input != output,
                in_place ? "in-place execution requires identical pointers" : "out-of-place execution requires distinct pointers");
        if (!plan->arbitrary_children.empty()) {
            require(plan->workspace != nullptr, "arbitrary axis workspace is not set");
            void* current = plan->workspace;
            void* alternate = static_cast<unsigned char*>(plan->workspace) + align_16(plan->packed_bytes);
            if (cudaMemcpyAsync(current, input, plan->input_bytes, cudaMemcpyDeviceToDevice, handle->stream) != cudaSuccess)
                throw std::runtime_error("arbitrary axis input copy failed");
            const auto& d = plan->descriptor;
            std::vector<std::uint32_t> order(d.rank);
            for (std::uint32_t axis = 0; axis < d.rank; ++axis) order[axis] = axis;
            for (std::size_t step = 0; step < d.rank; ++step) {
                const auto axis = d.rank - 1 - static_cast<std::uint32_t>(step);
                auto target = order;
                target.erase(std::find(target.begin(), target.end(), axis));
                target.push_back(axis);
                if (target != order) {
                    std::vector<std::uint32_t> permutation(d.rank);
                    for (std::uint32_t j = 0; j < d.rank; ++j)
                        permutation[j] = std::find(order.begin(), order.end(), target[j]) - order.begin();
                    const auto layout = packed_nd_layout_in_order(d.extents, order, d.batch);
                    cuntt::detail::launch_permute_nd(current, alternate, layout, permutation.data(),
                                                    plan->element_bytes, handle->stream);
                    std::swap(current, alternate);
                    order = std::move(target);
                }
                check_child(cubutterflyExecute(handle, plan->arbitrary_children[step].get(), current, alternate));
                std::swap(current, alternate);
            }
            std::vector<std::uint32_t> permutation(d.rank);
            for (std::uint32_t j = 0; j < d.rank; ++j)
                permutation[j] = std::find(order.begin(), order.end(), j) - order.begin();
            const auto layout = packed_nd_layout_in_order(d.extents, order, d.batch);
            cuntt::detail::launch_permute_nd(current, output, layout, permutation.data(),
                                            plan->element_bytes, handle->stream);
            return;
        }
        if (plan->numeric_child) {
            require(plan->workspace != nullptr, "numeric adapter workspace is not set");
            auto* first = static_cast<unsigned char*>(plan->workspace);
            auto* second = first + align_16(plan->boundary_input_bytes);
            const auto& d = plan->descriptor;
            if (plan->full_spectrum_real) {
                const auto layout = packed_nd_layout(d.extents, d.batch);
                const bool bf16 = d.storage == CUBUTTERFLY_DATA_BF16;
                if (d.fft_type == CUBUTTERFLY_FFT_R2C)
                    cuntt::detail::launch_low_real_to_complex(input, first, layout, bf16, handle->stream);
                else cuntt::detail::launch_low_expand_half_spectrum(input, first, layout, bf16, handle->stream);
                check_child(cubutterflyExecute(handle, plan->numeric_child.get(), first, second));
                if (d.fft_type == CUBUTTERFLY_FFT_R2C)
                    cuntt::detail::launch_low_extract_half_spectrum(second, output, layout, bf16, handle->stream);
                else cuntt::detail::launch_low_complex_to_real(second, output, layout, bf16, handle->stream);
                cuntt::detail::launch_low_output_scale(output, plan->output_bytes / sizeof(std::uint16_t),
                    d.fft_output_scale, bf16, handle->stream, d.compute == CUBUTTERFLY_COMPUTE_FP32);
            } else {
                cuntt::detail::launch_fp32_to_fp64(input, first, plan->input_bytes / sizeof(float), handle->stream);
                check_child(cubutterflyExecute(handle, plan->numeric_child.get(), first, second));
                cuntt::detail::launch_fp64_to_fp32(second, output, plan->output_bytes / sizeof(float), handle->stream);
            }
            return;
        }
        if (plan->boundary_child) {
            require(plan->workspace != nullptr, "semantic boundary workspace is not set");
            auto* first = static_cast<unsigned char*>(plan->workspace);
            auto* second = first + align_16(plan->boundary_input_bytes);
            const auto& d = plan->descriptor;
            auto in = packed_nd_layout(cubutterfly::detail::semantic_input_extents(d), d.batch);
            auto out = packed_nd_layout(cubutterfly::detail::semantic_output_extents(d), d.batch);
            std::uint32_t in_orders[CUBUTTERFLY_MAX_EXECUTION_RANK]{};
            std::uint32_t out_orders[CUBUTTERFLY_MAX_EXECUTION_RANK]{};
            in.batch_stride = d.input_batch_stride;
            out.batch_stride = d.output_batch_stride;
            for (std::uint32_t axis = 0; axis < d.rank; ++axis) {
                in.strides[axis] = d.input_strides[axis]; out.strides[axis] = d.output_strides[axis];
                in_orders[axis] = static_cast<std::uint32_t>(d.input_orders[axis]);
                out_orders[axis] = static_cast<std::uint32_t>(d.output_orders[axis]);
            }
            cuntt::detail::launch_gather_ordered_nd(input, first, in, in_orders,
                boundary_element_bytes(d, true), handle->stream);
            check_child(cubutterflyExecute(handle, plan->boundary_child.get(), first, second));
            cuntt::detail::launch_scatter_ordered_nd(second, output, out, out_orders,
                boundary_element_bytes(d, false), handle->stream);
            return;
        }
        if (plan->xd_selected) {
            const auto status = cubutterflyExecute(handle, plan->xd_selected.get(), input, output);
            if (status != CUBUTTERFLY_STATUS_SUCCESS)
                throw std::runtime_error(std::string("xD child execute: ") + cubutterflyGetStatusString(status));
            return;
        }
        if (plan->direct_real_fft) {
            plan->direct_real_fft->execute(input, output, handle->stream);
            return;
        }
        if (plan->xd_fft) { plan->xd_fft->execute(input, output, handle->stream); return; }
        if (plan->xd_resident) { plan->xd_resident->execute(input, output, handle->stream); return; }
        if (plan->kind == cubutterflyPlan::Kind::DirectFft) {
            plan->direct_fft->execute(input, output, handle->stream);
            return;
        }
        if (plan->kind == cubutterflyPlan::Kind::BluesteinFft ||
            plan->kind == cubutterflyPlan::Kind::BluesteinNtt) {
            require(plan->workspace != nullptr, "Bluestein plan requires a bound workspace");
            if (plan->descriptor.rank == 1) {
                if (plan->kind == cubutterflyPlan::Kind::BluesteinNtt)
                    plan->bluestein_ntt_axes[0]->execute(static_cast<const std::uint64_t*>(input),
                                                         static_cast<std::uint64_t*>(output), plan->workspace,
                                                         plan->descriptor.input_strides[0],
                                                         plan->descriptor.input_batch_stride,
                                                         plan->descriptor.output_strides[0],
                                                         plan->descriptor.output_batch_stride,
                                                         handle->stream);
                else
                    plan->bluestein_axes[0]->execute(input, output, plan->workspace,
                                                     plan->descriptor.input_strides[0],
                                                     plan->descriptor.input_batch_stride,
                                                     plan->descriptor.output_strides[0],
                                                     plan->descriptor.output_batch_stride,
                                                     handle->stream);
            } else {
                auto* first = static_cast<unsigned char*>(plan->workspace);
                auto* second = first + align_16(plan->packed_bytes);
                auto* work = second + align_16(plan->packed_bytes);
                const std::size_t rows = plan->descriptor.extents[0];
                const std::size_t cols = plan->descriptor.extents[1];
                cuntt::detail::launch_pack_2d(input, first, rows, cols, rows, cols,
                                              plan->descriptor.batch,
                                              plan->descriptor.input_strides[0],
                                              plan->descriptor.input_strides[1],
                                              plan->descriptor.input_batch_stride,
                                              plan->element_bytes, handle->stream);
                if (plan->kind == cubutterflyPlan::Kind::BluesteinNtt)
                    plan->bluestein_ntt_axes[0]->execute(reinterpret_cast<const std::uint64_t*>(first),
                                                         reinterpret_cast<std::uint64_t*>(second), work,
                                                         1, cols, 1, cols, handle->stream);
                else
                    plan->bluestein_axes[0]->execute(first, second, work, 1, cols, 1, cols, handle->stream);
                cuntt::detail::launch_transpose_2d(second, first, rows, cols,
                                                   plan->descriptor.batch, plan->element_bytes, handle->stream);
                if (plan->kind == cubutterflyPlan::Kind::BluesteinNtt)
                    plan->bluestein_ntt_axes[1]->execute(reinterpret_cast<const std::uint64_t*>(first),
                                                         reinterpret_cast<std::uint64_t*>(second), work,
                                                         1, rows, 1, rows, handle->stream);
                else
                    plan->bluestein_axes[1]->execute(first, second, work, 1, rows, 1, rows, handle->stream);
                cuntt::detail::launch_transpose_2d(second, first, cols, rows,
                                                   plan->descriptor.batch, plan->element_bytes, handle->stream);
                cuntt::detail::launch_scatter_2d(first, output, rows, cols, rows, cols,
                                                 plan->descriptor.batch,
                                                 plan->descriptor.output_strides[0],
                                                 plan->descriptor.output_strides[1],
                                                 plan->descriptor.output_batch_stride,
                                                 plan->element_bytes, handle->stream);
            }
            return;
        }
        if (plan->composite) {
            require(plan->workspace_bytes == 0 || plan->workspace != nullptr,
                    "composite plan requires a bound workspace");
            auto* first = static_cast<unsigned char*>(plan->workspace);
            auto* second = plan->composite_buffers >= 2
                               ? first + align_16(plan->packed_bytes)
                               : first;
            if (plan->xd) {
                require(plan->descriptor.rank >= 2 && plan->descriptor.rank <= cuntt::detail::kGeneralMaxRank,
                        "generic xD execution requires rank two through eight");
                const auto& extents = plan->physical_extents;
                const auto original_layout = packed_nd_layout(extents, plan->descriptor.batch);
                const auto& d = plan->descriptor;
                const bool embedding = d.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
                const bool inverse = d.direction == CUBUTTERFLY_DIRECTION_INVERSE;
                auto logical_layout = packed_nd_layout(embedding && inverse ? extents : d.extents, d.batch);
                auto output_layout = packed_nd_layout(embedding && !inverse ? extents : d.extents, d.batch);
                logical_layout.batch_stride = d.input_batch_stride;
                output_layout.batch_stride = d.output_batch_stride;
                for (std::uint32_t axis = 0; axis < d.rank; ++axis) {
                    logical_layout.strides[axis] = d.input_strides[axis];
                    output_layout.strides[axis] = d.output_strides[axis];
                }
                cuntt::detail::GeneralNdLayout physical_layout = original_layout;
                cuntt::detail::launch_pack_nd(input, first, logical_layout, physical_layout,
                                               plan->element_bytes, handle->stream);
                void* current = first;
                void* alternate = second;
                for (std::size_t axis_plan_index = 0;
                     axis_plan_index < plan->descriptor.rank; ++axis_plan_index) {
                    const auto& schedule = plan->xd_permutation_steps[axis_plan_index];
                    if (schedule.required) {
                        cuntt::detail::launch_permute_nd(current, alternate, schedule.input_layout,
                                                         schedule.permutation.data(), plan->element_bytes,
                                                         handle->stream);
                        std::swap(current, alternate);
                    }
                    if (plan->kind == cubutterflyPlan::Kind::Ntt) {
                        auto& axis_plan = plan->ntt_axes[axis_plan_index];
                        void* axis_output = alternate;
                        if (plan->descriptor.storage == CUBUTTERFLY_DATA_UINT32)
                            axis_plan->execute_async(reinterpret_cast<const std::uint32_t*>(current),
                                                     reinterpret_cast<std::uint32_t*>(axis_output));
                        else
                            axis_plan->execute_async(reinterpret_cast<const std::uint64_t*>(current),
                                                     reinterpret_cast<std::uint64_t*>(axis_output));
                        std::swap(current, alternate);
                    } else {
                        auto& axis_plan = plan->butterfly_axes[axis_plan_index];
                        execute_butterfly(*axis_plan, current, current, plan->descriptor.storage);
                    }
                }
                void* final_packed = current;
                if (plan->xd_final_permutation_required) {
                    cuntt::detail::launch_permute_nd(current, alternate, plan->xd_final_input_layout,
                                                     plan->xd_final_permutation.data(), plan->element_bytes,
                                                     handle->stream);
                    final_packed = alternate;
                }
                cuntt::detail::launch_scatter_nd(final_packed, output, physical_layout,
                                                 output_layout, plan->element_bytes,
                                                 handle->stream);
                return;
            }
            // Internal adapters may place an already-packed contiguous input
            // in the first composite buffer.  That buffer is intentionally
            // owned by the plan, so the first 2-D pack is redundant and can be
            // skipped.  Public callers still take the regular pack path.
            const bool input_is_first_workspace = plan->descriptor.rank == 2 &&
                                                   plan->workspace != nullptr &&
                                                   input == plan->workspace;
            const bool inverse_embedding = plan->descriptor.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING &&
                                           plan->descriptor.direction == CUBUTTERFLY_DIRECTION_INVERSE;
            const auto& logical = plan->descriptor.extents;
            const auto& physical = plan->physical_extents;
            const bool direct_butterfly_1d = plan->direct_output_boundary &&
                                              plan->descriptor.rank == 1 &&
                                              plan->kind == cubutterflyPlan::Kind::Butterfly;
            if (plan->descriptor.rank == 1) {
                const std::size_t input_n = inverse_embedding ? physical[0] : logical[0];
                if (!plan->fused_input_boundary) {
                    void* pack_destination = direct_butterfly_1d ? output : first;
                    cuntt::detail::launch_pack_1d(input, pack_destination, input_n, physical[0], plan->descriptor.batch,
                                                  plan->descriptor.input_strides[0], plan->descriptor.input_batch_stride,
                                                  plan->element_bytes, handle->stream);
                }
            } else {
                const std::size_t input_rows = inverse_embedding ? physical[0] : logical[0];
                const std::size_t input_cols = inverse_embedding ? physical[1] : logical[1];
                if (!input_is_first_workspace && !plan->direct_input_boundary) {
                    cuntt::detail::launch_pack_2d(input, first, input_rows, input_cols, physical[0], physical[1],
                                                  plan->descriptor.batch, plan->descriptor.input_strides[0],
                                                  plan->descriptor.input_strides[1], plan->descriptor.input_batch_stride,
                                                  plan->element_bytes, handle->stream);
                }
            }
            void* packed_result = first;
            if (plan->kind == cubutterflyPlan::Kind::Ntt) {
                void* ntt_destination = plan->direct_output_boundary && plan->descriptor.rank == 1
                                            ? output
                                            : static_cast<void*>(second);
                const void* ntt_input = plan->direct_input_boundary ? input : first;
                if (plan->direct_input_boundary && ntt_input == ntt_destination)
                    ntt_destination = first;
                if (plan->descriptor.storage == CUBUTTERFLY_DATA_UINT32)
                    plan->ntt_axes[0]->execute_async(reinterpret_cast<const std::uint32_t*>(ntt_input), reinterpret_cast<std::uint32_t*>(ntt_destination));
                else
                    plan->ntt_axes[0]->execute_async(reinterpret_cast<const std::uint64_t*>(ntt_input), reinterpret_cast<std::uint64_t*>(ntt_destination));
                packed_result = ntt_destination;
                if (plan->descriptor.rank == 2) {
                    void* transposed = ntt_destination == second ? first : second;
                    cuntt::detail::launch_transpose_2d(ntt_destination, transposed, physical[0], physical[1],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    if (plan->descriptor.storage == CUBUTTERFLY_DATA_UINT32)
                        plan->ntt_axes[1]->execute_async(reinterpret_cast<const std::uint32_t*>(transposed), reinterpret_cast<std::uint32_t*>(ntt_destination));
                    else
                        plan->ntt_axes[1]->execute_async(reinterpret_cast<const std::uint64_t*>(transposed), reinterpret_cast<std::uint64_t*>(ntt_destination));
                    void* final_destination = plan->direct_output_boundary
                                                  ? output
                                                  : transposed;
                    cuntt::detail::launch_transpose_2d(ntt_destination, final_destination, physical[1], physical[0],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    packed_result = final_destination;
                }
            } else {
                void* butterfly_first = direct_butterfly_1d ? output : static_cast<void*>(first);
                if (plan->direct_input_boundary && input == butterfly_first)
                    butterfly_first = second;
                if (plan->fused_input_boundary) {
                    plan->butterfly_axes[0]->execute_zero_extended_async(
                        static_cast<const float*>(input), static_cast<float*>(output),
                        logical[0], plan->descriptor.input_batch_stride,
                        plan->descriptor.input_strides[0]);
                } else {
                    execute_butterfly(*plan->butterfly_axes[0],
                                      plan->direct_input_boundary ? input : butterfly_first, butterfly_first,
                                      plan->descriptor.storage);
                }
                packed_result = butterfly_first;
                if (plan->descriptor.rank == 2) {
                    void* transposed = butterfly_first == first ? second : first;
                    cuntt::detail::launch_transpose_2d(butterfly_first, transposed, physical[0], physical[1],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    execute_butterfly(*plan->butterfly_axes[1], transposed, transposed, plan->descriptor.storage);
                    void* final_destination = plan->direct_output_boundary
                                                  ? output
                                                  : butterfly_first;
                    cuntt::detail::launch_transpose_2d(transposed, final_destination, physical[1], physical[0],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    packed_result = final_destination;
                }
            }
            const bool forward_embedding = plan->descriptor.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING &&
                                            plan->descriptor.direction == CUBUTTERFLY_DIRECTION_FORWARD;
            if (plan->direct_output_boundary) {
                return;
            } else if (plan->descriptor.rank == 1) {
                const std::size_t output_n = forward_embedding ? physical[0] : logical[0];
                cuntt::detail::launch_scatter_1d(packed_result, output, output_n, physical[0], plan->descriptor.batch,
                                                 plan->descriptor.output_strides[0], plan->descriptor.output_batch_stride,
                                                 plan->element_bytes, handle->stream);
            } else {
                const std::size_t output_rows = forward_embedding ? physical[0] : logical[0];
                const std::size_t output_cols = forward_embedding ? physical[1] : logical[1];
                cuntt::detail::launch_scatter_2d(packed_result, output, output_rows, output_cols, physical[0], physical[1],
                                                 plan->descriptor.batch, plan->descriptor.output_strides[0],
                                                 plan->descriptor.output_strides[1], plan->descriptor.output_batch_stride,
                                                 plan->element_bytes, handle->stream);
            }
            return;
        }
        if (plan->kind == cubutterflyPlan::Kind::Ntt) {
            if (plan->ntt->config().word_bits == 32)
                plan->ntt->execute_async(static_cast<const std::uint32_t*>(input), static_cast<std::uint32_t*>(output));
            else
                plan->ntt->execute_async(static_cast<const std::uint64_t*>(input), static_cast<std::uint64_t*>(output));
        } else {
            const auto precision = plan->butterfly->config().precision;
            cubutterflyDataType_t storage = CUBUTTERFLY_DATA_FP32;
            if (plan->butterfly->config().op == cuntt::ButterflyOperator::Fft) {
                if (precision == cuntt::ButterflyPrecision::Fp16) storage = CUBUTTERFLY_DATA_COMPLEX_FP16;
                else if (precision == cuntt::ButterflyPrecision::Bf16) storage = CUBUTTERFLY_DATA_COMPLEX_BF16;
                else if (precision == cuntt::ButterflyPrecision::Fp64) storage = CUBUTTERFLY_DATA_COMPLEX_FP64;
                else storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
            } else if (precision == cuntt::ButterflyPrecision::Fp16) storage = CUBUTTERFLY_DATA_FP16;
            else if (precision == cuntt::ButterflyPrecision::Bf16) storage = CUBUTTERFLY_DATA_BF16;
            else if (precision == cuntt::ButterflyPrecision::Fp64) storage = CUBUTTERFLY_DATA_FP64;
            else if (precision == cuntt::ButterflyPrecision::Uint32) storage = CUBUTTERFLY_DATA_UINT32;
            else if (precision == cuntt::ButterflyPrecision::Uint64) storage = CUBUTTERFLY_DATA_UINT64;
            execute_butterfly(*plan->butterfly, input, output, storage);
        }
    });
}

namespace {

// Submit the already-bound, direct portion of a plan without rebuilding the
// generic dispatcher state.  This is intentionally a narrow steady-state
// path: plans that still need semantic staging fall back to the complete
// executor, and the public entry point falls back whenever the caller stream
// changed or has not yet been propagated to the plan's children.
void execute_steady_bound(cubutterflyHandle_t handle, cubutterflyPlan_t plan,
                          const void* input, void* output) {
    if (plan->xd_selected) {
        execute_steady_bound(handle, plan->xd_selected.get(), input, output);
        return;
    }
    if (plan->direct_real_fft) {
        plan->direct_real_fft->execute(input, output, handle->stream);
        return;
    }
    if (plan->xd_fft) {
        plan->xd_fft->execute(input, output, handle->stream);
        return;
    }
    if (plan->xd_resident) {
        plan->xd_resident->execute(input, output, handle->stream);
        return;
    }
    if (plan->kind == cubutterflyPlan::Kind::DirectFft) {
        plan->direct_fft->execute(input, output, handle->stream);
        return;
    }
    const auto status = cubutterflyExecute(handle, plan, input, output);
    if (status != CUBUTTERFLY_STATUS_SUCCESS)
        throw std::runtime_error(std::string("steady child execute: ") +
                                 cubutterflyGetStatusString(status));
}

}  // namespace

cubutterflyStatus_t cubutterflyExecuteSteady(cubutterflyHandle_t handle,
                                             cubutterflyPlan_t plan,
                                             const void* input, void* output) {
    return protect([&] {
        require(handle != nullptr && plan != nullptr && input != nullptr && output != nullptr,
                "execution arguments must be non-null");
        const bool in_place = plan->descriptor.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE;
        require(in_place ? input == output : input != output,
                in_place ? "in-place execution requires identical pointers" :
                           "out-of-place execution requires distinct pointers");
        // The first call and a stream change use the complete path so all
        // child plans receive the same stream before the fast path is used.
        if (plan->bound_stream != handle->stream) {
            const auto status = cubutterflyExecute(handle, plan, input, output);
            if (status != CUBUTTERFLY_STATUS_SUCCESS)
                throw std::runtime_error(std::string("steady stream bind: ") +
                                         cubutterflyGetStatusString(status));
            return;
        }
        execute_steady_bound(handle, plan, input, output);
    });
}
