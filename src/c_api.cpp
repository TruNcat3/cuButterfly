#include "cubutterfly/cubutterfly.h"

#include "cuntt/butterfly.hpp"
#include "cuntt/ntt.hpp"
#include "bluestein_fft.hpp"
#include "bluestein_ntt.hpp"
#include "direct_fft.hpp"
#include "general_kernels.hpp"
#include "generated_application_profiles.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#ifndef CUBUTTERFLY_HAS_CUFFTDX
#define CUBUTTERFLY_HAS_CUFFTDX 0
#endif

struct cubutterflyHandle {
    cudaStream_t stream = nullptr;
    cubutterflyLogCallback_t logger = nullptr;
    void* logger_data = nullptr;
    std::string cache_path;
};

struct cubutterflyDescriptor {
    cubutterflyOperator_t op = CUBUTTERFLY_OPERATOR_FFT;
    cubutterflyDataType_t storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
    cubutterflyComputeType_t compute = CUBUTTERFLY_COMPUTE_FP32;
    cubutterflyDirection_t direction = CUBUTTERFLY_DIRECTION_FORWARD;
    cubutterflyPlacement_t placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    cubutterflyLengthMode_t length_mode = CUBUTTERFLY_LENGTH_STANDARD;
    cubutterflyAlgorithmPolicy_t algorithm_policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
    std::string explicit_algorithm;
    std::uint32_t rank = 1;
    std::vector<std::size_t> extents{256};
    std::vector<std::size_t> input_strides;
    std::vector<std::size_t> output_strides;
    std::size_t input_batch_stride = 0;
    std::size_t output_batch_stride = 0;
    std::size_t batch = 1;
    bool normalize_inverse = true;
    std::uint64_t modulus = cuntt::kDefaultModulus;
    std::uint32_t word_bits = 64;
    std::vector<std::vector<cubutterflyMatrix2x2_t>> stage_matrices{2};
};

struct cubutterflyPlan {
    enum class Kind { Butterfly, Ntt, DirectFft, BluesteinFft, BluesteinNtt } kind = Kind::Butterfly;
    std::unique_ptr<cuntt::ButterflyPlan> butterfly;
    std::unique_ptr<cuntt::Plan> ntt;
    std::vector<std::unique_ptr<cuntt::detail::BluesteinFftPlan>> bluestein_axes;
    std::vector<std::unique_ptr<cuntt::detail::BluesteinNttPlan>> bluestein_ntt_axes;
    std::unique_ptr<cuntt::detail::DirectFftPlan> direct_fft;
    std::vector<std::unique_ptr<cuntt::ButterflyPlan>> butterfly_axes;
    std::vector<std::unique_ptr<cuntt::Plan>> ntt_axes;
    cubutterflyDescriptor descriptor;
    bool composite = false;
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
};

namespace {

template <typename Function>
cubutterflyStatus_t protect(Function&& function) noexcept {
    try {
        function();
        return CUBUTTERFLY_STATUS_SUCCESS;
    } catch (const std::invalid_argument&) {
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    } catch (const std::overflow_error&) {
        return CUBUTTERFLY_STATUS_INVALID_VALUE;
    } catch (const std::bad_alloc&) {
        return CUBUTTERFLY_STATUS_ALLOC_FAILED;
    } catch (const std::logic_error&) {
        return CUBUTTERFLY_STATUS_INTERNAL_ERROR;
    } catch (const std::runtime_error&) {
        return CUBUTTERFLY_STATUS_BACKEND_ERROR;
    } catch (...) {
        return CUBUTTERFLY_STATUS_INTERNAL_ERROR;
    }
}

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}

std::string workload_key(const cubutterflyDescriptor& descriptor);

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
        const std::size_t add = (extents[axis] - 1) * strides[axis];
        require(extent <= static_cast<std::size_t>(-1) - add, "strided extent overflows size_t");
        extent += add;
    }
    if (descriptor.batch > 1) {
        require(batch_stride >= extent, "batch stride is smaller than one transform extent");
        const std::size_t add = (descriptor.batch - 1) * batch_stride;
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
                strides[axis - 1] = strides[axis] * extents[axis];
        }
        require(strides.size() == descriptor.rank, "stride rank does not match shape rank");
        if (batch_stride == 0) {
            batch_stride = 1;
            for (std::uint32_t axis = 0; axis < descriptor.rank; ++axis)
                batch_stride += (extents[axis] - 1) * strides[axis];
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
        stride *= descriptor.extents[axis];
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
        stride *= extents[axis];
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
        case CUBUTTERFLY_DATA_UINT64: break;
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
    if (descriptor.op == CUBUTTERFLY_OPERATOR_FFT)
        require(complex, "FFT requires a complex storage type");
    else if (descriptor.op == CUBUTTERFLY_OPERATOR_NTT)
        require(descriptor.storage == CUBUTTERFLY_DATA_UINT32 || descriptor.storage == CUBUTTERFLY_DATA_UINT64,
                "NTT requires uint32 or uint64 storage");
    else if (descriptor.op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || descriptor.op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA)
        require(descriptor.storage == CUBUTTERFLY_DATA_UINT32, "zeta transforms require uint32 storage");
    else
        require(real, "real butterfly operator requires a real floating-point storage type");
}

bool v100_profile_device() {
    const auto device = cuntt::current_device_info();
    return device.compute_major == cuntt::detail::kApplicationProfileComputeMajor &&
           device.compute_minor == cuntt::detail::kApplicationProfileComputeMinor &&
           device.multiprocessors == cuntt::detail::kApplicationProfileMultiprocessors;
}

bool measured_v100_shape(const cubutterflyDescriptor& descriptor, std::uint32_t log_n) {
    if (!v100_profile_device() || descriptor.rank != 1 ||
        descriptor.length_mode != CUBUTTERFLY_LENGTH_STANDARD ||
        descriptor.direction != CUBUTTERFLY_DIRECTION_FORWARD)
        return false;
    if (descriptor.input_strides.size() != 1 || descriptor.output_strides.size() != 1 ||
        descriptor.input_strides[0] != 1 || descriptor.output_strides[0] != 1 ||
        descriptor.input_batch_stride != descriptor.extents[0] ||
        descriptor.output_batch_stride != descriptor.extents[0])
        return false;
    if (descriptor.op == CUBUTTERFLY_OPERATOR_FFT && !CUBUTTERFLY_HAS_CUFFTDX)
        return false;
    for (const auto& profile : cuntt::detail::kApplicationProfiles) {
        if (profile.op != static_cast<int>(descriptor.op) ||
            profile.storage != static_cast<int>(descriptor.storage) ||
            profile.direction != static_cast<int>(descriptor.direction) ||
            profile.placement != static_cast<int>(descriptor.placement) || profile.log_n != log_n)
            continue;
        if (profile.modulus != 0 && profile.modulus != descriptor.modulus)
            continue;
        if (std::find(profile.batches, profile.batches + profile.batch_count, descriptor.batch) !=
            profile.batches + profile.batch_count)
            return true;
    }
    return false;
}

bool measured_v100_axis(const cubutterflyDescriptor& descriptor, std::uint32_t log_n,
                        std::size_t batch) {
    if (!v100_profile_device() || descriptor.direction != CUBUTTERFLY_DIRECTION_FORWARD)
        return false;
    if (descriptor.op == CUBUTTERFLY_OPERATOR_FFT && !CUBUTTERFLY_HAS_CUFFTDX)
        return false;
    for (const auto& profile : cuntt::detail::kApplicationProfiles) {
        if (profile.op != static_cast<int>(descriptor.op) ||
            profile.storage != static_cast<int>(descriptor.storage) ||
            profile.direction != static_cast<int>(descriptor.direction) || profile.log_n != log_n)
            continue;
        if (profile.modulus != 0 && profile.modulus != descriptor.modulus)
            continue;
        if (std::find(profile.batches, profile.batches + profile.batch_count, batch) !=
            profile.batches + profile.batch_count)
            return true;
    }
    return false;
}

void log_profile_miss(cubutterflyHandle_t handle, const cubutterflyDescriptor& descriptor,
                      const std::string& reason, const std::string& algorithm) {
    const auto device = cuntt::current_device_info();
    std::ostringstream message;
    if (!v100_profile_device()) {
        message << "no static performance profile for device='" << device.name << "' sm_"
                << device.compute_major << device.compute_minor << " multiprocessors="
                << device.multiprocessors;
        log_event(handle, CUBUTTERFLY_LOG_WARNING, "hardware-profile-miss",
                  message.str() + "; using " + algorithm + "; set MEASURE with a cache path to calibrate this workload");
        return;
    }
    message << "V100 profile has no exact entry for " << workload_key(descriptor)
            << "; " << reason << "; using " << algorithm
            << "; set MEASURE with a cache path to record a reusable choice";
    log_event(handle, CUBUTTERFLY_LOG_WARNING, "workload-profile-miss", message.str());
}

void choose_butterfly_mapping(cuntt::ButterflyConfig& config, bool measured) {
    config.compute_unit = cuntt::ComputeUnit::Radix4;
    config.tile_threads = 128;
    if (measured && config.op == cuntt::ButterflyOperator::Fwht &&
        config.precision == cuntt::ButterflyPrecision::Fp32) {
        if (config.log_n == 8 || (config.log_n == 15 && config.batch >= 16)) {
            config.backend = cuntt::ButterflyBackend::TemporalTile;
            config.compute_unit = cuntt::ComputeUnit::Radix2;
            config.local_exchange = cuntt::LocalExchange::WarpRegister;
            config.tile_threads = 256;
            return;
        }
        if (config.log_n == 15) {
            config.backend = cuntt::ButterflyBackend::OnlineReorder;
            config.local_stages = 8;
            config.reorder_columns = 1;
            config.tile_threads = 256;
            return;
        }
    }
    if (config.log_n <= 10) {
        config.backend = cuntt::ButterflyBackend::TemporalTile;
        return;
    }
    if (measured && config.log_n >= 18) {
        config.backend = cuntt::ButterflyBackend::OnlineReorder;
        config.local_stages = config.log_n / 2;
        config.reorder_columns = 1;
        config.tile_threads = 256;
        return;
    }
    config.backend = cuntt::ButterflyBackend::Hierarchical;
    config.local_stages = std::min<std::uint32_t>(10, config.log_n - 1);
    config.tile_threads = 256;
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
        case CUBUTTERFLY_DATA_UINT64: break;
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
    choose_butterfly_mapping(config, false);
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
    config.auto_allocate_workspace = false;
    if (config.log_n >= 12 && config.log_n <= 20) {
        config.backend = cuntt::Backend::Hybrid2D;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
    } else {
        config.backend = cuntt::Backend::Baseline;
        config.compute_unit = cuntt::ComputeUnit::Radix2;
    }
    return config;
}

std::size_t align_16(std::size_t value) {
    require(value <= static_cast<std::size_t>(-1) - 15, "workspace size overflows size_t");
    return (value + 15) & ~std::size_t{15};
}

const char* operator_name(cubutterflyOperator_t op) {
    switch (op) {
        case CUBUTTERFLY_OPERATOR_FFT: return "fft";
        case CUBUTTERFLY_OPERATOR_NTT: return "ntt";
        case CUBUTTERFLY_OPERATOR_FWHT: return "fwht";
        case CUBUTTERFLY_OPERATOR_SUBSET_ZETA: return "subset-zeta";
        case CUBUTTERFLY_OPERATOR_SUPERSET_ZETA: return "superset-zeta";
        case CUBUTTERFLY_OPERATOR_STRUCTURED_2X2: return "structured-2x2";
    }
    return "unknown";
}

std::string workload_key(const cubutterflyDescriptor& descriptor) {
    const auto device = cuntt::current_device_info();
    std::ostringstream key;
    key << "v1|sm" << device.compute_major << device.compute_minor << "-mp" << device.multiprocessors
        << '|' << operator_name(descriptor.op) << "|storage" << static_cast<int>(descriptor.storage)
        << "|compute" << static_cast<int>(descriptor.compute) << "|direction" << static_cast<int>(descriptor.direction)
        << "|placement" << static_cast<int>(descriptor.placement) << "|length-mode" << static_cast<int>(descriptor.length_mode)
        << "|rank" << descriptor.rank << "|shape";
    for (std::uint32_t axis = 0; axis < descriptor.rank; ++axis) {
        if (axis != 0)
            key << 'x';
        key << descriptor.extents[axis];
    }
    key << "|batch" << descriptor.batch << "|in";
    for (const auto stride : descriptor.input_strides)
        key << stride << ',';
    key << descriptor.input_batch_stride << "|out";
    for (const auto stride : descriptor.output_strides)
        key << stride << ',';
    key << descriptor.output_batch_stride;
    if (descriptor.op == CUBUTTERFLY_OPERATOR_NTT)
        key << "|modulus" << descriptor.modulus << "|word" << descriptor.word_bits;
    if (descriptor.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
        key << "|matrices" << std::setprecision(std::numeric_limits<double>::max_digits10);
        for (std::uint32_t axis = 0; axis < descriptor.rank; ++axis) {
            if (axis != 0)
                key << '/';
            for (const auto& matrix : descriptor.stage_matrices[axis])
                key << matrix.m00 << ',' << matrix.m01 << ',' << matrix.m10 << ',' << matrix.m11 << ';';
        }
    }
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
    if (descriptor.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
        require(source_axis < descriptor.stage_matrices.size(), "structured axis is out of range");
        axis.stage_matrices[0] = descriptor.stage_matrices[source_axis];
        axis.stage_matrices[1].clear();
    }
    return axis;
}

std::string lookup_cache(const std::string& path, const std::string& key) {
    if (path.empty())
        return {};
    std::ifstream input(path);
    std::string line;
    std::string match;
    while (std::getline(input, line)) {
        const auto separator = line.find('\t');
        if (separator != std::string::npos && line.compare(0, separator, key) == 0 && separator == key.size())
            match = line.substr(separator + 1);
    }
    return match;
}

void store_cache(const std::string& path, const std::string& key, const std::string& algorithm) {
    if (path.empty())
        return;
    std::ofstream output(path, std::ios::app);
    if (!output)
        throw std::runtime_error("could not open the cuButterfly selection cache for writing");
    output << key << '\t' << algorithm << '\n';
}

void apply_butterfly_algorithm(cuntt::ButterflyConfig& config, const std::string& algorithm) {
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
    throw std::invalid_argument("butterfly configuration has no public algorithm ID");
}

std::string ntt_algorithm_id(const cuntt::PlanConfig& config) {
    if (config.backend == cuntt::Backend::Baseline)
        return "ntt-baseline-radix2";
    if (config.backend == cuntt::Backend::Tile256)
        return "ntt-tile256-radix2";
    if (config.backend == cuntt::Backend::Hybrid2D)
        return config.compute_unit == cuntt::ComputeUnit::Radix4 ? "ntt-hybrid-radix4"
                                                                 : "ntt-hybrid-radix2";
    throw std::invalid_argument("NTT configuration has no public algorithm ID");
}

void apply_ntt_algorithm(cuntt::PlanConfig& config, const std::string& algorithm) {
    if (algorithm == "ntt-baseline-radix2") {
        config.backend = cuntt::Backend::Baseline;
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
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
        throw std::runtime_error("could not create candidate measurement events");
    cudaEventRecord(start, stream);
    for (int repeat = 0; repeat < 5; ++repeat)
        execute();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return milliseconds / 5.0;
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
    cudaMemsetAsync(input.pointer, 0, bytes, stream);
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
    cudaMemsetAsync(input.pointer, 0, logical_bytes, stream);
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

double measure_butterfly(cuntt::ButterflyConfig config, cubutterflyDataType_t storage,
                         cudaStream_t stream) {
    config.auto_allocate_workspace = false;
    cuntt::ButterflyPlan plan(config);
    plan.set_stream(stream);
    DeviceAllocation input(plan.data_size());
    DeviceAllocation output(config.placement == cuntt::ButterflyPlacement::InPlace ? 0 : plan.data_size());
    DeviceAllocation workspace(plan.workspace_size());
    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace.pointer, plan.workspace_size());
    cudaMemsetAsync(input.pointer, 0, plan.data_size(), stream);
    void* destination = output.pointer == nullptr ? input.pointer : output.pointer;
    for (int warmup = 0; warmup < 2; ++warmup)
        execute_butterfly(plan, input.pointer, destination, storage);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
        throw std::runtime_error("could not create tuning events");
    cudaEventRecord(start, stream);
    for (int repeat = 0; repeat < 5; ++repeat)
        execute_butterfly(plan, input.pointer, destination, storage);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return milliseconds / 5.0;
}

std::pair<std::string, double> tune_butterfly(const cuntt::ButterflyConfig& base,
                                               cubutterflyDataType_t storage,
                                               cudaStream_t stream) {
    std::vector<std::string> candidates{"temporal-radix2", "temporal-radix4",
                                        "temporal-warp-register-radix2",
                                        "hierarchical-radix4", "online-radix4"};
    if (base.op == cuntt::ButterflyOperator::Fft)
        candidates.push_back("cufft");
    std::pair<std::string, double> best{"", std::numeric_limits<double>::infinity()};
    for (const auto& name : candidates) {
        try {
            auto candidate = base;
            apply_butterfly_algorithm(candidate, name);
            const double milliseconds = measure_butterfly(candidate, storage, stream);
            if (milliseconds < best.second)
                best = {name, milliseconds};
        } catch (const std::exception&) {
        }
    }
    require(!best.first.empty(), "no legal butterfly candidate survived measurement");
    return best;
}

double measure_ntt(cuntt::PlanConfig config, cudaStream_t stream) {
    config.auto_allocate_workspace = false;
    cuntt::Plan plan(config);
    plan.set_stream(stream);
    DeviceAllocation input(plan.data_size());
    DeviceAllocation output(plan.data_size());
    DeviceAllocation workspace(plan.workspace_size());
    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace.pointer, plan.workspace_size());
    cudaMemsetAsync(input.pointer, 0, plan.data_size(), stream);
    auto execute = [&] {
        if (config.word_bits == 32)
            plan.execute_async(static_cast<const std::uint32_t*>(input.pointer), static_cast<std::uint32_t*>(output.pointer));
        else
            plan.execute_async(static_cast<const std::uint64_t*>(input.pointer), static_cast<std::uint64_t*>(output.pointer));
    };
    execute();
    execute();
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
        throw std::runtime_error("could not create tuning events");
    cudaEventRecord(start, stream);
    for (int repeat = 0; repeat < 5; ++repeat)
        execute();
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0F;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return milliseconds / 5.0;
}

std::pair<std::string, double> tune_ntt(const cuntt::PlanConfig& base, cudaStream_t stream) {
    const std::vector<std::string> candidates{"ntt-baseline-radix2", "ntt-tile256-radix2",
                                               "ntt-hybrid-radix2", "ntt-hybrid-radix4"};
    std::pair<std::string, double> best{"", std::numeric_limits<double>::infinity()};
    for (const auto& name : candidates) {
        try {
            auto candidate = base;
            apply_ntt_algorithm(candidate, name);
            const double milliseconds = measure_ntt(candidate, stream);
            if (milliseconds < best.second)
                best = {name, milliseconds};
        } catch (const std::exception&) {
        }
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

ButterflyAxisSelection select_butterfly_axis(cubutterflyHandle_t handle,
                                             const cubutterflyDescriptor& descriptor,
                                             std::uint32_t source_axis, std::size_t length,
                                             std::size_t batch) {
    auto config = make_axis_butterfly_config(descriptor, source_axis, length, batch);
    const auto axis_descriptor = make_axis_descriptor(descriptor, source_axis, length, batch);
    const auto key = workload_key(axis_descriptor);
    const auto cached = descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT
                            ? lookup_cache(handle->cache_path, key)
                            : std::string{};
    ButterflyAxisSelection selected;
    if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
        apply_butterfly_algorithm(config, descriptor.explicit_algorithm);
        selected.source = CUBUTTERFLY_SELECTION_EXPLICIT;
        selected.reason = "applied the explicit algorithm to the physical axis";
    } else if (!cached.empty()) {
        apply_butterfly_algorithm(config, cached);
        selected.source = CUBUTTERFLY_SELECTION_USER_CACHE;
        selected.reason = "matched the physical axis in the user selection cache";
    } else if (descriptor.algorithm_policy == CUBUTTERFLY_ALGORITHM_MEASURE) {
        const auto tuned = tune_butterfly(config, descriptor.storage, handle->stream);
        apply_butterfly_algorithm(config, tuned.first);
        store_cache(handle->cache_path, key, tuned.first);
        selected.source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
        selected.reason = "measured physical-axis candidates; kernel_ms=" + std::to_string(tuned.second);
    } else {
        const bool profiled = measured_v100_axis(descriptor, integer_log2(length), batch);
        choose_butterfly_mapping(config, profiled);
        selected.source = profiled ? CUBUTTERFLY_SELECTION_MEASURED_TABLE
                                   : CUBUTTERFLY_SELECTION_STATIC_MODEL;
        selected.reason = profiled ? "matched the built-in V100 physical-axis profile"
                                   : "selected the physical axis with the static resource model";
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
        apply_ntt_algorithm(config, descriptor.explicit_algorithm);
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
        const bool profiled = measured_v100_axis(descriptor, integer_log2(length), batch);
        selected.source = profiled ? CUBUTTERFLY_SELECTION_MEASURED_TABLE
                                   : CUBUTTERFLY_SELECTION_STATIC_MODEL;
        selected.reason = profiled ? "matched the built-in V100 physical NTT-axis profile"
                                   : "selected the physical NTT axis with the static resource model";
    }
    selected.algorithm = ntt_algorithm_id(config);
    selected.config = std::move(config);
    return selected;
}

cubutterflySelectionSource_t combined_source(
    cubutterflySelectionSource_t current, cubutterflySelectionSource_t next) {
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

}  // namespace

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

cubutterflyStatus_t cubutterflySetOperator(cubutterflyDescriptor_t descriptor, cubutterflyOperator_t op) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->op = op; });
}

cubutterflyStatus_t cubutterflySetDataType(cubutterflyDescriptor_t descriptor, cubutterflyDataType_t storage, cubutterflyComputeType_t compute) {
    return protect([&] { require(descriptor != nullptr, "descriptor must be non-null"); descriptor->storage = storage; descriptor->compute = compute; });
}

cubutterflyStatus_t cubutterflySetShape(cubutterflyDescriptor_t descriptor, uint32_t rank, const size_t* extents, size_t batch) {
    return protect([&] {
        require(descriptor != nullptr && extents != nullptr, "descriptor and extents must be non-null");
        require(rank == 1 || rank == 2, "rank must be one or two");
        require(batch != 0, "batch must be positive");
        descriptor->rank = rank;
        descriptor->extents.assign(extents, extents + rank);
        require(std::all_of(descriptor->extents.begin(), descriptor->extents.end(), [](std::size_t n) { return n != 0; }),
                "extents must be positive");
        descriptor->batch = batch;
        descriptor->input_strides.clear();
        descriptor->output_strides.clear();
        descriptor->input_batch_stride = 0;
        descriptor->output_batch_stride = 0;
    });
}

cubutterflyStatus_t cubutterflySetStrides(cubutterflyDescriptor_t descriptor, const size_t* input_strides, size_t input_batch_stride,
                                          const size_t* output_strides, size_t output_batch_stride) {
    return protect([&] {
        require(descriptor != nullptr && input_strides != nullptr && output_strides != nullptr, "descriptor and strides must be non-null");
        descriptor->input_strides.assign(input_strides, input_strides + descriptor->rank);
        descriptor->output_strides.assign(output_strides, output_strides + descriptor->rank);
        descriptor->input_batch_stride = input_batch_stride;
        descriptor->output_batch_stride = output_batch_stride;
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

cubutterflyStatus_t cubutterflySetStageMatrices(cubutterflyDescriptor_t descriptor, uint32_t axis,
                                                const cubutterflyMatrix2x2_t* matrices, size_t count) {
    return protect([&] {
        require(descriptor != nullptr && matrices != nullptr && count != 0, "matrix arguments must be non-null and non-empty");
        require(axis < 2, "matrix axis must be zero or one");
        descriptor->stage_matrices[axis].assign(matrices, matrices + count);
    });
}

cubutterflyStatus_t cubutterflyCreatePlan(cubutterflyHandle_t handle, cubutterflyDescriptor_t descriptor, cubutterflyPlan_t* plan) {
    return protect([&] {
        require(handle != nullptr && descriptor != nullptr && plan != nullptr, "handle, descriptor, and plan pointer must be non-null");
        auto resolved = *descriptor;
        validate_storage(resolved);
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
        created->input_bytes = strided_elements(resolved, true, input_extents) * created->element_bytes;
        created->output_bytes = strided_elements(resolved, false, output_extents) * created->element_bytes;
        const bool measured = !standard_arbitrary && resolved.rank == 1 && resolved.length_mode == CUBUTTERFLY_LENGTH_STANDARD &&
                              resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT && measured_v100_shape(resolved, log_n);
        created->selection_source = resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT
                                        ? CUBUTTERFLY_SELECTION_EXPLICIT
                                        : (measured ? CUBUTTERFLY_SELECTION_MEASURED_TABLE : CUBUTTERFLY_SELECTION_STATIC_MODEL);
        created->composite = resolved.rank == 2 || resolved.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
        created->direct_output_boundary = created->composite && output_extents == physical_extents &&
                                          packed_side(resolved, false, output_extents);
        const std::string cache_key = workload_key(resolved);
        const std::string cached_algorithm = resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT
                                                 ? lookup_cache(handle->cache_path, cache_key)
                                                 : std::string{};
        if (standard_arbitrary) {
            require(resolved.placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE,
                    "Bluestein FFT currently requires out-of-place execution");
            const bool ntt_bluestein = resolved.op == CUBUTTERFLY_OPERATOR_NTT;
            const char* base_algorithm = ntt_bluestein ? "bluestein-ntt-power2-core" : "bluestein-cufft-power2-core";
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
                    const double direct_ms = packed_layout(resolved)
                                                 ? measure_direct_fft_candidate(
                                                       resolved, created->input_bytes, handle->stream)
                                                 : std::numeric_limits<double>::infinity();
                    direct_fft = direct_ms <= bluestein_ms;
                    const std::string selected_id = direct_fft ? "cufft-direct" : base_algorithm;
                    store_cache(handle->cache_path, cache_key, selected_id);
                    arbitrary_selection_source = CUBUTTERFLY_SELECTION_RUNTIME_MEASURED;
                    arbitrary_selection_reason = "measured exact-length candidates; direct_ms=" +
                        std::to_string(direct_ms) + "; bluestein_ms=" + std::to_string(bluestein_ms);
                    log_event(handle, CUBUTTERFLY_LOG_INFO, "measure-complete",
                              arbitrary_selection_reason + "; algorithm=" + selected_id);
                } else {
                    direct_fft = packed_layout(resolved);
                    arbitrary_selection_reason = "static policy prefers the vendor native exact-length candidate";
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
                if (resolved.explicit_algorithm == base_algorithm) {
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
                        ntt_core_selections[0].config.compute_unit));
                    created->workspace_bytes = created->bluestein_ntt_axes[0]->workspace_size();
                } else {
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch,
                        inverse, resolved.normalize_inverse, fp64));
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
                        ntt_core_selections[0].config.compute_unit));
                    created->bluestein_ntt_axes.push_back(std::make_unique<cuntt::detail::BluesteinNttPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch * resolved.extents[1],
                        resolved.modulus, inverse, ntt_core_selections[1].config.backend,
                        ntt_core_selections[1].config.compute_unit));
                } else {
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[1], physical_extents[1], resolved.batch * resolved.extents[0],
                        inverse, resolved.normalize_inverse, fp64));
                    created->bluestein_axes.push_back(std::make_unique<cuntt::detail::BluesteinFftPlan>(
                        resolved.extents[0], physical_extents[0], resolved.batch * resolved.extents[1],
                        inverse, resolved.normalize_inverse, fp64));
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
                    created->algorithm = "rank2-transpose-bluestein-cufft-power2-core";
            }
            log_event(handle, CUBUTTERFLY_LOG_INFO, "algorithm-selection",
                      created->reason + "; algorithm=" + created->algorithm);
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT)
                log_profile_miss(handle, resolved, created->reason, created->algorithm);
            *plan = created.release();
            return;
        }
        if (created->composite) {
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
                auto selected = select_butterfly_axis(handle, resolved, source_axis, length, batch);
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
                const std::size_t row_transforms = resolved.batch * physical_extents[0];
                const std::size_t column_transforms = resolved.batch * physical_extents[1];
                if (created->kind == cubutterflyPlan::Kind::Ntt) {
                    add_ntt_axis(1, physical_extents[1], row_transforms);
                    add_ntt_axis(0, physical_extents[0], column_transforms);
                } else {
                    add_butterfly_axis(1, physical_extents[1], row_transforms);
                    add_butterfly_axis(0, physical_extents[0], column_transforms);
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
            created->composite_buffers = resolved.rank == 2
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
            algorithm << (resolved.rank == 2 ? "rank2-transpose" : "zero-extended") << '[';
            for (std::size_t axis = 0; axis < axis_algorithms.size(); ++axis)
                algorithm << (axis == 0 ? "" : ",") << axis_algorithms[axis];
            algorithm << ']';
            if (created->direct_output_boundary)
                algorithm << "+direct-output";
            if (created->fused_input_boundary)
                algorithm << "+fused-input";
            created->algorithm = algorithm.str();
            std::ostringstream reason;
            for (std::size_t axis = 0; axis < axis_reasons.size(); ++axis)
                reason << (axis == 0 ? "" : "; ") << "axis" << axis << ": " << axis_reasons[axis];
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
            config.auto_allocate_workspace = false;
            config.auto_select = measured;
            if (resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) {
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
            } else if (log_n >= 12 && log_n <= 20) {
                config.backend = cuntt::Backend::Hybrid2D;
                config.compute_unit = cuntt::ComputeUnit::Radix4;
            } else {
                config.backend = cuntt::Backend::Baseline;
                config.compute_unit = cuntt::ComputeUnit::Radix2;
            }
            created->kind = cubutterflyPlan::Kind::Ntt;
            if (created->reason.empty())
                created->reason = created->selection_source == CUBUTTERFLY_SELECTION_USER_CACHE
                                      ? "matched an exact workload key in the user selection cache"
                                      : (measured ? "matched the built-in V100 measured profile"
                                                  : "no measured profile entry; selected by the static resource model");
            created->ntt = std::make_unique<cuntt::Plan>(config);
            created->ntt->set_stream(handle->stream);
            created->algorithm = std::string("ntt-") + cuntt::backend_name(created->ntt->config().backend) +
                                 "-" + cuntt::compute_unit_name(created->ntt->config().compute_unit);
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
            config.auto_allocate_workspace = false;
            config.auto_select = measured;
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
                choose_butterfly_mapping(config, measured);
            }
            created->kind = cubutterflyPlan::Kind::Butterfly;
            if (created->reason.empty())
                created->reason = created->selection_source == CUBUTTERFLY_SELECTION_USER_CACHE
                                      ? "matched an exact workload key in the user selection cache"
                                      : (measured ? "matched the built-in V100 measured profile"
                                                  : "no measured profile entry; selected by the static resource model");
            created->butterfly = std::make_unique<cuntt::ButterflyPlan>(config);
            created->butterfly->set_stream(handle->stream);
            created->algorithm = std::string(cuntt::butterfly_backend_name(created->butterfly->config().backend)) +
                                 "-" + cuntt::compute_unit_name(created->butterfly->config().compute_unit);
            created->workspace_bytes = created->butterfly->workspace_size();
        }
        if (!measured && resolved.algorithm_policy == CUBUTTERFLY_ALGORITHM_DEFAULT && cached_algorithm.empty())
            log_profile_miss(handle, resolved, created->reason, created->algorithm);
        *plan = created.release();
    });
}

cubutterflyStatus_t cubutterflyDestroyPlan(cubutterflyPlan_t plan) { delete plan; return CUBUTTERFLY_STATUS_SUCCESS; }

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

cubutterflyStatus_t cubutterflyPlanSetWorkspace(cubutterflyPlan_t plan, void* workspace, size_t bytes) {
    return protect([&] {
        require(plan != nullptr, "plan must be non-null");
        if (plan->kind == cubutterflyPlan::Kind::DirectFft) {
            require(bytes == 0, "direct FFT does not require caller workspace");
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
        const bool in_place = plan->descriptor.placement == CUBUTTERFLY_PLACEMENT_IN_PLACE;
        require(in_place ? input == output : input != output,
                in_place ? "in-place execution requires identical pointers" : "out-of-place execution requires distinct pointers");
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
                cuntt::detail::launch_pack_2d(input, first, input_rows, input_cols, physical[0], physical[1],
                                              plan->descriptor.batch, plan->descriptor.input_strides[0],
                                              plan->descriptor.input_strides[1], plan->descriptor.input_batch_stride,
                                              plan->element_bytes, handle->stream);
            }
            void* packed_result = first;
            if (plan->kind == cubutterflyPlan::Kind::Ntt) {
                for (auto& axis_plan : plan->ntt_axes)
                    axis_plan->set_stream(handle->stream);
                void* ntt_destination = plan->direct_output_boundary && plan->descriptor.rank == 1
                                            ? output
                                            : static_cast<void*>(second);
                if (plan->descriptor.storage == CUBUTTERFLY_DATA_UINT32)
                    plan->ntt_axes[0]->execute_async(reinterpret_cast<const std::uint32_t*>(first), reinterpret_cast<std::uint32_t*>(ntt_destination));
                else
                    plan->ntt_axes[0]->execute_async(reinterpret_cast<const std::uint64_t*>(first), reinterpret_cast<std::uint64_t*>(ntt_destination));
                packed_result = ntt_destination;
                if (plan->descriptor.rank == 2) {
                    cuntt::detail::launch_transpose_2d(second, first, physical[0], physical[1],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    if (plan->descriptor.storage == CUBUTTERFLY_DATA_UINT32)
                        plan->ntt_axes[1]->execute_async(reinterpret_cast<const std::uint32_t*>(first), reinterpret_cast<std::uint32_t*>(second));
                    else
                        plan->ntt_axes[1]->execute_async(reinterpret_cast<const std::uint64_t*>(first), reinterpret_cast<std::uint64_t*>(second));
                    void* final_destination = plan->direct_output_boundary
                                                  ? output
                                                  : static_cast<void*>(first);
                    cuntt::detail::launch_transpose_2d(second, final_destination, physical[1], physical[0],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    packed_result = final_destination;
                }
            } else {
                for (auto& axis_plan : plan->butterfly_axes)
                    axis_plan->set_stream(handle->stream);
                void* butterfly_first = direct_butterfly_1d ? output : static_cast<void*>(first);
                if (plan->fused_input_boundary) {
                    plan->butterfly_axes[0]->execute_zero_extended_async(
                        static_cast<const float*>(input), static_cast<float*>(output),
                        logical[0], plan->descriptor.input_batch_stride,
                        plan->descriptor.input_strides[0]);
                } else {
                    execute_butterfly(*plan->butterfly_axes[0], butterfly_first, butterfly_first,
                                      plan->descriptor.storage);
                }
                packed_result = butterfly_first;
                if (plan->descriptor.rank == 2) {
                    cuntt::detail::launch_transpose_2d(first, second, physical[0], physical[1],
                                                       plan->descriptor.batch, plan->element_bytes, handle->stream);
                    execute_butterfly(*plan->butterfly_axes[1], second, second, plan->descriptor.storage);
                    void* final_destination = plan->direct_output_boundary
                                                  ? output
                                                  : static_cast<void*>(first);
                    cuntt::detail::launch_transpose_2d(second, final_destination, physical[1], physical[0],
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
            plan->ntt->set_stream(handle->stream);
            if (plan->ntt->config().word_bits == 32)
                plan->ntt->execute_async(static_cast<const std::uint32_t*>(input), static_cast<std::uint32_t*>(output));
            else
                plan->ntt->execute_async(static_cast<const std::uint64_t*>(input), static_cast<std::uint64_t*>(output));
        } else {
            plan->butterfly->set_stream(handle->stream);
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
            execute_butterfly(*plan->butterfly, input, output, storage);
        }
    });
}
