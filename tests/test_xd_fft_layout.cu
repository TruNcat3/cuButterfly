#include "../src/xd_real_fft.hpp"
#include "../src/xd_axis_fft.hpp"
#include "../src/xd_pair_fft.hpp"
#include "../src/xd_register_fft.hpp"
#include "../src/xd_cooperative_pair_fft.hpp"
#include "../src/xd_local_fft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream"); }
    ~Stream() {
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    operator cudaStream_t() const { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes_ != 0)
            check_cuda(cudaMalloc(&data_, bytes_), "cudaMalloc");
    }
    ~DeviceBuffer() {
        if (data_ != nullptr)
            cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() const { return data_; }
    std::size_t bytes() const { return bytes_; }

  private:
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

class CufftPlan {
  public:
    CufftPlan(int rank, std::vector<int> dimensions,
              std::vector<int> input_embed, int input_distance,
              std::vector<int> output_embed, int output_distance,
              cufftType type, int batch) {
        cufftHandle handle = 0;
        try {
            check_cufft(cufftCreate(&handle), "create native oracle plan");
            std::size_t work_bytes = 0;
            check_cufft(cufftMakePlanMany(handle, rank, dimensions.data(), input_embed.data(), 1,
                                          input_distance, output_embed.data(), 1, output_distance,
                                          type, batch, &work_bytes),
                        "make native oracle plan");
            handle_ = handle;
        } catch (...) {
            if (handle != 0)
                cufftDestroy(handle);
            throw;
        }
    }

    ~CufftPlan() {
        if (handle_ != 0)
            cufftDestroy(handle_);
    }

    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;

    cufftHandle handle() const { return handle_; }

    void set_stream(cudaStream_t stream) {
        check_cufft(cufftSetStream(handle_, stream), "set native oracle stream");
    }

  private:
    cufftHandle handle_ = 0;
};

struct Shape {
    std::vector<std::size_t> extents;
    const char* name;
};

struct Statistics {
    std::size_t cases = 0;
    std::size_t scalar_checks = 0;
    std::size_t input_checks = 0;
    std::size_t axis_cap_checks = 0;
    std::size_t axis_exchange_checks = 0;
    std::size_t axis_ept_checks = 0;
    std::size_t register_axis_checks = 0;
    std::size_t axis_resource_checks = 0;
    std::size_t axis_resource_invalid_checks = 0;
    std::size_t local_core_checks = 0;
    std::size_t local_direct_checks = 0;
    std::size_t pair_groups = 0;
    bool pair_observed = false;
    bool register_axis_observed = false;
    bool cooperative_pair_observed = false;
    double fp32_pair_max_relative_l2 = 0.0;
    double fp32_pair_max_absolute_error = 0.0;
    bool distribution_only = false;
    std::size_t threshold_failures = 0;
    std::size_t threshold_failed_scalars = 0;
};

struct PlanOptions {
    bool normalize_inverse = true;
    double output_scale = 1.0;
    bool packed_axes = true;
    bool fused_axes = false;
    unsigned axis_ept = 8;
    bool paired_axes = false;
    cuntt::detail::XdAxisCore axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
    cuntt::detail::XdPairCore pair_core = cuntt::detail::XdPairCore::Thread;
    const char* expected_core = nullptr;
    std::size_t expected_register_length = 0;
    bool fp64_reference = false;
    unsigned register_lanes = 4;
    unsigned pair_layout = 0;
    std::size_t batch_tile = 0;
    unsigned axis_ffts_per_block_cap = 0;
    unsigned axis_exchange_policy = 0;
    std::vector<cuntt::detail::XdAxisResources> axis_resources;
    bool fp64_diagnostics = false;
};

static_assert(!cuntt::detail::xd_axis_fft_u32_index_fits(0, 1, 1));
static_assert(!cuntt::detail::xd_axis_fft_u32_index_fits(128, 0, 1));
static_assert(cuntt::detail::xd_axis_fft_u32_index_fits(128, 1, 33554431));
static_assert(!cuntt::detail::xd_axis_fft_u32_index_fits(128, 1, 33554432));
static_assert(cuntt::detail::xd_axis_fft_u32_index_fits(64, 17, 51));
static_assert(!cuntt::detail::xd_axis_fft_u32_index_fits(64, 17, 50));
static_assert(!cuntt::detail::xd_axis_fft_u32_index_fits(
    128, static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) + 1, 1));

std::size_t product(const std::vector<std::size_t>& values) {
    return std::accumulate(values.begin(), values.end(), std::size_t{1},
                           [](std::size_t left, std::size_t right) { return left * right; });
}

std::vector<std::vector<std::size_t>> axis_orders(std::size_t rank) {
    std::vector<std::size_t> order(rank);
    std::iota(order.begin(), order.end(), 0);
    std::vector<std::vector<std::size_t>> result;
    do {
        result.push_back(order);
    } while (std::next_permutation(order.begin(), order.end()));
    return result;
}

std::vector<std::vector<std::size_t>> real_axis_orders(std::size_t rank) {
    if (rank == 2)
        return {{0}};
    if (rank == 3)
        return {{0, 1}, {1, 0}};
    throw std::invalid_argument("real oracle test only supports rank two and three");
}

std::string order_string(const std::vector<std::size_t>& order) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < order.size(); ++i)
        text << (i == 0 ? "" : ",") << order[i];
    text << ']';
    return text.str();
}

std::string shape_string(const std::vector<std::size_t>& extents) {
    std::ostringstream text;
    text << '[';
    for (std::size_t i = 0; i < extents.size(); ++i)
        text << (i == 0 ? "" : "x") << extents[i];
    text << ']';
    return text.str();
}

std::size_t mapping_pair_groups(const std::string& mapping) {
    constexpr const char* key = "\"pair_groups\":";
    const auto key_position = mapping.find(key);
    if (key_position == std::string::npos)
        throw std::runtime_error("xD FFT mapping is missing pair_groups");
    const auto value_begin = key_position + std::strlen(key);
    const auto value_end = mapping.find_first_not_of("0123456789", value_begin);
    if (value_end == value_begin)
        throw std::runtime_error("xD FFT mapping has an invalid pair_groups value");
    return std::stoull(mapping.substr(value_begin, value_end - value_begin));
}

bool mapping_has_core(const std::string& mapping, const char* core) {
    const std::string needle = std::string("\"core\":\"") + core + "\"";
    return mapping.find(needle) != std::string::npos;
}

std::vector<unsigned> mapping_unsigned_values(const std::string& mapping,
                                              const char* field) {
    const std::string needle = std::string("\"") + field + "\":";
    std::vector<unsigned> values;
    std::size_t search = 0;
    while (true) {
        const auto position = mapping.find(needle, search);
        if (position == std::string::npos)
            break;
        auto value_begin = position + needle.size();
        value_begin = mapping.find_first_not_of(" \t\r\n", value_begin);
        if (value_begin == std::string::npos ||
            value_begin == mapping.size() ||
            mapping[value_begin] < '0' || mapping[value_begin] > '9')
            throw std::runtime_error(std::string("xD FFT mapping has an invalid ") + field + " value");
        const auto value_end = mapping.find_first_not_of("0123456789", value_begin);
        const auto parsed = std::stoull(mapping.substr(value_begin, value_end - value_begin));
        if (parsed > std::numeric_limits<unsigned>::max())
            throw std::runtime_error(std::string("xD FFT mapping ") + field + " value overflows unsigned");
        values.push_back(static_cast<unsigned>(parsed));
        search = value_end;
    }
    return values;
}

std::vector<std::string> mapping_string_values(const std::string& mapping,
                                               const char* field) {
    const std::string needle = std::string("\"") + field + "\":\"";
    std::vector<std::string> values;
    std::size_t search = 0;
    while (true) {
        const auto position = mapping.find(needle, search);
        if (position == std::string::npos)
            break;
        const auto value_begin = position + needle.size();
        const auto value_end = mapping.find('\"', value_begin);
        if (value_end == std::string::npos)
            throw std::runtime_error(std::string("xD FFT mapping has an unterminated ") + field);
        values.push_back(mapping.substr(value_begin, value_end - value_begin));
        search = value_end + 1;
    }
    return values;
}

void check_register_axis_mapping(const std::string& mapping, unsigned lanes,
                                 std::size_t required_length,
                                 const std::string& label, Statistics& statistics) {
    const std::string needle = "\"core\":\"register-cooperative-axis\"";
    std::size_t search = 0;
    bool found = false;
    bool observed_required_length = required_length == 0;
    while (true) {
        const auto position = mapping.find(needle, search);
        if (position == std::string::npos)
            break;
        const auto object_begin = mapping.rfind('{', position);
        const auto object_end = mapping.find('}', position);
        if (object_begin == std::string::npos || object_end == std::string::npos ||
            object_begin >= object_end)
            throw std::runtime_error(label + ": malformed register axis mapping object");
        const std::string axis = mapping.substr(object_begin, object_end - object_begin + 1);
        const auto lengths = mapping_unsigned_values(axis, "length");
        const auto lane_values = mapping_unsigned_values(axis, "cooperative_lanes");
        const auto twiddles = mapping_string_values(axis, "twiddle_policy");
        if (lengths.size() != 1)
            throw std::runtime_error(label + ": register axis has no unique length");
        if (lane_values.size() != 1 || lane_values.front() != lanes)
            throw std::runtime_error(label + ": register axis cooperative_lanes does not match " +
                                     std::to_string(lanes));
        const std::size_t length = lengths.front();
        const char* expected_twiddle =
            length == 256 ? "constant-roots-256" :
            (length == 64 || length == 128) ? "constant-roots-128" : nullptr;
        if (expected_twiddle == nullptr || twiddles.size() != 1 ||
            twiddles.front() != expected_twiddle)
            throw std::runtime_error(label + ": register axis has an invalid twiddle policy for length " +
                                     std::to_string(length));
        found = true;
        observed_required_length = observed_required_length || length == required_length;
        search = object_end + 1;
    }
    if (!found)
        throw std::runtime_error(label + ": mapping has no register-cooperative-axis object");
    if (!observed_required_length)
        throw std::runtime_error(label + ": mapping has no register axis of required length " +
                                 std::to_string(required_length));
    ++statistics.register_axis_checks;
}

std::string mapping_axis_string(const std::string& axis, const char* field,
                                const std::string& label) {
    const auto values = mapping_string_values(axis, field);
    if (values.size() != 1)
        throw std::runtime_error(label + ": axis resource metadata has no unique " + field);
    return values.front();
}

unsigned mapping_axis_unsigned(const std::string& axis, const char* field,
                               const std::string& label) {
    const auto values = mapping_unsigned_values(axis, field);
    if (values.size() != 1)
        throw std::runtime_error(label + ": axis resource metadata has no unique " + field);
    return values.front();
}

std::vector<unsigned> mapping_axis_array_values(const std::string& axis, const char* field,
                                                const std::string& label) {
    const std::string needle = std::string("\"") + field + "\":[";
    const auto begin_position = axis.find(needle);
    if (begin_position == std::string::npos)
        throw std::runtime_error(label + ": axis resource metadata has no " + field + " array");
    const auto value_begin = begin_position + needle.size();
    const auto value_end = axis.find(']', value_begin);
    if (value_end == std::string::npos)
        throw std::runtime_error(label + ": axis resource metadata has an unterminated " + field + " array");
    std::vector<unsigned> values;
    std::size_t position = value_begin;
    while (position < value_end) {
        position = axis.find_first_not_of(" \t\r\n,", position);
        if (position == std::string::npos || position >= value_end)
            break;
        if (axis[position] < '0' || axis[position] > '9')
            throw std::runtime_error(label + ": axis resource metadata has an invalid " + field + " array");
        const auto number_end = axis.find_first_not_of("0123456789", position);
        const auto parsed = std::stoull(axis.substr(position, number_end - position));
        if (parsed > std::numeric_limits<unsigned>::max())
            throw std::runtime_error(label + ": axis resource metadata " + field + " value overflows unsigned");
        values.push_back(static_cast<unsigned>(parsed));
        position = number_end;
        position = axis.find_first_not_of(" \t\r\n", position);
        if (position < value_end && axis[position] != ',')
            throw std::runtime_error(label + ": axis resource metadata has malformed " + field + " array");
    }
    return values;
}

std::string expected_axis_resource_core(cuntt::detail::XdAxisCore core) {
    if (core == cuntt::detail::XdAxisCore::CooperativeRegister)
        return "register-cooperative-axis";
    if (core == cuntt::detail::XdAxisCore::ThreadShared)
        return "thread-shared-axis";
    return "cufftdx-axis-tiled";
}

std::string expected_axis_resource_metadata_core(cuntt::detail::XdAxisCore core) {
    if (core == cuntt::detail::XdAxisCore::CooperativeRegister)
        return "cooperative-register";
    if (core == cuntt::detail::XdAxisCore::ThreadShared)
        return "thread-shared";
    return "cufftdx-block";
}

void check_axis_resource_vector(
    const std::string& mapping,
    const std::vector<cuntt::detail::XdAxisResources>& expected,
    const std::string& label) {
    constexpr const char* key = "\"axis_resources\":[";
    const auto key_position = mapping.find(key);
    if (key_position == std::string::npos)
        throw std::runtime_error(label + ": mapping is missing axis_resources metadata");
    const auto array_begin = key_position + std::strlen(key);
    const auto array_end = mapping.find(']', array_begin);
    if (array_end == std::string::npos)
        throw std::runtime_error(label + ": axis_resources metadata is not closed");

    const std::string resources = mapping.substr(array_begin, array_end - array_begin);
    std::size_t search = 0;
    std::size_t index = 0;
    while (true) {
        const auto object_begin = resources.find('{', search);
        if (object_begin == std::string::npos)
            break;
        const auto object_end = resources.find('}', object_begin);
        if (object_end == std::string::npos)
            throw std::runtime_error(label + ": malformed axis_resources metadata object");
        if (index >= expected.size())
            throw std::runtime_error(label + ": mapping contains too many axis resources");
        const std::string resource = resources.substr(object_begin,
                                                       object_end - object_begin + 1);
        const auto& wanted = expected[index];
        if (mapping_axis_string(resource, "core", label) !=
            expected_axis_resource_metadata_core(wanted.core) ||
            mapping_axis_unsigned(resource, "ept", label) != wanted.ept ||
            mapping_axis_unsigned(resource, "register_lanes", label) != wanted.register_lanes ||
            mapping_axis_unsigned(resource, "block_cap", label) != wanted.block_cap ||
            mapping_axis_unsigned(resource, "exchange", label) != wanted.exchange) {
            throw std::runtime_error(label + ": axis_resources metadata does not match logical axis " +
                                     std::to_string(index));
        }
        const bool has_local_twiddle_policy =
            resource.find("\"local_twiddle_policy\":") != std::string::npos;
        if (wanted.local_twiddle_policy == 0) {
            if (has_local_twiddle_policy)
                throw std::runtime_error(label + ": policy-zero resource serialized local_twiddle_policy");
        } else {
            if (wanted.core != cuntt::detail::XdAxisCore::ThreadShared ||
                !has_local_twiddle_policy ||
                mapping_axis_unsigned(resource, "local_twiddle_policy", label) !=
                    wanted.local_twiddle_policy)
                throw std::runtime_error(label + ": local_twiddle_policy does not match logical axis " +
                                         std::to_string(index));
        }
        ++index;
        search = object_end + 1;
    }
    if (index != expected.size())
        throw std::runtime_error(label + ": axis_resources metadata has the wrong length");
}

void check_axis_resource_mapping(
    const std::string& mapping,
    const std::vector<cuntt::detail::XdAxisResources>& expected,
    const std::string& label, Statistics& statistics) {
    if (expected.empty())
        throw std::invalid_argument(label + ": axis resource mapping check needs resources");
    check_axis_resource_vector(mapping, expected, label);
    const bool expects_local_core = std::any_of(
        expected.begin(), expected.end(), [](const auto& resource) {
            return resource.core == cuntt::detail::XdAxisCore::ThreadShared;
        });
    if (expects_local_core && mapping.find("\"axis_core\":\"thread-shared\"") ==
                                  std::string::npos)
        throw std::runtime_error(label + ": mapping has no thread-shared axis_core");
    std::vector<bool> observed(expected.size(), false);
    std::size_t object_count = 0;
    const std::string needle = "\"axis\":";
    std::size_t search = 0;
    while (true) {
        const auto position = mapping.find(needle, search);
        if (position == std::string::npos)
            break;
        const auto object_begin = mapping.rfind('{', position);
        const auto object_end = mapping.find('}', position);
        if (object_begin == std::string::npos || object_end == std::string::npos ||
            object_begin >= object_end)
            throw std::runtime_error(label + ": malformed axis resource metadata object");
        const std::string axis = mapping.substr(object_begin, object_end - object_begin + 1);
        const auto logical_axis = mapping_axis_unsigned(axis, "axis", label);
        if (logical_axis >= expected.size())
            throw std::runtime_error(label + ": mapping contains an unexpected logical axis " +
                                     std::to_string(logical_axis));
        const auto& resource = expected[logical_axis];
        const auto core = mapping_axis_string(axis, "core", label);
        if (core != expected_axis_resource_core(resource.core))
            throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                     " used core " + core + ", expected " +
                                     expected_axis_resource_core(resource.core));
        const auto length = mapping_axis_unsigned(axis, "length", label);
        const auto suffix = mapping_axis_unsigned(axis, "axis_suffix", label);
        const auto elements_per_thread = mapping_axis_unsigned(axis, "elements_per_thread", label);
        const auto cooperative_lanes = mapping_axis_unsigned(axis, "cooperative_lanes", label);
        const auto ffts_per_block = mapping_axis_unsigned(axis, "ffts_per_block", label);
        if (resource.core == cuntt::detail::XdAxisCore::CooperativeRegister) {
            if (cooperative_lanes != resource.register_lanes)
                throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                         " has cooperative_lanes=" +
                                         std::to_string(cooperative_lanes));
            if (resource.register_lanes == 0 ||
                elements_per_thread != length / resource.register_lanes)
                throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                         " has an invalid register elements_per_thread value");
            if (axis.find("\"twiddle_policy\":\"constant-roots-") ==
                std::string::npos)
                throw std::runtime_error(label + ": register axis " +
                                         std::to_string(logical_axis) +
                                         " is missing its constant twiddle policy");
        } else if (resource.core == cuntt::detail::XdAxisCore::ThreadShared) {
            if (length != 64 && length != 128 && length != 256)
                throw std::runtime_error(label + ": thread-shared axis has unsupported length " +
                                         std::to_string(length));
            const bool fp64 = mapping.find("\"fp64\":true") != std::string::npos;
            const unsigned expected_ept = length == 64 ? 8U :
                                          static_cast<unsigned>(length / 16);
            const unsigned expected_lines = cuntt::detail::xd_local_fft_lines_per_block(
                length, fp64, resource.block_cap);
            if (cooperative_lanes != 0 || elements_per_thread != expected_ept ||
                ffts_per_block != expected_lines)
                throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                         " has unexpected thread-shared resource values");
            const auto twiddle = mapping_axis_string(axis, "twiddle_policy", label);
            const std::string expected_twiddle_root = resource.local_twiddle_policy == 1 ?
                "recurrence-root" : "constant-roots";
            const std::string expected_twiddle = expected_twiddle_root +
                (length == 256 ? "-256" : length == 64 ? "-64" : "-128");
            if (twiddle != expected_twiddle)
                throw std::runtime_error(label + ": thread-shared axis " +
                                         std::to_string(logical_axis) +
                                         " has unexpected twiddle_policy=" + twiddle);
            if (mapping_axis_string(axis, "io_exchange", label) != "soa-xor-local")
                throw std::runtime_error(label + ": thread-shared axis " +
                                         std::to_string(logical_axis) +
                                         " has unexpected io_exchange");
            const std::vector<unsigned> expected_factors = {
                length == 64 ? 8U : 16U,
                length == 64 ? 8U : static_cast<unsigned>(length / 16)};
            if (mapping_axis_array_values(axis, "local_factors", label) != expected_factors)
                throw std::runtime_error(label + ": thread-shared axis " +
                                         std::to_string(logical_axis) +
                                         " has unexpected local_factors");
        } else {
            const unsigned expected_ept = cuntt::detail::xd_axis_fft_ept(length, resource.ept);
            if (cooperative_lanes != 0 || elements_per_thread != expected_ept)
                throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                         " has unexpected block resource values");
            const unsigned expected_ffts = cuntt::detail::xd_axis_fft_block_batch(
                length, mapping.find("\"fp64\":true") != std::string::npos,
                resource.ept, resource.block_cap);
            if (ffts_per_block != expected_ffts)
                throw std::runtime_error(label + ": axis " + std::to_string(logical_axis) +
                                         " has unexpected ffts_per_block");
            if (resource.exchange != 0 && suffix > 1) {
                const auto exchange = mapping_axis_string(axis, "io_exchange", label);
                const std::string expected_exchange = resource.exchange == 1 ?
                    "direct-strided-register" : resource.exchange == 2 ?
                    "unrolled-staged" : resource.exchange == 3 ?
                    "unrolled-staged-u32" : resource.exchange == 4 ?
                    "packed-contiguous-register" : "";
                if (expected_exchange.empty() || exchange != expected_exchange)
                    throw std::runtime_error(label + ": axis " +
                                             std::to_string(logical_axis) +
                                             " has unexpected io_exchange=" + exchange);
            }
        }
        observed[logical_axis] = true;
        ++object_count;
        search = object_end + 1;
    }
    if (object_count == 0)
        throw std::runtime_error(label + ": mapping has no per-axis resource objects");
    for (std::size_t axis = 0; axis < observed.size(); ++axis) {
        if (!observed[axis])
            throw std::runtime_error(label + ": mapping is missing logical axis " +
                                     std::to_string(axis));
    }
    if (expects_local_core)
        ++statistics.local_core_checks;
    ++statistics.axis_resource_checks;
}

void check_axis_block_cap_mapping(const std::string& mapping, unsigned cap,
                                  const std::string& label, Statistics& statistics,
                                  bool require_observed_cap) {
    const auto cap_values = mapping_unsigned_values(mapping, "axis_ffts_per_block_cap");
    if (cap_values.empty())
        throw std::runtime_error(label + ": mapping is missing axis_ffts_per_block_cap");
    for (const auto actual_cap : cap_values) {
        if (actual_cap != cap)
            throw std::runtime_error(label + ": mapping has axis_ffts_per_block_cap=" +
                                     std::to_string(actual_cap) + ", expected " +
                                     std::to_string(cap));
    }

    const auto ffts_per_block = mapping_unsigned_values(mapping, "ffts_per_block");
    if (ffts_per_block.empty())
        throw std::runtime_error(label + ": mapping has no per-axis ffts_per_block values");
    bool observed_cap = false;
    for (const auto actual : ffts_per_block) {
        if (actual > cap)
            throw std::runtime_error(label + ": per-axis ffts_per_block=" +
                                     std::to_string(actual) + " exceeds cap " +
                                     std::to_string(cap));
        observed_cap = observed_cap || actual == cap;
    }
    if (require_observed_cap && !observed_cap)
        throw std::runtime_error(label + ": no axis reached the requested ffts_per_block cap");
    ++statistics.axis_cap_checks;
}

void check_axis_exchange_mapping(const std::string& mapping, unsigned policy,
                                 const std::string& label, Statistics& statistics,
                                 bool require_contiguous) {
    if (policy < 1U || policy > 4U)
        throw std::invalid_argument("xD axis exchange test requires policy 1 through 4");
    const auto policy_values = mapping_unsigned_values(mapping, "axis_exchange_policy");
    if (policy_values.empty())
        throw std::runtime_error(label + ": mapping is missing axis_exchange_policy");
    for (const auto actual : policy_values) {
        if (actual != policy)
            throw std::runtime_error(label + ": mapping has axis_exchange_policy=" +
                                     std::to_string(actual) + ", expected " +
                                     std::to_string(policy));
    }

    const std::string expected = policy == 1U ? "direct-strided-register" :
                                 policy == 2U ? "unrolled-staged" :
                                 policy == 3U ? "unrolled-staged-u32" : "packed-contiguous-register";
    const auto exchanges = mapping_string_values(mapping, "io_exchange");
    if (exchanges.empty())
        throw std::runtime_error(label + ": mapping has no per-axis io_exchange values");
    bool observed_expected = false;
    bool observed_contiguous = false;
    for (const auto& exchange : exchanges) {
        if (exchange == expected)
            observed_expected = true;
        else if (exchange == "contiguous-register")
            observed_contiguous = true;
        else
            throw std::runtime_error(label + ": unknown io_exchange=" + exchange);
    }
    if (!observed_expected)
        throw std::runtime_error(label + ": no axis used the requested io_exchange policy");
    if (require_contiguous && policy != 4 && !observed_contiguous)
        throw std::runtime_error(label + ": mapping has no contiguous-register axis");
    if (policy == 4) {
        const auto core_suffixes = mapping_unsigned_values(mapping, "core_suffix");
        const auto core_policies = mapping_unsigned_values(mapping, "core_exchange_policy");
        if (core_suffixes.empty() || core_policies.size() != core_suffixes.size() ||
            std::any_of(core_suffixes.begin(), core_suffixes.end(),
                        [](unsigned suffix) { return suffix != 1; }) ||
            std::any_of(core_policies.begin(), core_policies.end(),
                        [](unsigned exchange) { return exchange != 0; }))
            throw std::runtime_error(label + ": packed core did not use contiguous policy 0");
    }
    ++statistics.axis_exchange_checks;
}

void check_axis_ept_mapping(const std::string& mapping, unsigned ept,
                            const std::string& label, Statistics& statistics) {
    const auto values = mapping_unsigned_values(mapping, "axis_ept");
    if (values.empty())
        throw std::runtime_error(label + ": mapping is missing axis_ept");
    for (const auto actual : values) {
        if (actual != ept)
            throw std::runtime_error(label + ": mapping has axis_ept=" +
                                     std::to_string(actual) + ", expected " +
                                     std::to_string(ept));
    }
    ++statistics.axis_ept_checks;
}

template <typename Real>
Real deterministic_value(std::size_t index, std::size_t batch, std::size_t lane) {
    // The distinct per-axis and per-batch terms make accidental transposes visible
    // without introducing a dependency on the implementation's address arithmetic.
    const std::uint64_t mixed = (static_cast<std::uint64_t>(index) * 0x9e3779b97f4a7c15ULL) ^
                                (static_cast<std::uint64_t>(batch) * 0xd1b54a32d192ed03ULL) ^
                                (static_cast<std::uint64_t>(lane) * 0x94d049bb133111ebULL);
    const double unit = static_cast<double>((mixed >> 11) & 0xfffff) / 1048575.0 - 0.5;
    return static_cast<Real>(unit + 0.03125 * static_cast<double>((index + 3 * batch + lane) % 17));
}

template <typename Real, typename Complex>
void fill_complex(std::vector<Complex>& values, std::size_t points, std::size_t batch) {
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < points; ++i) {
            values[b * points + i].x = deterministic_value<Real>(i, b, 0);
            values[b * points + i].y = deterministic_value<Real>(i, b, 1);
        }
    }
}

template <typename Real>
void fill_real(std::vector<Real>& values, std::size_t points, std::size_t batch) {
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t i = 0; i < points; ++i)
            values[b * points + i] = deterministic_value<Real>(i, b, 2);
}

void report_numeric_distribution(std::vector<double>& errors, double error_squared,
                                  double reference_squared, std::size_t failures,
                                  double worst_ratio, const std::string& label) {
    std::sort(errors.begin(), errors.end());
    const auto quantile = [&](double fraction) {
        return errors[static_cast<std::size_t>(fraction * (errors.size() - 1))];
    };
    const double relative_l2 = reference_squared == 0.0
        ? (error_squared == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
        : std::sqrt(error_squared / reference_squared);
    std::cout << std::setprecision(17) << label << ": distribution-only"
              << " relative_l2=" << relative_l2 << " abs_error_p50=" << quantile(0.50)
              << " p90=" << quantile(0.90) << " p99=" << quantile(0.99)
              << " p999=" << quantile(0.999) << " max=" << errors.back()
              << " threshold_failed_scalars=" << failures << " worst_ratio=" << worst_ratio << '\n';
}

template <typename Real, typename Complex>
void compare_complex(const std::vector<Complex>& actual, const std::vector<Complex>& expected,
                     const std::string& label, Statistics& statistics) {
    const double relative_tolerance = std::is_same_v<Real, double> ? 5e-11 : 2e-4;
    const double absolute_tolerance = std::is_same_v<Real, double> ? 5e-11 : 3e-4;
    double max_error = 0.0;
    double max_reference = 0.0;
    double worst_ratio = 0.0;
    std::size_t worst = 0;
    char worst_component = 'x';
    std::vector<double> errors;
    double error_squared = 0.0, reference_squared = 0.0;
    std::size_t failures = 0;
    if (statistics.distribution_only)
        errors.reserve(actual.size() * 2);
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double components[2] = {static_cast<double>(actual[i].x) - static_cast<double>(expected[i].x),
                                      static_cast<double>(actual[i].y) - static_cast<double>(expected[i].y)};
        const double references[2] = {static_cast<double>(expected[i].x), static_cast<double>(expected[i].y)};
        for (unsigned component = 0; component < 2; ++component) {
            const double error = std::abs(components[component]);
            const double reference = std::abs(references[component]);
            const double limit = absolute_tolerance + relative_tolerance * std::max(1.0, reference);
            ++statistics.scalar_checks;
            const double ratio = std::isfinite(error) ? error / limit
                                                       : std::numeric_limits<double>::infinity();
            if (statistics.distribution_only) {
                errors.push_back(std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
                error_squared += error * error;
                reference_squared += reference * reference;
                failures += ratio > 1.0;
            }
            if (ratio > worst_ratio) {
                worst_ratio = ratio;
                max_error = error;
                max_reference = reference;
                worst = i;
                worst_component = component == 0 ? 'x' : 'y';
            }
        }
    }
    if (statistics.distribution_only) {
        statistics.threshold_failures += failures != 0;
        statistics.threshold_failed_scalars += failures;
        report_numeric_distribution(errors, error_squared, reference_squared,
                                                   failures, worst_ratio, label);
        return;
    }
    if (worst_ratio > 1.0) {
        std::ostringstream message;
        message << label << ": mismatch at element " << worst << worst_component
                << ", error=" << max_error << ", reference=" << max_reference;
        throw std::runtime_error(message.str());
    }
}

template <typename Real, typename Complex>
void report_real_fp64_diagnostics(const std::vector<Complex>& actual,
                                  const std::vector<Complex>& native,
                                  const std::vector<cufftDoubleComplex>& reference,
                                  const std::string& label) {
    const auto report = [&](const char* comparison, const auto& measured, const auto& oracle) {
        std::vector<double> errors;
        errors.reserve(2 * measured.size());
        double error_squared = 0.0, reference_squared = 0.0;
        for (std::size_t i = 0; i < measured.size(); ++i) {
            for (unsigned component = 0; component < 2; ++component) {
                const double value = component ? measured[i].y : measured[i].x;
                const double target = component ? oracle[i].y : oracle[i].x;
                const double error = std::abs(value - target);
                errors.push_back(std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
                error_squared += error * error;
                reference_squared += target * target;
            }
        }
        std::sort(errors.begin(), errors.end());
        const auto quantile = [&](double fraction) {
            return errors[static_cast<std::size_t>(fraction * (errors.size() - 1))];
        };
        std::cout << std::setprecision(17) << label << ": " << comparison
                  << " relative_l2=" << std::sqrt(error_squared / reference_squared)
                  << " abs_error_p50=" << quantile(0.50)
                  << " p90=" << quantile(0.90) << " p99=" << quantile(0.99)
                  << " p999=" << quantile(0.999) << " max=" << errors.back() << '\n';
    };
    report("xD_vs_fp64", actual, reference);
    report("native_fp32_vs_fp64", native, reference);
    report("xD_vs_native_fp32", actual, native);
    const double relative_tolerance = std::is_same_v<Real, double> ? 5e-11 : 2e-4;
    const double absolute_tolerance = std::is_same_v<Real, double> ? 5e-11 : 3e-4;
    std::size_t failures = 0, worst = 0;
    unsigned worst_component = 0;
    double worst_ratio = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        for (unsigned component = 0; component < 2; ++component) {
            const double value = component ? actual[i].y : actual[i].x;
            const double target = component ? native[i].y : native[i].x;
            const double limit = absolute_tolerance + relative_tolerance * std::max(1.0, std::abs(target));
            const double error = std::abs(value - target);
            const double ratio = std::isfinite(error) ? error / limit : std::numeric_limits<double>::infinity();
            failures += ratio > 1.0;
            if (ratio > worst_ratio) {
                worst_ratio = ratio;
                worst = i;
                worst_component = component;
            }
        }
    }
    std::cout << std::setprecision(17) << label << ": native_threshold_failures=" << failures
              << " worst_ratio=" << worst_ratio << " index=" << worst
              << (worst_component ? 'y' : 'x')
              << " actual=" << (worst_component ? actual[worst].y : actual[worst].x)
              << " native=" << (worst_component ? native[worst].y : native[worst].x)
              << " fp64=" << (worst_component ? reference[worst].y : reference[worst].x) << '\n';
    const char* dump_path = std::getenv("CUBUTTERFLY_XD_ERROR_DUMP");
    if (failures && dump_path && *dump_path) {
        std::ofstream dump(dump_path);
        if (!dump)
            throw std::runtime_error("cannot open xD error diagnostic CSV: " + std::string(dump_path));
        dump << "index,component,actual,native_fp32,fp64,actual_minus_fp64,native_minus_fp64,actual_minus_native\n"
             << std::setprecision(17);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            for (unsigned component = 0; component < 2; ++component) {
                const double value = component ? actual[i].y : actual[i].x;
                const double target = component ? native[i].y : native[i].x;
                const double truth = component ? reference[i].y : reference[i].x;
                dump << i << ',' << (component ? 'y' : 'x') << ',' << value << ',' << target
                     << ',' << truth << ',' << value - truth << ',' << target - truth
                     << ',' << value - target << '\n';
            }
        }
        if (!dump)
            throw std::runtime_error("cannot write xD error diagnostic CSV: " + std::string(dump_path));
        std::cout << label << ": full_error_csv=" << dump_path << '\n';
    }
}

template <typename Real, typename Complex>
void compare_complex_fp64_norm(const std::vector<Complex>& actual,
                               const std::vector<cufftDoubleComplex>& expected,
                               std::size_t transform_points, const std::string& label,
                               Statistics& statistics) {
    // A cancellation-dominated bin can have a large relative error even when
    // the transform's overall forward error is at the expected O(epsilon log N)
    // level. Use an FP64 oracle and a norm-wise error check in that case;
    // a wrong axis permutation produces a large error norm on this input.
    const double unit_roundoff =
        0.5 * static_cast<double>(std::numeric_limits<Real>::epsilon());
    const auto stages = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(std::log2(static_cast<double>(transform_points)))));
    const double stage_roundoff = static_cast<double>(stages) * unit_roundoff;
    const double tolerance = 8.0 * stage_roundoff / (1.0 - stage_roundoff);
    double error_squared = 0.0;
    double reference_squared = 0.0;
    double max_error = 0.0;
    std::size_t max_error_index = 0;
    std::vector<double> errors;
    if (statistics.distribution_only)
        errors.reserve(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::hypot(static_cast<double>(actual[i].x) - expected[i].x,
                                        static_cast<double>(actual[i].y) - expected[i].y);
        const double reference = std::hypot(expected[i].x, expected[i].y);
        error_squared += error * error;
        reference_squared += reference * reference;
        if (statistics.distribution_only)
            errors.push_back(std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
        if (error > max_error) {
            max_error = error;
            max_error_index = i;
        }
    }
    const double relative_error = std::sqrt(error_squared / reference_squared);
    statistics.fp32_pair_max_relative_l2 =
        std::max(statistics.fp32_pair_max_relative_l2, relative_error);
    statistics.fp32_pair_max_absolute_error =
        std::max(statistics.fp32_pair_max_absolute_error, max_error);
    statistics.scalar_checks += actual.size() * 2;
    if (statistics.distribution_only) {
        const bool failed = !std::isfinite(relative_error) || relative_error > tolerance;
        statistics.threshold_failures += failed;
        report_numeric_distribution(errors, error_squared, reference_squared,
                                                   0, relative_error / tolerance, label);
        std::cout << label << ": fp64_norm_threshold_failure=" << failed << '\n';
        return;
    }
    if (!std::isfinite(relative_error) || relative_error > tolerance) {
        std::ostringstream message;
        message << label << ": FP64-reference L2 relative error=" << relative_error
                << ", tolerance=" << tolerance << ", max error=" << max_error
                << " at element " << max_error_index;
        throw std::runtime_error(message.str());
    }
}

template <typename Real>
void compare_real(const std::vector<Real>& actual, const std::vector<Real>& expected,
                  const std::string& label, Statistics& statistics) {
    const double relative_tolerance = std::is_same_v<Real, double> ? 5e-11 : 2e-4;
    const double absolute_tolerance = std::is_same_v<Real, double> ? 5e-11 : 3e-4;
    double max_error = 0.0;
    double max_reference = 0.0;
    double worst_ratio = 0.0;
    std::size_t worst = 0;
    std::vector<double> errors;
    double error_squared = 0.0, reference_squared = 0.0;
    std::size_t failures = 0;
    if (statistics.distribution_only)
        errors.reserve(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
        const double reference = std::abs(static_cast<double>(expected[i]));
        const double limit = absolute_tolerance + relative_tolerance * std::max(1.0, reference);
        ++statistics.scalar_checks;
        const double ratio = std::isfinite(error) ? error / limit
                                                   : std::numeric_limits<double>::infinity();
        if (statistics.distribution_only) {
            errors.push_back(std::isfinite(error) ? error : std::numeric_limits<double>::infinity());
            error_squared += error * error;
            reference_squared += reference * reference;
            failures += ratio > 1.0;
        }
        if (ratio > worst_ratio) {
            worst_ratio = ratio;
            max_error = error;
            max_reference = reference;
            worst = i;
        }
    }
    if (statistics.distribution_only) {
        statistics.threshold_failures += failures != 0;
        statistics.threshold_failed_scalars += failures;
        report_numeric_distribution(errors, error_squared, reference_squared,
                                                failures, worst_ratio, label);
        return;
    }
    if (worst_ratio > 1.0) {
        std::ostringstream message;
        message << label << ": mismatch at element " << worst
                << ", error=" << max_error << ", reference=" << max_reference;
        throw std::runtime_error(message.str());
    }
}

template <typename Value>
void require_input_unchanged(const void* device_input, const std::vector<Value>& expected,
                             const std::string& label, Statistics& statistics) {
    std::vector<Value> actual(expected.size());
    check_cuda(cudaMemcpy(actual.data(), device_input, actual.size() * sizeof(Value), cudaMemcpyDeviceToHost),
               "read back xD input");
    ++statistics.input_checks;
    if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(Value)) != 0)
        throw std::runtime_error(label + ": input was modified");
}

template <typename Real, typename Complex>
CufftPlan make_complex_oracle(const std::vector<std::size_t>& extents, std::size_t batch,
                              bool inverse) {
    std::vector<int> dimensions(extents.size());
    for (std::size_t i = 0; i < extents.size(); ++i)
        dimensions[i] = static_cast<int>(extents[i]);
    const int points = static_cast<int>(product(extents));
    const cufftType type = std::is_same_v<Real, double> ? CUFFT_Z2Z : CUFFT_C2C;
    (void)inverse;
    return CufftPlan(static_cast<int>(extents.size()), dimensions, dimensions, points,
                     dimensions, points, type, static_cast<int>(batch));
}

template <typename Real, typename Complex>
void run_complex_case(const Shape& shape, std::size_t batch,
                      const std::vector<std::size_t>& order, bool inverse,
                      Statistics& statistics, const PlanOptions& options = {},
                      bool expect_pair = false) {
    const std::size_t points = product(shape.extents);
    const std::size_t count = points * batch;
    const bool fp64 = std::is_same_v<Real, double>;
    const std::string label = std::string("complex ") + (fp64 ? "fp64 " : "fp32 ") +
                              (inverse ? "inverse " : "forward ") + shape.name +
                              " extents=" + shape_string(shape.extents) +
                              " order=" + order_string(order);
    std::vector<Complex> input(count), expected(count), first(count), second(count);
    fill_complex<Real>(input, points, batch);

    Stream stream;
    cuntt::detail::XdRealFftPlan plan(shape.extents, batch, inverse, options.normalize_inverse, fp64, true,
                                      options.output_scale, options.packed_axes, order,
                                      options.fused_axes, options.axis_ept,
                                      options.paired_axes, options.axis_core, options.pair_core,
                                      options.register_lanes, options.pair_layout, options.batch_tile,
                                      options.axis_ffts_per_block_cap,
                                      options.axis_exchange_policy,
                                      options.axis_resources);
    const std::string mapping = plan.mapping_json();
    if (!options.axis_resources.empty())
        check_axis_resource_mapping(mapping, options.axis_resources, label, statistics);
    if (options.axis_ffts_per_block_cap != 0)
        check_axis_block_cap_mapping(
            mapping, options.axis_ffts_per_block_cap, label, statistics,
            std::any_of(shape.extents.begin(), shape.extents.end(),
                        [](std::size_t extent) { return extent == 64; }));
    if (options.axis_exchange_policy != 0)
        check_axis_ept_mapping(mapping, options.axis_ept, label, statistics);
    if (options.axis_exchange_policy != 0)
        check_axis_exchange_mapping(mapping, options.axis_exchange_policy, label,
                                    statistics, true);
    const std::size_t pair_groups = mapping_pair_groups(mapping);
    statistics.pair_groups += pair_groups;
    statistics.pair_observed = statistics.pair_observed || pair_groups != 0;
    if (options.expected_core != nullptr) {
        if (!mapping_has_core(mapping, options.expected_core))
            throw std::runtime_error(label + ": expected mapping core " + options.expected_core);
        if (std::strcmp(options.expected_core, "register-cooperative-axis") == 0)
            statistics.register_axis_observed = true;
        else if (std::strcmp(options.expected_core, "register-cooperative-pair") == 0)
            statistics.cooperative_pair_observed = true;
        if (std::strcmp(options.expected_core, "register-cooperative-axis") == 0)
            check_register_axis_mapping(mapping, options.register_lanes,
                                         options.expected_register_length, label, statistics);
    }
    if (expect_pair && pair_groups == 0)
        throw std::runtime_error(label + ": expected a fused pair group");
    const bool use_fp64_reference = !fp64 && (pair_groups != 0 || options.fp64_reference);
    DeviceBuffer d_input(count * sizeof(Complex));
    DeviceBuffer d_output(count * sizeof(Complex));
    DeviceBuffer d_oracle_input(count * sizeof(Complex));
    DeviceBuffer d_oracle_output(count * sizeof(Complex));
    DeviceBuffer d_fp64_oracle_input(use_fp64_reference ? count * sizeof(cufftDoubleComplex) : 0);
    DeviceBuffer d_fp64_oracle_output(use_fp64_reference ? count * sizeof(cufftDoubleComplex) : 0);
    DeviceBuffer workspace_a(plan.workspace_size());
    DeviceBuffer workspace_b(plan.workspace_size());
    check_cuda(cudaMemcpy(d_input.data(), input.data(), d_input.bytes(), cudaMemcpyHostToDevice),
               "upload xD complex input");
    check_cuda(cudaMemcpy(d_oracle_input.data(), input.data(), d_oracle_input.bytes(), cudaMemcpyHostToDevice),
               "upload native complex input");

    CufftPlan oracle = make_complex_oracle<Real, Complex>(shape.extents, batch, inverse);
    oracle.set_stream(stream);
    if (fp64) {
        check_cufft(cufftExecZ2Z(oracle.handle(), static_cast<cufftDoubleComplex*>(d_oracle_input.data()),
                                 static_cast<cufftDoubleComplex*>(d_oracle_output.data()),
                                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                    "execute native Z2Z oracle");
    } else {
        check_cufft(cufftExecC2C(oracle.handle(), static_cast<cufftComplex*>(d_oracle_input.data()),
                                 static_cast<cufftComplex*>(d_oracle_output.data()),
                                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                    "execute native C2C oracle");
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize native complex oracle");
    check_cuda(cudaMemcpy(expected.data(), d_oracle_output.data(), d_oracle_output.bytes(), cudaMemcpyDeviceToHost),
               "download native complex oracle");
    if (inverse && options.normalize_inverse) {
        const Real scale = static_cast<Real>(1.0 / static_cast<double>(points));
        for (auto& value : expected) {
            value.x *= scale;
            value.y *= scale;
        }
    }
    if (options.output_scale != 1.0) {
        const Real scale = static_cast<Real>(options.output_scale);
        for (auto& value : expected) {
            value.x *= scale;
            value.y *= scale;
        }
    }
    std::vector<cufftDoubleComplex> expected_fp64;
    if (use_fp64_reference) {
        std::vector<cufftDoubleComplex> input_fp64(count);
        expected_fp64.resize(count);
        for (std::size_t i = 0; i < count; ++i) {
            input_fp64[i].x = static_cast<double>(input[i].x);
            input_fp64[i].y = static_cast<double>(input[i].y);
        }
        check_cuda(cudaMemcpy(d_fp64_oracle_input.data(), input_fp64.data(),
                              d_fp64_oracle_input.bytes(), cudaMemcpyHostToDevice),
                   "upload native fp64 complex reference input");
        CufftPlan fp64_oracle = make_complex_oracle<double, cufftDoubleComplex>(shape.extents, batch, inverse);
        fp64_oracle.set_stream(stream);
        check_cufft(cufftExecZ2Z(fp64_oracle.handle(),
                                 static_cast<cufftDoubleComplex*>(d_fp64_oracle_input.data()),
                                 static_cast<cufftDoubleComplex*>(d_fp64_oracle_output.data()),
                                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                    "execute native fp64 complex reference");
        check_cuda(cudaStreamSynchronize(stream), "synchronize native fp64 complex reference");
        check_cuda(cudaMemcpy(expected_fp64.data(), d_fp64_oracle_output.data(),
                              d_fp64_oracle_output.bytes(), cudaMemcpyDeviceToHost),
                   "download native fp64 complex reference");
        if (inverse && options.normalize_inverse) {
            const double scale = 1.0 / static_cast<double>(points);
            for (auto& value : expected_fp64) {
                value.x *= scale;
                value.y *= scale;
            }
        }
        if (options.output_scale != 1.0) {
            for (auto& value : expected_fp64) {
                value.x *= options.output_scale;
                value.y *= options.output_scale;
            }
        }
    }

    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace_a.data(), workspace_a.bytes());
    else
        plan.set_workspace(nullptr, 0);
    plan.execute(d_input.data(), d_output.data(), stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize xD complex first execution");
    check_cuda(cudaMemcpy(first.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
               "download xD complex first execution");
    if (use_fp64_reference)
        compare_complex_fp64_norm<Real>(first, expected_fp64, points,
                                        label + " first execution", statistics);
    else
        compare_complex<Real>(first, expected, label + " first execution", statistics);
    require_input_unchanged(d_input.data(), input, label + " after first execution", statistics);

    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace_b.data(), workspace_b.bytes());
    else
        plan.set_workspace(nullptr, 0);
    plan.execute(d_input.data(), d_output.data(), stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize xD complex rebound execution");
    check_cuda(cudaMemcpy(second.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
               "download xD complex rebound execution");
    if (use_fp64_reference)
        compare_complex_fp64_norm<Real>(second, expected_fp64, points,
                                        label + " rebound execution", statistics);
    else
        compare_complex<Real>(second, expected, label + " rebound execution", statistics);
    compare_complex<Real>(second, first, label + " repeatability", statistics);
    require_input_unchanged(d_input.data(), input, label + " after rebound execution", statistics);

    if (options.axis_exchange_policy == 4) {
        plan.execute(d_input.data(), d_input.data(), stream);
        check_cuda(cudaStreamSynchronize(stream), "synchronize xD packed-core exact in-place execution");
        check_cuda(cudaMemcpy(second.data(), d_input.data(), d_input.bytes(), cudaMemcpyDeviceToHost),
                   "download xD packed-core exact in-place output");
        if (use_fp64_reference)
            compare_complex_fp64_norm<Real>(second, expected_fp64, points,
                                            label + " exact in-place execution", statistics);
        else
            compare_complex<Real>(second, expected, label + " exact in-place execution", statistics);
    }
    ++statistics.cases;
}

template <typename Real, typename Complex>
CufftPlan make_real_oracle(const std::vector<std::size_t>& extents, std::size_t batch, bool inverse) {
    std::vector<int> dimensions(extents.size());
    std::vector<int> input_embed(extents.size());
    std::vector<int> output_embed(extents.size());
    for (std::size_t i = 0; i < extents.size(); ++i) {
        dimensions[i] = static_cast<int>(extents[i]);
        input_embed[i] = static_cast<int>(extents[i]);
        output_embed[i] = static_cast<int>(extents[i]);
    }
    const std::size_t compact_last = extents.back() / 2 + 1;
    const std::size_t logical_points = product(extents);
    const std::size_t compact_points = logical_points / extents.back() * compact_last;
    const int input_distance = static_cast<int>(inverse ? compact_points : logical_points);
    const int output_distance = static_cast<int>(inverse ? logical_points : compact_points);
    const cufftType type = std::is_same_v<Real, double>
                               ? (inverse ? CUFFT_Z2D : CUFFT_D2Z)
                               : (inverse ? CUFFT_C2R : CUFFT_R2C);
    if (!inverse)
        output_embed.back() = static_cast<int>(compact_last);
    else
        input_embed.back() = static_cast<int>(compact_last);
    return CufftPlan(static_cast<int>(extents.size()), dimensions, input_embed, input_distance,
                     output_embed, output_distance, type,
                     static_cast<int>(batch));
}

template <typename Real, typename Complex>
void run_real_case(const Shape& shape, std::size_t batch,
                   const std::vector<std::size_t>& order, bool inverse,
                   Statistics& statistics, const PlanOptions& options = {}) {
    const std::size_t logical_points = product(shape.extents);
    const std::size_t compact_last = shape.extents.back() / 2 + 1;
    const std::size_t compact_points = logical_points / shape.extents.back() * compact_last;
    const std::size_t logical_count = logical_points * batch;
    const std::size_t compact_count = compact_points * batch;
    const bool fp64 = std::is_same_v<Real, double>;
    const std::string label = std::string("real ") + (fp64 ? "fp64 " : "fp32 ") +
                              (inverse ? "C2R inverse " : "R2C forward ") + shape.name +
                              " extents=" + shape_string(shape.extents) +
                              " order=" + order_string(order);

    std::vector<Real> real_input;
    std::vector<Complex> complex_input;
    if (!inverse) {
        real_input.resize(logical_count);
        fill_real(real_input, logical_points, batch);
    }
    std::vector<Complex> expected_complex(compact_count);
    std::vector<Real> expected_real(logical_count);
    std::vector<Complex> first_complex(compact_count), second_complex(compact_count);
    std::vector<Real> first_real(logical_count), second_real(logical_count);

    Stream stream;
    cuntt::detail::XdRealFftPlan plan(shape.extents, batch, inverse, options.normalize_inverse, fp64, false,
                                      options.output_scale, options.packed_axes, order,
                                      options.fused_axes, options.axis_ept,
                                      options.paired_axes, options.axis_core, options.pair_core,
                                      options.register_lanes, options.pair_layout, options.batch_tile,
                                      options.axis_ffts_per_block_cap,
                                      options.axis_exchange_policy,
                                      options.axis_resources);
    const std::string mapping = plan.mapping_json();
    if (!options.axis_resources.empty())
        check_axis_resource_mapping(mapping, options.axis_resources, label, statistics);
    if (options.axis_ffts_per_block_cap != 0) {
        const bool has_capped_complex_axis =
            shape.extents.size() > 1 &&
            std::any_of(shape.extents.begin(), shape.extents.end() - 1,
                        [](std::size_t extent) { return extent == 64; });
        check_axis_block_cap_mapping(mapping, options.axis_ffts_per_block_cap, label, statistics,
                                     has_capped_complex_axis);
    }
    if (options.axis_exchange_policy != 0)
        check_axis_ept_mapping(mapping, options.axis_ept, label, statistics);
    if (options.axis_exchange_policy != 0)
        check_axis_exchange_mapping(mapping, options.axis_exchange_policy, label,
                                    statistics, false);
    if (options.expected_core != nullptr) {
        if (!mapping_has_core(mapping, options.expected_core))
            throw std::runtime_error(label + ": expected mapping core " + options.expected_core);
        if (std::strcmp(options.expected_core, "register-cooperative-axis") == 0)
            statistics.register_axis_observed = true;
        else if (std::strcmp(options.expected_core, "register-cooperative-pair") == 0)
            statistics.cooperative_pair_observed = true;
        if (std::strcmp(options.expected_core, "register-cooperative-axis") == 0)
            check_register_axis_mapping(mapping, options.register_lanes,
                                         options.expected_register_length, label, statistics);
    }
    DeviceBuffer d_input(inverse ? compact_count * sizeof(Complex) : logical_count * sizeof(Real));
    DeviceBuffer d_output(inverse ? logical_count * sizeof(Real) : compact_count * sizeof(Complex));
    DeviceBuffer d_oracle_input(inverse ? compact_count * sizeof(Complex) : logical_count * sizeof(Real));
    DeviceBuffer d_oracle_output(inverse ? logical_count * sizeof(Real) : compact_count * sizeof(Complex));
    const bool use_fp64_reference = !fp64 && !inverse && options.fp64_reference;
    const bool collect_fp64_reference = use_fp64_reference || (!fp64 && !inverse && options.fp64_diagnostics);
    DeviceBuffer d_fp64_oracle_input(collect_fp64_reference ? logical_count * sizeof(double) : 0);
    DeviceBuffer d_fp64_oracle_output(
        collect_fp64_reference ? compact_count * sizeof(cufftDoubleComplex) : 0);
    DeviceBuffer workspace_a(plan.workspace_size());
    DeviceBuffer workspace_b(plan.workspace_size());
    if (inverse) {
        // Build the inverse input from a native multidimensional real transform.
        // This supplies the Hermitian-consistent compact spectrum required by
        // C2R, while keeping the input oracle independent of the xD lowering.
        std::vector<Real> real_seed(logical_count);
        fill_real(real_seed, logical_points, batch);
        DeviceBuffer d_seed(logical_count * sizeof(Real));
        check_cuda(cudaMemcpy(d_seed.data(), real_seed.data(), d_seed.bytes(), cudaMemcpyHostToDevice),
                   "upload native real seed");
        CufftPlan forward_oracle = make_real_oracle<Real, Complex>(shape.extents, batch, false);
        forward_oracle.set_stream(stream);
        if (fp64) {
            check_cufft(cufftExecD2Z(forward_oracle.handle(), static_cast<double*>(d_seed.data()),
                                     static_cast<cufftDoubleComplex*>(d_oracle_input.data())),
                        "execute native D2Z spectrum seed");
        } else {
            check_cufft(cufftExecR2C(forward_oracle.handle(), static_cast<float*>(d_seed.data()),
                                     static_cast<cufftComplex*>(d_oracle_input.data())),
                        "execute native R2C spectrum seed");
        }
        check_cuda(cudaStreamSynchronize(stream), "synchronize native real spectrum seed");
        complex_input.resize(compact_count);
        check_cuda(cudaMemcpy(complex_input.data(), d_oracle_input.data(), d_oracle_input.bytes(),
                              cudaMemcpyDeviceToHost),
                   "download native compact spectrum seed");
        check_cuda(cudaMemcpy(d_input.data(), complex_input.data(), d_input.bytes(), cudaMemcpyHostToDevice),
                   "upload xD inverse compact input");
    } else {
        check_cuda(cudaMemcpy(d_input.data(), real_input.data(), d_input.bytes(), cudaMemcpyHostToDevice),
                   "upload xD real input");
        check_cuda(cudaMemcpy(d_oracle_input.data(), real_input.data(), d_oracle_input.bytes(), cudaMemcpyHostToDevice),
                   "upload native real input");
    }

    CufftPlan oracle = make_real_oracle<Real, Complex>(shape.extents, batch, inverse);
    oracle.set_stream(stream);
    if (inverse) {
        if (fp64) {
            check_cufft(cufftExecZ2D(oracle.handle(), static_cast<cufftDoubleComplex*>(d_oracle_input.data()),
                                     static_cast<double*>(d_oracle_output.data())),
                        "execute native Z2D oracle");
        } else {
            check_cufft(cufftExecC2R(oracle.handle(), static_cast<cufftComplex*>(d_oracle_input.data()),
                                     static_cast<float*>(d_oracle_output.data())),
                        "execute native C2R oracle");
        }
    } else {
        if (fp64) {
            check_cufft(cufftExecD2Z(oracle.handle(), static_cast<double*>(d_oracle_input.data()),
                                     static_cast<cufftDoubleComplex*>(d_oracle_output.data())),
                        "execute native D2Z oracle");
        } else {
            check_cufft(cufftExecR2C(oracle.handle(), static_cast<float*>(d_oracle_input.data()),
                                     static_cast<cufftComplex*>(d_oracle_output.data())),
                        "execute native R2C oracle");
        }
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize native real oracle");
    if (inverse) {
        check_cuda(cudaMemcpy(expected_real.data(), d_oracle_output.data(), d_oracle_output.bytes(), cudaMemcpyDeviceToHost),
                   "download native C2R oracle");
        if (options.normalize_inverse) {
            const Real scale = static_cast<Real>(1.0 / static_cast<double>(logical_points));
            for (auto& value : expected_real)
                value *= scale;
        }
        if (options.output_scale != 1.0) {
            const Real scale = static_cast<Real>(options.output_scale);
            for (auto& value : expected_real)
                value *= scale;
        }
    } else {
        check_cuda(cudaMemcpy(expected_complex.data(), d_oracle_output.data(), d_oracle_output.bytes(), cudaMemcpyDeviceToHost),
                   "download native R2C oracle");
        if (options.output_scale != 1.0) {
            const Real scale = static_cast<Real>(options.output_scale);
            for (auto& value : expected_complex) {
                value.x *= scale;
                value.y *= scale;
            }
        }
    }

    std::vector<cufftDoubleComplex> expected_fp64;
    if (collect_fp64_reference) {
        // Preserve the exact FP32 input values before widening so cancellation
        // errors are measured against the same samples consumed by the plan.
        std::vector<double> real_input_fp64(logical_count);
        for (std::size_t i = 0; i < logical_count; ++i)
            real_input_fp64[i] = static_cast<double>(real_input[i]);
        expected_fp64.resize(compact_count);
        check_cuda(cudaMemcpy(d_fp64_oracle_input.data(), real_input_fp64.data(),
                              d_fp64_oracle_input.bytes(), cudaMemcpyHostToDevice),
                   "upload native fp64 real reference input");
        CufftPlan fp64_oracle = make_real_oracle<double, cufftDoubleComplex>(
            shape.extents, batch, false);
        fp64_oracle.set_stream(stream);
        check_cufft(cufftExecD2Z(
                        fp64_oracle.handle(),
                        static_cast<double*>(d_fp64_oracle_input.data()),
                        static_cast<cufftDoubleComplex*>(d_fp64_oracle_output.data())),
                    "execute native fp64 D2Z real reference");
        check_cuda(cudaStreamSynchronize(stream),
                   "synchronize native fp64 real reference");
        check_cuda(cudaMemcpy(expected_fp64.data(), d_fp64_oracle_output.data(),
                              d_fp64_oracle_output.bytes(), cudaMemcpyDeviceToHost),
                   "download native fp64 real reference");
        if (options.output_scale != 1.0) {
            for (auto& value : expected_fp64) {
                value.x *= options.output_scale;
                value.y *= options.output_scale;
            }
        }
    }

    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace_a.data(), workspace_a.bytes());
    else
        plan.set_workspace(nullptr, 0);
    plan.execute(d_input.data(), d_output.data(), stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize xD real first execution");
    if (inverse) {
        check_cuda(cudaMemcpy(first_real.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
                   "download xD C2R first execution");
        compare_real(first_real, expected_real, label + " first execution", statistics);
        require_input_unchanged(d_input.data(), complex_input, label + " after first execution", statistics);
    } else {
        check_cuda(cudaMemcpy(first_complex.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
                   "download xD R2C first execution");
        if (options.fp64_diagnostics && collect_fp64_reference)
            report_real_fp64_diagnostics<Real>(first_complex, expected_complex, expected_fp64,
                                               label + " first execution");
        if (use_fp64_reference) {
            Statistics xD_reference_diagnostic;
            Statistics native_reference_diagnostic;
            compare_complex_fp64_norm<Real>(
                first_complex, expected_fp64, logical_points,
                label + " xD FP64-reference diagnostic", xD_reference_diagnostic);
            compare_complex_fp64_norm<Real>(
                expected_complex, expected_fp64, logical_points,
                label + " native FP32-oracle FP64-reference diagnostic",
                native_reference_diagnostic);
            std::ostringstream diagnostic;
            diagnostic << std::setprecision(17)
                       << label << ": FP64-reference diagnostics xD_output_relative_l2="
                       << xD_reference_diagnostic.fp32_pair_max_relative_l2
                       << ", native_fp32_oracle_relative_l2="
                       << native_reference_diagnostic.fp32_pair_max_relative_l2 << '\n';
            std::cout << diagnostic.str();
            compare_complex_fp64_norm<Real>(first_complex, expected_fp64, logical_points,
                                            label + " first execution", statistics);
        } else {
            compare_complex<Real>(first_complex, expected_complex,
                                  label + " first execution", statistics);
        }
        require_input_unchanged(d_input.data(), real_input, label + " after first execution", statistics);
    }

    if (plan.workspace_size() != 0)
        plan.set_workspace(workspace_b.data(), workspace_b.bytes());
    else
        plan.set_workspace(nullptr, 0);
    plan.execute(d_input.data(), d_output.data(), stream);
    check_cuda(cudaStreamSynchronize(stream), "synchronize xD real rebound execution");
    if (inverse) {
        check_cuda(cudaMemcpy(second_real.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
                   "download xD C2R rebound execution");
        compare_real(second_real, expected_real, label + " rebound execution", statistics);
        compare_real(second_real, first_real, label + " repeatability", statistics);
        require_input_unchanged(d_input.data(), complex_input, label + " after rebound execution", statistics);
    } else {
        check_cuda(cudaMemcpy(second_complex.data(), d_output.data(), d_output.bytes(), cudaMemcpyDeviceToHost),
                   "download xD R2C rebound execution");
        if (use_fp64_reference)
            compare_complex_fp64_norm<Real>(second_complex, expected_fp64, logical_points,
                                            label + " rebound execution", statistics);
        else
            compare_complex<Real>(second_complex, expected_complex,
                                  label + " rebound execution", statistics);
        compare_complex<Real>(second_complex, first_complex, label + " repeatability", statistics);
        require_input_unchanged(d_input.data(), real_input, label + " after rebound execution", statistics);
    }
    ++statistics.cases;
}

template <typename Real, typename Complex>
void run_local_core_direct_case(bool inverse, Statistics& statistics,
                                std::size_t length = 128,
                                unsigned block_cap = 0,
                                unsigned twiddle_policy = 0) {
    constexpr std::size_t suffix = 1;
    const bool fp64 = std::is_same_v<Real, double>;
    // Keep a tail for every direct launch. FP64 length-256 cap16 deliberately
    // exercises the production alias where the cap reduces the CTA to eight
    // lines; FP32 uses the full 16-line cap.
    const std::size_t lines = length == 256 && fp64 && block_cap == 16 ? 9 :
                              length == 256 ? 17 : 9;
    const std::string label = std::string("local-core direct ") +
                              (fp64 ? "fp64 " : "fp32 ") +
                              "length" + std::to_string(length) + " " +
                              (twiddle_policy == 1 ? "recurrence " : "constant ") +
                              (inverse ? "inverse" : "forward");
    const double scale = inverse ? 0.75 / static_cast<double>(length) : 0.75;
    std::vector<Complex> input(length * lines), expected(length * lines), actual(length * lines);
    fill_complex<Real>(input, length, lines);

    Stream stream;
    DeviceBuffer d_input(input.size() * sizeof(Complex));
    DeviceBuffer d_oracle_input(input.size() * sizeof(Complex));
    DeviceBuffer d_oracle_output(input.size() * sizeof(Complex));
    check_cuda(cudaMemcpy(d_input.data(), input.data(), d_input.bytes(), cudaMemcpyHostToDevice),
               "upload local direct input");
    check_cuda(cudaMemcpy(d_oracle_input.data(), input.data(), d_oracle_input.bytes(),
                          cudaMemcpyHostToDevice),
               "upload local direct oracle input");

    CufftPlan oracle = make_complex_oracle<Real, Complex>({length}, lines, inverse);
    oracle.set_stream(stream);
    if (fp64) {
        check_cufft(cufftExecZ2Z(oracle.handle(),
                                 static_cast<cufftDoubleComplex*>(d_oracle_input.data()),
                                 static_cast<cufftDoubleComplex*>(d_oracle_output.data()),
                                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                    "execute local direct Z2Z oracle");
    } else {
        check_cufft(cufftExecC2C(oracle.handle(),
                                 static_cast<cufftComplex*>(d_oracle_input.data()),
                                 static_cast<cufftComplex*>(d_oracle_output.data()),
                                 inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                    "execute local direct C2C oracle");
    }
    check_cuda(cudaStreamSynchronize(stream), "synchronize local direct oracle");
    check_cuda(cudaMemcpy(expected.data(), d_oracle_output.data(), d_oracle_output.bytes(),
                          cudaMemcpyDeviceToHost),
               "download local direct oracle");
    for (auto& value : expected) {
        value.x = static_cast<Real>(static_cast<double>(value.x) * scale);
        value.y = static_cast<Real>(static_cast<double>(value.y) * scale);
    }

    cuntt::detail::launch_xd_local_fft(
        d_input.data(), d_input.data(), length, suffix, lines, fp64, inverse, scale,
        stream, block_cap, twiddle_policy);
    check_cuda(cudaStreamSynchronize(stream), "synchronize local direct exact-inplace launch");
    check_cuda(cudaMemcpy(actual.data(), d_input.data(), d_input.bytes(), cudaMemcpyDeviceToHost),
               "download local direct exact-inplace output");
    compare_complex<Real>(actual, expected, label, statistics);
    ++statistics.local_direct_checks;
    ++statistics.cases;
}

template <typename Real, typename Complex>
void run_complex_precision(const std::vector<Shape>& shapes, std::size_t batch,
                           Statistics& statistics, const PlanOptions& options = {}) {
    for (const auto& shape : shapes) {
        for (const auto& order : axis_orders(shape.extents.size())) {
            for (const bool inverse : {false, true})
                run_complex_case<Real, Complex>(shape, batch, order, inverse, statistics, options);
        }
    }
}

template <typename Real, typename Complex>
void run_real_precision(const std::vector<Shape>& shapes, std::size_t batch,
                        Statistics& statistics, const PlanOptions& options = {}) {
    for (const auto& shape : shapes) {
        for (const auto& order : real_axis_orders(shape.extents.size())) {
            for (const bool inverse : {false, true})
                run_real_case<Real, Complex>(shape, batch, order, inverse, statistics, options);
        }
    }
}

template <typename Real, typename Complex>
void run_fused_rectangular_precision(std::size_t batch, Statistics& statistics,
                                     const PlanOptions& options) {
    // The long axis is deliberately selected in its strided position for
    // [length, 8] and its contiguous position for [8, length].  Keeping one
    // order per shape bounds plan creation while covering every fused length
    // in both physical layouts.
    const std::vector<std::size_t> lengths = {8, 16, 32, 64, 128, 256, 512, 1024};
    for (const auto length : lengths) {
        const Shape strided{{length, 8}, "fused-rectangular-strided"};
        const Shape contiguous{{8, length}, "fused-rectangular-contiguous"};
        for (const bool inverse : {false, true}) {
            run_complex_case<Real, Complex>(strided, batch, {0, 1}, inverse, statistics, options);
            run_complex_case<Real, Complex>(contiguous, batch, {1, 0}, inverse, statistics, options);
        }
    }
}

template <typename Real, typename Complex>
std::size_t run_pair_precision(std::size_t batch, Statistics& statistics) {
    const bool fp64 = std::is_same_v<Real, double>;
    const PlanOptions pair_options_ept8{/*normalize_inverse=*/false, /*output_scale=*/1.25,
                                        /*packed_axes=*/true, /*fused_axes=*/true,
                                        /*axis_ept=*/8, /*paired_axes=*/true};
    const PlanOptions pair_options_ept16{/*normalize_inverse=*/false, /*output_scale=*/1.25,
                                         /*packed_axes=*/true, /*fused_axes=*/true,
                                         /*axis_ept=*/16, /*paired_axes=*/true};
    const std::vector<Shape> rank2_shapes = {
        {{8, 16}, "pair-rank2-8x16"},
        {{16, 32}, "pair-rank2-16x32"},
        {{32, 16}, "pair-rank2-32x16"},
        {{32, 32}, "pair-rank2-32x32"},
        {{64, 32}, "pair-rank2-64x32"},
        {{64, 64}, "pair-rank2-64x64"},
    };
    const std::vector<Shape> rank3_shapes = {
        {{5, 16, 32}, "pair-rank3-prefix5"},
        {{16, 32, 32}, "pair-rank3-16x32x32"},
        {{64, 64, 64}, "pair-rank3-64x64x64"},
    };

    std::size_t candidates = 0;
    for (const auto& shape : rank2_shapes) {
        if (!cuntt::detail::xd_pair_fft_available(shape.extents[0], shape.extents[1], fp64))
            continue;
        ++candidates;
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(shape, batch, {1, 0}, inverse,
                                             statistics, pair_options_ept8, true);
    }
    for (const auto& shape : rank3_shapes) {
        // Both rank-3 orders pair the final two dimensions only when that
        // plane is supported for this precision.  In particular, FP64
        // 64-axis shapes are filtered by the compiled Thread codelet inventory.
        if (!cuntt::detail::xd_pair_fft_available(shape.extents[1], shape.extents[2], fp64))
            continue;
        ++candidates;
        for (const PlanOptions& options : {pair_options_ept8, pair_options_ept16}) {
            for (const bool inverse : {false, true}) {
                run_complex_case<Real, Complex>(shape, batch, {2, 1, 0}, inverse,
                                                 statistics, options, true);
                // A fused fallback on the unsupported prefix-5 axis can leave
                // the final pair non-contiguous.  Keep that case in the matrix,
                // but require a pair group only where the schedule can form one.
                const bool expect_pair = shape.extents[0] != 5;
                run_complex_case<Real, Complex>(shape, batch, {0, 2, 1}, inverse,
                                                 statistics, options, expect_pair);
            }
        }
    }
    return candidates;
}

template <typename Real, typename Complex>
std::size_t run_register_axis_precision(std::size_t batch, Statistics& statistics, unsigned lanes = 4) {
    const bool fp64 = std::is_same_v<Real, double>;
    PlanOptions options;
    options.normalize_inverse = true;
    options.output_scale = 1.25;
    options.packed_axes = true;
    options.fused_axes = true;
    options.axis_ept = 8;
    options.paired_axes = false;
    options.axis_core = cuntt::detail::XdAxisCore::CooperativeRegister;
    options.pair_core = cuntt::detail::XdPairCore::Thread;
    options.expected_core = "register-cooperative-axis";
    options.register_lanes = lanes;

    const std::vector<Shape> rank2_shapes = {
        {{64, 128}, "register-axis-rank2-64x128"},
        {{128, 64}, "register-axis-rank2-128x64"},
    };
    const Shape rank3_shape{{16, 64, 128}, "register-axis-rank3-16x64x128"};
    const Shape real_shape{{64, 65}, "register-axis-real-64x65"};
    std::size_t candidates = 0;

    // Put the supported register length in a non-contiguous position with a
    // suffix greater than one.  The opposite rank-2 shapes exercise both
    // supported lengths without expanding the general permutation matrix.
    for (const auto& shape : rank2_shapes) {
        const std::vector<std::size_t> order = shape.extents[0] == 64
                                                   ? std::vector<std::size_t>{0, 1}
                                                   : std::vector<std::size_t>{1, 0};
        ++candidates;
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(shape, batch, order, inverse, statistics, options);
    }

    // The large FP32 cases use the existing FP64 norm-wise oracle.  This
    // keeps cancellation-sensitive coverage meaningful without weakening the
    // ordinary scalar tolerance for the rest of the mode.
    PlanOptions large_options = options;
    large_options.fp64_reference = !fp64;
    ++candidates;
    for (const auto& order : {std::vector<std::size_t>{0, 1, 2},
                              std::vector<std::size_t>{2, 1, 0}}) {
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(rank3_shape, batch, order, inverse,
                                             statistics, large_options);
    }

    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_shape, batch, {0}, inverse, statistics, options);

    // Lanes 8/16 also cover the newly supported 256-point register axis.  Keep
    // the lane-4 matrix unchanged so its historical 64/128 contract remains a
    // direct comparison point.  The rank-3 case deliberately uses B=3 and a
    // two-batch tile to exercise the final partial tile.
    if (lanes != 4) {
        PlanOptions register256_options = options;
        register256_options.expected_register_length = 256;
        register256_options.fp64_reference = !fp64;
        const Shape rank2_256_shape{{256, 32}, "register-axis-rank2-256x32"};
        ++candidates;
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(rank2_256_shape, batch, {0, 1}, inverse,
                                             statistics, register256_options);

        register256_options.batch_tile = 2;
        register256_options.fp64_reference = !fp64;
        const Shape rank3_256_shape{{8, 256, 32}, "register-axis-rank3-8x256x32"};
        ++candidates;
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(rank3_256_shape, batch, {2, 1, 0}, inverse,
                                             statistics, register256_options);

        PlanOptions real256_options = options;
        real256_options.expected_register_length = 256;
        const Shape real256_shape{{256, 33}, "register-axis-real-256x33"};
        ++candidates;
        for (const bool inverse : {false, true})
            run_real_case<Real, Complex>(real256_shape, batch, {0}, inverse,
                                         statistics, real256_options);
    }
    return candidates;
}

template <typename Real, typename Complex>
std::size_t run_axis_block_caps_precision(Statistics& statistics) {
    const std::size_t batch = 3;
    std::size_t candidates = 0;
    for (const unsigned cap : {8U, 16U}) {
        for (const unsigned ept : {8U, 16U}) {
            PlanOptions options;
            options.normalize_inverse = true;
            options.output_scale = 1.0;
            options.packed_axes = true;
            options.fused_axes = true;
            options.axis_ept = ept;
            options.paired_axes = false;
            options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
            options.pair_core = cuntt::detail::XdPairCore::Thread;
            options.axis_ffts_per_block_cap = cap;
            // Different axis cores may round cancellation-heavy outputs
            // differently. Use the existing higher-precision, normwise FFT
            // oracle instead of treating cuFFT FP32 as exact arithmetic.
            options.fp64_reference = std::is_same_v<Real, float>;

            // The default order contains both a contiguous axis (suffix 1)
            // and a strided axis (suffix 64), while the reverse order checks
            // the same pair under the opposite dependency order.
            const Shape square64{{64, 64}, "axis-cap-64x64"};
            for (const auto& order : {std::vector<std::size_t>{1, 0},
                                      std::vector<std::size_t>{0, 1}}) {
                ++candidates;
                for (const bool inverse : {false, true})
                    run_complex_case<Real, Complex>(square64, 3, order, inverse,
                                                     statistics, options);
            }

            // Keep the larger single-batch case bounded while exercising the
            // 256-point axis and both EPT policies.
            const Shape square256{{256, 256}, "axis-cap-256x256"};
            ++candidates;
            for (const bool inverse : {false, true})
                run_complex_case<Real, Complex>(square256, 1, {1, 0}, inverse,
                                                 statistics, options);

            // B=3 with tile=2 forces a B=1 tail. The cap must survive both
            // the body and tail XdRealFftPlan mappings.
            options.batch_tile = 2;
            const Shape rank3{{16, 32, 64}, "axis-cap-rank3-tail"};
            for (const auto& order : {std::vector<std::size_t>{2, 1, 0},
                                      std::vector<std::size_t>{0, 2, 1}}) {
                ++candidates;
                for (const bool inverse : {false, true})
                    run_complex_case<Real, Complex>(rank3, batch, order, inverse,
                                                     statistics, options);
            }

            // Exercise the shared cap path through the public R2C/C2R
            // composition.  The length-64 complex axis is followed by the
            // real length-32 axis, so this also checks a non-square shape.
            const Shape real_shape{{64, 32}, "axis-cap-real-64x32"};
            options.batch_tile = 0;  // Compact real boundaries do not use C2C batch tiling.
            ++candidates;
            for (const bool inverse : {false, true})
                run_real_case<Real, Complex>(real_shape, batch, {0}, inverse,
                                             statistics, options);
        }
    }
    return candidates;
}

template <typename Real, typename Complex>
std::size_t run_axis_exchange_precision(Statistics& statistics) {
    const std::size_t batch = 3;
    const std::array<std::size_t, 8> lengths = {8, 16, 32, 64, 128, 256, 512, 1024};
    std::size_t candidates = 0;
    for (const unsigned policy : {1U, 2U, 3U}) {
        std::size_t combination = 0;
        for (const unsigned cap : {0U, 8U, 16U}) {
            for (const unsigned ept : {4U, 8U, 16U}) {
                PlanOptions options;
                options.normalize_inverse = true;
                options.output_scale = 1.125;
                options.packed_axes = true;
                options.fused_axes = true;
                options.axis_ept = ept;
                options.paired_axes = false;
                options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
                options.pair_core = cuntt::detail::XdPairCore::Thread;
                options.axis_ffts_per_block_cap = cap;
                options.axis_exchange_policy = policy;
                options.batch_tile = 2;
                options.fp64_reference = std::is_same_v<Real, float>;

                const auto length = lengths[combination % lengths.size()];
                ++combination;
                const Shape shape{{length, 8}, "axis-exchange-c2c"};
                ++candidates;
                for (const bool inverse : {false, true})
                    run_complex_case<Real, Complex>(shape, batch, {1, 0}, inverse,
                                                     statistics, options);
            }
        }
        // Complete EPT4's length coverage; the cap/EPT cross-product above
        // already exercises its length-8, length-64 and length-512 variants.
        for (const auto length : {16U, 32U, 128U, 256U, 1024U}) {
            PlanOptions options;
            options.normalize_inverse = true;
            options.output_scale = 1.125;
            options.packed_axes = true;
            options.fused_axes = true;
            options.axis_ept = 4;
            options.paired_axes = false;
            options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
            options.axis_exchange_policy = policy;
            options.batch_tile = 2;
            options.fp64_reference = std::is_same_v<Real, float>;
            const Shape shape{{length, 8}, "axis-exchange-c2c-ept4"};
            ++candidates;
            for (const bool inverse : {false, true})
                run_complex_case<Real, Complex>(shape, batch, {1, 0}, inverse,
                                                 statistics, options);
        }
    }

    // The compact real boundary leaves a non-power-of-two complex-axis suffix
    // (32 / 2 + 1 = 17). Keep one bounded representative for each policy and
    // precision; the C2C matrix above covers every cap/EPT combination.
    for (const unsigned policy : {1U, 2U, 3U}) {
        PlanOptions options;
        options.normalize_inverse = true;
        options.output_scale = 0.75;
        options.packed_axes = true;
        options.fused_axes = true;
        options.axis_ept = policy == 3 ? 4 : 16;
        options.paired_axes = false;
        options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
        options.pair_core = cuntt::detail::XdPairCore::Thread;
        options.axis_ffts_per_block_cap = 8;
        options.axis_exchange_policy = policy;
        options.fp64_reference = std::is_same_v<Real, float>;
        const Shape real_shape{{64, 32}, "axis-exchange-real-suffix17"};
        ++candidates;
        for (const bool inverse : {false, true})
            run_real_case<Real, Complex>(real_shape, batch, {0}, inverse,
                                         statistics, options);
    }
    return candidates;
}

template <typename Real, typename Complex>
std::size_t run_cooperative_pair_precision(std::size_t batch, Statistics& statistics, unsigned layout = 0) {
    const bool fp64 = std::is_same_v<Real, double>;
    PlanOptions options;
    options.normalize_inverse = true;
    options.output_scale = 1.25;
    options.packed_axes = true;
    options.fused_axes = true;
    options.axis_ept = 8;
    options.paired_axes = true;
    options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
    options.pair_core = cuntt::detail::XdPairCore::CooperativeRegister;
    options.expected_core = "register-cooperative-pair";
    options.pair_layout = layout;

    const std::vector<Shape> rank2_shapes = {
        {{32, 32}, "cooperative-pair-rank2-32x32"},
        {{64, 64}, "cooperative-pair-rank2-64x64"},
        {{64, 128}, "cooperative-pair-rank2-64x128"},
        {{128, 64}, "cooperative-pair-rank2-128x64"},
        {{128, 128}, "cooperative-pair-rank2-128x128"},
    };
    const Shape rank3_shape{{16, 64, 64}, "cooperative-pair-rank3-16x64x64"};
    std::size_t candidates = 0;

    for (const auto& shape : rank2_shapes) {
        if (!cuntt::detail::xd_cooperative_pair_fft_available(shape.extents[0], shape.extents[1], fp64, layout))
            continue;
        ++candidates;
        for (const bool inverse : {false, true})
            run_complex_case<Real, Complex>(shape, batch, {1, 0}, inverse,
                                             statistics, options, true);
    }

    if (cuntt::detail::xd_cooperative_pair_fft_available(rank3_shape.extents[1],
                                                         rank3_shape.extents[2], fp64, layout)) {
        ++candidates;
        PlanOptions large_options = options;
        large_options.fp64_reference = !fp64;
        for (const auto& order : {std::vector<std::size_t>{2, 1, 0},
                                  std::vector<std::size_t>{0, 2, 1}}) {
            for (const bool inverse : {false, true})
                run_complex_case<Real, Complex>(rank3_shape, batch, order, inverse,
                                                 statistics, large_options, true);
        }
    }
    return candidates;
}

template <typename Real, typename Complex>
void run_batch_tile_precision(Statistics& statistics) {
    PlanOptions options;
    options.fused_axes = true;
    options.axis_core = cuntt::detail::XdAxisCore::CooperativeRegister;
    options.register_lanes = 8;
    options.fp64_reference = std::is_same_v<Real,float>;
    options.output_scale = 1.25;
    const Shape shape{{64,16,16},"batch-tile-register"};
    for (std::size_t tile : {2U,3U}) {
        options.batch_tile = tile;
        for (bool inverse : {false,true})
            run_complex_case<Real,Complex>(shape,5,{2,1,0},inverse,statistics,options);
    }
    // Exercise the wrapper's reusable vendor workspace and a nonempty tail.
    options.fused_axes = false;
    options.axis_core = cuntt::detail::XdAxisCore::CufftdxBlock;
    options.register_lanes = 4;
    options.batch_tile = 2;
    const Shape fallback{{3,4,8},"batch-tile-packed-fallback"};
    for (bool inverse : {false,true})
        run_complex_case<Real,Complex>(fallback,5,{0,2,1},inverse,statistics,options);
}

cuntt::detail::XdAxisResources make_axis_resource(cuntt::detail::XdAxisCore core,
                                                   unsigned ept, unsigned register_lanes,
                                                   unsigned block_cap, unsigned exchange,
                                                   unsigned local_twiddle_policy = 0) {
    cuntt::detail::XdAxisResources resource{};
    resource.core = core;
    resource.ept = ept;
    resource.register_lanes = register_lanes;
    resource.block_cap = block_cap;
    resource.exchange = exchange;
    resource.local_twiddle_policy = local_twiddle_policy;
    return resource;
}

template <typename Callable>
void expect_axis_resource_rejection(Callable&& construct, const std::string& label,
                                    Statistics& statistics) {
    bool rejected = false;
    try {
        construct();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error(label + ": invalid axis resource configuration was accepted");
    ++statistics.axis_resource_invalid_checks;
}

void run_axis_resource_invalid_cases(Statistics& statistics) {
    using cuntt::detail::XdAxisCore;
    const auto register64 = make_axis_resource(XdAxisCore::CooperativeRegister, 8, 4, 0, 0);
    const auto block32 = make_axis_resource(XdAxisCore::CufftdxBlock, 16, 4, 8, 0);
    const auto register32 = make_axis_resource(XdAxisCore::CooperativeRegister, 8, 4, 0, 0);
    const std::vector<cuntt::detail::XdAxisResources> valid = {register64, block32};
    const auto construct = [&](const std::vector<cuntt::detail::XdAxisResources>& resources,
                               bool packed = true, bool fused = true, bool paired = false) {
        cuntt::detail::XdRealFftPlan plan(
            {64, 32}, 3, false, true, false, true, 1.0, packed, {1, 0}, fused, 8,
            paired, XdAxisCore::CufftdxBlock, cuntt::detail::XdPairCore::Thread,
            4, 0, 0, 0, 0, resources);
    };
    expect_axis_resource_rejection([&] { construct({register64}); },
                                   "axis resources vector length", statistics);

    auto bad_ept = valid;
    bad_ept[1].ept = 7;
    expect_axis_resource_rejection([&] { construct(bad_ept); },
                                   "axis resources ept", statistics);
    auto bad_lanes = valid;
    bad_lanes[0].register_lanes = 7;
    expect_axis_resource_rejection([&] { construct(bad_lanes); },
                                   "axis resources register lanes", statistics);
    auto bad_cap = valid;
    bad_cap[1].block_cap = 4;
    expect_axis_resource_rejection([&] { construct(bad_cap); },
                                   "axis resources block cap", statistics);
    auto bad_exchange = valid;
    bad_exchange[1].exchange = 5;
    expect_axis_resource_rejection([&] { construct(bad_exchange); },
                                   "axis resources exchange", statistics);
    auto nonblock_packed_core = valid;
    nonblock_packed_core[0].exchange = 4;
    expect_axis_resource_rejection([&] { construct(nonblock_packed_core); },
                                   "axis resources packed nonblock core", statistics);
    auto bad_twiddle_policy = valid;
    bad_twiddle_policy[0].local_twiddle_policy = 2;
    expect_axis_resource_rejection([&] { construct(bad_twiddle_policy); },
                                   "axis resources local twiddle policy range", statistics);
    auto nonlocal_twiddle_policy = valid;
    nonlocal_twiddle_policy[1].local_twiddle_policy = 1;
    expect_axis_resource_rejection([&] { construct(nonlocal_twiddle_policy); },
                                   "axis resources nonlocal twiddle policy", statistics);
    expect_axis_resource_rejection([&] { construct(valid, false, true, false); },
                                   "axis resources unpacked plan", statistics);
    expect_axis_resource_rejection([&] { construct(valid, true, true, true); },
                                   "axis resources paired plan", statistics);

    const auto block64 = make_axis_resource(XdAxisCore::CufftdxBlock, 8, 4, 8, 0);
    expect_axis_resource_rejection(
        [&] { construct({block64, register32}); },
        "axis resources register override on contiguous axis", statistics);

    const auto register256 = make_axis_resource(XdAxisCore::CooperativeRegister, 8, 4, 0, 0);
    const auto construct_length256 = [&] {
        cuntt::detail::XdRealFftPlan plan(
            {256, 32}, 3, false, true, false, true, 1.0, true, {1, 0}, true, 8,
            false, XdAxisCore::CufftdxBlock, cuntt::detail::XdPairCore::Thread,
            4, 0, 0, 0, 0, {register256, block32});
    };
    expect_axis_resource_rejection([&] { construct_length256(); },
                                   "axis resources length 256 lane4", statistics);
}

template <typename Real, typename Complex>
std::size_t run_axis_resources_precision(std::size_t batch, Statistics& statistics) {
    using cuntt::detail::XdAxisCore;
    const bool fp64 = std::is_same_v<Real, double>;
    std::size_t candidates = 0;

    PlanOptions rank2;
    rank2.normalize_inverse = true;
    rank2.output_scale = 1.25;
    rank2.packed_axes = true;
    rank2.fused_axes = true;
    rank2.axis_resources = {
        make_axis_resource(XdAxisCore::CooperativeRegister, 8, 4, 0, 0),
        make_axis_resource(XdAxisCore::CufftdxBlock, 16, 4, 8, 0),
    };
    const Shape rank2_shape{{64, 32}, "axis-resources-rank2-64x32"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank2_shape, batch, {1, 0}, inverse,
                                         statistics, rank2);

    PlanOptions rank3 = rank2;
    rank3.fp64_reference = !fp64;
    rank3.batch_tile = 2;
    rank3.axis_resources = {
        make_axis_resource(XdAxisCore::CufftdxBlock, 4, 4, 8, 1),
        make_axis_resource(XdAxisCore::CooperativeRegister, 8, 8, 0, 0),
        make_axis_resource(XdAxisCore::CufftdxBlock, 16, 4, 16, 2),
    };
    const Shape rank3_shape{{64, 64, 32}, "axis-resources-rank3-64x64x32"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank3_shape, batch, {2, 0, 1}, inverse,
                                         statistics, rank3);

    PlanOptions real = rank2;
    real.output_scale = 0.75;
    real.batch_tile = 0;
    real.axis_resources = {
        make_axis_resource(XdAxisCore::CooperativeRegister, 8, 4, 0, 0),
    };
    const Shape real_shape{{64, 33}, "axis-resources-real-64x33-compact17"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_shape, batch, {0}, inverse,
                                      statistics, real);

    PlanOptions real_rank3 = rank2;
    real_rank3.output_scale = 0.625;
    real_rank3.batch_tile = 0;
    real_rank3.axis_resources = {
        make_axis_resource(XdAxisCore::CufftdxBlock, 4, 4, 8, 1),
        make_axis_resource(XdAxisCore::CufftdxBlock, 16, 4, 8, 3),
    };
    const Shape real_rank3_shape{{64, 64, 33}, "axis-resources-real-rank3-64x64x33-compact17"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_rank3_shape, batch, {0, 1}, inverse,
                                      statistics, real_rank3);
    return candidates;
}

template <typename Real, typename Complex>
std::size_t run_local_core_precision(std::size_t batch, Statistics& statistics) {
    using cuntt::detail::XdAxisCore;
    const bool fp64 = std::is_same_v<Real, double>;
    std::size_t candidates = 0;
    const auto local = [](unsigned cap, unsigned twiddle_policy = 0) {
        return make_axis_resource(XdAxisCore::ThreadShared, 8, 4, cap, 0,
                                  twiddle_policy);
    };
    const auto block = [](unsigned ept, unsigned cap) {
        return make_axis_resource(XdAxisCore::CufftdxBlock, ept, 4, cap, 0);
    };

    // Keep both directions and both local line caps while mixing one local
    // axis with the established block core on the other logical axis.
    PlanOptions rank2_cap0;
    rank2_cap0.normalize_inverse = true;
    rank2_cap0.output_scale = 1.25;
    rank2_cap0.packed_axes = true;
    rank2_cap0.fused_axes = true;
    rank2_cap0.axis_core = XdAxisCore::ThreadShared;
    rank2_cap0.batch_tile = 2;
    rank2_cap0.fp64_reference = !fp64;
    rank2_cap0.axis_resources = {local(0), block(8, 8)};
    const Shape rank2_shape{{128, 256}, "local-core-rank2-128x256"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank2_shape, batch, {1, 0}, inverse,
                                         statistics, rank2_cap0);

    PlanOptions rank2_cap16 = rank2_cap0;
    rank2_cap16.output_scale = 0.875;
    rank2_cap16.axis_resources = {block(8, 8), local(16)};
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank2_shape, batch, {0, 1}, inverse,
                                         statistics, rank2_cap16);

    // Recurrence is an explicit per-axis resource choice. Keep both local
    // roots in one reordered, batch-tiled C2C plan; FP32 also uses the FP64
    // oracle while FP64 length-256 uses the established cap16 alias.
    PlanOptions recurrence = rank2_cap0;
    recurrence.output_scale = 0.9375;
    recurrence.axis_resources = {
        local(0, 1), local(16, 1),
    };
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank2_shape, batch, {0, 1}, inverse,
                                         statistics, recurrence);

    // The rank-3 case puts both local lengths in one reordered, tiled plan,
    // with a short block-core axis retained as a mixed-path control.
    PlanOptions rank3 = rank2_cap0;
    rank3.output_scale = 1.125;
    rank3.axis_resources = {block(8, 8), local(16), local(0)};
    const Shape rank3_shape{{8, 128, 256}, "local-core-rank3-8x128x256"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_complex_case<Real, Complex>(rank3_shape, batch, {2, 0, 1}, inverse,
                                         statistics, rank3);

    // Compact real axes end in suffix 17 for length 33. Exercise R2C/C2R
    // with the same input-preservation and scaling checks as the C2C cases.
    PlanOptions real_rank2 = rank2_cap0;
    real_rank2.batch_tile = 0;
    real_rank2.output_scale = 0.75;
    real_rank2.axis_resources = {local(0)};
    const Shape real_shape{{128, 33}, "local-core-real-128x33-compact17"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_shape, batch, {0}, inverse,
                                      statistics, real_rank2);

    PlanOptions real_recurrence = rank2_cap0;
    real_recurrence.batch_tile = 0;
    real_recurrence.output_scale = 0.6875;
    real_recurrence.axis_resources = {local(0, 1)};
    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_shape, batch, {0}, inverse,
                                      statistics, real_recurrence);

    PlanOptions real_rank3 = rank2_cap0;
    real_rank3.batch_tile = 0;
    real_rank3.output_scale = 0.625;
    real_rank3.axis_resources = {local(0), local(16)};
    const Shape real_shape_rank3{{128, 256, 33}, "local-core-real-rank3-128x256x33-compact17"};
    ++candidates;
    for (const bool inverse : {false, true})
        run_real_case<Real, Complex>(real_shape_rank3, batch, {1, 0}, inverse,
                                      statistics, real_rank3);

    ++candidates;
    for (const bool inverse : {false, true})
        run_local_core_direct_case<Real, Complex>(inverse, statistics);
    for (const bool inverse : {false, true}) {
        run_local_core_direct_case<Real, Complex>(inverse, statistics, 128, 0, 1);
        run_local_core_direct_case<Real, Complex>(
            inverse, statistics, 256, 16U, 1);
    }
    return candidates;
}

template <typename Real, typename Complex>
void run_packed_core_precision(Statistics& statistics) {
    PlanOptions options;
    options.fused_axes = true;
    options.axis_exchange_policy = 4;
    options.output_scale = 0.875;
    options.expected_core = "cufftdx-axis-tiled";
    options.fp64_diagnostics = !statistics.distribution_only && !std::is_same_v<Real, double>;
    for (const bool inverse : {false, true}) {
        for (const auto& order : std::vector<std::vector<std::size_t>>{{0, 1}, {1, 0}}) {
            run_complex_case<Real, Complex>({{2048, 8}, "packed-core-long"}, 3, order,
                                             inverse, statistics, options);
            run_complex_case<Real, Complex>({{8, 4096}, "packed-core-long-reordered"}, 3,
                                             order, inverse, statistics, options);
        }
        run_complex_case<Real, Complex>({{8, 16, 32}, "packed-core-retained-rank3"}, 3,
                                         {0, 2, 1}, inverse, statistics, options);
        run_real_case<Real, Complex>({{2048, 33}, "packed-core-real-long-compact17"}, 3,
                                      {0}, inverse, statistics, options);
        run_real_case<Real, Complex>({{64, 8, 33}, "packed-core-real-retained-compact17"}, 3,
                                      {0, 1}, inverse, statistics, options);
        auto tiled = options;
        tiled.batch_tile = 2;
        run_complex_case<Real, Complex>({{64, 32}, "packed-core-batch-tail"}, 3,
                                         {1, 0}, inverse, statistics, tiled);
        auto mixed = options;
        mixed.axis_exchange_policy = 0;
        mixed.axis_resources = {
            make_axis_resource(cuntt::detail::XdAxisCore::CufftdxBlock, 8, 4, 0, 4),
            make_axis_resource(cuntt::detail::XdAxisCore::CufftdxBlock, 8, 4, 0, 1)};
        run_complex_case<Real, Complex>({{64, 32}, "packed-core-mixed-resources"}, 3,
                                         {0, 1}, inverse, statistics, mixed);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool fused_mode = false;
        bool wide_mode = false;
        bool pair_mode = false;
        bool register_axis_mode = false;
        bool cooperative_pair_mode = false;
        bool batch_tile_mode = false;
        bool axis_block_caps_mode = false;
        bool axis_exchange_mode = false;
        bool packed_core_mode = false;
        bool packed_core_errors_mode = false;
        bool axis_resources_mode = false;
        bool local_core_mode = false;
        unsigned register_lanes = 4, pair_layout = 0;
        if (argc > 2)
            throw std::invalid_argument("usage: test_xd_fft_layout [--fused|--wide|--pair|--register-axis|--cooperative-pair|--axis-block-caps|--axis-exchange|--packed-core|--packed-core-errors|--axis-resources|--local-core]");
        if (argc == 2) {
            fused_mode = std::strcmp(argv[1], "--fused") == 0;
            wide_mode = std::strcmp(argv[1], "--wide") == 0;
            pair_mode = std::strcmp(argv[1], "--pair") == 0;
            register_axis_mode = std::strcmp(argv[1], "--register-axis") == 0;
            cooperative_pair_mode = std::strcmp(argv[1], "--cooperative-pair") == 0;
            if (std::strcmp(argv[1], "--register-axis-l8") == 0) { register_axis_mode = true; register_lanes = 8; }
            if (std::strcmp(argv[1], "--register-axis-l16") == 0) { register_axis_mode = true; register_lanes = 16; }
            if (std::strcmp(argv[1], "--cooperative-pair-layout1") == 0) { cooperative_pair_mode = true; pair_layout = 1; }
            batch_tile_mode = std::strcmp(argv[1], "--batch-tile") == 0;
            axis_block_caps_mode = std::strcmp(argv[1], "--axis-block-caps") == 0;
            axis_exchange_mode = std::strcmp(argv[1], "--axis-exchange") == 0;
            packed_core_errors_mode = std::strcmp(argv[1], "--packed-core-errors") == 0;
            packed_core_mode = std::strcmp(argv[1], "--packed-core") == 0 || packed_core_errors_mode;
            axis_resources_mode = std::strcmp(argv[1], "--axis-resources") == 0;
            local_core_mode = std::strcmp(argv[1], "--local-core") == 0;
            if (!fused_mode && !wide_mode && !pair_mode && !register_axis_mode &&
                !cooperative_pair_mode && !batch_tile_mode && !axis_block_caps_mode &&
                !axis_exchange_mode && !packed_core_mode && !axis_resources_mode && !local_core_mode)
                throw std::invalid_argument("usage: test_xd_fft_layout [--fused|--wide|--pair|--register-axis|--cooperative-pair|--axis-block-caps|--axis-exchange|--packed-core|--packed-core-errors|--axis-resources|--local-core]");
        }
        const bool fused_layout_mode = fused_mode || wide_mode;
        if (fused_layout_mode && !cuntt::detail::xd_axis_fft_available(8)) {
            std::cout << "xD FFT fused layout regression SKIPPED: xd_axis_fft_available(8) is false\n";
            return 77;
        }
        if (register_axis_mode) {
            const bool basic_register_configuration =
                cuntt::detail::xd_register_fft_configuration_available(64, register_lanes) &&
                cuntt::detail::xd_register_fft_configuration_available(128, register_lanes);
            if (!basic_register_configuration) {
                std::cout << "xD FFT register-axis regression SKIPPED: xD register FFT backend is unavailable\n";
                return 77;
            }
            const bool register256_available =
                cuntt::detail::xd_register_fft_configuration_available(256, register_lanes);
            if (register_lanes == 4 && register256_available)
                throw std::runtime_error("xD register-axis lane4 unexpectedly supports length 256");
            if (register_lanes != 4 && !register256_available)
                throw std::runtime_error("xD register-axis length 256 is unavailable for requested lanes");
        }
        if (pair_mode && !cuntt::detail::xd_pair_fft_available(8, 16, false) &&
            !cuntt::detail::xd_pair_fft_available(8, 16, true)) {
            std::cout << "xD FFT pair layout regression SKIPPED: xD pair FFT backend is unavailable\n";
            return 77;
        }
        if (cooperative_pair_mode &&
            !cuntt::detail::xd_cooperative_pair_fft_available(64, 64, false) &&
            !cuntt::detail::xd_cooperative_pair_fft_available(64, 64, true)) {
            std::cout << "xD FFT cooperative-pair regression SKIPPED: xD cooperative pair FFT backend is unavailable\n";
            return 77;
        }
        if (axis_block_caps_mode &&
            (!cuntt::detail::xd_axis_fft_available(64) ||
             !cuntt::detail::xd_axis_fft_available(256))) {
            std::cout << "xD FFT axis-block-cap regression SKIPPED: required axis FFT backend is unavailable\n";
            return 77;
        }
        if (axis_exchange_mode &&
            (!cuntt::detail::xd_axis_fft_available(8) ||
             !cuntt::detail::xd_axis_fft_available(1024))) {
            std::cout << "xD FFT axis-exchange regression SKIPPED: required axis FFT backend is unavailable\n";
            return 77;
        }
        if (packed_core_mode &&
            (!cuntt::detail::xd_axis_fft_configuration_available(2048, true, 8) ||
             !cuntt::detail::xd_axis_fft_configuration_available(4096, true, 8))) {
            std::cout << "xD FFT packed-core regression SKIPPED: required axis configurations unavailable\n";
            return 77;
        }
        if (axis_resources_mode &&
            (!cuntt::detail::xd_axis_fft_available(32) ||
             !cuntt::detail::xd_axis_fft_available(64) ||
             !cuntt::detail::xd_register_fft_configuration_available(64, 4) ||
             !cuntt::detail::xd_register_fft_configuration_available(64, 8))) {
            std::cout << "xD FFT axis-resources regression SKIPPED: required per-axis backend is unavailable\n";
            return 77;
        }
        if (local_core_mode &&
            (!cuntt::detail::xd_local_fft_available(128) ||
             !cuntt::detail::xd_local_fft_available(256))) {
            std::cout << "xD FFT local-core regression SKIPPED: required local FFT backend is unavailable\n";
            return 77;
        }

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "query CUDA devices");
        if (device_count == 0)
            throw std::runtime_error("no CUDA device available");

        const std::vector<Shape> complex_shapes = {
            {{35, 17}, "rank2-rectangular"},
            {{35, 17, 9}, "rank3-rectangular-ragged"},
            {{16, 32, 32}, "rank3-regular"},
            {{3, 1, 37}, "rank3-unit-axis"},
        };
        const std::vector<Shape> real_shapes = {
            {{35, 17}, "rank2-odd"},
            {{16, 32}, "rank2-even"},
            {{35, 17, 9}, "rank3-odd"},
            {{16, 32, 32}, "rank3-even"},
            {{3, 1, 37}, "rank3-unit-odd"},
        };
        constexpr std::size_t batch = 3;
        Statistics statistics;
        statistics.distribution_only = packed_core_errors_mode;
        std::size_t pair_candidates = 0;

        if (!fused_layout_mode && !pair_mode && !register_axis_mode &&
            !cooperative_pair_mode && !batch_tile_mode && !axis_block_caps_mode &&
            !axis_exchange_mode && !packed_core_mode && !axis_resources_mode && !local_core_mode) {
            // Every complex rank-2/rank-3 permutation is tested with packed_axes=true.
            run_complex_precision<float, cufftComplex>(complex_shapes, batch, statistics);
            run_complex_precision<double, cufftDoubleComplex>(complex_shapes, batch, statistics);
            // Rank-3 real cases exercise both orders of the two complex axes. Rank-2
            // real cases necessarily have one complex axis and therefore one order.
            run_real_precision<float, cufftComplex>(real_shapes, batch, statistics);
            run_real_precision<double, cufftDoubleComplex>(real_shapes, batch, statistics);
        } else if (fused_layout_mode) {
            const unsigned axis_ept = wide_mode ? 16U : 8U;
            const PlanOptions fused_options{/*normalize_inverse=*/true, /*output_scale=*/1.0,
                                             /*packed_axes=*/true, /*fused_axes=*/true,
                                             /*axis_ept=*/axis_ept, /*paired_axes=*/false};
            const std::vector<Shape> mixed_shapes = {
                {{8, 5, 16}, "fused-mixed-supported-unsupported"},
            };
            const std::vector<Shape> real_fused_shapes = {
                {{16, 8, 9}, "fused-real-odd-last"},
                {{16, 8, 32}, "fused-real-even-last"},
            };
            run_fused_rectangular_precision<float, cufftComplex>(batch, statistics, fused_options);
            run_fused_rectangular_precision<double, cufftDoubleComplex>(batch, statistics, fused_options);
            // The unsupported length 5 must take the physical fallback path in
            // every rank-3 order while supported 8/16 axes use the fused core.
            run_complex_precision<float, cufftComplex>(mixed_shapes, batch, statistics, fused_options);
            run_complex_precision<double, cufftDoubleComplex>(mixed_shapes, batch, statistics, fused_options);
            // Real inverse inputs are generated from a native forward transform,
            // so these cases also check a valid Hermitian compact spectrum.
            run_real_precision<float, cufftComplex>(real_fused_shapes, batch, statistics, fused_options);
            run_real_precision<double, cufftDoubleComplex>(real_fused_shapes, batch, statistics, fused_options);

            const PlanOptions scaled_complex{/*normalize_inverse=*/false, /*output_scale=*/1.25,
                                              /*packed_axes=*/true, /*fused_axes=*/true,
                                              /*axis_ept=*/axis_ept, /*paired_axes=*/false};
            const PlanOptions scaled_real{/*normalize_inverse=*/false, /*output_scale=*/0.75,
                                          /*packed_axes=*/true, /*fused_axes=*/true,
                                          /*axis_ept=*/axis_ept, /*paired_axes=*/false};
            const Shape scaled_complex_shape{{8, 16}, "fused-scaled-complex"};
            const Shape scaled_real_shape{{16, 8, 9}, "fused-scaled-real"};
            run_complex_case<float, cufftComplex>(scaled_complex_shape, batch, {0, 1}, true,
                                                  statistics, scaled_complex);
            run_complex_case<double, cufftDoubleComplex>(scaled_complex_shape, batch, {1, 0}, true,
                                                         statistics, scaled_complex);
            run_real_case<float, cufftComplex>(scaled_real_shape, batch, {0, 1}, true,
                                               statistics, scaled_real);
            run_real_case<double, cufftDoubleComplex>(scaled_real_shape, batch, {1, 0}, true,
                                                      statistics, scaled_real);
        } else if (pair_mode) {
            pair_candidates += run_pair_precision<float, cufftComplex>(batch, statistics);
            pair_candidates += run_pair_precision<double, cufftDoubleComplex>(batch, statistics);
            if (pair_candidates == 0)
                throw std::runtime_error("xD pair layout regression found no available pair shapes");
            if (!statistics.pair_observed)
                throw std::runtime_error("xD pair layout regression observed no fused pair group");
        } else if (register_axis_mode) {
            const std::size_t register_candidates =
                run_register_axis_precision<float, cufftComplex>(batch, statistics,register_lanes) +
                run_register_axis_precision<double, cufftDoubleComplex>(batch, statistics,register_lanes);
            if (register_candidates == 0 || !statistics.register_axis_observed)
                throw std::runtime_error("xD register-axis regression observed no cooperative axis core");
        } else if (batch_tile_mode) {
            run_batch_tile_precision<float,cufftComplex>(statistics);
            run_batch_tile_precision<double,cufftDoubleComplex>(statistics);
        } else if (axis_block_caps_mode) {
            const std::size_t axis_cap_candidates =
                run_axis_block_caps_precision<float, cufftComplex>(statistics) +
                run_axis_block_caps_precision<double, cufftDoubleComplex>(statistics);
            if (axis_cap_candidates == 0 || statistics.axis_cap_checks == 0)
                throw std::runtime_error("xD axis-block-cap regression observed no capped axis metadata");
        } else if (axis_exchange_mode) {
            const std::size_t axis_exchange_candidates =
                run_axis_exchange_precision<float, cufftComplex>(statistics) +
                run_axis_exchange_precision<double, cufftDoubleComplex>(statistics);
            if (axis_exchange_candidates == 0 || statistics.axis_exchange_checks == 0)
                throw std::runtime_error("xD axis-exchange regression observed no exchange metadata");
        } else if (packed_core_mode) {
            run_packed_core_precision<float, cufftComplex>(statistics);
            run_packed_core_precision<double, cufftDoubleComplex>(statistics);
        } else if (axis_resources_mode) {
            run_axis_resource_invalid_cases(statistics);
            const std::size_t axis_resource_candidates =
                run_axis_resources_precision<float, cufftComplex>(batch, statistics) +
                run_axis_resources_precision<double, cufftDoubleComplex>(batch, statistics);
            if (axis_resource_candidates == 0 || statistics.axis_resource_checks == 0 ||
                statistics.axis_resource_invalid_checks == 0)
                throw std::runtime_error("xD axis-resources regression observed no resource coverage");
        } else if (local_core_mode) {
            const std::size_t local_core_candidates =
                run_local_core_precision<float, cufftComplex>(batch, statistics) +
                run_local_core_precision<double, cufftDoubleComplex>(batch, statistics);
            if (local_core_candidates == 0 || statistics.local_core_checks == 0 ||
                statistics.local_direct_checks != 12)
                throw std::runtime_error("xD local-core regression observed no thread-shared local core");
        } else {
            const std::size_t cooperative_pair_candidates =
                run_cooperative_pair_precision<float, cufftComplex>(batch, statistics,pair_layout) +
                run_cooperative_pair_precision<double, cufftDoubleComplex>(batch, statistics,pair_layout);
            if (cooperative_pair_candidates == 0 || !statistics.cooperative_pair_observed)
                throw std::runtime_error("xD cooperative-pair regression observed no cooperative pair core");
        }

        std::cout << (statistics.distribution_only
                          ? "xD FFT layout distribution-only completed (numerical acceptance not asserted): cases="
                          : "xD FFT layout regression passed: cases=") << statistics.cases
                  << ", scalar_checks=" << statistics.scalar_checks
                  << ", input_preservation_checks=" << statistics.input_checks
                  << (axis_block_caps_mode
                          ? std::string(", axis_cap_checks=") + std::to_string(statistics.axis_cap_checks)
                          : axis_exchange_mode
                          ? std::string(", axis_exchange_checks=") +
                                      std::to_string(statistics.axis_exchange_checks) +
                                      ", axis_ept_checks=" +
                                      std::to_string(statistics.axis_ept_checks)
                                : std::string{})
                  << ", batch=" << batch
                  << ", register_lanes=" << register_lanes << ", pair_layout=" << pair_layout
                  << ", native cufftMakePlanMany rank2/rank3 oracle"
                  << ((pair_mode || register_axis_mode || cooperative_pair_mode || local_core_mode)
                          ? ", FP64 norm oracle for FP32 large cases" : "")
                  << (cooperative_pair_mode ? ", paired_axes=true, pair_core=cooperative-register" :
                      pair_mode ? ", paired_axes=true" :
                      register_axis_mode ? ", fused_axes=true, axis_core=cooperative-register" :
                      axis_block_caps_mode ? ", fused_axes=true, axis_core=cufftdx-block, axis_ffts_per_block_cap=8/16" :
                      axis_exchange_mode ? ", fused_axes=true, axis_exchange_policy=1/2/3, axis_ept=4/8/16" :
                      packed_core_mode ? ", fused_axes=true, axis_exchange_policy=4, packed-core" :
                      local_core_mode ? ", fused_axes=true, axis_core=thread-shared, local-core" :
                      axis_resources_mode ? ", fused_axes=true, per-axis resources" :
                      fused_layout_mode ? ", fused_axes=true" : ", fused_axes=false")
                  << (wide_mode ? ", axis_ept=16" : "")
                  << ", pair_groups=" << statistics.pair_groups
                  << ", fp32_pair_max_relative_l2=" << statistics.fp32_pair_max_relative_l2
                  << ", fp32_pair_max_absolute_error=" << statistics.fp32_pair_max_absolute_error
                  << ", register_axis_checks=" << statistics.register_axis_checks
                  << ", axis_resource_checks=" << statistics.axis_resource_checks
                  << ", axis_resource_invalid_checks=" << statistics.axis_resource_invalid_checks
                  << ", local_core_checks=" << statistics.local_core_checks
                  << ", local_direct_checks=" << statistics.local_direct_checks
                  << ", distribution_only=" << (statistics.distribution_only ? "true" : "false")
                  << ", threshold_failures=" << statistics.threshold_failures
                  << ", threshold_failed_scalars=" << statistics.threshold_failed_scalars
                  << ", packed_axes=true, normalization/scaling checks, workspace rebinding\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "xD FFT layout regression FAILED: " << error.what() << '\n';
        return 1;
    }
}
