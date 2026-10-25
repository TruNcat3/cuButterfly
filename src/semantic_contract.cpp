#include "semantic_contract.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace cubutterfly::detail {
namespace {
using nlohmann::json;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template <typename T> void enum_range(T value, int last) {
    require(static_cast<int>(value) >= 0 && static_cast<int>(value) <= last,
            "invalid descriptor enum");
}
std::size_t multiply(std::size_t a, std::size_t b) {
    require(!b || a <= std::numeric_limits<std::size_t>::max() / b,
            "descriptor span overflows size_t");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    require(a <= std::numeric_limits<std::size_t>::max() - b,
            "descriptor span overflows size_t");
    return a + b;
}
bool power_two(std::size_t n) { return n && !(n & (n - 1)); }
std::size_t next_power(std::size_t n) {
    require(n != 0, "zero extent");
    std::size_t p = 1;
    while (p < n) p = multiply(p, 2);
    return p;
}
std::size_t log_two(std::size_t n) {
    std::size_t result = 0;
    while (n > 1) { n /= 2; ++result; }
    return result;
}
std::vector<std::size_t> side_extents(const cubutterflyDescriptor& d, bool input) {
    require(d.rank > 0 && d.extents.size() == d.rank, "rank and shape mismatch");
    auto result = d.extents;
    const bool inverse = d.direction == CUBUTTERFLY_DIRECTION_INVERSE;
    if (d.length_mode == CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING && input == inverse)
        for (auto& n : result) n = next_power(n);
    if (d.op == CUBUTTERFLY_OPERATOR_FFT &&
        ((input && d.fft_type == CUBUTTERFLY_FFT_C2R) ||
         (!input && d.fft_type == CUBUTTERFLY_FFT_R2C)))
        result.back() = result.back() / 2 + 1;
    return result;
}
enum class LayoutProof { Nonoverlapping, Overlapping, Unknown };
LayoutProof layout_proof(const std::vector<std::size_t>& extents,
                         const std::vector<std::size_t>& strides,
                         std::size_t batch, std::size_t batch_stride) {
    require(!extents.empty() && extents.size() == strides.size() && batch > 0 && batch_stride > 0,
            "invalid layout dimensions");
    std::vector<std::pair<std::size_t, std::size_t>> dimensions;
    std::size_t span = 1;
    for (std::size_t axis = 0; axis < extents.size(); ++axis) {
        require(extents[axis] > 0 && strides[axis] > 0, "layout extents and strides must be positive");
        span = add(span, multiply(extents[axis] - 1, strides[axis]));
        if (extents[axis] > 1) dimensions.emplace_back(strides[axis], extents[axis]);
    }
    add(span, multiply(batch - 1, batch_stride));
    if (batch > 1) dimensions.emplace_back(batch_stride, batch);
    std::sort(dimensions.begin(), dimensions.end());
    std::size_t covered = 1;
    bool hierarchical = true;
    for (const auto& dimension : dimensions) {
        if (dimension.first < covered) hierarchical = false;
        covered = add(covered, multiply(dimension.second - 1, dimension.first));
    }
    if (hierarchical) return LayoutProof::Nonoverlapping;
    // A pairwise bounded integer relation establishes a collision immediately.
    for (std::size_t a = 0; a < dimensions.size(); ++a)
        for (std::size_t b = a + 1; b < dimensions.size(); ++b) {
            const auto divisor = std::gcd(dimensions[a].first, dimensions[b].first);
            if (dimensions[a].second - 1 >= dimensions[b].first / divisor &&
                dimensions[b].second - 1 >= dimensions[a].first / divisor)
                return LayoutProof::Overlapping;
        }
    constexpr std::size_t exact_limit = 65536;
    std::size_t points = 1;
    for (const auto& dimension : dimensions) {
        if (dimension.second > exact_limit / points) return LayoutProof::Unknown;
        points *= dimension.second;
    }
    std::unordered_set<std::size_t> addresses;
    addresses.reserve(points);
    for (std::size_t index = 0; index < points; ++index) {
        std::size_t remaining = index, address = 0;
        for (const auto& dimension : dimensions) {
            address += (remaining % dimension.second) * dimension.first;
            remaining /= dimension.second;
        }
        if (!addresses.insert(address).second) return LayoutProof::Overlapping;
    }
    return LayoutProof::Nonoverlapping;
}
void normalize_layout(std::vector<std::size_t>& strides, std::size_t& batch_stride,
                      const std::vector<std::size_t>& extents, std::size_t batch) {
    if (strides.empty()) {
        strides.resize(extents.size());
        std::size_t stride = 1;
        for (std::size_t axis = extents.size(); axis-- > 0;) {
            strides[axis] = stride;
            stride = multiply(stride, extents[axis]);
        }
    }
    require(strides.size() == extents.size(), "stride rank mismatch");
    std::size_t span = 1;
    for (std::size_t axis = 0; axis < extents.size(); ++axis) {
        require(strides[axis] > 0, "axis stride must be nonzero");
        span = add(span, multiply(extents[axis] - 1, strides[axis]));
    }
    if (!batch_stride) batch_stride = span;
    require(layout_proof(extents, strides, batch, batch_stride) != LayoutProof::Overlapping,
            "descriptor layout overlaps");
}
void normalize_orders(std::vector<cubutterflyLogicalOrder_t>& orders,
                      const std::vector<std::size_t>& extents) {
    if (orders.empty()) orders.assign(extents.size(), CUBUTTERFLY_ORDER_NATURAL);
    require(orders.size() == extents.size(), "logical order rank mismatch");
    for (std::size_t axis = 0; axis < orders.size(); ++axis) {
        enum_range(orders[axis], 1);
        require(orders[axis] == CUBUTTERFLY_ORDER_NATURAL || power_two(extents[axis]),
                "bit reversal requires a power-of-two side extent");
    }
}
void exact_fields(const json& value, std::initializer_list<const char*> fields) {
    require(value.is_object() && value.size() == fields.size(), "JSON fields mismatch");
    for (const char* field : fields) require(value.contains(field), "missing JSON field");
}
std::uint64_t unsigned_integer(const json& value, std::uint64_t maximum) {
    require(value.is_number_integer(), "JSON integer required");
    if (value.is_number_unsigned()) {
        const auto result = value.get<std::uint64_t>();
        require(result <= maximum, "JSON integer out of range");
        return result;
    }
    const auto result = value.get<std::int64_t>();
    require(result >= 0 && static_cast<std::uint64_t>(result) <= maximum,
            "JSON integer out of range");
    return static_cast<std::uint64_t>(result);
}
template <typename T> T integer(const json& value) {
    return static_cast<T>(unsigned_integer(value, std::numeric_limits<T>::max()));
}
template <typename T> T enumeration(const json& value, int last) {
    return static_cast<T>(unsigned_integer(value, static_cast<std::uint64_t>(last)));
}
std::vector<std::size_t> size_array(const json& value) {
    require(value.is_array(), "JSON array required");
    std::vector<std::size_t> result;
    for (const auto& item : value) result.push_back(integer<std::size_t>(item));
    return result;
}
std::vector<cubutterflyLogicalOrder_t> order_array(const json& value) {
    require(value.is_array(), "JSON order array required");
    std::vector<cubutterflyLogicalOrder_t> result;
    for (const auto& item : value)
        result.push_back(enumeration<cubutterflyLogicalOrder_t>(item, 1));
    return result;
}
double real_number(const json& value) {
    require(value.is_number(), "JSON number required");
    const double result = value.get<double>();
    require(std::isfinite(result), "JSON number must be finite");
    return result;
}
}

std::vector<std::size_t> semantic_input_extents(const cubutterflyDescriptor& d) {
    return side_extents(d, true);
}
std::vector<std::size_t> semantic_output_extents(const cubutterflyDescriptor& d) {
    return side_extents(d, false);
}
bool layout_is_nonoverlapping(const std::vector<std::size_t>& extents,
                              const std::vector<std::size_t>& strides,
                              std::size_t batch, std::size_t batch_stride) {
    return layout_proof(extents, strides, batch, batch_stride) == LayoutProof::Nonoverlapping;
}

cubutterflyDescriptor normalize_descriptor(const cubutterflyDescriptor& source) {
    auto d = source;
    enum_range(d.op, 5); enum_range(d.storage, 9); enum_range(d.compute, 4);
    enum_range(d.direction, 1); enum_range(d.placement, 1); enum_range(d.length_mode, 1);
    enum_range(d.execution_mode, 2); enum_range(d.fft_type, 2); enum_range(d.algorithm_policy, 2);
    enum_range(d.input_ntt_layout, 2); enum_range(d.output_ntt_layout, 2);
    require((d.algorithm_policy == CUBUTTERFLY_ALGORITHM_EXPLICIT) == !d.explicit_algorithm.empty(),
            "explicit selection policy and algorithm must be supplied together");
    require(d.rank > 0 && d.extents.size() == d.rank, "rank and shape mismatch");
    require(d.batch > 0, "batch must be positive");
    for (auto n : d.extents) require(n > 0, "extent must be positive");
    require(std::isfinite(d.fft_output_scale), "FFT output scale must be finite");
    require(std::isfinite(d.output_scale), "output scale must be finite");
    enum_range(d.output_rounding, 1);
    const bool real = d.storage <= CUBUTTERFLY_DATA_FP64;
    const bool complex = d.storage >= CUBUTTERFLY_DATA_COMPLEX_FP16 &&
                         d.storage <= CUBUTTERFLY_DATA_COMPLEX_FP64;
    const bool integer_storage = d.storage >= CUBUTTERFLY_DATA_UINT32;
    const bool low = d.storage == CUBUTTERFLY_DATA_FP16 || d.storage == CUBUTTERFLY_DATA_BF16 ||
                     d.storage == CUBUTTERFLY_DATA_COMPLEX_FP16 || d.storage == CUBUTTERFLY_DATA_COMPLEX_BF16;
    const auto expected = integer_storage
        ? (d.storage == CUBUTTERFLY_DATA_UINT32 ? CUBUTTERFLY_COMPUTE_UINT32 : CUBUTTERFLY_COMPUTE_UINT64)
        : ((d.storage == CUBUTTERFLY_DATA_FP64 || d.storage == CUBUTTERFLY_DATA_COMPLEX_FP64)
           ? CUBUTTERFLY_COMPUTE_FP64 : CUBUTTERFLY_COMPUTE_FP32);
    require(d.compute == CUBUTTERFLY_COMPUTE_NATIVE ||
                (integer_storage ? (d.compute == CUBUTTERFLY_COMPUTE_UINT32 ||
                                    d.compute == CUBUTTERFLY_COMPUTE_UINT64)
                                 : (d.compute == CUBUTTERFLY_COMPUTE_FP32 ||
                                    d.compute == CUBUTTERFLY_COMPUTE_FP64)),
            "compute domain is incompatible with storage");
    if (d.compute == CUBUTTERFLY_COMPUTE_NATIVE && !low) d.compute = expected;
    if (d.op == CUBUTTERFLY_OPERATOR_FFT) {
        require(d.fft_type == CUBUTTERFLY_FFT_C2C ? complex : real,
                "FFT transform and storage are incompatible");
        if (d.fft_type != CUBUTTERFLY_FFT_C2C)
            require((d.fft_type == CUBUTTERFLY_FFT_R2C) ==
                        (d.direction == CUBUTTERFLY_DIRECTION_FORWARD),
                    "R2C requires forward direction and C2R requires inverse direction");
    } else {
        require(d.fft_type == CUBUTTERFLY_FFT_C2C && d.fft_output_scale == 1.0,
                "FFT parameters require FFT operator");
        if (d.op == CUBUTTERFLY_OPERATOR_NTT) require(integer_storage, "NTT requires integer storage");
    }
    if (d.op != CUBUTTERFLY_OPERATOR_FFT && d.op != CUBUTTERFLY_OPERATOR_NTT &&
        d.length_mode == CUBUTTERFLY_LENGTH_STANDARD)
        for (auto n : d.extents) require(power_two(n), "butterfly extent must be power of two");
    if (d.op == CUBUTTERFLY_OPERATOR_NTT) {
        require(d.word_bits == (d.storage == CUBUTTERFLY_DATA_UINT32 ? 32U : 64U),
                "NTT word bits must match storage");
        require(d.modulus > 2 && (d.modulus & 1) &&
                    (d.word_bits == 64 || d.modulus <= std::numeric_limits<std::uint32_t>::max()),
                "invalid NTT modulus");
    } else {
        require(d.modulus == cuntt::kDefaultModulus && d.word_bits == 64,
                "modulus and word bits require NTT operator");
        require(d.input_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_NATURAL &&
                    d.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_NATURAL,
                "NTT layouts require NTT operator");
    }
    const auto input = side_extents(d, true), output = side_extents(d, false);
    normalize_orders(d.input_orders, input); normalize_orders(d.output_orders, output);
    // The legacy rank-one bit-reversed boundary names the same logical order.
    for (bool is_input : {true, false}) {
        auto& layout = is_input ? d.input_ntt_layout : d.output_ntt_layout;
        auto& orders = is_input ? d.input_orders : d.output_orders;
        const auto& extents = is_input ? input : output;
        if (layout == CUBUTTERFLY_NTT_LAYOUT_BIT_REVERSED) {
            require(d.rank == 1 && power_two(extents[0]), "legacy bit reversal requires rank-one power-of-two shape");
            const auto& supplied_orders = is_input ? source.input_orders : source.output_orders;
            require(supplied_orders.empty() || supplied_orders[0] == CUBUTTERFLY_ORDER_BIT_REVERSED,
                    "legacy bit reversal conflicts with explicit logical order");
            orders[0] = CUBUTTERFLY_ORDER_BIT_REVERSED;
            layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
        }
    }
    if (d.input_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC ||
        d.output_ntt_layout == CUBUTTERFLY_NTT_LAYOUT_APPT_STATIC) {
        require(std::all_of(d.input_orders.begin(), d.input_orders.end(), [](auto o) { return o == CUBUTTERFLY_ORDER_NATURAL; }) &&
                std::all_of(d.output_orders.begin(), d.output_orders.end(), [](auto o) { return o == CUBUTTERFLY_ORDER_NATURAL; }),
                "APPT layout cannot be combined with logical bit reversal");
    }
    normalize_layout(d.input_strides, d.input_batch_stride, input, d.batch);
    normalize_layout(d.output_strides, d.output_batch_stride, output, d.batch);
    d.stage_matrices.resize(std::max<std::size_t>(3, std::max<std::size_t>(d.rank, d.stage_matrices.size())));
    for (std::size_t axis = 0; axis < d.stage_matrices.size(); ++axis) {
        const auto& matrices = d.stage_matrices[axis];
        require(axis < d.rank || matrices.empty(), "matrices supplied beyond rank");
        require(d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2 || matrices.empty(), "matrices require structured operator");
        if (d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2 && axis < d.rank) {
            const auto length = next_power(d.extents[axis]);
            require(matrices.size() == 1 || matrices.size() == log_two(length),
                    "structured axis requires one matrix or one per stage");
        }
        for (const auto& m : matrices) {
            require(std::isfinite(m.m00) && std::isfinite(m.m01) && std::isfinite(m.m10) && std::isfinite(m.m11),
                    "matrix entries must be finite");
            if (d.direction == CUBUTTERFLY_DIRECTION_INVERSE) {
                const double determinant = m.m00 * m.m11 - m.m01 * m.m10;
                require(std::isfinite(determinant) && determinant != 0, "inverse matrix must be nonsingular");
            }
        }
    }
    if (d.direction == CUBUTTERFLY_DIRECTION_FORWARD) d.normalize_inverse = false;
    else if (d.op == CUBUTTERFLY_OPERATOR_NTT) d.normalize_inverse = true;
    else if (d.op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || d.op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA ||
             d.op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) d.normalize_inverse = false;
    return d;
}

json semantic_json(const cubutterflyDescriptor& descriptor) {
    const auto d = normalize_descriptor(descriptor);
    auto matrices = json::array();
    for (std::size_t axis = 0; axis < d.rank; ++axis) {
        auto list = json::array();
        for (const auto& m : d.stage_matrices[axis]) list.push_back({m.m00, m.m01, m.m10, m.m11});
        matrices.push_back(std::move(list));
    }
    json result{{"schema", "cubutterfly-semantic-v1"}, {"operator", int(d.op)}, {"storage", int(d.storage)},
            {"compute", int(d.compute)}, {"shape", d.extents}, {"batch", d.batch}, {"inverse", int(d.direction)},
            {"normalize_inverse", d.normalize_inverse}, {"placement", int(d.placement)}, {"fft_type", int(d.fft_type)},
            {"fft_output_scale", d.fft_output_scale}, {"length_mode", int(d.length_mode)},
            {"input_strides", d.input_strides}, {"output_strides", d.output_strides},
            {"input_batch_stride", d.input_batch_stride}, {"output_batch_stride", d.output_batch_stride},
            {"input_orders", d.input_orders}, {"output_orders", d.output_orders},
            {"ntt_input_layout", int(d.input_ntt_layout)}, {"ntt_output_layout", int(d.output_ntt_layout)},
            {"modulus", d.op == CUBUTTERFLY_OPERATOR_NTT ? json(d.modulus) : json(nullptr)},
            {"word_bits", d.op == CUBUTTERFLY_OPERATOR_NTT ? json(d.word_bits) : json(nullptr)}, {"matrices", matrices}};
    if (d.output_scale != 1.0) result["output_scale"] = d.output_scale;
    if (d.output_rounding != CUBUTTERFLY_ROUNDING_STAGE_STORAGE)
        result["output_rounding"] = int(d.output_rounding);
    return result;
}

json descriptor_json(const cubutterflyDescriptor& descriptor) {
    const auto d = normalize_descriptor(descriptor);
    return {{"schema", "cubutterfly-descriptor-v1"}, {"semantics", semantic_json(d)},
            {"selection", {{"execution_mode", int(d.execution_mode)}, {"algorithm_policy", int(d.algorithm_policy)},
                           {"explicit_algorithm", d.explicit_algorithm}}}};
}

cubutterflyDescriptor descriptor_from_json(const json& value) {
    exact_fields(value, {"schema", "semantics", "selection"});
    require(value.at("schema").is_string() && value.at("schema") == "cubutterfly-descriptor-v1", "invalid descriptor schema");
    const auto& s = value.at("semantics");
    require(s.is_object(), "JSON semantics object required");
    auto required_semantics = s;
    required_semantics.erase("output_scale");
    required_semantics.erase("output_rounding");
    exact_fields(required_semantics, {"schema", "operator", "storage", "compute", "shape", "batch", "inverse", "normalize_inverse",
                     "placement", "fft_type", "fft_output_scale", "length_mode", "input_strides", "output_strides",
                     "input_batch_stride", "output_batch_stride", "modulus", "word_bits", "matrices",
                     "input_orders", "output_orders", "ntt_input_layout", "ntt_output_layout"});
    require(s.at("schema").is_string() && s.at("schema") == "cubutterfly-semantic-v1", "invalid semantic schema");
    cubutterflyDescriptor d;
    d.op = enumeration<cubutterflyOperator_t>(s.at("operator"), 5);
    d.storage = enumeration<cubutterflyDataType_t>(s.at("storage"), 9);
    d.compute = enumeration<cubutterflyComputeType_t>(s.at("compute"), 4);
    d.direction = enumeration<cubutterflyDirection_t>(s.at("inverse"), 1);
    d.placement = enumeration<cubutterflyPlacement_t>(s.at("placement"), 1);
    d.fft_type = enumeration<cubutterflyFftTransformType_t>(s.at("fft_type"), 2);
    d.length_mode = enumeration<cubutterflyLengthMode_t>(s.at("length_mode"), 1);
    d.input_ntt_layout = enumeration<cubutterflyNttLayout_t>(s.at("ntt_input_layout"), 2);
    d.output_ntt_layout = enumeration<cubutterflyNttLayout_t>(s.at("ntt_output_layout"), 2);
    d.extents = size_array(s.at("shape"));
    require(d.extents.size() <= std::numeric_limits<std::uint32_t>::max(), "JSON rank out of range");
    d.rank = static_cast<std::uint32_t>(d.extents.size());
    d.batch = integer<std::size_t>(s.at("batch"));
    d.input_strides = size_array(s.at("input_strides")); d.output_strides = size_array(s.at("output_strides"));
    d.input_batch_stride = integer<std::size_t>(s.at("input_batch_stride"));
    d.output_batch_stride = integer<std::size_t>(s.at("output_batch_stride"));
    d.input_orders = order_array(s.at("input_orders")); d.output_orders = order_array(s.at("output_orders"));
    require(s.at("normalize_inverse").is_boolean(), "JSON boolean required");
    d.normalize_inverse = s.at("normalize_inverse").get<bool>();
    d.fft_output_scale = real_number(s.at("fft_output_scale"));
    if (s.contains("output_scale")) d.output_scale = real_number(s.at("output_scale"));
    if (s.contains("output_rounding"))
        d.output_rounding = enumeration<cubutterflyOutputRounding_t>(s.at("output_rounding"), 1);
    if (d.op == CUBUTTERFLY_OPERATOR_NTT) {
        d.modulus = integer<std::uint64_t>(s.at("modulus")); d.word_bits = integer<std::uint32_t>(s.at("word_bits"));
    } else require(s.at("modulus").is_null() && s.at("word_bits").is_null(), "non-NTT modulus fields must be null");
    const auto& matrices = s.at("matrices");
    require(matrices.is_array() && matrices.size() == d.rank, "JSON matrix rank mismatch");
    d.stage_matrices.resize(std::max<std::size_t>(3, d.rank));
    for (std::size_t axis = 0; axis < d.rank; ++axis) {
        require(matrices[axis].is_array(), "JSON matrix list required");
        for (const auto& m : matrices[axis]) {
            require(m.is_array() && m.size() == 4, "JSON matrix requires four entries");
            d.stage_matrices[axis].push_back({real_number(m[0]), real_number(m[1]), real_number(m[2]), real_number(m[3])});
        }
    }
    const auto& selection = value.at("selection");
    exact_fields(selection, {"execution_mode", "algorithm_policy", "explicit_algorithm"});
    d.execution_mode = enumeration<cubutterflyExecutionMode_t>(selection.at("execution_mode"), 2);
    d.algorithm_policy = enumeration<cubutterflyAlgorithmPolicy_t>(selection.at("algorithm_policy"), 2);
    require(selection.at("explicit_algorithm").is_string(), "JSON algorithm string required");
    d.explicit_algorithm = selection.at("explicit_algorithm").get<std::string>();
    return normalize_descriptor(d);
}
}
