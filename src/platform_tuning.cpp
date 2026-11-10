#include "platform_tuning.hpp"
#include "plan_registry.hpp"
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace cuntt::detail {
namespace {
using Json=nlohmann::json;
std::string rule_text() {
    const auto* path=std::getenv("CUBUTTERFLY_PLATFORM_TUNING");
    if (!path || !*path) return {};
    std::ifstream input(path);
    if (!input) throw std::invalid_argument(std::string("cannot read platform tuning file: ")+path);
    return {std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>()};
}
Json hardware_identity() {
    const auto d=current_device_info();
    return {{"device",d.name},{"compute_capability",std::to_string(d.compute_major)+"."+std::to_string(d.compute_minor)},
            {"global_memory_bytes",d.global_memory_bytes}};
}
std::size_t effective_batch_stride(const ButterflyConfig& c) {
    if (c.batch_stride || c.log_n >= std::numeric_limits<std::size_t>::digits)
        return c.batch_stride;
    return ((std::size_t{1} << c.log_n) - 1) * c.element_stride + 1;
}
std::string stage_matrices(const ButterflyConfig& c) {
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (std::size_t index=0; index<c.stage_matrices.size(); ++index) {
        if (index) text << 'x';
        const auto& matrix=c.stage_matrices[index];
        text << matrix.m00 << ':' << matrix.m01 << ':' << matrix.m10 << ':' << matrix.m11;
    }
    return text.str();
}
const char* normalization(const ButterflyConfig& c) noexcept {
    return c.inverse && c.normalize_inverse &&
                   (c.op == ButterflyOperator::Fft || c.op == ButterflyOperator::Fwht)
               ? "inverse"
               : "none";
}
Json semantics(const PlanConfig& c) {
    return {{"operator","ntt"},{"precision","word"+std::to_string(c.word_bits)},
            {"logN",c.log_n},{"batch",c.batch},{"modulus",c.modulus},
            {"direction",c.inverse ? "inverse" : "forward"},
            {"input_order",input_order_name(c.input_order)},{"output_order",output_order_name(c.output_order)}};
}
Json semantics(const ButterflyConfig& c) {
    return {{"operator",butterfly_operator_name(c.op)},{"precision",butterfly_precision_name(c.precision)},
            {"logN",c.log_n},{"batch",c.batch},{"direction",c.inverse ? "inverse" : "forward"},
            {"placement",butterfly_placement_name(c.placement)},{"normalization",normalization(c)},
            // Keep the original C++ spelling as an input alias for existing
            // profiles while exposing the registry's canonical field above.
            {"normalize_inverse",c.normalize_inverse},
            {"accumulation",butterfly_accumulation_name(c.accumulation)},
            {"output_scale",c.output_scale},{"output_rounding",c.final_storage_rounding ? 1 : 0},
            {"element_stride",c.element_stride},{"batch_stride",effective_batch_stride(c)},
            {"stage_matrices",stage_matrices(c)}};
}
void warn_invalid_mapping(const std::exception& error) {
    std::cerr << "cuButterfly: skipping invalid platform tuning mapping: " << error.what() << '\n';
}
template<class Config>
std::vector<Config> apply_rule_candidates(const Config& base, const Json& document, const Json& hardware) {
    std::vector<Config> out;
    for (const auto& mapping:match_platform_rules(document,hardware,semantics(base))) {
        try {
            auto c=base; apply_serialized_mapping(mapping.dump(),c); c.auto_select=false;
            out.push_back(std::move(c));
        } catch (const std::exception& error) {
            warn_invalid_mapping(error);
        }
    }
    return out;
}
template<class Config> std::vector<Config> file_candidates(const Config& base) {
    const auto text=rule_text();
    if (text.empty()) return {};
    return apply_rule_candidates(base,Json::parse(text),hardware_identity());
}

std::string hex_encode(const std::string& value) {
    static constexpr char digits[]="0123456789abcdef";
    if (value.size() > std::numeric_limits<std::size_t>::max()/2)
        throw std::length_error("platform tuning identity is too large");
    std::string encoded;
    encoded.reserve(value.size()*2);
    for (const unsigned char byte : value) {
        encoded.push_back(digits[byte >> 4]);
        encoded.push_back(digits[byte & 0x0f]);
    }
    return encoded;
}
}

std::vector<nlohmann::json> match_platform_rules(const Json& document,const Json& hardware,const Json& semantic) {
    if (document.value("schema","")!="cubutterfly-platform-tuning-v1" || !document.at("targets").is_array())
        throw std::invalid_argument("unsupported platform tuning schema");
    std::vector<Json> out;
    for (const auto& target:document.at("targets")) {
        const auto& match=target.at("hardware");
        if (!match.is_object() || match.size()!=3 || !match.contains("device") ||
            !match.contains("compute_capability") || !match.contains("global_memory_bytes"))
            throw std::invalid_argument("platform tuning needs exact device, compute capability and memory identity");
        if (match!=hardware) continue;
        for (const auto& rule:target.at("rules")) {
            const auto& filters=rule.at("match");
            if (!filters.is_object() || !filters.contains("operator") || !filters.contains("precision") ||
                !filters.contains("direction") || !rule.contains("mapping"))
                throw std::invalid_argument("platform rule requires operator, precision, direction and mapping");
            // Different operators have different semantic fields. Inspect an
            // operator's rule only when its family matches.
            if (filters.at("operator")!=semantic.at("operator")) continue;
            bool matches=true;
            for (const auto& [key,value]:filters.items()) {
                if (!semantic.contains(key)) throw std::invalid_argument("unknown platform semantic field: "+key);
                if (value.is_object()) {
                    if (value.empty() || !semantic.at(key).is_number())
                        throw std::invalid_argument("platform ranges require a numeric semantic field");
                    for (const auto& [bound,limit]:value.items()) {
                        if (!limit.is_number() || (bound!="min" && bound!="max"))
                            throw std::invalid_argument("invalid platform range bound");
                        if ((bound=="min" && semantic.at(key)<limit) || (bound=="max" && semantic.at(key)>limit)) matches=false;
                    }
                } else if (value!=semantic.at(key)) matches=false;
            }
            if (matches) out.push_back(rule.at("mapping"));
        }
    }
    return out;
}

std::vector<PlanConfig> platform_rule_candidates(const PlanConfig& config,
    const Json& document, const Json& hardware) {
    return apply_rule_candidates(config,document,hardware);
}

std::vector<ButterflyConfig> platform_rule_candidates(const ButterflyConfig& config,
    const Json& document, const Json& hardware) {
    return apply_rule_candidates(config,document,hardware);
}

std::string platform_tuning_identity() {
    // Encode the content, not the pathname: editing a profile must invalidate
    // process-local measured-plan caches as well as later campaign identities.
    // The encoding keeps the C API's line-oriented cache key single-line and
    // collision-free even when the source JSON is pretty-printed.
    return "platform-tuning-v1-hex|"+hex_encode(rule_text());
}

std::vector<PlanConfig> platform_candidates(const PlanConfig& config) {
    auto out=file_candidates(config);
    try {
        const auto prior=select_legacy_ntt_mapping(config);
        auto c=config; apply_serialized_mapping(serialize_mapping(prior.config),c); c.auto_select=false;
        out.push_back(std::move(c));
    } catch (const std::invalid_argument&) {}
    return out;
}
std::vector<ButterflyConfig> platform_candidates(const ButterflyConfig& config) {
    auto out=file_candidates(config);
    try {
        const auto prior=select_legacy_butterfly_mapping(config);
        auto c=config; apply_serialized_mapping(serialize_mapping(prior.config),c); c.auto_select=false;
        out.push_back(std::move(c));
    } catch (const std::invalid_argument&) {}
    return out;
}
} // namespace cuntt::detail
