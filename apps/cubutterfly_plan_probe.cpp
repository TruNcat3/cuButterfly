#include <cubutterfly/mapping.hpp>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>

using Json = nlohmann::json;

Json hardware() {
    const auto h = cuntt::query_hardware_resource_model();
    return {{"device", h.device_name}, {"sm_count", h.sm_count}, {"memory_bytes", h.memory_bytes},
        {"max_blocks_per_sm", h.max_blocks_per_sm}, {"max_threads_per_sm", h.max_threads_per_sm},
        {"registers_per_sm", h.registers_per_sm}, {"shared_bytes_per_sm", h.shared_bytes_per_sm},
        {"max_threads_per_block", h.max_threads_per_block}};
}

template<class Plan> Json describe(const Plan& plan) {
    Json edges = Json::array();
    for (const auto& edge : plan.dataflow_plan().execution_boundaries)
        edges.push_back({{"producer_group", edge.producer_group}, {"consumer_group", edge.consumer_group},
            {"storage", static_cast<int>(edge.storage)}, {"bytes", edge.bytes}, {"buffers", edge.buffers},
            {"producer_layout", edge.producer_layout}, {"consumer_layout", edge.consumer_layout}});
    return {{"status", "resolved"}, {"mapping_json", cubutterfly::serialize_mapping(plan.config())},
        {"execution_groups_json", Json::parse(cubutterfly::execution_groups_json(plan.dataflow_plan()))},
        {"execution_boundaries", edges}, {"workspace_bytes", plan.workspace_size()},
        {"runtime_fingerprint", cubutterfly::runtime_fingerprint()}, {"descriptor_source", "resolved-plan"},
        {"hardware", hardware()}};
}

Json probe(const Json& point) {
    const auto mapping = point.at("mapping_json").get<std::string>();
    if (point.value("operator", "fft") == "ntt") {
        cuntt::PlanConfig config;
        config.log_n = point.at("logN"); config.batch = point.at("batch");
        config.word_bits = std::stoi(point.value("precision", "word32").substr(4));
        config.modulus = point.value("modulus", cuntt::kDefaultModulus);
        config.inverse = point.value("direction", "forward") == "inverse";
        auto placement = point.value("placement", "natural");
        if (placement == "out-of-place" || placement == "in-place") placement = "natural";
        config.output_order = cuntt::parse_output_order(point.value("output_order", placement));
        config.input_order = cuntt::parse_input_order(point.value("input_order", "natural"));
        cubutterfly::apply_serialized_mapping(mapping, config);
        return describe(cuntt::Plan(config));
    }
    cuntt::ButterflyConfig config;
    config.op = cuntt::parse_butterfly_operator(point.value("operator", "fft"));
    config.precision = cuntt::parse_butterfly_precision(point.value("precision", "fp32"));
    config.accumulation = cuntt::parse_butterfly_accumulation(point.value("accumulation", "native"));
    config.placement = cuntt::parse_butterfly_placement(point.value("placement", "out-of-place"));
    config.log_n = point.at("logN"); config.batch = point.at("batch");
    config.inverse = point.value("direction", "forward") == "inverse";
    config.normalize_inverse = point.value("normalization", "inverse") != "none";
    config.element_stride = point.value("element_stride", std::size_t{1});
    config.batch_stride = point.value("batch_stride", std::size_t{0});
    if (point.contains("stage_matrix")) {
        auto text = point.at("stage_matrix").get<std::string>();
        for (auto& ch : text) if (ch == ',') ch = ' ';
        std::istringstream stream(text);
        cuntt::ButterflyMatrix2x2 matrix;
        if (!(stream >> matrix.m00 >> matrix.m01 >> matrix.m10 >> matrix.m11))
            throw std::invalid_argument("stage_matrix requires four coefficients");
        config.stage_matrices.push_back(matrix);
    }
    cubutterfly::apply_serialized_mapping(mapping, config);
    return describe(cuntt::ButterflyPlan(config));
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--hardware") std::cout << hardware() << '\n';
        else if (argc == 3 && std::string(argv[1]) == "--point-json")
            std::cout << probe(Json::parse(argv[2])) << '\n';
        else throw std::invalid_argument("usage: cubutterfly_plan_probe --hardware | --point-json JSON");
        return 0;
    } catch (const std::exception& error) {
        std::cout << Json{{"status", "unavailable"}, {"reason", error.what()}} << '\n';
        return 1;
    }
}
