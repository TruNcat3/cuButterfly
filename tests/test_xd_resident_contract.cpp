#include "xd_resident.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>

int main() try {
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
        cuntt::XdResidentConfig config;
        config.extents = {4, 4};
        if (scenario == 0) config.threads = 0;
        if (scenario == 1) config.axis_order = {0};
        if (scenario == 2) config.axis_order = {0, 0};
        if (scenario == 3) config.extents.assign(9, 2);
        // Semantic rejection occurs before any CUDA device query/allocation.
        cuntt::XdResidentPlan plan(config);
        const auto mapping = nlohmann::json::parse(plan.mapping_json());
        if (plan.available() || mapping.at("available").get<bool>() ||
            mapping.at("reason").get<std::string>().empty() ||
            !mapping.at("axes").empty())
            throw std::runtime_error("rejected resident plan lost diagnostic mapping");
    }
    std::cout << "xD resident rejected mapping contract passed: 4 cases\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
