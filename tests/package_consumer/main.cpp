#include <string_view>

#include "cuntt/ntt.hpp"
#include <cubutterfly/plan.hpp>

int main() {
    cuntt::ButterflyConfig config;
    auto mapping = cubutterfly::serialize_mapping(config);
    cuntt::ButterflyConfig replay;
    cubutterfly::apply_serialized_mapping(mapping, replay);
    return std::string_view(cuntt::backend_name(cuntt::Backend::Hybrid2D)) == "hybrid2d" &&
           cubutterfly::serialize_mapping(replay) == mapping ? 0 : 1;
}
