#include <string_view>

#include "cuntt/ntt.hpp"

int main() {
    return std::string_view(cuntt::backend_name(cuntt::Backend::Hybrid2D)) == "hybrid2d" ? 0 : 1;
}
