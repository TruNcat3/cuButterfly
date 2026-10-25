#include "xd_real_fft.hpp"

#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct MixedValidationPlan {
    std::unique_ptr<cuntt::detail::XdRealFftPlan> plan;
    std::string mapping;
};
}

extern "C" {
const char* xd_mixed_error() { return last_error.c_str(); }

void* xd_mixed_create(const std::size_t* shape, unsigned rank, std::size_t batch,
                     int inverse, int normalize, int fp64, int complex_only,
                     double output_scale, const unsigned* cores,
                     const unsigned* exchanges, const std::size_t* order,
                     std::size_t batch_tile, int c2r_prefix_scale) {
    last_error.clear();
    try {
        using namespace cuntt::detail;
        const unsigned count = complex_only ? rank : rank - 1;
        std::vector<XdAxisResources> resources(count);
        for (unsigned axis = 0; axis < count; ++axis) {
            resources[axis].core = cores[axis] ? XdAxisCore::Vkfft : XdAxisCore::CufftdxBlock;
            resources[axis].exchange = exchanges[axis];
        }
        auto value = std::make_unique<MixedValidationPlan>();
        value->plan = std::make_unique<XdRealFftPlan>(
            std::vector<std::size_t>(shape, shape + rank), batch, inverse != 0,
            normalize != 0, fp64 != 0, complex_only != 0, output_scale,
            true, std::vector<std::size_t>(order, order + count), true, 8,
            false, XdAxisCore::CufftdxBlock, XdPairCore::Thread, 4, 0,
            batch_tile, 0, 0, resources, c2r_prefix_scale != 0);
        value->mapping = value->plan->mapping_json();
        return value.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

const char* xd_mixed_mapping(void* handle) {
    return static_cast<MixedValidationPlan*>(handle)->mapping.c_str();
}
std::size_t xd_mixed_workspace_size(void* handle) {
    return static_cast<MixedValidationPlan*>(handle)->plan->workspace_size();
}
int xd_mixed_set_workspace(void* handle, void* workspace, std::size_t bytes) {
    try {
        static_cast<MixedValidationPlan*>(handle)->plan->set_workspace(workspace, bytes);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
int xd_mixed_execute(void* handle, const void* input, void* output, void* stream) {
    try {
        static_cast<MixedValidationPlan*>(handle)->plan->execute(
            input, output, static_cast<cudaStream_t>(stream));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
void xd_mixed_destroy(void* handle) { delete static_cast<MixedValidationPlan*>(handle); }
}
