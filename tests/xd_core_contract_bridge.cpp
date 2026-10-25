#include "xd_real_fft.hpp"

#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
struct ContractPlan {
    std::unique_ptr<cuntt::detail::XdRealFftPlan> plan;
    std::string mapping;
};
}

extern "C" {
const char* xd_contract_error() { return last_error.c_str(); }

void* xd_contract_create(const std::size_t* shape, unsigned rank, std::size_t batch,
                         int inverse, int normalize, int fp64, int complex_only,
                         double output_scale, unsigned core, unsigned cap,
                         unsigned twiddle, unsigned exchange) {
    last_error.clear();
    try {
        using namespace cuntt::detail;
        const unsigned count = complex_only ? rank : rank - 1;
        std::vector<XdAxisResources> resources(count);
        std::vector<std::size_t> order(count);
        for (unsigned axis = 0; axis < count; ++axis) {
            order[axis] = axis;
            resources[axis].core = core == 0 ? XdAxisCore::ThreadShared : XdAxisCore::CufftdxBlock;
            resources[axis].block_cap = cap;
            resources[axis].local_twiddle_policy = twiddle;
            resources[axis].exchange = exchange;
        }
        auto value = std::make_unique<ContractPlan>();
        value->plan = std::make_unique<XdRealFftPlan>(
            std::vector<std::size_t>(shape, shape + rank), batch, inverse != 0,
            normalize != 0, fp64 != 0, complex_only != 0, output_scale,
            true, order, true, 8, false,
            XdAxisCore::CufftdxBlock, XdPairCore::Thread, 4, 0, 0, 0,
            exchange, resources, false);
        value->mapping = value->plan->mapping_json();
        return value.release();
    } catch (const std::exception& error) {
        last_error = error.what();
        return nullptr;
    }
}

const char* xd_contract_mapping(void* handle) {
    return static_cast<ContractPlan*>(handle)->mapping.c_str();
}
std::size_t xd_contract_workspace_size(void* handle) {
    return static_cast<ContractPlan*>(handle)->plan->workspace_size();
}
int xd_contract_set_workspace(void* handle, void* workspace, std::size_t bytes) {
    try {
        static_cast<ContractPlan*>(handle)->plan->set_workspace(workspace, bytes);
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
int xd_contract_execute(void* handle, const void* input, void* output, void* stream) {
    try {
        static_cast<ContractPlan*>(handle)->plan->execute(input, output, static_cast<cudaStream_t>(stream));
        return 0;
    } catch (const std::exception& error) {
        last_error = error.what();
        return 1;
    }
}
void xd_contract_destroy(void* handle) { delete static_cast<ContractPlan*>(handle); }
}
