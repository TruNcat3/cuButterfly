// Original kernels remain source-identical. The second namespace fixes the
// provider at compile time; Graph preparation replaces only kernel identity.
#include <heongpu/kernel/bootstrapping.cuh>
#include "tfhe_shared_ntt.cuh"
#include "tfhe_graph_contract.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

#include CUBUTTERFLY_HEONGPU_TFHE_BOOTSTRAP_SOURCE

namespace cubutterfly_tfhe_variant { using namespace heongpu; }
#define heongpu cubutterfly_tfhe_variant
#define SmallForwardNTT cubutterfly_tfhe_forward
#define SmallInverseNTT cubutterfly_tfhe_inverse
#include CUBUTTERFLY_HEONGPU_TFHE_BOOTSTRAP_SOURCE
#undef SmallForwardNTT
#undef SmallInverseNTT
#undef heongpu

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Entry {
    void* original;
    void* replacement;
    std::vector<std::size_t> sizes;
};
template<class... Args>
Entry entry(void (*original)(Args...), void (*replacement)(Args...)) {
    return {reinterpret_cast<void*>(original), reinterpret_cast<void*>(replacement), {sizeof(Args)...}};
}
std::vector<unsigned char> arguments(const cudaKernelNodeParams& params,
                                     const std::vector<std::size_t>& sizes) {
    if (!params.kernelParams || params.extra)
        throw std::runtime_error("TFHE Graph requires explicit kernel parameter arrays");
    std::vector<unsigned char> bytes;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const auto* value = static_cast<const unsigned char*>(params.kernelParams[i]);
        bytes.insert(bytes.end(), value, value + sizes[i]);
    }
    return bytes;
}
}

std::size_t heongpu::cubutterfly_tfhe_fix_graph(cudaGraph_t graph, bool shared,
                                              std::size_t bootstraps) {
    // Deduction requires identical complete parameter types before substitution.
    const std::array<Entry, 4> entries{{
        entry(tfhe_bootstrapping_kernel_unique_step1, cubutterfly_tfhe_variant::tfhe_bootstrapping_kernel_unique_step1),
        entry(tfhe_bootstrapping_kernel_regular_step1, cubutterfly_tfhe_variant::tfhe_bootstrapping_kernel_regular_step1),
        entry(tfhe_bootstrapping_kernel_unique_step2, cubutterfly_tfhe_variant::tfhe_bootstrapping_kernel_unique_step2),
        entry(tfhe_bootstrapping_kernel_regular_step2, cubutterfly_tfhe_variant::tfhe_bootstrapping_kernel_regular_step2),
    }};
    const std::vector<void*> unchanged{
        reinterpret_cast<void*>(tfhe_nand_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_and_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_and_first_not_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_nor_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_or_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_xnor_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_xor_pre_comp_kernel),
        reinterpret_cast<void*>(tfhe_not_comp_kernel),
        reinterpret_cast<void*>(tfhe_sample_extraction_kernel),
        reinterpret_cast<void*>(tfhe_key_switching_kernel),
    };
    std::array<const void*, 4> originals{};
    for (std::size_t i = 0; i < entries.size(); ++i) originals[i] = entries[i].original;
    std::array<std::size_t, 4> counts{};
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count));
    for (auto node : nodes) {
        cudaGraphNodeType type;
        check(cudaGraphNodeGetType(node, &type));
        if (type == cudaGraphNodeTypeGraph)
            throw std::runtime_error("unexpected child graph in TFHE complete capture");
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams params{};
        check(cudaGraphKernelNodeGetParams(node, &params));
        const std::size_t match = cubutterfly::heongpu_tfhe::identity_index(params.func, originals);
        if (match == entries.size()) {
            if (std::find(unchanged.begin(), unchanged.end(), params.func) == unchanged.end())
                throw std::runtime_error("unknown kernel identity in TFHE Graph; refusing partial replacement");
            continue;
        }
        ++counts[match];
        const auto before = arguments(params, entries[match].sizes);
        if (shared) {
            const auto saved = params;
            params.func = entries[match].replacement;
            check(cudaGraphKernelNodeSetParams(node, &params));
            check(cudaGraphKernelNodeGetParams(node, &params));
            if (params.func != entries[match].replacement ||
                params.gridDim.x != saved.gridDim.x || params.gridDim.y != saved.gridDim.y || params.gridDim.z != saved.gridDim.z ||
                params.blockDim.x != saved.blockDim.x || params.blockDim.y != saved.blockDim.y || params.blockDim.z != saved.blockDim.z ||
                params.sharedMemBytes != saved.sharedMemBytes || arguments(params, entries[match].sizes) != before)
                throw std::runtime_error("TFHE Graph replacement changed launch/parameter contract");
        }
    }
    cubutterfly::heongpu_tfhe::validate_bootstrap_counts(counts, bootstraps);
    return counts[0] + counts[1] + counts[2] + counts[3];
}
