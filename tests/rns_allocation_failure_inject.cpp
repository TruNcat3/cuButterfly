#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <atomic>
#include <cstddef>

namespace {
std::atomic<bool> fail_next{false};
std::atomic<unsigned> failures{0};
}

extern "C" void arm_next_cuda_malloc_failure() {
    fail_next.store(true, std::memory_order_release);
}

extern "C" unsigned rns_injected_cuda_malloc_failures() {
    return failures.load(std::memory_order_acquire);
}

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** pointer, std::size_t bytes) {
    if (fail_next.exchange(false, std::memory_order_acq_rel)) {
        failures.fetch_add(1, std::memory_order_release);
        return cudaErrorMemoryAllocation;
    }
    using Allocate = cudaError_t (CUDARTAPI *)(void**, std::size_t);
    static const auto allocate = reinterpret_cast<Allocate>(dlsym(RTLD_NEXT, "cudaMalloc"));
    return allocate ? allocate(pointer, bytes) : cudaErrorUnknown;
}
