#include "xd_vkfft_axis.hpp"

#include <cstdint>
#include <exception>
#include <string>

namespace {
thread_local std::string error;
thread_local std::string metadata;
}

extern "C" {
const char* vka_error() { return error.c_str(); }
void* vka_create(std::size_t length, std::size_t suffix, std::size_t lines,
                 int fp64, int inverse) {
    try {
        return new cuntt::detail::XdVkfftAxisPlan(length, suffix, lines, fp64 != 0, inverse != 0);
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
}
int vka_execute(void* plan, std::uintptr_t input, std::uintptr_t output, std::uintptr_t stream) {
    try {
        static_cast<cuntt::detail::XdVkfftAxisPlan*>(plan)->execute(
            reinterpret_cast<const void*>(input), reinterpret_cast<void*>(output),
            reinterpret_cast<cudaStream_t>(stream));
        return 0;
    } catch (const std::exception& exception) {
        error = exception.what();
        return -1;
    }
}
const char* vka_metadata(void* plan) {
    metadata = static_cast<cuntt::detail::XdVkfftAxisPlan*>(plan)->metadata_json();
    return metadata.c_str();
}
void vka_destroy(void* plan) { delete static_cast<cuntt::detail::XdVkfftAxisPlan*>(plan); }
}
