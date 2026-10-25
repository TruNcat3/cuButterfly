#pragma once
#if __has_include("cufftdx_provider_capabilities.hpp")
#include "cufftdx_provider_capabilities.hpp"
#else
// CPU-only geometry tools can run without a configured provider. Absence is
// unknown capability, never evidence that a mathematical request is invalid.
namespace cuntt::detail {
enum class CufftDxSupport { Supported, Unsupported, Unknown };
inline constexpr const char* cufftdx_provider_fingerprint = "unconfigured";
inline CufftDxSupport cufftdx_provider_support_both(unsigned, unsigned, unsigned, unsigned) noexcept {
    return CufftDxSupport::Unknown;
}
}
#endif
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstdint>
#if __has_include("jit_config.hpp")
#include "jit_config.hpp"
#endif

namespace cuntt::detail {
class ProviderImplementationMissing : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

inline bool cufftdx_provider_matches_runtime() {
    // Runtime provider overrides may select a different package from the one
    // used to generate the C++ table. Only Python's actual-package check can
    // establish support then. Do this once per enumeration, not per candidate.
    const auto* override_root = std::getenv("CUBUTTERFLY_MATHDX_ROOT");
#ifdef CUBUTTERFLY_JIT_MATHDX
    std::error_code error;
    const auto configured = std::filesystem::weakly_canonical(CUBUTTERFLY_JIT_MATHDX, error);
    if (error || !std::filesystem::exists(configured, error) || error) return false;
    if (override_root) {
        const auto actual = std::filesystem::weakly_canonical(override_root, error);
        if (error || actual != configured) return false;
    }
#else
    if (override_root) return false;
#endif
#if __has_include("cufftdx_provider_capabilities.hpp")
    // A package can be upgraded in place while an installed binary remains.
    // Verify actual bytes, not just the path or modification timestamps.
    // FNV binds configuration contents; SHA256 provenance stays in metadata.
    for (const auto& source : cufftdx_provider_sources) {
        std::ifstream stream(source.path, std::ios::binary);
        if (!stream) return false;
        std::uint64_t digest = UINT64_C(14695981039346656037);
        char buffer[4096];
        while (stream.read(buffer, sizeof(buffer)) || stream.gcount())
            for (std::streamsize i=0; i<stream.gcount(); ++i) {
                digest ^= static_cast<unsigned char>(buffer[i]);
                digest *= UINT64_C(1099511628211);
            }
        if (!stream.eof() || digest != source.fnv1a64) return false;
    }
    return true;
#else
    return false;
#endif
}

inline const char* cufftdx_support_name(CufftDxSupport status) noexcept {
    switch (status) {
    case CufftDxSupport::Supported: return "supported";
    case CufftDxSupport::Unsupported: return "provider-implementation-missing";
    case CufftDxSupport::Unknown: return "provider-capability-unknown";
    }
    return "provider-capability-unknown";
}

inline void require_cufftdx_provider(unsigned sm, unsigned precision_bits, unsigned size, unsigned ept) {
    if (!cufftdx_provider_matches_runtime()) return;
    if (cufftdx_provider_support_both(sm, precision_bits, size, ept) == CufftDxSupport::Unsupported)
        throw ProviderImplementationMissing("provider-implementation-missing: cuFFTDx SM" + std::to_string(sm) +
            " FP" + std::to_string(precision_bits) + " local N=" + std::to_string(size) +
            " EPT=" + std::to_string(ept) +
            " has no implementation for the generated forward/inverse module; the FFT request remains valid");
    // An unrecognized provider version/SM is resolved by actual compilation.
    // Do not remove candidates or advertise unsupported semantics on a guess.
}
}
