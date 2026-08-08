#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cuntt/butterfly.hpp"

namespace cuntt {

const char* butterfly_precision_name(ButterflyPrecision precision) noexcept {
    switch (precision) {
        case ButterflyPrecision::Fp32:
            return "fp32";
        case ButterflyPrecision::Fp64:
            return "fp64";
        case ButterflyPrecision::Fp16:
            return "fp16";
        case ButterflyPrecision::Bf16:
            return "bf16";
        case ButterflyPrecision::Fp16Fp32:
            return "fp16-fp32";
        case ButterflyPrecision::Uint32:
            return "uint32";
    }
    return "unknown";
}

ButterflyPrecision parse_butterfly_precision(const std::string& name) {
    if (name == "fp32")
        return ButterflyPrecision::Fp32;
    if (name == "fp64")
        return ButterflyPrecision::Fp64;
    if (name == "fp16")
        return ButterflyPrecision::Fp16;
    if (name == "bf16")
        return ButterflyPrecision::Bf16;
    if (name == "fp16-fp32")
        return ButterflyPrecision::Fp16Fp32;
    if (name == "uint32")
        return ButterflyPrecision::Uint32;
    throw std::invalid_argument("unknown butterfly precision: " + name);
}

const char* butterfly_accumulation_name(ButterflyAccumulation accumulation) noexcept {
    return accumulation == ButterflyAccumulation::Fp32 ? "fp32" : "native";
}

ButterflyAccumulation parse_butterfly_accumulation(const std::string& name) {
    if (name == "native")
        return ButterflyAccumulation::Native;
    if (name == "fp32")
        return ButterflyAccumulation::Fp32;
    throw std::invalid_argument("unknown butterfly accumulation: " + name);
}

const char* butterfly_placement_name(ButterflyPlacement placement) noexcept {
    return placement == ButterflyPlacement::InPlace ? "in-place" : "out-of-place";
}

ButterflyPlacement parse_butterfly_placement(const std::string& name) {
    if (name == "in-place")
        return ButterflyPlacement::InPlace;
    if (name == "out-of-place")
        return ButterflyPlacement::OutOfPlace;
    throw std::invalid_argument("unknown butterfly placement: " + name);
}

std::vector<ButterflyCapability> butterfly_capabilities() {
    return {
        {ButterflyBackend::TemporalTile, 1, 15, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
         "generated and optional cuFFTDx/TurboFFT FFT cores have design-point-specific ranges"},
        {ButterflyBackend::Hierarchical, 6, 20, true, true, true, false, false, false, false, true, true, true, true, false, true, true, true,
         "local_stages=5..10 and local_stages<logN"},
        {ButterflyBackend::OnlineReorder, 6, 20, true, true, true, false, false, false, false, true, true, true, true, false, true, true, true,
         "scalar dimensions are <=10; cuFFTDx mixed block/direct dimensions are <=12"},
        {ButterflyBackend::WarpHybrid, 8, 8, true, false, false, false, false, false, false, true, true, true, true, false, true, true, true, "N=256"},
        {ButterflyBackend::StagePipeline, 8, 8, true, false, false, false, false, false, false, true, true, true, true, false, true, true, true,
         "FP64 complex exceeds V100 CTA shared memory"},
        {ButterflyBackend::CuFft, 1, 20, false, false, false, false, false, false, false, false, false, true, true, false, false, true, true,
         "FFT only; internal unit is opaque"},
    };
}

const char* complex_multiply_name(ComplexMultiply multiply) noexcept {
    switch (multiply) {
        case ComplexMultiply::FourMul:
            return "four-mul";
        case ComplexMultiply::Gauss3:
            return "gauss3";
    }
    return "unknown";
}

ComplexMultiply parse_complex_multiply(const std::string& name) {
    if (name == "four-mul")
        return ComplexMultiply::FourMul;
    if (name == "gauss3")
        return ComplexMultiply::Gauss3;
    throw std::invalid_argument("unknown complex multiply: " + name);
}

const char* cross_twiddle_mode_name(CrossTwiddleMode mode) noexcept {
    switch (mode) {
        case CrossTwiddleMode::Table:
            return "table";
        case CrossTwiddleMode::Recurrence:
            return "recurrence";
    }
    return "unknown";
}

CrossTwiddleMode parse_cross_twiddle_mode(const std::string& name) {
    if (name == "table")
        return CrossTwiddleMode::Table;
    if (name == "recurrence")
        return CrossTwiddleMode::Recurrence;
    throw std::invalid_argument("unknown cross twiddle mode: " + name);
}

const char* direct_boundary_name(DirectBoundary boundary) noexcept {
    switch (boundary) {
        case DirectBoundary::Strided:
            return "direct-strided";
        case DirectBoundary::TiledTranspose:
            return "tiled-transpose";
        case DirectBoundary::PrefixTiledTranspose:
            return "prefix-tiled-transpose";
    }
    return "unknown";
}

DirectBoundary parse_direct_boundary(const std::string& name) {
    if (name == "direct-strided")
        return DirectBoundary::Strided;
    if (name == "tiled-transpose")
        return DirectBoundary::TiledTranspose;
    if (name == "prefix-tiled-transpose")
        return DirectBoundary::PrefixTiledTranspose;
    throw std::invalid_argument("unknown direct boundary: " + name);
}

const char* fft_boundary_residency_name(FftBoundaryResidency residency) noexcept {
    switch (residency) {
        case FftBoundaryResidency::GlobalScratch:
            return "global-scratch";
        case FftBoundaryResidency::Fused:
            return "fused";
    }
    return "unknown";
}

FftBoundaryResidency parse_fft_boundary_residency(const std::string& name) {
    if (name == "global-scratch")
        return FftBoundaryResidency::GlobalScratch;
    if (name == "fused")
        return FftBoundaryResidency::Fused;
    throw std::invalid_argument("unknown FFT boundary residency: " + name);
}

const char* local_exchange_name(LocalExchange exchange) noexcept {
    switch (exchange) {
        case LocalExchange::SharedMemory:
            return "shared";
        case LocalExchange::WarpRegister:
            return "warp-register";
    }
    return "unknown";
}

LocalExchange parse_local_exchange(const std::string& name) {
    if (name == "shared")
        return LocalExchange::SharedMemory;
    if (name == "warp-register")
        return LocalExchange::WarpRegister;
    throw std::invalid_argument("unknown local exchange: " + name);
}

const char* shared_layout_name(SharedLayout layout) noexcept {
    switch (layout) {
        case SharedLayout::Linear:
            return "linear";
        case SharedLayout::XorSwizzle:
            return "xor-swizzle";
    }
    return "unknown";
}

SharedLayout parse_shared_layout(const std::string& name) {
    if (name == "linear")
        return SharedLayout::Linear;
    if (name == "xor-swizzle")
        return SharedLayout::XorSwizzle;
    throw std::invalid_argument("unknown shared layout: " + name);
}

const char* fft_core_name(FftCore core) noexcept {
    switch (core) {
        case FftCore::Scalar:
            return "scalar";
        case FftCore::ThreadDft8:
            return "thread-dft8";
        case FftCore::CtaDft8:
            return "cta-dft8";
        case FftCore::WmmaDft8:
            return "wmma-dft8";
        case FftCore::CufftDxBlock:
            return "cufftdx-block";
        case FftCore::CufftDxDirect:
            return "cufftdx-direct";
        case FftCore::CufftDxResident:
            return "cufftdx-resident";
        case FftCore::TurboFftGenerated:
            return "turbofft-generated";
    }
    return "unknown";
}

FftCore parse_fft_core(const std::string& name) {
    if (name == "scalar")
        return FftCore::Scalar;
    if (name == "thread-dft8")
        return FftCore::ThreadDft8;
    if (name == "cta-dft8")
        return FftCore::CtaDft8;
    if (name == "wmma-dft8")
        return FftCore::WmmaDft8;
    if (name == "cufftdx-block")
        return FftCore::CufftDxBlock;
    if (name == "cufftdx-direct")
        return FftCore::CufftDxDirect;
    if (name == "cufftdx-resident")
        return FftCore::CufftDxResident;
    if (name == "turbofft-generated")
        return FftCore::TurboFftGenerated;
    throw std::invalid_argument("unknown FFT core: " + name);
}

const char* butterfly_operator_name(ButterflyOperator op) noexcept {
    switch (op) {
        case ButterflyOperator::Fwht:
            return "fwht";
        case ButterflyOperator::Fft:
            return "fft";
        case ButterflyOperator::SubsetZeta:
            return "subset-zeta";
        case ButterflyOperator::SupersetZeta:
            return "superset-zeta";
        case ButterflyOperator::Structured2x2:
            return "structured-2x2";
        case ButterflyOperator::XorZeta:
            return "xor-zeta";
    }
    return "unknown";
}

ButterflyOperator parse_butterfly_operator(const std::string& name) {
    if (name == "fwht") {
        return ButterflyOperator::Fwht;
    }
    if (name == "fft") {
        return ButterflyOperator::Fft;
    }
    if (name == "subset-zeta" || name == "or-zeta") {
        return ButterflyOperator::SubsetZeta;
    }
    if (name == "superset-zeta" || name == "and-zeta") {
        return ButterflyOperator::SupersetZeta;
    }
    if (name == "structured-2x2" || name == "structured") {
        return ButterflyOperator::Structured2x2;
    }
    if (name == "xor-zeta") {
        return ButterflyOperator::XorZeta;
    }
    throw std::invalid_argument("unknown butterfly operator: " + name);
}

const char* butterfly_backend_name(ButterflyBackend backend) noexcept {
    switch (backend) {
        case ButterflyBackend::TemporalTile:
            return "temporal-tile";
        case ButterflyBackend::Hierarchical:
            return "hierarchical";
        case ButterflyBackend::OnlineReorder:
            return "online-reorder";
        case ButterflyBackend::WarpHybrid:
            return "warp-hybrid";
        case ButterflyBackend::StagePipeline:
            return "stage-pipeline";
        case ButterflyBackend::CuFft:
            return "cufft";
    }
    return "unknown";
}

ButterflyBackend parse_butterfly_backend(const std::string& name) {
    if (name == "temporal-tile") {
        return ButterflyBackend::TemporalTile;
    }
    if (name == "hierarchical") {
        return ButterflyBackend::Hierarchical;
    }
    if (name == "online-reorder") {
        return ButterflyBackend::OnlineReorder;
    }
    if (name == "warp-hybrid") {
        return ButterflyBackend::WarpHybrid;
    }
    if (name == "stage-pipeline") {
        return ButterflyBackend::StagePipeline;
    }
    if (name == "cufft") {
        return ButterflyBackend::CuFft;
    }
    throw std::invalid_argument("unknown butterfly backend: " + name);
}

void reference_fwht(std::vector<float>& values, bool inverse, bool normalize_inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("FWHT input size must be a nonzero power of two");
    }
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                const float left             = values[base + offset];
                const float right            = values[base + half + offset];
                values[base + offset]        = left + right;
                values[base + half + offset] = left - right;
            }
        }
    }
    if (inverse && normalize_inverse) {
        const float scale = 1.0F / static_cast<float>(values.size());
        for (float& value : values) {
            value *= scale;
        }
    }
}

void reference_fwht(std::vector<double>& values, bool inverse, bool normalize_inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("FWHT input size must be a nonzero power of two");
    }
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                const double left            = values[base + offset];
                const double right           = values[base + half + offset];
                values[base + offset]        = left + right;
                values[base + half + offset] = left - right;
            }
        }
    }
    if (inverse && normalize_inverse) {
        const double scale = 1.0 / static_cast<double>(values.size());
        for (double& value : values)
            value *= scale;
    }
}

void reference_fft(std::vector<Complex32>& values, bool inverse, bool normalize_inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("FFT input size must be a nonzero power of two");
    }
    std::size_t reversed = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        std::size_t bit = values.size() >> 1;
        while ((reversed & bit) != 0) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (index < reversed) {
            std::swap(values[index], values[reversed]);
        }
    }

    constexpr double kPi = 3.141592653589793238462643383279502884;
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                const double    direction    = inverse ? 1.0 : -1.0;
                const double    angle        = direction * kPi * static_cast<double>(offset) / static_cast<double>(half);
                const double    wr           = std::cos(angle);
                const double    wi           = std::sin(angle);
                const Complex32 a            = values[base + offset];
                const Complex32 b            = values[base + half + offset];
                const float     vr           = static_cast<float>(b.real * wr - b.imag * wi);
                const float     vi           = static_cast<float>(b.real * wi + b.imag * wr);
                values[base + offset]        = {a.real + vr, a.imag + vi};
                values[base + half + offset] = {a.real - vr, a.imag - vi};
            }
        }
    }
    if (inverse && normalize_inverse) {
        const float scale = 1.0F / static_cast<float>(values.size());
        for (Complex32& value : values) {
            value.real *= scale;
            value.imag *= scale;
        }
    }
}

void reference_fft(std::vector<Complex64>& values, bool inverse, bool normalize_inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("FFT input size must be a nonzero power of two");
    }
    std::size_t reversed = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        std::size_t bit = values.size() >> 1;
        while ((reversed & bit) != 0) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (index < reversed)
            std::swap(values[index], values[reversed]);
    }
    constexpr double kPi = 3.141592653589793238462643383279502884;
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                const double    angle        = (inverse ? 1.0 : -1.0) * kPi * static_cast<double>(offset) / static_cast<double>(half);
                const double    wr           = std::cos(angle);
                const double    wi           = std::sin(angle);
                const Complex64 a            = values[base + offset];
                const Complex64 b            = values[base + half + offset];
                const double    vr           = b.real * wr - b.imag * wi;
                const double    vi           = b.real * wi + b.imag * wr;
                values[base + offset]        = {a.real + vr, a.imag + vi};
                values[base + half + offset] = {a.real - vr, a.imag - vi};
            }
        }
    }
    if (inverse && normalize_inverse) {
        const double scale = 1.0 / static_cast<double>(values.size());
        for (Complex64& value : values) {
            value.real *= scale;
            value.imag *= scale;
        }
    }
}

void reference_subset_zeta(std::vector<std::uint32_t>& values, bool inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("subset zeta input size must be a nonzero power of two");
    }
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                if (inverse) {
                    values[base + half + offset] -= values[base + offset];
                } else {
                    values[base + half + offset] += values[base + offset];
                }
            }
        }
    }
}

void reference_superset_zeta(std::vector<std::uint32_t>& values, bool inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("superset zeta input size must be a nonzero power of two");
    }
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                if (inverse) {
                    values[base + offset] -= values[base + half + offset];
                } else {
                    values[base + offset] += values[base + half + offset];
                }
            }
        }
    }
}

void reference_xor_zeta(std::vector<std::uint32_t>& values, bool inverse) {
    reference_subset_zeta(values, inverse);
}

namespace {

template <typename Real>
void reference_structured_impl(std::vector<Real>& values, const std::vector<ButterflyMatrix2x2>& matrices, bool inverse) {
    if (values.empty() || (values.size() & (values.size() - 1)) != 0) {
        throw std::invalid_argument("structured 2x2 input size must be a nonzero power of two");
    }
    std::uint32_t log_n = 0;
    for (std::size_t n = values.size(); n > 1; n >>= 1) {
        ++log_n;
    }
    if (matrices.size() != 1 && matrices.size() != log_n) {
        throw std::invalid_argument("structured 2x2 requires one matrix or one matrix per stage");
    }
    for (std::uint32_t stage = 0; stage < log_n; ++stage) {
        const auto& source = matrices.size() == 1 ? matrices.front() : matrices[stage];
        double m00 = source.m00;
        double m01 = source.m01;
        double m10 = source.m10;
        double m11 = source.m11;
        if (!std::isfinite(m00) || !std::isfinite(m01) || !std::isfinite(m10) || !std::isfinite(m11)) {
            throw std::invalid_argument("structured 2x2 matrix entries must be finite");
        }
        if (inverse) {
            const double determinant = m00 * m11 - m01 * m10;
            if (!std::isfinite(determinant) || determinant == 0.0) {
                throw std::invalid_argument("structured 2x2 inverse requires nonsingular matrices");
            }
            m00 = source.m11 / determinant;
            m01 = -source.m01 / determinant;
            m10 = -source.m10 / determinant;
            m11 = source.m00 / determinant;
        }
        const std::size_t half = std::size_t{1} << stage;
        for (std::size_t base = 0; base < values.size(); base += 2 * half) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                const Real left  = values[base + offset];
                const Real right = values[base + half + offset];
                values[base + offset] = static_cast<Real>(m00) * left + static_cast<Real>(m01) * right;
                values[base + half + offset] = static_cast<Real>(m10) * left + static_cast<Real>(m11) * right;
            }
        }
    }
}

}  // namespace

void reference_structured_2x2(std::vector<float>& values, const std::vector<ButterflyMatrix2x2>& stage_matrices, bool inverse) {
    reference_structured_impl(values, stage_matrices, inverse);
}

void reference_structured_2x2(std::vector<double>& values, const std::vector<ButterflyMatrix2x2>& stage_matrices, bool inverse) {
    reference_structured_impl(values, stage_matrices, inverse);
}

}  // namespace cuntt
