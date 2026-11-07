// Finite public-API regression for scalar OnlineReorder boundary ownership.
// Arithmetic controls (unit, precision, normalization, layout, strides) are
// identical within each pair; only independent prefix/suffix grouping differs.
#include "cuntt/butterfly.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace cuntt;
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
    void* p = nullptr;
    explicit Buffer(std::size_t bytes) { cuda_check(cudaMalloc(&p, bytes)); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
};
template<class T> T value(int i) { return static_cast<T>((i % 17 - 8) * 0.03125); }
template<> std::uint32_t value<std::uint32_t>(int i) { return static_cast<std::uint32_t>(i % 17 + 1); }
template<> std::uint64_t value<std::uint64_t>(int i) {
    return (std::uint64_t{1} << (32 + i % 9)) + static_cast<std::uint64_t>(i % 17 + 1);
}
template<> Fp16 value<Fp16>(int i) { return __float2half(value<float>(i)); }
template<> Bf16 value<Bf16>(int i) { return __float2bfloat16(value<float>(i)); }
template<> Complex32 value<Complex32>(int i) { return {value<float>(i), value<float>(i + 5)}; }
template<> Complex64 value<Complex64>(int i) { return {value<double>(i), value<double>(i + 5)}; }
template<> Complex16 value<Complex16>(int i) {
    return {__float2half(value<float>(i)), __float2half(value<float>(i + 5))};
}
template<> ComplexBf16 value<ComplexBf16>(int i) {
    return {__float2bfloat16(value<float>(i)), __float2bfloat16(value<float>(i + 5))};
}
std::size_t passed = 0;
template<class T> void compare(ButterflyConfig c, unsigned groups, unsigned columns) {
    const std::size_t n = std::size_t{1} << c.log_n;
    c.batch = 3;  // tail CTA groups, not a multiple of grouping factors
    if (!c.batch_stride) c.batch_stride = (n - 1) * c.element_stride + 1 + 13;
    const std::size_t extent = (c.batch - 1) * c.batch_stride + (n - 1) * c.element_stride + 1;
    const std::size_t bytes = extent * sizeof(T), guard = 256;
    std::vector<unsigned char> input(bytes + 2 * guard, 0xA5);
    for (std::size_t b = 0; b < c.batch; ++b)
        for (std::size_t j = 0; j < n; ++j) {
            T v = value<T>(static_cast<int>(j * 7 + b * 11));
            std::memcpy(input.data() + guard + (b * c.batch_stride + j * c.element_stride) * sizeof(T), &v, sizeof(T));
        }
    Buffer x(input.size()), y(input.size());
    cuda_check(cudaMemcpy(x.p, input.data(), input.size(), cudaMemcpyHostToDevice));
    auto run = [&](unsigned g, unsigned col) {
        auto requested = c;
        requested.data_tiles_per_cta = g;
        requested.reorder_columns = col;
        ButterflyPlan plan(requested);
        const auto& resolved = plan.config();
        if (resolved.data_tiles_per_cta != g || resolved.reorder_columns != col ||
            resolved.compute_unit != c.compute_unit || resolved.precision != c.precision ||
            resolved.fft_core != FftCore::Scalar || resolved.backend != ButterflyBackend::OnlineReorder)
            throw std::runtime_error("public plan silently changed requested organization/arithmetic");
        cuda_check(cudaMemset(y.p, 0xA5, input.size()));
        plan.execute_async(reinterpret_cast<const T*>(static_cast<unsigned char*>(x.p) + guard),
                           reinterpret_cast<T*>(static_cast<unsigned char*>(y.p) + guard));
        cuda_check(cudaDeviceSynchronize());
        std::vector<unsigned char> output(input.size()), observed(input.size());
        cuda_check(cudaMemcpy(output.data(), y.p, output.size(), cudaMemcpyDeviceToHost));
        cuda_check(cudaMemcpy(observed.data(), x.p, observed.size(), cudaMemcpyDeviceToHost));
        if (observed != input) throw std::runtime_error("input or input guards modified");
        std::vector<bool> written(output.size(), false);
        for (std::size_t b = 0; b < c.batch; ++b)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < sizeof(T); ++k)
                    written[guard + (b * c.batch_stride + j * c.element_stride) * sizeof(T) + k] = true;
        for (std::size_t j = 0; j < output.size(); ++j)
            if (!written[j] && output[j] != 0xA5) throw std::runtime_error("output guard/stride hole modified");
        return output;
    };
    const auto baseline = run(1, 1);
    if (baseline != run(groups, columns))
        throw std::runtime_error("bitwise output mismatch: " + std::string(butterfly_operator_name(c.op)) +
                                 "/" + butterfly_precision_name(c.precision) + "/" + compute_unit_name(c.compute_unit));
    ++passed;
    std::cout << "PASS " << passed << " " << butterfly_operator_name(c.op) << " "
              << butterfly_precision_name(c.precision) << " " << compute_unit_name(c.compute_unit)
              << " N=" << n << " inverse=" << c.inverse << " norm=" << c.normalize_inverse
              << " G=" << groups << " C=" << columns << '\n';
}
ButterflyConfig config() {
    ButterflyConfig c;
    c.backend = ButterflyBackend::OnlineReorder;
    c.fft_core = FftCore::Scalar;
    c.local_stages = 5;
    c.log_n = 8;
    c.tile_threads = 64;
    return c;
}
void dispatch_fft(ButterflyConfig c, unsigned groups, unsigned columns) {
    switch (c.precision) {
        case ButterflyPrecision::Fp16: compare<Complex16>(c, groups, columns); break;
        case ButterflyPrecision::Bf16: compare<ComplexBf16>(c, groups, columns); break;
        case ButterflyPrecision::Fp32: compare<Complex32>(c, groups, columns); break;
        case ButterflyPrecision::Fp64: compare<Complex64>(c, groups, columns); break;
        default: throw std::runtime_error("unexpected FFT precision");
    }
}
void invalid_groups(unsigned groups) {
    auto c = config();
    c.data_tiles_per_cta = groups;
    try { ButterflyPlan plan(c); }
    catch (const std::invalid_argument&) { ++passed; return; }
    throw std::runtime_error("invalid grouping accepted: " + std::to_string(groups));
}
} // namespace

int main() {
    try {
        const ComputeUnit units[] = {ComputeUnit::Radix2, ComputeUnit::Radix4, ComputeUnit::Radix8};
        const ButterflyPrecision precisions[] = {ButterflyPrecision::Fp16, ButterflyPrecision::Bf16,
                                                 ButterflyPrecision::Fp32, ButterflyPrecision::Fp64};
        for (auto precision : precisions)
            for (auto unit : units)
                for (bool inverse : {false, true}) {
                    auto c = config();
                    c.op = ButterflyOperator::Fft;
                    c.precision = precision;
                    c.compute_unit = unit;
                    c.inverse = inverse;
                    c.element_stride = inverse ? 2 : 1;
                    c.shared_layout = inverse ? SharedLayout::WriterAligned : SharedLayout::Linear;
                    dispatch_fft(c, 4, 4);
                }
        for (auto op : {ButterflyOperator::Fwht, ButterflyOperator::SubsetZeta,
                        ButterflyOperator::SupersetZeta, ButterflyOperator::XorZeta, ButterflyOperator::Structured2x2})
            for (auto unit : units)
                for (bool inverse : {false, true}) {
                    auto c = config();
                    c.op = op;
                    c.compute_unit = unit;
                    c.inverse = inverse;
                    c.element_stride = inverse ? 2 : 1;
                    c.shared_layout = inverse ? SharedLayout::WriterAligned : SharedLayout::Linear;
                    if (op == ButterflyOperator::Structured2x2)
                        c.stage_matrices = {{1.0, 0.25, -0.5, 1.0}};
                    if (op == ButterflyOperator::SubsetZeta || op == ButterflyOperator::SupersetZeta ||
                        op == ButterflyOperator::XorZeta) {
                        c.precision = ButterflyPrecision::Uint32;
                        c.normalize_inverse = false;
                        compare<std::uint32_t>(c, 4, 4);
                    } else {
                        compare<float>(c, 4, 4);
                    }
                }
        // Explicit 2-byte scalar storage/native arithmetic coverage.
        for (auto op : {ButterflyOperator::Fwht, ButterflyOperator::Structured2x2})
            for (auto precision : {ButterflyPrecision::Fp16, ButterflyPrecision::Bf16})
                for (bool inverse : {false, true}) {
                    auto low = config();
                    low.op = op;
                    low.precision = precision;
                    low.accumulation = ButterflyAccumulation::Native;
                    low.compute_unit = ComputeUnit::Radix2;
                    low.inverse = inverse;
                    low.element_stride = inverse ? 2 : 1;
                    low.shared_layout = inverse ? SharedLayout::WriterAligned : SharedLayout::Linear;
                    if (op == ButterflyOperator::Structured2x2)
                        low.stage_matrices = {{1.0, 0.25, -0.5, 1.0}};
                    if (precision == ButterflyPrecision::Fp16) compare<Fp16>(low, 4, 4);
                    else compare<Bf16>(low, 4, 4);
                }
        // UINT64 inputs carry nonzero high words; inverse arithmetic wraps
        // modulo 2^64, as required by public zeta/Mobius semantics.
        for (auto op : {ButterflyOperator::SubsetZeta, ButterflyOperator::SupersetZeta,
                        ButterflyOperator::XorZeta})
            for (bool inverse : {false, true}) {
                auto integer = config();
                integer.op = op;
                integer.precision = ButterflyPrecision::Uint64;
                integer.compute_unit = ComputeUnit::Radix2;
                integer.inverse = inverse;
                integer.normalize_inverse = false;
                integer.element_stride = inverse ? 2 : 1;
                integer.shared_layout = inverse ? SharedLayout::WriterAligned : SharedLayout::Linear;
                compare<std::uint64_t>(integer, 4, 4);
            }
        // Boundary capacities G*L=1024, C*R=1024, plus G1/C1 compatibility
        // and unnormalized inverse on the same finite arithmetic controls.
        for (auto precision : precisions) {
            auto c = config();
            c.op = ButterflyOperator::Fft;
            c.precision = precision;
            c.log_n = 12;
            c.inverse = true;
            c.normalize_inverse = false;
            c.tile_threads = 256;
            dispatch_fft(c, 32, 8);
        }
        // Larger dynamic-shared groups: true logical extent and byte limits.
        for (auto precision : {ButterflyPrecision::Fp16, ButterflyPrecision::Bf16, ButterflyPrecision::Fp32}) {
            auto extended = config();
            extended.op = ButterflyOperator::Fft;
            extended.precision = precision;
            extended.log_n = 12;
            dispatch_fft(extended, 32, 32);
            extended.shared_layout = SharedLayout::WriterAligned;
            dispatch_fft(extended, 128, 32);
        }
        {
            auto invalid = config();
            invalid.op = ButterflyOperator::Fft;
            invalid.precision = ButterflyPrecision::Fp64;
            invalid.log_n = 12;
            invalid.reorder_columns = 32; // 32*128*sizeof(Complex64)=64KiB >48KiB
            bool rejected = false;
            try { ButterflyPlan plan(invalid); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("64KiB FP64 suffix grouping accepted");
            ++passed;
        }
        auto c = config();
        compare<float>(c, 1, 1);
        c.inverse = true;
        c.normalize_inverse = false;
        compare<float>(c, 8, 8);
        invalid_groups(0);
        invalid_groups(3);
        invalid_groups(64); // G exceeds N/L=8 for this N256 configuration
        std::cout << "PASS total=" << passed << " (80 bitwise pairs + 4 rejected configurations)\n";
        std::cout << "NTT not covered: ButterflyOperator has no NTT variant; NTT has a separate public Plan and no OnlineReorder backend/grouping controls.\n";
        std::cout << "FFT natural-input bit reversal is exercised through public operator traits, not manually pre-permuted input.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
