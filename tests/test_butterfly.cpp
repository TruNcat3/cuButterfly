#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace {

constexpr std::size_t kBatch = 17;

void test_fwht(cuntt::ButterflyBackend backend, std::uint32_t stage_space, std::uint32_t warp_stages = 5, std::uint32_t pipeline_warps = 8,
               std::uint32_t tile_threads = 128, cuntt::ComputeUnit compute_unit = cuntt::ComputeUnit::Radix2, std::uint32_t log_n = 8,
               bool inverse = false, bool normalize_inverse = true, std::uint32_t local_stages = 10, std::uint32_t reorder_columns = 1,
               cuntt::LocalExchange local_exchange = cuntt::LocalExchange::SharedMemory) {
    const std::size_t                     n = std::size_t{1} << log_n;
    std::mt19937                          random(0xf017U + stage_space);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float>                    input(kBatch * n);
    std::generate(input.begin(), input.end(), [&] { return distribution(random); });

    cuntt::ButterflyConfig config;
    config.op                = cuntt::ButterflyOperator::Fwht;
    config.backend           = backend;
    config.batch             = kBatch;
    config.log_n             = log_n;
    config.inverse           = inverse;
    config.normalize_inverse = normalize_inverse;
    config.stage_space       = stage_space;
    config.warp_stages       = warp_stages;
    config.pipeline_warps    = pipeline_warps;
    config.tile_threads      = tile_threads;
    config.compute_unit      = compute_unit;
    config.local_stages      = local_stages;
    config.reorder_columns   = reorder_columns;
    config.local_exchange    = local_exchange;
    cuntt::ButterflyPlan plan(config);
    std::vector<float>   output;
    plan.execute(input, output, 0, 1);

    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        std::vector<float> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
        cuntt::reference_fwht(expected, inverse, normalize_inverse);
        for (std::size_t index = 0; index < n; ++index) {
            if (std::abs(expected[index] - output[transform * n + index]) > 1.0e-4F) {
                throw std::runtime_error("FWHT mismatch backend=" + std::string(cuntt::butterfly_backend_name(backend)) +
                                         " Us=" + std::to_string(stage_space));
            }
        }
    }
    std::cout << "PASS FWHT backend=" << cuntt::butterfly_backend_name(backend) << " Us=" << stage_space << '\n';
}

void test_fft(cuntt::ButterflyBackend backend, std::uint32_t stage_space, std::uint32_t warp_stages = 5, std::uint32_t pipeline_warps = 8,
              std::uint32_t tile_threads = 128, cuntt::ComputeUnit compute_unit = cuntt::ComputeUnit::Radix2, std::uint32_t log_n = 8,
              bool inverse = false, bool normalize_inverse = true, std::uint32_t local_stages = 10, std::uint32_t reorder_columns = 1,
              cuntt::ComplexMultiply complex_multiply = cuntt::ComplexMultiply::FourMul,
              cuntt::FftCore fft_core = cuntt::FftCore::Scalar,
              cuntt::ButterflyPrecision precision = cuntt::ButterflyPrecision::Fp32) {
    const std::size_t                     n = std::size_t{1} << log_n;
    std::mt19937                          random(0xff70U + stage_space);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<cuntt::Complex32>         input(kBatch * n);
    std::generate(input.begin(), input.end(), [&] { return cuntt::Complex32{distribution(random), distribution(random)}; });

    cuntt::ButterflyConfig config;
    config.op                = cuntt::ButterflyOperator::Fft;
    config.backend           = backend;
    config.batch             = kBatch;
    config.log_n             = log_n;
    config.inverse           = inverse;
    config.normalize_inverse = normalize_inverse;
    config.stage_space       = stage_space;
    config.warp_stages       = warp_stages;
    config.pipeline_warps    = pipeline_warps;
    config.tile_threads      = tile_threads;
    config.compute_unit      = backend == cuntt::ButterflyBackend::CuFft ? cuntt::ComputeUnit::Auto : compute_unit;
    config.complex_multiply  = complex_multiply;
    config.local_stages      = local_stages;
    config.reorder_columns   = reorder_columns;
    config.fft_core          = fft_core;
    config.precision         = precision;
    cuntt::ButterflyPlan          plan(config);
    std::vector<cuntt::Complex32> output;
    plan.execute(input, output, 0, 1);

    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        std::vector<cuntt::Complex32> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
        cuntt::reference_fft(expected, inverse, normalize_inverse);
        for (std::size_t index = 0; index < n; ++index) {
            const auto& actual = output[transform * n + index];
            const float error  = std::hypot(expected[index].real - actual.real, expected[index].imag - actual.imag);
            const float tolerance = precision == cuntt::ButterflyPrecision::Fp16Fp32 ? 1.0e-2F : 2.0e-4F;
            if (error > tolerance) {
                throw std::runtime_error("FFT mismatch backend=" + std::string(cuntt::butterfly_backend_name(backend)) +
                                         " Us=" + std::to_string(stage_space) + " error=" + std::to_string(error));
            }
        }
    }
    std::cout << "PASS FFT backend=" << cuntt::butterfly_backend_name(backend) << " Us=" << stage_space << '\n';
}

void test_xor_zeta(cuntt::ButterflyBackend backend, std::uint32_t stage_space, std::uint32_t warp_stages = 5, std::uint32_t pipeline_warps = 8,
                   std::uint32_t tile_threads = 128, cuntt::ComputeUnit compute_unit = cuntt::ComputeUnit::Radix2, std::uint32_t log_n = 8,
                   bool inverse = false, std::uint32_t local_stages = 10, std::uint32_t reorder_columns = 1) {
    const std::size_t                            n = std::size_t{1} << log_n;
    std::mt19937                                 random(0x207aU + stage_space);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1023);
    std::vector<std::uint32_t>                   input(kBatch * n);
    std::generate(input.begin(), input.end(), [&] { return distribution(random); });

    cuntt::ButterflyConfig config;
    config.op              = cuntt::ButterflyOperator::XorZeta;
    config.backend         = backend;
    config.batch           = kBatch;
    config.log_n           = log_n;
    config.inverse         = inverse;
    config.stage_space     = stage_space;
    config.warp_stages     = warp_stages;
    config.pipeline_warps  = pipeline_warps;
    config.tile_threads    = tile_threads;
    config.compute_unit    = compute_unit;
    config.local_stages    = local_stages;
    config.reorder_columns = reorder_columns;
    cuntt::ButterflyPlan       plan(config);
    std::vector<std::uint32_t> output;
    plan.execute(input, output, 0, 1);

    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        std::vector<std::uint32_t> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
        cuntt::reference_xor_zeta(expected, inverse);
        if (!std::equal(expected.begin(), expected.end(), output.begin() + transform * n)) {
            throw std::runtime_error("XOR zeta mismatch backend=" + std::string(cuntt::butterfly_backend_name(backend)) +
                                     " Us=" + std::to_string(stage_space));
        }
    }
    std::cout << "PASS XOR-zeta backend=" << cuntt::butterfly_backend_name(backend) << " Us=" << stage_space << '\n';
}

void test_fp64(std::uint32_t log_n, bool inverse) {
    const std::size_t                      n = std::size_t{1} << log_n;
    std::mt19937                           random(0x6400U + log_n + static_cast<unsigned int>(inverse));
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    std::vector<double> fwht_input(kBatch * n);
    std::generate(fwht_input.begin(), fwht_input.end(), [&] { return distribution(random); });
    cuntt::ButterflyConfig fwht_config;
    fwht_config.op           = cuntt::ButterflyOperator::Fwht;
    fwht_config.precision    = cuntt::ButterflyPrecision::Fp64;
    fwht_config.backend      = log_n <= 10 ? cuntt::ButterflyBackend::TemporalTile : cuntt::ButterflyBackend::Hierarchical;
    fwht_config.compute_unit = cuntt::ComputeUnit::Radix4;
    fwht_config.log_n        = log_n;
    fwht_config.batch        = kBatch;
    fwht_config.inverse      = inverse;
    cuntt::ButterflyPlan fwht_plan(fwht_config);
    std::vector<double>  fwht_output;
    fwht_plan.execute(fwht_input, fwht_output, 0, 1);
    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        std::vector<double> expected(fwht_input.begin() + transform * n, fwht_input.begin() + (transform + 1) * n);
        cuntt::reference_fwht(expected, inverse);
        for (std::size_t index = 0; index < n; ++index) {
            if (std::abs(expected[index] - fwht_output[transform * n + index]) > 1.0e-10)
                throw std::runtime_error("FP64 FWHT mismatch");
        }
    }

    std::vector<cuntt::Complex64> fft_input(kBatch * n);
    std::generate(fft_input.begin(), fft_input.end(), [&] { return cuntt::Complex64{distribution(random), distribution(random)}; });
    const std::vector<cuntt::ButterflyBackend> fft_backends =
        log_n <= 10 ? std::vector<cuntt::ButterflyBackend>{cuntt::ButterflyBackend::TemporalTile, cuntt::ButterflyBackend::CuFft}
                    : std::vector<cuntt::ButterflyBackend>{cuntt::ButterflyBackend::Hierarchical, cuntt::ButterflyBackend::OnlineReorder,
                                                           cuntt::ButterflyBackend::CuFft};
    for (const auto backend : fft_backends) {
        cuntt::ButterflyConfig fft_config;
        fft_config.op           = cuntt::ButterflyOperator::Fft;
        fft_config.precision    = cuntt::ButterflyPrecision::Fp64;
        fft_config.backend      = backend;
        fft_config.compute_unit = backend == cuntt::ButterflyBackend::CuFft ? cuntt::ComputeUnit::Auto : cuntt::ComputeUnit::Radix4;
        fft_config.log_n        = log_n;
        fft_config.batch        = kBatch;
        fft_config.inverse      = inverse;
        cuntt::ButterflyPlan          fft_plan(fft_config);
        std::vector<cuntt::Complex64> fft_output;
        fft_plan.execute(fft_input, fft_output, 0, 1);
        for (std::size_t transform = 0; transform < kBatch; ++transform) {
            std::vector<cuntt::Complex64> expected(fft_input.begin() + transform * n, fft_input.begin() + (transform + 1) * n);
            cuntt::reference_fft(expected, inverse);
            for (std::size_t index = 0; index < n; ++index) {
                const auto actual = fft_output[transform * n + index];
                if (std::hypot(expected[index].real - actual.real, expected[index].imag - actual.imag) > 1.0e-10) {
                    throw std::runtime_error("FP64 FFT mismatch");
                }
            }
        }
    }
}

void test_layout(cuntt::ButterflyBackend backend, std::uint32_t log_n, cuntt::ButterflyPlacement placement) {
    const std::size_t                     n              = std::size_t{1} << log_n;
    const std::size_t                     element_stride = 2;
    const std::size_t                     stride         = (n - 1) * element_stride + 1 + 13;
    const std::size_t                     extent         = (kBatch - 1) * stride + (n - 1) * element_stride + 1;
    std::mt19937                          random(0x1a700U + log_n + static_cast<unsigned int>(backend));
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<cuntt::Complex32>         input(extent);
    std::generate(input.begin(), input.end(), [&] { return cuntt::Complex32{distribution(random), distribution(random)}; });

    cuntt::ButterflyConfig config;
    config.op           = cuntt::ButterflyOperator::Fft;
    config.backend      = backend;
    config.compute_unit = backend == cuntt::ButterflyBackend::TemporalTile ? cuntt::ComputeUnit::Radix4 : cuntt::ComputeUnit::Radix2;
    if (backend == cuntt::ButterflyBackend::CuFft)
        config.compute_unit = cuntt::ComputeUnit::Auto;
    config.log_n          = log_n;
    config.batch          = kBatch;
    config.batch_stride   = stride;
    config.element_stride = element_stride;
    config.placement      = placement;
    config.inverse        = placement == cuntt::ButterflyPlacement::InPlace;
    cuntt::ButterflyPlan          plan(config);
    std::vector<cuntt::Complex32> output;
    const bool                    repeated_in_place = placement == cuntt::ButterflyPlacement::InPlace;
    plan.execute(input, output, repeated_in_place ? 2 : 0, repeated_in_place ? 3 : 1);
    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        const std::size_t             base = transform * stride;
        std::vector<cuntt::Complex32> expected(n);
        for (std::size_t index = 0; index < n; ++index)
            expected[index] = input[base + index * element_stride];
        cuntt::reference_fft(expected, config.inverse);
        for (std::size_t index = 0; index < n; ++index) {
            const auto actual = output[base + index * element_stride];
            if (std::hypot(expected[index].real - actual.real, expected[index].imag - actual.imag) > 2.0e-4F) {
                throw std::runtime_error("strided/in-place FFT mismatch");
            }
        }
    }
}

}  // namespace

int main() {
    try {
        for (const std::uint32_t tile_threads : {32U, 64U, 128U, 256U}) {
            test_fwht(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads);
            test_fft(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads);
            test_xor_zeta(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads);
            test_fwht(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads, cuntt::ComputeUnit::Radix4);
            test_fft(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads, cuntt::ComputeUnit::Radix4);
            test_xor_zeta(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, tile_threads, cuntt::ComputeUnit::Radix4);
        }
        for (const std::uint32_t stage_space : {1U, 2U, 4U, 8U}) {
            test_fwht(cuntt::ButterflyBackend::StagePipeline, stage_space);
            test_fft(cuntt::ButterflyBackend::StagePipeline, stage_space);
            test_xor_zeta(cuntt::ButterflyBackend::StagePipeline, stage_space);
        }
        for (const std::uint32_t stage_space : {1U, 2U, 4U}) {
            test_fwht(cuntt::ButterflyBackend::StagePipeline, stage_space, 5, 4);
            test_fft(cuntt::ButterflyBackend::StagePipeline, stage_space, 5, 4);
            test_xor_zeta(cuntt::ButterflyBackend::StagePipeline, stage_space, 5, 4);
        }
        for (std::uint32_t warp_stages = 0; warp_stages <= 5; ++warp_stages) {
            test_fwht(cuntt::ButterflyBackend::WarpHybrid, 1, warp_stages);
            test_fft(cuntt::ButterflyBackend::WarpHybrid, 1, warp_stages);
            test_xor_zeta(cuntt::ButterflyBackend::WarpHybrid, 1, warp_stages);
        }
        test_fft(cuntt::ButterflyBackend::CuFft, 1);
        for (const bool inverse : {false, true}) {
            test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 128, cuntt::ComputeUnit::Radix8, 3, inverse, true, 0, 0,
                     cuntt::ComplexMultiply::FourMul, cuntt::FftCore::ThreadDft8, cuntt::ButterflyPrecision::Fp32);
            test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 256, cuntt::ComputeUnit::Radix8, 3, inverse, true, 0, 0,
                     cuntt::ComplexMultiply::FourMul, cuntt::FftCore::WmmaDft8, cuntt::ButterflyPrecision::Fp16Fp32);
        }
        for (std::uint32_t log_n = 3; log_n <= 10; ++log_n) {
            for (const std::uint32_t tile_threads : {32U, 64U, 128U, 256U}) {
                if (tile_threads * 8 < (1U << log_n))
                    continue;
                for (const bool inverse : {false, true}) {
                    test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, tile_threads, cuntt::ComputeUnit::Radix8, log_n, inverse, true, 0, 0,
                             cuntt::ComplexMultiply::FourMul, cuntt::FftCore::CtaDft8, cuntt::ButterflyPrecision::Fp32);
                }
            }
        }
        for (std::uint32_t log_n = 1; log_n <= 10; ++log_n) {
            for (const auto unit : {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8}) {
                for (const bool inverse : {false, true}) {
                    test_fwht(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, unit, log_n, inverse);
                    test_fft(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, unit, log_n, inverse);
                    test_xor_zeta(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, unit, log_n, inverse);
                }
            }
            test_fft(cuntt::ButterflyBackend::CuFft, 1, 5, 8, 128, cuntt::ComputeUnit::Auto, log_n, true);
        }
        for (std::uint32_t log_n = 3; log_n <= 15; ++log_n) {
            for (const bool inverse : {false, true}) {
                test_fwht(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 0, cuntt::ComputeUnit::Radix2, log_n, inverse, true, 0, 0,
                          cuntt::LocalExchange::WarpRegister);
            }
        }
        for (const std::uint32_t log_n : {1U, 5U, 8U, 10U}) {
            test_fp64(log_n, false);
            test_fp64(log_n, true);
        }
        test_fp64(11, false);
        test_fp64(11, true);
        test_fwht(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, cuntt::ComputeUnit::Radix4, 5, true, false);
        test_fft(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, cuntt::ComputeUnit::Radix4, 5, true, false);
        test_fft(cuntt::ButterflyBackend::CuFft, 1, 5, 8, 128, cuntt::ComputeUnit::Auto, 5, true, false);
        for (const auto unit : {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8}) {
            for (const bool inverse : {false, true}) {
                test_fwht(cuntt::ButterflyBackend::Hierarchical, 0, 0, 0, 128, unit, 11, inverse);
                test_fft(cuntt::ButterflyBackend::Hierarchical, 0, 0, 0, 128, unit, 11, inverse);
                test_xor_zeta(cuntt::ButterflyBackend::Hierarchical, 0, 0, 0, 128, unit, 11, inverse);
                test_fwht(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 128, unit, 11, inverse, true, 5);
                test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 128, unit, 11, inverse, true, 5);
                test_xor_zeta(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 128, unit, 11, inverse, 5);
            }
        }
        test_fwht(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 256, cuntt::ComputeUnit::Radix4, 16, false, true, 8, 4);
        test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 256, cuntt::ComputeUnit::Radix4, 16, false, true, 8, 4);
        test_xor_zeta(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 256, cuntt::ComputeUnit::Radix4, 16, false, 8, 4);
        test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 128, cuntt::ComputeUnit::Radix8, 10, false, true, 10, 1,
                 cuntt::ComplexMultiply::Gauss3);
        test_fft(cuntt::ButterflyBackend::Hierarchical, 0, 0, 0, 128, cuntt::ComputeUnit::Radix8, 11, false, true, 8, 1,
                 cuntt::ComplexMultiply::Gauss3);
        test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 128, cuntt::ComputeUnit::Radix8, 11, false, true, 5, 1,
                 cuntt::ComplexMultiply::Gauss3);
        test_fft(cuntt::ButterflyBackend::WarpHybrid, 0, 5, 0, 128, cuntt::ComputeUnit::Radix2, 8, false, true, 10, 1,
                 cuntt::ComplexMultiply::Gauss3);
        test_fft(cuntt::ButterflyBackend::StagePipeline, 2, 0, 8, 128, cuntt::ComputeUnit::Radix2, 8, false, true, 10, 1,
                 cuntt::ComplexMultiply::Gauss3);
        test_fft(cuntt::ButterflyBackend::CuFft, 0, 0, 0, 128, cuntt::ComputeUnit::Auto, 12, true);
        for (const auto placement : {cuntt::ButterflyPlacement::OutOfPlace, cuntt::ButterflyPlacement::InPlace}) {
            test_layout(cuntt::ButterflyBackend::TemporalTile, 10, placement);
            test_layout(cuntt::ButterflyBackend::WarpHybrid, 8, placement);
            test_layout(cuntt::ButterflyBackend::StagePipeline, 8, placement);
            test_layout(cuntt::ButterflyBackend::CuFft, 10, placement);
            test_layout(cuntt::ButterflyBackend::Hierarchical, 11, placement);
            test_layout(cuntt::ButterflyBackend::OnlineReorder, 11, placement);
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All cuButterfly tests passed\n";
    return 0;
}
