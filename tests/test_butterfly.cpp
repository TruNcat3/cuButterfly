#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace {

constexpr std::size_t kBatch = 17;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

class TestStream {
  public:
    TestStream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~TestStream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class TestDeviceBuffer {
  public:
    explicit TestDeviceBuffer(std::size_t bytes) {
        if (bytes != 0)
            check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
    }
    ~TestDeviceBuffer() { cudaFree(pointer_); }
    void* get() const noexcept { return pointer_; }

  private:
    void* pointer_ = nullptr;
};

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
              cuntt::ButterflyPrecision precision = cuntt::ButterflyPrecision::Fp32,
              cuntt::CrossTwiddleMode cross_twiddle = cuntt::CrossTwiddleMode::Table,
              std::uint32_t prefix_threads = 0, std::uint32_t suffix_threads = 0,
              std::uint32_t prefix_ept = 8, std::uint32_t suffix_ept = 8) {
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
    config.cross_twiddle     = cross_twiddle;
    config.prefix_threads    = prefix_threads;
    config.suffix_threads    = suffix_threads;
    config.prefix_ept        = prefix_ept;
    config.suffix_ept        = suffix_ept;
    cuntt::ButterflyPlan          plan(config);
    std::vector<cuntt::Complex32> output;
    plan.execute(input, output, 0, 1);

    for (std::size_t transform = 0; transform < kBatch; ++transform) {
        std::vector<cuntt::Complex32> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
        cuntt::reference_fft(expected, inverse, normalize_inverse);
        for (std::size_t index = 0; index < n; ++index) {
            const auto& actual = output[transform * n + index];
            const float error  = std::hypot(expected[index].real - actual.real, expected[index].imag - actual.imag);
            const float tolerance = precision == cuntt::ButterflyPrecision::Fp16Fp32
                                        ? 1.0e-2F
                                        : (fft_core == cuntt::FftCore::TurboFftGenerated && log_n == 10
                                               ? 3.0e-4F
                                               : 2.0e-4F * std::sqrt(std::max(1.0F, static_cast<float>(n) / 1024.0F)));
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
                   bool inverse = false, std::uint32_t local_stages = 10, std::uint32_t reorder_columns = 1,
                   cuntt::ButterflyOperator op = cuntt::ButterflyOperator::XorZeta) {
    const std::size_t                            n = std::size_t{1} << log_n;
    std::mt19937                                 random(0x207aU + stage_space);
    std::uniform_int_distribution<std::uint32_t> distribution(0, 1023);
    std::vector<std::uint32_t>                   input(kBatch * n);
    std::generate(input.begin(), input.end(), [&] { return distribution(random); });

    cuntt::ButterflyConfig config;
    config.op              = op;
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
        if (op == cuntt::ButterflyOperator::SupersetZeta) {
            cuntt::reference_superset_zeta(expected, inverse);
        } else if (op == cuntt::ButterflyOperator::SubsetZeta) {
            cuntt::reference_subset_zeta(expected, inverse);
        } else {
            cuntt::reference_xor_zeta(expected, inverse);
        }
        if (!std::equal(expected.begin(), expected.end(), output.begin() + transform * n)) {
            throw std::runtime_error(std::string(cuntt::butterfly_operator_name(op)) + " mismatch backend=" +
                                     std::string(cuntt::butterfly_backend_name(backend)) +
                                     " Us=" + std::to_string(stage_space));
        }
    }
    std::cout << "PASS " << cuntt::butterfly_operator_name(op) << " backend=" << cuntt::butterfly_backend_name(backend)
              << " Us=" << stage_space << '\n';
}

void test_extended_zeta_operators() {
    for (const auto op : {cuntt::ButterflyOperator::SubsetZeta, cuntt::ButterflyOperator::SupersetZeta}) {
        for (const bool inverse : {false, true}) {
            for (const auto unit : {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8}) {
                test_xor_zeta(cuntt::ButterflyBackend::TemporalTile, 1, 5, 8, 128, unit, 8, inverse, 0, 1, op);
            }
            test_xor_zeta(cuntt::ButterflyBackend::StagePipeline, 2, 5, 8, 128,
                          cuntt::ComputeUnit::Radix2, 8, inverse, 0, 1, op);
            test_xor_zeta(cuntt::ButterflyBackend::WarpHybrid, 1, 4, 8, 128,
                          cuntt::ComputeUnit::Radix2, 8, inverse, 0, 1, op);
            test_xor_zeta(cuntt::ButterflyBackend::Hierarchical, 0, 0, 0, 128,
                          cuntt::ComputeUnit::Radix4, 11, inverse, 5, 1, op);
            test_xor_zeta(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 128,
                          cuntt::ComputeUnit::Radix4, 11, inverse, 5, 2, op);
        }
    }
}

std::vector<cuntt::ButterflyMatrix2x2> structured_matrices(std::uint32_t log_n) {
    std::vector<cuntt::ButterflyMatrix2x2> matrices;
    matrices.reserve(log_n);
    for (std::uint32_t stage = 0; stage < log_n; ++stage) {
        const double delta = 0.01 * static_cast<double>(stage);
        matrices.push_back({1.0 + delta, 0.125, -0.0625, 0.875 - 0.5 * delta});
    }
    return matrices;
}

void test_structured_reference_roundtrip() {
    constexpr std::uint32_t log_n = 6;
    const auto matrices = structured_matrices(log_n);
    std::vector<double> values(std::size_t{1} << log_n);
    std::iota(values.begin(), values.end(), -17.0);
    const auto original = values;
    cuntt::reference_structured_2x2(values, matrices, false);
    cuntt::reference_structured_2x2(values, matrices, true);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (std::abs(values[index] - original[index]) > 1.0e-10)
            throw std::runtime_error("structured-2x2 CPU reference roundtrip mismatch");
    }
    std::cout << "PASS structured-2x2 CPU reference roundtrip\n";
}

void test_structured_2x2(cuntt::ButterflyBackend backend, cuntt::ButterflyPrecision precision,
                         cuntt::ComputeUnit unit, std::uint32_t log_n, bool inverse,
                         std::uint32_t local_stages = 0, std::uint32_t stage_space = 0,
                         std::uint32_t warp_stages = 0,
                         cuntt::LocalExchange exchange = cuntt::LocalExchange::SharedMemory,
                         bool broadcast = false) {
    const std::size_t n = std::size_t{1} << log_n;
    const auto matrices = broadcast ? std::vector<cuntt::ButterflyMatrix2x2>{{1.0, 0.25, -0.5, 1.0}}
                                    : structured_matrices(log_n);
    cuntt::ButterflyConfig config;
    config.op              = cuntt::ButterflyOperator::Structured2x2;
    config.backend         = backend;
    config.precision       = precision;
    config.compute_unit    = unit;
    config.log_n           = log_n;
    config.batch           = kBatch;
    config.inverse         = inverse;
    config.stage_matrices  = matrices;
    config.local_stages    = local_stages;
    config.stage_space     = stage_space;
    config.warp_stages     = warp_stages;
    config.local_exchange  = exchange;
    config.reorder_columns = 1;

    cuntt::ButterflyPlan plan(config);
    std::mt19937 random(0x2b2bU + log_n + static_cast<unsigned int>(inverse));
    if (precision == cuntt::ButterflyPrecision::Fp64) {
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        std::vector<double> input(kBatch * n);
        std::generate(input.begin(), input.end(), [&] { return distribution(random); });
        std::vector<double> output;
        plan.execute(input, output, 0, 1);
        for (std::size_t transform = 0; transform < kBatch; ++transform) {
            std::vector<double> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
            cuntt::reference_structured_2x2(expected, matrices, inverse);
            for (std::size_t index = 0; index < n; ++index) {
                const double error = std::abs(expected[index] - output[transform * n + index]);
                if (error > 1.0e-10 * std::max(1.0, std::abs(expected[index])))
                    throw std::runtime_error("FP64 structured-2x2 mismatch");
            }
        }
    } else {
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        std::vector<float> input(kBatch * n);
        std::generate(input.begin(), input.end(), [&] { return distribution(random); });
        std::vector<float> output;
        plan.execute(input, output, 0, 1);
        for (std::size_t transform = 0; transform < kBatch; ++transform) {
            std::vector<float> expected(input.begin() + transform * n, input.begin() + (transform + 1) * n);
            cuntt::reference_structured_2x2(expected, matrices, inverse);
            for (std::size_t index = 0; index < n; ++index) {
                const float error = std::abs(expected[index] - output[transform * n + index]);
                if (error > 2.0e-4F * std::max(1.0F, std::abs(expected[index])))
                    throw std::runtime_error("FP32 structured-2x2 mismatch");
            }
        }
    }
    std::cout << "PASS structured-2x2 backend=" << cuntt::butterfly_backend_name(backend)
              << " precision=" << cuntt::butterfly_precision_name(precision) << '\n';
}

void test_structured_operators() {
    test_structured_reference_roundtrip();
    for (const bool inverse : {false, true}) {
        for (std::uint32_t log_n = 3; log_n <= 15; ++log_n) {
            test_structured_2x2(cuntt::ButterflyBackend::TemporalTile, cuntt::ButterflyPrecision::Fp32,
                                cuntt::ComputeUnit::Radix2, log_n, inverse, 0, 0, 0,
                                cuntt::LocalExchange::WarpRegister);
        }
        for (const std::uint32_t log_n : {8U, 12U, 15U}) {
            test_structured_2x2(cuntt::ButterflyBackend::TemporalTile, cuntt::ButterflyPrecision::Fp32,
                                cuntt::ComputeUnit::Radix2, log_n, inverse, 0, 0, 0,
                                cuntt::LocalExchange::WarpRegister, true);
        }
        for (const auto unit : {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8}) {
            test_structured_2x2(cuntt::ButterflyBackend::TemporalTile, cuntt::ButterflyPrecision::Fp32,
                                unit, 8, inverse);
        }
        test_structured_2x2(cuntt::ButterflyBackend::StagePipeline, cuntt::ButterflyPrecision::Fp32,
                            cuntt::ComputeUnit::Radix2, 8, inverse, 0, 4);
        test_structured_2x2(cuntt::ButterflyBackend::WarpHybrid, cuntt::ButterflyPrecision::Fp32,
                            cuntt::ComputeUnit::Radix2, 8, inverse, 0, 0, 4);
        test_structured_2x2(cuntt::ButterflyBackend::Hierarchical, cuntt::ButterflyPrecision::Fp32,
                            cuntt::ComputeUnit::Radix4, 11, inverse, 5);
        test_structured_2x2(cuntt::ButterflyBackend::OnlineReorder, cuntt::ButterflyPrecision::Fp32,
                            cuntt::ComputeUnit::Radix4, 11, inverse, 5);
        test_structured_2x2(cuntt::ButterflyBackend::TemporalTile, cuntt::ButterflyPrecision::Fp64,
                            cuntt::ComputeUnit::Radix4, 8, inverse);
        test_structured_2x2(cuntt::ButterflyBackend::OnlineReorder, cuntt::ButterflyPrecision::Fp64,
                            cuntt::ComputeUnit::Radix4, 11, inverse, 5);
    }
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

void test_layout(cuntt::ButterflyBackend backend, std::uint32_t log_n, cuntt::ButterflyPlacement placement,
                 cuntt::FftCore fft_core = cuntt::FftCore::Scalar, std::uint32_t local_stages = 10) {
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
    config.fft_core       = fft_core;
    config.local_stages   = local_stages;
    if (fft_core == cuntt::FftCore::CufftDxResident || fft_core == cuntt::FftCore::CufftDxDirect)
        config.tile_threads = 512;
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

void test_device_api() {
    constexpr std::uint32_t log_n = 11;
    constexpr std::size_t batch = 3;
    const std::size_t points = std::size_t{1} << log_n;

    cuntt::ButterflyConfig config;
    config.op = cuntt::ButterflyOperator::Fwht;
    config.precision = cuntt::ButterflyPrecision::Fp32;
    config.backend = cuntt::ButterflyBackend::OnlineReorder;
    config.compute_unit = cuntt::ComputeUnit::Radix4;
    config.log_n = log_n;
    config.local_stages = 5;
    config.batch = batch;
    config.auto_allocate_workspace = false;

    cuntt::ButterflyPlan plan(config);
    if (plan.data_size() != batch * points * sizeof(float) || plan.workspace_size() != plan.data_size())
        throw std::runtime_error("device API size query mismatch");
    if (plan.workspace() != nullptr)
        throw std::runtime_error("external-workspace plan unexpectedly allocated scratch");

    std::vector<float> input(batch * points);
    std::mt19937 random(0xd3a1U);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::generate(input.begin(), input.end(), [&] { return distribution(random); });
    std::vector<float> output(input.size());

    TestStream stream;
    TestDeviceBuffer device_input(plan.data_size());
    TestDeviceBuffer device_output(plan.data_size());
    TestDeviceBuffer workspace(plan.workspace_size());
    plan.set_stream(stream.get());
    if (plan.stream() != stream.get())
        throw std::runtime_error("plan did not retain the configured CUDA stream");

    bool rejected_missing_workspace = false;
    try {
        plan.execute_async(static_cast<const float*>(device_input.get()), static_cast<float*>(device_output.get()));
    } catch (const std::invalid_argument&) {
        rejected_missing_workspace = true;
    }
    if (!rejected_missing_workspace)
        throw std::runtime_error("device API accepted a missing external workspace");

    bool rejected_small_workspace = false;
    try {
        plan.set_workspace(workspace.get(), plan.workspace_size() - 1);
    } catch (const std::invalid_argument&) {
        rejected_small_workspace = true;
    }
    if (!rejected_small_workspace)
        throw std::runtime_error("device API accepted an undersized external workspace");

    plan.set_workspace(workspace.get(), plan.workspace_size());
    check_cuda(cudaMemcpyAsync(device_input.get(), input.data(), plan.data_size(), cudaMemcpyHostToDevice, stream.get()),
               "device API H2D");
    plan.execute_async(static_cast<const float*>(device_input.get()), static_cast<float*>(device_output.get()));
    check_cuda(cudaMemcpyAsync(output.data(), device_output.get(), plan.data_size(), cudaMemcpyDeviceToHost, stream.get()),
               "device API D2H");
    check_cuda(cudaStreamSynchronize(stream.get()), "device API stream synchronize");

    for (std::size_t transform = 0; transform < batch; ++transform) {
        std::vector<float> expected(input.begin() + transform * points, input.begin() + (transform + 1) * points);
        cuntt::reference_fwht(expected);
        for (std::size_t index = 0; index < points; ++index) {
            if (std::abs(expected[index] - output[transform * points + index]) > 1.0e-4F)
                throw std::runtime_error("asynchronous external-workspace FWHT mismatch");
        }
    }

    config.backend = cuntt::ButterflyBackend::TemporalTile;
    config.log_n = 8;
    config.batch = 1;
    config.local_stages = 0;
    config.auto_allocate_workspace = true;
    cuntt::ButterflyPlan in_place_plan(config);
    TestDeviceBuffer in_place_data(in_place_plan.data_size());
    TestDeviceBuffer distinct_output(in_place_plan.data_size());
    check_cuda(cudaMemset(in_place_data.get(), 0, in_place_plan.data_size()), "device API memset");
    bool rejected_alias = false;
    try {
        in_place_plan.execute_async(static_cast<const float*>(in_place_data.get()), static_cast<float*>(in_place_data.get()));
    } catch (const std::invalid_argument&) {
        rejected_alias = true;
    }
    if (!rejected_alias)
        throw std::runtime_error("out-of-place device API accepted aliased pointers");
    in_place_plan.execute_async(static_cast<const float*>(in_place_data.get()), static_cast<float*>(distinct_output.get()));
    check_cuda(cudaDeviceSynchronize(), "device API no-workspace synchronize");

    config.placement = cuntt::ButterflyPlacement::InPlace;
    cuntt::ButterflyPlan true_in_place_plan(config);
    bool rejected_distinct = false;
    try {
        true_in_place_plan.execute_async(static_cast<const float*>(in_place_data.get()), static_cast<float*>(distinct_output.get()));
    } catch (const std::invalid_argument&) {
        rejected_distinct = true;
    }
    if (!rejected_distinct)
        throw std::runtime_error("in-place device API accepted distinct pointers");
    true_in_place_plan.execute_async(static_cast<const float*>(in_place_data.get()), static_cast<float*>(in_place_data.get()));
    check_cuda(cudaDeviceSynchronize(), "in-place device API synchronize");

    std::cout << "PASS asynchronous device-pointer API\n";
}

void test_runtime_selector() {
    cuntt::ButterflyConfig config;
    config.op          = cuntt::ButterflyOperator::Fwht;
    config.precision   = cuntt::ButterflyPrecision::Fp32;
    config.placement   = cuntt::ButterflyPlacement::OutOfPlace;
    config.log_n       = 8;
    config.batch       = 4;
    config.auto_select = true;
    cuntt::ButterflyPlan plan(config);
    if (!plan.config().auto_select || plan.config().backend != cuntt::ButterflyBackend::TemporalTile ||
        plan.config().local_exchange != cuntt::LocalExchange::WarpRegister || !plan.selection().calibrated ||
        plan.selection().implementation != "cuButterfly-warp-register" || plan.selection().predicted_kernel_ms <= 0.0) {
        throw std::runtime_error("FWHT runtime selector did not resolve the calibrated mapping");
    }
    std::vector<float> input(config.batch * (1ULL << config.log_n), 1.0F);
    std::vector<float> output;
    plan.execute(input, output, 0, 1);
    for (std::size_t transform = 0; transform < config.batch; ++transform) {
        if (output[transform * (1ULL << config.log_n)] != static_cast<float>(1ULL << config.log_n)) {
            throw std::runtime_error("auto-selected FWHT produced an incorrect result");
        }
    }

    config.log_n = 15;
    config.batch = 1;
    cuntt::ButterflyPlan low_batch(config);
    if (low_batch.config().backend != cuntt::ButterflyBackend::OnlineReorder) {
        throw std::runtime_error("FWHT selector missed the low-batch online mapping");
    }
    config.batch = 16;
    cuntt::ButterflyPlan saturated(config);
    if (saturated.config().backend != cuntt::ButterflyBackend::TemporalTile ||
        saturated.config().local_exchange != cuntt::LocalExchange::WarpRegister) {
        throw std::runtime_error("FWHT selector missed the saturated warp-register mapping");
    }

#if CUBUTTERFLY_TEST_CUFFTDX
    config.op        = cuntt::ButterflyOperator::Fft;
    config.placement = cuntt::ButterflyPlacement::InPlace;
    config.log_n     = 14;
    config.batch     = 4;
    cuntt::ButterflyPlan fft_plan(config);
    if (fft_plan.config().fft_core != cuntt::FftCore::CufftDxDirect ||
        fft_plan.selection().implementation != "cuButterfly-cuFFTDx-direct") {
        throw std::runtime_error("FFT runtime selector did not resolve the direct processing unit");
    }
#endif

    config.op        = cuntt::ButterflyOperator::Fwht;
    config.placement = cuntt::ButterflyPlacement::OutOfPlace;
    config.log_n     = 9;
    try {
        cuntt::ButterflyPlan unsupported(config);
        throw std::runtime_error("runtime selector accepted an uncalibrated FWHT length");
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASS calibrated butterfly runtime selector\n";
}

}  // namespace

int main() {
    try {
        test_device_api();
        test_runtime_selector();
        test_extended_zeta_operators();
        test_structured_operators();
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
#ifdef CUBUTTERFLY_TEST_CUFFTDX
        for (std::uint32_t log_n = 3; log_n <= 10; ++log_n) {
            for (const bool inverse : {false, true}) {
                test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 32, cuntt::ComputeUnit::Auto, log_n, inverse, true, 0, 0,
                         cuntt::ComplexMultiply::FourMul, cuntt::FftCore::CufftDxBlock, cuntt::ButterflyPrecision::Fp32);
            }
        }
        for (const auto placement : {cuntt::ButterflyPlacement::OutOfPlace, cuntt::ButterflyPlacement::InPlace}) {
            test_layout(cuntt::ButterflyBackend::TemporalTile, 10, placement, cuntt::FftCore::CufftDxBlock);
            test_layout(cuntt::ButterflyBackend::TemporalTile, 12, placement, cuntt::FftCore::CufftDxDirect);
            test_layout(cuntt::ButterflyBackend::OnlineReorder, 12, placement, cuntt::FftCore::CufftDxBlock, 6);
            test_layout(cuntt::ButterflyBackend::OnlineReorder, 12, placement, cuntt::FftCore::CufftDxResident, 6);
            test_layout(cuntt::ButterflyBackend::OnlineReorder, 14, placement, cuntt::FftCore::CufftDxResident, 7);
        }
        for (const auto& point : {std::pair{12U, 6U}, std::pair{14U, 7U}}) {
            for (const auto cross_twiddle : {cuntt::CrossTwiddleMode::Table, cuntt::CrossTwiddleMode::Recurrence}) {
                for (const bool inverse : {false, true}) {
                    test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 512, cuntt::ComputeUnit::Auto,
                             point.first, inverse, true, point.second, 1, cuntt::ComplexMultiply::FourMul,
                             cuntt::FftCore::CufftDxResident, cuntt::ButterflyPrecision::Fp32, cross_twiddle);
                }
            }
        }
        for (std::uint32_t log_n = 11; log_n <= 14; ++log_n) {
            for (const std::uint32_t tile_threads : {256U, 512U, 1024U}) {
                if (log_n == 14 && tile_threads == 256)
                    continue;
                test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, tile_threads, cuntt::ComputeUnit::Auto,
                         log_n, false, true, 0, 0, cuntt::ComplexMultiply::FourMul,
                         cuntt::FftCore::CufftDxDirect, cuntt::ButterflyPrecision::Fp32);
            }
            test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 512, cuntt::ComputeUnit::Auto,
                     log_n, true, true, 0, 0, cuntt::ComplexMultiply::FourMul,
                     cuntt::FftCore::CufftDxDirect, cuntt::ButterflyPrecision::Fp32);
        }
        for (const auto& point : {std::pair{12U, 6U}, std::pair{16U, 6U}, std::pair{16U, 10U}}) {
            for (const auto cross_twiddle : {cuntt::CrossTwiddleMode::Table, cuntt::CrossTwiddleMode::Recurrence}) {
                for (const bool inverse : {false, true}) {
                    test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 32, cuntt::ComputeUnit::Auto,
                             point.first, inverse, true, point.second, 1, cuntt::ComplexMultiply::FourMul,
                             cuntt::FftCore::CufftDxBlock, cuntt::ButterflyPrecision::Fp32, cross_twiddle);
                }
            }
        }
        for (const auto& threads : {std::pair{128U, 1024U}, std::pair{1024U, 128U}}) {
            test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 32, cuntt::ComputeUnit::Auto,
                     16, false, true, 8, 1, cuntt::ComplexMultiply::FourMul,
                     cuntt::FftCore::CufftDxBlock, cuntt::ButterflyPrecision::Fp32,
                     cuntt::CrossTwiddleMode::Recurrence, threads.first, threads.second);
        }
        for (const auto& ept : {std::pair{4U, 16U}, std::pair{16U, 4U}}) {
            test_fft(cuntt::ButterflyBackend::OnlineReorder, 0, 0, 0, 256, cuntt::ComputeUnit::Auto,
                     18, false, true, 9, 1, cuntt::ComplexMultiply::FourMul,
                     cuntt::FftCore::CufftDxBlock, cuntt::ButterflyPrecision::Fp32,
                     cuntt::CrossTwiddleMode::Recurrence, 256, 256, ept.first, ept.second);
        }
#endif
#ifdef CUBUTTERFLY_TEST_TURBOFFT
        for (std::uint32_t log_n = 7; log_n <= 10; ++log_n) {
            test_fft(cuntt::ButterflyBackend::TemporalTile, 0, 0, 0, 32, cuntt::ComputeUnit::Auto, log_n, false, false, 0, 0,
                     cuntt::ComplexMultiply::FourMul, cuntt::FftCore::TurboFftGenerated, cuntt::ButterflyPrecision::Fp32);
        }
#endif
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
