#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/butterfly.hpp"

namespace {

std::string take_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

void print_usage() {
    std::cout
        << "cubutterfly_bench [--operator fwht|fft|xor-zeta] [--backend temporal-tile|hierarchical|online-reorder|warp-hybrid|stage-pipeline|cufft]\n"
        << "                    [--logN 8] [--batch 16384] [--inverse]\n"
        << "                    [--normalization none|inverse]\n"
        << "                    [--auto-select]\n"
        << "                    [--placement in-place|out-of-place]\n"
        << "                    [--batch-stride N]\n"
        << "                    [--element-stride N]\n"
        << "                    [--stage-space 1|2|4|8]\n"
        << "                    [--stage-handoff atomic|named-barrier]\n"
        << "                    [--tile-threads 32|64|128|256; resident also 512|1024] [--local-stages 5..10; cuFFTDx online 5..12] [--reorder-columns power-of-two]\n"
        << "                    [--prefix-threads 128|256|512|1024] [--suffix-threads 128|256|512|1024]\n"
        << "                    [--prefix-ept N] [--suffix-ept N]\n"
        << "                    [--warp-stages 0..5]\n"
        << "                    [--pipeline-warps 4|8]\n"
        << "                    [--compute-unit auto|radix2|radix4|radix8]\n"
        << "                    [--complex-multiply four-mul|gauss3]\n"
        << "                    [--cross-twiddle table|recurrence]\n"
        << "                    [--direct-boundary direct-strided|tiled-transpose]\n"
        << "                    [--local-exchange shared|warp-register]\n"
        << "                    [--fft-core scalar|thread-dft8|cta-dft8|wmma-dft8|cufftdx-block|cufftdx-direct|cufftdx-resident|turbofft-generated]\n"
        << "                    [--precision fp32|fp64|fp16-fp32|uint32]\n"
        << "                    [--warmup 20] [--repeat 100] [--verify] [--csv]\n"
        << "                    [--list-capabilities]\n";
}

void print_capabilities() {
    std::cout << "backend,min_logN,max_logN,radix2,radix4,radix8,warp_register,fft_thread_dft8,fft_cta_dft8,fft_wmma_dft8,fft_four_mul,fft_gauss3,fp32,fp64,fp16_fp32,uint32,in_place,strided_layout,constraint\n";
    for (const auto& capability : cuntt::butterfly_capabilities()) {
        std::cout << cuntt::butterfly_backend_name(capability.backend) << ',' << capability.min_log_n << ',' << capability.max_log_n << ','
                  << capability.radix2 << ',' << capability.radix4 << ',' << capability.radix8 << ',' << capability.warp_register << ','
                  << capability.fft_thread_dft8 << ',' << capability.fft_cta_dft8 << ',' << capability.fft_wmma_dft8 << ','
                  << capability.fft_four_mul << ','
                  << capability.fft_gauss3 << ',' << capability.fp32 << ',' << capability.fp64 << ',' << capability.fp16_fp32 << ','
                  << capability.uint32 << ','
                  << capability.in_place << ',' << capability.strided_layout << ",\"" << capability.constraint << "\"\n";
    }
}

float complex_error(cuntt::Complex32 expected, cuntt::Complex32 actual) {
    return std::hypot(expected.real - actual.real, expected.imag - actual.imag);
}

double complex_error(cuntt::Complex64 expected, cuntt::Complex64 actual) {
    return std::hypot(expected.real - actual.real, expected.imag - actual.imag);
}

float fft_tolerance_fp32(std::size_t n) {
    return 2.0e-4F * std::sqrt(std::max(1.0F, static_cast<float>(n) / 1024.0F));
}

double fft_tolerance_fp64(std::size_t n) {
    return 1.0e-10 * std::sqrt(std::max(1.0, static_cast<double>(n) / 1024.0));
}

}  // namespace

int main(int argc, char** argv) {
    cuntt::ButterflyConfig config;
    std::uint32_t          warmup = 20;
    std::uint32_t          repeat = 100;
    bool                   verify = false;
    bool                   csv    = false;

    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--list-capabilities") {
                print_capabilities();
                return 0;
            }
            if (arg == "--operator") {
                config.op = cuntt::parse_butterfly_operator(take_arg(index, argc, argv));
            } else if (arg == "--backend") {
                config.backend = cuntt::parse_butterfly_backend(take_arg(index, argc, argv));
            } else if (arg == "--logN") {
                config.log_n = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--batch") {
                config.batch = std::stoull(take_arg(index, argc, argv));
            } else if (arg == "--batch-stride") {
                config.batch_stride = std::stoull(take_arg(index, argc, argv));
            } else if (arg == "--element-stride") {
                config.element_stride = std::stoull(take_arg(index, argc, argv));
            } else if (arg == "--stage-space") {
                config.stage_space = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--stage-handoff") {
                config.stage_handoff = cuntt::parse_stage_handoff(take_arg(index, argc, argv));
            } else if (arg == "--tile-threads") {
                config.tile_threads = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--prefix-threads") {
                config.prefix_threads = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--suffix-threads") {
                config.suffix_threads = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--prefix-ept") {
                config.prefix_ept = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--suffix-ept") {
                config.suffix_ept = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--local-stages") {
                config.local_stages = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--reorder-columns") {
                config.reorder_columns = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--warp-stages") {
                config.warp_stages = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--pipeline-warps") {
                config.pipeline_warps = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--compute-unit") {
                config.compute_unit = cuntt::parse_compute_unit(take_arg(index, argc, argv));
            } else if (arg == "--complex-multiply") {
                config.complex_multiply = cuntt::parse_complex_multiply(take_arg(index, argc, argv));
            } else if (arg == "--cross-twiddle") {
                config.cross_twiddle = cuntt::parse_cross_twiddle_mode(take_arg(index, argc, argv));
            } else if (arg == "--direct-boundary") {
                config.direct_boundary = cuntt::parse_direct_boundary(take_arg(index, argc, argv));
            } else if (arg == "--local-exchange") {
                config.local_exchange = cuntt::parse_local_exchange(take_arg(index, argc, argv));
            } else if (arg == "--fft-core") {
                config.fft_core = cuntt::parse_fft_core(take_arg(index, argc, argv));
            } else if (arg == "--precision") {
                config.precision = cuntt::parse_butterfly_precision(take_arg(index, argc, argv));
            } else if (arg == "--placement") {
                config.placement = cuntt::parse_butterfly_placement(take_arg(index, argc, argv));
            } else if (arg == "--inverse") {
                config.inverse = true;
            } else if (arg == "--auto-select") {
                config.auto_select = true;
            } else if (arg == "--normalization") {
                const std::string mode = take_arg(index, argc, argv);
                if (mode == "none")
                    config.normalize_inverse = false;
                else if (mode == "inverse")
                    config.normalize_inverse = true;
                else
                    throw std::invalid_argument("normalization must be none or inverse");
            } else if (arg == "--warmup") {
                warmup = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--repeat") {
                repeat = std::stoul(take_arg(index, argc, argv));
            } else if (arg == "--verify") {
                verify = true;
            } else if (arg == "--csv") {
                csv = true;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        cuntt::ButterflyPlan plan(config);
        config = plan.config();
        std::mt19937                          random(0x43554246U + static_cast<unsigned int>(config.batch));
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        cuntt::ButterflyStats                 stats;
        float                                 max_error = 0.0F;
        bool                                  correct   = true;
        const std::size_t                     n         = std::size_t{1} << config.log_n;
        const std::size_t                     extent    = (config.batch - 1) * config.batch_stride + (n - 1) * config.element_stride + 1;

        if (config.op == cuntt::ButterflyOperator::Fwht) {
            if (config.precision == cuntt::ButterflyPrecision::Fp64) {
                std::vector<double> input(extent);
                std::generate(input.begin(), input.end(), [&] { return static_cast<double>(distribution(random)); });
                std::vector<double> output;
                stats = plan.execute(input, output, warmup, repeat);
                if (verify) {
                    double error = 0.0;
                    for (std::size_t transform = 0; transform < config.batch; ++transform) {
                        const auto          base = transform * config.batch_stride;
                        std::vector<double> expected(n);
                        for (std::size_t index = 0; index < n; ++index)
                            expected[index] = input[base + index * config.element_stride];
                        cuntt::reference_fwht(expected, config.inverse, config.normalize_inverse);
                        for (std::size_t index = 0; index < n; ++index)
                            error = std::max(error, std::abs(expected[index] - output[base + index * config.element_stride]));
                    }
                    max_error = static_cast<float>(error);
                    correct   = error <= fft_tolerance_fp64(n);
                }
            } else {
                std::vector<float> input(extent);
                std::generate(input.begin(), input.end(), [&] { return distribution(random); });
                std::vector<float> output;
                stats = plan.execute(input, output, warmup, repeat);
                if (verify) {
                    for (std::size_t transform = 0; transform < config.batch; ++transform) {
                        const auto         base = transform * config.batch_stride;
                        std::vector<float> expected(n);
                        for (std::size_t index = 0; index < n; ++index)
                            expected[index] = input[base + index * config.element_stride];
                        cuntt::reference_fwht(expected, config.inverse, config.normalize_inverse);
                        for (std::size_t index = 0; index < n; ++index)
                            max_error = std::max(max_error, std::abs(expected[index] - output[base + index * config.element_stride]));
                    }
                    correct = max_error <= 1.0e-4F;
                }
            }
        } else if (config.op == cuntt::ButterflyOperator::Fft) {
            if (config.precision == cuntt::ButterflyPrecision::Fp64) {
                std::vector<cuntt::Complex64> input(extent);
                std::generate(input.begin(), input.end(), [&] { return cuntt::Complex64{distribution(random), distribution(random)}; });
                std::vector<cuntt::Complex64> output;
                stats = plan.execute(input, output, warmup, repeat);
                if (verify) {
                    double error = 0.0;
                    for (std::size_t transform = 0; transform < config.batch; ++transform) {
                        const auto                    base = transform * config.batch_stride;
                        std::vector<cuntt::Complex64> expected(n);
                        for (std::size_t index = 0; index < n; ++index)
                            expected[index] = input[base + index * config.element_stride];
                        cuntt::reference_fft(expected, config.inverse, config.normalize_inverse);
                        for (std::size_t index = 0; index < n; ++index)
                            error = std::max(error, complex_error(expected[index], output[base + index * config.element_stride]));
                    }
                    max_error = static_cast<float>(error);
                    correct   = error <= 1.0e-10;
                }
            } else {
                std::vector<cuntt::Complex32> input(extent);
                std::generate(input.begin(), input.end(), [&] { return cuntt::Complex32{distribution(random), distribution(random)}; });
                std::vector<cuntt::Complex32> output;
                stats = plan.execute(input, output, warmup, repeat);
                if (verify) {
                    for (std::size_t transform = 0; transform < config.batch; ++transform) {
                        const auto                    base = transform * config.batch_stride;
                        std::vector<cuntt::Complex32> expected(n);
                        for (std::size_t index = 0; index < n; ++index)
                            expected[index] = input[base + index * config.element_stride];
                        cuntt::reference_fft(expected, config.inverse, config.normalize_inverse);
                        for (std::size_t index = 0; index < n; ++index)
                            max_error = std::max(max_error, complex_error(expected[index], output[base + index * config.element_stride]));
                    }
                    const float tolerance = config.precision == cuntt::ButterflyPrecision::Fp16Fp32 ? 1.0e-2F : fft_tolerance_fp32(n);
                    correct               = max_error <= tolerance;
                }
            }
        } else {
            std::uniform_int_distribution<std::uint32_t> integer_distribution(0, 1023);
            std::vector<std::uint32_t>                   input(extent);
            std::generate(input.begin(), input.end(), [&] { return integer_distribution(random); });
            std::vector<std::uint32_t> output;
            stats = plan.execute(input, output, warmup, repeat);
            if (verify) {
                for (std::size_t transform = 0; transform < config.batch; ++transform) {
                    const auto                 base = transform * config.batch_stride;
                    std::vector<std::uint32_t> expected(n);
                    for (std::size_t index = 0; index < n; ++index)
                        expected[index] = input[base + index * config.element_stride];
                    cuntt::reference_xor_zeta(expected, config.inverse);
                    for (std::size_t index = 0; index < n; ++index) {
                        if (expected[index] != output[base + index * config.element_stride]) {
                            correct   = false;
                            max_error = std::max(
                                max_error, static_cast<float>(std::abs(static_cast<std::int64_t>(expected[index]) -
                                                                       static_cast<std::int64_t>(output[base + index * config.element_stride]))));
                        }
                    }
                }
            }
        }

        const auto device = cuntt::current_device_info();
        std::cout << std::fixed << std::setprecision(6);
        if (csv) {
            std::cout << "device,compute_capability,operator,precision,direction,normalization,placement,auto_select,backend,compute_unit,complex_multiply,cross_twiddle,direct_boundary,local_"
                         "exchange,fft_core,stage_space,stage_handoff,tile_threads,prefix_threads,suffix_threads,prefix_ept,suffix_ept,prefix_units_per_cta,suffix_units_per_cta,local_stages,reorder_columns,warp_stages,pipeline_warps,logN,N,batch,element_stride,batch_"
                         "stride,warmup,repeat,h2d_ms,kernel_ms,d2h_ms,"
                         "transforms_s,Gbutterfly_s,points_s,max_error,correct\n";
            std::cout << '"' << device.name << "\"," << device.compute_major << '.' << device.compute_minor << ','
                      << cuntt::butterfly_operator_name(config.op) << ',' << cuntt::butterfly_precision_name(config.precision) << ','
                      << (config.inverse ? "inverse" : "forward") << ',' << (config.normalize_inverse ? "inverse" : "none") << ','
                      << cuntt::butterfly_placement_name(config.placement) << ',' << static_cast<int>(config.auto_select) << ','
                      << cuntt::butterfly_backend_name(config.backend) << ','
                      << cuntt::compute_unit_name(config.compute_unit) << ',' << cuntt::complex_multiply_name(config.complex_multiply) << ','
                      << cuntt::cross_twiddle_mode_name(config.cross_twiddle) << ','
                      << cuntt::direct_boundary_name(config.direct_boundary) << ','
                      << cuntt::local_exchange_name(config.local_exchange) << ',' << cuntt::fft_core_name(config.fft_core) << ',' << config.stage_space << ','
                      << cuntt::stage_handoff_name(config.stage_handoff) << ',' << config.tile_threads << ','
                      << config.prefix_threads << ',' << config.suffix_threads << ',' << config.prefix_ept << ',' << config.suffix_ept << ','
                      << config.prefix_units_per_cta << ',' << config.suffix_units_per_cta << ',' << config.local_stages << ','
                      << config.reorder_columns << ',' << config.warp_stages << ',' << config.pipeline_warps << ','
                      << config.log_n << ',' << n << ',' << config.batch << ',' << config.element_stride << ',' << config.batch_stride << ','
                      << warmup << ',' << repeat << ',' << stats.h2d_ms << ',' << stats.kernel_ms << ',' << stats.d2h_ms << ','
                      << stats.transforms_per_second << ',' << stats.butterflies_per_second / 1.0e9 << ',' << stats.points_per_second << ','
                      << max_error << ',' << (verify ? static_cast<int>(correct) : -1) << '\n';
        } else {
            std::cout << "device: " << device.name << " (sm_" << device.compute_major << device.compute_minor << ")\n"
                      << "operator: " << cuntt::butterfly_operator_name(config.op) << "\n"
                      << "precision: " << cuntt::butterfly_precision_name(config.precision) << "\n"
                      << "direction: " << (config.inverse ? "inverse" : "forward") << "\n"
                      << "normalization: " << (config.normalize_inverse ? "inverse" : "none") << "\n"
                      << "placement: " << cuntt::butterfly_placement_name(config.placement) << "\n"
                      << "auto_select: " << (config.auto_select ? "yes" : "no") << "\n"
                      << "backend: " << cuntt::butterfly_backend_name(config.backend) << "\n"
                      << "compute_unit: " << cuntt::compute_unit_name(config.compute_unit) << "\n"
                      << "complex_multiply: " << cuntt::complex_multiply_name(config.complex_multiply) << "\n"
                      << "cross_twiddle: " << cuntt::cross_twiddle_mode_name(config.cross_twiddle) << "\n"
                      << "direct_boundary: " << cuntt::direct_boundary_name(config.direct_boundary) << "\n"
                      << "local_exchange: " << cuntt::local_exchange_name(config.local_exchange) << "\n"
                      << "fft_core: " << cuntt::fft_core_name(config.fft_core) << "\n"
                      << "stage_space: " << config.stage_space << "\n"
                      << "stage_handoff: " << cuntt::stage_handoff_name(config.stage_handoff) << "\n"
                      << "tile_threads: " << config.tile_threads << "\n"
                      << "prefix_threads: " << config.prefix_threads << "\n"
                      << "suffix_threads: " << config.suffix_threads << "\n"
                      << "prefix_ept: " << config.prefix_ept << "\n"
                      << "suffix_ept: " << config.suffix_ept << "\n"
                      << "prefix_units_per_cta: " << config.prefix_units_per_cta << "\n"
                      << "suffix_units_per_cta: " << config.suffix_units_per_cta << "\n"
                      << "local_stages: " << config.local_stages << "\n"
                      << "reorder_columns: " << config.reorder_columns << "\n"
                      << "warp_stages: " << config.warp_stages << "\n"
                      << "pipeline_warps: " << config.pipeline_warps << "\n"
                      << "shape: logN=" << config.log_n << ", N=" << n << ", batch=" << config.batch << "\n"
                      << "batch_stride: " << config.batch_stride << "\n"
                      << "element_stride: " << config.element_stride << "\n"
                      << "kernel_ms: " << stats.kernel_ms << "\n"
                      << "Gbutterfly_s: " << stats.butterflies_per_second / 1.0e9 << "\n"
                      << "max_error: " << max_error << "\n"
                      << "correct: " << (verify ? (correct ? "yes" : "no") : "not checked") << '\n';
        }
        return verify && !correct ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage();
        return 1;
    }
}
