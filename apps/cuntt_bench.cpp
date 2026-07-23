#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"

namespace {

std::string take_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

void print_usage() {
    std::cout << "cuntt_bench [--logN 16] [--batch 1] [--backend baseline|tile256|hybrid2d|compact-stage|stage-pipeline]\n"
              << "            [--stage-space 1|2|4|8]\n"
              << "            [--stage-handoff atomic|named-barrier]\n"
              << "            [--n1-log 8] [--rows-per-block 4] [--threads-per-block 256]\n"
              << "            [--compute-unit auto|radix2|radix4|radix8]\n"
              << "            [--cross-twiddle first|second|fused|fused-barrett]\n"
              << "            [--mod-multiply shoup|barrett]\n"
              << "            [--word-bits 32|64]\n"
              << "            [--output-order natural|bit-reversed]\n"
              << "            [--warmup 5] [--repeat 20] [--modulus Q] [--inverse] [--verify] [--csv]\n";
}

std::uint32_t reverse_bits(std::uint32_t value, std::uint32_t bits) {
    std::uint32_t reversed = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | ((value >> bit) & 1U);
    }
    return reversed;
}

bool verify_output(const std::vector<std::uint64_t>& input, const std::vector<std::uint64_t>& output, const cuntt::PlanConfig& config,
                   std::size_t& mismatch_index) {
    const std::size_t n = 1ULL << config.log_n;
    for (std::size_t batch_index = 0; batch_index < config.batch; ++batch_index) {
        std::vector<std::uint64_t> expected(input.begin() + batch_index * n, input.begin() + (batch_index + 1) * n);
        cuntt::reference_ntt(expected, config.modulus, config.inverse);
        if (config.output_order == cuntt::OutputOrder::BitReversed) {
            std::vector<std::uint64_t> reordered(n);
            for (std::size_t index = 0; index < n; ++index) {
                const std::size_t reversed = reverse_bits(static_cast<std::uint32_t>(index), config.log_n);
                reordered[index]           = expected[reversed];
            }
            expected.swap(reordered);
        }
        const auto output_begin = output.begin() + batch_index * n;
        const auto mismatch     = std::mismatch(expected.begin(), expected.end(), output_begin);
        if (mismatch.first != expected.end()) {
            mismatch_index = batch_index * n + static_cast<std::size_t>(mismatch.first - expected.begin());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    cuntt::PlanConfig config;
    std::uint32_t     warmup = 5;
    std::uint32_t     repeat = 20;
    bool              verify = false;
    bool              csv    = false;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }
            if (arg == "--logN") {
                config.log_n = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--batch") {
                config.batch = static_cast<std::size_t>(std::stoull(take_arg(i, argc, argv)));
            } else if (arg == "--backend") {
                config.backend = cuntt::parse_backend(take_arg(i, argc, argv));
            } else if (arg == "--stage-space") {
                config.stage_space = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--stage-handoff") {
                config.stage_handoff = cuntt::parse_stage_handoff(take_arg(i, argc, argv));
            } else if (arg == "--n1-log") {
                config.n1_log = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--rows-per-block") {
                config.rows_per_block = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--threads-per-block") {
                config.threads_per_block = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--compute-unit") {
                config.compute_unit = cuntt::parse_compute_unit(take_arg(i, argc, argv));
            } else if (arg == "--cross-twiddle") {
                config.cross_twiddle_placement = cuntt::parse_cross_twiddle_placement(take_arg(i, argc, argv));
            } else if (arg == "--mod-multiply") {
                config.modular_multiply = cuntt::parse_modular_multiply(take_arg(i, argc, argv));
            } else if (arg == "--word-bits") {
                config.word_bits = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--output-order") {
                config.output_order = cuntt::parse_output_order(take_arg(i, argc, argv));
            } else if (arg == "--warmup") {
                warmup = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--repeat") {
                repeat = static_cast<std::uint32_t>(std::stoul(take_arg(i, argc, argv)));
            } else if (arg == "--modulus") {
                config.modulus = std::stoull(take_arg(i, argc, argv));
            } else if (arg == "--inverse") {
                config.inverse = true;
            } else if (arg == "--verify") {
                verify = true;
            } else if (arg == "--csv") {
                csv = true;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        const auto  device = cuntt::current_device_info();
        cuntt::Plan plan(config);
        config                       = plan.config();
        const std::size_t          n = plan.points_per_transform();
        std::mt19937_64            random(0x43554e5454ULL + config.log_n + config.batch);
        std::vector<std::uint64_t> input(n * config.batch);
        for (auto& value : input) {
            value = random() % config.modulus;
        }

        std::vector<std::uint64_t> output;
        const auto                 stats = plan.execute(input, output, warmup, repeat);

        bool        correct        = true;
        std::size_t mismatch_index = 0;
        if (verify) {
            correct = verify_output(input, output, config, mismatch_index);
        }

        std::cout << std::fixed << std::setprecision(6);
        if (csv) {
            std::cout << "device,compute_capability,backend,stage_space,stage_handoff,compute_unit,word_bits,logN,N,batch,modulus,modulus_bits,n1_"
                         "log,rows_per_block,threads_"
                         "per_block,inverse,warmup,repeat,h2d_ms,kernel_ms,"
                         "d2h_ms,kernel_ntt_s,end_to_end_ntt_s,kernel_points_s,cross_twiddle,mod_multiply,output_order,correct\n";
            std::cout << '"' << device.name << "\"," << device.compute_major << '.' << device.compute_minor << ','
                      << cuntt::backend_name(config.backend) << ',' << config.stage_space << ',' << cuntt::stage_handoff_name(config.stage_handoff)
                      << ',' << cuntt::compute_unit_name(config.compute_unit) << ',' << config.word_bits << ',' << config.log_n << ',' << n << ','
                      << config.batch << ',' << config.modulus << ',' << (64U - static_cast<std::uint32_t>(__builtin_clzll(config.modulus))) << ','
                      << config.n1_log << ',' << config.rows_per_block << ',' << config.threads_per_block << ',' << static_cast<int>(config.inverse)
                      << ',' << warmup << ',' << repeat << ',' << stats.h2d_ms << ',' << stats.kernel_ms << ',' << stats.d2h_ms << ','
                      << stats.kernel_ntt_per_second << ',' << stats.end_to_end_ntt_per_second << ',' << stats.kernel_points_per_second << ','
                      << cuntt::cross_twiddle_placement_name(config.cross_twiddle_placement) << ','
                      << cuntt::modular_multiply_name(config.modular_multiply) << ',' << cuntt::output_order_name(config.output_order) << ','
                      << (verify ? static_cast<int>(correct) : -1) << '\n';
        } else {
            std::cout << "device: " << device.name << " (sm_" << device.compute_major << device.compute_minor << ")\n"
                      << "backend: " << cuntt::backend_name(config.backend) << "\n"
                      << "stage_space: " << config.stage_space << "\n"
                      << "stage_handoff: " << cuntt::stage_handoff_name(config.stage_handoff) << "\n"
                      << "compute_unit: " << cuntt::compute_unit_name(config.compute_unit) << "\n"
                      << "word_bits: " << config.word_bits << "\n"
                      << "shape: logN=" << config.log_n << ", N=" << n << ", batch=" << config.batch << "\n"
                      << "mapping: n1_log=" << config.n1_log << ", rows_per_block=" << config.rows_per_block
                      << ", threads_per_block=" << config.threads_per_block
                      << ", cross_twiddle=" << cuntt::cross_twiddle_placement_name(config.cross_twiddle_placement) << "\n"
                      << "mod_multiply: " << cuntt::modular_multiply_name(config.modular_multiply) << "\n"
                      << "output_order: " << cuntt::output_order_name(config.output_order) << "\n"
                      << "direction: " << (config.inverse ? "inverse" : "forward") << "\n"
                      << "h2d_ms: " << stats.h2d_ms << "\n"
                      << "kernel_ms: " << stats.kernel_ms << "\n"
                      << "d2h_ms: " << stats.d2h_ms << "\n"
                      << "kernel_ntt_s: " << stats.kernel_ntt_per_second << "\n"
                      << "end_to_end_ntt_s: " << stats.end_to_end_ntt_per_second << "\n"
                      << "kernel_points_s: " << stats.kernel_points_per_second << "\n"
                      << "correct: " << (verify ? (correct ? "yes" : "no") : "not checked") << "\n";
        }

        if (verify && !correct) {
            std::cerr << "verification mismatch at flat index " << mismatch_index << '\n';
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage();
        return 1;
    }
    return 0;
}
