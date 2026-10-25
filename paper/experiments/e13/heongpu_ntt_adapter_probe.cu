#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gpuntt/common/nttparameters.cuh>
#include <gpuntt/ntt_merge/ntt.cuh>
#include <gpuntt/ntt_merge/ntt_cpu.cuh>

#include "cuntt/ntt.hpp"

namespace {

using U64 = std::uint64_t;
using U128 = unsigned __int128;

constexpr std::uint32_t kLogN = 12;
constexpr std::size_t kN = std::size_t{1} << kLogN;
// HEonGPU generate_proper_primes(2*N, 60, 1) returns this first candidate.
constexpr U64 kExpectedQ0 = 1152921504606830593ULL;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

U64 mod_mul(U64 left, U64 right, U64 modulus) {
    return static_cast<U64>((static_cast<U128>(left) * right) % modulus);
}

U64 mod_pow(U64 base, U64 exponent, U64 modulus) {
    U64 result = 1;
    base %= modulus;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exponent >>= 1U;
    }
    return result;
}

bool is_prime(U64 value) {
    if (value < 2) {
        return false;
    }
    for (const U64 small : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL,
                            19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
        if (value == small) {
            return true;
        }
        if ((value % small) == 0) {
            return false;
        }
    }

    U64 odd = value - 1;
    unsigned power = 0;
    while ((odd & 1U) == 0) {
        odd >>= 1U;
        ++power;
    }
    // Deterministic for all uint64 values.
    for (const U64 base : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL,
                           9780504ULL, 1795265022ULL}) {
        if ((base % value) == 0) {
            continue;
        }
        U64 witness = mod_pow(base, odd, value);
        if (witness == 1 || witness == value - 1) {
            continue;
        }
        bool probably_prime = false;
        for (unsigned round = 1; round < power; ++round) {
            witness = mod_mul(witness, witness, value);
            if (witness == value - 1) {
                probably_prime = true;
                break;
            }
        }
        if (!probably_prime) {
            return false;
        }
    }
    return true;
}

U64 generate_heongpu_q0() {
    const U64 factor = 2 * static_cast<U64>(kN);
    U64 candidate = (((U64{1} << 60U) - 1) / factor) * factor + 1;
    while (candidate > (U64{1} << 59U)) {
        if (is_prime(candidate)) {
            return candidate;
        }
        candidate -= factor;
    }
    throw std::runtime_error("could not reproduce HEonGPU q0");
}

U64 minimal_primitive_root(U64 modulus, U64 degree) {
    if ((modulus - 1) % degree != 0) {
        throw std::runtime_error("q0 does not support the requested root order");
    }
    const U64 quotient = (modulus - 1) / degree;
    U64 root = 0;
    for (U64 base = 2; base < 1000000; ++base) {
        const U64 candidate = mod_pow(base, quotient, modulus);
        if (mod_pow(candidate, degree / 2, modulus) == modulus - 1) {
            root = candidate;
            break;
        }
    }
    if (root == 0) {
        throw std::runtime_error("could not find primitive 2N-th root");
    }

    // This is the canonicalization used by HEonGPU's
    // find_minimal_primitive_root: minimize over all odd powers.
    U64 minimum = root;
    const U64 root_squared = mod_mul(root, root, modulus);
    U64 current = root;
    for (U64 exponent = 0; exponent < degree; exponent += 2) {
        minimum = std::min(minimum, current);
        current = mod_mul(current, root_squared, modulus);
    }
    return minimum;
}

std::uint32_t reverse_bits(std::uint32_t value, std::uint32_t bits) {
    std::uint32_t reversed = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1U) | ((value >> bit) & 1U);
    }
    return reversed;
}

std::vector<U64> make_input(U64 modulus) {
    std::mt19937_64 random(0xe13a4096ULL);
    std::vector<U64> values(kN);
    for (auto& value : values) {
        value = random() % modulus;
    }
    return values;
}

std::vector<U64> transform_natural(const std::vector<U64>& input, U64 modulus,
                                   U64 omega, U64 psi) {
    std::vector<U64> output(kN, 0);
    for (std::size_t k = 0; k < kN; ++k) {
        const U64 omega_k = mod_pow(omega, static_cast<U64>(k), modulus);
        U64 psi_power = 1;
        U64 omega_power = 1;
        U64 accumulator = 0;
        for (std::size_t j = 0; j < kN; ++j) {
            const U64 twisted = mod_mul(input[j], psi_power, modulus);
            const U64 term = mod_mul(twisted, omega_power, modulus);
            accumulator = static_cast<U64>(
                (static_cast<U128>(accumulator) + term) % modulus);
            psi_power = mod_mul(psi_power, psi, modulus);
            omega_power = mod_mul(omega_power, omega_k, modulus);
        }
        output[k] = accumulator;
    }
    return output;
}

std::vector<U64> cyclic_natural(const std::vector<U64>& input, U64 modulus,
                                U64 omega) {
    return transform_natural(input, modulus, omega, 1);
}

std::vector<U64> bit_reversed(const std::vector<U64>& natural) {
    std::vector<U64> output(kN);
    for (std::uint32_t index = 0; index < kN; ++index) {
        output[reverse_bits(index, kLogN)] = natural[index];
    }
    return output;
}

std::vector<U64> natural_from_bit_reversed(const std::vector<U64>& input) {
    std::vector<U64> output(kN);
    for (std::uint32_t index = 0; index < kN; ++index) {
        output[index] = input[reverse_bits(index, kLogN)];
    }
    return output;
}

std::size_t mismatch_count(const std::vector<U64>& expected,
                           const std::vector<U64>& actual) {
    if (expected.size() != actual.size()) {
        return std::numeric_limits<std::size_t>::max();
    }
    std::size_t count = 0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        count += expected[index] != actual[index];
    }
    return count;
}

std::uint32_t find_frequency_multiplier(U64 base, U64 target, U64 modulus) {
    // Both roots have exact order N, so their discrete-log relation is an odd
    // exponent modulo N. A bounded scan keeps this probe independent of any
    // library root-generation or discrete-log helper.
    U64 current = base;
    for (std::uint32_t exponent = 1; exponent < kN; exponent += 2) {
        if (current == target) {
            return exponent;
        }
        current = mod_mul(current, mod_mul(base, base, modulus), modulus);
    }
    return 0;
}

std::uint32_t inverse_frequency_multiplier(std::uint32_t multiplier) {
    if (multiplier == 0) {
        return 0;
    }
    for (std::uint32_t inverse = 1; inverse < kN; inverse += 2) {
        if ((static_cast<std::uint64_t>(multiplier) * inverse) % kN == 1) {
            return inverse;
        }
    }
    return 0;
}

std::vector<U64> reindex_frequency_for_heongpu(
    const std::vector<U64>& cu_natural, std::uint32_t multiplier_inverse) {
    std::vector<U64> aligned(kN);
    for (std::uint32_t index = 0; index < kN; ++index) {
        const auto cu_index =
            (static_cast<std::uint64_t>(index) * multiplier_inverse) % kN;
        aligned[index] = cu_natural[cu_index];
    }
    return aligned;
}

std::vector<U64> reindex_frequency_for_cu(const std::vector<U64>& heongpu_natural,
                                          std::uint32_t multiplier) {
    std::vector<U64> cu_input(kN);
    for (std::uint32_t index = 0; index < kN; ++index) {
        const auto heongpu_index =
            (static_cast<std::uint64_t>(index) * multiplier) % kN;
        cu_input[index] = heongpu_natural[heongpu_index];
    }
    return cu_input;
}

template <typename T> class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                              count_ * sizeof(T)),
                   "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    T* data() const noexcept { return data_; }

  private:
    std::size_t count_ = 0;
    T* data_ = nullptr;
};

std::vector<U64> run_cu_plan(const std::vector<U64>& input, U64 modulus,
                             bool inverse, cuntt::OutputOrder output_order,
                             cudaStream_t stream) {
    cuntt::PlanConfig config;
    config.log_n = kLogN;
    config.batch = 1;
    config.modulus = modulus;
    config.inverse = inverse;
    config.backend = output_order == cuntt::OutputOrder::BitReversed
                         ? cuntt::Backend::SharedIterative
                         : cuntt::Backend::Baseline;
    if (config.backend == cuntt::Backend::SharedIterative) {
        // SharedIterative is the general N=4096 bit-reversed-output path;
        // CompactStage is intentionally restricted to forward logN=20.
        config.compute_unit = cuntt::ComputeUnit::Radix2;
        config.stage_partition = {6, 6};
        config.threads_per_block = 128;
        config.dataflow_layout = inverse ? cuntt::DataflowLayout::Linear
                                          : cuntt::DataflowLayout::HermesXor;
    }
    config.word_bits = 64;
    config.input_order = cuntt::InputOrder::Natural;
    config.output_order = output_order;
    config.auto_allocate_workspace = true;

    cuntt::Plan plan(config);
    plan.set_stream(stream);
    DeviceBuffer<U64> device_input(kN);
    DeviceBuffer<U64> device_output(kN);
    std::vector<U64> output(kN);
    check_cuda(cudaMemcpyAsync(device_input.data(), input.data(),
                               kN * sizeof(U64), cudaMemcpyHostToDevice,
                               stream),
               "cuButterfly input copy");
    plan.execute_async(device_input.data(), device_output.data());
    check_cuda(cudaMemcpyAsync(output.data(), device_output.data(),
                               kN * sizeof(U64), cudaMemcpyDeviceToHost,
                               stream),
               "cuButterfly output copy");
    check_cuda(cudaStreamSynchronize(stream), "cuButterfly stream synchronize");
    return output;
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

struct ProbeResult {
    U64 q0 = 0;
    U64 psi = 0;
    U64 omega = 0;
    U64 cu_omega = 0;
    U64 n_inverse = 0;
    std::size_t gpu_forward_mismatches = 0;
    std::size_t gpu_cpu_mismatches = 0;
    std::size_t gpu_inverse_mismatches = 0;
    std::size_t cu_twist_mismatches = 0;
    std::size_t cu_twist_frequency_permutation_mismatches = 0;
    bool q0_matches_source = false;
    bool psi_order_ok = false;
    bool psi_square_ok = false;
    bool n_inverse_ok = false;
    bool gpu_forward_ok = false;
    bool gpu_cpu_forward_ok = false;
    bool gpu_inverse_ok = false;
    bool caller_stream_ok = false;
    bool cu_root_matches = false;
    bool cu_frequency_permutation_found = false;
    bool cu_frequency_permutation_inverse_ok = false;
    bool adapter_bidirectional_ok = false;
    std::uint32_t cu_frequency_multiplier = 0;
    std::uint32_t cu_frequency_multiplier_inverse = 0;
    bool cu_cyclic_forward_ok = false;
    bool cu_cyclic_inverse_ok = false;
    bool cu_twist_roundtrip_ok = false;
    bool cu_twist_matches_heongpu = false;
    std::string device_name;
    int device_index = -1;
};

void print_json(const ProbeResult& result, const std::string& status,
                const std::string& error = {}) {
    std::cout << "{\n"
              << "  \"status\": \"" << status << "\",\n"
              << "  \"n\": " << kN << ",\n"
              << "  \"log_n\": " << kLogN << ",\n"
              << "  \"device_index\": " << result.device_index << ",\n"
              << "  \"device_name\": \"" << result.device_name << "\",\n"
              << "  \"q0\": " << result.q0 << ",\n"
              << "  \"psi\": " << result.psi << ",\n"
              << "  \"omega\": " << result.omega << ",\n"
              << "  \"cu_omega\": " << result.cu_omega << ",\n"
              << "  \"n_inverse\": " << result.n_inverse << ",\n"
              << "  \"q0_matches_source\": "
              << json_bool(result.q0_matches_source) << ",\n"
              << "  \"psi_order_ok\": " << json_bool(result.psi_order_ok)
              << ",\n"
              << "  \"psi_square_ok\": " << json_bool(result.psi_square_ok)
              << ",\n"
              << "  \"n_inverse_ok\": " << json_bool(result.n_inverse_ok)
              << ",\n"
              << "  \"cu_root_matches\": "
              << json_bool(result.cu_root_matches) << ",\n"
              << "  \"gpu_forward_mismatches\": "
              << result.gpu_forward_mismatches << ",\n"
              << "  \"gpu_cpu_mismatches\": " << result.gpu_cpu_mismatches
              << ",\n"
              << "  \"gpu_inverse_mismatches\": "
              << result.gpu_inverse_mismatches << ",\n"
              << "  \"cu_twist_mismatches\": "
              << result.cu_twist_mismatches << ",\n"
              << "  \"cu_twist_frequency_permutation_mismatches\": "
              << result.cu_twist_frequency_permutation_mismatches << ",\n"
              << "  \"gpu_forward_ok\": "
              << json_bool(result.gpu_forward_ok) << ",\n"
              << "  \"gpu_cpu_forward_ok\": "
              << json_bool(result.gpu_cpu_forward_ok) << ",\n"
              << "  \"gpu_inverse_ok\": "
              << json_bool(result.gpu_inverse_ok) << ",\n"
              << "  \"caller_stream_ok\": "
              << json_bool(result.caller_stream_ok) << ",\n"
              << "  \"cu_cyclic_forward_ok\": "
              << json_bool(result.cu_cyclic_forward_ok) << ",\n"
              << "  \"cu_cyclic_inverse_ok\": "
              << json_bool(result.cu_cyclic_inverse_ok) << ",\n"
              << "  \"cu_frequency_permutation_found\": "
              << json_bool(result.cu_frequency_permutation_found) << ",\n"
              << "  \"cu_frequency_multiplier\": "
              << result.cu_frequency_multiplier << ",\n"
              << "  \"cu_frequency_multiplier_inverse\": "
              << result.cu_frequency_multiplier_inverse << ",\n"
              << "  \"cu_frequency_permutation_inverse_ok\": "
              << json_bool(result.cu_frequency_permutation_inverse_ok)
              << ",\n"
              << "  \"adapter_bidirectional_ok\": "
              << json_bool(result.adapter_bidirectional_ok) << ",\n"
              << "  \"cu_twist_roundtrip_ok\": "
              << json_bool(result.cu_twist_roundtrip_ok) << ",\n"
              << "  \"cu_twist_matches_heongpu\": "
              << json_bool(result.cu_twist_matches_heongpu) << ",\n"
              << "  \"error\": \"" << error << "\"\n"
              << "}\n";
}

ProbeResult run_probe() {
    ProbeResult result;
    check_cuda(cudaGetDevice(&result.device_index), "cudaGetDevice");
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, result.device_index),
               "cudaGetDeviceProperties");
    result.device_name = properties.name;

    result.q0 = generate_heongpu_q0();
    result.q0_matches_source = result.q0 == kExpectedQ0;
    result.psi = minimal_primitive_root(result.q0, 2 * static_cast<U64>(kN));
    result.omega = mod_mul(result.psi, result.psi, result.q0);
    result.psi_order_ok =
        mod_pow(result.psi, 2 * static_cast<U64>(kN), result.q0) == 1 &&
        mod_pow(result.psi, static_cast<U64>(kN), result.q0) == result.q0 - 1;
    result.psi_square_ok =
        mod_pow(result.omega, static_cast<U64>(kN), result.q0) == 1 &&
        mod_pow(result.omega, static_cast<U64>(kN / 2), result.q0) != 1;

    Modulus64 modulus(result.q0);
    gpuntt::NTTFactors<U64> factors(modulus, result.omega, result.psi);
    gpuntt::NTTParameters<U64> parameters(
        static_cast<int>(kLogN), factors, gpuntt::ReductionPolynomial::X_N_plus);
    result.n_inverse = parameters.n_inv;
    result.n_inverse_ok =
        mod_mul(static_cast<U64>(kN), result.n_inverse, result.q0) == 1;

    const std::vector<U64> input = make_input(result.q0);
    const std::vector<U64> expected_natural =
        transform_natural(input, result.q0, result.omega, result.psi);
    const std::vector<U64> expected_bit_reversed =
        bit_reversed(expected_natural);

    auto cpu_input = input;
    gpuntt::NTTCPU<U64> cpu(parameters);
    const auto cpu_forward = cpu.ntt(cpu_input);
    result.gpu_cpu_mismatches = mismatch_count(expected_bit_reversed, cpu_forward);
    result.gpu_cpu_forward_ok = result.gpu_cpu_mismatches == 0;
    auto cpu_inverse_input = cpu_forward;
    const auto cpu_inverse = cpu.intt(cpu_inverse_input);
    if (mismatch_count(input, cpu_inverse) != 0) {
        throw std::runtime_error("pinned CPU GPU-NTT inverse did not round-trip");
    }

    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "create caller stream");
    unsigned int stream_flags = 0;
    check_cuda(cudaStreamGetFlags(stream, &stream_flags),
               "query caller stream flags");

    DeviceBuffer<U64> device_values(kN);
    DeviceBuffer<U64> device_inverse(kN);
    DeviceBuffer<Root<U64>> device_forward_roots(kN);
    DeviceBuffer<Root<U64>> device_inverse_roots(kN);
    DeviceBuffer<Modulus<U64>> device_modulus(1);
    DeviceBuffer<Ninverse<U64>> device_n_inverse(1);

    const auto forward_roots = parameters.gpu_root_of_unity_table_generator(
        parameters.forward_root_of_unity_table);
    const auto inverse_roots = parameters.gpu_root_of_unity_table_generator(
        parameters.inverse_root_of_unity_table);
    check_cuda(cudaMemcpyAsync(device_forward_roots.data(), forward_roots.data(),
                               kN * sizeof(Root<U64>),
                               cudaMemcpyHostToDevice, stream),
               "copy forward roots");
    check_cuda(cudaMemcpyAsync(device_inverse_roots.data(), inverse_roots.data(),
                               kN * sizeof(Root<U64>),
                               cudaMemcpyHostToDevice, stream),
               "copy inverse roots");
    check_cuda(cudaMemcpyAsync(device_modulus.data(), &modulus,
                               sizeof(modulus), cudaMemcpyHostToDevice, stream),
               "copy q0");
    check_cuda(cudaMemcpyAsync(device_n_inverse.data(), &parameters.n_inv,
                               sizeof(parameters.n_inv), cudaMemcpyHostToDevice,
                               stream),
               "copy N inverse");
    check_cuda(cudaMemcpyAsync(device_values.data(), input.data(),
                               kN * sizeof(U64), cudaMemcpyHostToDevice, stream),
               "copy GPU-NTT input");

    gpuntt::ntt_rns_configuration<U64> forward_config = {
        .n_power = static_cast<int>(kLogN),
        .ntt_type = gpuntt::FORWARD,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .stream = stream};
    gpuntt::ntt_rns_configuration<U64> inverse_config = {
        .n_power = static_cast<int>(kLogN),
        .ntt_type = gpuntt::INVERSE,
        .ntt_layout = gpuntt::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
        .zero_padding = false,
        .mod_inverse = device_n_inverse.data(),
        .stream = stream};

    // This mirrors HEonGPU's raised-ciphertext forward call: in-place forward,
    // followed by an out-of-place inverse at the q0 boundary.
    gpuntt::GPU_NTT_Inplace(device_values.data(), device_forward_roots.data(),
                            device_modulus.data(), forward_config, 1, 1);
    std::vector<U64> gpu_forward(kN);
    check_cuda(cudaMemcpyAsync(gpu_forward.data(), device_values.data(),
                               kN * sizeof(U64), cudaMemcpyDeviceToHost, stream),
               "copy GPU-NTT forward");
    gpuntt::GPU_INTT(device_values.data(), device_inverse.data(),
                     device_inverse_roots.data(), device_modulus.data(),
                     inverse_config, 1, 1);
    std::vector<U64> gpu_inverse(kN);
    check_cuda(cudaMemcpyAsync(gpu_inverse.data(), device_inverse.data(),
                               kN * sizeof(U64), cudaMemcpyDeviceToHost, stream),
               "copy GPU-NTT inverse");
    check_cuda(cudaStreamSynchronize(stream), "synchronize GPU-NTT caller stream");

    result.gpu_forward_mismatches =
        mismatch_count(expected_bit_reversed, gpu_forward);
    result.gpu_inverse_mismatches = mismatch_count(input, gpu_inverse);
    result.gpu_forward_ok = result.gpu_forward_mismatches == 0;
    result.gpu_inverse_ok = result.gpu_inverse_mismatches == 0;
    result.caller_stream_ok =
        (stream_flags & cudaStreamNonBlocking) != 0 && result.gpu_forward_ok &&
        result.gpu_inverse_ok;

    result.cu_omega = cuntt::find_primitive_power_of_two_root(kLogN, result.q0);
    result.cu_root_matches = result.cu_omega == result.omega;
    result.cu_frequency_multiplier =
        find_frequency_multiplier(result.omega, result.cu_omega, result.q0);
    result.cu_frequency_multiplier_inverse =
        inverse_frequency_multiplier(result.cu_frequency_multiplier);
    result.cu_frequency_permutation_found =
        result.cu_frequency_multiplier != 0 &&
        result.cu_frequency_multiplier_inverse != 0;
    const auto cu_natural_expected =
        cyclic_natural(input, result.q0, result.cu_omega);
    const auto cu_forward_natural =
        run_cu_plan(input, result.q0, false, cuntt::OutputOrder::Natural, stream);
    result.cu_cyclic_forward_ok =
        mismatch_count(cu_natural_expected, cu_forward_natural) == 0;
    const auto cu_inverse = run_cu_plan(cu_forward_natural, result.q0, true,
                                        cuntt::OutputOrder::Natural, stream);
    result.cu_cyclic_inverse_ok = mismatch_count(input, cu_inverse) == 0;

    // Attempt the adapter composition: psi^j pre-twist, cuButterfly cyclic
    // plan, and bit-reversed output. A root mismatch must remain observable.
    std::vector<U64> twisted_input(kN);
    U64 psi_power = 1;
    for (std::size_t index = 0; index < kN; ++index) {
        twisted_input[index] = mod_mul(input[index], psi_power, result.q0);
        psi_power = mod_mul(psi_power, result.psi, result.q0);
    }
    const auto cu_twisted_bit_reversed = run_cu_plan(
        twisted_input, result.q0, false, cuntt::OutputOrder::BitReversed, stream);
    result.cu_twist_mismatches =
        mismatch_count(expected_bit_reversed, cu_twisted_bit_reversed);
    result.cu_twist_matches_heongpu = result.cu_twist_mismatches == 0;

    const auto cu_twisted_natural =
        natural_from_bit_reversed(cu_twisted_bit_reversed);
    const U64 psi_inverse = mod_pow(result.psi, result.q0 - 2, result.q0);
    U64 psi_inverse_power = 1;
    if (result.cu_frequency_permutation_found) {
        const auto cu_twisted_aligned = reindex_frequency_for_heongpu(
            cu_twisted_natural, result.cu_frequency_multiplier_inverse);
        result.cu_twist_frequency_permutation_mismatches = mismatch_count(
            expected_natural, cu_twisted_aligned);

        // Inverse adapter direction: map HEonGPU frequency k to cuButterfly
        // frequency k*r, then undo the cyclic inverse's psi^j pre-twist.
        const auto cu_frequency_input = reindex_frequency_for_cu(
            expected_natural, result.cu_frequency_multiplier);
        const auto cu_frequency_inverse = run_cu_plan(
            cu_frequency_input, result.q0, true, cuntt::OutputOrder::Natural,
            stream);
        std::vector<U64> inverse_aligned(kN);
        for (std::size_t index = 0; index < kN; ++index) {
            inverse_aligned[index] =
                mod_mul(cu_frequency_inverse[index], psi_inverse_power,
                        result.q0);
            psi_inverse_power =
                mod_mul(psi_inverse_power, psi_inverse, result.q0);
        }
        result.cu_frequency_permutation_inverse_ok =
            mismatch_count(input, inverse_aligned) == 0;
    }
    const auto cu_twisted_inverse = run_cu_plan(
        cu_twisted_natural, result.q0, true, cuntt::OutputOrder::Natural, stream);
    std::vector<U64> untwisted(kN);
    psi_inverse_power = 1;
    for (std::size_t index = 0; index < kN; ++index) {
        untwisted[index] =
            mod_mul(cu_twisted_inverse[index], psi_inverse_power, result.q0);
        psi_inverse_power = mod_mul(psi_inverse_power, psi_inverse, result.q0);
    }
    result.cu_twist_roundtrip_ok = mismatch_count(input, untwisted) == 0;
    check_cuda(cudaStreamDestroy(stream), "destroy caller stream");

    return result;
}

}  // namespace

int main() {
    ProbeResult result;
    try {
        result = run_probe();
        const bool pinned_ok = result.q0_matches_source && result.psi_order_ok &&
                               result.psi_square_ok && result.n_inverse_ok &&
                               result.gpu_forward_ok && result.gpu_cpu_forward_ok &&
                               result.gpu_inverse_ok && result.caller_stream_ok;
        result.adapter_bidirectional_ok =
            result.cu_frequency_permutation_found &&
            result.cu_twist_frequency_permutation_mismatches == 0 &&
            result.cu_frequency_permutation_inverse_ok;
        const bool cu_controls_ok = result.cu_cyclic_forward_ok &&
                                    result.cu_cyclic_inverse_ok;
        const bool probe_ok = pinned_ok && cu_controls_ok &&
                              result.cu_twist_roundtrip_ok &&
                              result.adapter_bidirectional_ok;
        const std::string status =
            probe_ok && result.cu_twist_matches_heongpu
                ? "adapter_forward_compatible"
                : (probe_ok && result.cu_frequency_permutation_found
                       ? "adapter_frequency_permutation_compatible"
                       : (pinned_ok && cu_controls_ok &&
                                  result.cu_twist_roundtrip_ok &&
                                  !result.cu_root_matches
                              ? "blocked_root_mismatch"
                              : "probe_failed"));
        print_json(result, status);
        return probe_ok ? 0 : 1;
    } catch (const std::exception& error) {
        print_json(result, "error", error.what());
        return 1;
    }
}
