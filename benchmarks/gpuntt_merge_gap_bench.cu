#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "gpuntt/ntt_merge/ntt.cuh"

namespace {

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    std::uint64_t result = 1;
    while (exponent != 0) {
        if ((exponent & 1U) != 0) {
            result = static_cast<std::uint64_t>((static_cast<unsigned __int128>(result) * base) % modulus);
        }
        base = static_cast<std::uint64_t>((static_cast<unsigned __int128>(base) * base) % modulus);
        exponent >>= 1;
    }
    return result;
}

void reference_ntt(std::vector<std::uint64_t>& values, std::uint64_t modulus, std::uint64_t root) {
    for (std::size_t i = 1, reversed = 0; i < values.size(); ++i) {
        std::size_t bit = values.size() >> 1;
        while ((reversed & bit) != 0) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (i < reversed) {
            std::swap(values[i], values[reversed]);
        }
    }
    for (std::size_t length = 2; length <= values.size(); length <<= 1) {
        const std::uint64_t step = mod_pow(root, values.size() / length, modulus);
        for (std::size_t base = 0; base < values.size(); base += length) {
            std::uint64_t omega = 1;
            for (std::size_t offset = 0; offset < length / 2; ++offset) {
                const std::uint64_t left = values[base + offset];
                const std::uint64_t right =
                    static_cast<std::uint64_t>((static_cast<unsigned __int128>(values[base + offset + length / 2]) * omega) % modulus);
                const std::uint64_t sum            = left + right;
                values[base + offset]              = sum >= modulus ? sum - modulus : sum;
                values[base + offset + length / 2] = left >= right ? left - right : modulus + left - right;
                omega                              = static_cast<std::uint64_t>((static_cast<unsigned __int128>(omega) * step) % modulus);
            }
        }
    }
}

std::size_t reverse_bits(std::size_t value, int bits) {
    std::size_t result = 0;
    for (int bit = 0; bit < bits; ++bit) {
        result = (result << 1) | ((value >> bit) & 1U);
    }
    return result;
}

__global__ void naturalize_kernel(const Data64* bit_reversed, Data64* natural, std::size_t count, int log_n) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const std::size_t local    = index & ((std::size_t{1} << log_n) - 1);
        const std::size_t batch    = index >> log_n;
        const std::size_t reversed = __brev(static_cast<unsigned int>(local)) >> (32 - log_n);
        natural[index]             = bit_reversed[(batch << log_n) + reversed];
    }
}

}  // namespace

#define CUDA_CHECK(call)                                                                              \
    do {                                                                                              \
        cudaError_t status = (call);                                                                  \
        if (status != cudaSuccess) {                                                                  \
            std::cerr << cudaGetErrorString(status) << " at " << __FILE__ << ':' << __LINE__ << '\n'; \
            return 1;                                                                                 \
        }                                                                                             \
    } while (0)

int main(int argc, char** argv) {
    const int         log_n      = argc > 1 ? std::atoi(argv[1]) : 16;
    const int         batch      = argc > 2 ? std::atoi(argv[2]) : 64;
    const int         warmup     = argc > 3 ? std::atoi(argv[3]) : 1000;
    const int         repeat     = argc > 4 ? std::atoi(argv[4]) : 200;
    const bool        verify     = argc > 5 && std::atoi(argv[5]) != 0;
    const bool        naturalize = argc > 6 && std::atoi(argv[6]) != 0;
    const std::size_t n          = std::size_t{1} << log_n;
    const std::size_t count      = n * batch;

    gpuntt::NTTParameters<Data64> parameters(log_n, gpuntt::ReductionPolynomial::X_N_minus);
    auto                          roots = parameters.gpu_root_of_unity_table_generator(parameters.forward_root_of_unity_table);

    std::vector<Data64> input(count);
    std::mt19937_64     random(0);
    for (auto& value : input)
        value = random() % parameters.modulus.value;

    Data64*       device_input   = nullptr;
    Data64*       device_output  = nullptr;
    Data64*       device_natural = nullptr;
    Root<Data64>* device_roots   = nullptr;
    CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(Data64)));
    CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(Data64)));
    if (naturalize) {
        CUDA_CHECK(cudaMalloc(&device_natural, count * sizeof(Data64)));
    }
    CUDA_CHECK(cudaMalloc(&device_roots, roots.size() * sizeof(Root<Data64>)));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(Data64), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_roots, roots.data(), roots.size() * sizeof(Root<Data64>), cudaMemcpyHostToDevice));

    gpuntt::ntt_configuration<Data64> config = {
        .n_power        = log_n,
        .ntt_type       = gpuntt::FORWARD,
        .ntt_layout     = gpuntt::NTTLayout::PerPolynomial,
        .reduction_poly = gpuntt::ReductionPolynomial::X_N_minus,
        .zero_padding   = false,
        .stream         = 0,
    };

    for (int i = 0; i < warmup; ++i) {
        gpuntt::GPU_NTT(device_input, device_output, device_roots, parameters.modulus, config, batch);
        if (naturalize) {
            naturalize_kernel<<<(count + 255) / 256, 256>>>(device_output, device_natural, count, log_n);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        gpuntt::GPU_NTT(device_input, device_output, device_roots, parameters.modulus, config, batch);
        if (naturalize) {
            naturalize_kernel<<<(count + 255) / 256, 256>>>(device_output, device_natural, count, log_n);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double invocation_ms  = elapsed_ms / repeat;
    const double ntt_per_second = 1000.0 * batch / invocation_ms;
    std::cout << "implementation,logN,batch,modulus,modulus_bits,warmup,repeat,"
                 "kernel_ms,kernel_ntt_s\n";
    std::cout << "GPU-NTT-Merge," << log_n << ',' << batch << ',' << parameters.modulus.value << ",60," << warmup << ',' << repeat << ','
              << invocation_ms << ',' << ntt_per_second << '\n';

    if (verify) {
        std::vector<Data64> actual(n);
        CUDA_CHECK(cudaMemcpy(actual.data(), naturalize ? device_natural : device_output, n * sizeof(Data64), cudaMemcpyDeviceToHost));
        std::vector<std::uint64_t> expected(input.begin(), input.begin() + n);
        reference_ntt(expected, parameters.modulus.value, parameters.root_of_unity);
        std::size_t natural_mismatches      = 0;
        std::size_t bit_reversed_mismatches = 0;
        for (std::size_t i = 0; i < n; ++i) {
            natural_mismatches += actual[i] != expected[i];
            bit_reversed_mismatches += actual[i] != expected[reverse_bits(i, log_n)];
        }
        std::cerr << "layout_check,natural_mismatches=" << natural_mismatches << ",bit_reversed_mismatches=" << bit_reversed_mismatches << '\n';
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
    if (device_natural != nullptr) {
        CUDA_CHECK(cudaFree(device_natural));
    }
    CUDA_CHECK(cudaFree(device_roots));
    return 0;
}
