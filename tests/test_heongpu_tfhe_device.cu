// Exact native/core-template/CPU comparison on the pinned TFHE domain.
#include "tfhe_shared_ntt.cuh"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned n = 1024;
constexpr Data64 q = 1152921504606877697ULL;
constexpr Data64 psi = 1689264667710614ULL;
Data64 mul(Data64 a, Data64 b) { return static_cast<unsigned __int128>(a) * b % q; }
Data64 power(Data64 a, Data64 e) {
    Data64 r = 1;
    for (; e; e >>= 1, a = mul(a, a)) if (e & 1) r = mul(r, a);
    return r;
}
unsigned reverse(unsigned x) {
    unsigned r = 0;
    for (unsigned i = 0; i < 10; ++i, x >>= 1) r = (r << 1) | (x & 1);
    return r;
}
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <bool Inverse, bool Adapted>
__global__ void transform(const Data64* input, Data64* output, const Root64* roots,
                          Modulus64 modulus, Ninverse64 scale) {
    __shared__ Data64 values[n];
    const unsigned i = threadIdx.x;
    values[i] = input[i];
    values[i + 512] = input[i + 512];
    __syncthreads();
    if constexpr (Adapted)
        cubutterfly::heongpu_tfhe::shared_ntt<10, Inverse>(values, roots, modulus, scale, false);
    else if constexpr (Inverse)
        heongpu::SmallInverseNTT(values, roots, modulus, scale, false);
    else
        heongpu::SmallForwardNTT(values, roots, modulus, false);
    output[i] = values[i];
    output[i + 512] = values[i + 512];
}
std::vector<Data64> reference(const std::vector<Data64>& input, bool inverse) {
    std::vector<Data64> output(n);
    const Data64 scale = inverse ? power(n, q - 2) : 1;
    for (unsigned k = 0; k < n; ++k) {
        const Data64 step = inverse ? power(psi, (2 * n - k) % (2 * n))
                                    : power(psi, 2 * reverse(k) + 1);
        Data64 sum = 0;
        if (inverse) {
            const Data64 ratio = mul(step, step);
            Data64 factor = step;
            for (unsigned j = 0; j < n; ++j, factor = mul(factor, ratio))
                sum = (sum + mul(input[reverse(j)] % q, factor)) % q;
        } else {
            Data64 factor = 1;
            for (unsigned j = 0; j < n; ++j, factor = mul(factor, step))
                sum = (sum + mul(input[j] % q, factor)) % q;
        }
        output[k] = mul(sum, scale);
    }
    return output;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
            std::mt19937_64 random(20261019);
            std::vector<Data64> input(n);
            for (auto& value : input) value = random() % q;
            if (reference(reference(input, false), true) != input)
                throw std::runtime_error("CPU forward/inverse reference roundtrip failed");
            std::cout << "TFHE CPU negacyclic DFT reference roundtrip passed\n";
            return 0;
        }
        cudaStream_t stream;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Data64 *device_input, *device_native, *device_adapted, *device_roots;
        check(cudaMalloc(&device_input, n * sizeof(Data64)));
        check(cudaMalloc(&device_native, n * sizeof(Data64)));
        check(cudaMalloc(&device_adapted, n * sizeof(Data64)));
        check(cudaMalloc(&device_roots, n * sizeof(Data64)));
        std::mt19937_64 random(20261019);
        for (bool inverse : {false, true}) {
            std::vector<Data64> roots(n);
            const Data64 root = inverse ? power(psi, q - 2) : psi;
            for (unsigned i = 0; i < n; ++i) roots[i] = power(root, reverse(i));
            check(cudaMemcpyAsync(device_roots, roots.data(), n * sizeof(Data64), cudaMemcpyHostToDevice, stream));
            for (unsigned pattern = 0; pattern < 3; ++pattern) {
                std::vector<Data64> input(n), native(n), adapted(n);
                for (unsigned i = 0; i < n; ++i)
                    input[i] = pattern == 0 ? random() % q
                             : pattern == 1 ? (i % 3 == 0 ? q : (i % 3 == 1 ? q - 1 : 0))
                                            : (i == 513 ? 1 : 0);
                const auto expected = reference(input, inverse);
                check(cudaMemcpyAsync(device_input, input.data(), n * sizeof(Data64), cudaMemcpyHostToDevice, stream));
                if (inverse) {
                    transform<true, false><<<1, 512, 0, stream>>>(device_input, device_native, device_roots, Modulus64(q), power(n, q - 2));
                    transform<true, true><<<1, 512, 0, stream>>>(device_input, device_adapted, device_roots, Modulus64(q), power(n, q - 2));
                } else {
                    transform<false, false><<<1, 512, 0, stream>>>(device_input, device_native, device_roots, Modulus64(q), 1);
                    transform<false, true><<<1, 512, 0, stream>>>(device_input, device_adapted, device_roots, Modulus64(q), 1);
                }
                check(cudaGetLastError());
                check(cudaMemcpyAsync(native.data(), device_native, n * sizeof(Data64), cudaMemcpyDeviceToHost, stream));
                check(cudaMemcpyAsync(adapted.data(), device_adapted, n * sizeof(Data64), cudaMemcpyDeviceToHost, stream));
                check(cudaStreamSynchronize(stream));
                if (native != expected || adapted != expected)
                    throw std::runtime_error("exact CPU/native/template mismatch: inverse=" + std::to_string(inverse) + " pattern=" + std::to_string(pattern));
            }
        }
        check(cudaFree(device_input)); check(cudaFree(device_native));
        check(cudaFree(device_adapted)); check(cudaFree(device_roots));
        check(cudaStreamDestroy(stream));
        std::cout << "TFHE shared device: 6 exact CPU/native/template cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
