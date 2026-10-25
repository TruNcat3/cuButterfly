// Direct pinned GPU-NTT control for the RNS core investigation.
// This deliberately calls the vendor ForwardCore/InverseCore dispatch through
// GPU_NTT_Inplace. It is a timing control, not a cuButterfly provider.
#include <cuntt/ntt.hpp>
#include <cuda_runtime.h>
#include "gpuntt/ntt_merge/ntt.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using Word = std::uint64_t;
using Wide = unsigned __int128;
constexpr Word kModuli[] = {1152921504606584833ULL, 576460756061519873ULL};

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
Word multiply(Word a, Word b, Word modulus) { return static_cast<Word>(Wide(a) * b % modulus); }
Word inverse(Word value, Word modulus) {
    Word result = 1;
    for (Word exponent = modulus - 2; exponent; exponent >>= 1, value = multiply(value, value, modulus))
        if (exponent & 1) result = multiply(result, value, modulus);
    return result;
}
Word reverse_bits(Word value, unsigned bits) {
    Word result = 0;
    for (unsigned i = 0; i < bits; ++i, value >>= 1) result = (result << 1) | (value & 1);
    return result;
}
std::vector<Root<Word>> roots(unsigned log_n, Word modulus, bool inverse_transform) {
    const std::size_t n = std::size_t{1} << log_n;
    const Word primitive = cuntt::find_primitive_power_of_two_root(log_n + 1, modulus);
    const Word root = inverse_transform ? inverse(primitive, modulus) : primitive;
    std::vector<Root<Word>> result(n);
    Word power = 1;
    for (std::size_t i = 0; i < n; ++i) {
        result[reverse_bits(i, log_n)] = power;
        power = multiply(power, root, modulus);
    }
    return result;
}

double run(unsigned log_n, unsigned batch, bool inverse_transform, unsigned repetitions) {
    const std::size_t n = std::size_t{1} << log_n;
    std::mt19937_64 random(0x20261019ULL + inverse_transform);
    std::vector<Word> host(n * batch);
    for (unsigned row = 0; row < batch; ++row)
        for (std::size_t i = 0; i < n; ++i)
            host[row * n + i] = random() % kModuli[row % 2];
    std::vector<Root<Word>> host_roots;
    for (Word modulus : kModuli) {
        const auto table = roots(log_n, modulus, inverse_transform);
        host_roots.insert(host_roots.end(), table.begin(), table.end());
    }

    Word* data = nullptr;
    Root<Word>* root_table = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&data), host.size() * sizeof(Word)));
    check(cudaMalloc(reinterpret_cast<void**>(&root_table), host_roots.size() * sizeof(Root<Word>)));
    check(cudaMemcpy(data, host.data(), host.size() * sizeof(Word), cudaMemcpyHostToDevice));
    check(cudaMemcpy(root_table, host_roots.data(), host_roots.size() * sizeof(Root<Word>), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Modulus<Word> host_moduli[] = {Modulus<Word>(kModuli[0]), Modulus<Word>(kModuli[1])};
    std::vector<Word> host_inverse{inverse(static_cast<Word>(n), kModuli[0]),
                                  inverse(static_cast<Word>(n), kModuli[1])};
    Modulus<Word>* device_moduli = nullptr;
    Word* device_inverse = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&device_moduli), 2 * sizeof(Modulus<Word>)));
    check(cudaMalloc(reinterpret_cast<void**>(&device_inverse), 2 * sizeof(Word)));
    check(cudaMemcpy(device_moduli, host_moduli, 2 * sizeof(Modulus<Word>), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_inverse, host_inverse.data(), 2 * sizeof(Word), cudaMemcpyHostToDevice));
    gpuntt::ntt_rns_configuration<Word> config{};
    config.n_power = static_cast<int>(log_n);
    config.ntt_type = inverse_transform ? gpuntt::INVERSE : gpuntt::FORWARD;
    config.ntt_layout = gpuntt::PerPolynomial;
    config.reduction_poly = gpuntt::ReductionPolynomial::X_N_plus;
    config.zero_padding = false;
    config.mod_inverse = device_inverse;
    config.stream = stream;

    auto launch = [&] {
        if (inverse_transform)
            gpuntt::GPU_INTT_Inplace(data, root_table, device_moduli, config, static_cast<int>(batch), 2);
        else
            gpuntt::GPU_NTT_Inplace(data, root_table, device_moduli, config, static_cast<int>(batch), 2);
    };
    for (unsigned i = 0; i < 10; ++i) launch();
    check(cudaStreamSynchronize(stream));
    cudaEvent_t begin = nullptr, end = nullptr;
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    check(cudaEventRecord(begin, stream));
    for (unsigned i = 0; i < repetitions; ++i) launch();
    check(cudaEventRecord(end, stream));
    check(cudaEventSynchronize(end));
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, begin, end));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream);
    cudaFree(device_inverse);
    cudaFree(device_moduli);
    cudaFree(root_table);
    cudaFree(data);
    return elapsed / repetitions;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const unsigned log_n = argc > 1 ? static_cast<unsigned>(std::stoul(argv[1])) : 14;
        const unsigned batch = argc > 2 ? static_cast<unsigned>(std::stoul(argv[2])) : 2;
        const unsigned repetitions = argc > 3 ? static_cast<unsigned>(std::stoul(argv[3])) : 80;
        for (bool inverse_transform : {false, true})
            std::cout << "provider,native-gpuntt,log_n," << log_n << ",batch," << batch
                      << ",inverse," << inverse_transform << ",mod_count,2"
                      << ",repetitions," << repetitions << ",median_ms," 
                      << run(log_n, batch, inverse_transform, repetitions) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
