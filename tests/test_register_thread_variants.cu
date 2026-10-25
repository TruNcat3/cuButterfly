#include "generated_unit_api.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

int main() {
    // Exact integer inputs and signed-permutation matrices avoid tolerance policy.
    unsigned cases = 0;
    for (unsigned log_n : {3U, 8U, 10U, 15U}) {
        const unsigned n = 1U << log_n;
        std::vector<float> input(n), expected(n), actual(n), matrices(log_n * 4);
        for (unsigned i = 0; i < n; ++i) input[i] = static_cast<int>(i % 7) - 3;
        float *source, *output, *coefficients;
        checked(cudaMalloc(&source, n * sizeof(float)));
        checked(cudaMalloc(&output, n * sizeof(float)));
        checked(cudaMalloc(&coefficients, matrices.size() * sizeof(float)));
        checked(cudaMemcpy(source, input.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        for (unsigned structured = 0; structured < 2; ++structured) {
            expected = input;
            for (unsigned stage = 0; stage < log_n; ++stage) {
                matrices[stage * 4] = structured ? 0 : 1;
                matrices[stage * 4 + 1] = 1;
                matrices[stage * 4 + 2] = 1;
                matrices[stage * 4 + 3] = structured ? 0 : -1;
                for (unsigned base = 0; base < n; base += 2U << stage)
                    for (unsigned i = 0; i < (1U << stage); ++i) {
                        float a = expected[base + i], b = expected[base + i + (1U << stage)];
                        expected[base + i] = structured ? b : a + b;
                        expected[base + i + (1U << stage)] = structured ? a : a - b;
                    }
            }
            checked(cudaMemcpy(coefficients, matrices.data(), matrices.size() * sizeof(float), cudaMemcpyHostToDevice));
            const auto options = structured
                ? cuntt::detail::generated_register_structured_thread_options(log_n)
                : cuntt::detail::generated_register_fwht_thread_options(log_n);
            for (unsigned threads : options) {
                if (structured)
                    cuntt::detail::launch_generated_register_structured(log_n, source, output, coefficients,
                                                                       1, n, 1, false, nullptr, threads);
                else
                    cuntt::detail::launch_generated_register_fwht(log_n, source, output, 1, n, 1, false, nullptr, threads);
                checked(cudaGetLastError());
                checked(cudaMemcpy(actual.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost));
                if (actual != expected) throw std::runtime_error("register thread variant changed arithmetic");
                ++cases;
            }
        }
        checked(cudaFree(source)); checked(cudaFree(output)); checked(cudaFree(coefficients));
    }
    std::cout << "register thread cases passed: " << cases << '\n';
}
