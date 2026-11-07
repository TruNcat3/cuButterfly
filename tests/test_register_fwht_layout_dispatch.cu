// Standalone: nvcc -std=c++17 -O3 -arch=sm_80 -Isrc this_file.cu -o test
#include "register_fwht.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Allocation {
    float* pointer = nullptr;
    explicit Allocation(std::size_t count) { checked(cudaMalloc(&pointer, count * sizeof(float))); }
    ~Allocation() { cudaFree(pointer); }
};

// Dense Walsh signs are independent of the GPU's staged factorization. Integer
// inputs and power-of-two normalization permit exact comparisons, no tolerance.
std::vector<float> reference(const std::vector<float>& values, unsigned n, bool normalize) {
    std::vector<float> result(n, 0);
    for (unsigned row = 0; row < n; ++row) {
        int sum = 0;
        for (unsigned col = 0; col < values.size(); ++col)
            sum += (__builtin_parity(row & col) ? -1 : 1) * static_cast<int>(values[col]);
        result[row] = normalize ? static_cast<float>(sum) / n : static_cast<float>(sum);
    }
    return result;
}

template <unsigned LogN, unsigned Threads>
void run_case(unsigned input_offset, unsigned output_offset, unsigned input_stride,
              unsigned output_stride, unsigned input_distance, unsigned output_distance,
              bool normalize, unsigned logical_points, bool zero_extended) {
    constexpr unsigned n = 1U << LogN;
    constexpr unsigned batches = 3, guard = 16;
    constexpr float sentinel = -123456.0F;
    const auto input_size = guard + input_offset + batches * input_distance + guard;
    const auto output_size = guard + output_offset + batches * output_distance + guard;
    std::vector<float> input(input_size, sentinel), expected(output_size, sentinel);
    for (unsigned b = 0; b < batches; ++b) {
        std::vector<float> values(logical_points);
        for (unsigned i = 0; i < logical_points; ++i) {
            values[i] = static_cast<int>((i * 13 + b * 7) % 17) - 8;
            input[guard + input_offset + b * input_distance + i * input_stride] = values[i];
        }
        auto transformed = reference(values, n, normalize);
        for (unsigned i = 0; i < n; ++i)
            expected[guard + output_offset + b * output_distance + i * output_stride] = transformed[i];
    }
    Allocation source(input_size), destination(output_size);
    std::vector<float> initial(output_size, sentinel), actual(output_size), input_after(input_size);
    checked(cudaMemcpy(source.pointer, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
    checked(cudaMemcpy(destination.pointer, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice));
    float* in = source.pointer + guard + input_offset;
    float* out = destination.pointer + guard + output_offset;
    if (zero_extended) {
        cuntt::detail::launch_register_fwht_zero_extended<LogN, Threads>(
            in, out, batches, input_distance, output_distance, input_stride,
            output_stride, logical_points, nullptr);
    } else {
        if (input_distance != output_distance || input_stride != output_stride)
            throw std::runtime_error("exact API has one shared layout");
        cuntt::detail::launch_register_fwht<LogN, Threads>(
            in, out, batches, input_distance, input_stride, normalize, nullptr);
    }
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    checked(cudaMemcpy(actual.data(), destination.pointer, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checked(cudaMemcpy(input_after.data(), source.pointer, input_after.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (input_after != input) throw std::runtime_error("input or its guard changed");
    if (actual != expected) throw std::runtime_error("Walsh output, layout holes, or output guards differ");
}

template <unsigned LogN, unsigned Threads>
unsigned suite() {
    constexpr unsigned n = 1U << LogN;
    for (bool normalize : {false, true}) {
        run_case<LogN, Threads>(0, 0, 1, 1, n, n, normalize, n, false);
        run_case<LogN, Threads>(0, 0, 1, 1, n + 4, n + 4, normalize, n, false);
        // Independent pointer offsets exercise both scalar-load and scalar-store fallbacks.
        run_case<LogN, Threads>(1, 0, 1, 1, n, n, normalize, n, false);
        run_case<LogN, Threads>(0, 1, 1, 1, n, n, normalize, n, false);
        run_case<LogN, Threads>(1, 3, 1, 1, n + 1, n + 1, normalize, n, false);
        run_case<LogN, Threads>(0, 0, 2, 2, 2 * n + 4, 2 * n + 4, normalize, n, false);
    }
    // Zero extension keeps its own input/output distances and strides, including
    // a non-vector-sized tail. Full output, holes, and both guards are checked.
    run_case<LogN, Threads>(0, 0, 1, 1, n, n + 4, false, n - 5, true);
    run_case<LogN, Threads>(1, 3, 1, 1, n + 1, n + 5, false, n - 5, true);
    run_case<LogN, Threads>(1, 0, 2, 3, 2 * n + 1, 3 * n + 4, false, n - 5, true);
    return 15;
}
} // namespace

int main() {
    try {
        const unsigned cases = suite<8, 32>() + suite<12, 128>();
        std::cout << "FWHT layout dispatch cases passed: " << cases << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
