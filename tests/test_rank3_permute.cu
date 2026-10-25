#include "general_kernels.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuntt::detail::GeneralNdLayout;
using cuntt::detail::launch_permute_nd;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct DeviceBuffer {
    void* data = nullptr;

    explicit DeviceBuffer(std::size_t bytes) { check_cuda(cudaMalloc(&data, bytes), "cudaMalloc"); }
    ~DeviceBuffer() {
        if (data != nullptr)
            cudaFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;

    Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreate"); }
    ~Stream() {
        if (value != nullptr)
            cudaStreamDestroy(value);
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

GeneralNdLayout packed_layout(const std::array<std::size_t, 3>& extents,
                              std::size_t batch, bool padded) {
    GeneralNdLayout layout{};
    layout.rank = 3;
    layout.batch = batch;
    layout.points = extents[0] * extents[1] * extents[2];
    if (padded) {
        // Deliberately make the logical innermost axis strided so launch
        // selection must retain the generic stride-aware fallback.
        layout.strides[0] = 47;
        layout.strides[1] = 9;
        layout.strides[2] = 2;
        layout.batch_stride = 151;
    } else {
        layout.strides[2] = 1;
        layout.strides[1] = extents[2];
        layout.strides[0] = extents[1] * extents[2];
        layout.batch_stride = layout.points;
    }
    for (std::size_t axis = 0; axis < 3; ++axis)
        layout.extents[axis] = extents[axis];
    return layout;
}

bool run_case(const std::array<std::size_t, 3>& extents, std::size_t batch,
              const std::array<std::uint32_t, 3>& permutation,
              std::size_t element_bytes, bool padded, const char* name) {
    try {
        const auto layout = packed_layout(extents, batch, padded);
        const std::size_t input_slots = batch * layout.batch_stride;
        const std::size_t output_slots = batch * layout.points;
        std::vector<std::uint8_t> host_input(input_slots * element_bytes);
        std::vector<std::uint8_t> expected(output_slots * element_bytes);
        std::vector<std::uint8_t> actual(expected.size());
        for (std::size_t index = 0; index < host_input.size(); ++index)
            host_input[index] = static_cast<std::uint8_t>((index * 37 + element_bytes * 11 + 5) & 0xff);

        // Independent reference: enumerate output coordinates and map them
        // back through the requested permutation into the strided input.
        const std::size_t output_extent1 = extents[permutation[1]];
        const std::size_t output_extent2 = extents[permutation[2]];
        for (std::size_t b = 0; b < batch; ++b) {
            for (std::size_t output0 = 0; output0 < extents[permutation[0]]; ++output0) {
                for (std::size_t output1 = 0; output1 < output_extent1; ++output1) {
                    for (std::size_t output2 = 0; output2 < output_extent2; ++output2) {
                        std::size_t input_coordinate[3]{};
                        input_coordinate[permutation[0]] = output0;
                        input_coordinate[permutation[1]] = output1;
                        input_coordinate[permutation[2]] = output2;
                        const std::size_t input_index =
                            b * layout.batch_stride +
                            input_coordinate[0] * layout.strides[0] +
                            input_coordinate[1] * layout.strides[1] +
                            input_coordinate[2] * layout.strides[2];
                        const std::size_t output_index =
                            b * layout.points +
                            (output0 * output_extent1 + output1) * output_extent2 + output2;
                        std::memcpy(expected.data() + output_index * element_bytes,
                                    host_input.data() + input_index * element_bytes,
                                    element_bytes);
                    }
                }
            }
        }

        Stream stream;
        DeviceBuffer input(host_input.size());
        DeviceBuffer output(actual.size());
        check_cuda(cudaMemcpyAsync(input.data, host_input.data(), host_input.size(),
                                   cudaMemcpyHostToDevice, stream.value),
                   "copy rank-3 input");
        launch_permute_nd(input.data, output.data, layout, permutation.data(), element_bytes,
                          stream.value);
        check_cuda(cudaGetLastError(), "launch rank-3 permutation test");
        check_cuda(cudaMemcpyAsync(actual.data(), output.data, actual.size(),
                                   cudaMemcpyDeviceToHost, stream.value),
                   "copy rank-3 output");
        check_cuda(cudaStreamSynchronize(stream.value), "synchronize rank-3 permutation test");
        if (actual != expected) {
            std::fprintf(stderr, "rank-3 permutation mismatch: %s (bytes=%zu, batch=%zu, padded=%d)\n",
                         name, element_bytes, batch, padded ? 1 : 0);
            return false;
        }
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "rank-3 permutation case failed: %s: %s\n", name, error.what());
        return false;
    }
}

bool run_all() {
    constexpr std::array<std::array<std::uint32_t, 3>, 6> permutations{{
        {{0, 1, 2}}, {{0, 2, 1}}, {{1, 0, 2}},
        {{1, 2, 0}}, {{2, 0, 1}}, {{2, 1, 0}},
    }};
    constexpr std::array<std::size_t, 4> element_bytes{{2, 4, 8, 16}};

    // Odd extents model compact spectra and force ragged shared-memory tiles.
    for (const auto bytes : element_bytes) {
        for (const auto& permutation : permutations) {
            if (!run_case({3, 5, 5}, 2, permutation, bytes, false, "packed-ragged"))
                return false;
        }
    }

    // A padded innermost stride must continue through the generic fallback.
    for (const auto& permutation : permutations) {
        if (!run_case({3, 5, 5}, 2, permutation, 4, true, "padded-inner-stride"))
            return false;
    }

    // These shapes intentionally exceed the portable rank-3 tiled grid
    // limits. Correctness proves that launch_permute_nd selected generic.
    if (!run_case({524289, 1, 1}, 1, {{1, 0, 2}}, 4, false, "same-inner-grid-y-fallback"))
        return false;
    if (!run_case({1, 2097121, 1}, 1, {{0, 2, 1}}, 4, false, "tiled-grid-y-fallback"))
        return false;
    if (!run_case({1, 1, 3}, 65536, {{1, 2, 0}}, 4, false, "slab-grid-z-fallback"))
        return false;
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || device_count == 0) {
        std::printf("SKIP rank-3 permutation tests: no CUDA device\n");
        return 0;
    }
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    return run_all() ? 0 : 1;
}
