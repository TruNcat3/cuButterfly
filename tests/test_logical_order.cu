#include "general_kernels.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using cuntt::detail::GeneralNdLayout;
using cuntt::detail::kGeneralMaxRank;
using cuntt::detail::launch_gather_ordered_nd;
using cuntt::detail::launch_scatter_ordered_nd;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct Buffer {
    void* data = nullptr;
    explicit Buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes), "allocate"); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "stream"); }
    ~Stream() { cudaStreamDestroy(value); }
};

std::size_t reference_address(std::size_t linear, const GeneralNdLayout& layout,
                              const std::uint32_t* flags) {
    std::size_t address = (linear / layout.points) * layout.batch_stride;
    std::size_t within = linear % layout.points;
    for (std::uint32_t axis = layout.rank; axis-- > 0;) {
        const auto coordinate = within % layout.extents[axis];
        within /= layout.extents[axis];
        std::size_t physical = coordinate;
        if (flags[axis]) {
            physical = 0;
            std::size_t mask = layout.extents[axis] / 2;
            for (std::size_t bit = 1; mask != 0; bit <<= 1, mask >>= 1)
                if (coordinate & bit)
                    physical |= mask;
        }
        address += physical * layout.strides[axis];
    }
    return address;
}

void run_case(std::uint32_t rank, std::size_t bytes, unsigned mode, bool interleaved) {
    GeneralNdLayout layout{};
    layout.rank = rank;
    layout.batch = 3;
    layout.points = 1;
    std::array<std::uint32_t, kGeneralMaxRank> flags{};
    for (std::uint32_t axis = 0; axis < rank; ++axis) {
        flags[axis] = mode == 0 ? 0 : (mode == 3 ? 1 : (axis + mode) % 2);
        layout.extents[axis] = axis == rank - 1 && rank > 2 ? 1 :
            (flags[axis] ? (axis % 2 == 0 ? 4 : 8) : (axis % 2 == 0 ? 3 : 2));
        layout.points *= layout.extents[axis];
    }
    // Axis zero is physically fastest (transpose of natural axis order).
    // Gaps between axis slabs and batch slabs remain untouched sentinels.
    std::size_t span = interleaved ? layout.batch * 2 : 2;
    for (std::uint32_t axis = 0; axis < rank; ++axis) {
        layout.strides[axis] = span;
        span = span * layout.extents[axis] + (interleaved ? layout.batch * 2 : 2);
    }
    layout.batch_stride = interleaved ? 1 : span + 7;
    const auto slots = span + (layout.batch - 1) * layout.batch_stride + 8;
    const auto count = layout.points * layout.batch;
    std::vector<std::uint8_t> input(slots * bytes, 0xa5);
    std::vector<std::uint8_t> expected_packed(count * bytes);
    std::vector<std::uint8_t> expected_output(slots * bytes, 0xa5);
    for (std::size_t linear = 0; linear < count; ++linear) {
        const auto address = reference_address(linear, layout, flags.data());
        for (std::size_t byte = 0; byte < bytes; ++byte) {
            const auto value = static_cast<std::uint8_t>((linear * 37 + (linear >> 8) * 11 + byte * 19) & 255);
            expected_packed[linear * bytes + byte] = value;
            input[address * bytes + byte] = value;
            expected_output[address * bytes + byte] = value;
        }
    }
    Stream stream;
    Buffer device_input(input.size()), packed(expected_packed.size()), output(input.size());
    check(cudaMemcpyAsync(device_input.data, input.data(), input.size(), cudaMemcpyHostToDevice,
                          stream.value), "upload input");
    check(cudaMemsetAsync(output.data, 0xa5, input.size(), stream.value), "sentinel output");
    launch_gather_ordered_nd(device_input.data, packed.data, layout, flags.data(), bytes, stream.value);
    std::vector<std::uint8_t> actual_packed(expected_packed.size());
    check(cudaMemcpyAsync(actual_packed.data(), packed.data, actual_packed.size(), cudaMemcpyDeviceToHost,
                          stream.value), "download gather");
    launch_scatter_ordered_nd(packed.data, output.data, layout, flags.data(), bytes, stream.value);
    std::vector<std::uint8_t> actual_output(input.size());
    check(cudaMemcpyAsync(actual_output.data(), output.data, actual_output.size(), cudaMemcpyDeviceToHost,
                          stream.value), "download scatter");
    // Also test scatter -> gather with independently uploaded natural data.
    check(cudaMemcpyAsync(packed.data, expected_packed.data(), expected_packed.size(), cudaMemcpyHostToDevice,
                          stream.value), "upload natural reference");
    launch_scatter_ordered_nd(packed.data, device_input.data, layout, flags.data(), bytes, stream.value);
    launch_gather_ordered_nd(device_input.data, packed.data, layout, flags.data(), bytes, stream.value);
    std::vector<std::uint8_t> roundtrip(expected_packed.size());
    check(cudaMemcpyAsync(roundtrip.data(), packed.data, roundtrip.size(), cudaMemcpyDeviceToHost,
                          stream.value), "download roundtrip");
    check(cudaStreamSynchronize(stream.value), "synchronize");
    if (actual_packed != expected_packed || actual_output != expected_output || roundtrip != expected_packed)
        throw std::runtime_error("ordered boundary reference/roundtrip/padding mismatch: rank=" +
            std::to_string(rank) + " bytes=" + std::to_string(bytes) + " mode=" + std::to_string(mode) +
            " interleaved=" + std::to_string(interleaved));
}

template <typename Exception = std::invalid_argument>
void reject(const GeneralNdLayout& layout, const std::uint32_t* flags, std::size_t bytes,
            bool same = false) {
    auto* input = reinterpret_cast<void*>(std::uintptr_t{16});
    auto* output = same ? input : reinterpret_cast<void*>(std::uintptr_t{32});
    for (const bool gather : {false, true}) {
        bool rejected = false;
        try {
            if (gather)
                launch_gather_ordered_nd(input, output, layout, flags, bytes, nullptr);
            else
                launch_scatter_ordered_nd(input, output, layout, flags, bytes, nullptr);
        } catch (const Exception&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid ordered boundary accepted");
    }
}

void invalid_cases() {
    GeneralNdLayout valid{};
    valid.rank = 1; valid.extents[0] = 4; valid.strides[0] = 1;
    valid.points = 4; valid.batch = 1; valid.batch_stride = 4;
    std::uint32_t flag = 1;
    auto bad = valid; bad.rank = 0; reject(bad, &flag, 4);
    bad.rank = kGeneralMaxRank + 1; reject(bad, &flag, 4);
    reject(valid, nullptr, 4);
    flag = 2; reject(valid, &flag, 4);
    flag = 1; bad = valid; bad.extents[0] = bad.points = 3; reject(bad, &flag, 4);
    bad = valid; bad.extents[0] = 0; reject(bad, &flag, 4);
    bad = valid; bad.points = 8; reject(bad, &flag, 4);
    bad = valid; bad.batch = 0; reject(bad, &flag, 4);
    bad = valid; bad.strides[0] = 0; reject(bad, &flag, 4);
    bad = valid; bad.batch = 2; bad.batch_stride = 0; reject(bad, &flag, 4);
    reject(valid, &flag, 3);
    reject(valid, &flag, 4, true);
    bad = valid; bad.strides[0] = std::numeric_limits<std::size_t>::max();
    reject<std::overflow_error>(bad, &flag, 4);
    bad = valid; bad.batch = std::numeric_limits<std::size_t>::max();
    reject<std::overflow_error>(bad, &flag, 4);
    bad = valid; bad.extents[0] = bad.points = 1; bad.batch_stride = 1;
    bad.batch = std::size_t{2147483647} * 256 + 1;
    reject<std::overflow_error>(bad, &flag, 4);
    bad = valid; bad.rank = 2; bad.extents[0] = std::size_t{1} << 63;
    bad.extents[1] = 4; bad.strides[1] = 1;
    std::uint32_t flags[2]{0, 0};
    reject<std::overflow_error>(bad, flags, 4);
}

}  // namespace

int main() {
    try {
        invalid_cases();
        int devices = 0;
        const auto status = cudaGetDeviceCount(&devices);
        if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
            (status == cudaSuccess && devices == 0)) {
            std::printf("SKIP logical-order GPU tests: no CUDA device\n");
            return 0;
        }
        check(status, "device count");
        for (std::uint32_t rank = 1; rank <= kGeneralMaxRank; ++rank)
            for (const std::size_t bytes : {2, 4, 8, 16})
                for (unsigned mode = 0; mode < 4; ++mode)
                    for (const bool interleaved : {false, true})
                        run_case(rank, bytes, mode, interleaved);
        std::printf("PASS logical-order gather/scatter (256 cases)\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
