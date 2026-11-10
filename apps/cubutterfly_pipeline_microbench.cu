#include "batch_pipeline.cuh"

#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::json;

constexpr std::uint64_t kProbeValueBase = 0xC0BF'5EED'0000'0000ULL;
constexpr std::array<std::uint32_t, 3> kGroupCounts{2, 3, 4};
constexpr std::array<std::uint32_t, 10> kTileCounts{1, 2, 3, 4, 6, 8, 12, 16, 24, 32};

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(expression) + " failed at " + file + ':' + std::to_string(line) + ": " +
                             cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

// The value and address are both observable by the host-side verification.
// One thread is enough to keep the callback payload at the dispatch floor.
__global__ void pipeline_probe_kernel(std::uint64_t* output, std::uint64_t index, std::uint64_t value) {
    if (threadIdx.x == 0) {
        output[index] = value;
    }
}

struct Options {
    std::uint64_t warmup = 10;
    std::uint64_t repeat = 100;
    std::uint64_t trials = 3;
};

std::uint64_t parse_u64(const char* text, const std::string& option) {
    if (text == nullptr || *text == '\0') {
        throw std::invalid_argument("missing value for " + option);
    }
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            throw std::invalid_argument("invalid value for " + option + ": " + text);
        }
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0' ||
        value > static_cast<unsigned long long>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::invalid_argument("invalid value for " + option + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option(argv[index]);
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + option);
        }
        const std::uint64_t value = parse_u64(argv[++index], option);
        if (option == "--warmup") {
            options.warmup = value;
        } else if (option == "--repeat") {
            options.repeat = value;
        } else if (option == "--trials") {
            options.trials = value;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (options.repeat == 0) {
        throw std::invalid_argument("--repeat must be greater than zero");
    }
    if (options.trials == 0) {
        throw std::invalid_argument("--trials must be greater than zero");
    }
    return options;
}

struct DeviceBuffer {
    std::uint64_t* pointer = nullptr;

    ~DeviceBuffer() {
        if (pointer != nullptr) {
            cudaFree(pointer);
        }
    }

    void allocate(std::size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), bytes));
    }

    void clear(std::size_t bytes) {
        CUDA_CHECK(cudaMemset(pointer, 0, bytes));
    }
};

struct Stream {
    cudaStream_t value = nullptr;

    Stream() {
        CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking));
    }

    ~Stream() {
        if (value != nullptr) {
            cudaStreamDestroy(value);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};

struct EventPair {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    EventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        try {
            CUDA_CHECK(cudaEventCreate(&stop));
        } catch (...) {
            cudaEventDestroy(start);
            start = nullptr;
            throw;
        }
    }

    ~EventPair() {
        if (stop != nullptr) {
            cudaEventDestroy(stop);
        }
        if (start != nullptr) {
            cudaEventDestroy(start);
        }
    }

    EventPair(const EventPair&) = delete;
    EventPair& operator=(const EventPair&) = delete;
};

std::uint64_t expected_value(std::size_t group, std::size_t tile) {
    return kProbeValueBase + (static_cast<std::uint64_t>(tile) << 8) + static_cast<std::uint64_t>(group);
}

void enqueue_probe(cuntt::detail::BatchPipeline& pipeline, std::size_t groups, std::size_t tiles,
                   cudaStream_t caller, std::uint64_t* output) {
    // A unit tile makes BatchPipeline visit exactly one callback per requested
    // tile, while the implementation supplies the two ready/consumed slots.
    pipeline.enqueue(tiles, 1, caller,
                     [=](auto group, auto offset, auto count, auto slot, auto stream) {
                         (void)count;
                         (void)slot;
                         const auto index = static_cast<std::uint64_t>(offset) * static_cast<std::uint64_t>(groups) +
                                            static_cast<std::uint64_t>(group);
                         pipeline_probe_kernel<<<1, 1, 0, stream>>>(output, index,
                                                                      expected_value(group, offset));
                     });
}

bool verify(const DeviceBuffer& output, std::size_t groups, std::size_t tiles) {
    const std::size_t count = groups * tiles;
    std::vector<std::uint64_t> host(count, 0);
    CUDA_CHECK(cudaMemcpy(host.data(), output.pointer, count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    for (std::size_t tile = 0; tile < tiles; ++tile) {
        for (std::size_t group = 0; group < groups; ++group) {
            const std::size_t index = tile * groups + group;
            if (host[index] != expected_value(group, tile)) {
                return false;
            }
        }
    }
    return true;
}

std::vector<double> measure_trials(cuntt::detail::BatchPipeline& pipeline, std::size_t groups, std::size_t tiles,
                                   cudaStream_t caller, DeviceBuffer& output, const Options& options,
                                   EventPair& events) {
    const std::size_t bytes = groups * tiles * sizeof(std::uint64_t);
    output.clear(bytes);
    for (std::uint64_t iteration = 0; iteration < options.warmup; ++iteration) {
        enqueue_probe(pipeline, groups, tiles, caller, output.pointer);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> trials;
    trials.reserve(static_cast<std::size_t>(options.trials));
    for (std::uint64_t trial = 0; trial < options.trials; ++trial) {
        output.clear(bytes);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(events.start, caller));
        for (std::uint64_t iteration = 0; iteration < options.repeat; ++iteration) {
            enqueue_probe(pipeline, groups, tiles, caller, output.pointer);
        }
        CUDA_CHECK(cudaEventRecord(events.stop, caller));
        CUDA_CHECK(cudaEventSynchronize(events.stop));

        float elapsed_ms = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop));
        trials.push_back(static_cast<double>(elapsed_ms) / static_cast<double>(options.repeat));
    }
    return trials;
}

template <typename Values>
Json array_json(const Values& values) {
    Json result = Json::array();
    for (const auto value : values) {
        result.push_back(value);
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        constexpr std::size_t max_groups = kGroupCounts.back();
        constexpr std::size_t max_tiles = kTileCounts.back();
        DeviceBuffer output;
        output.allocate(max_groups * max_tiles * sizeof(std::uint64_t));
        EventPair events;

        Json rows = Json::array();
        rows.get_ref<Json::array_t&>().reserve(kGroupCounts.size() * kTileCounts.size());
        for (const std::uint32_t groups : kGroupCounts) {
            cuntt::detail::BatchPipeline pipeline(groups);
            Stream caller;
            for (const std::uint32_t tiles : kTileCounts) {
                const auto trial_kernel_ms = measure_trials(pipeline, groups, tiles, caller.value, output,
                                                            options, events);
                const bool correct = verify(output, groups, tiles);
                rows.push_back(Json{{"groups", groups},
                                    {"tiles", tiles},
                                    {"trial_kernel_ms", trial_kernel_ms},
                                    {"correct", correct}});
            }
        }

        Json profile{{"schema", "cubutterfly-pipeline-schedule-v1"},
                     {"hardware", Json{{"device", std::string(properties.name)},
                                        {"compute_capability", std::to_string(properties.major) + "." +
                                                                     std::to_string(properties.minor)},
                                        {"global_memory_bytes", static_cast<std::uint64_t>(properties.totalGlobalMem)},
                                        {"sm_count", properties.multiProcessorCount}}},
                     {"protocol", Json{{"warmup", options.warmup},
                                        {"repeat", options.repeat},
                                        {"trials", options.trials},
                                        {"launch_mode", "BatchPipeline::enqueue"},
                                        {"groups", array_json(kGroupCounts)},
                                        {"tiles", array_json(kTileCounts)},
                                        {"tile_batch", 1}}},
                     {"measurement", Json{{"kernel", "one-CTA one-thread distinct uint64 store"},
                                           {"purpose", "minimum BatchPipeline event/submission scheduling floor; not a full operator concurrency calibration"},
                                           {"timing", "cudaEvent elapsed time over repeat enqueue calls; allocation, initialization, correctness and warmup excluded"},
                                           {"dependencies", "one non-blocking caller stream per group count; BatchPipeline ready/consumed two-slot edges"}}},
                     {"rows", rows}};

        std::cout << profile.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cubutterfly_pipeline_microbench: " << error.what() << '\n';
        return 1;
    }
}
