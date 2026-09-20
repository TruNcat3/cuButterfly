#include <cuda_runtime.h>

#include <nlohmann/json.hpp>

#include <algorithm>
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

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(expression) + " failed at " + file + ':' + std::to_string(line) + ": " +
                             cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

// One thread per CTA performs a distinct, observable store. The store keeps
// the launch from becoming an empty kernel without making atomics or useful
// butterfly arithmetic part of the dispatch measurement.
__global__ void schedule_probe_kernel(volatile std::uint64_t* output) {
    if (threadIdx.x == 0) {
        const std::uint64_t cta = static_cast<std::uint64_t>(blockIdx.x);
        output[cta] = cta + static_cast<std::uint64_t>(blockDim.x);
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

    void release() {
        if (pointer == nullptr) {
            return;
        }
        CUDA_CHECK(cudaFree(pointer));
        pointer = nullptr;
    }
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

    void destroy() {
        if (stop != nullptr) {
            CUDA_CHECK(cudaEventDestroy(stop));
            stop = nullptr;
        }
        if (start != nullptr) {
            CUDA_CHECK(cudaEventDestroy(start));
            start = nullptr;
        }
    }
};

void launch_probe(std::uint32_t grid_ctas, std::uint32_t threads, std::uint64_t* output) {
    schedule_probe_kernel<<<grid_ctas, threads>>>(output);
    // This catches launch-configuration errors without synchronizing the
    // kernel, so it does not add device work to the event interval.
    CUDA_CHECK(cudaGetLastError());
}

double measure_row(std::uint32_t grid_ctas, std::uint32_t threads, std::uint64_t* output, const Options& options,
                   EventPair& events) {
    for (std::uint64_t iteration = 0; iteration < options.warmup; ++iteration) {
        launch_probe(grid_ctas, threads, output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(events.start));
    for (std::uint64_t iteration = 0; iteration < options.repeat; ++iteration) {
        launch_probe(grid_ctas, threads, output);
    }
    CUDA_CHECK(cudaEventRecord(events.stop));
    CUDA_CHECK(cudaEventSynchronize(events.stop));

    float elapsed_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop));
    return static_cast<double>(elapsed_ms) / static_cast<double>(options.repeat);
}

double median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("cannot compute a median of zero trials");
    }
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return (values[middle - 1] + values[middle]) / 2.0;
}

std::vector<std::uint32_t> supported_thread_shapes(const cudaDeviceProp& properties) {
    if (properties.maxThreadsPerBlock <= 0) {
        throw std::runtime_error("device reports no threads per block");
    }
    const std::vector<std::uint32_t> candidates{32, 64, 128, 256, 512, 1024};
    std::vector<std::uint32_t> shapes;
    for (const std::uint32_t threads : candidates) {
        if (threads <= static_cast<std::uint32_t>(properties.maxThreadsPerBlock)) {
            shapes.push_back(threads);
        }
    }
    return shapes;
}

std::vector<std::uint32_t> grid_shapes(const cudaDeviceProp& properties) {
    if (properties.multiProcessorCount <= 0) {
        throw std::runtime_error("device reports no streaming multiprocessors");
    }
    const std::uint64_t sm_count = static_cast<std::uint64_t>(properties.multiProcessorCount);
    const std::vector<std::uint64_t> candidates{1, sm_count, sm_count * 8, sm_count * 64};
    std::vector<std::uint32_t> grids;
    for (const std::uint64_t grid : candidates) {
        if (grid > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
            throw std::runtime_error("requested grid exceeds CUDA grid dimension range");
        }
        const auto grid_value = static_cast<std::uint32_t>(grid);
        if (std::find(grids.begin(), grids.end(), grid_value) == grids.end()) {
            grids.push_back(grid_value);
        }
    }
    return grids;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        int device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        const std::vector<std::uint32_t> thread_shapes = supported_thread_shapes(properties);
        const std::vector<std::uint32_t> grids = grid_shapes(properties);
        if (thread_shapes.empty()) {
            throw std::runtime_error("device supports no requested thread shape");
        }

        const std::uint32_t max_grid = *std::max_element(grids.begin(), grids.end());
        const std::uint64_t output_bytes_u64 = static_cast<std::uint64_t>(max_grid) * sizeof(std::uint64_t);
        if (output_bytes_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("output allocation size exceeds host size_t range");
        }

        DeviceBuffer output;
        output.allocate(static_cast<std::size_t>(output_bytes_u64));
        EventPair events;

        Json rows = Json::array();
        rows.get_ref<Json::array_t&>().reserve(thread_shapes.size() * grids.size());
        for (const std::uint32_t threads : thread_shapes) {
            for (const std::uint32_t grid : grids) {
                std::vector<double> trial_kernel_ms;
                trial_kernel_ms.reserve(static_cast<std::size_t>(options.trials));
                for (std::uint64_t trial = 0; trial < options.trials; ++trial) {
                    trial_kernel_ms.push_back(measure_row(grid, threads, output.pointer, options, events));
                }
                rows.push_back(Json{{"threads", threads},
                                    {"grid_ctas", grid},
                                    {"trial_kernel_ms", trial_kernel_ms},
                                    {"median_kernel_ms", median(trial_kernel_ms)}});
            }
        }

        Json profile{{"schema", "cubutterfly-schedule-profile-v1"},
                     {"hardware", Json{{"device", std::string(properties.name)},
                                        {"compute_capability", std::to_string(properties.major) + "." +
                                                                     std::to_string(properties.minor)},
                                        {"global_memory_bytes", static_cast<std::uint64_t>(properties.totalGlobalMem)},
                                        {"sm_count", properties.multiProcessorCount}}},
                     {"protocol", Json{{"warmup", options.warmup},
                                        {"repeat", options.repeat},
                                        {"trials", options.trials},
                                        {"launch_mode", "direct"}}},
                     {"measurement", Json{{"kernel", "one-thread-per-CTA distinct output store"},
                                           {"purpose", "measure launch/dispatch lower-bound only; not a butterfly compute/per-core model"},
                                           {"timing", "cudaEvent elapsed time over direct kernel launches; allocation, initialization, event creation and warmup excluded"}}},
                     {"rows", rows}};

        events.destroy();
        output.release();
        std::cout << profile.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cubutterfly_schedule_microbench: " << error.what() << '\n';
        return 1;
    }
}
