#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <vkFFT.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
    }
}

void check_cufft(cufftResult status, const char* expression) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(expression) + " failed with cuFFT status " + std::to_string(status));
    }
}

void check_vkfft(VkFFTResult status, const char* expression) {
    if (status != VKFFT_SUCCESS) {
        throw std::runtime_error(std::string(expression) + ": " + getVkFFTErrorString(status));
    }
}

void check_driver(CUresult status, const char* expression) {
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(status, &message);
        throw std::runtime_error(std::string(expression) + ": " + (message == nullptr ? "unknown CUDA driver error" : message));
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)
#define CHECK_CUFFT(expr) check_cufft((expr), #expr)
#define CHECK_VKFFT(expr) check_vkfft((expr), #expr)
#define CHECK_DRIVER(expr) check_driver((expr), #expr)

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) { CHECK_CUDA(cudaMalloc(&pointer_, bytes)); }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* get() const noexcept { return pointer_; }

  private:
    void* pointer_ = nullptr;
};

class Event {
  public:
    Event() { CHECK_CUDA(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_{};
};

struct Options {
    std::string   library     = "vkfft";
    std::uint32_t log_n       = 12;
    std::uint64_t batch       = 0;
    std::uint64_t total_points = 1ULL << 22;
    std::uint32_t warmup      = 20;
    std::uint32_t repeat      = 100;
    bool          verify      = false;
    bool          csv         = false;
};

const char* take_arg(int& index, int argc, char** argv) {
    if (++index == argc) {
        throw std::invalid_argument("missing option value");
    }
    return argv[index];
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--library") {
            options.library = take_arg(index, argc, argv);
        } else if (argument == "--logN") {
            options.log_n = static_cast<std::uint32_t>(std::stoul(take_arg(index, argc, argv)));
        } else if (argument == "--batch") {
            options.batch = std::stoull(take_arg(index, argc, argv));
        } else if (argument == "--total-points") {
            options.total_points = std::stoull(take_arg(index, argc, argv));
        } else if (argument == "--warmup") {
            options.warmup = static_cast<std::uint32_t>(std::stoul(take_arg(index, argc, argv)));
        } else if (argument == "--repeat") {
            options.repeat = static_cast<std::uint32_t>(std::stoul(take_arg(index, argc, argv)));
        } else if (argument == "--verify") {
            options.verify = true;
        } else if (argument == "--csv") {
            options.csv = true;
        } else if (argument == "--help") {
            std::cout << "vkfft_bench [--library cufft|vkfft] [--logN 12] [--batch N | --total-points N] "
                         "[--warmup 20] [--repeat 100] [--verify] [--csv]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    if (options.library != "cufft" && options.library != "vkfft") {
        throw std::invalid_argument("library must be cufft or vkfft");
    }
    if (options.log_n < 1 || options.log_n > 30 || options.repeat == 0) {
        throw std::invalid_argument("logN must be in [1, 30] and repeat must be positive");
    }
    const std::uint64_t n = 1ULL << options.log_n;
    if (options.batch == 0) {
        options.batch = std::max<std::uint64_t>(1, options.total_points / n);
    }
    if (options.batch > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("batch exceeds the cuFFT integer limit");
    }
    return options;
}

template <typename Launch>
double time_launch(Launch&& launch, std::uint32_t warmup, std::uint32_t repeat) {
    for (std::uint32_t iteration = 0; iteration < warmup; ++iteration) {
        launch();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    Event start;
    Event stop;
    CHECK_CUDA(cudaEventRecord(start.get()));
    for (std::uint32_t iteration = 0; iteration < repeat; ++iteration) {
        launch();
    }
    CHECK_CUDA(cudaEventRecord(stop.get()));
    CHECK_CUDA(cudaEventSynchronize(stop.get()));
    float elapsed_ms = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
    return elapsed_ms / repeat;
}

std::vector<cufftComplex> make_input(std::uint64_t elements) {
    std::mt19937 generator(20260723);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<cufftComplex> input(elements);
    for (auto& value : input) {
        value.x = distribution(generator);
        value.y = distribution(generator);
    }
    return input;
}

std::vector<cufftComplex> cufft_reference(const std::vector<cufftComplex>& input, std::uint32_t log_n, std::uint64_t batch) {
    int n = 1 << log_n;
    const std::size_t bytes = input.size() * sizeof(cufftComplex);
    DeviceBuffer buffer(bytes);
    CHECK_CUDA(cudaMemcpy(buffer.get(), input.data(), bytes, cudaMemcpyHostToDevice));
    cufftHandle plan = 0;
    CHECK_CUFFT(cufftPlanMany(&plan, 1, &n, &n, 1, n, &n, 1, n,
                              CUFFT_C2C, static_cast<int>(batch)));
    CHECK_CUFFT(cufftExecC2C(plan, static_cast<cufftComplex*>(buffer.get()), static_cast<cufftComplex*>(buffer.get()), CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<cufftComplex> output(input.size());
    CHECK_CUDA(cudaMemcpy(output.data(), buffer.get(), bytes, cudaMemcpyDeviceToHost));
    CHECK_CUFFT(cufftDestroy(plan));
    return output;
}

struct ErrorStats {
    double max_abs = 0.0;
    double relative_l2 = 0.0;
};

ErrorStats compare(const std::vector<cufftComplex>& actual, const std::vector<cufftComplex>& expected) {
    long double error_norm = 0.0;
    long double reference_norm = 0.0;
    double max_abs = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double real_error = static_cast<double>(actual[index].x) - expected[index].x;
        const double imag_error = static_cast<double>(actual[index].y) - expected[index].y;
        const double absolute_error = std::hypot(real_error, imag_error);
        max_abs = std::max(max_abs, absolute_error);
        error_norm += real_error * real_error + imag_error * imag_error;
        reference_norm += static_cast<double>(expected[index].x) * expected[index].x +
                          static_cast<double>(expected[index].y) * expected[index].y;
    }
    return {max_abs, std::sqrt(static_cast<double>(error_norm / std::max(reference_norm, 1.0L)))};
}

struct Result {
    double plan_ms = 0.0;
    double kernel_ms = 0.0;
    ErrorStats error{};
};

Result run_cufft(const Options& options, const std::vector<cufftComplex>& input,
                 const std::vector<cufftComplex>* reference) {
    int n = 1 << options.log_n;
    const std::size_t bytes = input.size() * sizeof(cufftComplex);
    DeviceBuffer buffer(bytes);
    CHECK_CUDA(cudaMemcpy(buffer.get(), input.data(), bytes, cudaMemcpyHostToDevice));
    cufftHandle plan = 0;
    const auto plan_start = std::chrono::steady_clock::now();
    CHECK_CUFFT(cufftPlanMany(&plan, 1, &n, &n, 1, n, &n, 1, n,
                              CUFFT_C2C, static_cast<int>(options.batch)));
    const auto plan_stop = std::chrono::steady_clock::now();
    auto launch = [&] {
        CHECK_CUFFT(cufftExecC2C(plan, static_cast<cufftComplex*>(buffer.get()), static_cast<cufftComplex*>(buffer.get()), CUFFT_FORWARD));
    };
    Result result;
    result.plan_ms = std::chrono::duration<double, std::milli>(plan_stop - plan_start).count();
    result.kernel_ms = time_launch(launch, options.warmup, options.repeat);
    if (reference != nullptr) {
        CHECK_CUDA(cudaMemcpy(buffer.get(), input.data(), bytes, cudaMemcpyHostToDevice));
        launch();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<cufftComplex> output(input.size());
        CHECK_CUDA(cudaMemcpy(output.data(), buffer.get(), bytes, cudaMemcpyDeviceToHost));
        result.error = compare(output, *reference);
    }
    CHECK_CUFFT(cufftDestroy(plan));
    return result;
}

Result run_vkfft(const Options& options, const std::vector<cufftComplex>& input,
                 const std::vector<cufftComplex>* reference) {
    const std::uint64_t n = 1ULL << options.log_n;
    std::uint64_t bytes = input.size() * sizeof(cufftComplex);
    DeviceBuffer storage(bytes);
    void* buffer = storage.get();
    CHECK_CUDA(cudaMemcpy(buffer, input.data(), bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaFree(nullptr));
    CHECK_DRIVER(cuInit(0));
    CUdevice device = 0;
    CHECK_DRIVER(cuDeviceGet(&device, 0));
    VkFFTConfiguration configuration{};
    configuration.FFTdim = 1;
    configuration.size[0] = n;
    configuration.numberBatches = options.batch;
    configuration.device = &device;
    configuration.buffer = &buffer;
    configuration.bufferSize = &bytes;
    configuration.makeForwardPlanOnly = 1;

    VkFFTApplication application{};
    const auto plan_start = std::chrono::steady_clock::now();
    CHECK_VKFFT(initializeVkFFT(&application, configuration));
    const auto plan_stop = std::chrono::steady_clock::now();
    VkFFTLaunchParams launch_parameters{};
    auto launch = [&] { CHECK_VKFFT(VkFFTAppend(&application, -1, &launch_parameters)); };

    Result result;
    result.plan_ms = std::chrono::duration<double, std::milli>(plan_stop - plan_start).count();
    result.kernel_ms = time_launch(launch, options.warmup, options.repeat);
    if (reference != nullptr) {
        CHECK_CUDA(cudaMemcpy(buffer, input.data(), bytes, cudaMemcpyHostToDevice));
        launch();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::vector<cufftComplex> output(input.size());
        CHECK_CUDA(cudaMemcpy(output.data(), buffer, bytes, cudaMemcpyDeviceToHost));
        result.error = compare(output, *reference);
    }
    deleteVkFFT(&application);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::uint64_t n = 1ULL << options.log_n;
        if (options.batch > std::numeric_limits<std::uint64_t>::max() / n) {
            throw std::invalid_argument("element count overflows uint64_t");
        }
        const std::uint64_t elements = n * options.batch;
        const auto input = make_input(elements);
        std::vector<cufftComplex> reference;
        if (options.verify) {
            reference = cufft_reference(input, options.log_n, options.batch);
        }
        const auto result = options.library == "cufft"
                                ? run_cufft(options, input, options.verify ? &reference : nullptr)
                                : run_vkfft(options, input, options.verify ? &reference : nullptr);

        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
        int cufft_version = 0;
        CHECK_CUFFT(cufftGetVersion(&cufft_version));
        const std::string library_version = options.library == "cufft" ? std::to_string(cufft_version)
                                                                       : std::to_string(VkFFTGetVersion());
        const double transforms_s = options.batch / (result.kernel_ms / 1000.0);
        const double butterflies_s = transforms_s * static_cast<double>(n / 2) * options.log_n;
        const bool correct = result.error.relative_l2 <= 5.0e-5;
        std::cout << std::fixed << std::setprecision(9);
        if (options.csv) {
            std::cout << "device,compute_capability,library,library_version,precision,direction,placement,logN,N,batch,warmup,repeat,plan_ms,kernel_ms,transforms_s,Gbutterfly_s,points_s,max_abs_error,relative_l2_error,correct\n";
            std::cout << '"' << properties.name << "\"," << properties.major << '.' << properties.minor << ',' << options.library
                      << ',' << library_version << ",fp32,forward,in-place," << options.log_n << ',' << n << ',' << options.batch << ',' << options.warmup << ','
                      << options.repeat << ',' << result.plan_ms << ',' << result.kernel_ms << ',' << transforms_s << ','
                      << butterflies_s / 1.0e9 << ',' << transforms_s * n << ',' << result.error.max_abs << ','
                      << result.error.relative_l2 << ',' << (options.verify ? static_cast<int>(correct) : -1) << '\n';
        } else {
            std::cout << "device: " << properties.name << " (sm_" << properties.major << properties.minor << ")\n"
                      << "library: " << options.library << "\nlogN: " << options.log_n << "\nbatch: " << options.batch
                      << "\nplan_ms: " << result.plan_ms << "\nkernel_ms: " << result.kernel_ms
                      << "\nrelative_l2_error: " << result.error.relative_l2 << '\n';
        }
        return options.verify && !correct ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
