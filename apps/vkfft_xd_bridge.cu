#include <cuda.h>
#include <cuda_runtime_api.h>

#include <vkFFT.h>

#include <cstdint>
#include <limits>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kOk = 0;
constexpr int kInvalidArgument = -1;
constexpr int kVkfftError = -2;
constexpr int kException = -4;
constexpr int kNullPlan = -5;

thread_local std::string g_last_error;

void set_error(const std::string& message) {
    g_last_error = message;
}

std::string cuda_error(cudaError_t status, const char* expression) {
    std::ostringstream message;
    message << expression << " failed with CUDA status " << static_cast<int>(status)
            << ": " << cudaGetErrorString(status);
    return message.str();
}

std::string driver_error(CUresult status, const char* expression) {
    const char* name = nullptr;
    const char* detail = nullptr;
    cuGetErrorName(status, &name);
    cuGetErrorString(status, &detail);
    std::ostringstream message;
    message << expression << " failed with driver status " << static_cast<int>(status);
    if (name != nullptr) {
        message << " (" << name << ")";
    }
    if (detail != nullptr) {
        message << ": " << detail;
    }
    return message.str();
}

std::string vkfft_error(VkFFTResult status, const char* expression) {
    std::ostringstream message;
    message << expression << " failed with VkFFT status " << static_cast<int>(status);
    return message.str();
}

struct VkfftXdPlan {
    VkFFTApplication application{};
    VkFFTLaunchParams launch{};
    VkFFTConfiguration configuration{};

    pfUINT sizes[VKFFT_MAX_FFT_DIMENSIONS]{};
    pfUINT buffer_size = 0;
    pfUINT input_buffer_size = 0;
    void* buffer = nullptr;
    void* input_buffer = nullptr;
    cudaStream_t stream = nullptr;
    CUdevice device{};

    std::uint32_t rank = 0;
    std::uint64_t batch = 0;
    bool fp64 = false;
    bool inverse = false;
    bool initialized = false;
};

void validate_shape(std::uint32_t rank, const std::uint64_t* shape,
                    std::uint64_t batch, bool fp64, bool inverse,
                    std::uintptr_t stream, std::uintptr_t input,
                    std::uintptr_t output, std::uint64_t* points,
                    std::uint64_t* bytes) {
    if (shape == nullptr || points == nullptr || bytes == nullptr) {
        throw std::invalid_argument("shape and output arguments must be non-null");
    }
    if (rank < 1 || rank > 3) {
        throw std::invalid_argument("rank must be in [1, 3]");
    }
    if (batch == 0) {
        throw std::invalid_argument("batch must be positive");
    }
    if (fp64 != false && fp64 != true) {
        throw std::invalid_argument("fp64 must be 0 or 1");
    }
    if (inverse != false && inverse != true) {
        throw std::invalid_argument("inverse must be 0 or 1");
    }
    if (input == 0 || output == 0) {
        throw std::invalid_argument("input and output pointers must be non-null");
    }
    if (input == output) {
        throw std::invalid_argument("input and output pointers must be distinct");
    }

    std::uint64_t element_count = 1;
    for (std::uint32_t axis = 0; axis < rank; ++axis) {
        if (shape[axis] == 0) {
            throw std::invalid_argument("FFT dimensions must be positive");
        }
        if (element_count > std::numeric_limits<std::uint64_t>::max() / shape[axis]) {
            throw std::overflow_error("FFT element count overflows uint64_t");
        }
        element_count *= shape[axis];
    }
    if (element_count > std::numeric_limits<std::uint64_t>::max() / batch) {
        throw std::overflow_error("batched FFT element count overflows uint64_t");
    }
    element_count *= batch;
    const std::uint64_t scalar_bytes = fp64 ? sizeof(double) : sizeof(float);
    if (element_count > std::numeric_limits<std::uint64_t>::max() / (2 * scalar_bytes)) {
        throw std::overflow_error("FFT buffer size overflows uint64_t");
    }
    *points = element_count;
    *bytes = element_count * (2 * scalar_bytes);
    (void)stream;
}

void initialize_plan(VkfftXdPlan* plan, std::uint32_t rank,
                     const std::uint64_t* shape, std::uint64_t batch,
                     bool fp64, bool inverse, std::uintptr_t stream,
                     std::uintptr_t input, std::uintptr_t output,
                     std::uint64_t bytes) {
    plan->rank = rank;
    plan->batch = batch;
    plan->fp64 = fp64;
    plan->inverse = inverse;
    plan->buffer = reinterpret_cast<void*>(output);
    plan->input_buffer = reinterpret_cast<void*>(input);
    plan->stream = reinterpret_cast<cudaStream_t>(stream);
    plan->buffer_size = static_cast<pfUINT>(bytes);
    plan->input_buffer_size = static_cast<pfUINT>(bytes);

    for (std::uint32_t axis = 0; axis < rank; ++axis) {
        // VkFFT uses WHD order, with size[0] as the contiguous axis.
        plan->sizes[axis] = static_cast<pfUINT>(shape[rank - 1 - axis]);
    }

    CUresult driver_status = cuInit(0);
    if (driver_status != CUDA_SUCCESS) {
        throw std::runtime_error(driver_error(driver_status, "cuInit"));
    }
    int device_ordinal = 0;
    cudaError_t runtime_status = cudaGetDevice(&device_ordinal);
    if (runtime_status != cudaSuccess) {
        throw std::runtime_error(cuda_error(runtime_status, "cudaGetDevice"));
    }
    driver_status = cuDeviceGet(&plan->device, device_ordinal);
    if (driver_status != CUDA_SUCCESS) {
        throw std::runtime_error(driver_error(driver_status, "cuDeviceGet"));
    }

    plan->configuration.FFTdim = rank;
    for (std::uint32_t axis = 0; axis < VKFFT_MAX_FFT_DIMENSIONS; ++axis) {
        plan->configuration.size[axis] = plan->sizes[axis];
    }
    plan->configuration.device = &plan->device;
    plan->configuration.stream = &plan->stream;
    plan->configuration.num_streams = 1;
    plan->configuration.numberBatches = static_cast<pfUINT>(batch);
    plan->configuration.doublePrecision = fp64 ? 1 : 0;
    plan->configuration.normalize = inverse ? 1 : 0;
    plan->configuration.makeForwardPlanOnly = inverse ? 0 : 1;
    plan->configuration.makeInversePlanOnly = inverse ? 1 : 0;

    // The main buffer is the output. inputBuffer is read-only and is never
    // written, which gives the Python wrapper strict out-of-place semantics.
    plan->configuration.buffer = &plan->buffer;
    plan->configuration.bufferSize = &plan->buffer_size;
    plan->configuration.isInputFormatted = 1;
    plan->configuration.inputBuffer = &plan->input_buffer;
    plan->configuration.inputBufferSize = &plan->input_buffer_size;

    VkFFTResult status = initializeVkFFT(&plan->application, plan->configuration);
    if (status != VKFFT_SUCCESS) {
        throw std::runtime_error(vkfft_error(status, "initializeVkFFT"));
    }
    plan->initialized = true;

    // Keep launch pointers explicit so the fixed Torch buffers are used on
    // every call and remain valid for CUDA graph capture.
    plan->launch.buffer = &plan->buffer;
    plan->launch.inputBuffer = &plan->input_buffer;
}

void destroy_plan(VkfftXdPlan* plan) noexcept {
    if (plan == nullptr) {
        return;
    }
    if (plan->initialized) {
        deleteVkFFT(&plan->application);
        plan->initialized = false;
    }
    delete plan;
}

}  // namespace

extern "C" {

int vkfft_xd_create(void** opaque_plan, std::uint32_t rank,
                    const std::uint64_t* shape, std::uint64_t batch,
                    int fp64, int inverse, std::uintptr_t stream,
                    std::uintptr_t input, std::uintptr_t output) noexcept {
    if (opaque_plan == nullptr) {
        set_error("opaque_plan must be non-null");
        return kInvalidArgument;
    }
    *opaque_plan = nullptr;
    try {
        if (fp64 != 0 && fp64 != 1) {
            throw std::invalid_argument("fp64 must be 0 or 1");
        }
        if (inverse != 0 && inverse != 1) {
            throw std::invalid_argument("inverse must be 0 or 1");
        }
        std::uint64_t points = 0;
        std::uint64_t bytes = 0;
        validate_shape(rank, shape, batch, fp64 != 0, inverse != 0, stream,
                       input, output, &points, &bytes);
        (void)points;
        auto* plan = new VkfftXdPlan();
        try {
            initialize_plan(plan, rank, shape, batch, fp64 != 0, inverse != 0,
                             stream, input, output, bytes);
        } catch (...) {
            destroy_plan(plan);
            throw;
        }
        *opaque_plan = plan;
        g_last_error.clear();
        return kOk;
    } catch (const std::invalid_argument& error) {
        set_error(error.what());
        return kInvalidArgument;
    } catch (const std::overflow_error& error) {
        set_error(error.what());
        return kInvalidArgument;
    } catch (const std::bad_alloc& error) {
        set_error(error.what());
        return kException;
    } catch (const std::exception& error) {
        set_error(error.what());
        return kException;
    } catch (...) {
        set_error("unknown exception in vkfft_xd_create");
        return kException;
    }
}

int vkfft_xd_execute(void* opaque_plan) noexcept {
    if (opaque_plan == nullptr) {
        set_error("plan must be non-null");
        return kNullPlan;
    }
    try {
        auto* plan = static_cast<VkfftXdPlan*>(opaque_plan);
        const VkFFTResult status = VkFFTAppend(
            &plan->application, plan->inverse ? 1 : -1, &plan->launch);
        if (status != VKFFT_SUCCESS) {
            set_error(vkfft_error(status, "VkFFTAppend"));
            return kVkfftError;
        }
        g_last_error.clear();
        return kOk;
    } catch (const std::exception& error) {
        set_error(error.what());
        return kException;
    } catch (...) {
        set_error("unknown exception in vkfft_xd_execute");
        return kException;
    }
}

int vkfft_xd_destroy(void* opaque_plan) noexcept {
    if (opaque_plan == nullptr) {
        g_last_error.clear();
        return kOk;
    }
    try {
        destroy_plan(static_cast<VkfftXdPlan*>(opaque_plan));
        g_last_error.clear();
        return kOk;
    } catch (const std::exception& error) {
        set_error(error.what());
        return kException;
    } catch (...) {
        set_error("unknown exception in vkfft_xd_destroy");
        return kException;
    }
}

int vkfft_xd_version() noexcept {
    try {
        return VkFFTGetVersion();
    } catch (...) {
        return 0;
    }
}

const char* vkfft_xd_last_error() noexcept {
    return g_last_error.c_str();
}

}  // extern "C"
