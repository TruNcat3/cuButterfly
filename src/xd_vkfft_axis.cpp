#include "xd_vkfft_axis.hpp"

#include <cuda.h>
#include <vkFFT.h>

#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace cuntt::detail {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("xD VkFFT axis byte count overflows size_t");
    return a * b;
}

void check_vkfft(VkFFTResult status, const char* operation) {
    if (status != VKFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + getVkFFTErrorString(status));
}

}  // namespace

class XdVkfftAxisPlan::Impl {
  public:
    Impl(std::size_t length, std::size_t suffix, std::size_t lines,
         bool fp64, bool inverse)
        : length_(length), suffix_(suffix), lines_(lines), fp64_(fp64), inverse_(inverse) {
        if (length < 2 || (length & (length - 1)) != 0)
            throw std::invalid_argument("xD VkFFT axis requires a power-of-two length >= 2");
        if (suffix == 0 || lines == 0 || lines % suffix != 0)
            throw std::invalid_argument("xD VkFFT axis requires positive lines divisible by suffix");
        bytes_ = checked_product(checked_product(length, lines), fp64 ? 16U : 8U);
        if (bytes_ > std::numeric_limits<pfUINT>::max())
            throw std::overflow_error("xD VkFFT axis byte count exceeds VkFFT index range");
        if (cuInit(0) != CUDA_SUCCESS || cudaGetDevice(&device_ordinal_) != cudaSuccess ||
            cuDeviceGet(&device_, device_ordinal_) != CUDA_SUCCESS)
            throw std::runtime_error("initialize xD VkFFT CUDA device");

        VkFFTConfiguration configuration{};
        configuration.FFTdim = suffix == 1 ? 1 : 2;
        configuration.size[0] = suffix == 1 ? length : suffix;
        if (suffix != 1) {
            configuration.size[1] = length;
            configuration.omitDimension[0] = 1;
        }
        configuration.numberBatches = lines / suffix;
        configuration.device = &device_;
        configuration.stream = &stream_;
        configuration.num_streams = 1;
        configuration.doublePrecision = fp64 ? 1 : 0;
        configuration.normalize = 0;
        configuration.makeForwardPlanOnly = inverse ? 0 : 1;
        configuration.makeInversePlanOnly = inverse ? 1 : 0;
        configuration.buffer = &output_;
        configuration.bufferSize = &bytes_;
        configuration.isInputFormatted = 1;
        configuration.inputBuffer = &input_;
        configuration.inputBufferSize = &bytes_;
        check_vkfft(initializeVkFFT(&application_, configuration), "initialize xD VkFFT axis");
        initialized_ = true;
        try {
            collect_resources();
        } catch (...) {
            deleteVkFFT(&application_);
            initialized_ = false;
            throw;
        }
    }

    ~Impl() {
        if (initialized_) deleteVkFFT(&application_);
    }

    void execute(const void* input, void* output, cudaStream_t stream) {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("xD VkFFT axis requires non-null buffers");
        input_ = const_cast<void*>(input);
        output_ = output;
        stream_ = stream;
        VkFFTLaunchParams launch{};
        launch.buffer = &output_;
        launch.inputBuffer = &input_;
        check_vkfft(VkFFTAppend(&application_, inverse_ ? 1 : -1, &launch),
                    "execute xD VkFFT axis");
    }

    void collect_resources() {
        const auto& configuration = application_.configuration;
        if (configuration.allocateTempBuffer && configuration.tempBufferSize)
            for (pfUINT i = 0; i < configuration.tempBufferNum; ++i)
                temp_bytes_ += static_cast<std::size_t>(configuration.tempBufferSize[i]);
        // VkFFT may share a LUT across uploads. Count each device allocation once.
        std::unordered_map<void*, std::size_t> tables;
        const auto* plan = inverse_ ? application_.localFFTPlan_inverse : application_.localFFTPlan;
        if (plan)
            for (pfUINT axis = 0; axis < configuration.FFTdim; ++axis)
                for (pfUINT upload = 0; upload < plan->numAxisUploads[axis]; ++upload) {
                    const auto& kernel = plan->axes[axis][upload];
                    if (kernel.bufferLUT)
                        tables[kernel.bufferLUT] = static_cast<std::size_t>(kernel.bufferLUTSize);
                    if (kernel.bufferRaderUintLUT)
                        tables[kernel.bufferRaderUintLUT] = static_cast<std::size_t>(kernel.bufferRaderUintLUTSize);
                }
        for (const auto& entry : tables) table_bytes_ += entry.second;
    }

    std::string metadata_json() const {
        std::ostringstream json;
        json << "{\"provider\":\"vkfft-axis\",\"vkfft_version\":" << VkFFTGetVersion()
             << ",\"length\":" << length_ << ",\"suffix\":" << suffix_
             << ",\"lines\":" << lines_ << ",\"prefix\":" << lines_ / suffix_
             << ",\"fft_dimension\":" << (suffix_ == 1 ? 1 : 2)
             << ",\"fp64\":" << (fp64_ ? "true" : "false")
             << ",\"inverse\":" << (inverse_ ? "true" : "false")
             << ",\"normalize\":false,\"supports_scale\":false"
             << ",\"caller_workspace_bytes\":0,\"owned_temp_bytes\":" << temp_bytes_
             << ",\"owned_lookup_table_bytes\":" << table_bytes_
             << ",\"owned_workspace_bytes\":" << temp_bytes_ + table_bytes_ << "}";
        return json.str();
    }

    VkFFTApplication application_{};
    CUdevice device_{};
    int device_ordinal_ = 0;
    cudaStream_t stream_ = nullptr;
    void* input_ = nullptr;
    void* output_ = nullptr;
    pfUINT bytes_ = 0;
    std::size_t length_ = 0;
    std::size_t suffix_ = 0;
    std::size_t lines_ = 0;
    std::size_t temp_bytes_ = 0;
    std::size_t table_bytes_ = 0;
    bool fp64_ = false;
    bool inverse_ = false;
    bool initialized_ = false;
};

bool xd_vkfft_axis_available() noexcept { return true; }

XdVkfftAxisPlan::XdVkfftAxisPlan(std::size_t length, std::size_t suffix, std::size_t lines,
                               bool fp64, bool inverse)
    : impl_(std::make_unique<Impl>(length, suffix, lines, fp64, inverse)) {}
XdVkfftAxisPlan::~XdVkfftAxisPlan() = default;
std::size_t XdVkfftAxisPlan::owned_workspace_size() const noexcept {
    return impl_->temp_bytes_ + impl_->table_bytes_;
}
void XdVkfftAxisPlan::execute(const void* input, void* output, cudaStream_t stream) {
    impl_->execute(input, output, stream);
}
std::string XdVkfftAxisPlan::metadata_json() const { return impl_->metadata_json(); }

}  // namespace cuntt::detail
