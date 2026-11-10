#include "direct_real_fft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cuntt::detail {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

std::size_t checked_product(std::size_t left, std::size_t right, const char* what) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error(std::string(what) + " overflows size_t");
    return left * right;
}

int checked_int(std::size_t value, const char* what) {
    if (value == 0 || value > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument(std::string(what) + " exceeds cuFFT int plan limit");
    return static_cast<int>(value);
}

unsigned blocks_for(std::size_t count, const char* what) {
    const auto blocks = (count + 255) / 256;
    if (blocks > static_cast<std::size_t>(UINT_MAX))
        throw std::overflow_error(std::string(what) + " grid exceeds CUDA grid.x");
    return static_cast<unsigned>(blocks);
}

template <typename Complex, typename Real>
__global__ void scale_complex_kernel(Complex* values, std::size_t count, Real scale) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index].x *= scale;
        values[index].y *= scale;
    }
}

template <typename Complex, typename Real>
__global__ void copy_scale_complex_kernel(const Complex* input, Complex* output,
                                          std::size_t count, Real scale) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        Complex value = input[index];
        value.x *= scale;
        value.y *= scale;
        output[index] = value;
    }
}

}  // namespace

class DirectRealFftPlan::Impl {
  public:
    Impl(const std::vector<std::size_t>& extents, std::size_t batch,
         bool inverse, bool normalize_inverse, bool fp64, double output_scale)
        : extents_(extents), batch_(batch), inverse_(inverse),
          normalize_inverse_(normalize_inverse), fp64_(fp64),
          output_scale_(output_scale) {
        if (extents_.empty() || extents_.size() > 3 || batch_ == 0 ||
            batch_ > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("invalid complete real FFT shape");
        if (!std::isfinite(output_scale_))
            throw std::invalid_argument("complete real FFT output scale must be finite");

        points_ = 1;
        for (std::size_t axis = 0; axis < extents_.size(); ++axis) {
            dimensions_.push_back(checked_int(extents_[axis], "real FFT extent"));
            points_ = checked_product(points_, extents_[axis], "real FFT point count");
        }
        compact_points_ = points_ / extents_.back() * (extents_.back() / 2 + 1);
        if (points_ > static_cast<std::size_t>(INT_MAX) ||
            compact_points_ > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("complete real FFT batch distance exceeds cuFFT int plan limit");
        if (inverse_)
            workspace_bytes_ = checked_product(
                checked_product(compact_points_, batch_, "complete C2R workspace elements"),
                fp64_ ? sizeof(cufftDoubleComplex) : sizeof(cufftComplex),
                "complete C2R workspace bytes");

        inembed_ = dimensions_;
        onembed_ = dimensions_;
        if (inverse_) {
            inembed_.back() = checked_int(extents_.back() / 2 + 1, "compact real FFT extent");
            idist_ = checked_int(compact_points_, "compact real FFT batch distance");
            odist_ = checked_int(points_, "real FFT batch distance");
        } else {
            onembed_.back() = checked_int(extents_.back() / 2 + 1, "compact real FFT extent");
            idist_ = checked_int(points_, "real FFT batch distance");
            odist_ = checked_int(compact_points_, "compact real FFT batch distance");
        }
        const auto type = fp64_ ? (inverse_ ? CUFFT_Z2D : CUFFT_D2Z)
                                : (inverse_ ? CUFFT_C2R : CUFFT_R2C);
        check_cufft(cufftPlanMany(&plan_, static_cast<int>(dimensions_.size()), dimensions_.data(),
                                  inembed_.data(), 1, idist_, onembed_.data(), 1, odist_,
                                  type, static_cast<int>(batch_)),
                    "create complete real FFT plan");
    }

    ~Impl() {
        if (plan_ != 0) cufftDestroy(plan_);
    }

    void execute(const void* input, void* output, cudaStream_t stream) {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("complete real FFT input and output must be non-null");
        if (stream != bound_stream_) {
            check_cufft(cufftSetStream(plan_, stream), "set complete real FFT stream");
            bound_stream_ = stream;
        }
        const double scale = output_scale_ *
            (inverse_ && normalize_inverse_ ? 1.0 / static_cast<double>(points_) : 1.0);
        const void* transform_input = input;
        if (inverse_) {
            if (workspace_ == nullptr)
                throw std::invalid_argument("complete C2R requires caller workspace for input preservation");
            const auto count = checked_product(compact_points_, batch_, "complete C2R input count");
            if (scale == 1.0) {
                check_cuda(cudaMemcpyAsync(workspace_, input, workspace_bytes_, cudaMemcpyDeviceToDevice, stream),
                           "copy complete C2R input");
            } else if (fp64_) {
                copy_scale_complex_kernel<cufftDoubleComplex, double>
                    <<<blocks_for(count, "complete FP64 C2R input scale"), 256, 0, stream>>>(
                        static_cast<const cufftDoubleComplex*>(input),
                        static_cast<cufftDoubleComplex*>(workspace_), count, scale);
                check_cuda(cudaGetLastError(), "launch complete FP64 C2R input scale");
            } else {
                copy_scale_complex_kernel<cufftComplex, float>
                    <<<blocks_for(count, "complete FP32 C2R input scale"), 256, 0, stream>>>(
                        static_cast<const cufftComplex*>(input),
                        static_cast<cufftComplex*>(workspace_), count, static_cast<float>(scale));
                check_cuda(cudaGetLastError(), "launch complete FP32 C2R input scale");
            }
            transform_input = workspace_;
        }
        if (fp64_) {
            if (inverse_)
                check_cufft(cufftExecZ2D(plan_, reinterpret_cast<cufftDoubleComplex*>(const_cast<void*>(transform_input)),
                                         static_cast<double*>(output)), "execute complete FP64 C2R");
            else
                check_cufft(cufftExecD2Z(plan_, const_cast<double*>(static_cast<const double*>(input)),
                                         reinterpret_cast<cufftDoubleComplex*>(output)), "execute complete FP64 R2C");
        } else {
            if (inverse_)
                check_cufft(cufftExecC2R(plan_, reinterpret_cast<cufftComplex*>(const_cast<void*>(transform_input)),
                                         static_cast<float*>(output)), "execute complete FP32 C2R");
            else
                check_cufft(cufftExecR2C(plan_, const_cast<float*>(static_cast<const float*>(input)),
                                         reinterpret_cast<cufftComplex*>(output)), "execute complete FP32 R2C");
        }
        // C2R normalization was folded into the protected input copy above,
        // so no second output traversal is needed for inverse transforms.
        if (scale == 1.0 || inverse_) return;
        const auto count = checked_product(compact_points_, batch_, "compact FFT output count");
        if (fp64_)
            scale_complex_kernel<<<blocks_for(count, "complete FP64 R2C scale"), 256, 0, stream>>>(
                reinterpret_cast<cufftDoubleComplex*>(output), count, static_cast<double>(scale));
        else
            scale_complex_kernel<<<blocks_for(count, "complete FP32 R2C scale"), 256, 0, stream>>>(
                reinterpret_cast<cufftComplex*>(output), count, static_cast<float>(scale));
        check_cuda(cudaGetLastError(), "launch complete real FFT scale");
    }

    std::string mapping_json() const {
        std::ostringstream json;
        json << "{\"schema_version\":1,\"kind\":\"xd-vendor-whole-real-fft\""
             << ",\"rank\":" << extents_.size() << ",\"batch\":" << batch_
             << ",\"inverse\":" << (inverse_ ? "true" : "false")
             << ",\"normalize_inverse\":" << (normalize_inverse_ ? "true" : "false")
             << ",\"fp64\":" << (fp64_ ? "true" : "false")
             << ",\"output_scale\":" << std::setprecision(17) << output_scale_
             << ",\"workspace_bytes\":" << workspace_size()
             << ",\"extents\":[";
        for (std::size_t i = 0; i < extents_.size(); ++i)
            json << (i ? "," : "") << extents_[i];
        json << "],\"boundary_launches\":0}";
        return json.str();
    }

  private:
    std::vector<std::size_t> extents_;
    std::vector<int> dimensions_;
    std::vector<int> inembed_;
    std::vector<int> onembed_;
    std::size_t batch_ = 0;
    std::size_t points_ = 0;
    std::size_t compact_points_ = 0;
    std::size_t workspace_bytes_ = 0;
    bool inverse_ = false;
    bool normalize_inverse_ = false;
    bool fp64_ = false;
    double output_scale_ = 1.0;
    int idist_ = 0;
    int odist_ = 0;
    cufftHandle plan_ = 0;
    cudaStream_t bound_stream_ = nullptr;
    void* workspace_ = nullptr;

  public:
    std::size_t workspace_size() const noexcept {
        return workspace_bytes_;
    }

    void set_workspace(void* workspace, std::size_t bytes) {
        const auto required = workspace_size();
        if (required == 0) {
            if (bytes != 0)
                throw std::invalid_argument("complete R2C does not accept workspace bytes");
            workspace_ = workspace;
            return;
        }
        if (workspace == nullptr && bytes == 0) {
            workspace_ = nullptr;
            return;
        }
        if (workspace == nullptr || bytes < required)
            throw std::invalid_argument("complete C2R workspace is too small");
        workspace_ = workspace;
    }
};

DirectRealFftPlan::DirectRealFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                                     bool inverse, bool normalize_inverse, bool fp64,
                                     double output_scale)
    : impl_(std::make_unique<Impl>(extents, batch, inverse, normalize_inverse, fp64, output_scale)) {}

DirectRealFftPlan::~DirectRealFftPlan() = default;

void DirectRealFftPlan::execute(const void* input, void* output, cudaStream_t stream) {
    impl_->execute(input, output, stream);
}

std::size_t DirectRealFftPlan::workspace_size() const noexcept { return impl_->workspace_size(); }

void DirectRealFftPlan::set_workspace(void* workspace, std::size_t bytes) {
    impl_->set_workspace(workspace, bytes);
}

std::string DirectRealFftPlan::mapping_json() const { return impl_->mapping_json(); }

}  // namespace cuntt::detail
