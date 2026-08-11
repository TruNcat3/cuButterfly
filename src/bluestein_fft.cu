#include "bluestein_fft.hpp"

#include "cuntt/butterfly.hpp"

#include <cufft.h>

#include <cmath>
#include <climits>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

template <typename Complex>
struct ComplexOps;

template <>
struct ComplexOps<Complex32> {
    __host__ __device__ static Complex32 make(double real, double imag) {
        return {static_cast<float>(real), static_cast<float>(imag)};
    }
};

template <>
struct ComplexOps<Complex64> {
    __host__ __device__ static Complex64 make(double real, double imag) { return {real, imag}; }
};

template <typename Complex>
__global__ void bluestein_pack_kernel(const Complex* input, Complex* work,
                                      std::size_t n, std::size_t m, std::size_t batch,
                                      std::size_t input_stride, std::size_t input_batch_stride,
                                      double sign) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / n;
    const std::size_t index = linear - transform * n;
    const auto value = input[transform * input_batch_stride + index * input_stride];
    const double angle = sign * 3.141592653589793238462643383279502884 *
                         static_cast<double>(index) * static_cast<double>(index) / static_cast<double>(n);
    const double c = cos(angle);
    const double s = sin(angle);
    work[transform * m + index] = ComplexOps<Complex>::make(value.real * c - value.imag * s,
                                                            value.real * s + value.imag * c);
}

template <typename Complex>
__global__ void bluestein_multiply_kernel(Complex* work, const Complex* kernel,
                                          std::size_t m, std::size_t batch) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= m * batch)
        return;
    const auto left = work[linear];
    const auto right = kernel[linear % m];
    work[linear] = ComplexOps<Complex>::make(left.real * right.real - left.imag * right.imag,
                                             left.real * right.imag + left.imag * right.real);
}

template <typename Complex>
__global__ void bluestein_output_kernel(const Complex* work, Complex* output,
                                        std::size_t n, std::size_t m, std::size_t batch,
                                        std::size_t output_stride, std::size_t output_batch_stride,
                                        double sign, double scale) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t count = n * batch;
    if (linear >= count)
        return;
    const std::size_t transform = linear / n;
    const std::size_t index = linear - transform * n;
    const auto value = work[transform * m + index];
    const double angle = sign * 3.141592653589793238462643383279502884 *
                         static_cast<double>(index) * static_cast<double>(index) / static_cast<double>(n);
    const double c = cos(angle) * scale;
    const double s = sin(angle) * scale;
    output[transform * output_batch_stride + index * output_stride] =
        ComplexOps<Complex>::make(value.real * c - value.imag * s,
                                  value.real * s + value.imag * c);
}

unsigned int blocks_for(std::size_t count) {
    const std::size_t blocks = (count + 255) / 256;
    if (blocks > UINT32_MAX)
        throw std::overflow_error("Bluestein CUDA grid exceeds grid.x");
    return static_cast<unsigned int>(blocks);
}

}  // namespace

class BluesteinFftPlan::Impl {
  public:
    Impl(std::size_t n, std::size_t m, std::size_t batch, bool inverse,
         bool normalize_inverse, bool fp64)
        : n_(n), m_(m), batch_(batch), inverse_(inverse),
          normalize_inverse_(normalize_inverse), fp64_(fp64) {
        if (n_ == 0 || m_ < 2 * n_ - 1 || batch_ == 0 || m_ > static_cast<std::size_t>(INT32_MAX))
            throw std::invalid_argument("invalid Bluestein shape");
        int length = static_cast<int>(m_);
        check_cufft(cufftPlanMany(&plan_, 1, &length, nullptr, 1, length, nullptr, 1, length,
                                  fp64_ ? CUFFT_Z2Z : CUFFT_C2C, static_cast<int>(batch_)),
                    "create Bluestein batch plan");
        try {
            build_kernel();
        } catch (...) {
            cufftDestroy(plan_);
            plan_ = 0;
            cudaFree(kernel_fft_);
            kernel_fft_ = nullptr;
            throw;
        }
    }

    ~Impl() {
        if (kernel_fft_ != nullptr)
            cudaFree(kernel_fft_);
        if (plan_ != 0)
            cufftDestroy(plan_);
    }

    std::size_t workspace_size() const noexcept {
        return m_ * batch_ * (fp64_ ? sizeof(Complex64) : sizeof(Complex32));
    }

    void execute(const void* input, void* output, void* workspace,
                 std::size_t input_stride, std::size_t input_batch_stride,
                 std::size_t output_stride, std::size_t output_batch_stride,
                 cudaStream_t stream) {
        if (input == nullptr || output == nullptr || workspace == nullptr)
            throw std::invalid_argument("Bluestein input, output, and workspace must be non-null");
        check_cufft(cufftSetStream(plan_, stream), "set Bluestein stream");
        check_cuda(cudaMemsetAsync(workspace, 0, workspace_size(), stream), "clear Bluestein workspace");
        if (fp64_)
            execute_typed(static_cast<const Complex64*>(input), static_cast<Complex64*>(output),
                          static_cast<Complex64*>(workspace), input_stride, input_batch_stride,
                          output_stride, output_batch_stride, stream);
        else
            execute_typed(static_cast<const Complex32*>(input), static_cast<Complex32*>(output),
                          static_cast<Complex32*>(workspace), input_stride, input_batch_stride,
                          output_stride, output_batch_stride, stream);
    }

  private:
    template <typename Complex>
    void execute_typed(const Complex* input, Complex* output, Complex* workspace,
                       std::size_t input_stride, std::size_t input_batch_stride,
                       std::size_t output_stride, std::size_t output_batch_stride,
                       cudaStream_t stream) {
        const double sign = inverse_ ? 1.0 : -1.0;
        bluestein_pack_kernel<<<blocks_for(n_ * batch_), 256, 0, stream>>>(
            input, workspace, n_, m_, batch_, input_stride, input_batch_stride, sign);
        check_cuda(cudaGetLastError(), "launch Bluestein pack");
        if constexpr (std::is_same_v<Complex, Complex64>)
            check_cufft(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(workspace),
                                     reinterpret_cast<cufftDoubleComplex*>(workspace), CUFFT_FORWARD),
                        "execute Bluestein forward FFT");
        else
            check_cufft(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(workspace),
                                     reinterpret_cast<cufftComplex*>(workspace), CUFFT_FORWARD),
                        "execute Bluestein forward FFT");
        bluestein_multiply_kernel<<<blocks_for(m_ * batch_), 256, 0, stream>>>(
            workspace, static_cast<const Complex*>(kernel_fft_), m_, batch_);
        check_cuda(cudaGetLastError(), "launch Bluestein pointwise multiply");
        if constexpr (std::is_same_v<Complex, Complex64>)
            check_cufft(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(workspace),
                                     reinterpret_cast<cufftDoubleComplex*>(workspace), CUFFT_INVERSE),
                        "execute Bluestein inverse FFT");
        else
            check_cufft(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(workspace),
                                     reinterpret_cast<cufftComplex*>(workspace), CUFFT_INVERSE),
                        "execute Bluestein inverse FFT");
        const double scale = 1.0 / static_cast<double>(m_) *
                             (inverse_ && normalize_inverse_ ? 1.0 / static_cast<double>(n_) : 1.0);
        bluestein_output_kernel<<<blocks_for(n_ * batch_), 256, 0, stream>>>(
            workspace, output, n_, m_, batch_, output_stride, output_batch_stride, sign, scale);
        check_cuda(cudaGetLastError(), "launch Bluestein output");
    }

    template <typename Complex>
    void build_kernel_typed() {
        std::vector<Complex> values(m_, ComplexOps<Complex>::make(0.0, 0.0));
        const double sign = inverse_ ? 1.0 : -1.0;
        for (std::size_t index = 0; index < n_; ++index) {
            const double angle = -sign * 3.141592653589793238462643383279502884 *
                                 static_cast<double>(index) * static_cast<double>(index) / static_cast<double>(n_);
            const auto value = ComplexOps<Complex>::make(std::cos(angle), std::sin(angle));
            values[index] = value;
            if (index != 0)
                values[m_ - index] = value;
        }
        const std::size_t bytes = m_ * sizeof(Complex);
        void* temporary = nullptr;
        cufftHandle kernel_plan = 0;
        check_cuda(cudaMalloc(&temporary, bytes), "allocate Bluestein kernel temporary");
        try {
            check_cuda(cudaMalloc(&kernel_fft_, bytes), "allocate Bluestein kernel FFT");
            check_cuda(cudaMemcpy(temporary, values.data(), bytes, cudaMemcpyHostToDevice), "upload Bluestein kernel");
            int length = static_cast<int>(m_);
            check_cufft(cufftPlan1d(&kernel_plan, length,
                                    fp64_ ? CUFFT_Z2Z : CUFFT_C2C, 1), "create Bluestein kernel plan");
            if constexpr (std::is_same_v<Complex, Complex64>)
                check_cufft(cufftExecZ2Z(kernel_plan, static_cast<cufftDoubleComplex*>(temporary),
                                         static_cast<cufftDoubleComplex*>(kernel_fft_), CUFFT_FORWARD),
                            "transform Bluestein kernel");
            else
                check_cufft(cufftExecC2C(kernel_plan, static_cast<cufftComplex*>(temporary),
                                         static_cast<cufftComplex*>(kernel_fft_), CUFFT_FORWARD),
                            "transform Bluestein kernel");
            check_cuda(cudaDeviceSynchronize(), "wait for Bluestein kernel initialization");
            cufftDestroy(kernel_plan);
            cudaFree(temporary);
        } catch (...) {
            if (kernel_plan != 0)
                cufftDestroy(kernel_plan);
            cudaFree(temporary);
            throw;
        }
    }

    void build_kernel() {
        if (fp64_)
            build_kernel_typed<Complex64>();
        else
            build_kernel_typed<Complex32>();
    }

    std::size_t n_;
    std::size_t m_;
    std::size_t batch_;
    bool inverse_;
    bool normalize_inverse_;
    bool fp64_;
    cufftHandle plan_ = 0;
    void* kernel_fft_ = nullptr;
};

BluesteinFftPlan::BluesteinFftPlan(std::size_t length, std::size_t convolution_length,
                                   std::size_t batch, bool inverse, bool normalize_inverse,
                                   bool fp64)
    : impl_(std::make_unique<Impl>(length, convolution_length, batch, inverse,
                                   normalize_inverse, fp64)) {}

BluesteinFftPlan::~BluesteinFftPlan() = default;

std::size_t BluesteinFftPlan::workspace_size() const noexcept { return impl_->workspace_size(); }

void BluesteinFftPlan::execute(const void* input, void* output, void* workspace,
                               std::size_t input_stride, std::size_t input_batch_stride,
                               std::size_t output_stride, std::size_t output_batch_stride,
                               cudaStream_t stream) {
    impl_->execute(input, output, workspace, input_stride, input_batch_stride,
                   output_stride, output_batch_stride, stream);
}

}  // namespace cuntt::detail
