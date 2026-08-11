#include "direct_fft.hpp"

#include "cuntt/butterfly.hpp"

#include <cufft.h>

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuntt::detail {
namespace {

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

template <typename Complex>
__global__ void scale_kernel(Complex* values, std::size_t count, double scale) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    values[index].real = static_cast<decltype(values[index].real)>(values[index].real * scale);
    values[index].imag = static_cast<decltype(values[index].imag)>(values[index].imag * scale);
}

unsigned int blocks_for(std::size_t count) {
    const auto blocks = (count + 255) / 256;
    if (blocks > UINT_MAX)
        throw std::overflow_error("direct FFT normalization grid exceeds grid.x");
    return static_cast<unsigned int>(blocks);
}

}  // namespace

class DirectFftPlan::Impl {
  public:
    Impl(const std::vector<std::size_t>& extents, std::size_t batch,
         bool inverse, bool normalize_inverse, bool fp64)
        : inverse_(inverse), normalize_inverse_(normalize_inverse), fp64_(fp64), batch_(batch) {
        if (extents.empty() || extents.size() > 2 || batch == 0 || batch > static_cast<std::size_t>(INT_MAX))
            throw std::invalid_argument("invalid direct FFT shape");
        std::vector<int> dimensions(extents.size());
        points_ = 1;
        for (std::size_t axis = 0; axis < extents.size(); ++axis) {
            if (extents[axis] == 0 || extents[axis] > static_cast<std::size_t>(INT_MAX) ||
                points_ > static_cast<std::size_t>(INT_MAX) / extents[axis])
                throw std::invalid_argument("direct FFT extent exceeds cuFFT limits");
            dimensions[axis] = static_cast<int>(extents[axis]);
            points_ *= extents[axis];
        }
        check_cufft(cufftPlanMany(&plan_, static_cast<int>(dimensions.size()), dimensions.data(),
                                  nullptr, 1, static_cast<int>(points_), nullptr, 1,
                                  static_cast<int>(points_), fp64_ ? CUFFT_Z2Z : CUFFT_C2C,
                                  static_cast<int>(batch_)), "create direct FFT plan");
    }

    ~Impl() {
        if (plan_ != 0)
            cufftDestroy(plan_);
    }

    void execute(const void* input, void* output, cudaStream_t stream) {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("direct FFT input and output must be non-null");
        check_cufft(cufftSetStream(plan_, stream), "set direct FFT stream");
        const int direction = inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD;
        if (fp64_)
            check_cufft(cufftExecZ2Z(plan_, reinterpret_cast<cufftDoubleComplex*>(const_cast<void*>(input)),
                                     static_cast<cufftDoubleComplex*>(output), direction),
                        "execute direct FP64 FFT");
        else
            check_cufft(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(const_cast<void*>(input)),
                                     static_cast<cufftComplex*>(output), direction),
                        "execute direct FP32 FFT");
        if (inverse_ && normalize_inverse_) {
            const std::size_t count = points_ * batch_;
            const double scale = 1.0 / static_cast<double>(points_);
            if (fp64_)
                scale_kernel<<<blocks_for(count), 256, 0, stream>>>(
                    static_cast<Complex64*>(output), count, scale);
            else
                scale_kernel<<<blocks_for(count), 256, 0, stream>>>(
                    static_cast<Complex32*>(output), count, scale);
            const auto status = cudaGetLastError();
            if (status != cudaSuccess)
                throw std::runtime_error(std::string("launch direct FFT normalization: ") +
                                         cudaGetErrorString(status));
        }
    }

  private:
    cufftHandle plan_ = 0;
    bool inverse_ = false;
    bool normalize_inverse_ = false;
    bool fp64_ = false;
    std::size_t batch_ = 0;
    std::size_t points_ = 0;
};

DirectFftPlan::DirectFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                             bool inverse, bool normalize_inverse, bool fp64)
    : impl_(std::make_unique<Impl>(extents, batch, inverse, normalize_inverse, fp64)) {}

DirectFftPlan::~DirectFftPlan() = default;

void DirectFftPlan::execute(const void* input, void* output, cudaStream_t stream) {
    impl_->execute(input, output, stream);
}

}  // namespace cuntt::detail
