#include "bluestein_ntt.hpp"

#include "cuntt/ntt.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuntt::detail {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

std::uint64_t shoup_host(std::uint64_t value, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(value) << 64) / modulus);
}

std::vector<std::uint64_t> prime_factors(std::uint64_t value) {
    std::vector<std::uint64_t> factors;
    for (std::uint64_t factor = 2; factor <= value / factor; ++factor) {
        if (value % factor != 0)
            continue;
        factors.push_back(factor);
        while (value % factor == 0)
            value /= factor;
    }
    if (value > 1)
        factors.push_back(value);
    return factors;
}

std::uint64_t find_root(std::uint64_t order, std::uint64_t modulus) {
    if (!cuntt::is_prime(modulus))
        throw std::invalid_argument("Bluestein NTT modulus must be prime");
    if (order < 2 || (modulus - 1) % order != 0)
        throw std::invalid_argument("modulus does not admit the required 2N-th Bluestein root");
    const auto factors = prime_factors(order);
    for (std::uint64_t candidate = 2; candidate < (1ULL << 20); ++candidate) {
        const auto root = cuntt::mod_pow(candidate, (modulus - 1) / order, modulus);
        bool primitive = cuntt::mod_pow(root, order, modulus) == 1;
        for (const auto factor : factors)
            primitive = primitive && cuntt::mod_pow(root, order / factor, modulus) != 1;
        if (primitive)
            return root;
    }
    throw std::runtime_error("could not find the required Bluestein NTT root");
}

__device__ __forceinline__ std::uint64_t mul_shoup(std::uint64_t value,
                                                    std::uint64_t multiplier,
                                                    std::uint64_t multiplier_shoup,
                                                    std::uint64_t modulus) {
    const std::uint64_t quotient = __umul64hi(value, multiplier_shoup);
    std::uint64_t reduced = value * multiplier - quotient * modulus;
    if (reduced >= modulus)
        reduced -= modulus;
    return reduced;
}

__global__ void pack_kernel(const std::uint64_t* input, std::uint64_t* work,
                            const std::uint64_t* chirp, const std::uint64_t* chirp_shoup,
                            std::size_t n, std::size_t m, std::size_t batch,
                            std::size_t input_stride, std::size_t input_batch_stride,
                            std::uint64_t modulus) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= n * batch)
        return;
    const std::size_t transform = linear / n;
    const std::size_t index = linear % n;
    const auto value = input[transform * input_batch_stride + index * input_stride] % modulus;
    work[transform * m + index] = mul_shoup(value, chirp[index], chirp_shoup[index], modulus);
}

__global__ void multiply_kernel(std::uint64_t* values, const std::uint64_t* kernel,
                                const std::uint64_t* kernel_shoup, std::size_t m,
                                std::size_t batch, std::uint64_t modulus) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= m * batch)
        return;
    const std::size_t index = linear % m;
    values[linear] = mul_shoup(values[linear], kernel[index], kernel_shoup[index], modulus);
}

__global__ void output_kernel(const std::uint64_t* work, std::uint64_t* output,
                              const std::uint64_t* chirp, const std::uint64_t* chirp_shoup,
                              std::size_t n, std::size_t m, std::size_t batch,
                              std::size_t output_stride, std::size_t output_batch_stride,
                              std::uint64_t scale, std::uint64_t scale_shoup,
                              std::uint64_t modulus) {
    const std::size_t linear = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear >= n * batch)
        return;
    const std::size_t transform = linear / n;
    const std::size_t index = linear % n;
    auto value = mul_shoup(work[transform * m + index], chirp[index], chirp_shoup[index], modulus);
    value = mul_shoup(value, scale, scale_shoup, modulus);
    output[transform * output_batch_stride + index * output_stride] = value;
}

unsigned int blocks_for(std::size_t count) {
    const auto blocks = (count + 255) / 256;
    if (blocks > UINT_MAX)
        throw std::overflow_error("Bluestein NTT CUDA grid exceeds grid.x");
    return static_cast<unsigned int>(blocks);
}

std::size_t align_16(std::size_t value) { return (value + 15) & ~std::size_t{15}; }

}  // namespace

class BluesteinNttPlan::Impl {
  public:
    Impl(std::size_t n, std::size_t m, std::size_t batch, std::uint64_t modulus, bool inverse)
        : Impl(n, m, batch, modulus, inverse, cuntt::Backend::Baseline,
               cuntt::ComputeUnit::Radix2) {}

    Impl(std::size_t n, std::size_t m, std::size_t batch, std::uint64_t modulus,
         bool inverse, cuntt::Backend core_backend, cuntt::ComputeUnit core_compute_unit)
        : n_(n), m_(m), batch_(batch), modulus_(modulus), inverse_(inverse),
          core_backend_(core_backend), core_compute_unit_(core_compute_unit),
          forward_(core_config(m, batch, modulus, false, core_backend, core_compute_unit)),
          inverse_core_(core_config(m, batch, modulus, true, core_backend, core_compute_unit)) {
        if (n_ < 2 || m_ < 2 * n_ - 1 || batch_ == 0)
            throw std::invalid_argument("invalid Bluestein NTT shape");
        if ((modulus_ - 1) % m_ != 0)
            throw std::invalid_argument("modulus does not admit the power-of-two Bluestein convolution root");
        try {
            build_tables();
        } catch (...) {
            cudaFree(kernel_shoup_);
            cudaFree(kernel_fft_);
            cudaFree(chirp_shoup_);
            cudaFree(chirp_);
            kernel_shoup_ = nullptr;
            kernel_fft_ = nullptr;
            chirp_shoup_ = nullptr;
            chirp_ = nullptr;
            throw;
        }
    }

    ~Impl() {
        cudaFree(kernel_shoup_);
        cudaFree(kernel_fft_);
        cudaFree(chirp_shoup_);
        cudaFree(chirp_);
    }

    std::size_t workspace_size() const noexcept {
        return 2 * data_bytes() + align_16(std::max(forward_.workspace_size(), inverse_core_.workspace_size()));
    }

    void execute(const std::uint64_t* input, std::uint64_t* output, void* workspace,
                 std::size_t input_stride, std::size_t input_batch_stride,
                 std::size_t output_stride, std::size_t output_batch_stride,
                 cudaStream_t stream) {
        if (input == nullptr || output == nullptr || workspace == nullptr)
            throw std::invalid_argument("Bluestein NTT input, output, and workspace must be non-null");
        auto* first = static_cast<std::uint64_t*>(workspace);
        auto* second = reinterpret_cast<std::uint64_t*>(static_cast<unsigned char*>(workspace) + data_bytes());
        auto* core_workspace = static_cast<unsigned char*>(workspace) + 2 * data_bytes();
        const auto core_bytes = std::max(forward_.workspace_size(), inverse_core_.workspace_size());
        forward_.set_stream(stream);
        inverse_core_.set_stream(stream);
        if (core_bytes != 0) {
            forward_.set_workspace(core_workspace, core_bytes);
            inverse_core_.set_workspace(core_workspace, core_bytes);
        }
        check_cuda(cudaMemsetAsync(first, 0, data_bytes(), stream), "clear Bluestein NTT input");
        pack_kernel<<<blocks_for(n_ * batch_), 256, 0, stream>>>(
            input, first, chirp_, chirp_shoup_, n_, m_, batch_, input_stride,
            input_batch_stride, modulus_);
        check_cuda(cudaGetLastError(), "launch Bluestein NTT pack");
        forward_.execute_async(first, second);
        multiply_kernel<<<blocks_for(m_ * batch_), 256, 0, stream>>>(
            second, kernel_fft_, kernel_shoup_, m_, batch_, modulus_);
        check_cuda(cudaGetLastError(), "launch Bluestein NTT multiply");
        inverse_core_.execute_async(second, first);
        output_kernel<<<blocks_for(n_ * batch_), 256, 0, stream>>>(
            first, output, chirp_, chirp_shoup_, n_, m_, batch_, output_stride,
            output_batch_stride, scale_, scale_shoup_, modulus_);
        check_cuda(cudaGetLastError(), "launch Bluestein NTT output");
    }

  private:
    static cuntt::PlanConfig core_config(std::size_t m, std::size_t batch,
                                         std::uint64_t modulus, bool inverse,
                                         cuntt::Backend backend,
                                         cuntt::ComputeUnit compute_unit) {
        if (m == 0 || (m & (m - 1)) != 0)
            throw std::invalid_argument("Bluestein NTT convolution length must be a power of two");
        std::uint32_t log_m = 0;
        for (auto value = m; value > 1; value >>= 1)
            ++log_m;
        cuntt::PlanConfig config;
        config.log_n = log_m;
        config.batch = batch;
        config.modulus = modulus;
        config.inverse = inverse;
        config.backend = backend;
        config.compute_unit = compute_unit;
        config.word_bits = 64;
        config.auto_allocate_workspace = false;
        return config;
    }

    std::size_t data_bytes() const noexcept { return m_ * batch_ * sizeof(std::uint64_t); }

    void build_tables() {
        auto root = find_root(2 * n_, modulus_);
        if (inverse_)
            root = cuntt::mod_pow(root, modulus_ - 2, modulus_);
        std::vector<std::uint64_t> chirp(n_);
        std::vector<std::uint64_t> chirp_shoup(n_);
        std::vector<std::uint64_t> kernel(m_, 0);
        for (std::size_t index = 0; index < n_; ++index) {
            const auto exponent = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(index) * index) % (2 * n_));
            chirp[index] = cuntt::mod_pow(root, exponent, modulus_);
            chirp_shoup[index] = shoup_host(chirp[index], modulus_);
            const auto inverse_chirp = cuntt::mod_pow(chirp[index], modulus_ - 2, modulus_);
            kernel[index] = inverse_chirp;
            if (index != 0)
                kernel[m_ - index] = inverse_chirp;
        }
        const auto chirp_bytes = n_ * sizeof(std::uint64_t);
        const auto kernel_bytes = m_ * sizeof(std::uint64_t);
        check_cuda(cudaMalloc(&chirp_, chirp_bytes), "allocate Bluestein NTT chirp");
        check_cuda(cudaMalloc(&chirp_shoup_, chirp_bytes), "allocate Bluestein NTT chirp Shoup table");
        check_cuda(cudaMalloc(&kernel_fft_, kernel_bytes), "allocate Bluestein NTT kernel");
        check_cuda(cudaMalloc(&kernel_shoup_, kernel_bytes), "allocate Bluestein NTT kernel Shoup table");
        std::uint64_t* temporary = nullptr;
        check_cuda(cudaMalloc(&temporary, kernel_bytes), "allocate Bluestein NTT initialization buffer");
        try {
            check_cuda(cudaMemcpy(chirp_, chirp.data(), chirp_bytes, cudaMemcpyHostToDevice), "upload Bluestein NTT chirp");
            check_cuda(cudaMemcpy(chirp_shoup_, chirp_shoup.data(), chirp_bytes, cudaMemcpyHostToDevice), "upload Bluestein NTT chirp Shoup table");
            check_cuda(cudaMemcpy(temporary, kernel.data(), kernel_bytes, cudaMemcpyHostToDevice), "upload Bluestein NTT kernel");
            auto init_config = core_config(m_, 1, modulus_, false, core_backend_, core_compute_unit_);
            init_config.auto_allocate_workspace = true;
            cuntt::Plan init_plan(init_config);
            init_plan.execute_async(temporary, kernel_fft_);
            check_cuda(cudaDeviceSynchronize(), "initialize Bluestein NTT kernel transform");
            std::vector<std::uint64_t> transformed(m_);
            check_cuda(cudaMemcpy(transformed.data(), kernel_fft_, kernel_bytes, cudaMemcpyDeviceToHost),
                       "download Bluestein NTT kernel transform");
            std::vector<std::uint64_t> transformed_shoup(m_);
            std::transform(transformed.begin(), transformed.end(), transformed_shoup.begin(),
                           [&](std::uint64_t value) { return shoup_host(value, modulus_); });
            check_cuda(cudaMemcpy(kernel_shoup_, transformed_shoup.data(), kernel_bytes, cudaMemcpyHostToDevice),
                       "upload Bluestein NTT kernel Shoup table");
            cudaFree(temporary);
        } catch (...) {
            cudaFree(temporary);
            throw;
        }
        scale_ = inverse_ ? cuntt::mod_pow(static_cast<std::uint64_t>(n_), modulus_ - 2, modulus_) : 1;
        scale_shoup_ = shoup_host(scale_, modulus_);
    }

    std::size_t n_;
    std::size_t m_;
    std::size_t batch_;
    std::uint64_t modulus_;
    bool inverse_;
    cuntt::Backend core_backend_;
    cuntt::ComputeUnit core_compute_unit_;
    cuntt::Plan forward_;
    cuntt::Plan inverse_core_;
    std::uint64_t* chirp_ = nullptr;
    std::uint64_t* chirp_shoup_ = nullptr;
    std::uint64_t* kernel_fft_ = nullptr;
    std::uint64_t* kernel_shoup_ = nullptr;
    std::uint64_t scale_ = 1;
    std::uint64_t scale_shoup_ = 0;
};

BluesteinNttPlan::BluesteinNttPlan(std::size_t length, std::size_t convolution_length,
                                   std::size_t batch, std::uint64_t modulus, bool inverse,
                                   cuntt::Backend core_backend,
                                   cuntt::ComputeUnit core_compute_unit)
    : impl_(std::make_unique<Impl>(length, convolution_length, batch, modulus, inverse,
                                   core_backend, core_compute_unit)) {}

BluesteinNttPlan::~BluesteinNttPlan() = default;

std::size_t BluesteinNttPlan::workspace_size() const noexcept { return impl_->workspace_size(); }

void BluesteinNttPlan::execute(const std::uint64_t* input, std::uint64_t* output,
                               void* workspace, std::size_t input_stride,
                               std::size_t input_batch_stride, std::size_t output_stride,
                               std::size_t output_batch_stride, cudaStream_t stream) {
    impl_->execute(input, output, workspace, input_stride, input_batch_stride,
                   output_stride, output_batch_stride, stream);
}

}  // namespace cuntt::detail
