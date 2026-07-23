#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "cuntt/butterfly.hpp"
#include "generated_unit_api.cuh"
#include "stage_pipeline.cuh"

namespace cuntt {
namespace {

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

void check_cufft(cufftResult status, const char* expression, const char* file, int line) {
    if (status == CUFFT_SUCCESS) {
        return;
    }
    std::ostringstream message;
    message << expression << " failed at " << file << ':' << line << " with cuFFT status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

#define CUB_CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
#define CUB_CUFFT_CHECK(expr) check_cufft((expr), #expr, __FILE__, __LINE__)

class Event {
  public:
    Event() { CUB_CUDA_CHECK(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

class DeviceBuffer {
  public:
    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }
    void allocate(std::size_t bytes) { CUB_CUDA_CHECK(cudaMalloc(&pointer_, bytes)); }
    template <typename T>
    T* as() const noexcept {
        return static_cast<T*>(pointer_);
    }

  private:
    void* pointer_ = nullptr;
};

template <typename Real>
struct FwhtOperator {
    using Value                            = Real;
    static constexpr bool kBitReverseInput = false;

    bool inverse;
    bool normalize_inverse;

    __device__ __forceinline__ void apply(std::uint32_t, std::uint32_t, Value& left, Value& right) const {
        const Real a = left;
        const Real b = right;
        left         = a + b;
        right        = a - b;
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const { return __shfl_xor_sync(0xffffffffU, value, mask); }

    __device__ __forceinline__ Value apply_lane(std::uint32_t, std::uint32_t, Value self, Value partner, bool right_lane) const {
        return right_lane ? partner - self : self + partner;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        return inverse && normalize_inverse ? value / static_cast<Real>(n) : value;
    }
};

template <typename Complex, typename Real>
struct FftOperator {
    using Value                            = Complex;
    static constexpr bool kBitReverseInput = true;

    const Complex*  twiddles;
    bool            inverse;
    bool            normalize_inverse;
    ComplexMultiply multiply;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const Complex root = twiddles[(1U << stage) - 1 + offset];
        Real          vr;
        Real          vi;
        if (multiply == ComplexMultiply::Gauss3) {
            const Real p0 = right.real * root.real;
            const Real p1 = right.imag * root.imag;
            const Real p2 = (right.real + right.imag) * (root.real + root.imag);
            vr            = p0 - p1;
            vi            = p2 - p0 - p1;
        } else {
            vr = right.real * root.real - right.imag * root.imag;
            vi = right.real * root.imag + right.imag * root.real;
        }
        const Real lr = left.real;
        const Real li = left.imag;
        left          = {lr + vr, li + vi};
        right         = {lr - vr, li - vi};
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const {
        return {__shfl_xor_sync(0xffffffffU, value.real, mask), __shfl_xor_sync(0xffffffffU, value.imag, mask)};
    }

    __device__ __forceinline__ Value apply_lane(std::uint32_t stage, std::uint32_t offset, Value self, Value partner, bool right_lane) const {
        const Complex left  = right_lane ? partner : self;
        const Complex right = right_lane ? self : partner;
        const Complex root  = twiddles[(1U << stage) - 1 + offset];
        Real          vr;
        Real          vi;
        if (multiply == ComplexMultiply::Gauss3) {
            const Real p0 = right.real * root.real;
            const Real p1 = right.imag * root.imag;
            const Real p2 = (right.real + right.imag) * (root.real + root.imag);
            vr            = p0 - p1;
            vi            = p2 - p0 - p1;
        } else {
            vr = right.real * root.real - right.imag * root.imag;
            vi = right.real * root.imag + right.imag * root.real;
        }
        return right_lane ? Value{left.real - vr, left.imag - vi} : Value{left.real + vr, left.imag + vi};
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t n) const {
        if (!inverse || !normalize_inverse) {
            return value;
        }
        const Real scale = Real{1} / static_cast<Real>(n);
        return {value.real * scale, value.imag * scale};
    }
};

struct XorZetaOperator {
    using Value                            = std::uint32_t;
    static constexpr bool kBitReverseInput = false;

    bool inverse;

    __device__ __forceinline__ void apply(std::uint32_t, std::uint32_t, Value& left, Value& right) const {
        right = inverse ? right - left : right + left;
    }

    __device__ __forceinline__ Value shuffle(Value value, std::uint32_t mask) const { return __shfl_xor_sync(0xffffffffU, value, mask); }

    __device__ __forceinline__ Value apply_lane(std::uint32_t, std::uint32_t, Value self, Value partner, bool right_lane) const {
        return right_lane ? (inverse ? self - partner : partner + self) : self;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const { return value; }
};

template <typename Complex, typename Real>
__global__ void scale_inverse_fft_kernel(Complex* values, std::uint64_t count, Real scale) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index].real *= scale;
        values[index].imag *= scale;
    }
}

template <typename Operator, std::uint32_t StageSpace, bool NamedBarrier, std::uint32_t PipelineWarps>
void launch_stage_pipeline(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    constexpr std::uint32_t kPipelines         = PipelineWarps / StageSpace;
    constexpr std::uint32_t kTokensPerPipeline = 2;
    const std::uint64_t     tiles              = config.batch;
    const auto              blocks = static_cast<std::uint32_t>((tiles + kPipelines * kTokensPerPipeline - 1) / (kPipelines * kTokensPerPipeline));
    for (std::uint32_t stage_base = 0; stage_base < 8; stage_base += StageSpace) {
        const auto*           stage_input     = stage_base == 0 ? input : output;
        constexpr std::size_t kHandoffPadding = NamedBarrier ? PipelineWarps * 2 * sizeof(int) : 0;
        detail::stage_pipeline_256_kernel<Operator, StageSpace, NamedBarrier, PipelineWarps><<<blocks, PipelineWarps * 32, kHandoffPadding>>>(
            stage_input, output, tiles, stage_base, config.batch_stride, config.batch_stride, config.element_stride, op);
        CUB_CUDA_CHECK(cudaGetLastError());
    }
}

template <typename Operator, bool NamedBarrier, std::uint32_t PipelineWarps>
void dispatch_stage_space(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    switch (config.stage_space) {
        case 1:
            launch_stage_pipeline<Operator, 1, NamedBarrier, PipelineWarps>(config, input, output, op);
            break;
        case 2:
            launch_stage_pipeline<Operator, 2, NamedBarrier, PipelineWarps>(config, input, output, op);
            break;
        case 4:
            launch_stage_pipeline<Operator, 4, NamedBarrier, PipelineWarps>(config, input, output, op);
            break;
        case 8:
            if constexpr (PipelineWarps == 8) {
                launch_stage_pipeline<Operator, 8, NamedBarrier, PipelineWarps>(config, input, output, op);
            } else {
                throw std::logic_error("Us=8 requires eight pipeline warps");
            }
            break;
        default:
            throw std::logic_error("unresolved butterfly stage_space");
    }
}

template <typename Operator, bool NamedBarrier>
void dispatch_pipeline_warps(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    if (config.pipeline_warps == 4) {
        dispatch_stage_space<Operator, NamedBarrier, 4>(config, input, output, op);
    } else {
        dispatch_stage_space<Operator, NamedBarrier, 8>(config, input, output, op);
    }
}

template <typename Operator, std::uint32_t LogN>
void launch_temporal_tile(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    constexpr std::size_t shared_bytes = (std::size_t{1} << LogN) * sizeof(typename Operator::Value);
    if (config.compute_unit == ComputeUnit::Radix8) {
        detail::hierarchical_prefix_kernel<Operator, LogN, false, false, true, true>
            <<<static_cast<unsigned int>(config.batch), config.tile_threads, shared_bytes>>>(input, output, config.batch, config.log_n,
                                                                                             config.batch_stride, config.element_stride, op);
    } else if (config.compute_unit == ComputeUnit::Radix4) {
        detail::temporal_tile_radix4_kernel<Operator, LogN><<<static_cast<unsigned int>(config.batch), config.tile_threads, shared_bytes>>>(
            input, output, config.batch, config.batch_stride, config.element_stride, op);
    } else {
        detail::temporal_tile_kernel<Operator, LogN><<<static_cast<unsigned int>(config.batch), config.tile_threads, shared_bytes>>>(
            input, output, config.batch, config.batch_stride, config.element_stride, op);
    }
    CUB_CUDA_CHECK(cudaGetLastError());
}

template <typename Operator>
void dispatch_temporal_length(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    switch (config.log_n) {
        case 1:
            launch_temporal_tile<Operator, 1>(config, input, output, op);
            break;
        case 2:
            launch_temporal_tile<Operator, 2>(config, input, output, op);
            break;
        case 3:
            launch_temporal_tile<Operator, 3>(config, input, output, op);
            break;
        case 4:
            launch_temporal_tile<Operator, 4>(config, input, output, op);
            break;
        case 5:
            launch_temporal_tile<Operator, 5>(config, input, output, op);
            break;
        case 6:
            launch_temporal_tile<Operator, 6>(config, input, output, op);
            break;
        case 7:
            launch_temporal_tile<Operator, 7>(config, input, output, op);
            break;
        case 8:
            launch_temporal_tile<Operator, 8>(config, input, output, op);
            break;
        case 9:
            launch_temporal_tile<Operator, 9>(config, input, output, op);
            break;
        case 10:
            launch_temporal_tile<Operator, 10>(config, input, output, op);
            break;
        default:
            throw std::logic_error("unresolved temporal butterfly length");
    }
}

template <typename Operator, std::uint32_t LocalLogN>
void launch_hierarchical_prefix(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
    constexpr std::size_t shared_bytes = (std::size_t{1} << LocalLogN) * sizeof(typename Operator::Value);
    const std::uint64_t   tiles        = static_cast<std::uint64_t>(config.batch) << (config.log_n - LocalLogN);
    if (config.compute_unit == ComputeUnit::Radix8) {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, false, false, true>
            <<<static_cast<unsigned int>(tiles), config.tile_threads, shared_bytes>>>(input, output, config.batch, config.log_n, config.batch_stride,
                                                                                      config.element_stride, op);
    } else if (config.compute_unit == ComputeUnit::Radix4) {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, true><<<static_cast<unsigned int>(tiles), config.tile_threads, shared_bytes>>>(
            input, output, config.batch, config.log_n, config.batch_stride, config.element_stride, op);
    } else {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, false><<<static_cast<unsigned int>(tiles), config.tile_threads, shared_bytes>>>(
            input, output, config.batch, config.log_n, config.batch_stride, config.element_stride, op);
    }
    CUB_CUDA_CHECK(cudaGetLastError());
}

template <typename Operator>
void dispatch_hierarchical_prefix(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output,
                                  Operator op) {
    switch (config.local_stages) {
        case 5:
            launch_hierarchical_prefix<Operator, 5>(config, input, output, op);
            break;
        case 6:
            launch_hierarchical_prefix<Operator, 6>(config, input, output, op);
            break;
        case 7:
            launch_hierarchical_prefix<Operator, 7>(config, input, output, op);
            break;
        case 8:
            launch_hierarchical_prefix<Operator, 8>(config, input, output, op);
            break;
        case 9:
            launch_hierarchical_prefix<Operator, 9>(config, input, output, op);
            break;
        case 10:
            launch_hierarchical_prefix<Operator, 10>(config, input, output, op);
            break;
        default:
            throw std::logic_error("unresolved hierarchical local_stages");
    }
}

template <typename Operator, std::uint32_t LocalLogN, std::uint32_t RemainingLogN>
void launch_online_reorder(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output,
                           typename Operator::Value* scratch, Operator op) {
    constexpr std::size_t prefix_shared_bytes = (std::size_t{1} << LocalLogN) * sizeof(typename Operator::Value);
    const std::size_t     suffix_shared_bytes = (std::size_t{1} << RemainingLogN) * config.reorder_columns * sizeof(typename Operator::Value);
    const std::uint64_t   prefix_tiles        = static_cast<std::uint64_t>(config.batch) << RemainingLogN;
    const std::uint32_t   local_n             = 1U << LocalLogN;
    const std::uint32_t   column_tiles        = (local_n + config.reorder_columns - 1) / config.reorder_columns;
    const std::uint64_t   suffix_tiles        = static_cast<std::uint64_t>(config.batch) * column_tiles;
    if (config.compute_unit == ComputeUnit::Radix8) {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, false, true, true>
            <<<static_cast<unsigned int>(prefix_tiles), config.tile_threads, prefix_shared_bytes>>>(input, scratch, config.batch, config.log_n,
                                                                                                    config.batch_stride, config.element_stride, op);
        CUB_CUDA_CHECK(cudaGetLastError());
        detail::online_reorder_suffix_kernel<Operator, RemainingLogN, false, true>
            <<<static_cast<unsigned int>(suffix_tiles), config.tile_threads, suffix_shared_bytes>>>(
                scratch, output, config.batch, config.log_n, LocalLogN, config.batch_stride, config.element_stride, config.reorder_columns, op);
    } else if (config.compute_unit == ComputeUnit::Radix4) {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, true, true>
            <<<static_cast<unsigned int>(prefix_tiles), config.tile_threads, prefix_shared_bytes>>>(input, scratch, config.batch, config.log_n,
                                                                                                    config.batch_stride, config.element_stride, op);
        CUB_CUDA_CHECK(cudaGetLastError());
        detail::online_reorder_suffix_kernel<Operator, RemainingLogN, true>
            <<<static_cast<unsigned int>(suffix_tiles), config.tile_threads, suffix_shared_bytes>>>(
                scratch, output, config.batch, config.log_n, LocalLogN, config.batch_stride, config.element_stride, config.reorder_columns, op);
    } else {
        detail::hierarchical_prefix_kernel<Operator, LocalLogN, false, true>
            <<<static_cast<unsigned int>(prefix_tiles), config.tile_threads, prefix_shared_bytes>>>(input, scratch, config.batch, config.log_n,
                                                                                                    config.batch_stride, config.element_stride, op);
        CUB_CUDA_CHECK(cudaGetLastError());
        detail::online_reorder_suffix_kernel<Operator, RemainingLogN, false>
            <<<static_cast<unsigned int>(suffix_tiles), config.tile_threads, suffix_shared_bytes>>>(
                scratch, output, config.batch, config.log_n, LocalLogN, config.batch_stride, config.element_stride, config.reorder_columns, op);
    }
    CUB_CUDA_CHECK(cudaGetLastError());
}

template <typename Operator, std::uint32_t LocalLogN>
void dispatch_online_reorder_remaining(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output,
                                       typename Operator::Value* scratch, Operator op) {
    switch (config.log_n - LocalLogN) {
        case 1:
            launch_online_reorder<Operator, LocalLogN, 1>(config, input, output, scratch, op);
            break;
        case 2:
            launch_online_reorder<Operator, LocalLogN, 2>(config, input, output, scratch, op);
            break;
        case 3:
            launch_online_reorder<Operator, LocalLogN, 3>(config, input, output, scratch, op);
            break;
        case 4:
            launch_online_reorder<Operator, LocalLogN, 4>(config, input, output, scratch, op);
            break;
        case 5:
            launch_online_reorder<Operator, LocalLogN, 5>(config, input, output, scratch, op);
            break;
        case 6:
            launch_online_reorder<Operator, LocalLogN, 6>(config, input, output, scratch, op);
            break;
        case 7:
            launch_online_reorder<Operator, LocalLogN, 7>(config, input, output, scratch, op);
            break;
        case 8:
            launch_online_reorder<Operator, LocalLogN, 8>(config, input, output, scratch, op);
            break;
        case 9:
            launch_online_reorder<Operator, LocalLogN, 9>(config, input, output, scratch, op);
            break;
        case 10:
            launch_online_reorder<Operator, LocalLogN, 10>(config, input, output, scratch, op);
            break;
        default:
            throw std::logic_error("unresolved online-reorder remaining stages");
    }
}

template <typename Operator>
void dispatch_online_reorder(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* output,
                             typename Operator::Value* scratch, Operator op) {
    switch (config.local_stages) {
        case 5:
            dispatch_online_reorder_remaining<Operator, 5>(config, input, output, scratch, op);
            break;
        case 6:
            dispatch_online_reorder_remaining<Operator, 6>(config, input, output, scratch, op);
            break;
        case 7:
            dispatch_online_reorder_remaining<Operator, 7>(config, input, output, scratch, op);
            break;
        case 8:
            dispatch_online_reorder_remaining<Operator, 8>(config, input, output, scratch, op);
            break;
        case 9:
            dispatch_online_reorder_remaining<Operator, 9>(config, input, output, scratch, op);
            break;
        case 10:
            dispatch_online_reorder_remaining<Operator, 10>(config, input, output, scratch, op);
            break;
        default:
            throw std::logic_error("unresolved online-reorder local_stages");
    }
}

template <typename Operator>
void launch_hierarchical(const ButterflyConfig& config, const typename Operator::Value* input, typename Operator::Value* desired_output,
                         typename Operator::Value* scratch, std::size_t data_bytes, Operator op) {
    using Value                          = typename Operator::Value;
    const std::uint32_t remaining_stages = config.log_n - config.local_stages;
    Value*              prefix_output    = config.placement == ButterflyPlacement::OutOfPlace && remaining_stages % 2 == 0 ? desired_output : scratch;
    dispatch_hierarchical_prefix(config, input, prefix_output, op);

    Value*                  source            = prefix_output;
    constexpr std::uint32_t kThreads          = 256;
    const std::uint64_t     total_butterflies = static_cast<std::uint64_t>(config.batch) << (config.log_n - 1);
    const auto              blocks            = static_cast<unsigned int>((total_butterflies + kThreads - 1) / kThreads);
    for (std::uint32_t stage = config.local_stages; stage < config.log_n; ++stage) {
        Value* destination = source == scratch ? desired_output : scratch;
        detail::hierarchical_stage_kernel<Operator><<<blocks, kThreads>>>(source, destination, config.batch, config.log_n, stage, config.batch_stride,
                                                                          config.element_stride, stage + 1 == config.log_n, op);
        CUB_CUDA_CHECK(cudaGetLastError());
        source = destination;
    }
    if (source != desired_output) {
        CUB_CUDA_CHECK(cudaMemcpyAsync(desired_output, source, data_bytes, cudaMemcpyDeviceToDevice));
    }
}

}  // namespace

class ButterflyPlan::Impl {
  public:
    explicit Impl(ButterflyConfig config) : config_(std::move(config)) {
        if (config_.log_n == 0 || config_.log_n > 20) {
            throw std::invalid_argument("butterfly log_n must be in [1, 20]");
        }
        if (config_.backend == ButterflyBackend::TemporalTile && config_.local_exchange == LocalExchange::SharedMemory && config_.log_n > 10) {
            throw std::invalid_argument("temporal-tile requires log_n in [1, 10]");
        }
        if (config_.local_exchange == LocalExchange::WarpRegister) {
            if (config_.backend != ButterflyBackend::TemporalTile || config_.op != ButterflyOperator::Fwht ||
                config_.precision != ButterflyPrecision::Fp32 || !detail::generated_register_fwht_available(config_.log_n)) {
                throw std::invalid_argument("warp-register exchange requires a generated temporal-tile FP32 FWHT design point");
            }
            config_.tile_threads = detail::generated_register_fwht_threads(config_.log_n);
        }
        if (config_.backend == ButterflyBackend::Hierarchical &&
            (config_.local_stages < 5 || config_.local_stages > 10 || config_.local_stages >= config_.log_n)) {
            throw std::invalid_argument("hierarchical requires local_stages in [5, 10] and smaller than log_n");
        }
        if (config_.backend == ButterflyBackend::OnlineReorder &&
            (config_.local_stages < 5 || config_.local_stages > 10 || config_.local_stages >= config_.log_n ||
             config_.log_n - config_.local_stages > 10)) {
            throw std::invalid_argument("online-reorder requires local and remaining stages in [1, 10], with local_stages >= 5");
        }
        if (config_.backend == ButterflyBackend::OnlineReorder) {
            const std::uint32_t local_n     = 1U << config_.local_stages;
            const std::uint32_t remaining_n = 1U << (config_.log_n - config_.local_stages);
            if (config_.reorder_columns == 0) {
                config_.reorder_columns = std::min(local_n, 1024U / remaining_n);
            }
            if (config_.reorder_columns == 0 || config_.reorder_columns > local_n || config_.reorder_columns * remaining_n > 1024 ||
                (config_.reorder_columns & (config_.reorder_columns - 1)) != 0) {
                throw std::invalid_argument("online-reorder columns must be a power of two and retain at most 1024 values per CTA");
            }
        }
        if (config_.batch == 0) {
            throw std::invalid_argument("butterfly batch must be positive");
        }
        if (config_.backend == ButterflyBackend::StagePipeline && config_.stage_space == 0) {
            config_.stage_space = 4;
        }
        if (config_.backend == ButterflyBackend::StagePipeline && config_.stage_space != 1 && config_.stage_space != 2 && config_.stage_space != 4 &&
            config_.stage_space != 8) {
            throw std::invalid_argument("butterfly stage_space must be 1, 2, 4, or 8");
        }
        if (config_.local_exchange == LocalExchange::SharedMemory &&
            (config_.backend == ButterflyBackend::TemporalTile || config_.backend == ButterflyBackend::Hierarchical ||
             config_.backend == ButterflyBackend::OnlineReorder) &&
            config_.tile_threads != 32 && config_.tile_threads != 64 && config_.tile_threads != 128 && config_.tile_threads != 256) {
            throw std::invalid_argument("butterfly tile_threads must be 32, 64, 128, or 256");
        }
        if (config_.backend == ButterflyBackend::WarpHybrid && config_.warp_stages > 5) {
            throw std::invalid_argument("butterfly warp_stages must be in [0, 5]");
        }
        if (config_.backend == ButterflyBackend::StagePipeline && config_.pipeline_warps != 4 && config_.pipeline_warps != 8) {
            throw std::invalid_argument("butterfly pipeline_warps must be 4 or 8");
        }
        if (config_.backend == ButterflyBackend::StagePipeline && config_.stage_space > config_.pipeline_warps) {
            throw std::invalid_argument("butterfly stage_space cannot exceed pipeline_warps");
        }
        if (config_.backend == ButterflyBackend::CuFft && config_.op != ButterflyOperator::Fft) {
            throw std::invalid_argument("cufft backend requires the FFT operator");
        }
        if (config_.op == ButterflyOperator::XorZeta) {
            config_.precision         = ButterflyPrecision::Uint32;
            config_.normalize_inverse = false;
        } else if (config_.precision == ButterflyPrecision::Uint32) {
            throw std::invalid_argument("uint32 precision requires xor-zeta");
        }
        if (config_.fft_core != FftCore::Scalar) {
            if (config_.op != ButterflyOperator::Fft || config_.backend != ButterflyBackend::TemporalTile ||
                config_.local_exchange != LocalExchange::SharedMemory) {
                throw std::invalid_argument("generated FFT codelets require temporal-tile FFT with shared exchange");
            }
            if (config_.fft_core == FftCore::ThreadDft8) {
                if (config_.precision != ButterflyPrecision::Fp32 || !detail::generated_thread_dft8_available(config_.log_n)) {
                    throw std::invalid_argument("thread-dft8 requires a generated FP32 design point");
                }
                config_.tile_threads = 128;
            } else if (config_.fft_core == FftCore::CtaDft8) {
                if (config_.precision != ButterflyPrecision::Fp32 ||
                    !detail::generated_cta_dft8_available(config_.log_n, config_.tile_threads)) {
                    throw std::invalid_argument("cta-dft8 requires a generated FP32 design point");
                }
            } else {
                if (config_.precision != ButterflyPrecision::Fp16Fp32 || !detail::generated_wmma_dft8_available(config_.log_n)) {
                    throw std::invalid_argument("wmma-dft8 requires a generated fp16-fp32 log_n=3 design point");
                }
                config_.tile_threads = 256;
            }
        } else if (config_.precision == ButterflyPrecision::Fp16Fp32) {
            throw std::invalid_argument("fp16-fp32 precision requires the wmma-dft8 FFT core");
        }
        if ((config_.backend == ButterflyBackend::WarpHybrid || config_.backend == ButterflyBackend::StagePipeline) && config_.log_n != 8) {
            throw std::invalid_argument("warp-hybrid and stage-pipeline currently require log_n=8");
        }
        if (config_.backend == ButterflyBackend::StagePipeline && config_.op == ButterflyOperator::Fft &&
            config_.precision == ButterflyPrecision::Fp64) {
            throw std::invalid_argument("FP64 complex stage-pipeline exceeds the V100 per-CTA shared-memory limit");
        }
        if (config_.compute_unit == ComputeUnit::Auto) {
            config_.compute_unit = config_.backend == ButterflyBackend::CuFft ? ComputeUnit::Auto : ComputeUnit::Radix2;
        }
        if (config_.fft_core != FftCore::Scalar) {
            config_.compute_unit = ComputeUnit::Radix8;
        }
        if (config_.local_exchange == LocalExchange::WarpRegister && config_.compute_unit != ComputeUnit::Radix2) {
            throw std::invalid_argument("warp-register exchange currently provides the radix2 arithmetic unit");
        }
        if ((config_.compute_unit == ComputeUnit::Radix4 || config_.compute_unit == ComputeUnit::Radix8) &&
            config_.backend != ButterflyBackend::TemporalTile && config_.backend != ButterflyBackend::Hierarchical &&
            config_.backend != ButterflyBackend::OnlineReorder) {
            throw std::invalid_argument("fused-radix butterfly processing units require temporal-tile, hierarchical, or online-reorder");
        }
        if (config_.backend == ButterflyBackend::CuFft) {
            config_.compute_unit = ComputeUnit::Auto;
            config_.fft_core     = FftCore::Scalar;
        }
        if (config_.complex_multiply != ComplexMultiply::FourMul &&
            (config_.op != ButterflyOperator::Fft || config_.backend == ButterflyBackend::CuFft)) {
            throw std::invalid_argument("Gauss complex multiplication requires a self-kernel FFT backend");
        }
        if (config_.batch > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("butterfly batch exceeds backend integer limits");
        }
        if (config_.backend != ButterflyBackend::StagePipeline) {
            config_.stage_space    = 0;
            config_.pipeline_warps = 0;
        }
        if (config_.backend != ButterflyBackend::TemporalTile && config_.backend != ButterflyBackend::Hierarchical &&
            config_.backend != ButterflyBackend::OnlineReorder) {
            config_.tile_threads = 0;
        }
        if (config_.backend != ButterflyBackend::Hierarchical && config_.backend != ButterflyBackend::OnlineReorder)
            config_.local_stages = 0;
        if (config_.backend != ButterflyBackend::OnlineReorder)
            config_.reorder_columns = 0;
        if (config_.backend != ButterflyBackend::WarpHybrid) {
            config_.warp_stages = 0;
        }

        points_ = std::size_t{1} << config_.log_n;
        if (config_.element_stride == 0) {
            throw std::invalid_argument("butterfly element_stride must be positive");
        }
        constexpr std::size_t kMaxSize = std::numeric_limits<std::size_t>::max();
        if (config_.element_stride > (kMaxSize - 1) / (points_ - 1)) {
            throw std::invalid_argument("butterfly transform extent overflows size_t");
        }
        const std::size_t transform_extent = (points_ - 1) * config_.element_stride + 1;
        if (config_.batch_stride == 0) {
            config_.batch_stride = transform_extent;
        }
        if (config_.batch_stride < transform_extent) {
            throw std::invalid_argument("butterfly batch_stride is smaller than one strided transform");
        }
        if (config_.batch > 1 && config_.batch_stride > (kMaxSize - transform_extent) / (config_.batch - 1)) {
            throw std::invalid_argument("butterfly batch extent overflows size_t");
        }
        if (config_.backend == ButterflyBackend::CuFft && (config_.batch_stride > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                                                           config_.element_stride > static_cast<std::size_t>(std::numeric_limits<int>::max()))) {
            throw std::invalid_argument("cuFFT strides exceed backend integer limits");
        }
        const std::size_t element_bytes = config_.op == ButterflyOperator::Fft
                                              ? (config_.precision == ButterflyPrecision::Fp64 ? sizeof(Complex64) : sizeof(Complex32))
                                              : (config_.precision == ButterflyPrecision::Fp64 ? sizeof(double) : sizeof(std::uint32_t));
        data_elements_                  = (config_.batch - 1) * config_.batch_stride + transform_extent;
        if (data_elements_ > kMaxSize / element_bytes) {
            throw std::invalid_argument("butterfly allocation size overflows size_t");
        }
        data_bytes_ = data_elements_ * element_bytes;
        device_input_.allocate(data_bytes_);
        if (config_.placement == ButterflyPlacement::OutOfPlace) {
            device_output_.allocate(data_bytes_);
        }
        if (config_.backend == ButterflyBackend::Hierarchical || config_.backend == ButterflyBackend::OnlineReorder) {
            device_scratch_.allocate(data_bytes_);
        }

        if (config_.op == ButterflyOperator::Fft && config_.backend != ButterflyBackend::CuFft) {
            constexpr double kPi = 3.141592653589793238462643383279502884;
            if (config_.precision == ButterflyPrecision::Fp64) {
                std::vector<Complex64> twiddles(points_ - 1);
                for (std::uint32_t stage = 0; stage < config_.log_n; ++stage) {
                    const std::uint32_t half = 1U << stage;
                    for (std::uint32_t offset = 0; offset < half; ++offset) {
                        const double angle          = (config_.inverse ? 1.0 : -1.0) * kPi * static_cast<double>(offset) / half;
                        twiddles[half - 1 + offset] = {std::cos(angle), std::sin(angle)};
                    }
                }
                device_twiddles_.allocate(twiddles.size() * sizeof(Complex64));
                CUB_CUDA_CHECK(
                    cudaMemcpy(device_twiddles_.as<Complex64>(), twiddles.data(), twiddles.size() * sizeof(Complex64), cudaMemcpyHostToDevice));
            } else {
                std::vector<Complex32> twiddles(points_ - 1);
                for (std::uint32_t stage = 0; stage < config_.log_n; ++stage) {
                    const std::uint32_t half = 1U << stage;
                    for (std::uint32_t offset = 0; offset < half; ++offset) {
                        const double angle          = (config_.inverse ? 1.0 : -1.0) * kPi * static_cast<double>(offset) / half;
                        twiddles[half - 1 + offset] = {static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle))};
                    }
                }
                device_twiddles_.allocate(twiddles.size() * sizeof(Complex32));
                CUB_CUDA_CHECK(
                    cudaMemcpy(device_twiddles_.as<Complex32>(), twiddles.data(), twiddles.size() * sizeof(Complex32), cudaMemcpyHostToDevice));
            }
        }

        if (config_.backend == ButterflyBackend::CuFft) {
            int             length         = static_cast<int>(points_);
            const cufftType type           = config_.precision == ButterflyPrecision::Fp64 ? CUFFT_Z2Z : CUFFT_C2C;
            const int       distance       = static_cast<int>(config_.batch_stride);
            const int       element_stride = static_cast<int>(config_.element_stride);
            CUB_CUFFT_CHECK(cufftPlanMany(&cufft_plan_, 1, &length, &length, element_stride, distance, &length, element_stride, distance, type,
                                          static_cast<int>(config_.batch)));
        }
    }

    ~Impl() {
        if (cufft_plan_ != 0) {
            cufftDestroy(cufft_plan_);
        }
    }

    const ButterflyConfig& config() const noexcept { return config_; }

    template <typename Value>
    ButterflyStats execute(const std::vector<Value>& input, std::vector<Value>& output, std::uint32_t warmup, std::uint32_t repeat,
                           ButterflyOperator required_op, ButterflyPrecision required_precision) {
        const bool mixed_complex32 = required_op == ButterflyOperator::Fft && required_precision == ButterflyPrecision::Fp32 &&
                                     config_.precision == ButterflyPrecision::Fp16Fp32;
        if (config_.op != required_op || (config_.precision != required_precision && !mixed_complex32)) {
            throw std::invalid_argument("input type does not match butterfly operator");
        }
        if (input.size() != data_elements_) {
            throw std::invalid_argument("butterfly input size does not match the strided batch extent");
        }
        if (repeat == 0) {
            throw std::invalid_argument("repeat must be positive");
        }
        output.resize(input.size());

        Event start;
        Event stop;
        CUB_CUDA_CHECK(cudaEventRecord(start.get()));
        CUB_CUDA_CHECK(cudaMemcpyAsync(device_input_.as<Value>(), input.data(), data_bytes_, cudaMemcpyHostToDevice));
        CUB_CUDA_CHECK(cudaEventRecord(stop.get()));
        CUB_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float h2d_ms = 0.0F;
        CUB_CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start.get(), stop.get()));

        for (std::uint32_t iteration = 0; iteration < warmup; ++iteration) {
            launch_once();
        }
        CUB_CUDA_CHECK(cudaDeviceSynchronize());
        CUB_CUDA_CHECK(cudaEventRecord(start.get()));
        for (std::uint32_t iteration = 0; iteration < repeat; ++iteration) {
            launch_once();
        }
        CUB_CUDA_CHECK(cudaEventRecord(stop.get()));
        CUB_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float total_kernel_ms = 0.0F;
        CUB_CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start.get(), stop.get()));

        // Repeated in-place launches consume their previous result. Restore the
        // API contract without including this correctness launch in kernel time.
        if (config_.placement == ButterflyPlacement::InPlace && (warmup != 0 || repeat > 1)) {
            CUB_CUDA_CHECK(cudaMemcpy(device_input_.as<Value>(), input.data(), data_bytes_, cudaMemcpyHostToDevice));
            launch_once();
            CUB_CUDA_CHECK(cudaDeviceSynchronize());
        }

        CUB_CUDA_CHECK(cudaEventRecord(start.get()));
        CUB_CUDA_CHECK(cudaMemcpyAsync(output.data(), result_buffer<Value>(), data_bytes_, cudaMemcpyDeviceToHost));
        CUB_CUDA_CHECK(cudaEventRecord(stop.get()));
        CUB_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float d2h_ms = 0.0F;
        CUB_CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start.get(), stop.get()));

        ButterflyStats stats;
        stats.h2d_ms                 = h2d_ms;
        stats.kernel_ms              = total_kernel_ms / repeat;
        stats.d2h_ms                 = d2h_ms;
        stats.transforms_per_second  = config_.batch / (stats.kernel_ms / 1000.0);
        stats.butterflies_per_second = stats.transforms_per_second * static_cast<double>(points_ / 2) * config_.log_n;
        stats.points_per_second      = stats.transforms_per_second * static_cast<double>(points_);
        return stats;
    }

  private:
    template <typename Value>
    Value* result_buffer() const noexcept {
        return config_.placement == ButterflyPlacement::InPlace ? device_input_.as<Value>() : device_output_.as<Value>();
    }

    void launch_once() {
        if (config_.backend == ButterflyBackend::CuFft) {
            if (config_.precision == ButterflyPrecision::Fp64) {
                CUB_CUFFT_CHECK(cufftExecZ2Z(cufft_plan_, reinterpret_cast<cufftDoubleComplex*>(device_input_.as<Complex64>()),
                                             reinterpret_cast<cufftDoubleComplex*>(result_buffer<Complex64>()),
                                             config_.inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
            } else {
                CUB_CUFFT_CHECK(cufftExecC2C(cufft_plan_, reinterpret_cast<cufftComplex*>(device_input_.as<Complex32>()),
                                             reinterpret_cast<cufftComplex*>(result_buffer<Complex32>()),
                                             config_.inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
            }
            if (config_.inverse && config_.normalize_inverse) {
                constexpr std::uint32_t kThreads = 256;
                const auto              count    = static_cast<std::uint64_t>(data_elements_);
                const auto              blocks   = static_cast<unsigned int>((count + kThreads - 1) / kThreads);
                if (config_.precision == ButterflyPrecision::Fp64) {
                    scale_inverse_fft_kernel<Complex64, double><<<blocks, kThreads>>>(result_buffer<Complex64>(), count, 1.0 / points_);
                } else {
                    scale_inverse_fft_kernel<Complex32, float>
                        <<<blocks, kThreads>>>(result_buffer<Complex32>(), count, 1.0F / static_cast<float>(points_));
                }
                CUB_CUDA_CHECK(cudaGetLastError());
            }
            return;
        }

        if (config_.op == ButterflyOperator::Fwht) {
            if (config_.precision == ButterflyPrecision::Fp64) {
                const FwhtOperator<double> op{config_.inverse, config_.normalize_inverse};
                launch_operator(device_input_.as<double>(), result_buffer<double>(), op);
            } else {
                const FwhtOperator<float> op{config_.inverse, config_.normalize_inverse};
                launch_operator(device_input_.as<float>(), result_buffer<float>(), op);
            }
        } else if (config_.op == ButterflyOperator::Fft) {
            if (config_.fft_core == FftCore::ThreadDft8) {
                detail::launch_generated_thread_dft8(config_.log_n, device_input_.as<Complex32>(), result_buffer<Complex32>(),
                                                     device_twiddles_.as<Complex32>(), config_.batch, config_.batch_stride,
                                                     config_.element_stride, config_.inverse && config_.normalize_inverse);
                CUB_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (config_.fft_core == FftCore::CtaDft8) {
                detail::launch_generated_cta_dft8(config_.log_n, device_input_.as<Complex32>(), result_buffer<Complex32>(),
                                                  device_twiddles_.as<Complex32>(), config_.batch, config_.batch_stride,
                                                  config_.element_stride, config_.tile_threads,
                                                  config_.inverse && config_.normalize_inverse);
                CUB_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (config_.fft_core == FftCore::WmmaDft8) {
                detail::launch_generated_wmma_dft8(config_.log_n, device_input_.as<Complex32>(), result_buffer<Complex32>(), config_.batch,
                                                   config_.batch_stride, config_.element_stride, config_.inverse,
                                                   config_.inverse && config_.normalize_inverse);
                CUB_CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (config_.precision == ButterflyPrecision::Fp64) {
                const FftOperator<Complex64, double> op{device_twiddles_.as<Complex64>(), config_.inverse, config_.normalize_inverse,
                                                        config_.complex_multiply};
                launch_operator(device_input_.as<Complex64>(), result_buffer<Complex64>(), op);
            } else {
                const FftOperator<Complex32, float> op{device_twiddles_.as<Complex32>(), config_.inverse, config_.normalize_inverse,
                                                       config_.complex_multiply};
                launch_operator(device_input_.as<Complex32>(), result_buffer<Complex32>(), op);
            }
        } else {
            const XorZetaOperator op{config_.inverse};
            launch_operator(device_input_.as<std::uint32_t>(), result_buffer<std::uint32_t>(), op);
        }
    }

    template <typename Operator>
    void launch_operator(const typename Operator::Value* input, typename Operator::Value* output, Operator op) {
        if (config_.local_exchange == LocalExchange::WarpRegister) {
            if constexpr (std::is_same_v<typename Operator::Value, float>) {
                detail::launch_generated_register_fwht(config_.log_n, input, output, config_.batch, config_.batch_stride,
                                                       config_.element_stride, op.inverse && op.normalize_inverse);
                CUB_CUDA_CHECK(cudaGetLastError());
                return;
            } else {
                throw std::logic_error("warp-register dispatch reached an unsupported value type");
            }
        }
        if (config_.backend == ButterflyBackend::TemporalTile) {
            dispatch_temporal_length(config_, input, output, op);
            return;
        }
        if (config_.backend == ButterflyBackend::Hierarchical) {
            launch_hierarchical(config_, input, output, device_scratch_.as<typename Operator::Value>(), data_bytes_, op);
            return;
        }
        if (config_.backend == ButterflyBackend::OnlineReorder) {
            dispatch_online_reorder(config_, input, output, device_scratch_.as<typename Operator::Value>(), op);
            return;
        }
        if (config_.backend == ButterflyBackend::WarpHybrid) {
            detail::warp_hybrid_256_kernel<Operator><<<static_cast<unsigned int>(config_.batch), 256>>>(
                input, output, config_.batch, config_.warp_stages, config_.batch_stride, config_.element_stride, op);
            CUB_CUDA_CHECK(cudaGetLastError());
            return;
        }
        if constexpr (sizeof(typename Operator::Value) > sizeof(Complex32)) {
            throw std::logic_error("selected stage-pipeline value type is not instantiated");
        } else {
            if (config_.stage_handoff == StageHandoff::NamedBarrier) {
                dispatch_pipeline_warps<Operator, true>(config_, input, output, op);
            } else {
                dispatch_pipeline_warps<Operator, false>(config_, input, output, op);
            }
        }
    }

    ButterflyConfig config_;
    std::size_t     data_bytes_    = 0;
    std::size_t     data_elements_ = 0;
    std::size_t     points_        = 0;
    DeviceBuffer    device_input_;
    DeviceBuffer    device_output_;
    DeviceBuffer    device_scratch_;
    DeviceBuffer    device_twiddles_;
    cufftHandle     cufft_plan_ = 0;
};

ButterflyPlan::ButterflyPlan(ButterflyConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {
}

ButterflyPlan::~ButterflyPlan()                                   = default;
ButterflyPlan::ButterflyPlan(ButterflyPlan&&) noexcept            = default;
ButterflyPlan& ButterflyPlan::operator=(ButterflyPlan&&) noexcept = default;

const ButterflyConfig& ButterflyPlan::config() const noexcept {
    return impl_->config();
}

ButterflyStats ButterflyPlan::execute(const std::vector<float>& input, std::vector<float>& output, std::uint32_t warmup, std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat, ButterflyOperator::Fwht, ButterflyPrecision::Fp32);
}

ButterflyStats ButterflyPlan::execute(const std::vector<double>& input, std::vector<double>& output, std::uint32_t warmup, std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat, ButterflyOperator::Fwht, ButterflyPrecision::Fp64);
}

ButterflyStats ButterflyPlan::execute(const std::vector<Complex32>& input, std::vector<Complex32>& output, std::uint32_t warmup,
                                      std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat, ButterflyOperator::Fft, ButterflyPrecision::Fp32);
}

ButterflyStats ButterflyPlan::execute(const std::vector<Complex64>& input, std::vector<Complex64>& output, std::uint32_t warmup,
                                      std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat, ButterflyOperator::Fft, ButterflyPrecision::Fp64);
}

ButterflyStats ButterflyPlan::execute(const std::vector<std::uint32_t>& input, std::vector<std::uint32_t>& output, std::uint32_t warmup,
                                      std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat, ButterflyOperator::XorZeta, ButterflyPrecision::Uint32);
}

}  // namespace cuntt
