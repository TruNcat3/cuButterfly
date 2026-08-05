#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuntt/ntt.hpp"
#include "mapping_selector.hpp"
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

#define CUNTT_CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

thread_local cudaStream_t active_stream = nullptr;

class LaunchStreamScope {
  public:
    explicit LaunchStreamScope(cudaStream_t stream) noexcept : previous_(active_stream) { active_stream = stream; }
    ~LaunchStreamScope() { active_stream = previous_; }

  private:
    cudaStream_t previous_;
};

class Event {
  public:
    Event() { CUNTT_CUDA_CHECK(cudaEventCreate(&event_)); }

    ~Event() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    Event(const Event&)            = delete;
    Event& operator=(const Event&) = delete;

    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_ = nullptr;
};

class DeviceBuffer {
  public:
    DeviceBuffer() = default;

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(std::size_t bytes) {
        if (pointer_ != nullptr) {
            throw std::logic_error("device buffer is already allocated");
        }
        CUNTT_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), bytes));
    }

    std::uint64_t* get() const noexcept { return pointer_; }

    template <typename T>
    T* as() const noexcept {
        return reinterpret_cast<T*>(pointer_);
    }

    void* data() const noexcept { return pointer_; }

  private:
    std::uint64_t* pointer_ = nullptr;
};

std::uint64_t shoup_precompute(std::uint64_t value, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(value) << 64) / modulus);
}

std::uint32_t shoup_precompute32(std::uint32_t value, std::uint32_t modulus) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(value) << 32) / modulus);
}

std::uint32_t modulus_bit_length(std::uint64_t modulus) {
    std::uint32_t bits = 0;
    while (modulus != 0) {
        ++bits;
        modulus >>= 1;
    }
    return bits;
}

std::uint32_t reverse_bits_host(std::uint32_t value, std::uint32_t bits) {
    std::uint32_t reversed = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | ((value >> bit) & 1U);
    }
    return reversed;
}

std::uint64_t barrett_precompute(std::uint64_t modulus, std::uint32_t modulus_bits) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << (2 * modulus_bits + 1)) / modulus);
}

__device__ __forceinline__ std::uint64_t mul_shoup(std::uint64_t value, std::uint64_t twiddle, std::uint64_t twiddle_shoup, std::uint64_t modulus) {
    const std::uint64_t quotient = __umul64hi(value, twiddle_shoup);
    std::uint64_t       reduced  = value * twiddle - quotient * modulus;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

__device__ __forceinline__ std::uint32_t mul_shoup(std::uint32_t value, std::uint32_t twiddle, std::uint32_t twiddle_shoup, std::uint32_t modulus) {
    const std::uint32_t quotient = __umulhi(value, twiddle_shoup);
    std::uint32_t       reduced  = value * twiddle - quotient * modulus;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

struct Wide128 {
    std::uint64_t low;
    std::uint64_t high;
};

__device__ __forceinline__ Wide128 wide_multiply(std::uint64_t left, std::uint64_t right) {
    return {left * right, __umul64hi(left, right)};
}

__device__ __forceinline__ Wide128 wide_shift_right(Wide128 value, std::uint32_t shift) {
    if (shift == 0) {
        return value;
    }
    if (shift < 64) {
        return {(value.low >> shift) | (value.high << (64 - shift)), value.high >> shift};
    }
    if (shift < 128) {
        return {value.high >> (shift - 64), 0};
    }
    return {0, 0};
}

__device__ __forceinline__ Wide128 wide_subtract(Wide128 left, Wide128 right) {
    const std::uint64_t low = left.low - right.low;
    return {low, left.high - right.high - (left.low < right.low)};
}

__device__ __forceinline__ std::uint64_t mul_barrett(std::uint64_t left, std::uint64_t right, std::uint64_t modulus, std::uint32_t modulus_bits,
                                                     std::uint64_t mu) {
    Wide128 product  = wide_multiply(left, right);
    Wide128 quotient = wide_shift_right(product, modulus_bits - 2);
    quotient         = wide_multiply(quotient.low, mu);
    quotient         = wide_shift_right(quotient, modulus_bits + 3);
    product          = wide_subtract(product, wide_multiply(quotient.low, modulus));
    return product.low >= modulus ? product.low - modulus : product.low;
}

__device__ __forceinline__ std::uint32_t mul_barrett(std::uint32_t left, std::uint32_t right, std::uint32_t modulus, std::uint32_t modulus_bits,
                                                     std::uint32_t mu) {
    std::uint64_t product  = static_cast<std::uint64_t>(left) * right;
    std::uint64_t quotient = product >> (modulus_bits - 2);
    quotient               = static_cast<std::uint64_t>(static_cast<std::uint32_t>(quotient)) * mu;
    quotient >>= modulus_bits + 3;
    product -= static_cast<std::uint64_t>(static_cast<std::uint32_t>(quotient)) * modulus;
    return static_cast<std::uint32_t>(product >= modulus ? product - modulus : product);
}

__device__ __forceinline__ void butterfly(std::uint64_t& left, std::uint64_t& right, std::uint64_t twiddle, std::uint64_t twiddle_shoup,
                                          std::uint64_t modulus) {
    const std::uint64_t u   = left;
    const std::uint64_t v   = mul_shoup(right, twiddle, twiddle_shoup, modulus);
    const std::uint64_t sum = u + v;
    left                    = sum >= modulus ? sum - modulus : sum;
    right                   = u >= v ? u - v : modulus + u - v;
}

struct NttStageOperator {
    using Value = std::uint64_t;
    static constexpr bool kBitReverseInput = true;

    const std::uint64_t* twiddles;
    const std::uint64_t* twiddles_shoup;
    std::uint64_t        modulus;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const std::uint32_t twiddle_base = (1U << stage) - 1;
        butterfly(left, right, twiddles[twiddle_base + offset], twiddles_shoup[twiddle_base + offset], modulus);
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const {
        return value;
    }
};

template <std::uint32_t StageSpace, bool NamedBarrier>
void launch_stage_pipeline_ntt256(const PlanConfig& config, const std::uint64_t* input, std::uint64_t* output,
                                  const std::uint64_t* twiddles, const std::uint64_t* twiddles_shoup) {
    constexpr std::uint32_t kPipelines         = 8 / StageSpace;
    constexpr std::uint32_t kTokensPerPipeline = 2;
    const std::uint64_t     tiles              = config.batch;
    const std::uint32_t     blocks = static_cast<std::uint32_t>((tiles + kPipelines * kTokensPerPipeline - 1) /
                                                             (kPipelines * kTokensPerPipeline));
    for (std::uint32_t stage_base = 0; stage_base < 8; stage_base += StageSpace) {
        const std::uint64_t* stage_input = stage_base == 0 ? input : output;
        constexpr std::size_t kHandoffPadding = NamedBarrier ? 16 * sizeof(int) : 0;
        const NttStageOperator op{twiddles, twiddles_shoup, config.modulus};
        detail::stage_pipeline_256_kernel<NttStageOperator, StageSpace, NamedBarrier><<<blocks, 256, kHandoffPadding, active_stream>>>(
            stage_input, output, tiles, stage_base, 256, 256, 1, op);
        CUNTT_CUDA_CHECK(cudaGetLastError());
    }
}

__global__ void bit_reverse_copy_kernel(const std::uint64_t* input, std::uint64_t* output, std::uint64_t total_points, std::uint32_t log_n) {
    const std::uint64_t global_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_index >= total_points) {
        return;
    }
    const std::uint64_t n              = 1ULL << log_n;
    const std::uint64_t local_index    = global_index & (n - 1);
    const std::uint32_t reversed       = __brev(static_cast<std::uint32_t>(local_index)) >> (32 - log_n);
    const std::uint64_t batch_index    = global_index >> log_n;
    output[batch_index * n + reversed] = input[global_index];
}

__device__ __forceinline__ void compact_butterfly(std::uint64_t& left, std::uint64_t& right, std::uint64_t root, std::uint64_t root_shoup,
                                                  std::uint64_t modulus) {
    const std::uint64_t u   = left;
    const std::uint64_t v   = mul_shoup(right, root, root_shoup, modulus);
    const std::uint64_t sum = u + v;
    left                    = sum >= modulus ? sum - modulus : sum;
    right                   = u >= v ? u - v : modulus + u - v;
}

template <std::uint32_t BlockX, std::uint32_t BlockY, std::uint32_t LogM, std::uint32_t OuterIterations, bool LastKernel, bool NaturalOutput = false>
__global__ void compact_stage_kernel(const std::uint64_t* input, std::uint64_t* output, const std::uint64_t* roots, std::uint64_t modulus,
                                     const std::uint64_t* roots_shoup) {
    constexpr std::uint32_t kLogN        = 20;
    constexpr std::uint32_t kSharedIndex = 8;
    constexpr std::uint32_t kPoints      = BlockX * BlockY;
    static_assert(kPoints == 256);

    __shared__ std::uint64_t shared[2 * kPoints];

    const std::uint32_t thread_x       = threadIdx.x;
    const std::uint32_t thread_y       = threadIdx.y;
    const std::uint32_t offset         = 1U << (kLogN - LogM - 1);
    const std::uint32_t stride         = offset >> (OuterIterations - 1);
    const std::uint64_t global_address = thread_x + static_cast<std::uint64_t>(thread_y) * stride + static_cast<std::uint64_t>(BlockX) * blockIdx.x +
                                         static_cast<std::uint64_t>(2U * blockIdx.y) * offset + (static_cast<std::uint64_t>(blockIdx.z) << kLogN);
    const std::uint32_t omega_address  = thread_x + thread_y * stride + BlockX * blockIdx.x + blockIdx.y * offset;
    const std::uint32_t shared_address = thread_x + thread_y * BlockX;

    shared[shared_address]           = input[global_address];
    shared[shared_address + kPoints] = input[global_address + offset];

    std::uint32_t distance          = 1U << kSharedIndex;
    std::int32_t  distance_log      = kSharedIndex;
    std::int32_t  root_shift        = kLogN - LogM - 1;
    std::uint32_t butterfly_address = ((shared_address >> distance_log) << distance_log) + shared_address;

    if constexpr (!LastKernel) {
#pragma unroll
        for (std::uint32_t iteration = 0; iteration < OuterIterations; ++iteration) {
            __syncthreads();
            const std::uint32_t root_index = omega_address >> root_shift;
            compact_butterfly(shared[butterfly_address], shared[butterfly_address + distance], roots[root_index], roots_shoup[root_index], modulus);
            distance >>= 1;
            --distance_log;
            --root_shift;
            butterfly_address = ((shared_address >> distance_log) << distance_log) + shared_address;
        }
        __syncthreads();
    } else {
#pragma unroll
        for (std::uint32_t iteration = 0; iteration < kSharedIndex - 5; ++iteration) {
            __syncthreads();
            const std::uint32_t root_index = omega_address >> root_shift;
            compact_butterfly(shared[butterfly_address], shared[butterfly_address + distance], roots[root_index], roots_shoup[root_index], modulus);
            distance >>= 1;
            --distance_log;
            --root_shift;
            butterfly_address = ((shared_address >> distance_log) << distance_log) + shared_address;
        }
        __syncthreads();
#pragma unroll
        for (std::uint32_t iteration = 0; iteration < 6; ++iteration) {
            const std::uint32_t root_index = omega_address >> root_shift;
            compact_butterfly(shared[butterfly_address], shared[butterfly_address + distance], roots[root_index], roots_shoup[root_index], modulus);
            distance >>= 1;
            --distance_log;
            --root_shift;
            if (iteration + 1 < 6) {
                butterfly_address = ((shared_address >> distance_log) << distance_log) + shared_address;
            }
        }
        __syncthreads();
    }

    if constexpr (NaturalOutput) {
        const std::uint64_t transform_base       = static_cast<std::uint64_t>(blockIdx.z) << kLogN;
        const std::uint32_t local_address        = static_cast<std::uint32_t>(global_address - transform_base);
        const std::uint32_t first_reversed       = __brev(local_address) >> (32 - kLogN);
        const std::uint32_t second_reversed      = __brev(local_address + offset) >> (32 - kLogN);
        output[transform_base + first_reversed]  = shared[shared_address];
        output[transform_base + second_reversed] = shared[shared_address + kPoints];
    } else {
        output[global_address]          = shared[shared_address];
        output[global_address + offset] = shared[shared_address + kPoints];
    }
}

void launch_compact_stage_log20(const PlanConfig& config, const std::uint64_t* input, std::uint64_t* output, const std::uint64_t* roots,
                                const std::uint64_t* roots_shoup, std::uint64_t* natural_output) {
    const auto batch = static_cast<unsigned int>(config.batch);

    compact_stage_kernel<16, 16, 0, 5, false><<<dim3(2048, 1, batch), dim3(16, 16), 0, active_stream>>>(input, output, roots, config.modulus,
                                                                                                        roots_shoup);
    CUNTT_CUDA_CHECK(cudaGetLastError());
    compact_stage_kernel<8, 32, 5, 6, false><<<dim3(64, 32, batch), dim3(8, 32), 0, active_stream>>>(output, output, roots, config.modulus,
                                                                                                     roots_shoup);
    CUNTT_CUDA_CHECK(cudaGetLastError());
    if (natural_output != nullptr) {
        compact_stage_kernel<256, 1, 11, 9, true, true>
            <<<dim3(1, 2048, batch), dim3(256, 1), 0, active_stream>>>(output, natural_output, roots, config.modulus, roots_shoup);
    } else {
        compact_stage_kernel<256, 1, 11, 9, true><<<dim3(1, 2048, batch), dim3(256, 1), 0, active_stream>>>(output, output, roots, config.modulus,
                                                                                                          roots_shoup);
    }
    CUNTT_CUDA_CHECK(cudaGetLastError());
}

__global__ void stage_kernel(std::uint64_t* values, std::uint64_t butterflies_per_transform, std::uint64_t total_butterflies, std::uint64_t n,
                             std::uint64_t half, std::uint64_t twiddle_offset, const std::uint64_t* twiddles, const std::uint64_t* twiddles_shoup,
                             std::uint64_t modulus) {
    const std::uint64_t global_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_index >= total_butterflies) {
        return;
    }

    const std::uint64_t batch_index     = global_index / butterflies_per_transform;
    const std::uint64_t local_butterfly = global_index - batch_index * butterflies_per_transform;
    const std::uint64_t group           = local_butterfly / half;
    const std::uint64_t offset          = local_butterfly - group * half;
    const std::uint64_t left_index      = batch_index * n + group * (half << 1) + offset;
    const std::uint64_t right_index     = left_index + half;

    std::uint64_t left  = values[left_index];
    std::uint64_t right = values[right_index];
    butterfly(left, right, twiddles[twiddle_offset + offset], twiddles_shoup[twiddle_offset + offset], modulus);
    values[left_index]  = left;
    values[right_index] = right;
}

__global__ void tile_kernel(std::uint64_t* values, std::uint32_t tile_log, const std::uint64_t* twiddles, const std::uint64_t* twiddles_shoup,
                            std::uint64_t modulus) {
    __shared__ std::uint64_t tile[256];

    const std::uint32_t tile_size = 1U << tile_log;
    const std::uint64_t base      = static_cast<std::uint64_t>(blockIdx.x) * tile_size;
    for (std::uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        tile[i] = values[base + i];
    }
    __syncthreads();

    for (std::uint32_t stage = 0; stage < tile_log; ++stage) {
        const std::uint32_t half           = 1U << stage;
        const std::uint32_t twiddle_offset = half - 1;
        const std::uint32_t butterflies    = tile_size >> 1;
        for (std::uint32_t index = threadIdx.x; index < butterflies; index += blockDim.x) {
            const std::uint32_t group       = index / half;
            const std::uint32_t offset      = index - group * half;
            const std::uint32_t left_index  = group * (half << 1) + offset;
            const std::uint32_t right_index = left_index + half;
            std::uint64_t       left        = tile[left_index];
            std::uint64_t       right       = tile[right_index];
            butterfly(left, right, twiddles[twiddle_offset + offset], twiddles_shoup[twiddle_offset + offset], modulus);
            tile[left_index]  = left;
            tile[right_index] = right;
        }
        __syncthreads();
    }

    for (std::uint32_t i = threadIdx.x; i < tile_size; i += blockDim.x) {
        values[base + i] = tile[i];
    }
}

constexpr std::uint32_t kHybridMinLocalLog = 6;
constexpr std::uint32_t kHybridMaxLocalLog = 10;

template <bool Fused, typename Word>
__device__ __forceinline__ Word local_twiddle(const Word* standard, const Word* fused, std::uint32_t row_base, std::uint32_t row,
                                              std::uint32_t local_n, std::uint32_t index) {
    if constexpr (Fused) {
        return fused[(row_base + row) * (local_n - 1) + index];
    }
    return standard[index];
}

template <bool Barrett, bool Fused, typename Word>
__device__ __forceinline__ Word local_twiddle_shoup(const Word* standard, const Word* fused, std::uint32_t row_base, std::uint32_t row,
                                                    std::uint32_t local_n, std::uint32_t index) {
    if constexpr (Barrett) {
        return 0;
    }
    return local_twiddle<Fused>(standard, fused, row_base, row, local_n, index);
}

template <bool Barrett, typename Word>
__device__ __forceinline__ void local_butterfly(Word& left, Word& right, Word twiddle, Word twiddle_shoup, Word modulus, std::uint32_t modulus_bits,
                                                Word barrett_mu) {
    const Word u = left;
    Word       v;
    if constexpr (Barrett) {
        v = mul_barrett(right, twiddle, modulus, modulus_bits, barrett_mu);
    } else {
        v = mul_shoup(right, twiddle, twiddle_shoup, modulus);
    }
    const Word sum = u + v;
    left           = sum >= modulus ? sum - modulus : sum;
    right          = u >= v ? u - v : modulus + u - v;
}

template <typename Word, std::uint32_t LocalLog, std::uint32_t RowsPerBlock, ComputeUnit Unit, bool Fused = false, bool Barrett = false>
__device__ void local_ntt(Word* tiles, const Word* twiddles, const Word* twiddles_shoup, Word modulus, const Word* fused_twiddles = nullptr,
                          const Word* fused_twiddles_shoup = nullptr, std::uint32_t row_base = 0, std::uint32_t modulus_bits = 0,
                          Word barrett_mu = 0) {
    constexpr std::uint32_t local_n = 1U << LocalLog;
    constexpr std::uint32_t stride  = local_n + 1;
    if constexpr (Unit == ComputeUnit::Radix2) {
        constexpr std::uint32_t butterflies_per_row = local_n / 2;
        for (std::uint32_t stage = 0; stage < LocalLog; ++stage) {
            const std::uint32_t half           = 1U << stage;
            const std::uint32_t twiddle_offset = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t local       = linear - row * butterflies_per_row;
                const std::uint32_t group       = local / half;
                const std::uint32_t offset      = local - group * half;
                const std::uint32_t left_index  = row * stride + group * (half << 1) + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    } else if constexpr (Unit == ComputeUnit::Radix4) {
        constexpr std::uint32_t radix4_units_per_row = local_n / 4;
        std::uint32_t           stage                = 0;
        for (; stage + 1 < LocalLog; stage += 2) {
            const std::uint32_t half0           = 1U << stage;
            const std::uint32_t twiddle_offset0 = half0 - 1;
            const std::uint32_t twiddle_offset1 = (half0 << 1) - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * radix4_units_per_row; linear += blockDim.x) {
                const std::uint32_t row      = linear / radix4_units_per_row;
                const std::uint32_t local    = linear - row * radix4_units_per_row;
                const std::uint32_t group    = local / half0;
                const std::uint32_t offset   = local - group * half0;
                const std::uint32_t i0       = row * stride + group * (half0 << 2) + offset;
                const std::uint32_t i1       = i0 + half0;
                const std::uint32_t i2       = i1 + half0;
                const std::uint32_t i3       = i2 + half0;
                Word                a        = tiles[i0];
                Word                b        = tiles[i1];
                Word                c        = tiles[i2];
                Word                d        = tiles[i3];
                const Word          twiddle0 = local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset0 + offset);
                const Word          twiddle0_shoup =
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset0 + offset);
                local_butterfly<Barrett>(a, b, twiddle0, twiddle0_shoup, modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(c, d, twiddle0, twiddle0_shoup, modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(
                    a, c, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset1 + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset1 + offset),
                    modulus, modulus_bits, barrett_mu);
                local_butterfly<Barrett>(b, d,
                                         local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset1 + half0 + offset),
                                         local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n,
                                                                             twiddle_offset1 + half0 + offset),
                                         modulus, modulus_bits, barrett_mu);
                tiles[i0] = a;
                tiles[i1] = b;
                tiles[i2] = c;
                tiles[i3] = d;
            }
            __syncthreads();
        }
        if (stage < LocalLog) {
            constexpr std::uint32_t butterflies_per_row = local_n / 2;
            constexpr std::uint32_t half                = 1U << (LocalLog - 1);
            constexpr std::uint32_t twiddle_offset      = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t offset      = linear - row * butterflies_per_row;
                const std::uint32_t left_index  = row * stride + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    } else {
        constexpr std::uint32_t radix8_units_per_row = local_n / 8;
        std::uint32_t           stage                = 0;
        for (; stage + 2 < LocalLog; stage += 3) {
            const std::uint32_t eighth          = 1U << stage;
            const std::uint32_t twiddle_offset0 = eighth - 1;
            const std::uint32_t twiddle_offset1 = (eighth << 1) - 1;
            const std::uint32_t twiddle_offset2 = (eighth << 2) - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * radix8_units_per_row; linear += blockDim.x) {
                const std::uint32_t row       = linear / radix8_units_per_row;
                const std::uint32_t local     = linear - row * radix8_units_per_row;
                const std::uint32_t group     = local / eighth;
                const std::uint32_t offset    = local - group * eighth;
                const std::uint32_t i0        = row * stride + group * (eighth << 3) + offset;
                const std::uint32_t i1        = i0 + eighth;
                const std::uint32_t i2        = i1 + eighth;
                const std::uint32_t i3        = i2 + eighth;
                const std::uint32_t i4        = i3 + eighth;
                const std::uint32_t i5        = i4 + eighth;
                const std::uint32_t i6        = i5 + eighth;
                const std::uint32_t i7        = i6 + eighth;
                Word                x0        = tiles[i0];
                Word                x1        = tiles[i1];
                Word                x2        = tiles[i2];
                Word                x3        = tiles[i3];
                Word                x4        = tiles[i4];
                Word                x5        = tiles[i5];
                Word                x6        = tiles[i6];
                Word                x7        = tiles[i7];
                const auto          butterfly = [&](Word& left, Word& right, std::uint32_t twiddle_index) {
                    local_butterfly<Barrett>(
                        left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_index),
                        local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_index), modulus,
                        modulus_bits, barrett_mu);
                };
                butterfly(x0, x1, twiddle_offset0 + offset);
                butterfly(x2, x3, twiddle_offset0 + offset);
                butterfly(x4, x5, twiddle_offset0 + offset);
                butterfly(x6, x7, twiddle_offset0 + offset);
                butterfly(x0, x2, twiddle_offset1 + offset);
                butterfly(x1, x3, twiddle_offset1 + eighth + offset);
                butterfly(x4, x6, twiddle_offset1 + offset);
                butterfly(x5, x7, twiddle_offset1 + eighth + offset);
                butterfly(x0, x4, twiddle_offset2 + offset);
                butterfly(x1, x5, twiddle_offset2 + eighth + offset);
                butterfly(x2, x6, twiddle_offset2 + 2 * eighth + offset);
                butterfly(x3, x7, twiddle_offset2 + 3 * eighth + offset);
                tiles[i0] = x0;
                tiles[i1] = x1;
                tiles[i2] = x2;
                tiles[i3] = x3;
                tiles[i4] = x4;
                tiles[i5] = x5;
                tiles[i6] = x6;
                tiles[i7] = x7;
            }
            __syncthreads();
        }
        for (; stage < LocalLog; ++stage) {
            constexpr std::uint32_t butterflies_per_row = local_n / 2;
            const std::uint32_t     half                = 1U << stage;
            const std::uint32_t     twiddle_offset      = half - 1;
            for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * butterflies_per_row; linear += blockDim.x) {
                const std::uint32_t row         = linear / butterflies_per_row;
                const std::uint32_t local       = linear - row * butterflies_per_row;
                const std::uint32_t group       = local / half;
                const std::uint32_t offset      = local - group * half;
                const std::uint32_t left_index  = row * stride + group * (half << 1) + offset;
                const std::uint32_t right_index = left_index + half;
                Word                left        = tiles[left_index];
                Word                right       = tiles[right_index];
                local_butterfly<Barrett>(
                    left, right, local_twiddle<Fused>(twiddles, fused_twiddles, row_base, row, local_n, twiddle_offset + offset),
                    local_twiddle_shoup<Barrett, Fused>(twiddles_shoup, fused_twiddles_shoup, row_base, row, local_n, twiddle_offset + offset),
                    modulus, modulus_bits, barrett_mu);
                tiles[left_index]  = left;
                tiles[right_index] = right;
            }
            __syncthreads();
        }
    }
}

template <typename Word, std::uint32_t N2Log, std::uint32_t RowsPerBlock, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett>
__global__ void hybrid2d_first_pass_kernel(const Word* input, Word* scratch, const Word* twiddles, const Word* twiddles_shoup,
                                           const Word* root_powers, const Word* root_powers_shoup, Word modulus, std::uint32_t n1_log,
                                           std::uint32_t modulus_bits, Word barrett_mu, std::uint32_t log_n) {
    constexpr std::uint32_t         N2      = 1U << N2Log;
    const std::uint32_t             n1_size = 1U << n1_log;
    extern __shared__ unsigned char shared_storage[];
    Word*                           tiles = reinterpret_cast<Word*>(shared_storage);

    const std::uint32_t blocks_per_transform = n1_size / RowsPerBlock;
    const std::uint64_t batch_index          = blockIdx.x / blocks_per_transform;
    const std::uint32_t row_base             = (blockIdx.x % blocks_per_transform) * RowsPerBlock;
    const std::uint64_t transform_base       = batch_index << log_n;

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N2; linear += blockDim.x) {
        const std::uint32_t n2           = linear / RowsPerBlock;
        const std::uint32_t row          = linear - n2 * RowsPerBlock;
        const std::uint32_t n1           = row_base + row;
        const std::uint32_t reversed     = __brev(n2) >> (32 - N2Log);
        tiles[row * (N2 + 1) + reversed] = input[transform_base + static_cast<std::uint64_t>(n2) * n1_size + n1];
    }
    __syncthreads();

    local_ntt<Word, N2Log, RowsPerBlock, Unit, false, Barrett>(tiles, twiddles, twiddles_shoup, modulus, nullptr, nullptr, 0, modulus_bits,
                                                               barrett_mu);

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N2; linear += blockDim.x) {
        const std::uint32_t k2    = linear / RowsPerBlock;
        const std::uint32_t row   = linear - k2 * RowsPerBlock;
        const std::uint32_t n1    = row_base + row;
        Word                value = tiles[row * (N2 + 1) + k2];
        if constexpr (Placement == CrossTwiddlePlacement::FirstPass) {
            const std::uint32_t exponent = n1 * k2;
            if constexpr (Barrett) {
                value = mul_barrett(value, root_powers[exponent], modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, root_powers[exponent], root_powers_shoup[exponent], modulus);
            }
        }
        scratch[transform_base + static_cast<std::uint64_t>(k2) * n1_size + n1] = value;
    }
}

template <typename Word, std::uint32_t N1Log, std::uint32_t RowsPerBlock, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett>
__global__ void hybrid2d_second_pass_kernel(const Word* scratch, Word* output, const Word* twiddles, const Word* twiddles_shoup, Word modulus,
                                            const Word* root_powers, const Word* root_powers_shoup, bool apply_scale, Word scale, Word scale_shoup,
                                            std::uint32_t modulus_bits, Word barrett_mu, std::uint32_t n2_log, std::uint32_t log_n) {
    constexpr std::uint32_t         N1 = 1U << N1Log;
    const std::uint32_t             n2 = 1U << n2_log;
    extern __shared__ unsigned char shared_storage[];
    Word*                           tiles = reinterpret_cast<Word*>(shared_storage);

    const std::uint32_t blocks_per_transform = n2 / RowsPerBlock;
    const std::uint64_t batch_index          = blockIdx.x / blocks_per_transform;
    const std::uint32_t row_base             = (blockIdx.x % blocks_per_transform) * RowsPerBlock;
    const std::uint64_t transform_base       = batch_index << log_n;

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N1; linear += blockDim.x) {
        const std::uint32_t row      = linear / N1;
        const std::uint32_t n1       = linear - row * N1;
        const std::uint32_t k2       = row_base + row;
        const std::uint32_t reversed = __brev(n1) >> (32 - N1Log);
        Word                value    = scratch[transform_base + static_cast<std::uint64_t>(k2) * N1 + n1];
        if constexpr (Placement == CrossTwiddlePlacement::SecondPass) {
            const std::uint32_t exponent = n1 * k2;
            if constexpr (Barrett) {
                value = mul_barrett(value, root_powers[exponent], modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, root_powers[exponent], root_powers_shoup[exponent], modulus);
            }
        }
        tiles[row * (N1 + 1) + reversed] = value;
    }
    __syncthreads();

    if constexpr (Placement == CrossTwiddlePlacement::Fused) {
        local_ntt<Word, N1Log, RowsPerBlock, Unit, true, Barrett>(tiles, twiddles, twiddles_shoup, modulus, root_powers, root_powers_shoup, row_base,
                                                                  modulus_bits, barrett_mu);
    } else {
        local_ntt<Word, N1Log, RowsPerBlock, Unit, false, Barrett>(tiles, twiddles, twiddles_shoup, modulus, nullptr, nullptr, 0, modulus_bits,
                                                                   barrett_mu);
    }

    for (std::uint32_t linear = threadIdx.x; linear < RowsPerBlock * N1; linear += blockDim.x) {
        const std::uint32_t k1    = linear / RowsPerBlock;
        const std::uint32_t row   = linear - k1 * RowsPerBlock;
        const std::uint32_t k2    = row_base + row;
        Word                value = tiles[row * (N1 + 1) + k1];
        if (apply_scale) {
            if constexpr (Barrett) {
                value = mul_barrett(value, scale, modulus, modulus_bits, barrett_mu);
            } else {
                value = mul_shoup(value, scale, scale_shoup, modulus);
            }
        }
        output[transform_base + static_cast<std::uint64_t>(k1) * n2 + k2] = value;
    }
}

__global__ void scale_kernel(std::uint64_t* values, std::uint64_t total_points, std::uint64_t scale, std::uint64_t scale_shoup,
                             std::uint64_t modulus) {
    const std::uint64_t global_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (global_index < total_points) {
        values[global_index] = mul_shoup(values[global_index], scale, scale_shoup, modulus);
    }
}

dim3 grid_for(std::uint64_t work_items, std::uint32_t threads) {
    const std::uint64_t blocks = (work_items + threads - 1) / threads;
    if (blocks == 0 || blocks > std::numeric_limits<unsigned int>::max()) {
        throw std::invalid_argument("CUDA grid dimension is outside the supported range");
    }
    return dim3(static_cast<unsigned int>(blocks));
}

template <typename Word, std::uint32_t LocalLog, std::uint32_t RowsPerBlock, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett>
void launch_hybrid2d_pass(const PlanConfig& config, bool first_pass, const Word* input, Word* scratch, Word* output, const Word* twiddles,
                          const Word* twiddles_shoup, const Word* root_powers, const Word* root_powers_shoup, std::uint64_t inverse_n,
                          std::uint64_t inverse_n_shoup) {
    const std::uint32_t   other_log            = config.log_n - LocalLog;
    const std::uint32_t   blocks_per_transform = (1U << other_log) / RowsPerBlock;
    const dim3            grid                 = grid_for(static_cast<std::uint64_t>(config.batch) * blocks_per_transform, 1);
    constexpr std::size_t shared_bytes         = RowsPerBlock * ((1U << LocalLog) + 1ULL) * sizeof(Word);
    const std::uint32_t   modulus_bits         = modulus_bit_length(config.modulus);
    const Word            barrett_mu           = static_cast<Word>(barrett_precompute(config.modulus, modulus_bits));
    if (first_pass) {
        hybrid2d_first_pass_kernel<Word, LocalLog, RowsPerBlock, Unit, Placement, Barrett>
            <<<grid, config.threads_per_block, shared_bytes, active_stream>>>(input, scratch, twiddles, twiddles_shoup, root_powers, root_powers_shoup,
                                                               static_cast<Word>(config.modulus), other_log, modulus_bits, barrett_mu, config.log_n);
    } else {
        hybrid2d_second_pass_kernel<Word, LocalLog, RowsPerBlock, Unit, Placement, Barrett><<<grid, config.threads_per_block, shared_bytes, active_stream>>>(
            scratch, output, twiddles, twiddles_shoup, static_cast<Word>(config.modulus), root_powers, root_powers_shoup, config.inverse,
            static_cast<Word>(inverse_n), static_cast<Word>(inverse_n_shoup), modulus_bits, barrett_mu, other_log, config.log_n);
    }
    CUNTT_CUDA_CHECK(cudaGetLastError());
}

template <typename Word, std::uint32_t LocalLog, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett>
void dispatch_hybrid2d_rows(const PlanConfig& config, bool first_pass, const Word* input, Word* scratch, Word* output, const Word* twiddles,
                            const Word* twiddles_shoup, const Word* root_powers, const Word* root_powers_shoup, std::uint64_t inverse_n,
                            std::uint64_t inverse_n_shoup) {
#define CUNTT_LAUNCH_HYBRID(ROWS)                                                                                                              \
    launch_hybrid2d_pass<Word, LocalLog, ROWS, Unit, Placement, Barrett>(config, first_pass, input, scratch, output, twiddles, twiddles_shoup, \
                                                                         root_powers, root_powers_shoup, inverse_n, inverse_n_shoup)
    switch (config.rows_per_block) {
        case 1:
            CUNTT_LAUNCH_HYBRID(1);
            break;
        case 2:
            CUNTT_LAUNCH_HYBRID(2);
            break;
        case 4:
            CUNTT_LAUNCH_HYBRID(4);
            break;
        default:
            throw std::logic_error("unsupported Hybrid2D rows_per_block dispatch");
    }
#undef CUNTT_LAUNCH_HYBRID
}

template <typename Word, ComputeUnit Unit, CrossTwiddlePlacement Placement, bool Barrett>
void dispatch_hybrid2d_log(const PlanConfig& config, std::uint32_t local_log, bool first_pass, const Word* input, Word* scratch, Word* output,
                           const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers, const Word* root_powers_shoup,
                           std::uint64_t inverse_n, std::uint64_t inverse_n_shoup) {
#define CUNTT_DISPATCH_LOCAL(LOG)                                                                                                                  \
    dispatch_hybrid2d_rows<Word, LOG, Unit, Placement, Barrett>(config, first_pass, input, scratch, output, twiddles, twiddles_shoup, root_powers, \
                                                                root_powers_shoup, inverse_n, inverse_n_shoup)
    switch (local_log) {
        case 6:
            CUNTT_DISPATCH_LOCAL(6);
            break;
        case 7:
            CUNTT_DISPATCH_LOCAL(7);
            break;
        case 8:
            CUNTT_DISPATCH_LOCAL(8);
            break;
        case 9:
            CUNTT_DISPATCH_LOCAL(9);
            break;
        case 10:
            CUNTT_DISPATCH_LOCAL(10);
            break;
        default:
            throw std::logic_error("unsupported Hybrid2D local transform dispatch");
    }
#undef CUNTT_DISPATCH_LOCAL
}

}  // namespace

class Plan::Impl {
  public:
    explicit Impl(PlanConfig config) : config_(std::move(config)) {
        const bool automatic_selection = config_.auto_select;
        if (automatic_selection) {
            auto selected = detail::select_ntt_mapping(std::move(config_));
            config_       = std::move(selected.config);
            selection_    = std::move(selected.info);
        }
        resolve_hybrid2d_mapping();
        validate_config();
        n_ = 1ULL << config_.log_n;
        if (config_.batch > std::numeric_limits<std::size_t>::max() / n_) {
            throw std::overflow_error("batch * N overflows size_t");
        }
        total_points_ = n_ * config_.batch;
        if (total_points_ > std::numeric_limits<std::size_t>::max() / (config_.word_bits / 8)) {
            throw std::overflow_error("device allocation size overflows size_t");
        }

        build_twiddles();
        const std::size_t word_bytes          = config_.word_bits / 8;
        data_bytes_                           = total_points_ * word_bytes;
        workspace_bytes_ = (config_.backend == Backend::Hybrid2D ||
                            (config_.backend == Backend::CompactStage && config_.output_order == OutputOrder::Natural))
                               ? data_bytes_
                               : 0;
        config_.auto_select = automatic_selection;
        const std::size_t twiddle_bytes       = twiddles_.size() * word_bytes;
        const std::size_t twiddle_shoup_bytes = twiddles_shoup_.size() * word_bytes;
        if (config_.auto_allocate_workspace && workspace_bytes_ != 0) {
            device_scratch_.allocate(workspace_bytes_);
        }
        if (config_.backend == Backend::CompactStage) {
            const std::size_t root_bytes = compact_roots_.size() * sizeof(std::uint64_t);
            device_compact_roots_.allocate(root_bytes);
            device_compact_roots_shoup_.allocate(root_bytes);
            CUNTT_CUDA_CHECK(cudaMemcpy(device_compact_roots_.get(), compact_roots_.data(), root_bytes, cudaMemcpyHostToDevice));
            CUNTT_CUDA_CHECK(cudaMemcpy(device_compact_roots_shoup_.get(), compact_roots_shoup_.data(), root_bytes, cudaMemcpyHostToDevice));
            return;
        }
        device_twiddles_.allocate(twiddle_bytes);
        device_twiddles_shoup_.allocate(twiddle_shoup_bytes);
        if (config_.word_bits == 32) {
            CUNTT_CUDA_CHECK(cudaMemcpy(device_twiddles_.as<std::uint32_t>(), twiddles32_.data(), twiddle_bytes, cudaMemcpyHostToDevice));
            CUNTT_CUDA_CHECK(
                cudaMemcpy(device_twiddles_shoup_.as<std::uint32_t>(), twiddles_shoup32_.data(), twiddle_shoup_bytes, cudaMemcpyHostToDevice));
        } else {
            CUNTT_CUDA_CHECK(cudaMemcpy(device_twiddles_.get(), twiddles_.data(), twiddle_bytes, cudaMemcpyHostToDevice));
            CUNTT_CUDA_CHECK(cudaMemcpy(device_twiddles_shoup_.get(), twiddles_shoup_.data(), twiddle_shoup_bytes, cudaMemcpyHostToDevice));
        }
        if (config_.backend == Backend::Hybrid2D) {
            const std::size_t root_power_bytes       = root_powers_.size() * word_bytes;
            const std::size_t root_power_shoup_bytes = root_powers_shoup_.size() * word_bytes;
            device_root_powers_.allocate(root_power_bytes);
            device_root_powers_shoup_.allocate(root_power_shoup_bytes);
            if (config_.word_bits == 32) {
                CUNTT_CUDA_CHECK(
                    cudaMemcpy(device_root_powers_.as<std::uint32_t>(), root_powers32_.data(), root_power_bytes, cudaMemcpyHostToDevice));
                CUNTT_CUDA_CHECK(cudaMemcpy(device_root_powers_shoup_.as<std::uint32_t>(), root_powers_shoup32_.data(), root_power_shoup_bytes,
                                            cudaMemcpyHostToDevice));
            } else {
                CUNTT_CUDA_CHECK(cudaMemcpy(device_root_powers_.get(), root_powers_.data(), root_power_bytes, cudaMemcpyHostToDevice));
                CUNTT_CUDA_CHECK(
                    cudaMemcpy(device_root_powers_shoup_.get(), root_powers_shoup_.data(), root_power_shoup_bytes, cudaMemcpyHostToDevice));
            }
        }
    }

    const PlanConfig& config() const noexcept { return config_; }
    const SelectionInfo& selection() const noexcept { return selection_; }

    std::size_t points_per_transform() const noexcept { return n_; }

    std::size_t data_size() const noexcept { return data_bytes_; }

    std::size_t workspace_size() const noexcept { return workspace_bytes_; }

    void set_stream(cudaStream_t stream) noexcept { stream_ = stream; }

    cudaStream_t stream() const noexcept { return stream_; }

    void set_workspace(void* workspace, std::size_t bytes) {
        if (workspace == nullptr) {
            if (bytes != 0) {
                throw std::invalid_argument("a null workspace must have zero bytes");
            }
            external_workspace_       = nullptr;
            external_workspace_bytes_ = 0;
            return;
        }
        if (bytes < workspace_bytes_) {
            throw std::invalid_argument("workspace is smaller than Plan::workspace_size()");
        }
        if (reinterpret_cast<std::uintptr_t>(workspace) % 16 != 0) {
            throw std::invalid_argument("workspace must be at least 16-byte aligned");
        }
        external_workspace_       = workspace;
        external_workspace_bytes_ = bytes;
    }

    void* workspace() const noexcept {
        return external_workspace_ != nullptr ? external_workspace_ : device_scratch_.data();
    }

    template <typename Word>
    void execute_async(const Word* input, Word* output, std::uint32_t expected_word_bits) {
        if (config_.word_bits != expected_word_bits) {
            throw std::invalid_argument("device pointer type does not match the configured NTT word width");
        }
        if (input == nullptr || output == nullptr) {
            throw std::invalid_argument("device input and output pointers must be non-null");
        }
        if (input == output) {
            throw std::invalid_argument("NTT device execution currently requires distinct input and output pointers");
        }
        void* scratch = workspace();
        if (workspace_bytes_ != 0 && scratch == nullptr) {
            throw std::logic_error("the selected NTT mapping requires a bound workspace");
        }
        LaunchStreamScope scope(stream_);
        launch_once(input, output, scratch);
    }

    RunStats execute(const std::vector<std::uint64_t>& input, std::vector<std::uint64_t>& output, std::uint32_t warmup, std::uint32_t repeat) {
        if (input.size() != total_points_) {
            throw std::invalid_argument("input size does not match plan batch * N");
        }
        if (repeat == 0) {
            throw std::invalid_argument("repeat must be positive");
        }
        for (const auto value : input) {
            if (value >= config_.modulus) {
                throw std::invalid_argument("input coefficients must be reduced modulo the plan modulus");
            }
        }
        output.resize(total_points_);
        std::vector<std::uint32_t> input32;
        std::vector<std::uint32_t> output32;
        const void*                host_input  = input.data();
        void*                      host_output = output.data();
        if (config_.word_bits == 32) {
            input32.assign(input.begin(), input.end());
            output32.resize(total_points_);
            host_input  = input32.data();
            host_output = output32.data();
        }

        ensure_host_buffers();
        void* scratch = workspace();
        if (workspace_bytes_ != 0 && scratch == nullptr) {
            throw std::logic_error("the selected NTT mapping requires a bound workspace");
        }

        Event start;
        Event stop;
        LaunchStreamScope scope(stream_);

        CUNTT_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
        CUNTT_CUDA_CHECK(cudaMemcpyAsync(device_input_.get(), host_input, data_bytes_, cudaMemcpyHostToDevice, stream_));
        CUNTT_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
        CUNTT_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float h2d_ms = 0.0F;
        CUNTT_CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start.get(), stop.get()));

        for (std::uint32_t i = 0; i < warmup; ++i) {
            launch_once(device_input_.data(), device_work_.data(), scratch);
        }
        CUNTT_CUDA_CHECK(cudaStreamSynchronize(stream_));

        CUNTT_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
        for (std::uint32_t i = 0; i < repeat; ++i) {
            launch_once(device_input_.data(), device_work_.data(), scratch);
        }
        CUNTT_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
        CUNTT_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float total_kernel_ms = 0.0F;
        CUNTT_CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, start.get(), stop.get()));

        CUNTT_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
        CUNTT_CUDA_CHECK(cudaMemcpyAsync(host_output, device_work_.data(), data_bytes_, cudaMemcpyDeviceToHost, stream_));
        CUNTT_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
        CUNTT_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        float d2h_ms = 0.0F;
        CUNTT_CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start.get(), stop.get()));
        if (config_.word_bits == 32) {
            std::copy(output32.begin(), output32.end(), output.begin());
        }

        RunStats stats;
        stats.h2d_ms                    = h2d_ms;
        stats.kernel_ms                 = total_kernel_ms / repeat;
        stats.d2h_ms                    = d2h_ms;
        const double kernel_seconds     = stats.kernel_ms / 1000.0;
        const double end_to_end_seconds = (stats.h2d_ms + stats.kernel_ms + stats.d2h_ms) / 1000.0;
        stats.kernel_ntt_per_second     = config_.batch / kernel_seconds;
        stats.end_to_end_ntt_per_second = config_.batch / end_to_end_seconds;
        stats.kernel_points_per_second  = stats.kernel_ntt_per_second * n_;
        return stats;
    }

  private:
    void ensure_host_buffers() {
        if (device_input_.data() == nullptr) {
            device_input_.allocate(data_bytes_);
            device_work_.allocate(data_bytes_);
        }
    }

    void resolve_hybrid2d_mapping() {
        if (config_.cross_twiddle_placement == CrossTwiddlePlacement::FusedBarrett) {
            config_.cross_twiddle_placement = CrossTwiddlePlacement::Fused;
            config_.modular_multiply        = ModularMultiply::Barrett;
        }
        if (config_.backend == Backend::StagePipeline) {
            config_.stage_space       = config_.stage_space == 0 ? 2 : config_.stage_space;
            config_.compute_unit      = ComputeUnit::Radix2;
            config_.n1_log            = 0;
            config_.rows_per_block    = 0;
            config_.threads_per_block = 256;
            return;
        }
        if (config_.backend == Backend::CompactStage) {
            config_.compute_unit      = ComputeUnit::Radix2;
            config_.n1_log            = 0;
            config_.rows_per_block    = 0;
            config_.threads_per_block = 256;
            return;
        }
        if (config_.backend != Backend::Hybrid2D) {
            return;
        }
        int device = 0;
        CUNTT_CUDA_CHECK(cudaGetDevice(&device));
        CUNTT_CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device));
        const bool is_v100 = device_properties_.major == 7 && device_properties_.minor == 0;
        if (config_.n1_log == 0) {
            config_.n1_log = config_.log_n >= 2 ? config_.log_n / 2 : 1;
        }
        if (config_.compute_unit == ComputeUnit::Auto) {
            config_.compute_unit = is_v100 && !(config_.word_bits == 64 && config_.log_n == 12) ? ComputeUnit::Radix4 : ComputeUnit::Radix2;
        }
        if (config_.threads_per_block == 0) {
            const bool          fused             = config_.cross_twiddle_placement == CrossTwiddlePlacement::Fused;
            const std::uint32_t preferred_threads = is_v100 && fused && config_.log_n >= 18 ? 512 : 256;
            config_.threads_per_block             = std::min<std::uint32_t>(preferred_threads, device_properties_.maxThreadsPerBlock);
            config_.threads_per_block -= config_.threads_per_block % device_properties_.warpSize;
        }
        if (config_.rows_per_block == 0) {
            const bool fused = config_.cross_twiddle_placement == CrossTwiddlePlacement::Fused;
            if (is_v100 && fused && config_.log_n == 20) {
                config_.rows_per_block = 2;
            } else if (config_.log_n <= 30 && config_.n1_log <= config_.log_n) {
                const std::uint32_t larger_factor  = 1U << std::max(config_.n1_log, config_.log_n - config_.n1_log);
                const std::size_t   four_row_bytes = 4ULL * (larger_factor + 1) * (config_.word_bits / 8);
                config_.rows_per_block             = four_row_bytes <= device_properties_.sharedMemPerBlock ? 4 : 2;
            } else {
                config_.rows_per_block = 1;
            }
        }
    }

    void validate_config() const {
        if (config_.log_n == 0 || config_.log_n > 30) {
            throw std::invalid_argument("log_n must be in [1, 30]");
        }
        if (config_.batch == 0) {
            throw std::invalid_argument("batch must be positive");
        }
        if (config_.modulus < 3 || config_.modulus >= (1ULL << 63)) {
            throw std::invalid_argument("modulus must be in [3, 2^63)");
        }
        if (!is_prime(config_.modulus)) {
            throw std::invalid_argument("modulus must be prime");
        }
        const std::uint64_t n = 1ULL << config_.log_n;
        if ((config_.modulus - 1) % n != 0) {
            throw std::invalid_argument("modulus - 1 is not divisible by N");
        }
        if (config_.word_bits != 32 && config_.word_bits != 64) {
            throw std::invalid_argument("word_bits must be 32 or 64");
        }
        if (config_.word_bits == 32 && (config_.backend != Backend::Hybrid2D || config_.modulus >= (1ULL << 31))) {
            throw std::invalid_argument("32-bit words require hybrid2d and modulus < 2^31");
        }
        if (config_.output_order == OutputOrder::BitReversed && config_.backend != Backend::CompactStage) {
            throw std::invalid_argument("bit-reversed output is currently supported only by compact-stage");
        }
        if (config_.backend == Backend::CompactStage) {
            if (config_.log_n != 20 || config_.word_bits != 64 || config_.inverse) {
                throw std::invalid_argument("compact-stage currently requires a forward, 64-bit logN=20 transform");
            }
            if (modulus_bit_length(config_.modulus) > 62) {
                throw std::invalid_argument("compact-stage supports moduli of at most 62 bits");
            }
            if (config_.batch > 65535) {
                throw std::invalid_argument("compact-stage batch exceeds the CUDA grid-z limit");
            }
        }
        if (config_.backend == Backend::StagePipeline) {
            if (config_.log_n != 8 || config_.word_bits != 64 || config_.inverse || config_.output_order != OutputOrder::Natural) {
                throw std::invalid_argument("stage-pipeline currently requires a forward, 64-bit, natural-order logN=8 transform");
            }
            if (config_.stage_space != 1 && config_.stage_space != 2 && config_.stage_space != 4 && config_.stage_space != 8) {
                throw std::invalid_argument("stage-pipeline stage_space must be 1, 2, 4, or 8");
            }
        }
        if (config_.modular_multiply == ModularMultiply::Barrett) {
            const std::uint32_t max_bits = config_.word_bits == 32 ? 30 : 62;
            if (modulus_bit_length(config_.modulus) > max_bits) {
                throw std::invalid_argument(
                    "Barrett multiplication supports at most 30-bit moduli with 32-bit words and 62-bit moduli with 64-bit words");
            }
            if (config_.backend != Backend::Hybrid2D) {
                throw std::invalid_argument("Barrett multiplication currently requires the hybrid2d backend");
            }
        }
        if (config_.backend == Backend::Hybrid2D) {
            if (config_.log_n < 2 * kHybridMinLocalLog || config_.log_n > 2 * kHybridMaxLocalLog) {
                throw std::invalid_argument("hybrid2d backend requires logN in [12, 20]");
            }
            const std::uint32_t n2_log = config_.log_n - config_.n1_log;
            if (config_.n1_log < kHybridMinLocalLog || config_.n1_log > kHybridMaxLocalLog || n2_log < kHybridMinLocalLog ||
                n2_log > kHybridMaxLocalLog) {
                throw std::invalid_argument("hybrid2d n1_log must be in [6, 10]");
            }
            if (config_.rows_per_block != 1 && config_.rows_per_block != 2 && config_.rows_per_block != 4) {
                throw std::invalid_argument("hybrid2d rows_per_block must be 1, 2, or 4");
            }
            if (config_.threads_per_block == 0 || config_.threads_per_block > static_cast<std::uint32_t>(device_properties_.maxThreadsPerBlock) ||
                config_.threads_per_block % static_cast<std::uint32_t>(device_properties_.warpSize) != 0) {
                throw std::invalid_argument("hybrid2d threads_per_block must be a supported multiple of the warp size");
            }
            const std::uint32_t larger_factor = 1U << std::max(config_.n1_log, config_.log_n - config_.n1_log);
            const std::size_t   shared_bytes  = static_cast<std::size_t>(config_.rows_per_block) * (larger_factor + 1) * (config_.word_bits / 8);
            if (shared_bytes > device_properties_.sharedMemPerBlock) {
                throw std::invalid_argument("hybrid2d mapping exceeds the device shared-memory limit per block");
            }
        }
    }

    void build_twiddles() {
        std::uint64_t root = find_primitive_power_of_two_root(config_.log_n, config_.modulus);
        if (config_.inverse) {
            root = mod_pow(root, config_.modulus - 2, config_.modulus);
        }
        const std::size_t n = 1ULL << config_.log_n;
        if (config_.backend == Backend::CompactStage) {
            std::vector<std::uint64_t> sequential_roots(n / 2);
            sequential_roots[0] = 1;
            for (std::size_t index = 1; index < sequential_roots.size(); ++index) {
                sequential_roots[index] =
                    static_cast<std::uint64_t>((static_cast<unsigned __int128>(sequential_roots[index - 1]) * root) % config_.modulus);
            }
            compact_roots_.resize(sequential_roots.size());
            compact_roots_shoup_.resize(sequential_roots.size());
            for (std::uint32_t index = 0; index < compact_roots_.size(); ++index) {
                compact_roots_[index]       = sequential_roots[reverse_bits_host(index, config_.log_n - 1)];
                compact_roots_shoup_[index] = shoup_precompute(compact_roots_[index], config_.modulus);
            }
            return;
        }
        twiddles_.resize(n - 1);
        twiddles_shoup_.resize(n - 1);
        for (std::uint32_t stage = 0; stage < config_.log_n; ++stage) {
            const std::size_t   half      = 1ULL << stage;
            const std::size_t   offset    = half - 1;
            const std::uint64_t step_root = mod_pow(root, n / (half << 1), config_.modulus);
            std::uint64_t       omega     = 1;
            for (std::size_t i = 0; i < half; ++i) {
                twiddles_[offset + i]       = omega;
                twiddles_shoup_[offset + i] = shoup_precompute(omega, config_.modulus);
                omega                       = static_cast<std::uint64_t>((static_cast<unsigned __int128>(omega) * step_root) % config_.modulus);
            }
        }
        if (config_.backend == Backend::Hybrid2D) {
            root_powers_.resize(n);
            root_powers_shoup_.resize(n);
            std::uint64_t omega = 1;
            for (std::size_t i = 0; i < n; ++i) {
                root_powers_[i]       = omega;
                root_powers_shoup_[i] = shoup_precompute(omega, config_.modulus);
                omega                 = static_cast<std::uint64_t>((static_cast<unsigned __int128>(omega) * root) % config_.modulus);
            }
            const bool fused = config_.cross_twiddle_placement == CrossTwiddlePlacement::Fused;
            if (fused) {
                const std::uint32_t        n1 = 1U << config_.n1_log;
                const std::uint32_t        n2 = 1U << (config_.log_n - config_.n1_log);
                std::vector<std::uint64_t> fused_twiddles(static_cast<std::size_t>(n2) * (n1 - 1));
                std::vector<std::uint64_t> fused_twiddles_shoup;
                if (config_.modular_multiply == ModularMultiply::Shoup) {
                    fused_twiddles_shoup.resize(fused_twiddles.size());
                }
                for (std::uint32_t k2 = 0; k2 < n2; ++k2) {
                    const std::size_t row_offset = static_cast<std::size_t>(k2) * (n1 - 1);
                    for (std::uint32_t half = 1; half < n1; half <<= 1) {
                        const std::uint64_t exponent_factor = n1 / (half << 1);
                        for (std::uint32_t offset = 0; offset < half; ++offset) {
                            const std::uint64_t exponent = exponent_factor * (static_cast<std::uint64_t>(n2) * offset + k2);
                            const std::size_t   index    = row_offset + (half - 1) + offset;
                            fused_twiddles[index]        = root_powers_[exponent];
                            if (config_.modular_multiply == ModularMultiply::Shoup) {
                                fused_twiddles_shoup[index] = root_powers_shoup_[exponent];
                            }
                        }
                    }
                }
                root_powers_.swap(fused_twiddles);
                if (config_.modular_multiply == ModularMultiply::Barrett) {
                    root_powers_shoup_.assign(1, 0);
                } else {
                    root_powers_shoup_.swap(fused_twiddles_shoup);
                }
            }
            if (config_.modular_multiply == ModularMultiply::Barrett) {
                twiddles_shoup_.assign(1, 0);
                if (!fused) {
                    root_powers_shoup_.assign(1, 0);
                }
            }
            if (config_.word_bits == 32) {
                twiddles32_.reserve(twiddles_.size());
                twiddles_shoup32_.reserve(config_.modular_multiply == ModularMultiply::Barrett ? 1 : twiddles_.size());
                for (const auto twiddle : twiddles_) {
                    const auto value = static_cast<std::uint32_t>(twiddle);
                    twiddles32_.push_back(value);
                    if (config_.modular_multiply == ModularMultiply::Shoup) {
                        twiddles_shoup32_.push_back(shoup_precompute32(value, static_cast<std::uint32_t>(config_.modulus)));
                    }
                }
                if (config_.modular_multiply == ModularMultiply::Barrett) {
                    twiddles_shoup32_.push_back(0);
                }
                root_powers32_.reserve(root_powers_.size());
                root_powers_shoup32_.reserve(config_.modular_multiply == ModularMultiply::Barrett ? 1 : root_powers_.size());
                for (const auto power : root_powers_) {
                    const auto value = static_cast<std::uint32_t>(power);
                    root_powers32_.push_back(value);
                    if (config_.modular_multiply != ModularMultiply::Barrett) {
                        root_powers_shoup32_.push_back(shoup_precompute32(value, static_cast<std::uint32_t>(config_.modulus)));
                    }
                }
                if (config_.modular_multiply == ModularMultiply::Barrett) {
                    root_powers_shoup32_.push_back(0);
                }
            }
        }
        if (config_.inverse) {
            inverse_n_       = mod_pow(n, config_.modulus - 2, config_.modulus);
            inverse_n_shoup_ = config_.word_bits == 32
                                   ? shoup_precompute32(static_cast<std::uint32_t>(inverse_n_), static_cast<std::uint32_t>(config_.modulus))
                                   : shoup_precompute(inverse_n_, config_.modulus);
        }
    }

    void launch_once(const void* input_pointer, void* output_pointer, void* scratch_pointer) {
        constexpr std::uint32_t kThreads = 256;
        const auto*             input    = static_cast<const std::uint64_t*>(input_pointer);
        auto*                   output   = static_cast<std::uint64_t*>(output_pointer);
        auto*                   scratch  = static_cast<std::uint64_t*>(scratch_pointer);
        if (config_.backend == Backend::StagePipeline) {
            const bool named_barrier = config_.stage_handoff == StageHandoff::NamedBarrier;
            switch (config_.stage_space) {
                case 1:
                    if (named_barrier) {
                        launch_stage_pipeline_ntt256<1, true>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    } else {
                        launch_stage_pipeline_ntt256<1, false>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    }
                    break;
                case 2:
                    if (named_barrier) {
                        launch_stage_pipeline_ntt256<2, true>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    } else {
                        launch_stage_pipeline_ntt256<2, false>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    }
                    break;
                case 4:
                    if (named_barrier) {
                        launch_stage_pipeline_ntt256<4, true>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    } else {
                        launch_stage_pipeline_ntt256<4, false>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    }
                    break;
                case 8:
                    if (named_barrier) {
                        launch_stage_pipeline_ntt256<8, true>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    } else {
                        launch_stage_pipeline_ntt256<8, false>(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get());
                    }
                    break;
                default:
                    throw std::logic_error("unresolved stage-pipeline stage_space");
            }
            return;
        }
        if (config_.backend == Backend::CompactStage) {
            if (config_.output_order == OutputOrder::Natural) {
                launch_compact_stage_log20(config_, input, scratch, device_compact_roots_.get(), device_compact_roots_shoup_.get(), output);
            } else {
                launch_compact_stage_log20(config_, input, output, device_compact_roots_.get(), device_compact_roots_shoup_.get(), nullptr);
            }
            return;
        }
        if (config_.backend == Backend::Hybrid2D) {
            const std::uint32_t n2_log = config_.log_n - config_.n1_log;
            const auto* input32   = static_cast<const std::uint32_t*>(input_pointer);
            auto*       output32  = static_cast<std::uint32_t*>(output_pointer);
            auto*       scratch32 = static_cast<std::uint32_t*>(scratch_pointer);
            const auto* input64   = input;
            auto*       output64  = output;
            auto*       scratch64 = scratch;
#define CUNTT_DISPATCH_PLACEMENT(WORD, TAG, UNIT, PLACEMENT, BARRETT)                                                                               \
    dispatch_hybrid2d_log<WORD, UNIT, PLACEMENT, BARRETT>(                                                                                          \
        config_, n2_log, true, input##TAG, scratch##TAG, output##TAG, device_twiddles_.as<WORD>(),                                                    \
        device_twiddles_shoup_.as<WORD>(), device_root_powers_.as<WORD>(), device_root_powers_shoup_.as<WORD>(), inverse_n_, inverse_n_shoup_);     \
    dispatch_hybrid2d_log<WORD, UNIT, PLACEMENT, BARRETT>(                                                                                          \
        config_, config_.n1_log, false, input##TAG, scratch##TAG, output##TAG, device_twiddles_.as<WORD>(),                                           \
        device_twiddles_shoup_.as<WORD>(), device_root_powers_.as<WORD>(), device_root_powers_shoup_.as<WORD>(), inverse_n_, inverse_n_shoup_)
#define CUNTT_DISPATCH_REDUCTION(WORD, TAG, UNIT, PLACEMENT)    \
    if (config_.modular_multiply == ModularMultiply::Barrett) { \
        CUNTT_DISPATCH_PLACEMENT(WORD, TAG, UNIT, PLACEMENT, true);  \
    } else {                                                    \
        CUNTT_DISPATCH_PLACEMENT(WORD, TAG, UNIT, PLACEMENT, false); \
    }
#define CUNTT_DISPATCH_UNIT(WORD, TAG, UNIT)                                           \
    if (config_.cross_twiddle_placement == CrossTwiddlePlacement::FirstPass) {         \
        CUNTT_DISPATCH_REDUCTION(WORD, TAG, UNIT, CrossTwiddlePlacement::FirstPass);   \
    } else if (config_.cross_twiddle_placement == CrossTwiddlePlacement::SecondPass) { \
        CUNTT_DISPATCH_REDUCTION(WORD, TAG, UNIT, CrossTwiddlePlacement::SecondPass);  \
    } else {                                                                           \
        CUNTT_DISPATCH_REDUCTION(WORD, TAG, UNIT, CrossTwiddlePlacement::Fused);       \
    }
            if (config_.word_bits == 32) {
                switch (config_.compute_unit) {
                    case ComputeUnit::Radix2:
                        CUNTT_DISPATCH_UNIT(std::uint32_t, 32, ComputeUnit::Radix2);
                        break;
                    case ComputeUnit::Radix4:
                        CUNTT_DISPATCH_UNIT(std::uint32_t, 32, ComputeUnit::Radix4);
                        break;
                    case ComputeUnit::Radix8:
                        CUNTT_DISPATCH_UNIT(std::uint32_t, 32, ComputeUnit::Radix8);
                        break;
                    case ComputeUnit::Auto:
                        throw std::logic_error("unresolved Hybrid2D compute unit");
                }
            } else {
                switch (config_.compute_unit) {
                    case ComputeUnit::Radix2:
                        CUNTT_DISPATCH_UNIT(std::uint64_t, 64, ComputeUnit::Radix2);
                        break;
                    case ComputeUnit::Radix4:
                        CUNTT_DISPATCH_UNIT(std::uint64_t, 64, ComputeUnit::Radix4);
                        break;
                    case ComputeUnit::Radix8:
                        CUNTT_DISPATCH_UNIT(std::uint64_t, 64, ComputeUnit::Radix8);
                        break;
                    case ComputeUnit::Auto:
                        throw std::logic_error("unresolved Hybrid2D compute unit");
                }
            }
#undef CUNTT_DISPATCH_UNIT
#undef CUNTT_DISPATCH_REDUCTION
#undef CUNTT_DISPATCH_PLACEMENT
            return;
        }
        bit_reverse_copy_kernel<<<grid_for(total_points_, kThreads), kThreads, 0, active_stream>>>(input, output, total_points_, config_.log_n);
        CUNTT_CUDA_CHECK(cudaGetLastError());

        std::uint32_t first_global_stage = 0;
        if (config_.backend == Backend::Tile256) {
            const std::uint32_t tile_log   = std::min<std::uint32_t>(config_.log_n, 8);
            const std::uint64_t tile_size  = 1ULL << tile_log;
            const std::uint64_t tile_count = total_points_ / tile_size;
            tile_kernel<<<grid_for(tile_count, 1), 128, 0, active_stream>>>(output, tile_log, device_twiddles_.get(), device_twiddles_shoup_.get(),
                                                                            config_.modulus);
            CUNTT_CUDA_CHECK(cudaGetLastError());
            first_global_stage = tile_log;
        }

        const std::uint64_t butterflies_per_transform = n_ >> 1;
        const std::uint64_t total_butterflies         = butterflies_per_transform * config_.batch;
        for (std::uint32_t stage = first_global_stage; stage < config_.log_n; ++stage) {
            const std::uint64_t half = 1ULL << stage;
            stage_kernel<<<grid_for(total_butterflies, kThreads), kThreads, 0, active_stream>>>(
                output, butterflies_per_transform, total_butterflies, n_, half, half - 1, device_twiddles_.get(), device_twiddles_shoup_.get(),
                config_.modulus);
            CUNTT_CUDA_CHECK(cudaGetLastError());
        }

        if (config_.inverse) {
            scale_kernel<<<grid_for(total_points_, kThreads), kThreads, 0, active_stream>>>(output, total_points_, inverse_n_, inverse_n_shoup_,
                                                                                            config_.modulus);
            CUNTT_CUDA_CHECK(cudaGetLastError());
        }
    }

    PlanConfig                 config_;
    SelectionInfo              selection_;
    std::size_t                n_            = 0;
    std::size_t                total_points_ = 0;
    std::size_t                data_bytes_ = 0;
    std::size_t                workspace_bytes_ = 0;
    std::vector<std::uint64_t> twiddles_;
    std::vector<std::uint64_t> twiddles_shoup_;
    std::vector<std::uint64_t> root_powers_;
    std::vector<std::uint64_t> root_powers_shoup_;
    std::vector<std::uint64_t> compact_roots_;
    std::vector<std::uint64_t> compact_roots_shoup_;
    std::vector<std::uint32_t> twiddles32_;
    std::vector<std::uint32_t> twiddles_shoup32_;
    std::vector<std::uint32_t> root_powers32_;
    std::vector<std::uint32_t> root_powers_shoup32_;
    std::uint64_t              inverse_n_       = 1;
    std::uint64_t              inverse_n_shoup_ = 0;
    DeviceBuffer               device_input_;
    DeviceBuffer               device_work_;
    DeviceBuffer               device_twiddles_;
    DeviceBuffer               device_twiddles_shoup_;
    DeviceBuffer               device_scratch_;
    DeviceBuffer               device_root_powers_;
    DeviceBuffer               device_root_powers_shoup_;
    DeviceBuffer               device_compact_roots_;
    DeviceBuffer               device_compact_roots_shoup_;
    cudaDeviceProp             device_properties_{};
    cudaStream_t               stream_ = nullptr;
    void*                      external_workspace_ = nullptr;
    std::size_t                external_workspace_bytes_ = 0;
};

DeviceInfo current_device_info() {
    DeviceInfo info;
    CUNTT_CUDA_CHECK(cudaGetDevice(&info.device_id));
    cudaDeviceProp properties{};
    CUNTT_CUDA_CHECK(cudaGetDeviceProperties(&properties, info.device_id));
    info.name                = properties.name;
    info.compute_major       = properties.major;
    info.compute_minor       = properties.minor;
    info.global_memory_bytes = properties.totalGlobalMem;
    info.multiprocessors     = properties.multiProcessorCount;
    return info;
}

Plan::Plan(PlanConfig config) : impl_(std::make_unique<Impl>(std::move(config))) {
}

Plan::~Plan() = default;

Plan::Plan(Plan&&) noexcept = default;

Plan& Plan::operator=(Plan&&) noexcept = default;

const PlanConfig& Plan::config() const noexcept {
    return impl_->config();
}

const SelectionInfo& Plan::selection() const noexcept {
    return impl_->selection();
}

std::size_t Plan::points_per_transform() const noexcept {
    return impl_->points_per_transform();
}

std::size_t Plan::data_size() const noexcept {
    return impl_->data_size();
}

std::size_t Plan::workspace_size() const noexcept {
    return impl_->workspace_size();
}

void Plan::set_stream(cudaStream_t stream) noexcept {
    impl_->set_stream(stream);
}

cudaStream_t Plan::stream() const noexcept {
    return impl_->stream();
}

void Plan::set_workspace(void* workspace, std::size_t bytes) {
    impl_->set_workspace(workspace, bytes);
}

void* Plan::workspace() const noexcept {
    return impl_->workspace();
}

void Plan::execute_async(const std::uint32_t* input, std::uint32_t* output) {
    impl_->execute_async(input, output, 32);
}

void Plan::execute_async(const std::uint64_t* input, std::uint64_t* output) {
    impl_->execute_async(input, output, 64);
}

RunStats Plan::execute(const std::vector<std::uint64_t>& input, std::vector<std::uint64_t>& output, std::uint32_t warmup, std::uint32_t repeat) {
    return impl_->execute(input, output, warmup, repeat);
}

}  // namespace cuntt
