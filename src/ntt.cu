#include <cuda_runtime.h>
#include <cuda/atomic>

#include <algorithm>
#include <chrono>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "cuntt/ntt.hpp"
#include "generated_hybrid_dataflow_config.hpp"
#include "hybrid_dataflow.cuh"
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

template <typename Word>
struct NttStageOperatorT {
    using Value = Word;
    static constexpr bool kBitReverseInput = true;

    const Word* twiddles;
    const Word* twiddles_shoup;
    Word        modulus;
    Word        scale;
    Word        scale_shoup;

    __device__ __forceinline__ void apply(std::uint32_t stage, std::uint32_t offset, Value& left, Value& right) const {
        const std::uint32_t twiddle_base = (1U << stage) - 1;
        const Value u = left;
        const Value v = mul_shoup(right, twiddles[twiddle_base + offset], twiddles_shoup[twiddle_base + offset], modulus);
        const Value sum = u + v;
        left  = sum >= modulus ? sum - modulus : sum;
        right = u >= v ? u - v : modulus + u - v;
    }

    __device__ __forceinline__ Value finalize(Value value, std::uint32_t) const {
        return scale == 1 ? value : mul_shoup(value, scale, scale_shoup, modulus);
    }

    __device__ __forceinline__ Value multiply(Value value, Value coefficient,
                                               Value coefficient_shoup) const {
        return mul_shoup(value, coefficient, coefficient_shoup, modulus);
    }

    __device__ __forceinline__ void apply_coefficient(
        Value coefficient, Value coefficient_shoup, Value& left, Value& right) const {
        const Value u = left;
        const Value v = mul_shoup(right, coefficient, coefficient_shoup, modulus);
        const Value sum = u + v;
        left = sum >= modulus ? sum - modulus : sum;
        right = u >= v ? u - v : modulus + u - v;
    }
};

using NttStageOperator = NttStageOperatorT<std::uint64_t>;

constexpr std::uint32_t kMaxStreamingSegments = 8;

struct NttStreamingDescriptor {
    std::uint32_t count;
    std::uint32_t log_n;
    std::uint32_t segment_log[kMaxStreamingSegments];
    std::uint32_t prefix_log[kMaxStreamingSegments];
    std::uint32_t remaining_log[kMaxStreamingSegments];
    std::uint32_t core[kMaxStreamingSegments];
    std::uint32_t role_begin[kMaxStreamingSegments + 1];
    std::uint32_t role_blocks[kMaxStreamingSegments];
    std::uint64_t counter_base[kMaxStreamingSegments + 1];
    std::uint64_t boundary_word_base[kMaxStreamingSegments];
    std::uint64_t state_base[kMaxStreamingSegments + 1];
    std::uint32_t boundary_slots[kMaxStreamingSegments];
    std::uint32_t producer_tasks_per_transform[kMaxStreamingSegments];
    std::uint32_t digit_input_shift[kMaxStreamingSegments];
    std::uint32_t digit_output_shift[kMaxStreamingSegments];
    std::uint32_t digit_mask[kMaxStreamingSegments];
};

__device__ __forceinline__ std::uint32_t reverse_streaming_digits(
    std::uint32_t logical, const NttStreamingDescriptor& descriptor) {
    std::uint32_t reordered = 0;
#pragma unroll
    for (std::uint32_t segment = 0; segment < kMaxStreamingSegments; ++segment) {
        if (segment < descriptor.count) {
            const std::uint32_t digit =
                (logical >> descriptor.digit_input_shift[segment]) & descriptor.digit_mask[segment];
            reordered |= digit << descriptor.digit_output_shift[segment];
        }
    }
    return reordered;
}

__device__ __forceinline__ unsigned long long streaming_globaltimer() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

bool uses_appt_pipeline(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(), config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::ApptPipeline;
                       });
}

bool uses_appt_online(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(), config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::ApptOnline ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRadix4 ||
                                  mapping.core == NttSubgraphCore::ApptOnlineFusedTail ||
                                  mapping.core == NttSubgraphCore::ApptOnlineSplitTail ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRegisterTail ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                                  mapping.core == NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinal ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalResident ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
                       });
}

bool uses_appt_fused_tail(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                  NttSubgraphCore::ApptOnlineFusedTail;
                       });
}

bool uses_appt_split_tail(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                  NttSubgraphCore::ApptOnlineSplitTail;
                       });
}

bool uses_appt_register_tail(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                      NttSubgraphCore::ApptOnlineRegisterTail ||
                                  mapping.core ==
                                      NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                                  mapping.core ==
                                      NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
                                  mapping.core ==
                                      NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                                  mapping.core ==
                                      NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinal ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalResident ||
                                  mapping.core == NttSubgraphCore::
                                      ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
                       });
}

bool uses_appt_register_tail_warp(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                  NttSubgraphCore::ApptOnlineRegisterTailWarp;
                       });
}

bool uses_appt_register_tail_column_warp(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::
                                                      ApptOnlineRegisterTailColumnWarp;
                       });
}

bool uses_appt_register_tail_radix8(const PlanConfig& config) {
    return !config.subgraph_mappings.empty() &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                  NttSubgraphCore::ApptOnlineRegisterTailRadix8;
                       });
}

bool uses_appt_register_tail_grouped(const PlanConfig& config) {
    return config.backend == Backend::HierarchicalDataflow &&
           !config.subgraph_mappings.empty() &&
           (config.subgraph_mappings.front().core ==
                NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinal ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResident ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter);
}

bool uses_appt_register_tail_grouped_writer_final(const PlanConfig& config) {
    return config.backend == Backend::HierarchicalDataflow &&
           !config.subgraph_mappings.empty() &&
           (config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinal ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResident ||
            config.subgraph_mappings.front().core == NttSubgraphCore::
                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter);
}

bool uses_appt_register_tail_grouped_writer_final_data_time(
    const PlanConfig& config) {
    return config.backend == Backend::HierarchicalDataflow &&
           !config.subgraph_mappings.empty() &&
           config.subgraph_mappings.front().core == NttSubgraphCore::
               ApptOnlineRegisterTailGroupedWriterFinalDataTime;
}

bool uses_appt_register_tail_grouped_writer_final_resident(
    const PlanConfig& config) {
    return config.backend == Backend::HierarchicalDataflow &&
           !config.subgraph_mappings.empty() &&
           config.subgraph_mappings.front().core == NttSubgraphCore::
               ApptOnlineRegisterTailGroupedWriterFinalResident;
}

bool uses_appt_register_tail_grouped_writer_final_resident_quarter(
    const PlanConfig& config) {
    return config.backend == Backend::HierarchicalDataflow &&
           !config.subgraph_mappings.empty() &&
           config.subgraph_mappings.front().core == NttSubgraphCore::
               ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
}

// A=(Us/Ts, Ud/Td) on CUDA. One warp is one physical stage (Us); independent
// tokens traverse shared-memory channels (Ud/Td); the same warp stages are
// reused after a grid-wide fold boundary (Ts). No token synchronizes with an
// unrelated token and no CTA is permanently assigned to a logical segment.
template <typename Word, std::uint32_t StageSpace, std::uint32_t RoleStages,
          std::uint32_t StaticLogN = 0>
__global__ void appt_pipeline_kernel(
    const Word* input, Word* output, Word* scratch, std::uint64_t transforms,
    std::uint32_t log_n, std::uint32_t data_time,
    std::uint32_t token_interleave, std::uint32_t pipeline_buffers,
    std::uint32_t pipeline_replicas,
    unsigned long long* trace, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kTokenPoints = 1U << StageSpace;
    constexpr std::uint32_t kValuesPerLane = (kTokenPoints + 31) / 32;
    constexpr std::uint32_t kRoles = (StageSpace + RoleStages - 1) / RoleStages;
    constexpr std::uint32_t kEdges = kRoles - 1;
    const auto grid = cg::this_grid();
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t role = warp / pipeline_replicas;
    const std::uint32_t replica = warp - role * pipeline_replicas;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t effective_log_n = StaticLogN == 0 ? log_n : StaticLogN;
    const std::uint64_t n = 1ULL << effective_log_n;
    const std::uint32_t fold_count =
        (effective_log_n + StageSpace - 1) / StageSpace;
    Word* state = (fold_count & 1U) != 0 ? scratch : output;

    extern __shared__ __align__(16) unsigned char shared_storage[];
    Word* channels = reinterpret_cast<Word*>(shared_storage);
    const std::size_t channel_words = static_cast<std::size_t>(kEdges) *
                                      pipeline_replicas * pipeline_buffers *
                                      token_interleave * kTokenPoints;
    int* flags = reinterpret_cast<int*>(channels + channel_words);

    for (std::uint32_t item = threadIdx.x;
         item < 2 * fold_count;
         item += blockDim.x) {
        trace[item] = (item & 1U) == 0 ? ~0ULL : 0;
    }
    grid.sync();

#pragma unroll
    for (std::uint32_t stage_base = 0, fold = 0;
         stage_base < effective_log_n; stage_base += StageSpace, ++fold) {
        const std::uint32_t active_stages =
            min(StageSpace, effective_log_n - stage_base);
        const bool final_fold = stage_base + active_stages == effective_log_n;
        const std::uint32_t active_roles =
            (active_stages + RoleStages - 1) / RoleStages;
        const std::uint32_t token_points = 1U << active_stages;
        const std::uint64_t tokens_per_transform = n >> active_stages;
        const std::uint64_t total_tokens = transforms * tokens_per_transform;
        const std::uint64_t packet_count = (total_tokens + data_time - 1) / data_time;

        for (std::uint32_t item = threadIdx.x;
             item < kEdges * pipeline_replicas * pipeline_buffers;
             item += blockDim.x) {
            flags[item] = 0;
        }
        __syncthreads();

        if (role < active_roles) {
            bool recorded_start = false;
            std::uint64_t processed_groups = 0;
            for (std::uint64_t packet =
                     static_cast<std::uint64_t>(blockIdx.x) * pipeline_replicas + replica;
                 packet < packet_count;
                 packet += static_cast<std::uint64_t>(gridDim.x) * pipeline_replicas) {
                for (std::uint32_t group_begin = 0; group_begin < data_time;
                     group_begin += token_interleave) {
                    if (packet * data_time + group_begin >= total_tokens) break;
                    const std::uint32_t ring_slot =
                        static_cast<std::uint32_t>(processed_groups % pipeline_buffers);
                    const std::size_t previous_flag =
                        ((static_cast<std::size_t>(role - (role != 0)) *
                          pipeline_replicas + replica) * pipeline_buffers) + ring_slot;
                    const std::size_t next_flag =
                        ((static_cast<std::size_t>(role) * pipeline_replicas + replica) *
                         pipeline_buffers) + ring_slot;

                    if (role != 0) {
                        detail::hybrid_wait_flag(&flags[previous_flag], 1, lane);
                    } else if (processed_groups >= pipeline_buffers) {
                        detail::hybrid_wait_flag(&flags[next_flag], 0, lane);
                    }
                    if (role + 1 < active_roles && processed_groups >= pipeline_buffers) {
                        detail::hybrid_wait_flag(&flags[next_flag], 0, lane);
                    }

                    for (std::uint32_t group_item = 0;
                         group_item < token_interleave; ++group_item) {
                        const std::uint32_t item = group_begin + group_item;
                        if (item >= data_time) continue;
                        const std::uint64_t sequence = packet * data_time + item;
                        if (sequence >= total_tokens) continue;
                        Word values[kValuesPerLane];
                        const std::uint64_t transform = sequence / tokens_per_transform;
                        const std::uint32_t token = static_cast<std::uint32_t>(
                            sequence - transform * tokens_per_transform);
                        const std::uint32_t stage_stride = 1U << stage_base;
                        const std::uint32_t subgraph_block = token / stage_stride;
                        const std::uint32_t subgraph_offset =
                            token - subgraph_block * stage_stride;
                        const std::uint32_t subgraph_base =
                            subgraph_block << (stage_base + active_stages);

#pragma unroll
                        for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
                            const std::uint32_t local = (slot << 5) + lane;
                            Word value = Word{0};
                            if (local < token_points) {
                                if (role == 0) {
                                    const std::uint32_t logical = subgraph_base +
                                        subgraph_offset + (local << stage_base);
                                    const std::uint32_t source = stage_base == 0
                                        ? (__brev(logical) >> (32 - effective_log_n)) : logical;
                                    value = stage_base == 0
                                        ? input[transform * n + source]
                                        : state[transform * n +
                                                static_cast<std::uint64_t>(token) *
                                                    token_points + local];
                                } else {
                                    const std::size_t channel =
                                        ((((static_cast<std::size_t>(role - 1) *
                                            pipeline_replicas + replica) * pipeline_buffers +
                                           ring_slot) * token_interleave +
                                          group_item) * kTokenPoints) + local;
                                    value = channels[channel];
                                }
                            }
                            values[slot] = value;
                        }
                        __syncwarp();

                        if (!recorded_start && lane == 0) {
                            atomicMin(trace + 2 * fold, streaming_globaltimer());
                            recorded_start = true;
                        }
#pragma unroll
                        for (std::uint32_t fused = 0; fused < RoleStages; ++fused) {
                            const std::uint32_t local_stage = role * RoleStages + fused;
                            if (local_stage < active_stages) {
                                detail::hybrid_apply_register_stage<Word, kValuesPerLane>(
                                    values, token_points, local_stage, stage_base,
                                    subgraph_offset, lane, op);
                            }
                        }

#pragma unroll
                        for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
                            const std::uint32_t local = (slot << 5) + lane;
                            if (local >= token_points) continue;
                            if (role + 1 < active_roles) {
                                const std::size_t channel =
                                    ((((static_cast<std::size_t>(role) * pipeline_replicas +
                                        replica) * pipeline_buffers + ring_slot) *
                                      token_interleave + group_item) *
                                     kTokenPoints) + local;
                                channels[channel] = values[slot];
                            } else if (!final_fold) {
                                state[transform * n + static_cast<std::uint64_t>(token) *
                                      token_points + local] = values[slot];
                            } else {
                                const std::uint32_t logical = subgraph_base +
                                    subgraph_offset + (local << stage_base);
                                output[transform * n + logical] =
                                    op.finalize(values[slot], static_cast<std::uint32_t>(n));
                            }
                        }
                        __syncwarp();
                    }
                    __threadfence_block();
                    if (role + 1 < active_roles && lane == 0) {
                        atomicExch(&flags[next_flag], 1);
                    }
                    if (role != 0 && lane == 0) {
                        atomicExch(&flags[previous_flag], 0);
                    }
                    ++processed_groups;
                }
            }
            if (recorded_start && lane == 0) {
                atomicMax(trace + 2 * fold + 1, streaming_globaltimer());
            }
        }
        __syncthreads();
        grid.sync();
        if (!final_fold) {
            Word* next_state = state == scratch ? output : scratch;
            const std::uint32_t next_stage_base = stage_base + active_stages;
            const std::uint32_t next_active_stages =
                min(StageSpace, effective_log_n - next_stage_base);
            constexpr std::uint32_t kTransposeTile = 32;
            constexpr std::uint32_t kTransposePitch = kTransposeTile + 1;
            Word* transpose_tile = channels;
            const std::uint32_t b_points = 1U << active_stages;
            const std::uint32_t d_points = 1U << next_active_stages;
            const std::uint32_t c_points = 1U << stage_base;
            const std::uint32_t a_points =
                1U << (effective_log_n - stage_base - active_stages -
                       next_active_stages);
            const std::uint32_t b_tiles = (b_points + kTransposeTile - 1) / kTransposeTile;
            const std::uint32_t d_tiles = (d_points + kTransposeTile - 1) / kTransposeTile;
            const std::uint64_t tiles_per_transform =
                static_cast<std::uint64_t>(a_points) * c_points * b_tiles * d_tiles;
            const std::uint64_t total_tiles = transforms * tiles_per_transform;

            for (std::uint64_t tile_index = blockIdx.x; tile_index < total_tiles;
                 tile_index += gridDim.x) {
                std::uint64_t coordinate = tile_index;
                const std::uint32_t d_tile = coordinate % d_tiles;
                coordinate /= d_tiles;
                const std::uint32_t b_tile = coordinate % b_tiles;
                coordinate /= b_tiles;
                const std::uint32_t c = coordinate % c_points;
                coordinate /= c_points;
                const std::uint32_t a = coordinate % a_points;
                const std::uint64_t transform = coordinate / a_points;
                const std::uint32_t b_base = b_tile * kTransposeTile;
                const std::uint32_t d_base = d_tile * kTransposeTile;

                for (std::uint32_t item = threadIdx.x;
                     item < kTransposeTile * kTransposeTile; item += blockDim.x) {
                    const std::uint32_t d_offset = item / kTransposeTile;
                    const std::uint32_t b_offset = item % kTransposeTile;
                    const std::uint32_t b = b_base + b_offset;
                    const std::uint32_t d = d_base + d_offset;
                    if (b < b_points && d < d_points) {
                        const std::uint32_t source_prefix =
                            (((a << next_active_stages) + d) << stage_base) + c;
                        const std::uint32_t source = (source_prefix << active_stages) + b;
                        transpose_tile[d_offset * kTransposePitch + b_offset] =
                            state[transform * n + source];
                    }
                }
                __syncthreads();
                for (std::uint32_t item = threadIdx.x;
                     item < kTransposeTile * kTransposeTile; item += blockDim.x) {
                    const std::uint32_t b_offset = item / kTransposeTile;
                    const std::uint32_t d_offset = item % kTransposeTile;
                    const std::uint32_t b = b_base + b_offset;
                    const std::uint32_t d = d_base + d_offset;
                    if (b < b_points && d < d_points) {
                        const std::uint32_t target_prefix =
                            (((a << active_stages) + b) << stage_base) + c;
                        const std::uint32_t target =
                            (target_prefix << next_active_stages) + d;
                        next_state[transform * n + target] =
                            transpose_tile[d_offset * kTransposePitch + b_offset];
                    }
                }
                __syncthreads();
            }
            grid.sync();
            state = next_state;
        }
    }
}

// Generated logN=20 schedule for A=(7/3, Ud/Td). CTA roles own complete
// dependency groups and publish directly in the next fold's token-major
// layout. Fold roles execute concurrently; there is no grid barrier after the
// initial readiness-state initialization.
template <typename Word, std::uint32_t OnlineAGroup,
          std::uint32_t OnlineCGroup, bool CtaRadix4 = false,
          bool ProfileRoleMetrics = false>
__global__ void appt_online_log20_kernel(
    const Word* input, Word* output, Word* state1, Word* state2,
    unsigned int* ready1, unsigned int* ready2, std::uint64_t transforms,
    std::uint32_t role0_blocks, std::uint32_t role1_blocks,
    unsigned long long* trace, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kTileRows = 32;
    constexpr std::uint32_t kTilePitch = 129;
    constexpr std::uint32_t kWarps = 8;
    const auto grid = cg::this_grid();
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    extern __shared__ __align__(16) unsigned char shared_storage[];
    Word* tile = reinterpret_cast<Word*>(shared_storage);

    const std::uint64_t ready1_count = transforms * 64;
    const std::uint64_t ready2_count = transforms * 128;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < ready1_count + ready2_count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        if (index < ready1_count) ready1[index] = 0;
        else ready2[index - ready1_count] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 6; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    if constexpr (ProfileRoleMetrics) {
        for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
             index < 12; index += gridDim.x * blockDim.x) {
            trace[6 + index] = 0;
        }
    }
    grid.sync();

    const std::uint32_t role1_begin = role0_blocks;
    const std::uint32_t role2_begin = role0_blocks + role1_blocks;
    const std::uint32_t role = blockIdx.x < role1_begin ? 0U
                                  : blockIdx.x < role2_begin ? 1U : 2U;
    const std::uint32_t local_block = role == 0 ? blockIdx.x
                                      : role == 1 ? blockIdx.x - role1_begin
                                                  : blockIdx.x - role2_begin;
    const std::uint32_t role_blocks = role == 0 ? role0_blocks
                                      : role == 1 ? role1_blocks
                                                  : gridDim.x - role2_begin;
    bool recorded_start = false;
    unsigned long long wait_time = 0;
    unsigned long long compute_time = 0;
    unsigned long long boundary_time = 0;
    unsigned long long completed_tasks = 0;

    if (role == 0) {
        const std::uint64_t task_count = transforms * 64;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            const std::uint64_t transform = task >> 6;
            const std::uint32_t a = static_cast<std::uint32_t>(task & 63U);
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace, streaming_globaltimer());
                recorded_start = true;
            }
            for (std::uint32_t d_base = 0; d_base < 128;
                 d_base += kTileRows) {
                unsigned long long compute_start = 0;
                unsigned long long boundary_start = 0;
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) compute_start = streaming_globaltimer();
                }
                if constexpr (CtaRadix4) {
                    for (std::uint32_t item = threadIdx.x;
                         item < 128 * kTileRows; item += blockDim.x) {
                        const std::uint32_t d_row = item / 128;
                        const std::uint32_t b = item & 127U;
                        const std::uint32_t token = a * 128 + d_base + d_row;
                        const std::uint32_t logical = token * 128 + b;
                        const std::uint32_t source = __brev(logical) >> 12;
                        tile[d_row * kTilePitch + b] =
                            input[transform * kN + source];
                    }
                    __syncthreads();
                    detail::hierarchical_resident_radix4<false, Word, 7,
                                                         kTileRows>(tile, op);
                } else {
#pragma unroll
                    for (std::uint32_t wave = 0; wave < kTileRows / kWarps; ++wave) {
                        const std::uint32_t d_row = wave * kWarps + warp;
                        const std::uint32_t d = d_base + d_row;
                        const std::uint32_t token = a * 128 + d;
                        Word values[4];
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            const std::uint32_t b = (slot << 5) + lane;
                            const std::uint32_t logical = token * 128 + b;
                            const std::uint32_t source = __brev(logical) >> 12;
                            values[slot] = input[transform * kN + source];
                        }
#pragma unroll
                        for (std::uint32_t stage = 0; stage < 7; ++stage) {
                            detail::hybrid_apply_register_stage<Word, 4>(
                                values, 128, stage, 0, 0, lane, op);
                        }
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            tile[d_row * kTilePitch + (slot << 5) + lane] =
                                values[slot];
                        }
                    }
                }
                __syncthreads();
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) {
                        const auto now = streaming_globaltimer();
                        compute_time += now - compute_start;
                        boundary_start = now;
                    }
                }
                for (std::uint32_t item = threadIdx.x; item < 128 * kTileRows;
                     item += blockDim.x) {
                    const std::uint32_t b = item / kTileRows;
                    const std::uint32_t d_row = item % kTileRows;
                    state1[transform * kN +
                           (static_cast<std::uint64_t>(a * 128 + b) * 128) +
                           d_base + d_row] = tile[d_row * kTilePitch + b];
                }
                __syncthreads();
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) {
                        boundary_time += streaming_globaltimer() - boundary_start;
                    }
                }
            }
            if (threadIdx.x == 0) {
                unsigned long long handoff_start = 0;
                if constexpr (ProfileRoleMetrics) {
                    handoff_start = streaming_globaltimer();
                }
                __threadfence();
                atomicExch(ready1 + task, 1U);
                if constexpr (ProfileRoleMetrics) {
                    boundary_time += streaming_globaltimer() - handoff_start;
                    ++completed_tasks;
                }
            }
        }
    } else if (role == 1) {
        static_assert(OnlineAGroup == 1 || OnlineAGroup == 2 ||
                          OnlineAGroup == 4,
                      "unsupported APPT online publication group");
        static_assert(OnlineCGroup == 1 || OnlineCGroup == 2 ||
                          OnlineCGroup == 4,
                      "unsupported APPT online output group");
        const std::uint64_t task_count = transforms * 64;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            if constexpr (OnlineAGroup == 1) {
                unsigned long long wait_start = 0;
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) wait_start = streaming_globaltimer();
                }
                if (threadIdx.x == 0) {
                    while (atomicAdd(ready1 + task, 0U) == 0U) __nanosleep(128);
                }
                __syncthreads();
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) {
                        wait_time += streaming_globaltimer() - wait_start;
                    }
                }
                if (!recorded_start && threadIdx.x == 0) {
                    atomicMin(trace + 2, streaming_globaltimer());
                    recorded_start = true;
                }
                const std::uint64_t transform = task >> 6;
                const std::uint32_t a = static_cast<std::uint32_t>(task & 63U);
                for (std::uint32_t c_base = 0; c_base < 128;
                     c_base += kTileRows) {
                    unsigned long long compute_start = 0;
                    unsigned long long boundary_start = 0;
                    if constexpr (ProfileRoleMetrics) {
                        if (threadIdx.x == 0) compute_start = streaming_globaltimer();
                    }
                    if constexpr (CtaRadix4) {
                        for (std::uint32_t item = threadIdx.x;
                             item < 128 * kTileRows; item += blockDim.x) {
                            const std::uint32_t c_row = item / 128;
                            const std::uint32_t b = item & 127U;
                            const std::uint32_t c = c_base + c_row;
                            const std::uint32_t token = a * 128 + c;
                            tile[c_row * kTilePitch + b] = state1[
                                transform * kN +
                                static_cast<std::uint64_t>(token) * 128 + b];
                        }
                        __syncthreads();
                        detail::hierarchical_resident_radix4_offset<
                            Word, 7, kTileRows>(tile, op, 7, c_base, 1);
                    } else {
#pragma unroll
                        for (std::uint32_t wave = 0; wave < kTileRows / kWarps; ++wave) {
                            const std::uint32_t c_row = wave * kWarps + warp;
                            const std::uint32_t c = c_base + c_row;
                            const std::uint32_t token = a * 128 + c;
                            Word values[4];
#pragma unroll
                            for (std::uint32_t slot = 0; slot < 4; ++slot) {
                                const std::uint32_t b = (slot << 5) + lane;
                                values[slot] = state1[
                                    transform * kN +
                                    static_cast<std::uint64_t>(token) * 128 + b];
                            }
#pragma unroll
                            for (std::uint32_t stage = 0; stage < 7; ++stage) {
                                detail::hybrid_apply_register_stage<Word, 4>(
                                    values, 128, stage, 7, c, lane, op);
                            }
#pragma unroll
                            for (std::uint32_t slot = 0; slot < 4; ++slot) {
                                tile[c_row * kTilePitch + (slot << 5) + lane] =
                                    values[slot];
                            }
                        }
                    }
                    __syncthreads();
                    if constexpr (ProfileRoleMetrics) {
                        if (threadIdx.x == 0) {
                            const auto now = streaming_globaltimer();
                            compute_time += now - compute_start;
                            boundary_start = now;
                        }
                    }
                    for (std::uint32_t item = threadIdx.x; item < 128 * kTileRows;
                         item += blockDim.x) {
                        const std::uint32_t b = item / kTileRows;
                        const std::uint32_t c_row = item % kTileRows;
                        const std::uint32_t c = c_base + c_row;
                        state2[transform * kN +
                               (static_cast<std::uint64_t>(b * 128 + c) * 64) + a] =
                            tile[c_row * kTilePitch + b];
                    }
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        __threadfence();
                        for (std::uint32_t c = c_base; c < c_base + kTileRows; ++c) {
                            atomicAdd(ready2 + transform * 128 + c, 1U);
                        }
                        if constexpr (ProfileRoleMetrics) {
                            boundary_time +=
                                streaming_globaltimer() - boundary_start;
                        }
                    }
                }
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) ++completed_tasks;
                }
            } else {
                constexpr std::uint32_t kCPerTask = 128 / OnlineAGroup;
                constexpr std::uint32_t kCMicro = kTileRows / OnlineAGroup;
                const std::uint64_t transform = task >> 6;
                const std::uint32_t local_task = static_cast<std::uint32_t>(task & 63U);
                const std::uint32_t c_base =
                    (local_task % OnlineAGroup) * kCPerTask;
                const std::uint32_t a_base =
                    (local_task / OnlineAGroup) * OnlineAGroup;
                if (threadIdx.x == 0) {
                    for (std::uint32_t a = a_base;
                         a < a_base + OnlineAGroup; ++a) {
                        while (atomicAdd(ready1 + transform * 64 + a, 0U) == 0U) {
                            __nanosleep(128);
                        }
                    }
                }
                __syncthreads();
                if (!recorded_start && threadIdx.x == 0) {
                    atomicMin(trace + 2, streaming_globaltimer());
                    recorded_start = true;
                }
                for (std::uint32_t c_micro = 0; c_micro < kCPerTask;
                     c_micro += kCMicro) {
#pragma unroll
                    for (std::uint32_t wave = 0; wave < kTileRows / kWarps;
                         ++wave) {
                        const std::uint32_t token_row = wave * kWarps + warp;
                        const std::uint32_t a_offset = token_row % OnlineAGroup;
                        const std::uint32_t c_offset = token_row / OnlineAGroup;
                        const std::uint32_t a = a_base + a_offset;
                        const std::uint32_t c = c_base + c_micro + c_offset;
                        const std::uint32_t token = a * 128 + c;
                        Word values[4];
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            const std::uint32_t b = (slot << 5) + lane;
                            values[slot] = state1[
                                transform * kN +
                                static_cast<std::uint64_t>(token) * 128 + b];
                        }
#pragma unroll
                        for (std::uint32_t stage = 0; stage < 7; ++stage) {
                            detail::hybrid_apply_register_stage<Word, 4>(
                                values, 128, stage, 7, c, lane, op);
                        }
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            tile[token_row * kTilePitch + (slot << 5) + lane] =
                                values[slot];
                        }
                    }
                    __syncthreads();
                    for (std::uint32_t item = threadIdx.x;
                         item < 128 * kTileRows; item += blockDim.x) {
                        const std::uint32_t a_offset = item % OnlineAGroup;
                        const std::uint32_t c_offset =
                            (item / OnlineAGroup) % kCMicro;
                        const std::uint32_t b = item / kTileRows;
                        const std::uint32_t token_row =
                            c_offset * OnlineAGroup + a_offset;
                        const std::uint32_t c = c_base + c_micro + c_offset;
                        state2[transform * kN +
                               (static_cast<std::uint64_t>(b * 128 + c) * 64) +
                               a_base + a_offset] =
                            tile[token_row * kTilePitch + b];
                    }
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        __threadfence();
                        for (std::uint32_t c = c_base + c_micro;
                             c < c_base + c_micro + kCMicro; ++c) {
                            atomicAdd(ready2 + transform * 128 + c,
                                      OnlineAGroup);
                        }
                    }
                }
            }
        }
    } else {
        const std::uint64_t task_count = transforms * 128;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            if constexpr (OnlineCGroup == 1) {
                unsigned long long wait_start = 0;
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) wait_start = streaming_globaltimer();
                }
                if (threadIdx.x == 0) {
                    while (atomicAdd(ready2 + task, 0U) < 64U) __nanosleep(128);
                }
                __syncthreads();
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) {
                        wait_time += streaming_globaltimer() - wait_start;
                    }
                }
                if (!recorded_start && threadIdx.x == 0) {
                    atomicMin(trace + 4, streaming_globaltimer());
                    recorded_start = true;
                }
                const std::uint64_t transform = task >> 7;
                const std::uint32_t c = static_cast<std::uint32_t>(task & 127U);
                for (std::uint32_t b_base = 0; b_base < 128;
                     b_base += CtaRadix4 ? kTileRows : kWarps) {
                    unsigned long long compute_start = 0;
                    unsigned long long boundary_start = 0;
                    if constexpr (ProfileRoleMetrics) {
                        if (threadIdx.x == 0) compute_start = streaming_globaltimer();
                    }
                    if constexpr (CtaRadix4) {
                        constexpr std::uint32_t kLocalStride = 65;
                        for (std::uint32_t item = threadIdx.x;
                             item < 64 * kTileRows; item += blockDim.x) {
                            const std::uint32_t b_row = item / 64;
                            const std::uint32_t a = item & 63U;
                            const std::uint32_t token = (b_base + b_row) * 128 + c;
                            tile[b_row * kLocalStride + a] = state2[
                                transform * kN +
                                static_cast<std::uint64_t>(token) * 64 + a];
                        }
                        __syncthreads();
                        detail::hierarchical_resident_radix4_offset<
                            Word, 6, kTileRows>(tile, op, 14,
                                b_base * 128 + c, 128);
                        if constexpr (ProfileRoleMetrics) {
                            if (threadIdx.x == 0) {
                                const auto now = streaming_globaltimer();
                                compute_time += now - compute_start;
                                boundary_start = now;
                            }
                        }
                        for (std::uint32_t item = threadIdx.x;
                             item < 64 * kTileRows; item += blockDim.x) {
                            const std::uint32_t b_row = item / 64;
                            const std::uint32_t a = item & 63U;
                            const std::uint32_t token = (b_base + b_row) * 128 + c;
                            const std::uint32_t logical = token + (a << 14);
                            output[transform * kN + logical] = op.finalize(
                                tile[b_row * kLocalStride + a],
                                static_cast<std::uint32_t>(kN));
                        }
                        __syncthreads();
                    } else {
                        const std::uint32_t b = b_base + warp;
                        const std::uint32_t token = b * 128 + c;
                        Word values[2];
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 2; ++slot) {
                            const std::uint32_t a = (slot << 5) + lane;
                            values[slot] = state2[
                                transform * kN +
                                static_cast<std::uint64_t>(token) * 64 + a];
                        }
#pragma unroll
                        for (std::uint32_t stage = 0; stage < 6; ++stage) {
                            detail::hybrid_apply_register_stage<Word, 2>(
                                values, 64, stage, 14, token, lane, op);
                        }
                        if constexpr (ProfileRoleMetrics) {
                            if (threadIdx.x == 0) {
                                const auto now = streaming_globaltimer();
                                compute_time += now - compute_start;
                                boundary_start = now;
                            }
                        }
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 2; ++slot) {
                            const std::uint32_t a = (slot << 5) + lane;
                            const std::uint32_t logical = token + (a << 14);
                            output[transform * kN + logical] =
                                op.finalize(values[slot], static_cast<std::uint32_t>(kN));
                        }
                    }
                    if constexpr (ProfileRoleMetrics) {
                        if (threadIdx.x == 0) {
                            boundary_time +=
                                streaming_globaltimer() - boundary_start;
                        }
                    }
                }
                if constexpr (ProfileRoleMetrics) {
                    if (threadIdx.x == 0) ++completed_tasks;
                }
            } else {
                constexpr std::uint32_t kBPerTask = 128 / OnlineCGroup;
                const std::uint64_t transform = task >> 7;
                const std::uint32_t local_task = static_cast<std::uint32_t>(task & 127U);
                const std::uint32_t c_base =
                    (local_task / OnlineCGroup) * OnlineCGroup;
                const std::uint32_t b_base =
                    (local_task % OnlineCGroup) * kBPerTask;
                if (threadIdx.x == 0) {
                    for (std::uint32_t c = c_base;
                         c < c_base + OnlineCGroup; ++c) {
                        while (atomicAdd(ready2 + transform * 128 + c, 0U) < 64U) {
                            __nanosleep(128);
                        }
                    }
                }
                __syncthreads();
                if (!recorded_start && threadIdx.x == 0) {
                    atomicMin(trace + 4, streaming_globaltimer());
                    recorded_start = true;
                }
                Word* warp_tile = tile + warp * OnlineCGroup * 64;
                for (std::uint32_t b_offset = warp; b_offset < kBPerTask;
                     b_offset += kWarps) {
                    const std::uint32_t b = b_base + b_offset;
#pragma unroll
                    for (std::uint32_t c_offset = 0;
                         c_offset < OnlineCGroup; ++c_offset) {
                        const std::uint32_t c = c_base + c_offset;
                        const std::uint32_t token = b * 128 + c;
                        Word values[2];
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 2; ++slot) {
                            const std::uint32_t a = (slot << 5) + lane;
                            values[slot] = state2[
                                transform * kN +
                                static_cast<std::uint64_t>(token) * 64 + a];
                        }
#pragma unroll
                        for (std::uint32_t stage = 0; stage < 6; ++stage) {
                            detail::hybrid_apply_register_stage<Word, 2>(
                                values, 64, stage, 14, token, lane, op);
                        }
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 2; ++slot) {
                            const std::uint32_t a = (slot << 5) + lane;
                            warp_tile[c_offset * 64 + a] =
                                op.finalize(values[slot],
                                            static_cast<std::uint32_t>(kN));
                        }
                    }
                    __syncwarp();
                    for (std::uint32_t item = lane;
                         item < 64 * OnlineCGroup; item += 32) {
                        const std::uint32_t c_offset = item % OnlineCGroup;
                        const std::uint32_t a = item / OnlineCGroup;
                        const std::uint32_t logical =
                            b * 128 + c_base + c_offset + (a << 14);
                        output[transform * kN + logical] =
                            warp_tile[c_offset * 64 + a];
                    }
                    __syncwarp();
                }
            }
        }
    }
    if (recorded_start && threadIdx.x == 0) {
        atomicMax(trace + 2 * role + 1, streaming_globaltimer());
    }
    if constexpr (ProfileRoleMetrics) {
        if (threadIdx.x == 0) {
            atomicAdd(trace + 6 + 4 * role, wait_time);
            atomicAdd(trace + 7 + 4 * role, compute_time);
            atomicAdd(trace + 8 + 4 * role, boundary_time);
            atomicAdd(trace + 9 + 4 * role, completed_tasks);
        }
    }
}

// APPT fused-tail schedule for logN=20. Fold 0 publishes one reordered global
// state. A tail CTA then owns a dependency-closed [64 a][128 d] tile for one
// middle frequency c and executes folds 1 and 2 entirely in shared memory.
template <typename Word>
__global__ void appt_online_fused_tail_log20_kernel(
    const Word* input, Word* output, Word* state1, unsigned int* ready1,
    std::uint64_t transforms, std::uint32_t producer_blocks,
    unsigned long long* trace, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kFirstRows = 32;
    constexpr std::uint32_t kTailRows = 64;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;
    const auto grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char shared_storage[];
    Word* tile = reinterpret_cast<Word*>(shared_storage);

    const std::uint64_t ready_count = transforms * 64;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < ready_count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready1[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 6; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    grid.sync();

    const bool producer = blockIdx.x < producer_blocks;
    const std::uint32_t local_block =
        producer ? blockIdx.x : blockIdx.x - producer_blocks;
    const std::uint32_t role_blocks =
        producer ? producer_blocks : gridDim.x - producer_blocks;
    bool recorded_start = false;

    if (producer) {
        const std::uint64_t task_count = transforms * 64;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            const std::uint64_t transform = task >> 6;
            const std::uint32_t a = static_cast<std::uint32_t>(task & 63U);
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace, streaming_globaltimer());
                recorded_start = true;
            }
            for (std::uint32_t c_base = 0; c_base < kColumns;
                 c_base += kFirstRows) {
                for (std::uint32_t item = threadIdx.x;
                     item < kColumns * kFirstRows; item += blockDim.x) {
                    const std::uint32_t c_row = item / kColumns;
                    const std::uint32_t b = item & (kColumns - 1);
                    const std::uint32_t token = a * kColumns + c_base + c_row;
                    const std::uint32_t logical = token * kColumns + b;
                    const std::uint32_t source = __brev(logical) >> 12;
                    tile[c_row * kPitch + b] =
                        input[transform * kN + source];
                }
                __syncthreads();
                detail::hierarchical_resident_radix4<
                    false, Word, 7, kFirstRows>(tile, op);
                for (std::uint32_t item = threadIdx.x;
                     item < kColumns * kFirstRows; item += blockDim.x) {
                    const std::uint32_t b = item / kFirstRows;
                    const std::uint32_t c_row = item % kFirstRows;
                    state1[transform * kN +
                           static_cast<std::uint64_t>(a * kColumns + b) *
                               kColumns +
                           c_base + c_row] = tile[c_row * kPitch + b];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                __threadfence();
                atomicExch(ready1 + task, 1U);
            }
        }
    } else {
        const std::uint64_t task_count = transforms * kColumns;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            const std::uint64_t transform = task >> 7;
            const std::uint32_t c = static_cast<std::uint32_t>(task & 127U);
            if (threadIdx.x < kTailRows) {
                while (atomicAdd(ready1 + transform * kTailRows + threadIdx.x,
                                 0U) == 0U) {
                    __nanosleep(128);
                }
            }
            __syncthreads();
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace + 2, streaming_globaltimer());
                recorded_start = true;
            }
            for (std::uint32_t item = threadIdx.x;
                 item < kTailRows * kColumns; item += blockDim.x) {
                const std::uint32_t a = item / kColumns;
                const std::uint32_t d = item & (kColumns - 1);
                tile[a * kPitch + d] = state1[
                    transform * kN +
                    static_cast<std::uint64_t>(a * kColumns + c) * kColumns + d];
            }
            __syncthreads();
            detail::hierarchical_resident_radix4_offset<
                Word, 7, kTailRows>(tile, op, 7, c, 0);
            if (threadIdx.x == 0) {
                atomicMax(trace + 3, streaming_globaltimer());
                atomicMin(trace + 4, streaming_globaltimer());
            }
            detail::hierarchical_resident_radix4_columns_offset<
                Word, 6, kColumns, kPitch>(tile, op, 14, c, kColumns);
            for (std::uint32_t item = threadIdx.x;
                 item < kTailRows * kColumns; item += blockDim.x) {
                const std::uint32_t a = item / kColumns;
                const std::uint32_t d = item & (kColumns - 1);
                const std::uint32_t logical =
                    d * kColumns + c + (a << 14);
                output[transform * kN + logical] = op.finalize(
                    tile[a * kPitch + d], static_cast<std::uint32_t>(kN));
            }
            __syncthreads();
        }
    }
    if (recorded_start && threadIdx.x == 0) {
        atomicMax(trace + (producer ? 1 : 5), streaming_globaltimer());
    }
}

// Residency-preserving fused tail. The low and high 32-row halves independently
// execute fold 1 and the first five stages of fold 2. Only the low half is
// published; the high-half role consumes it and performs the final stage.
template <typename Word>
__global__ void appt_online_split_tail_log20_kernel(
    const Word* input, Word* output, Word* state1, Word* low_half,
    unsigned int* ready1, unsigned int* ready_low, std::uint64_t transforms,
    std::uint32_t role0_blocks, std::uint32_t role1_blocks,
    unsigned long long* trace, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kRows = 32;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;
    const auto grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char shared_storage[];
    Word* tile = reinterpret_cast<Word*>(shared_storage);

    const std::uint64_t ready1_count = transforms * 64;
    const std::uint64_t ready_low_count = transforms * kColumns;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < ready1_count + ready_low_count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        if (index < ready1_count) ready1[index] = 0;
        else ready_low[index - ready1_count] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 6; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    grid.sync();

    const std::uint32_t role1_begin = role0_blocks;
    const std::uint32_t role2_begin = role0_blocks + role1_blocks;
    const std::uint32_t role = blockIdx.x < role1_begin ? 0U
                                  : blockIdx.x < role2_begin ? 1U : 2U;
    const std::uint32_t local_block = role == 0 ? blockIdx.x
                                      : role == 1 ? blockIdx.x - role1_begin
                                                  : blockIdx.x - role2_begin;
    const std::uint32_t role_blocks = role == 0 ? role0_blocks
                                      : role == 1 ? role1_blocks
                                                  : gridDim.x - role2_begin;
    bool recorded_start = false;

    if (role == 0) {
        const std::uint64_t task_count = transforms * 64;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            const std::uint64_t transform = task >> 6;
            const std::uint32_t a = static_cast<std::uint32_t>(task & 63U);
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace, streaming_globaltimer());
                recorded_start = true;
            }
            for (std::uint32_t c_base = 0; c_base < kColumns;
                 c_base += kRows) {
                for (std::uint32_t item = threadIdx.x;
                     item < kColumns * kRows; item += blockDim.x) {
                    const std::uint32_t c_row = item / kColumns;
                    const std::uint32_t b = item & (kColumns - 1);
                    const std::uint32_t token = a * kColumns + c_base + c_row;
                    const std::uint32_t logical = token * kColumns + b;
                    const std::uint32_t source = __brev(logical) >> 12;
                    tile[c_row * kPitch + b] = input[transform * kN + source];
                }
                __syncthreads();
                detail::hierarchical_resident_radix4<false, Word, 7, kRows>(
                    tile, op);
                for (std::uint32_t item = threadIdx.x;
                     item < kColumns * kRows; item += blockDim.x) {
                    const std::uint32_t b = item / kRows;
                    const std::uint32_t c_row = item % kRows;
                    state1[transform * kN +
                           static_cast<std::uint64_t>(a * kColumns + b) *
                               kColumns +
                           c_base + c_row] = tile[c_row * kPitch + b];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                __threadfence();
                atomicExch(ready1 + task, 1U);
            }
        }
    } else {
        const std::uint64_t task_count = transforms * kColumns;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            const std::uint64_t transform = task >> 7;
            const std::uint32_t c = static_cast<std::uint32_t>(task & 127U);
            const std::uint32_t a_base = role == 1 ? 0U : kRows;
            if (threadIdx.x < kRows) {
                while (atomicAdd(ready1 + transform * 64 + a_base + threadIdx.x,
                                 0U) == 0U) {
                    __nanosleep(128);
                }
            }
            __syncthreads();
            if (role == 2 && threadIdx.x == 0) {
                while (atomicAdd(ready_low + task, 0U) == 0U) {
                    __nanosleep(128);
                }
            }
            __syncthreads();
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace + 2 * role, streaming_globaltimer());
                recorded_start = true;
            }
            for (std::uint32_t item = threadIdx.x;
                 item < kRows * kColumns; item += blockDim.x) {
                const std::uint32_t a = item / kColumns;
                const std::uint32_t d = item & (kColumns - 1);
                tile[a * kPitch + d] = state1[
                    transform * kN +
                    static_cast<std::uint64_t>((a_base + a) * kColumns + c) *
                        kColumns +
                    d];
            }
            __syncthreads();
            detail::hierarchical_resident_radix4_offset<Word, 7, kRows>(
                tile, op, 7, c, 0);
            detail::hierarchical_resident_radix4_columns_offset<
                Word, 5, kColumns, kPitch>(tile, op, 14, c, kColumns);

            if (role == 1) {
                for (std::uint32_t item = threadIdx.x;
                     item < kRows * kColumns; item += blockDim.x) {
                    const std::uint32_t a = item / kColumns;
                    const std::uint32_t d = item & (kColumns - 1);
                    const std::uint32_t token = d * kColumns + c;
                    low_half[transform * (kN / 2) +
                             static_cast<std::uint64_t>(token) * kRows + a] =
                        tile[a * kPitch + d];
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    __threadfence();
                    atomicExch(ready_low + task, 1U);
                }
            } else {
                for (std::uint32_t item = threadIdx.x;
                     item < kRows * kColumns; item += blockDim.x) {
                    const std::uint32_t a = item / kColumns;
                    const std::uint32_t d = item & (kColumns - 1);
                    const std::uint32_t token = d * kColumns + c;
                    Word low = low_half[
                        transform * (kN / 2) +
                        static_cast<std::uint64_t>(token) * kRows + a];
                    Word high = tile[a * kPitch + d];
                    op.apply(19, token + (a << 14), low, high);
                    output[transform * kN + token + (a << 14)] =
                        op.finalize(low, static_cast<std::uint32_t>(kN));
                    output[transform * kN + token + ((a + kRows) << 14)] =
                        op.finalize(high, static_cast<std::uint32_t>(kN));
                }
                __syncthreads();
            }
        }
    }
    if (recorded_start && threadIdx.x == 0) {
        atomicMax(trace + 2 * role + 1, streaming_globaltimer());
    }
}

// A 32-row shared tile is reused for both tail halves. The unused portion of a
// residency-sized shared-memory budget retains most low-half values; only the
// remainder occupies registers while the high half is computed.
template <std::uint32_t FragmentWidth>
__device__ __forceinline__ std::uint32_t appt_static_index_log20(
    std::uint32_t natural) {
    static_assert(FragmentWidth == 8 || FragmentWidth == 16 ||
                  FragmentWidth == 32);
    constexpr std::uint32_t mask = FragmentWidth - 1;
    const std::uint32_t a = natural >> 14;
    const std::uint32_t d = (natural >> 7) & 127U;
    const std::uint32_t c = natural & 127U;
    return (a << 14) | (c << 7) |
           ((d & ~mask) | ((d ^ c) & mask));
}

template <typename Word, std::uint32_t ValuesPerLane, typename Operator>
__device__ __forceinline__ void appt_apply_reused_register_stage(
    Word (&values)[ValuesPerLane], std::uint32_t token_points,
    std::uint32_t local_stage, std::uint32_t lane,
    const Word* coefficients, const Word* coefficients_shoup, Operator op) {
    const std::uint32_t half = 1U << local_stage;
    const std::uint32_t coefficient_base = half - 1;
    if (local_stage < 5) {
        const int lane_delta = 1 << local_stage;
        Word coefficient = Word{0};
        Word coefficient_shoup = Word{0};
        if ((lane & lane_delta) == 0) {
            const std::uint32_t offset = lane & (half - 1U);
            coefficient = coefficients[coefficient_base + offset];
            coefficient_shoup =
                coefficients_shoup[coefficient_base + offset];
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < ValuesPerLane; ++slot) {
            Word left = values[slot];
            Word right = detail::hybrid_shuffle_xor(values[slot], lane_delta);
            Word right_result = Word{0};
            const std::uint32_t left_index = (slot << 5) + lane;
            if (left_index < token_points && (lane & lane_delta) == 0) {
                op.apply_coefficient(coefficient, coefficient_shoup, left,
                                     right);
                right_result = right;
            }
            const Word received_right =
                detail::hybrid_shuffle_xor(right_result, lane_delta);
            if ((lane & lane_delta) == 0) {
                values[slot] = left;
            } else if (left_index < token_points) {
                values[slot] = received_right;
            }
        }
    } else if (local_stage == 5) {
        // Register slots are spaced by one warp. Every stage-5 pair owned by
        // a lane therefore uses the same coefficient, independent of slot.
        const Word coefficient = coefficients[coefficient_base + lane];
        const Word coefficient_shoup =
            coefficients_shoup[coefficient_base + lane];
#pragma unroll
        for (std::uint32_t slot = 0; slot < ValuesPerLane; slot += 2U) {
            if (slot + 1U < ValuesPerLane) {
                const std::uint32_t left_index = (slot << 5) + lane;
                if (left_index < token_points) {
                    Word left = values[slot];
                    Word right = values[slot + 1U];
                    op.apply_coefficient(coefficient, coefficient_shoup, left,
                                         right);
                    values[slot] = left;
                    values[slot + 1U] = right;
                }
            }
        }
    } else {
        const std::uint32_t slot_delta = 1U << (local_stage - 5);
#pragma unroll
        for (std::uint32_t slot = 0; slot < ValuesPerLane; ++slot) {
            if ((slot & slot_delta) == 0 &&
                slot + slot_delta < ValuesPerLane) {
                const std::uint32_t left_index = (slot << 5) + lane;
                if (left_index < token_points) {
                    Word left = values[slot];
                    Word right = values[slot + slot_delta];
                    const std::uint32_t offset = left_index & (half - 1);
                    op.apply_coefficient(
                        coefficients[coefficient_base + offset],
                        coefficients_shoup[coefficient_base + offset], left,
                        right);
                    values[slot] = left;
                    values[slot + slot_delta] = right;
                }
            }
        }
    }
}

template <typename Word, std::uint32_t ValuesPerLane, typename Operator>
__device__ __forceinline__ void appt_apply_cached_register_stage(
    Word (&values)[ValuesPerLane], std::uint32_t token_points,
    std::uint32_t local_stage, std::uint32_t lane,
    const Word* coefficients, const Word* coefficients_shoup, Operator op) {
    const std::uint32_t half = 1U << local_stage;
    const std::uint32_t coefficient_base = half - 1;
    if (local_stage < 5) {
        const int lane_delta = 1 << local_stage;
#pragma unroll
        for (std::uint32_t slot = 0; slot < ValuesPerLane; ++slot) {
            Word left = values[slot];
            Word right = detail::hybrid_shuffle_xor(values[slot], lane_delta);
            Word right_result = Word{0};
            const std::uint32_t left_index = (slot << 5) + lane;
            if (left_index < token_points && (lane & lane_delta) == 0) {
                const std::uint32_t offset = left_index & (half - 1);
                op.apply_coefficient(coefficients[coefficient_base + offset],
                                     coefficients_shoup[coefficient_base + offset],
                                     left, right);
                right_result = right;
            }
            const Word received_right =
                detail::hybrid_shuffle_xor(right_result, lane_delta);
            if ((lane & lane_delta) == 0) {
                values[slot] = left;
            } else if (left_index < token_points) {
                values[slot] = received_right;
            }
        }
    } else {
        const std::uint32_t slot_delta = 1U << (local_stage - 5);
#pragma unroll
        for (std::uint32_t slot = 0; slot < ValuesPerLane; ++slot) {
            if ((slot & slot_delta) == 0 &&
                slot + slot_delta < ValuesPerLane) {
                const std::uint32_t left_index = (slot << 5) + lane;
                if (left_index < token_points) {
                    Word left = values[slot];
                    Word right = values[slot + slot_delta];
                    const std::uint32_t offset = left_index & (half - 1);
                    op.apply_coefficient(
                        coefficients[coefficient_base + offset],
                        coefficients_shoup[coefficient_base + offset], left,
                        right);
                    values[slot] = left;
                    values[slot + slot_delta] = right;
                }
            }
        }
    }
}

template <bool WarpRows, bool WarpColumns, bool Radix8Core,
          bool GroupedState, typename Word, std::uint32_t Threads,
          typename Operator>
__device__ __forceinline__ void appt_register_tail_half(
    Word* tile, const Word* state1, std::uint64_t transform,
    std::uint32_t c, std::uint32_t a_base, const Word* coefficients,
    const Word* coefficients_shoup, Operator op) {
    constexpr std::uint32_t kRows = 32;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;
    constexpr std::uint32_t kWarps = Threads / 32;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    if constexpr (WarpRows) {
#pragma unroll
        for (std::uint32_t wave = 0; wave < kRows / kWarps; ++wave) {
            const std::uint32_t a = wave * kWarps + warp;
            Word values[4];
#pragma unroll
            for (std::uint32_t slot = 0; slot < 4; ++slot) {
                const std::uint32_t d = (slot << 5) + lane;
                values[slot] = state1[
                    transform * (1ULL << 20) +
                    static_cast<std::uint64_t>((a_base + a) * kColumns + c) *
                        kColumns +
                    d];
            }
#pragma unroll
            for (std::uint32_t stage = 0; stage < 7; ++stage) {
                appt_apply_cached_register_stage<Word, 4>(
                    values, kColumns, stage, lane, coefficients,
                    coefficients_shoup, op);
            }
#pragma unroll
            for (std::uint32_t slot = 0; slot < 4; ++slot) {
                tile[a * kPitch + (slot << 5) + lane] = values[slot];
            }
        }
        __syncthreads();
    } else {
        for (std::uint32_t item = threadIdx.x;
             item < kRows * kColumns; item += blockDim.x) {
            const std::uint32_t a = GroupedState
                                        ? item & (kRows - 1)
                                        : item / kColumns;
            const std::uint32_t d = GroupedState
                                        ? item / kRows
                                        : item & (kColumns - 1);
            const std::uint64_t state_index = GroupedState
                ? (static_cast<std::uint64_t>(c) * kColumns + d) * 64 +
                      a_base + a
                : static_cast<std::uint64_t>((a_base + a) * kColumns + c) *
                      kColumns + d;
            tile[a * kPitch + d] =
                state1[transform * (1ULL << 20) + state_index];
        }
        __syncthreads();
        if constexpr (Radix8Core) {
            detail::hierarchical_resident_radix8_cached<Word, 7, kRows>(
                tile, coefficients, coefficients_shoup, op);
        } else {
            detail::hierarchical_resident_radix4_cached<Word, 7, kRows>(
                tile, coefficients, coefficients_shoup, op);
        }
    }

    if constexpr (WarpColumns) {
        for (std::uint32_t d = warp; d < kColumns; d += kWarps) {
            Word values[1] = {tile[lane * kPitch + d]};
#pragma unroll
            for (std::uint32_t stage = 0; stage < 5; ++stage) {
                detail::hybrid_apply_register_stage<Word, 1>(
                    values, kRows, stage, 14, d * kColumns + c, lane, op);
            }
            tile[lane * kPitch + d] = values[0];
        }
        __syncthreads();
    } else {
        if constexpr (Radix8Core) {
            detail::hierarchical_resident_radix8_columns_offset<
                Word, 5, kColumns, kPitch>(tile, op, 14, c, kColumns);
        } else {
            detail::hierarchical_resident_radix4_columns_offset<
                Word, 5, kColumns, kPitch>(tile, op, 14, c, kColumns);
        }
    }
}

template <typename Word, typename Operator>
__device__ __forceinline__ void appt_register_final_half(
    Word* tile, const Word* state, std::uint64_t transform, std::uint32_t c,
    std::uint32_t a_base, Operator op) {
    constexpr std::uint32_t kRows = 32;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;
    for (std::uint32_t item = threadIdx.x; item < kRows * kColumns;
         item += blockDim.x) {
        const std::uint32_t a = item & (kRows - 1);
        const std::uint32_t d = item / kRows;
        const std::uint64_t state_index =
            (static_cast<std::uint64_t>(c) * kColumns + d) * 64 +
            a_base + a;
        tile[a * kPitch + d] =
            state[transform * (1ULL << 20) + state_index];
    }
    __syncthreads();
    detail::hierarchical_resident_radix4_columns_offset<
        Word, 5, kColumns, kPitch>(tile, op, 14, c, kColumns);
}

__device__ __forceinline__ bool appt_decode_data_time_task(
    std::uint64_t task, std::uint64_t spatial_tasks,
    std::uint64_t transforms, std::uint32_t data_time,
    std::uint64_t& transform, std::uint64_t& spatial_task) {
    const std::uint64_t transform_group = task / spatial_tasks;
    spatial_task = task % spatial_tasks;
    transform = transform_group * data_time;
    return transform < transforms;
}

template <typename Word, std::uint32_t FragmentWidth, bool StaticInput,
          typename Operator>
__device__ __forceinline__ void appt_resident_2d_half(
    Word* tile, Word* coefficients, Word* coefficients_shoup,
    const Word* input, std::uint64_t transform, std::uint32_t a,
    std::uint32_t c_base, Operator op) {
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kRows = 64;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;

    detail::hierarchical_load_coefficient_tree<Word, 7>(
        coefficients, coefficients_shoup, op, 0, 0);
    for (std::uint32_t item = threadIdx.x; item < kRows * kColumns;
         item += blockDim.x) {
        const std::uint32_t c = item / kColumns;
        const std::uint32_t b = item & (kColumns - 1);
        const std::uint32_t logical =
            (a * kColumns + c_base + c) * kColumns + b;
        const std::uint32_t source = __brev(logical) >> 12;
        const std::uint32_t input_index =
            StaticInput ? appt_static_index_log20<FragmentWidth>(source)
                        : source;
        tile[c * kPitch + b] = input[transform * kN + input_index];
    }
    __syncthreads();
    detail::hierarchical_resident_radix4_cached<Word, 7, kRows>(
        tile, coefficients, coefficients_shoup, op);

    detail::hierarchical_resident_radix4_columns_offset<
        Word, 6, kColumns, kPitch>(tile, op, 7, 0, 1);
}

template <typename Word, std::uint32_t FragmentWidth, bool StaticInput,
          typename Operator>
__device__ __forceinline__ void appt_resident_2d_quarter(
    Word* tile, const Word* coefficients, const Word* coefficients_shoup,
    const Word* input, std::uint64_t transform, std::uint32_t a,
    std::uint32_t c_base, Operator op) {
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kRows = 32;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;

    for (std::uint32_t item = threadIdx.x; item < kRows * kColumns;
         item += blockDim.x) {
        const std::uint32_t c = item / kColumns;
        const std::uint32_t b = item & (kColumns - 1);
        const std::uint32_t logical =
            (a * kColumns + c_base + c) * kColumns + b;
        const std::uint32_t source = __brev(logical) >> 12;
        const std::uint32_t input_index =
            StaticInput ? appt_static_index_log20<FragmentWidth>(source)
                        : source;
        tile[c * kPitch + b] = input[transform * kN + input_index];
    }
    __syncthreads();
    detail::hierarchical_resident_radix4_cached<Word, 7, kRows>(
        tile, coefficients, coefficients_shoup, op);
    detail::hierarchical_resident_radix4_columns_offset<
        Word, 5, kColumns, kPitch>(tile, op, 7, 0, 1);
}

template <typename Word, std::uint32_t Threads, std::uint32_t MinBlocksPerSm,
          bool AggregateReady, bool WarpRows, bool WarpColumns,
          bool Radix8Core, std::uint32_t ProducerAGroup,
          bool WriterFinal, bool TransformInterleave,
          bool Resident2D, bool ResidentQuarter,
          std::uint32_t FragmentWidth,
          bool StaticInput, bool NaturalOutput>
__global__ __launch_bounds__(Threads, MinBlocksPerSm)
void appt_online_register_tail_log20_kernel(
    const Word* input, Word* output, Word* state1, unsigned int* ready_count,
    std::uint64_t transforms, std::uint32_t producer_blocks,
    std::uint32_t tail_blocks, std::uint32_t writer_tiles_per_cta,
    std::uint32_t data_time, std::uint32_t data_time_role_mask,
    unsigned long long* trace,
    NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLogN = 20;
    constexpr std::uint64_t kN = 1ULL << kLogN;
    constexpr std::uint32_t kRows = 32;
    constexpr std::uint32_t kColumns = 128;
    constexpr std::uint32_t kPitch = 129;
    constexpr std::uint32_t kResidentRows = 64;
    constexpr std::uint32_t kTileWords =
        (Resident2D ? kResidentRows : kRows) * kPitch;
    constexpr std::uint32_t kValueWords = kRows * kColumns;
    constexpr std::uint32_t kCoefficientCount = (1U << 7) - 1;
    constexpr std::uint32_t kCoefficientWords = 2 * kCoefficientCount;
    constexpr std::uint32_t kSharedBudgetBytes =
        Resident2D
            ? 96 * 1024
            : ResidentQuarter
            ? 48 * 1024
            : sizeof(Word) == sizeof(std::uint64_t) ? 48 * 1024 : 32 * 1024;
    constexpr std::uint32_t kSharedBudgetWords =
        kSharedBudgetBytes / sizeof(Word);
    static_assert(kSharedBudgetWords - kTileWords > kCoefficientWords);
    constexpr std::uint32_t kSavedCapacityWords =
        kSharedBudgetWords - kTileWords - kCoefficientWords;
    constexpr std::uint32_t kSavedWords =
        kSavedCapacityWords < kValueWords
            ? kSavedCapacityWords
            : kValueWords;
    constexpr std::uint32_t kFirstRegisterSlot = kSavedWords / Threads;
    constexpr std::uint32_t kRegisterSlots =
        (kValueWords - kSavedWords + Threads - 1) / Threads;
    constexpr std::uint32_t kItemsPerThread = kValueWords / Threads;
    static_assert(ProducerAGroup == 1 || ProducerAGroup == 8 ||
                  ProducerAGroup == 16 || ProducerAGroup == 32);
    static_assert(!WriterFinal || ProducerAGroup != 1);
    static_assert(!TransformInterleave || WriterFinal);
    static_assert(!Resident2D ||
                  (ProducerAGroup == 32 && WriterFinal && TransformInterleave));
    static_assert(!ResidentQuarter ||
                  (ProducerAGroup == 32 && WriterFinal && TransformInterleave));
    static_assert(!(Resident2D && ResidentQuarter));
    constexpr bool kGroupedProducer = ProducerAGroup != 1;
    constexpr std::uint32_t kProducerCMicro = kRows / ProducerAGroup;
    constexpr std::uint32_t kProducerAGroups = 64 / ProducerAGroup;
    constexpr std::uint32_t kProducerTasksPerAGroup =
        kColumns / kProducerCMicro;
    const auto grid = cg::this_grid();
    extern __shared__ __align__(16) unsigned char shared_storage[];
    Word* tile = reinterpret_cast<Word*>(shared_storage);
    Word* coefficient_storage = tile + kTileWords;
    Word* saved_low = coefficient_storage + kCoefficientWords;

    constexpr std::uint32_t kFragmentGroups = kColumns / FragmentWidth;
    const std::uint64_t input_counter_count = transforms *
        (kGroupedProducer ? kProducerAGroups : (AggregateReady ? 2 : 64));
    const std::uint64_t output_counter_count =
        NaturalOutput ? transforms * kFragmentGroups : 0;
    const std::uint64_t counter_count = input_counter_count +
                                        output_counter_count;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < counter_count;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready_count[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 6; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    grid.sync();

    const bool producer = blockIdx.x < producer_blocks;
    const bool tail = !producer &&
                      blockIdx.x < producer_blocks + tail_blocks;
    const std::uint32_t local_block = producer
                                          ? blockIdx.x
                                          : tail
                                          ? blockIdx.x - producer_blocks
                                          : blockIdx.x - producer_blocks - tail_blocks;
    const std::uint32_t role_blocks = producer
                                          ? producer_blocks
                                          : tail
                                          ? tail_blocks
                                          : gridDim.x - producer_blocks - tail_blocks;
    bool recorded_start = false;

    if (producer) {
        if constexpr (ResidentQuarter) {
            constexpr std::uint64_t kResidentSpatialTasks = 64;
            constexpr std::uint32_t kQuarterWords = kRows * kColumns;
            constexpr std::uint32_t kQuarterItemsPerThread =
                kQuarterWords / Threads;
            constexpr std::uint32_t kRetainedWords = 3 * kQuarterWords;
            constexpr std::uint32_t kRetainedItemsPerThread =
                kRetainedWords / Threads;
            constexpr std::uint32_t kRetainedSavedWordsRaw =
                kSavedCapacityWords < kRetainedWords
                    ? kSavedCapacityWords
                    : kRetainedWords;
            constexpr std::uint32_t kRetainedSavedWords =
                (kRetainedSavedWordsRaw / Threads) * Threads;
            constexpr std::uint32_t kRetainedSavedSlots =
                kRetainedSavedWords / Threads;
            constexpr std::uint32_t kRetainedRegisterSlots =
                kRetainedItemsPerThread - kRetainedSavedSlots;
            static_assert(kQuarterWords % Threads == 0);
            static_assert(kRetainedWords % Threads == 0);
            static_assert(kRetainedRegisterSlots > 0);

            const bool role_data_time = (data_time_role_mask & 1U) != 0;
            const std::uint64_t task_count = role_data_time
                ? ((transforms + data_time - 1) / data_time) *
                      kResidentSpatialTasks
                : transforms * kResidentSpatialTasks;
            Word* coefficients = coefficient_storage;
            Word* coefficients_shoup = coefficients + kCoefficientCount;
            detail::hierarchical_load_coefficient_tree<Word, 7>(
                coefficients, coefficients_shoup, op, 0, 0);
            for (std::uint64_t task = local_block; task < task_count;
                 task += role_blocks) {
                std::uint64_t transform = 0;
                std::uint64_t spatial_task = 0;
                if (role_data_time) {
                    if (!appt_decode_data_time_task(
                            task, kResidentSpatialTasks, transforms, data_time,
                            transform, spatial_task)) {
                        continue;
                    }
                } else {
                    transform = task / kResidentSpatialTasks;
                    spatial_task = task % kResidentSpatialTasks;
                }
                const std::uint32_t a =
                    static_cast<std::uint32_t>(spatial_task);
                const std::uint64_t transform_base = transform;
                const std::uint32_t transform_iterations =
                    role_data_time ? data_time : 1;
                for (std::uint32_t time = 0; time < transform_iterations;
                     ++time) {
                    const std::uint64_t transform = transform_base + time;
                    if (transform >= transforms) break;
                    if (!recorded_start && threadIdx.x == 0) {
                        atomicMin(trace, streaming_globaltimer());
                        recorded_start = true;
                    }

                    Word retained_values[kRetainedRegisterSlots];
                    appt_resident_2d_quarter<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, 0, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kQuarterItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const Word value =
                            tile[(item / kColumns) * kPitch +
                                 (item & (kColumns - 1))];
                        if (slot < kRetainedSavedSlots) {
                            saved_low[item] = value;
                        } else {
                            retained_values[slot - kRetainedSavedSlots] = value;
                        }
                    }
                    __syncthreads();

                    appt_resident_2d_quarter<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, kRows, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kQuarterItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const std::uint32_t c = item / kColumns;
                        const std::uint32_t b = item & (kColumns - 1);
                        Word low = slot < kRetainedSavedSlots
                                       ? saved_low[item]
                                       : retained_values[
                                             slot - kRetainedSavedSlots];
                        Word high = tile[c * kPitch + b];
                        op.apply(12, (c << 7) + b, low, high);
                        if (slot < kRetainedSavedSlots) {
                            saved_low[item] = low;
                        } else {
                            retained_values[slot - kRetainedSavedSlots] = low;
                        }
                        const std::uint32_t high_slot =
                            kQuarterItemsPerThread + slot;
                        const std::uint32_t high_item = kQuarterWords + item;
                        if (high_slot < kRetainedSavedSlots) {
                            saved_low[high_item] = high;
                        } else {
                            retained_values[high_slot - kRetainedSavedSlots] =
                                high;
                        }
                    }
                    __syncthreads();

                    appt_resident_2d_quarter<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, 2 * kRows, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kQuarterItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const std::uint32_t retained_slot =
                            2 * kQuarterItemsPerThread + slot;
                        const std::uint32_t retained_item =
                            2 * kQuarterWords + item;
                        const Word value =
                            tile[(item / kColumns) * kPitch +
                                 (item & (kColumns - 1))];
                        if (retained_slot < kRetainedSavedSlots) {
                            saved_low[retained_item] = value;
                        } else {
                            retained_values[
                                retained_slot - kRetainedSavedSlots] = value;
                        }
                    }
                    __syncthreads();

                    appt_resident_2d_quarter<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, 3 * kRows, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kQuarterItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const std::uint32_t c = item / kColumns;
                        const std::uint32_t b = item & (kColumns - 1);
                        const std::uint32_t q2_slot =
                            2 * kQuarterItemsPerThread + slot;
                        const std::uint32_t q2_item =
                            2 * kQuarterWords + item;
                        Word q2 = q2_slot < kRetainedSavedSlots
                                      ? saved_low[q2_item]
                                      : retained_values[
                                            q2_slot - kRetainedSavedSlots];
                        Word q3 = tile[c * kPitch + b];
                        op.apply(12, (c << 7) + b, q2, q3);

                        Word q0 = slot < kRetainedSavedSlots
                                      ? saved_low[item]
                                      : retained_values[
                                            slot - kRetainedSavedSlots];
                        op.apply(13, (c << 7) + b, q0, q2);
                        const std::uint64_t state0 =
                            transform * kN +
                            (static_cast<std::uint64_t>(b) * kColumns + c) *
                                64 +
                            a;
                        state1[state0] = q0;
                        state1[state0 +
                               static_cast<std::uint64_t>(2 * kRows) * 64] = q2;

                        const std::uint32_t q1_slot =
                            kQuarterItemsPerThread + slot;
                        const std::uint32_t q1_item = kQuarterWords + item;
                        Word q1 = q1_slot < kRetainedSavedSlots
                                      ? saved_low[q1_item]
                                      : retained_values[
                                            q1_slot - kRetainedSavedSlots];
                        op.apply(13, ((c + kRows) << 7) + b, q1, q3);
                        const std::uint64_t state1_index =
                            transform * kN +
                            (static_cast<std::uint64_t>(b) * kColumns + c +
                             kRows) *
                                64 +
                            a;
                        state1[state1_index] = q1;
                        state1[state1_index +
                               static_cast<std::uint64_t>(2 * kRows) * 64] = q3;
                    }
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        __threadfence();
                        atomicAdd(ready_count + transform * kProducerAGroups +
                                      (a >= kRows),
                                  1U);
                    }
                }
            }
        } else if constexpr (Resident2D) {
            constexpr std::uint64_t kResidentSpatialTasks = 64;
            constexpr std::uint32_t kResidentValueWords =
                kResidentRows * kColumns;
            constexpr std::uint32_t kResidentSavedWords =
                kSavedCapacityWords < kResidentValueWords
                    ? kSavedCapacityWords
                    : kResidentValueWords;
            constexpr std::uint32_t kResidentFirstRegisterSlot =
                kResidentSavedWords / Threads;
            constexpr std::uint32_t kResidentRegisterSlots =
                (kResidentValueWords - kResidentSavedWords + Threads - 1) /
                Threads;
            constexpr std::uint32_t kResidentItemsPerThread =
                kResidentValueWords / Threads;
            const bool role_data_time =
                (data_time_role_mask & 1U) != 0;
            const std::uint64_t task_count = role_data_time
                ? ((transforms + data_time - 1) / data_time) *
                      kResidentSpatialTasks
                : transforms * kResidentSpatialTasks;
            Word* coefficients = coefficient_storage;
            Word* coefficients_shoup = coefficients + kCoefficientCount;
            for (std::uint64_t task = local_block; task < task_count;
                 task += role_blocks) {
                std::uint64_t transform = 0;
                std::uint64_t spatial_task = 0;
                if (role_data_time) {
                    if (!appt_decode_data_time_task(
                            task, kResidentSpatialTasks, transforms, data_time,
                            transform, spatial_task)) {
                        continue;
                    }
                } else {
                    transform = task / kResidentSpatialTasks;
                    spatial_task = task % kResidentSpatialTasks;
                }
                const std::uint32_t a =
                    static_cast<std::uint32_t>(spatial_task);
                const std::uint64_t transform_base = transform;
                const std::uint32_t transform_iterations =
                    role_data_time ? data_time : 1;
                for (std::uint32_t time = 0; time < transform_iterations;
                     ++time) {
                    const std::uint64_t transform = transform_base + time;
                    if (transform >= transforms) break;
                    if (!recorded_start && threadIdx.x == 0) {
                        atomicMin(trace, streaming_globaltimer());
                        recorded_start = true;
                    }

                    Word low_values[kResidentRegisterSlots == 0
                                        ? 1
                                        : kResidentRegisterSlots];
                    appt_resident_2d_half<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, 0, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kResidentItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const std::uint32_t c = item / kColumns;
                        const std::uint32_t b = item & (kColumns - 1);
                        const std::uint32_t register_slot =
                            slot >= kResidentFirstRegisterSlot
                                ? slot - kResidentFirstRegisterSlot
                                : 0;
                        if (item < kResidentSavedWords) {
                            saved_low[item] = tile[c * kPitch + b];
                        } else {
                            low_values[register_slot] = tile[c * kPitch + b];
                        }
                    }
                    __syncthreads();

                    appt_resident_2d_half<Word, FragmentWidth, StaticInput>(
                        tile, coefficients, coefficients_shoup, input,
                        transform, a, kResidentRows, op);
#pragma unroll
                    for (std::uint32_t slot = 0;
                         slot < kResidentItemsPerThread; ++slot) {
                        const std::uint32_t item =
                            threadIdx.x + slot * Threads;
                        const std::uint32_t c = item / kColumns;
                        const std::uint32_t b = item & (kColumns - 1);
                        const std::uint32_t register_slot =
                            slot >= kResidentFirstRegisterSlot
                                ? slot - kResidentFirstRegisterSlot
                                : 0;
                        Word low = item < kResidentSavedWords
                                       ? saved_low[item]
                                       : low_values[register_slot];
                        Word high = tile[c * kPitch + b];
                        op.apply(13, (c << 7) + b, low, high);
                        const std::uint64_t state_base =
                            transform * kN +
                            (static_cast<std::uint64_t>(b) * kColumns + c) *
                                64 +
                            a;
                        state1[state_base] = low;
                        state1[state_base +
                               static_cast<std::uint64_t>(kResidentRows) * 64] =
                            high;
                    }
                    __syncthreads();
                    if (threadIdx.x == 0) {
                        __threadfence();
                        atomicAdd(ready_count + transform * kProducerAGroups +
                                      (a >= kRows),
                                  1U);
                    }
                }
            }
        } else {
        const bool role_data_time =
            TransformInterleave && (data_time_role_mask & 1U) != 0;
        Word* coefficients = coefficient_storage;
        Word* coefficients_shoup = coefficients + kCoefficientCount;
        detail::hierarchical_load_coefficient_tree<Word, 7>(
            coefficients, coefficients_shoup, op, 0, 0);
        constexpr std::uint64_t kProducerSpatialTasks =
            kGroupedProducer ? 256 : 64;
        const std::uint64_t task_count = role_data_time
            ? ((transforms + data_time - 1) / data_time) *
                  kProducerSpatialTasks
            : transforms * kProducerSpatialTasks;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            std::uint64_t transform = 0;
            std::uint64_t spatial_task = 0;
            if constexpr (TransformInterleave) {
                if (role_data_time) {
                    if (!appt_decode_data_time_task(
                            task, kProducerSpatialTasks, transforms, data_time,
                            transform, spatial_task)) {
                        continue;
                    }
                } else {
                    transform = task / kProducerSpatialTasks;
                    spatial_task = task % kProducerSpatialTasks;
                }
            } else {
                transform = task / kProducerSpatialTasks;
                spatial_task = task % kProducerSpatialTasks;
            }
            const std::uint32_t local_task =
                static_cast<std::uint32_t>(spatial_task);
            // Finish low-a groups first so the retained low tail can overlap
            // production of the high half.
            const std::uint32_t a = kGroupedProducer
                ? (local_task / kProducerTasksPerAGroup) * ProducerAGroup
                : local_task;
            const std::uint32_t a_group = kGroupedProducer
                                              ? local_task /
                                                    kProducerTasksPerAGroup
                                              : 0;
            const std::uint32_t grouped_c_base = kGroupedProducer
                ? (local_task % kProducerTasksPerAGroup) * kProducerCMicro
                : 0;
            const std::uint64_t transform_base = transform;
            const std::uint32_t transform_iterations =
                role_data_time ? data_time : 1;
            for (std::uint32_t time = 0; time < transform_iterations; ++time) {
            const std::uint64_t transform = transform_base + time;
            if (transform >= transforms) break;
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace, streaming_globaltimer());
                recorded_start = true;
            }
            const std::uint32_t c_begin = kGroupedProducer ? grouped_c_base : 0;
            const std::uint32_t c_end = kGroupedProducer
                                            ? grouped_c_base + kProducerCMicro
                                            : kColumns;
            const std::uint32_t c_step = kGroupedProducer
                                             ? kProducerCMicro
                                             : kRows;
            for (std::uint32_t c_base = c_begin; c_base < c_end;
                 c_base += c_step) {
                if constexpr (WarpRows) {
                    constexpr std::uint32_t kWarps = Threads / 32;
                    const std::uint32_t warp = threadIdx.x >> 5;
                    const std::uint32_t lane = threadIdx.x & 31U;
#pragma unroll
                    for (std::uint32_t wave = 0;
                         wave < kRows / kWarps; ++wave) {
                        const std::uint32_t c_row = wave * kWarps + warp;
                        const std::uint32_t token =
                            a * kColumns + c_base + c_row;
                        Word values[4];
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            const std::uint32_t b = (slot << 5) + lane;
                            const std::uint32_t logical = token * kColumns + b;
                            const std::uint32_t source = __brev(logical) >> 12;
                            const std::uint32_t input_index =
                                StaticInput
                                    ? appt_static_index_log20<FragmentWidth>(source)
                                    : source;
                            values[slot] =
                                input[transform * kN + input_index];
                        }
#pragma unroll
                        for (std::uint32_t stage = 0; stage < 7; ++stage) {
                            appt_apply_cached_register_stage<Word, 4>(
                                values, kColumns, stage, lane, coefficients,
                                coefficients_shoup, op);
                        }
#pragma unroll
                        for (std::uint32_t slot = 0; slot < 4; ++slot) {
                            tile[c_row * kPitch + (slot << 5) + lane] =
                                values[slot];
                        }
                    }
                    __syncthreads();
                } else if constexpr (kGroupedProducer) {
                    for (std::uint32_t item = threadIdx.x;
                         item < kColumns * kRows; item += blockDim.x) {
                        const std::uint32_t a_offset = item % ProducerAGroup;
                        const std::uint32_t remaining = item / ProducerAGroup;
                        const std::uint32_t b = remaining & (kColumns - 1);
                        const std::uint32_t c_offset = remaining / kColumns;
                        const std::uint32_t row =
                            a_offset * kProducerCMicro + c_offset;
                        const std::uint32_t logical =
                            ((a + a_offset) * kColumns + c_base + c_offset) *
                                kColumns +
                            b;
                        const std::uint32_t source = __brev(logical) >> 12;
                        const std::uint32_t input_index =
                            StaticInput
                                ? appt_static_index_log20<FragmentWidth>(source)
                                : source;
                        tile[row * kPitch + b] =
                            input[transform * kN + input_index];
                    }
                    __syncthreads();
                    if constexpr (Radix8Core) {
                        detail::hierarchical_resident_radix8_cached<Word, 7,
                                                                    kRows>(
                            tile, coefficients, coefficients_shoup, op);
                    } else {
                        detail::hierarchical_resident_radix4_cached<Word, 7,
                                                                    kRows>(
                            tile, coefficients, coefficients_shoup, op);
                    }
                } else {
                    for (std::uint32_t item = threadIdx.x;
                         item < kColumns * kRows; item += blockDim.x) {
                        const std::uint32_t c_row = item / kColumns;
                        const std::uint32_t b = item & (kColumns - 1);
                        const std::uint32_t token =
                            a * kColumns + c_base + c_row;
                        const std::uint32_t logical = token * kColumns + b;
                        const std::uint32_t source = __brev(logical) >> 12;
                        const std::uint32_t input_index =
                            StaticInput
                                ? appt_static_index_log20<FragmentWidth>(source)
                                : source;
                        tile[c_row * kPitch + b] =
                            input[transform * kN + input_index];
                    }
                    __syncthreads();
                    if constexpr (Radix8Core) {
                        detail::hierarchical_resident_radix8_cached<Word, 7,
                                                                    kRows>(
                            tile, coefficients, coefficients_shoup, op);
                    } else {
                        detail::hierarchical_resident_radix4_cached<Word, 7,
                                                                    kRows>(
                            tile, coefficients, coefficients_shoup, op);
                    }
                }
                for (std::uint32_t item = threadIdx.x;
                     item < kColumns * kRows; item += blockDim.x) {
                    if constexpr (kGroupedProducer) {
                        const std::uint32_t c_offset =
                            item % kProducerCMicro;
                        const std::uint32_t remaining =
                            item / kProducerCMicro;
                        const std::uint32_t a_offset =
                            remaining % ProducerAGroup;
                        const std::uint32_t b = remaining / ProducerAGroup;
                        const std::uint32_t row =
                            a_offset * kProducerCMicro + c_offset;
                        const std::uint64_t state_index =
                            (static_cast<std::uint64_t>(b) * kColumns +
                             c_base + c_offset) *
                                64 +
                            a + a_offset;
                        state1[transform * kN + state_index] =
                            tile[row * kPitch + b];
                    } else {
                        const std::uint32_t b = item / kRows;
                        const std::uint32_t c_row = item % kRows;
                        state1[transform * kN +
                               static_cast<std::uint64_t>(a * kColumns + b) *
                                   kColumns +
                               c_base + c_row] = tile[c_row * kPitch + b];
                    }
                }
                __syncthreads();
            }
            if constexpr (kGroupedProducer) {
                __threadfence();
                __syncthreads();
                if (threadIdx.x == 0) {
                    atomicAdd(ready_count + transform * kProducerAGroups +
                                  a_group,
                              1U);
                }
            } else {
                if (threadIdx.x == 0) {
                    __threadfence();
                    if constexpr (AggregateReady) {
                        atomicAdd(ready_count + 2 * transform + (a >= kRows),
                                  1U);
                    } else {
                        atomicExch(ready_count + task, 1U);
                    }
                }
            }
            }
        }
        }
    } else if (tail) {
        const bool role_data_time =
            TransformInterleave && (data_time_role_mask & 2U) != 0;
        constexpr std::uint64_t kTailSpatialTasks = kColumns;
        const std::uint64_t task_count = role_data_time
            ? ((transforms + data_time - 1) / data_time) * kTailSpatialTasks
            : transforms * kTailSpatialTasks;
        for (std::uint64_t task = local_block; task < task_count;
             task += role_blocks) {
            std::uint64_t transform = 0;
            std::uint64_t spatial_task = 0;
            if constexpr (TransformInterleave) {
                if (role_data_time) {
                    if (!appt_decode_data_time_task(
                            task, kTailSpatialTasks, transforms, data_time,
                            transform, spatial_task)) {
                        continue;
                    }
                } else {
                    transform = task / kTailSpatialTasks;
                    spatial_task = task % kTailSpatialTasks;
                }
            } else {
                transform = task / kTailSpatialTasks;
                spatial_task = task % kTailSpatialTasks;
            }
            const std::uint32_t c =
                static_cast<std::uint32_t>(spatial_task);
            Word* coefficients = coefficient_storage;
            Word* coefficients_shoup = coefficients + kCoefficientCount;
            if constexpr (!(Resident2D || ResidentQuarter)) {
                detail::hierarchical_load_coefficient_tree<Word, 7>(
                    coefficients, coefficients_shoup, op, 7, c);
            }
            const std::uint64_t transform_base = transform;
            const std::uint32_t transform_iterations =
                role_data_time ? data_time : 1;
            for (std::uint32_t time = 0; time < transform_iterations; ++time) {
            const std::uint64_t transform = transform_base + time;
            if (transform >= transforms) break;
            Word low_values[kRegisterSlots == 0 ? 1 : kRegisterSlots];

            if constexpr (kGroupedProducer) {
                constexpr std::uint32_t kHalfAGroups =
                    kRows / ProducerAGroup;
                if (threadIdx.x < kHalfAGroups) {
                    while (atomicAdd(ready_count +
                                         transform * kProducerAGroups +
                                         threadIdx.x,
                                     0U) < ((Resident2D || ResidentQuarter)
                                                ? kRows
                                                : kProducerTasksPerAGroup)) {
                        __nanosleep(128);
                    }
                }
            } else if constexpr (AggregateReady) {
                if (threadIdx.x == 0) {
                    while (atomicAdd(ready_count + 2 * transform, 0U) < kRows) {
                        __nanosleep(128);
                    }
                }
            } else {
                if (threadIdx.x < kRows) {
                    while (atomicAdd(
                               ready_count + transform * 64 + threadIdx.x,
                               0U) == 0U) {
                        __nanosleep(128);
                    }
                }
            }
            __syncthreads();
            if (!recorded_start && threadIdx.x == 0) {
                atomicMin(trace + 2, streaming_globaltimer());
                recorded_start = true;
            }
            if constexpr (Resident2D || ResidentQuarter) {
                appt_register_final_half(tile, state1, transform, c, 0, op);
            } else {
                appt_register_tail_half<WarpRows, WarpColumns, Radix8Core,
                                        kGroupedProducer, Word, Threads>(
                    tile, state1, transform, c, 0, coefficients,
                    coefficients_shoup, op);
            }
#pragma unroll
            for (std::uint32_t slot = 0; slot < kItemsPerThread; ++slot) {
                const std::uint32_t item = threadIdx.x + slot * Threads;
                const std::uint32_t a = item / kColumns;
                const std::uint32_t d = item & (kColumns - 1);
                const std::uint32_t register_slot =
                    slot >= kFirstRegisterSlot ? slot - kFirstRegisterSlot : 0;
                if (item < kSavedWords) {
                    saved_low[item] = tile[a * kPitch + d];
                } else {
                    low_values[register_slot] = tile[a * kPitch + d];
                }
            }
            __syncthreads();

            if constexpr (kGroupedProducer) {
                constexpr std::uint32_t kHalfAGroups =
                    kRows / ProducerAGroup;
                if (threadIdx.x < kHalfAGroups) {
                    while (atomicAdd(ready_count +
                                         transform * kProducerAGroups +
                                         kHalfAGroups + threadIdx.x,
                                     0U) < ((Resident2D || ResidentQuarter)
                                                ? kRows
                                                : kProducerTasksPerAGroup)) {
                        __nanosleep(128);
                    }
                }
            } else if constexpr (AggregateReady) {
                if (threadIdx.x == 0) {
                    while (atomicAdd(ready_count + 2 * transform + 1, 0U) <
                           kRows) {
                        __nanosleep(128);
                    }
                }
            } else {
                if (threadIdx.x < kRows) {
                    while (atomicAdd(ready_count + transform * 64 + kRows +
                                         threadIdx.x,
                                     0U) == 0U) {
                        __nanosleep(128);
                    }
                }
            }
            __syncthreads();
            if constexpr (Resident2D || ResidentQuarter) {
                appt_register_final_half(
                    tile, state1, transform, c, kRows, op);
            } else {
                appt_register_tail_half<WarpRows, WarpColumns, Radix8Core,
                                        kGroupedProducer, Word, Threads>(
                    tile, state1, transform, c, kRows, coefficients,
                    coefficients_shoup, op);
            }
#pragma unroll
            for (std::uint32_t slot = 0; slot < kItemsPerThread; ++slot) {
                const std::uint32_t item = threadIdx.x + slot * Threads;
                const std::uint32_t a = item / kColumns;
                const std::uint32_t d = item & (kColumns - 1);
                const std::uint32_t token = d * kColumns + c;
                const std::uint32_t register_slot =
                    slot >= kFirstRegisterSlot ? slot - kFirstRegisterSlot : 0;
                Word low = item < kSavedWords
                               ? saved_low[item]
                               : low_values[register_slot];
                Word high = tile[a * kPitch + d];
                if constexpr (!WriterFinal) {
                    op.apply(19, token + (a << 14), low, high);
                }
                const std::uint32_t natural_low = token + (a << 14);
                const std::uint32_t natural_high =
                    token + ((a + kRows) << 14);
                Word* publish_state = kGroupedProducer
                                          ? state1 + transforms * kN
                                          : state1;
                Word* publish = NaturalOutput ? publish_state : output;
                publish[transform * kN +
                        appt_static_index_log20<FragmentWidth>(natural_low)] =
                    WriterFinal
                        ? low
                        : op.finalize(low, static_cast<std::uint32_t>(kN));
                publish[transform * kN +
                        appt_static_index_log20<FragmentWidth>(natural_high)] =
                    WriterFinal
                        ? high
                        : op.finalize(high, static_cast<std::uint32_t>(kN));
            }
            __syncthreads();
            if constexpr (NaturalOutput) {
                if (threadIdx.x == 0) {
                    __threadfence();
                    atomicAdd(ready_count + input_counter_count +
                                  transform * kFragmentGroups +
                                  c / FragmentWidth,
                              1U);
                }
            }
            }
        }
    } else {
        const bool role_data_time =
            TransformInterleave && (data_time_role_mask & 4U) != 0;
        constexpr std::uint32_t kWriterPitch = FragmentWidth + 1;
        constexpr std::uint32_t kDGroups = kColumns / 32;
        constexpr std::uint32_t kWriterHalfWords = 32 * kWriterPitch;
        constexpr std::uint32_t kWriterTileWords =
            (WriterFinal ? 2 : 1) * kWriterHalfWords;
        constexpr std::uint32_t kWriterWarpsByStorage =
            kSharedBudgetWords / kWriterTileWords;
        constexpr std::uint32_t kWriterWarps =
            kWriterWarpsByStorage < Threads / 32
                ? kWriterWarpsByStorage
                : Threads / 32;
        static_assert(kWriterWarps > 0);
        const std::uint32_t writer_warp = threadIdx.x >> 5;
        const std::uint32_t lane = threadIdx.x & 31U;
        constexpr std::uint32_t kWriterARows = WriterFinal ? kRows : 64;
        constexpr std::uint64_t kWriterSpatialTasks =
            kWriterARows * kFragmentGroups * kDGroups;
        const std::uint64_t task_count = role_data_time
            ? ((transforms + data_time - 1) / data_time) * kWriterSpatialTasks
            : transforms * kWriterSpatialTasks;
        const Word* writer_state = kGroupedProducer
                                       ? state1 + transforms * kN
                                       : state1;
        if (writer_warp < kWriterWarps) {
            Word* writer_tile = tile + writer_warp * kWriterTileWords;
            Word* writer_high = writer_tile + kWriterHalfWords;
            for (std::uint64_t task_base =
                     (static_cast<std::uint64_t>(local_block) * kWriterWarps +
                      writer_warp) * writer_tiles_per_cta;
                 task_base < task_count;
                 task_base += static_cast<std::uint64_t>(role_blocks) *
                              kWriterWarps * writer_tiles_per_cta) {
                for (std::uint32_t tile_index = 0;
                     tile_index < writer_tiles_per_cta; ++tile_index) {
                    const std::uint64_t task = task_base + tile_index;
                    if (task >= task_count) break;
                    std::uint64_t transform = 0;
                    std::uint64_t coordinates = 0;
                    if constexpr (TransformInterleave) {
                        if (role_data_time) {
                            if (!appt_decode_data_time_task(
                                    task, kWriterSpatialTasks, transforms,
                                    data_time, transform, coordinates)) {
                                continue;
                            }
                        } else {
                            transform = task / kWriterSpatialTasks;
                            coordinates = task % kWriterSpatialTasks;
                        }
                    } else {
                        transform = task / kWriterSpatialTasks;
                        coordinates = task % kWriterSpatialTasks;
                    }
                    const std::uint32_t d_base =
                        static_cast<std::uint32_t>(coordinates % kDGroups) * 32;
                    coordinates /= kDGroups;
                    const std::uint32_t c_group = static_cast<std::uint32_t>(
                        coordinates % kFragmentGroups);
                    coordinates /= kFragmentGroups;
                    const std::uint32_t a = static_cast<std::uint32_t>(
                        coordinates % kWriterARows);
                    const std::uint64_t transform_base = transform;
                    const std::uint32_t transform_iterations =
                        role_data_time ? data_time : 1;
                    for (std::uint32_t time = 0;
                         time < transform_iterations; ++time) {
                    const std::uint64_t transform = transform_base + time;
                    if (transform >= transforms) break;
                    if (lane == 0) {
                        while (atomicAdd(ready_count + input_counter_count +
                                             transform * kFragmentGroups +
                                             c_group,
                                         0U) < FragmentWidth) {
                            __nanosleep(128);
                        }
                    }
                    __syncwarp();
                    if (!recorded_start && threadIdx.x == 0) {
                        atomicMin(trace + 4, streaming_globaltimer());
                        recorded_start = true;
                    }
                    const std::uint32_t c_base =
                        c_group * FragmentWidth;
                    for (std::uint32_t item = lane;
                         item < FragmentWidth * 32; item += 32) {
                        const std::uint32_t c_local = item / 32;
                        const std::uint32_t d = d_base + (item & 31U);
                        const std::uint32_t natural =
                            (a << 14) | (d << 7) | (c_base + c_local);
                        writer_tile[(d - d_base) * kWriterPitch + c_local] =
                            writer_state[transform * kN +
                                   appt_static_index_log20<FragmentWidth>(natural)];
                        if constexpr (WriterFinal) {
                            const std::uint32_t natural_high =
                                ((a + kRows) << 14) | (d << 7) |
                                (c_base + c_local);
                            writer_high[(d - d_base) * kWriterPitch +
                                        c_local] =
                                writer_state[
                                    transform * kN +
                                    appt_static_index_log20<FragmentWidth>(
                                        natural_high)];
                        }
                    }
                    __syncwarp();
                    for (std::uint32_t item = lane;
                         item < FragmentWidth * 32; item += 32) {
                        const std::uint32_t d_local =
                            item / FragmentWidth;
                        const std::uint32_t c_local =
                            item % FragmentWidth;
                        const std::uint32_t natural =
                            (a << 14) | ((d_base + d_local) << 7) |
                            (c_base + c_local);
                        if constexpr (WriterFinal) {
                            const std::uint32_t natural_high =
                                natural + (kRows << 14);
                            Word low = writer_tile[
                                d_local * kWriterPitch + c_local];
                            Word high = writer_high[
                                d_local * kWriterPitch + c_local];
                            op.apply(19, natural, low, high);
                            output[transform * kN + natural] = op.finalize(
                                low, static_cast<std::uint32_t>(kN));
                            output[transform * kN + natural_high] = op.finalize(
                                high, static_cast<std::uint32_t>(kN));
                        } else {
                            output[transform * kN + natural] =
                                writer_tile[d_local * kWriterPitch + c_local];
                        }
                    }
                    __syncwarp();
                    }
                }
            }
        }
    }
    if (recorded_start && threadIdx.x == 0) {
        atomicMax(trace + (producer ? 1 : tail ? 3 : 5),
                  streaming_globaltimer());
    }
}

template <typename Word>
__global__ void hierarchical_streaming_kernel(
    const Word* input, Word* output, Word* boundaries, unsigned int* ready,
    std::uint64_t transforms, const Word* twiddles, const Word* twiddles_shoup,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* consume_state, unsigned int* consumed_epoch,
    unsigned long long* trace, NttStageOperatorT<Word> op,
    NttStreamingDescriptor descriptor) {
    namespace cg = cooperative_groups;
    const auto grid = cg::this_grid();
    const std::uint64_t n = 1ULL << descriptor.log_n;
    const std::uint64_t counter_count = descriptor.counter_base[descriptor.count];
    const std::uint64_t state_count = descriptor.state_base[descriptor.count];

    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < counter_count; index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready[index] = 0;
    }
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < state_count; index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        consume_state[index] = 0;
        consumed_epoch[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 2 * descriptor.count; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    grid.sync();

    // Interleave roles in launch order so each residency wave contains both
    // producers and consumers. role_blocks still permits service-time weighting.
    std::uint32_t segment = 0;
    std::uint32_t role_block = 0;
    std::uint32_t launch_slot = blockIdx.x;
    bool role_found = false;
    for (std::uint32_t round = 0; !role_found; ++round) {
        for (std::uint32_t candidate = 0; candidate < descriptor.count; ++candidate) {
            if (round < descriptor.role_blocks[candidate]) {
                if (launch_slot == 0) {
                    segment = candidate;
                    role_block = round;
                    role_found = true;
                    break;
                }
                --launch_slot;
            }
        }
    }

    const std::uint32_t segment_log = descriptor.segment_log[segment];
    const std::uint32_t prefix_log = descriptor.prefix_log[segment];
    const std::uint32_t remaining_log = descriptor.remaining_log[segment];
    const std::uint32_t segment_n = 1U << segment_log;
    const std::uint32_t prefix_n = 1U << prefix_log;
    const std::uint32_t remaining_n = 1U << remaining_log;
    const std::uint32_t task_log = prefix_log + remaining_log;
    const std::uint64_t tasks_per_transform = static_cast<std::uint64_t>(prefix_n) * remaining_n;
    const std::uint64_t total_tasks = transforms * tasks_per_transform;
    const std::uint32_t role_blocks = descriptor.role_blocks[segment];
    const std::uint32_t lane = threadIdx.x & (warpSize - 1U);
    const std::uint32_t warp = threadIdx.x / warpSize;
    const std::uint32_t warps_per_block = blockDim.x / warpSize;
    const std::uint32_t role_workers = role_blocks * warps_per_block;
    const std::uint32_t role_worker = role_block * warps_per_block + warp;
    extern __shared__ __align__(16) unsigned char storage[];
    Word* tile = reinterpret_cast<Word*>(storage) +
                 static_cast<std::uint64_t>(warp) * segment_n;
    bool trace_started = false;

    for (std::uint64_t task = role_worker; task < total_tasks; task += role_workers) {
        const std::uint64_t transform = task >> task_log;
        const std::uint64_t within = task & (tasks_per_transform - 1U);
        const std::uint32_t prefix = static_cast<std::uint32_t>(within >> remaining_log);
        const std::uint32_t remainder = static_cast<std::uint32_t>(within) & (remaining_n - 1U);
        if (segment + 1 < descriptor.count &&
            descriptor.boundary_slots[segment] < transforms &&
            transform >= descriptor.boundary_slots[segment]) {
            const std::uint32_t slot =
                static_cast<std::uint32_t>(transform % descriptor.boundary_slots[segment]);
            const unsigned int required_epoch = static_cast<unsigned int>(
                transform - descriptor.boundary_slots[segment] + 1);
            unsigned int* completed = consumed_epoch + descriptor.state_base[segment] + slot;
            while (atomicAdd(completed, 0U) < required_epoch) {
#if __CUDA_ARCH__ >= 700
                __nanosleep(64);
#endif
            }
        }
        const std::uint32_t source_slot = segment == 0 ? 0 : static_cast<std::uint32_t>(
            descriptor.boundary_slots[segment - 1] == transforms
                ? transform
                : transform % descriptor.boundary_slots[segment - 1]);
        const Word* source = segment == 0 ? input : boundaries +
            descriptor.boundary_word_base[segment - 1] +
            static_cast<std::uint64_t>(source_slot) * n;

        // A full-scratch edge has a private counter for every downstream
        // subgraph task. Producers aggregate all of that task's dependencies,
        // so only one warp lane polls readiness instead of every data lane.
        const bool aggregated_source = segment != 0 &&
            descriptor.boundary_slots[segment - 1] == transforms;
        if (aggregated_source) {
            if (lane == 0) {
                const std::uint32_t producer_segment_log = descriptor.segment_log[segment - 1];
                const std::uint32_t producer_prefix = prefix >> producer_segment_log;
                const std::uint64_t waves_per_transform =
                    static_cast<std::uint64_t>(1U << descriptor.prefix_log[segment - 1]) *
                    remaining_n;
                const std::uint64_t wave =
                    static_cast<std::uint64_t>(producer_prefix) * remaining_n + remainder;
                unsigned int* counter = ready + descriptor.counter_base[segment] +
                    static_cast<std::uint64_t>(source_slot) * waves_per_transform + wave;
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(*counter);
                while (readiness.load(cuda::memory_order_acquire) < segment_n) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
            }
            __syncwarp();
        }

        for (std::uint32_t element = lane; element < segment_n; element += warpSize) {
            const std::uint64_t logical =
                (static_cast<std::uint64_t>(prefix) * segment_n + element) * remaining_n + remainder;
            if (segment != 0 && !aggregated_source) {
                const std::uint32_t producer_segment_log = descriptor.segment_log[segment - 1];
                const std::uint32_t producer_prefix_log = descriptor.prefix_log[segment - 1];
                const std::uint32_t producer_remaining_log = descriptor.remaining_log[segment - 1];
                const std::uint32_t producer_remaining_n = 1U << producer_remaining_log;
                const std::uint32_t producer_prefix =
                    static_cast<std::uint32_t>(logical >> (producer_segment_log + producer_remaining_log));
                const std::uint32_t producer_remainder =
                    static_cast<std::uint32_t>(logical) & (producer_remaining_n - 1U);
                const std::uint64_t producer_tasks_per_transform =
                    static_cast<std::uint64_t>(1U << producer_prefix_log) * producer_remaining_n;
                const std::uint64_t producer_within =
                    static_cast<std::uint64_t>(producer_prefix) * producer_remaining_n + producer_remainder;
                const std::uint32_t producer_slot = static_cast<std::uint32_t>(
                    transform % descriptor.boundary_slots[segment - 1]);
                const std::uint64_t producer_token =
                    static_cast<std::uint64_t>(producer_slot) * producer_tasks_per_transform + producer_within;
                unsigned int* token = ready + descriptor.counter_base[segment] + producer_token;
                const unsigned int required_epoch = static_cast<unsigned int>(transform + 1);
                while (atomicAdd(token, 0U) < required_epoch) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
                __threadfence();
            }
            const std::uint32_t reversed = __brev(element) >> (32 - segment_log);
            tile[reversed] = segment == 0 ? source[transform * n + logical] : source[logical];
        }
        __syncwarp();
        if (segment != 0 && descriptor.boundary_slots[segment - 1] < transforms &&
            lane == 0) {
            const std::uint32_t edge = segment - 1;
            const std::uint32_t slot = static_cast<std::uint32_t>(
                transform % descriptor.boundary_slots[edge]);
            unsigned long long* state = consume_state + descriptor.state_base[edge] + slot;
            const unsigned int epoch = static_cast<unsigned int>(transform + 1);
            unsigned long long observed = atomicAdd(state, 0ULL);
            unsigned int count = 0;
            while (true) {
                const unsigned int observed_epoch = static_cast<unsigned int>(observed >> 32);
                const unsigned int observed_count = static_cast<unsigned int>(observed);
                count = observed_epoch == epoch ? observed_count + 1U : 1U;
                const unsigned long long desired =
                    (static_cast<unsigned long long>(epoch) << 32) | count;
                const unsigned long long prior = atomicCAS(state, observed, desired);
                if (prior == observed) break;
                observed = prior;
            }
            if (count == tasks_per_transform) {
                __threadfence();
                atomicExch(consumed_epoch + descriptor.state_base[edge] + slot, epoch);
            }
        }
        __syncwarp();
        if (lane == 0 && !trace_started) {
            atomicMin(trace + 2 * segment, streaming_globaltimer());
            trace_started = true;
        }

        std::uint32_t stage = 0;
        while (stage < segment_log) {
            const std::uint32_t stages_left = segment_log - stage;
            const std::uint32_t core = descriptor.core[segment];
            if (core == static_cast<std::uint32_t>(NttSubgraphCore::Radix8) && stages_left >= 3) {
                const std::uint32_t half = 1U << stage;
                for (std::uint32_t work = lane; work < segment_n / 8; work += warpSize) {
                    const std::uint32_t group = work >> stage;
                    const std::uint32_t offset = work & (half - 1U);
                    const std::uint32_t base = group * (half << 3) + offset;
                    Word value[8];
#pragma unroll
                    for (std::uint32_t item = 0; item < 8; ++item) value[item] = tile[base + item * half];
                    op.apply(stage, offset, value[0], value[1]);
                    op.apply(stage, offset, value[2], value[3]);
                    op.apply(stage, offset, value[4], value[5]);
                    op.apply(stage, offset, value[6], value[7]);
                    op.apply(stage + 1, offset, value[0], value[2]);
                    op.apply(stage + 1, half + offset, value[1], value[3]);
                    op.apply(stage + 1, offset, value[4], value[6]);
                    op.apply(stage + 1, half + offset, value[5], value[7]);
                    op.apply(stage + 2, offset, value[0], value[4]);
                    op.apply(stage + 2, half + offset, value[1], value[5]);
                    op.apply(stage + 2, 2 * half + offset, value[2], value[6]);
                    op.apply(stage + 2, 3 * half + offset, value[3], value[7]);
#pragma unroll
                    for (std::uint32_t item = 0; item < 8; ++item) tile[base + item * half] = value[item];
                }
                stage += 3;
            } else if (core != static_cast<std::uint32_t>(NttSubgraphCore::Radix2) && stages_left >= 2) {
                const std::uint32_t half = 1U << stage;
                for (std::uint32_t work = lane; work < segment_n / 4; work += warpSize) {
                    const std::uint32_t group = work >> stage;
                    const std::uint32_t offset = work & (half - 1U);
                    const std::uint32_t base = group * (half << 2) + offset;
                    Word value0 = tile[base];
                    Word value1 = tile[base + half];
                    Word value2 = tile[base + 2 * half];
                    Word value3 = tile[base + 3 * half];
                    op.apply(stage, offset, value0, value1);
                    op.apply(stage, offset, value2, value3);
                    op.apply(stage + 1, offset, value0, value2);
                    op.apply(stage + 1, half + offset, value1, value3);
                    tile[base] = value0;
                    tile[base + half] = value1;
                    tile[base + 2 * half] = value2;
                    tile[base + 3 * half] = value3;
                }
                stage += 2;
            } else {
                const std::uint32_t half = 1U << stage;
                for (std::uint32_t butterfly_index = lane;
                     butterfly_index < segment_n / 2; butterfly_index += warpSize) {
                    const std::uint32_t group = butterfly_index >> stage;
                    const std::uint32_t offset = butterfly_index & (half - 1U);
                    const std::uint32_t left_index = group * (half << 1) + offset;
                    op.apply(stage, offset, tile[left_index], tile[left_index + half]);
                }
                ++stage;
            }
            __syncwarp();
        }

        const bool final_segment = segment + 1 == descriptor.count;
        const std::uint32_t destination_slot = final_segment ? 0 : static_cast<std::uint32_t>(
            descriptor.boundary_slots[segment] == transforms
                ? transform
                : transform % descriptor.boundary_slots[segment]);
        Word* destination = final_segment ? output : boundaries +
            descriptor.boundary_word_base[segment] +
            static_cast<std::uint64_t>(destination_slot) * n;
        for (std::uint32_t frequency = lane; frequency < segment_n; frequency += warpSize) {
            Word value = tile[frequency];
            if (!final_segment) {
                const std::uint64_t exponent =
                    static_cast<std::uint64_t>(prefix_n) * frequency * remainder;
                value = op.multiply(value, root_powers[exponent], root_powers_shoup[exponent]);
            } else {
                value = op.finalize(value, frequency);
            }
            std::uint32_t logical = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(prefix) * segment_n + frequency) * remaining_n + remainder);
            if (final_segment) {
                logical = reverse_streaming_digits(logical, descriptor);
            }
            if (final_segment) destination[transform * n + logical] = value;
            else destination[logical] = value;
        }
        __syncwarp();

        if (!final_segment) {
            if (descriptor.boundary_slots[segment] == transforms) {
                const std::uint32_t next_remaining_log = descriptor.remaining_log[segment + 1];
                const std::uint32_t next_remaining_n = 1U << next_remaining_log;
                const std::uint64_t waves_per_transform =
                    static_cast<std::uint64_t>(prefix_n) * next_remaining_n;
                const std::uint32_t later_remainder = remainder & (next_remaining_n - 1U);
                const std::uint64_t wave =
                    static_cast<std::uint64_t>(prefix) * next_remaining_n + later_remainder;
                if (lane == 0) {
                    unsigned int* counter = ready + descriptor.counter_base[segment + 1] +
                        static_cast<std::uint64_t>(destination_slot) * waves_per_transform + wave;
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(*counter);
                    readiness.fetch_add(1U, cuda::memory_order_acq_rel);
                }
            } else {
                __threadfence();
                if (lane == 0) {
                    const std::uint64_t within_task = task - transform * tasks_per_transform;
                    const std::uint64_t token_index =
                        static_cast<std::uint64_t>(destination_slot) * tasks_per_transform + within_task;
                    unsigned int* token = ready + descriptor.counter_base[segment + 1] + token_index;
                    atomicExch(token, static_cast<unsigned int>(transform + 1));
                }
            }
        }
        __syncwarp();
    }
    if (lane == 0 && trace_started) {
        atomicMax(trace + 2 * segment + 1, streaming_globaltimer());
    }
}

template <typename Word, std::uint32_t RowsPerBlock>
__global__ void hierarchical_streaming_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLocalLog = 10;
    constexpr std::uint32_t kLocalN = 1U << kLocalLog;
    constexpr std::uint32_t kN = kLocalN * kLocalN;
    constexpr std::uint32_t kStride = kLocalN + 1;
    constexpr std::uint32_t kGroupsPerTransform = kLocalN / RowsPerBlock;
    extern __shared__ __align__(16) unsigned char storage[];
    __shared__ unsigned int consumer_ready;
    Word* state = reinterpret_cast<Word*>(storage);

    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < transforms; index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 4; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    cg::this_grid().sync();

    const std::uint32_t consumer_blocks = gridDim.x - producer_blocks;
    const std::uint32_t paired_blocks = min(producer_blocks, consumer_blocks);
    bool consumer_role = false;
    std::uint32_t role_block = 0;
    if (blockIdx.x < 2U * paired_blocks) {
        consumer_role = (blockIdx.x & 1U) != 0;
        role_block = blockIdx.x >> 1;
    } else {
        consumer_role = consumer_blocks > producer_blocks;
        role_block = paired_blocks + blockIdx.x - 2U * paired_blocks;
    }
    const std::uint32_t role_blocks = consumer_role ? consumer_blocks : producer_blocks;
    const std::uint64_t total_groups = transforms * kGroupsPerTransform;
    bool trace_started = false;

    for (std::uint64_t group = role_block; group < total_groups; group += role_blocks) {
        const std::uint64_t transform = group >> 8;
        const std::uint32_t group_in_transform = static_cast<std::uint32_t>(group) & 255U;
        const std::uint32_t row_base = group_in_transform * RowsPerBlock;
        const std::uint64_t transform_base = transform * kN;

        if (consumer_role) {
            if (threadIdx.x == 0) {
                consumer_ready = 0;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(ready[transform]);
                while (readiness.load(cuda::memory_order_acquire) < kGroupsPerTransform) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> block_readiness(consumer_ready);
                block_readiness.store(1U, cuda::memory_order_release);
            }
            if ((threadIdx.x & 31U) == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> block_readiness(consumer_ready);
                while (block_readiness.load(cuda::memory_order_acquire) == 0U) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
            }
            __syncwarp();
            if (threadIdx.x == 0 && !trace_started) {
                atomicMin(trace + 2, streaming_globaltimer());
                trace_started = true;
            }
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kLocalN; linear += blockDim.x) {
                const std::uint32_t row = linear >> kLocalLog;
                const std::uint32_t second_index = linear & (kLocalN - 1U);
                const std::uint32_t first_index = row_base + row;
                const std::uint32_t reversed = __brev(second_index) >> (32 - kLocalLog);
                state[row * kStride + reversed] =
                    boundary[transform_base + static_cast<std::uint64_t>(first_index) *
                                              kLocalN + second_index];
            }
            __syncthreads();
            detail::hierarchical_resident_radix4<true, Word, kLocalLog, RowsPerBlock>(
                state, op, root_powers, root_powers_shoup, row_base);
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kLocalN; linear += blockDim.x) {
                const std::uint32_t second_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - second_index * RowsPerBlock;
                const std::uint32_t first_index = row_base + row;
                output[transform_base + static_cast<std::uint64_t>(second_index) *
                                        kLocalN + first_index] =
                    op.finalize(state[row * kStride + second_index], kN);
            }
            __syncthreads();
        } else {
            if (threadIdx.x == 0 && !trace_started) {
                atomicMin(trace, streaming_globaltimer());
                trace_started = true;
            }
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kLocalN; linear += blockDim.x) {
                const std::uint32_t first_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - first_index * RowsPerBlock;
                const std::uint32_t second_index = row_base + row;
                const std::uint32_t reversed = __brev(first_index) >> (32 - kLocalLog);
                state[row * kStride + reversed] =
                    input[transform_base + static_cast<std::uint64_t>(first_index) *
                                           kLocalN + second_index];
            }
            __syncthreads();
            detail::hierarchical_resident_radix4<false, Word, kLocalLog, RowsPerBlock>(state, op);
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kLocalN; linear += blockDim.x) {
                const std::uint32_t first_index = linear / RowsPerBlock;
                const std::uint32_t row = linear - first_index * RowsPerBlock;
                const std::uint32_t second_index = row_base + row;
                boundary[transform_base + static_cast<std::uint64_t>(first_index) *
                                          kLocalN + second_index] =
                    state[row * kStride + first_index];
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(ready[transform]);
                // The CTA barrier makes every lane's boundary writes happen
                // before this release; the consumer's acquire publishes the
                // complete packet, not just lane zero's stores.
                readiness.fetch_add(1U, cuda::memory_order_release);
            }
        }
    }
    if (threadIdx.x == 0 && trace_started) {
        atomicMax(trace + (consumer_role ? 3 : 1), streaming_globaltimer());
    }
}

// Generated two-level physical template. Both roles use the same resident
// radix-4 codelet shape; FirstLogN/SecondLogN describe the logical stage fold,
// while each role's data_time independently controls traversal of transforms
// at a fixed spatial group. The first emitted instance is 10+10 on V100.
template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t RowsPerBlock>
__global__ void hierarchical_homogeneous_two_level_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kFirstN = 1U << FirstLogN;
    constexpr std::uint32_t kSecondN = 1U << SecondLogN;
    constexpr std::uint32_t kN = kFirstN * kSecondN;
    constexpr std::uint32_t kFirstStride = kFirstN + 1;
    constexpr std::uint32_t kSecondStride = kSecondN + 1;
    constexpr std::uint32_t kProducerGroups = kSecondN / RowsPerBlock;
    constexpr std::uint32_t kConsumerGroups = kFirstN / RowsPerBlock;
    extern __shared__ __align__(16) unsigned char storage[];
    __shared__ unsigned int consumer_ready;
    Word* state = reinterpret_cast<Word*>(storage);

    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < transforms;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 4; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    cg::this_grid().sync();

    const std::uint32_t consumer_blocks = gridDim.x - producer_blocks;
    const std::uint32_t paired_blocks = min(producer_blocks, consumer_blocks);
    bool consumer_role = false;
    std::uint32_t role_block = 0;
    if (blockIdx.x < 2U * paired_blocks) {
        consumer_role = (blockIdx.x & 1U) != 0;
        role_block = blockIdx.x >> 1;
    } else {
        consumer_role = consumer_blocks > producer_blocks;
        role_block = paired_blocks + blockIdx.x - 2U * paired_blocks;
    }
    const std::uint32_t role_blocks =
        consumer_role ? consumer_blocks : producer_blocks;
    const std::uint32_t groups =
        consumer_role ? kConsumerGroups : kProducerGroups;
    const std::uint32_t data_time =
        consumer_role ? consumer_data_time : producer_data_time;
    const std::uint64_t transform_packets =
        (transforms + data_time - 1U) / data_time;
    const std::uint64_t tasks = transform_packets * groups;
    bool trace_started = false;

    for (std::uint64_t task = role_block; task < tasks; task += role_blocks) {
        const std::uint32_t group_in_transform =
            static_cast<std::uint32_t>(task % groups);
        const std::uint64_t transform_begin = (task / groups) * data_time;
        const std::uint32_t row_base = group_in_transform * RowsPerBlock;

        // D_t is deliberately inside the spatial task: a resident CTA reuses
        // its role and codelet while walking homogeneous batch subgraphs.
        for (std::uint32_t time = 0; time < data_time; ++time) {
            const std::uint64_t transform = transform_begin + time;
            if (transform >= transforms) break;
            const std::uint64_t transform_base = transform * kN;

            if (consumer_role) {
                if (threadIdx.x == 0) consumer_ready = 0;
                __syncthreads();
                if (threadIdx.x == 0) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        readiness(ready[transform]);
                    while (readiness.load(cuda::memory_order_acquire) <
                           kProducerGroups) {
#if __CUDA_ARCH__ >= 700
                        __nanosleep(64);
#endif
                    }
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                        block_readiness(consumer_ready);
                    block_readiness.store(1U, cuda::memory_order_release);
                }
                if ((threadIdx.x & 31U) == 0) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                        block_readiness(consumer_ready);
                    while (block_readiness.load(cuda::memory_order_acquire) == 0U) {
#if __CUDA_ARCH__ >= 700
                        __nanosleep(64);
#endif
                    }
                }
                __syncwarp();
                if (threadIdx.x == 0 && !trace_started) {
                    atomicMin(trace + 2, streaming_globaltimer());
                    trace_started = true;
                }
                for (std::uint32_t linear = threadIdx.x;
                     linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                    const std::uint32_t row = linear >> SecondLogN;
                    const std::uint32_t second_index = linear & (kSecondN - 1U);
                    const std::uint32_t first_index = row_base + row;
                    const std::uint32_t reversed =
                        __brev(second_index) >> (32 - SecondLogN);
                    state[row * kSecondStride + reversed] =
                        boundary[transform_base +
                                 static_cast<std::uint64_t>(first_index) *
                                     kSecondN +
                                 second_index];
                }
                __syncthreads();
                detail::hierarchical_resident_radix4<
                    true, Word, SecondLogN, RowsPerBlock>(
                    state, op, root_powers, root_powers_shoup, row_base);
                for (std::uint32_t linear = threadIdx.x;
                     linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                    const std::uint32_t second_index = linear / RowsPerBlock;
                    const std::uint32_t row = linear - second_index * RowsPerBlock;
                    const std::uint32_t first_index = row_base + row;
                    output[transform_base +
                           static_cast<std::uint64_t>(second_index) * kFirstN +
                           first_index] =
                        op.finalize(state[row * kSecondStride + second_index],
                                    kN);
                }
                __syncthreads();
            } else {
                if (threadIdx.x == 0 && !trace_started) {
                    atomicMin(trace, streaming_globaltimer());
                    trace_started = true;
                }
                for (std::uint32_t linear = threadIdx.x;
                     linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                    const std::uint32_t first_index = linear / RowsPerBlock;
                    const std::uint32_t row = linear - first_index * RowsPerBlock;
                    const std::uint32_t second_index = row_base + row;
                    const std::uint32_t reversed =
                        __brev(first_index) >> (32 - FirstLogN);
                    state[row * kFirstStride + reversed] =
                        input[transform_base +
                              static_cast<std::uint64_t>(first_index) *
                                  kSecondN +
                              second_index];
                }
                __syncthreads();
                detail::hierarchical_resident_radix4<
                    false, Word, FirstLogN, RowsPerBlock>(state, op);
                for (std::uint32_t linear = threadIdx.x;
                     linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                    const std::uint32_t first_index = linear / RowsPerBlock;
                    const std::uint32_t row = linear - first_index * RowsPerBlock;
                    const std::uint32_t second_index = row_base + row;
                    boundary[transform_base +
                             static_cast<std::uint64_t>(first_index) * kSecondN +
                             second_index] =
                        state[row * kFirstStride + first_index];
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        readiness(ready[transform]);
                    readiness.fetch_add(1U, cuda::memory_order_release);
                }
            }
        }
    }
    if (threadIdx.x == 0 && trace_started) {
        atomicMax(trace + (consumer_role ? 3 : 1), streaming_globaltimer());
    }
}

template <bool FusedRows, typename Word, std::uint32_t LocalLogN,
          typename Operator>
__device__ __forceinline__ void hierarchical_warp_register_subgraph(
    Word (&values)[1U << (LocalLogN - 5)], std::uint32_t row,
    const Word* fused_twiddles, const Word* fused_twiddles_shoup,
    Operator op) {
    static_assert(LocalLogN >= 5 && LocalLogN <= 10);
    constexpr std::uint32_t kLocalN = 1U << LocalLogN;
    constexpr std::uint32_t kValuesPerLane = 1U << (LocalLogN - 5);
    const Word* coefficients = FusedRows
                                   ? fused_twiddles +
                                         static_cast<std::uint64_t>(row) *
                                             (kLocalN - 1U)
                                   : op.twiddles;
    const Word* coefficients_shoup =
        FusedRows
            ? fused_twiddles_shoup +
                  static_cast<std::uint64_t>(row) * (kLocalN - 1U)
            : op.twiddles_shoup;
#pragma unroll
    for (std::uint32_t stage = 0; stage < LocalLogN; ++stage) {
        appt_apply_cached_register_stage<Word, kValuesPerLane>(
            values, kLocalN, stage, threadIdx.x & 31U, coefficients,
            coefficients_shoup, op);
    }
}

// Warp-granular instance of the homogeneous two-level template. A warp owns a
// dependency-closed local transform in registers, so eight subgraphs can make
// progress independently inside a 256-thread CTA. The producer publishes the
// second coordinate in bit-reversed physical order; this online remap turns
// the consumer's local bit-reversal gather into a contiguous load.
template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t WarpsPerCta>
__global__ void hierarchical_homogeneous_warp_two_level_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kFirstN = 1U << FirstLogN;
    constexpr std::uint32_t kSecondN = 1U << SecondLogN;
    constexpr std::uint32_t kN = kFirstN * kSecondN;
    constexpr std::uint32_t kFirstValues = 1U << (FirstLogN - 5);
    constexpr std::uint32_t kSecondValues = 1U << (SecondLogN - 5);
    static_assert(FirstLogN >= 5 && FirstLogN <= 10);
    static_assert(SecondLogN >= 5 && SecondLogN <= 10);
    static_assert(WarpsPerCta * 32 == 256);

    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < transforms;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 4; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    cg::this_grid().sync();

    const std::uint32_t consumer_blocks = gridDim.x - producer_blocks;
    const std::uint32_t paired_blocks = min(producer_blocks, consumer_blocks);
    bool consumer_role = false;
    std::uint32_t role_block = 0;
    if (blockIdx.x < 2U * paired_blocks) {
        consumer_role = (blockIdx.x & 1U) != 0;
        role_block = blockIdx.x >> 1;
    } else {
        consumer_role = consumer_blocks > producer_blocks;
        role_block = paired_blocks + blockIdx.x - 2U * paired_blocks;
    }
    const std::uint32_t role_blocks =
        consumer_role ? consumer_blocks : producer_blocks;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t rows = consumer_role ? kFirstN : kSecondN;
    const std::uint32_t data_time =
        consumer_role ? consumer_data_time : producer_data_time;
    const std::uint64_t transform_packets =
        (transforms + data_time - 1U) / data_time;
    const std::uint64_t tasks = transform_packets * rows;
    const std::uint64_t warp_task =
        static_cast<std::uint64_t>(role_block) * WarpsPerCta + warp;
    const std::uint64_t task_stride =
        static_cast<std::uint64_t>(role_blocks) * WarpsPerCta;
    bool trace_started = false;

    for (std::uint64_t task = warp_task; task < tasks; task += task_stride) {
        const std::uint32_t row = static_cast<std::uint32_t>(task % rows);
        const std::uint64_t transform_begin = (task / rows) * data_time;
#pragma unroll 1
        for (std::uint32_t time = 0; time < data_time; ++time) {
            const std::uint64_t transform = transform_begin + time;
            if (transform >= transforms) break;
            const std::uint64_t transform_base = transform * kN;

            if (consumer_role) {
                unsigned int published = 0;
                if (lane == 0) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        readiness(ready[transform]);
                    do {
                        published = readiness.load(cuda::memory_order_acquire);
                        if (published < kSecondN) {
#if __CUDA_ARCH__ >= 700
                            __nanosleep(64);
#endif
                        }
                    } while (published < kSecondN);
                }
                published = __shfl_sync(0xffffffffU, published, 0);
                if (lane == 0 && !trace_started) {
                    atomicMin(trace + 2, streaming_globaltimer());
                    trace_started = true;
                }
                Word values[kSecondValues];
#pragma unroll
                for (std::uint32_t slot = 0; slot < kSecondValues; ++slot) {
                    const std::uint32_t local = (slot << 5) + lane;
                    values[slot] = boundary[
                        transform_base +
                        static_cast<std::uint64_t>(row) * kSecondN + local];
                }
                hierarchical_warp_register_subgraph<true, Word, SecondLogN>(
                    values, row, root_powers, root_powers_shoup, op);
#pragma unroll
                for (std::uint32_t slot = 0; slot < kSecondValues; ++slot) {
                    const std::uint32_t second_index = (slot << 5) + lane;
                    output[transform_base +
                           static_cast<std::uint64_t>(second_index) * kFirstN +
                           row] = op.finalize(values[slot], kN);
                }
                __syncwarp();
            } else {
                if (lane == 0 && !trace_started) {
                    atomicMin(trace, streaming_globaltimer());
                    trace_started = true;
                }
                Word values[kFirstValues];
#pragma unroll
                for (std::uint32_t slot = 0; slot < kFirstValues; ++slot) {
                    const std::uint32_t local = (slot << 5) + lane;
                    const std::uint32_t first_index =
                        __brev(local) >> (32 - FirstLogN);
                    values[slot] = input[
                        transform_base +
                        static_cast<std::uint64_t>(first_index) * kSecondN + row];
                }
                hierarchical_warp_register_subgraph<false, Word, FirstLogN>(
                    values, row, nullptr, nullptr, op);
                const std::uint32_t physical_second =
                    __brev(row) >> (32 - SecondLogN);
#pragma unroll
                for (std::uint32_t slot = 0; slot < kFirstValues; ++slot) {
                    const std::uint32_t first_index = (slot << 5) + lane;
                    boundary[transform_base +
                             static_cast<std::uint64_t>(first_index) *
                                 kSecondN +
                             physical_second] = values[slot];
                }
                __syncwarp();
                if (lane == 0) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        readiness(ready[transform]);
                    readiness.fetch_add(1U, cuda::memory_order_release);
                }
            }
        }
    }
    if (lane == 0 && trace_started) {
        atomicMax(trace + (consumer_role ? 3 : 1), streaming_globaltimer());
    }
}

// Four warps compute the first eight stages of one 1024-point row in
// registers, exchange one 4x256 tile, and finish the final two stages. Named
// 128-thread barriers keep the two warp groups in a CTA independent.
__device__ __forceinline__ void hierarchical_warp_group_barrier(
    std::uint32_t barrier) {
    asm volatile("bar.sync %0, 128;" : : "r"(barrier) : "memory");
}

__device__ __forceinline__ std::uint32_t
hierarchical_warp256_io_swizzle(std::uint32_t index) {
    return index ^ (index >> 5);
}

template <std::uint32_t Rows>
__device__ __forceinline__ constexpr std::uint32_t
hierarchical_warp256_io_stride() {
    static_assert(Rows == 2 || Rows == 4 || Rows == 8);
    // Odd skews spread the interleaved row-packet transpose across banks.
    return 1024U + (Rows == 2 ? 15U : 7U);
}

template <std::uint32_t Rows>
__device__ __forceinline__ std::uint32_t
hierarchical_warp256_io_index(std::uint32_t row, std::uint32_t major) {
    return row * hierarchical_warp256_io_stride<Rows>() +
           hierarchical_warp256_io_swizzle(major);
}

template <typename Word, bool ConsumerRole, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t CoefficientReuseStages = 0>
__device__ __forceinline__ void hierarchical_warp256_row(
    const Word* source, Word* destination, Word* exchange,
    std::uint64_t transform_base, std::uint32_t row,
    std::uint32_t quarter, Word* writer_tile, std::uint32_t writer_row,
    const Word* root_powers,
    const Word* root_powers_shoup, NttStageOperatorT<Word> op) {
    constexpr std::uint32_t kLocalLog = 10;
    constexpr std::uint32_t kLocalN = 1U << kLocalLog;
    constexpr std::uint32_t kQuarterLog = 8;
    constexpr std::uint32_t kQuarterN = 1U << kQuarterLog;
    constexpr std::uint32_t kValuesPerLane = kQuarterN / 32;
    constexpr std::uint32_t kWriterRows = StaticWriter
        ? (WriterRows == 0 ? 32U / sizeof(Word) : WriterRows) : 1U;
    constexpr bool kRowMajorIo =
        StaticReader && sizeof(Word) == 4 && kWriterRows == 4;
    static_assert(!StaticReader || StaticWriter);
    const std::uint32_t lane = threadIdx.x & 31U;
    Word values[kValuesPerLane];

#pragma unroll
    for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
        const std::uint32_t local = (slot << 5) + lane;
        if constexpr (StaticReader) {
            const std::uint32_t value_index =
                quarter * kQuarterN + local;
            if constexpr (kRowMajorIo) {
                values[slot] = writer_tile[
                    hierarchical_warp256_io_index<kWriterRows>(writer_row,
                                                                value_index)];
            } else {
                const std::uint32_t major =
                    hierarchical_warp256_io_swizzle(value_index);
                const std::uint32_t physical_row =
                    writer_row ^ (major & (kWriterRows - 1U));
                values[slot] =
                    writer_tile[major * kWriterRows + physical_row];
            }
        } else if constexpr (ConsumerRole) {
            values[slot] = source[
                transform_base + static_cast<std::uint64_t>(row) * kLocalN +
                quarter * kQuarterN + local];
        } else {
            const std::uint32_t first_index =
                ((__brev(local) >> (32 - kQuarterLog)) << 2) |
                (__brev(quarter) >> 30);
            values[slot] = source[
                transform_base +
                static_cast<std::uint64_t>(first_index) * kLocalN + row];
        }
    }

    const Word* coefficients = ConsumerRole
                                   ? root_powers +
                                         static_cast<std::uint64_t>(row) *
                                             (kLocalN - 1U)
                                   : op.twiddles;
    const Word* coefficients_shoup =
        ConsumerRole
            ? root_powers_shoup +
                  static_cast<std::uint64_t>(row) * (kLocalN - 1U)
            : op.twiddles_shoup;
#pragma unroll
    for (std::uint32_t stage = 0; stage < kQuarterLog; ++stage) {
        if constexpr (CoefficientReuseStages != 0) {
            if (stage < CoefficientReuseStages) {
                appt_apply_reused_register_stage<Word, kValuesPerLane>(
                    values, kQuarterN, stage, lane, coefficients,
                    coefficients_shoup, op);
            } else {
                appt_apply_cached_register_stage<Word, kValuesPerLane>(
                    values, kQuarterN, stage, lane, coefficients,
                    coefficients_shoup, op);
            }
        } else {
            appt_apply_cached_register_stage<Word, kValuesPerLane>(
                values, kQuarterN, stage, lane, coefficients,
                coefficients_shoup, op);
        }
    }
#pragma unroll
    for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
        exchange[quarter * kQuarterN + (slot << 5) + lane] = values[slot];
    }

    const std::uint32_t group = (threadIdx.x >> 5) >> 2;
    hierarchical_warp_group_barrier(group + 1U);

#pragma unroll
    for (std::uint32_t wave = 0; wave < 2; ++wave) {
        const std::uint32_t offset = quarter * 64U + (wave << 5) + lane;
        Word a = exchange[offset];
        Word b = exchange[kQuarterN + offset];
        Word c = exchange[2U * kQuarterN + offset];
        Word d = exchange[3U * kQuarterN + offset];
        op.apply_coefficient(coefficients[255U + offset],
                             coefficients_shoup[255U + offset], a, b);
        op.apply_coefficient(coefficients[255U + offset],
                             coefficients_shoup[255U + offset], c, d);
        op.apply_coefficient(coefficients[511U + offset],
                             coefficients_shoup[511U + offset], a, c);
        op.apply_coefficient(coefficients[511U + kQuarterN + offset],
                             coefficients_shoup[511U + kQuarterN + offset],
                             b, d);
        const Word finished[4] = {a, b, c, d};
#pragma unroll
        for (std::uint32_t output_quarter = 0; output_quarter < 4;
             ++output_quarter) {
            const std::uint32_t local = output_quarter * kQuarterN + offset;
            Word result = finished[output_quarter];
            if constexpr (ConsumerRole) {
                result = op.finalize(result, kLocalN * kLocalN);
            }
            if constexpr (StaticWriter) {
                if constexpr (kRowMajorIo) {
                    writer_tile[
                        hierarchical_warp256_io_index<kWriterRows>(writer_row,
                                                                    local)] =
                        result;
                } else {
                    const std::uint32_t major = StaticReader
                        ? hierarchical_warp256_io_swizzle(local) : local;
                    const std::uint32_t physical_row =
                        writer_row ^ (major & (kWriterRows - 1U));
                    writer_tile[major * kWriterRows + physical_row] = result;
                }
            } else if constexpr (ConsumerRole) {
                destination[
                    transform_base +
                    static_cast<std::uint64_t>(local) * kLocalN + row] =
                    result;
            } else {
                const std::uint32_t physical_second =
                    __brev(row) >> (32 - kLocalLog);
                destination[
                    transform_base +
                    static_cast<std::uint64_t>(local) * kLocalN +
                    physical_second] = finished[output_quarter];
            }
        }
    }
    hierarchical_warp_group_barrier(group + 1U);
}

template <typename Word, bool ConsumerRole, std::uint32_t WriterRows,
          std::uint32_t WarpTileLog,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false>
__device__ __forceinline__ void hierarchical_warp_small_static_io_row(
    Word* exchange, std::uint32_t row, std::uint32_t quarter,
    Word* writer_tile, std::uint32_t writer_row,
    const Word* root_powers, const Word* root_powers_shoup,
    NttStageOperatorT<Word> op) {
    static_assert(WarpTileLog == 6 || WarpTileLog == 7);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kWarpTileN = 1U << WarpTileLog;
    constexpr std::uint32_t kValuesPerLane = kWarpTileN / 32U;
    constexpr std::uint32_t kWaves = kLocalN / (4U * kWarpTileN);
    constexpr bool kRowMajorIo = sizeof(Word) == 4 && WriterRows == 4;
    static_assert(!VectorRadix4Prefix ||
                  (sizeof(Word) == 4 && WarpTileLog == 7 &&
                   kValuesPerLane == 4));
    static_assert(!(PackedStage6 && DistributedPackedStage6));
    static_assert(!(PackedStage6 || DistributedPackedStage6) ||
                  (VectorRadix4Prefix && CoefficientReuseStages == 6));
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t group_thread = quarter * 32U + lane;
    const std::uint32_t group = (threadIdx.x >> 5) >> 2;
    const Word* coefficients = ConsumerRole
                                   ? root_powers +
                                         static_cast<std::uint64_t>(row) *
                                             (kLocalN - 1U)
                                   : op.twiddles;
    const Word* coefficients_shoup =
        ConsumerRole
            ? root_powers_shoup +
                  static_cast<std::uint64_t>(row) * (kLocalN - 1U)
            : op.twiddles_shoup;
#pragma unroll
    for (std::uint32_t wave = 0; wave < kWaves; ++wave) {
        const std::uint32_t tile_base =
            (wave * 4U + quarter) * kWarpTileN;
        Word values[kValuesPerLane];
#pragma unroll
        for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
            const std::uint32_t local =
                tile_base + (VectorRadix4Prefix ? (lane << 2) + slot
                                                : (slot << 5) + lane);
            if constexpr (kRowMajorIo) {
                values[slot] = writer_tile[
                    hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                               local)];
            } else {
                const std::uint32_t major =
                    hierarchical_warp256_io_swizzle(local);
                const std::uint32_t physical_row =
                    writer_row ^ (major & (WriterRows - 1U));
                values[slot] =
                    writer_tile[major * WriterRows + physical_row];
            }
        }
        if constexpr (VectorRadix4Prefix) {
            const Word stage0_coefficient = coefficients[0];
            const Word stage0_coefficient_shoup = coefficients_shoup[0];
            op.apply_coefficient(stage0_coefficient,
                                 stage0_coefficient_shoup, values[0],
                                 values[1]);
            op.apply_coefficient(stage0_coefficient,
                                 stage0_coefficient_shoup, values[2],
                                 values[3]);
            op.apply_coefficient(coefficients[1], coefficients_shoup[1],
                                 values[0], values[2]);
            op.apply_coefficient(coefficients[2], coefficients_shoup[2],
                                 values[1], values[3]);
#pragma unroll
            for (std::uint32_t stage = 2; stage < WarpTileLog; ++stage) {
                const std::uint32_t half = 1U << stage;
                const std::uint32_t coefficient_base = half - 1U;
                const int lane_delta = 1 << (stage - 2U);
                Word lane_coefficient = Word{0};
                Word lane_coefficient_shoup = Word{0};
                Word lane_coefficient_upper = Word{0};
                Word lane_coefficient_upper_shoup = Word{0};
                Word packed_coefficients[4] = {};
                Word packed_coefficients_shoup[4] = {};
                if constexpr (PackedStage6 || DistributedPackedStage6) {
                    if (stage == 6U) {
                        constexpr std::uint32_t kPackedStage6Base =
                            kLocalN * (kLocalN - 1U);
                        const std::uint64_t packed_row_offset = ConsumerRole
                            ? 64U + static_cast<std::uint64_t>(row) * 64U : 0U;
                        const Word* packed_stage6 = root_powers +
                            kPackedStage6Base + packed_row_offset;
                        const Word* packed_stage6_shoup = root_powers_shoup +
                            kPackedStage6Base + packed_row_offset;
                        if constexpr (PackedStage6) {
                            if (lane < 16U) {
                                const uint4 values4 =
                                    *reinterpret_cast<const uint4*>(
                                        packed_stage6 + (lane << 2U));
                                const uint4 shoup4 =
                                    *reinterpret_cast<const uint4*>(
                                        packed_stage6_shoup + (lane << 2U));
                                packed_coefficients[0] = values4.x;
                                packed_coefficients[1] = values4.y;
                                packed_coefficients[2] = values4.z;
                                packed_coefficients[3] = values4.w;
                                packed_coefficients_shoup[0] = shoup4.x;
                                packed_coefficients_shoup[1] = shoup4.y;
                                packed_coefficients_shoup[2] = shoup4.z;
                                packed_coefficients_shoup[3] = shoup4.w;
                            }
                        } else {
                            const uint2 values2 =
                                *reinterpret_cast<const uint2*>(
                                    packed_stage6 + (lane << 1U));
                            const uint2 shoup2 =
                                *reinterpret_cast<const uint2*>(
                                    packed_stage6_shoup + (lane << 1U));
                            lane_coefficient = values2.x;
                            lane_coefficient_upper = values2.y;
                            lane_coefficient_shoup = shoup2.x;
                            lane_coefficient_upper_shoup = shoup2.y;
                        }
                    }
                }
                if constexpr (CoefficientReuseStages != 0) {
                    if (stage < CoefficientReuseStages && lane < half) {
                        const std::uint32_t distributed_offset =
                            stage == 6U ? (lane << 1U) : lane;
                        lane_coefficient = coefficients[
                            coefficient_base + distributed_offset];
                        lane_coefficient_shoup = coefficients_shoup[
                            coefficient_base + distributed_offset];
                    }
                    if constexpr (CoefficientReuseStages > 6U) {
                        if (stage == 6U) {
                            lane_coefficient_upper =
                                coefficients[coefficient_base +
                                             (lane << 1U) + 1U];
                            lane_coefficient_upper_shoup = coefficients_shoup[
                                coefficient_base + (lane << 1U) + 1U];
                        }
                    }
                }
#pragma unroll
                for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
                    Word coefficient = Word{0};
                    Word coefficient_shoup = Word{0};
                    if constexpr (CoefficientReuseStages != 0) {
                        if (stage < CoefficientReuseStages) {
                            const std::uint32_t offset =
                                ((lane << 2) + slot) & (half - 1U);
                            int source_lane =
                                static_cast<int>(offset & 31U);
                            Word distributed_coefficient = lane_coefficient;
                            Word distributed_coefficient_shoup =
                                lane_coefficient_shoup;
                            if constexpr (CoefficientReuseStages > 6U) {
                                if (stage == 6U) {
                                    source_lane =
                                        static_cast<int>((offset >> 1U) & 31U);
                                    if ((offset & 1U) != 0U) {
                                        distributed_coefficient =
                                            lane_coefficient_upper;
                                        distributed_coefficient_shoup =
                                            lane_coefficient_upper_shoup;
                                    }
                                }
                            }
                            coefficient = __shfl_sync(
                                0xffffffffU, distributed_coefficient,
                                source_lane);
                            coefficient_shoup = __shfl_sync(
                                0xffffffffU, distributed_coefficient_shoup,
                                source_lane);
                        }
                    }
                    if constexpr (DistributedPackedStage6) {
                        if (stage == 6U) {
                            const std::uint32_t offset =
                                ((lane << 2) + slot) & (half - 1U);
                            const int source_lane =
                                static_cast<int>((offset >> 1U) & 31U);
                            const Word distributed_coefficient =
                                (offset & 1U) == 0U ? lane_coefficient
                                                   : lane_coefficient_upper;
                            const Word distributed_coefficient_shoup =
                                (offset & 1U) == 0U
                                    ? lane_coefficient_shoup
                                    : lane_coefficient_upper_shoup;
                            coefficient = __shfl_sync(
                                0xffffffffU, distributed_coefficient,
                                source_lane);
                            coefficient_shoup = __shfl_sync(
                                0xffffffffU, distributed_coefficient_shoup,
                                source_lane);
                        }
                    }
                    Word left = values[slot];
                    Word right = detail::hybrid_shuffle_xor(values[slot],
                                                            lane_delta);
                    Word right_result = Word{0};
                    if ((lane & lane_delta) == 0) {
                        if constexpr (PackedStage6 || DistributedPackedStage6) {
                            if (stage == 6U) {
                                if constexpr (PackedStage6) {
                                    coefficient = packed_coefficients[slot];
                                    coefficient_shoup =
                                        packed_coefficients_shoup[slot];
                                }
                            } else if (stage >= CoefficientReuseStages) {
                                const std::uint32_t logical =
                                    (lane << 2) + slot;
                                const std::uint32_t offset =
                                    logical & (half - 1U);
                                coefficient =
                                    coefficients[coefficient_base + offset];
                                coefficient_shoup = coefficients_shoup[
                                    coefficient_base + offset];
                            }
                        } else if constexpr (CoefficientReuseStages == 0) {
                            const std::uint32_t logical =
                                (lane << 2) + slot;
                            const std::uint32_t offset =
                                logical & (half - 1U);
                            coefficient =
                                coefficients[coefficient_base + offset];
                            coefficient_shoup = coefficients_shoup[
                                coefficient_base + offset];
                        } else if (stage >= CoefficientReuseStages) {
                            const std::uint32_t logical =
                                (lane << 2) + slot;
                            const std::uint32_t offset =
                                logical & (half - 1U);
                            coefficient =
                                coefficients[coefficient_base + offset];
                            coefficient_shoup = coefficients_shoup[
                                coefficient_base + offset];
                        }
                        op.apply_coefficient(coefficient, coefficient_shoup,
                                             left, right);
                        right_result = right;
                    }
                    const Word received_right =
                        detail::hybrid_shuffle_xor(right_result, lane_delta);
                    values[slot] = (lane & lane_delta) == 0 ? left
                                                            : received_right;
                }
            }
        } else {
#pragma unroll
            for (std::uint32_t stage = 0; stage < WarpTileLog; ++stage) {
                if constexpr (CoefficientReuseStages != 0) {
                    if (stage < CoefficientReuseStages) {
                        appt_apply_reused_register_stage<Word, kValuesPerLane>(
                            values, kWarpTileN, stage, lane, coefficients,
                            coefficients_shoup, op);
                    } else {
                        appt_apply_cached_register_stage<Word, kValuesPerLane>(
                            values, kWarpTileN, stage, lane, coefficients,
                            coefficients_shoup, op);
                    }
                } else {
                    appt_apply_cached_register_stage<Word, kValuesPerLane>(
                        values, kWarpTileN, stage, lane, coefficients,
                        coefficients_shoup, op);
                }
            }
        }
#pragma unroll
        for (std::uint32_t slot = 0; slot < kValuesPerLane; ++slot) {
            const std::uint32_t local =
                tile_base + (VectorRadix4Prefix ? (lane << 2) + slot
                                                : (slot << 5) + lane);
            exchange[local] = values[slot];
        }
    }
    hierarchical_warp_group_barrier(group + 1U);

    if constexpr (WarpTileLog == 7) {
        const std::uint32_t offset = group_thread;
        Word values[8];
#pragma unroll
        for (std::uint32_t tile = 0; tile < 8U; ++tile) {
            values[tile] = exchange[tile * 128U + offset];
        }
#pragma unroll
        for (std::uint32_t tile = 0; tile < 8U; tile += 2U) {
            op.apply_coefficient(coefficients[127U + offset],
                                 coefficients_shoup[127U + offset],
                                 values[tile], values[tile + 1U]);
        }
#pragma unroll
        for (std::uint32_t tile = 0; tile < 8U; tile += 4U) {
            op.apply_coefficient(coefficients[255U + offset],
                                 coefficients_shoup[255U + offset],
                                 values[tile], values[tile + 2U]);
            op.apply_coefficient(coefficients[383U + offset],
                                 coefficients_shoup[383U + offset],
                                 values[tile + 1U], values[tile + 3U]);
        }
#pragma unroll
        for (std::uint32_t tile = 0; tile < 4U; ++tile) {
            op.apply_coefficient(coefficients[511U + tile * 128U + offset],
                                 coefficients_shoup[511U + tile * 128U +
                                                    offset],
                                 values[tile], values[tile + 4U]);
        }
#pragma unroll
        for (std::uint32_t tile = 0; tile < 8U; ++tile) {
            exchange[tile * 128U + offset] = values[tile];
        }
        hierarchical_warp_group_barrier(group + 1U);
    } else {
#pragma unroll
        for (std::uint32_t stage = WarpTileLog; stage < 10U; ++stage) {
            const std::uint32_t half = 1U << stage;
            const std::uint32_t coefficient_base = half - 1U;
            for (std::uint32_t work = group_thread; work < kLocalN / 2U;
                 work += 128U) {
                const std::uint32_t butterfly_group = work / half;
                const std::uint32_t offset = work & (half - 1U);
                const std::uint32_t left_index =
                    butterfly_group * (half << 1U) + offset;
                Word left = exchange[left_index];
                Word right = exchange[left_index + half];
                op.apply_coefficient(
                    coefficients[coefficient_base + offset],
                    coefficients_shoup[coefficient_base + offset], left,
                    right);
                exchange[left_index] = left;
                exchange[left_index + half] = right;
            }
            hierarchical_warp_group_barrier(group + 1U);
        }
    }

    for (std::uint32_t local = group_thread; local < kLocalN;
         local += 128U) {
        Word result = exchange[local];
        if constexpr (ConsumerRole) {
            result = op.finalize(result, kLocalN * kLocalN);
        }
        if constexpr (kRowMajorIo) {
            writer_tile[
                hierarchical_warp256_io_index<WriterRows>(writer_row, local)] =
                result;
        } else {
            const std::uint32_t major =
                hierarchical_warp256_io_swizzle(local);
            const std::uint32_t physical_row =
                writer_row ^ (major & (WriterRows - 1U));
            writer_tile[major * WriterRows + physical_row] = result;
        }
    }
    hierarchical_warp_group_barrier(group + 1U);
}

__device__ __forceinline__ void hierarchical_warp_prefix_barrier(
    std::uint32_t barrier) {
    asm volatile("bar.sync %0, 96;" : : "r"(barrier) : "memory");
}

__device__ __forceinline__ void hierarchical_warp_pair_barrier(
    std::uint32_t barrier) {
    asm volatile("bar.sync %0, 64;" : : "r"(barrier) : "memory");
}

// Cooperative 4 -> 2x2 -> 4 continuation. Each warp owns two adjacent
// 128-point prefixes and their stage-7 join. Two warp pairs independently
// finish stage 8 before the complete group executes stage 9. There are no
// polling tokens or permanently idle prefix/merge roles.
template <typename Word, bool ConsumerRole, std::uint32_t WriterRows>
__device__ __forceinline__ void
hierarchical_warp128_cooperative_static_io_row(
    Word* exchange, std::uint32_t row, std::uint32_t quarter,
    Word* writer_tile, std::uint32_t writer_row,
    const Word* root_powers, const Word* root_powers_shoup,
    NttStageOperatorT<Word> op) {
    static_assert(sizeof(Word) == 4 && WriterRows == 4);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kTileN = 128;
    constexpr std::uint32_t kValuesPerLane = 4;
    const std::uint32_t lane = threadIdx.x & 31U;
    const Word* coefficients = ConsumerRole
        ? root_powers + static_cast<std::uint64_t>(row) * (kLocalN - 1U)
        : op.twiddles;
    const Word* coefficients_shoup = ConsumerRole
        ? root_powers_shoup +
              static_cast<std::uint64_t>(row) * (kLocalN - 1U)
        : op.twiddles_shoup;

#pragma unroll
    for (std::uint32_t pair_tile = 0; pair_tile < 2U; ++pair_tile) {
        const std::uint32_t tile = quarter * 2U + pair_tile;
        Word values[kValuesPerLane];
#pragma unroll
        for (std::uint32_t value = 0; value < kValuesPerLane; ++value) {
            const std::uint32_t local =
                tile * kTileN + (value << 5) + lane;
            values[value] = writer_tile[
                hierarchical_warp256_io_index<WriterRows>(writer_row, local)];
        }
#pragma unroll
        for (std::uint32_t stage = 0; stage < 7U; ++stage) {
            appt_apply_cached_register_stage<Word, kValuesPerLane>(
                values, kTileN, stage, lane, coefficients,
                coefficients_shoup, op);
        }
#pragma unroll
        for (std::uint32_t value = 0; value < kValuesPerLane; ++value) {
            exchange[tile * kTileN + (value << 5) + lane] = values[value];
        }
    }
    __syncwarp();

#pragma unroll
    for (std::uint32_t wave = 0; wave < 4U; ++wave) {
        const std::uint32_t offset = (wave << 5) + lane;
        const std::uint32_t first = quarter * 2U * kTileN + offset;
        Word left = exchange[first];
        Word right = exchange[first + kTileN];
        op.apply_coefficient(coefficients[127U + offset],
                             coefficients_shoup[127U + offset], left, right);
        exchange[first] = left;
        exchange[first + kTileN] = right;
    }

    const std::uint32_t half_group = quarter >> 1U;
    hierarchical_warp_pair_barrier(6U + half_group);
#pragma unroll
    for (std::uint32_t wave = 0; wave < 4U; ++wave) {
        const std::uint32_t offset =
            ((quarter & 1U) * kTileN) + (wave << 5) + lane;
        const std::uint32_t base = half_group * 4U * kTileN;
        Word left = exchange[base + offset];
        Word right = exchange[base + 2U * kTileN + offset];
        op.apply_coefficient(coefficients[255U + offset],
                             coefficients_shoup[255U + offset], left, right);
        exchange[base + offset] = left;
        exchange[base + 2U * kTileN + offset] = right;
    }

    hierarchical_warp_group_barrier(1U);
#pragma unroll
    for (std::uint32_t wave = 0; wave < 4U; ++wave) {
        const std::uint32_t offset = quarter * kTileN + (wave << 5) + lane;
        Word left = exchange[offset];
        Word right = exchange[512U + offset];
        op.apply_coefficient(coefficients[511U + offset],
                             coefficients_shoup[511U + offset], left, right);
        if constexpr (ConsumerRole) {
            left = op.finalize(left, kLocalN * kLocalN);
            right = op.finalize(right, kLocalN * kLocalN);
        }
        writer_tile[
            hierarchical_warp256_io_index<WriterRows>(writer_row, offset)] =
            left;
        writer_tile[hierarchical_warp256_io_index<WriterRows>(
            writer_row, 512U + offset)] = right;
    }
    hierarchical_warp_group_barrier(1U);
}

// Three prefix warps compute dependency-closed 128-point tiles while one merge
// warp finishes the previous row. Alternating shared slots make the overlap
// explicit instead of relying on warp scheduling across a row-wide barrier.
template <typename Word, bool ConsumerRole, std::uint32_t WriterRows,
          std::uint32_t WarpTileLog>
__device__ __forceinline__ void
hierarchical_warp_small_static_io_packet_pipeline(
    Word* exchange, Word* writer_tile, std::uint32_t row_base,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned int* flags, NttStageOperatorT<Word> op) {
    static_assert(WarpTileLog == 6 || WarpTileLog == 7);
    static_assert(sizeof(Word) == 4 && WriterRows == 4);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kBuffers = 2;
    constexpr std::uint32_t kWarpTileN = 1U << WarpTileLog;
    constexpr std::uint32_t kValuesPerLane = kWarpTileN / 32U;
    constexpr std::uint32_t kTiles = kLocalN / kWarpTileN;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;

    if (threadIdx.x < kBuffers) {
        flags[threadIdx.x] = 0U;
        flags[kBuffers + threadIdx.x] = 1U;
    }
    hierarchical_warp_group_barrier(1U);

    if (warp < 3U) {
        const std::uint32_t prefix_warp = warp;
#pragma unroll
        for (std::uint32_t writer_row = 0; writer_row < WriterRows;
             ++writer_row) {
            const std::uint32_t slot = writer_row & (kBuffers - 1U);
            if (prefix_warp == 0U && lane == 0U) {
                while (atomicAdd(flags + kBuffers + slot, 0U) == 0U) {
                    __nanosleep(32);
                }
                atomicExch(flags + kBuffers + slot, 0U);
                atomicExch(flags + slot, 0U);
            }
            hierarchical_warp_prefix_barrier(5U);

            const std::uint32_t row = row_base + writer_row;
            const Word* coefficients = ConsumerRole
                ? root_powers + static_cast<std::uint64_t>(row) *
                                    (kLocalN - 1U)
                : op.twiddles;
            const Word* coefficients_shoup = ConsumerRole
                ? root_powers_shoup + static_cast<std::uint64_t>(row) *
                                          (kLocalN - 1U)
                : op.twiddles_shoup;
            Word* slot_exchange = exchange + slot * kLocalN;
#pragma unroll
            for (std::uint32_t tile = prefix_warp; tile < kTiles;
                 tile += 3U) {
                Word values[kValuesPerLane];
#pragma unroll
                for (std::uint32_t value = 0; value < kValuesPerLane;
                     ++value) {
                    const std::uint32_t local =
                        tile * kWarpTileN + (value << 5) + lane;
                    values[value] = writer_tile[
                        hierarchical_warp256_io_index<WriterRows>(
                            writer_row, local)];
                }
#pragma unroll
                for (std::uint32_t stage = 0; stage < WarpTileLog; ++stage) {
                    appt_apply_cached_register_stage<Word, kValuesPerLane>(
                        values, kWarpTileN, stage, lane, coefficients,
                        coefficients_shoup, op);
                }
#pragma unroll
                for (std::uint32_t value = 0; value < kValuesPerLane;
                     ++value) {
                    slot_exchange[tile * kWarpTileN + (value << 5) + lane] =
                        values[value];
                }
            }
            hierarchical_warp_prefix_barrier(5U);
            if (lane == 0U) atomicAdd(flags + slot, 1U);
        }
    } else {
#pragma unroll
        for (std::uint32_t writer_row = 0; writer_row < WriterRows;
             ++writer_row) {
            const std::uint32_t slot = writer_row & (kBuffers - 1U);
            if (lane == 0U) {
                while (atomicAdd(flags + slot, 0U) < 3U) {
                    __nanosleep(32);
                }
            }
            __syncwarp();

            const std::uint32_t row = row_base + writer_row;
            const Word* coefficients = ConsumerRole
                ? root_powers + static_cast<std::uint64_t>(row) *
                                    (kLocalN - 1U)
                : op.twiddles;
            const Word* coefficients_shoup = ConsumerRole
                ? root_powers_shoup + static_cast<std::uint64_t>(row) *
                                          (kLocalN - 1U)
                : op.twiddles_shoup;
            Word* slot_exchange = exchange + slot * kLocalN;
            if constexpr (WarpTileLog == 7) {
#pragma unroll
                for (std::uint32_t wave = 0; wave < 4U; ++wave) {
                    const std::uint32_t offset = (wave << 5) + lane;
                    Word values[8];
#pragma unroll
                    for (std::uint32_t tile = 0; tile < 8U; ++tile) {
                        values[tile] =
                            slot_exchange[tile * 128U + offset];
                    }
#pragma unroll
                    for (std::uint32_t tile = 0; tile < 8U; tile += 2U) {
                        op.apply_coefficient(
                            coefficients[127U + offset],
                            coefficients_shoup[127U + offset], values[tile],
                            values[tile + 1U]);
                    }
#pragma unroll
                    for (std::uint32_t tile = 0; tile < 8U; tile += 4U) {
                        op.apply_coefficient(
                            coefficients[255U + offset],
                            coefficients_shoup[255U + offset], values[tile],
                            values[tile + 2U]);
                        op.apply_coefficient(
                            coefficients[383U + offset],
                            coefficients_shoup[383U + offset],
                            values[tile + 1U], values[tile + 3U]);
                    }
#pragma unroll
                    for (std::uint32_t tile = 0; tile < 4U; ++tile) {
                        op.apply_coefficient(
                            coefficients[511U + tile * 128U + offset],
                            coefficients_shoup[511U + tile * 128U + offset],
                            values[tile], values[tile + 4U]);
                    }
#pragma unroll
                    for (std::uint32_t tile = 0; tile < 8U; ++tile) {
                        Word result = values[tile];
                        if constexpr (ConsumerRole) {
                            result = op.finalize(result, kLocalN * kLocalN);
                        }
                        const std::uint32_t local = tile * 128U + offset;
                        writer_tile[
                            hierarchical_warp256_io_index<WriterRows>(
                                writer_row, local)] = result;
                    }
                }
            } else {
#pragma unroll
                for (std::uint32_t stage = WarpTileLog; stage < 10U;
                     ++stage) {
                    const std::uint32_t half = 1U << stage;
                    const std::uint32_t coefficient_base = half - 1U;
                    for (std::uint32_t work = lane;
                         work < kLocalN / 2U; work += 32U) {
                        const std::uint32_t butterfly_group = work / half;
                        const std::uint32_t offset = work & (half - 1U);
                        const std::uint32_t left_index =
                            butterfly_group * (half << 1U) + offset;
                        Word left = slot_exchange[left_index];
                        Word right = slot_exchange[left_index + half];
                        op.apply_coefficient(
                            coefficients[coefficient_base + offset],
                            coefficients_shoup[coefficient_base + offset],
                            left, right);
                        slot_exchange[left_index] = left;
                        slot_exchange[left_index + half] = right;
                    }
                    __syncwarp();
                }
                for (std::uint32_t local = lane; local < kLocalN;
                     local += 32U) {
                    Word result = slot_exchange[local];
                    if constexpr (ConsumerRole) {
                        result = op.finalize(result, kLocalN * kLocalN);
                    }
                    writer_tile[
                        hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                   local)] =
                        result;
                }
            }
            __syncwarp();
            if (lane == 0U) {
                atomicExch(flags + kBuffers + slot, 1U);
            }
        }
    }
    hierarchical_warp_group_barrier(1U);
}

// Tile-online form of the prefix/merge pipeline. Prefix warps publish
// independent 128-point tiles. The merge warp consumes pairs immediately for
// stage 7, groups of four for stage 8, and only waits for the full row at
// stage 9. One shared row is sufficient because all warps rendezvous only
// after the completed row has been written to the static packet.
template <typename Word, bool ConsumerRole, std::uint32_t WriterRows>
__device__ __forceinline__ void
hierarchical_warp128_static_io_packet_online(
    Word* exchange, Word* writer_tile, std::uint32_t row_base,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned int* tile_ready, NttStageOperatorT<Word> op) {
    static_assert(sizeof(Word) == 4 && WriterRows == 4);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kTileN = 128;
    constexpr std::uint32_t kTiles = 8;
    constexpr std::uint32_t kValuesPerLane = 4;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;

#pragma unroll
    for (std::uint32_t writer_row = 0; writer_row < WriterRows;
         ++writer_row) {
        if (threadIdx.x < kTiles) tile_ready[threadIdx.x] = 0U;
        hierarchical_warp_group_barrier(1U);

        const std::uint32_t row = row_base + writer_row;
        const Word* coefficients = ConsumerRole
            ? root_powers +
                  static_cast<std::uint64_t>(row) * (kLocalN - 1U)
            : op.twiddles;
        const Word* coefficients_shoup = ConsumerRole
            ? root_powers_shoup +
                  static_cast<std::uint64_t>(row) * (kLocalN - 1U)
            : op.twiddles_shoup;

        if (warp < 3U) {
#pragma unroll
            for (std::uint32_t tile = warp; tile < kTiles; tile += 3U) {
                Word values[kValuesPerLane];
#pragma unroll
                for (std::uint32_t value = 0; value < kValuesPerLane;
                     ++value) {
                    const std::uint32_t local =
                        tile * kTileN + (value << 5) + lane;
                    values[value] = writer_tile[
                        hierarchical_warp256_io_index<WriterRows>(
                            writer_row, local)];
                }
#pragma unroll
                for (std::uint32_t stage = 0; stage < 7U; ++stage) {
                    appt_apply_cached_register_stage<Word, kValuesPerLane>(
                        values, kTileN, stage, lane, coefficients,
                        coefficients_shoup, op);
                }
#pragma unroll
                for (std::uint32_t value = 0; value < kValuesPerLane;
                     ++value) {
                    exchange[tile * kTileN + (value << 5) + lane] =
                        values[value];
                }
                __syncwarp();
                if (lane == 0U) atomicExch(tile_ready + tile, 1U);
            }
        } else {
#pragma unroll
            for (std::uint32_t group = 0; group < 2U; ++group) {
#pragma unroll
                for (std::uint32_t pair = 0; pair < 2U; ++pair) {
                    const std::uint32_t first_tile = group * 4U + pair * 2U;
                    if (lane == 0U) {
                        while (atomicAdd(tile_ready + first_tile, 0U) == 0U ||
                               atomicAdd(tile_ready + first_tile + 1U, 0U) ==
                                   0U) {
                            __nanosleep(32);
                        }
                    }
                    __syncwarp();
#pragma unroll
                    for (std::uint32_t wave = 0; wave < 4U; ++wave) {
                        const std::uint32_t offset = (wave << 5) + lane;
                        Word left =
                            exchange[first_tile * kTileN + offset];
                        Word right =
                            exchange[(first_tile + 1U) * kTileN + offset];
                        op.apply_coefficient(
                            coefficients[127U + offset],
                            coefficients_shoup[127U + offset], left, right);
                        exchange[first_tile * kTileN + offset] = left;
                        exchange[(first_tile + 1U) * kTileN + offset] = right;
                    }
                }
                __syncwarp();
#pragma unroll
                for (std::uint32_t wave = 0; wave < 8U; ++wave) {
                    const std::uint32_t offset = (wave << 5) + lane;
                    const std::uint32_t base = group * 4U * kTileN;
                    Word left = exchange[base + offset];
                    Word right = exchange[base + 2U * kTileN + offset];
                    op.apply_coefficient(
                        coefficients[255U + offset],
                        coefficients_shoup[255U + offset], left, right);
                    exchange[base + offset] = left;
                    exchange[base + 2U * kTileN + offset] = right;
                }
            }
            __syncwarp();
#pragma unroll
            for (std::uint32_t wave = 0; wave < 16U; ++wave) {
                const std::uint32_t offset = (wave << 5) + lane;
                Word left = exchange[offset];
                Word right = exchange[512U + offset];
                op.apply_coefficient(coefficients[511U + offset],
                                     coefficients_shoup[511U + offset], left,
                                     right);
                if constexpr (ConsumerRole) {
                    left = op.finalize(left, kLocalN * kLocalN);
                    right = op.finalize(right, kLocalN * kLocalN);
                }
                writer_tile[
                    hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                               offset)] = left;
                writer_tile[hierarchical_warp256_io_index<WriterRows>(
                    writer_row, 512U + offset)] = right;
            }
        }
        hierarchical_warp_group_barrier(1U);
    }
}

template <typename Word, bool ConsumerRole, std::uint32_t WriterRows>
__device__ __forceinline__ void
hierarchical_warp128_packet_shared_radix4(
    Word* state, std::uint32_t row_base, const Word* root_powers,
    const Word* root_powers_shoup, NttStageOperatorT<Word> op) {
    static_assert(sizeof(Word) == 4 && WriterRows == 4);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kUnitsPerRow = kLocalN / 4;
    constexpr std::uint32_t kRows = WriterRows;
    const std::uint32_t group_thread = threadIdx.x & 127U;
    const std::uint32_t group = (threadIdx.x >> 5) >> 2;

#pragma unroll
    for (std::uint32_t stage = 0; stage < 10; stage += 2) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = group_thread;
             linear < kRows * kUnitsPerRow; linear += 128U) {
            const std::uint32_t writer_row = linear / kUnitsPerRow;
            const std::uint32_t unit = linear & (kUnitsPerRow - 1U);
            const std::uint32_t group_index = unit / half;
            const std::uint32_t offset = unit - group_index * half;
            const std::uint32_t i0 = group_index * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            Word a = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i0)];
            Word b = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i1)];
            Word c = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i2)];
            Word d = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i3)];
            const Word* coefficients = ConsumerRole
                ? root_powers + static_cast<std::uint64_t>(row_base + writer_row) *
                                    (kLocalN - 1U)
                : op.twiddles;
            const Word* coefficients_shoup = ConsumerRole
                ? root_powers_shoup +
                      static_cast<std::uint64_t>(row_base + writer_row) *
                          (kLocalN - 1U)
                : op.twiddles_shoup;
            const std::uint32_t first_base = half - 1U;
            const std::uint32_t second_base = (half << 1) - 1U;
            op.apply_coefficient(coefficients[first_base + offset],
                                 coefficients_shoup[first_base + offset], a, b);
            op.apply_coefficient(coefficients[first_base + offset],
                                 coefficients_shoup[first_base + offset], c, d);
            op.apply_coefficient(coefficients[second_base + offset],
                                 coefficients_shoup[second_base + offset], a, c);
            op.apply_coefficient(coefficients[second_base + half + offset],
                                 coefficients_shoup[second_base + half + offset],
                                 b, d);
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i0)] = a;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i1)] = b;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i2)] = c;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i3)] = d;
        }
        hierarchical_warp_group_barrier(group + 1U);
    }
    if constexpr (ConsumerRole) {
        for (std::uint32_t linear = group_thread;
             linear < kRows * kLocalN; linear += 128U) {
            const std::uint32_t writer_row = linear >> 10;
            const std::uint32_t local = linear & (kLocalN - 1U);
            const std::uint32_t index =
                hierarchical_warp256_io_index<WriterRows>(writer_row, local);
            state[index] = op.finalize(state[index], kLocalN * kLocalN);
        }
        hierarchical_warp_group_barrier(group + 1U);
    }
}

template <typename Word, std::uint32_t WriterRows>
__device__ __forceinline__ void
hierarchical_packet_streaming_radix4_unit(
    Word* state, const Word* staging, std::uint32_t row_base,
    std::uint32_t q_begin, std::uint32_t q_count,
    std::uint32_t writer_row, std::uint32_t q_offset,
    std::uint32_t residue, const Word* root_powers,
    const Word* root_powers_shoup, NttStageOperatorT<Word> op) {
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kQuadrants = 4;
    const std::uint32_t reversed =
        __brev(((q_begin + q_offset) << 2) + residue) >> 24;
    const std::uint32_t i0 = reversed << 2;
    const std::uint32_t i1 = i0 + 1U;
    const std::uint32_t i2 = i0 + 2U;
    const std::uint32_t i3 = i0 + 3U;
    const std::uint32_t words_per_row =
        kQuadrants * q_count * WriterRows;
    const std::uint32_t row_offset = writer_row * words_per_row;
    const std::uint32_t q_word = q_offset * WriterRows + residue;
    Word a = staging[row_offset + q_word];
    Word b = staging[row_offset + 2U * q_count * WriterRows + q_word];
    Word c = staging[row_offset + q_count * WriterRows + q_word];
    Word d = staging[row_offset + 3U * q_count * WriterRows + q_word];
    const Word* coefficients =
        root_powers + static_cast<std::uint64_t>(row_base + writer_row) *
                          (kLocalN - 1U);
    const Word* coefficients_shoup =
        root_powers_shoup +
        static_cast<std::uint64_t>(row_base + writer_row) *
            (kLocalN - 1U);
    op.apply_coefficient(coefficients[0], coefficients_shoup[0], a, b);
    op.apply_coefficient(coefficients[0], coefficients_shoup[0], c, d);
    op.apply_coefficient(coefficients[1], coefficients_shoup[1], a, c);
    op.apply_coefficient(coefficients[2], coefficients_shoup[2], b, d);
    state[hierarchical_warp256_io_index<WriterRows>(writer_row, i0)] = a;
    state[hierarchical_warp256_io_index<WriterRows>(writer_row, i1)] = b;
    state[hierarchical_warp256_io_index<WriterRows>(writer_row, i2)] = c;
    state[hierarchical_warp256_io_index<WriterRows>(writer_row, i3)] = d;
}

template <typename Word, std::uint32_t WriterRows>
__device__ __forceinline__ void
hierarchical_warp128_packet_streaming_radix4_consumer(
    Word* state, Word* staging, const Word* boundary, unsigned int* ready,
    std::uint64_t transform, std::uint32_t row_base,
    std::uint32_t packet_wave_q, std::uint32_t packet_poll_sleep,
    std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers,
    const Word* root_powers,
    const Word* root_powers_shoup, NttStageOperatorT<Word> op) {
    static_assert(sizeof(Word) == 4 && WriterRows == 4);
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kN = kLocalN * kLocalN;
    constexpr std::uint32_t kRowPackets = kLocalN / WriterRows;
    constexpr std::uint32_t kQuadrants = 4;
    constexpr std::uint32_t kQCount = kRowPackets / kQuadrants;
    constexpr std::uint32_t kStagingWords = kLocalN;
    constexpr std::uint32_t kWordsPerQ = kQuadrants * WriterRows * 4U;
    constexpr std::uint32_t kMaxPacketWaveQ = kStagingWords / kWordsPerQ;
    const std::uint32_t group_thread = threadIdx.x & 127U;
    const std::uint32_t group = (threadIdx.x >> 5) >> 2;
    const std::uint32_t wave_q = min(packet_wave_q, kMaxPacketWaveQ);

    // One q-group contains the four producer packets needed by four
    // dependency-closed radix-4 units after the boundary bit reversal.
    for (std::uint32_t q_begin = 0; q_begin < kQCount;
         q_begin += wave_q) {
        const std::uint32_t q_count = min(wave_q, kQCount - q_begin);
        if (packet_readiness_mode != 0U) {
            for (std::uint32_t quadrant = group_thread;
                 quadrant < kQuadrants; quadrant += 128U) {
                const std::uint32_t packet =
                    q_begin + quadrant * kQCount;
                const std::uint32_t word = packet >> 5;
                const std::uint32_t shift = packet & 31U;
                const std::uint32_t mask =
                    ((1U << q_count) - 1U) << shift;
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    readiness(ready[transform * 8U + word]);
                while ((readiness.load(cuda::memory_order_acquire) & mask) !=
                       mask) {
                    __nanosleep(packet_poll_sleep);
                }
            }
        } else {
            for (std::uint32_t check = group_thread;
                 check < q_count * kQuadrants;
                 check += 128U) {
                const std::uint32_t q = q_begin + check / kQuadrants;
                const std::uint32_t quadrant = check & (kQuadrants - 1U);
                const std::uint32_t packet = q + quadrant * kQCount;
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    readiness(ready[transform * kRowPackets + packet]);
                while (readiness.load(cuda::memory_order_acquire) == 0U) {
                    __nanosleep(packet_poll_sleep);
                }
            }
        }
        hierarchical_warp_group_barrier(group + 1U);

        const std::uint32_t wave_words = q_count * WriterRows * 16U;
        for (std::uint32_t linear = group_thread; linear < wave_words;
             linear += 128U) {
            const std::uint32_t words_per_row =
                kQuadrants * q_count * WriterRows;
            const std::uint32_t writer_row = linear / words_per_row;
            const std::uint32_t within_row = linear % words_per_row;
            const std::uint32_t quadrant =
                within_row / (q_count * WriterRows);
            const std::uint32_t within_quadrant =
                within_row % (q_count * WriterRows);
            const std::uint32_t q_offset =
                within_quadrant / WriterRows;
            const std::uint32_t packet_element =
                within_quadrant & (WriterRows - 1U);
            const std::uint32_t packet =
                q_begin + q_offset + quadrant * kQCount;
            const std::uint32_t second_index =
                packet * WriterRows + packet_element;
            staging[linear] = boundary[
                transform * kN +
                static_cast<std::uint64_t>(row_base + writer_row) * kLocalN +
                second_index];
        }
        hierarchical_warp_group_barrier(group + 1U);

        if (packet_compute_layout != 0U) {
            const std::uint32_t writer_row = group_thread >> 5;
            const std::uint32_t lane = group_thread & 31U;
            for (std::uint32_t row_linear = lane;
                 row_linear < q_count * 4U; row_linear += 32U) {
                hierarchical_packet_streaming_radix4_unit<Word, WriterRows>(
                    state, staging, row_base, q_begin, q_count, writer_row,
                    row_linear >> 2, row_linear & 3U, root_powers,
                    root_powers_shoup, op);
            }
        } else {
            const std::uint32_t wave_units = q_count * WriterRows * 4U;
            for (std::uint32_t linear = group_thread; linear < wave_units;
                 linear += 128U) {
                const std::uint32_t q_offset =
                    linear / (WriterRows * 4U);
                const std::uint32_t within = linear % (WriterRows * 4U);
                hierarchical_packet_streaming_radix4_unit<Word, WriterRows>(
                    state, staging, row_base, q_begin, q_count, within >> 2,
                    q_offset, within & 3U, root_powers, root_powers_shoup, op);
            }
        }
        if (packet_fold_wave_barriers == 0U) {
            hierarchical_warp_group_barrier(group + 1U);
        }
    }
    if (packet_fold_wave_barriers != 0U) {
        hierarchical_warp_group_barrier(group + 1U);
    }

#pragma unroll
    for (std::uint32_t stage = 2; stage < 10; stage += 2) {
        const std::uint32_t half = 1U << stage;
        for (std::uint32_t linear = group_thread;
             linear < WriterRows * (kLocalN / 4); linear += 128U) {
            const std::uint32_t writer_row = linear / (kLocalN / 4);
            const std::uint32_t unit = linear & (kLocalN / 4 - 1U);
            const std::uint32_t group_index = unit / half;
            const std::uint32_t offset = unit - group_index * half;
            const std::uint32_t i0 = group_index * (half << 2) + offset;
            const std::uint32_t i1 = i0 + half;
            const std::uint32_t i2 = i1 + half;
            const std::uint32_t i3 = i2 + half;
            Word a = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i0)];
            Word b = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i1)];
            Word c = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i2)];
            Word d = state[hierarchical_warp256_io_index<WriterRows>(writer_row,
                                                                      i3)];
            const Word* coefficients =
                root_powers +
                static_cast<std::uint64_t>(row_base + writer_row) *
                    (kLocalN - 1U);
            const Word* coefficients_shoup =
                root_powers_shoup +
                static_cast<std::uint64_t>(row_base + writer_row) *
                    (kLocalN - 1U);
            const std::uint32_t first_base = half - 1U;
            const std::uint32_t second_base = (half << 1) - 1U;
            op.apply_coefficient(coefficients[first_base + offset],
                                 coefficients_shoup[first_base + offset], a, b);
            op.apply_coefficient(coefficients[first_base + offset],
                                 coefficients_shoup[first_base + offset], c, d);
            op.apply_coefficient(coefficients[second_base + offset],
                                 coefficients_shoup[second_base + offset], a, c);
            op.apply_coefficient(coefficients[second_base + half + offset],
                                 coefficients_shoup[second_base + half + offset],
                                 b, d);
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i0)] = a;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i1)] = b;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i2)] = c;
            state[hierarchical_warp256_io_index<WriterRows>(writer_row, i3)] = d;
        }
        hierarchical_warp_group_barrier(group + 1U);
    }
    for (std::uint32_t linear = group_thread;
         linear < WriterRows * kLocalN; linear += 128U) {
        const std::uint32_t writer_row = linear >> 10;
        const std::uint32_t local = linear & (kLocalN - 1U);
        const std::uint32_t index =
            hierarchical_warp256_io_index<WriterRows>(writer_row, local);
        state[index] = op.finalize(state[index], kN);
    }
    hierarchical_warp_group_barrier(group + 1U);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 8, bool WarpPipeline = false,
          bool WarpCooperative = false,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false,
          bool PacketSharedRadix4 = false>
__device__ __forceinline__ void
hierarchical_homogeneous_warp256_10x10_body(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op,
    unsigned char* storage) {
    namespace cg = cooperative_groups;
    constexpr std::uint32_t kLocalN = 1024;
    constexpr std::uint32_t kN = kLocalN * kLocalN;
    constexpr std::uint32_t kWarpGroups = WarpGroups;
    constexpr std::uint32_t kWriterRows = StaticWriter
        ? (WriterRows == 0 ? 32U / sizeof(Word) : WriterRows) : 1U;
    constexpr bool kRowMajorIo =
        StaticReader && sizeof(Word) == 4 && kWriterRows == 4;
    constexpr std::uint32_t kWriterStride = [] {
        if constexpr (kRowMajorIo) {
            return hierarchical_warp256_io_stride<kWriterRows>();
        }
        return kLocalN;
    }();
    constexpr std::uint32_t kWriterTileWords = kWriterRows * kWriterStride;
    static_assert(WarpGroups == 1 || WarpGroups == 2);
    static_assert(!StaticReader || StaticWriter);
    static_assert(WarpTileLog >= 6 && WarpTileLog <= 8);
    static_assert(WarpTileLog == 8 || (StaticWriter && StaticReader));
    static_assert(!WarpPipeline ||
                  (WarpGroups == 1 && StaticWriter && StaticReader &&
                   sizeof(Word) == 4 && kWriterRows == 4 && WarpTileLog < 8));
    static_assert(!WarpCooperative ||
                  (WarpGroups == 1 && StaticWriter && StaticReader &&
                   sizeof(Word) == 4 && kWriterRows == 4 && WarpTileLog == 7));
    static_assert(!(WarpPipeline && WarpCooperative));
    static_assert(!PacketSharedRadix4 ||
                  (StaticWriter && StaticReader && sizeof(Word) == 4 &&
                   kWriterRows == 4 && WarpTileLog == 7));
    constexpr std::uint32_t kExchangeBuffers = 1U;
    constexpr std::uint32_t kExchangeStride = kLocalN;
    Word* exchange = reinterpret_cast<Word*>(storage);
    Word* writer_tiles =
        exchange + kWarpGroups * kExchangeBuffers * kExchangeStride;
    auto* pipeline_flags = reinterpret_cast<unsigned int*>(
        writer_tiles + kWarpGroups * kWriterTileWords);

    const bool packet_streaming = PacketSharedRadix4 && packet_wave_q != 0U;
    const std::uint64_t readiness_tokens = packet_streaming
        ? transforms * (packet_readiness_mode != 0U
                            ? 8U
                            : (kLocalN / kWriterRows))
        : transforms;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < readiness_tokens;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 4; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    cg::this_grid().sync();

    const std::uint32_t consumer_blocks = gridDim.x - producer_blocks;
    const std::uint32_t paired_blocks = min(producer_blocks, consumer_blocks);
    bool consumer_role = false;
    std::uint32_t role_block = 0;
    if (blockIdx.x < 2U * paired_blocks) {
        consumer_role = (blockIdx.x & 1U) != 0;
        role_block = blockIdx.x >> 1;
    } else {
        consumer_role = consumer_blocks > producer_blocks;
        role_block = paired_blocks + blockIdx.x - 2U * paired_blocks;
    }
    const std::uint32_t role_blocks =
        consumer_role ? consumer_blocks : producer_blocks;
    const std::uint32_t warp = threadIdx.x >> 5;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp_group = warp >> 2;
    const std::uint32_t quarter = warp & 3U;
    const std::uint32_t data_time =
        consumer_role ? consumer_data_time : producer_data_time;
    const std::uint64_t transform_packets =
        (transforms + data_time - 1U) / data_time;
    const std::uint32_t row_packets = kLocalN / kWriterRows;
    const std::uint64_t tasks = transform_packets * row_packets;
    const std::uint64_t group_task =
        static_cast<std::uint64_t>(role_block) * kWarpGroups + warp_group;
    const std::uint64_t task_stride =
        static_cast<std::uint64_t>(role_blocks) * kWarpGroups;
    bool trace_started = false;

    for (std::uint64_t task = group_task; task < tasks;
         task += task_stride) {
        const std::uint32_t row_packet_ordinal =
            static_cast<std::uint32_t>(task % row_packets);
        const std::uint32_t row_packet =
            packet_streaming && !consumer_role
                ? (row_packet_ordinal >> 2) +
                      (row_packet_ordinal & 3U) * (row_packets >> 2)
                : row_packet_ordinal;
        const std::uint32_t row_base = row_packet * kWriterRows;
        const std::uint64_t transform_begin =
            (task / row_packets) * data_time;
        for (std::uint32_t time = 0; time < data_time; ++time) {
            const std::uint64_t transform = transform_begin + time;
            if (transform >= transforms) break;
            const std::uint32_t group_thread = quarter * 32U + lane;
            Word* writer_tile =
                writer_tiles + warp_group * kWriterTileWords;
            if (consumer_role) {
                if (!packet_streaming) {
                    unsigned int published = 0;
                    if (lane == 0 && quarter == 0) {
                        cuda::atomic_ref<unsigned int,
                                         cuda::thread_scope_device>
                            readiness(ready[transform]);
                        do {
                            published =
                                readiness.load(cuda::memory_order_acquire);
                            if (published < kLocalN) __nanosleep(64);
                        } while (published < kLocalN);
                    }
                    if constexpr (!PacketSharedRadix4) {
                        published = __shfl_sync(0xffffffffU, published, 0);
                    }
                    hierarchical_warp_group_barrier(warp_group + 1U);
                }
                if (lane == 0 && quarter == 0 && !trace_started) {
                    atomicMin(trace + 2, streaming_globaltimer());
                    trace_started = true;
                }
                if constexpr (StaticReader) {
                    if (!packet_streaming) {
                        for (std::uint32_t index = group_thread;
                             index < kLocalN * kWriterRows; index += 128U) {
                            const std::uint32_t writer_row = index / kLocalN;
                            const std::uint32_t second_index =
                                index & (kLocalN - 1U);
                            const std::uint32_t local =
                                __brev(second_index) >> 22;
                            const Word value = boundary[
                                transform * kN +
                                static_cast<std::uint64_t>(row_base +
                                                           writer_row) *
                                    kLocalN + second_index];
                            if constexpr (kRowMajorIo) {
                                writer_tile[
                                    hierarchical_warp256_io_index<kWriterRows>(
                                        writer_row, local)] = value;
                            } else {
                                const std::uint32_t major =
                                    hierarchical_warp256_io_swizzle(local);
                                const std::uint32_t physical_row =
                                    writer_row ^
                                    (major & (kWriterRows - 1U));
                                writer_tile[major * kWriterRows + physical_row] =
                                    value;
                            }
                        }
                        hierarchical_warp_group_barrier(warp_group + 1U);
                    }
                }
                if constexpr (PacketSharedRadix4) {
                    if (packet_streaming) {
                        hierarchical_warp128_packet_streaming_radix4_consumer<
                            Word, kWriterRows>(
                            writer_tile,
                            exchange + warp_group * kExchangeBuffers *
                                           kExchangeStride,
                            boundary, ready, transform, row_base, packet_wave_q,
                            packet_poll_sleep, packet_readiness_mode,
                            packet_compute_layout,
                            packet_fold_wave_barriers,
                            root_powers, root_powers_shoup, op);
                    } else {
                        hierarchical_warp128_packet_shared_radix4<
                            Word, true, kWriterRows>(
                            writer_tile, row_base, root_powers,
                            root_powers_shoup, op);
                    }
                } else if constexpr (WarpPipeline) {
                    hierarchical_warp128_static_io_packet_online<
                        Word, true, kWriterRows>(
                        exchange, writer_tile, row_base, root_powers,
                        root_powers_shoup, pipeline_flags, op);
                } else {
                    for (std::uint32_t writer_row = 0;
                         writer_row < kWriterRows; ++writer_row) {
                        const std::uint32_t row = row_base + writer_row;
                        if constexpr (WarpCooperative) {
                            hierarchical_warp128_cooperative_static_io_row<
                                Word, true, kWriterRows>(
                                exchange, row, quarter, writer_tile,
                                writer_row, root_powers, root_powers_shoup, op);
                        } else if constexpr (WarpTileLog == 8) {
                            hierarchical_warp256_row<Word, true, StaticWriter,
                                                     StaticReader, WriterRows,
                                                     CoefficientReuseStages>(
                                boundary, output,
                                exchange + warp_group * kExchangeBuffers *
                                               kExchangeStride,
                                transform * kN, row, quarter, writer_tile,
                                writer_row, root_powers, root_powers_shoup,
                                op);
                        } else {
                            hierarchical_warp_small_static_io_row<
                                Word, true, kWriterRows, WarpTileLog,
                                CoefficientReuseStages, VectorRadix4Prefix,
                                PackedStage6, DistributedPackedStage6>(
                                exchange + warp_group * kExchangeBuffers *
                                               kExchangeStride,
                                row, quarter, writer_tile, writer_row,
                                root_powers, root_powers_shoup, op);
                        }
                    }
                }
            } else {
                if (lane == 0 && quarter == 0 && !trace_started) {
                    atomicMin(trace, streaming_globaltimer());
                    trace_started = true;
                }
                if constexpr (StaticReader) {
                    for (std::uint32_t index = group_thread;
                         index < kLocalN * kWriterRows; index += 128U) {
                        const std::uint32_t first_index =
                            index / kWriterRows;
                        const std::uint32_t logical_row =
                            index % kWriterRows;
                        const std::uint32_t local =
                            __brev(first_index) >> 22;
                        const Word value = input[
                            transform * kN +
                            static_cast<std::uint64_t>(first_index) * kLocalN +
                            row_base + logical_row];
                        if constexpr (kRowMajorIo) {
                            writer_tile[
                                hierarchical_warp256_io_index<kWriterRows>(
                                    logical_row, local)] = value;
                        } else {
                            const std::uint32_t major =
                                hierarchical_warp256_io_swizzle(local);
                            const std::uint32_t physical_row =
                                logical_row ^ (major & (kWriterRows - 1U));
                            writer_tile[major * kWriterRows + physical_row] =
                                value;
                        }
                    }
                    hierarchical_warp_group_barrier(warp_group + 1U);
                }
                if constexpr (PacketSharedRadix4) {
                    hierarchical_warp128_packet_shared_radix4<
                        Word, false, kWriterRows>(
                        writer_tile, row_base, root_powers,
                        root_powers_shoup, op);
                } else if constexpr (WarpPipeline) {
                    hierarchical_warp128_static_io_packet_online<
                        Word, false, kWriterRows>(
                        exchange, writer_tile, row_base, root_powers,
                        root_powers_shoup, pipeline_flags, op);
                } else {
                    for (std::uint32_t writer_row = 0;
                         writer_row < kWriterRows; ++writer_row) {
                        const std::uint32_t physical_row =
                            row_base + writer_row;
                        const std::uint32_t row = StaticReader
                            ? physical_row : StaticWriter
                            ? __brev(physical_row) >> 22 : physical_row;
                        if constexpr (WarpCooperative) {
                            hierarchical_warp128_cooperative_static_io_row<
                                Word, false, kWriterRows>(
                                exchange, row, quarter, writer_tile,
                                writer_row, root_powers, root_powers_shoup, op);
                        } else if constexpr (WarpTileLog == 8) {
                            hierarchical_warp256_row<Word, false, StaticWriter,
                                                     StaticReader, WriterRows,
                                                     CoefficientReuseStages>(
                                input, boundary,
                                exchange + warp_group * kExchangeBuffers *
                                               kExchangeStride,
                                transform * kN, row, quarter, writer_tile,
                                writer_row, root_powers, root_powers_shoup,
                                op);
                        } else {
                            hierarchical_warp_small_static_io_row<
                                Word, false, kWriterRows, WarpTileLog,
                                CoefficientReuseStages, VectorRadix4Prefix,
                                PackedStage6, DistributedPackedStage6>(
                                exchange + warp_group * kExchangeBuffers *
                                               kExchangeStride,
                                row, quarter, writer_tile, writer_row,
                                root_powers, root_powers_shoup, op);
                        }
                    }
                }
            }
            if constexpr (StaticWriter) {
                hierarchical_warp_group_barrier(warp_group + 1U);
                Word* destination = consumer_role ? output : boundary;
                for (std::uint32_t index = group_thread;
                     index < kLocalN * kWriterRows; index += 128U) {
                    const std::uint32_t local = index / kWriterRows;
                    const std::uint32_t logical_row = index % kWriterRows;
                    Word staged;
                    if constexpr (kRowMajorIo) {
                        staged = writer_tile[
                            hierarchical_warp256_io_index<kWriterRows>(
                                logical_row, local)];
                    } else {
                        const std::uint32_t major = StaticReader
                            ? hierarchical_warp256_io_swizzle(local) : local;
                        const std::uint32_t staged_row =
                            logical_row ^ (major & (kWriterRows - 1U));
                        staged = writer_tile[
                            major * kWriterRows + staged_row];
                    }
                    destination[
                        transform * kN +
                        static_cast<std::uint64_t>(local) * kLocalN +
                        row_base + logical_row] =
                        staged;
                }
                hierarchical_warp_group_barrier(warp_group + 1U);
            }
            if (!consumer_role && lane == 0 && quarter == 0) {
                if (packet_streaming) {
                    if (packet_readiness_mode != 0U) {
                        const std::uint32_t word = row_packet >> 5;
                        const std::uint32_t bit = 1U << (row_packet & 31U);
                        cuda::atomic_ref<unsigned int,
                                         cuda::thread_scope_device>
                            packet_ready(ready[transform * 8U + word]);
                        packet_ready.fetch_or(bit,
                                              cuda::memory_order_release);
                    } else {
                        cuda::atomic_ref<unsigned int,
                                         cuda::thread_scope_device>
                            packet_ready(
                                ready[transform * row_packets + row_packet]);
                        packet_ready.store(1U, cuda::memory_order_release);
                    }
                } else {
                    cuda::atomic_ref<unsigned int,
                                     cuda::thread_scope_device>
                        readiness(ready[transform]);
                    readiness.fetch_add(kWriterRows,
                                        cuda::memory_order_release);
                }
            }
        }
    }
    if (lane == 0 && quarter == 0 && trace_started) {
        atomicMax(trace + (consumer_role ? 3 : 1), streaming_globaltimer());
    }
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 8,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false,
          bool PacketSharedRadix4 = false>
__global__ void hierarchical_homogeneous_warp256_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        false, false, CoefficientReuseStages, VectorRadix4Prefix, PackedStage6,
        DistributedPackedStage6, PacketSharedRadix4>(
        input, output, boundary, ready, transforms, root_powers,
        root_powers_shoup, trace, producer_blocks, producer_data_time,
        consumer_data_time, packet_wave_q, packet_poll_sleep,
        packet_readiness_mode, packet_compute_layout,
        packet_fold_wave_barriers, op, storage);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 8,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false,
          bool PacketSharedRadix4 = false>
__global__ __launch_bounds__(128, 4)
void hierarchical_homogeneous_warp_small_occupancy4_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    static_assert(WarpGroups == 1 && WarpTileLog < 8);
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        false, false, CoefficientReuseStages, VectorRadix4Prefix, PackedStage6,
        DistributedPackedStage6, PacketSharedRadix4>(
        input, output, boundary, ready, transforms, root_powers,
        root_powers_shoup, trace, producer_blocks, producer_data_time,
        consumer_data_time, packet_wave_q, packet_poll_sleep,
        packet_readiness_mode, packet_compute_layout,
        packet_fold_wave_barriers, op, storage);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 8,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false,
          bool PacketSharedRadix4 = false>
__global__ __launch_bounds__(256, 2)
void hierarchical_homogeneous_warp_small_occupancy2_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    static_assert(WarpGroups == 2 && WarpTileLog < 8);
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        false, false, CoefficientReuseStages, VectorRadix4Prefix, PackedStage6,
        DistributedPackedStage6, PacketSharedRadix4>(
        input, output, boundary, ready, transforms, root_powers,
        root_powers_shoup, trace, producer_blocks, producer_data_time,
        consumer_data_time, packet_wave_q, packet_poll_sleep,
        packet_readiness_mode, packet_compute_layout,
        packet_fold_wave_barriers, op, storage);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 7>
__global__ __launch_bounds__(128, 4)
void hierarchical_homogeneous_warp_small_pipeline_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    static_assert(WarpGroups == 1 && WarpTileLog < 8);
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        true>(input, output, boundary, ready, transforms, root_powers,
              root_powers_shoup, trace, producer_blocks, producer_data_time,
              consumer_data_time, packet_wave_q, packet_poll_sleep,
              packet_readiness_mode, packet_compute_layout,
              packet_fold_wave_barriers, op, storage);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 7>
__global__ __launch_bounds__(128, 4)
void hierarchical_homogeneous_warp128_cooperative_occupancy4_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    static_assert(WarpGroups == 1 && WarpTileLog == 7);
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        false, true>(input, output, boundary, ready, transforms, root_powers,
                     root_powers_shoup, trace, producer_blocks,
                     producer_data_time, consumer_data_time, packet_wave_q,
                     packet_poll_sleep, packet_readiness_mode,
                     packet_compute_layout, packet_fold_wave_barriers, op,
                     storage);
}

template <typename Word, std::uint32_t WarpGroups, bool StaticWriter = false,
          bool StaticReader = false, std::uint32_t WriterRows = 0,
          std::uint32_t WarpTileLog = 7>
__global__ __launch_bounds__(128, 3)
void hierarchical_homogeneous_warp128_cooperative_occupancy3_10x10_kernel(
    const Word* input, Word* output, Word* boundary, unsigned int* ready,
    std::uint64_t transforms, const Word* root_powers,
    const Word* root_powers_shoup, unsigned long long* trace,
    std::uint32_t producer_blocks, std::uint32_t producer_data_time,
    std::uint32_t consumer_data_time, std::uint32_t packet_wave_q,
    std::uint32_t packet_poll_sleep, std::uint32_t packet_readiness_mode,
    std::uint32_t packet_compute_layout,
    std::uint32_t packet_fold_wave_barriers, NttStageOperatorT<Word> op) {
    static_assert(WarpGroups == 1 && WarpTileLog == 7);
    extern __shared__ __align__(16) unsigned char storage[];
    hierarchical_homogeneous_warp256_10x10_body<
        Word, WarpGroups, StaticWriter, StaticReader, WriterRows, WarpTileLog,
        false, true>(input, output, boundary, ready, transforms, root_powers,
                     root_powers_shoup, trace, producer_blocks,
                     producer_data_time, consumer_data_time, packet_wave_q,
                     packet_poll_sleep, packet_readiness_mode,
                     packet_compute_layout, packet_fold_wave_barriers, op,
                     storage);
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t ThirdLogN, std::uint32_t RowsPerBlock>
__global__ void hierarchical_streaming_three_level_kernel(
    const Word* input, Word* output, Word* boundary0, Word* boundary1,
    unsigned int* ready0, unsigned int* ready1, std::uint64_t transforms,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, std::uint32_t segment0_blocks,
    std::uint32_t segment1_blocks, NttStageOperatorT<Word> op) {
    namespace cg = cooperative_groups;
    static_assert(FirstLogN + SecondLogN + ThirdLogN == 20);
    static_assert(FirstLogN >= 6 && FirstLogN <= 8);
    static_assert(SecondLogN >= 6 && SecondLogN <= 8);
    static_assert(ThirdLogN >= 6 && ThirdLogN <= 8);
    constexpr std::uint32_t kN = 1U << 20;
    constexpr std::uint32_t kFirstN = 1U << FirstLogN;
    constexpr std::uint32_t kSecondN = 1U << SecondLogN;
    constexpr std::uint32_t kThirdN = 1U << ThirdLogN;
    constexpr std::uint32_t kFirstStride = kFirstN + 1;
    constexpr std::uint32_t kSecondStride = kSecondN + 1;
    constexpr std::uint32_t kThirdStride = kThirdN + 1;
    constexpr std::uint32_t kFirstGroups = kFirstN / RowsPerBlock;
    constexpr std::uint32_t kSecondGroups = kSecondN / RowsPerBlock;
    extern __shared__ __align__(16) unsigned char storage[];
    Word* state = reinterpret_cast<Word*>(storage);

    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < transforms * kThirdN;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready0[index] = 0;
    }
    for (std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < transforms * kFirstN;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        ready1[index] = 0;
    }
    for (std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < 6; index += gridDim.x * blockDim.x) {
        trace[index] = (index & 1U) == 0 ? ~0ULL : 0ULL;
    }
    cg::this_grid().sync();

    const std::uint32_t segment2_blocks = gridDim.x - segment0_blocks - segment1_blocks;
    const std::uint32_t role_blocks[3] = {
        segment0_blocks, segment1_blocks, segment2_blocks};
    std::uint32_t segment = 0;
    std::uint32_t role_block = 0;
    std::uint32_t launch_slot = blockIdx.x;
    bool role_found = false;
    for (std::uint32_t round = 0; !role_found; ++round) {
#pragma unroll
        for (std::uint32_t candidate = 0; candidate < 3; ++candidate) {
            if (round < role_blocks[candidate]) {
                if (launch_slot == 0) {
                    segment = candidate;
                    role_block = round;
                    role_found = true;
                    break;
                }
                --launch_slot;
            }
        }
    }

    const std::uint32_t groups_per_transform =
        segment == 0 ? kThirdN * kSecondGroups
                     : segment == 1 ? kThirdN * kFirstGroups
                                    : kFirstN * kSecondGroups;
    const std::uint64_t total_groups = transforms * groups_per_transform;
    bool trace_started = false;

    for (std::uint64_t group = role_block; group < total_groups;
         group += role_blocks[segment]) {
        const std::uint64_t transform = group / groups_per_transform;
        const std::uint32_t within = static_cast<std::uint32_t>(
            group - transform * groups_per_transform);
        const std::uint64_t transform_base = transform * kN;

        if (segment == 0) {
            const std::uint32_t wave = within / kSecondGroups;
            const std::uint32_t row_base =
                (within % kSecondGroups) * RowsPerBlock;
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                const std::uint32_t row = linear / kFirstN;
                const std::uint32_t element = linear - row * kFirstN;
                const std::uint32_t remainder =
                    (row_base + row) * kThirdN + wave;
                const std::uint32_t reversed =
                    __brev(element) >> (32 - FirstLogN);
                state[row * kFirstStride + reversed] =
                    input[transform_base + static_cast<std::uint64_t>(element) *
                                               (kSecondN * kThirdN) + remainder];
            }
            __syncthreads();
            if (threadIdx.x == 0 && !trace_started) {
                atomicMin(trace, streaming_globaltimer());
                trace_started = true;
            }
            detail::hierarchical_resident_radix4<false, Word, FirstLogN,
                                                  RowsPerBlock>(
                state, op);
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kFirstN; linear += blockDim.x) {
                const std::uint32_t row = linear / kFirstN;
                const std::uint32_t frequency = linear - row * kFirstN;
                const std::uint32_t remainder =
                    (row_base + row) * kThirdN + wave;
                const std::uint64_t exponent =
                    static_cast<std::uint64_t>(frequency) * remainder;
                const Word value = op.multiply(
                    state[row * kFirstStride + frequency], root_powers[exponent],
                    root_powers_shoup[exponent]);
                boundary0[transform_base + static_cast<std::uint64_t>(frequency) *
                                                (kSecondN * kThirdN) + remainder] = value;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(
                    ready0[transform * kThirdN + wave]);
                readiness.fetch_add(RowsPerBlock, cuda::memory_order_release);
            }
        } else if (segment == 1) {
            const std::uint32_t wave = within / kFirstGroups;
            const std::uint32_t prefix_base =
                (within % kFirstGroups) * RowsPerBlock;
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(
                    ready0[transform * kThirdN + wave]);
                while (readiness.load(cuda::memory_order_acquire) < kSecondN) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
            }
            __syncthreads();
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                const std::uint32_t row = linear / kSecondN;
                const std::uint32_t element = linear - row * kSecondN;
                const std::uint32_t prefix = prefix_base + row;
                const std::uint64_t logical =
                    (static_cast<std::uint64_t>(prefix) * kSecondN + element) *
                        kThirdN + wave;
                const std::uint32_t reversed =
                    __brev(element) >> (32 - SecondLogN);
                state[row * kSecondStride + reversed] =
                    boundary0[transform_base + logical];
            }
            __syncthreads();
            if (threadIdx.x == 0 && !trace_started) {
                atomicMin(trace + 2, streaming_globaltimer());
                trace_started = true;
            }
            detail::hierarchical_resident_radix4<false, Word, SecondLogN,
                                                  RowsPerBlock>(
                state, op);
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kSecondN; linear += blockDim.x) {
                const std::uint32_t row = linear / kSecondN;
                const std::uint32_t frequency = linear - row * kSecondN;
                const std::uint32_t prefix = prefix_base + row;
                const std::uint64_t exponent =
                    static_cast<std::uint64_t>(kFirstN) * frequency * wave;
                const Word value = op.multiply(
                    state[row * kSecondStride + frequency], root_powers[exponent],
                    root_powers_shoup[exponent]);
                const std::uint64_t logical =
                    (static_cast<std::uint64_t>(prefix) * kSecondN + frequency) *
                        kThirdN + wave;
                boundary1[transform_base + logical] = value;
            }
            __syncthreads();
            if (threadIdx.x < RowsPerBlock) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(
                    ready1[transform * kFirstN + prefix_base + threadIdx.x]);
                readiness.fetch_add(1U, cuda::memory_order_release);
            }
        } else {
            const std::uint32_t upper = within / kSecondGroups;
            const std::uint32_t lower_base =
                (within % kSecondGroups) * RowsPerBlock;
            if (threadIdx.x == 0) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> readiness(
                    ready1[transform * kFirstN + upper]);
                while (readiness.load(cuda::memory_order_acquire) < kThirdN) {
#if __CUDA_ARCH__ >= 700
                    __nanosleep(64);
#endif
                }
            }
            __syncthreads();
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kThirdN; linear += blockDim.x) {
                const std::uint32_t row = linear / kThirdN;
                const std::uint32_t element = linear - row * kThirdN;
                const std::uint32_t prefix =
                    upper * kSecondN + lower_base + row;
                const std::uint32_t reversed =
                    __brev(element) >> (32 - ThirdLogN);
                state[row * kThirdStride + reversed] =
                    boundary1[transform_base + static_cast<std::uint64_t>(prefix) *
                                                  kThirdN + element];
            }
            __syncthreads();
            if (threadIdx.x == 0 && !trace_started) {
                atomicMin(trace + 4, streaming_globaltimer());
                trace_started = true;
            }
            detail::hierarchical_resident_radix4<false, Word, ThirdLogN,
                                                  RowsPerBlock>(
                state, op);
            for (std::uint32_t linear = threadIdx.x;
                 linear < RowsPerBlock * kThirdN; linear += blockDim.x) {
                const std::uint32_t row = linear / kThirdN;
                const std::uint32_t frequency = linear - row * kThirdN;
                const std::uint32_t lower = lower_base + row;
                const std::uint32_t reordered =
                    upper + (lower << FirstLogN) +
                    (frequency << (FirstLogN + SecondLogN));
                output[transform_base + reordered] =
                    op.finalize(state[row * kThirdStride + frequency], kN);
            }
            __syncthreads();
        }
    }

    if (threadIdx.x == 0 && trace_started) {
        atomicMax(trace + 2 * segment + 1, streaming_globaltimer());
    }
}

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
        const NttStageOperator op{twiddles, twiddles_shoup, config.modulus, 1, 0};
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

std::size_t hybrid_dataflow_shared_bytes(const PlanConfig& config) {
    const std::size_t word_bytes = config.word_bits / 8;
    const std::size_t states = config.dataflow_state_mode == DataflowStateMode::PingPong ? 2 : 1;
    const std::size_t state_words = states * (std::size_t{1} << config.log_n);
    if (config.compute_unit == ComputeUnit::Radix4) {
        return state_words * word_bytes;
    }
    const std::size_t roles = (config.stage_space + config.role_stages - 1) / config.role_stages;
    const std::size_t channel_words = static_cast<std::size_t>(roles - 1) * config.pipeline_buffers *
                                      (std::size_t{1} << config.stage_space) * config.data_time;
    const std::size_t flags = static_cast<std::size_t>(roles - 1) * config.pipeline_buffers *
                              config.role_stages * sizeof(int);
    return (state_words + channel_words) * word_bytes + flags;
}

template <typename Word, std::uint32_t FlowTileLogN, std::uint32_t StageSpace,
          std::uint32_t DataTime, std::uint32_t RoleStages, std::uint32_t TargetCtasPerSm,
          std::uint32_t TokenInterleave>
void launch_hybrid_dataflow_point(const PlanConfig& config, const Word* input, Word* output,
                                  const Word* twiddles, const Word* twiddles_shoup,
                                  std::uint64_t inverse_n, std::uint64_t inverse_n_shoup) {
    const detail::HybridDataflowRuntime runtime{config.log_n, config.data_space, config.data_time, config.pipeline_buffers,
                                                config.dataflow_layout, config.dataflow_state_mode, config.stage_handoff};
    const Word scale = config.inverse ? static_cast<Word>(inverse_n) : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup) : static_cast<Word>(0);
    const NttStageOperatorT<Word> op{twiddles, twiddles_shoup, static_cast<Word>(config.modulus), scale, scale_shoup};
    const std::size_t shared_bytes = hybrid_dataflow_shared_bytes(config);
    if (config.compute_unit == ComputeUnit::Radix4) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            detail::hybrid_resident_radix4_kernel<Word, FlowTileLogN, TargetCtasPerSm,
                                                  NttStageOperatorT<Word>>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
        detail::hybrid_resident_radix4_kernel<Word, FlowTileLogN, TargetCtasPerSm,
                                              NttStageOperatorT<Word>>
            <<<static_cast<unsigned int>(config.batch), 256, shared_bytes, active_stream>>>(
                input, output, config.batch, runtime, op);
        CUNTT_CUDA_CHECK(cudaGetLastError());
        return;
    }
    CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
        detail::hybrid_dataflow_kernel<Word, FlowTileLogN, StageSpace, DataTime, RoleStages,
                                       TargetCtasPerSm, TokenInterleave, NttStageOperatorT<Word>>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
    constexpr std::uint32_t roles = (StageSpace + RoleStages - 1) / RoleStages;
    detail::hybrid_dataflow_kernel<Word, FlowTileLogN, StageSpace, DataTime, RoleStages,
                                   TargetCtasPerSm, TokenInterleave, NttStageOperatorT<Word>>
        <<<static_cast<unsigned int>(config.batch), roles * RoleStages * 32, shared_bytes, active_stream>>>(
            input, output, config.batch, runtime, op);
    CUNTT_CUDA_CHECK(cudaGetLastError());
}

template <typename Word>
void dispatch_hybrid_dataflow(const PlanConfig& config, const Word* input, Word* output,
                              const Word* twiddles, const Word* twiddles_shoup,
                              std::uint64_t inverse_n, std::uint64_t inverse_n_shoup) {
#define CUNTT_TRY_FLOW(FLOW, US, TD, UR, RB, TI)                                                          \
    if (config.flow_tile_log_n == FLOW && config.stage_space == US && config.data_time == TD &&           \
        config.role_stages == UR && config.target_ctas_per_sm == RB && config.token_interleave == TI) {   \
        launch_hybrid_dataflow_point<Word, FLOW, US, TD, UR, RB, TI>(config, input, output, twiddles,     \
                                                                  twiddles_shoup, inverse_n, inverse_n_shoup); \
        return;                                                                                            \
    }
    CUNTT_FOR_EACH_HYBRID_DATAFLOW_KERNEL(CUNTT_TRY_FLOW)
#undef CUNTT_TRY_FLOW
    throw std::logic_error("requested HybridDataflow point was not generated");
}

template <bool Hybrid2DCore, typename Word, std::uint32_t FirstLogN,
          std::uint32_t SecondLogN, std::uint32_t RowsPerBlock>
void launch_hierarchical_dataflow_point_implementation(
    const PlanConfig& config, const Word* input, Word* output, Word* boundary,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    using Operator = NttStageOperatorT<Word>;
    const Word scale = config.inverse ? static_cast<Word>(inverse_n) : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup) : static_cast<Word>(0);
    Operator op{twiddles, twiddles_shoup, static_cast<Word>(config.modulus), scale, scale_shoup};
    constexpr std::uint32_t first_n = 1U << FirstLogN;
    constexpr std::uint32_t second_n = 1U << SecondLogN;
    constexpr std::uint32_t local_n = first_n > second_n ? first_n : second_n;
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(RowsPerBlock) * (local_n + 1) * sizeof(Word);
    auto kernel = detail::hierarchical_dataflow_kernel<Word, FirstLogN, SecondLogN,
                                                       RowsPerBlock, Operator, Hybrid2DCore>;
    CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    const std::uint64_t first_groups = static_cast<std::uint64_t>(config.batch) * second_n / RowsPerBlock;
    const std::uint64_t second_groups = static_cast<std::uint64_t>(config.batch) * first_n / RowsPerBlock;
    const std::uint64_t resident_grid = static_cast<std::uint64_t>(multiprocessors) * config.target_ctas_per_sm;
    const unsigned int grid = static_cast<unsigned int>(
        std::max<std::uint64_t>(1, std::min(std::max(first_groups, second_groups), resident_grid)));
    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundary;
    std::uint64_t transforms = config.batch;
    std::uint32_t data_time = config.data_time;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    void* arguments[] = {&input_arg, &output_arg, &boundary_arg, &transforms, &data_time,
                         &roots_arg, &roots_shoup_arg, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(config.threads_per_block),
        arguments, shared_bytes, active_stream));
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t RowsPerBlock>
void launch_hierarchical_dataflow_point(
    const PlanConfig& config, const Word* input, Word* output, Word* boundary,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if constexpr (FirstLogN == 10 && SecondLogN == 10) {
        if (config.hierarchical_core == HierarchicalCore::Hybrid2DRadix4) {
            launch_hierarchical_dataflow_point_implementation<true, Word, FirstLogN,
                SecondLogN, RowsPerBlock>(config, input, output, boundary, twiddles,
                twiddles_shoup, root_powers, root_powers_shoup, inverse_n,
                inverse_n_shoup, multiprocessors);
            return;
        }
    }
    launch_hierarchical_dataflow_point_implementation<false, Word, FirstLogN,
        SecondLogN, RowsPerBlock>(config, input, output, boundary, twiddles,
        twiddles_shoup, root_powers, root_powers_shoup, inverse_n,
        inverse_n_shoup, multiprocessors);
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN>
void dispatch_hierarchical_dataflow_rows(
    const PlanConfig& config, const Word* input, Word* output, Word* boundary,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
#define CUNTT_LAUNCH_HIERARCHICAL_ROWS(ROWS)                                              \
    launch_hierarchical_dataflow_point<Word, FirstLogN, SecondLogN, ROWS>(                \
        config, input, output, boundary, twiddles, twiddles_shoup, root_powers,            \
        root_powers_shoup, inverse_n, inverse_n_shoup, multiprocessors)
    switch (config.rows_per_block) {
        case 1: CUNTT_LAUNCH_HIERARCHICAL_ROWS(1); break;
        case 2: CUNTT_LAUNCH_HIERARCHICAL_ROWS(2); break;
        case 4: CUNTT_LAUNCH_HIERARCHICAL_ROWS(4); break;
        default: throw std::logic_error("unsupported HierarchicalDataflow rows_per_block");
    }
#undef CUNTT_LAUNCH_HIERARCHICAL_ROWS
}

template <typename Word, std::uint32_t FirstLogN>
void dispatch_hierarchical_dataflow_second(
    const PlanConfig& config, std::uint32_t second_log_n, const Word* input,
    Word* output, Word* boundary, const Word* twiddles, const Word* twiddles_shoup,
    const Word* root_powers, const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
#define CUNTT_HIERARCHICAL_SECOND(LOG)                                                    \
    dispatch_hierarchical_dataflow_rows<Word, FirstLogN, LOG>(                           \
        config, input, output, boundary, twiddles, twiddles_shoup, root_powers,           \
        root_powers_shoup, inverse_n, inverse_n_shoup, multiprocessors)
    switch (second_log_n) {
        case 6: CUNTT_HIERARCHICAL_SECOND(6); break;
        case 7: CUNTT_HIERARCHICAL_SECOND(7); break;
        case 8: CUNTT_HIERARCHICAL_SECOND(8); break;
        case 9: CUNTT_HIERARCHICAL_SECOND(9); break;
        case 10: CUNTT_HIERARCHICAL_SECOND(10); break;
        default: throw std::logic_error("unsupported HierarchicalDataflow second layer");
    }
#undef CUNTT_HIERARCHICAL_SECOND
}

template <typename Word>
void dispatch_hierarchical_dataflow(
    const PlanConfig& config, const Word* input, Word* output, Word* boundary,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    const std::uint32_t first_log_n = config.log_n - config.n1_log;
#define CUNTT_HIERARCHICAL_FIRST(LOG)                                                    \
    dispatch_hierarchical_dataflow_second<Word, LOG>(                                   \
        config, config.n1_log, input, output, boundary, twiddles, twiddles_shoup,        \
        root_powers, root_powers_shoup, inverse_n, inverse_n_shoup, multiprocessors)
    switch (first_log_n) {
        case 6: CUNTT_HIERARCHICAL_FIRST(6); break;
        case 7: CUNTT_HIERARCHICAL_FIRST(7); break;
        case 8: CUNTT_HIERARCHICAL_FIRST(8); break;
        case 9: CUNTT_HIERARCHICAL_FIRST(9); break;
        case 10: CUNTT_HIERARCHICAL_FIRST(10); break;
        default: throw std::logic_error("unsupported HierarchicalDataflow first layer");
    }
#undef CUNTT_HIERARCHICAL_FIRST
}

template <bool Hybrid2DCore, typename Word, std::uint32_t FirstLogN,
          std::uint32_t SecondLogN, std::uint32_t RowsPerBlock>
std::uint32_t hierarchical_dataflow_residency_implementation(std::uint32_t threads) {
    using Operator = NttStageOperatorT<Word>;
    constexpr std::uint32_t first_n = 1U << FirstLogN;
    constexpr std::uint32_t second_n = 1U << SecondLogN;
    constexpr std::uint32_t local_n = first_n > second_n ? first_n : second_n;
    constexpr std::size_t shared_bytes =
        static_cast<std::size_t>(RowsPerBlock) * (local_n + 1) * sizeof(Word);
    auto kernel = detail::hierarchical_dataflow_kernel<Word, FirstLogN, SecondLogN,
                                                       RowsPerBlock, Operator, Hybrid2DCore>;
    CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int blocks = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, static_cast<int>(threads), shared_bytes));
    return static_cast<std::uint32_t>(blocks);
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t RowsPerBlock>
std::uint32_t hierarchical_dataflow_residency_point(const PlanConfig& config) {
    if constexpr (FirstLogN == 10 && SecondLogN == 10) {
        if (config.hierarchical_core == HierarchicalCore::Hybrid2DRadix4) {
            return hierarchical_dataflow_residency_implementation<true, Word,
                FirstLogN, SecondLogN, RowsPerBlock>(config.threads_per_block);
        }
    }
    return hierarchical_dataflow_residency_implementation<false, Word, FirstLogN,
        SecondLogN, RowsPerBlock>(config.threads_per_block);
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN>
std::uint32_t hierarchical_dataflow_residency_rows(const PlanConfig& config) {
    switch (config.rows_per_block) {
        case 1: return hierarchical_dataflow_residency_point<Word, FirstLogN, SecondLogN, 1>(config);
        case 2: return hierarchical_dataflow_residency_point<Word, FirstLogN, SecondLogN, 2>(config);
        case 4: return hierarchical_dataflow_residency_point<Word, FirstLogN, SecondLogN, 4>(config);
        default: return 0;
    }
}

template <typename Word, std::uint32_t FirstLogN>
std::uint32_t hierarchical_dataflow_residency_second(const PlanConfig& config) {
    switch (config.n1_log) {
        case 6: return hierarchical_dataflow_residency_rows<Word, FirstLogN, 6>(config);
        case 7: return hierarchical_dataflow_residency_rows<Word, FirstLogN, 7>(config);
        case 8: return hierarchical_dataflow_residency_rows<Word, FirstLogN, 8>(config);
        case 9: return hierarchical_dataflow_residency_rows<Word, FirstLogN, 9>(config);
        case 10: return hierarchical_dataflow_residency_rows<Word, FirstLogN, 10>(config);
        default: return 0;
    }
}

template <typename Word>
std::uint32_t hierarchical_dataflow_residency(const PlanConfig& config) {
    switch (config.log_n - config.n1_log) {
        case 6: return hierarchical_dataflow_residency_second<Word, 6>(config);
        case 7: return hierarchical_dataflow_residency_second<Word, 7>(config);
        case 8: return hierarchical_dataflow_residency_second<Word, 8>(config);
        case 9: return hierarchical_dataflow_residency_second<Word, 9>(config);
        case 10: return hierarchical_dataflow_residency_second<Word, 10>(config);
        default: return 0;
    }
}

struct StreamingWorkspaceLayout {
    std::uint64_t boundary_words = 0;
    std::uint64_t token_count = 0;
    std::uint64_t state_count = 0;
    std::size_t ready_offset = 0;
    std::size_t state_offset = 0;
    std::size_t completed_offset = 0;
    std::size_t trace_offset = 0;
    std::size_t total_bytes = 0;
};

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

StreamingWorkspaceLayout hierarchical_streaming_workspace_layout(const PlanConfig& config) {
    StreamingWorkspaceLayout layout;
    if (uses_appt_split_tail(config)) {
        const std::uint64_t n = 1ULL << config.log_n;
        layout.boundary_words = config.batch * (n + n / 2);
        layout.ready_offset = static_cast<std::size_t>(
            layout.boundary_words * (config.word_bits / 8));
        layout.token_count = config.batch * (64 + 128);
        layout.trace_offset = align_up(
            layout.ready_offset + layout.token_count * sizeof(unsigned int),
            alignof(unsigned long long));
        layout.total_bytes = layout.trace_offset +
                             2 * config.stage_partition.size() *
                                 sizeof(unsigned long long);
        return layout;
    }
    if (uses_appt_register_tail(config)) {
        const std::uint64_t n = 1ULL << config.log_n;
        const bool grouped = uses_appt_register_tail_grouped(config);
        layout.boundary_words = config.batch * n * (grouped ? 2 : 1);
        layout.ready_offset = static_cast<std::size_t>(
            layout.boundary_words * (config.word_bits / 8));
        const std::uint64_t input_tokens = config.batch *
            (grouped
                 ? 64 / config.subgraph_mappings.front().data_space
                 : (config.word_bits == 32 ? 2 : 64));
        const std::uint64_t output_tokens =
            config.output_order == OutputOrder::Natural
                ? config.batch *
                      (128 / config.appt_role_mapping.fragment_width)
                : 0;
        layout.token_count = input_tokens + output_tokens;
        layout.trace_offset = align_up(
            layout.ready_offset + layout.token_count * sizeof(unsigned int),
            alignof(unsigned long long));
        layout.total_bytes = layout.trace_offset +
                             2 * config.stage_partition.size() *
                                 sizeof(unsigned long long);
        return layout;
    }
    if (uses_appt_fused_tail(config)) {
        const std::uint64_t n = 1ULL << config.log_n;
        layout.boundary_words = config.batch * n;
        layout.ready_offset = static_cast<std::size_t>(
            layout.boundary_words * (config.word_bits / 8));
        layout.token_count = config.batch * 64;
        layout.trace_offset = align_up(
            layout.ready_offset + layout.token_count * sizeof(unsigned int),
            alignof(unsigned long long));
        layout.total_bytes = layout.trace_offset +
                             2 * config.stage_partition.size() *
                                 sizeof(unsigned long long);
        return layout;
    }
    if (uses_appt_online(config)) {
        const std::uint64_t n = 1ULL << config.log_n;
        layout.boundary_words = 2 * config.batch * n;
        layout.ready_offset = static_cast<std::size_t>(
            layout.boundary_words * (config.word_bits / 8));
        layout.token_count = config.batch * (64 + 128);
        layout.trace_offset = align_up(
            layout.ready_offset + layout.token_count * sizeof(unsigned int),
            alignof(unsigned long long));
        layout.total_bytes = layout.trace_offset + 2 * config.stage_partition.size() *
                             sizeof(unsigned long long) +
                             (config.profile_appt_roles ? 12 * sizeof(unsigned long long) : 0);
        return layout;
    }
    if (uses_appt_pipeline(config)) {
        const std::uint64_t n = 1ULL << config.log_n;
        layout.boundary_words = config.batch * n;
        layout.trace_offset = static_cast<std::size_t>(
            layout.boundary_words * (config.word_bits / 8));
        layout.total_bytes = layout.trace_offset + 2 * config.stage_partition.size() *
                             sizeof(unsigned long long);
        return layout;
    }
    const std::uint64_t n = 1ULL << config.log_n;
    const bool packet_streaming =
        config.log_n == 20 && config.stage_partition.size() == 2 &&
        config.subgraph_mappings.size() == 2 &&
        config.subgraph_mappings.front().core == NttSubgraphCore::
            HomogeneousWarp128PacketSharedRadix4StaticIo &&
        config.subgraph_mappings.back().token_interleave != 0U;
    std::uint32_t prefix_log = 0;
    for (std::size_t segment = 0; segment < config.stage_partition.size(); ++segment) {
        const std::uint32_t stages = config.stage_partition[segment];
        const std::uint32_t remaining_log = config.log_n - prefix_log - stages;
        if (segment + 1 < config.stage_partition.size()) {
            const auto& boundary = config.boundary_mappings[segment];
            const std::uint64_t slots = boundary.storage == BoundaryStorage::FullScratch
                                            ? config.batch
                                            : std::min<std::size_t>(config.batch, boundary.buffers);
            const bool aggregate_readiness = slots == config.batch;
            const std::uint32_t next_remaining_log =
                remaining_log - config.stage_partition[segment + 1];
            const std::uint32_t token_task_log = aggregate_readiness
                                                     ? prefix_log + next_remaining_log
                                                     : prefix_log + remaining_log;
            const std::uint64_t tasks_per_transform = 1ULL << token_task_log;
            layout.boundary_words += slots * n;
            layout.token_count += slots * tasks_per_transform;
            if (slots < config.batch) layout.state_count += slots;
        }
        prefix_log += stages;
    }
    if (packet_streaming) {
        layout.token_count = config.batch * (1U << 8);
    }
    layout.ready_offset = static_cast<std::size_t>(layout.boundary_words * (config.word_bits / 8));
    layout.state_offset = align_up(layout.ready_offset + layout.token_count * sizeof(unsigned int),
                                   alignof(unsigned long long));
    layout.completed_offset = layout.state_offset + layout.state_count * sizeof(unsigned long long);
    layout.trace_offset = align_up(layout.completed_offset + layout.state_count * sizeof(unsigned int),
                                   alignof(unsigned long long));
    layout.total_bytes = layout.trace_offset +
                         2 * config.stage_partition.size() * sizeof(unsigned long long);
    return layout;
}

std::size_t hierarchical_streaming_workspace_bytes(const PlanConfig& config) {
    return hierarchical_streaming_workspace_layout(config).total_bytes;
}

bool uses_specialized_hierarchical_streaming_10x10(const PlanConfig& config) {
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(), config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::DataflowRadix4 &&
                                  mapping.threads_per_block == 256;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch;
}

bool uses_homogeneous_hierarchical_streaming_10x10(const PlanConfig& config) {
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                      NttSubgraphCore::HomogeneousRadix4 &&
                                  mapping.threads_per_block == 256 &&
                                  mapping.units_per_cta == 4 &&
                                  mapping.data_time > 0;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch;
}

bool uses_homogeneous_warp_hierarchical_streaming_10x10(
    const PlanConfig& config) {
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core ==
                                      NttSubgraphCore::HomogeneousWarpRadix2 &&
                                  mapping.threads_per_block == 256 &&
                                  mapping.units_per_cta == 8 &&
                                  mapping.data_time > 0;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch &&
           config.modular_multiply == ModularMultiply::Shoup;
}

bool uses_homogeneous_warp256_hierarchical_streaming_10x10(
    const PlanConfig& config) {
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::
                                                      HomogeneousWarp256Radix2 &&
                                  (mapping.threads_per_block == 128 ||
                                   mapping.threads_per_block == 256) &&
                                  mapping.units_per_cta ==
                                      mapping.threads_per_block / 32 &&
                                  mapping.data_time > 0;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch &&
           config.modular_multiply == ModularMultiply::Shoup;
}

bool uses_homogeneous_warp256_static_hierarchical_streaming_10x10(
    const PlanConfig& config) {
    const std::uint32_t full_rows = config.word_bits == 32 ? 8U : 4U;
    const std::uint32_t half_rows = full_rows / 2U;
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::
                                      HomogeneousWarp256StaticRadix2 &&
                                  (mapping.threads_per_block == 128 ||
                                   mapping.threads_per_block == 256) &&
                                  mapping.units_per_cta ==
                                      mapping.threads_per_block / 32 &&
                                  mapping.data_time > 0;
                       }) &&
           (config.subgraph_mappings.front().data_space == 0 ||
            config.subgraph_mappings.front().data_space == half_rows ||
            config.subgraph_mappings.front().data_space == full_rows) &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [&](const NttSubgraphMapping& mapping) {
                           return mapping.data_space ==
                                      config.subgraph_mappings.front().data_space &&
                                  mapping.coefficient_reuse_stages ==
                                      config.subgraph_mappings.front()
                                          .coefficient_reuse_stages;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch &&
           config.modular_multiply == ModularMultiply::Shoup;
}

bool uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
    const PlanConfig& config,
    NttSubgraphCore core =
        NttSubgraphCore::HomogeneousWarp256StaticIoRadix2) {
    const std::uint32_t full_rows = config.word_bits == 32 ? 8U : 4U;
    const std::uint32_t half_rows = full_rows / 2U;
    return config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [&](const NttSubgraphMapping& mapping) {
                           const bool coefficient_reuse =
                               core == NttSubgraphCore::
                                           HomogeneousWarp256CoefficientReuseStaticIoRadix2;
                           return mapping.core == core &&
                                  (coefficient_reuse
                                       ? config.word_bits == 32 &&
                                             mapping.threads_per_block == 128
                                       : mapping.threads_per_block == 128 ||
                                             mapping.threads_per_block == 256) &&
                                  mapping.units_per_cta ==
                                      mapping.threads_per_block / 32 &&
                                  mapping.data_time > 0;
                       }) &&
           (core == NttSubgraphCore::
                        HomogeneousWarp256CoefficientReuseStaticIoRadix2
                ? config.subgraph_mappings.front().data_space == half_rows
                : config.subgraph_mappings.front().data_space == 0 ||
                      config.subgraph_mappings.front().data_space == half_rows ||
                      config.subgraph_mappings.front().data_space == full_rows) &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [&](const NttSubgraphMapping& mapping) {
                           return mapping.data_space ==
                                      config.subgraph_mappings.front().data_space &&
                                  mapping.coefficient_reuse_stages ==
                                      config.subgraph_mappings.front()
                                          .coefficient_reuse_stages;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch &&
           config.modular_multiply == ModularMultiply::Shoup;
}

bool uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
    const PlanConfig& config, NttSubgraphCore core) {
    const std::uint32_t coefficient_reuse_stages =
        config.subgraph_mappings.empty()
            ? 0U
            : config.subgraph_mappings.front().coefficient_reuse_stages;
    const bool vector_radix4 =
        core == NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo ||
        core == NttSubgraphCore::
                    HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
        core == NttSubgraphCore::
                    HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
    const bool packed_stage6 =
        core == NttSubgraphCore::
                    HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
        core == NttSubgraphCore::
                    HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
    return config.word_bits == 32 && config.log_n == 20 &&
           config.stage_partition == std::vector<std::uint32_t>{10, 10} &&
           config.subgraph_mappings.size() == 2 &&
           std::all_of(config.subgraph_mappings.begin(),
                       config.subgraph_mappings.end(),
                       [&](const NttSubgraphMapping& mapping) {
                           const bool single_group =
                               mapping.threads_per_block == 128 &&
                               mapping.units_per_cta == 4;
                           const bool dual_group =
                               (core == NttSubgraphCore::
                                            HomogeneousWarp128StaticIoRadix2 ||
                                core == NttSubgraphCore::
                                            HomogeneousWarp128CoefficientReuseStaticIoRadix2 ||
                                core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4StaticIo ||
                                core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
                                core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo) &&
                               mapping.threads_per_block == 256 &&
                               mapping.units_per_cta == 8;
                           const bool packet_shared =
                               core == NttSubgraphCore::
                                           HomogeneousWarp128PacketSharedRadix4StaticIo &&
                               ((mapping.threads_per_block == 128 &&
                                 mapping.units_per_cta == 4) ||
                                (mapping.threads_per_block == 256 &&
                                 mapping.units_per_cta == 8));
                           return mapping.core == core &&
                                  (single_group || dual_group || packet_shared) &&
                                  mapping.data_space == 4 &&
                                  mapping.coefficient_reuse_stages ==
                                      config.subgraph_mappings.front()
                                          .coefficient_reuse_stages &&
                                  mapping.data_time > 0;
                       }) &&
           config.boundary_mappings.size() == 1 &&
           config.boundary_mappings[0].storage == BoundaryStorage::FullScratch &&
           (!vector_radix4 ||
            coefficient_reuse_stages == 0U ||
            (coefficient_reuse_stages >= 3U &&
             coefficient_reuse_stages <= 7U)) &&
           (!packed_stage6 || coefficient_reuse_stages == 6U) &&
           (core != NttSubgraphCore::
                        HomogeneousWarp128PacketSharedRadix4StaticIo ||
            coefficient_reuse_stages == 0U) &&
           config.modular_multiply == ModularMultiply::Shoup;
}

bool uses_specialized_hierarchical_streaming_three_level(
    const PlanConfig& config) {
    const bool supported_partition =
        config.stage_partition == std::vector<std::uint32_t>{6, 7, 7} ||
        config.stage_partition == std::vector<std::uint32_t>{7, 6, 7} ||
        config.stage_partition == std::vector<std::uint32_t>{7, 7, 6} ||
        config.stage_partition == std::vector<std::uint32_t>{6, 6, 8} ||
        config.stage_partition == std::vector<std::uint32_t>{6, 8, 6} ||
        config.stage_partition == std::vector<std::uint32_t>{8, 6, 6};
    return config.log_n == 20 &&
           supported_partition &&
           config.subgraph_mappings.size() == 3 &&
           std::all_of(config.subgraph_mappings.begin(), config.subgraph_mappings.end(),
                       [](const NttSubgraphMapping& mapping) {
                           return mapping.core == NttSubgraphCore::DataflowRadix4 &&
                                  ((mapping.threads_per_block == 128 &&
                                    mapping.units_per_cta == 4) ||
                                   (mapping.threads_per_block == 256 &&
                                    (mapping.units_per_cta == 8 ||
                                     mapping.units_per_cta == 16 ||
                                     mapping.units_per_cta == 32)));
                       }) &&
           std::all_of(config.subgraph_mappings.begin(), config.subgraph_mappings.end(),
                       [&](const NttSubgraphMapping& mapping) {
                           return mapping.units_per_cta ==
                                      config.subgraph_mappings[0].units_per_cta &&
                                  mapping.threads_per_block ==
                                      config.subgraph_mappings[0].threads_per_block;
                       }) &&
           config.boundary_mappings.size() == 2 &&
           std::all_of(config.boundary_mappings.begin(), config.boundary_mappings.end(),
                       [](const NttBoundaryMapping& mapping) {
                           return mapping.storage == BoundaryStorage::FullScratch;
                       });
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t ThirdLogN, std::uint32_t RowsPerBlock>
void launch_specialized_hierarchical_streaming_three_level(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, NttStageOperatorT<Word> op, int multiprocessors) {
    constexpr std::uint64_t kN = 1ULL << 20;
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundary0 = static_cast<Word*>(workspace);
    Word* boundary1 = boundary0 + config.batch * kN;
    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* ready0 = reinterpret_cast<unsigned int*>(workspace_bytes + layout.ready_offset);
    auto* ready1 = ready0 + config.batch * (1U << ThirdLogN);
    auto kernel = hierarchical_streaming_three_level_kernel<
        Word, FirstLogN, SecondLogN, ThirdLogN, RowsPerBlock>;
    constexpr std::uint32_t kLargestLog =
        FirstLogN > SecondLogN
            ? (FirstLogN > ThirdLogN ? FirstLogN : ThirdLogN)
            : (SecondLogN > ThirdLogN ? SecondLogN : ThirdLogN);
    constexpr std::size_t kSharedWords =
        RowsPerBlock * ((1U << kLargestLog) + 1U);
    const std::size_t shared_bytes = kSharedWords * sizeof(Word);
    int occupancy = 0;
    const std::uint32_t threads =
        config.subgraph_mappings.front().threads_per_block;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = std::max(3U, resident_per_sm *
        static_cast<std::uint32_t>(multiprocessors));

    const std::uint32_t total_weight =
        config.subgraph_mappings[0].cta_weight +
        config.subgraph_mappings[1].cta_weight +
        config.subgraph_mappings[2].cta_weight;
    const std::uint32_t distributable = grid - 3U;
    std::uint32_t segment0_blocks = 1U + distributable *
        config.subgraph_mappings[0].cta_weight / total_weight;
    std::uint32_t segment1_blocks = 1U + distributable *
        config.subgraph_mappings[1].cta_weight / total_weight;
    if (segment0_blocks + segment1_blocks >= grid) {
        segment1_blocks = grid - segment0_blocks - 1U;
    }

    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary0_arg = boundary0;
    Word* boundary1_arg = boundary1;
    unsigned int* ready0_arg = ready0;
    unsigned int* ready1_arg = ready1;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    unsigned long long* trace_arg = trace;
    void* arguments[] = {
        &input_arg, &output_arg, &boundary0_arg, &boundary1_arg,
        &ready0_arg, &ready1_arg, &transforms, &roots_arg, &roots_shoup_arg,
        &trace_arg, &segment0_blocks, &segment1_blocks, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t ThirdLogN>
bool launch_specialized_hierarchical_streaming_three_level_rows(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, NttStageOperatorT<Word> op,
    int multiprocessors) {
    switch (config.subgraph_mappings[0].units_per_cta) {
    case 4:
        launch_specialized_hierarchical_streaming_three_level<
            Word, FirstLogN, SecondLogN, ThirdLogN, 4>(
            config, input, output, workspace, root_powers, root_powers_shoup,
            trace, op, multiprocessors);
        return true;
    case 8:
        launch_specialized_hierarchical_streaming_three_level<
            Word, FirstLogN, SecondLogN, ThirdLogN, 8>(
            config, input, output, workspace, root_powers, root_powers_shoup,
            trace, op, multiprocessors);
        return true;
    case 16:
        launch_specialized_hierarchical_streaming_three_level<
            Word, FirstLogN, SecondLogN, ThirdLogN, 16>(
            config, input, output, workspace, root_powers, root_powers_shoup,
            trace, op, multiprocessors);
        return true;
    case 32:
        launch_specialized_hierarchical_streaming_three_level<
            Word, FirstLogN, SecondLogN, ThirdLogN, 32>(
            config, input, output, workspace, root_powers, root_powers_shoup,
            trace, op, multiprocessors);
        return true;
    default:
        return false;
    }
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t RowsPerBlock>
void launch_homogeneous_hierarchical_streaming_two_level(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, NttStageOperatorT<Word> op,
    int multiprocessors) {
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundary = static_cast<Word*>(workspace);
    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* ready = reinterpret_cast<unsigned int*>(workspace_bytes +
                                                   layout.ready_offset);
    auto kernel = hierarchical_homogeneous_two_level_kernel<
        Word, FirstLogN, SecondLogN, RowsPerBlock>;
    constexpr std::uint32_t kLargerLog =
        FirstLogN > SecondLogN ? FirstLogN : SecondLogN;
    constexpr std::size_t kSharedWords =
        RowsPerBlock * ((1U << kLargerLog) + 1U);
    const std::size_t shared_bytes = kSharedWords * sizeof(Word);
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, 256, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = resident_per_sm *
                         static_cast<std::uint32_t>(multiprocessors);
    grid = std::max(grid, 2U);
    const std::uint32_t producer_weight =
        config.subgraph_mappings[0].cta_weight;
    const std::uint32_t consumer_weight =
        config.subgraph_mappings[1].cta_weight;
    const std::uint32_t total_weight = producer_weight + consumer_weight;
    std::uint32_t producer_blocks = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(grid) * producer_weight +
         total_weight / 2U) /
        total_weight);
    producer_blocks = std::max(1U, std::min(grid - 1U, producer_blocks));

    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundary;
    unsigned int* ready_arg = ready;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    unsigned long long* trace_arg = trace;
    std::uint32_t producer_blocks_arg = producer_blocks;
    std::uint32_t producer_data_time =
        config.subgraph_mappings[0].data_time;
    std::uint32_t consumer_data_time =
        config.subgraph_mappings[1].data_time;
    void* arguments[] = {
        &input_arg,          &output_arg,         &boundary_arg,
        &ready_arg,          &transforms,         &roots_arg,
        &roots_shoup_arg,    &trace_arg,          &producer_blocks_arg,
        &producer_data_time, &consumer_data_time, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(256), arguments,
        shared_bytes, active_stream));
}

template <typename Word, std::uint32_t FirstLogN, std::uint32_t SecondLogN,
          std::uint32_t WarpsPerCta>
void launch_homogeneous_warp_hierarchical_streaming_two_level(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, NttStageOperatorT<Word> op,
    int multiprocessors) {
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundary = static_cast<Word*>(workspace);
    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* ready = reinterpret_cast<unsigned int*>(workspace_bytes +
                                                   layout.ready_offset);
    auto kernel = hierarchical_homogeneous_warp_two_level_kernel<
        Word, FirstLogN, SecondLogN, WarpsPerCta>;
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, 256, 0));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = resident_per_sm *
                         static_cast<std::uint32_t>(multiprocessors);
    grid = std::max(grid, 2U);
    const std::uint32_t producer_weight =
        config.subgraph_mappings[0].cta_weight;
    const std::uint32_t consumer_weight =
        config.subgraph_mappings[1].cta_weight;
    const std::uint32_t total_weight = producer_weight + consumer_weight;
    std::uint32_t producer_blocks = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(grid) * producer_weight +
         total_weight / 2U) /
        total_weight);
    producer_blocks = std::max(1U, std::min(grid - 1U, producer_blocks));

    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundary;
    unsigned int* ready_arg = ready;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    unsigned long long* trace_arg = trace;
    std::uint32_t producer_blocks_arg = producer_blocks;
    std::uint32_t producer_data_time =
        config.subgraph_mappings[0].data_time;
    std::uint32_t consumer_data_time =
        config.subgraph_mappings[1].data_time;
    void* arguments[] = {
        &input_arg,          &output_arg,         &boundary_arg,
        &ready_arg,          &transforms,         &roots_arg,
        &roots_shoup_arg,    &trace_arg,          &producer_blocks_arg,
        &producer_data_time, &consumer_data_time, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(256), arguments, 0,
        active_stream));
}

template <typename Word, std::uint32_t Threads, std::uint32_t WarpGroups,
          bool StaticWriter = false, bool StaticReader = false,
          std::uint32_t WriterRows = 0, std::uint32_t WarpTileLog = 8,
          bool WarpPipeline = false, bool WarpCooperative = false,
          std::uint32_t CooperativeMinBlocksPerSm = 4,
          std::uint32_t CoefficientReuseStages = 0,
          bool VectorRadix4Prefix = false, bool PackedStage6 = false,
          bool DistributedPackedStage6 = false,
          bool PacketSharedRadix4 = false>
void launch_homogeneous_warp256_hierarchical_streaming_10x10(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* root_powers, const Word* root_powers_shoup,
    unsigned long long* trace, NttStageOperatorT<Word> op,
    int multiprocessors) {
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundary = static_cast<Word*>(workspace);
    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* ready = reinterpret_cast<unsigned int*>(workspace_bytes +
                                                   layout.ready_offset);
    static_assert(Threads == WarpGroups * 128U);
    auto kernel = [] {
        if constexpr (WarpPipeline) {
            return hierarchical_homogeneous_warp_small_pipeline_10x10_kernel<
                Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                WarpTileLog>;
        } else if constexpr (WarpCooperative) {
            static_assert(CooperativeMinBlocksPerSm == 3 ||
                          CooperativeMinBlocksPerSm == 4);
            if constexpr (CooperativeMinBlocksPerSm == 3) {
                return hierarchical_homogeneous_warp128_cooperative_occupancy3_10x10_kernel<
                    Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                    WarpTileLog>;
            } else {
                return hierarchical_homogeneous_warp128_cooperative_occupancy4_10x10_kernel<
                    Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                    WarpTileLog>;
            }
        } else if constexpr (WarpTileLog < 8) {
            if constexpr (Threads == 256) {
                return hierarchical_homogeneous_warp_small_occupancy2_10x10_kernel<
                    Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                    WarpTileLog, CoefficientReuseStages, VectorRadix4Prefix,
                    PackedStage6, DistributedPackedStage6,
                    PacketSharedRadix4>;
            } else {
                return hierarchical_homogeneous_warp_small_occupancy4_10x10_kernel<
                    Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                    WarpTileLog, CoefficientReuseStages, VectorRadix4Prefix,
                    PackedStage6, DistributedPackedStage6,
                    PacketSharedRadix4>;
            }
        } else {
            return hierarchical_homogeneous_warp256_10x10_kernel<
                Word, WarpGroups, StaticWriter, StaticReader, WriterRows,
                WarpTileLog, CoefficientReuseStages, VectorRadix4Prefix,
                PackedStage6, DistributedPackedStage6,
                PacketSharedRadix4>;
        }
    }();
    constexpr std::uint32_t kWriterRows =
        StaticWriter ? (WriterRows == 0 ? 32U / sizeof(Word) : WriterRows)
                     : 0U;
    constexpr bool kRowMajorIo =
        StaticReader && sizeof(Word) == 4 && kWriterRows == 4;
    constexpr std::uint32_t kWriterStride = kRowMajorIo
        ? 1024U + (kWriterRows == 2 ? 15U : 7U) : 1024U;
    constexpr std::uint32_t kExchangeBuffers = 1U;
    constexpr std::uint32_t kExchangeStride = 1024U;
    constexpr std::uint32_t kPipelineFlagWords = WarpPipeline ? 8U : 0U;
    constexpr std::size_t kSharedWords =
        WarpGroups *
            (kExchangeBuffers * kExchangeStride +
             kWriterRows * kWriterStride) +
        kPipelineFlagWords;
    const std::size_t shared_bytes = kSharedWords * sizeof(Word);
    if (shared_bytes > 48U * 1024U) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, Threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = resident_per_sm *
                         static_cast<std::uint32_t>(multiprocessors);
    grid = std::max(grid, 2U);
    const std::uint32_t producer_weight =
        config.subgraph_mappings[0].cta_weight;
    const std::uint32_t consumer_weight =
        config.subgraph_mappings[1].cta_weight;
    const std::uint32_t total_weight = producer_weight + consumer_weight;
    std::uint32_t producer_blocks = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(grid) * producer_weight +
         total_weight / 2U) /
        total_weight);
    producer_blocks = std::max(1U, std::min(grid - 1U, producer_blocks));

    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundary;
    unsigned int* ready_arg = ready;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    unsigned long long* trace_arg = trace;
    std::uint32_t producer_blocks_arg = producer_blocks;
    std::uint32_t producer_data_time =
        config.subgraph_mappings[0].data_time;
    std::uint32_t consumer_data_time =
        config.subgraph_mappings[1].data_time;
    std::uint32_t packet_wave_q =
        PacketSharedRadix4
            ? config.subgraph_mappings[1].token_interleave
            : 0U;
    std::uint32_t packet_poll_sleep =
        PacketSharedRadix4 ? config.ready_window * 32U : 0U;
    std::uint32_t packet_readiness_mode =
        PacketSharedRadix4 &&
                config.packet_readiness_mode == PacketReadinessMode::WaveBitmap
            ? 1U
            : 0U;
    std::uint32_t packet_compute_layout =
        PacketSharedRadix4 &&
                config.packet_compute_layout == PacketComputeLayout::WarpRows
            ? 1U
            : 0U;
    std::uint32_t packet_fold_wave_barriers =
        PacketSharedRadix4 && config.packet_fold_wave_barriers ? 1U : 0U;
    void* arguments[] = {
        &input_arg,          &output_arg,         &boundary_arg,
        &ready_arg,          &transforms,         &roots_arg,
        &roots_shoup_arg,    &trace_arg,          &producer_blocks_arg,
        &producer_data_time, &consumer_data_time, &packet_wave_q,
        &packet_poll_sleep,  &packet_readiness_mode,
        &packet_compute_layout, &packet_fold_wave_barriers, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(Threads), arguments,
        shared_bytes, active_stream));
}

template <typename Word>
bool launch_specialized_hierarchical_streaming(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    constexpr std::uint32_t kRowsPerBlock = 4;
    const bool specialized_three_level =
        uses_specialized_hierarchical_streaming_three_level(config);
    const bool homogeneous_10x10 =
        uses_homogeneous_hierarchical_streaming_10x10(config);
    const bool homogeneous_warp_10x10 =
        uses_homogeneous_warp_hierarchical_streaming_10x10(config);
    const bool homogeneous_warp256_10x10 =
        uses_homogeneous_warp256_hierarchical_streaming_10x10(config);
    const bool homogeneous_warp256_static_10x10 =
        uses_homogeneous_warp256_static_hierarchical_streaming_10x10(config);
    const bool homogeneous_warp256_static_io_10x10 =
        uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
            config);
    const bool homogeneous_warp256_coefficient_reuse_static_io_10x10 =
        uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::
                        HomogeneousWarp256CoefficientReuseStaticIoRadix2);
    const bool homogeneous_warp128_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::HomogeneousWarp128StaticIoRadix2);
    const bool homogeneous_warp128_coefficient_reuse_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::
                        HomogeneousWarp128CoefficientReuseStaticIoRadix2);
    const bool homogeneous_warp128_vector_radix4_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config,
            NttSubgraphCore::HomogeneousWarp128VectorRadix4StaticIo);
    const bool homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::
                        HomogeneousWarp128VectorRadix4PackedStage6StaticIo);
    const bool
        homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10 =
            uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                config, NttSubgraphCore::
                            HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo);
    const bool homogeneous_warp128_packet_shared_radix4_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::
                        HomogeneousWarp128PacketSharedRadix4StaticIo);
    const bool homogeneous_warp64_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config, NttSubgraphCore::HomogeneousWarp64StaticIoRadix2);
    const bool homogeneous_warp128_pipeline_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config,
            NttSubgraphCore::HomogeneousWarp128PipelineStaticIoRadix2) &&
        config.pipeline_buffers == 1;
    const bool homogeneous_warp128_cooperative_static_io_10x10 =
        uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
            config,
            NttSubgraphCore::HomogeneousWarp128CooperativeStaticIoRadix2);
    if (!specialized_three_level &&
        !uses_specialized_hierarchical_streaming_10x10(config) &&
        !homogeneous_10x10 && !homogeneous_warp_10x10 &&
        !homogeneous_warp256_10x10 &&
        !homogeneous_warp256_static_10x10 &&
        !homogeneous_warp256_static_io_10x10 &&
        !homogeneous_warp256_coefficient_reuse_static_io_10x10 &&
        !homogeneous_warp128_static_io_10x10 &&
        !homogeneous_warp128_coefficient_reuse_static_io_10x10 &&
        !homogeneous_warp128_vector_radix4_static_io_10x10 &&
        !homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10 &&
        !homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10 &&
        !homogeneous_warp128_packet_shared_radix4_static_io_10x10 &&
        !homogeneous_warp64_static_io_10x10 &&
        !homogeneous_warp128_pipeline_static_io_10x10 &&
        !homogeneous_warp128_cooperative_static_io_10x10) {
        return false;
    }

    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundary = static_cast<Word*>(workspace);
    auto* ready = reinterpret_cast<unsigned int*>(workspace_bytes + layout.ready_offset);
    auto* trace = reinterpret_cast<unsigned long long*>(workspace_bytes + layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n) : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup) : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup, static_cast<Word>(config.modulus),
                               scale, scale_shoup};
    if (specialized_three_level) {
        if (config.stage_partition == std::vector<std::uint32_t>{6, 7, 7}) {
            return launch_specialized_hierarchical_streaming_three_level_rows<
                Word, 6, 7, 7>(config, input, output, workspace, root_powers,
                               root_powers_shoup, trace, op, multiprocessors);
        }
        if (config.stage_partition == std::vector<std::uint32_t>{7, 6, 7}) {
            return launch_specialized_hierarchical_streaming_three_level_rows<
                Word, 7, 6, 7>(config, input, output, workspace, root_powers,
                               root_powers_shoup, trace, op, multiprocessors);
        }
        if (config.stage_partition == std::vector<std::uint32_t>{7, 7, 6}) {
            return launch_specialized_hierarchical_streaming_three_level_rows<
                Word, 7, 7, 6>(config, input, output, workspace, root_powers,
                               root_powers_shoup, trace, op, multiprocessors);
        }
        if (config.stage_partition == std::vector<std::uint32_t>{6, 6, 8}) {
            return launch_specialized_hierarchical_streaming_three_level_rows<
                Word, 6, 6, 8>(config, input, output, workspace, root_powers,
                               root_powers_shoup, trace, op, multiprocessors);
        }
        if (config.stage_partition == std::vector<std::uint32_t>{6, 8, 6}) {
            return launch_specialized_hierarchical_streaming_three_level_rows<
                Word, 6, 8, 6>(config, input, output, workspace, root_powers,
                               root_powers_shoup, trace, op, multiprocessors);
        }
        return launch_specialized_hierarchical_streaming_three_level_rows<
            Word, 8, 6, 6>(config, input, output, workspace, root_powers,
                           root_powers_shoup, trace, op, multiprocessors);
    }
    if (homogeneous_10x10 &&
        (config.subgraph_mappings[0].data_time != 1U ||
         config.subgraph_mappings[1].data_time != 1U)) {
        launch_homogeneous_hierarchical_streaming_two_level<Word, 10, 10,
                                                            kRowsPerBlock>(
            config, input, output, workspace, root_powers,
            root_powers_shoup, trace, op, multiprocessors);
        return true;
    }
    if (homogeneous_warp_10x10) {
        launch_homogeneous_warp_hierarchical_streaming_two_level<
            Word, 10, 10, 8>(config, input, output, workspace, root_powers,
                             root_powers_shoup, trace, op, multiprocessors);
        return true;
    }
    if (homogeneous_warp256_10x10) {
        if (config.subgraph_mappings[0].threads_per_block == 128) {
            launch_homogeneous_warp256_hierarchical_streaming_10x10<Word, 128,
                                                                    1>(
                config, input, output, workspace, root_powers,
                root_powers_shoup, trace, op, multiprocessors);
        } else {
            launch_homogeneous_warp256_hierarchical_streaming_10x10<Word, 256,
                                                                    2>(
                config, input, output, workspace, root_powers,
                root_powers_shoup, trace, op, multiprocessors);
        }
        return true;
    }
    if (homogeneous_warp256_static_10x10) {
        constexpr std::uint32_t kHalfWriterRows = 16U / sizeof(Word);
        const bool half_packet =
            config.subgraph_mappings[0].data_space == kHalfWriterRows;
        if (config.subgraph_mappings[0].threads_per_block == 128) {
            if (half_packet) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, false, kHalfWriterRows>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            }
        } else {
            if (half_packet) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true, false, kHalfWriterRows>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            }
        }
        return true;
    }
    if (homogeneous_warp256_static_io_10x10 ||
        homogeneous_warp256_coefficient_reuse_static_io_10x10) {
        constexpr std::uint32_t kHalfWriterRows = 16U / sizeof(Word);
        const bool half_packet =
            config.subgraph_mappings[0].data_space == kHalfWriterRows;
        if (homogeneous_warp256_coefficient_reuse_static_io_10x10) {
            const auto launch_reuse = [&](auto reuse_stages) {
                constexpr std::uint32_t kReuseStages =
                    decltype(reuse_stages)::value;
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, true, kHalfWriterRows, 8, false,
                    false, 4, kReuseStages>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            };
            switch (config.subgraph_mappings[0].coefficient_reuse_stages) {
                case 1:
                    launch_reuse(std::integral_constant<std::uint32_t, 1>{});
                    break;
                case 2:
                    launch_reuse(std::integral_constant<std::uint32_t, 2>{});
                    break;
                case 3:
                    launch_reuse(std::integral_constant<std::uint32_t, 3>{});
                    break;
                case 4:
                    launch_reuse(std::integral_constant<std::uint32_t, 4>{});
                    break;
                case 5:
                    launch_reuse(std::integral_constant<std::uint32_t, 5>{});
                    break;
                case 6:
                    launch_reuse(std::integral_constant<std::uint32_t, 6>{});
                    break;
                default:
                    return false;
            }
            return true;
        }
        if (config.subgraph_mappings[0].threads_per_block == 128) {
            if (half_packet) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, true, kHalfWriterRows>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            }
        } else {
            if (half_packet) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true, true, kHalfWriterRows>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            }
        }
        return true;
    }
    if (homogeneous_warp128_static_io_10x10 ||
        homogeneous_warp128_coefficient_reuse_static_io_10x10 ||
        homogeneous_warp128_vector_radix4_static_io_10x10 ||
        homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10 ||
        homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10 ||
        homogeneous_warp128_packet_shared_radix4_static_io_10x10 ||
        homogeneous_warp128_pipeline_static_io_10x10 ||
        homogeneous_warp128_cooperative_static_io_10x10 ||
        homogeneous_warp64_static_io_10x10) {
        if constexpr (sizeof(Word) != 4) {
            return false;
        } else {
            if (homogeneous_warp128_packet_shared_radix4_static_io_10x10) {
                if (config.subgraph_mappings[0].threads_per_block == 128) {
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, 128, 1, true, true, 4, 7, false, false, 4, 0,
                        false, false, false, true>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                } else {
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, 256, 2, true, true, 4, 7, false, false, 4, 0,
                        false, false, false, true>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                }
            } else if (homogeneous_warp128_pipeline_static_io_10x10) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, true, 4, 7, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else if (homogeneous_warp128_cooperative_static_io_10x10) {
                if (config.target_ctas_per_sm <= 3U) {
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, 128, 1, true, true, 4, 7, false, true, 3>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                } else {
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, 128, 1, true, true, 4, 7, false, true, 4>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                }
            } else if (
                homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true, true, 4, 7, false, false, 4, 6,
                    true, false, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else if (
                homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10) {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 256, 2, true, true, 4, 7, false, false, 4, 6,
                    true, true>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            } else if (homogeneous_warp128_vector_radix4_static_io_10x10) {
                const auto launch_vector = [&](auto reuse_stages) {
                    constexpr std::uint32_t kReuseStages =
                        decltype(reuse_stages)::value;
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, 256, 2, true, true, 4, 7, false, false, 4,
                        kReuseStages, true>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                };
                switch (config.subgraph_mappings[0]
                            .coefficient_reuse_stages) {
                    case 0:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 0>{});
                        break;
                    case 3:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 3>{});
                        break;
                    case 4:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 4>{});
                        break;
                    case 5:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 5>{});
                        break;
                    case 6:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 6>{});
                        break;
                    case 7:
                        launch_vector(
                            std::integral_constant<std::uint32_t, 7>{});
                        break;
                    default:
                        return false;
                }
            } else if (homogeneous_warp128_static_io_10x10 ||
                       homogeneous_warp128_coefficient_reuse_static_io_10x10) {
                const bool coefficient_reuse =
                    homogeneous_warp128_coefficient_reuse_static_io_10x10;
                const auto launch_reuse = [&](auto threads, auto groups,
                                              auto reuse_stages) {
                    constexpr std::uint32_t kThreads = decltype(threads)::value;
                    constexpr std::uint32_t kGroups = decltype(groups)::value;
                    constexpr std::uint32_t kReuseStages =
                        decltype(reuse_stages)::value;
                    launch_homogeneous_warp256_hierarchical_streaming_10x10<
                        Word, kThreads, kGroups, true, true, 4, 7, false,
                        false, 4, kReuseStages>(
                        config, input, output, workspace, root_powers,
                        root_powers_shoup, trace, op, multiprocessors);
                };
                const auto dispatch_reuse = [&](auto threads, auto groups) {
                    switch (config.subgraph_mappings[0]
                                .coefficient_reuse_stages) {
                        case 1:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 1>{});
                            return true;
                        case 2:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 2>{});
                            return true;
                        case 3:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 3>{});
                            return true;
                        case 4:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 4>{});
                            return true;
                        case 5:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 5>{});
                            return true;
                        case 6:
                            launch_reuse(
                                threads, groups,
                                std::integral_constant<std::uint32_t, 6>{});
                            return true;
                        default:
                            return false;
                    }
                };
                if (config.subgraph_mappings[0].threads_per_block == 256U) {
                    if (coefficient_reuse) {
                        if (!dispatch_reuse(
                                std::integral_constant<std::uint32_t, 256>{},
                                std::integral_constant<std::uint32_t, 2>{})) {
                            return false;
                        }
                    } else {
                        launch_homogeneous_warp256_hierarchical_streaming_10x10<
                            Word, 256, 2, true, true, 4, 7>(
                            config, input, output, workspace, root_powers,
                            root_powers_shoup, trace, op, multiprocessors);
                    }
                } else {
                    if (coefficient_reuse) {
                        if (!dispatch_reuse(
                                std::integral_constant<std::uint32_t, 128>{},
                                std::integral_constant<std::uint32_t, 1>{})) {
                            return false;
                        }
                    } else {
                        launch_homogeneous_warp256_hierarchical_streaming_10x10<
                            Word, 128, 1, true, true, 4, 7>(
                            config, input, output, workspace, root_powers,
                            root_powers_shoup, trace, op, multiprocessors);
                    }
                }
            } else {
                launch_homogeneous_warp256_hierarchical_streaming_10x10<
                    Word, 128, 1, true, true, 4, 6>(
                    config, input, output, workspace, root_powers,
                    root_powers_shoup, trace, op, multiprocessors);
            }
            return true;
        }
    }
    // The homogeneous template's D_t=1 point is exactly the mature 10+10
    // physical codelet. Selecting it directly removes the runtime packet-loop
    // overhead without changing the logical schedule.
    auto kernel = hierarchical_streaming_10x10_kernel<Word, kRowsPerBlock>;
    constexpr std::size_t kSharedWords = kRowsPerBlock * ((1U << 10) + 1U);
    const std::size_t shared_bytes = kSharedWords * sizeof(Word);
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, 256, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = resident_per_sm * static_cast<std::uint32_t>(multiprocessors);
    grid = std::max(grid, 2U);
    const std::uint32_t producer_weight = config.subgraph_mappings[0].cta_weight;
    const std::uint32_t consumer_weight = config.subgraph_mappings[1].cta_weight;
    const std::uint32_t total_weight = producer_weight + consumer_weight;
    std::uint32_t producer_blocks = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(grid) * producer_weight + total_weight / 2U) /
        total_weight);
    producer_blocks = std::max(1U, std::min(grid - 1U, producer_blocks));

    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundary;
    unsigned int* ready_arg = ready;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    unsigned long long* trace_arg = trace;
    std::uint32_t producer_blocks_arg = producer_blocks;
    void* arguments[] = {&input_arg, &output_arg, &boundary_arg, &ready_arg, &transforms,
                         &roots_arg, &roots_shoup_arg, &trace_arg,
                         &producer_blocks_arg, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(256), arguments,
        shared_bytes, active_stream));
    return true;
}

template <typename Word, std::uint32_t StageSpace, std::uint32_t RoleStages,
          std::uint32_t StaticLogN = 0>
void launch_appt_pipeline_impl(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    const auto& mapping = config.subgraph_mappings.front();
    const std::uint32_t buffers = config.boundary_mappings.empty()
                                      ? 2U : config.boundary_mappings.front().buffers;
    constexpr std::uint32_t kRoles = (StageSpace + RoleStages - 1) / RoleStages;
    const std::uint32_t replicas = mapping.units_per_cta / kRoles;
    auto kernel = appt_pipeline_kernel<Word, StageSpace, RoleStages, StaticLogN>;
    const std::uint32_t threads = mapping.units_per_cta * 32;
    const std::size_t shared_bytes =
        std::max(
            static_cast<std::size_t>(kRoles - 1) * replicas * buffers *
                mapping.token_interleave * (1U << StageSpace) * sizeof(Word) +
                static_cast<std::size_t>(kRoles - 1) * buffers * sizeof(int),
            static_cast<std::size_t>(32 * 33) * sizeof(Word));
    if (shared_bytes > 48 * 1024) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    if (resident_per_sm == 0) {
        throw std::runtime_error("appt-pipeline has zero cooperative residency");
    }
    const std::uint32_t grid = resident_per_sm *
                               static_cast<std::uint32_t>(multiprocessors);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n) : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup) : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup,
                               static_cast<Word>(config.modulus), scale, scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    Word* scratch_arg = static_cast<Word*>(workspace);
    std::uint64_t transforms = config.batch;
    std::uint32_t log_n = config.log_n;
    std::uint32_t data_time = mapping.data_time;
    std::uint32_t token_interleave = mapping.token_interleave;
    std::uint32_t pipeline_buffers = buffers;
    std::uint32_t pipeline_replicas = replicas;
    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* trace = reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(workspace) + layout.trace_offset);
    void* arguments[] = {&input_arg, &output_arg, &scratch_arg, &transforms,
                         &log_n, &data_time, &token_interleave, &pipeline_buffers,
                         &pipeline_replicas, &trace, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
}

template <typename Word, std::uint32_t StageSpace>
void launch_appt_pipeline_roles(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    const bool fused_roles = config.subgraph_mappings.front().role_stages == 2;
    if constexpr (StageSpace == 7) {
        if (config.log_n == 20) {
            if (fused_roles) {
                launch_appt_pipeline_impl<Word, StageSpace, 2, 20>(
                    config, input, output, workspace, twiddles, twiddles_shoup,
                    inverse_n, inverse_n_shoup, multiprocessors);
            } else {
                launch_appt_pipeline_impl<Word, StageSpace, 1, 20>(
                    config, input, output, workspace, twiddles, twiddles_shoup,
                    inverse_n, inverse_n_shoup, multiprocessors);
            }
            return;
        }
    }
    if (fused_roles) {
        launch_appt_pipeline_impl<Word, StageSpace, 2>(config, input, output,
            workspace, twiddles, twiddles_shoup, inverse_n,
            inverse_n_shoup, multiprocessors);
    } else {
        launch_appt_pipeline_impl<Word, StageSpace, 1>(config, input, output,
            workspace, twiddles, twiddles_shoup, inverse_n,
            inverse_n_shoup, multiprocessors);
    }
}

template <typename Word, std::uint32_t FragmentWidth, bool WarpRows,
          bool WarpColumns, bool Radix8Core, std::uint32_t ProducerAGroup,
          bool WriterFinal, bool TransformInterleave,
          bool Resident2D, bool ResidentQuarter, bool StaticInput,
          bool NaturalOutput>
void launch_appt_register_tail_impl(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t min_blocks_per_sm =
        Resident2D
            ? 1
            : ResidentQuarter
            ? 2
            : sizeof(Word) == sizeof(std::uint64_t) ? 2 : 3;
    constexpr bool aggregate_ready = sizeof(Word) == sizeof(std::uint32_t);
    auto kernel = appt_online_register_tail_log20_kernel<
        Word, threads, min_blocks_per_sm, aggregate_ready, WarpRows,
        WarpColumns, Radix8Core, ProducerAGroup, WriterFinal,
        TransformInterleave, Resident2D, ResidentQuarter, FragmentWidth,
        StaticInput,
        NaturalOutput>;
    const std::size_t shared_bytes =
        Resident2D || ResidentQuarter
            ? (Resident2D ? 96 * 1024 : 48 * 1024)
            : sizeof(Word) == sizeof(std::uint64_t) ? 48 * 1024 : 32 * 1024;
    if constexpr (Resident2D) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    const std::uint32_t grid = resident_per_sm *
                               static_cast<std::uint32_t>(multiprocessors);
    constexpr std::uint32_t required_roles = NaturalOutput ? 3 : 2;
    if (grid < required_roles) {
        throw std::runtime_error(
            "appt-online-register-tail has insufficient resident CTA roles");
    }

    const auto& roles = config.appt_role_mapping;
    const std::uint32_t producer_weight = roles.producer_weight;
    const std::uint32_t total_weight = producer_weight +
                                       roles.tail_weight +
                                       (NaturalOutput ? roles.writer_weight : 0U);
    std::uint32_t producer_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) *
               producer_weight / total_weight);
    producer_blocks = std::min(producer_blocks, grid - required_roles + 1);
    std::uint32_t tail_blocks = NaturalOutput
        ? std::max<std::uint32_t>(
              1, static_cast<std::uint64_t>(grid) * roles.tail_weight /
                     total_weight)
        : grid - producer_blocks;
    if constexpr (NaturalOutput) {
        tail_blocks = std::min(tail_blocks, grid - producer_blocks - 1);
    }

    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* state1 = static_cast<Word*>(workspace);
    auto* ready1 = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(workspace) + layout.ready_offset);
    auto* trace = reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(workspace) + layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n)
                                      : static_cast<Word>(1);
    const Word scale_shoup = config.inverse
                                 ? static_cast<Word>(inverse_n_shoup)
                                 : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup,
                               static_cast<Word>(config.modulus), scale,
                               scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    std::uint64_t transforms = config.batch;
    std::uint32_t writer_tiles_per_cta =
        config.appt_role_mapping.writer_tiles_per_cta;
    std::uint32_t data_time =
        config.subgraph_mappings.front().data_time;
    std::uint32_t data_time_role_mask =
        config.appt_role_mapping.data_time_role_mask;
    void* arguments[] = {&input_arg, &output_arg, &state1, &ready1,
                         &transforms, &producer_blocks, &tail_blocks,
                         &writer_tiles_per_cta, &data_time,
                         &data_time_role_mask, &trace, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
}

template <typename Word, std::uint32_t FragmentWidth, bool WarpRows,
          bool WarpColumns, bool Radix8Core,
          std::uint32_t ProducerAGroup, bool WriterFinal,
          bool TransformInterleave, bool Resident2D = false,
          bool ResidentQuarter = false>
void dispatch_appt_register_tail_layout(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    const bool static_input = config.input_order == InputOrder::ApptStatic;
    const bool natural_output = config.output_order == OutputOrder::Natural;
#define CUNTT_LAUNCH_APPT_REGISTER(STATIC_INPUT, NATURAL_OUTPUT)                 \
    launch_appt_register_tail_impl<Word, FragmentWidth, WarpRows, WarpColumns, \
                                   Radix8Core, ProducerAGroup, WriterFinal,    \
                                   TransformInterleave, Resident2D,           \
                                   ResidentQuarter,                           \
                                   STATIC_INPUT,                              \
                                   NATURAL_OUTPUT>(                            \
        config, input, output, workspace, twiddles, twiddles_shoup, inverse_n, \
        inverse_n_shoup, multiprocessors)
    if constexpr (WriterFinal) {
        if (!natural_output) {
            throw std::logic_error(
                "APPT writer-final core requires natural output");
        }
        if (static_input) {
            CUNTT_LAUNCH_APPT_REGISTER(true, true);
        } else {
            CUNTT_LAUNCH_APPT_REGISTER(false, true);
        }
    } else {
        if (static_input) {
            if (natural_output) {
                CUNTT_LAUNCH_APPT_REGISTER(true, true);
            } else {
                CUNTT_LAUNCH_APPT_REGISTER(true, false);
            }
        } else if (natural_output) {
            CUNTT_LAUNCH_APPT_REGISTER(false, true);
        } else {
            CUNTT_LAUNCH_APPT_REGISTER(false, false);
        }
    }
#undef CUNTT_LAUNCH_APPT_REGISTER
}

template <typename Word>
bool launch_appt_register_tail(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (!uses_appt_register_tail(config)) return false;
    const bool warp_core = uses_appt_register_tail_warp(config);
    const bool column_warp_core =
        uses_appt_register_tail_column_warp(config);
    const bool radix8_core = uses_appt_register_tail_radix8(config);
    const bool resident_core =
        uses_appt_register_tail_grouped_writer_final_resident(config);
    const bool resident_quarter_core =
        uses_appt_register_tail_grouped_writer_final_resident_quarter(config);
    if (resident_quarter_core) {
#define CUNTT_DISPATCH_APPT_RESIDENT_QUARTER(FRAGMENT)                        \
        dispatch_appt_register_tail_layout<Word, FRAGMENT, false, false,      \
                                           false, 32, true, true, false,      \
                                           true>(                             \
            config, input, output, workspace, twiddles, twiddles_shoup,       \
            inverse_n, inverse_n_shoup, multiprocessors)
        switch (config.appt_role_mapping.fragment_width) {
            case 8: CUNTT_DISPATCH_APPT_RESIDENT_QUARTER(8); break;
            case 16: CUNTT_DISPATCH_APPT_RESIDENT_QUARTER(16); break;
            case 32: CUNTT_DISPATCH_APPT_RESIDENT_QUARTER(32); break;
            default:
                throw std::logic_error(
                    "unresolved APPT resident-quarter fragment width");
        }
#undef CUNTT_DISPATCH_APPT_RESIDENT_QUARTER
        return true;
    }
    if (resident_core) {
#define CUNTT_DISPATCH_APPT_RESIDENT(FRAGMENT)                                \
        dispatch_appt_register_tail_layout<Word, FRAGMENT, false, false,      \
                                           false, 32, true, true, true>(      \
            config, input, output, workspace, twiddles, twiddles_shoup,       \
            inverse_n, inverse_n_shoup, multiprocessors)
        switch (config.appt_role_mapping.fragment_width) {
            case 8: CUNTT_DISPATCH_APPT_RESIDENT(8); break;
            case 16: CUNTT_DISPATCH_APPT_RESIDENT(16); break;
            case 32: CUNTT_DISPATCH_APPT_RESIDENT(32); break;
            default:
                throw std::logic_error("unresolved APPT resident fragment width");
        }
#undef CUNTT_DISPATCH_APPT_RESIDENT
        return true;
    }
#define CUNTT_DISPATCH_APPT_GROUPED(FRAGMENT, A_GROUP, WRITER_FINAL, DATA_TIME) \
    dispatch_appt_register_tail_layout<Word, FRAGMENT, false, false, false,    \
                                       A_GROUP, WRITER_FINAL, DATA_TIME>(      \
        config, input, output, workspace, twiddles, twiddles_shoup, inverse_n, \
        inverse_n_shoup, multiprocessors)
#define CUNTT_DISPATCH_APPT_GROUPED_WIDTH(FRAGMENT, WRITER_FINAL, DATA_TIME)   \
    do {                                                                       \
        switch (config.subgraph_mappings.front().data_space) {                 \
            case 8:                                                            \
                CUNTT_DISPATCH_APPT_GROUPED(FRAGMENT, 8, WRITER_FINAL, DATA_TIME); \
                break;                                                         \
            case 16:                                                           \
                CUNTT_DISPATCH_APPT_GROUPED(FRAGMENT, 16, WRITER_FINAL, DATA_TIME); \
                break;                                                         \
            case 32:                                                           \
                CUNTT_DISPATCH_APPT_GROUPED(FRAGMENT, 32, WRITER_FINAL, DATA_TIME); \
                break;                                                         \
            default:                                                           \
                throw std::logic_error("unresolved APPT producer group");     \
        }                                                                      \
    } while (false)
#define CUNTT_DISPATCH_APPT_FRAGMENT(FRAGMENT)                                 \
    do {                                                                       \
        if (warp_core) {                                                       \
            dispatch_appt_register_tail_layout<Word, FRAGMENT, true, true,     \
                                               false, 1, false, false>(        \
                config, input, output, workspace, twiddles, twiddles_shoup,    \
                inverse_n, inverse_n_shoup, multiprocessors);                  \
        } else if (column_warp_core) {                                         \
            dispatch_appt_register_tail_layout<Word, FRAGMENT, false, true,    \
                                               false, 1, false, false>(        \
                config, input, output, workspace, twiddles, twiddles_shoup,    \
                inverse_n, inverse_n_shoup, multiprocessors);                  \
        } else if (radix8_core) {                                              \
            dispatch_appt_register_tail_layout<Word, FRAGMENT, false, false,   \
                                               true, 1, false, false>(         \
                config, input, output, workspace, twiddles, twiddles_shoup,    \
                inverse_n, inverse_n_shoup, multiprocessors);                  \
        } else {                                                               \
            dispatch_appt_register_tail_layout<Word, FRAGMENT, false, false,   \
                                               false, 1, false, false>(        \
                config, input, output, workspace, twiddles, twiddles_shoup,    \
                inverse_n, inverse_n_shoup, multiprocessors);                  \
        }                                                                      \
    } while (false)
    const bool grouped_core = uses_appt_register_tail_grouped(config);
    const bool writer_final_core =
        uses_appt_register_tail_grouped_writer_final(config);
    const bool data_time_core =
        uses_appt_register_tail_grouped_writer_final_data_time(config);
#define CUNTT_DISPATCH_SELECTED_GROUPED(FRAGMENT)                              \
    do {                                                                       \
        if (data_time_core) {                                                  \
            CUNTT_DISPATCH_APPT_GROUPED_WIDTH(FRAGMENT, true, true);          \
        } else if (writer_final_core) {                                        \
            CUNTT_DISPATCH_APPT_GROUPED_WIDTH(FRAGMENT, true, false);         \
        } else {                                                               \
            CUNTT_DISPATCH_APPT_GROUPED_WIDTH(FRAGMENT, false, false);        \
        }                                                                      \
    } while (false)
    switch (config.appt_role_mapping.fragment_width) {
        case 8:
            if (grouped_core) CUNTT_DISPATCH_SELECTED_GROUPED(8);
            else CUNTT_DISPATCH_APPT_FRAGMENT(8);
            break;
        case 16:
            if (grouped_core) CUNTT_DISPATCH_SELECTED_GROUPED(16);
            else CUNTT_DISPATCH_APPT_FRAGMENT(16);
            break;
        case 32:
            if (grouped_core) CUNTT_DISPATCH_SELECTED_GROUPED(32);
            else CUNTT_DISPATCH_APPT_FRAGMENT(32);
            break;
        default:
            throw std::logic_error("unresolved APPT fragment width");
    }
#undef CUNTT_DISPATCH_APPT_FRAGMENT
#undef CUNTT_DISPATCH_SELECTED_GROUPED
#undef CUNTT_DISPATCH_APPT_GROUPED_WIDTH
#undef CUNTT_DISPATCH_APPT_GROUPED
    return true;
}

template <typename Word>
bool launch_appt_split_tail(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (!uses_appt_split_tail(config)) return false;
    auto kernel = appt_online_split_tail_log20_kernel<Word>;
    constexpr std::uint32_t threads = 256;
    const std::size_t shared_bytes = 32 * 129 * sizeof(Word);
    if (shared_bytes > 48 * 1024) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
    }
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    const std::uint32_t grid = resident_per_sm *
                               static_cast<std::uint32_t>(multiprocessors);
    if (grid < 3) {
        throw std::runtime_error(
            "appt-online-split-tail requires three resident CTA roles");
    }

    const std::uint32_t weight0 = config.subgraph_mappings[0].cta_weight;
    const std::uint32_t weight1 = config.subgraph_mappings[1].cta_weight;
    const std::uint32_t weight2 = config.subgraph_mappings[2].cta_weight;
    const std::uint32_t total_weight = weight0 + weight1 + weight2;
    std::uint32_t role0_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) * weight0 / total_weight);
    std::uint32_t role1_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) * weight1 / total_weight);
    if (role0_blocks + role1_blocks >= grid) {
        role0_blocks = std::max<std::uint32_t>(1, grid / 3);
        role1_blocks = std::max<std::uint32_t>(1, grid / 3);
    }

    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* state1 = static_cast<Word*>(workspace);
    auto* low_half = state1 + config.batch * (1ULL << config.log_n);
    auto* ready1 = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(workspace) + layout.ready_offset);
    auto* ready_low = ready1 + config.batch * 64;
    auto* trace = reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(workspace) + layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n)
                                      : static_cast<Word>(1);
    const Word scale_shoup = config.inverse
                                 ? static_cast<Word>(inverse_n_shoup)
                                 : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup,
                               static_cast<Word>(config.modulus), scale,
                               scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    std::uint64_t transforms = config.batch;
    void* arguments[] = {&input_arg, &output_arg, &state1, &low_half,
                         &ready1, &ready_low, &transforms, &role0_blocks,
                         &role1_blocks, &trace, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
    return true;
}

template <typename Word>
bool launch_appt_fused_tail(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (!uses_appt_fused_tail(config)) return false;
    auto kernel = appt_online_fused_tail_log20_kernel<Word>;
    constexpr std::uint32_t threads = 256;
    const std::size_t shared_bytes = 64 * 129 * sizeof(Word);
    CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    const std::uint32_t grid = resident_per_sm *
                               static_cast<std::uint32_t>(multiprocessors);
    if (grid < 2) {
        throw std::runtime_error(
            "appt-online-fused-tail requires two resident CTA roles");
    }

    std::uint32_t total_weight = 0;
    for (const auto& mapping : config.subgraph_mappings) {
        total_weight += mapping.cta_weight;
    }
    std::uint32_t producer_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) *
               config.subgraph_mappings.front().cta_weight / total_weight);
    producer_blocks = std::min(producer_blocks, grid - 1);

    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* state1 = static_cast<Word*>(workspace);
    auto* ready1 = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(workspace) + layout.ready_offset);
    auto* trace = reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(workspace) + layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n)
                                      : static_cast<Word>(1);
    const Word scale_shoup = config.inverse
                                 ? static_cast<Word>(inverse_n_shoup)
                                 : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup,
                               static_cast<Word>(config.modulus), scale,
                               scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    std::uint64_t transforms = config.batch;
    void* arguments[] = {&input_arg, &output_arg, &state1, &ready1,
                         &transforms, &producer_blocks, &trace, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
    return true;
}

template <typename Word>
bool launch_appt_online(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (!uses_appt_online(config)) return false;
    if (launch_appt_register_tail(config, input, output, workspace, twiddles,
                                  twiddles_shoup, inverse_n, inverse_n_shoup,
                                  multiprocessors)) {
        return true;
    }
    if (launch_appt_split_tail(config, input, output, workspace, twiddles,
                               twiddles_shoup, inverse_n, inverse_n_shoup,
                               multiprocessors)) {
        return true;
    }
    if (launch_appt_fused_tail(config, input, output, workspace, twiddles,
                               twiddles_shoup, inverse_n, inverse_n_shoup,
                               multiprocessors)) {
        return true;
    }
    const auto micro_group = [](std::uint32_t data_space) {
        return data_space == 8 ? 2U : data_space == 32 ? 4U : 1U;
    };
    const std::uint32_t publication_group =
        micro_group(config.subgraph_mappings[1].data_space);
    const std::uint32_t output_group =
        micro_group(config.subgraph_mappings[2].data_space);
    const bool cta_radix4 = config.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRadix4;
    void* kernel = nullptr;
#define CUNTT_SELECT_APPT_ONLINE(PUBLISH, OUTPUT)                              \
    reinterpret_cast<void*>(appt_online_log20_kernel<Word, PUBLISH, OUTPUT>)
    if (cta_radix4) {
        kernel = config.profile_appt_roles
            ? reinterpret_cast<void*>(
                  appt_online_log20_kernel<Word, 1, 1, true, true>)
            : reinterpret_cast<void*>(
                  appt_online_log20_kernel<Word, 1, 1, true, false>);
    } else if (config.profile_appt_roles) {
        kernel = reinterpret_cast<void*>(
            appt_online_log20_kernel<Word, 1, 1, false, true>);
    } else if (publication_group == 1 && output_group == 1)
        kernel = CUNTT_SELECT_APPT_ONLINE(1, 1);
    else if (publication_group == 1 && output_group == 2)
        kernel = CUNTT_SELECT_APPT_ONLINE(1, 2);
    else if (publication_group == 1 && output_group == 4)
        kernel = CUNTT_SELECT_APPT_ONLINE(1, 4);
    else if (publication_group == 2 && output_group == 1)
        kernel = CUNTT_SELECT_APPT_ONLINE(2, 1);
    else if (publication_group == 2 && output_group == 2)
        kernel = CUNTT_SELECT_APPT_ONLINE(2, 2);
    else if (publication_group == 2 && output_group == 4)
        kernel = CUNTT_SELECT_APPT_ONLINE(2, 4);
    else if (publication_group == 4 && output_group == 1)
        kernel = CUNTT_SELECT_APPT_ONLINE(4, 1);
    else if (publication_group == 4 && output_group == 2)
        kernel = CUNTT_SELECT_APPT_ONLINE(4, 2);
    else
        kernel = CUNTT_SELECT_APPT_ONLINE(4, 4);
#undef CUNTT_SELECT_APPT_ONLINE
    constexpr std::uint32_t threads = 256;
    const std::size_t shared_bytes = 32 * 129 * sizeof(Word);
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, threads, shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    const std::uint32_t grid = resident_per_sm *
                               static_cast<std::uint32_t>(multiprocessors);
    if (grid < 3) {
        throw std::runtime_error("appt-online requires at least three resident CTAs");
    }

    const std::uint32_t weight0 = config.subgraph_mappings[0].cta_weight;
    const std::uint32_t weight1 = config.subgraph_mappings[1].cta_weight;
    const std::uint32_t weight2 = config.subgraph_mappings[2].cta_weight;
    const std::uint32_t total_weight = weight0 + weight1 + weight2;
    std::uint32_t role0_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) * weight0 / total_weight);
    std::uint32_t role1_blocks = std::max<std::uint32_t>(
        1, static_cast<std::uint64_t>(grid) * weight1 / total_weight);
    if (role0_blocks + role1_blocks >= grid) {
        role0_blocks = std::max<std::uint32_t>(1, grid / 3);
        role1_blocks = std::max<std::uint32_t>(1, grid / 3);
    }

    const auto layout = hierarchical_streaming_workspace_layout(config);
    auto* state1 = static_cast<Word*>(workspace);
    auto* state2 = state1 + config.batch * (1ULL << config.log_n);
    auto* ready1 = reinterpret_cast<unsigned int*>(
        static_cast<unsigned char*>(workspace) + layout.ready_offset);
    auto* ready2 = ready1 + config.batch * 64;
    auto* trace = reinterpret_cast<unsigned long long*>(
        static_cast<unsigned char*>(workspace) + layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n)
                                      : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup)
                                            : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup,
                               static_cast<Word>(config.modulus), scale,
                               scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    std::uint64_t transforms = config.batch;
    void* arguments[] = {&input_arg, &output_arg, &state1, &state2,
                         &ready1, &ready2, &transforms, &role0_blocks,
                         &role1_blocks, &trace, &op};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        kernel, dim3(grid), dim3(threads), arguments,
        shared_bytes, active_stream));
    return true;
}

template <typename Word>
bool launch_appt_pipeline(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (!uses_appt_pipeline(config)) return false;
    switch (config.stage_partition.front()) {
        case 4:
            launch_appt_pipeline_roles<Word, 4>(config, input, output, workspace,
                twiddles, twiddles_shoup, inverse_n, inverse_n_shoup, multiprocessors);
            break;
        case 5:
            launch_appt_pipeline_roles<Word, 5>(config, input, output, workspace,
                twiddles, twiddles_shoup, inverse_n, inverse_n_shoup, multiprocessors);
            break;
        case 6:
            launch_appt_pipeline_roles<Word, 6>(config, input, output, workspace,
                twiddles, twiddles_shoup, inverse_n, inverse_n_shoup, multiprocessors);
            break;
        case 7:
            launch_appt_pipeline_roles<Word, 7>(config, input, output, workspace,
                twiddles, twiddles_shoup, inverse_n, inverse_n_shoup, multiprocessors);
            break;
        case 8:
            launch_appt_pipeline_roles<Word, 8>(config, input, output, workspace,
                twiddles, twiddles_shoup, inverse_n, inverse_n_shoup, multiprocessors);
            break;
        default:
            throw std::logic_error("unresolved appt-pipeline stage-space");
    }
    return true;
}

template <typename Word>
void launch_hierarchical_streaming(
    const PlanConfig& config, const Word* input, Word* output, void* workspace,
    const Word* twiddles, const Word* twiddles_shoup, const Word* root_powers,
    const Word* root_powers_shoup, std::uint64_t inverse_n,
    std::uint64_t inverse_n_shoup, int multiprocessors) {
    if (launch_appt_online(config, input, output, workspace, twiddles,
                           twiddles_shoup, inverse_n, inverse_n_shoup,
                           multiprocessors)) {
        return;
    }
    if (launch_appt_pipeline(config, input, output, workspace, twiddles,
                             twiddles_shoup, inverse_n, inverse_n_shoup,
                             multiprocessors)) {
        return;
    }
    if (launch_specialized_hierarchical_streaming(
            config, input, output, workspace, twiddles, twiddles_shoup,
            root_powers, root_powers_shoup, inverse_n, inverse_n_shoup,
            multiprocessors)) {
        return;
    }
    NttStreamingDescriptor descriptor{};
    descriptor.count = static_cast<std::uint32_t>(config.stage_partition.size());
    descriptor.log_n = config.log_n;
    std::uint32_t prefix_log = 0;
    std::uint32_t input_shift = config.log_n;
    std::uint32_t output_shift = 0;
    std::uint32_t max_segment_log = 0;
    std::uint32_t launch_threads = 0;
    std::uint64_t counter_count = 0;
    std::uint64_t boundary_words = 0;
    std::uint64_t state_count = 0;
    descriptor.counter_base[0] = 0;
    for (std::uint32_t segment = 0; segment < descriptor.count; ++segment) {
        const std::uint32_t segment_log = config.stage_partition[segment];
        descriptor.segment_log[segment] = segment_log;
        descriptor.prefix_log[segment] = prefix_log;
        descriptor.remaining_log[segment] = config.log_n - prefix_log - segment_log;
        descriptor.core[segment] = static_cast<std::uint32_t>(config.subgraph_mappings[segment].core);
        input_shift -= segment_log;
        descriptor.digit_input_shift[segment] = input_shift;
        descriptor.digit_output_shift[segment] = output_shift;
        descriptor.digit_mask[segment] = (1U << segment_log) - 1U;
        output_shift += segment_log;
        prefix_log += segment_log;
        max_segment_log = std::max(max_segment_log, segment_log);
        launch_threads = std::max(launch_threads, config.subgraph_mappings[segment].threads_per_block);
        if (segment + 1 < descriptor.count) {
            const auto& boundary = config.boundary_mappings[segment];
            const std::uint32_t slots = static_cast<std::uint32_t>(
                boundary.storage == BoundaryStorage::FullScratch
                    ? config.batch
                    : std::min<std::size_t>(config.batch, boundary.buffers));
            const std::uint32_t producer_tasks_per_transform =
                1U << (descriptor.prefix_log[segment] + descriptor.remaining_log[segment]);
            const bool aggregate_readiness = slots == config.batch;
            const std::uint32_t next_remaining_log =
                descriptor.remaining_log[segment] - config.stage_partition[segment + 1];
            const std::uint32_t counter_tasks_per_transform = aggregate_readiness
                ? 1U << (descriptor.prefix_log[segment] + next_remaining_log)
                : producer_tasks_per_transform;
            descriptor.boundary_word_base[segment] = boundary_words;
            descriptor.boundary_slots[segment] = slots;
            descriptor.producer_tasks_per_transform[segment] = producer_tasks_per_transform;
            boundary_words += static_cast<std::uint64_t>(slots) << config.log_n;
            descriptor.counter_base[segment + 1] = counter_count;
            counter_count += static_cast<std::uint64_t>(slots) * counter_tasks_per_transform;
            descriptor.state_base[segment] = state_count;
            if (slots < config.batch) state_count += slots;
        }
    }
    descriptor.counter_base[descriptor.count] = counter_count;
    descriptor.state_base[descriptor.count] = state_count;

    const std::size_t shared_bytes = (1ULL << max_segment_log) * sizeof(Word) *
                                     (launch_threads / 32U);
    auto kernel = hierarchical_streaming_kernel<Word>;
    if (shared_bytes > 48U * 1024U) {
        CUNTT_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared_bytes)));
    }
    int occupancy = 0;
    CUNTT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, kernel, static_cast<int>(launch_threads), shared_bytes));
    const std::uint32_t resident_per_sm = std::min<std::uint32_t>(
        config.target_ctas_per_sm, static_cast<std::uint32_t>(occupancy));
    std::uint32_t grid = resident_per_sm * static_cast<std::uint32_t>(multiprocessors);
    grid = std::max(grid, descriptor.count);

    std::uint32_t total_weight = 0;
    for (const auto& mapping : config.subgraph_mappings) {
        total_weight += mapping.cta_weight;
    }
    descriptor.role_begin[0] = 0;
    std::uint32_t assigned = 0;
    const std::uint32_t distributable = grid - descriptor.count;
    for (std::uint32_t segment = 0; segment < descriptor.count; ++segment) {
        std::uint32_t blocks = 1;
        if (segment + 1 == descriptor.count) {
            blocks += grid - assigned - (descriptor.count - segment);
        } else if (total_weight != 0) {
            blocks += distributable * config.subgraph_mappings[segment].cta_weight / total_weight;
        }
        assigned += blocks;
        descriptor.role_blocks[segment] = blocks;
        descriptor.role_begin[segment + 1] = assigned;
    }

    const auto workspace_layout = hierarchical_streaming_workspace_layout(config);
    auto* workspace_bytes = static_cast<unsigned char*>(workspace);
    Word* boundaries = static_cast<Word*>(workspace);
    auto* ready = reinterpret_cast<unsigned int*>(workspace_bytes + workspace_layout.ready_offset);
    auto* consume_state = reinterpret_cast<unsigned long long*>(workspace_bytes + workspace_layout.state_offset);
    auto* consumed_epoch = reinterpret_cast<unsigned int*>(workspace_bytes + workspace_layout.completed_offset);
    auto* trace = reinterpret_cast<unsigned long long*>(workspace_bytes + workspace_layout.trace_offset);
    const Word scale = config.inverse ? static_cast<Word>(inverse_n) : static_cast<Word>(1);
    const Word scale_shoup = config.inverse ? static_cast<Word>(inverse_n_shoup) : static_cast<Word>(0);
    NttStageOperatorT<Word> op{twiddles, twiddles_shoup, static_cast<Word>(config.modulus), scale, scale_shoup};
    const Word* input_arg = input;
    Word* output_arg = output;
    Word* boundary_arg = boundaries;
    unsigned int* ready_arg = ready;
    unsigned long long* consume_state_arg = consume_state;
    unsigned int* consumed_epoch_arg = consumed_epoch;
    unsigned long long* trace_arg = trace;
    std::uint64_t transforms = config.batch;
    const Word* roots_arg = root_powers;
    const Word* roots_shoup_arg = root_powers_shoup;
    void* arguments[] = {&input_arg, &output_arg, &boundary_arg, &ready_arg, &transforms,
                         &twiddles, &twiddles_shoup, &roots_arg, &roots_shoup_arg,
                         &consume_state_arg, &consumed_epoch_arg, &trace_arg, &op, &descriptor};
    CUNTT_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(kernel), dim3(grid), dim3(launch_threads), arguments,
        shared_bytes, active_stream));
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
        lower_resident_execution_groups();
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
        if (config_.backend == Backend::HierarchicalDataflow) {
            workspace_bytes_ = hierarchical_streaming_workspace_bytes(config_);
        } else {
            workspace_bytes_ = (config_.backend == Backend::Hybrid2D ||
                                config_.backend == Backend::HierarchicalBarrier ||
                                (config_.backend == Backend::CompactStage &&
                                 config_.output_order == OutputOrder::Natural))
                                   ? data_bytes_
                                   : 0;
        }
        config_.auto_select = automatic_selection;
        logical_config_.auto_select = automatic_selection;
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
        if (config_.backend == Backend::Hybrid2D || config_.backend == Backend::HierarchicalBarrier ||
            config_.backend == Backend::HierarchicalDataflow) {
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

    const PlanConfig& config() const noexcept { return logical_config_; }
    const SelectionInfo& selection() const noexcept { return selection_; }

    std::size_t points_per_transform() const noexcept { return n_; }

    std::size_t data_size() const noexcept { return data_bytes_; }

    std::size_t workspace_size() const noexcept { return workspace_bytes_; }

    ApptLayoutInfo appt_layout_info() const {
        if (!uses_appt_register_tail(config_)) {
            throw std::logic_error(
                "plan does not expose an APPT static-layout contract");
        }
        ApptLayoutInfo info;
        info.log_n = config_.log_n;
        info.stage_partition = config_.stage_partition;
        info.fragment_width = config_.appt_role_mapping.fragment_width;
        std::uint32_t width = info.fragment_width;
        while (width > 1) {
            ++info.bank_bits;
            width >>= 1;
        }
        info.xor_permutation = true;
        info.compatibility_id =
            "appt-ntt-log20-7x7x6-xor-fw" +
            std::to_string(info.fragment_width) + "-v1";
        return info;
    }

    std::vector<PipelineSegmentTrace> pipeline_trace() const {
        if (config_.backend != Backend::HierarchicalDataflow) {
            return {};
        }
        const auto layout = hierarchical_streaming_workspace_layout(config_);
        const auto* trace_pointer = reinterpret_cast<const unsigned char*>(workspace()) +
                                    layout.trace_offset;
        std::vector<PipelineSegmentTrace> result(config_.stage_partition.size());
        CUNTT_CUDA_CHECK(cudaMemcpyAsync(result.data(), trace_pointer,
                                         result.size() * sizeof(PipelineSegmentTrace),
                                         cudaMemcpyDeviceToHost, stream_));
        CUNTT_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return result;
    }

    std::vector<PipelineRoleMetrics> pipeline_role_metrics() const {
        if (!uses_appt_online(config_) || !config_.profile_appt_roles) {
            return {};
        }
        const auto layout = hierarchical_streaming_workspace_layout(config_);
        const auto* metrics_pointer =
            reinterpret_cast<const unsigned char*>(workspace()) +
            layout.trace_offset + 6 * sizeof(unsigned long long);
        std::vector<PipelineRoleMetrics> result(3);
        CUNTT_CUDA_CHECK(cudaMemcpyAsync(
            result.data(), metrics_pointer,
            result.size() * sizeof(PipelineRoleMetrics), cudaMemcpyDeviceToHost,
            stream_));
        CUNTT_CUDA_CHECK(cudaStreamSynchronize(stream_));
        return result;
    }

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

        const auto timed_region_start = std::chrono::steady_clock::now();
        CUNTT_CUDA_CHECK(cudaEventRecord(start.get(), stream_));
        for (std::uint32_t i = 0; i < repeat; ++i) {
            launch_once(device_input_.data(), device_work_.data(), scratch);
        }
        CUNTT_CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
        CUNTT_CUDA_CHECK(cudaEventSynchronize(stop.get()));
        const auto timed_region_stop = std::chrono::steady_clock::now();
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
        stats.timed_region_wall_ms      =
            std::chrono::duration<double, std::milli>(
                timed_region_stop - timed_region_start).count() / repeat;
        stats.host_timing_overhead_ms   = std::max(
            0.0, stats.timed_region_wall_ms - stats.kernel_ms);
        stats.d2h_ms                    = d2h_ms;
        const double kernel_seconds     = stats.kernel_ms / 1000.0;
        const double end_to_end_seconds = (stats.h2d_ms + stats.kernel_ms + stats.d2h_ms) / 1000.0;
        stats.kernel_ntt_per_second     = config_.batch / kernel_seconds;
        stats.end_to_end_ntt_per_second = config_.batch / end_to_end_seconds;
        stats.kernel_points_per_second  = stats.kernel_ntt_per_second * n_;
        return stats;
    }

  private:
    void lower_resident_execution_groups() {
        logical_config_ = config_;
        if (config_.backend != Backend::HierarchicalDataflow) {
            return;
        }

        const bool has_resident_boundary = std::any_of(
            config_.boundary_mappings.begin(), config_.boundary_mappings.end(),
            [](const NttBoundaryMapping& boundary) {
                return boundary.storage == BoundaryStorage::ResidentFused;
            });
        if (!has_resident_boundary && config_.execution_group_mappings.empty()) {
            logical_config_.execution_group_mappings =
                config_.subgraph_mappings;
            return;
        }

        std::vector<std::uint32_t> execution_partition;
        std::vector<NttBoundaryMapping> execution_boundaries;
        std::vector<std::pair<std::size_t, std::size_t>> logical_groups;
        std::size_t group_begin = 0;
        std::uint32_t group_stages = config_.stage_partition.front();
        for (std::size_t boundary = 0;
             boundary < config_.boundary_mappings.size(); ++boundary) {
            if (config_.boundary_mappings[boundary].storage ==
                BoundaryStorage::ResidentFused) {
                group_stages += config_.stage_partition[boundary + 1];
                continue;
            }
            execution_partition.push_back(group_stages);
            execution_boundaries.push_back(config_.boundary_mappings[boundary]);
            logical_groups.emplace_back(group_begin, boundary + 1);
            group_begin = boundary + 1;
            group_stages = config_.stage_partition[boundary + 1];
        }
        execution_partition.push_back(group_stages);
        logical_groups.emplace_back(group_begin, config_.stage_partition.size());

        if (execution_partition.size() < 2) {
            throw std::invalid_argument(
                "fully resident-fused NTT requires a whole-transform physical core");
        }
        for (const auto stages : execution_partition) {
            if (stages > 10) {
                throw std::invalid_argument(
                    "resident-fused NTT execution groups currently support at most 10 stages");
            }
        }

        std::vector<NttSubgraphMapping> execution_mappings;
        if (!config_.execution_group_mappings.empty()) {
            if (config_.execution_group_mappings.size() !=
                execution_partition.size()) {
                throw std::invalid_argument(
                    "execution_group_mappings count does not match resident-fused lowering");
            }
            execution_mappings = config_.execution_group_mappings;
        } else {
            execution_mappings.reserve(logical_groups.size());
            for (const auto& group : logical_groups) {
                NttSubgraphMapping mapping = config_.subgraph_mappings[group.first];
                mapping.cta_weight = 0;
                for (std::size_t segment = group.first; segment < group.second;
                     ++segment) {
                    mapping.cta_weight +=
                        config_.subgraph_mappings[segment].cta_weight;
                }
                execution_mappings.push_back(mapping);
            }
        }

        logical_config_.execution_group_mappings = execution_mappings;
        config_.stage_partition = std::move(execution_partition);
        config_.subgraph_mappings = std::move(execution_mappings);
        config_.boundary_mappings = std::move(execution_boundaries);
        config_.execution_group_mappings.clear();
        if (uses_specialized_hierarchical_streaming_10x10(config_)) {
            selection_.implementation =
                "hierarchical-dataflow-wave-10x10-resident-radix4";
            selection_.reason =
                "resident lowering targets the generated V100 10+10 radix-4 execution core";
        } else if (uses_specialized_hierarchical_streaming_three_level(config_)) {
            selection_.implementation =
                "hierarchical-dataflow-wave-three-level-resident-radix4";
            selection_.reason =
                "resident lowering targets a generated V100 three-level radix-4 execution core";
        }
        selection_.implementation += has_resident_boundary
                                         ? "-resident-fused-m"
                                         : "-execution-remap-m";
        selection_.implementation +=
            std::to_string(logical_config_.stage_partition.size()) + "-g" +
            std::to_string(config_.stage_partition.size());
        selection_.reason += has_resident_boundary
                                 ? "; logical subgraphs are lowered into resident physical execution groups"
                                 : "; explicit physical execution-group mappings replace logical codelet mappings";
    }

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
        if (config_.backend == Backend::HybridDataflow) {
            const bool role_stages_automatic = config_.role_stages == 0;
            const bool data_time_automatic = config_.data_time == 0;
            const bool token_interleave_automatic = config_.token_interleave == 0;
            const bool compute_unit_automatic = config_.compute_unit == ComputeUnit::Auto;
            const ComputeUnit requested_compute_unit = config_.compute_unit;
            bool generated_default_applied = false;
            int device = 0;
            CUNTT_CUDA_CHECK(cudaGetDevice(&device));
            CUNTT_CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device));
            if (config_.flow_tile_log_n == 0) {
                config_.flow_tile_log_n = config_.log_n <= 8 ? config_.log_n : (config_.log_n + 1) / 2;
                config_.flow_tile_log_n = std::min<std::uint32_t>(8, config_.flow_tile_log_n);
            }
            if (config_.stage_space == 0) {
                config_.stage_space = config_.flow_tile_log_n;
            }
            if (config_.data_space == 0) {
                config_.data_space = config_.word_bits == 32 ? 32 : 16;
            }
            if (config_.pipeline_buffers == 0) {
                config_.pipeline_buffers = 2;
            }
            if (config_.target_ctas_per_sm == 0) {
                config_.target_ctas_per_sm = 1;
            }
            if (config_.token_interleave == 0) {
                config_.token_interleave = 1;
            }
            const bool primary_policy = config_.pipeline_buffers == 2 &&
                                        config_.dataflow_layout == DataflowLayout::HermesXor &&
                                        config_.dataflow_state_mode == DataflowStateMode::InPlace &&
                                        config_.stage_handoff == StageHandoff::NamedBarrier;
            const bool v100 = device_properties_.major == 7 && device_properties_.minor == 0;
            if (role_stages_automatic && data_time_automatic && token_interleave_automatic && primary_policy && v100) {
                generated_default_applied = detail::apply_generated_hybrid_dataflow_default(
                    config_, static_cast<std::uint32_t>(device_properties_.multiProcessorCount));
                if (!compute_unit_automatic) {
                    config_.compute_unit = requested_compute_unit;
                }
            }
            if (config_.role_stages == 0) {
                config_.role_stages = 1;
            }
            if (config_.data_time == 0) {
                if (!primary_policy) {
                    config_.data_time = 1;
                } else {
                    const std::size_t shared_limit = std::max<std::size_t>(device_properties_.sharedMemPerBlock,
                                                                           device_properties_.sharedMemPerBlockOptin);
                    constexpr std::array<std::uint32_t, 4> candidates = {8U, 4U, 2U, 1U};
                    const std::size_t candidate_begin = config_.role_stages > 1 ? 0 : 1;
                    for (std::size_t index = candidate_begin; index < candidates.size(); ++index) {
                        const std::uint32_t candidate = candidates[index];
                        config_.data_time = candidate;
                        if (hybrid_dataflow_shared_bytes(config_) <= shared_limit &&
                            detail::generated_hybrid_dataflow_available(config_)) {
                            break;
                        }
                    }
                }
            }
            if (config_.compute_unit == ComputeUnit::Auto) {
                config_.compute_unit = ComputeUnit::Radix2;
            }
            config_.n1_log            = 0;
            config_.rows_per_block    = 0;
            config_.threads_per_block = config_.compute_unit == ComputeUnit::Radix4
                                            ? 256
                                            : ((config_.stage_space + config_.role_stages - 1) /
                                               config_.role_stages) * config_.role_stages * 32;
            dataflow_shared_bytes_    = hybrid_dataflow_shared_bytes(config_);
            if (role_stages_automatic && data_time_automatic && token_interleave_automatic) {
                selection_.target = v100 ? "v100-sm70" : "unrecognized-device";
                selection_.implementation = generated_default_applied
                                                ? "hybrid-dataflow-generated-static-table"
                                                : "hybrid-dataflow-resource-fallback";
                selection_.confidence = generated_default_applied ? "measured" : "fallback";
                selection_.reason = generated_default_applied
                                        ? "matched generated numeric/length/batch preference"
                                        : "no generated preference matched; selected Ur=1 and a legal Td";
            }
            return;
        }
        if (config_.backend == Backend::HierarchicalBarrier) {
            const bool target_ctas_automatic = config_.target_ctas_per_sm == 0;
            int device = 0;
            CUNTT_CUDA_CHECK(cudaGetDevice(&device));
            CUNTT_CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device));
            if (config_.n1_log == 0) {
                config_.n1_log = config_.log_n / 2;
            }
            const std::uint32_t first_log = config_.log_n - config_.n1_log;
            if (config_.compute_unit == ComputeUnit::Auto) {
                config_.compute_unit = ComputeUnit::Radix4;
            }
            if (config_.rows_per_block == 0) {
                const std::uint32_t larger_factor = 1U << std::max(config_.n1_log, first_log);
                const std::size_t four_row_bytes =
                    4ULL * (larger_factor + 1) * (config_.word_bits / 8);
                const std::size_t shared_limit = std::max<std::size_t>(
                    device_properties_.sharedMemPerBlock, device_properties_.sharedMemPerBlockOptin);
                config_.rows_per_block = four_row_bytes <= shared_limit ? 4 : 2;
            }
            if (config_.threads_per_block == 0) {
                config_.threads_per_block = 256;
            }
            if (config_.data_time == 0) {
                config_.data_time = 1;
            }
            if (config_.target_ctas_per_sm == 0) {
                config_.target_ctas_per_sm = 1;
            }
            if (config_.flow_tile_log_n == 0) {
                config_.flow_tile_log_n = std::max(config_.n1_log, first_log);
            }
            if (config_.stage_space == 0) {
                config_.stage_space = config_.flow_tile_log_n;
            }
            if (config_.data_space == 0) {
                config_.data_space = config_.rows_per_block;
            }
            const bool dispatchable = config_.n1_log >= 6 && config_.n1_log <= 10 &&
                                      first_log >= 6 && first_log <= 10 &&
                                      (config_.rows_per_block == 1 || config_.rows_per_block == 2 ||
                                       config_.rows_per_block == 4);
            if (dispatchable) {
                const std::uint32_t actual_residency = config_.word_bits == 32
                    ? hierarchical_dataflow_residency<std::uint32_t>(config_)
                    : hierarchical_dataflow_residency<std::uint64_t>(config_);
                if (actual_residency == 0) {
                    throw std::invalid_argument(
                        "hierarchical-barrier point has zero cooperative CTA residency");
                }
                if (target_ctas_automatic) {
                    config_.target_ctas_per_sm = actual_residency;
                } else if (config_.target_ctas_per_sm > actual_residency) {
                    throw std::invalid_argument(
                        "hierarchical-barrier target_ctas_per_sm exceeds measured kernel occupancy");
                }
            }
            config_.role_stages = 0;
            config_.token_interleave = 0;
            config_.pipeline_buffers = 0;
            selection_.target = device_properties_.major == 7 && device_properties_.minor == 0
                                    ? "v100-sm70" : "unrecognized-device";
            selection_.implementation = std::string("hierarchical-barrier-") +
                                        hierarchical_core_name(config_.hierarchical_core);
            selection_.confidence = "fallback";
            selection_.reason = "balanced two-layer split with persistent cooperative CTAs";
            return;
        }
        if (config_.backend == Backend::HierarchicalDataflow) {
            int device = 0;
            CUNTT_CUDA_CHECK(cudaGetDevice(&device));
            CUNTT_CUDA_CHECK(cudaGetDeviceProperties(&device_properties_, device));
            const bool is_v100 = device_properties_.major == 7 &&
                                 device_properties_.minor == 0;
            if (config_.stage_partition.empty()) {
                if (is_v100 && config_.log_n == 20) {
                    // The generated 10+10 point reuses the mature resident
                    // radix-4 core and wins once independent transforms fill
                    // the two-role pipeline.
                    config_.stage_partition = {10, 10};
                } else {
                    // Three segments retain an outer independent digit at
                    // large N, allowing a single transform to form a wavefront.
                    const std::uint32_t segments = config_.log_n <= 16 ? 2 : 3;
                    const std::uint32_t base = config_.log_n / segments;
                    const std::uint32_t remainder = config_.log_n % segments;
                    config_.stage_partition.resize(segments, base);
                    for (std::uint32_t segment = 0; segment < remainder; ++segment) {
                        ++config_.stage_partition[segment];
                    }
                }
            }
            const bool v100_10x10 = is_v100 && config_.log_n == 20 &&
                                     config_.stage_partition ==
                                         std::vector<std::uint32_t>{10, 10};
            const bool v100_7x7x6 = is_v100 && config_.log_n == 20 &&
                                     config_.stage_partition ==
                                         std::vector<std::uint32_t>{7, 7, 6};
            const bool v100_three_level =
                is_v100 && config_.log_n == 20 &&
                (config_.stage_partition ==
                     std::vector<std::uint32_t>{6, 7, 7} ||
                 config_.stage_partition ==
                     std::vector<std::uint32_t>{7, 6, 7} ||
                 v100_7x7x6 ||
                 config_.stage_partition ==
                     std::vector<std::uint32_t>{6, 6, 8} ||
                 config_.stage_partition ==
                     std::vector<std::uint32_t>{6, 8, 6} ||
                 config_.stage_partition ==
                     std::vector<std::uint32_t>{8, 6, 6});
            const bool appt_pipeline = uses_appt_pipeline(config_);
            const bool appt_online = uses_appt_online(config_);
            const bool appt_mapping = appt_pipeline || appt_online;
            const auto automatic_cta_weight = [&](std::size_t segment) {
                if (v100_7x7x6 && appt_online) {
                    const bool register_tail =
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTail ||
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                        config_.subgraph_mappings.front().core ==
                            NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal ||
                        config_.subgraph_mappings.front().core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                        config_.subgraph_mappings.front().core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalResident ||
                        config_.subgraph_mappings.front().core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
                    if (register_tail) {
                        constexpr std::uint32_t u32_batch1[] = {8, 6, 6};
                        constexpr std::uint32_t u32_batched[] = {7, 7, 6};
                        constexpr std::uint32_t u64_weights[] = {10, 5, 5};
                        return config_.word_bits == 32
                                   ? (config_.batch == 1
                                          ? u32_batch1[segment]
                                          : u32_batched[segment])
                                   : u64_weights[segment];
                    }
                    const bool split_tail =
                        config_.subgraph_mappings.front().core ==
                        NttSubgraphCore::ApptOnlineSplitTail;
                    if (split_tail) {
                        constexpr std::uint32_t split_weights[] = {7, 6, 7};
                        return split_weights[segment];
                    }
                    const bool fused_tail =
                        config_.subgraph_mappings.front().core ==
                        NttSubgraphCore::ApptOnlineFusedTail;
                    if (fused_tail) {
                        constexpr std::uint32_t fused_weights[] = {7, 7, 6};
                        return fused_weights[segment];
                    }
                    const bool cta_radix4 =
                        config_.subgraph_mappings.front().core ==
                        NttSubgraphCore::ApptOnlineRadix4;
                    if (config_.word_bits == 32) {
                        constexpr std::uint32_t warp_batch1[] = {6, 6, 8};
                        constexpr std::uint32_t warp_batched[] = {7, 7, 6};
                        constexpr std::uint32_t cta[] = {7, 6, 7};
                        return cta_radix4 ? cta[segment]
                                          : config_.batch == 1
                                          ? warp_batch1[segment]
                                          : warp_batched[segment];
                    }
                    constexpr std::uint32_t batch1[] = {4, 8, 8};
                    constexpr std::uint32_t warp_batched[] = {6, 8, 6};
                    constexpr std::uint32_t cta_batched[] = {5, 7, 8};
                    return config_.batch == 1
                               ? batch1[segment]
                               : cta_radix4 ? cta_batched[segment]
                                            : warp_batched[segment];
                }
                // The second subgraph also performs the fused cross-twiddle
                // and finalization. Measured service time on V100 therefore
                // favors a small consumer surplus over a stage-count split.
                if (v100_10x10) return segment == 0 ? 9U : 11U;
                // The final 6-stage subgraph owns the output permutation, so
                // its measured service demand is larger than its stage count.
                if (v100_7x7x6) return segment < 2 ? 6U : 8U;
                return config_.stage_partition[segment];
            };
            if (config_.subgraph_mappings.empty()) {
                config_.subgraph_mappings.resize(config_.stage_partition.size());
                for (std::size_t segment = 0; segment < config_.subgraph_mappings.size(); ++segment) {
                    auto& mapping = config_.subgraph_mappings[segment];
                    mapping.core = NttSubgraphCore::DataflowRadix4;
                    mapping.threads_per_block = 256;
                    mapping.units_per_cta = v100_three_level ? 32U : 1U;
                    mapping.cta_weight = automatic_cta_weight(segment);
                }
            } else {
                if (config_.subgraph_mappings.size() != config_.stage_partition.size()) {
                    throw std::invalid_argument(
                        "hierarchical-dataflow subgraph mapping count must match stage_partition");
                }
                for (std::size_t segment = 0; segment < config_.subgraph_mappings.size(); ++segment) {
                    auto& mapping = config_.subgraph_mappings[segment];
                    const bool coefficient_reuse_core =
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp256CoefficientReuseStaticIoRadix2 ||
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128CoefficientReuseStaticIoRadix2;
                    const bool vector_radix4_core =
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4StaticIo ||
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
                    const bool packed_stage6_core =
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
                        mapping.core == NttSubgraphCore::
                                            HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
                    if (coefficient_reuse_core &&
                        mapping.coefficient_reuse_stages == 0) {
                        mapping.coefficient_reuse_stages =
                            mapping.core == NttSubgraphCore::
                                                HomogeneousWarp256CoefficientReuseStaticIoRadix2
                                ? 1U
                                : 4U;
                    }
                    if (packed_stage6_core &&
                        mapping.coefficient_reuse_stages == 0) {
                        mapping.coefficient_reuse_stages = 6U;
                    }
                    if ((coefficient_reuse_core &&
                         (mapping.coefficient_reuse_stages < 1 ||
                          mapping.coefficient_reuse_stages > 6)) ||
                        (vector_radix4_core &&
                         mapping.coefficient_reuse_stages != 0 &&
                         (mapping.coefficient_reuse_stages < 3 ||
                          mapping.coefficient_reuse_stages > 7)) ||
                        (packed_stage6_core &&
                         mapping.coefficient_reuse_stages != 6) ||
                        (!coefficient_reuse_core && !vector_radix4_core &&
                         mapping.coefficient_reuse_stages != 0)) {
                        throw std::invalid_argument(
                            "coefficient reuse stages must be [1,6] on radix-2 reuse cores, 0 or [3,7] on vector-radix4, exactly 6 on packed-stage6 vector-radix4, and 0 otherwise");
                    }
                    if (appt_mapping) {
                        const std::uint32_t physical_stages = config_.stage_partition.front();
                        mapping.role_stages = mapping.role_stages == 0 ? 1U : mapping.role_stages;
                        const std::uint32_t physical_roles =
                            (physical_stages + mapping.role_stages - 1) / mapping.role_stages;
                        if (mapping.units_per_cta < physical_roles) {
                            mapping.units_per_cta = physical_roles;
                        }
                        mapping.threads_per_block = mapping.units_per_cta * 32;
                        mapping.data_space = mapping.data_space == 0 ? 16U : mapping.data_space;
                        mapping.data_time = mapping.data_time == 0 ? 8U : mapping.data_time;
                        mapping.token_interleave = mapping.token_interleave == 0
                                                       ? 1U : mapping.token_interleave;
                    }
                    if (mapping.units_per_cta == 0) {
                        mapping.units_per_cta = v100_7x7x6 ? 32U : 1U;
                    }
                    if (mapping.cta_weight == 0) {
                        mapping.cta_weight = automatic_cta_weight(segment);
                    }
                }
            }
            const bool automatic_boundaries = config_.boundary_mappings.empty();
            if (config_.boundary_mappings.empty()) {
                config_.boundary_mappings.resize(config_.stage_partition.size() - 1);
            }
            if (appt_mapping && automatic_boundaries) {
                for (auto& boundary : config_.boundary_mappings) {
                    boundary.storage = BoundaryStorage::Ring;
                    boundary.buffers = 2;
                }
            }
            if (config_.target_ctas_per_sm == 0) {
                const bool v100_log20_u32 = device_properties_.major == 7 &&
                                             device_properties_.minor == 0 &&
                                             config_.log_n == 20 &&
                                             config_.word_bits == 32;
                config_.target_ctas_per_sm = appt_mapping
                                                  ? uses_appt_fused_tail(config_)
                                                        ? config_.word_bits == 32 ? 2U : 1U
                                                        : uses_appt_register_tail_grouped_writer_final_resident(config_)
                                                        ? 1U
                                                        : uses_appt_register_tail_grouped_writer_final_resident_quarter(config_)
                                                        ? 2U
                                                        : uses_appt_register_tail(config_)
                                                        ? config_.word_bits == 32 ? 3U : 2U
                                                        : uses_appt_split_tail(config_)
                                                        ? config_.word_bits == 32 ? 3U : 2U
                                                        : config_.word_bits == 32 ? 3U : 2U
                                                  : v100_7x7x6 && config_.word_bits == 32
                                                  ? 3
                                                  : v100_log20_u32 ? 4 : 2;
            }
            const bool packet_shared_radix4 =
                !config_.subgraph_mappings.empty() &&
                config_.subgraph_mappings.front().core == NttSubgraphCore::
                    HomogeneousWarp128PacketSharedRadix4StaticIo;
            if (packet_shared_radix4) {
                // Five 128-thread CTAs fit the arithmetic resource estimate,
                // but the cooperative producer/consumer role mix can leave a
                // full SM spinning before every producer packet is issued.
                config_.target_ctas_per_sm =
                    std::min<std::uint32_t>(config_.target_ctas_per_sm, 4U);
                const std::uint32_t wave =
                    config_.subgraph_mappings.back().token_interleave;
                if (wave != 0U && wave != 4U && wave != 8U &&
                    wave != 16U) {
                    throw std::invalid_argument(
                        "packet-shared radix-4 token interleave must be 0, 4, 8, or 16 q-groups");
                }
                const std::uint32_t poll_window = config_.ready_window;
                if (poll_window != 0U && poll_window != 1U &&
                    poll_window != 2U && poll_window != 4U &&
                    poll_window != 8U && poll_window != 16U) {
                    throw std::invalid_argument(
                        "packet-shared radix-4 ready window must be 1, 2, 4, 8, or 16 poll quanta");
                }
                if (config_.packet_readiness_mode ==
                        PacketReadinessMode::WaveBitmap &&
                    wave == 0U) {
                    throw std::invalid_argument(
                        "wave-bitmap packet readiness requires a nonzero packet wave");
                }
                if (config_.packet_compute_layout ==
                        PacketComputeLayout::WarpRows &&
                    wave == 0U) {
                    throw std::invalid_argument(
                        "warp-row packet compute layout requires a nonzero packet wave");
                }
                if (config_.packet_fold_wave_barriers && wave == 0U) {
                    throw std::invalid_argument(
                        "packet wave-barrier folding requires a nonzero packet wave");
                }
            } else if (config_.packet_readiness_mode !=
                           PacketReadinessMode::PerPacket ||
                       config_.packet_compute_layout !=
                           PacketComputeLayout::InterleavedRows ||
                       config_.packet_fold_wave_barriers) {
                throw std::invalid_argument(
                    "packet readiness and compute layout are only valid for the packet-shared radix-4 core");
            }
            if (uses_appt_register_tail(config_)) {
                auto& roles = config_.appt_role_mapping;
                if (roles.fragment_width == 0) {
                    roles.fragment_width = 32;
                }
                if (roles.writer_tiles_per_cta == 0) {
                    roles.writer_tiles_per_cta = 1;
                }
                if (roles.producer_weight == 0 && roles.tail_weight == 0 &&
                    roles.writer_weight == 0) {
                    if (config_.output_order == OutputOrder::ApptStatic) {
                        if (config_.word_bits == 32) {
                            roles.producer_weight = config_.batch >= 16 ? 7 : 8;
                            roles.tail_weight = config_.batch >= 16 ? 14 : 12;
                        } else {
                            roles.producer_weight = config_.batch == 1 ? 11 : 10;
                            roles.tail_weight = config_.batch == 1 ? 9 : 10;
                        }
                    } else if (config_.word_bits == 32) {
                        roles.producer_weight = config_.batch == 1 ? 8 : 7;
                        roles.tail_weight = config_.batch == 1
                                                ? 12
                                                : config_.batch >= 16 ? 15 : 14;
                    } else {
                        roles.producer_weight = 11;
                        roles.tail_weight = 9;
                    }
                    roles.writer_weight = 2;
                }
            }
            if (config_.ready_window == 0) {
                config_.ready_window = packet_shared_radix4 ? 2U : 8U;
            }
            config_.compute_unit = ComputeUnit::Radix4;
            config_.n1_log = 0;
            config_.rows_per_block = 0;
            config_.threads_per_block = 0;
            selection_.target = device_properties_.major == 7 && device_properties_.minor == 0
                                    ? "v100-sm70" : "unrecognized-device";
            const bool uses_ring = std::any_of(
                config_.boundary_mappings.begin(), config_.boundary_mappings.end(),
                [](const NttBoundaryMapping& boundary) {
                    return boundary.storage == BoundaryStorage::Ring;
                });
            const bool specialized_10x10 =
                uses_specialized_hierarchical_streaming_10x10(config_);
            const bool homogeneous_10x10 =
                uses_homogeneous_hierarchical_streaming_10x10(config_);
            const bool homogeneous_warp_10x10 =
                uses_homogeneous_warp_hierarchical_streaming_10x10(config_);
            const bool homogeneous_warp256_10x10 =
                uses_homogeneous_warp256_hierarchical_streaming_10x10(config_);
            const bool homogeneous_warp256_static_10x10 =
                uses_homogeneous_warp256_static_hierarchical_streaming_10x10(
                    config_);
            const bool homogeneous_warp256_static_io_10x10 =
                uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
                    config_);
            const bool
                homogeneous_warp256_coefficient_reuse_static_io_10x10 =
                    uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
                        config_, NttSubgraphCore::
                                     HomogeneousWarp256CoefficientReuseStaticIoRadix2);
            const bool homogeneous_warp128_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_,
                    NttSubgraphCore::HomogeneousWarp128StaticIoRadix2);
            const bool homogeneous_warp128_coefficient_reuse_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::
                                 HomogeneousWarp128CoefficientReuseStaticIoRadix2);
            const bool homogeneous_warp128_vector_radix4_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::
                                 HomogeneousWarp128VectorRadix4StaticIo);
            const bool
                homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10 =
                    uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                        config_, NttSubgraphCore::
                                     HomogeneousWarp128VectorRadix4PackedStage6StaticIo);
            const bool
                homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10 =
                    uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                        config_, NttSubgraphCore::
                                     HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo);
            const bool homogeneous_warp128_packet_shared_radix4_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::
                                 HomogeneousWarp128PacketSharedRadix4StaticIo);
            const bool homogeneous_warp64_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::HomogeneousWarp64StaticIoRadix2);
            const bool homogeneous_warp128_pipeline_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::
                                 HomogeneousWarp128PipelineStaticIoRadix2) &&
                config_.pipeline_buffers == 1;
            const bool homogeneous_warp128_cooperative_static_io_10x10 =
                uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                    config_, NttSubgraphCore::
                                 HomogeneousWarp128CooperativeStaticIoRadix2);
            const bool specialized_three_level =
                uses_specialized_hierarchical_streaming_three_level(config_);
            const bool appt_online_radix4 =
                appt_online && config_.subgraph_mappings.front().core ==
                                   NttSubgraphCore::ApptOnlineRadix4;
            const bool appt_online_fused_tail =
                appt_online && config_.subgraph_mappings.front().core ==
                                   NttSubgraphCore::ApptOnlineFusedTail;
            const bool appt_online_split_tail =
                appt_online && config_.subgraph_mappings.front().core ==
                                   NttSubgraphCore::ApptOnlineSplitTail;
            const bool appt_online_register_tail =
                appt_online && uses_appt_register_tail(config_);
            const bool appt_online_register_tail_warp =
                appt_online && uses_appt_register_tail_warp(config_);
            const bool appt_online_register_tail_column_warp =
                appt_online && uses_appt_register_tail_column_warp(config_);
            const bool appt_online_register_tail_radix8 =
                appt_online && uses_appt_register_tail_radix8(config_);
            const bool appt_online_register_tail_grouped =
                appt_online && uses_appt_register_tail_grouped(config_);
            const bool appt_online_register_tail_grouped_writer_final =
                appt_online &&
                uses_appt_register_tail_grouped_writer_final(config_);
            const bool appt_online_register_tail_grouped_writer_final_data_time =
                appt_online &&
                uses_appt_register_tail_grouped_writer_final_data_time(config_);
            const bool appt_online_register_tail_grouped_writer_final_resident =
                appt_online &&
                uses_appt_register_tail_grouped_writer_final_resident(config_);
            const bool
                appt_online_register_tail_grouped_writer_final_resident_quarter =
                    appt_online &&
                    uses_appt_register_tail_grouped_writer_final_resident_quarter(
                        config_);
            selection_.implementation =
                                        appt_online_register_tail_grouped_writer_final_resident_quarter
                                            ? "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-resident-quarter"
                                        : appt_online_register_tail_grouped_writer_final_resident
                                            ? "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-resident"
                                        : appt_online_register_tail_grouped_writer_final_data_time
                                            ? "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-data-time"
                                        : appt_online_register_tail_grouped_writer_final
                                            ? "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final"
                                        : appt_online_register_tail_grouped
                                            ? "hierarchical-dataflow-appt-online-register-tail-grouped"
                                        : appt_online_register_tail_radix8
                                            ? "hierarchical-dataflow-appt-online-register-tail-radix8"
                                        : appt_online_register_tail_column_warp
                                            ? "hierarchical-dataflow-appt-online-register-tail-column-warp"
                                        : appt_online_register_tail_warp
                                            ? "hierarchical-dataflow-appt-online-register-tail-warp"
                                        : appt_online_register_tail
                                            ? "hierarchical-dataflow-appt-online-register-tail"
                                        : appt_online_split_tail
                                            ? "hierarchical-dataflow-appt-online-split-tail"
                                        : appt_online_fused_tail
                                            ? "hierarchical-dataflow-appt-online-fused-tail"
                                        : appt_online_radix4
                                            ? "hierarchical-dataflow-appt-online-radix4"
                                        : appt_online
                                            ? "hierarchical-dataflow-appt-online-warp-radix2"
                                            : appt_pipeline
                                            ? "hierarchical-dataflow-appt-pipeline"
                                            : uses_ring
                                            ? "hierarchical-dataflow-readiness-ring"
                                            : homogeneous_10x10
                                            ? "hierarchical-dataflow-homogeneous-radix4-10x10"
                                            : homogeneous_warp_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp-radix2-10x10"
                                            : homogeneous_warp256_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp256-radix2-10x10"
                                            : homogeneous_warp256_static_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp256-static-radix2-10x10"
                                            : homogeneous_warp128_pipeline_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-pipeline-static-io-radix2-10x10"
                                            : homogeneous_warp128_cooperative_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-cooperative-static-io-radix2-10x10"
                                            : homogeneous_warp128_coefficient_reuse_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-coefficient-reuse-static-io-radix2-10x10"
                                            : homogeneous_warp128_vector_radix4_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-vector-radix4-static-io-10x10"
                                            : homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io-10x10"
                                            : homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-vector-radix4-packed-stage6-static-io-10x10"
                                            : homogeneous_warp128_packet_shared_radix4_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-packet-shared-radix4-static-io-10x10"
                                            : homogeneous_warp128_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp128-static-io-radix2-10x10"
                                            : homogeneous_warp64_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp64-static-io-radix2-10x10"
                                            : homogeneous_warp256_coefficient_reuse_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp256-coefficient-reuse-static-io-radix2-10x10"
                                            : homogeneous_warp256_static_io_10x10
                                            ? "hierarchical-dataflow-homogeneous-warp256-static-io-radix2-10x10"
                                            : specialized_10x10
                                                  ? "hierarchical-dataflow-wave-10x10-resident-radix4"
                                                  : specialized_three_level
                                                        ? (config_.stage_partition ==
                                                                   std::vector<std::uint32_t>{7, 7, 6}
                                                               ? "hierarchical-dataflow-wave-7x7x6-resident-radix4"
                                                               : "hierarchical-dataflow-wave-three-level-resident-radix4")
                                                  : "hierarchical-dataflow-wave-warp-radix4";
            if (homogeneous_warp128_packet_shared_radix4_static_io_10x10 &&
                config_.packet_readiness_mode ==
                    PacketReadinessMode::WaveBitmap) {
                selection_.implementation += "-wave-bitmap";
            }
            if (homogeneous_warp128_packet_shared_radix4_static_io_10x10 &&
                config_.packet_compute_layout ==
                    PacketComputeLayout::WarpRows) {
                selection_.implementation += "-warp-rows";
            }
            if (homogeneous_warp128_packet_shared_radix4_static_io_10x10 &&
                config_.packet_fold_wave_barriers) {
                selection_.implementation += "-folded-wave-barriers";
            }
            selection_.confidence = "experimental";
            selection_.reason = appt_online
                                    ? "persistent CTA fold roles publish online-reordered packet groups without fold-wide barriers"
                                    : appt_pipeline
                                    ? "fixed warp stages stream independent register tokens and are reused across stage folds"
                                    : uses_ring
                                    ? "fixed-role cooperative subgraphs with epoch-protected bounded boundaries"
                                    : homogeneous_10x10
                                    ? "generated homogeneous two-level radix-4 roles with independent batch-time traversal"
                                    : homogeneous_warp_10x10
                                    ? "eight dependency-closed warp-register subgraphs per persistent CTA role"
                                    : homogeneous_warp256_10x10
                                    ? "two independent four-warp subgraphs with an eight-stage register prefix"
                                    : homogeneous_warp256_static_10x10
                                    ? "warp256 subgraphs with XOR-swizzled full-sector boundary and final writers"
                                    : homogeneous_warp128_pipeline_static_io_10x10
                                    ? "three prefix warps publish 128-point tiles to one merge warp through a single-slot online stage pipeline"
                                    : homogeneous_warp128_cooperative_static_io_10x10
                                    ? "four warp-owned 256-point continuations merge through two warp pairs before a four-warp final stage"
                                    : homogeneous_warp128_coefficient_reuse_static_io_10x10
                                    ? "dual row subgraphs reuse per-lane coefficients across register slots"
                                    : homogeneous_warp128_vector_radix4_packed_stage6_distributed_static_io_10x10
                                    ? "all 32 lanes issue aligned stage-6 coefficient pairs from a prepacked side table"
                                    : homogeneous_warp128_vector_radix4_packed_stage6_static_io_10x10
                                    ? "dual vector-radix4 row subgraphs issue aligned stage-6 coefficient vectors from a prepacked side table"
                                    : homogeneous_warp128_packet_shared_radix4_static_io_10x10
                                    ? "independent 128-thread packets fuse four rows through the resident shared radix-4 core"
                                    : homogeneous_warp128_static_io_10x10
                                    ? "128-point warp prefixes reduce the static-IO register live set before shared completion"
                                    : homogeneous_warp64_static_io_10x10
                                    ? "64-point warp prefixes maximize static-IO residency before shared completion"
                                    : homogeneous_warp256_coefficient_reuse_static_io_10x10
                                    ? "256-point warp prefixes explicitly reuse per-lane coefficients across register slots"
                                    : homogeneous_warp256_static_io_10x10
                                    ? "warp256 subgraphs with coalesced row-packet input and XOR-swizzled full-sector writers"
                                    : specialized_10x10
                                          ? "transform-wave readiness with the generated resident radix-4 core"
                                          : specialized_three_level
                                                ? "intra-transform dependency waves with generated resident radix-4 cores"
                                          : "warp-parallel subgraphs with dependency-wave readiness";
            return;
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
        if (config_.word_bits == 32 &&
            ((config_.backend != Backend::Hybrid2D && config_.backend != Backend::HybridDataflow &&
              config_.backend != Backend::HierarchicalBarrier &&
              config_.backend != Backend::HierarchicalDataflow) || config_.modulus >= (1ULL << 31))) {
            throw std::invalid_argument("32-bit words require hybrid2d or a dataflow backend and modulus < 2^31");
        }
        if (config_.output_order == OutputOrder::BitReversed && config_.backend != Backend::CompactStage) {
            throw std::invalid_argument("bit-reversed output is currently supported only by compact-stage");
        }
        const bool appt_static_capable =
            config_.backend == Backend::HierarchicalDataflow &&
            uses_appt_register_tail(config_);
        if (config_.input_order == InputOrder::ApptStatic &&
            !appt_static_capable) {
            throw std::invalid_argument(
                "APPT static input is supported only by hierarchical register-tail");
        }
        if (config_.output_order == OutputOrder::ApptStatic &&
            !appt_static_capable) {
            throw std::invalid_argument(
                "APPT static output is supported only by hierarchical register-tail");
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
        if (config_.backend == Backend::HybridDataflow) {
            if (config_.log_n < 6) {
                throw std::invalid_argument("hybrid-dataflow currently requires logN >= 6");
            }
            if (config_.flow_tile_log_n < 5 || config_.flow_tile_log_n > 8 || config_.flow_tile_log_n > config_.log_n) {
                throw std::invalid_argument("hybrid-dataflow flow_tile_log_n must be a generated value in [5, 8] not exceeding logN");
            }
            if (config_.role_stages == 0 || config_.role_stages > config_.stage_space) {
                throw std::invalid_argument("hybrid-dataflow role_stages must be in [1, stage_space]");
            }
            if (config_.target_ctas_per_sm < 1 || config_.target_ctas_per_sm > 3) {
                throw std::invalid_argument("hybrid-dataflow target_ctas_per_sm must be in [1, 3]");
            }
            if (config_.token_interleave < 1 || config_.token_interleave > 2) {
                throw std::invalid_argument("hybrid-dataflow token_interleave must be 1 or 2");
            }
            if (config_.token_interleave > 1 &&
                (config_.role_stages == 1 || config_.data_time < config_.role_stages * config_.token_interleave)) {
                throw std::invalid_argument(
                    "hybrid-dataflow token_interleave=2 requires fused roles and at least 2 * role_stages tokens");
            }
            if (!detail::generated_hybrid_dataflow_available(config_)) {
                throw std::invalid_argument(
                    "hybrid-dataflow point is not present in the generated design manifest");
            }
            if (config_.modular_multiply != ModularMultiply::Shoup ||
                (config_.compute_unit != ComputeUnit::Radix2 && config_.compute_unit != ComputeUnit::Radix4) ||
                config_.output_order != OutputOrder::Natural) {
                throw std::invalid_argument(
                    "hybrid-dataflow currently requires radix2/radix4 Shoup arithmetic and natural-order output");
            }
            const std::size_t shared_limit = std::max<std::size_t>(device_properties_.sharedMemPerBlock,
                                                                   device_properties_.sharedMemPerBlockOptin);
            if (dataflow_shared_bytes_ > shared_limit) {
                throw std::invalid_argument("hybrid-dataflow resident state requires " + std::to_string(dataflow_shared_bytes_) +
                                            " shared bytes, exceeding the device limit " + std::to_string(shared_limit));
            }
            if (dataflow_shared_bytes_ * config_.target_ctas_per_sm >
                static_cast<std::size_t>(device_properties_.sharedMemPerMultiprocessor)) {
                throw std::invalid_argument("hybrid-dataflow CTA residency target exceeds SM shared memory");
            }
            if (config_.threads_per_block > static_cast<std::uint32_t>(device_properties_.maxThreadsPerBlock)) {
                throw std::invalid_argument("hybrid-dataflow fused roles exceed the device thread-block limit");
            }
            if (config_.batch > static_cast<std::size_t>(device_properties_.maxGridSize[0])) {
                throw std::invalid_argument("hybrid-dataflow batch exceeds the CUDA grid-x limit");
            }
        }
        if (config_.backend == Backend::HierarchicalBarrier) {
            constexpr std::uint32_t min_local_log = 6;
            constexpr std::uint32_t max_local_log = 10;
            const std::uint32_t first_log = config_.log_n - config_.n1_log;
            if (config_.log_n < 12 || config_.log_n > 20 ||
                config_.n1_log < min_local_log || config_.n1_log > max_local_log ||
                first_log < min_local_log || first_log > max_local_log) {
                throw std::invalid_argument(
                    "hierarchical-barrier requires a two-layer logN=12..20 split with each layer in [6, 10]");
            }
            if (config_.compute_unit != ComputeUnit::Radix4 ||
                config_.modular_multiply != ModularMultiply::Shoup ||
                config_.cross_twiddle_placement != CrossTwiddlePlacement::Fused ||
                config_.output_order != OutputOrder::Natural) {
                throw std::invalid_argument(
                    "hierarchical-barrier currently requires radix4, fused Shoup twiddles, and natural-order output");
            }
            if (config_.hierarchical_core == HierarchicalCore::Hybrid2DRadix4 &&
                (config_.log_n != 20 || config_.n1_log != 10)) {
                throw std::invalid_argument(
                    "hierarchical hybrid2d-radix4 core is currently generated only for the 10+10 split");
            }
            if (config_.rows_per_block != 1 && config_.rows_per_block != 2 &&
                config_.rows_per_block != 4) {
                throw std::invalid_argument(
                    "hierarchical-barrier rows_per_block must be 1, 2, or 4");
            }
            if (config_.data_time < 1 || config_.data_time > 16) {
                throw std::invalid_argument("hierarchical-barrier data_time must be in [1, 16]");
            }
            if (config_.target_ctas_per_sm < 1) {
                throw std::invalid_argument(
                    "hierarchical-barrier target_ctas_per_sm must be positive");
            }
            if (config_.threads_per_block == 0 ||
                config_.threads_per_block > static_cast<std::uint32_t>(device_properties_.maxThreadsPerBlock) ||
                config_.threads_per_block % static_cast<std::uint32_t>(device_properties_.warpSize) != 0) {
                throw std::invalid_argument(
                    "hierarchical-barrier threads_per_block must be a supported warp multiple");
            }
            if (!device_properties_.cooperativeLaunch) {
                throw std::invalid_argument(
                    "hierarchical-barrier requires cooperative grid launch support");
            }
            const std::uint32_t larger_factor = 1U << std::max(config_.n1_log, first_log);
            const std::size_t shared_bytes = static_cast<std::size_t>(config_.rows_per_block) *
                                             (larger_factor + 1) * (config_.word_bits / 8);
            const std::size_t shared_limit = std::max<std::size_t>(
                device_properties_.sharedMemPerBlock, device_properties_.sharedMemPerBlockOptin);
            if (shared_bytes > shared_limit ||
                shared_bytes * config_.target_ctas_per_sm >
                    static_cast<std::size_t>(device_properties_.sharedMemPerMultiprocessor)) {
                throw std::invalid_argument(
                    "hierarchical-barrier mapping exceeds the requested CTA shared-memory residency");
            }
            if (config_.threads_per_block * config_.target_ctas_per_sm >
                static_cast<std::uint32_t>(device_properties_.maxThreadsPerMultiProcessor)) {
                throw std::invalid_argument(
                    "hierarchical-barrier mapping exceeds the requested CTA thread residency");
            }
        }
        if (config_.backend == Backend::HierarchicalDataflow) {
            const bool appt_pipeline = uses_appt_pipeline(config_);
            const bool appt_online = uses_appt_online(config_);
            const bool appt_mapping = appt_pipeline || appt_online;
            const bool any_appt_pipeline = std::any_of(
                config_.subgraph_mappings.begin(), config_.subgraph_mappings.end(),
                [](const NttSubgraphMapping& mapping) {
                    return mapping.core == NttSubgraphCore::ApptPipeline ||
                           mapping.core == NttSubgraphCore::ApptOnline ||
                           mapping.core == NttSubgraphCore::ApptOnlineRadix4 ||
                           mapping.core == NttSubgraphCore::ApptOnlineFusedTail ||
                           mapping.core == NttSubgraphCore::ApptOnlineSplitTail ||
                           mapping.core == NttSubgraphCore::ApptOnlineRegisterTail ||
                           mapping.core == NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                           mapping.core == NttSubgraphCore::ApptOnlineRegisterTailColumnWarp ||
                           mapping.core == NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                           mapping.core == NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                           mapping.core ==
                               NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal ||
                           mapping.core == NttSubgraphCore::
                               ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                           mapping.core == NttSubgraphCore::
                               ApptOnlineRegisterTailGroupedWriterFinalResident ||
                           mapping.core == NttSubgraphCore::
                               ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
                });
            if (any_appt_pipeline && !appt_mapping) {
                throw std::invalid_argument(
                    "appt-pipeline cannot be mixed with fixed-role subgraph cores");
            }
            if (config_.log_n < 12 || config_.log_n > 20) {
                throw std::invalid_argument("hierarchical-dataflow currently requires logN=12..20");
            }
            if (config_.stage_partition.size() < 2 ||
                config_.stage_partition.size() > kMaxStreamingSegments) {
                throw std::invalid_argument("hierarchical-dataflow requires 2..8 subgraphs");
            }
            std::uint32_t stage_sum = 0;
            for (std::size_t segment = 0; segment < config_.stage_partition.size(); ++segment) {
                const auto stages = config_.stage_partition[segment];
                if ((!appt_mapping && (stages < 2 || stages > 10)) ||
                    (appt_mapping && (stages == 0 || stages > config_.stage_partition.front()))) {
                    throw std::invalid_argument(
                        "hierarchical-dataflow subgraphs require 2..10 stages");
                }
                stage_sum += stages;
            }
            if (stage_sum != config_.log_n) {
                throw std::invalid_argument("hierarchical-dataflow stage_partition must sum to logN");
            }
            if (config_.subgraph_mappings.size() != config_.stage_partition.size() ||
                config_.boundary_mappings.size() + 1 != config_.stage_partition.size()) {
                throw std::invalid_argument("hierarchical-dataflow descriptor counts do not match stage_partition");
            }
            if (appt_mapping) {
                const std::uint32_t stage_space = config_.stage_partition.front();
                if (stage_space < 4 || stage_space > 8) {
                    throw std::invalid_argument("appt-pipeline stage-space must be in [4, 8]");
                }
                for (std::size_t segment = 0; segment + 1 < config_.stage_partition.size(); ++segment) {
                    if (config_.stage_partition[segment] != stage_space) {
                        throw std::invalid_argument(
                            "appt-pipeline non-tail folds must reuse one fixed stage-space");
                    }
                }
                const auto& physical = config_.subgraph_mappings.front();
                if ((physical.data_space != 8 && physical.data_space != 16 &&
                     physical.data_space != 32) ||
                    physical.data_time == 0 || physical.data_time > 64 ||
                    (physical.role_stages != 1 && physical.role_stages != 2) ||
                    (physical.token_interleave != 1 && physical.token_interleave != 2)) {
                    throw std::invalid_argument("invalid appt-pipeline (Ud,Td) mapping");
                }
                const std::uint32_t physical_roles =
                    (stage_space + physical.role_stages - 1) / physical.role_stages;
                if (appt_pipeline &&
                    (physical.units_per_cta % physical_roles != 0 ||
                     physical.units_per_cta / physical_roles > 2)) {
                    throw std::invalid_argument(
                        "appt-pipeline supports one or two complete pipeline replicas per CTA");
                }
                for (const auto& mapping : config_.subgraph_mappings) {
                    const bool valid_data_space =
                        mapping.data_space == 8 || mapping.data_space == 16 ||
                        mapping.data_space == 32;
                    if (!valid_data_space ||
                        mapping.threads_per_block != physical.units_per_cta * 32 ||
                        mapping.units_per_cta != physical.units_per_cta ||
                        (!appt_online && mapping.data_space != physical.data_space) ||
                        mapping.data_time != physical.data_time ||
                        mapping.role_stages != physical.role_stages ||
                        mapping.token_interleave != physical.token_interleave) {
                        throw std::invalid_argument(
                            "appt-pipeline folds must reuse one physical mapping; "
                            "appt-online may vary data-space by fold");
                    }
                }
                const std::uint32_t buffers = config_.boundary_mappings.front().buffers;
                if (buffers < 1 || buffers > 3 ||
                    std::any_of(config_.boundary_mappings.begin(), config_.boundary_mappings.end(),
                                [&](const NttBoundaryMapping& boundary) {
                                    return boundary.storage != BoundaryStorage::Ring ||
                                           boundary.buffers != buffers;
                                })) {
                    throw std::invalid_argument(
                        "appt-pipeline requires 1..3 equal ring buffers between physical stages");
                }
                if (appt_online &&
                    (config_.log_n != 20 ||
                     config_.stage_partition != std::vector<std::uint32_t>{7, 7, 6} ||
                     physical.threads_per_block != 256)) {
                    throw std::invalid_argument(
                        "appt-online currently requires logN=20, partition 7+7+6, and 256 threads");
                }
                if (appt_online) {
                    const auto online_core = config_.subgraph_mappings.front().core;
                    if (std::any_of(config_.subgraph_mappings.begin(),
                                    config_.subgraph_mappings.end(),
                                    [&](const NttSubgraphMapping& mapping) {
                                        return mapping.core != online_core;
                                    })) {
                        throw std::invalid_argument(
                            "appt-online folds must select one physical subgraph core");
                    }
                    if (online_core == NttSubgraphCore::ApptOnlineRadix4 &&
                        (config_.subgraph_mappings[1].data_space != 16 ||
                         config_.subgraph_mappings[2].data_space != 16)) {
                        throw std::invalid_argument(
                            "appt-online-radix4 currently requires fine 1/1 publication");
                    }
                    if (online_core == NttSubgraphCore::ApptOnlineFusedTail &&
                        (config_.subgraph_mappings[1].data_space != 16 ||
                         config_.subgraph_mappings[2].data_space != 16 ||
                         config_.profile_appt_roles)) {
                        throw std::invalid_argument(
                            "appt-online-fused-tail requires fine publication and disables intrusive role profiling");
                    }
                    if (online_core == NttSubgraphCore::ApptOnlineSplitTail &&
                        (config_.subgraph_mappings[1].data_space != 16 ||
                         config_.subgraph_mappings[2].data_space != 16 ||
                         config_.profile_appt_roles)) {
                        throw std::invalid_argument(
                            "appt-online-split-tail requires fine publication and disables intrusive role profiling");
                    }
                    const bool register_tail_core =
                        online_core == NttSubgraphCore::ApptOnlineRegisterTail ||
                        online_core ==
                            NttSubgraphCore::ApptOnlineRegisterTailWarp ||
                        online_core == NttSubgraphCore::
                                           ApptOnlineRegisterTailColumnWarp ||
                        online_core ==
                            NttSubgraphCore::ApptOnlineRegisterTailRadix8 ||
                        online_core ==
                            NttSubgraphCore::ApptOnlineRegisterTailGrouped ||
                        online_core ==
                            NttSubgraphCore::ApptOnlineRegisterTailGroupedWriterFinal ||
                        online_core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                        online_core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalResident ||
                        online_core == NttSubgraphCore::
                            ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter;
                    if (register_tail_core &&
                        (config_.subgraph_mappings[1].data_space != 16 ||
                         config_.subgraph_mappings[2].data_space != 16 ||
                         config_.profile_appt_roles)) {
                        throw std::invalid_argument(
                            "appt-online-register-tail requires fine publication and disables intrusive role profiling");
                    }
                    if (register_tail_core) {
                        if ((online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinal ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResident ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter) &&
                            config_.output_order != OutputOrder::Natural) {
                            throw std::invalid_argument(
                                "appt-online-register-tail-grouped-writer-final requires natural-order output");
                        }
                        const auto& roles = config_.appt_role_mapping;
                        if ((roles.fragment_width != 8 &&
                             roles.fragment_width != 16 &&
                             roles.fragment_width != 32) ||
                            (roles.writer_tiles_per_cta != 1 &&
                             roles.writer_tiles_per_cta != 2 &&
                             roles.writer_tiles_per_cta != 4) ||
                            roles.producer_weight == 0 || roles.tail_weight == 0 ||
                            (config_.output_order == OutputOrder::Natural &&
                             roles.writer_weight == 0)) {
                            throw std::invalid_argument(
                                "invalid APPT producer/tail/writer physical mapping");
                        }
                        if ((online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalDataTime ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResident ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter) &&
                            (roles.data_time_role_mask == 0 ||
                             roles.data_time_role_mask > 7)) {
                            throw std::invalid_argument(
                                "APPT data-time role mask must be in [1, 7]");
                        }
                        if ((online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResident ||
                             online_core == NttSubgraphCore::
                                                ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter) &&
                            physical.data_space != 32) {
                            throw std::invalid_argument(
                                "APPT resident 2D core requires data-space 32");
                        }
                    }
                }
                if (config_.profile_appt_roles &&
                    (!appt_online || config_.subgraph_mappings[1].data_space != 16 ||
                     config_.subgraph_mappings[2].data_space != 16)) {
                    throw std::invalid_argument(
                        "APPT role profiling currently requires the fine 1/1 online mapping");
                }
            }
            for (const auto& mapping : config_.subgraph_mappings) {
                if (mapping.core == NttSubgraphCore::Hybrid2DRadix4) {
                    throw std::invalid_argument(
                        "hybrid2d-radix4 is not yet adapted to the true-streaming subgraph ABI");
                }
                if (mapping.threads_per_block == 0 ||
                    mapping.threads_per_block > static_cast<std::uint32_t>(device_properties_.maxThreadsPerBlock) ||
                    mapping.threads_per_block % static_cast<std::uint32_t>(device_properties_.warpSize) != 0 ||
                    mapping.units_per_cta == 0 ||
                    mapping.cta_weight == 0) {
                    throw std::invalid_argument("invalid hierarchical-dataflow subgraph CTA mapping");
                }
            }
            for (const auto& boundary : config_.boundary_mappings) {
                if (boundary.buffers == 0 ||
                    (boundary.storage == BoundaryStorage::FullScratch && boundary.buffers != 1)) {
                    throw std::invalid_argument("invalid hierarchical-dataflow boundary buffer count");
                }
            }
            const bool register_layout = uses_appt_register_tail(config_) &&
                (config_.output_order == OutputOrder::Natural ||
                 config_.output_order == OutputOrder::ApptStatic) &&
                (config_.input_order == InputOrder::Natural ||
                 config_.input_order == InputOrder::ApptStatic);
            if (config_.modular_multiply != ModularMultiply::Shoup ||
                (!register_layout &&
                 (config_.output_order != OutputOrder::Natural ||
                  config_.input_order != InputOrder::Natural))) {
                throw std::invalid_argument(
                    "hierarchical-dataflow requires Shoup arithmetic; static layout is supported only by APPT register-tail");
            }
            if (config_.target_ctas_per_sm == 0 || !device_properties_.cooperativeLaunch) {
                throw std::invalid_argument(
                    "hierarchical-dataflow requires a positive CTA target and cooperative launch support");
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
        if (config_.backend == Backend::Hybrid2D || config_.backend == Backend::HierarchicalBarrier ||
            config_.backend == Backend::HierarchicalDataflow) {
            root_powers_.resize(n);
            root_powers_shoup_.resize(n);
            std::uint64_t omega = 1;
            for (std::size_t i = 0; i < n; ++i) {
                root_powers_[i]       = omega;
                root_powers_shoup_[i] = shoup_precompute(omega, config_.modulus);
                omega                 = static_cast<std::uint64_t>((static_cast<unsigned __int128>(omega) * root) % config_.modulus);
            }
            const bool specialized_streaming =
                config_.backend == Backend::HierarchicalDataflow &&
                (uses_specialized_hierarchical_streaming_10x10(config_) ||
                 uses_homogeneous_hierarchical_streaming_10x10(config_) ||
                 uses_homogeneous_warp_hierarchical_streaming_10x10(config_) ||
                 uses_homogeneous_warp256_hierarchical_streaming_10x10(config_) ||
                 uses_homogeneous_warp256_static_hierarchical_streaming_10x10(
                     config_) ||
                 uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
                     config_) ||
                 uses_homogeneous_warp256_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp256CoefficientReuseStaticIoRadix2) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_,
                     NttSubgraphCore::HomogeneousWarp128StaticIoRadix2) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128CoefficientReuseStaticIoRadix2) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128VectorRadix4StaticIo) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128VectorRadix4PackedStage6StaticIo) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128PacketSharedRadix4StaticIo) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_,
                     NttSubgraphCore::HomogeneousWarp64StaticIoRadix2) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128PipelineStaticIoRadix2) ||
                 uses_homogeneous_warp_small_static_io_hierarchical_streaming_10x10(
                     config_, NttSubgraphCore::
                                  HomogeneousWarp128CooperativeStaticIoRadix2));
            const bool fused = (config_.backend != Backend::HierarchicalDataflow ||
                                specialized_streaming) &&
                               config_.cross_twiddle_placement == CrossTwiddlePlacement::Fused;
            if (fused) {
                const std::uint32_t split_log = specialized_streaming ? 10U : config_.n1_log;
                const std::uint32_t        n1 = 1U << split_log;
                const std::uint32_t        n2 = 1U << (config_.log_n - split_log);
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
            const bool packed_stage6 =
                config_.backend == Backend::HierarchicalDataflow &&
                !config_.subgraph_mappings.empty() &&
                (config_.subgraph_mappings.front().core == NttSubgraphCore::
                     HomogeneousWarp128VectorRadix4PackedStage6StaticIo ||
                 config_.subgraph_mappings.front().core == NttSubgraphCore::
                     HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo);
            if (packed_stage6) {
                constexpr std::size_t kLocalN = 1024;
                constexpr std::size_t kStage6Base = 63;
                constexpr std::size_t kStage6Size = 64;
                constexpr std::size_t kFusedWords = kLocalN * (kLocalN - 1);
                constexpr std::size_t kPackedWords =
                    kStage6Size + kLocalN * kStage6Size;
                if (root_powers_.size() != kFusedWords ||
                    root_powers_shoup_.size() != kFusedWords) {
                    throw std::logic_error(
                        "packed stage-6 coefficients require fused 10x10 root trees");
                }
                root_powers_.reserve(kFusedWords + kPackedWords);
                root_powers_shoup_.reserve(kFusedWords + kPackedWords);
                root_powers_.insert(root_powers_.end(),
                                    twiddles_.begin() + kStage6Base,
                                    twiddles_.begin() + kStage6Base + kStage6Size);
                root_powers_shoup_.insert(
                    root_powers_shoup_.end(),
                    twiddles_shoup_.begin() + kStage6Base,
                    twiddles_shoup_.begin() + kStage6Base + kStage6Size);
                for (std::size_t row = 0; row < kLocalN; ++row) {
                    const std::size_t source = row * (kLocalN - 1) + kStage6Base;
                    for (std::size_t offset = 0; offset < kStage6Size; ++offset) {
                        root_powers_.push_back(root_powers_[source + offset]);
                        root_powers_shoup_.push_back(
                            root_powers_shoup_[source + offset]);
                    }
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
        if (config_.backend == Backend::HybridDataflow && config_.word_bits == 32) {
            twiddles32_.reserve(twiddles_.size());
            twiddles_shoup32_.reserve(twiddles_shoup_.size());
            for (std::size_t index = 0; index < twiddles_.size(); ++index) {
                const auto twiddle = static_cast<std::uint32_t>(twiddles_[index]);
                twiddles32_.push_back(twiddle);
                twiddles_shoup32_.push_back(
                    shoup_precompute32(twiddle, static_cast<std::uint32_t>(config_.modulus)));
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
        if (config_.backend == Backend::HybridDataflow) {
            if (config_.word_bits == 32) {
                dispatch_hybrid_dataflow(config_, static_cast<const std::uint32_t*>(input_pointer),
                                         static_cast<std::uint32_t*>(output_pointer), device_twiddles_.as<std::uint32_t>(),
                                         device_twiddles_shoup_.as<std::uint32_t>(), inverse_n_, inverse_n_shoup_);
            } else {
                dispatch_hybrid_dataflow(config_, input, output, device_twiddles_.get(), device_twiddles_shoup_.get(),
                                         inverse_n_, inverse_n_shoup_);
            }
            return;
        }
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
        if (config_.backend == Backend::HierarchicalDataflow) {
            if (config_.word_bits == 32) {
                launch_hierarchical_streaming(
                    config_, static_cast<const std::uint32_t*>(input_pointer),
                    static_cast<std::uint32_t*>(output_pointer), scratch_pointer,
                    device_twiddles_.as<std::uint32_t>(),
                    device_twiddles_shoup_.as<std::uint32_t>(),
                    device_root_powers_.as<std::uint32_t>(),
                    device_root_powers_shoup_.as<std::uint32_t>(), inverse_n_,
                    inverse_n_shoup_, device_properties_.multiProcessorCount);
            } else {
                launch_hierarchical_streaming(
                    config_, input, output, scratch_pointer, device_twiddles_.get(),
                    device_twiddles_shoup_.get(), device_root_powers_.get(),
                    device_root_powers_shoup_.get(), inverse_n_, inverse_n_shoup_,
                    device_properties_.multiProcessorCount);
            }
            return;
        }
        if (config_.backend == Backend::HierarchicalBarrier) {
            if (config_.word_bits == 32) {
                dispatch_hierarchical_dataflow(
                    config_, static_cast<const std::uint32_t*>(input_pointer),
                    static_cast<std::uint32_t*>(output_pointer),
                    static_cast<std::uint32_t*>(scratch_pointer),
                    device_twiddles_.as<std::uint32_t>(),
                    device_twiddles_shoup_.as<std::uint32_t>(),
                    device_root_powers_.as<std::uint32_t>(),
                    device_root_powers_shoup_.as<std::uint32_t>(), inverse_n_,
                    inverse_n_shoup_, device_properties_.multiProcessorCount);
            } else {
                dispatch_hierarchical_dataflow(
                    config_, input, output, scratch, device_twiddles_.get(),
                    device_twiddles_shoup_.get(), device_root_powers_.get(),
                    device_root_powers_shoup_.get(), inverse_n_, inverse_n_shoup_,
                    device_properties_.multiProcessorCount);
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
    PlanConfig                 logical_config_;
    SelectionInfo              selection_;
    std::size_t                n_            = 0;
    std::size_t                total_points_ = 0;
    std::size_t                data_bytes_ = 0;
    std::size_t                workspace_bytes_ = 0;
    std::size_t                dataflow_shared_bytes_ = 0;
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

ApptLayoutInfo Plan::appt_layout_info() const {
    return impl_->appt_layout_info();
}

std::vector<PipelineSegmentTrace> Plan::pipeline_trace() const {
    return impl_->pipeline_trace();
}

std::vector<PipelineRoleMetrics> Plan::pipeline_role_metrics() const {
    return impl_->pipeline_role_metrics();
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
