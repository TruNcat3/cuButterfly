#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kModulus = 576460756061519873ULL;

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    throw std::runtime_error(std::string(expression) + " failed at " + file + ':' + std::to_string(line) + ": " + cudaGetErrorString(status));
}

#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

std::uint64_t shoup_precompute(std::uint64_t value, std::uint64_t modulus) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(value) << 64) / modulus);
}

__device__ __forceinline__ std::uint64_t mul_shoup(std::uint64_t value, std::uint64_t root, std::uint64_t root_shoup, std::uint64_t modulus) {
    const std::uint64_t quotient = __umul64hi(value, root_shoup);
    std::uint64_t       reduced  = value * root - quotient * modulus;
    if (reduced >= modulus) {
        reduced -= modulus;
    }
    return reduced;
}

__global__ void global_copy_kernel(const std::uint64_t* input, std::uint64_t* output, std::uint64_t count) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index];
    }
}

__global__ void initialize_kernel(std::uint64_t* values, std::uint64_t count) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = index + 1;
    }
}

__global__ void butterfly_kernel(std::uint64_t* values, std::uint64_t butterflies, std::uint32_t iterations, std::uint64_t root,
                                 std::uint64_t root_shoup, std::uint64_t modulus) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= butterflies) {
        return;
    }

    std::uint64_t left  = values[2 * index];
    std::uint64_t right = values[2 * index + 1];
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const std::uint64_t product = mul_shoup(right, root, root_shoup, modulus);
        const std::uint64_t sum     = left + product;
        right                       = left >= product ? left - product : modulus + left - product;
        left                        = sum >= modulus ? sum - modulus : sum;
    }
    values[2 * index]     = left;
    values[2 * index + 1] = right;
}

__global__ void shared_exchange_kernel(const std::uint64_t* input, std::uint64_t* output, std::uint32_t iterations) {
    extern __shared__ std::uint64_t shared[];
    const std::uint32_t             local = threadIdx.x;
    const std::uint64_t             base  = (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + local) * 2;
    shared[local]                         = input[base];
    shared[blockDim.x + local]            = input[base + 1];

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        __syncthreads();
        const std::uint32_t peer   = local ^ 1U;
        const std::uint64_t first  = shared[peer];
        const std::uint64_t second = shared[blockDim.x + peer];
        __syncthreads();
        shared[local]              = second;
        shared[blockDim.x + local] = first;
    }
    __syncthreads();
    output[base]     = shared[local];
    output[base + 1] = shared[blockDim.x + local];
}

__global__ void barrier_kernel(std::uint64_t* output, std::uint32_t iterations) {
    std::uint64_t value = threadIdx.x;
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        value += iteration & 1U;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = value;
    }
}

template <std::uint32_t StageSpace>
__global__ void stage_pipeline_kernel(std::uint64_t* values, std::uint64_t total_tiles, std::uint32_t stage_base, std::uint64_t root,
                                      std::uint64_t root_shoup, std::uint64_t modulus) {
    constexpr std::uint32_t kTilePoints        = 256;
    constexpr std::uint32_t kWarpsPerBlock     = 8;
    constexpr std::uint32_t kTokensPerPipeline = 2;
    constexpr std::uint32_t kPipelines         = kWarpsPerBlock / StageSpace;
    static_assert(StageSpace == 1 || StageSpace == 2 || StageSpace == 4 || StageSpace == 8);

    __shared__ std::uint64_t buffers[kWarpsPerBlock * kTokensPerPipeline * kTilePoints];
    __shared__ int           ready[kWarpsPerBlock * kTokensPerPipeline];

    const std::uint32_t warp     = threadIdx.x / warpSize;
    const std::uint32_t lane     = threadIdx.x % warpSize;
    const std::uint32_t pipeline = warp / StageSpace;
    const std::uint32_t role     = warp % StageSpace;
    if (threadIdx.x < kWarpsPerBlock * kTokensPerPipeline) {
        ready[threadIdx.x] = 0;
    }
    __syncthreads();

    for (std::uint32_t token = 0; token < kTokensPerPipeline; ++token) {
        const std::uint64_t tile = static_cast<std::uint64_t>(blockIdx.x) * (kPipelines * kTokensPerPipeline) + pipeline * kTokensPerPipeline + token;
        const bool          active = tile < total_tiles;
        if (role != 0) {
            const std::uint32_t previous_flag = ((pipeline * StageSpace + role - 1) * kTokensPerPipeline) + token;
            while (atomicAdd(&ready[previous_flag], 0) == 0) {
            }
        }
        __syncwarp();

        const std::uint32_t stage       = stage_base + role;
        const std::uint32_t half        = 1U << stage;
        const std::uint32_t source      = ((pipeline * StageSpace + (role == 0 ? 0 : role - 1)) * kTokensPerPipeline + token) * kTilePoints;
        const std::uint32_t target      = ((pipeline * StageSpace + role) * kTokensPerPipeline + token) * kTilePoints;
        const std::uint64_t global_base = tile * kTilePoints;

        for (std::uint32_t butterfly_index = lane; butterfly_index < kTilePoints / 2; butterfly_index += warpSize) {
            const std::uint32_t group       = butterfly_index / half;
            const std::uint32_t offset      = butterfly_index - group * half;
            const std::uint32_t left_index  = group * (2 * half) + offset;
            const std::uint32_t right_index = left_index + half;
            std::uint64_t       left        = 0;
            std::uint64_t       right       = 0;
            if (active) {
                if (role == 0) {
                    left  = values[global_base + left_index];
                    right = values[global_base + right_index];
                } else {
                    left  = buffers[source + left_index];
                    right = buffers[source + right_index];
                }
            }
            right                         = mul_shoup(right, root, root_shoup, modulus);
            const std::uint64_t sum       = left + right;
            buffers[target + left_index]  = sum >= modulus ? sum - modulus : sum;
            buffers[target + right_index] = left >= right ? left - right : modulus + left - right;
        }
        __syncwarp();
        __threadfence_block();

        if (role + 1 < StageSpace) {
            if (lane == 0) {
                const std::uint32_t flag = ((pipeline * StageSpace + role) * kTokensPerPipeline) + token;
                atomicExch(&ready[flag], 1);
            }
        } else if (active) {
            for (std::uint32_t index = lane; index < kTilePoints; index += warpSize) {
                values[global_base + index] = buffers[target + index];
            }
        }
    }
}

template <typename Launch>
double time_kernel(Launch&& launch, std::uint32_t warmup, std::uint32_t repeat) {
    for (std::uint32_t iteration = 0; iteration < warmup; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (std::uint32_t iteration = 0; iteration < repeat; ++iteration) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / repeat;
}

template <std::uint32_t StageSpace>
double time_stage_pipeline(std::uint64_t* values, std::uint64_t points, std::uint32_t warmup, std::uint32_t repeat) {
    constexpr std::uint32_t kStages            = 8;
    constexpr std::uint32_t kTilePoints        = 256;
    constexpr std::uint32_t kTokensPerPipeline = 2;
    constexpr std::uint32_t kPipelines         = 8 / StageSpace;
    const std::uint64_t     total_tiles        = points / kTilePoints;
    const std::uint32_t grid = static_cast<std::uint32_t>((total_tiles + kPipelines * kTokensPerPipeline - 1) / (kPipelines * kTokensPerPipeline));
    const std::uint64_t root = 7;
    const std::uint64_t root_shoup = shoup_precompute(root, kModulus);
    return time_kernel(
        [&] {
            for (std::uint32_t stage_base = 0; stage_base < kStages; stage_base += StageSpace) {
                stage_pipeline_kernel<StageSpace><<<grid, 256>>>(values, total_tiles, stage_base, root, root_shoup, kModulus);
                CUDA_CHECK(cudaGetLastError());
            }
        },
        warmup, repeat);
}

std::uint64_t parse_u64(const char* value, const char* option) {
    char*               end    = nullptr;
    const std::uint64_t parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0') {
        throw std::invalid_argument(std::string("invalid value for ") + option);
    }
    return parsed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::uint64_t points     = 1ULL << 22;
        std::uint32_t threads    = 256;
        std::uint32_t blocks     = 640;
        std::uint32_t iterations = 64;
        std::uint32_t warmup     = 5;
        std::uint32_t repeat     = 20;
        for (int index = 1; index < argc; ++index) {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string("missing value for ") + argv[index]);
            }
            const std::string option = argv[index];
            const auto        value  = parse_u64(argv[++index], option.c_str());
            if (option == "--points") {
                points = value;
            } else if (option == "--threads") {
                threads = value;
            } else if (option == "--blocks") {
                blocks = value;
            } else if (option == "--iterations") {
                iterations = value;
            } else if (option == "--warmup") {
                warmup = value;
            } else if (option == "--repeat") {
                repeat = value;
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        if (points < 256 || points % 256 != 0 || threads == 0 || threads > 1024 || blocks == 0 || iterations == 0 || repeat == 0) {
            throw std::invalid_argument("invalid benchmark dimensions");
        }

        int            device = 0;
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        std::uint64_t* input  = nullptr;
        std::uint64_t* output = nullptr;
        CUDA_CHECK(cudaMalloc(&input, points * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc(&output, points * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMemset(output, 0, points * sizeof(std::uint64_t)));

        const std::uint32_t copy_blocks      = static_cast<std::uint32_t>((points + threads - 1) / threads);
        const std::uint32_t butterfly_blocks = static_cast<std::uint32_t>((points / 2 + threads - 1) / threads);
        const std::uint64_t shared_points    = static_cast<std::uint64_t>(blocks) * threads * 2;
        if (shared_points > points) {
            throw std::invalid_argument("--points must cover 2 * blocks * threads shared-exchange values");
        }
        initialize_kernel<<<copy_blocks, threads>>>(input, points);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        const std::uint64_t root       = 7;
        const std::uint64_t root_shoup = shoup_precompute(root, kModulus);
        const auto          copy_ms    = time_kernel(
            [&] {
                global_copy_kernel<<<copy_blocks, threads>>>(input, output, points);
                CUDA_CHECK(cudaGetLastError());
            },
            warmup, repeat);
        const auto butterfly_ms = time_kernel(
            [&] {
                butterfly_kernel<<<butterfly_blocks, threads>>>(input, points / 2, iterations, root, root_shoup, kModulus);
                CUDA_CHECK(cudaGetLastError());
            },
            warmup, repeat);
        const auto shared_ms = time_kernel(
            [&] {
                shared_exchange_kernel<<<blocks, threads, 2ULL * threads * sizeof(std::uint64_t)>>>(input, output, iterations);
                CUDA_CHECK(cudaGetLastError());
            },
            warmup, repeat);
        const auto barrier_ms = time_kernel(
            [&] {
                barrier_kernel<<<blocks, threads>>>(output, iterations);
                CUDA_CHECK(cudaGetLastError());
            },
            warmup, repeat);
        CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
        const auto pipeline_us1_ms = time_stage_pipeline<1>(output, points, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
        const auto pipeline_us2_ms = time_stage_pipeline<2>(output, points, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
        const auto pipeline_us4_ms = time_stage_pipeline<4>(output, points, warmup, repeat);
        CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
        const auto pipeline_us8_ms = time_stage_pipeline<8>(output, points, warmup, repeat);

        std::vector<std::uint64_t> reference(points);
        std::vector<std::uint64_t> actual(points);
        CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
        time_stage_pipeline<1>(output, points, 0, 1);
        CUDA_CHECK(cudaMemcpy(reference.data(), output, points * sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
        bool pipeline_correct = true;
#define VERIFY_PIPELINE(STAGE_SPACE)                                                                       \
    CUDA_CHECK(cudaMemcpy(output, input, points * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));       \
    time_stage_pipeline<STAGE_SPACE>(output, points, 0, 1);                                                \
    CUDA_CHECK(cudaMemcpy(actual.data(), output, points * sizeof(std::uint64_t), cudaMemcpyDeviceToHost)); \
    pipeline_correct = pipeline_correct && actual == reference
        VERIFY_PIPELINE(2);
        VERIFY_PIPELINE(4);
        VERIFY_PIPELINE(8);
#undef VERIFY_PIPELINE

        const double copy_gb_s            = (2.0 * points * sizeof(std::uint64_t)) / (copy_ms * 1.0e6);
        const double butterfly_gops_s     = (static_cast<double>(points / 2) * iterations) / (butterfly_ms * 1.0e6);
        const double shared_gb_s          = (static_cast<double>(blocks) * threads * 32.0 * iterations) / (shared_ms * 1.0e6);
        const double barrier_gcta_s       = (static_cast<double>(blocks) * iterations) / (barrier_ms * 1.0e6);
        const double pipeline_butterflies = static_cast<double>(points / 2) * 8;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "device,compute_capability,sm_count,points,blocks,threads,iterations,warmup,repeat,global_copy_ms,global_copy_GB_s,"
                     "butterfly_ms,butterfly_Gbutterfly_s,shared_exchange_ms,shared_exchange_GB_s,barrier_ms,barrier_GCTA_s,"
                     "pipeline_us1_ms,pipeline_us1_Gbutterfly_s,pipeline_us2_ms,pipeline_us2_Gbutterfly_s,"
                     "pipeline_us4_ms,pipeline_us4_Gbutterfly_s,pipeline_us8_ms,pipeline_us8_Gbutterfly_s,pipeline_correct\n";
        std::cout << '"' << properties.name << "\"," << properties.major << '.' << properties.minor << ',' << properties.multiProcessorCount << ','
                  << points << ',' << blocks << ',' << threads << ',' << iterations << ',' << warmup << ',' << repeat << ',' << copy_ms << ','
                  << copy_gb_s << ',' << butterfly_ms << ',' << butterfly_gops_s << ',' << shared_ms << ',' << shared_gb_s << ',' << barrier_ms << ','
                  << barrier_gcta_s << ',' << pipeline_us1_ms << ',' << pipeline_butterflies / (pipeline_us1_ms * 1.0e6) << ',' << pipeline_us2_ms
                  << ',' << pipeline_butterflies / (pipeline_us2_ms * 1.0e6) << ',' << pipeline_us4_ms << ','
                  << pipeline_butterflies / (pipeline_us4_ms * 1.0e6) << ',' << pipeline_us8_ms << ','
                  << pipeline_butterflies / (pipeline_us8_ms * 1.0e6) << ',' << static_cast<int>(pipeline_correct) << '\n';

        CUDA_CHECK(cudaFree(input));
        CUDA_CHECK(cudaFree(output));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
