// Standalone, strict core x dataflow ablation for radix-16 FFTs.
//
// This executable intentionally does not use the public planner or lowering.
// It fixes the logical radix-16 algorithm and changes only (1) the
// small leaf codelet and (2) whether the first two leaf stages are written
// back to HBM between stages.  That makes the dataflow comparison independent
// of the production search path.

#include <cuda_runtime.h>
#include <cufft.h>

#include "cuntt/butterfly.hpp"
#include "fft_register_tile.cuh"
#include "fft_thread_codelet.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Complex = cuntt::Complex32;

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void check_cufft(cufftResult status, const char* what) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(what) + " failed with cuFFT status " + std::to_string(status));
    }
}

struct DeviceBuffer {
    Complex* value = nullptr;

    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(Complex)), "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(value); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Event {
    cudaEvent_t value = nullptr;

    Event() { check_cuda(cudaEventCreate(&value), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(value); }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

struct CufftPlan {
    cufftHandle value = 0;

    CufftPlan(unsigned log_n, std::uint64_t batch) {
        int n = 1 << log_n;
        if (batch > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("batch is too large for cufftPlanMany");
        }
        check_cufft(cufftPlanMany(&value, 1, &n, nullptr, 1, n,
                                  nullptr, 1, n, CUFFT_C2C, static_cast<int>(batch)),
                    "cufftPlanMany");
    }

    ~CufftPlan() {
        if (value != 0) cufftDestroy(value);
    }

    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
};

enum class CoreKind { Native, CufftDxThread };
enum class Organization { StagewiseHbm, ResidentPrefix2 };

const char* core_name(CoreKind core) {
    return core == CoreKind::Native ? "native" : "cufftdx-thread";
}

const char* organization_name(Organization organization) {
    return organization == Organization::StagewiseHbm ? "stagewise-hbm" : "resident-prefix-2";
}

struct Options {
    unsigned log_n = 8;
    std::uint64_t batch = 1;
    unsigned trials = 3;
    unsigned warmup = 100;
    unsigned repeat = 100;
    std::string core = "both";
    std::string organization = "both";
    bool csv = false;
    bool verify_only = false;
};

std::uint64_t parse_u64(const char* argument, const char* name) {
    try {
        std::size_t consumed = 0;
        const std::string text(argument);
        const auto value = std::stoull(text, &consumed, 10);
        if (consumed != text.size() || value == 0) throw std::invalid_argument("not positive");
        return value;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        auto require_value = [&](const char* name) -> const char* {
            if (index + 1 >= argc) throw std::invalid_argument(std::string(name) + " requires a value");
            return argv[++index];
        };
        if (argument == "--logN") options.log_n = static_cast<unsigned>(parse_u64(require_value("--logN"), "--logN"));
        else if (argument == "--batch") options.batch = parse_u64(require_value("--batch"), "--batch");
        else if (argument == "--trials") options.trials = static_cast<unsigned>(parse_u64(require_value("--trials"), "--trials"));
        else if (argument == "--warmup") options.warmup = static_cast<unsigned>(parse_u64(require_value("--warmup"), "--warmup"));
        else if (argument == "--repeat") options.repeat = static_cast<unsigned>(parse_u64(require_value("--repeat"), "--repeat"));
        else if (argument == "--core") options.core = require_value("--core");
        else if (argument == "--organization") options.organization = require_value("--organization");
        else if (argument == "--csv") options.csv = true;
        else if (argument == "--verify-only") options.verify_only = true;
        else if (argument == "--help" || argument == "-h") {
            std::cout << "usage: cubutterfly_core_dataflow_ablation --logN {8|12|16} --batch N\n"
                      << "       [--core native|cufftdx-thread|both]\n"
                      << "       [--organization stagewise-hbm|resident-prefix-2|both]\n"
                      << "       [--trials N] [--warmup N] [--repeat N] [--csv] [--verify-only]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.log_n != 8 && options.log_n != 12 && options.log_n != 16) {
        throw std::invalid_argument("--logN must be one of 8, 12, or 16");
    }
    if (options.core != "native" && options.core != "cufftdx-thread" && options.core != "both") {
        throw std::invalid_argument("--core must be native, cufftdx-thread, or both");
    }
    if (options.organization != "stagewise-hbm" && options.organization != "resident-prefix-2" &&
        options.organization != "both") {
        throw std::invalid_argument("--organization must be stagewise-hbm, resident-prefix-2, or both");
    }
    return options;
}

template <unsigned LogN>
__device__ __forceinline__ unsigned reverse_base16(unsigned value) {
    static_assert(LogN % 4 == 0, "radix-16 digit reversal requires a multiple of four bits");
    unsigned result = 0;
#pragma unroll
    for (unsigned digit = 0; digit < LogN / 4; ++digit) {
        const unsigned nibble = (value >> (4 * digit)) & 0xfU;
        result |= nibble << (LogN - 4 - 4 * digit);
    }
    return result;
}

template <unsigned LogN>
__global__ void digit_reverse_kernel(const Complex* input, Complex* output, std::uint64_t batch) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    const std::uint64_t element = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t count = batch * n;
    if (element >= count) return;
    const std::uint64_t transform = element / n;
    const unsigned position = static_cast<unsigned>(element & (n - 1));
    output[transform * n + reverse_base16<LogN>(position)] = input[element];
}

struct NativeLeaf {
    __device__ __forceinline__ static void run(Complex (&values)[16]) {
        cuntt::detail::register_tile::NativeCodelet<4, false>{}(values);
    }
};

struct CufftDxThreadLeaf {
    __device__ __forceinline__ static void run(Complex (&values)[16]) {
        cuntt::detail::register_tile::DxCodelet<4, false>{}(values);
    }
};

__device__ __forceinline__ unsigned shared_index(unsigned row, unsigned column) {
    return row * 16U + (column ^ row);
}

template <unsigned LogN, unsigned Stage, class Core>
__global__ void radix16_global_stage(const Complex* input, Complex* output, std::uint64_t batch) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    constexpr std::uint64_t m = std::uint64_t{1} << (4 * Stage);
    constexpr std::uint64_t leaves_per_transform = n / 16;
    const std::uint64_t leaf = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total_leaves = batch * leaves_per_transform;
    if (leaf >= total_leaves) return;

    const std::uint64_t transform = leaf / leaves_per_transform;
    const std::uint64_t local_leaf = leaf % leaves_per_transform;
    const std::uint64_t group = local_leaf / m;
    const std::uint64_t j = local_leaf % m;
    const std::uint64_t base = transform * n + group * 16 * m;
    Complex values[16];
#pragma unroll
    for (unsigned q = 0; q < 16; ++q) {
        values[q] = input[base + j + q * m];
        const float turns = static_cast<float>(q * j) / static_cast<float>(16 * m);
        values[q] = cuntt::detail::register_tile::multiply(
            values[q], cuntt::detail::register_tile::root<false, Complex>(turns));
    }
    Core::run(values);
#pragma unroll
    for (unsigned q = 0; q < 16; ++q) output[base + j + q * m] = values[q];
}

template <unsigned LogN, class Core>
__global__ void resident_prefix_kernel(const Complex* input, Complex* output, std::uint64_t batch) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    constexpr std::uint64_t tile_size = 256;
    constexpr std::uint64_t tiles_per_transform = n / tile_size;
    constexpr unsigned leaves_per_tile = 16;
    constexpr unsigned tile_values = 256;
    const unsigned tile_slot = threadIdx.x / leaves_per_tile;
    const unsigned lane = threadIdx.x % leaves_per_tile;
    const std::uint64_t tile_id = static_cast<std::uint64_t>(blockIdx.x) * 16 + tile_slot;
    const std::uint64_t total_tiles = batch * tiles_per_transform;
    const bool active = tile_id < total_tiles;
    const std::uint64_t transform = active ? tile_id / tiles_per_transform : 0;
    const std::uint64_t tile_number = active ? tile_id % tiles_per_transform : 0;
    const std::uint64_t base = transform * n + tile_number * tile_size;
    extern __shared__ Complex shared[];
    Complex* tile = shared + static_cast<std::size_t>(tile_slot) * tile_values;

    // APPT/writer-aligned physical placement for the 16x16 shared tile.  The
    // xor is an address permutation only: every read and write below uses the
    // same bijection, while a fixed row/column access has distinct banks.
#pragma unroll
    for (unsigned index = 0; index < 16; ++index) {
        tile[shared_index(lane, index)] =
            active ? input[base + lane * 16 + index] : Complex{0.0F, 0.0F};
    }
    __syncthreads();

    // Stage zero: 16 contiguous radix-16 leaves.
    if (active) {
        Complex values[16];
#pragma unroll
        for (unsigned q = 0; q < 16; ++q) values[q] = tile[shared_index(lane, q)];
        Core::run(values);
#pragma unroll
        for (unsigned q = 0; q < 16; ++q) tile[shared_index(lane, q)] = values[q];
    }
    __syncthreads();

    // Stage one: the same 256-value tile is transposed logically through
    // shared memory; no global boundary is inserted between the two stages.
    if (active) {
        Complex values[16];
#pragma unroll
        for (unsigned q = 0; q < 16; ++q) {
            values[q] = tile[shared_index(q, lane)];
            const float turns = static_cast<float>(q * lane) / 256.0F;
            values[q] = cuntt::detail::register_tile::multiply(
                values[q], cuntt::detail::register_tile::root<false, Complex>(turns));
        }
        Core::run(values);
#pragma unroll
        for (unsigned q = 0; q < 16; ++q) tile[shared_index(q, lane)] = values[q];
    }
    __syncthreads();

#pragma unroll
    for (unsigned index = 0; index < 16; ++index) {
        if (active) output[base + lane * 16 + index] = tile[shared_index(lane, index)];
    }
}

template <unsigned LogN>
void launch_digit_reverse(const Complex* input, Complex* output, std::uint64_t batch) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    const std::uint64_t count = batch * n;
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    digit_reverse_kernel<LogN><<<blocks, threads>>>(input, output, batch);
    check_cuda(cudaGetLastError(), "digit-reverse launch");
}

template <unsigned LogN, unsigned Stage, class Core>
void launch_global_stage(const Complex* input, Complex* output, std::uint64_t batch,
                         unsigned threads) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    const std::uint64_t leaves = batch * n / 16;
    const unsigned blocks = static_cast<unsigned>((leaves + threads - 1) / threads);
    radix16_global_stage<LogN, Stage, Core><<<blocks, threads>>>(input, output, batch);
    check_cuda(cudaGetLastError(), "global stage launch");
}

template <unsigned LogN, class Core>
void launch_resident_prefix(const Complex* input, Complex* output, std::uint64_t batch) {
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    constexpr std::uint64_t tiles = n / 256;
    const std::uint64_t total_tiles = batch * tiles;
    const unsigned blocks = static_cast<unsigned>((total_tiles + 16 - 1) / 16);
    constexpr unsigned threads = 16 * 16;
    constexpr std::size_t shared_bytes = 16U * 256U * sizeof(Complex);
    resident_prefix_kernel<LogN, Core><<<blocks, threads, shared_bytes>>>(input, output, batch);
    check_cuda(cudaGetLastError(), "resident-prefix launch");
}

template <unsigned LogN, class Core>
void enqueue_stagewise(const Complex* input, Complex* output, Complex* ping, Complex* pong,
                       std::uint64_t batch) {
    constexpr unsigned stages = LogN / 4;
    constexpr std::uint64_t n = std::uint64_t{1} << LogN;
    launch_digit_reverse<LogN>(input, ping, batch);
    const std::uint64_t leaves = batch * n / 16;
    const unsigned threads = static_cast<unsigned>(std::min<std::uint64_t>(256, leaves));
    const Complex* source = ping;
    for (unsigned stage = 0; stage < stages; ++stage) {
        Complex* destination = stage + 1 == stages ? output : (source == ping ? pong : ping);
        switch (stage) {
            case 0: launch_global_stage<LogN, 0, Core>(source, destination, batch, threads); break;
            case 1: launch_global_stage<LogN, 1, Core>(source, destination, batch, threads); break;
            case 2: launch_global_stage<LogN, 2, Core>(source, destination, batch, threads); break;
            case 3: launch_global_stage<LogN, 3, Core>(source, destination, batch, threads); break;
        }
        source = destination;
    }
}

template <unsigned LogN, class Core>
void enqueue_resident_prefix(const Complex* input, Complex* output, Complex* ping, Complex* pong,
                             std::uint64_t batch) {
    constexpr unsigned stages = LogN / 4;
    launch_digit_reverse<LogN>(input, ping, batch);
    // The resident prefix covers exactly stages 0 and 1.  For N=256 this is
    // the whole transform; for larger N, later stages intentionally return to
    // the same stagewise HBM path as the baseline.
    Complex* prefix_destination = stages == 2 ? output : pong;
    launch_resident_prefix<LogN, Core>(ping, prefix_destination, batch);
    if (stages == 2) return;
    const Complex* source = prefix_destination;
    for (unsigned stage = 2; stage < stages; ++stage) {
        Complex* destination = stage + 1 == stages ? output : (source == ping ? pong : ping);
        switch (stage) {
            case 2: launch_global_stage<LogN, 2, Core>(source, destination, batch, 256); break;
            case 3: launch_global_stage<LogN, 3, Core>(source, destination, batch, 256); break;
        }
        source = destination;
    }
}

template <unsigned LogN, class Core>
void enqueue(const Complex* input, Complex* output, Complex* ping, Complex* pong,
             std::uint64_t batch, Organization organization) {
    if (organization == Organization::StagewiseHbm) {
        enqueue_stagewise<LogN, Core>(input, output, ping, pong, batch);
    } else {
        enqueue_resident_prefix<LogN, Core>(input, output, ping, pong, batch);
    }
}

template <class Core>
void enqueue_runtime(unsigned log_n, const Complex* input, Complex* output, Complex* ping, Complex* pong,
                     std::uint64_t batch, Organization organization) {
    switch (log_n) {
        case 8: enqueue<8, Core>(input, output, ping, pong, batch, organization); break;
        case 12: enqueue<12, Core>(input, output, ping, pong, batch, organization); break;
        case 16: enqueue<16, Core>(input, output, ping, pong, batch, organization); break;
        default: throw std::invalid_argument("unsupported logN");
    }
}

struct Verification {
    bool correct = false;
    double max_error = 0.0;
    double max_reference = 0.0;
    double l2_error = 0.0;
    double l2_reference = 0.0;
};

Verification verify_output(const std::vector<Complex>& expected, const std::vector<Complex>& actual,
                           std::size_t transform_size) {
    Verification result;
    double error_squared = 0.0;
    double reference_squared = 0.0;
    for (const auto& value : expected) {
        const double magnitude = std::hypot(static_cast<double>(value.real), static_cast<double>(value.imag));
        result.max_reference = std::max(result.max_reference, magnitude);
        reference_squared += magnitude * magnitude;
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const double error = std::hypot(static_cast<double>(actual[index].real) - expected[index].real,
                                        static_cast<double>(actual[index].imag) - expected[index].imag);
        if (!std::isfinite(error)) {
            result.max_error = std::numeric_limits<double>::infinity();
            break;
        }
        result.max_error = std::max(result.max_error, error);
        error_squared += error * error;
    }
    result.l2_error = std::sqrt(error_squared);
    result.l2_reference = std::sqrt(reference_squared);
    const double relative_error = result.max_reference == 0.0 ? result.max_error : result.max_error / result.max_reference;
    const double absolute_tolerance = 2e-4 * std::sqrt(static_cast<double>(transform_size) / 1024.0);
    // Match the existing FP32 FFT acceptance guard.  L2 is retained as a
    // diagnostic rather than replacing the established maximum-error guard.
    result.correct = std::isfinite(result.max_error) && std::isfinite(result.l2_error) &&
                     relative_error <= 2e-4 && result.max_error <= absolute_tolerance;
    return result;
}

template <class Core>
Verification verify_variant(unsigned log_n, Organization organization, const Complex* input, Complex* output,
                            Complex* ping, Complex* pong, std::uint64_t batch,
                            const std::vector<Complex>& expected, std::vector<Complex>& actual) {
    enqueue_runtime<Core>(log_n, input, output, ping, pong, batch, organization);
    check_cuda(cudaGetLastError(), "verification enqueue");
    check_cuda(cudaMemcpy(actual.data(), output, actual.size() * sizeof(Complex), cudaMemcpyDeviceToHost),
               "copy verification output");
    const Verification result = verify_output(expected, actual, actual.size() / static_cast<std::size_t>(batch));
    if (!result.correct) {
        throw std::runtime_error(std::string("correctness failed for ") + organization_name(organization) +
                                 ": relative error " + std::to_string(result.max_error / result.max_reference));
    }
    return result;
}

struct Variant {
    CoreKind core;
    Organization organization;
    Verification verification;
};

void enqueue_variant(const Variant& variant, unsigned log_n, const Complex* input, Complex* output,
                     Complex* ping, Complex* pong, std::uint64_t batch) {
    if (variant.core == CoreKind::Native) {
        enqueue_runtime<NativeLeaf>(log_n, input, output, ping, pong, batch, variant.organization);
    } else {
        enqueue_runtime<CufftDxThreadLeaf>(log_n, input, output, ping, pong, batch, variant.organization);
    }
}

void measure_variants(unsigned log_n, std::uint64_t batch, const Options& options,
                      const Complex* input, Complex* output, Complex* ping, Complex* pong,
                      std::vector<Variant>& variants, int device, const cudaDeviceProp& properties) {
    for (auto& variant : variants) {
        for (unsigned index = 0; index < options.warmup; ++index) {
            enqueue_variant(variant, log_n, input, output, ping, pong, batch);
        }
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronize");
    Event start;
    Event stop;
    // Rotate the variant order per trial so the four cells are interleaved in
    // the same process and see the same clock/thermal epoch.
    for (unsigned trial = 0; trial < options.trials; ++trial) {
        for (std::size_t offset = 0; offset < variants.size(); ++offset) {
            const std::size_t index = (static_cast<std::size_t>(trial) + offset) % variants.size();
            auto& variant = variants[index];
            check_cuda(cudaEventRecord(start.value), "record start event");
            for (unsigned repeat = 0; repeat < options.repeat; ++repeat) {
                enqueue_variant(variant, log_n, input, output, ping, pong, batch);
            }
            check_cuda(cudaEventRecord(stop.value), "record stop event");
            check_cuda(cudaEventSynchronize(stop.value), "synchronize stop event");
            float elapsed = 0.0F;
            check_cuda(cudaEventElapsedTime(&elapsed, start.value, stop.value), "elapsed event time");
            const double kernel_ms = static_cast<double>(elapsed) / options.repeat;
            const double relative_error = variant.verification.max_reference == 0.0
                ? variant.verification.max_error
                : variant.verification.max_error / variant.verification.max_reference;
            const double relative_l2 = variant.verification.l2_reference == 0.0
                ? variant.verification.l2_error
                : variant.verification.l2_error / variant.verification.l2_reference;
            if (options.csv) {
                std::cout << device << ',' << properties.name << ',' << properties.major << '.' << properties.minor << ','
                          << log_n << ',' << batch << ',' << core_name(variant.core) << ','
                          << organization_name(variant.organization) << ',' << trial << ',' << std::setprecision(9)
                          << kernel_ms << ',' << (variant.verification.correct ? 1 : 0) << ','
                          << variant.verification.max_error << ',' << variant.verification.max_reference << ','
                          << relative_error << ',' << variant.verification.l2_error << ','
                          << variant.verification.l2_reference << ',' << relative_l2 << ','
                          << (batch * (std::uint64_t{1} << log_n)) << '\n';
            } else {
                std::cout << "logN=" << log_n << " batch=" << batch << " core=" << core_name(variant.core)
                          << " organization=" << organization_name(variant.organization) << " trial=" << trial
                          << " kernel_ms=" << std::setprecision(9) << kernel_ms
                          << " correct=" << (variant.verification.correct ? "yes" : "no")
                          << " relative_max_error=" << relative_error << " relative_l2_error=" << relative_l2 << '\n';
            }
        }
    }
}

void append_core(const std::string& selection, std::vector<CoreKind>& cores) {
    if (selection == "native" || selection == "both") cores.push_back(CoreKind::Native);
    if (selection == "cufftdx-thread" || selection == "both") cores.push_back(CoreKind::CufftDxThread);
}

void append_organization(const std::string& selection, std::vector<Organization>& organizations) {
    if (selection == "stagewise-hbm" || selection == "both") organizations.push_back(Organization::StagewiseHbm);
    if (selection == "resident-prefix-2" || selection == "both") organizations.push_back(Organization::ResidentPrefix2);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp properties{};
        check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        const std::size_t n = std::size_t{1} << options.log_n;
        if (options.batch > std::numeric_limits<std::size_t>::max() / n) {
            throw std::invalid_argument("batch and N overflow host size");
        }
        const std::size_t count = n * static_cast<std::size_t>(options.batch);
        DeviceBuffer input(count), output(count), ping(count), pong(count);
        std::vector<Complex> values(count), expected(count), actual(count);
        std::mt19937 generator(127);
        std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
        for (auto& value : values) value = {distribution(generator), distribution(generator)};
        check_cuda(cudaMemcpy(input.value, values.data(), count * sizeof(Complex), cudaMemcpyHostToDevice),
                   "copy input");

        CufftPlan reference(options.log_n, options.batch);
        check_cufft(cufftExecC2C(reference.value, reinterpret_cast<cufftComplex*>(input.value),
                                 reinterpret_cast<cufftComplex*>(output.value), CUFFT_FORWARD),
                    "cufftExecC2C");
        check_cuda(cudaMemcpy(expected.data(), output.value, count * sizeof(Complex), cudaMemcpyDeviceToHost),
                   "copy cuFFT reference");

        std::vector<CoreKind> cores;
        std::vector<Organization> organizations;
        append_core(options.core, cores);
        append_organization(options.organization, organizations);
        std::vector<Variant> variants;
        for (const auto organization : organizations) {
            for (const auto core : cores) {
                Verification verification;
                if (core == CoreKind::Native) {
                    verification = verify_variant<NativeLeaf>(options.log_n, organization, input.value, output.value,
                                                              ping.value, pong.value, options.batch, expected, actual);
                } else {
                    verification = verify_variant<CufftDxThreadLeaf>(options.log_n, organization, input.value, output.value,
                                                                      ping.value, pong.value, options.batch, expected, actual);
                }
                variants.push_back(Variant{core, organization, verification});
                std::cerr << "verified core=" << core_name(core)
                          << " organization=" << organization_name(organization)
                          << " relative_max_error=" << (verification.max_reference == 0.0
                              ? verification.max_error : verification.max_error / verification.max_reference)
                          << " relative_l2_error=" << (verification.l2_reference == 0.0
                              ? verification.l2_error : verification.l2_error / verification.l2_reference) << '\n';
            }
        }
        if (options.verify_only) return 0;
        if (options.csv) {
            std::cout << "device,device_name,compute_capability,logN,batch,core,organization,trial,kernel_ms,correct,"
                         "max_error,max_reference,relative_error,l2_error,l2_reference,relative_l2_error,points\n";
        } else {
            std::cout << "device=" << device << " device_name=" << properties.name
                      << " compute_capability=" << properties.major << '.' << properties.minor
                      << " logN=" << options.log_n << " batch=" << options.batch << '\n';
        }
        measure_variants(options.log_n, options.batch, options, input.value, output.value, ping.value, pong.value,
                         variants, device, properties);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
