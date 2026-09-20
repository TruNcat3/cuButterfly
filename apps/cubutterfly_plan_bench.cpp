#include <cubutterfly/cubutterfly.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cubutterflyStatus_t status, const char* operation) {
    if (status != CUBUTTERFLY_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + cubutterflyGetStatusString(status));
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

std::size_t checked_product(const std::vector<std::size_t>& values, std::size_t multiplier = 1) {
    std::size_t result = multiplier;
    for (const auto value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value)
            throw std::overflow_error("shape product overflows size_t");
        result *= value;
    }
    return result;
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double unit_from_hash(std::uint64_t value) {
    return static_cast<double>(mix64(value) >> 11) * (1.0 / 9007199254740992.0);
}

template <typename Complex, typename Scalar>
void fill_fft_input(std::vector<Complex>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto seed = 0x243f6a8885a308d3ULL + static_cast<std::uint64_t>(index);
        const double real = 0.25 * (unit_from_hash(seed) - 0.5);
        const double imag = 0.25 * (unit_from_hash(seed ^ 0x13198a2e03707344ULL) - 0.5);
        values[index] = Complex{static_cast<Scalar>(real), static_cast<Scalar>(imag)};
    }
}

float median(std::vector<float> values) {
    if (values.empty())
        throw std::invalid_argument("at least one timing trial is required");
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
        return (values[middle - 1] + values[middle]) * 0.5F;
    return values[middle];
}

struct VerificationMetrics {
    bool finite = true;
    double relative_max = 0.0;
    double relative_l2 = 0.0;
    double max_absolute = 0.0;
};

template <typename Complex>
VerificationMetrics compare_fft_output(const std::vector<Complex>& actual,
                                       const std::vector<Complex>& expected) {
    VerificationMetrics metrics;
    long double max_error = 0.0L;
    long double max_reference = 0.0L;
    long double squared_error = 0.0L;
    long double squared_reference = 0.0L;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const long double actual_real = static_cast<long double>(actual[index].x);
        const long double actual_imag = static_cast<long double>(actual[index].y);
        const long double expected_real = static_cast<long double>(expected[index].x);
        const long double expected_imag = static_cast<long double>(expected[index].y);
        if (!std::isfinite(actual_real) || !std::isfinite(actual_imag) ||
            !std::isfinite(expected_real) || !std::isfinite(expected_imag)) {
            metrics.finite = false;
            continue;
        }
        const long double real_error = actual_real - expected_real;
        const long double imag_error = actual_imag - expected_imag;
        const long double absolute_error = std::sqrt(real_error * real_error + imag_error * imag_error);
        const long double reference_magnitude =
            std::sqrt(expected_real * expected_real + expected_imag * expected_imag);
        max_error = std::max(max_error, absolute_error);
        max_reference = std::max(max_reference, reference_magnitude);
        squared_error += real_error * real_error + imag_error * imag_error;
        squared_reference += expected_real * expected_real + expected_imag * expected_imag;
    }
    if (!metrics.finite) {
        const auto infinity = std::numeric_limits<double>::infinity();
        metrics.relative_max = infinity;
        metrics.relative_l2 = infinity;
        metrics.max_absolute = infinity;
        return metrics;
    }
    metrics.relative_max = static_cast<double>(max_error / std::max(max_reference, 1.0e-30L));
    metrics.relative_l2 = static_cast<double>(std::sqrt(squared_error) /
        std::max(std::sqrt(squared_reference), 1.0e-30L));
    metrics.max_absolute = static_cast<double>(max_error);
    return metrics;
}

std::string take(int& index, int argc, char** argv) {
    if (++index >= argc)
        throw std::invalid_argument("missing command-line value");
    return argv[index];
}

std::vector<std::size_t> parse_shape(const std::string& text) {
    std::vector<std::size_t> extents;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find('x', begin);
        const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
        std::size_t consumed = 0;
        const auto extent = std::stoull(token, &consumed);
        if (consumed != token.size() || extent == 0)
            throw std::invalid_argument("shape extents must be positive integers");
        extents.push_back(extent);
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    if (extents.empty() || extents.size() > 2)
        throw std::invalid_argument("shape must be N or RowsxColumns");
    return extents;
}

std::array<double, 4> parse_matrix(const std::string& text) {
    std::array<double, 4> values{};
    std::size_t begin = 0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto end = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
        std::size_t consumed = 0;
        values[index] = std::stod(token, &consumed);
        if (consumed != token.size() || (index + 1 != values.size()) != (end != std::string::npos))
            throw std::invalid_argument("matrix must contain exactly four comma-separated values");
        begin = end == std::string::npos ? text.size() : end + 1;
    }
    return values;
}

void logger(cubutterflyLogLevel_t level, const char* event, const char* message, void*) {
    const char* label = level == CUBUTTERFLY_LOG_ERROR ? "error" :
                        (level == CUBUTTERFLY_LOG_WARNING ? "warning" : "info");
    std::cerr << "cubutterfly[" << label << "][" << event << "]: " << message << '\n';
}

void usage() {
    std::cout << "cubutterfly_plan_bench --operator fft|ntt|fwht|subset-zeta|superset-zeta|structured-2x2 "
                 "--shape N|RowsxColumns [--batch N] [--precision fp32|fp64|uint32|uint64] "
                 "[--length-mode standard|embedding] [--policy default|measure] [--algorithm ID] "
                 "[--matrix m00,m01,m10,m11] [--cache PATH] "
                 "[--inverse] [--in-place] [--compare-cufft] [--verify] [--trials N] [--warmup N] [--repeat N]\n";
}

}  // namespace

int main(int argc, char** argv) {
    cubutterflyOperator_t op = CUBUTTERFLY_OPERATOR_FFT;
    cubutterflyDataType_t storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
    cubutterflyComputeType_t compute = CUBUTTERFLY_COMPUTE_FP32;
    cubutterflyLengthMode_t length_mode = CUBUTTERFLY_LENGTH_STANDARD;
    cubutterflyAlgorithmPolicy_t policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
    cubutterflyPlacement_t placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    std::vector<std::size_t> shape{256};
    std::size_t batch = 1;
    std::uint64_t modulus = 1152921504606584833ULL;
    std::string cache;
    int warmup = 10;
    int repeat = 100;
    int trials = 1;
    bool inverse = false;
    bool compare_cufft = false;
    bool verify = false;
    std::string explicit_algorithm;
    std::array<double, 4> matrix{1.0, 0.25, -0.5, 1.0};
    try {
        for (int index = 1; index < argc; ++index) {
            const std::string arg = argv[index];
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else if (arg == "--shape") {
                shape = parse_shape(take(index, argc, argv));
            } else if (arg == "--batch") {
                batch = std::stoull(take(index, argc, argv));
            } else if (arg == "--warmup") {
                warmup = std::stoi(take(index, argc, argv));
            } else if (arg == "--repeat") {
                repeat = std::stoi(take(index, argc, argv));
            } else if (arg == "--trials") {
                trials = std::stoi(take(index, argc, argv));
            } else if (arg == "--cache") {
                cache = take(index, argc, argv);
            } else if (arg == "--modulus") {
                modulus = std::stoull(take(index, argc, argv));
            } else if (arg == "--algorithm") {
                explicit_algorithm = take(index, argc, argv);
                policy = CUBUTTERFLY_ALGORITHM_EXPLICIT;
            } else if (arg == "--matrix") {
                matrix = parse_matrix(take(index, argc, argv));
            } else if (arg == "--inverse") {
                inverse = true;
            } else if (arg == "--in-place") {
                placement = CUBUTTERFLY_PLACEMENT_IN_PLACE;
            } else if (arg == "--compare-cufft") {
                compare_cufft = true;
            } else if (arg == "--verify") {
                verify = true;
            } else if (arg == "--operator") {
                const auto value = take(index, argc, argv);
                if (value == "fft") op = CUBUTTERFLY_OPERATOR_FFT;
                else if (value == "ntt") op = CUBUTTERFLY_OPERATOR_NTT;
                else if (value == "fwht") op = CUBUTTERFLY_OPERATOR_FWHT;
                else if (value == "subset-zeta") op = CUBUTTERFLY_OPERATOR_SUBSET_ZETA;
                else if (value == "superset-zeta") op = CUBUTTERFLY_OPERATOR_SUPERSET_ZETA;
                else if (value == "structured-2x2") op = CUBUTTERFLY_OPERATOR_STRUCTURED_2X2;
                else throw std::invalid_argument("unknown operator");
            } else if (arg == "--precision") {
                const auto value = take(index, argc, argv);
                if (value == "fp32") { storage = op == CUBUTTERFLY_OPERATOR_FFT ? CUBUTTERFLY_DATA_COMPLEX_FP32 : CUBUTTERFLY_DATA_FP32; compute = CUBUTTERFLY_COMPUTE_FP32; }
                else if (value == "fp64") { storage = op == CUBUTTERFLY_OPERATOR_FFT ? CUBUTTERFLY_DATA_COMPLEX_FP64 : CUBUTTERFLY_DATA_FP64; compute = CUBUTTERFLY_COMPUTE_FP64; }
                else if (value == "uint32") { storage = CUBUTTERFLY_DATA_UINT32; compute = CUBUTTERFLY_COMPUTE_UINT32; }
                else if (value == "uint64") { storage = CUBUTTERFLY_DATA_UINT64; compute = CUBUTTERFLY_COMPUTE_UINT64; }
                else throw std::invalid_argument("unknown precision");
            } else if (arg == "--length-mode") {
                const auto value = take(index, argc, argv);
                if (value == "standard") length_mode = CUBUTTERFLY_LENGTH_STANDARD;
                else if (value == "embedding") length_mode = CUBUTTERFLY_LENGTH_ZERO_EXTENDED_EMBEDDING;
                else throw std::invalid_argument("unknown length mode");
            } else if (arg == "--policy") {
                const auto value = take(index, argc, argv);
                if (value == "default") policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
                else if (value == "measure") policy = CUBUTTERFLY_ALGORITHM_MEASURE;
                else throw std::invalid_argument("unknown policy");
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }
        if (op == CUBUTTERFLY_OPERATOR_NTT && storage != CUBUTTERFLY_DATA_UINT32 && storage != CUBUTTERFLY_DATA_UINT64) {
            storage = CUBUTTERFLY_DATA_UINT64;
            compute = CUBUTTERFLY_COMPUTE_UINT64;
        }
        if ((op == CUBUTTERFLY_OPERATOR_SUBSET_ZETA || op == CUBUTTERFLY_OPERATOR_SUPERSET_ZETA)) {
            storage = CUBUTTERFLY_DATA_UINT32;
            compute = CUBUTTERFLY_COMPUTE_UINT32;
        }
        if (op == CUBUTTERFLY_OPERATOR_FFT) {
            if (storage == CUBUTTERFLY_DATA_FP32) storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
            if (storage == CUBUTTERFLY_DATA_FP64) storage = CUBUTTERFLY_DATA_COMPLEX_FP64;
        } else {
            if (storage == CUBUTTERFLY_DATA_COMPLEX_FP32) storage = CUBUTTERFLY_DATA_FP32;
            if (storage == CUBUTTERFLY_DATA_COMPLEX_FP64) storage = CUBUTTERFLY_DATA_FP64;
        }
        if (warmup < 0 || repeat <= 0 || trials <= 0)
            throw std::invalid_argument("warmup must be nonnegative, repeat and trials must be positive");
        if (compare_cufft && (op != CUBUTTERFLY_OPERATOR_FFT || inverse ||
                              length_mode != CUBUTTERFLY_LENGTH_STANDARD))
            throw std::invalid_argument("cuFFT comparison currently requires a forward standard FFT");
        if (verify && (op != CUBUTTERFLY_OPERATOR_FFT || inverse ||
                       length_mode != CUBUTTERFLY_LENGTH_STANDARD ||
                       placement != CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE))
            throw std::invalid_argument("full cuFFT verification requires a forward standard out-of-place FFT");
        if (verify)
            compare_cufft = true;

        cubutterflyHandle_t handle = nullptr;
        cubutterflyDescriptor_t descriptor = nullptr;
        cubutterflyPlan_t plan = nullptr;
        cudaStream_t stream = nullptr;
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        check(cubutterflyCreate(&handle), "create handle");
        check(cubutterflySetStream(handle, stream), "set stream");
        check(cubutterflySetLogCallback(handle, logger, nullptr), "set logger");
        if (!cache.empty())
            check(cubutterflySetCachePath(handle, cache.c_str()), "set cache path");
        check(cubutterflyCreateDescriptor(&descriptor), "create descriptor");
        check(cubutterflySetOperator(descriptor, op), "set operator");
        check(cubutterflySetDataType(descriptor, storage, compute), "set data type");
        check(cubutterflySetShape(descriptor, static_cast<std::uint32_t>(shape.size()), shape.data(), batch), "set shape");
        check(cubutterflySetDirection(descriptor, inverse ? CUBUTTERFLY_DIRECTION_INVERSE : CUBUTTERFLY_DIRECTION_FORWARD, 1), "set direction");
        check(cubutterflySetPlacement(descriptor, placement), "set placement");
        check(cubutterflySetLengthMode(descriptor, length_mode), "set length mode");
        check(cubutterflySetAlgorithmPolicy(descriptor, policy,
                                            explicit_algorithm.empty() ? nullptr : explicit_algorithm.c_str()),
              "set algorithm policy");
        if (op == CUBUTTERFLY_OPERATOR_NTT)
            check(cubutterflySetModulus(descriptor, modulus, storage == CUBUTTERFLY_DATA_UINT32 ? 32 : 64), "set modulus");
        if (op == CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) {
            const cubutterflyMatrix2x2_t stage{matrix[0], matrix[1], matrix[2], matrix[3]};
            for (std::uint32_t axis = 0; axis < shape.size(); ++axis)
                check(cubutterflySetStageMatrices(descriptor, axis, &stage, 1), "set stage matrix");
        }
        check(cubutterflyCreatePlan(handle, descriptor, &plan), "create plan");

        std::size_t input_bytes = 0, output_bytes = 0, workspace_bytes = 0;
        check(cubutterflyPlanGetInputSize(plan, &input_bytes), "query input size");
        check(cubutterflyPlanGetOutputSize(plan, &output_bytes), "query output size");
        check(cubutterflyPlanGetWorkspaceSize(plan, &workspace_bytes), "query workspace size");
        void* input = nullptr;
        void* output = nullptr;
        void* workspace = nullptr;
        void* reference_input = nullptr;
        void* reference_output = nullptr;
        check_cuda(cudaMalloc(&input, std::max(input_bytes, output_bytes)), "allocate input");
        if (placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE)
            check_cuda(cudaMalloc(&output, output_bytes), "allocate output");
        else
            output = input;
        if (workspace_bytes != 0) {
            check_cuda(cudaMalloc(&workspace, workspace_bytes), "allocate workspace");
            check(cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes), "bind workspace");
        }

        std::vector<std::uint8_t> host_input;
        std::size_t fft_elements = 0;
        std::size_t expected_fft_bytes = 0;
        if (op == CUBUTTERFLY_OPERATOR_FFT) {
            fft_elements = checked_product(shape, batch);
            const std::size_t element_bytes =
                storage == CUBUTTERFLY_DATA_COMPLEX_FP64 ? sizeof(cufftDoubleComplex) : sizeof(cufftComplex);
            if (fft_elements > std::numeric_limits<std::size_t>::max() / element_bytes)
                throw std::overflow_error("FFT input size overflows size_t");
            expected_fft_bytes = fft_elements * element_bytes;
            if (input_bytes < expected_fft_bytes || output_bytes < expected_fft_bytes)
                throw std::runtime_error("FFT plan buffer size is smaller than its logical shape");
            host_input.assign(std::max(input_bytes, expected_fft_bytes), 0);
            if (storage == CUBUTTERFLY_DATA_COMPLEX_FP64) {
                std::vector<cufftDoubleComplex> values(fft_elements);
                fill_fft_input<cufftDoubleComplex, double>(values);
                std::memcpy(host_input.data(), values.data(), expected_fft_bytes);
            } else {
                std::vector<cufftComplex> values(fft_elements);
                fill_fft_input<cufftComplex, float>(values);
                std::memcpy(host_input.data(), values.data(), expected_fft_bytes);
            }
            check_cuda(cudaMemcpy(input, host_input.data(), input_bytes, cudaMemcpyHostToDevice),
                       "initialize FFT input");
        } else {
            check_cuda(cudaMemsetAsync(input, 0, input_bytes, stream), "initialize input");
        }

        cudaEvent_t start = nullptr, stop = nullptr;
        check_cuda(cudaEventCreate(&start), "create start event");
        check_cuda(cudaEventCreate(&stop), "create stop event");
        std::vector<float> kernel_trials;
        kernel_trials.reserve(static_cast<std::size_t>(trials));
        for (int trial = 0; trial < trials; ++trial) {
            for (int index = 0; index < warmup; ++index)
                check(cubutterflyExecute(handle, plan, input, output), "warmup execute");
            check_cuda(cudaEventRecord(start, stream), "record start");
            for (int index = 0; index < repeat; ++index)
                check(cubutterflyExecute(handle, plan, input, output), "timed execute");
            check_cuda(cudaEventRecord(stop, stream), "record stop");
            check_cuda(cudaEventSynchronize(stop), "wait for benchmark");
            float elapsed = 0.0F;
            check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "read benchmark time");
            kernel_trials.push_back(elapsed / repeat);
        }

        float cufft_ms = 0.0F;
        std::vector<float> cufft_trials;
        bool correct = true;
        double relative_error = 0.0;
        double relative_l2_error = 0.0;
        double max_absolute_error = 0.0;
        double tolerance = 0.0;
        std::size_t verified_batches = 0;
        if (compare_cufft) {
            check_cuda(cudaMalloc(&reference_input, input_bytes), "allocate cuFFT reference input");
            check_cuda(cudaMalloc(&reference_output, output_bytes), "allocate cuFFT reference output");
            check_cuda(cudaMemcpy(reference_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice),
                       "initialize cuFFT reference input");
            if (batch > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("cuFFT comparison batch exceeds int range");
            std::vector<int> dimensions(shape.size());
            std::size_t distance = 1;
            for (std::size_t axis = 0; axis < shape.size(); ++axis) {
                if (shape[axis] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("cuFFT comparison extent exceeds int range");
                dimensions[axis] = static_cast<int>(shape[axis]);
                if (distance > static_cast<std::size_t>(std::numeric_limits<int>::max()) / shape[axis])
                    throw std::invalid_argument("cuFFT comparison transform distance exceeds int range");
                distance *= shape[axis];
            }
            cufftHandle baseline = 0;
            check_cufft(cufftPlanMany(&baseline, static_cast<int>(dimensions.size()), dimensions.data(),
                                      nullptr, 1, static_cast<int>(distance), nullptr, 1,
                                      static_cast<int>(distance),
                                      storage == CUBUTTERFLY_DATA_COMPLEX_FP64 ? CUFFT_Z2Z : CUFFT_C2C,
                                      static_cast<int>(batch)), "create cuFFT comparison plan");
            check_cufft(cufftSetStream(baseline, stream), "set cuFFT comparison stream");
            auto execute_cufft = [&] {
                if (storage == CUBUTTERFLY_DATA_COMPLEX_FP64)
                    check_cufft(cufftExecZ2Z(baseline, static_cast<cufftDoubleComplex*>(reference_input),
                                             static_cast<cufftDoubleComplex*>(reference_output), CUFFT_FORWARD),
                                "execute cuFFT comparison");
                else
                    check_cufft(cufftExecC2C(baseline, static_cast<cufftComplex*>(reference_input),
                                             static_cast<cufftComplex*>(reference_output), CUFFT_FORWARD),
                                "execute cuFFT comparison");
            };
            cufft_trials.reserve(static_cast<std::size_t>(trials));
            for (int trial = 0; trial < trials; ++trial) {
                for (int index = 0; index < warmup; ++index)
                    execute_cufft();
                check_cuda(cudaEventRecord(start, stream), "record cuFFT start");
                for (int index = 0; index < repeat; ++index)
                    execute_cufft();
                check_cuda(cudaEventRecord(stop, stream), "record cuFFT stop");
                check_cuda(cudaEventSynchronize(stop), "wait for cuFFT benchmark");
                float elapsed = 0.0F;
                check_cuda(cudaEventElapsedTime(&elapsed, start, stop), "read cuFFT benchmark time");
                cufft_trials.push_back(elapsed / repeat);
            }
            cufft_ms = median(cufft_trials);

            if (verify) {
                verified_batches = batch;
                if (storage == CUBUTTERFLY_DATA_COMPLEX_FP64) {
                    std::vector<cufftDoubleComplex> actual(fft_elements);
                    std::vector<cufftDoubleComplex> expected(fft_elements);
                    check_cuda(cudaMemcpy(actual.data(), output, expected_fft_bytes, cudaMemcpyDeviceToHost),
                               "read cuButterfly output for verification");
                    check_cuda(cudaMemcpy(expected.data(), reference_output, expected_fft_bytes,
                                          cudaMemcpyDeviceToHost), "read cuFFT output for verification");
                    const auto metrics = compare_fft_output(actual, expected);
                    relative_error = metrics.relative_max;
                    relative_l2_error = metrics.relative_l2;
                    max_absolute_error = metrics.max_absolute;
                    tolerance = 1.0e-10;
                } else {
                    std::vector<cufftComplex> actual(fft_elements);
                    std::vector<cufftComplex> expected(fft_elements);
                    check_cuda(cudaMemcpy(actual.data(), output, expected_fft_bytes, cudaMemcpyDeviceToHost),
                               "read cuButterfly output for verification");
                    check_cuda(cudaMemcpy(expected.data(), reference_output, expected_fft_bytes,
                                          cudaMemcpyDeviceToHost), "read cuFFT output for verification");
                    const auto metrics = compare_fft_output(actual, expected);
                    relative_error = metrics.relative_max;
                    relative_l2_error = metrics.relative_l2;
                    max_absolute_error = metrics.max_absolute;
                    tolerance = 2.0e-4;
                }
                correct = std::isfinite(relative_error) && std::isfinite(relative_l2_error) &&
                          std::isfinite(max_absolute_error) && relative_error <= tolerance;
            }
            check_cufft(cufftDestroy(baseline), "destroy cuFFT comparison plan");
        }
        std::size_t name_bytes = 0;
        check(cubutterflyPlanGetAlgorithmName(plan, nullptr, &name_bytes), "query algorithm name size");
        std::vector<char> algorithm(name_bytes);
        check(cubutterflyPlanGetAlgorithmName(plan, algorithm.data(), &name_bytes), "query algorithm name");
        std::vector<std::size_t> physical(shape.size());
        check(cubutterflyPlanGetPhysicalExtents(plan, physical.data(), static_cast<std::uint32_t>(physical.size())), "query physical shape");
        std::cout << "algorithm=" << algorithm.data() << ",logical_shape=";
        for (std::size_t axis = 0; axis < shape.size(); ++axis)
            std::cout << (axis ? "x" : "") << shape[axis];
        std::cout << ",physical_shape=";
        for (std::size_t axis = 0; axis < physical.size(); ++axis)
            std::cout << (axis ? "x" : "") << physical[axis];
        const float kernel_ms = median(kernel_trials);
        const bool verification_failed = verify && !correct;
        std::cout << std::setprecision(9)
                  << ",batch=" << batch << ",workspace_bytes=" << workspace_bytes
                  << ",trials=" << trials << ",kernel_ms=" << kernel_ms
                  << ",trial_kernel_ms=";
        for (std::size_t index = 0; index < kernel_trials.size(); ++index)
            std::cout << (index ? ":" : "") << kernel_trials[index];
        if (compare_cufft)
            std::cout << ",cufft_ms=" << cufft_ms << ",trial_cufft_ms=";
        if (compare_cufft) {
            for (std::size_t index = 0; index < cufft_trials.size(); ++index)
                std::cout << (index ? ":" : "") << cufft_trials[index];
            std::cout << ",speedup_vs_cufft=" << cufft_ms / kernel_ms;
        }
        if (verify)
            std::cout << ",verified_batches=" << verified_batches
                      << ",correct=" << (correct ? 1 : 0)
                      << ",relative_error=" << relative_error
                      << ",relative_l2_error=" << relative_l2_error
                      << ",max_absolute_error=" << max_absolute_error
                      << ",tolerance=" << tolerance;
        std::cout << '\n';

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(workspace);
        cudaFree(reference_output);
        cudaFree(reference_input);
        if (placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE)
            cudaFree(output);
        cudaFree(input);
        cubutterflyDestroyPlan(plan);
        cubutterflyDestroyDescriptor(descriptor);
        cubutterflyDestroy(handle);
        cudaStreamDestroy(stream);
        if (verification_failed)
            throw std::runtime_error("cuFFT full-output verification failed");
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
