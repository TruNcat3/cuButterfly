#include <cubutterfly/cubutterfly.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <array>
#include <cstdint>
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
                 "[--inverse] [--in-place] [--compare-cufft] [--warmup N] [--repeat N]\n";
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
    bool inverse = false;
    bool compare_cufft = false;
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
        if (warmup < 0 || repeat <= 0)
            throw std::invalid_argument("warmup must be nonnegative and repeat must be positive");
        if (compare_cufft && (op != CUBUTTERFLY_OPERATOR_FFT || inverse ||
                              length_mode != CUBUTTERFLY_LENGTH_STANDARD))
            throw std::invalid_argument("cuFFT comparison currently requires a forward standard FFT");

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
        check_cuda(cudaMalloc(&input, std::max(input_bytes, output_bytes)), "allocate input");
        if (placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE)
            check_cuda(cudaMalloc(&output, output_bytes), "allocate output");
        else
            output = input;
        if (workspace_bytes != 0) {
            check_cuda(cudaMalloc(&workspace, workspace_bytes), "allocate workspace");
            check(cubutterflyPlanSetWorkspace(plan, workspace, workspace_bytes), "bind workspace");
        }
        check_cuda(cudaMemsetAsync(input, 0, input_bytes, stream), "initialize input");
        for (int index = 0; index < warmup; ++index)
            check(cubutterflyExecute(handle, plan, input, output), "warmup execute");
        cudaEvent_t start = nullptr, stop = nullptr;
        check_cuda(cudaEventCreate(&start), "create start event");
        check_cuda(cudaEventCreate(&stop), "create stop event");
        check_cuda(cudaEventRecord(start, stream), "record start");
        for (int index = 0; index < repeat; ++index)
            check(cubutterflyExecute(handle, plan, input, output), "timed execute");
        check_cuda(cudaEventRecord(stop, stream), "record stop");
        check_cuda(cudaEventSynchronize(stop), "wait for benchmark");
        float total_ms = 0.0F;
        check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "read benchmark time");
        float cufft_ms = 0.0F;
        if (compare_cufft) {
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
                    check_cufft(cufftExecZ2Z(baseline, static_cast<cufftDoubleComplex*>(input),
                                             static_cast<cufftDoubleComplex*>(output), CUFFT_FORWARD),
                                "execute cuFFT comparison");
                else
                    check_cufft(cufftExecC2C(baseline, static_cast<cufftComplex*>(input),
                                             static_cast<cufftComplex*>(output), CUFFT_FORWARD),
                                "execute cuFFT comparison");
            };
            for (int index = 0; index < warmup; ++index)
                execute_cufft();
            check_cuda(cudaEventRecord(start, stream), "record cuFFT start");
            for (int index = 0; index < repeat; ++index)
                execute_cufft();
            check_cuda(cudaEventRecord(stop, stream), "record cuFFT stop");
            check_cuda(cudaEventSynchronize(stop), "wait for cuFFT benchmark");
            check_cuda(cudaEventElapsedTime(&cufft_ms, start, stop), "read cuFFT benchmark time");
            cufft_ms /= repeat;
            cufftDestroy(baseline);
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
        std::cout << ",batch=" << batch << ",workspace_bytes=" << workspace_bytes
                  << ",kernel_ms=" << total_ms / repeat;
        if (compare_cufft)
            std::cout << ",cufft_ms=" << cufft_ms << ",speedup_vs_cufft="
                      << cufft_ms / (total_ms / repeat);
        std::cout << '\n';

        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        cudaFree(workspace);
        if (placement == CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE)
            cudaFree(output);
        cudaFree(input);
        cubutterflyDestroyPlan(plan);
        cubutterflyDestroyDescriptor(descriptor);
        cubutterflyDestroy(handle);
        cudaStreamDestroy(stream);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
