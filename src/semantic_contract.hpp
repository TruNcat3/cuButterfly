#pragma once

#include "cubutterfly/cubutterfly.h"
#include "cuntt/ntt.hpp"
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct cubutterflyDescriptor {
    cubutterflyOperator_t op = CUBUTTERFLY_OPERATOR_FFT;
    cubutterflyDataType_t storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
    cubutterflyComputeType_t compute = CUBUTTERFLY_COMPUTE_FP32;
    cubutterflyDirection_t direction = CUBUTTERFLY_DIRECTION_FORWARD;
    cubutterflyPlacement_t placement = CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE;
    cubutterflyLengthMode_t length_mode = CUBUTTERFLY_LENGTH_STANDARD;
    cubutterflyExecutionMode_t execution_mode = CUBUTTERFLY_EXECUTION_AUTO;
    cubutterflyFftTransformType_t fft_type = CUBUTTERFLY_FFT_C2C;
    double fft_output_scale = 1.0;
    double output_scale = 1.0;
    cubutterflyOutputRounding_t output_rounding = CUBUTTERFLY_ROUNDING_STAGE_STORAGE;
    cubutterflyAlgorithmPolicy_t algorithm_policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
    std::string explicit_algorithm;
    std::uint32_t rank = 1;
    std::vector<std::size_t> extents{256};
    std::vector<std::size_t> input_strides;
    std::vector<std::size_t> output_strides;
    std::size_t input_batch_stride = 0;
    std::size_t output_batch_stride = 0;
    std::size_t batch = 1;
    bool normalize_inverse = true;
    std::uint64_t modulus = cuntt::kDefaultModulus;
    std::uint32_t word_bits = 64;
    cubutterflyNttLayout_t input_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    cubutterflyNttLayout_t output_ntt_layout = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    std::vector<std::vector<cubutterflyMatrix2x2_t>> stage_matrices{3};
    std::vector<cubutterflyLogicalOrder_t> input_orders;
    std::vector<cubutterflyLogicalOrder_t> output_orders;
};

namespace cubutterfly::detail {
cubutterflyDescriptor normalize_descriptor(const cubutterflyDescriptor& descriptor);
nlohmann::json semantic_json(const cubutterflyDescriptor& descriptor);
nlohmann::json descriptor_json(const cubutterflyDescriptor& descriptor);
cubutterflyDescriptor descriptor_from_json(const nlohmann::json& json);
std::vector<std::size_t> semantic_input_extents(const cubutterflyDescriptor& descriptor);
std::vector<std::size_t> semantic_output_extents(const cubutterflyDescriptor& descriptor);
// False can mean either overlap or that the bounded check could not prove it.
bool layout_is_nonoverlapping(const std::vector<std::size_t>& extents,
                              const std::vector<std::size_t>& strides,
                              std::size_t batch, std::size_t batch_stride);
}
