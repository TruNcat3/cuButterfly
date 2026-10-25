#pragma once
#include <cubutterfly/cubutterfly.h>
#include <cubutterfly/mapping.hpp>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cubutterfly {

inline void check(cubutterflyStatus_t status) {
    if (status != CUBUTTERFLY_STATUS_SUCCESS) {
        const char* detail = cubutterflyGetLastErrorString();
        const std::string saved = detail ? detail : "";
        const std::string summary = cubutterflyGetStatusString(status);
        throw std::runtime_error(saved.empty() ? summary : summary + ": " + saved);
    }
}

// All six applications share this semantic descriptor. Execution mapping is a
// separate planner choice; no device name or backend default belongs here.
struct Transform {
    cubutterflyOperator_t op = CUBUTTERFLY_OPERATOR_FFT;
    cubutterflyDataType_t storage = CUBUTTERFLY_DATA_COMPLEX_FP32;
    cubutterflyComputeType_t compute = CUBUTTERFLY_COMPUTE_NATIVE;
    std::vector<std::size_t> extents{256};
    std::size_t batch = 1;
    std::vector<std::size_t> input_strides, output_strides;
    std::size_t input_batch_stride = 0, output_batch_stride = 0;
    bool inverse = false, normalize_inverse = true, in_place = false;
    cubutterflyLengthMode_t length_mode = CUBUTTERFLY_LENGTH_STANDARD;
    std::uint64_t modulus = cuntt::kDefaultModulus;
    std::uint32_t word_bits = 64;
    cubutterflyNttLayout_t input_order = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    cubutterflyNttLayout_t output_order = CUBUTTERFLY_NTT_LAYOUT_NATURAL;
    std::vector<cubutterflyLogicalOrder_t> input_orders, output_orders;
    std::vector<std::vector<cubutterflyMatrix2x2_t>> stage_matrices;
    /* Convenience mode for callers that keep lowering policy with the
     * workload description.  PlanOptions can override it. */
    cubutterflyExecutionMode_t execution_mode = CUBUTTERFLY_EXECUTION_AUTO;
    cubutterflyFftTransformType_t fft_type = CUBUTTERFLY_FFT_C2C;
    double fft_output_scale = 1.0;
    double output_scale = 1.0;
    cubutterflyOutputRounding_t output_rounding = CUBUTTERFLY_ROUNDING_STAGE_STORAGE;
};

struct PlanOptions {
    cubutterflyAlgorithmPolicy_t policy = CUBUTTERFLY_ALGORITHM_DEFAULT;
    std::string mapping;  // Complete JSON from a previous plan, or a legacy explicit ID.
    bool allocate_workspace = true;
    /* AUTO means "use Transform::execution_mode". */
    cubutterflyExecutionMode_t execution_mode = CUBUTTERFLY_EXECUTION_AUTO;
};

class Context {
    struct State {
        cubutterflyHandle_t handle = nullptr;
        State() { check(cubutterflyCreate(&handle)); }
        ~State() { cubutterflyDestroy(handle); }
    };
    std::shared_ptr<State> state_ = std::make_shared<State>();
  public:
    void set_stream(cudaStream_t stream) { check(cubutterflySetStream(state_->handle, stream)); }
    void set_cache_path(const std::string& path) { check(cubutterflySetCachePath(state_->handle, path.c_str())); }
    void set_logger(cubutterflyLogCallback_t logger, void* user = nullptr) {
        check(cubutterflySetLogCallback(state_->handle, logger, user));
    }
    cubutterflyHandle_t native_handle() const noexcept { return state_->handle; }
};

class Descriptor {
    using Owner = std::unique_ptr<std::remove_pointer_t<cubutterflyDescriptor_t>,
                                  decltype(&cubutterflyDestroyDescriptor)>;
    Owner descriptor_{nullptr, cubutterflyDestroyDescriptor};
    Descriptor() {
        cubutterflyDescriptor_t raw = nullptr;
        check(cubutterflyCreateDescriptor(&raw));
        descriptor_.reset(raw);
    }
    using StringQuery = cubutterflyStatus_t (*)(cubutterflyDescriptor_t, char*, std::size_t*);
    std::string query(StringQuery function) const {
        std::size_t bytes = 0;
        check(function(native_handle(), nullptr, &bytes));
        std::string result(bytes, '\0');
        check(function(native_handle(), result.data(), &bytes));
        if (!result.empty() && result.back() == '\0') result.pop_back();
        return result;
    }
  public:
    Descriptor(const Transform& transform, const PlanOptions& options = {}) : Descriptor() {
        const auto raw = native_handle();
        const auto rank = transform.extents.size();
        if (rank > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("rank exceeds the uint32 C ABI range");
        auto axis_size = [rank](std::size_t count) {
            if (count && count != rank)
                throw std::invalid_argument("one entry is required per axis");
        };
        axis_size(transform.input_strides.size());
        axis_size(transform.output_strides.size());
        axis_size(transform.input_orders.size());
        axis_size(transform.output_orders.size());
        const bool legacy = transform.input_order != CUBUTTERFLY_NTT_LAYOUT_NATURAL ||
                            transform.output_order != CUBUTTERFLY_NTT_LAYOUT_NATURAL;
        if (legacy && transform.op != CUBUTTERFLY_OPERATOR_NTT)
            throw std::invalid_argument("legacy NTT layouts require an NTT operator");
        if (transform.op != CUBUTTERFLY_OPERATOR_NTT &&
            (transform.modulus != cuntt::kDefaultModulus || transform.word_bits != 64))
            throw std::invalid_argument("modulus and word_bits require an NTT operator");
        if (legacy && (!transform.input_orders.empty() || !transform.output_orders.empty()))
            throw std::invalid_argument("legacy NTT layouts and logical orders are ambiguous");
        check(cubutterflySetOperator(raw, transform.op));
        check(cubutterflySetDataType(raw, transform.storage, transform.compute));
        check(cubutterflySetShape(raw, rank, transform.extents.data(), transform.batch));
        check(cubutterflySetExecutionMode(raw, options.execution_mode != CUBUTTERFLY_EXECUTION_AUTO
              ? options.execution_mode : transform.execution_mode));
        check(cubutterflySetFftTransformType(raw, transform.fft_type));
        check(cubutterflySetFftOutputScale(raw, transform.fft_output_scale));
        check(cubutterflySetOutputScale(raw, transform.output_scale));
        check(cubutterflySetOutputRounding(raw, transform.output_rounding));
        check(cubutterflySetStrides(raw,
              transform.input_strides.empty() ? nullptr : transform.input_strides.data(), transform.input_batch_stride,
              transform.output_strides.empty() ? nullptr : transform.output_strides.data(), transform.output_batch_stride));
        check(cubutterflySetDirection(raw, transform.inverse ? CUBUTTERFLY_DIRECTION_INVERSE : CUBUTTERFLY_DIRECTION_FORWARD,
                                     transform.normalize_inverse));
        check(cubutterflySetPlacement(raw, transform.in_place ? CUBUTTERFLY_PLACEMENT_IN_PLACE : CUBUTTERFLY_PLACEMENT_OUT_OF_PLACE));
        check(cubutterflySetLengthMode(raw, transform.length_mode));
        check(cubutterflySetAlgorithmPolicy(raw, options.mapping.empty() ? options.policy : CUBUTTERFLY_ALGORITHM_EXPLICIT,
                                           options.mapping.c_str()));
        if (transform.op == CUBUTTERFLY_OPERATOR_NTT) {
            check(cubutterflySetModulus(raw, transform.modulus, transform.word_bits));
            check(cubutterflySetNttLayouts(raw, transform.input_order, transform.output_order));
        }
        if (!legacy) check(cubutterflySetLogicalOrders(raw,
            transform.input_orders.empty() ? nullptr : transform.input_orders.data(),
            transform.output_orders.empty() ? nullptr : transform.output_orders.data()));
        for (std::size_t axis = 0; axis < transform.stage_matrices.size(); ++axis) {
            const auto& matrices = transform.stage_matrices[axis];
            if (matrices.empty()) continue;
            if (axis > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("matrix axis exceeds the uint32 C ABI range");
            check(cubutterflySetStageMatrices(raw, axis, matrices.data(), matrices.size()));
        }
    }
    Descriptor(Descriptor&&) noexcept = default;
    Descriptor& operator=(Descriptor&&) noexcept = default;
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    static Descriptor from_json(const std::string& json) {
        Descriptor result;
        check(cubutterflyDescriptorSetJson(result.native_handle(), json.c_str()));
        return result;
    }
    void validate() const { check(cubutterflyDescriptorValidate(native_handle())); }
    std::string json() const { return query(cubutterflyDescriptorGetJson); }
    std::string capabilities_json() const { return query(cubutterflyDescriptorGetCapabilities); }
    cubutterflyDescriptor_t native_handle() const noexcept { return descriptor_.get(); }
};

class Plan {
    struct State {
        Context context;
        cubutterflyPlan_t handle = nullptr;
        void* workspace = nullptr;
        explicit State(const Context& value) : context(value) {}
        ~State() { cubutterflyDestroyPlan(handle); if (workspace) cudaFree(workspace); }
    };
    std::unique_ptr<State> state_;
    using StringQuery = cubutterflyStatus_t (*)(cubutterflyPlan_t, char*, std::size_t*);
    std::string query(StringQuery function) const {
        std::size_t bytes = 0;
        check(function(state_->handle, nullptr, &bytes));
        std::string value(bytes, '\0');
        check(function(state_->handle, value.data(), &bytes));
        if (!value.empty()) value.pop_back();
        return value;
    }
  public:
    Plan(const Context& context, const Transform& transform, const PlanOptions& options = {})
        : Plan(context, Descriptor(transform, options), options.allocate_workspace) {}
    Plan(const Context& context, const Descriptor& descriptor, bool allocate_workspace = true)
        : state_(std::make_unique<State>(context)) {
        check(cubutterflyCreatePlan(context.native_handle(), descriptor.native_handle(), &state_->handle));
        const auto bytes = workspace_size();
        if (allocate_workspace && bytes) {
            const auto status = cudaMalloc(&state_->workspace, bytes);
            if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
            check(cubutterflyPlanSetWorkspace(state_->handle, state_->workspace, bytes));
        }
    }
    Plan(Plan&&) noexcept = default;
    Plan& operator=(Plan&&) noexcept = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    std::size_t workspace_size() const { std::size_t n; check(cubutterflyPlanGetWorkspaceSize(state_->handle, &n)); return n; }
    std::size_t input_size() const { std::size_t n; check(cubutterflyPlanGetInputSize(state_->handle, &n)); return n; }
    std::size_t output_size() const { std::size_t n; check(cubutterflyPlanGetOutputSize(state_->handle, &n)); return n; }
    void set_workspace(void* pointer, std::size_t bytes) { check(cubutterflyPlanSetWorkspace(state_->handle, pointer, bytes)); }
    void execute(const void* input, void* output) {
        check(cubutterflyExecute(state_->context.native_handle(), state_->handle, input, output));
    }
    void execute_steady(const void* input, void* output) {
        check(cubutterflyExecuteSteady(state_->context.native_handle(), state_->handle, input, output));
    }
    std::string mapping_json() const { return query(cubutterflyPlanGetMappingJson); }
    std::string descriptor_json() const { return query(cubutterflyPlanGetDescriptorJson); }
    std::string selection_reason() const { return query(cubutterflyPlanGetSelectionReason); }
    std::string algorithm_name() const { return query(cubutterflyPlanGetAlgorithmName); }
    cubutterflySelectionSource_t selection_source() const {
        cubutterflySelectionSource_t source;
        check(cubutterflyPlanGetSelectionSource(state_->handle, &source));
        return source;
    }
    cubutterflyPlan_t native_handle() const noexcept { return state_->handle; }
};
} // namespace cubutterfly
