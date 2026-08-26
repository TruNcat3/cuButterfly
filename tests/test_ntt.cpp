#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"

namespace {

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

class TestStream {
  public:
    TestStream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create test stream"); }
    ~TestStream() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

class TestDeviceBuffer {
  public:
    explicit TestDeviceBuffer(std::size_t bytes) { check_cuda(cudaMalloc(&pointer_, bytes), "allocate test device buffer"); }
    ~TestDeviceBuffer() { cudaFree(pointer_); }

    template <typename T>
    T* as() const noexcept {
        return static_cast<T*>(pointer_);
    }

    void* data() const noexcept { return pointer_; }

  private:
    void* pointer_ = nullptr;
};

std::vector<std::uint64_t> make_input(std::size_t count, std::uint64_t modulus, std::uint64_t seed) {
    std::mt19937_64            random(seed);
    std::vector<std::uint64_t> values(count);
    for (auto& value : values) {
        value = random() % modulus;
    }
    return values;
}

void require_equal(const std::vector<std::uint64_t>& expected, const std::vector<std::uint64_t>& actual, const std::string& context) {
    if (expected.size() != actual.size()) {
        throw std::runtime_error(context + " size mismatch: expected " + std::to_string(expected.size()) + ", got " + std::to_string(actual.size()));
    }
    const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin());
    if (mismatch.first == expected.end()) {
        return;
    }
    const std::size_t index = static_cast<std::size_t>(mismatch.first - expected.begin());
    throw std::runtime_error(context + " mismatch at index " + std::to_string(index) + ": expected " + std::to_string(*mismatch.first) + ", got " +
                             std::to_string(*mismatch.second));
}

std::vector<std::uint64_t> reference_batch(const std::vector<std::uint64_t>& input, std::uint32_t log_n, std::size_t batch, std::uint64_t modulus,
                                           bool inverse) {
    const std::size_t          n = 1ULL << log_n;
    std::vector<std::uint64_t> expected(input.size());
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        std::vector<std::uint64_t> slice(input.begin() + batch_index * n, input.begin() + (batch_index + 1) * n);
        cuntt::reference_ntt(slice, modulus, inverse);
        std::copy(slice.begin(), slice.end(), expected.begin() + batch_index * n);
    }
    return expected;
}

std::uint32_t reverse_bits(std::uint32_t value, std::uint32_t bits) {
    std::uint32_t reversed = 0;
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
        reversed = (reversed << 1) | ((value >> bit) & 1U);
    }
    return reversed;
}

template <typename Callable>
void require_invalid_argument(Callable&& callable, const std::string& context) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(context + " did not throw std::invalid_argument");
}

void test_compact_stage_backend() {
    constexpr std::uint32_t kLogN    = 20;
    constexpr std::uint64_t kModulus = 1152921504577486849ULL;
    const std::size_t       n        = 1ULL << kLogN;
    const auto              input    = make_input(n, kModulus, 0xc000);
    const auto              expected = reference_batch(input, kLogN, 1, kModulus, false);

    for (const auto order : {cuntt::OutputOrder::Natural, cuntt::OutputOrder::BitReversed}) {
        cuntt::PlanConfig config;
        config.log_n        = kLogN;
        config.modulus      = kModulus;
        config.backend      = cuntt::Backend::CompactStage;
        config.output_order = order;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);

        if (order == cuntt::OutputOrder::Natural) {
            require_equal(expected, output, "compact-stage natural output");
        } else {
            std::vector<std::uint64_t> bit_reversed(n);
            for (std::uint32_t index = 0; index < n; ++index) {
                bit_reversed[index] = expected[reverse_bits(index, kLogN)];
            }
            require_equal(bit_reversed, output, "compact-stage bit-reversed output");
        }
        std::cout << "PASS compact-stage output_order=" << cuntt::output_order_name(order) << '\n';
    }
}

void test_device_api() {
    constexpr std::uint32_t kLogN  = 12;
    constexpr std::size_t   kBatch = 2;
    const auto input    = make_input((1ULL << kLogN) * kBatch, cuntt::kDefaultModulus, 0xd300);
    const auto expected = reference_batch(input, kLogN, kBatch, cuntt::kDefaultModulus, false);

    cuntt::PlanConfig config;
    config.log_n                    = kLogN;
    config.batch                    = kBatch;
    config.backend                  = cuntt::Backend::Hybrid2D;
    config.auto_allocate_workspace = false;
    cuntt::Plan plan(config);
    if (plan.data_size() != input.size() * sizeof(std::uint64_t) || plan.workspace_size() != plan.data_size() || plan.workspace() != nullptr) {
        throw std::runtime_error("hybrid2d device API size query mismatch");
    }

    TestStream       stream;
    TestDeviceBuffer device_input(plan.data_size());
    TestDeviceBuffer device_output(plan.data_size());
    TestDeviceBuffer workspace(plan.workspace_size());
    plan.set_stream(stream.get());
    if (plan.stream() != stream.get()) {
        throw std::runtime_error("NTT stream getter mismatch");
    }

    try {
        plan.execute_async(device_input.as<std::uint64_t>(), device_output.as<std::uint64_t>());
        throw std::runtime_error("missing NTT workspace was accepted");
    } catch (const std::logic_error&) {
    }
    require_invalid_argument([&] { plan.set_workspace(workspace.data(), plan.workspace_size() - 1); }, "undersized NTT workspace");
    plan.set_workspace(workspace.data(), plan.workspace_size());

    std::vector<std::uint64_t> output(input.size());
    check_cuda(cudaMemcpyAsync(device_input.data(), input.data(), plan.data_size(), cudaMemcpyHostToDevice, stream.get()), "queue NTT input copy");
    plan.execute_async(device_input.as<std::uint64_t>(), device_output.as<std::uint64_t>());
    check_cuda(cudaMemcpyAsync(output.data(), device_output.data(), plan.data_size(), cudaMemcpyDeviceToHost, stream.get()), "queue NTT output copy");
    check_cuda(cudaStreamSynchronize(stream.get()), "synchronize NTT test stream");
    require_equal(expected, output, "hybrid2d asynchronous device API");

    require_invalid_argument([&] { plan.execute_async(device_input.as<std::uint32_t>(), device_output.as<std::uint32_t>()); },
                             "NTT device pointer width mismatch");
    require_invalid_argument([&] { plan.execute_async(device_input.as<std::uint64_t>(), device_input.as<std::uint64_t>()); },
                             "NTT device pointer alias");

    cuntt::PlanConfig pipeline_config;
    pipeline_config.log_n                    = 8;
    pipeline_config.batch                    = 1;
    pipeline_config.backend                  = cuntt::Backend::StagePipeline;
    pipeline_config.auto_allocate_workspace = false;
    cuntt::Plan pipeline_plan(pipeline_config);
    if (pipeline_plan.workspace_size() != 0 || pipeline_plan.workspace() != nullptr) {
        throw std::runtime_error("stage-pipeline unexpectedly requires workspace");
    }
    std::cout << "PASS NTT asynchronous device API\n";
}

void test_runtime_selector() {
    cuntt::PlanConfig config;
    config.log_n       = 16;
    config.batch       = 4;
    config.word_bits   = 64;
    config.output_order = cuntt::OutputOrder::Natural;
    config.auto_select = true;
    cuntt::Plan plan(config);
    if (!plan.config().auto_select || plan.config().backend != cuntt::Backend::Hybrid2D ||
        plan.config().compute_unit != cuntt::ComputeUnit::Radix4 || !plan.selection().calibrated ||
        plan.selection().implementation != "cuNTT-Hybrid2D-radix4" || plan.selection().predicted_kernel_ms <= 0.0) {
        throw std::runtime_error("NTT runtime selector did not resolve the calibrated Hybrid2D mapping");
    }

    constexpr std::uint64_t kModulus30 = 1073479681ULL;
    config.log_n       = 12;
    config.batch       = 16;
    config.word_bits   = 32;
    config.modulus     = kModulus30;
    cuntt::Plan word32_plan(config);
    if (word32_plan.config().compute_unit != cuntt::ComputeUnit::Radix4 || word32_plan.config().word_bits != 32) {
        throw std::runtime_error("NTT runtime selector did not preserve the 32-bit semantic contract");
    }

    config.log_n    = 14;
    config.word_bits = 64;
    config.modulus   = cuntt::kDefaultModulus;
    try {
        cuntt::Plan unsupported(config);
        throw std::runtime_error("runtime selector accepted an uncalibrated NTT length");
    } catch (const std::invalid_argument&) {
    }
    std::cout << "PASS calibrated NTT runtime selector\n";
}

void test_stage_pipeline_backend() {
    constexpr std::uint32_t kLogN    = 8;
    constexpr std::size_t   kBatch   = 17;
    const auto              input    = make_input((1ULL << kLogN) * kBatch, cuntt::kDefaultModulus, 0x5a6e);
    const auto              expected = reference_batch(input, kLogN, kBatch, cuntt::kDefaultModulus, false);

    for (const auto handoff : {cuntt::StageHandoff::Atomic, cuntt::StageHandoff::NamedBarrier}) {
        for (const std::uint32_t stage_space : {1U, 2U, 4U, 8U}) {
            cuntt::PlanConfig config;
            config.log_n         = kLogN;
            config.batch         = kBatch;
            config.backend       = cuntt::Backend::StagePipeline;
            config.stage_space   = stage_space;
            config.stage_handoff = handoff;
            cuntt::Plan                plan(config);
            std::vector<std::uint64_t> output;
            plan.execute(input, output, 0, 1);
            require_equal(expected, output, "stage-pipeline Us=" + std::to_string(stage_space));
            std::cout << "PASS stage-pipeline handoff=" << cuntt::stage_handoff_name(handoff) << " Us=" << stage_space << " batch=" << kBatch << '\n';
        }
    }
}

void test_hybrid_dataflow_backend() {
    struct TestCase {
        std::uint32_t log_n;
        std::size_t batch;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
    };
    constexpr std::uint64_t kModulus30 = 1073479681ULL;
    constexpr std::array<TestCase, 5> kCases = {{{6, 17, 64, cuntt::kDefaultModulus, false},
                                                 {8, 3, 64, cuntt::kDefaultModulus, true},
                                                 {10, 2, 64, cuntt::kDefaultModulus, false},
                                                 {8, 5, 32, kModulus30, false},
                                                 {10, 2, 32, kModulus30, true}}};

    for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
        const auto& test_case = kCases[case_index];
        const auto input = make_input((1ULL << test_case.log_n) * test_case.batch, test_case.modulus,
                                      0x7000 + case_index);
        const auto expected = reference_batch(input, test_case.log_n, test_case.batch,
                                              test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.batch = test_case.batch;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        config.backend = cuntt::Backend::HybridDataflow;
        cuntt::Plan plan(config);
        if (test_case.log_n == 10 &&
            (plan.config().role_stages != 5 || plan.config().data_time != 10 ||
             plan.config().token_interleave != 1 || plan.config().compute_unit != cuntt::ComputeUnit::Radix4)) {
            throw std::runtime_error("V100 HybridDataflow logN=10 default did not select Ur=5,Td=10,Ti=1,radix4");
        }
        if (plan.workspace_size() != 0 || plan.workspace() != nullptr) {
            throw std::runtime_error("hybrid-dataflow unexpectedly requires global workspace");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hybrid-dataflow primary point");
        std::cout << "PASS hybrid-dataflow logN=" << test_case.log_n << " batch=" << test_case.batch
                  << " word_bits=" << test_case.word_bits
                  << " direction=" << (test_case.inverse ? "inverse" : "forward") << '\n';
    }

    struct Ablation {
        cuntt::DataflowLayout layout;
        cuntt::DataflowStateMode state;
        cuntt::StageHandoff handoff;
        std::uint32_t buffers;
    };
    constexpr std::array<Ablation, 4> kAblations = {{
        {cuntt::DataflowLayout::Linear, cuntt::DataflowStateMode::InPlace,
         cuntt::StageHandoff::NamedBarrier, 2},
        {cuntt::DataflowLayout::HermesXor, cuntt::DataflowStateMode::PingPong,
         cuntt::StageHandoff::NamedBarrier, 2},
        {cuntt::DataflowLayout::HermesXor, cuntt::DataflowStateMode::InPlace,
         cuntt::StageHandoff::Atomic, 2},
        {cuntt::DataflowLayout::HermesXor, cuntt::DataflowStateMode::InPlace,
         cuntt::StageHandoff::NamedBarrier, 3},
    }};
    const auto input = make_input(1U << 8, cuntt::kDefaultModulus, 0x7100);
    const auto expected = reference_batch(input, 8, 1, cuntt::kDefaultModulus, false);
    for (const auto& ablation : kAblations) {
        cuntt::PlanConfig config;
        config.log_n = 8;
        config.backend = cuntt::Backend::HybridDataflow;
        config.flow_tile_log_n = 8;
        config.stage_space = 8;
        config.data_space = 16;
        config.pipeline_buffers = ablation.buffers;
        config.dataflow_layout = ablation.layout;
        config.dataflow_state_mode = ablation.state;
        config.stage_handoff = ablation.handoff;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hybrid-dataflow ablation");
    }
    std::cout << "PASS hybrid-dataflow generated ablations\n";

    const auto packed_input = make_input(1U << 10, cuntt::kDefaultModulus, 0x7200);
    const auto packed_expected = reference_batch(packed_input, 10, 1, cuntt::kDefaultModulus, false);
    for (const std::uint32_t data_time : {1U, 2U, 4U, 8U}) {
        cuntt::PlanConfig config;
        config.log_n = 10;
        config.backend = cuntt::Backend::HybridDataflow;
        config.flow_tile_log_n = 5;
        config.stage_space = 5;
        config.data_space = 16;
        config.data_time = data_time;
        config.pipeline_buffers = 2;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(packed_input, output, 0, 1);
        require_equal(packed_expected, output,
                      "hybrid-dataflow data_time=" + std::to_string(data_time));
    }
    std::cout << "PASS hybrid-dataflow data-time unfolding\n";

    struct FusedRoleCase {
        std::uint32_t log_n;
        std::uint32_t flow_tile_log_n;
        std::uint32_t stage_space;
        std::uint32_t role_stages;
        std::uint32_t data_time;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
    };
    constexpr std::array<FusedRoleCase, 9> kFusedRoleCases = {{{10, 5, 5, 2, 4, 64, cuntt::kDefaultModulus, false},
                                                               {10, 5, 5, 5, 4, 64, cuntt::kDefaultModulus, false},
                                                               {10, 5, 5, 5, 4, 64, cuntt::kDefaultModulus, true},
                                                               {10, 5, 5, 2, 4, 32, kModulus30, false},
                                                               {10, 5, 5, 5, 4, 32, kModulus30, false},
                                                               {12, 6, 6, 2, 4, 64, cuntt::kDefaultModulus, false},
                                                               {12, 6, 6, 3, 4, 64, cuntt::kDefaultModulus, false},
                                                               {12, 6, 6, 6, 4, 64, cuntt::kDefaultModulus, false},
                                                               {12, 6, 6, 2, 8, 64, cuntt::kDefaultModulus, false}}};
    for (std::size_t case_index = 0; case_index < kFusedRoleCases.size(); ++case_index) {
        const auto& test_case = kFusedRoleCases[case_index];
        const auto fused_input = make_input(1U << test_case.log_n, test_case.modulus, 0x7300 + case_index);
        const auto fused_expected = reference_batch(fused_input, test_case.log_n, 1,
                                                    test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.backend = cuntt::Backend::HybridDataflow;
        config.flow_tile_log_n = test_case.flow_tile_log_n;
        config.stage_space = test_case.stage_space;
        config.role_stages = test_case.role_stages;
        config.data_space = test_case.word_bits == 32 ? 32 : 16;
        config.data_time = test_case.data_time;
        config.pipeline_buffers = 2;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(fused_input, output, 0, 1);
        require_equal(fused_expected, output,
                      "hybrid-dataflow role_stages=" + std::to_string(test_case.role_stages));
    }
    std::cout << "PASS hybrid-dataflow register-resident fused roles\n";

    struct TokenInterleaveCase {
        std::uint32_t log_n;
        std::uint32_t flow_tile_log_n;
        std::uint32_t role_stages;
        std::uint32_t data_time;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
    };
    constexpr std::array<TokenInterleaveCase, 3> kTokenInterleaveCases = {{
        {10, 5, 5, 10, 64, cuntt::kDefaultModulus, false},
        {10, 5, 5, 10, 64, cuntt::kDefaultModulus, true},
        {12, 6, 6, 12, 32, kModulus30, false},
    }};
    for (std::size_t case_index = 0; case_index < kTokenInterleaveCases.size(); ++case_index) {
        const auto& test_case = kTokenInterleaveCases[case_index];
        const auto interleave_input = make_input(1U << test_case.log_n, test_case.modulus,
                                                 0x7380 + case_index);
        const auto interleave_expected = reference_batch(interleave_input, test_case.log_n, 1,
                                                         test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.backend = cuntt::Backend::HybridDataflow;
        config.flow_tile_log_n = test_case.flow_tile_log_n;
        config.stage_space = test_case.role_stages;
        config.role_stages = test_case.role_stages;
        config.data_space = test_case.word_bits == 32 ? 32 : 16;
        config.data_time = test_case.data_time;
        config.token_interleave = 2;
        config.pipeline_buffers = 2;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(interleave_input, output, 0, 1);
        require_equal(interleave_expected, output, "hybrid-dataflow token_interleave=2");
    }
    std::cout << "PASS hybrid-dataflow two-token software pipeline\n";

    const auto occupancy_input = make_input(1U << 12, cuntt::kDefaultModulus, 0x7400);
    const auto occupancy_expected = reference_batch(occupancy_input, 12, 1, cuntt::kDefaultModulus, false);
    cuntt::PlanConfig occupancy_config;
    occupancy_config.log_n = 12;
    occupancy_config.backend = cuntt::Backend::HybridDataflow;
    occupancy_config.flow_tile_log_n = 6;
    occupancy_config.stage_space = 6;
    occupancy_config.role_stages = 6;
    occupancy_config.target_ctas_per_sm = 3;
    occupancy_config.data_space = 16;
    occupancy_config.data_time = 8;
    occupancy_config.pipeline_buffers = 2;
    cuntt::Plan occupancy_plan(occupancy_config);
    std::vector<std::uint64_t> occupancy_output;
    occupancy_plan.execute(occupancy_input, occupancy_output, 0, 1);
    require_equal(occupancy_expected, occupancy_output,
                  "hybrid-dataflow target_ctas_per_sm=3");
    std::cout << "PASS HybridDataflow generated CTA-residency target\n";

    cuntt::PlanConfig low_batch_config;
    low_batch_config.log_n = 12;
    low_batch_config.batch = 16;
    low_batch_config.backend = cuntt::Backend::HybridDataflow;
    cuntt::Plan low_batch_plan(low_batch_config);
    if (low_batch_plan.config().role_stages != 6 || low_batch_plan.config().data_time != 12 ||
        low_batch_plan.config().token_interleave != 1 ||
        low_batch_plan.config().compute_unit != cuntt::ComputeUnit::Radix4) {
        throw std::runtime_error("V100 HybridDataflow low-batch default did not select Ur=6,Td=12,Ti=1,radix4");
    }
    if (low_batch_plan.selection().implementation != "hybrid-dataflow-generated-static-table") {
        throw std::runtime_error("V100 HybridDataflow low-batch static-table source was not reported");
    }
    cuntt::PlanConfig saturated_config = low_batch_config;
    saturated_config.batch = 256;
    cuntt::Plan saturated_plan(saturated_config);
    if (saturated_plan.config().role_stages != 6 || saturated_plan.config().data_time != 12 ||
        saturated_plan.config().target_ctas_per_sm != 1 || saturated_plan.config().token_interleave != 1 ||
        saturated_plan.config().compute_unit != cuntt::ComputeUnit::Radix4) {
        throw std::runtime_error("V100 HybridDataflow saturated default did not select Ur=6,Td=12,Ti=1,radix4");
    }
    if (saturated_plan.selection().implementation != "hybrid-dataflow-generated-static-table" ||
        saturated_plan.selection().confidence != "measured") {
        throw std::runtime_error("V100 HybridDataflow static-table source was not reported");
    }
    cuntt::PlanConfig uint32_boundary_config;
    uint32_boundary_config.log_n = 12;
    uint32_boundary_config.batch = 160;
    uint32_boundary_config.word_bits = 32;
    uint32_boundary_config.modulus = kModulus30;
    uint32_boundary_config.backend = cuntt::Backend::HybridDataflow;
    cuntt::Plan uint32_low_plan(uint32_boundary_config);
    if (uint32_low_plan.config().role_stages != 6 || uint32_low_plan.config().data_time != 12) {
        throw std::runtime_error("V100 uint32 HybridDataflow 2*SM boundary did not preserve Td=12");
    }
    uint32_boundary_config.batch = 161;
    cuntt::Plan uint32_high_plan(uint32_boundary_config);
    if (uint32_high_plan.config().role_stages != 6 || uint32_high_plan.config().data_time != 6) {
        throw std::runtime_error("V100 uint32 HybridDataflow 2*SM+1 boundary did not select Td=6");
    }
    std::cout << "PASS HybridDataflow V100 fused-role defaults\n";

    struct ResidentRadix4Case {
        std::uint32_t log_n;
        std::size_t batch;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
    };
    constexpr std::array<ResidentRadix4Case, 5> kResidentRadix4Cases = {{
        {7, 3, 64, cuntt::kDefaultModulus, false},
        {10, 17, 64, cuntt::kDefaultModulus, false},
        {10, 3, 64, cuntt::kDefaultModulus, true},
        {12, 17, 64, cuntt::kDefaultModulus, false},
        {12, 17, 32, kModulus30, false},
    }};
    for (std::size_t case_index = 0; case_index < kResidentRadix4Cases.size(); ++case_index) {
        const auto& test_case = kResidentRadix4Cases[case_index];
        const auto radix4_input = make_input((1ULL << test_case.log_n) * test_case.batch,
                                             test_case.modulus, 0x7500 + case_index);
        const auto radix4_expected = reference_batch(radix4_input, test_case.log_n, test_case.batch,
                                                     test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.batch = test_case.batch;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        config.backend = cuntt::Backend::HybridDataflow;
        config.compute_unit = cuntt::ComputeUnit::Radix4;
        cuntt::Plan plan(config);
        if (plan.config().threads_per_block != 256 || plan.workspace_size() != 0) {
            throw std::runtime_error("HybridDataflow resident radix-4 launch contract is incorrect");
        }
        std::vector<std::uint64_t> output;
        plan.execute(radix4_input, output, 0, 1);
        require_equal(radix4_expected, output, "hybrid-dataflow resident radix4");
    }
    std::cout << "PASS HybridDataflow resident radix-4 physical core\n";
}

void test_hierarchical_dataflow_backend() {
    constexpr std::uint64_t kModulus30 = 1073479681ULL;
    struct Case {
        std::uint32_t log_n;
        std::size_t batch;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
        std::uint32_t n1_log;
        std::uint32_t rows;
        std::uint32_t data_time;
    };
    constexpr std::array<Case, 4> kCases = {{
        {12, 1, 64, cuntt::kDefaultModulus, false, 6, 4, 1},
        {13, 3, 32, kModulus30, true, 6, 2, 2},
        {14, 5, 64, cuntt::kDefaultModulus, true, 7, 1, 4},
        {16, 2, 64, cuntt::kDefaultModulus, false, 6, 4, 8},
    }};
    for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
        const auto& test_case = kCases[case_index];
        const auto input = make_input((1ULL << test_case.log_n) * test_case.batch,
                                      test_case.modulus, 0x7600 + case_index);
        const auto expected = reference_batch(input, test_case.log_n, test_case.batch,
                                              test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.batch = test_case.batch;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        config.backend = cuntt::Backend::HierarchicalBarrier;
        config.n1_log = test_case.n1_log;
        config.rows_per_block = test_case.rows;
        config.data_time = test_case.data_time;
        cuntt::Plan plan(config);
        if (plan.config().compute_unit != cuntt::ComputeUnit::Radix4 ||
            plan.workspace_size() != input.size() * (test_case.word_bits / 8) ||
            plan.config().target_ctas_per_sm == 0) {
            throw std::runtime_error("HierarchicalDataflow plan resource contract is incorrect");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hierarchical-dataflow cooperative graph");
    }
    {
        constexpr std::size_t n = 1ULL << 20;
        std::vector<std::uint64_t> input(n, 0);
        input[0] = 1;
        cuntt::PlanConfig config;
        config.log_n = 20;
        config.word_bits = 32;
        config.modulus = 998244353;
        config.backend = cuntt::Backend::HierarchicalBarrier;
        config.n1_log = 10;
        config.rows_per_block = 4;
        config.hierarchical_core = cuntt::HierarchicalCore::Hybrid2DRadix4;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(), [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error("hierarchical Hybrid2D radix-4 core produced an incorrect delta transform");
        }
    }
    std::cout << "PASS HierarchicalDataflow cooperative multi-CTA graph\n";
}

void test_true_hierarchical_streaming_backend() {
    constexpr std::uint64_t kModulus30 = 1073479681ULL;
    struct Case {
        std::uint32_t log_n;
        std::size_t batch;
        std::uint32_t word_bits;
        std::uint64_t modulus;
        bool inverse;
        std::vector<std::uint32_t> partition;
    };
    const std::array<Case, 7> cases = {{
        {12, 2, 64, cuntt::kDefaultModulus, false, {}},
        {13, 3, 32, kModulus30, true, {7, 6}},
        {15, 2, 64, cuntt::kDefaultModulus, false, {5, 5, 5}},
        {12, 2, 32, kModulus30, false, {3, 3, 3, 3}},
        {12, 2, 64, cuntt::kDefaultModulus, true, {2, 2, 2, 3, 3}},
        {12, 2, 32, kModulus30, false, {2, 2, 2, 2, 2, 2}},
        {16, 1, 32, kModulus30, false, {2, 2, 2, 2, 2, 2, 2, 2}},
    }};
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& test_case = cases[case_index];
        const auto input = make_input((1ULL << test_case.log_n) * test_case.batch,
                                      test_case.modulus, 0x7a00 + case_index);
        const auto expected = reference_batch(input, test_case.log_n, test_case.batch,
                                              test_case.modulus, test_case.inverse);
        cuntt::PlanConfig config;
        config.log_n = test_case.log_n;
        config.batch = test_case.batch;
        config.word_bits = test_case.word_bits;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = test_case.partition;
        cuntt::Plan plan(config);
        if (plan.config().stage_partition.size() < 2 || plan.workspace_size() <= plan.data_size()) {
            throw std::runtime_error("true HierarchicalDataflow descriptor/workspace contract is incorrect");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hierarchical-dataflow readiness stream");
        const auto trace = plan.pipeline_trace();
        if (trace.size() != plan.config().stage_partition.size()) {
            throw std::runtime_error("true HierarchicalDataflow trace shape is incorrect");
        }
        for (std::size_t segment = 1; segment < trace.size(); ++segment) {
            if (trace[segment].start_globaltimer == 0 ||
                trace[segment].end_globaltimer <= trace[segment].start_globaltimer) {
                throw std::runtime_error("HierarchicalDataflow subgraph trace is invalid");
            }
            if (trace.size() >= 3 &&
                trace[segment].start_globaltimer >= trace[segment - 1].end_globaltimer) {
                throw std::runtime_error("adjacent HierarchicalDataflow subgraphs did not overlap");
            }
        }
    }
    for (const auto core : {cuntt::NttSubgraphCore::Radix2,
                            cuntt::NttSubgraphCore::Radix4,
                            cuntt::NttSubgraphCore::Radix8,
                            cuntt::NttSubgraphCore::DataflowRadix4}) {
        std::vector<std::uint64_t> input(1ULL << 12, 0);
        input[0] = 1;
        cuntt::PlanConfig config;
        config.log_n = 12;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {6, 6};
        config.subgraph_mappings.resize(2);
        for (auto& mapping : config.subgraph_mappings) mapping.core = core;
        cuntt::Plan plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(), [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error("true HierarchicalDataflow physical core failed delta verification");
        }
    }
    {
        constexpr std::uint32_t log_n = 12;
        constexpr std::size_t batch = 2;
        const auto input = make_input((1ULL << log_n) * batch,
                                      cuntt::kDefaultModulus, 0x4d344732);
        const auto expected = reference_batch(input, log_n, batch,
                                              cuntt::kDefaultModulus, false);
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.batch = batch;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {3, 3, 3, 3};
        config.boundary_mappings = {
            {cuntt::BoundaryStorage::ResidentFused, 1},
            {cuntt::BoundaryStorage::FullScratch, 1},
            {cuntt::BoundaryStorage::ResidentFused, 1},
        };
        config.execution_group_mappings.resize(2);
        config.execution_group_mappings[0].core =
            cuntt::NttSubgraphCore::Radix2;
        config.execution_group_mappings[1].core =
            cuntt::NttSubgraphCore::Radix4;
        for (auto& mapping : config.execution_group_mappings) {
            mapping.cta_weight = 6;
        }
        cuntt::Plan plan(config);
        if (plan.config().stage_partition !=
                std::vector<std::uint32_t>{3, 3, 3, 3} ||
            plan.config().execution_group_mappings.size() != 2 ||
            plan.config().execution_group_mappings[0].core !=
                cuntt::NttSubgraphCore::Radix2 ||
            plan.config().execution_group_mappings[1].core !=
                cuntt::NttSubgraphCore::Radix4 ||
            plan.selection().implementation.find("resident-fused-m4-g2") ==
                std::string::npos) {
            throw std::runtime_error(
                "resident lowering did not preserve logical M or expose physical G");
        }
        if (plan.workspace_size() >= 2 * input.size() * sizeof(std::uint64_t)) {
            throw std::runtime_error(
                "resident lowering materialized a fused logical boundary");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "resident-fused M4/G2 lowering");
        if (plan.pipeline_trace().size() != 2) {
            throw std::runtime_error(
                "resident-fused trace must report physical execution groups");
        }
    }
    {
        constexpr std::uint32_t log_n = 12;
        constexpr std::size_t batch = 2;
        const auto input = make_input((1ULL << log_n) * batch,
                                      cuntt::kDefaultModulus, 0x7a77);
        const auto expected = reference_batch(input, log_n, batch,
                                              cuntt::kDefaultModulus, false);
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.batch = batch;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {8, 4};
        config.subgraph_mappings.resize(2);
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::ApptPipeline;
            mapping.data_space = 16;
            mapping.data_time = 4;
            mapping.role_stages = 1;
            mapping.token_interleave = 1;
        }
        config.target_ctas_per_sm = 1;
        cuntt::Plan plan(config);
        if (plan.selection().implementation != "hierarchical-dataflow-appt-pipeline" ||
            plan.config().subgraph_mappings.front().units_per_cta != 8 ||
            plan.config().boundary_mappings.front().storage != cuntt::BoundaryStorage::Ring ||
            plan.workspace_size() < plan.data_size() ||
            plan.workspace_size() > plan.data_size() + 64) {
            throw std::runtime_error("APPT pipeline mapping contract is incorrect");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "APPT four-axis pipeline");
        const auto trace = plan.pipeline_trace();
        if (trace.size() != 2 || trace[0].start_globaltimer == 0 ||
            trace[0].end_globaltimer <= trace[0].start_globaltimer ||
            trace[1].start_globaltimer < trace[0].end_globaltimer) {
            throw std::runtime_error("APPT stage-time fold trace is invalid");
        }

        for (auto& mapping : config.subgraph_mappings) {
            mapping.units_per_cta = 8;
            mapping.role_stages = 2;
            mapping.token_interleave = 2;
        }
        cuntt::Plan replicated_plan(config);
        if (replicated_plan.config().subgraph_mappings.front().units_per_cta != 8 ||
            replicated_plan.config().subgraph_mappings.front().role_stages != 2 ||
            replicated_plan.config().subgraph_mappings.front().token_interleave != 2) {
            throw std::runtime_error("APPT replicated pipeline mapping contract is incorrect");
        }
        replicated_plan.execute(input, output, 0, 1);
        require_equal(expected, output, "APPT replicated grouped-token pipeline");
    }
    {
        constexpr std::uint32_t log_n = 12;
        constexpr std::size_t batch = 5;
        const auto input = make_input((1ULL << log_n) * batch, cuntt::kDefaultModulus, 0x7b00);
        const auto expected = reference_batch(input, log_n, batch, cuntt::kDefaultModulus, false);
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.batch = batch;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {6, 6};
        config.boundary_mappings = {{cuntt::BoundaryStorage::Ring, 2}};
        cuntt::Plan plan(config);
        if (plan.workspace_size() >= input.size() * sizeof(std::uint64_t)) {
            throw std::runtime_error("ring boundary did not bound the streaming workspace");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hierarchical-dataflow bounded ring");
    }
    {
        constexpr std::uint32_t log_n = 20;
        constexpr std::size_t batch = 2;
        std::vector<std::uint64_t> input((1ULL << log_n) * batch, 0);
        for (std::size_t transform = 0; transform < batch; ++transform) {
            input[transform << log_n] = 1;
        }
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.batch = batch;
        config.modulus = 576460756061519873ULL;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        cuntt::Plan plan(config);
        if (plan.config().stage_partition != std::vector<std::uint32_t>{10, 10}) {
            throw std::runtime_error("V100 logN=20 did not select the generated 10+10 stream");
        }
        if (plan.config().subgraph_mappings[0].cta_weight != 9 ||
            plan.config().subgraph_mappings[1].cta_weight != 11) {
            throw std::runtime_error("V100 logN=20 did not select the measured 9:11 role balance");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error("specialized 10+10 streaming core failed delta verification");
        }
    }
    {
        constexpr std::uint32_t log_n = 20;
        constexpr std::size_t batch = 3;
        std::vector<std::uint64_t> input((1ULL << log_n) * batch, 0);
        for (std::size_t transform = 0; transform < batch; ++transform) {
            input[transform << log_n] = 1;
        }
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.batch = batch;
        config.word_bits = 32;
        config.modulus = 998244353ULL;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {10, 10};
        config.subgraph_mappings.resize(2);
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::HomogeneousRadix4;
            mapping.threads_per_block = 256;
            mapping.units_per_cta = 4;
        }
        config.subgraph_mappings[0].data_time = 2;
        config.subgraph_mappings[1].data_time = 3;
        config.subgraph_mappings[0].cta_weight = 9;
        config.subgraph_mappings[1].cta_weight = 11;
        config.boundary_mappings = {{cuntt::BoundaryStorage::FullScratch, 1}};
        cuntt::Plan plan(config);
        if (plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-radix4-10x10") {
            throw std::runtime_error(
                "10+10 homogeneous template did not select its generated kernel");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error(
                "10+10 homogeneous role data-time traversal failed delta verification");
        }

        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::HomogeneousWarpRadix2;
            mapping.units_per_cta = 8;
        }
        config.subgraph_mappings[0].data_time = 2;
        config.subgraph_mappings[1].data_time = 1;
        cuntt::Plan warp_plan(config);
        if (warp_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp-radix2-10x10") {
            throw std::runtime_error(
                "10+10 warp template did not select its generated kernel");
        }
        warp_plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error(
                "10+10 warp-register subgraphs failed delta verification");
        }

        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::HomogeneousWarp256Radix2;
        }
        cuntt::Plan warp256_plan(config);
        if (warp256_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp256-radix2-10x10") {
            throw std::runtime_error(
                "10+10 warp256 template did not select its generated kernel");
        }
        warp256_plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error(
                "10+10 warp256 subgraphs failed delta verification");
        }

        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = (index * 48271ULL + 17ULL) % config.modulus;
        }
        auto reference_config = config;
        for (auto& mapping : reference_config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::DataflowRadix4;
            mapping.units_per_cta = 4;
            mapping.data_time = 1;
        }
        cuntt::Plan reference_plan(reference_config);
        std::vector<std::uint64_t> reference_output;
        reference_plan.execute(input, reference_output, 0, 1);
        warp_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 warp-register subgraphs disagree with v0.6");
        }
        warp256_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 warp256 subgraphs disagree with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core =
                cuntt::NttSubgraphCore::HomogeneousWarp256StaticRadix2;
        }
        cuntt::Plan warp256_static_plan(config);
        warp256_static_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 static-writer warp256 subgraphs disagree with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core =
                cuntt::NttSubgraphCore::HomogeneousWarp256StaticIoRadix2;
        }
        cuntt::Plan warp256_static_io_plan(config);
        warp256_static_io_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 static-IO warp256 subgraphs disagree with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.data_space = 4;
            mapping.threads_per_block = 128;
            mapping.units_per_cta = 4;
        }
        config.target_ctas_per_sm = 3;
        cuntt::Plan warp256_half_packet_plan(config);
        warp256_half_packet_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 half-packet warp256 subgraphs disagree with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp256CoefficientReuseStaticIoRadix2;
        }
        cuntt::Plan warp256_coefficient_reuse_plan(config);
        warp256_coefficient_reuse_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 coefficient-reuse warp256 disagrees with v0.6");
        }
        for (const std::uint32_t reuse_stages : {1U, 2U, 3U, 4U, 5U, 6U}) {
            for (auto& mapping : config.subgraph_mappings) {
                mapping.coefficient_reuse_stages = reuse_stages;
            }
            cuntt::Plan warp256_partial_reuse_plan(config);
            warp256_partial_reuse_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "10+10 partial coefficient-reuse warp256 disagrees with v0.6");
            }
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.coefficient_reuse_stages = 0;
        }
        for (const auto core : {
                 cuntt::NttSubgraphCore::HomogeneousWarp128StaticIoRadix2,
                 cuntt::NttSubgraphCore::HomogeneousWarp64StaticIoRadix2}) {
            for (auto& mapping : config.subgraph_mappings) {
                mapping.core = core;
            }
            cuntt::Plan small_warp_plan(config);
            small_warp_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "10+10 small-warp static-IO subgraphs disagree with v0.6");
            }
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core =
                cuntt::NttSubgraphCore::HomogeneousWarp128StaticIoRadix2;
            mapping.threads_per_block = 256;
            mapping.units_per_cta = 8;
        }
        config.target_ctas_per_sm = 2;
        cuntt::Plan dual_warp128_plan(config);
        dual_warp128_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 dual warp128 groups disagree with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128CoefficientReuseStaticIoRadix2;
        }
        cuntt::Plan dual_warp128_coefficient_reuse_plan(config);
        dual_warp128_coefficient_reuse_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 dual warp128 coefficient reuse disagrees with v0.6");
        }
        for (const std::uint32_t reuse_stages : {1U, 2U, 3U, 4U, 5U, 6U}) {
            for (auto& mapping : config.subgraph_mappings) {
                mapping.coefficient_reuse_stages = reuse_stages;
            }
            cuntt::Plan dual_warp128_partial_reuse_plan(config);
            dual_warp128_partial_reuse_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "10+10 partial coefficient-reuse dual warp128 disagrees with v0.6");
            }
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.coefficient_reuse_stages = 0;
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128VectorRadix4StaticIo;
        }
        cuntt::Plan dual_warp128_vector_radix4_plan(config);
        if (dual_warp128_vector_radix4_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp128-vector-radix4-static-io-10x10") {
            throw std::runtime_error(
                "10+10 vector-radix4 dual warp128 selection is incorrect");
        }
        dual_warp128_vector_radix4_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 vector-radix4 dual warp128 disagrees with v0.6");
        }
        for (const std::uint32_t reuse_stages : {3U, 4U, 5U, 6U, 7U}) {
            for (auto& mapping : config.subgraph_mappings) {
                mapping.coefficient_reuse_stages = reuse_stages;
            }
            cuntt::Plan vector_radix4_reuse_plan(config);
            if (vector_radix4_reuse_plan.selection().implementation !=
                "hierarchical-dataflow-homogeneous-warp128-vector-radix4-static-io-10x10") {
                throw std::runtime_error(
                    "10+10 vector-radix4 coefficient reuse did not select the specialized kernel");
            }
            vector_radix4_reuse_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "10+10 vector-radix4 coefficient reuse disagrees with v0.6");
            }
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128VectorRadix4PackedStage6StaticIo;
            mapping.coefficient_reuse_stages = 6;
        }
        cuntt::Plan packed_stage6_plan(config);
        if (packed_stage6_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp128-vector-radix4-packed-stage6-static-io-10x10") {
            throw std::runtime_error(
                "10+10 packed-stage6 vector-radix4 selection is incorrect");
        }
        packed_stage6_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 packed-stage6 vector-radix4 disagrees with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128VectorRadix4PackedStage6DistributedStaticIo;
        }
        cuntt::Plan packed_stage6_distributed_plan(config);
        if (packed_stage6_distributed_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp128-vector-radix4-packed-stage6-distributed-static-io-10x10") {
            throw std::runtime_error(
                "10+10 distributed packed-stage6 selection is incorrect");
        }
        packed_stage6_distributed_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 distributed packed-stage6 disagrees with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.coefficient_reuse_stages = 0;
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128PacketSharedRadix4StaticIo;
            mapping.threads_per_block = 128;
            mapping.units_per_cta = 4;
            mapping.token_interleave = 0;
        }
        config.target_ctas_per_sm = 4;
        cuntt::Plan packet_shared_radix4_plan(config);
        if (packet_shared_radix4_plan.selection().implementation !=
            "hierarchical-dataflow-homogeneous-warp128-packet-shared-radix4-static-io-10x10") {
            throw std::runtime_error(
                "10+10 packet-shared radix-4 selection is incorrect");
        }
        packet_shared_radix4_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 packet-shared radix-4 disagrees with v0.6");
        }
        config.subgraph_mappings[1].token_interleave = 16;
        cuntt::Plan packet_streaming_radix4_plan(config);
        for (std::uint32_t repetition = 0; repetition < 3; ++repetition) {
            packet_streaming_radix4_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "10+10 packet-streaming radix-4 disagrees with v0.6 on repetition " +
                    std::to_string(repetition));
            }
        }
        for (const std::uint32_t ready_window : {1U, 2U, 4U, 8U, 16U}) {
            config.ready_window = ready_window;
            cuntt::Plan polling_plan(config);
            polling_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "packet-streaming radix-4 polling window disagrees with v0.6");
            }
        }
        config.ready_window = 2;
        config.packet_readiness_mode =
            cuntt::PacketReadinessMode::WaveBitmap;
        cuntt::Plan bitmap_readiness_plan(config);
        for (std::uint32_t repetition = 0; repetition < 3; ++repetition) {
            bitmap_readiness_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "packet-streaming wave-bitmap readiness disagrees with v0.6");
            }
        }
        config.packet_compute_layout =
            cuntt::PacketComputeLayout::WarpRows;
        cuntt::Plan bitmap_warp_rows_plan(config);
        bitmap_warp_rows_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "packet-streaming bitmap warp-row layout disagrees with v0.6");
        }
        config.packet_fold_wave_barriers = true;
        cuntt::Plan folded_bitmap_warp_rows_plan(config);
        for (std::uint32_t repetition = 0; repetition < 3; ++repetition) {
            folded_bitmap_warp_rows_plan.execute(input, output, 0, 1);
            if (output != reference_output) {
                throw std::runtime_error(
                    "folded packet wave barriers disagree with v0.6");
            }
        }
        config.packet_fold_wave_barriers = false;
        config.packet_readiness_mode =
            cuntt::PacketReadinessMode::PerPacket;
        cuntt::Plan packet_warp_rows_plan(config);
        packet_warp_rows_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "packet-streaming per-packet warp-row layout disagrees with v0.6");
        }
        config.packet_compute_layout =
            cuntt::PacketComputeLayout::InterleavedRows;
        config.ready_window = 3;
        try {
            cuntt::Plan invalid_polling_window(config);
            throw std::runtime_error(
                "packet-streaming radix-4 accepted an unsupported polling window");
        } catch (const std::invalid_argument&) {
        }
        config.ready_window = 0;
        config.packet_readiness_mode =
            cuntt::PacketReadinessMode::WaveBitmap;
        config.subgraph_mappings[1].token_interleave = 0;
        try {
            cuntt::Plan invalid_bitmap_aggregate(config);
            throw std::runtime_error(
                "wave-bitmap readiness accepted an aggregate packet schedule");
        } catch (const std::invalid_argument&) {
        }
        config.packet_readiness_mode =
            cuntt::PacketReadinessMode::PerPacket;
        config.packet_compute_layout =
            cuntt::PacketComputeLayout::WarpRows;
        try {
            cuntt::Plan invalid_warp_rows_aggregate(config);
            throw std::runtime_error(
                "warp-row packet layout accepted an aggregate packet schedule");
        } catch (const std::invalid_argument&) {
        }
        config.packet_compute_layout =
            cuntt::PacketComputeLayout::InterleavedRows;
        config.packet_fold_wave_barriers = true;
        try {
            cuntt::Plan invalid_folded_aggregate(config);
            throw std::runtime_error(
                "folded packet barriers accepted an aggregate packet schedule");
        } catch (const std::invalid_argument&) {
        }
        config.packet_fold_wave_barriers = false;
        config.subgraph_mappings[1].token_interleave = 32;
        try {
            cuntt::Plan invalid_packet_wave(config);
            throw std::runtime_error(
                "packet-streaming radix-4 accepted a wave wider than its staging tile");
        } catch (const std::invalid_argument&) {
        }
        config.subgraph_mappings[1].token_interleave = 0;
        for (auto& mapping : config.subgraph_mappings) {
            mapping.threads_per_block = 128;
            mapping.units_per_cta = 4;
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128PipelineStaticIoRadix2;
        }
        config.pipeline_buffers = 1;
        config.target_ctas_per_sm = 4;
        cuntt::Plan pipeline_plan(config);
        pipeline_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 prefix/merge warp pipeline disagrees with v0.6");
        }
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::
                HomogeneousWarp128CooperativeStaticIoRadix2;
        }
        config.target_ctas_per_sm = 4;
        cuntt::Plan cooperative_plan(config);
        cooperative_plan.execute(input, output, 0, 1);
        if (output != reference_output) {
            throw std::runtime_error(
                "10+10 cooperative continuation disagrees with v0.6");
        }
    }
    {
        constexpr std::uint32_t log_n = 20;
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.word_bits = 64;
        config.modulus = 576460756061519873ULL;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {10, 10};
        config.target_ctas_per_sm = 3;
        config.subgraph_mappings.resize(2);
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core =
                cuntt::NttSubgraphCore::HomogeneousWarp256StaticIoRadix2;
            mapping.threads_per_block = 128;
            mapping.units_per_cta = 4;
            mapping.data_space = 2;
            mapping.data_time = 1;
        }
        config.subgraph_mappings[0].cta_weight = 9;
        config.subgraph_mappings[1].cta_weight = 11;
        config.boundary_mappings = {{cuntt::BoundaryStorage::FullScratch, 1}};

        auto reference_config = config;
        reference_config.target_ctas_per_sm = 2;
        for (auto& mapping : reference_config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::DataflowRadix4;
            mapping.data_space = 0;
        }
        const auto input = make_input(1ULL << log_n, config.modulus,
                                      0x48414c465041434bULL);
        std::vector<std::uint64_t> expected;
        std::vector<std::uint64_t> output;
        cuntt::Plan(reference_config).execute(input, expected, 0, 1);
        cuntt::Plan(config).execute(input, output, 0, 1);
        require_equal(expected, output,
                      "uint64 10+10 half-packet static IO");
    }
    for (const auto word_bits : {32U, 64U}) {
        constexpr std::uint32_t log_n = 20;
        std::vector<std::uint64_t> input(1ULL << log_n, 0);
        input[0] = 1;
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.word_bits = word_bits;
        config.modulus = word_bits == 32 ? 998244353ULL : 576460756061519873ULL;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {7, 7, 6};
        config.target_ctas_per_sm = 1;
        cuntt::Plan plan(config);
        if (plan.selection().implementation !=
            "hierarchical-dataflow-wave-7x7x6-resident-radix4") {
            throw std::runtime_error("V100 7+7+6 did not select the resident wave kernel");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error("specialized 7+7+6 streaming core failed delta verification");
        }
        const auto trace = plan.pipeline_trace();
        if (trace.size() != 3 ||
            trace[1].start_globaltimer >= trace[0].end_globaltimer) {
            throw std::runtime_error("specialized 7+7+6 did not expose an intra-transform wave");
        }
    }
    {
        constexpr std::uint32_t log_n = 20;
        const std::vector<std::vector<std::uint32_t>> partitions = {
            {6, 7, 7}, {7, 6, 7}, {7, 7, 6},
            {6, 6, 8}, {6, 8, 6}, {8, 6, 6}};
        cuntt::PlanConfig reference_config;
        reference_config.log_n = log_n;
        reference_config.word_bits = 32;
        reference_config.modulus = 998244353ULL;
        reference_config.backend = cuntt::Backend::HierarchicalDataflow;
        reference_config.stage_partition = {10, 10};
        reference_config.target_ctas_per_sm = 4;
        const auto input = make_input(1ULL << log_n, reference_config.modulus,
                                      0x54485245454c564cULL);
        std::vector<std::uint64_t> expected;
        cuntt::Plan(reference_config).execute(input, expected, 0, 1);
        for (const auto& partition : partitions) {
            auto config = reference_config;
            config.stage_partition = partition;
            config.target_ctas_per_sm = 2;
            cuntt::Plan plan(config);
            if (plan.selection().implementation.find(
                    "three-level-resident-radix4") == std::string::npos &&
                partition != std::vector<std::uint32_t>{7, 7, 6}) {
                throw std::runtime_error(
                    "three-level partition did not select its resident kernel");
            }
            std::vector<std::uint64_t> output;
            plan.execute(input, output, 0, 1);
            require_equal(expected, output,
                          "parameterized three-level resident partition");
        }
        auto packet_config = reference_config;
        packet_config.stage_partition = {6, 6, 8};
        packet_config.target_ctas_per_sm = 4;
        packet_config.subgraph_mappings.resize(3);
        for (auto& mapping : packet_config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::DataflowRadix4;
            mapping.threads_per_block = 128;
            mapping.units_per_cta = 4;
            mapping.data_time = 1;
        }
        packet_config.subgraph_mappings[0].cta_weight = 4;
        packet_config.subgraph_mappings[1].cta_weight = 4;
        packet_config.subgraph_mappings[2].cta_weight = 12;
        std::vector<std::uint64_t> packet_output;
        cuntt::Plan(packet_config).execute(input, packet_output, 0, 1);
        require_equal(expected, packet_output,
                      "three-level 128-thread four-row packet");
    }
    {
        constexpr std::uint32_t log_n = 20;
        std::vector<std::uint64_t> input(1ULL << log_n, 0);
        input[0] = 1;
        cuntt::PlanConfig config;
        config.log_n = log_n;
        config.word_bits = 32;
        config.modulus = 998244353ULL;
        config.backend = cuntt::Backend::HierarchicalDataflow;
        config.stage_partition = {7, 7, 6};
        config.target_ctas_per_sm = 3;
        config.subgraph_mappings.resize(3);
        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::ApptOnline;
            mapping.units_per_cta = 8;
            mapping.data_space = 16;
            mapping.data_time = 4;
            mapping.role_stages = 2;
            mapping.token_interleave = 2;
        }
        config.subgraph_mappings[0].cta_weight = 6;
        config.subgraph_mappings[1].cta_weight = 6;
        config.subgraph_mappings[2].cta_weight = 8;
        config.boundary_mappings = {
            {cuntt::BoundaryStorage::Ring, 2},
            {cuntt::BoundaryStorage::Ring, 2},
        };
        cuntt::Plan plan(config);
        if (plan.selection().implementation !=
                "hierarchical-dataflow-appt-online-warp-radix2" ||
            plan.workspace_size() <= 2 * plan.data_size()) {
            throw std::runtime_error("APPT online-fold mapping contract is incorrect");
        }
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error("APPT online-fold core failed delta verification");
        }
        const auto trace = plan.pipeline_trace();
        if (trace.size() != 3 ||
            trace[0].start_globaltimer == 0 ||
            trace[0].end_globaltimer <= trace[0].start_globaltimer ||
            trace[1].start_globaltimer == 0 ||
            trace[1].end_globaltimer <= trace[1].start_globaltimer ||
            trace[2].start_globaltimer == 0 ||
            trace[2].end_globaltimer <= trace[2].start_globaltimer) {
            throw std::runtime_error("APPT online-fold trace contract is invalid");
        }

        config.profile_appt_roles = true;
        cuntt::Plan profile_plan(config);
        profile_plan.execute(input, output, 0, 1);
        const auto role_metrics = profile_plan.pipeline_role_metrics();
        if (role_metrics.size() != 3 || role_metrics[0].tasks != 64 ||
            role_metrics[1].tasks != 64 || role_metrics[2].tasks != 128 ||
            role_metrics[0].compute_globaltimer == 0 ||
            role_metrics[1].boundary_globaltimer == 0 ||
            role_metrics[2].compute_globaltimer == 0) {
            throw std::runtime_error("APPT online per-role metrics are invalid");
        }
        config.profile_appt_roles = false;

        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::ApptOnlineRadix4;
            mapping.cta_weight = 0;
        }
        config.target_ctas_per_sm = 0;
        cuntt::Plan radix4_plan(config);
        if (radix4_plan.selection().implementation !=
                "hierarchical-dataflow-appt-online-radix4" ||
            radix4_plan.config().target_ctas_per_sm != 3 ||
            radix4_plan.config().subgraph_mappings[0].cta_weight != 7 ||
            radix4_plan.config().subgraph_mappings[1].cta_weight != 6 ||
            radix4_plan.config().subgraph_mappings[2].cta_weight != 7) {
            throw std::runtime_error(
                "APPT radix-4 physical-core calibration is incorrect");
        }
        radix4_plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error(
                "APPT radix-4 physical core failed delta verification");
        }
        config.profile_appt_roles = true;
        cuntt::Plan radix4_profile_plan(config);
        radix4_profile_plan.execute(input, output, 0, 1);
        const auto radix4_metrics = radix4_profile_plan.pipeline_role_metrics();
        if (radix4_metrics.size() != 3 ||
            radix4_metrics[0].compute_globaltimer == 0 ||
            radix4_metrics[2].boundary_globaltimer == 0) {
            throw std::runtime_error(
                "APPT radix-4 per-role metrics are invalid");
        }
        config.profile_appt_roles = false;

        for (const auto word_bits : {32U, 64U}) {
            cuntt::PlanConfig register_config = config;
            register_config.word_bits = word_bits;
            register_config.modulus = word_bits == 32
                                          ? 998244353ULL
                                          : 576460756061519873ULL;
            register_config.target_ctas_per_sm = 0;
            for (auto& mapping : register_config.subgraph_mappings) {
                mapping.core =
                    cuntt::NttSubgraphCore::ApptOnlineRegisterTail;
                mapping.cta_weight = 0;
            }
            cuntt::Plan register_plan(register_config);
            const std::array<std::uint32_t, 3> expected_weights =
                word_bits == 32
                    ? std::array<std::uint32_t, 3>{8, 6, 6}
                    : std::array<std::uint32_t, 3>{10, 5, 5};
            if (register_plan.selection().implementation !=
                    "hierarchical-dataflow-appt-online-register-tail" ||
                register_plan.config().target_ctas_per_sm !=
                    (word_bits == 32 ? 3U : 2U)) {
                throw std::runtime_error(
                    "APPT register-tail physical mapping is incorrect");
            }
            for (std::size_t segment = 0; segment < expected_weights.size();
                 ++segment) {
                if (register_plan.config().subgraph_mappings[segment].cta_weight !=
                    expected_weights[segment]) {
                    throw std::runtime_error(
                        "APPT register-tail role calibration is incorrect");
                }
            }
            register_plan.execute(input, output, 0, 1);
            if (!std::all_of(output.begin(), output.end(),
                             [](std::uint64_t value) { return value == 1; })) {
                throw std::runtime_error(
                    "APPT register-tail core failed delta verification");
            }
            const auto physical_input = make_input(
                input.size(), register_config.modulus,
                0x47524f55504544ULL + word_bits);
            std::vector<std::uint64_t> physical_expected;
            register_plan.execute(physical_input, physical_expected, 0, 1);

            const std::array<cuntt::NttSubgraphCore, 8> experimental_cores = {
                cuntt::NttSubgraphCore::ApptOnlineRegisterTailWarp,
                cuntt::NttSubgraphCore::ApptOnlineRegisterTailColumnWarp,
                cuntt::NttSubgraphCore::ApptOnlineRegisterTailRadix8,
                cuntt::NttSubgraphCore::ApptOnlineRegisterTailGrouped,
                cuntt::NttSubgraphCore::
                    ApptOnlineRegisterTailGroupedWriterFinal,
                cuntt::NttSubgraphCore::
                    ApptOnlineRegisterTailGroupedWriterFinalDataTime,
                cuntt::NttSubgraphCore::
                    ApptOnlineRegisterTailGroupedWriterFinalResident,
                cuntt::NttSubgraphCore::
                    ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter,
            };
            const std::array<const char*, 8> expected_implementations = {
                "hierarchical-dataflow-appt-online-register-tail-warp",
                "hierarchical-dataflow-appt-online-register-tail-column-warp",
                "hierarchical-dataflow-appt-online-register-tail-radix8",
                "hierarchical-dataflow-appt-online-register-tail-grouped",
                "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final",
                "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-data-time",
                "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-resident",
                "hierarchical-dataflow-appt-online-register-tail-grouped-writer-final-resident-quarter",
            };
            for (std::size_t core_index = 0;
                 core_index < experimental_cores.size(); ++core_index) {
                auto physical_config = register_config;
                for (auto& mapping : physical_config.subgraph_mappings) {
                    mapping.core = experimental_cores[core_index];
                }
                if (experimental_cores[core_index] == cuntt::NttSubgraphCore::
                        ApptOnlineRegisterTailGroupedWriterFinalResident ||
                    experimental_cores[core_index] == cuntt::NttSubgraphCore::
                        ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter) {
                    physical_config.subgraph_mappings[0].data_space = 32;
                    physical_config.subgraph_mappings[1].data_space = 16;
                    physical_config.subgraph_mappings[2].data_space = 16;
                }
                cuntt::Plan physical_plan(physical_config);
                if (physical_plan.selection().implementation !=
                    expected_implementations[core_index]) {
                    throw std::runtime_error(
                        "APPT register-tail physical-core selection is incorrect");
                }
                if (experimental_cores[core_index] == cuntt::NttSubgraphCore::
                        ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter &&
                    physical_plan.config().target_ctas_per_sm != 2) {
                    throw std::runtime_error(
                        "APPT resident-quarter CTA residency is incorrect");
                }
                physical_plan.execute(physical_input, output, 0, 1);
                require_equal(
                    physical_expected, output,
                    "APPT register-tail physical core " +
                        std::string(expected_implementations[core_index]));
            }
        }

        for (auto& mapping : config.subgraph_mappings) {
            mapping.core = cuntt::NttSubgraphCore::ApptOnline;
        }

        config.subgraph_mappings[1].data_space = 8;
        config.subgraph_mappings[2].data_space = 32;
        cuntt::Plan grouped_plan(config);
        grouped_plan.execute(input, output, 0, 1);
        if (!std::all_of(output.begin(), output.end(),
                         [](std::uint64_t value) { return value == 1; })) {
            throw std::runtime_error(
                "APPT online per-fold data-space mapping failed delta verification");
        }
    }
    std::cout << "PASS true HierarchicalDataflow readiness stream\n";
}

void test_forward_backend(cuntt::Backend backend) {
    for (std::uint32_t log_n = 1; log_n <= 16; ++log_n) {
        const std::size_t batch    = log_n <= 10 ? 3 : 1;
        const std::size_t n        = 1ULL << log_n;
        const auto        input    = make_input(n * batch, cuntt::kDefaultModulus, 0x1000 + log_n);
        const auto        expected = reference_batch(input, log_n, batch, cuntt::kDefaultModulus, false);

        cuntt::PlanConfig config;
        config.log_n   = log_n;
        config.batch   = batch;
        config.backend = backend;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        const auto                 stats = plan.execute(input, output, 0, 1);
        if (stats.kernel_ms <= 0.0) {
            throw std::runtime_error("kernel timing must be positive");
        }
        require_equal(expected, output, std::string(cuntt::backend_name(backend)) + " logN=" + std::to_string(log_n));
        std::cout << "PASS forward backend=" << cuntt::backend_name(backend) << " logN=" << log_n << " batch=" << batch << '\n';
    }
}

void test_inverse_backend(cuntt::Backend backend) {
    constexpr std::array<std::uint32_t, 4> kLogSizes = {4, 8, 13, 16};
    for (const auto log_n : kLogSizes) {
        const std::size_t batch    = log_n <= 13 ? 2 : 1;
        const std::size_t n        = 1ULL << log_n;
        const auto        input    = make_input(n * batch, cuntt::kDefaultModulus, 0x5000 + log_n);
        const auto        expected = reference_batch(input, log_n, batch, cuntt::kDefaultModulus, true);

        cuntt::PlanConfig config;
        config.log_n   = log_n;
        config.batch   = batch;
        config.inverse = true;
        config.backend = backend;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, std::string(cuntt::backend_name(backend)) + " inverse logN=" + std::to_string(log_n));
        std::cout << "PASS inverse backend=" << cuntt::backend_name(backend) << " logN=" << log_n << " batch=" << batch << '\n';
    }
}

void test_custom_modulus(cuntt::Backend backend) {
    constexpr std::uint64_t kModulus = 998244353;
    constexpr std::uint32_t kLogN    = 12;
    constexpr std::size_t   kBatch   = 2;
    const std::size_t       n        = 1ULL << kLogN;
    const auto              input    = make_input(n * kBatch, kModulus, 0x7000 + static_cast<int>(backend));

    for (const bool inverse : {false, true}) {
        const auto        expected = reference_batch(input, kLogN, kBatch, kModulus, inverse);
        cuntt::PlanConfig config;
        config.log_n   = kLogN;
        config.batch   = kBatch;
        config.modulus = kModulus;
        config.inverse = inverse;
        config.backend = backend;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, std::string(cuntt::backend_name(backend)) + " custom modulus");
        std::cout << "PASS custom-modulus backend=" << cuntt::backend_name(backend) << " direction=" << (inverse ? "inverse" : "forward") << '\n';
    }
}

void test_roundtrip(cuntt::Backend backend, std::uint32_t log_n, std::size_t batch) {
    const std::size_t n     = 1ULL << log_n;
    const auto        input = make_input(n * batch, cuntt::kDefaultModulus, 0x9000 + log_n);

    cuntt::PlanConfig forward_config;
    forward_config.log_n   = log_n;
    forward_config.batch   = batch;
    forward_config.backend = backend;
    cuntt::Plan                forward_plan(forward_config);
    std::vector<std::uint64_t> transformed;
    forward_plan.execute(input, transformed, 0, 1);

    cuntt::PlanConfig inverse_config = forward_config;
    inverse_config.inverse           = true;
    cuntt::Plan                inverse_plan(inverse_config);
    std::vector<std::uint64_t> recovered;
    inverse_plan.execute(transformed, recovered, 0, 1);
    require_equal(input, recovered, "roundtrip logN=" + std::to_string(log_n));
    std::cout << "PASS roundtrip backend=" << cuntt::backend_name(backend) << " logN=" << log_n << " batch=" << batch << '\n';
}

void test_hybrid2d_backend() {
    struct TestCase {
        std::size_t   batch;
        std::uint64_t modulus;
        bool          inverse;
    };
    constexpr std::uint32_t           kLogN  = 16;
    constexpr std::array<TestCase, 5> kCases = {{{1, cuntt::kDefaultModulus, false},
                                                 {2, cuntt::kDefaultModulus, true},
                                                 {5, cuntt::kDefaultModulus, false},
                                                 {2, 998244353, false},
                                                 {2, 998244353, true}}};

    for (std::size_t case_index = 0; case_index < kCases.size(); ++case_index) {
        const auto&       test_case = kCases[case_index];
        const std::size_t n         = 1ULL << kLogN;
        const auto        input     = make_input(n * test_case.batch, test_case.modulus, 0xa000 + case_index);
        const auto        expected  = reference_batch(input, kLogN, test_case.batch, test_case.modulus, test_case.inverse);

        cuntt::PlanConfig config;
        config.log_n   = kLogN;
        config.batch   = test_case.batch;
        config.modulus = test_case.modulus;
        config.inverse = test_case.inverse;
        config.backend = cuntt::Backend::Hybrid2D;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hybrid2d batch=" + std::to_string(test_case.batch) + (test_case.inverse ? " inverse" : " forward"));
        std::cout << "PASS hybrid2d batch=" << test_case.batch << " modulus=" << test_case.modulus
                  << " direction=" << (test_case.inverse ? "inverse" : "forward") << '\n';
    }

    const std::size_t n        = 1ULL << kLogN;
    const auto        input    = make_input(n, cuntt::kDefaultModulus, 0xa100);
    const auto        expected = reference_batch(input, kLogN, 1, cuntt::kDefaultModulus, false);
    for (std::uint32_t n1_log = 6; n1_log <= 10; ++n1_log) {
        cuntt::PlanConfig config;
        config.log_n   = kLogN;
        config.backend = cuntt::Backend::Hybrid2D;
        config.n1_log  = n1_log;
        cuntt::Plan                plan(config);
        std::vector<std::uint64_t> output;
        plan.execute(input, output, 0, 1);
        require_equal(expected, output, "hybrid2d n1_log=" + std::to_string(n1_log));
        std::cout << "PASS hybrid2d n1_log=" << n1_log << '\n';
    }

    for (const auto placement :
         {cuntt::CrossTwiddlePlacement::FirstPass, cuntt::CrossTwiddlePlacement::SecondPass, cuntt::CrossTwiddlePlacement::Fused}) {
        for (const auto multiply : {cuntt::ModularMultiply::Shoup, cuntt::ModularMultiply::Barrett}) {
            cuntt::PlanConfig placement_config;
            placement_config.log_n                   = kLogN;
            placement_config.backend                 = cuntt::Backend::Hybrid2D;
            placement_config.cross_twiddle_placement = placement;
            placement_config.modular_multiply        = multiply;
            cuntt::Plan                placement_plan(placement_config);
            std::vector<std::uint64_t> placement_output;
            placement_plan.execute(input, placement_output, 0, 1);
            require_equal(expected, placement_output,
                          std::string("hybrid2d cross_twiddle=") + cuntt::cross_twiddle_placement_name(placement) +
                              " mod_multiply=" + cuntt::modular_multiply_name(multiply));
            std::cout << "PASS hybrid2d cross_twiddle=" << cuntt::cross_twiddle_placement_name(placement)
                      << " mod_multiply=" << cuntt::modular_multiply_name(multiply) << '\n';
        }
    }

    cuntt::PlanConfig legacy_barrett_config;
    legacy_barrett_config.log_n                   = kLogN;
    legacy_barrett_config.backend                 = cuntt::Backend::Hybrid2D;
    legacy_barrett_config.cross_twiddle_placement = cuntt::CrossTwiddlePlacement::FusedBarrett;
    cuntt::Plan legacy_barrett_plan(legacy_barrett_config);
    if (legacy_barrett_plan.config().cross_twiddle_placement != cuntt::CrossTwiddlePlacement::Fused ||
        legacy_barrett_plan.config().modular_multiply != cuntt::ModularMultiply::Barrett) {
        throw std::runtime_error("legacy fused-barrett was not normalized");
    }

    struct LengthCase {
        std::uint32_t log_n;
        std::uint64_t modulus;
    };
    constexpr std::array<LengthCase, 4> kLengthCases = {
        {{12, cuntt::kDefaultModulus}, {14, cuntt::kDefaultModulus}, {18, cuntt::kDefaultModulus}, {20, 1152921504577486849ULL}}};
    constexpr std::array<cuntt::ComputeUnit, 3> kUnits = {cuntt::ComputeUnit::Radix2, cuntt::ComputeUnit::Radix4, cuntt::ComputeUnit::Radix8};
    for (const auto& length_case : kLengthCases) {
        const std::size_t length          = 1ULL << length_case.log_n;
        const auto        length_input    = make_input(length, length_case.modulus, 0xa200 + length_case.log_n);
        const auto        length_expected = reference_batch(length_input, length_case.log_n, 1, length_case.modulus, false);
        for (const auto unit : kUnits) {
            cuntt::PlanConfig config;
            config.log_n        = length_case.log_n;
            config.modulus      = length_case.modulus;
            config.backend      = cuntt::Backend::Hybrid2D;
            config.compute_unit = unit;
            cuntt::Plan                plan(config);
            std::vector<std::uint64_t> output;
            plan.execute(length_input, output, 0, 1);
            require_equal(length_expected, output, "hybrid2d multi-length");
            std::cout << "PASS hybrid2d logN=" << length_case.log_n << " unit=" << cuntt::compute_unit_name(unit) << '\n';
        }
    }

    constexpr std::uint64_t kModulus30 = 1073479681ULL;
    const auto              input32    = make_input(1ULL << 16, kModulus30, 0xa300);
    const auto              expected32 = reference_batch(input32, 16, 1, kModulus30, false);
    for (const auto unit : kUnits) {
        cuntt::PlanConfig forward_config;
        forward_config.log_n        = 16;
        forward_config.modulus      = kModulus30;
        forward_config.backend      = cuntt::Backend::Hybrid2D;
        forward_config.compute_unit = unit;
        forward_config.word_bits    = 32;
        cuntt::Plan                forward_plan(forward_config);
        std::vector<std::uint64_t> transformed;
        forward_plan.execute(input32, transformed, 0, 1);
        require_equal(expected32, transformed, "hybrid2d 32-bit forward");

        cuntt::PlanConfig inverse_config = forward_config;
        inverse_config.inverse           = true;
        cuntt::Plan                inverse_plan(inverse_config);
        std::vector<std::uint64_t> recovered;
        inverse_plan.execute(transformed, recovered, 0, 1);
        require_equal(input32, recovered, "hybrid2d 32-bit roundtrip");
        std::cout << "PASS hybrid2d word_bits=32 unit=" << cuntt::compute_unit_name(unit) << '\n';
    }

    cuntt::PlanConfig barrett32_config;
    barrett32_config.log_n                   = 16;
    barrett32_config.modulus                 = kModulus30;
    barrett32_config.backend                 = cuntt::Backend::Hybrid2D;
    barrett32_config.word_bits               = 32;
    barrett32_config.cross_twiddle_placement = cuntt::CrossTwiddlePlacement::Fused;
    barrett32_config.modular_multiply        = cuntt::ModularMultiply::Barrett;
    cuntt::Plan                barrett32_plan(barrett32_config);
    std::vector<std::uint64_t> barrett32_output;
    barrett32_plan.execute(input32, barrett32_output, 0, 1);
    require_equal(expected32, barrett32_output, "hybrid2d 32-bit Barrett");
    std::cout << "PASS hybrid2d word_bits=32 mod_multiply=barrett\n";
}

void test_validation() {
    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n   = 4;
            config.modulus = 65;
            cuntt::Plan plan(config);
        },
        "composite modulus");

    cuntt::PlanConfig config;
    config.log_n = 4;
    config.batch = 1;
    cuntt::Plan                plan(config);
    std::vector<std::uint64_t> input(1ULL << config.log_n, 0);
    input.front() = config.modulus;
    std::vector<std::uint64_t> output;
    require_invalid_argument([&] { plan.execute(input, output, 0, 1); }, "unreduced coefficient");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig merge_config;
            merge_config.log_n   = 11;
            merge_config.backend = cuntt::Backend::Hybrid2D;
            cuntt::Plan merge_plan(merge_config);
        },
        "hybrid2d unsupported length");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.backend        = cuntt::Backend::Hybrid2D;
            config.rows_per_block = 3;
            cuntt::Plan plan(config);
        },
        "hybrid2d unsupported row mapping");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n        = 18;
            config.backend      = cuntt::Backend::CompactStage;
            config.output_order = cuntt::OutputOrder::BitReversed;
            cuntt::Plan plan(config);
        },
        "compact-stage unsupported length");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.output_order = cuntt::OutputOrder::BitReversed;
            cuntt::Plan plan(config);
        },
        "bit-reversed output on unsupported backend");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.modular_multiply = cuntt::ModularMultiply::Barrett;
            cuntt::Plan plan(config);
        },
        "Barrett on unsupported backend");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n            = 12;
            config.backend          = cuntt::Backend::Hybrid2D;
            config.word_bits        = 32;
            config.modulus          = 2013265921ULL;
            config.modular_multiply = cuntt::ModularMultiply::Barrett;
            cuntt::Plan plan(config);
        },
        "32-bit Barrett modulus width");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 8;
            config.backend = cuntt::Backend::HybridDataflow;
            config.flow_tile_log_n = 8;
            config.stage_space = 8;
            config.data_space = 32;
            cuntt::Plan plan(config);
        },
        "hybrid-dataflow ungenerated point");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 10;
            config.backend = cuntt::Backend::HybridDataflow;
            config.compute_unit = cuntt::ComputeUnit::Radix8;
            cuntt::Plan plan(config);
        },
        "hybrid-dataflow unsupported radix8 core");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 11;
            config.backend = cuntt::Backend::HierarchicalBarrier;
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow unsupported length");

    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 12;
            config.backend = cuntt::Backend::HierarchicalBarrier;
            config.compute_unit = cuntt::ComputeUnit::Radix2;
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow unsupported physical unit");
    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 16;
            config.backend = cuntt::Backend::HierarchicalBarrier;
            config.hierarchical_core = cuntt::HierarchicalCore::Hybrid2DRadix4;
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow ungenerated Hybrid2D core");
    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 12;
            config.backend = cuntt::Backend::HierarchicalDataflow;
            config.stage_partition = {6, 6};
            config.boundary_mappings = {{cuntt::BoundaryStorage::Ring, 0}};
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow zero ring buffers");
    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 12;
            config.backend = cuntt::Backend::HierarchicalDataflow;
            config.stage_partition = {6, 6};
            config.boundary_mappings = {
                {cuntt::BoundaryStorage::ResidentFused, 1}};
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow fully resident transform without whole core");
    require_invalid_argument(
        [] {
            cuntt::PlanConfig config;
            config.log_n = 20;
            config.backend = cuntt::Backend::HierarchicalDataflow;
            config.stage_partition = {6, 6, 8};
            config.boundary_mappings = {
                {cuntt::BoundaryStorage::ResidentFused, 1},
                {cuntt::BoundaryStorage::FullScratch, 1}};
            cuntt::Plan plan(config);
        },
        "hierarchical-dataflow resident group above physical stage limit");
    if (cuntt::parse_boundary_storage("resident-fused") !=
            cuntt::BoundaryStorage::ResidentFused ||
        std::string(cuntt::boundary_storage_name(
            cuntt::BoundaryStorage::ResidentFused)) != "resident-fused") {
        throw std::runtime_error("resident-fused boundary naming is unstable");
    }
    std::cout << "PASS API validation\n";
}

void test_appt_static_layout() {
    constexpr std::uint64_t n = 1ULL << 20;
    for (const std::uint32_t fragment_width : {8U, 16U, 32U}) {
        cuntt::ApptLayoutInfo layout;
        layout.log_n = 20;
        layout.stage_partition = {7, 7, 6};
        layout.fragment_width = fragment_width;
        layout.xor_permutation = true;
        std::vector<unsigned char> visited(n, 0);
        for (std::uint64_t natural = 0; natural < n; ++natural) {
            const std::uint64_t physical =
                cuntt::appt_static_index(natural, layout);
            if (physical >= n || visited[physical] != 0 ||
                cuntt::appt_natural_index(physical, layout) != natural) {
                throw std::runtime_error(
                    "APPT static layout is not a bijection");
            }
            visited[physical] = 1;
        }
    }
    std::cout << "PASS APPT static-layout bijection\n";
}

}  // namespace

int main() {
    try {
        const auto device = cuntt::current_device_info();
        std::cout << "Testing on " << device.name << " (sm_" << device.compute_major << device.compute_minor << ")\n";
        test_device_api();
        test_runtime_selector();
        test_forward_backend(cuntt::Backend::Baseline);
        test_forward_backend(cuntt::Backend::Tile256);
        test_inverse_backend(cuntt::Backend::Baseline);
        test_inverse_backend(cuntt::Backend::Tile256);
        test_custom_modulus(cuntt::Backend::Baseline);
        test_custom_modulus(cuntt::Backend::Tile256);
        test_roundtrip(cuntt::Backend::Baseline, 13, 2);
        test_roundtrip(cuntt::Backend::Tile256, 8, 3);
        test_roundtrip(cuntt::Backend::Tile256, 13, 2);
        test_roundtrip(cuntt::Backend::Tile256, 16, 1);
        test_hybrid2d_backend();
        test_roundtrip(cuntt::Backend::Hybrid2D, 16, 3);
        test_compact_stage_backend();
        test_stage_pipeline_backend();
        test_hybrid_dataflow_backend();
        test_hierarchical_dataflow_backend();
        test_true_hierarchical_streaming_backend();
        test_appt_static_layout();
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All cuNTT tests passed\n";
    return 0;
}
