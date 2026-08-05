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
    std::cout << "PASS API validation\n";
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
        test_validation();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All cuNTT tests passed\n";
    return 0;
}
