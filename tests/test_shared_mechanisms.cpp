#include "cubutterfly/mapping.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct GraphStream {
    cudaStream_t value = nullptr;
    GraphStream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~GraphStream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};

template <typename Value>
struct GraphBuffer {
    Value* data = nullptr;
    explicit GraphBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Value)));
    }
    ~GraphBuffer() { cudaFree(data); }
};

struct CapturedGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~CapturedGraph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    template <typename Launch>
    void capture(cudaStream_t stream, Launch launch) {
        check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        try {
            launch();
        } catch (...) {
            cudaStreamEndCapture(stream, &graph);
            throw;
        }
        check_cuda(cudaStreamEndCapture(stream, &graph));
        check_cuda(cudaGraphInstantiate(&executable, graph, 0));
    }
};

template <typename Value>
bool graph_value_equal(Value actual, Value expected) { return actual == expected; }

bool graph_value_equal(cuntt::Complex32 actual, cuntt::Complex32 expected) {
    return actual.real == expected.real && actual.imag == expected.imag;
}

bool graph_value_equal(cuntt::Complex64 actual, cuntt::Complex64 expected) {
    return actual.real == expected.real && actual.imag == expected.imag;
}

template <typename Plan, typename Value>
void check_graph_replay(Plan& plan, const std::vector<Value>& input, const std::string& label) {
    GraphStream stream;
    GraphBuffer<Value> device_input(input.size()), first_output(input.size()), second_output(input.size());
    const auto bytes = input.size() * sizeof(Value);
    const auto previous_stream = plan.stream();
    plan.set_stream(stream.value);
    struct StreamBinding {
        Plan& plan;
        cudaStream_t previous, current;
        ~StreamBinding() {
            cudaStreamSynchronize(current);
            plan.set_stream(previous);
        }
    } binding{plan, previous_stream, stream.value};
    auto changed = input;
    std::rotate(changed.begin(), changed.begin() + 1, changed.end());
    const auto equal = [](const auto& first, const auto& second) {
        return std::equal(first.begin(), first.end(), second.begin(),
                          [](auto a, auto b) { return graph_value_equal(a, b); });
    };
    require(!equal(input, changed), label + " graph input did not change");
    auto upload = [&](const auto& values) {
        check_cuda(cudaMemcpyAsync(device_input.data, values.data(), bytes,
                                   cudaMemcpyHostToDevice, stream.value));
    };
    auto download = [&](const auto& buffer) {
        std::vector<Value> values(input.size());
        check_cuda(cudaMemcpyAsync(values.data(), buffer.data, bytes,
                                   cudaMemcpyDeviceToHost, stream.value));
        check_cuda(cudaStreamSynchronize(stream.value));
        return values;
    };
    auto direct = [&](const auto& values) {
        upload(values);
        plan.execute_async(device_input.data, first_output.data);
        check_cuda(cudaStreamSynchronize(stream.value));
        return download(first_output);
    };
    // The same plan and core supply exact direct results; graph execution
    // changes scheduling only and introduces no new floating-point threshold.
    const auto expected = direct(input);
    const auto changed_expected = direct(changed);
    require(!equal(expected, changed_expected), label + " direct output did not change");
    for (unsigned capture = 0; capture < 2; ++capture) {
        direct(input);  // Warm up and retire all work before each capture.
        CapturedGraph graph;
        graph.capture(stream.value, [&] {
            plan.execute_async(device_input.data, first_output.data);
            plan.execute_async(device_input.data, second_output.data);
        });
        for (bool use_changed : {false, true, false}) {
            upload(use_changed ? changed : input);
            check_cuda(cudaMemsetAsync(first_output.data, 0xa5, bytes, stream.value));
            check_cuda(cudaMemsetAsync(second_output.data, 0xa5, bytes, stream.value));
            check_cuda(cudaGraphLaunch(graph.executable, stream.value));
            check_cuda(cudaStreamSynchronize(stream.value));
            const auto& reference = use_changed ? changed_expected : expected;
            require(equal(download(first_output), reference), label + " first captured output mismatch");
            require(equal(download(second_output), reference), label + " second captured output mismatch");
        }
        require(equal(direct(changed), changed_expected), label + " direct call after graph mismatch");
    }
    std::cout << label << " graph=two-captures/two-calls/changed-and-restored-input direct_equal=true\n";
}

template <typename Callable>
void require_invalid(Callable callable, const std::string& label) {
    try {
        callable();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(label + " accepted an invalid shared organization");
}

struct Organization {
    unsigned log_n = 8;
    std::size_t batch = 5;
    unsigned batch_tile = 1;
    unsigned threads = 128;
    bool aligned = false;
    bool inverse = false;
    bool overlap = false;
    bool fusion = false;
    bool leaves = false;
    int plain_group = -1;

    std::vector<unsigned> logical_partition() const {
        if (leaves) return {2, 3, 5};
        return fusion ? std::vector<unsigned>{2, 2, 4} : std::vector<unsigned>{4, 4};
    }

    std::vector<unsigned> physical_partition() const {
        return leaves ? std::vector<unsigned>{5, 5} : std::vector<unsigned>{4, 4};
    }

    std::vector<std::vector<unsigned>> local_partition() const {
        if (!leaves) return {};
        std::vector<std::vector<unsigned>> parts{{2, 3}, {3, 2}};
        if (plain_group >= 0) parts.at(plain_group).clear();
        return parts;
    }
};

template <typename Config>
void check_mapping(const Config& config) {
    const auto mapping = cubutterfly::serialize_mapping(config);
    auto replay = config;
    replay.batch_tile_count = 1;
    replay.stage_overlap = false;
    replay.local_stage_partitions.clear();
    replay.stage_partition = {config.log_n};
    cubutterfly::apply_serialized_mapping(mapping, replay);
    require(cubutterfly::serialize_mapping(replay) == mapping, "mapping roundtrip changed shared choices");
    require(replay.stage_partition == config.stage_partition &&
            replay.local_stage_partitions == config.local_stage_partitions &&
            replay.batch_tile_count == config.batch_tile_count && replay.stage_overlap == config.stage_overlap,
            "mapping replay lost shared organization");
    require(replay.batch == config.batch && replay.log_n == config.log_n && replay.inverse == config.inverse,
            "mapping replay changed transform semantics");
}

void check_flow(const cuntt::MixedDataflowPlan& flow, const Organization& organization,
                std::size_t value_bytes, std::size_t workspace_bytes) {
    const auto physical = organization.physical_partition();
    const auto logical = organization.logical_partition();
    require(flow.executable && flow.graph.batch == organization.batch && flow.graph.log_n == organization.log_n &&
            flow.graph.inverse == organization.inverse && flow.stage_partition == logical,
            "shared plan changed logical semantics");
    require(flow.dispatch == (organization.overlap ? cuntt::DataflowDispatch::BatchPipeline
                                                  : cuntt::DataflowDispatch::SharedIterative),
            "shared dispatch differs from requested schedule");
    require(flow.batch_tile_count == organization.batch_tile && flow.execution_groups.size() == physical.size(),
            "shared physical group or batch tile mismatch");
    require(flow.execution_boundaries.size() == physical.size() - 1, "fused edge was materialized");
    if (organization.fusion || organization.leaves)
        require(flow.boundary_policies.front() == cuntt::BoundaryPolicy::ResidentFused,
                "logical resident boundary disappeared");
    unsigned first = 0;
    for (std::size_t index = 0; index < physical.size(); ++index) {
        const auto& group = flow.execution_groups[index];
        const auto batch_space = organization.overlap ? 1U : organization.batch_tile;
        const auto launch_batches = organization.overlap ? std::min(organization.batch, std::size_t(organization.batch_tile))
                                                        : (organization.batch - 1) / batch_space + 1;
        const auto expected_grid = launch_batches << (organization.log_n - physical[index]);
        require(group.first_stage == first && group.stage_count == physical[index] &&
                group.threads == organization.threads && group.batch_space == batch_space &&
                group.grid_ctas == expected_grid &&
                group.batch_time == (organization.batch - 1) / organization.batch_tile + 1,
                "IR physical launch shape differs from shared kernel");
        require(group.shared_layout == (organization.aligned ? "writer-aligned" : "linear"),
                "shared writer layout was ignored");
        require(group.live_shared_bytes == 2 * (std::size_t{1} << physical[index]) * value_bytes,
                "serial CTA batches changed the shared live state");
        if (organization.leaves && !organization.local_partition()[index].empty())
            require(group.local_stage_partition == organization.local_partition()[index] &&
                    group.codelet == "native-register-subgraph" && group.compiler_resources_known,
                    "resident leaf was not compiled or projected");
        else
            require(group.local_stage_partition.empty(), "plain group unexpectedly acquired resident leaves");
        first += physical[index];
    }
    const auto boundary = organization.overlap ? cuntt::BoundaryPolicy::RingBuffer : cuntt::BoundaryPolicy::GlobalScratch;
    require(flow.execution_boundaries.front().storage == boundary, "physical boundary schedule mismatch");
    const auto expected_workspace = (std::size_t{1} << organization.log_n) * value_bytes *
        (organization.overlap ? 2 * organization.batch_tile : organization.batch);
    require(workspace_bytes == expected_workspace, "workspace does not match physical boundary schedule");
}

template <typename Value, typename Reference>
std::vector<Value> reference_batches(const std::vector<Value>& input, const Organization& organization,
                                     Reference reference) {
    const auto n = std::size_t{1} << organization.log_n;
    auto expected = input;
    for (std::size_t batch = 0; batch < organization.batch; ++batch) {
        std::vector<Value> slice(input.begin() + batch * n, input.begin() + (batch + 1) * n);
        reference(slice);
        std::copy(slice.begin(), slice.end(), expected.begin() + batch * n);
    }
    return expected;
}

void reference_fft_same_precision(std::vector<cuntt::Complex32>& values, bool inverse, bool normalize_inverse) {
    require(!values.empty() && (values.size() & (values.size() - 1)) == 0,
            "FFT reference requires a nonzero power of two");
    std::size_t reversed = 0;
    for (std::size_t index = 1; index < values.size(); ++index) {
        std::size_t bit = values.size() >> 1;
        while (reversed & bit) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (index < reversed) std::swap(values[index], values[reversed]);
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    for (std::size_t half = 1; half < values.size(); half <<= 1) {
        for (std::size_t offset = 0; offset < half; ++offset) {
            const double angle = (inverse ? 1.0 : -1.0) * pi * static_cast<double>(offset) / half;
            // Coefficient setup follows the runtime's double-to-float path;
            // every butterfly product and sum below remains FP32.
            const float wr = static_cast<float>(std::cos(angle));
            const float wi = static_cast<float>(std::sin(angle));
            for (std::size_t base = 0; base < values.size(); base += 2 * half) {
                const auto left = values[base + offset];
                const auto right = values[base + half + offset];
                const float real = right.real * wr - right.imag * wi;
                const float imag = right.real * wi + right.imag * wr;
                values[base + offset] = {left.real + real, left.imag + imag};
                values[base + half + offset] = {left.real - real, left.imag - imag};
            }
        }
    }
    if (inverse && normalize_inverse) {
        const float scale = 1.0F / static_cast<float>(values.size());
        for (auto& value : values) {
            value.real *= scale;
            value.imag *= scale;
        }
    }
}

void reference_fft_same_precision(std::vector<cuntt::Complex64>& values, bool inverse, bool normalize_inverse) {
    cuntt::reference_fft(values, inverse, normalize_inverse);
}

void record_errors(const std::string& label, const std::vector<double>& actual, const std::vector<double>& expected) {
    require(actual.size() == expected.size(), label + " output extent mismatch");
    double max_abs = 0;
    long double squared_error = 0, squared_reference = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        require(std::isfinite(actual[index]), label + " produced a nonfinite value");
        const auto error = actual[index] - expected[index];
        max_abs = std::max(max_abs, std::abs(error));
        squared_error += static_cast<long double>(error) * error;
        squared_reference += static_cast<long double>(expected[index]) * expected[index];
    }
    const auto l2 = std::sqrt(squared_error);
    const auto relative_l2 = squared_reference == 0 ? l2 : l2 / std::sqrt(squared_reference);
    std::cout << label << " reference=same-precision max_abs=" << max_abs << " l2=" << l2
              << " relative_l2=" << relative_l2 << '\n';
}

template <typename Real = float, typename Integer = std::uint32_t>
void run_butterfly(cuntt::ButterflyOperator op, const Organization& organization, const std::string& label,
                   bool graph = false) {
    using Complex = std::conditional_t<std::is_same_v<Real, double>, cuntt::Complex64, cuntt::Complex32>;
    cuntt::ButterflyConfig config;
    config.op = op;
    config.backend = cuntt::ButterflyBackend::SharedIterative;
    config.compute_unit = cuntt::ComputeUnit::Radix2;
    config.log_n = organization.log_n;
    config.batch = organization.batch;
    config.batch_tile_count = organization.batch_tile;
    config.tile_threads = organization.threads;
    config.shared_layout = organization.aligned ? cuntt::SharedLayout::WriterAligned : cuntt::SharedLayout::Linear;
    config.inverse = organization.inverse;
    config.stage_overlap = organization.overlap;
    config.stage_partition = organization.logical_partition();
    config.local_stage_partitions = organization.local_partition();
    config.precision = std::is_same_v<Real, double> ? cuntt::ButterflyPrecision::Fp64 : cuntt::ButterflyPrecision::Fp32;
    if (organization.fusion || organization.leaves) {
        config.boundaries.resize(2);
        config.boundaries.front().residency = cuntt::FftBoundaryResidency::Fused;
    }
    if (op == cuntt::ButterflyOperator::SubsetZeta || op == cuntt::ButterflyOperator::SupersetZeta)
        config.precision = sizeof(Integer) == 8 ? cuntt::ButterflyPrecision::Uint64 : cuntt::ButterflyPrecision::Uint32;
    if (op == cuntt::ButterflyOperator::Structured2x2)
        config.stage_matrices = {{0.75, 0.125, -0.25, 1.0}};
    cuntt::ButterflyPlan plan(config);
    check_mapping(plan.config());
    const bool integer = op == cuntt::ButterflyOperator::SubsetZeta || op == cuntt::ButterflyOperator::SupersetZeta;
    const auto value_bytes = integer ? sizeof(Integer) : op == cuntt::ButterflyOperator::Fft ? sizeof(Complex) : sizeof(Real);
    check_flow(plan.dataflow_plan(), organization, value_bytes, plan.workspace_size());
    const auto count = organization.batch << organization.log_n;
    std::mt19937_64 random(100 + organization.batch_tile);
    if (integer) {
        std::vector<Integer> input(count), output;
        for (auto& value : input) value = random();
        const auto expected = reference_batches(input, organization, [&](auto& slice) {
            if (op == cuntt::ButterflyOperator::SubsetZeta) cuntt::reference_subset_zeta(slice, config.inverse);
            else cuntt::reference_superset_zeta(slice, config.inverse);
        });
        plan.execute(input, output, 0, 1);
        require(output == expected, label + " integer output mismatch");
        if (graph) check_graph_replay(plan, input, label);
        std::cout << label << " exact=true\n";
    } else if (op == cuntt::ButterflyOperator::Fft) {
        std::vector<Complex> input(count), output;
        for (auto& value : input)
            value = {Real(int(random() % 1024) - 512) / 512, Real(int(random() % 1024) - 512) / 512};
        const auto expected = reference_batches(input, organization, [&](auto& slice) {
            reference_fft_same_precision(slice, config.inverse, config.normalize_inverse);
        });
        plan.execute(input, output, 0, 1);
        require(output.size() == expected.size(), label + " output extent mismatch");
        std::vector<double> actual_components, expected_components;
        for (std::size_t index = 0; index < output.size(); ++index) {
            actual_components.insert(actual_components.end(), {output[index].real, output[index].imag});
            expected_components.insert(expected_components.end(), {expected[index].real, expected[index].imag});
        }
        record_errors(label, actual_components, expected_components);
        if (graph) check_graph_replay(plan, input, label);
    } else {
        std::vector<Real> input(count), output;
        for (auto& value : input) value = Real(int(random() % 1024) - 512) / 512;
        const auto expected = reference_batches(input, organization, [&](auto& slice) {
            if (op == cuntt::ButterflyOperator::Fwht)
                cuntt::reference_fwht(slice, config.inverse, config.normalize_inverse);
            else cuntt::reference_structured_2x2(slice, config.stage_matrices, config.inverse);
        });
        plan.execute(input, output, 0, 1);
        record_errors(label, std::vector<double>(output.begin(), output.end()),
                      std::vector<double>(expected.begin(), expected.end()));
        if (graph) check_graph_replay(plan, input, label);
    }
}

void run_ntt(const Organization& organization, const std::string& label, bool graph = false) {
    cuntt::PlanConfig config;
    config.backend = cuntt::Backend::SharedIterative;
    config.compute_unit = cuntt::ComputeUnit::Radix2;
    config.log_n = organization.log_n;
    config.batch = organization.batch;
    config.batch_tile_count = organization.batch_tile;
    config.threads_per_block = organization.threads;
    config.stage_overlap = organization.overlap;
    config.inverse = organization.inverse;
    config.dataflow_layout = organization.aligned ? cuntt::DataflowLayout::HermesXor : cuntt::DataflowLayout::Linear;
    config.stage_partition = organization.logical_partition();
    config.local_stage_partitions = organization.local_partition();
    config.word_bits = organization.batch_tile % 2 ? 32 : 64;
    config.modulus = config.word_bits == 32 ? 998244353 : cuntt::kDefaultModulus;
    config.output_order = organization.inverse ? cuntt::OutputOrder::BitReversed : cuntt::OutputOrder::Natural;
    if (organization.fusion || organization.leaves) {
        config.boundary_mappings.resize(2);
        config.boundary_mappings.front().storage = cuntt::BoundaryStorage::ResidentFused;
    }
    cuntt::Plan plan(config);
    check_mapping(plan.config());
    require(plan.config().stage_partition == config.stage_partition &&
            plan.config().boundary_mappings.size() == config.boundary_mappings.size(),
            "NTT config exposed physical partition instead of logical replay");
    check_flow(plan.dataflow_plan(), organization, config.word_bits / 8, plan.workspace_size());
    std::mt19937_64 random(200 + organization.batch_tile);
    std::vector<std::uint64_t> input(organization.batch << organization.log_n), output;
    for (auto& value : input) value = random() % config.modulus;
    const auto expected = reference_batches(input, organization, [&](auto& slice) {
        cuntt::reference_ntt(slice, config.modulus, config.inverse);
        if (config.output_order == cuntt::OutputOrder::BitReversed) {
            for (unsigned index = 0; index < slice.size(); ++index) {
                unsigned reversed = 0;
                for (unsigned bit = 0; bit < config.log_n; ++bit)
                    reversed = (reversed << 1) | ((index >> bit) & 1U);
                if (index < reversed) std::swap(slice[index], slice[reversed]);
            }
        }
    });
    plan.execute(input, output, 0, 1);
    require(output == expected, label + " NTT exact output mismatch");
    if (graph) {
        if (config.word_bits == 32) {
            const std::vector<std::uint32_t> typed_input(input.begin(), input.end());
            check_graph_replay(plan, typed_input, label);
        } else check_graph_replay(plan, input, label);
    }
    std::cout << label << " word_bits=" << config.word_bits << " exact=true\n";
}

void run_all(const Organization& organization, const std::string& label, bool wide = false,
             bool graph = false) {
    for (const auto op : {cuntt::ButterflyOperator::Fft, cuntt::ButterflyOperator::Fwht,
                          cuntt::ButterflyOperator::SubsetZeta, cuntt::ButterflyOperator::SupersetZeta,
                          cuntt::ButterflyOperator::Structured2x2})
        if (wide) run_butterfly<double, std::uint64_t>(op, organization, label + " " + cuntt::butterfly_operator_name(op), graph);
        else run_butterfly(op, organization, label + " " + cuntt::butterfly_operator_name(op), graph);
    if (!wide) run_ntt(organization, label + " ntt", graph);
}

void negative_cases() {
    for (unsigned tile : {0U, 5U}) {
        cuntt::ButterflyConfig butterfly;
        butterfly.backend = cuntt::ButterflyBackend::SharedIterative;
        butterfly.log_n = 8;
        butterfly.batch_tile_count = tile;
        require_invalid([&] { cuntt::ButterflyPlan plan(butterfly); }, "butterfly bulk batch tile");
        cuntt::PlanConfig ntt;
        ntt.backend = cuntt::Backend::SharedIterative;
        ntt.log_n = 8;
        ntt.batch_tile_count = tile;
        require_invalid([&] { cuntt::Plan plan(ntt); }, "NTT bulk batch tile");
    }
    cuntt::ButterflyConfig butterfly;
    butterfly.backend = cuntt::ButterflyBackend::SharedIterative;
    butterfly.log_n = 8;
    butterfly.stage_partition = {4, 4};
    butterfly.boundaries.resize(1);
    butterfly.boundaries.front().residency = cuntt::FftBoundaryResidency::Fused;
    butterfly.stage_overlap = true;
    require_invalid([&] { cuntt::ButterflyPlan plan(butterfly); }, "butterfly fully fused overlap");
    cuntt::PlanConfig ntt;
    ntt.backend = cuntt::Backend::SharedIterative;
    ntt.log_n = 8;
    ntt.stage_partition = {4, 4};
    ntt.boundary_mappings.resize(1);
    ntt.boundary_mappings.front().storage = cuntt::BoundaryStorage::ResidentFused;
    ntt.stage_overlap = true;
    require_invalid([&] { cuntt::Plan plan(ntt); }, "NTT fully fused overlap");
    butterfly.stage_overlap = false;
    butterfly.batch = std::size_t{1} << 32;
    butterfly.batch_tile_count = 4;
    require_invalid([&] { cuntt::ButterflyPlan plan(butterfly); }, "butterfly bulk batch counter overflow");
    ntt.stage_overlap = false;
    ntt.batch = std::size_t{1} << 32;
    ntt.batch_tile_count = 4;
    require_invalid([&] { cuntt::Plan plan(ntt); }, "NTT bulk batch counter overflow");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(12);
        const bool resident_only = argc == 2 && std::string(argv[1]) == "--resident-only";
        const bool wide = argc == 2 && std::string(argv[1]) == "--wide";
        const bool graph = argc == 2 && std::string(argv[1]) == "--graph";
        require(argc == 1 || resident_only || wide || graph,
                "usage: test_shared_mechanisms [--resident-only|--wide|--graph]");
        if (graph) {
            for (bool inverse : {false, true}) for (unsigned tile : {2U, 3U, 5U, 6U}) {
                Organization organization;
                organization.batch_tile = tile;
                organization.overlap = true;
                organization.inverse = inverse;
                organization.aligned = inverse;
                const auto label = "graph-tile" + std::to_string(tile) + (inverse ? "-inverse" : "-forward");
                run_all(organization, label, false, true);
                run_all(organization, label + "-wide", true, true);
            }
            std::cout << "shared graph contract cases passed: 88\n";
        } else if (wide) {
            Organization organization;
            organization.batch = 7;
            organization.batch_tile = 4;
            organization.threads = 256;
            run_all(organization, "wide-bulk4", true);
            organization.batch = 5;
            organization.batch_tile = 3;
            organization.threads = 96;
            organization.aligned = true;
            organization.inverse = true;
            organization.fusion = true;
            organization.overlap = true;
            run_all(organization, "wide-logical-fusion-overlap", true);
            std::cout << "shared wide contract cases passed: 10\n";
        } else if (resident_only) {
            const char* mode = std::getenv("CUBUTTERFLY_COMPILE_MODE");
            require(mode && (std::string(mode) == "research" || std::string(mode) == "on-demand"),
                    "resident checks require CUBUTTERFLY_COMPILE_MODE=research or on-demand");
            Organization organization;
            organization.log_n = 10;
            organization.batch_tile = 4;
            organization.threads = 96;
            organization.aligned = true;
            organization.inverse = true;
            organization.leaves = true;
            run_all(organization, "resident-leaves-fused-bulk4");
            organization.batch_tile = 3;
            organization.overlap = true;
            run_all(organization, "resident-leaves-fused-overlap");
            for (int group : {0, 1}) {
                organization.plain_group = group;
                run_all(organization, "resident-mixed-plain-group" + std::to_string(group));
            }
            std::cout << "shared resident contract cases passed: 24\n";
        } else {
            for (unsigned tile : {1U, 2U, 3U, 4U}) {
                Organization organization;
                organization.batch = tile % 2 ? 5 : 7;
                organization.batch_tile = tile;
                organization.threads = tile == 1 ? 96 : tile == 2 ? 512 : tile == 3 ? 128 : 256;
                organization.aligned = tile % 2 == 0;
                organization.inverse = tile % 2 == 0;
                run_all(organization, "bulk" + std::to_string(tile));
            }
            for (bool overlap : {false, true}) {
                Organization organization;
                organization.batch = 7;
                organization.batch_tile = 3;
                organization.threads = 96;
                organization.aligned = true;
                organization.inverse = true;
                organization.fusion = true;
                organization.overlap = overlap;
                run_all(organization, overlap ? "logical-fusion-overlap" : "logical-fusion-bulk");
            }
            for (unsigned tile : {16U, 17U}) for (bool inverse : {false, true}) {
                Organization organization;
                organization.batch = 16;
                organization.batch_tile = tile;
                organization.overlap = true;
                organization.inverse = inverse;
                run_all(organization, "one-tile" + std::to_string(tile));
                run_all(organization, "one-tile-wide" + std::to_string(tile), true);
            }
            negative_cases();
            std::cout << "shared contract cases passed: 80; negative cases passed: 8\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "shared mechanisms: " << error.what() << '\n';
        return 1;
    }
}
