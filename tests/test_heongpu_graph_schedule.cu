// Standalone: nvcc -std=c++17 -Iadapters/heongpu tests/test_heongpu_graph_schedule.cu -o /tmp/test_heongpu_graph_schedule
// --cpu-only runs pure statistics/API checks without initializing CUDA.
#include "graph_schedule.hpp"

#include <array>
#include <iostream>
#include <limits>
#include <string>

namespace {
using namespace cubutterfly_heongpu;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception, class Function>
void expect_throw(Function function, const char* message) {
    try { function(); }
    catch (const Exception&) { return; }
    throw std::runtime_error(message);
}

void test_cpu() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    require(graph_schedule_median_ms({3, 1, 2}) == 2, "odd median");
    require(graph_schedule_median_ms({4, 1, 3, 2}) == 2.5, "even median");
    require(graph_schedule_median_ms({1, 1}) == 1, "median ties");
    const double tiny = std::numeric_limits<double>::denorm_min();
    require(graph_schedule_median_ms({tiny, tiny}) == tiny, "median underflow");
    const double huge = std::numeric_limits<double>::max();
    require(graph_schedule_median_ms({huge, huge}) == huge, "median overflow");
    expect_throw<std::invalid_argument>([] { graph_schedule_median_ms({}); }, "empty samples accepted");
    for (double invalid : {nan, inf, -inf, 0.0, -1.0}) {
        expect_throw<std::invalid_argument>([&] { graph_schedule_median_ms({1, invalid}); },
                                             "invalid sample accepted");
    }
    require(graph_schedule_select({3, 1, 2}) == 1, "minimum selection");
    require(graph_schedule_select({2, 1, 1}) == 1, "tie did not keep first candidate");
    require(graph_schedule_select({nan, 0, -1, inf, 4}) == 4, "invalid median selected");
    expect_throw<std::invalid_argument>([] { graph_schedule_select({}); }, "empty medians accepted");
    expect_throw<std::invalid_argument>([&] { graph_schedule_select({nan, 0, -1, inf}); },
                                         "all-invalid medians accepted");
    expect_throw<std::invalid_argument>([] { GraphSchedule schedule({}); }, "empty candidates accepted");
    expect_throw<std::invalid_argument>([] { GraphSchedule schedule(std::vector<GraphScheduleCandidate>{{"empty", {}}}); },
                                         "empty callable accepted");
    GraphSchedule schedule({{"noop", [](cudaStream_t) {}}});
    require(!schedule.tuned() && schedule.measurements().empty(), "new schedule is tuned");
    expect_throw<std::logic_error>([&] { schedule.launch(nullptr); }, "untuned launch accepted");
    expect_throw<std::logic_error>([&] { schedule.selected_index(); }, "untuned selection accepted");
    for (cudaStream_t stream : {static_cast<cudaStream_t>(nullptr), cudaStreamLegacy, cudaStreamPerThread})
        expect_throw<std::invalid_argument>([&] { schedule.tune(stream); }, "default stream accepted");
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
};

struct Device {
    float* data = nullptr;
    explicit Device(std::size_t size) { check(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(float))); }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
};

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ~Graph() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    Graph() = default;
    Graph(const Graph&) = delete;
};

__global__ void affine(const float* input, float* output, int n, float scale, float bias) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * scale + bias;
}

void test_gpu() {
    constexpr int n = 65536;
    Stream stream;
    Device input(n), scratch(n), output(n);
    Graph direct, staged;
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    affine<<<n / 256, 256, 0, stream.value>>>(input.data, output.data, n, 2, 1);
    check(cudaStreamEndCapture(stream.value, &direct.graph));
    check(cudaGraphInstantiate(&direct.executable, direct.graph, nullptr, nullptr, 0));
    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    affine<<<n / 256, 256, 0, stream.value>>>(input.data, scratch.data, n, 2, 0);
    affine<<<n / 256, 256, 0, stream.value>>>(scratch.data, output.data, n, 1, 1);
    check(cudaStreamEndCapture(stream.value, &staged.graph));
    check(cudaGraphInstantiate(&staged.executable, staged.graph, nullptr, nullptr, 0));

    const auto validate = [&](cudaGraphExec_t graph, float value) {
        std::vector<float> host(n, value), result(n);
        check(cudaMemcpyAsync(input.data, host.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        check(cudaGraphLaunch(graph, stream.value));
        check(cudaMemcpyAsync(result.data(), output.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
        check(cudaStreamSynchronize(stream.value));
        for (float actual : result) require(actual == value * 2 + 1, "candidate correctness");
    };
    // Validate all candidates before giving them to the scheduler, using changed
    // input and then restoring the original input at the same device address.
    for (float value : {3.0f, -2.0f, 3.0f}) {
        validate(direct.executable, value);
        validate(staged.executable, value);
    }

    std::array<std::size_t, 2> launches{{0, 0}};
    std::vector<int> order;
    const std::vector<GraphScheduleCandidate> candidates{
        {"direct", [&](cudaStream_t s) { ++launches[0]; order.push_back(0); check(cudaGraphLaunch(direct.executable, s)); }},
        {"staged", [&](cudaStream_t s) { ++launches[1]; order.push_back(1); check(cudaGraphLaunch(staged.executable, s)); }}
    };
    GraphSchedule schedule(candidates);
    const GraphScheduleOptions options{2, 5, 3};
    std::size_t before = 0, after = 0;
    schedule.tune(stream.value, options, [&] { ++before; }, [&] { ++after; });
    require(before == options.rounds && after == 2 * options.rounds, "guard callback counts");
    require(schedule.tuned() && schedule.selected_index() < 2, "no measured selection");
    require(schedule.selected_candidate().name == candidates[schedule.selected_index()].name, "selected name");
    require(schedule.measurements().size() == 2, "missing candidate measurements");
    std::vector<double> medians;
    for (std::size_t i = 0; i < 2; ++i) {
        const auto& result = schedule.measurements()[i];
        require(result.name == candidates[i].name, "measurement identity");
        require(result.round_ms.size() == options.rounds, "missing real samples");
        require(result.median_ms == graph_schedule_median_ms(result.round_ms), "stored median");
        require(launches[i] == options.warmup + options.rounds * options.replays, "tune did not replay every candidate");
        medians.push_back(result.median_ms);
    }
    require(schedule.selected_index() == graph_schedule_select(medians), "selection differs from measured minimum");
    std::size_t position = 2 * options.warmup;
    for (std::size_t round = 0; round < options.rounds; ++round)
        for (std::size_t offset = 0; offset < 2; ++offset)
            for (std::size_t replay = 0; replay < options.replays; ++replay)
                require(order.at(position++) == static_cast<int>((round + offset) % 2), "round order did not rotate");
    const auto counts = launches;
    const auto selected = schedule.selected_index();
    std::vector<float> changed(n, -7), result(n);
    check(cudaMemcpyAsync(input.data, changed.data(), n * sizeof(float), cudaMemcpyHostToDevice, stream.value));
    check(cudaMemsetAsync(output.data, 0, n * sizeof(float), stream.value));
    schedule.launch(stream.value);
    check(cudaMemcpyAsync(result.data(), output.data, n * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    check(cudaStreamSynchronize(stream.value));
    for (float actual : result) require(actual == -13, "selected launch output");
    require(launches[selected] == counts[selected] + 1 && launches[1 - selected] == counts[1 - selected],
            "launch did not dispatch only the selected candidate");

    // A second instance must actually measure its own rounds, not reuse results.
    GraphSchedule second(candidates);
    const auto prior = launches;
    second.tune(stream.value, {1, 3, 2});
    require(second.measurements()[0].round_ms.size() == 3 && schedule.measurements()[0].round_ms.size() == 5,
            "instances share measurement state");
    require(launches[0] == prior[0] + 7 && launches[1] == prior[1] + 7,
            "second tune did not collect independent samples");

    expect_throw<std::runtime_error>([&] {
        second.tune(stream.value, {0, 1, 1}, [] { throw std::runtime_error("guard failure"); });
    }, "guard failure swallowed");
    require(!second.tuned() && second.measurements().empty(), "failed tune retained stale selection");
    expect_throw<std::logic_error>([&] { second.launch(stream.value); }, "failed tune still launches");
    expect_throw<std::invalid_argument>([&] { second.tune(stream.value, {0, 0, 1}); }, "zero rounds accepted");
    expect_throw<std::invalid_argument>([&] { second.tune(stream.value, {0, 1, 0}); }, "zero replays accepted");

    check(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
    expect_throw<std::logic_error>([&] { second.tune(stream.value); }, "tuning inside capture accepted");
    Graph empty;
    check(cudaStreamEndCapture(stream.value, &empty.graph));
}
} // namespace

int main(int argc, char** argv) {
    try {
        test_cpu();
        if (argc == 2 && std::string(argv[1]) == "--cpu-only") {
            std::cout << "graph schedule CPU checks passed\n";
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("usage: test_heongpu_graph_schedule [--cpu-only]");
        test_gpu();
        std::cout << "graph schedule CPU and GPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
