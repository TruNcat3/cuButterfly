#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"
#include "cubutterfly/mapping.hpp"
#include "stage_probe.hpp"

namespace {

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
    }
}

class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t bytes) {
        check_cuda(cudaMalloc(&pointer_, bytes), "allocate stage probe buffer");
    }

    ~DeviceBuffer() {
        cudaFree(pointer_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void* data() const noexcept { return pointer_; }

  private:
    void* pointer_ = nullptr;
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_output_order_inventory() {
    cuntt::PlanConfig config;
    config.log_n = 16;
    config.batch = 16;
    config.output_order = cuntt::OutputOrder::BitReversed;
    const auto candidates = cubutterfly::portable_ntt_candidates(config);
    require(!candidates.empty(), "bit-reversed semantic cell has no search candidates");
    for (const auto& candidate : candidates) {
        require(candidate.output_order == config.output_order,
                "candidate projection changed output semantics");
        require(candidate.backend == cuntt::Backend::SharedIterative,
                "bit-reversed inventory includes a legacy family without this writeback");
    }
    config.output_order = cuntt::OutputOrder::Natural;
    const auto natural = cubutterfly::portable_ntt_candidates(config);
    require(std::any_of(natural.begin(), natural.end(), [](const auto& c) {
        return c.backend == cuntt::Backend::Hybrid2D;
    }), "output-order filter removed natural-order optimized candidates");
}

void test_shared_iterative_probe() {
    constexpr std::uint32_t log_n = 4;
    constexpr std::size_t batch = 1;
    constexpr std::uint64_t modulus = cuntt::kDefaultModulus;
    const std::size_t points = batch * (1ULL << log_n);
    std::vector<std::uint64_t> input(points);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = index + 1;
    }
    auto expected = input;
    cuntt::reference_ntt(expected, modulus, false);

    cuntt::PlanConfig config;
    config.log_n = log_n;
    config.batch = batch;
    config.stage_partition = {2, 2};
    config.backend = cuntt::Backend::SharedIterative;
    config.threads_per_block = 32;
    config.auto_allocate_workspace = false;
    cuntt::Plan plan(config);
    cuntt::detail::StageProbeAdapter probe(plan);

    require(probe.groups().size() == 2, "shared probe did not expose two physical groups");
    for (const auto& group : probe.groups()) {
        require(group.independent, "shared probe group is not independently executable");
        require(group.execution.compiler_resources_known,
                "shared probe did not report compiler resources");
        require(group.execution.threads == config.threads_per_block,
                "shared probe thread count mismatch");
        require(group.execution.live_shared_bytes != 0,
                "shared probe shared-memory query is empty");
    }
    require(probe.groups()[0].execution.first_stage == 0 &&
                probe.groups()[0].execution.stage_count == 2 &&
                probe.groups()[1].execution.first_stage == 2 &&
                probe.groups()[1].execution.stage_count == 2,
            "shared probe stage intervals are not contiguous");

    DeviceBuffer device_input(input.size() * sizeof(std::uint64_t));
    DeviceBuffer device_intermediate(input.size() * sizeof(std::uint64_t));
    DeviceBuffer device_output(input.size() * sizeof(std::uint64_t));
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stage probe stream");
    try {
        check_cuda(cudaMemcpyAsync(device_input.data(), input.data(), input.size() * sizeof(std::uint64_t),
                                   cudaMemcpyHostToDevice, stream),
                   "copy stage probe input");
        probe.launch_group(0, device_input.data(), device_intermediate.data(), nullptr, stream);
        probe.launch_group(1, device_intermediate.data(), device_output.data(), nullptr, stream);
        std::vector<std::uint64_t> output(input.size());
        check_cuda(cudaMemcpyAsync(output.data(), device_output.data(), output.size() * sizeof(std::uint64_t),
                                   cudaMemcpyDeviceToHost, stream),
                   "copy stage probe output");
        check_cuda(cudaStreamSynchronize(stream), "synchronize stage probe stream");
        require(output == expected, "shared probe group execution mismatch");
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    check_cuda(cudaStreamDestroy(stream), "destroy stage probe stream");
}

void test_physical_launches(cuntt::Backend backend, bool inverse) {
    cuntt::PlanConfig config;
    config.log_n = backend==cuntt::Backend::Hybrid2D ? 12 : 8;
    config.backend = backend;
    config.n1_log = 6;
    config.stage_space = 2;
    config.inverse = inverse;
    config.batch = 2;
    cuntt::Plan plan(config);
    cuntt::detail::StageProbeAdapter probe(plan);
    const auto n=plan.points_per_transform();
    const auto bytes=plan.data_size();
    std::vector<std::uint64_t> host(n*config.batch),expected;
    for (std::size_t i=0;i<host.size();++i) host[i]=(i*31+7)%101;
    expected=host;
    for (std::size_t b=0;b<config.batch;++b) {
        std::vector<std::uint64_t> transform(host.begin()+b*n,host.begin()+(b+1)*n);
        cuntt::reference_ntt(transform,config.modulus,inverse);
        std::copy(transform.begin(),transform.end(),expected.begin()+b*n);
    }
    DeviceBuffer first(bytes),second(bytes),workspace(std::max<std::size_t>(1,plan.workspace_size()));
    void* source=first.data(); void* destination=second.data();
    check_cuda(cudaMemcpy(source,host.data(),bytes,cudaMemcpyHostToDevice),"copy physical probe input");
    for (std::size_t i=0;i<probe.groups().size();++i) {
        require(probe.groups()[i].independent,"physical launch not exposed");
        if (probe.groups()[i].in_place) probe.launch_group(i,source,source,workspace.data(),nullptr);
        else {
            probe.launch_group(i,source,destination,workspace.data(),nullptr);
            std::swap(source,destination);
        }
    }
    check_cuda(cudaMemcpy(host.data(),source,bytes,cudaMemcpyDeviceToHost),"read physical probe output");
    require(host==expected,"physical launch chain differs from reference");
    if (backend==cuntt::Backend::Hybrid2D) require(probe.groups().size()==2,"Hybrid2D must expose two physical launches");
}

void test_compact_stage_probe(cuntt::OutputOrder order) {
    cuntt::PlanConfig config;
    config.log_n = 20;
    config.batch = 4;
    config.modulus = 576460756061519873ULL;
    config.backend = cuntt::Backend::CompactStage;
    config.output_order = order;
    cuntt::Plan plan(config);
    cuntt::detail::StageProbeAdapter probe(plan);
    require(probe.groups().size() == 3, "compact-stage must expose three physical launches");
    const unsigned partition[] = {5, 6, 9};
    unsigned first_stage = 0;
    for (unsigned i = 0; i < 3; ++i) {
        const auto& group = probe.groups()[i];
        require(group.independent, "compact-stage launch is not independent");
        require(group.execution.first_stage == first_stage &&
                    group.execution.stage_count == partition[i], "compact-stage interval mismatch");
        require(group.execution.threads == 256 && group.execution.grid_ctas == 2048 * config.batch &&
                    group.execution.launch_count == 1, "compact-stage physical shape mismatch");
        require(group.execution.compiler_resources_known && group.execution.compiler_local_resources_known &&
                    group.execution.live_shared_bytes == 512 * sizeof(std::uint64_t),
                "compact-stage compiler resources missing");
        require(group.in_place == (i == 1 || (i == 2 && order == cuntt::OutputOrder::BitReversed)),
                "compact-stage alias contract mismatch");
        first_stage += partition[i];
    }
    const auto n = plan.points_per_transform();
    const auto bytes = plan.data_size();
    std::vector<std::uint64_t> input(n * config.batch), expected(input.size()), actual;
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = (i * 31 + 7) % 101;
    for (std::size_t b = 0; b < config.batch; ++b) {
        std::vector<std::uint64_t> reference(input.begin() + b * n, input.begin() + (b + 1) * n);
        cuntt::reference_ntt(reference, config.modulus, false);
        for (unsigned i = 0; i < n; ++i) {
            unsigned reversed = 0;
            for (unsigned bit = 0; bit < config.log_n; ++bit) reversed = (reversed << 1) | ((i >> bit) & 1U);
            expected[b * n + i] = reference[order == cuntt::OutputOrder::BitReversed ? reversed : i];
        }
    }
    plan.execute(input, actual, 0, 1);
    require(actual == expected, "compact-stage full plan differs from reference");
    DeviceBuffer first(bytes), second(bytes);
    void* source = first.data();
    void* destination = second.data();
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create compact-stage stream");
    try {
        check_cuda(cudaMemcpyAsync(source, input.data(), bytes, cudaMemcpyHostToDevice, stream), "copy compact-stage input");
        for (unsigned i = 0; i < 3; ++i) {
            if (probe.groups()[i].in_place) probe.launch_group(i, source, source, nullptr, stream);
            else {
                probe.launch_group(i, source, destination, nullptr, stream);
                std::swap(source, destination);
            }
        }
        check_cuda(cudaMemcpyAsync(actual.data(), source, bytes, cudaMemcpyDeviceToHost, stream), "read compact-stage output");
        check_cuda(cudaStreamSynchronize(stream), "synchronize compact-stage stream");
        require(actual == expected, "compact-stage physical launch chain differs from reference");
    } catch (...) {
        cudaStreamDestroy(stream);
        throw;
    }
    check_cuda(cudaStreamDestroy(stream), "destroy compact-stage stream");
}

template <typename Word>
void test_shared_output_order(unsigned log_n, const std::vector<std::uint32_t>& partition,
                              bool inverse, cuntt::OutputOrder order, bool overlap) {
    cuntt::PlanConfig config;
    config.log_n = log_n;
    config.batch = 5; // The last pipeline tile is partial.
    config.word_bits = sizeof(Word) * 8;
    config.modulus = sizeof(Word) == 4 ? 998244353ULL : 576460756061519873ULL;
    config.backend = cuntt::Backend::SharedIterative;
    config.stage_partition = partition;
    config.threads_per_block = 128;
    config.output_order = order;
    config.inverse = inverse;
    config.stage_overlap = overlap;
    config.batch_tile_count = 2;
    config.dataflow_layout = inverse ? cuntt::DataflowLayout::Linear : cuntt::DataflowLayout::HermesXor;
    cuntt::Plan plan(config);
    // Independent service probes intentionally use bulk plans. The pipeline
    // plan is checked separately below against the same reference.
    config.stage_overlap = false;
    cuntt::Plan bulk(config);
    cuntt::detail::StageProbeAdapter probe(bulk);
    const auto n = plan.points_per_transform();
    const auto bytes = plan.data_size();
    std::vector<std::uint64_t> input(n * config.batch), expected(input.size()), actual;
    std::mt19937_64 random(20260917 + log_n);
    for (auto& value : input) value = random() % config.modulus;
    for (std::size_t b = 0; b < config.batch; ++b) {
        std::vector<std::uint64_t> reference(input.begin() + b*n, input.begin() + (b+1)*n);
        cuntt::reference_ntt(reference, config.modulus, inverse);
        for (unsigned i = 0; i < n; ++i) {
            unsigned reversed = 0;
            for (unsigned bit = 0; bit < log_n; ++bit) reversed = (reversed << 1) | ((i >> bit) & 1U);
            expected[b*n+i] = reference[order == cuntt::OutputOrder::BitReversed ? reversed : i];
        }
    }
    plan.execute(input, actual, 0, 1);
    require(actual == expected, "shared full plan output order mismatch");
    bulk.execute(input, actual, 0, 1);
    require(actual == expected, "shared bulk output order mismatch");
    require(probe.groups().size() == partition.size(), "output order added a physical permutation launch");
    // Chain physical groups independently: this catches early permutation and
    // a group-entry path accidentally retaining natural-order terminal stores.
    std::vector<Word> packed(input.begin(), input.end()), result(input.size());
    DeviceBuffer first(bytes), second(bytes);
    void* source = first.data();
    void* destination = second.data();
    check_cuda(cudaMemcpy(source, packed.data(), bytes, cudaMemcpyHostToDevice), "copy order input");
    for (unsigned group = 0; group < partition.size(); ++group) {
        require(probe.groups()[group].execution.compiler_resources_known, "missing order kernel resources");
        probe.launch_group(group, source, destination, nullptr, nullptr);
        std::swap(source, destination);
    }
    check_cuda(cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost), "read ordered probe chain");
    require(std::equal(result.begin(), result.end(), expected.begin()), "shared group output order mismatch");
    std::cout << "PASS shared output order word" << config.word_bits << " logN=" << log_n
              << " groups=" << partition.size() << " inverse=" << inverse
              << " order=" << cuntt::output_order_name(order) << " overlap=" << overlap << '\n';
}

}  // namespace

int main() {
    int device_count = 0;
    const auto status = cudaGetDeviceCount(&device_count);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && device_count == 0)) {
        std::cout << "SKIP NTT stage probe tests: no CUDA device\n";
        return 0;
    }
    check_cuda(status, "query CUDA devices");

    try {
        test_output_order_inventory();
        test_shared_iterative_probe();
        for (const auto order : {cuntt::OutputOrder::Natural, cuntt::OutputOrder::BitReversed})
            test_compact_stage_probe(order);
        for (bool inverse : {false, true})
            for (const auto order : {cuntt::OutputOrder::Natural, cuntt::OutputOrder::BitReversed})
                for (const auto& partition : std::vector<std::vector<std::uint32_t>>{{6}, {3,3}, {2,2,2}, {5,7}}) {
                    const unsigned log_n = partition == std::vector<std::uint32_t>{5,7} ? 12 : 6;
                    const bool overlap = partition.size() > 1;
                    test_shared_output_order<std::uint32_t>(log_n, partition, inverse, order, overlap);
                    test_shared_output_order<std::uint64_t>(log_n, partition, inverse, order, overlap);
                }
        for (const auto backend : {cuntt::Backend::Baseline,cuntt::Backend::Tile256,
                                   cuntt::Backend::Hybrid2D,cuntt::Backend::StagePipeline})
            for (bool inverse : {false,true}) {
                // This existing lowering supports forward transforms only.
                if (inverse && backend==cuntt::Backend::StagePipeline) continue;
                test_physical_launches(backend,inverse);
            }
    } catch (const std::exception& error) {
        std::cerr << "FAIL NTT stage probe tests: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASS NTT stage probe tests\n";
    return 0;
}
