#include "cuntt/butterfly.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

class Stream {
  public:
    Stream() { check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags"); }
    ~Stream() { if (stream_) cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const noexcept { return stream_; }

  private:
    cudaStream_t stream_ = nullptr;
};

template <typename Value> class PinnedArray {
  public:
    explicit PinnedArray(std::size_t count) : count_(count) {
        if (count_ != 0)
            check_cuda(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(Value)), "cudaMallocHost");
    }
    ~PinnedArray() { if (data_) cudaFreeHost(data_); }
    PinnedArray(const PinnedArray&) = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;
    Value* data() noexcept { return data_; }
    const Value* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return count_; }

  private:
    Value* data_ = nullptr;
    std::size_t count_ = 0;
};

// The data pointer is surrounded by a byte canary.  The separate value
// sentinel used by a submission checks logical padding without relying on the
// allocator's red zones.
class GuardedDeviceBuffer {
  public:
    static constexpr std::size_t kGuardBytes = 128;

    GuardedDeviceBuffer() = default;
    explicit GuardedDeviceBuffer(std::size_t bytes) { allocate(bytes); }
    ~GuardedDeviceBuffer() { if (raw_) cudaFree(raw_); }
    GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
    GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;

    void allocate(std::size_t bytes) {
        if (raw_)
            throw std::logic_error("guarded device buffer allocated twice");
        bytes_ = bytes;
        if (bytes_ == 0)
            return;
        check_cuda(cudaMalloc(&raw_, bytes_ + 2 * kGuardBytes), "cudaMalloc");
        data_ = static_cast<unsigned char*>(raw_) + kGuardBytes;
    }

    void* data() const noexcept { return data_; }
    void* raw() const noexcept { return raw_; }
    std::size_t bytes() const noexcept { return bytes_; }
    std::size_t allocation_bytes() const noexcept { return bytes_ + 2 * kGuardBytes; }

    void mark_async(cudaStream_t stream, unsigned char value) const {
        if (raw_)
            check_cuda(cudaMemsetAsync(raw_, value, allocation_bytes(), stream), "cudaMemsetAsync");
    }

    void check_guards(const std::string& label, unsigned char expected) const {
        if (!raw_)
            return;
        std::array<unsigned char, kGuardBytes> left{}, right{};
        auto* right_address = static_cast<unsigned char*>(raw_) + kGuardBytes + bytes_;
        check_cuda(cudaMemcpy(left.data(), raw_, left.size(), cudaMemcpyDeviceToHost), "cudaMemcpy left guard");
        check_cuda(cudaMemcpy(right.data(), right_address, right.size(), cudaMemcpyDeviceToHost), "cudaMemcpy right guard");
        if (!std::all_of(left.begin(), left.end(), [expected](unsigned char value) { return value == expected; }) ||
            !std::all_of(right.begin(), right.end(), [expected](unsigned char value) { return value == expected; }))
            throw std::runtime_error(label + " canary changed");
    }

  private:
    void* raw_ = nullptr;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
};

template <typename Value> struct ValueTraits;
template <> struct ValueTraits<cuntt::Complex32> { using Real = float; };
template <> struct ValueTraits<cuntt::Complex64> { using Real = double; };

template <typename Value> Value sentinel_value() {
    using Real = typename ValueTraits<Value>::Real;
    return {Real(1234.0), Real(-4321.0)};
}

struct CaseSpec {
    const char* name = "";
    cuntt::FftCore core = cuntt::FftCore::CufftDxBlock;
    bool fp64 = false;
    std::uint32_t log_n = 0;
    std::vector<std::uint32_t> factors;
    std::uint32_t factor_ept = 1;
    std::uint32_t factor_columns = 1;
    std::uint32_t factor_slices = 1;
    std::uint32_t prefetch_depth = 0;
    std::size_t batch = 1;
    std::size_t element_stride = 1;
    std::size_t batch_stride = 0;
    bool inverse = false;
    bool in_place = false;
};

std::size_t transform_extent(const CaseSpec& spec) {
    const auto n = std::size_t{1} << spec.log_n;
    return (n - 1) * spec.element_stride + 1;
}

std::size_t data_elements(const CaseSpec& spec) {
    return (spec.batch - 1) * spec.batch_stride + transform_extent(spec);
}

template <typename Value> struct InvocationData {
    std::vector<Value> input;
    std::vector<Value> expected;
};

template <typename Value> std::vector<InvocationData<Value>> make_inputs(const CaseSpec& spec) {
    using Real = typename ValueTraits<Value>::Real;
    const auto n = std::size_t{1} << spec.log_n;
    const auto extent = data_elements(spec);
    const auto sentinel = sentinel_value<Value>();
    std::vector<InvocationData<Value>> result;
    result.reserve(3);
    for (unsigned invocation = 0; invocation < 3; ++invocation) {
        InvocationData<Value> data;
        data.input.assign(extent, sentinel);
        data.expected.assign(extent, sentinel);
        std::mt19937 generator(0x51ceU + spec.log_n * 17U + invocation * 131U);
        std::uniform_real_distribution<Real> distribution(Real(-1), Real(1));
        for (std::size_t batch = 0; batch < spec.batch; ++batch) {
            std::vector<Value> transform(n);
            for (std::size_t index = 0; index < n; ++index) {
                const auto address = batch * spec.batch_stride + index * spec.element_stride;
                data.input[address] = {distribution(generator), distribution(generator)};
                transform[index] = data.input[address];
            }
            cuntt::reference_fft(transform, spec.inverse, true);
            for (std::size_t index = 0; index < n; ++index)
                data.expected[batch * spec.batch_stride + index * spec.element_stride] = transform[index];
        }
        result.push_back(std::move(data));
    }
    return result;
}

template <typename Value> cuntt::ButterflyConfig make_config(const CaseSpec& spec, bool overlap) {
    cuntt::ButterflyConfig config;
    config.op = cuntt::ButterflyOperator::Fft;
    config.backend = cuntt::ButterflyBackend::FactorStreamed;
    config.fft_core = spec.core;
    config.precision = spec.fp64 ? cuntt::ButterflyPrecision::Fp64 : cuntt::ButterflyPrecision::Fp32;
    config.compute_unit = cuntt::ComputeUnit::Auto;
    config.log_n = spec.log_n;
    config.batch = spec.batch;
    config.stage_partition = {spec.log_n};
    config.factor_partition = spec.factors;
    config.factor_ept = spec.factor_ept;
    config.factor_columns = spec.factor_columns;
    config.factor_slices = spec.factor_slices;
    config.factor_overlap = overlap;
    config.data_tiles_per_cta = 5;
    config.prefetch_depth = spec.prefetch_depth;
    config.shared_layout = cuntt::SharedLayout::WriterAligned;
    config.cross_twiddle = cuntt::CrossTwiddleMode::Recurrence;
    config.direct_boundary = cuntt::DirectBoundary::Strided;
    config.local_exchange = cuntt::LocalExchange::SharedMemory;
    config.element_stride = spec.element_stride;
    config.batch_stride = spec.batch_stride;
    config.inverse = spec.inverse;
    config.normalize_inverse = true;
    config.placement = spec.in_place ? cuntt::ButterflyPlacement::InPlace : cuntt::ButterflyPlacement::OutOfPlace;
    config.auto_allocate_workspace = false;
    return config;
}

template <typename Value> class Submission {
  public:
    Submission(const InvocationData<Value>& data, Value sentinel, bool in_place)
        : count_(data.input.size()), bytes_(count_ * sizeof(Value)), in_place_(in_place), input_(bytes_),
          output_(in_place ? 0 : bytes_), upload_(count_), untouched_(count_), result_(count_) {
        std::copy(data.input.begin(), data.input.end(), upload_.data());
        std::fill(untouched_.data(), untouched_.data() + count_, sentinel);
        std::fill(result_.data(), result_.data() + count_, sentinel);
    }

    void enqueue(cuntt::ButterflyPlan& plan, cudaStream_t stream) {
        input_.mark_async(stream, 0xa5);
        check_cuda(cudaMemcpyAsync(input_.data(), upload_.data(), bytes_, cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync input");
        auto* destination = static_cast<Value*>(input_.data());
        if (!in_place_) {
            output_.mark_async(stream, 0xa5);
            check_cuda(cudaMemcpyAsync(output_.data(), untouched_.data(), bytes_, cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync output initialization");
            destination = static_cast<Value*>(output_.data());
        }
        plan.set_stream(stream);
        plan.execute_async(static_cast<const Value*>(input_.data()), destination);
        check_cuda(cudaMemcpyAsync(result_.data(), destination, bytes_, cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync result");
    }

    void check_guards(unsigned index) const {
        input_.check_guards("submission " + std::to_string(index) + " input", 0xa5);
        if (!in_place_)
            output_.check_guards("submission " + std::to_string(index) + " output", 0xa5);
    }

    const Value* result() const noexcept { return result_.data(); }
    std::size_t count() const noexcept { return count_; }

  private:
    std::size_t count_ = 0;
    std::size_t bytes_ = 0;
    bool in_place_ = false;
    GuardedDeviceBuffer input_;
    GuardedDeviceBuffer output_;
    PinnedArray<Value> upload_;
    PinnedArray<Value> untouched_;
    PinnedArray<Value> result_;
};

template <typename Value>
std::vector<std::vector<Value>> execute_mode(const CaseSpec& spec, bool overlap,
                                             const std::vector<InvocationData<Value>>& inputs) {
    const auto sentinel = sentinel_value<Value>();
    std::vector<std::unique_ptr<Submission<Value>>> submissions;
    submissions.reserve(inputs.size());
    for (const auto& input : inputs)
        submissions.push_back(std::make_unique<Submission<Value>>(input, sentinel, spec.in_place));

    GuardedDeviceBuffer workspace;
    Stream caller0;
    Stream caller1;
    std::vector<std::vector<Value>> results(inputs.size(), std::vector<Value>(data_elements(spec)));
    {
        auto config = make_config<Value>(spec, overlap);
        cuntt::ButterflyPlan plan(config);
        if (plan.config().factor_slices != spec.factor_slices || plan.config().factor_overlap != overlap ||
            plan.dataflow_plan().execution_groups.size() != spec.factors.size())
            throw std::runtime_error("factor pipeline plan lost its slice or execution-group contract");
        const auto expected_workspace = (spec.factors.size() - 1) * plan.data_size();
        if (plan.workspace_size() != expected_workspace)
            throw std::runtime_error("factor pipeline workspace is not one full buffer per live edge");
        workspace.allocate(plan.workspace_size());
        check_cuda(cudaMemset(workspace.raw(), 0x5a, workspace.allocation_bytes()), "cudaMemset workspace");
        // The setup memset is issued on the legacy default stream.  Make its
        // completion explicit before nonblocking caller streams can consume
        // this externally owned scratch allocation.
        check_cuda(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize setup");
        plan.set_workspace(workspace.data(), plan.workspace_size());
        if (plan.workspace() != workspace.data())
            throw std::runtime_error("external factor workspace was not installed");

        const std::array<cudaStream_t, 2> callers{caller0.get(), caller1.get()};
        for (std::size_t index = 0; index < submissions.size(); ++index) {
            if (!overlap && index != 0)
                // The serial-slice control has no FactorPipeline event fence
                // between different caller streams.  Drain the previous
                // caller before reusing its shared external workspace.
                check_cuda(cudaStreamSynchronize(callers[(index - 1) % callers.size()]),
                           "cudaStreamSynchronize serial caller");
            submissions[index]->enqueue(plan, callers[index % callers.size()]);
        }
        for (auto caller : callers)
            check_cuda(cudaStreamSynchronize(caller), "cudaStreamSynchronize caller");
        for (std::size_t index = 0; index < submissions.size(); ++index) {
            submissions[index]->check_guards(static_cast<unsigned>(index));
            std::copy(submissions[index]->result(), submissions[index]->result() + submissions[index]->count(),
                      results[index].begin());
        }
    }
    // The plan's destructor drains its private factor streams before this
    // external allocation is released; checking afterward covers late events.
    workspace.check_guards("factor pipeline workspace", 0x5a);
    return results;
}

template <typename Value>
void check_results(const CaseSpec& spec, const std::vector<InvocationData<Value>>& inputs,
                   const std::vector<std::vector<Value>>& results, const std::string& label) {
    const auto n = std::size_t{1} << spec.log_n;
    const auto extent = transform_extent(spec);
    const auto sentinel = sentinel_value<Value>();
    const double tolerance = spec.fp64 ? 3.0e-10 : 8.0e-4 * std::sqrt(std::max(1.0, double(n) / 1024.0));
    double max_error = 0.0;
    double max_reference = 0.0;
    for (std::size_t invocation = 0; invocation < inputs.size(); ++invocation) {
        if (results[invocation].size() != inputs[invocation].expected.size())
            throw std::runtime_error(label + " result extent mismatch");
        for (std::size_t address = 0; address < results[invocation].size(); ++address) {
            const auto local = address % spec.batch_stride;
            const bool valid = local < extent && local % spec.element_stride == 0;
            const auto& actual = results[invocation][address];
            if (!valid) {
                if (actual.real != sentinel.real || actual.imag != sentinel.imag)
                    throw std::runtime_error(label + " overwrote stride or batch padding");
                continue;
            }
            const auto& expected = inputs[invocation].expected[address];
            const double error = std::hypot(double(actual.real) - double(expected.real),
                                            double(actual.imag) - double(expected.imag));
            if (!std::isfinite(error))
                throw std::runtime_error(label + " produced a non-finite value");
            max_error = std::max(max_error, error);
            max_reference = std::max(max_reference, std::hypot(double(expected.real), double(expected.imag)));
        }
    }
    if (max_error > tolerance * std::max(1.0, max_reference)) {
        std::ostringstream message;
        message << label << " disagrees with CPU reference: max_error=" << max_error
                << " max_reference=" << max_reference << " tolerance=" << tolerance;
        throw std::runtime_error(message.str());
    }
}

template <typename Value>
void check_scheduler_agreement(const CaseSpec& spec, const std::vector<InvocationData<Value>>& inputs,
                               const std::vector<std::vector<Value>>& serial,
                               const std::vector<std::vector<Value>>& overlap) {
    const auto extent = transform_extent(spec);
    const auto sentinel = sentinel_value<Value>();
    const auto n = std::size_t{1} << spec.log_n;
    const double tolerance = (spec.fp64 ? 6.0e-10 : 1.6e-3 * std::sqrt(std::max(1.0, double(n) / 1024.0)));
    for (std::size_t invocation = 0; invocation < inputs.size(); ++invocation) {
        for (std::size_t address = 0; address < serial[invocation].size(); ++address) {
            const auto local = address % spec.batch_stride;
            const bool valid = local < extent && local % spec.element_stride == 0;
            if (!valid) {
                if (serial[invocation][address].real != sentinel.real || serial[invocation][address].imag != sentinel.imag ||
                    overlap[invocation][address].real != sentinel.real || overlap[invocation][address].imag != sentinel.imag)
                    throw std::runtime_error("serial and overlapping factor schedulers changed padding differently");
                continue;
            }
            const double error = std::hypot(double(serial[invocation][address].real) - double(overlap[invocation][address].real),
                                            double(serial[invocation][address].imag) - double(overlap[invocation][address].imag));
            if (!std::isfinite(error) || error > tolerance)
                throw std::runtime_error("serial and overlapping factor schedulers disagree");
        }
    }
}

template <typename Value> void run_case(const CaseSpec& spec) {
    const auto inputs = make_inputs<Value>(spec);
    const auto serial = execute_mode(spec, false, inputs);
    check_results(spec, inputs, serial, std::string(spec.name) + " serial");
    const auto overlap = execute_mode(spec, true, inputs);
    check_results(spec, inputs, overlap, std::string(spec.name) + " overlap");
    check_scheduler_agreement(spec, inputs, serial, overlap);
    std::cout << "PASS factor pipeline " << spec.name << " core=" << cuntt::fft_core_name(spec.core)
              << " slices=" << spec.factor_slices << " groups=" << spec.factors.size() << '\n';
}

std::vector<CaseSpec> cases() {
    // All shapes use five CTA tiles so at least one launch has a non-full tail.
    return {
        {"native-fp64-two-factor-inverse-stride2", cuntt::FftCore::RegisterTile, true, 8,
         {4, 4}, 4, 8, 2, 1, 3, 2, (std::size_t{1} << 8) * 2 + 7, true, true},
        {"cufftdx-fp64-four-factor-unequal", cuntt::FftCore::CufftDxBlock, true, 19,
         {4, 5, 6, 4}, 4, 8, 2, 2, 1, 1, (std::size_t{1} << 19) + 13, true, false},
        {"cufftdx-fp32-three-factor-unequal-tail", cuntt::FftCore::CufftDxBlock, false, 12,
         {3, 4, 5}, 2, 8, 4, 1, 3, 1, (std::size_t{1} << 12) + 11, false, false},
        {"native-fp32-four-factor-tail", cuntt::FftCore::RegisterTile, false, 16,
         {4, 4, 4, 4}, 4, 8, 2, 2, 3, 1, (std::size_t{1} << 16) + 9, false, false},
    };
}

}  // namespace

int main(int argc, char** argv) {
#ifndef CUBUTTERFLY_TEST_CUFFTDX
    std::cout << "SKIP factor pipeline tests: cuFFTDx support is not enabled\n";
    return 0;
#else
    try {
        std::string selected_case;
        for (int index = 1; index < argc; ++index) {
            if (std::string(argv[index]) != "--case" || index + 1 >= argc)
                throw std::invalid_argument("usage: test_factor_pipeline [--case CASE_NAME]");
            selected_case = argv[++index];
        }
        const auto* policy = std::getenv("CUBUTTERFLY_COMPILE_MODE");
        if (!policy || std::string(policy) != "research")
            throw std::runtime_error("factor pipeline tests require CUBUTTERFLY_COMPILE_MODE=research");
        bool selected = selected_case.empty();
        for (const auto& spec : cases()) {
            if (!selected_case.empty() && selected_case != spec.name)
                continue;
            selected = true;
            if (spec.fp64)
                run_case<cuntt::Complex64>(spec);
            else
                run_case<cuntt::Complex32>(spec);
        }
        if (!selected)
            throw std::invalid_argument("unknown factor pipeline case: " + selected_case);
        std::cout << "PASS factor pipeline scheduler coverage\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
