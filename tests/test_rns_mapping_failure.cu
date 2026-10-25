#include <cuntt/rns_ntt.hpp>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Word = std::uint64_t;
using Wide = unsigned __int128;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Device {
    Word* data = nullptr;
    explicit Device(std::size_t count) {
        check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(Word)));
    }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

Word multiply(Word a, Word b, Word q) { return Wide(a) * b % q; }

Word power(Word a, Word exponent, Word q) {
    Word value = 1;
    for (; exponent; exponent >>= 1, a = multiply(a, a, q))
        if (exponent & 1) value = multiply(value, a, q);
    return value;
}

unsigned reverse_bits(unsigned value, unsigned bits) {
    unsigned result = 0;
    for (unsigned bit = 0; bit < bits; ++bit, value >>= 1)
        result = (result << 1) | (value & 1);
    return result;
}

// Independent forward negacyclic DIT reference, with native bit-reversed output.
std::vector<Word> reference(std::vector<Word> values, Word root, Word q, unsigned log_n) {
    const unsigned n = 1U << log_n;
    Word twist = 1;
    for (auto& value : values) {
        value = multiply(value, twist, q);
        twist = multiply(twist, root, q);
    }
    for (unsigned index = 0; index < n; ++index) {
        const unsigned reversed = reverse_bits(index, log_n);
        if (index < reversed) std::swap(values[index], values[reversed]);
    }
    const Word omega = multiply(root, root, q);
    for (unsigned length = 2; length <= n; length <<= 1) {
        const Word step = power(omega, n / length, q);
        for (unsigned base = 0; base < n; base += length) {
            Word twiddle = 1;
            for (unsigned index = 0; index < length / 2; ++index) {
                const Word a = values[base + index];
                const Word b = multiply(values[base + index + length / 2], twiddle, q);
                values[base + index] = (Wide(a) + b) % q;
                values[base + index + length / 2] = (Wide(a) + q - b) % q;
                twiddle = multiply(twiddle, step, q);
            }
        }
    }
    std::vector<Word> output(n);
    for (unsigned index = 0; index < n; ++index)
        output[reverse_bits(index, log_n)] = values[index];
    return output;
}

bool same_mapping(const cuntt::RnsMapping& a, const cuntt::RnsMapping& b) {
    return a.n1_log == b.n1_log && a.rows_per_block == b.rows_per_block &&
           a.threads_per_block == b.threads_per_block && a.compute_unit == b.compute_unit &&
           a.cross_twiddle == b.cross_twiddle &&
           a.physical_boundary_traversal == b.physical_boundary_traversal &&
           a.core_family == b.core_family && a.second_rows_per_block == b.second_rows_per_block &&
           a.second_threads_per_block == b.second_threads_per_block &&
           a.arithmetic == b.arithmetic && a.exchange == b.exchange;
}

bool same_selection(const cuntt::SelectionInfo& a, const cuntt::SelectionInfo& b) {
    return a.automatic == b.automatic && a.calibrated == b.calibrated &&
           a.target == b.target && a.implementation == b.implementation &&
           a.confidence == b.confidence && a.reason == b.reason &&
           a.predicted_kernel_ms == b.predicted_kernel_ms;
}

void execute_and_compare(cuntt::RnsPlan& plan, const Device& arena,
                         const std::vector<Word>& input, const std::vector<Word>& expected,
                         const char* label) {
    check(cudaMemcpy(arena.data, input.data(), input.size() * sizeof(Word), cudaMemcpyHostToDevice));
    plan.execute_async(arena.data, arena.data);
    std::vector<Word> actual(input.size());
    check(cudaMemcpy(actual.data(), arena.data, actual.size() * sizeof(Word), cudaMemcpyDeviceToHost));
    for (std::size_t index = 0; index < actual.size(); ++index)
        if (actual[index] != expected[index])
            throw std::runtime_error(std::string(label) + ": mismatch at coefficient " +
                                     std::to_string(index));
}
} // namespace

int main() try {
    using Arm = void (*)();
    using FailureCount = unsigned (*)();
    const auto arm = reinterpret_cast<Arm>(dlsym(RTLD_DEFAULT, "arm_next_cuda_malloc_failure"));
    const auto failure_count = reinterpret_cast<FailureCount>(
        dlsym(RTLD_DEFAULT, "rns_injected_cuda_malloc_failures"));
    require(arm && failure_count, "load the cudaMalloc failure shim with LD_PRELOAD");

    constexpr unsigned log_n = 12;
    constexpr unsigned n = 1U << log_n;
    constexpr Word q = 1152921504577486849ULL;
    const Word root = cuntt::find_primitive_power_of_two_root(log_n + 1, q);
    cuntt::RnsPlanConfig config;
    config.log_n = log_n;
    config.domains.push_back({q, root});
    config.polynomials.push_back({0, 0, 0});
    cuntt::RnsPlan plan(config);
    cuntt::RnsMapping fused;
    fused.n1_log = 6;
    fused.rows_per_block = 8;
    fused.threads_per_block = 128;
    fused.second_rows_per_block = 1;
    fused.second_threads_per_block = 64;
    fused.core_family = cuntt::RnsCoreFamily::FusedCTGS;
    fused.exchange = cuntt::RnsExchange::DependencyScoped;
    plan.set_mapping(fused, cuntt::RnsPlacement::InPlace);
    require(plan.workspace_bytes() == 0, "native fused plan retained private scratch");

    Device arena(n);
    std::vector<Word> input(n);
    Word state = 0x7682a5b109ULL;
    for (auto& value : input) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        value = state % q;
    }
    input[0] = 0;
    input[1] = q - 1;
    input[2] = q / 2;
    const auto expected = reference(input, root, q, log_n);
    execute_and_compare(plan, arena, input, expected, "initial fused mapping");

    const auto previous_mapping = plan.mapping();
    const auto previous_selection = plan.selection();
    const auto previous_workspace = plan.workspace_bytes();
    const auto previous_measurements = plan.measurements().size();
    const unsigned previous_failures = failure_count();
    cuntt::RnsMapping hybrid;
    hybrid.n1_log = 6;
    bool threw = false;
    arm();
    try {
        plan.set_mapping(hybrid, cuntt::RnsPlacement::InPlace);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "set_mapping did not propagate the injected allocation failure");
    require(failure_count() == previous_failures + 1, "failure injection was not consumed exactly once");
    require(same_mapping(plan.mapping(), previous_mapping), "failed set_mapping changed mapping");
    require(same_selection(plan.selection(), previous_selection), "failed set_mapping changed selection");
    require(plan.workspace_bytes() == previous_workspace, "failed set_mapping changed workspace");
    require(plan.measurements().size() == previous_measurements, "failed set_mapping changed measurements");
    // Private masks have no public accessor; executing the preserved mapping
    // checks the restored state through its public behavior.
    execute_and_compare(plan, arena, input, expected, "fused mapping after allocation failure");

    plan.set_mapping(hybrid, cuntt::RnsPlacement::InPlace);
    require(same_mapping(plan.mapping(), hybrid), "retry did not install the hybrid mapping");
    require(plan.workspace_bytes() == n * sizeof(Word), "retry did not allocate hybrid scratch");
    require(failure_count() == previous_failures + 1, "one-shot injection failed another allocation");
    execute_and_compare(plan, arena, input, expected, "hybrid mapping after retry");
    std::cout << "RNS mapping allocation failure recovery passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
