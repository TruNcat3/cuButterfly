#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuntt {

constexpr std::uint64_t kDefaultModulus = 1152921504606584833ULL;

enum class Backend {
    Baseline,
    Tile256,
    Hybrid2D,
    CompactStage,
    StagePipeline,
    Merge256 = Hybrid2D,
};

const char* backend_name(Backend backend) noexcept;
Backend     parse_backend(const std::string& name);

enum class StageHandoff {
    Atomic,
    NamedBarrier,
};

const char*  stage_handoff_name(StageHandoff handoff) noexcept;
StageHandoff parse_stage_handoff(const std::string& name);

enum class OutputOrder {
    Natural,
    BitReversed,
};

const char* output_order_name(OutputOrder order) noexcept;
OutputOrder parse_output_order(const std::string& name);

enum class ComputeUnit {
    Auto,
    Radix2,
    Radix4,
    Radix8,
};

const char* compute_unit_name(ComputeUnit unit) noexcept;
ComputeUnit parse_compute_unit(const std::string& name);

enum class CrossTwiddlePlacement {
    FirstPass,
    SecondPass,
    Fused,
    FusedBarrett,
};

const char*           cross_twiddle_placement_name(CrossTwiddlePlacement placement) noexcept;
CrossTwiddlePlacement parse_cross_twiddle_placement(const std::string& name);

enum class ModularMultiply {
    Shoup,
    Barrett,
};

const char*     modular_multiply_name(ModularMultiply multiply) noexcept;
ModularMultiply parse_modular_multiply(const std::string& name);

struct PlanConfig {
    std::uint32_t log_n   = 16;
    std::size_t   batch   = 1;
    std::uint64_t modulus = kDefaultModulus;
    bool          inverse = false;
    Backend       backend = Backend::Tile256;
    // Explicit stage-space unfolding for StagePipeline. Zero selects the default.
    std::uint32_t stage_space   = 0;
    StageHandoff  stage_handoff = StageHandoff::NamedBarrier;
    // Hybrid2D mapping. Zero selects a device-specific default.
    std::uint32_t         n1_log                  = 0;
    std::uint32_t         rows_per_block          = 0;
    std::uint32_t         threads_per_block       = 0;
    ComputeUnit           compute_unit            = ComputeUnit::Auto;
    std::uint32_t         word_bits               = 64;
    CrossTwiddlePlacement cross_twiddle_placement = CrossTwiddlePlacement::Fused;
    ModularMultiply       modular_multiply        = ModularMultiply::Shoup;
    OutputOrder           output_order            = OutputOrder::Natural;
};

struct RunStats {
    double h2d_ms                    = 0.0;
    double kernel_ms                 = 0.0;
    double d2h_ms                    = 0.0;
    double kernel_ntt_per_second     = 0.0;
    double end_to_end_ntt_per_second = 0.0;
    double kernel_points_per_second  = 0.0;
};

struct DeviceInfo {
    int         device_id = 0;
    std::string name;
    int         compute_major       = 0;
    int         compute_minor       = 0;
    std::size_t global_memory_bytes = 0;
    int         multiprocessors     = 0;
};

DeviceInfo current_device_info();

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus);
bool          is_prime(std::uint64_t value);
std::uint64_t find_primitive_power_of_two_root(std::uint32_t log_n, std::uint64_t modulus);
void          reference_ntt(std::vector<std::uint64_t>& values, std::uint64_t modulus, bool inverse = false);

class Plan {
  public:
    explicit Plan(PlanConfig config);
    ~Plan();

    Plan(const Plan&)            = delete;
    Plan& operator=(const Plan&) = delete;
    Plan(Plan&&) noexcept;
    Plan& operator=(Plan&&) noexcept;

    const PlanConfig& config() const noexcept;
    std::size_t       points_per_transform() const noexcept;

    RunStats execute(const std::vector<std::uint64_t>& input, std::vector<std::uint64_t>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt
