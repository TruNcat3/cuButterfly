#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuntt/ntt.hpp"

namespace cuntt {

struct alignas(8) Complex32 {
    float real;
    float imag;
};

struct alignas(16) Complex64 {
    double real;
    double imag;
};

// One real-valued local butterfly: [left', right']^T = M [left, right]^T.
struct ButterflyMatrix2x2 {
    double m00 = 1.0;
    double m01 = 0.0;
    double m10 = 0.0;
    double m11 = 1.0;
};

enum class ButterflyPrecision {
    Fp32,
    Fp64,
    Fp16Fp32,
    Uint32,
};

const char*        butterfly_precision_name(ButterflyPrecision precision) noexcept;
ButterflyPrecision parse_butterfly_precision(const std::string& name);

enum class ButterflyPlacement {
    OutOfPlace,
    InPlace,
};

const char*        butterfly_placement_name(ButterflyPlacement placement) noexcept;
ButterflyPlacement parse_butterfly_placement(const std::string& name);

enum class ButterflyOperator {
    Fwht,
    Fft,
    SubsetZeta,
    SupersetZeta,
    Structured2x2,
    // Legacy name retained for source and CLI compatibility. Its established
    // operation is the subset zeta/Mobius transform, not an XOR convolution.
    XorZeta,
};

const char*       butterfly_operator_name(ButterflyOperator op) noexcept;
ButterflyOperator parse_butterfly_operator(const std::string& name);

enum class ButterflyBackend {
    TemporalTile,
    Hierarchical,
    OnlineReorder,
    WarpHybrid,
    StagePipeline,
    CuFft,
};

const char*      butterfly_backend_name(ButterflyBackend backend) noexcept;
ButterflyBackend parse_butterfly_backend(const std::string& name);

struct ButterflyCapability {
    ButterflyBackend backend;
    std::uint32_t    min_log_n;
    std::uint32_t    max_log_n;
    bool             radix2;
    bool             radix4;
    bool             radix8;
    bool             warp_register;
    bool             fft_thread_dft8;
    bool             fft_cta_dft8;
    bool             fft_wmma_dft8;
    bool             fft_four_mul;
    bool             fft_gauss3;
    bool             fp32;
    bool             fp64;
    bool             fp16_fp32;
    bool             uint32;
    bool             in_place;
    bool             strided_layout;
    const char*      constraint;
};

enum class ComplexMultiply {
    FourMul,
    Gauss3,
};

const char*     complex_multiply_name(ComplexMultiply multiply) noexcept;
ComplexMultiply parse_complex_multiply(const std::string& name);

enum class CrossTwiddleMode {
    Table,
    Recurrence,
};

const char*      cross_twiddle_mode_name(CrossTwiddleMode mode) noexcept;
CrossTwiddleMode parse_cross_twiddle_mode(const std::string& name);

enum class DirectBoundary {
    Strided,
    TiledTranspose,
    PrefixTiledTranspose,
};

const char*    direct_boundary_name(DirectBoundary boundary) noexcept;
DirectBoundary parse_direct_boundary(const std::string& name);

enum class FftBoundaryResidency {
    GlobalScratch,
    Fused,
};

const char*          fft_boundary_residency_name(FftBoundaryResidency residency) noexcept;
FftBoundaryResidency parse_fft_boundary_residency(const std::string& name);

enum class LocalExchange {
    SharedMemory,
    WarpRegister,
};

const char*   local_exchange_name(LocalExchange exchange) noexcept;
LocalExchange parse_local_exchange(const std::string& name);

enum class SharedLayout {
    Linear,
    XorSwizzle,
};

const char*  shared_layout_name(SharedLayout layout) noexcept;
SharedLayout parse_shared_layout(const std::string& name);

enum class FftCore {
    Scalar,
    ThreadDft8,
    CtaDft8,
    WmmaDft8,
    CufftDxBlock,
    CufftDxDirect,
    CufftDxResident,
    TurboFftGenerated,
};

const char* fft_core_name(FftCore core) noexcept;
FftCore     parse_fft_core(const std::string& name);

struct FftSegmentMapping {
    FftCore         core     = FftCore::CufftDxBlock;
    LocalExchange   exchange = LocalExchange::SharedMemory;
    std::uint32_t   threads  = 0;
    std::uint32_t   ept      = 8;
};

struct FftBoundaryMapping {
    CrossTwiddleMode cross_twiddle = CrossTwiddleMode::Table;
    DirectBoundary   layout        = DirectBoundary::Strided;
    FftBoundaryResidency residency = FftBoundaryResidency::GlobalScratch;
};

std::vector<ButterflyCapability> butterfly_capabilities();

struct ButterflyConfig {
    ButterflyOperator  op                = ButterflyOperator::Fwht;
    ButterflyBackend   backend           = ButterflyBackend::StagePipeline;
    ButterflyPrecision precision         = ButterflyPrecision::Fp32;
    ButterflyPlacement placement         = ButterflyPlacement::OutOfPlace;
    std::uint32_t      log_n             = 8;
    // Structured2x2 accepts one broadcast matrix or one matrix per stage.
    std::vector<ButterflyMatrix2x2> stage_matrices;
    // Ordered stage counts for each algorithmic decomposition segment.
    std::vector<std::uint32_t> stage_partition;
    std::vector<FftSegmentMapping> segment_mappings;
    std::vector<FftBoundaryMapping> boundaries;
    // Physical processing groups after fused logical boundaries are lowered.
    std::vector<FftSegmentMapping> execution_group_mappings;
    std::size_t        batch             = 1;
    std::size_t        batch_stride      = 0;
    std::size_t        element_stride    = 1;
    bool               inverse           = false;
    bool               normalize_inverse = true;
    bool               auto_select       = false;
    // Allocate plan-owned scratch by default. Disable this when an application
    // supplies workspace with set_workspace().
    bool               auto_allocate_workspace = true;
    std::uint32_t      stage_space       = 0;
    StageHandoff       stage_handoff     = StageHandoff::NamedBarrier;
    std::uint32_t      tile_threads      = 128;
    std::uint32_t      prefix_threads    = 0;
    std::uint32_t      suffix_threads    = 0;
    std::uint32_t      prefix_ept        = 8;
    std::uint32_t      suffix_ept        = 8;
    std::uint32_t      prefix_units_per_cta = 0;
    std::uint32_t      suffix_units_per_cta = 0;
    std::uint32_t      local_stages      = 10;
    std::uint32_t      reorder_columns   = 1;
    std::uint32_t      warp_stages       = 5;
    std::uint32_t      pipeline_warps    = 8;
    ComputeUnit        compute_unit      = ComputeUnit::Radix2;
    ComplexMultiply    complex_multiply  = ComplexMultiply::FourMul;
    CrossTwiddleMode   cross_twiddle     = CrossTwiddleMode::Table;
    DirectBoundary     direct_boundary   = DirectBoundary::Strided;
    LocalExchange      local_exchange    = LocalExchange::SharedMemory;
    SharedLayout       shared_layout     = SharedLayout::Linear;
    FftCore            fft_core          = FftCore::Scalar;
};

struct ButterflyStats {
    double h2d_ms                 = 0.0;
    double kernel_ms              = 0.0;
    double d2h_ms                 = 0.0;
    double transforms_per_second  = 0.0;
    double butterflies_per_second = 0.0;
    double points_per_second      = 0.0;
};

void reference_fwht(std::vector<float>& values, bool inverse = false, bool normalize_inverse = true);
void reference_fwht(std::vector<double>& values, bool inverse = false, bool normalize_inverse = true);
void reference_fft(std::vector<Complex32>& values, bool inverse = false, bool normalize_inverse = true);
void reference_fft(std::vector<Complex64>& values, bool inverse = false, bool normalize_inverse = true);
void reference_subset_zeta(std::vector<std::uint32_t>& values, bool inverse = false);
void reference_superset_zeta(std::vector<std::uint32_t>& values, bool inverse = false);
void reference_structured_2x2(std::vector<float>& values, const std::vector<ButterflyMatrix2x2>& stage_matrices,
                              bool inverse = false);
void reference_structured_2x2(std::vector<double>& values, const std::vector<ButterflyMatrix2x2>& stage_matrices,
                              bool inverse = false);
// Compatibility wrapper for reference_subset_zeta().
void reference_xor_zeta(std::vector<std::uint32_t>& values, bool inverse = false);

class ButterflyPlan {
  public:
    explicit ButterflyPlan(ButterflyConfig config);
    ~ButterflyPlan();

    ButterflyPlan(const ButterflyPlan&)            = delete;
    ButterflyPlan& operator=(const ButterflyPlan&) = delete;
    ButterflyPlan(ButterflyPlan&&) noexcept;
    ButterflyPlan& operator=(ButterflyPlan&&) noexcept;

    const ButterflyConfig& config() const noexcept;
    const SelectionInfo&   selection() const noexcept;

    std::size_t data_size() const noexcept;
    std::size_t workspace_size() const noexcept;

    // A plan is ordered on one CUDA stream. Changing the stream also updates
    // an internal cuFFT plan when that backend is selected.
    void         set_stream(cudaStream_t stream);
    cudaStream_t stream() const noexcept;

    // The workspace must remain valid until all work submitted by this plan
    // completes. Passing nullptr restores plan-owned workspace when enabled.
    void  set_workspace(void* workspace, std::size_t bytes);
    void* workspace() const noexcept;

    // Device-pointer execution is allocation-free, asynchronous, and does not
    // perform host/device copies or stream synchronization.
    void execute_async(const float* input, float* output);
    void execute_async(const double* input, double* output);
    void execute_async(const Complex32* input, Complex32* output);
    void execute_async(const Complex64* input, Complex64* output);
    void execute_async(const std::uint32_t* input, std::uint32_t* output);

    ButterflyStats execute(const std::vector<float>& input, std::vector<float>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);
    ButterflyStats execute(const std::vector<double>& input, std::vector<double>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);
    ButterflyStats execute(const std::vector<Complex32>& input, std::vector<Complex32>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);
    ButterflyStats execute(const std::vector<Complex64>& input, std::vector<Complex64>& output, std::uint32_t warmup = 1, std::uint32_t repeat = 1);
    ButterflyStats execute(const std::vector<std::uint32_t>& input, std::vector<std::uint32_t>& output, std::uint32_t warmup = 1,
                           std::uint32_t repeat = 1);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt
