#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "cuntt/butterfly.hpp"
#include "general_kernels.hpp"

namespace cuntt {

// Operators supported by the small multidimensional resident lowering.  The
// transform is packed and every axis is a power of two; the enclosing plan
// layer is responsible for selecting this lowering only for eligible shapes.
enum class XdOperator {
    Fft,
    Fwht,
    SubsetZeta,
    SupersetZeta,
    Structured2x2,
    Ntt,
};

// Storage is also the pointer contract passed to execute().  FFT uses the
// two complex storage variants; the other operators use real/integer values.
enum class XdStorageType {
    Fp16,
    Bf16,
    Fp32,
    Fp64,
    ComplexFp32,
    ComplexFp64,
    Uint32,
    Uint64,
};

// Low-precision FWHT requires explicit FP32 shared accumulation here; native
// low-precision rounding remains on the existing lowering. FP32/FP64 and
// Uint32/Uint64 arithmetic match their respective storage types.
enum class XdComputeType {
    Native,
    Fp32,
    Fp64,
    Uint32,
    Uint64,
};

using XdResidentType = XdStorageType;
using XdMatrix2x2 = ButterflyMatrix2x2;

struct XdResidentConfig {
    XdOperator     op                = XdOperator::Fwht;
    XdStorageType  storage           = XdStorageType::Fp32;
    XdComputeType  compute           = XdComputeType::Native;
    std::vector<std::size_t> extents;
    std::size_t    batch             = 1;
    bool           inverse           = false;
    bool           normalize_inverse = true;
    std::uint64_t  modulus           = 0;
    // For a flat per-stage list, entries are ordered from the innermost
    // (last packed extent) axis to the outermost axis, with stage zero first
    // within each axis.  A one-entry list broadcasts one matrix everywhere.
    std::vector<ButterflyMatrix2x2> stage_matrices;
    // A CTA processes one complete transform. Any warp multiple within the
    // device block limit is legal; the automatic inventory is finite.
    std::uint32_t  threads           = 128;
    // Actual execution order of independent axes. Empty preserves the legacy
    // forward/inverse traversal. Per-axis arithmetic and coefficients retain
    // their original semantic axis identity.
    std::vector<std::size_t> axis_order;
};

struct XdResidentAvailability {
    bool        available = false;
    std::string reason;
};

class XdResidentPlan {
  public:
    explicit XdResidentPlan(XdResidentConfig config);
    ~XdResidentPlan();

    XdResidentPlan(const XdResidentPlan&) = delete;
    XdResidentPlan& operator=(const XdResidentPlan&) = delete;
    XdResidentPlan(XdResidentPlan&&) noexcept;
    XdResidentPlan& operator=(XdResidentPlan&&) noexcept;

    // This check is side-effect free apart from querying the current CUDA
    // device.  It is useful to let a composite lowering reject a shape before
    // constructing a plan and allocating coefficient tables.
    static XdResidentAvailability check(const XdResidentConfig& config);
    static bool supported(const XdResidentConfig& config, std::string* reason = nullptr);

    bool available() const noexcept;
    bool is_available() const noexcept { return available(); }
    const std::string& availability_reason() const noexcept;
    const std::string& reason() const noexcept { return availability_reason(); }
    const XdResidentConfig& config() const noexcept;

    std::size_t points_per_transform() const noexcept;
    std::size_t data_size() const noexcept;
    std::size_t workspace_size() const noexcept;
    std::uint32_t threads() const noexcept;

    // The JSON is intentionally self-contained: it records the axis order,
    // shared/resource bytes, and the zero-workspace contract used by callers
    // to distinguish this path from global-boundary composition.
    std::string mapping_json() const;

    // Asynchronous, allocation-free execution.  The input/output pointers are
    // device pointers and may be identical for an in-place submission.
    void execute(const void* input, void* output, cudaStream_t stream = nullptr) const;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cuntt

// Existing CUDA implementation code keeps most internal processing-unit
// interfaces in cuntt::detail.  These aliases make the standalone xD header
// usable from that layer without duplicating the public declarations.
namespace cuntt::detail {
using ::cuntt::XdMatrix2x2;
using ::cuntt::XdOperator;
using ::cuntt::XdResidentAvailability;
using ::cuntt::XdResidentConfig;
using ::cuntt::XdResidentPlan;
using ::cuntt::XdResidentType;
using ::cuntt::XdStorageType;
using ::cuntt::XdComputeType;
}  // namespace cuntt::detail
