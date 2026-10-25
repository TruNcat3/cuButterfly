#pragma once

#include <cuntt/ntt.hpp>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cuntt {

// An additive word64 interface. Roots describe the FORWARD transform even for
// inverse execution. Negacyclic uses a primitive 2N-th root, cyclic an N-th root.
// All fields are copied at construction; no external table lifetime is required.
enum class RnsRing { Cyclic, Negacyclic };
enum class RnsOrder { Natural, BitReversed };
enum class RnsPlacement { Separate, InPlace };
enum class RnsCoreFamily { Hybrid2D, FusedCTGS, GpuNttMerge };
// Modular multiply provider for the fused RNS core.  Shoup is the native
// default; Barrett mirrors the arithmetic provider used by the pinned
// GPU-NTT merge kernels and is only legal for its documented modulus range.
enum class RnsArithmetic { Shoup, Barrett };
// DependencyScoped uses a warp barrier only when every shared-memory value
// stays in the same warp between adjacent compute/load/store groups. All other
// exchanges retain the block barrier. This does not change arithmetic or order.
enum class RnsExchange { Block, DependencyScoped };
struct RnsDomain {
    std::uint64_t modulus = 0;
    std::uint64_t root = 0;
};
struct RnsPolynomial {
    std::uint32_t domain = 0;
    // Word offsets from execute_async's input/output pointers. Independent of
    // domain ordering; output intervals must be disjoint.
    std::size_t input_offset = 0;
    std::size_t output_offset = 0;
};
struct RnsPlanConfig {
    std::uint32_t log_n = 12;
    bool inverse = false;
    RnsRing ring = RnsRing::Negacyclic;
    RnsOrder input_order = RnsOrder::Natural;
    RnsOrder output_order = RnsOrder::BitReversed;
    std::vector<RnsDomain> domains;
    std::vector<RnsPolynomial> polynomials;
};
struct RnsMapping {
    std::uint32_t n1_log = 0;
    std::uint32_t rows_per_block = 1;
    std::uint32_t threads_per_block = 128;
    ComputeUnit compute_unit = ComputeUnit::Radix4;
    CrossTwiddlePlacement cross_twiddle = CrossTwiddlePlacement::FirstPass;
    // Traverse a bit-reversed boundary in physical address order. The transform
    // semantics and local arithmetic are unchanged; natural boundaries retain
    // their ordinary traversal. False preserves the original mapping.
    bool physical_boundary_traversal = false;
    // FusedCTGS embeds negacyclic roots into descending CT / ascending GS
    // stages. Its local core is radix2 or register-fused radix4; cross/traversal
    // are fixed to FirstPass/false; no separate cross or twist is executed.
    RnsCoreFamily core_family = RnsCoreFamily::Hybrid2D;
    // Execution-stage overrides: zero inherits the corresponding first-pass
    // field. "Second" means execution order, including inverse transforms.
    std::uint32_t second_rows_per_block = 0;
    std::uint32_t second_threads_per_block = 0;
    RnsArithmetic arithmetic = RnsArithmetic::Shoup;
    RnsExchange exchange = RnsExchange::Block;
};
struct RnsMeasurement {
    RnsMapping mapping;
    double kernel_ms = 0.0;
};

// Reuses Hybrid2D or fused-root CT/GS arithmetic and two-pass organization, with RNS
// domains and boundary semantics bound per polynomial. Current compiled domain:
// 12 <= log_n <= 20, prime moduli < 2^63, Shoup, radix 2/4/8, two passes whose
// local logs are 6..10 for Hybrid2D and 2..10 for FusedCTGS. FusedCTGS supports
// independent per-pass rows (through 128) and 32/64/128/256 threads, subject to resource and
// tile-count limits. It supports negacyclic radix2/4; radix4 retains
// adjacent stages in registers and uses a radix2 remainder for odd local logs.
// These are bounded candidate families, not the entire NTT space.  FusedCTGS
// additionally exposes the Shoup and GPU-NTT-compatible Barrett arithmetic
// providers as search dimensions; Barrett is rejected for moduli >= 2^62.
// GpuNttMerge is cuButterfly's GPU-NTT-compatible provider for independently
// addressable,
// N-aligned physical rows with an explicit per-polynomial domain sequence and
// mirrors the public GPU-NTT Merge contract. It does not link the vendor
// GPU-NTT library; the direct vendor control is a separate optional benchmark.
// Logical polynomial order may be a permutation of those rows. Exact
// contiguous rows use a dense fast path; sparse or permuted rows retain the
// compact offset path.
// Construction and tune() allocate/prepare outside Graph capture and timing.
// execute_async() allocates nothing and supports input == output. Keep the plan
// alive until all uses (including captured graphs) finish; one stream at a time.
// Each input coefficient must be canonical in [0, its polynomial's modulus).
// Outputs are canonical. Arbitrary uint64/lazy residue encodings are not accepted
// implicitly; callers must supply their declared representation.
class RnsPlan {
  public:
    explicit RnsPlan(const RnsPlanConfig& config);
    ~RnsPlan();
    RnsPlan(const RnsPlan&) = delete;
    RnsPlan& operator=(const RnsPlan&) = delete;
    const RnsPlanConfig& config() const noexcept;
    const RnsMapping& mapping() const noexcept;
    const std::vector<RnsMeasurement>& measurements() const noexcept;
    const SelectionInfo& selection() const noexcept;
    // Returns the active private two-pass scratch allocation.  An in-place
    // native-boundary FusedCTGS/GPU-NTT mapping releases this buffer after
    // tune(), because both passes reuse the caller's arena.
    std::size_t workspace_bytes() const noexcept;
    void set_mapping(const RnsMapping& mapping);
    // Replay a prepared mapping with its measured placement. In-place fused
    // native-boundary mappings do not retain unused private two-pass scratch.
    void set_mapping(const RnsMapping& mapping, RnsPlacement placement);
    // Bounded CUDA Graph search on private deterministic data, then confirm
    // three finalists. Includes coupled mappings and independent per-pass
    // threads/rows with equal tile capacities; not the full Cartesian product
    // of independent per-pass mappings. Explicit mappings can replay other
    // legal combinations.
    // Not a replacement for installation-wide cost-model calibration.
    // The legacy overload measures separate buffers. Specify placement when
    // the actual call aliases its pointers; aliasing can change candidate cost.
    void tune(cudaStream_t stream = nullptr, unsigned repetitions = 5);
    void tune(RnsPlacement placement, cudaStream_t stream = nullptr,
              unsigned repetitions = 5);
    void execute_async(const std::uint64_t* input, std::uint64_t* output,
                       cudaStream_t stream = nullptr);
  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace cuntt
