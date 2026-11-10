#include "xd_real_fft.hpp"
#include "xd_vkfft_axis.hpp"

#include "general_kernels.hpp"
#include "xd_axis_fft.hpp"
#include "xd_pair_fft.hpp"
#include "xd_pair_core.hpp"
#include "xd_register_fft.hpp"
#include "xd_local_fft.hpp"
#include "xd_cooperative_pair_fft.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <climits>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace cuntt::detail {
namespace {

constexpr std::size_t kWorkspaceAlignment = 256;
constexpr unsigned kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuFFT status " + std::to_string(status));
}

std::size_t checked_product(std::size_t left, std::size_t right, const char* what) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
        throw std::overflow_error(std::string(what) + " overflows size_t");
    return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right, const char* what) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
        throw std::overflow_error(std::string(what) + " overflows size_t");
    return left + right;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    const auto remainder = value % alignment;
    return remainder == 0 ? value : checked_add(value, alignment - remainder, "workspace alignment");
}

int checked_int(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument(std::string(what) + " exceeds cuFFT int plan limit");
    return static_cast<int>(value);
}

unsigned blocks_for(std::size_t count, const char* what) {
    const auto blocks = (count + kThreads - 1) / kThreads;
    if (blocks > static_cast<std::size_t>(UINT_MAX))
        throw std::overflow_error(std::string(what) + " grid exceeds CUDA grid.x");
    return static_cast<unsigned>(blocks);
}

template <typename Real>
__global__ void scale_real_kernel(Real* values, std::size_t count, Real scale) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        values[index] *= scale;
}

template <typename Complex, typename Real>
__global__ void scale_complex_kernel(Complex* values, std::size_t count, Real scale) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index].x *= scale;
        values[index].y *= scale;
    }
}

struct AxisPlan {
    cufftHandle handle = 0;
    std::unique_ptr<XdVkfftAxisPlan> vkfft_core;
    std::size_t axis = 0;
    std::size_t axis_length = 0;
    std::size_t suffix = 0;
    std::size_t prefix_count = 0;
    std::size_t prefix_stride = 0;
    std::size_t launches = 0;
    std::size_t batch_transforms = 0;
    std::size_t batch_distance = 0;
    std::size_t element_stride = 0;
    std::size_t work_bytes = 0;
    bool packed = false;
    bool fused = false;
    bool register_core = false;
    bool local_core = false;
    XdPairCore pair_core = XdPairCore::Thread;
    XdAxisResources resource{};
    std::size_t paired_axis = 0;
    std::size_t pair_rows = 0;
    std::size_t pair_planes = 0;
    GeneralNdLayout input_layout{};
    std::vector<std::size_t> input_order;
    std::vector<std::uint32_t> to_axis_last;
};

void destroy_plan(cufftHandle& handle) noexcept {
    if (handle != 0) {
        cufftDestroy(handle);
        handle = 0;
    }
}

}  // namespace

class XdRealFftPlan::Impl {
  public:
    Impl(const std::vector<std::size_t>& extents, std::size_t batch,
         bool inverse, bool normalize_inverse, bool fp64, bool complex_only,
         double output_scale, bool packed_axes,
         const std::vector<std::size_t>& axis_order, bool fused_axes,
         unsigned axis_ept, bool paired_axes, XdAxisCore axis_core, XdPairCore pair_core,
         unsigned register_lanes, unsigned pair_layout, std::size_t batch_tile,
         unsigned axis_ffts_per_block_cap, unsigned axis_exchange_policy,
         const std::vector<XdAxisResources>& axis_resources, bool c2r_prefix_scale)
        : extents_(extents), batch_(batch), inverse_(inverse),
          normalize_inverse_(normalize_inverse), fp64_(fp64),
          complex_only_(complex_only), output_scale_(output_scale),
          packed_axes_(packed_axes), fused_axes_(fused_axes), axis_ept_(axis_ept),
          paired_axes_(paired_axes), axis_core_(axis_core), pair_core_(pair_core),
          register_lanes_(register_lanes), pair_layout_(pair_layout), batch_tile_(batch_tile),
          axis_order_(axis_order), axis_ffts_per_block_cap_(axis_ffts_per_block_cap),
          axis_exchange_policy_(axis_exchange_policy), axis_resources_(axis_resources),
          c2r_prefix_scale_(c2r_prefix_scale) {
        validate_shape();
        if (batch_tile_ != 0) {
            if (!complex_only_ || !packed_axes_ || batch_tile_ >= batch_)
                throw std::invalid_argument("xD batch tiles require packed C2C and a tile smaller than batch");
            batch_body_ = std::make_unique<Impl>(extents_,batch_tile_,inverse_,normalize_inverse_,
                fp64_,complex_only_,output_scale_,packed_axes_,axis_order_,fused_axes_,axis_ept_,
                paired_axes_,axis_core_,pair_core_,register_lanes_,pair_layout_,0,axis_ffts_per_block_cap_,axis_exchange_policy_,axis_resources_,false);
            workspace_bytes_ = batch_body_->workspace_size();
            const auto remainder = batch_ % batch_tile_;
            if (remainder) {
                batch_tail_ = std::make_unique<Impl>(extents_,remainder,inverse_,normalize_inverse_,
                    fp64_,complex_only_,output_scale_,packed_axes_,axis_order_,fused_axes_,axis_ept_,
                    paired_axes_,axis_core_,pair_core_,register_lanes_,pair_layout_,0,axis_ffts_per_block_cap_,axis_exchange_policy_,axis_resources_,false);
                workspace_bytes_ = std::max(workspace_bytes_,batch_tail_->workspace_size());
            }
            return;
        }
        build_plans();
    }

    ~Impl() {
        destroy_plan(real_plan_);
        for (auto& plan : axis_plans_)
            destroy_plan(plan.handle);
    }

    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

    std::size_t workspace_size() const noexcept { return workspace_bytes_; }

    bool can_fuse_c2r_scale() const noexcept {
        return inverse_ && !complex_only_ && !axis_plans_.empty() && axis_plans_.back().fused;
    }

    void set_workspace(void* workspace, std::size_t bytes) {
        if (workspace == nullptr) {
            if (bytes != 0)
                throw std::invalid_argument("null xD FFT workspace must have zero bytes");
            // A later non-null call installs a fresh work area on every plan.
            // Keeping the cuFFT handles alive here avoids rebuilding plans
            // when a caller temporarily releases its candidate workspace.
            workspace_ = nullptr;
            workspace_capacity_ = 0;
            if (batch_body_) batch_body_->set_workspace(nullptr,0);
            if (batch_tail_) batch_tail_->set_workspace(nullptr,0);
            return;
        }
        if (bytes < workspace_bytes_)
            throw std::invalid_argument("xD FFT workspace is smaller than workspace_size()");
        if (reinterpret_cast<std::uintptr_t>(workspace) % alignof(std::max_align_t) != 0)
            throw std::invalid_argument("xD FFT workspace is not sufficiently aligned");
        if (batch_body_) batch_body_->set_workspace(workspace,bytes);
        if (batch_tail_) batch_tail_->set_workspace(workspace,bytes);
        for (const auto& plan : axis_plans_)
            if (plan.handle != 0)
                check_cufft(cufftSetWorkArea(plan.handle, workspace), "set xD C2C work area");
        if (real_plan_ != 0)
            check_cufft(cufftSetWorkArea(real_plan_, workspace), "set xD real work area");
        workspace_ = workspace;
        workspace_capacity_ = bytes;
    }

    void execute(const void* input, void* output, cudaStream_t stream) {
        if (input == nullptr || output == nullptr)
            throw std::invalid_argument("xD FFT input and output must be non-null");
        if (workspace_bytes_ != 0 && workspace_ == nullptr)
            throw std::invalid_argument("xD FFT workspace must be set before execute");
        if (batch_body_) {
            const auto bytes_per_transform = logical_points_ * (fp64_ ? sizeof(cufftDoubleComplex) : sizeof(cufftComplex));
            const auto* source = static_cast<const unsigned char*>(input);
            auto* destination = static_cast<unsigned char*>(output);
            std::size_t first = 0;
            for (; batch_ - first >= batch_tile_; first += batch_tile_)
                batch_body_->execute(source + first * bytes_per_transform,
                                     destination + first * bytes_per_transform,stream);
            if (batch_tail_)
                batch_tail_->execute(source + first * bytes_per_transform,
                                     destination + first * bytes_per_transform,stream);
            return;
        }
        update_stream(stream);
        if (fp64_)
            execute_typed<cufftDoubleReal, cufftDoubleComplex>(input, output, stream);
        else
            execute_typed<float, cufftComplex>(input, output, stream);
    }

    std::string mapping_json() const {
        if (batch_body_) {
            auto mapping = nlohmann::json::parse(batch_body_->mapping_json());
            // Top-level axes describe one complete tile. The tail records its
            // actual launch sizes; the semantic batch remains the user's B.
            const auto body_mapping = mapping;
            mapping["tile_body"] = body_mapping;
            if (batch_tail_) mapping["tile_tail"] = nlohmann::json::parse(batch_tail_->mapping_json());
            mapping["batch"] = batch_;
            mapping["batch_tile"] = batch_tile_;
            mapping["batch_chunks"] = batch_ / batch_tile_ + (batch_tail_ ? 1 : 0);
            mapping["axes_scope"] = "per-batch-tile";
            // Rank-N C2R can consume the caller's compact spectrum directly
            // when axis plans exist.  Keep this as an explicit capability in
            // the replay record so the selector does not confuse the optimized
            // path with the legacy copy fallback.
            mapping["c2r_input_copy_elision"] = inverse_ && !complex_only_ && !axis_plans_.empty();
            mapping["c2r_input_boundary"] = inverse_ && !complex_only_
                                                  ? (axis_plans_.empty() ? "scratch-copy" : "direct-compact")
                                                  : "not-applicable";
            const auto tile_cost = body_mapping.at("cost");
            auto total_cost = tile_cost;
            const std::size_t body_chunks = batch_ / batch_tile_;
            const std::size_t chunks = body_chunks + (batch_tail_ ? 1U : 0U);
            total_cost["batch_chunks"] = chunks;
            total_cost["per_tile"] = tile_cost;
            for (const char* field : {"prefix_launches", "permutation_launches", "restore_permutation_launches",
                                      "scale_kernel_launches", "c2r_d2d_copy_launches",
                                      "real_boundary_launches", "total_launches"})
                total_cost[field] = tile_cost.at(field).get<std::size_t>() * body_chunks;
            if (batch_tail_) {
                const auto tail = nlohmann::json::parse(batch_tail_->mapping_json());
                const auto tail_cost = tail.value("cost", nlohmann::json::object());
                if (tail_cost.contains("provider_owned_workspace_bytes"))
                    total_cost["provider_owned_workspace_bytes"] = checked_add(
                        total_cost.value("provider_owned_workspace_bytes", std::size_t{0}),
                        tail_cost.at("provider_owned_workspace_bytes").get<std::size_t>(),
                        "xD tiled provider workspace");
                for (const char* field : {"prefix_launches", "permutation_launches", "restore_permutation_launches",
                                          "scale_kernel_launches", "c2r_d2d_copy_launches",
                                          "real_boundary_launches", "total_launches"})
                    total_cost[field] = total_cost.at(field).get<std::size_t>() +
                        tail_cost.value(field, 0U);
            }
            mapping["cost"] = std::move(total_cost);
            return mapping.dump();
        }
        std::ostringstream json;
        json << "{\"kind\":\"xd-real-fft\",\"rank\":" << extents_.size();
        // Preserve exact replay of pre-cap mappings when using the old policy.
        if (axis_ffts_per_block_cap_)
            json << ",\"axis_ffts_per_block_cap\":" << axis_ffts_per_block_cap_;
        if (axis_exchange_policy_)
            json << ",\"axis_exchange_policy\":" << axis_exchange_policy_;
        json
             << ",\"batch\":" << batch_ << ",\"inverse\":" << (inverse_ ? "true" : "false")
             << ",\"batch_tile\":0,\"batch_chunks\":1"
             << ",\"normalize_inverse\":" << (normalize_inverse_ ? "true" : "false")
             << ",\"fp64\":" << (fp64_ ? "true" : "false")
             << ",\"complex_only\":" << (complex_only_ ? "true" : "false")
             << ",\"packed_axes\":" << (packed_axes_ ? "true" : "false")
             << ",\"c2r_input_copy_elision\":"
             << (inverse_ && !complex_only_ && !axis_plans_.empty() ? "true" : "false")
             << ",\"c2r_input_boundary\":\""
             << (inverse_ && !complex_only_ ? (axis_plans_.empty() ? "scratch-copy" : "direct-compact") : "not-applicable")
             << "\""
             << ",\"cost\":" << mapping_cost_json().dump()
             << ",\"fused_axes\":" << (fused_axes_ ? "true" : "false")
             << ",\"axis_ept\":" << axis_ept_
             << ",\"register_lanes\":" << register_lanes_
             << ",\"pair_layout\":" << pair_layout_
             << ",\"paired_axes\":" << (paired_axes_ ? "true" : "false")
             << ",\"pair_groups\":" << pair_groups_
             << ",\"axis_core\":\"" << xd_axis_core_name(axis_core_) << "\""
             << ",\"pair_core\":\"" << xd_pair_core_name(pair_core_) << "\""
             << ",\"layout_policy\":\"" << (packed_axes_ ? "retain-axis-layout-v1" : "natural-strided") << "\""
             << ",\"output_scale\":" << std::setprecision(17) << output_scale_
             << ",\"axis_order\":[";
        for (std::size_t i = 0; i < axis_order_.size(); ++i)
            json << (i == 0 ? "" : ",") << axis_order_[i];
        json << "]"
             << ",\"extents\":[";
        for (std::size_t i = 0; i < extents_.size(); ++i)
            json << (i == 0 ? "" : ",") << extents_[i];
        json << "]";
        if (c2r_prefix_scale_)
            json << ",\"c2r_prefix_scale\":true,\"c2r_scale_placement\":\"last-complex-axis\"";
        if (!complex_only_)
            json << ",\"compact_last_extent\":" << compact_last_extent_;
        if (!axis_resources_.empty()) {
            json << ",\"axis_resources\":[";
            for (std::size_t i = 0; i < axis_resources_.size(); ++i) {
                const auto& resource = axis_resources_[i];
                if (i != 0)
                    json << ',';
                json << "{\"core\":\""
                     << xd_axis_core_name(resource.core)
                     << "\",\"ept\":" << resource.ept
                     << ",\"register_lanes\":" << resource.register_lanes
                     << ",\"block_cap\":" << resource.block_cap
                     << ",\"exchange\":" << resource.exchange;
                if (resource.local_twiddle_policy != 0)
                    json << ",\"local_twiddle_policy\":"
                         << resource.local_twiddle_policy;
                if (resource.ffts_per_block != 0)
                    json << ",\"ffts_per_block\":" << resource.ffts_per_block;
                json << "}";
            }
            json << "]";
        }
        json << ",\"axes\":[";
        for (std::size_t i = 0; i < axis_plans_.size(); ++i) {
            const auto& plan = axis_plans_[i];
            const unsigned ept = axis_resources_.empty() ? axis_ept_ : plan.resource.ept;
            const unsigned lanes = axis_resources_.empty() ? register_lanes_ :
                                                                  plan.resource.register_lanes;
            const unsigned block_cap = axis_resources_.empty() ? axis_ffts_per_block_cap_ :
                                                                        plan.resource.block_cap;
            const unsigned exchange = axis_resources_.empty() ? axis_exchange_policy_ :
                                                                      plan.resource.exchange;
            if (i != 0)
                json << ',';
            json << "{\"axis\":" << plan.axis
                 << ",\"length\":" << plan.axis_length
                 << ",\"stride\":" << plan.element_stride
                 << ",\"batch\":" << plan.batch_transforms
                 << ",\"idist\":" << plan.batch_distance
                 << ",\"prefix_loops\":" << plan.launches
                 << ",\"packed\":" << (plan.packed ? "true" : "false")
                 << ",\"core\":\""
                 << (plan.vkfft_core ? "vkfft-axis" :
                     plan.pair_rows ? xd_pair_execution_name(plan.pair_core) :
                     plan.local_core ? "thread-shared-axis" :
                     plan.register_core ? "register-cooperative-axis" :
                     plan.fused ? "cufftdx-axis-tiled" :
                     plan.packed ? "cufft-rank1-batched" : "cufft-rank1-strided")
                 << "\",\"elements_per_thread\":" << (plan.local_core ? (plan.axis_length == 64 ? 8 : plan.axis_length / 16) : plan.register_core ? plan.axis_length / lanes : plan.fused && !plan.pair_rows ? xd_axis_fft_ept(plan.axis_length, ept) : 0)
                 << ",\"ffts_per_block\":" << (plan.local_core ? xd_local_fft_lines_per_block(plan.axis_length, fp64_, block_cap) : plan.register_core ? 128 / lanes : plan.fused && !plan.pair_rows ? xd_axis_fft_block_batch(plan.axis_length, fp64_, ept, block_cap, plan.resource.ffts_per_block) : 0)
                 << ",\"cooperative_lanes\":" << (plan.register_core ? lanes : plan.pair_rows ? xd_pair_cooperative_lanes(plan.pair_core) : 0)
                 << ",\"twiddle_policy\":\"" << (plan.pair_rows ? xd_pair_twiddle_policy(plan.pair_core) : plan.local_core &&
                     plan.resource.local_twiddle_policy == 1 ?
                         (plan.axis_length == 256 ? "recurrence-root-256" :
                          plan.axis_length == 64 ? "recurrence-root-64" :
                          "recurrence-root-128") :
                     (plan.register_core || plan.local_core) && plan.axis_length == 256 ?
                         "constant-roots-256" :
                     plan.local_core && plan.axis_length == 64 ?
                         "constant-roots-64" :
                     plan.register_core || plan.local_core ?
                         "constant-roots-128" : "core-defined") << "\""
                 << ",\"pair_rows\":" << plan.pair_rows
                 << ",\"pair_planes\":" << plan.pair_planes
                 << ",\"paired_axis\":" << plan.paired_axis
                 << ",\"axis_suffix\":" << plan.suffix
                 << ",\"boundary\":\""
                 << (plan.packed ? "permute_nd" : "none") << "\",\"input_axis_order\":[";
            for (std::size_t j = 0; j < plan.input_order.size(); ++j)
                json << (j == 0 ? "" : ",") << plan.input_order[j];
            json << "],\"permutation\":[";
            for (std::size_t j = 0; j < plan.to_axis_last.size(); ++j)
                json << (j == 0 ? "" : ",") << plan.to_axis_last[j];
            json << "]";
            if (plan.pair_rows && plan.pair_core == XdPairCore::Radix4Register)
                json << ",\"pair_axis_radices\":[4,4,2]"
                     << ",\"io_exchange\":\"" << (fp64_ ? "padded-shared-complex128" : "padded-shared-complex64") << "\""
                     << ",\"transform_semantics\":\"separable-axes\""
                     << ",\"supports_fused_output_scale\":true";
            if (plan.vkfft_core)
                json << ",\"provider\":" << plan.vkfft_core->metadata_json()
                     << ",\"supports_fused_output_scale\":false";
            if (plan.local_core)
                json << ",\"io_exchange\":\"soa-xor-local\",\"local_factors\":["
                     << (plan.axis_length == 64 ? 8 : 16) << ","
                     << (plan.axis_length == 64 ? 8 : plan.axis_length / 16) << "]";
            if (exchange && plan.fused && !plan.pair_rows && !plan.register_core && !plan.local_core)
                json << ",\"io_exchange\":\"" << (plan.packed ? "packed-contiguous-register" :
                    plan.suffix == 1 ? "contiguous-register" :
                    exchange == 1 ? "direct-strided-register" :
                    exchange == 3 ? "unrolled-staged-u32" : "unrolled-staged") << "\"";
            if (exchange == 4 && plan.fused)
                json << ",\"core_exchange_policy\":0,\"core_suffix\":1"
                     << ",\"fft_launches\":" << plan.launches
                     << ",\"permutation_launches\":" << (plan.packed ? 1 : 0);
            json << "}";
        }
        json << "],\"output_permutation\":[";
        if (restore_output_) {
            for (std::size_t j = 0; j < restore_order_.size(); ++j)
                json << (j == 0 ? "" : ",") << restore_order_[j];
        }
        json << "]}";
        return json.str();
    }

    nlohmann::json mapping_cost_json() const {
        std::size_t prefix_launches = 0;
        std::size_t permutation_launches = restore_output_ ? 1U : 0U;
        for (const auto& plan : axis_plans_) {
            prefix_launches += plan.launches;
            permutation_launches += plan.packed ? 1U : 0U;
        }
        const bool core_scales = !axis_plans_.empty() && axis_plans_.back().fused;
        const double inverse_scale = output_scale_ *
            (inverse_ && normalize_inverse_ ? 1.0 / static_cast<double>(logical_points_) : 1.0);
        const bool scale_kernel = complex_only_
            ? (inverse_scale != 1.0 && !core_scales)
            : (inverse_ ? inverse_scale != 1.0 && !c2r_prefix_scale_ : output_scale_ != 1.0 && !core_scales);
        const bool d2d_copy = inverse_ && !complex_only_ && axis_plans_.empty();
        const std::size_t real_boundary = complex_only_ ? 0U : 1U;
        nlohmann::json cost{
            {"prefix_launches", prefix_launches},
            {"permutation_launches", permutation_launches},
            {"restore_permutation_launches", restore_output_ ? 1U : 0U},
            {"scale_kernel_launches", scale_kernel ? 1U : 0U},
            {"c2r_d2d_copy_launches", d2d_copy ? 1U : 0U},
            {"real_boundary_launches", real_boundary},
            {"total_launches", prefix_launches + permutation_launches +
                (scale_kernel ? 1U : 0U) + (d2d_copy ? 1U : 0U) + real_boundary},
            {"workspace_bytes", workspace_bytes_},
            {"compact_boundary_bytes", compact_bytes_},
            {"source", "lowering-instance"}
        };
        std::size_t provider_workspace = 0;
        bool has_provider = false;
        for (const auto& plan : axis_plans_) if (plan.vkfft_core) {
            has_provider = true;
            provider_workspace = checked_add(provider_workspace,
                plan.vkfft_core->owned_workspace_size(), "xD provider workspace");
        }
        if (has_provider) {
            cost["provider_owned_workspace_bytes"] = provider_workspace;
            cost["prefix_launch_count_scope"] = "axis invocations; provider may issue multiple kernels";
        }
        return cost;
    }

  private:
    using Real32 = float;
    using Real64 = double;

    void validate_shape() {
        if (extents_.empty() || extents_.size() > kGeneralMaxRank)
            throw std::invalid_argument("xD FFT rank must be one through " +
                                        std::to_string(kGeneralMaxRank));
        if (batch_ == 0)
            throw std::invalid_argument("xD real FFT batch must be positive");
        if (!std::isfinite(output_scale_))
            throw std::invalid_argument("xD FFT output scale must be finite");
        if (c2r_prefix_scale_ && (!inverse_ || complex_only_ || !fused_axes_))
            throw std::invalid_argument("C2R prefix scaling requires inverse real FFT with fused axes");
        if (axis_ept_ != 4 && axis_ept_ != 8 && axis_ept_ != 16)
            throw std::invalid_argument("xD FFT axis EPT must be 4, 8 or 16");
        if (axis_ffts_per_block_cap_ != 0 && axis_ffts_per_block_cap_ != 8 && axis_ffts_per_block_cap_ != 16)
            throw std::invalid_argument("xD FFT axis FFTs per block cap must be 0, 8 or 16");
        if (axis_exchange_policy_ > 4)
            throw std::invalid_argument("xD FFT axis exchange policy must be in the range 0 through 4");
        if (axis_exchange_policy_ == 4 &&
            (!packed_axes_ || !fused_axes_ || paired_axes_ || axis_core_ != XdAxisCore::CufftdxBlock))
            throw std::invalid_argument("xD packed-core exchange requires packed fused cuFFTDx axes without pairs");
        if (register_lanes_ != 4 && register_lanes_ != 8 && register_lanes_ != 16)
            throw std::invalid_argument("xD FFT register lanes must be 4, 8 or 16");
        if (pair_layout_ > 1)
            throw std::invalid_argument("xD FFT pair layout must be padded or XOR");
        xd_pair_core_name(pair_core_);
        if (pair_layout_ && pair_core_ != XdPairCore::CooperativeRegister)
            throw std::invalid_argument("only the cooperative pair core implements XOR layout");
        if (!paired_axes_ && pair_core_ != XdPairCore::Thread)
            throw std::invalid_argument("a custom pair core requires paired axes");
        if (paired_axes_ && !fused_axes_)
            throw std::invalid_argument("xD paired axes require tiled complex-axis cores");
        xd_axis_core_name(axis_core_);  // Validate the enum even for an empty axis list.
        if (axis_core_ != XdAxisCore::CufftdxBlock && !fused_axes_)
            throw std::invalid_argument("xD custom axis core requires tiled axes");
        checked_int(batch_, "xD FFT batch");
        logical_points_ = 1;
        for (const auto extent : extents_) {
            if (extent == 0)
                throw std::invalid_argument("xD FFT extents must be positive");
            checked_int(extent, "xD FFT extent");
            logical_points_ = checked_product(logical_points_, extent, "xD FFT point count");
        }
        compact_last_extent_ = complex_only_ ? extents_.back() : extents_.back() / 2 + 1;
        compact_points_ = 1;
        for (std::size_t i = 0; i + 1 < extents_.size(); ++i)
            compact_points_ = checked_product(compact_points_, extents_[i], "xD FFT compact point count");
        compact_points_ = checked_product(compact_points_, compact_last_extent_, "xD FFT compact point count");
        compact_points_ = checked_product(compact_points_, batch_, "xD FFT compact point count");
        const auto complex_bytes = fp64_ ? sizeof(cufftDoubleComplex) : sizeof(cufftComplex);
        compact_bytes_ = checked_product(compact_points_, complex_bytes, "xD FFT compact bytes");
        const auto real_points = checked_product(logical_points_, batch_, "xD FFT real point count");
        real_bytes_ = checked_product(real_points, fp64_ ? sizeof(double) : sizeof(float), "xD FFT real bytes");
        const std::size_t complex_axis_count = complex_only_ ? extents_.size() : extents_.size() - 1;
        if (!axis_resources_.empty()) {
            if (!fused_axes_ || !packed_axes_ || paired_axes_)
                throw std::invalid_argument(
                    "xD per-axis resources require fused packed axes without paired axes");
            if (axis_resources_.size() != complex_axis_count)
                throw std::invalid_argument(
                    "xD per-axis resources must cover every logical complex axis");
            for (const auto& resource : axis_resources_)
                validate_xd_axis_resource(resource);
        }
        if (axis_order_.empty()) {
            axis_order_.reserve(complex_axis_count);
            for (std::size_t axis = complex_axis_count; axis-- > 0;)
                axis_order_.push_back(axis);
        }
        if (axis_order_.size() != complex_axis_count)
            throw std::invalid_argument("xD FFT axis order has the wrong number of complex axes");
        std::vector<bool> seen(complex_axis_count, false);
        for (const auto axis : axis_order_) {
            if (axis >= complex_axis_count || seen[axis])
                throw std::invalid_argument("xD FFT axis order is not a permutation");
            seen[axis] = true;
        }
    }

    void build_plans() {
        try {
            if (!complex_only_) {
                if (inverse_)
                    real_plan_ = make_real_plan(CUFFT_C2R);
                else
                    real_plan_ = make_real_plan(CUFFT_R2C);
            }

            std::vector<std::size_t> working_dims = extents_;
            if (!complex_only_)
                working_dims.back() = compact_last_extent_;
            std::vector<std::size_t> physical_order(extents_.size());
            std::iota(physical_order.begin(), physical_order.end(), 0);
            for (std::size_t step = 0; step < axis_order_.size(); ++step) {
                const auto axis = axis_order_[step];
                const auto position = static_cast<std::size_t>(
                    std::find(physical_order.begin(), physical_order.end(), axis) - physical_order.begin());
                AxisPlan plan;
                // Fuse only a dependency-closed contiguous plane. The axis
                // kernels inside the group execute last-axis then penultimate
                // axis, exactly matching this pair in the logical schedule.
                const auto paired_position = step + 1 < axis_order_.size()
                    ? static_cast<std::size_t>(std::find(physical_order.begin(), physical_order.end(),
                          axis_order_[step + 1]) - physical_order.begin()) : working_dims.size();
                const bool contiguous_pair = position + 1 == working_dims.size() && position > 0 &&
                    paired_position + 1 == position;
                const bool pair = paired_axes_ && paired_position < working_dims.size() &&
                    (contiguous_pair || packed_axes_) &&
                    xd_pair_core_available(pair_core_, working_dims[paired_position],
                                           working_dims[position], fp64_, pair_layout_);
                if (pair) {
                    plan.fused = true;
                    plan.axis_length = working_dims[position];
                    plan.pair_rows = working_dims[paired_position];
                    if (!contiguous_pair) {
                        // Compact-real's Hermitian axis remains a plane index.
                        // Pack only the two scheduled complex axes, then retain
                        // this layout until the ordinary final permutation.
                        plan.packed = true;
                        plan.input_layout = make_layout(working_dims);
                        for (std::size_t p = 0; p < working_dims.size(); ++p)
                            if (p != paired_position && p != position)
                                plan.to_axis_last.push_back(static_cast<std::uint32_t>(p));
                        plan.to_axis_last.push_back(static_cast<std::uint32_t>(paired_position));
                        plan.to_axis_last.push_back(static_cast<std::uint32_t>(position));
                        has_packed_axis_ = true;
                    }
                    plan.paired_axis = axis_order_[++step];
                    plan.pair_planes = compact_points_ / (plan.pair_rows * plan.axis_length);
                    plan.pair_core = pair_core_;
                    prepare_xd_pair_core(plan.pair_core, plan.pair_rows, plan.axis_length,
                                         fp64_, inverse_, pair_layout_);
                    plan.suffix = plan.element_stride = 1;
                    plan.batch_transforms = plan.pair_planes;
                    plan.batch_distance = plan.pair_rows * plan.axis_length;
                    plan.launches = 1;
                    ++pair_groups_;
                } else {
                    plan = make_axis_plan(position, working_dims,
                                           axis_resources_.empty() ? nullptr :
                                               &axis_resources_[axis]);
                }
                try {
                    plan.axis = axis;
                    plan.input_order = physical_order;
                    if (plan.packed) {
                        std::vector<std::size_t> next_dims, next_order;
                        for (const auto source_axis : plan.to_axis_last) {
                            next_dims.push_back(working_dims[source_axis]);
                            next_order.push_back(physical_order[source_axis]);
                        }
                        working_dims = std::move(next_dims);
                        physical_order = std::move(next_order);
                    }
                    axis_plans_.push_back(std::move(plan));
                    // AxisPlan owns the handle only after it is in the vector.
                    plan.handle = 0;
                } catch (...) {
                    destroy_plan(plan.handle);
                    throw;
                }
            }
            if (c2r_prefix_scale_ && !can_fuse_c2r_scale())
                throw std::invalid_argument("C2R prefix scaling requires an available final fused complex axis");
            final_layout_ = make_layout(working_dims);
            restore_order_.resize(extents_.size());
            for (std::size_t physical = 0; physical < physical_order.size(); ++physical) {
                restore_order_[physical_order[physical]] = static_cast<std::uint32_t>(physical);
                restore_output_ = restore_output_ || physical_order[physical] != physical;
            }
            std::size_t max_work = real_work_bytes_;
            for (const auto& plan : axis_plans_)
                max_work = std::max(max_work, plan.work_bytes);
            plan_work_bytes_ = max_work;
            data_offset_ = align_up(plan_work_bytes_, kWorkspaceAlignment);
            // Only C2R needs an input copy: cuFFT may overwrite its compact
            // spectrum. C2C preserves input through an out-of-place first axis.
            const auto data_bytes = inverse_ && !complex_only_ ? compact_bytes_ : 0;
            packed_axis_offset_ = align_up(
                checked_add(data_offset_, data_bytes, "xD FFT packed-axis offset"),
                kWorkspaceAlignment);
            workspace_bytes_ = has_packed_axis_
                                   ? checked_add(packed_axis_offset_, compact_bytes_,
                                                 "xD FFT workspace bytes")
                                   : checked_add(data_offset_, data_bytes,
                                                 "xD FFT workspace bytes");
        } catch (...) {
            destroy_plan(real_plan_);
            for (auto& plan : axis_plans_)
                destroy_plan(plan.handle);
            axis_plans_.clear();
            throw;
        }
    }

    cufftHandle make_real_plan(cufftType type) {
        const auto n = checked_int(extents_.back(), "xD real FFT length");
        std::size_t prefix_extent = 1;
        for (std::size_t i = 0; i + 1 < extents_.size(); ++i)
            prefix_extent = checked_product(prefix_extent, extents_[i], "xD real FFT prefix count");
        const auto prefix = checked_product(batch_, prefix_extent, "xD real FFT transform count");
        const auto compact = checked_int(compact_last_extent_, "xD real FFT compact length");
        const auto distance_in = type == CUFFT_R2C ? n : compact;
        const auto distance_out = type == CUFFT_R2C ? compact : n;
        cufftHandle handle = 0;
        try {
            check_cufft(cufftCreate(&handle), "create xD real plan");
            check_cufft(cufftSetAutoAllocation(handle, 0), "disable xD real plan allocation");
            int dimensions[1] = {n};
            int inembed[1] = {type == CUFFT_R2C ? n : compact};
            int onembed[1] = {type == CUFFT_R2C ? compact : n};
            size_t work = 0;
            check_cufft(cufftMakePlanMany(handle, 1, dimensions, inembed, 1,
                                         checked_int(distance_in, "xD real input distance"),
                                         onembed, 1,
                                         checked_int(distance_out, "xD real output distance"),
                                         fp64_ ? (type == CUFFT_R2C ? CUFFT_D2Z : CUFFT_Z2D) : type,
                                         checked_int(prefix, "xD real transform count"), &work),
                        "make xD real plan");
            real_work_bytes_ = work;
            return handle;
        } catch (...) {
            destroy_plan(handle);
            throw;
        }
    }

    AxisPlan make_axis_plan(std::size_t axis, const std::vector<std::size_t>& dims,
                            const XdAxisResources* resource) {
        AxisPlan result;
        result.axis = axis;
        result.axis_length = dims[axis];
        result.suffix = 1;
        for (std::size_t i = axis + 1; i < dims.size(); ++i)
            result.suffix = checked_product(result.suffix, dims[i], "xD axis suffix");
        result.prefix_count = 1;
        for (std::size_t i = 0; i < axis; ++i)
            result.prefix_count = checked_product(result.prefix_count, dims[i], "xD axis prefix");
        result.prefix_count = checked_product(result.prefix_count, batch_, "xD axis prefix batch");
        result.element_stride = result.suffix;
        const auto exchange = resource != nullptr ? resource->exchange : axis_exchange_policy_;
        const auto requested_core = resource != nullptr ? resource->core : axis_core_;
        if (fused_axes_ && requested_core == XdAxisCore::Vkfft) {
            if (!xd_vkfft_axis_available())
                throw std::invalid_argument("VkFFT axis provider is not enabled in this build");
            validate_xd_axis_resource(resource ? *resource : XdAxisResources{
                axis_core_, axis_ept_, register_lanes_, axis_ffts_per_block_cap_, exchange});
            if (resource) result.resource = *resource;
            result.batch_transforms = checked_product(result.prefix_count, result.suffix,
                                                       "xD VkFFT axis lines");
            result.batch_distance = result.suffix == 1 ? result.axis_length : 1;
            result.launches = 1;
            result.vkfft_core = std::make_unique<XdVkfftAxisPlan>(result.axis_length,
                result.suffix, result.batch_transforms, fp64_, inverse_);
            // An external axis provider is not a whole xD transform. Keep its
            // natural layout; only the requested axis is transformed. It does
            // not expose arbitrary output scaling, unlike our device cores.
            return result;
        }
        if (resource != nullptr) {
            result.resource = *resource;
            if (resource->core == XdAxisCore::CooperativeRegister) {
                if (result.suffix == 1)
                    throw std::invalid_argument(
                        "xD cooperative-register resource requires a non-contiguous axis");
                if (!xd_register_fft_configuration_available(result.axis_length,
                                                              resource->register_lanes))
                    throw std::invalid_argument(
                        "xD cooperative-register resource is unavailable for this axis");
                result.register_core = true;
                result.fused = true;
            } else if (resource->core == XdAxisCore::ThreadShared) {
                if (!xd_local_fft_available(result.axis_length) ||
                    xd_local_fft_lines_per_block(result.axis_length, fp64_, resource->block_cap) == 0)
                    throw std::invalid_argument("xD thread-shared resource is unavailable for this axis");
                result.local_core = true;
                result.fused = true;
            } else {
                // A requested block resource may be absent from this build or
                // target SM. Keep the established cuFFT fallback in that
                // case, while the axis metadata below reports the fallback
                // from the actual plan flags rather than claiming a tiled
                // launch.
                result.fused = xd_axis_fft_configuration_available(
                    result.axis_length, fp64_, resource->ept, resource->block_cap,
                    resource->ffts_per_block, exchange == 4 ? 0U : exchange);
                if (resource->ffts_per_block != 0 && !result.fused)
                    throw std::invalid_argument("exact xD FFTsPerBlock resource is unavailable for this axis");
            }
        } else {
            result.register_core = fused_axes_ && axis_core_ == XdAxisCore::CooperativeRegister &&
                                   result.suffix > 1 &&
                                   xd_register_fft_configuration_available(result.axis_length, register_lanes_);
            result.local_core = fused_axes_ && axis_core_ == XdAxisCore::ThreadShared &&
                                xd_local_fft_available(result.axis_length);
            result.fused = result.register_core || result.local_core ||
                           (fused_axes_ && xd_axis_fft_configuration_available(
                               result.axis_length, fp64_, axis_ept_, axis_ffts_per_block_cap_));
        }
        if (result.fused) {
            result.batch_transforms = checked_product(result.prefix_count, result.suffix,
                                                       "xD fused axis lines");
            if (exchange == 3 && !result.register_core && !result.local_core && result.suffix != 1 &&
                !xd_axis_fft_u32_index_fits(result.axis_length, result.suffix, result.batch_transforms))
                throw std::invalid_argument("xD unrolled32 axis exceeds the 32-bit element index range");
            result.launches = 1;
            // Policies 0..3 decode all prefixes in the current layout.
            // Policy 4 continues below to retain a contiguous axis-last layout.
            result.batch_distance = result.suffix == 1 ? result.axis_length : 1;
            if (exchange != 4 || result.suffix == 1)
                return result;
        }
        result.packed = packed_axes_ && result.suffix != 1;
        if (result.packed) {
            // Move this physical axis last and retain that layout for the
            // next logical axis. Restore the public layout only at the end.
            result.input_layout = make_layout(dims);
            result.to_axis_last.reserve(dims.size());
            for (std::size_t candidate = 0; candidate < dims.size(); ++candidate) {
                if (candidate != axis)
                    result.to_axis_last.push_back(static_cast<std::uint32_t>(candidate));
            }
            result.to_axis_last.push_back(static_cast<std::uint32_t>(axis));
            result.element_stride = 1;
            result.batch_transforms = checked_product(result.prefix_count, result.suffix,
                                                       "xD packed axis batch");
            result.batch_distance = result.axis_length;
            result.prefix_stride = 0;
            result.launches = 1;
            has_packed_axis_ = true;
        } else if (result.suffix == 1) {
            result.batch_transforms = result.prefix_count;
            result.batch_distance = result.axis_length;
            result.prefix_stride = 0;
            result.launches = 1;
        } else {
            result.batch_transforms = result.suffix;
            result.batch_distance = 1;
            result.prefix_stride = checked_product(result.axis_length, result.suffix, "xD axis prefix stride");
            result.launches = result.prefix_count;
        }
        if (result.fused)
            return result;
        checked_int(result.axis_length, "xD axis length");
        checked_int(result.element_stride, "xD axis stride");
        checked_int(result.batch_distance, "xD axis distance");
        checked_int(result.batch_transforms, "xD axis batch");

        cufftHandle handle = 0;
        try {
            check_cufft(cufftCreate(&handle), "create xD axis plan");
            check_cufft(cufftSetAutoAllocation(handle, 0), "disable xD axis allocation");
            int n[1] = {checked_int(result.axis_length, "xD axis length")};
            int embed[1] = {n[0]};
            size_t work = 0;
            check_cufft(cufftMakePlanMany(
                            handle, 1, n, embed, checked_int(result.element_stride, "xD axis stride"),
                            checked_int(result.batch_distance, "xD axis distance"), embed,
                            checked_int(result.element_stride, "xD axis stride"),
                            checked_int(result.batch_distance, "xD axis distance"),
                            fp64_ ? CUFFT_Z2Z : CUFFT_C2C,
                            checked_int(result.batch_transforms, "xD axis batch"), &work),
                        "make xD axis plan");
            result.handle = handle;
            result.work_bytes = work;
            return result;
        } catch (...) {
            destroy_plan(handle);
            throw;
        }
    }

    GeneralNdLayout make_layout(const std::vector<std::size_t>& dims) const {
        if (dims.empty() || dims.size() > kGeneralMaxRank)
            throw std::invalid_argument("xD FFT permutation rank is unsupported");
        GeneralNdLayout layout;
        layout.rank = static_cast<std::uint32_t>(dims.size());
        layout.batch = batch_;
        layout.points = 1;
        for (std::size_t axis = dims.size(); axis-- > 0;) {
            layout.extents[axis] = dims[axis];
            layout.strides[axis] = layout.points;
            layout.points = checked_product(layout.points, dims[axis], "xD permutation point count");
        }
        layout.batch_stride = layout.points;
        return layout;
    }

    void update_stream(cudaStream_t stream) {
        if (stream_initialized_ && stream_ == stream)
            return;
        if (real_plan_ != 0)
            check_cufft(cufftSetStream(real_plan_, stream), "set xD real stream");
        for (const auto& plan : axis_plans_)
            if (plan.handle != 0)
                check_cufft(cufftSetStream(plan.handle, stream), "set xD axis stream");
        stream_ = stream;
        stream_initialized_ = true;
    }

    template <typename Real, typename Complex>
    void execute_typed(const void* input, void* output, cudaStream_t stream) {
        const auto* input_real = static_cast<const Real*>(input);
        auto* output_real = static_cast<Real*>(output);
        const auto* input_complex = static_cast<const Complex*>(input);
        auto* output_complex = static_cast<Complex*>(output);
        if (complex_only_) {
            execute_complex_only(input_complex, output_complex, stream);
            return;
        }
        if (!inverse_) {
            if constexpr (std::is_same_v<Real, double>)
                check_cufft(cufftExecD2Z(real_plan_, const_cast<cufftDoubleReal*>(input_real),
                                         reinterpret_cast<cufftDoubleComplex*>(output_complex)),
                            "execute xD D2Z");
            else
                check_cufft(cufftExecR2C(real_plan_, const_cast<cufftReal*>(input_real),
                                         reinterpret_cast<cufftComplex*>(output_complex)),
                            "execute xD R2C");
            const bool core_scales = !axis_plans_.empty() && axis_plans_.back().fused;
            execute_c2c_axes(output_complex, output_complex, CUFFT_FORWARD, false,
                             core_scales ? output_scale_ : 1.0);
            if (output_scale_ != 1.0 && !core_scales)
                launch_complex_scale(output_complex, compact_points_, output_scale_, stream);
            return;
        }

        auto* scratch = reinterpret_cast<Complex*>(static_cast<unsigned char*>(workspace_) + data_offset_);
        const double scale = output_scale_ * (normalize_inverse_ ? 1.0 / static_cast<double>(logical_points_) : 1.0);
        if (!axis_plans_.empty()) {
            // Axis plans can read the caller's compact spectrum while writing
            // their first result into owned scratch; avoid the redundant D2D
            // copy that is required by the legacy no-axis path.
            execute_c2c_axes(const_cast<Complex*>(input_complex), scratch,
                             CUFFT_INVERSE, true, c2r_prefix_scale_ ? scale : 1.0);
        } else {
            check_cuda(cudaMemcpyAsync(scratch, input_complex, compact_bytes_, cudaMemcpyDeviceToDevice, stream),
                       "copy xD inverse compact input");
            execute_c2c_axes(scratch, scratch, CUFFT_INVERSE);
        }
        if constexpr (std::is_same_v<Real, double>)
            check_cufft(cufftExecZ2D(real_plan_, reinterpret_cast<cufftDoubleComplex*>(scratch), output_real),
                        "execute xD Z2D");
        else
            check_cufft(cufftExecC2R(real_plan_, reinterpret_cast<cufftComplex*>(scratch), output_real),
                        "execute xD C2R");
        if (scale != 1.0 && !c2r_prefix_scale_)
            launch_real_scale(output_real, checked_product(logical_points_, batch_, "xD inverse output count"),
                              static_cast<Real>(scale), stream);
    }

    template <typename Complex>
    void execute_complex_only(const Complex* input, Complex* output, cudaStream_t stream) {
        const double scale = output_scale_ * (inverse_ && normalize_inverse_
                                 ? 1.0 / static_cast<double>(logical_points_) : 1.0);
        const bool core_scales = !axis_plans_.empty() && axis_plans_.back().fused;
        execute_c2c_axes(const_cast<Complex*>(input), output,
                         inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD, true, core_scales ? scale : 1.0);
        if (scale != 1.0 && !core_scales)
            launch_complex_scale<Complex, double>(output, compact_points_, scale, stream);
    }

    template <typename Complex>
    void execute_c2c_axes(Complex* input, Complex* output, int direction,
                          bool first_out_of_place = false, double final_scale = 1.0) {
        Complex* current_input = input;
        auto* packed = has_packed_axis_ ? reinterpret_cast<Complex*>(
            static_cast<unsigned char*>(workspace_) + packed_axis_offset_) : nullptr;
        for (std::size_t index = 0; index < axis_plans_.size(); ++index) {
            const auto& plan = axis_plans_[index];
            Complex* current_output = first_out_of_place && index == 0 ? output : current_input;
            if (plan.packed) {
                // Ping-pong only between owned destinations; never overwrite
                // the caller's input. Each permutation has distinct buffers.
                auto* destination = current_input == packed ? output : packed;
                launch_permute_nd(current_input, destination, plan.input_layout,
                                  plan.to_axis_last.data(), sizeof(Complex), stream_);
                current_input = destination;
                current_output = destination;
            }
            // Let the last FFT write into the buffer needed by the final
            // boundary. Both cores support out-of-place C2C, so no copy is needed
            // to repair ping-pong parity when retaining several axis layouts.
            if (index + 1 == axis_plans_.size())
                current_output = restore_output_ ? packed : output;
            if (plan.vkfft_core) {
                plan.vkfft_core->execute(current_input, current_output, stream_);
                current_input = current_output;
                continue;
            }
            if (plan.fused) {
                const double scale = index + 1 == axis_plans_.size() ? final_scale : 1.0;
                if (plan.pair_rows)
                    launch_xd_pair_core(plan.pair_core, current_input, current_output,
                                        plan.pair_rows, plan.axis_length, plan.pair_planes,
                                        fp64_, direction == CUFFT_INVERSE, scale, stream_, pair_layout_);
                else if (plan.local_core)
                    launch_xd_local_fft(current_input, current_output, plan.axis_length,
                                        plan.suffix, plan.batch_transforms, fp64_,
                                        direction == CUFFT_INVERSE, scale, stream_,
                                        axis_resources_.empty() ? axis_ffts_per_block_cap_ :
                                                                   plan.resource.block_cap,
                                        axis_resources_.empty() ? 0U :
                                                                   plan.resource.local_twiddle_policy);
                else if (plan.register_core)
                    launch_xd_register_fft(current_input, current_output, plan.axis_length,
                                           plan.suffix, plan.batch_transforms, fp64_,
                                           direction == CUFFT_INVERSE, scale, stream_,
                                           axis_resources_.empty() ? register_lanes_ :
                                                                      plan.resource.register_lanes);
                else
                    launch_xd_axis_fft(current_input, current_output, plan.axis_length,
                                       plan.packed ? 1U : plan.suffix, plan.batch_transforms, fp64_,
                                       direction == CUFFT_INVERSE, scale, stream_,
                                       axis_resources_.empty() ? axis_ept_ : plan.resource.ept,
                                       axis_resources_.empty() ? axis_ffts_per_block_cap_ :
                                                                  plan.resource.block_cap,
                                       (axis_resources_.empty() ? axis_exchange_policy_ :
                                                                  plan.resource.exchange) == 4
                                           ? 0U : (axis_resources_.empty() ? axis_exchange_policy_ :
                                                                             plan.resource.exchange),
                                       plan.resource.ffts_per_block);
                current_input = current_output;
                continue;
            }
            for (std::size_t prefix = 0; prefix < plan.launches; ++prefix) {
                auto* in = current_input;
                auto* out = current_output;
                if (plan.prefix_stride != 0) {
                    const auto offset = prefix * plan.prefix_stride;
                    in += offset;
                    out += offset;
                }
                if (fp64_)
                    check_cufft(cufftExecZ2Z(plan.handle,
                                             reinterpret_cast<cufftDoubleComplex*>(in),
                                             reinterpret_cast<cufftDoubleComplex*>(out), direction),
                                "execute xD Z2Z axis");
                else
                    check_cufft(cufftExecC2C(plan.handle,
                                             reinterpret_cast<cufftComplex*>(in),
                                             reinterpret_cast<cufftComplex*>(out), direction),
                                "execute xD C2C axis");
            }
            current_input = current_output;
        }
        if (restore_output_)
            launch_permute_nd(current_input, output, final_layout_,
                              restore_order_.data(), sizeof(Complex), stream_);
    }

    template <typename Real>
    void launch_real_scale(Real* output, std::size_t count, Real scale, cudaStream_t stream) {
        scale_real_kernel<<<blocks_for(count, "xD real normalization"), kThreads, 0, stream>>>(output, count, scale);
        check_cuda(cudaGetLastError(), "launch xD real normalization");
    }

    template <typename Complex, typename Real>
    void launch_complex_scale(Complex* output, std::size_t count, Real scale, cudaStream_t stream) {
        scale_complex_kernel<<<blocks_for(count, "xD complex normalization"), kThreads, 0, stream>>>(output, count, scale);
        check_cuda(cudaGetLastError(), "launch xD complex normalization");
    }

    std::vector<std::size_t> extents_;
    std::size_t batch_ = 0;
    bool inverse_ = false;
    bool normalize_inverse_ = false;
    bool fp64_ = false;
    bool complex_only_ = false;
    double output_scale_ = 1.0;
    bool packed_axes_ = false;
    bool fused_axes_ = false;
    unsigned axis_ept_ = 8;
    bool paired_axes_ = false;
    std::size_t pair_groups_ = 0;
    XdAxisCore axis_core_ = XdAxisCore::CufftdxBlock;
    XdPairCore pair_core_ = XdPairCore::Thread;
    unsigned register_lanes_ = 4;
    unsigned pair_layout_ = 0;
    std::size_t batch_tile_ = 0;
    std::unique_ptr<Impl> batch_body_;
    std::unique_ptr<Impl> batch_tail_;
    std::vector<std::size_t> axis_order_;
    std::vector<XdAxisResources> axis_resources_;
    bool c2r_prefix_scale_ = false;
    unsigned axis_ffts_per_block_cap_ = 0;
    unsigned axis_exchange_policy_ = 0;
    bool has_packed_axis_ = false;
    bool restore_output_ = false;
    GeneralNdLayout final_layout_{};
    std::vector<std::uint32_t> restore_order_;
    std::size_t logical_points_ = 0;
    std::size_t compact_last_extent_ = 0;
    std::size_t compact_points_ = 0;
    std::size_t compact_bytes_ = 0;
    std::size_t real_bytes_ = 0;
    std::size_t real_work_bytes_ = 0;
    std::size_t plan_work_bytes_ = 0;
    std::size_t data_offset_ = 0;
    std::size_t packed_axis_offset_ = 0;
    std::size_t workspace_bytes_ = 0;
    std::size_t workspace_capacity_ = 0;
    void* workspace_ = nullptr;
    cufftHandle real_plan_ = 0;
    std::vector<AxisPlan> axis_plans_;
    cudaStream_t stream_ = nullptr;
    bool stream_initialized_ = false;
};

XdRealFftPlan::XdRealFftPlan(const std::vector<std::size_t>& extents, std::size_t batch,
                             bool inverse, bool normalize_inverse, bool fp64,
                             bool complex_only, double output_scale, bool packed_axes,
                             const std::vector<std::size_t>& axis_order, bool fused_axes,
                             unsigned axis_ept, bool paired_axes,
                             XdAxisCore axis_core, XdPairCore pair_core,
                             unsigned register_lanes, unsigned pair_layout, std::size_t batch_tile,
                             unsigned axis_ffts_per_block_cap, unsigned axis_exchange_policy,
                             const std::vector<XdAxisResources>& axis_resources, bool c2r_prefix_scale)
    : impl_(std::make_unique<Impl>(extents, batch, inverse, normalize_inverse,
                                   fp64, complex_only, output_scale, packed_axes, axis_order,
                                   fused_axes, axis_ept, paired_axes, axis_core, pair_core,
                                   register_lanes, pair_layout, batch_tile, axis_ffts_per_block_cap,
                                   axis_exchange_policy, axis_resources, c2r_prefix_scale)) {}

XdRealFftPlan::~XdRealFftPlan() = default;

XdRealFftPlan::XdRealFftPlan(XdRealFftPlan&&) noexcept = default;

XdRealFftPlan& XdRealFftPlan::operator=(XdRealFftPlan&&) noexcept = default;

std::size_t XdRealFftPlan::workspace_size() const noexcept { return impl_->workspace_size(); }

void XdRealFftPlan::set_workspace(void* workspace, std::size_t bytes) {
    impl_->set_workspace(workspace, bytes);
}

void XdRealFftPlan::execute(const void* input, void* output, cudaStream_t stream) {
    impl_->execute(input, output, stream);
}

std::string XdRealFftPlan::mapping_json() const { return impl_->mapping_json(); }

bool XdRealFftPlan::can_fuse_c2r_scale() const noexcept { return impl_->can_fuse_c2r_scale(); }

}  // namespace cuntt::detail
