#pragma once

#include <cuntt/mixed_dataflow.hpp>
#include <cuda_runtime_api.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuntt {
class Plan;
class ButterflyPlan;
namespace detail {

struct StageProbeGroup {
    ExecutionGroup execution;
    bool independent = false;
    std::string reason;
    // The physical kernel updates its input allocation. The probe restores
    // that allocation outside each timing interval, without adding a copy to
    // the measured kernel or to the original plan.
    bool in_place = false;
};

// Internal diagnostic view. The originating plan, its coefficients and all
// endpoint/workspace allocations must outlive every submitted invocation.
// Construction and resource queries are outside benchmark timing.
class StageProbeAdapter {
public:
    explicit StageProbeAdapter(Plan&);
    explicit StageProbeAdapter(ButterflyPlan&);
    const std::vector<StageProbeGroup>& groups() const { return groups_; }
    void launch_group(unsigned group, const void* input, void* output,
                      void* workspace, cudaStream_t stream) const {
        if (group >= groups_.size() || !groups_[group].independent)
            throw std::invalid_argument("physical group has no independent probe entry");
        group_launch_(group, input, output, workspace, stream);
    }
    void launch_plan(const void* input, void* output, void* workspace,
                     cudaStream_t stream) const {
        plan_launch_(input, output, workspace, stream);
    }
private:
    std::vector<StageProbeGroup> groups_;
    std::function<void(unsigned, const void*, void*, void*, cudaStream_t)> group_launch_;
    std::function<void(const void*, void*, void*, cudaStream_t)> plan_launch_;
};

}  // namespace detail
}  // namespace cuntt
