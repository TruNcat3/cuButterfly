#pragma once
#include <cuntt/butterfly.hpp>
#include <cubutterfly/module.h>
namespace cuntt::detail {
bool on_demand_compilation_enabled();
bool specialize_shared_templates();
const cubutterflyModuleV2* prepare_shared_module(const ButterflyConfig& config);
const cubutterflyModuleV2* prepare_shared_module(const PlanConfig& config);
cubutterflyModuleGroupLaunchV2 prepared_group_launcher(const cubutterflyModuleV2* module);
cubutterflyModuleGroupLaunchV1 prepared_group_launcher(const cubutterflyModuleV1* module);
cubutterflyModuleRangeLaunchV1 prepared_range_launcher(const cubutterflyModuleV1* module);
void attach_module_resources(MixedDataflowPlan& plan, const cubutterflyModuleV2* module, bool inverse);
void attach_module_resources(MixedDataflowPlan& plan, const cubutterflyModuleV1* module, bool inverse);
// Called only while constructing a plan. Compilation and CUDA resource queries
// are never part of execution timing.
void prepare_register_module(const ButterflyConfig& config);
const cubutterflyModuleV1* prepared_register_module(const ButterflyConfig& config);
bool launch_prepared_register_module(const ButterflyConfig& config, const void* input,
                                    void* output, void* scratch, cudaStream_t stream);
}
