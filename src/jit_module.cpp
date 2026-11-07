#include "jit_module.hpp"
#include "jit_config.hpp"
#include "cufftdx_provider.hpp"
#include "fft_local_core_contract.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace cuntt::detail {
namespace {
std::string setting(const char* name, const std::string& fallback) {
    const auto* value = std::getenv(name);
    return value ? value : fallback;
}
std::string cache_path() {
    return setting("CUBUTTERFLY_JIT_CACHE", setting("XDG_CACHE_HOME", setting("HOME", ".")+"/.cache")+"/cubutterfly/modules");
}
std::string mapping(const ButterflyConfig& c) {
    if (fp16_local_block(c)) {
        fp16_local_block_shape(c);
        return nlohmann::json{{"backend", "temporal-tile"}, {"fft_core", "cufftdx-block"},
            {"precision", "fp16"}, {"accumulation", "native"}, {"logN", c.log_n},
            {"tile_threads", c.tile_threads}, {"prefix_ept", c.prefix_ept},
            {"packing", "half2-transform-pair-v1"}}.dump();
    }
    if(c.backend==ButterflyBackend::FactorStreamed) {
        auto point=nlohmann::json{{"backend","factor-streamed"},{"precision",butterfly_precision_name(c.precision)},
            {"logN",c.log_n},{"stage_partition",c.stage_partition},{"factor_partition",c.factor_partition},
            {"factor_ept",c.factor_ept},{"factor_columns",c.factor_columns},
            {"data_tiles_per_cta",c.data_tiles_per_cta},{"prefetch_depth",c.prefetch_depth}};
        if (c.precision == ButterflyPrecision::Fp16) {
            point["accumulation"] = "native";
            point["packing"] = "half2-transform-pair-v1";
        }
        // Preserve the legacy cache point for the all-dynamic policy.  A
        // static-unrolled policy is a distinct lowering and must be visible
        // in the key rather than reusing a dynamic module.
        if (std::any_of(c.factor_io_policies.begin(), c.factor_io_policies.end(),
                        [](const std::string& policy) { return policy != "dynamic"; }))
            point["factor_io_policies"] = c.factor_io_policies;
        if(c.fft_core==FftCore::RegisterTile) point["fft_core"]="register-tile";
        if(c.factor_slices>1) point["factor_slices"]=c.factor_slices;
        return point.dump();
    }
    return nlohmann::json{{"fft_core", "register-tile"}, {"precision", butterfly_precision_name(c.precision)},
        {"logN", c.log_n}, {"local_stages", c.local_stages}, {"prefix_threads", c.prefix_threads},
        {"prefix_ept", c.prefix_ept}, {"suffix_threads", c.suffix_threads}, {"suffix_ept", c.suffix_ept},
        {"prefix_codelet_lanes", c.prefix_codelet_lanes}, {"suffix_core", c.suffix_core},
        {"prefix_codelet", c.prefix_codelet}, {"prefix_shared_layout", c.prefix_shared_layout},
        {"prefix_twiddle", c.prefix_twiddle},
        {"local_stage_partitions",c.local_stage_partitions},{"exchange_chunks",c.exchange_chunks}}.dump();
}
std::string key(const ButterflyConfig& c) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) throw std::runtime_error("cannot identify the module's CUDA device");
    return std::to_string(device)+"/"+mapping(c);
}
std::string run_compiler(const std::vector<std::string>& command) {
    int descriptors[2];
    if (pipe(descriptors)) throw std::runtime_error("cannot open module compiler output pipe");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, descriptors[0]);
    posix_spawn_file_actions_addclose(&actions, descriptors[1]);
    std::vector<char*> argv;
    for (const auto& arg : command) argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    pid_t process = -1;
    const auto spawned = posix_spawnp(&process, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(descriptors[1]);
    if (spawned) { close(descriptors[0]); throw std::runtime_error("could not start specialization compiler"); }
    std::string output;
    char buffer[4096];
    for (ssize_t n; (n = read(descriptors[0], buffer, sizeof(buffer))) > 0;) output.append(buffer, n);
    close(descriptors[0]);
    int status = 0;
    while (waitpid(process, &status, 0) < 0) if (errno != EINTR) throw std::runtime_error("cannot wait for module compiler");
    if (!WIFEXITED(status) || WEXITSTATUS(status)) throw std::runtime_error("module compilation failed: " + output);
    while (!output.empty() && (output.back()=='\n' || output.back()=='\r')) output.pop_back();
    return output;
}
struct Module {
    void* handle = nullptr;
    const cubutterflyModuleV1* api = nullptr;
    const cubutterflyModuleV2* generic = nullptr;
    cubutterflyModuleGroupLaunchV2 group_launch = nullptr;
    cubutterflyModuleGroupLaunchV1 register_group_launch = nullptr;
    cubutterflyModuleRangeLaunchV1 range_launch = nullptr;
    cubutterflyModuleLocalBytesV1 local_bytes = nullptr;
    ~Module() { if (handle) dlclose(handle); }
};
std::map<std::string, std::unique_ptr<Module>> modules;
std::mutex module_mutex;

std::filesystem::path template_root() {
    if (const auto* configured=std::getenv("CUBUTTERFLY_TEMPLATE_ROOT")) return configured;
    std::error_code error;
    const auto executable=std::filesystem::read_symlink("/proc/self/exe",error);
    for (const auto& path : {std::filesystem::path(CUBUTTERFLY_SOURCE_ROOT),
            executable.parent_path().parent_path()/"share/cuButterfly/templates",
            std::filesystem::path(CUBUTTERFLY_JIT_PYTHON).parent_path().parent_path()/"share/cuButterfly/templates",
            std::filesystem::path(CUBUTTERFLY_INSTALLED_ROOT)})
        if(std::filesystem::exists(path/"scripts/compile_module.py")) return path;
    throw std::runtime_error("specialization templates were not installed; set CUBUTTERFLY_TEMPLATE_ROOT");
}

std::string compile(const std::string& point, unsigned sm) {
    const auto root=template_root();
    auto mathdx=setting("CUBUTTERFLY_MATHDX_ROOT",CUBUTTERFLY_JIT_MATHDX);
    if(!std::filesystem::exists(mathdx+"/include/cufftdx.hpp")) mathdx=(root/"mathdx").string();
    return run_compiler({setting("CUBUTTERFLY_JIT_PYTHON",std::filesystem::exists(CUBUTTERFLY_JIT_PYTHON) ? CUBUTTERFLY_JIT_PYTHON : "python3"),
        (root/"scripts/compile_module.py").string(),"--mapping",point,"--root",root.string(),
        "--mathdx",mathdx,"--nvcc",setting("CUBUTTERFLY_NVCC",std::filesystem::exists(CUBUTTERFLY_JIT_NVCC) ? CUBUTTERFLY_JIT_NVCC : "nvcc"),
        "--sm",std::to_string(sm),"--cache",cache_path(),"--timeout",setting("CUBUTTERFLY_COMPILE_TIMEOUT",specialize_shared_templates() ? "0" : "600")});
}

const cubutterflyModuleV2* prepare_shared(nlohmann::json point) {
    const auto info=current_device_info();
    const auto sm=info.compute_major*10+info.compute_minor;
    int device=0; if(cudaGetDevice(&device)!=cudaSuccess) throw std::runtime_error("cannot identify module device");
    const auto id=std::to_string(device)+"/"+point.dump();
    std::lock_guard<std::mutex> guard(module_mutex);
    if (modules.count(id)) return modules.at(id)->generic;
    const auto path=compile(point.dump(),sm);
    auto module=std::make_unique<Module>();
    module->handle=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!module->handle) throw std::runtime_error(std::string("cannot load generic specialization: ")+dlerror());
    const auto entry=reinterpret_cast<cubutterflyModuleEntryV2>(dlsym(module->handle,"cubutterfly_module_v2"));
    if(!entry) throw std::runtime_error("generic specialization has no V2 entry");
    module->generic=entry();
    module->group_launch=reinterpret_cast<cubutterflyModuleGroupLaunchV2>(
        dlsym(module->handle,"cubutterfly_module_launch_group_v2"));
    if (!module->group_launch) throw std::runtime_error("generic module has no group launch entry");
    const auto* api=module->generic;
    if(!api || api->abi_version!=2 || api->struct_size!=sizeof(cubutterflyModuleV2) || api->sm!=static_cast<unsigned>(sm) ||
        api->group_count!=point.at("stage_partition").size()) throw std::runtime_error("generic module ABI or mapping mismatch");
    const auto hardware=query_hardware_resource_model();
    for(unsigned g=0;g<api->group_count;++g) {
        cubutterflyModuleResourcesV1 resource{};
        if(api->resources(0,g,&resource)!=cudaSuccess) throw std::invalid_argument("generic specialization cannot satisfy device resource limits");
        if(resource.threads>hardware.max_threads_per_block || std::uint64_t(resource.static_shared_bytes)+resource.dynamic_shared_bytes>hardware.shared_bytes_per_block ||
            std::uint64_t(resource.threads)*resource.registers_per_thread>hardware.registers_per_block)
            throw std::invalid_argument("generic specialization exceeds device resources");
    }
    modules.emplace(id,std::move(module));
    return api;
}
}

CompilePolicy compile_policy() noexcept {
    const auto* value = std::getenv("CUBUTTERFLY_COMPILE_MODE");
    const std::string_view policy(value ? value : CUBUTTERFLY_DEFAULT_CODEGEN_MODE);
    if (policy == "precompiled") return CompilePolicy::Precompiled;
    if (policy == "auto") return CompilePolicy::Auto;
    if (policy == "on-demand") return CompilePolicy::OnDemand;
    if (policy == "research") return CompilePolicy::Research;
    return CompilePolicy::Invalid;
}

bool on_demand_compilation_enabled() {
    const auto policy = compile_policy();
    if (policy == CompilePolicy::Invalid)
        throw std::invalid_argument("CUBUTTERFLY_COMPILE_MODE must be precompiled, auto, on-demand, or research");
    return policy != CompilePolicy::Precompiled;
}

bool specialize_shared_templates() {
    const auto policy = compile_policy();
    if (policy == CompilePolicy::Invalid)
        throw std::invalid_argument("CUBUTTERFLY_COMPILE_MODE must be precompiled, auto, on-demand, or research");
    return policy == CompilePolicy::Research;
}

const cubutterflyModuleV2* prepare_shared_module(const ButterflyConfig& c) {
    nlohmann::json point{{"backend","shared-iterative"},{"operator",butterfly_operator_name(c.op)},
        {"precision",butterfly_precision_name(c.precision)},{"accumulation",butterfly_accumulation_name(c.accumulation)},
        {"complex_multiply",complex_multiply_name(c.complex_multiply)},{"logN",c.log_n},
        {"stage_partition",c.stage_partition},{"threads",c.tile_threads},
        {"data_tiles_per_cta",c.data_tiles_per_cta},
        {"local_stage_partitions",c.local_stage_partitions},{"exchange_chunks",c.exchange_chunks},
        {"writer_aligned",c.shared_layout==SharedLayout::WriterAligned},
        {"batch_tile",c.stage_overlap ? 1U : c.batch_tile_count}};
    if (!c.execution_group_mappings.empty()) {
        point["group_threads"] = nlohmann::json::array();
        for (const auto& group : c.execution_group_mappings)
            point["group_threads"].push_back(group.threads);
    }
    return prepare_shared(std::move(point));
}
const cubutterflyModuleV2* prepare_shared_module(const PlanConfig& c) {
    nlohmann::json point{{"backend","shared-iterative"},{"operator","ntt"},
        {"precision","word"+std::to_string(c.word_bits)},{"logN",c.log_n},{"stage_partition",c.stage_partition},
        {"threads",c.threads_per_block},{"writer_aligned",c.dataflow_layout==DataflowLayout::HermesXor},
        {"local_stage_partitions",c.local_stage_partitions},{"exchange_chunks",c.exchange_chunks},
        {"output_order",output_order_name(c.output_order)},
        {"batch_tile",c.stage_overlap ? 1U : c.batch_tile_count}};
    if (!c.execution_group_mappings.empty()) {
        point["group_threads"] = nlohmann::json::array();
        for (const auto& group : c.execution_group_mappings)
            point["group_threads"].push_back(group.threads_per_block);
    }
    return prepare_shared(std::move(point));
}
template <typename ModuleApi>
void attach_resources(MixedDataflowPlan& plan, const ModuleApi* module, bool inverse) {
    if(!module || module->group_count!=plan.execution_groups.size()) throw std::logic_error("module/IR group mismatch");
    cubutterflyModuleLocalBytesV1 local_bytes=nullptr;
    { std::lock_guard<std::mutex> guard(module_mutex);
      for(const auto& entry:modules) if(static_cast<const void*>(entry.second->api)==module || static_cast<const void*>(entry.second->generic)==module)
          local_bytes=entry.second->local_bytes;
    }
    for(unsigned g=0;g<module->group_count;++g) {
        cubutterflyModuleResourcesV1 resource{};
        if(module->resources(inverse,g,&resource)!=cudaSuccess) throw std::runtime_error("cannot read module resources");
        auto& group=plan.execution_groups[g];
        group.compiler_resources_known=true; group.compiler_registers_per_thread=resource.registers_per_thread;
        group.static_shared_bytes=resource.static_shared_bytes;
        group.dynamic_shared_bytes=resource.dynamic_shared_bytes;
        group.shared_components_known=true;
        group.live_shared_bytes=std::uint64_t(resource.dynamic_shared_bytes)+resource.static_shared_bytes;
        group.threads=resource.threads;
        group.compiler_local_bytes_per_thread=0;
        group.compiler_local_resources_known=false;
        if(local_bytes) {
            if(local_bytes(inverse,g,&group.compiler_local_bytes_per_thread)!=cudaSuccess)
                throw std::runtime_error("cannot read compiler local-memory allocation");
            group.compiler_local_resources_known=true;
        }
    }
}
void attach_module_resources(MixedDataflowPlan& plan, const cubutterflyModuleV2* module, bool inverse) {
    attach_resources(plan,module,inverse);
}
void attach_module_resources(MixedDataflowPlan& plan, const cubutterflyModuleV1* module, bool inverse) {
    attach_resources(plan,module,inverse);
}
cubutterflyModuleGroupLaunchV2 prepared_group_launcher(const cubutterflyModuleV2* module) {
    std::lock_guard<std::mutex> guard(module_mutex);
    for (const auto& entry : modules) if (entry.second->generic == module)
        return entry.second->group_launch;
    throw std::logic_error("shared module has not been prepared");
}
cubutterflyModuleGroupLaunchV1 prepared_group_launcher(const cubutterflyModuleV1* module) {
    std::lock_guard<std::mutex> guard(module_mutex);
    for (const auto& entry : modules) if (entry.second->api == module)
        return entry.second->register_group_launch;
    throw std::logic_error("register module has not been prepared");
}
cubutterflyModuleRangeLaunchV1 prepared_range_launcher(const cubutterflyModuleV1* module) {
    std::lock_guard<std::mutex> guard(module_mutex);
    for(const auto& entry:modules) if(entry.second->api==module) return entry.second->range_launch;
    throw std::logic_error("factor module has not been prepared");
}

void prepare_register_module(const ButterflyConfig& c) {
    if (!on_demand_compilation_enabled()) throw std::invalid_argument("mapping requires compilation; enable CUBUTTERFLY_COMPILE_MODE=research");
    const auto id = key(c);
    std::lock_guard<std::mutex> guard(module_mutex);
    if (modules.count(id)) return;
    const auto device = current_device_info();
    const auto sm = device.compute_major*10 + device.compute_minor;
    const auto require_provider = [&](unsigned size, unsigned ept) {
        const auto precision_bits = cufftdx_arithmetic_precision_bits(c);
        if (!precision_bits)
            throw ProviderImplementationMissing("local-arithmetic-not-implemented: cuButterfly cuFFTDx adapter does not implement the requested precision/accumulation/normalization contract");
        require_cufftdx_provider(sm*10, *precision_bits, size, ept);
    };
    if (fp16_local_block(c)) {
        fp16_local_block_shape(c);
        require_provider(1U<<c.log_n, c.prefix_ept);
    } else if (c.backend==ButterflyBackend::FactorStreamed) {
        if (c.fft_core==FftCore::CufftDxBlock)
            for (auto stage : c.factor_partition)
                require_provider(1U<<stage, c.factor_ept);
    } else if (c.suffix_core == "cufftdx-block") {
        require_provider(1U<<(c.log_n-c.local_stages), c.suffix_ept);
    }
    const auto path = compile(mapping(c),sm);
    auto module = std::make_unique<Module>();
    module->handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module->handle) throw std::runtime_error(std::string("cannot load specialization: ")+dlerror());
    const auto entry = reinterpret_cast<cubutterflyModuleEntryV1>(dlsym(module->handle, "cubutterfly_module_v1"));
    if (!entry) throw std::runtime_error("module has no versioned cuButterfly entry point");
    module->api = entry();
    module->local_bytes=reinterpret_cast<cubutterflyModuleLocalBytesV1>(dlsym(module->handle,"cubutterfly_module_local_bytes_v1"));
    module->register_group_launch=reinterpret_cast<cubutterflyModuleGroupLaunchV1>(
        dlsym(module->handle,"cubutterfly_module_launch_group_v1"));
    module->range_launch=reinterpret_cast<cubutterflyModuleRangeLaunchV1>(
        dlsym(module->handle,"cubutterfly_module_launch_range_v1"));
    if(c.factor_slices>1 && !module->range_launch) throw std::runtime_error("factor module has no partial launch entry");
    if (!module->register_group_launch) throw std::runtime_error("register module has no group launch entry");
    if (!module->api || module->api->abi_version!=1 || module->api->struct_size!=sizeof(cubutterflyModuleV1) ||
        module->api->sm!=static_cast<unsigned>(sm) || module->api->group_count!=(fp16_local_block(c) ? 1 :
            c.backend==ButterflyBackend::FactorStreamed ? c.factor_partition.size() : 2))
        throw std::runtime_error("specialization ABI or hardware mismatch");
    const auto hardware = query_hardware_resource_model();
    for (unsigned direction=0; direction<2; ++direction) for (unsigned group=0; group<module->api->group_count; ++group) {
        cubutterflyModuleResourcesV1 resource{};
        if (module->api->resources(direction, group, &resource)!=cudaSuccess)
            throw std::runtime_error("cannot query compiled specialization resources");
        if (resource.threads>hardware.max_threads_per_block ||
            std::uint64_t(resource.dynamic_shared_bytes)+resource.static_shared_bytes>hardware.shared_bytes_per_block ||
            std::uint64_t(resource.threads)*resource.registers_per_thread>hardware.registers_per_block)
            throw std::invalid_argument("compiled specialization exceeds active-device resources");
    }
    modules.emplace(id, std::move(module));
}

const cubutterflyModuleV1* prepared_register_module(const ButterflyConfig& c) {
    const auto id = key(c);
    std::lock_guard<std::mutex> guard(module_mutex);
    const auto found=modules.find(id);
    return found==modules.end() ? nullptr : found->second->api;
}

bool launch_prepared_register_module(const ButterflyConfig& c, const void* input,
                                    void* output, void* scratch, cudaStream_t stream) {
    const auto* api=prepared_register_module(c);
    if(!api) return false;
    const cubutterflyModuleInvocationV1 invocation{input, output, scratch, c.batch, c.batch_stride,
                                                  c.element_stride, c.inverse, c.normalize_inverse, stream};
    const auto status = api->launch(&invocation);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(status)));
    return true;
}
}
