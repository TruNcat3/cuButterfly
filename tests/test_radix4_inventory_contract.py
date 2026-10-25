"""CPU checks of the actual register inventory, validation and mapping replay."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_provider_cpp_contract import _json_include


ROOT = Path(__file__).resolve().parents[1]


def test_prebuilt_register_inventory_candidate_and_replay_contract(tmpdir):
    tmp_path = Path(str(tmpdir))
    compiler, includes = shutil.which("c++"), _json_include()
    cuda = Path("/usr/local/cuda/include")
    if not compiler or not includes or not cuda.is_dir():
        pytest.skip("CPU compiler, CUDA host headers and JSON headers required")
    registry = (ROOT / "src/plan_registry.cpp").read_text()
    helpers = registry[registry.index("std::string string_value("):registry.index("std::string registry_path()")]
    macros = registry[registry.index("#define BUTTERFLY_ENUM_FIELDS"):registry.index("#define NTT_ENUM_FIELDS")]
    replay = registry[registry.index("std::string serialize_mapping(const ButterflyConfig& c)"):
                      registry.index("std::string serialize_mapping(const PlanConfig& c)")]
    start = registry.index("        if (base.precision == ButterflyPrecision::Fp32) {",
                           registry.index("std::vector<ButterflyConfig> portable_butterfly_candidates("))
    end = registry.index("        if (on_demand_compilation_enabled() &&", start)
    factory = registry[start:end]
    dispatch = (ROOT / "src/fft_register_dispatch.cu").read_text()
    compiled = dispatch[dispatch.index("bool compiled_mapping("):dispatch.index("template <unsigned Local")]
    validator = dispatch[dispatch.index("void validate_register_tile("):dispatch.index("void launch_register_tile(")]
    inventory = dispatch[dispatch.index("std::vector<ButterflyConfig> register_tile_mappings()"):
                         dispatch.rindex("\n}")]
    reference = (ROOT / "src/reference.cpp").read_text()
    handoff = reference[reference.index("const char* stage_handoff_name("):reference.index("const char* output_order_name(")]
    compute = reference[reference.index("const char* compute_unit_name("):reference.index("const char* hierarchical_core_name(")]
    source = tmp_path / "contract.cpp"
    source.write_text(r'''
#include "plan_registry.hpp"
#include "resident_mapping.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <set>
#include <limits>
static bool research = false;
static unsigned prepared = 0;
int fake_device(int* device) { *device = 0; return cudaSuccess; }
int fake_properties(cudaDeviceProp* props, int) {
    props->major=8; props->minor=0; props->maxThreadsPerBlock=1024;
    props->sharedMemPerBlockOptin=164*1024;
    props->maxGridSize[0]=std::numeric_limits<int>::max();
    return cudaSuccess;
}
#define cudaGetDevice fake_device
#define cudaGetDeviceProperties fake_properties
namespace cuntt {
''' + handoff + compute + r'''
namespace detail {
using Json=nlohmann::json;
bool on_demand_compilation_enabled() { return research; }
void require_cufftdx_provider(unsigned, unsigned, unsigned, unsigned) {}
void prepare_register_module(const ButterflyConfig&) { ++prepared; }
''' + helpers + macros + replay + compiled + validator + r'''
#define register_tile_mappings raw_register_tile_mappings
''' + inventory + r'''
#undef register_tile_mappings
std::vector<ButterflyConfig> register_tile_mappings() {
    auto mappings=raw_register_tile_mappings();
    // Exercise providers whose inventory already carries physical metadata.
    for(auto& c:mappings) {
        c.segment_mappings={{FftCore::RegisterTile,LocalExchange::SharedMemory,c.prefix_threads,c.prefix_ept},
                            {FftCore::CufftDxBlock,LocalExchange::SharedMemory,c.suffix_threads,c.suffix_ept}};
        c.segment_mappings[0].codelet=c.prefix_codelet;
        c.execution_group_mappings=c.segment_mappings;
    }
    return mappings;
}
std::vector<ButterflyConfig> inventory_candidates(const ButterflyConfig& base) {
    std::vector<ButterflyConfig> candidates;
''' + factory + r'''
    return candidates;
}
}}
int main() {
    using namespace cuntt; using namespace cuntt::detail;
    const auto inventory=register_tile_mappings();
    assert(!inventory.empty() && prepared==0);
    unsigned radix_count=0;
    for(const auto& c:inventory) {
        assert(compiled_mapping(c));
        if(c.prefix_codelet=="radix4") { ++radix_count; assert(c.local_stages==8); }
    }
    assert(radix_count>0);
    for(const auto& c:inventory) if(c.prefix_codelet=="native") {
        auto legacy=Json::parse(serialize_mapping(c));
        legacy.erase("prefix_codelet"); legacy.erase("prefix_shared_layout");
        auto replay=c; replay.prefix_codelet="radix4"; replay.prefix_shared_layout="xor";
        apply_serialized_mapping(legacy.dump(),replay);
        assert(replay.prefix_codelet=="native" && replay.prefix_shared_layout=="linear");
        break;
    }
    for(bool online:{false,true}) {
        research=online;
        for(unsigned log_n:{16U,18U,20U,22U}) {
            ButterflyConfig base; base.op=ButterflyOperator::Fft; base.log_n=log_n; base.batch=3;
            const auto candidates=inventory_candidates(base);
            std::set<std::string> identities;
            unsigned expected=0;
            for(const auto& c:inventory) if(c.log_n==log_n)
                expected+=online && c.prefix_codelet=="native" ? (c.local_stages==8 ? 5 : 6) : 1;
            assert(candidates.size()==expected);
            for(const auto& c:candidates) {
                const auto serialized=serialize_mapping(c);
                assert(identities.insert(serialized).second);
                ButterflyConfig replay=base;
                apply_serialized_mapping(serialized,replay);
                assert(serialize_mapping(replay)==serialized);
                assert(replay.segment_mappings[0].codelet==c.prefix_codelet);
                assert(replay.execution_group_mappings[0].codelet==c.prefix_codelet);
                assert(replay.execution_group_mappings[1].codelet=="native");
                if(c.prefix_codelet=="radix4" && !online) assert(compiled_mapping(c));
                if(!online) validate_register_tile(replay);
            }
        }
    }
    assert(prepared==0);
    research=false;
    for(const auto& c:inventory) if(c.prefix_codelet=="radix4") {
        for(unsigned change=0;change<4;++change) {
            auto invalid=c;
            if(change==0) invalid.prefix_shared_layout="xor";
            if(change==1) invalid.precision=ButterflyPrecision::Fp64;
            if(change==2) invalid.local_stages=6;
            if(change==3) invalid.prefix_codelet_lanes=2;
            bool rejected=false;
            try { validate_register_tile(invalid); }
            catch(const std::invalid_argument&) { rejected=true; }
            assert(rejected);
        }
    }
}
''')
    executable = tmp_path / "contract"
    subprocess.run([compiler, "-std=c++17", "-O0", "-ffunction-sections", "-fdata-sections",
                    "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"), "-I" + str(includes),
                    "-I" + str(cuda), str(source), str(ROOT / "src/butterfly_reference.cpp"),
                    "-Wl,--gc-sections", "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=90)
    subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
