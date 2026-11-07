"""CPU integration of the real provider wrapper and candidate function body.

Only hardware discovery and linked, precompiled inventories are stubbed. The
candidate function and capacity partition helper are extracted verbatim from
plan_registry.cpp; partition and organization algorithms use their real headers.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cufftdx_capabilities import load_database, render_header


def _json_include():
    candidates = [Path("/usr/include"), ROOT / "external/json/single_include"]
    candidates += list(ROOT.glob("build*/_deps/nlohmann_json-src/single_include"))
    # An isolated source checkout may reuse existing build dependencies read-only.
    candidates += list((ROOT.parent / "cuButterfly").glob(
        "build*/_deps/nlohmann_json-src/single_include"))
    return next((path for path in candidates if (path / "nlohmann/json.hpp").is_file()), None)


@pytest.fixture(scope="module")
def cpu_contract(tmpdir_factory):
    compiler = shutil.which("c++")
    includes = _json_include()
    installed_mathdx = ROOT / "external/mathdx/nvidia/mathdx"
    cuda = Path("/usr/local/cuda/include")
    if not compiler or not includes or not installed_mathdx.is_dir() or not cuda.is_dir():
        pytest.skip("CPU compiler, CUDA host headers, JSON headers and MathDx required")
    work = Path(str(tmpdir_factory.mktemp("provider-cpp-contract")))
    mathdx = work / "private-mathdx"
    installed_db = load_database(installed_mathdx)
    assert not installed_db.diagnostics
    for source in installed_db.sources:
        source = Path(source)
        target = mathdx / source.relative_to(installed_mathdx.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    db = load_database(mathdx)
    assert not db.diagnostics
    (work / "cufftdx_provider_capabilities.hpp").write_text(render_header(db))
    (work / "jit_config.hpp").write_text(
        '#define CUBUTTERFLY_JIT_MATHDX ' + json.dumps(str(mathdx.resolve())) + "\n")
    source = (ROOT / "src/plan_registry.cpp").read_text()
    start = source.index("std::vector<ButterflyConfig> portable_butterfly_candidates(")
    end = source.index("std::vector<PlanConfig> portable_ntt_candidates(", start)
    actual_candidates = source[start:end]
    start = source.index("std::vector<std::vector<std::uint32_t>> capacity_partitions(")
    end = source.index("std::string string_value(", start)
    actual_capacity = source[start:end]
    c_api = (ROOT / "src/c_api.cpp").read_text()
    start = c_api.index("thread_local std::string last_error;")
    end = c_api.index("void require(bool condition, const char* message)", start)
    actual_protect = c_api[start:end]
    translation_unit = r'''
#include "cubutterfly/cubutterfly.h"
#include "plan_registry.hpp"
#include "cufftdx_provider.hpp"
#include "fft_local_core_contract.hpp"
#include "temporal_tile_contract.hpp"
#include "online_reorder_contract.hpp"
#include "shared_tile_contract.hpp"
#include "fwht_contract.hpp"
#include "partition_space.hpp"
#include "shared_organization_candidates.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
using Json = nlohmann::json;
static unsigned test_sm = 800;
static unsigned test_shared = 164*1024;
namespace cuntt {
DeviceInfo current_device_info() {
    DeviceInfo info{};
    info.compute_major=test_sm/100; info.compute_minor=(test_sm%100)/10;
    return info;
}
const char* butterfly_precision_name(ButterflyPrecision precision) noexcept {
    switch (precision) {
    case ButterflyPrecision::Fp16: return "fp16";
    case ButterflyPrecision::Bf16: return "bf16";
    case ButterflyPrecision::Fp64: return "fp64";
    default: return "fp32";
    }
}
const char* butterfly_accumulation_name(ButterflyAccumulation accumulation) noexcept {
    return accumulation==ButterflyAccumulation::Native ? "native" : "fp32";
}
HardwareResourceModel query_hardware_resource_model() {
    HardwareResourceModel h;
    h.compute_major=test_sm/100; h.compute_minor=(test_sm%100)/10;
    h.max_threads_per_block=1024; h.shared_bytes_per_block=test_shared;
    return h;
}
std::vector<FftOnlineProcessingUnit> fft_online_processing_units() { return {}; }
namespace detail {
bool on_demand_compilation_enabled() { return true; }
std::vector<ButterflyConfig> platform_candidates(const ButterflyConfig&) { return {}; }
std::vector<ButterflyConfig> register_tile_mappings() { return {}; }
std::vector<std::uint32_t> generated_register_fwht_thread_options(std::uint32_t) { return {}; }
std::vector<std::uint32_t> generated_register_structured_thread_options(std::uint32_t) { return {}; }
std::string serialize_mapping(const ButterflyConfig&) { throw std::logic_error("unused linked inventory"); }
void apply_serialized_mapping(const std::string&, ButterflyConfig&) { throw std::logic_error("unused linked inventory"); }
''' + actual_capacity + actual_candidates + "\n}}\n" + actual_protect + r'''
int main(int argc, char** argv) {
    using namespace cuntt;
    using namespace cuntt::detail;
    if (std::string(argv[1]) == "status") {
        auto missing=protect([] { require_cufftdx_provider(800,64,1024,32); });
        const auto missing_message=last_error;
        auto invalid=protect([] { throw std::invalid_argument("invalid FFT geometry"); });
        std::cout << Json{{"missing",missing},{"invalid",invalid},
            {"not_supported",CUBUTTERFLY_STATUS_NOT_SUPPORTED},
            {"invalid_value",CUBUTTERFLY_STATUS_INVALID_VALUE},
            {"message",missing_message}}.dump();
        return 0;
    }
    if (std::string(argv[1]) == "require") {
        Json result;
        result["matches"] = cufftdx_provider_matches_runtime();
        result["fingerprint"] = cufftdx_provider_fingerprint;
        for (unsigned size : {32U,1024U,4096U}) for (unsigned ept : {16U,32U}) {
            const auto key=std::to_string(size)+":"+std::to_string(ept);
            try { require_cufftdx_provider(std::stoul(argv[2]),64,size,ept); result[key]="accepted"; }
            catch(const std::invalid_argument& error) { result[key]=error.what(); }
        }
        std::cout << result.dump();
        return 0;
    }
    test_sm=std::stoul(argv[4]);
    if(argc>5) test_shared=std::stoul(argv[5]);
    ButterflyConfig request;
    request.op=ButterflyOperator::Fft;
    request.precision=std::string(argv[3])=="fp64" ? ButterflyPrecision::Fp64 : ButterflyPrecision::Fp32;
    request.log_n=std::stoul(argv[2]); request.batch=1;
    std::string audit;
    auto candidates=portable_butterfly_candidates(request,&audit);
    Json result={{"audit",Json::parse(audit)},{"candidate_count",candidates.size()},
                 {"register_dx",0},{"completed_ept_candidates",0},{"suffix_epts",Json::object()},
                 {"register_ept32",0},{"bad_suffix",0},
                 {"register_ept16",0},{"factor_native",0},{"factor_dx",0},{"native_suffix",0},
                 {"native_missing_1024",0},{"native_missing_4096",0},{"native_over_capacity",0}};
    for(const auto& c:candidates) {
        if(c.backend==ButterflyBackend::OnlineReorder && c.fft_core==FftCore::RegisterTile) {
            if(c.suffix_core=="cufftdx-block") {
                result["register_dx"]=result["register_dx"].get<unsigned>()+1;
                if(c.suffix_ept!=16 && c.suffix_ept!=32 && c.suffix_ept!=8 && c.suffix_ept!=4)
                    result["completed_ept_candidates"]=result["completed_ept_candidates"].get<unsigned>()+1;
                auto& epts=result["suffix_epts"][std::to_string(1U<<(c.log_n-c.local_stages))];
                if(epts.is_null()) epts=Json::array();
                if(std::find(epts.begin(),epts.end(),Json(c.suffix_ept))==epts.end()) epts.push_back(c.suffix_ept);
            }
            if(c.suffix_core=="native") {
                result["native_suffix"]=result["native_suffix"].get<unsigned>()+1;
                const auto n=1U<<(c.log_n-c.local_stages);
                const auto columns=c.suffix_threads*c.suffix_ept/n;
                const auto bytes=std::uint64_t(n)*columns*(c.precision==ButterflyPrecision::Fp64?16:8);
                if(bytes>test_shared) result["native_over_capacity"]=result["native_over_capacity"].get<unsigned>()+1;
                if(c.suffix_ept==32 && columns==2 && (n==1024 || n==4096)) {
                    const auto key=n==1024?"native_missing_1024":"native_missing_4096";
                    result[key]=result[key].get<unsigned>()+1;
                }
            }
            if(c.suffix_ept==32) {
                result["register_ept32"]=result["register_ept32"].get<unsigned>()+1;
                const auto suffix=c.log_n-c.local_stages;
                if((suffix==10 || suffix==12) && c.suffix_core=="cufftdx-block") result["bad_suffix"]=result["bad_suffix"].get<unsigned>()+1;
            }
            if(c.suffix_ept==16) result["register_ept16"]=result["register_ept16"].get<unsigned>()+1;
        }
        if(c.backend==ButterflyBackend::FactorStreamed) {
            const auto key=c.fft_core==FftCore::RegisterTile ? "factor_native" : "factor_dx";
            result[key]=result[key].get<unsigned>()+1;
        }
    }
    std::cout << result.dump();
}
'''
    cpp = work / "contract.cpp"
    cpp.write_text(translation_unit)
    executable = work / "contract"
    subprocess.run([compiler, "-std=c++17", "-O0", "-I"+str(work),
                    "-I"+str(ROOT / "include"), "-I"+str(ROOT / "src"),
                    "-I"+str(cuda), "-I"+str(includes), str(cpp), "-o", str(executable)],
                   check=True, capture_output=True, text=True, timeout=120)
    # Compare unrelated inventory dimensions with the previous precision lookup.
    # This isolates provider classification from changes to partition seeding.
    legacy_start = translation_unit.index("        const auto arithmetic_bits = cufftdx_arithmetic_precision_bits(original);")
    legacy_end = translation_unit.index("        const auto status = provider_table_matches", legacy_start)
    legacy_source = (translation_unit[:legacy_start] +
        "        const auto bits = original.precision == ButterflyPrecision::Fp64 ? 64U : "
        "original.precision == ButterflyPrecision::Fp16 ? 16U : 32U;\n" +
        translation_unit[legacy_end:])
    legacy_cpp = work / "legacy_precision.cpp"
    legacy_cpp.write_text(legacy_source)
    legacy_executable = work / "legacy_precision"
    subprocess.run([compiler, "-std=c++17", "-O0", "-I"+str(work),
                    "-I"+str(ROOT / "include"), "-I"+str(ROOT / "src"),
                    "-I"+str(cuda), "-I"+str(includes), str(legacy_cpp), "-o", str(legacy_executable)],
                   check=True, capture_output=True, text=True, timeout=120)
    def run(*args, override=None, legacy_precision=False):
        environment = dict(os.environ, CUBUTTERFLY_PARTITION_BUDGET="1")
        environment.pop("CUBUTTERFLY_MATHDX_ROOT", None)
        if override is not None:
            environment["CUBUTTERFLY_MATHDX_ROOT"] = str(override)
        result = subprocess.run([str(legacy_executable if legacy_precision else executable), *map(str, args)], env=environment,
                                check=True, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    return run, mathdx, work, db.metadata()["fingerprint"]


def test_require_classifies_missing_provider_not_invalid_transform(cpu_contract):
    run, _, _, fingerprint = cpu_contract
    result = run("require", 800)
    assert result["matches"] is True
    assert result["fingerprint"] == fingerprint
    for size in (1024, 4096):
        assert result[f"{size}:16"] == "accepted"
        message = result[f"{size}:32"]
        assert "provider-implementation-missing" in message
        assert "FFT request remains valid" in message
        assert f"local N={size} EPT=32" in message
    assert result["32:32"] == "accepted"


def test_unknown_sm_and_runtime_provider_override_defer(cpu_contract):
    run, mathdx, work, _ = cpu_contract
    unknown = run("require", 810)
    assert unknown["1024:32"] == "accepted"
    matching = run("require", 800, override=mathdx.resolve())
    assert matching["matches"] is True
    assert "provider-implementation-missing" in matching["1024:32"]
    different = work / "different-mathdx"
    different.mkdir(exist_ok=True)
    mismatch = run("require", 800, override=different)
    assert mismatch["matches"] is False
    assert mismatch["1024:32"] == "accepted"
    assert mismatch["4096:32"] == "accepted"


def test_actual_c_api_protect_distinguishes_provider_from_invalid_request(cpu_contract):
    run, _, _, _ = cpu_contract
    result = run("status")
    assert result["missing"] == result["not_supported"]
    assert result["invalid"] == result["invalid_value"]
    assert "provider-implementation-missing" in result["message"]
    assert "FFT request remains valid" in result["message"]


def test_same_path_provider_upgrade_or_missing_source_disables_old_table(cpu_contract):
    run, mathdx, _, _ = cpu_contract
    source = mathdx / "include/cufftdx/include/database/records/800/database_fp64_fwd.hpp.inc"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n// changed at the same installation path\n")
        changed = run("require", 800)
        assert changed["matches"] is False
        assert changed["1024:32"] == "accepted"
        assert changed["4096:32"] == "accepted"
        source.unlink()
        missing = run("require", 800)
        assert missing["matches"] is False
        assert missing["1024:32"] == "accepted"
    finally:
        source.write_bytes(original)
    assert run("require", 800)["matches"] is True


@pytest.mark.parametrize("log_n", [16, 18])
def test_actual_fp64_register_candidates_exclude_missing_suffix(cpu_contract, log_n):
    run, _, _, _ = cpu_contract
    result = run("candidates", log_n, "fp64", 800)
    assert result["bad_suffix"] == 0
    assert result["register_ept16"] > 0
    legacy = run("candidates", log_n, "fp64", 800, legacy_precision=True)
    # Native factor availability depends on bounded partition seeds and their
    # capacity envelope, independently of the cuFFTDx precision classifier.
    assert result["factor_native"] == legacy["factor_native"]
    assert result["factor_dx"] == legacy["factor_dx"]
    assert result["native_suffix"] == 0
    assert 2 in result["suffix_epts"]["1024"]
    assert set(result["suffix_epts"]["1024"]) == {2, 4, 8, 16}
    assert set(result["suffix_epts"]["4096"]) == {4, 8, 16}
    assert result["completed_ept_candidates"] > 0
    _, mathdx, _, _ = cpu_contract
    database = load_database(mathdx)
    for size, epts in result["suffix_epts"].items():
        for ept in epts:
            assert database.check_both(800, "fp64", int(size), ept) == "supported"
    checks = result["audit"]["checks"]
    rejected = checks["800:64:1024:32"]
    assert rejected["classification"] == "provider-implementation-missing"
    assert rejected["invalid_transform"] is False
    assert rejected["action"] == "exclude-provider-combination"
    assert rejected["directions"] == ["forward", "inverse"]
    if log_n == 18:
        assert checks["800:64:4096:32"]["classification"] == "provider-implementation-missing"


def test_actual_candidates_keep_other_precision_ept32_and_unknown(cpu_contract):
    run, _, work, _ = cpu_contract
    fp32 = run("candidates", 16, "fp32", 800)
    assert fp32["bad_suffix"] > 0
    assert fp32["native_suffix"] == 0
    legacy = run("candidates", 16, "fp32", 800, legacy_precision=True)
    assert fp32["factor_native"] > 0
    assert fp32["factor_native"] == legacy["factor_native"]
    assert fp32["factor_dx"] == legacy["factor_dx"]
    assert fp32["suffix_epts"]["1024"][:4] == [16, 32, 8, 4]
    assert 2 in fp32["suffix_epts"]["1024"]
    # FP64 N32/EPT32 is checked by require above. It cannot appear in the
    # register suffix inventory: <=4 columns gives <32 physical threads.
    unknown = run("candidates", 16, "fp64", 810)
    assert unknown["bad_suffix"] > 0
    legacy_unknown = run("candidates", 16, "fp64", 810, legacy_precision=True)
    assert unknown["factor_native"] == legacy_unknown["factor_native"]
    assert unknown["native_suffix"] == 0
    assert 2 in unknown["suffix_epts"]["1024"]
    assert 128 in unknown["suffix_epts"]["1024"]
    check = unknown["audit"]["checks"]["810:64:1024:32"]
    assert check["classification"] == "provider-capability-unknown"
    assert check["action"] == "defer-to-compilation"
    mismatch = run("candidates", 16, "fp64", 800, override=work)
    assert mismatch["bad_suffix"] > 0
    assert mismatch["audit"]["provider_table_matches_runtime"] is False
