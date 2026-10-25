"""CPU regressions for explicit suffix implementation and group-local identity."""
import json
from pathlib import Path
import sys
import shutil
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibration_seeds
import calibration_space
import calibrate_local_hardware
import resident_mapping
from research_compile_requests import compile_request
import stage_service_model as service
from test_provider_cpp_contract import cpu_contract, _json_include


def mapping(core=None, chunk=2):
    result = dict(backend="online-reorder", fft_core="register-tile", local_stages=4,
                  stage_partition=[4, 12], prefix_threads=32, prefix_ept=4,
                  suffix_threads=256, suffix_ept=32, exchange_chunks=[0, chunk])
    if core is not None:
        result["suffix_core"] = core
    return result


def point(value):
    return calibration_space.mapping_point(
        dict(operator="fft", precision="fp64", logN=16, batch=64), value)


def test_native_request_and_default_cache_identity():
    default, explicit, native = [point(mapping(core)) for core in (None, "cufftdx-block", "native")]
    assert compile_request(default) == compile_request(explicit)
    assert compile_request(native)["suffix_core"] == "native"
    assert calibration_seeds.canonical_mapping(mapping()) == calibration_seeds.canonical_mapping(mapping("cufftdx-block"))
    assert calibration_seeds.canonical_mapping(mapping()) != calibration_seeds.canonical_mapping(mapping("native"))
    key = calibrate_local_hardware._stage_template_key
    assert key(default) == key(explicit)
    assert key(default) != key(native)
    assert calibration_space._strategy_identity(mapping(), []) != calibration_space._strategy_identity(mapping("native"), [])
    calibration_seeds.validate_mapping(mapping("cufftdx-block"), mapping())
    for requested, resolved in ((mapping(), mapping("native")), (mapping("native"), mapping())):
        with pytest.raises(ValueError, match="suffix_core"):
            calibration_seeds.validate_mapping(requested, resolved)


def test_native_cannot_be_ignored_by_policy_or_unrelated_lowering():
    with pytest.raises(ValueError, match="on-demand"):
        compile_request(point(mapping("native")), "precompiled")
    for change in ({"backend": "temporal-tile"}, {"fft_core": "scalar"}, {"operator": "fwht"}):
        value = point({**mapping("native"), **change})
        value.update({k: v for k, v in change.items() if k == "operator"})
        with pytest.raises(ValueError, match="suffix_core"):
            compile_request(value)


def test_projection_reserves_full_native_tile_independent_of_exchange_chunk():
    for chunk in (0, 1, 2, 16):
        sample = calibration_space.predicted_sample(point(mapping("native", chunk)))
        groups = sample["execution_groups_json"]
        assert [group["core"] for group in groups] == ["register-tile", "register-tile"]
        assert groups[1]["codelet"] == "native"
        assert groups[1]["live_shared_bytes"] == 4096 * 2 * 16
        assert groups[1]["exchange_chunk"] == chunk
    request = compile_request(point(mapping("native")))
    assert "backend" not in request
    assert resident_mapping.register_geometry(request)["suffix_core"] == "native"


def test_physical_group_conflicts_are_rejected():
    for field in ("segment_mappings", "execution_group_mappings"):
        value = mapping("native")
        value[field] = [{"core": "register-tile", "codelet": "native"},
                        {"core": "cufftdx-block", "codelet": "native"}]
        with pytest.raises(ValueError, match="disagrees"):
            compile_request(point(value))
        value[field][1]["core"] = "register-tile"
        assert compile_request(point(value))["suffix_core"] == "native"


def test_accuracy_metadata_and_service_key_are_suffix_local():
    default = calibration_space.predicted_sample(point(mapping()))
    native = calibration_space.predicted_sample(point(mapping("native")))
    old_groups, native_groups = default["execution_groups_json"], native["execution_groups_json"]
    assert service.curve_key(default, old_groups[0]) == service.curve_key(native, native_groups[0])
    assert service.curve_key(default, old_groups[1]) != service.curve_key(native, native_groups[1])
    state = service._selected_mapping_state(mapping("native"), native, native_groups[1])
    assert state["suffix_core"] == "native"
    assert "suffix_core" not in service._selected_mapping_state(mapping("native"), native, native_groups[0])
    # Compiler-resource metadata cannot alias the old cuFFTDx implementation.
    assert service.shape_key(default, old_groups[1]) != service.shape_key(native, native_groups[1])


def test_experimental_native_suffix_is_explicit_only(cpu_contract):
    run, _, _, _ = cpu_contract
    for shared_bytes in (164*1024, 65536):
        result = run("candidates", 16, "fp64", 800, shared_bytes)
        assert result["bad_suffix"] == 0
        assert result["native_suffix"] == 0
        assert result["register_ept16"] > 0
        assert result["factor_native"] > 0


def test_real_cpp_mapping_serialization_and_replay(tmpdir):
    tmp_path = Path(str(tmpdir))
    root = Path(__file__).resolve().parents[1]
    compiler, includes = shutil.which("c++"), _json_include()
    cuda = Path("/usr/local/cuda/include")
    if not compiler or not includes or not cuda.is_dir():
        pytest.skip("CPU compiler, CUDA host headers and JSON headers required")
    source = (root / "src/plan_registry.cpp").read_text()
    helpers = source[source.index("std::string string_value("):source.index("std::string registry_path()")]
    macros = source[source.index("#define BUTTERFLY_ENUM_FIELDS"):source.index("#define NTT_ENUM_FIELDS")]
    functions = source[source.index("std::string serialize_mapping(const ButterflyConfig& c)"):
                       source.index("std::string serialize_mapping(const PlanConfig& c)")]
    reference = (root / "src/reference.cpp").read_text()
    handoff = reference[reference.index("const char* stage_handoff_name("):
                        reference.index("const char* output_order_name(")]
    compute = reference[reference.index("const char* compute_unit_name("):
                        reference.index("const char* hierarchical_core_name(")]
    cpp = tmp_path / "mapping.cpp"
    cpp.write_text('''
#include "plan_registry.hpp"
#include "resident_mapping.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <algorithm>
#include <iostream>
namespace cuntt {
''' + handoff + compute + '''
namespace detail { using Json=nlohmann::json;
''' + helpers + macros + functions + '''
}}
int main() {
 using namespace cuntt; using namespace cuntt::detail; using Json=nlohmann::json;
 ButterflyConfig c; c.op=ButterflyOperator::Fft; c.backend=ButterflyBackend::OnlineReorder;
 c.fft_core=FftCore::RegisterTile; c.stage_partition={4,12}; c.suffix_core="native";
 const auto native=serialize_mapping(c); ButterflyConfig replay=c;
 apply_serialized_mapping(native,replay);
 Json result={{"native_roundtrip",serialize_mapping(replay)==native}};
 auto legacy=Json::parse(native); legacy.erase("suffix_core");
 apply_serialized_mapping(legacy.dump(),replay);
 result["legacy_default"]=replay.suffix_core=="cufftdx-block";
 result["default_omitted"]=!Json::parse(serialize_mapping(replay)).contains("suffix_core");
 auto compact=Json::parse(native);
 compact["segment_mappings"]=Json::array({Json{{"exchange","shared"}},Json{{"exchange","shared"}}});
 apply_serialized_mapping(compact.dump(),replay);
 result["derived_native_core"]=replay.segment_mappings[1].core==FftCore::RegisterTile;
 compact["segment_mappings"][1]["core"]="cufftdx-block";
 try { apply_serialized_mapping(compact.dump(),replay); result["conflict_rejected"]=false; }
 catch(const std::invalid_argument&) {result["conflict_rejected"]=true;}
 auto csv=Json::parse(native); csv.erase("segment_mappings"); csv.erase("execution_group_mappings"); csv.erase("boundaries");
 csv["group_threads"]="32x256"; csv["group_ept"]="4x32";
 csv["group_cores"]="register-tile:cufftdx-block";
 try { apply_serialized_mapping(csv.dump(),replay); result["csv_conflict_rejected"]=false; }
 catch(const std::invalid_argument&) {result["csv_conflict_rejected"]=true;}
 std::cout<<result.dump();
}
''')
    executable = tmp_path / "mapping"
    subprocess.run([compiler, "-std=c++17", "-O0", "-ffunction-sections", "-fdata-sections",
                    "-I" + str(root / "include"), "-I" + str(root / "src"), "-I" + str(includes),
                    "-I" + str(cuda),
                    str(cpp), str(root / "src/butterfly_reference.cpp"), "-Wl,--gc-sections",
                    "-o", str(executable)], check=True, capture_output=True, text=True, timeout=90)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=10)
    assert all(json.loads(result.stdout).values())
