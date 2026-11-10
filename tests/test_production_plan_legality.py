"""CPU checks for production constructor legality gates and their ordering."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def constructor_source(filename, start, end):
    source = (ROOT / "src" / filename).read_text()
    return source[source.index(start):source.index(end, source.index(start))]


@pytest.fixture(scope="module")
def constructors():
    return {
        "butterfly": constructor_source("butterfly.cu", "explicit Impl(ButterflyConfig", "    ~Impl()"),
        "ntt": constructor_source("ntt.cu", "explicit Impl(PlanConfig", "    const PlanConfig& config()"),
    }


def test_every_constructor_certification_has_a_final_gate(constructors):
    for source in constructors.values():
        assert source.count("query_hardware_resource_model()") == 1
        assert source.count("dataflow_plan_.executable = true;") > 0
        for prefix in source.split("dataflow_plan_.executable = true;")[:-1]:
            assert prefix.rstrip().endswith("validate_dataflow();")
        assert source.index("validate_dataflow();") < source.index(".allocate(")
    # The compact path returns before the common end of the NTT constructor.
    compact = constructors["ntt"].split("if (config_.backend == Backend::CompactStage) {")[-1]
    assert compact.index("validate_dataflow();") < compact.index("return;")


def test_jit_calls_follow_candidate_gate_and_precede_final_gate(constructors):
    butterfly = constructors["butterfly"]
    for call in ("detail::prepare_register_module(config_);", "detail::validate_register_tile(config_);"):
        prefix = butterfly[:butterfly.index(call)]
        assert prefix.rstrip().endswith("require_valid_dataflow(candidate);")
    for source in constructors.values():
        assert source.rindex("detail::attach_module_resources(") < source.rindex("validate_dataflow();")
        assert source.index("validate_dataflow();") < source.index("detail::prepare_shared_module(")


@pytest.mark.parametrize("kind", ["butterfly", "ntt"])
def test_actual_gate_rejects_with_reason_and_preserves_unknown_resources(tmp_path, constructors, kind):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required")
    source = constructors[kind]
    name = "require_valid_dataflow" if kind == "butterfly" else "validate_dataflow"
    start = source.index("const auto " + name + " =")
    gate = source[start:source.index("\n        };", start) + len("\n        };")]
    call = f"{name}(dataflow_plan_);" if kind == "butterfly" else f"{name}();"
    harness = r'''
#include <cassert>
#include <stdexcept>
#include <string>
struct MixedDataflowPlan {
    bool executable = false, compiler_resources_known = false;
    std::string rejection_reason;
};
struct HardwareResourceModel { unsigned max_threads_per_block = 0; };
bool validate_plan(MixedDataflowPlan& plan, const HardwareResourceModel& hardware) {
    plan.executable = false;
    plan.rejection_reason = "threads exceed static profile";
    return hardware.max_threads_per_block == 1024;
}
int main() {
    HardwareResourceModel hardware{1024};
    MixedDataflowPlan dataflow_plan_;
''' + gate + call + r'''
    assert(!dataflow_plan_.executable && !dataflow_plan_.compiler_resources_known);
    hardware.max_threads_per_block = 32;
    try {
''' + call + r'''
        return 1;
    } catch (const std::invalid_argument& error) {
        assert(std::string(error.what()).find("threads exceed static profile") != std::string::npos);
    }
}
'''
    unit = tmp_path / "gate.cpp"
    unit.write_text(harness)
    executable = tmp_path / "gate"
    subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
