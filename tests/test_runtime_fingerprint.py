"""CPU-only checks for policy identity and generated-input fingerprinting."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _function(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.mark.parametrize("default", ["precompiled", "on-demand"])
def test_compile_policy_and_runtime_identity(tmpdir, default):
    tmp_path = Path(str(tmpdir))
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    header = (ROOT / "src/jit_module.hpp").read_text()
    enum = re.search(r"enum class CompilePolicy \{[^}]+\};", header).group()
    jit = (ROOT / "src/jit_module.cpp").read_text()
    registry = (ROOT / "src/plan_registry.cpp").read_text()
    functions = "\n".join([
        _function(jit, "CompilePolicy compile_policy() noexcept"),
        _function(jit, "bool on_demand_compilation_enabled()"),
        _function(jit, "bool specialize_shared_templates()"),
        _function(registry, "const char* runtime_fingerprint() noexcept"),
        _function(registry, "const char* local_selector_fingerprint() noexcept"),
    ])
    source = tmp_path / "policy.cpp"
    source.write_text("""
#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#define CUBUTTERFLY_RUNTIME_FINGERPRINT "fixture"
#define CUBUTTERFLY_LOCAL_SELECTOR_FINGERPRINT "selector-fixture"
""" + f'#define CUBUTTERFLY_DEFAULT_CODEGEN_MODE "{default}"\n' + enum + functions + """
int main() {
    assert(std::string(local_selector_fingerprint()) == "selector-fixture");
    for (const auto* mode : {"precompiled", "auto", "on-demand", "research"}) {
        setenv("CUBUTTERFLY_COMPILE_MODE", mode, 1);
        assert(std::string(runtime_fingerprint()) == std::string("fixture:") + mode);
        assert(on_demand_compilation_enabled() == (std::string(mode) != "precompiled"));
        assert(specialize_shared_templates() == (std::string(mode) == "research"));
    }
    unsetenv("CUBUTTERFLY_COMPILE_MODE");
    assert(std::string(runtime_fingerprint()) == std::string("fixture:") + CUBUTTERFLY_DEFAULT_CODEGEN_MODE);
    assert(on_demand_compilation_enabled() == (std::string(CUBUTTERFLY_DEFAULT_CODEGEN_MODE) != "precompiled"));
    for (const auto* invalid : {"", "typo", "Research"}) {
        setenv("CUBUTTERFLY_COMPILE_MODE", invalid, 1);
        assert(std::string(runtime_fingerprint()) == "fixture:invalid");
        bool on_demand_rejected = false, shared_rejected = false;
        try { on_demand_compilation_enabled(); } catch (const std::invalid_argument&) { on_demand_rejected = true; }
        try { specialize_shared_templates(); } catch (const std::invalid_argument&) { shared_rejected = true; }
        assert(on_demand_rejected && shared_rejected);
    }
}
""")
    binary = tmp_path / "policy"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_generated_contents_and_behavior_settings_change_identity(tmpdir):
    tmp_path = Path(str(tmpdir))
    inputs = [tmp_path / name for name in [
        "generated_hybrid_dataflow_config.hpp", "generated_runtime_selector.hpp",
    ]]
    local_selector = tmp_path / "generated_local_selector.hpp"
    local_selector.write_text("initial local selector\n")
    for path in inputs:
        path.write_text("initial contents\n")
    output = tmp_path / "build_fingerprint.hpp"

    def generate(base="behavior-options-a"):
        subprocess.run([
            "cmake", f"-DBASE_FINGERPRINT={base}",
            "-DINPUTS=" + ";".join(str(path) for path in inputs),
            f"-DLOCAL_SELECTOR={local_selector}",
            f"-DTEMPLATE={ROOT / 'src/build_fingerprint.hpp.in'}",
            f"-DOUTPUT={output}", "-P", str(ROOT / "cmake/runtime_fingerprint.cmake"),
        ], check=True)
        return dict(re.findall(r'#define (CUBUTTERFLY_\w+_FINGERPRINT) "([0-9a-f]+)"', output.read_text()))

    current = generate()
    assert set(current) == {"CUBUTTERFLY_RUNTIME_FINGERPRINT", "CUBUTTERFLY_LOCAL_SELECTOR_FINGERPRINT"}
    runtime_key, selector_key = "CUBUTTERFLY_RUNTIME_FINGERPRINT", "CUBUTTERFLY_LOCAL_SELECTOR_FINGERPRINT"
    local_selector.write_text("calibrated local selector\n")
    changed = generate()
    assert changed[runtime_key] == current[runtime_key]
    assert changed[selector_key] != current[selector_key]
    current = changed
    for path in inputs:
        path.write_text("calibrated or regenerated contents\n")
        changed = generate()
        assert changed[runtime_key] != current[runtime_key]
        assert changed[selector_key] == current[selector_key]
        current = changed
    timestamp = output.stat().st_mtime_ns
    assert generate() == current
    assert output.stat().st_mtime_ns == timestamp
    changed = generate("behavior-options-b")
    assert changed[runtime_key] != current[runtime_key]
    assert changed[selector_key] == current[selector_key]


def test_calibration_uses_build_compile_policy_and_rejects_invalid_override(tmp_path, monkeypatch):
    from calibrate_local_hardware import effective_compile_mode
    monkeypatch.delenv("CUBUTTERFLY_COMPILE_MODE", raising=False)
    assert effective_compile_mode(tmp_path) == "build-default-unknown"
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "jit_config.hpp").write_text('#define CUBUTTERFLY_DEFAULT_CODEGEN_MODE "on-demand"\n')
    assert effective_compile_mode(tmp_path) == "on-demand"
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "research")
    assert effective_compile_mode(tmp_path) == "research"
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "")
    with pytest.raises(ValueError, match="COMPILE_MODE"):
        effective_compile_mode(tmp_path)
