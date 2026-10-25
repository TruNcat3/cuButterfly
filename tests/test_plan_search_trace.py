"""CPU-only checks for optional planner diagnostics; no CUDA initialization."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = r'''
#include "plan_search_trace.hpp"
#include <stdexcept>
#include <string>
#include <cstdio>

int main(int argc, char** argv) {
    int cleanup_count = 0;
    cuntt::detail::PlanSearchTrace trace("cpu-test");
    trace.context([] { return nlohmann::json{{"candidate_index", 7}}; });
    {
        cuntt::detail::PlanSearchTrace::Phase phase(&trace, "construction");
    }
    try {
        cuntt::detail::PlanSearchTrace::Phase phase(&trace, "reference_check");
        throw std::runtime_error("candidate failure");
    } catch (const std::runtime_error&) {}
    {
        auto cleanup = cuntt::detail::PlanSearchTrace::on_exit(&trace, "destruction", [&] { ++cleanup_count; });
    }
    if (argc > 1 && std::string(argv[1]) == "serialize-failure") {
        trace.event("failure", []() -> nlohmann::json { throw std::runtime_error("serialization failure"); });
    }
    trace.event("result", [] { return nlohmann::json{{"kernel_ms", 0.125}}; });
    std::printf("cleanup=%d\n", cleanup_count);
    return cleanup_count == 1 ? 0 : 1;
}
'''


@pytest.fixture(scope="module")
def trace_binary(tmpdir_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    includes = sorted(ROOT.glob("build*/_deps/nlohmann_json-src/single_include"))
    if not includes:
        pytest.skip("local nlohmann_json headers unavailable")
    directory = Path(str(tmpdir_factory.mktemp("plan-search-trace")))
    source = directory / "trace.cpp"
    source.write_text(SOURCE)
    binary = directory / "trace"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                    f"-I{ROOT / 'src'}", f"-I{includes[0]}", str(source), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


def run_trace(binary, destination, *args):
    environment = os.environ.copy()
    environment.pop("CUBUTTERFLY_PLAN_TRACE", None)
    if destination is not None:
        environment["CUBUTTERFLY_PLAN_TRACE"] = str(destination)
    result = subprocess.run([str(binary), *args], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "cleanup=1\n"
    return result


@pytest.mark.parametrize("destination", [None, "", "0"])
def test_trace_disabled(trace_binary, destination):
    assert run_trace(trace_binary, destination).stderr == ""


def test_trace_jsonl_and_append(trace_binary, tmpdir):
    destination = Path(str(tmpdir)) / "trace.jsonl"
    for _ in range(2):
        assert run_trace(trace_binary, destination).stderr == ""
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(rows) == 8
    for index in (0, 4):
        run = rows[index:index + 4]
        assert [row["phase"] for row in run] == ["construction", "reference_check", "destruction", "result"]
        assert [row["outcome"] for row in run[:3]] == ["complete", "exception", "complete"]
        assert all(row["wall_ms"] >= 0 for row in run[:3])
        assert all(row["candidate_index"] == 7 for row in run)
        assert all(row["schema"] == "cubutterfly-plan-search-trace-v1" for row in run)
        assert len({row["search_id"] for row in run}) == 1
        assert run[-1]["kernel_ms"] == 0.125


def test_trace_stderr(trace_binary):
    rows = [json.loads(line) for line in run_trace(trace_binary, "1").stderr.splitlines()]
    assert len(rows) == 4


def test_trace_open_failure_is_reported(trace_binary, tmpdir):
    result = run_trace(trace_binary, Path(str(tmpdir)) / "missing" / "trace.jsonl")
    assert "cannot open trace destination" in result.stderr


def test_trace_write_failure_is_reported(trace_binary):
    if not Path("/dev/full").exists():
        pytest.skip("/dev/full unavailable")
    result = run_trace(trace_binary, "/dev/full")
    assert "cannot write trace destination" in result.stderr


def test_trace_serialization_failure_is_reported(trace_binary, tmpdir):
    destination = Path(str(tmpdir)) / "trace.jsonl"
    result = run_trace(trace_binary, destination, "serialize-failure")
    assert "cannot serialize trace event" in result.stderr
    assert len(destination.read_text().splitlines()) == 3
