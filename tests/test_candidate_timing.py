"""CPU fault injection for the production CUDA candidate timing helper."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    "create-start", "create-stop", "record-start", "record-stop", "synchronize", "elapsed",
    *[f"execute-{call}" for call in range(1, 8)],
    "zero", "nan", "negative", "infinity", "success",
]


@pytest.fixture(scope="module")
def timing_binary(tmpdir_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/c_api.cpp").read_text()
    start = source.index("template <typename Execute>\ndouble measure_cuda_candidate(")
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    function = source[start:end]
    directory = Path(str(tmpdir_factory.mktemp("candidate-timing")))
    cpp = directory / "timing.cpp"
    cpp.write_text(r"""
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

struct Event { int id; };
using cudaEvent_t = Event*;
using cudaStream_t = void*;
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
int api_calls = 0, fail_api_call = 0, created = 0, destroyed = 0;
bool live[3] = {};
float duration = 25.0F;
cudaStream_t expected_stream = reinterpret_cast<void*>(42);

cudaError_t result() { return ++api_calls == fail_api_call ? 1 : cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected error"; }
cudaError_t cudaEventCreate(cudaEvent_t* event) {
    auto status = result();
    if (status != cudaSuccess) return status;
    const int id = ++created;
    assert(id <= 2 && !live[id]);
    live[id] = true;
    *event = new Event{id};
    return status;
}
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    assert(event && live[event->id] && stream == expected_stream);
    assert(event->id == (api_calls == 2 ? 1 : 2));
    return result();
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    assert(event && event->id == 2 && live[2]);
    return result();
}
cudaError_t cudaEventElapsedTime(float* milliseconds, cudaEvent_t start, cudaEvent_t stop) {
    assert(start && stop && start->id == 1 && stop->id == 2 && live[1] && live[2]);
    auto status = result();
    if (status == cudaSuccess) *milliseconds = duration;
    return status;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
    assert(event && live[event->id]);
    live[event->id] = false;
    ++destroyed;
    delete event;
    return cudaSuccess;
}
""" + function + r"""
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string scenario(argv[1]);
    const char* fault_names[] = {
        "create-start", "create-stop", "record-start", "record-stop", "synchronize", "elapsed"
    };
    for (int i = 0; i < 6; ++i)
        if (scenario == fault_names[i]) fail_api_call = i + 1;
    int fail_execute_call = 0;
    if (scenario.rfind("execute-", 0) == 0) fail_execute_call = std::stoi(scenario.substr(8));
    if (scenario == "zero") duration = 0.0F;
    if (scenario == "nan") duration = std::numeric_limits<float>::quiet_NaN();
    if (scenario == "negative") duration = -1.0F;
    if (scenario == "infinity") duration = std::numeric_limits<float>::infinity();
    int executions = 0;
    bool returned = false, threw = false;
    double measured = -1.0;
    try {
        measured = measure_cuda_candidate(expected_stream, [&] {
            if (++executions == fail_execute_call) throw std::runtime_error("execute failed");
        });
        returned = true;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(!live[1] && !live[2]);
    assert(destroyed == created);
    const int expected_created = fail_api_call == 1 || (fail_execute_call > 0 && fail_execute_call <= 2)
        ? 0 : fail_api_call == 2 ? 1 : 2;
    assert(created == expected_created);
    if (scenario == "success") {
        assert(returned && !threw && measured == 5.0);
        assert(executions == 7 && api_calls == 6);
    } else {
        assert(threw && !returned && measured == -1.0);
        if (fail_api_call) assert(api_calls == fail_api_call);
        if (fail_execute_call) assert(executions == fail_execute_call);
    }
}
""")
    binary = directory / "timing"
    subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary),
    ], check=True)
    return binary


@pytest.mark.parametrize("scenario", CASES)
def test_candidate_timing_rejects_failures_and_releases_events(timing_binary, scenario):
    subprocess.run([str(timing_binary), scenario], check=True)
