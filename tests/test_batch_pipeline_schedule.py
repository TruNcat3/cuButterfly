"""Exercise the real scheduler with a CPU-only CUDA dependency recorder."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

CUDA_STUB = r"""
#pragma once
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
using cudaStream_t = struct StreamHandle*;
using cudaEvent_t = struct EventHandle*;
using cudaError_t = int;
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive,
    cudaStreamCaptureStatusInvalidated
};
constexpr int cudaSuccess = 0;
constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
namespace fake {
struct Node {
    std::string kind;
    cudaStream_t stream;
    cudaEvent_t event;
    std::vector<int> parents;
    std::size_t group = 0, offset = 0, count = 0, slot = 0;
};
inline std::vector<Node> nodes;
inline std::map<cudaStream_t, int> stream_tail;
inline std::map<cudaEvent_t, int> event_tail;
inline std::vector<int> synchronized_event_records;
inline cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
inline int next_stream = 1, next_event = 1;
inline int append(const char* kind, cudaStream_t stream, cudaEvent_t event = nullptr) {
    Node node{kind, stream, event, {}};
    if (stream_tail.count(stream)) node.parents.push_back(stream_tail.at(stream));
    const int index = static_cast<int>(nodes.size());
    nodes.push_back(node);
    stream_tail[stream] = index;
    return index;
}
inline bool precedes(int before, int after) {
    if (before == after) return true;
    std::vector<int> pending{after};
    std::vector<bool> seen(nodes.size(), false);
    while (!pending.empty()) {
        const int index = pending.back();
        pending.pop_back();
        if (index == before) return true;
        if (seen[index]) continue;
        seen[index] = true;
        for (int parent : nodes[index].parents) pending.push_back(parent);
    }
    return false;
}
}
inline const char* cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
    *status = fake::capture_status;
    return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) {
    *s = reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(fake::next_stream++));
    return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
    *e = reinterpret_cast<cudaEvent_t>(static_cast<std::uintptr_t>(fake::next_event++));
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    fake::event_tail[event] = fake::append("record", stream, event);
    return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned) {
    const int node = fake::append("wait", stream, event);
    // An event never recorded is satisfied; waits snapshot the latest record.
    if (fake::event_tail.count(event))
        fake::nodes[node].parents.push_back(fake::event_tail.at(event));
    return cudaSuccess;
}
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    assert(fake::event_tail.count(event));
    fake::synchronized_event_records.push_back(fake::event_tail.at(event));
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
"""

FIXTURE = r"""
#include "batch_pipeline.cuh"
#include <cstdlib>
#include <iostream>

struct Invocation {
    int upload, download, start, end;
    cudaStream_t caller;
    std::size_t batch, tile;
    std::vector<std::vector<int>> launches;
};

Invocation submit(cuntt::detail::BatchPipeline& pipeline, std::size_t groups,
                  std::size_t batch, std::size_t tile, cudaStream_t caller) {
    Invocation call{};
    call.batch = batch; call.tile = tile; call.caller = caller;
    call.launches.resize((batch - 1) / tile + 1, std::vector<int>(groups, -1));
    call.upload = fake::append("upload", caller);
    call.start = static_cast<int>(fake::nodes.size());
    pipeline.enqueue(batch, tile, caller,
        [&](std::size_t group, std::size_t offset, std::size_t count,
            std::size_t slot, cudaStream_t stream) {
            assert(offset % tile == 0);
            const auto index = offset / tile;
            assert(index < call.launches.size() && group < groups);
            assert(call.launches[index][group] == -1);
            assert(count == std::min(tile, batch - offset));
            assert(slot == index % 2);
            const int node = fake::append("launch", stream);
            fake::nodes[node].group = group; fake::nodes[node].offset = offset;
            fake::nodes[node].count = count; fake::nodes[node].slot = slot;
            call.launches[index][group] = node;
        });
    call.end = static_cast<int>(fake::nodes.size());
    call.download = fake::append("download", caller);
    return call;
}

void verify(const Invocation& call, std::size_t groups, bool capturing = false) {
    const auto tiles = call.launches.size();
    if (!capturing) {
        assert(fake::nodes[call.start].kind == "wait");
        assert(fake::nodes[call.start].stream == call.caller);
    }
    std::size_t records = 0, waits = 0;
    std::vector<int> record_nodes;
    for (int i = call.start; i < call.end; ++i) {
        const auto& node = fake::nodes[i];
        if (node.kind == "record") { ++records; record_nodes.push_back(i); }
        if (node.kind == "wait") ++waits;
    }
    if (tiles == 1) {
        assert(records == (capturing ? 0U : 1U));
        assert(waits == (capturing ? 0U : 1U));
        if (!capturing) {
            assert(fake::nodes[call.end - 1].kind == "record");
            assert(fake::nodes[call.end - 1].stream == call.caller);
        }
        for (int launch : call.launches[0])
            assert(fake::nodes[launch].stream == call.caller);
    } else {
        const auto reuses = tiles > 2 ? tiles - 2 : 0;
        assert(records == 2 + (groups - 1) * (tiles + reuses));
        assert(waits == (capturing ? 2U : 3U) + (groups - 1) * (tiles + reuses));
        if (capturing) {
            for (std::size_t g = 0; g < groups; ++g) {
                assert(fake::nodes[call.launches[0][g]].stream != call.caller);
                for (std::size_t other = 0; other < g; ++other)
                    assert(fake::nodes[call.launches[0][g]].stream !=
                           fake::nodes[call.launches[0][other]].stream);
            }
            // The next producer can start before the preceding tile's final
            // consumer. A serialized caller-stream fallback fails this check.
            assert(!fake::precedes(call.launches[0].back(), call.launches[1][0]));
        }
        // Only group zero joins begin; all other groups join via ready.
        const int begin_record = record_nodes.front();
        std::size_t begin_waits = 0;
        for (int i = call.start; i < call.end; ++i)
            if (fake::nodes[i].kind == "wait" &&
                fake::nodes[i].event == fake::nodes[begin_record].event) {
                ++begin_waits;
                assert(fake::nodes[i].stream == fake::nodes[call.launches[0][0]].stream);
            }
        assert(begin_waits == 1);
        // Every internal record must serve a wait in this invocation. This
        // rejects consumed records for final slots with no future overwrite.
        for (std::size_t r = 1; r + 1 < record_nodes.size(); ++r) {
            const int record = record_nodes[r];
            bool used = false;
            for (int i = record + 1; i < call.end; ++i)
                if (fake::nodes[i].kind == "wait" &&
                    std::find(fake::nodes[i].parents.begin(), fake::nodes[i].parents.end(), record)
                        != fake::nodes[i].parents.end()) used = true;
            assert(used);
        }
        const auto old_records = 2 + 2 * tiles * (groups - 1);
        const auto old_waits = 2 + groups + (tiles + reuses) * (groups - 1);
        assert(records + waits < old_records + old_waits);
    }
    for (std::size_t t = 0; t < tiles; ++t)
        for (std::size_t g = 0; g < groups; ++g) {
            const int launch = call.launches[t][g];
            assert(launch >= 0);
            assert(fake::precedes(call.upload, launch));
            assert(fake::precedes(launch, call.download));
            if (g) assert(fake::precedes(call.launches[t][g - 1], launch));
            // The consumer must finish before its producer overwrites this
            // boundary's ring slot two tiles later.
            if (t >= 2 && g + 1 < groups)
                assert(fake::precedes(call.launches[t - 2][g + 1], launch));
        }
}

int main(int argc, char** argv) {
    assert(argc == 5 || argc == 6);
    const std::size_t groups = std::strtoul(argv[1], nullptr, 10);
    const std::size_t first_batch = std::strtoul(argv[2], nullptr, 10);
    const std::size_t second_batch = std::strtoul(argv[3], nullptr, 10);
    const std::size_t tile = std::strtoul(argv[4], nullptr, 10);
    if (argc == 6) {
        cuntt::detail::BatchPipeline pipeline(groups);
        // Seed an ordinary done event, so accidentally waiting on a graph-
        // external event cannot pass merely because it has never been set.
        const auto warmup = submit(pipeline, groups, 1, tile,
                                   reinterpret_cast<cudaStream_t>(200));
        const auto ordinary_done = fake::nodes[warmup.end - 1].event;
        const auto ordinary_done_record = fake::event_tail.at(ordinary_done);
        fake::capture_status = cudaStreamCaptureStatusActive;
        const auto captured_first = submit(pipeline, groups, first_batch, tile,
                                           reinterpret_cast<cudaStream_t>(100));
        const auto captured_second = submit(pipeline, groups, second_batch, tile,
                                            reinterpret_cast<cudaStream_t>(100));
        // This checks capture-time submission only, not CUDA graph replay.
        // Capture must leave ordinary done untouched and retain the private
        // stream DAG, joining repeated calls through their shared caller.
        verify(captured_first, groups, true);
        verify(captured_second, groups, true);
        assert(fake::event_tail.at(ordinary_done) == ordinary_done_record);
        for (int i = captured_first.start; i < captured_second.end; ++i)
            if (fake::nodes[i].kind == "wait" || fake::nodes[i].kind == "record")
                assert(fake::nodes[i].event != ordinary_done);
        for (const auto& old_tile : captured_first.launches)
            for (int before : old_tile)
                for (const auto& new_tile : captured_second.launches)
                    for (int after : new_tile) assert(fake::precedes(before, after));
        assert(fake::synchronized_event_records.empty());
        fake::capture_status = cudaStreamCaptureStatusNone;
        std::cout << "PASS capture submission dependencies\n";
        return 0;
    }
    Invocation first{}, second{};
    {
    cuntt::detail::BatchPipeline pipeline(groups);
    first = submit(pipeline, groups, first_batch, tile, reinterpret_cast<cudaStream_t>(100));
    second = submit(pipeline, groups, second_batch, tile, reinterpret_cast<cudaStream_t>(101));
    verify(first, groups); verify(second, groups);
    // Both callers use the same workspace. Every earlier reader/writer must
    // finish before the next invocation can touch any ring slot.
    for (const auto& old_tile : first.launches)
        for (int before : old_tile)
            for (const auto& new_tile : second.launches)
                for (int after : new_tile) assert(fake::precedes(before, after));
    const auto& first_wait = fake::nodes[second.start];
    bool waits_for_first_done = false;
    for (int parent : first_wait.parents)
        if (parent >= first.start && parent < first.end &&
            fake::nodes[parent].kind == "record") waits_for_first_done = true;
    assert(waits_for_first_done);
    assert(fake::synchronized_event_records.empty());
    }
    // Private stream drains cannot retire the one-tile work on caller 101.
    // Destruction must explicitly await the last done record before freeing
    // plan-owned scratch, including when this invocation used the caller.
    int last_done = -1;
    for (int i = second.start; i < second.end; ++i)
        if (fake::nodes[i].kind == "record") last_done = i;
    assert(last_done >= 0);
    assert(fake::synchronized_event_records == std::vector<int>{last_done});
    for (const auto* call : {&first, &second})
        for (const auto& launches : call->launches)
            for (int launch : launches) assert(fake::precedes(launch, last_done));
    std::cout << "PASS dependency DAG and event counts\n";
}
"""


@pytest.fixture(scope="module")
def pipeline_fixture(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is required for the CPU scheduler fixture")
    directory = tmp_path_factory.mktemp("batch-pipeline-schedule")
    (directory / "cuda_runtime.h").write_text(CUDA_STUB)
    source = directory / "fixture.cpp"
    source.write_text(FIXTURE)
    binary = directory / "fixture"
    subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(directory),
         "-I", str(ROOT / "src"), str(source), "-o", str(binary)],
        check=True, capture_output=True, text=True,
    )
    return binary


@pytest.mark.parametrize("groups", [2, 3, 5])
@pytest.mark.parametrize(
    "first_batch,second_batch,tile",
    [(1, 1, 4), (4, 4, 4), (16, 16, 16), (8, 8, 4), (12, 12, 4), (20, 20, 4),
     (5, 5, 4), (9, 9, 4), (17, 17, 4), (4, 17, 4), (17, 4, 4)],
)
def test_batch_pipeline_dependencies(pipeline_fixture, groups, first_batch, second_batch, tile):
    result = subprocess.run(
        [str(pipeline_fixture), str(groups), str(first_batch), str(second_batch), str(tile)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS dependency DAG and event counts" in result.stdout


@pytest.mark.parametrize("groups", [2, 3, 5])
@pytest.mark.parametrize("first_batch,second_batch", [(1, 1), (8, 8), (9, 9), (17, 17), (4, 17), (17, 4)])
def test_batch_pipeline_capture_submission(pipeline_fixture, groups, first_batch, second_batch):
    result = subprocess.run(
        [str(pipeline_fixture), str(groups), str(first_batch), str(second_batch), "4", "capture"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS capture submission dependencies" in result.stdout
