"""FP64 cuFFTDx resource provenance and optional physical capture checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage_composition import resolve_batch_services


def test_resource_query_combines_actual_static_and_dynamic_bytes_and_reports_failure(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler required")
    source = (ROOT / "src/cufftdx_fft_fp64.cu").read_text()
    start = source.index("template <class Kernel>\nvoid query_online_resources(")
    end = source.index("template <unsigned Size", start)
    harness = r'''
#include <cassert>
#include <cstddef>
#include <stdexcept>
struct ExecutionGroup {
    std::size_t live_shared_bytes=0, compiler_local_bytes_per_thread=0;
    int compiler_registers_per_thread=0;
    bool compiler_resources_known=false, compiler_local_resources_known=false;
};
struct cudaFuncAttributes { std::size_t sharedSizeBytes, localSizeBytes; int numRegs; };
constexpr int cudaSuccess=0;
int cudaFuncGetAttributes(cudaFuncAttributes* attributes, int kernel) {
    if (kernel == 0) return 1;
    *attributes = {256, 16, 37};
    return cudaSuccess;
}
const char* cudaGetErrorString(int) { return "query failed"; }
''' + source[start:end] + r'''
int main() {
    ExecutionGroup group;
    query_online_resources(1, 8192, group);
    assert(group.live_shared_bytes == 8448);
    assert(group.compiler_registers_per_thread == 37);
    assert(group.compiler_local_bytes_per_thread == 16);
    assert(group.compiler_resources_known && group.compiler_local_resources_known);
    ExecutionGroup unknown;
    try { query_online_resources(0, 8192, unknown); return 1; }
    catch (const std::runtime_error&) {}
    assert(!unknown.compiler_resources_known && !unknown.compiler_local_resources_known);
}
'''
    unit = tmp_path / "query.cpp"
    unit.write_text(harness)
    executable = tmp_path / "query"
    subprocess.run([compiler, "-std=c++17", str(unit), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


@pytest.mark.parametrize("batch", [9, 10, 11])
@pytest.mark.parametrize("tile", [2, 4, 8])
def test_fp64_ring_resources_match_actual_serial_capture(batch, tile):
    binary = os.environ.get("CUBUTTERFLY_STAGE_TEST_BINARY")
    if not binary:
        pytest.skip("set CUBUTTERFLY_STAGE_TEST_BINARY on an idle CUDA device")
    mapping = dict(backend="online-reorder", fft_core="cufftdx-block",
                   stage_partition=[7, 9], local_stages=7,
                   prefix_threads=128, suffix_threads=128, prefix_ept=4, suffix_ept=4,
                   shared_layout="linear", stage_overlap=True, batch_tile_count=tile)
    point = dict(operator="fft", precision="fp64", logN=16, batch=batch,
                 backend="online-reorder", fft_core="cufftdx-block",
                 stage_overlap=True, batch_tile_count=tile, mapping_json=json.dumps(mapping))

    def describe(request):
        completed = subprocess.run([binary, "--describe-only", "--point-json", json.dumps(request)],
                                   check=True, capture_output=True, text=True)
        return json.loads(completed.stdout)

    source = describe(point)
    for group in source["groups"]:
        assert group["live_shared_bytes"] > 0
        assert group["compiler_resources_known"]
        assert group["compiler_local_resources_known"]
    projection = resolve_batch_services(source["sample"], source, describe)
    assert projection["status"] == "resolved", projection
    for view in (projection["full"], projection["tail"]):
        if view is None:
            continue
        for original, actual in zip(source["groups"], view["groups"]):
            assert len(actual["actual_kernels"]) == 1
            for field in ("live_shared_bytes", "compiler_registers_per_thread",
                          "compiler_local_bytes_per_thread"):
                assert original[field] == actual["actual_kernels"][0][field]
