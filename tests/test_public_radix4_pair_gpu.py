"""Opt-in exact-mapping numerical/memory checks; no search or timing.

Set CUBUTTERFLY_GPU_TESTS=1, CUBUTTERFLY_TEST_LIBRARY and CUDA_VISIBLE_DEVICES
(exclusive GPU UUID). Run pytest -s to retain report-only SciPy error metrics.
An alternate completed experiment can be supplied via CUBUTTERFLY_RADIX4_SUMMARY.
Preserve the summary's CUBUTTERFLY_COMPILE_MODE (research for the default
artifact); changing or omitting it changes the strict target fingerprint.
"""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "radix4_validation_runner", ROOT / "scripts/run_public_radix4_pair_experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CASES = [dict(case, normalize_inverse=normalize)
         for case in runner.cases(("fp32", "fp64"), "all")
         if case["batch"] == 1 and (case["shape"] == [32, 32] or case["suite"] == "compact-real")
         for normalize in ((False, True) if case["inverse"] else (False,))]
CASES += [dict(case, c2r_prefix_scale=True) for case in CASES
          if case["fft_type"] == "c2r" and case["normalize_inverse"]]


def load_summary():
    path = Path(os.environ.get("CUBUTTERFLY_RADIX4_SUMMARY", str(
        ROOT / "results/radix4_completion_20261024/xd/summary.json")))
    if not path.is_file():
        if os.environ.get("CUBUTTERFLY_GPU_TESTS") == "1":
            pytest.fail("prepare the public radix4 experiment or set CUBUTTERFLY_RADIX4_SUMMARY")
        pytest.skip("saved experiment mapping is an optional local artifact")
    return json.loads(path.read_text()), str(path)


def check_compile_mode(summary):
    expected = summary["environment"].get("CUBUTTERFLY_COMPILE_MODE")
    actual = os.environ.get("CUBUTTERFLY_COMPILE_MODE")
    if actual != expected:
        pytest.fail("compile mode mismatch: summary records CUBUTTERFLY_COMPILE_MODE="
                    f"{expected!r}, current environment has {actual!r}; preserve the recorded mode. "
                    "Exact mapping target fingerprint validation remains enabled.")


def mapping_for(case):
    summary, path = load_summary()
    matches = [row for row in summary["cases"] if row["name"] == case["name"]]
    assert len(matches) == 1, "missing or ambiguous completed case"
    row = matches[0]
    assert row["status"] == "measured_and_errors_reported"
    mapping = copy.deepcopy(row["providers"]["radix4"]["mapping"])
    # Extend the measured normalized inverse to the public unnormalized semantic
    # contract, preserving every organization/core parameter. Never select auto.
    mapping["contract"]["normalize_inverse"] = case["normalize_inverse"]
    mapping["physical"]["normalize_inverse"] = case["normalize_inverse"]
    if case["fft_type"] == "c2r" and row["normalize_inverse"] and not case["normalize_inverse"]:
        cost = mapping["physical"]["cost"]
        cost["total_launches"] -= cost["scale_kernel_launches"]
        cost["scale_kernel_launches"] = 0
    if case.get("c2r_prefix_scale"):
        physical = mapping["physical"]
        physical["c2r_prefix_scale"] = True
        physical["c2r_scale_placement"] = "last-complex-axis"
        physical["cost"]["total_launches"] -= physical["cost"]["scale_kernel_launches"]
        physical["cost"]["scale_kernel_launches"] = 0
    return mapping, path


def assert_organization(mapping, case):
    runner.assert_radix4(mapping)
    physical = mapping["physical"]
    assert physical["packed_axes"] is True
    assert physical["paired_axes"] is True
    pairs = [axis for axis in physical["axes"] if axis["core"] == "radix4-register-pair"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["pair_rows"] == pair["length"] == 32
    assert pair["pair_planes"] == (33 if case["suite"] == "compact-real" else 1)
    if case["suite"] == "compact-real":
        assert pair["packed"] is True
        assert pair["boundary"] == "permute_nd"
        assert pair["permutation"] == [2, 0, 1]
        assert physical["output_permutation"], "compact pair must restore the logical layout"


def test_saved_mapping_cpu_contracts():
    assert len(CASES) == 14
    for case in CASES:
        mapping, _ = mapping_for(case)
        assert_organization(mapping, case)
        assert mapping["contract"]["inverse"] == int(case["inverse"])
        assert mapping["contract"]["normalize_inverse"] == case["normalize_inverse"]


def test_compile_mode_preflight_cpu(monkeypatch):
    summary = {"environment": {"CUBUTTERFLY_COMPILE_MODE": "research"}}
    monkeypatch.delenv("CUBUTTERFLY_COMPILE_MODE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="compile mode mismatch"):
        check_compile_mode(summary)
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "research")
    check_compile_mode(summary)


def test_unscaled_compact_cost_cpu():
    for case in CASES:
        if case["fft_type"] == "c2r" and not case["normalize_inverse"]:
            normalized, _ = mapping_for(dict(case, normalize_inverse=True))
            unscaled, _ = mapping_for(case)
            expected = copy.deepcopy(normalized)
            expected["contract"]["normalize_inverse"] = False
            expected["physical"]["normalize_inverse"] = False
            cost = expected["physical"]["cost"]
            cost["total_launches"] -= cost["scale_kernel_launches"]
            cost["scale_kernel_launches"] = 0
            assert unscaled == expected
            assert cost["total_launches"] == 4


@pytest.fixture(scope="module")
def gpu_runtime():
    library_path = os.environ.get("CUBUTTERFLY_TEST_LIBRARY")
    if os.environ.get("CUBUTTERFLY_GPU_TESTS") != "1" or not library_path:
        pytest.skip("requires explicit GPU opt-in and CUBUTTERFLY_TEST_LIBRARY")
    assert Path(library_path).is_file()
    summary, _ = load_summary()
    check_compile_mode(summary)
    # All CPU references are complete before importing/initializing CUDA.
    prepared = {}
    for case in CASES:
        data, reference = runner.cpu_arrays(case, 20261024)
        if case["inverse"] and not case["normalize_inverse"]:
            reference *= math.prod(case["shape"])
        prepared[(case["name"], case["normalize_inverse"])] = data, reference
    import numpy as np
    import torch
    assert torch.cuda.is_available(), "explicit GPU opt-in requires CUDA"
    sys.path.insert(0, str(ROOT / "python"))
    import cubutterfly as api
    return api, api.Library(library_path), torch, np, prepared


def case_id(case):
    return (case["name"] + ("_normalized" if case["normalize_inverse"] else "_unscaled")
            + ("_prefix_scale" if case.get("c2r_prefix_scale") else ""))


@pytest.mark.parametrize("case", CASES, ids=case_id)
def test_public_radix4_exact_mapping_gpu(gpu_runtime, case):
    api, library, torch, np, prepared = gpu_runtime
    data, reference = prepared[(case["name"], case["normalize_inverse"])]
    mapping, mapping_source = mapping_for(case)
    storage = ("COMPLEX_" if case["fft_type"] == "c2c" else "") + case["precision"].upper()
    transform = api.Transform(
        storage=getattr(api.DataType, storage), compute=getattr(api.ComputeType, case["precision"].upper()),
        extents=case["shape"], batch=1, inverse=case["inverse"],
        normalize_inverse=case["normalize_inverse"], fft_type=getattr(api.FftType, case["fft_type"].upper()),
        execution_mode=api.ExecutionMode.XD)
    options = api.PlanOptions(policy=api.AlgorithmPolicy.EXPLICIT,
                              mapping=json.dumps(mapping), execution_mode=api.ExecutionMode.XD)
    stream = torch.cuda.Stream()
    with api.Plan(library, transform, options, stream=stream.cuda_stream) as plan:
        details = runner.metadata(plan)
        assert details["mapping"] == mapping
        assert details["selection_source_id"] == 4
        assert_organization(details["mapping"], case)
        # Guard both sides while keeping contiguous, naturally aligned views.
        incoming = torch.full((data.size + 64,), 7, dtype=torch.from_numpy(data).dtype, device="cuda")
        outgoing = torch.full((reference.size + 64,), 7, dtype=torch.from_numpy(reference).dtype, device="cuda")
        source = incoming[32:-32].view(data.shape)
        target = outgoing[32:-32].view(reference.shape)
        source.copy_(torch.from_numpy(data))
        original = incoming.cpu().numpy().copy()
        size = plan.workspace_size()
        workspace = torch.full((size + 512,), 0xA5, dtype=torch.uint8, device="cuda")
        if size:
            plan.set_workspace(workspace[256:-256], size)
        stream.wait_stream(torch.cuda.current_stream())
        plan.execute(source, target)
        stream.synchronize()
        eager = target.cpu().numpy().copy()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.execute(source, target)
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        actual = target.cpu().numpy().copy()
        metrics, _ = runner.errors(actual, reference)
        print(json.dumps(dict(case=case, mapping_source=mapping_source,
                              reference="same-precision scipy.fft; report-only, no threshold",
                              cpu_reference_error=metrics, metadata=details), allow_nan=False))
        assert metrics["nonfinite"] == 0
        np.testing.assert_array_equal(actual, eager)
        np.testing.assert_array_equal(incoming.cpu().numpy(), original)
        output = outgoing.cpu().numpy()
        assert np.all(output[:32] == 7) and np.all(output[-32:] == 7)
        scratch = workspace.cpu().numpy()
        assert np.all(scratch[:256] == 0xA5) and np.all(scratch[-256:] == 0xA5)
        del graph
