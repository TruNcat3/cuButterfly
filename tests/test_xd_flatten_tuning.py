"""Opt-in GPU checks of public xD MEASURE and exact mapping replay.

Set CUBUTTERFLY_LIBRARY and run under the GPU isolation guard. Collection does
not import torch or initialize CUDA. FWHT/structured use FP32; zeta's supported
storage is UINT32. Small integer matrices/inputs make exact equality meaningful,
without adding a floating-point tolerance. Error reports are printed and kept
in pytest properties (and optional --junitxml output).
"""

import copy
import json
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ("FWHT", "SUBSET_ZETA", "SUPERSET_ZETA", "STRUCTURED_2X2")
MATRICES = ((1, 1, 0, 1), (1, 0, 1, 1))


@pytest.fixture(scope="module")
def gpu_api():
    path = os.environ.get("CUBUTTERFLY_LIBRARY")
    if not path:
        pytest.skip("set CUBUTTERFLY_LIBRARY to opt in to GPU runtime tests")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    sys.path.insert(0, str(ROOT / "python"))
    import cubutterfly as cb

    return torch, cb, cb.Library(path)


def _reference(np, values, operator, inverse):
    work = values.astype(np.float64).copy()
    for axis in (1, 2):
        moved = np.moveaxis(work, axis, -1).copy()
        if operator == "FWHT":
            matrix = (1, 1, 1, -1)
        elif operator == "SUBSET_ZETA":
            matrix = (1, 0, -1 if inverse else 1, 1)
        elif operator == "SUPERSET_ZETA":
            matrix = (1, -1 if inverse else 1, 0, 1)
        else:
            a, b, c, d = MATRICES[axis - 1]
            matrix = (d, -b, -c, a) if inverse else (a, b, c, d)
        a, b, c, d = matrix
        for half in (1, 2, 4):
            for start in range(0, 8, 2 * half):
                left = moved[..., start:start + half].copy()
                right = moved[..., start + half:start + 2 * half].copy()
                moved[..., start:start + half] = a * left + b * right
                moved[..., start + half:start + 2 * half] = c * left + d * right
        work = np.moveaxis(moved, -1, axis)
    if operator == "FWHT" and inverse:
        work /= 64
    if "ZETA" in operator:
        return np.remainder(work, 2**32).astype(np.uint32)
    return work.astype(np.float32)


def _resolved_flattened_mapping(parent, cached, operator):
    """Account for documented plan construction, without ignoring fields.

    The tuning cache stores the pre-construction config; mapping_json reports
    the constructed config (butterfly.cu:1272-1294) plus its semantic contract
    (c_api.cpp:4747-4754). Every other field must remain identical.
    """
    physical = copy.deepcopy(cached)
    backend = physical["backend"]
    if backend != "stage-pipeline":
        physical.update(stage_space=0, pipeline_warps=0)
    if backend not in ("temporal-tile", "hierarchical", "online-reorder",
                       "shared-iterative", "factor-streamed"):
        physical["tile_threads"] = 0
    if backend not in ("hierarchical", "online-reorder", "shared-iterative", "factor-streamed"):
        physical["local_stages"] = 0
    if backend != "online-reorder":
        physical["reorder_columns"] = 0
    if backend != "warp-hybrid":
        physical["warp_stages"] = 0
    if not physical["stage_partition"]:
        physical["stage_partition"] = (
            [physical["local_stages"], 6 - physical["local_stages"]]
            if backend == "online-reorder" else [6]
        )
    contract = copy.deepcopy(parent["contract"])
    contract.update(shape=[64], input_strides=[1], output_strides=[1],
                    input_orders=[0], output_orders=[0], matrices=[[]])
    if operator == "STRUCTURED_2X2":
        contract["matrices"] = [[list(MATRICES[1])] * 3 + [list(MATRICES[0])] * 3]
    physical["contract"] = contract
    return dict(parent, lowering="flattened", physical=physical)


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("inverse", (False, True), ids=("forward", "inverse"))
def test_public_xd_measure_and_explicit_flatten_replay(
    gpu_api, tmp_path, monkeypatch, record_property, operator, inverse
):
    torch, cb, library = gpu_api
    np = pytest.importorskip("numpy")
    # Bound this correctness check, rather than conducting a full calibration.
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_BUDGET", "8")
    cache = tmp_path / "plans.cache"
    stream = torch.cuda.Stream()
    zeta = "ZETA" in operator
    transform = cb.Transform(
        op=getattr(cb.Operator, operator),
        storage=cb.DataType.UINT32 if zeta else cb.DataType.FP32,
        compute=cb.ComputeType.UINT32 if zeta else cb.ComputeType.FP32,
        extents=[8, 8], batch=2, inverse=inverse,
        normalize_inverse=operator == "FWHT" and inverse,
        stage_matrices=[[MATRICES[0]], [MATRICES[1]]]
        if operator == "STRUCTURED_2X2" else [],
        execution_mode=cb.ExecutionMode.XD,
    )
    values = (np.arange(128).reshape(2, 8, 8) % 5).astype(
        np.uint32 if zeta else np.float32
    )
    expected = _reference(np, values, operator, inverse)
    with torch.cuda.stream(stream):
        source = torch.from_numpy(values.copy()).cuda()
        output = torch.empty_like(source)
    reports = []

    def check_plan(options, label, expected_mapping=None):
        with cb.Plan(library, transform, options, stream=stream.cuda_stream,
                     cache_path=str(cache)) as plan:
            mapping = json.loads(plan.mapping_json())
            if expected_mapping is not None:
                assert mapping == expected_mapping
            workspace = torch.empty(plan.workspace_size(), dtype=torch.uint8, device=source.device)
            if workspace.numel():
                plan.set_workspace(workspace, workspace.numel())
            with torch.cuda.stream(stream):
                plan.execute(source, output)
            stream.synchronize()
            actual = output.cpu().numpy().copy()
            delta = actual.astype(np.float64) - expected.astype(np.float64)
            report = dict(label=label, mapping=mapping, reason=plan.selection_reason(),
                          max_abs=float(np.abs(delta).max()),
                          rmse=float(np.sqrt(np.mean(delta * delta))),
                          mismatch_count=int(np.count_nonzero(actual != expected)),
                          elements=int(expected.size), finite=bool(np.isfinite(actual).all()))
            reports.append(report)
            record_property(label, json.dumps(report, sort_keys=True))
            print(json.dumps(dict(operator=operator, inverse=inverse, **report), sort_keys=True))
            assert np.array_equal(source.cpu().numpy(), values), "out-of-place plan changed its input"
            assert np.array_equal(actual, expected), report
            return mapping

    measured = check_plan(cb.PlanOptions(policy=cb.AlgorithmPolicy.MEASURE), "measure")
    # A rank1 N=64 entry proves the flattened child was actually tuned, even if
    # complete-plan selection ultimately preferred an axis lowering.
    children = []
    for line in cache.read_text().splitlines():
        key, text = line.split("\t", 1)
        semantics = json.loads(key.split("|semantics:", 1)[1].split("|xd-", 1)[0])
        if semantics["shape"] == [64] and semantics["batch"] == 2:
            children.append(json.loads(text))
    assert children, "MEASURE did not persist the flattened rank1 child"
    check_plan(cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT,
                             mapping=json.dumps(measured)), "explicit-selected", measured)
    flattened = dict(measured, lowering="flattened", physical=children[-1])
    canonical = _resolved_flattened_mapping(measured, children[-1], operator)
    check_plan(cb.PlanOptions(policy=cb.AlgorithmPolicy.EXPLICIT,
                             mapping=json.dumps(flattened)), "explicit-flattened", canonical)
