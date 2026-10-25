"""Opt-in public FWHT final-storage regressions; no GPU work at import.

Run with CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY=/path/libcuntt.so.
All candidate selection uses the public DEFAULT/MEASURE policies.
"""
import ctypes as ct
from dataclasses import replace
import json

import pytest

from test_semantic_execution import runtime


CASES = [(n, inverse, normalize, scale)
         for n in (32, 64, 128)
         for inverse, normalize, scale in (
             (False, False, 1.0), (False, True, 0.375),
             (True, False, -0.75), (True, True, 1.0), (True, True, 0.375))]


def _values(torch, n, batch=3):
    # FP16 inputs and every FP32 butterfly sum are exact dyadic values.
    generator = torch.Generator().manual_seed(61013 + n)
    return (torch.randint(-1024, 1025, (batch, n), generator=generator).float() / 1024).half()


def _oracle(torch, values, inverse, normalize, scale, stage_storage=False):
    result = values.float()
    width = 1
    while width < result.shape[-1]:
        paired = result.reshape(values.shape[0], -1, 2 * width)
        left, right = paired[..., :width], paired[..., width:]
        result = torch.cat((left + right, left - right), dim=-1).reshape_as(result)
        if stage_storage:
            result = result.half().float()
        width *= 2
    if inverse and normalize:
        result = result / values.shape[-1]
    return (result * torch.tensor(scale, dtype=torch.float32)).half()


def _addresses(n, batch, stride, batch_stride, reversed_order):
    bits = n.bit_length() - 1
    order = [int(f"{i:0{bits}b}"[::-1], 2) if reversed_order else i for i in range(n)]
    return [row * batch_stride + index * stride for row in range(batch) for index in order]


def _selection(library, plan):
    function = library._dll.cubutterflyPlanGetSelectionSource
    function.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
    function.restype = ct.c_int
    source = ct.c_int()
    library._call("PlanGetSelectionSource", plan.native_handle, ct.byref(source))
    return source.value


def _metadata(library, plan):
    return dict(selection_source=_selection(library, plan),
                reason=plan.selection_reason(), algorithm=plan.algorithm_name(),
                mapping=json.loads(plan.mapping_json()),
                descriptor=json.loads(plan.descriptor_json()), workspace_bytes=plan.workspace_size())


def _execute(runtime, plan, transform, values, metadata, stage_storage=False):
    _, _, torch, _, _ = runtime
    n, batch = transform.extents[0], transform.batch
    incoming = _addresses(n, batch, transform.input_strides[0] if transform.input_strides else 1,
                          transform.input_batch_stride or n,
                          bool(transform.input_orders and transform.input_orders[0]))
    outgoing = _addresses(n, batch, transform.output_strides[0] if transform.output_strides else 1,
                          transform.output_batch_stride or n,
                          bool(transform.output_orders and transform.output_orders[0]))
    size = max(max(incoming), max(outgoing)) + 17
    source = torch.full((size,), 123.5, dtype=torch.float16, device="cuda")
    input_indices = torch.tensor(incoming, device="cuda")
    output_indices = torch.tensor(outgoing, device="cuda")
    source.index_copy_(0, input_indices, values.flatten().to("cuda"))
    output = source if transform.in_place else torch.full_like(source, -123.5)
    source_before, output_before = source.clone(), output.clone()
    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
    if plan.workspace_size():
        plan.set_workspace(workspace, plan.workspace_size())
    assert plan.input_size() <= source.numel() * source.element_size(), metadata
    assert plan.output_size() <= output.numel() * output.element_size(), metadata
    plan.execute(source, output)
    torch.cuda.synchronize()
    expected = _oracle(torch, values, transform.inverse, transform.normalize_inverse,
                       transform.output_scale, stage_storage=stage_storage)
    actual = output.index_select(0, output_indices).reshape_as(expected).cpu()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), metadata
    untouched = torch.ones(size, dtype=torch.bool, device="cuda")
    untouched[output_indices] = False
    assert torch.equal(output[untouched].view(torch.int16), output_before[untouched].view(torch.int16)), metadata
    if not transform.in_place:
        assert torch.equal(source.view(torch.int16), source_before.view(torch.int16)), metadata
    return actual


def _transform(api, n, inverse, normalize, scale):
    return api.Transform(op=api.Operator.FWHT, storage=api.DataType.FP16,
                         compute=api.ComputeType.FP32, extents=[n], batch=3,
                         inverse=inverse, normalize_inverse=normalize,
                         output_scale=scale, output_rounding=api.OutputRounding.FINAL_STORAGE)


@pytest.mark.parametrize("n,inverse,normalize,scale", CASES)
def test_measure_default_cache_and_final_narrow(runtime, n, inverse, normalize, scale,
                                               tmp_path, monkeypatch, record_property):
    api, library, torch, _, _ = runtime
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_BUDGET", "3")
    transform = _transform(api, n, inverse, normalize, scale)
    values = _values(torch, n)
    assert not torch.equal(_oracle(torch, values, False, False, 1.0).view(torch.int16),
                           _oracle(torch, values, False, False, 1.0, stage_storage=True).view(torch.int16)), \
        "input must distinguish full-FP32 arithmetic from intermediate FP16 rounding"
    if scale != 1.0:
        narrowed_first = (_oracle(torch, values, inverse, normalize, 1.0).float() * scale).half()
        assert not torch.equal(_oracle(torch, values, inverse, normalize, scale).view(torch.int16),
                               narrowed_first.view(torch.int16)), "input must distinguish scale-before-narrow"
    cache = tmp_path / "selection.cache"
    stream = torch.cuda.current_stream().cuda_stream
    mappings = []
    for policy, expected_source in ((api.AlgorithmPolicy.MEASURE, 2), (api.AlgorithmPolicy.DEFAULT, 1)):
        with api.Plan(library, transform, api.PlanOptions(policy=policy), stream=stream, cache_path=cache) as plan:
            metadata = _metadata(library, plan)
            record_property(policy.name, json.dumps(metadata, sort_keys=True))
            assert metadata["selection_source"] == expected_source, metadata
            assert metadata["mapping"]["output_epilogue"] == {"scale": scale, "final_storage_rounding": True}, metadata
            assert metadata["mapping"]["numeric_lowering"]["scale_position"] == "fp32-before-final-fp16-narrow", metadata
            if n == 128 or inverse:
                assert metadata["mapping"]["numeric_lowering"]["kind"] == "promoted-full-fp32-final-storage", metadata
            _execute(runtime, plan, transform, values, metadata)
            mappings.append(plan.mapping_json())
    assert json.loads(mappings[0]) == json.loads(mappings[1])
    with pytest.raises(api.Error, match="semantic contract mismatch|epilogue contract mismatch"):
        with api.Plan(library, replace(transform, output_scale=scale * 2),
                      api.PlanOptions(mapping=mappings[0]), stream=stream):
            pass


@pytest.mark.parametrize("n", (32, 64))
def test_default_reaches_short_provider(runtime, n, tmp_path, monkeypatch, record_property):
    api, library, torch, _, _ = runtime
    registry = tmp_path / "empty-registry.json"
    registry.write_text(json.dumps({"schema": "cubutterfly-registry-v1", "targets": []}))
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    transform = _transform(api, n, False, False, 0.375)
    with api.Plan(library, transform, stream=torch.cuda.current_stream().cuda_stream,
                  cache_path=tmp_path / "cold.cache") as plan:
        metadata = _metadata(library, plan)
        record_property("default-short-provider", json.dumps(metadata, sort_keys=True))
        assert metadata["mapping"]["numeric_lowering"]["kind"] == "short-fwht-fp32-final-storage", metadata
        assert metadata["mapping"]["numeric_lowering"]["storage_conversion_kernels"] == 0, metadata
        _execute(runtime, plan, transform, _values(torch, n), metadata)


def test_boundary_stride_order_inplace_preserves_epilogue(runtime, tmp_path, monkeypatch, record_property):
    api, library, torch, _, _ = runtime
    monkeypatch.setenv("CUBUTTERFLY_SEARCH_BUDGET", "3")
    transform = replace(_transform(api, 64, True, True, 0.375), in_place=True,
                        input_strides=[2], output_strides=[2],
                        input_batch_stride=137, output_batch_stride=137,
                        input_orders=[1], output_orders=[0])
    cache = tmp_path / "boundary.cache"
    mappings = []
    for policy, expected_source in ((api.AlgorithmPolicy.MEASURE, 2), (api.AlgorithmPolicy.DEFAULT, 1)):
        with api.Plan(library, transform, api.PlanOptions(policy=policy),
                      stream=torch.cuda.current_stream().cuda_stream, cache_path=cache) as plan:
            metadata = _metadata(library, plan)
            record_property("boundary-" + policy.name, json.dumps(metadata, sort_keys=True))
            assert metadata["selection_source"] == expected_source, metadata
            assert metadata["mapping"]["kind"] == "semantic-boundary", metadata
            assert metadata["mapping"]["child"]["output_epilogue"] == {"scale": 0.375, "final_storage_rounding": True}, metadata
            _execute(runtime, plan, transform, _values(torch, 64), metadata)
            mappings.append(plan.mapping_json())
    assert json.loads(mappings[0]) == json.loads(mappings[1])


@pytest.mark.parametrize("n", (32, 64))
def test_default_stage_and_final_rounding_have_distinct_semantics(runtime, n, tmp_path,
                                                                monkeypatch, record_property):
    api, library, torch, _, _ = runtime
    registry = tmp_path / "rounding-registry.json"
    registry.write_text(json.dumps({"schema": "cubutterfly-registry-v1", "targets": []}))
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    final = _transform(api, n, False, False, 1.0)
    values = _values(torch, n)
    stream = torch.cuda.current_stream().cuda_stream
    with api.Plan(library, final, stream=stream, cache_path=tmp_path / "rounding.cache") as plan:
        final_metadata = _metadata(library, plan)
        record_property("final-default", json.dumps(final_metadata, sort_keys=True))
        assert final_metadata["mapping"]["numeric_lowering"]["kind"] == "short-fwht-fp32-final-storage", final_metadata
        final_actual = _execute(runtime, plan, final, values, final_metadata)

    stage = replace(final, output_rounding=api.OutputRounding.STAGE_STORAGE)
    with api.Plan(library, stage, stream=stream, cache_path=tmp_path / "rounding.cache") as plan:
        baseline_metadata = _metadata(library, plan)
        assert baseline_metadata["mapping"]["local_exchange"] != "warp-register", baseline_metadata
        _execute(runtime, plan, stage, values, baseline_metadata, stage_storage=True)
    # Rank-one physical mappings contain a contract, while boundary mappings
    # expose the target identity used by the registry's hardware/build gate.
    with api.Plan(library, replace(final, input_orders=[1]), stream=stream,
                  cache_path=tmp_path / "identity.cache") as plan:
        boundary_mapping = json.loads(plan.mapping_json())
        assert boundary_mapping["kind"] == "semantic-boundary", boundary_mapping
        target = boundary_mapping["target"]

    # The faster legacy short record must be filtered; a same-key valid generic
    # record must be selected, proving that the hardware and semantic keys match.
    old_mapping = dict(final_metadata["mapping"])
    for field in ("contract", "target", "output_epilogue", "numeric_lowering"):
        old_mapping.pop(field, None)
    old_record = dict(status="measured", candidate_id="legacy-final-only-short",
                      runtime_fingerprint=target["runtime_fingerprint"], kernel_ms=0.000001,
                      key=dict(operator="fwht", precision="fp16", placement="out-of-place",
                               logN=n.bit_length() - 1, batch=3, direction="forward", accumulation="fp32",
                               element_stride=1, batch_stride=n, stage_matrices=""), mapping=old_mapping)
    generic_mapping = dict(baseline_metadata["mapping"])
    generic_mapping.pop("contract", None)
    generic_record = dict(old_record, candidate_id="matching-stage-storage-control",
                          kernel_ms=1.0, mapping=generic_mapping)
    registry.write_text(json.dumps({"schema": "cubutterfly-registry-v1", "targets": [
        {"hardware": {"device": target["name"],
                      "compute_capability": f'{target["sm_major"]}.{target["sm_minor"]}',
                      "global_memory_bytes": target["memory_bytes"]},
         "records": [old_record, generic_record]}]}))
    with api.Plan(library, stage, stream=stream, cache_path=tmp_path / "rounding.cache") as plan:
        stage_metadata = _metadata(library, plan)
        record_property("stage-default", json.dumps(stage_metadata, sort_keys=True))
        assert stage_metadata["mapping"]["local_exchange"] != "warp-register", stage_metadata
        assert stage_metadata["selection_source"] == 0, stage_metadata
        assert "exact hardware/semantic match" in stage_metadata["reason"], stage_metadata
        assert stage_metadata["mapping"] == baseline_metadata["mapping"], stage_metadata
        assert "numeric_lowering" not in stage_metadata["mapping"], stage_metadata
        stage_actual = _execute(runtime, plan, stage, values, stage_metadata, stage_storage=True)
    assert not torch.equal(stage_actual.view(torch.int16), final_actual.view(torch.int16))
