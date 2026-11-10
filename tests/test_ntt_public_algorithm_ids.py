"""Public NTT algorithm identification and exact replay regressions.

Run with CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY=/path/libcuntt.so.
"""
import json

import pytest

from test_semantic_execution import runtime


MODULUS = 576460756061519873


@pytest.mark.parametrize("backend,n,inverse", [
    (None, 1048576, False),
    (None, 1048576, True),
    ("compact-stage", 1048576, False),
    ("stage-pipeline", 256, False),
    ("hybrid-dataflow", 256, False),
    ("hybrid-dataflow", 256, True),
], ids=["automatic-log20-forward", "automatic-log20-inverse",
        "compact-stage-log20-forward", "stage-pipeline-forward",
        "hybrid-dataflow-forward", "hybrid-dataflow-inverse"])
def test_public_ntt_algorithm_and_mapping_replay(runtime, backend, n, inverse):
    api, library, torch, np, _ = runtime
    transform = api.Transform(
        op=api.Operator.NTT, storage=api.DataType.UINT64,
        compute=api.ComputeType.UINT64, extents=[n], batch=1,
        modulus=MODULUS, word_bits=64, inverse=inverse,
        normalize_inverse=inverse, input_orders=[0], output_orders=[0],
        input_strides=[1], output_strides=[1],
        input_batch_stride=n, output_batch_stride=n)
    options = api.PlanOptions()
    if backend is not None:
        options.mapping = json.dumps({
            "schema_version": 1, "kind": "ntt", "backend": backend,
            "compute_unit": "radix2"})
    stream = torch.cuda.current_stream().cuda_stream
    host_input = np.zeros(n, dtype=np.uint64)
    host_input[0] = 1
    expected = np.full(n, pow(n, -1, MODULUS) if inverse else 1, dtype=np.uint64)
    device_input = torch.from_numpy(host_input).to(device="cuda")
    device_output = torch.empty_like(device_input)

    def check(plan, semantics):
        assert json.loads(plan.descriptor_json())["semantics"] == semantics
        workspace = torch.empty(max(1, plan.workspace_size()), device="cuda", dtype=torch.uint8)
        plan.set_workspace(workspace, plan.workspace_size())
        plan.execute(device_input, device_output)
        torch.cuda.synchronize()
        np.testing.assert_array_equal(device_output.cpu().numpy(), expected)
        np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)

    with api.Descriptor(library, transform, options) as descriptor:
        semantics = json.loads(descriptor.json())["semantics"]
        assert semantics["shape"] == [n] and semantics["batch"] == 1
        assert semantics["modulus"] == MODULUS and semantics["word_bits"] == 64
        assert semantics["input_orders"] == semantics["output_orders"] == [0]
        assert semantics["inverse"] == int(inverse)
        with api.Plan(library, descriptor, stream=stream) as plan:
            algorithm = plan.algorithm_name()
            mapping = plan.mapping_json()
            if backend is not None:
                assert json.loads(algorithm)["backend"] == backend
                assert json.loads(mapping)["backend"] == backend
            check(plan, semantics)
    for replay in (algorithm, mapping):
        with api.Plan(library, transform, api.PlanOptions(mapping=replay), stream=stream) as plan:
            assert json.loads(plan.mapping_json()) == json.loads(mapping)
            check(plan, semantics)


@pytest.mark.parametrize("n,inverse", [
    (256, False),
    (1048576, True),
], ids=["small-size", "inverse"])
def test_public_compact_stage_rejects_unsupported_configuration(runtime, n, inverse):
    api, library, torch, _, _ = runtime
    transform = api.Transform(
        op=api.Operator.NTT, storage=api.DataType.UINT64,
        compute=api.ComputeType.UINT64, extents=[n], batch=1,
        modulus=MODULUS, word_bits=64, inverse=inverse,
        normalize_inverse=inverse, input_orders=[0], output_orders=[0],
        input_strides=[1], output_strides=[1],
        input_batch_stride=n, output_batch_stride=n)
    options = api.PlanOptions(mapping=json.dumps({
        "schema_version": 1, "kind": "ntt", "backend": "compact-stage",
        "compute_unit": "radix2"}))
    with pytest.raises(api.Error, match="compact-stage currently requires a forward, 64-bit logN=20 transform"):
        with api.Plan(library, transform, options,
                      stream=torch.cuda.current_stream().cuda_stream):
            pytest.fail("compact-stage accepted an unsupported configuration")
