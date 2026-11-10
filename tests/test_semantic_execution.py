"""Opt-in public Python API correctness tests, with no timing or GPU work at import.

Run with CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY=/absolute/libcuntt.so.
All plans use DEFAULT selection (or exact mapping replay) and manual workspace.
"""
from dataclasses import replace
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((8,), (4, 8), (2, 2, 2, 4))
OPERATORS = ("FFT", "NTT", "FWHT", "SUBSET_ZETA", "SUPERSET_ZETA", "STRUCTURED_2X2")
CASES = [(op, shape, inverse, orders, False, "fp32")
         for op, shape, inverse, orders in itertools.product(OPERATORS, SHAPES, (False, True), range(3))]
CASES += [(op, shape, inverse, 2, True, "fp32")
          for op, shape, inverse in itertools.product(("FFT", "FWHT", "NTT"), SHAPES, (False, True))]
CASES += [(op, shape, inverse, 2, False, precision)
          for (op, precision), shape, inverse in itertools.product(
              (("FFT", "fp64"), ("FWHT", "fp16"), ("FWHT", "bf16"),
               ("STRUCTURED_2X2", "fp16"), ("STRUCTURED_2X2", "bf16")), SHAPES, (False, True))]


@pytest.fixture(scope="module")
def runtime():
    path = os.environ.get("CUBUTTERFLY_TEST_LIBRARY")
    if os.environ.get("CUBUTTERFLY_GPU_TESTS") != "1" or not path:
        pytest.skip("requires CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY")
    if not Path(path).is_file():
        pytest.fail("CUBUTTERFLY_TEST_LIBRARY does not name an existing library")
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    if not torch.cuda.is_available():
        pytest.skip("opted in, but torch CUDA is unavailable")
    sys.path.insert(0, str(ROOT / "python"))
    import cubutterfly as api
    spec = importlib.util.spec_from_file_location("semantic_math_reference", ROOT / "tests/check_xd.py")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    return api, api.Library(path), torch, np, reference


def _layout(shape, variant):
    strides = [0] * len(shape)
    span = variant + 2
    axes = range(len(shape)) if variant else reversed(range(len(shape)))
    for axis in axes:
        strides[axis] = span
        span = span * shape[axis] + variant + 1
    return strides, span + 7


def _addresses(np, shape, strides, batch_stride, orders):
    result = []
    for batch in range(2):
        for coords in itertools.product(*(range(n) for n in shape)):
            address = batch * batch_stride
            for coord, extent, stride, order in zip(coords, shape, strides, orders):
                if order:
                    coord = int(f"{coord:0{extent.bit_length() - 1}b}"[::-1], 2)
                address += coord * stride
            result.append(address)
    assert len(set(result)) == len(result)
    return np.asarray(result, dtype=np.int64)


def _transform(api, op, shape, inverse, orders, in_place, precision, packed=False):
    integer = op in ("NTT", "SUBSET_ZETA", "SUPERSET_ZETA")
    storage = api.DataType.UINT32 if integer else getattr(
        api.DataType, ("COMPLEX_" if op == "FFT" else "") + precision.upper())
    compute = api.ComputeType.UINT32 if integer else (
        api.ComputeType.FP64 if precision == "fp64" else api.ComputeType.FP32)
    rank = len(shape)
    if orders == 0:
        incoming, outgoing = [0] * rank, [1] * rank
    elif orders == 1:
        incoming, outgoing = [1] * rank, [0] * rank
    else:
        incoming = [axis % 2 for axis in range(rank)]
        outgoing = [1 - flag for flag in incoming]
    ins, ib = _layout(shape, 0)
    outs, ob = _layout(shape, 1)
    if in_place:
        outs, ob = list(ins), ib
    if packed:
        ins = [math.prod(shape[axis + 1:]) for axis in range(rank)]
        outs, ib, ob = list(ins), math.prod(shape), math.prod(shape)
    kwargs = dict(op=getattr(api.Operator, op), storage=storage, compute=compute,
                  extents=list(shape), batch=2, inverse=inverse, normalize_inverse=True,
                  in_place=in_place, input_strides=ins, output_strides=outs,
                  input_batch_stride=ib, output_batch_stride=ob,
                  input_orders=incoming, output_orders=outgoing)
    if op == "NTT":
        kwargs.update(modulus=257, word_bits=32)
    if op == "STRUCTURED_2X2":
        kwargs["stage_matrices"] = [[(1.0, .25, -.5, 1.0)] for _ in shape]
    return api.Transform(**kwargs)


def _values_and_reference(runtime, transform, op, precision):
    _, _, torch, np, reference = runtime
    shape = (2, *transform.extents)
    count = math.prod(shape)
    integer = op in ("NTT", "SUBSET_ZETA", "SUPERSET_ZETA")
    if integer:
        values = (np.arange(count, dtype=np.uint32) * 7 + 3) % 29
        if op == "NTT":
            expected = reference._ntt_reference(values.tolist(), transform.extents, 2, 257, transform.inverse)
        else:
            expected = reference._zeta_reference(values, shape, int(transform.op), transform.inverse)
        return values, np.asarray(expected, dtype=np.uint32), torch.uint32, 73
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
             "fp32": torch.float32, "fp64": torch.float64}[precision]
    values = ((torch.arange(count, dtype=torch.float64) % 17) - 8) / 16
    if op == "FFT":
        dtype = torch.complex128 if precision == "fp64" else torch.complex64
        values = torch.complex(values, values.roll(3) / 2)
    values = values.to(dtype)
    work = values.to(torch.complex128 if op == "FFT" else torch.float64).reshape(shape)
    if op == "FFT":
        operation = torch.fft.ifftn if transform.inverse else torch.fft.fftn
        expected = operation(work, dim=tuple(range(1, len(shape))))
    elif op == "FWHT":
        expected = reference._fwht_reference(torch, work, range(1, len(shape)))
        if transform.inverse:
            expected /= math.prod(transform.extents)
    else:
        expected = reference._structured_reference(torch, work, transform.extents,
                                                   transform.stage_matrices, transform.inverse)
    return work.reshape(-1).numpy(), expected.reshape(-1).numpy(), dtype, 73


def _execute_and_check(runtime, plan, transform, op, precision):
    _, _, torch, np, _ = runtime
    values, expected, dtype, sentinel = _values_and_reference(runtime, transform, op, precision)
    incoming = _addresses(np, transform.extents, transform.input_strides,
                          transform.input_batch_stride, transform.input_orders)
    outgoing = _addresses(np, transform.extents, transform.output_strides,
                          transform.output_batch_stride, transform.output_orders)
    input_slots, output_slots = int(incoming.max()) + 9, int(outgoing.max()) + 9
    if transform.in_place:
        input_slots = output_slots = max(input_slots, output_slots)
    host_input = np.full(input_slots, sentinel, dtype=values.dtype)
    host_input[incoming] = values
    host_output = host_input.copy() if transform.in_place else np.full(output_slots, sentinel, dtype=values.dtype)
    device_input = torch.from_numpy(host_input).to(device="cuda", dtype=dtype)
    device_output = device_input if transform.in_place else torch.from_numpy(host_output).to(device="cuda", dtype=dtype)
    assert plan.input_size() <= input_slots * device_input.element_size()
    assert plan.output_size() <= output_slots * device_output.element_size()
    workspace = torch.empty(max(1, plan.workspace_size()), device="cuda", dtype=torch.uint8)
    plan.set_workspace(workspace, plan.workspace_size())
    plan.execute(device_input, device_output)
    torch.cuda.synchronize()
    actual = device_output.cpu()
    if dtype == torch.bfloat16:
        actual = actual.float()
    actual = actual.numpy()
    if dtype == torch.uint32:
        np.testing.assert_array_equal(actual[outgoing], expected)
    else:
        tolerance = {"fp64": 1e-10, "fp32": 3e-5, "fp16": .004, "bf16": .04}[precision]
        np.testing.assert_allclose(actual[outgoing], expected, atol=tolerance, rtol=tolerance)
    padding = np.ones(output_slots, dtype=bool)
    padding[outgoing] = False
    np.testing.assert_array_equal(actual[padding], host_output[padding])
    if not transform.in_place:
        remaining = device_input.cpu()
        if dtype == torch.bfloat16:
            remaining = remaining.float()
        np.testing.assert_array_equal(remaining.numpy(), host_input)


def _case_id(case):
    op, shape, inverse, orders, in_place, precision = case
    if op in ("NTT", "SUBSET_ZETA", "SUPERSET_ZETA"):
        precision = "uint32"
    return f"{op}-{precision}-{'x'.join(map(str, shape))}-{'inv' if inverse else 'fwd'}-o{orders}-{'ip' if in_place else 'oop'}"


@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_public_semantic_execution(runtime, case):
    api, library, torch, _, _ = runtime
    op, shape, inverse, orders, in_place, precision = case
    transform = _transform(api, op, shape, inverse, orders, in_place, precision)
    options = api.PlanOptions(policy=api.AlgorithmPolicy.DEFAULT)
    with api.Descriptor(library, transform, options) as descriptor:
        descriptor.validate()
        normalized = json.loads(descriptor.json())["semantics"]
        for field in ("input_orders", "output_orders", "input_strides", "output_strides",
                      "input_batch_stride", "output_batch_stride"):
            assert normalized[field] == getattr(transform, field)
        assert normalized["compute"] == int(transform.compute)
        assert normalized["storage"] == int(transform.storage)
        assert normalized["shape"] == list(shape)
        assert normalized["batch"] == 2
        assert normalized["inverse"] == int(inverse)
        assert normalized["placement"] == int(in_place)
        expected_normalization = bool(inverse and (
            op == "NTT" or (op in ("FFT", "FWHT") and transform.normalize_inverse)))
        assert normalized["normalize_inverse"] is expected_normalization
        assert json.loads(descriptor.json())["selection"]["algorithm_policy"] == int(api.AlgorithmPolicy.DEFAULT)
        if op == "NTT":
            assert normalized["modulus"] == 257
            assert normalized["word_bits"] == 32
        else:
            assert normalized["modulus"] is None
            assert normalized["word_bits"] is None
        if op == "STRUCTURED_2X2":
            assert normalized["matrices"] == [[[1.0, .25, -.5, 1.0]] for _ in shape]
        with api.Plan(library, descriptor, stream=torch.cuda.current_stream().cuda_stream) as plan:
            assert json.loads(plan.descriptor_json())["semantics"] == normalized
            mapping = plan.mapping_json()
            _execute_and_check(runtime, plan, transform, op, precision)
    with api.Plan(library, transform, api.PlanOptions(mapping=mapping),
                  stream=torch.cuda.current_stream().cuda_stream) as replay:
        assert json.loads(replay.descriptor_json())["semantics"] == normalized
        assert json.loads(replay.mapping_json()) == json.loads(mapping)
        _execute_and_check(runtime, replay, transform, op, precision)


@pytest.mark.parametrize("inverse", [False, True], ids=["forward", "inverse"])
def test_ntt_natural_to_reversed_keeps_native_path(runtime, inverse):
    api, library, torch, _, _ = runtime
    transform = _transform(api, "NTT", (8,), inverse, 0, False, "fp32", packed=True)
    with api.Descriptor(library, transform) as descriptor:
        normalized = json.loads(descriptor.json())["semantics"]
        assert normalized["normalize_inverse"] is inverse
        with api.Plan(library, descriptor, stream=torch.cuda.current_stream().cuda_stream) as plan:
            assert json.loads(plan.descriptor_json())["semantics"] == normalized
            assert "semantic-boundary" not in plan.algorithm_name()
            assert json.loads(plan.mapping_json()).get("kind") not in ("semantic-boundary", "composition", "xd")
            _execute_and_check(runtime, plan, transform, "NTT", "fp32")


@pytest.mark.parametrize("mutation", ["order", "stride", "precision"])
def test_mapping_rejects_changed_semantics(runtime, mutation):
    api, library, torch, _, _ = runtime
    transform = _transform(api, "FFT", (4, 8), False, 2, False, "fp32")
    with api.Plan(library, transform, stream=torch.cuda.current_stream().cuda_stream) as plan:
        mapping = plan.mapping_json()
    if mutation == "order":
        changed = replace(transform, input_orders=[1 - order for order in transform.input_orders])
    elif mutation == "stride":
        changed = replace(transform, input_strides=[stride * 2 for stride in transform.input_strides],
                          input_batch_stride=transform.input_batch_stride * 2)
    else:
        changed = replace(transform, storage=api.DataType.COMPLEX_FP64, compute=api.ComputeType.FP64)
    with api.Descriptor(library, changed) as descriptor:
        descriptor.validate()  # The changed semantics must themselves be valid.
    with pytest.raises(api.Error, match="(?i)mapping|contract|semantic"):
        with api.Plan(library, changed, api.PlanOptions(mapping=mapping),
                      stream=torch.cuda.current_stream().cuda_stream):
            pytest.fail("same mapping accepted a changed semantic contract")
