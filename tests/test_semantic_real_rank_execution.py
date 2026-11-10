"""Opt-in rank-4/rank-8 real FFT correctness and exact public mapping replay.

The CPU oracle uses every value of the actual quantized input. GPU execution
requires CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY; collection and
ordinary test runs do not allocate GPU memory or collect performance timings.
"""
import itertools
import json
import math

import pytest

from test_semantic_execution import runtime, _addresses, _layout


SHAPES = ((2, 4, 2, 8), (2, 2, 2, 2, 2, 2, 2, 8))
CASES = list(itertools.product(("R2C", "C2R"), ("fp32", "fp64"),
                               SHAPES, range(3), (False, True), (False, True)))


def _case_id(case):
    kind, precision, shape, order, strided, in_place = case
    return (f"{kind}-{precision}-rank{len(shape)}-orders{order}-"
            f"{'strided' if strided else 'compact'}-{'ip' if in_place else 'oop'}")


def _transform(api, case):
    kind, precision, shape, order, strided, in_place = case
    half = (*shape[:-1], shape[-1] // 2 + 1)
    input_shape, output_shape = (shape, half) if kind == "R2C" else (half, shape)
    if strided:
        incoming, ib = _layout(input_shape, 0)
        outgoing, ob = _layout(output_shape, 1)
    else:
        incoming = [math.prod(input_shape[axis + 1:]) for axis in range(len(shape))]
        outgoing = [math.prod(output_shape[axis + 1:]) for axis in range(len(shape))]
        ib, ob = math.prod(input_shape), math.prod(output_shape)
    if order == 0:
        input_orders = output_orders = [0] * len(shape)
    elif order == 1:
        input_orders = [1] * (len(shape) - 1) + [0]
        output_orders = [0] * len(shape)
    else:
        input_orders = [axis % 2 for axis in range(len(shape) - 1)] + [0]
        output_orders = [1 - flag for flag in input_orders[:-1]] + [0]
    # The Hermitian shortened axis has no bit-reversal interpretation.
    return api.Transform(
        op=api.Operator.FFT, storage=getattr(api.DataType, precision.upper()),
        compute=getattr(api.ComputeType, precision.upper()), extents=list(shape),
        batch=2, inverse=kind == "C2R", normalize_inverse=order != 1,
        in_place=in_place, fft_type=getattr(api.FftType, kind),
        fft_output_scale=-.5 if strided else .75,
        input_strides=incoming, output_strides=outgoing,
        input_batch_stride=ib, output_batch_stride=ob,
        input_orders=input_orders, output_orders=output_orders)


def _execute_and_check(runtime, plan, transform, precision):
    _, _, torch, np, _ = runtime
    shape = tuple(transform.extents)
    half = (*shape[:-1], shape[-1] // 2 + 1)
    inverse = transform.inverse
    input_shape, output_shape = (half, shape) if inverse else (shape, half)
    real_dtype = np.float64 if precision == "fp64" else np.float32
    complex_dtype = np.complex128 if precision == "fp64" else np.complex64
    real = (((np.arange(2 * math.prod(shape)) % 19) - 9) / 16).astype(real_dtype)
    real = real.reshape(2, *shape)
    axes = tuple(range(1, len(shape) + 1))
    if inverse:
        values = np.fft.rfftn(real, axes=axes).astype(complex_dtype)
        expected = np.fft.irfftn(values, s=shape, axes=axes)
        if not transform.normalize_inverse:
            expected *= math.prod(shape)
        output_dtype = real_dtype
    else:
        values = real
        expected = np.fft.rfftn(values, axes=axes)
        output_dtype = complex_dtype
    expected = (expected * transform.fft_output_scale).astype(output_dtype).reshape(-1)
    values = values.reshape(-1)
    incoming = _addresses(np, input_shape, transform.input_strides,
                          transform.input_batch_stride, transform.input_orders)
    outgoing = _addresses(np, output_shape, transform.output_strides,
                          transform.output_batch_stride, transform.output_orders)
    ib = (int(incoming.max()) + 9) * values.dtype.itemsize
    ob = (int(outgoing.max()) + 9) * expected.dtype.itemsize
    alignment = max(values.dtype.itemsize, expected.dtype.itemsize)
    round_size = lambda size: (size + alignment - 1) // alignment * alignment
    if transform.in_place:
        ib = ob = round_size(max(ib, ob))
    else:
        ib, ob = round_size(ib), round_size(ob)
    host_input = np.full(ib, 0xA5, dtype=np.uint8)
    host_input.view(values.dtype)[incoming] = values
    host_output = host_input.copy() if transform.in_place else np.full(ob, 0xA5, dtype=np.uint8)
    device_input = torch.from_numpy(host_input).to(device="cuda")
    device_output = device_input if transform.in_place else torch.from_numpy(host_output).to(device="cuda")
    assert plan.input_size() <= ib and plan.output_size() <= ob
    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
    plan.set_workspace(workspace, plan.workspace_size())
    plan.execute(device_input, device_output)
    torch.cuda.synchronize()
    actual = device_output.cpu().numpy()
    tolerance = 1e-10 if precision == "fp64" else 3e-5
    np.testing.assert_allclose(actual.view(expected.dtype)[outgoing], expected,
                               atol=tolerance, rtol=tolerance)
    touched = (outgoing[:, None] * expected.dtype.itemsize +
               np.arange(expected.dtype.itemsize, dtype=np.int64)).reshape(-1)
    padding = np.ones(ob, dtype=bool)
    padding[touched] = False
    np.testing.assert_array_equal(actual[padding], host_output[padding])
    if not transform.in_place:
        np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_public_high_rank_real_semantic_execution(runtime, case):
    api, library, torch, _, _ = runtime
    transform = _transform(api, case)
    precision = case[1]
    stream = torch.cuda.current_stream().cuda_stream
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantic = json.loads(descriptor.json())["semantics"]
        with api.Plan(library, descriptor, stream=stream) as plan:
            mapping = plan.mapping_json()
            assert json.loads(plan.descriptor_json())["semantics"] == semantic
            _execute_and_check(runtime, plan, transform, precision)
    with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
        assert json.loads(replay.mapping_json()) == json.loads(mapping)
        assert json.loads(replay.descriptor_json())["semantics"] == semantic
        _execute_and_check(runtime, replay, transform, precision)
