"""Opt-in real FFT semantic boundaries, including byte-shared in-place I/O.

Uses the CUBUTTERFLY_GPU_TESTS/CUBUTTERFLY_TEST_LIBRARY runtime fixture. Import
and collection perform no GPU work. Every case checks full logical output,
untouched output bytes, and exact mapping replay; no timings are collected.
"""
import itertools
import json
import math

import pytest

from test_semantic_execution import runtime, _addresses, _layout


CASES = list(itertools.product(("R2C", "C2R"), ("fp32", "fp64"),
                               ((8,), (4, 8)), (False, True)))


def _case_id(case):
    kind, precision, shape, in_place = case
    return f"{kind}-{precision}-{'x'.join(map(str, shape))}-{'ip' if in_place else 'oop'}"


def _transform(api, kind, precision, shape, in_place):
    half = (*shape[:-1], shape[-1] // 2 + 1)
    input_shape, output_shape = (shape, half) if kind == "R2C" else (half, shape)
    incoming, ib = _layout(input_shape, 0)
    outgoing, ob = _layout(output_shape, 1)
    # The shortened axis is natural; the leading full axis exercises the
    # semantic order adapter on both real and complex element types.
    orders = [1, 0] if len(shape) == 2 else [0]
    return api.Transform(
        op=api.Operator.FFT, storage=getattr(api.DataType, precision.upper()),
        compute=getattr(api.ComputeType, precision.upper()), extents=list(shape),
        batch=2, inverse=kind == "C2R", normalize_inverse=True, in_place=in_place,
        fft_type=getattr(api.FftType, kind), fft_output_scale=.75,
        input_strides=incoming, output_strides=outgoing,
        input_batch_stride=ib, output_batch_stride=ob,
        input_orders=orders, output_orders=list(orders))


def _values(runtime, kind, precision, shape):
    _, _, torch, np, _ = runtime
    real_dtype = torch.float64 if precision == "fp64" else torch.float32
    real = ((torch.arange(2 * math.prod(shape), dtype=torch.float64) % 19) - 9) / 16
    real = real.to(real_dtype).reshape(2, *shape)
    axes = tuple(range(1, len(shape) + 1))
    if kind == "R2C":
        values = real
        expected = torch.fft.rfftn(real, dim=axes) * .75
    else:
        # A valid Hermitian half spectrum ensures that C2R's real-valued
        # interpretation is fully specified even at DC/Nyquist boundaries.
        values = torch.fft.rfftn(real, dim=axes)
        expected = torch.fft.irfftn(values, s=shape, dim=axes) * .75
    return np.ascontiguousarray(values.numpy()).reshape(-1), expected.numpy().reshape(-1)


def _execute_and_check(runtime, plan, transform, kind, precision):
    _, _, torch, np, _ = runtime
    shape = tuple(transform.extents)
    half = (*shape[:-1], shape[-1] // 2 + 1)
    input_shape, output_shape = (shape, half) if kind == "R2C" else (half, shape)
    values, expected = _values(runtime, kind, precision, shape)
    incoming = _addresses(np, input_shape, transform.input_strides,
                          transform.input_batch_stride, transform.input_orders)
    outgoing = _addresses(np, output_shape, transform.output_strides,
                          transform.output_batch_stride, transform.output_orders)
    input_element_bytes, output_element_bytes = values.dtype.itemsize, expected.dtype.itemsize
    input_bytes = (int(incoming.max()) + 9) * input_element_bytes
    output_bytes = (int(outgoing.max()) + 9) * output_element_bytes
    # Byte allocations must support views of either scalar or complex dtype
    # when both plan sides point to the same allocation.
    alignment = max(input_element_bytes, output_element_bytes)
    round_size = lambda size: (size + alignment - 1) // alignment * alignment
    if transform.in_place:
        input_bytes = output_bytes = round_size(max(input_bytes, output_bytes))
    else:
        input_bytes, output_bytes = round_size(input_bytes), round_size(output_bytes)
    host_input = np.full(input_bytes, 0xA5, dtype=np.uint8)
    host_input.view(values.dtype)[incoming] = values
    host_output = host_input.copy() if transform.in_place else np.full(output_bytes, 0xA5, dtype=np.uint8)
    device_input = torch.from_numpy(host_input).to(device="cuda")
    device_output = device_input if transform.in_place else torch.from_numpy(host_output).to(device="cuda")
    assert device_input.dtype == torch.uint8 and device_output.dtype == torch.uint8
    assert plan.input_size() <= input_bytes
    assert plan.output_size() <= output_bytes
    if transform.in_place:
        assert device_input.data_ptr() == device_output.data_ptr()
        assert input_element_bytes != output_element_bytes
    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
    plan.set_workspace(workspace, plan.workspace_size())
    plan.execute(device_input, device_output)
    torch.cuda.synchronize()
    actual = device_output.cpu().numpy()
    tolerance = 1e-10 if precision == "fp64" else 3e-5
    np.testing.assert_allclose(actual.view(expected.dtype)[outgoing], expected,
                               atol=tolerance, rtol=tolerance)
    touched = (outgoing[:, None] * output_element_bytes +
               np.arange(output_element_bytes, dtype=np.int64)).reshape(-1)
    padding = np.ones(output_bytes, dtype=bool)
    padding[touched] = False
    np.testing.assert_array_equal(actual[padding], host_output[padding])
    if not transform.in_place:
        np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)


@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_public_real_semantic_execution(runtime, case):
    api, library, torch, _, _ = runtime
    kind, precision, shape, in_place = case
    transform = _transform(api, kind, precision, shape, in_place)
    stream = torch.cuda.current_stream().cuda_stream
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantic = json.loads(descriptor.json())["semantics"]
        for name in ("input_strides", "output_strides", "input_batch_stride",
                     "output_batch_stride", "input_orders", "output_orders"):
            assert semantic[name] == getattr(transform, name)
        assert semantic["fft_type"] == int(transform.fft_type)
        assert semantic["fft_output_scale"] == .75
        assert semantic["placement"] == int(in_place)
        assert semantic["inverse"] == int(kind == "C2R")
        with api.Plan(library, descriptor, stream=stream) as plan:
            assert json.loads(plan.descriptor_json())["semantics"] == semantic
            mapping = plan.mapping_json()
            _execute_and_check(runtime, plan, transform, kind, precision)
    with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
        assert json.loads(replay.mapping_json()) == json.loads(mapping)
        assert json.loads(replay.descriptor_json())["semantics"] == semantic
        _execute_and_check(runtime, replay, transform, kind, precision)
