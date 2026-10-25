"""Opt-in exact-length rank-3 FFT lowering, boundary and replay diagnostics.

The NumPy oracle uses all quantized input elements. Full floating error arrays
are retained; finite execution is not approval of a floating acceptance limit.
"""
import itertools
import json
import math

import pytest

from test_semantic_execution import runtime, _addresses, _layout


CASES = [(shape, precision, inverse, normalized, padded, in_place, order)
         for shape in ((3, 5, 7), (3, 4, 5))
         for precision, inverse, normalized, padded, in_place in itertools.product(
             ("fp32", "fp64"), (False, True), (False, True), (False, True), (False, True))
         for order in (range(3) if shape[1] == 4 else (0,))]


def _case_id(case):
    shape, precision, inverse, normalized, padded, in_place, order = case
    return (f"{'x'.join(map(str, shape))}-{precision}-inv{int(inverse)}-"
            f"norm{int(normalized)}-pad{int(padded)}-ip{int(in_place)}-order{order}")


def _transform(api, case):
    shape, precision, inverse, normalized, padded, in_place, order = case
    if padded:
        incoming, ib = _layout(shape, 0)
        outgoing, ob = _layout(shape, 1)
    else:
        incoming = [math.prod(shape[axis + 1:]) for axis in range(len(shape))]
        outgoing = list(incoming)
        ib = ob = math.prod(shape)
    input_orders = [0, int(order == 1), 0]
    output_orders = [0, int(order == 2), 0]
    return api.Transform(storage=getattr(api.DataType, "COMPLEX_" + precision.upper()),
                         compute=getattr(api.ComputeType, precision.upper()),
                         extents=list(shape), batch=2, inverse=inverse,
                         normalize_inverse=normalized, in_place=in_place,
                         input_strides=incoming, output_strides=outgoing,
                         input_batch_stride=ib, output_batch_stride=ob,
                         input_orders=input_orders, output_orders=output_orders)


def _execute(runtime, plan, transform, precision, artifact):
    _, _, torch, np, _ = runtime
    shape = tuple(transform.extents)
    dtype = np.complex64 if precision == "fp32" else np.complex128
    count = 2 * math.prod(shape)
    real = ((np.arange(count) * 7 % 31) - 15) / 32
    imag = ((np.arange(count) * 11 % 29) - 14) / 64
    source = (real + 1j * imag).astype(dtype).reshape(2, *shape)
    axes = tuple(range(1, len(shape) + 1))
    expected = (np.fft.ifftn(source, axes=axes) if transform.inverse
                else np.fft.fftn(source, axes=axes))
    if transform.inverse and not transform.normalize_inverse:
        expected *= math.prod(shape)
    expected = expected.reshape(-1)
    source = source.reshape(-1)
    incoming = _addresses(np, shape, transform.input_strides,
                          transform.input_batch_stride, transform.input_orders)
    outgoing = _addresses(np, shape, transform.output_strides,
                          transform.output_batch_stride, transform.output_orders)
    ni, no = int(incoming.max()) + 9, int(outgoing.max()) + 9
    if transform.in_place:
        ni = no = max(ni, no)
    host_input = np.full(ni, complex(73, -19), dtype=dtype)
    host_input[incoming] = source
    host_output = host_input.copy() if transform.in_place else np.full(no, complex(73, -19), dtype=dtype)
    device_input = torch.from_numpy(host_input).to("cuda")
    device_output = device_input if transform.in_place else torch.from_numpy(host_output).to("cuda")
    assert plan.input_size() <= host_input.nbytes
    assert plan.output_size() <= host_output.nbytes
    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
    plan.set_workspace(workspace, plan.workspace_size())
    plan.execute(device_input, device_output)
    torch.cuda.synchronize()
    raw = device_output.cpu().numpy()
    padding = np.ones(no, dtype=bool)
    padding[outgoing] = False
    np.testing.assert_array_equal(raw[padding], host_output[padding])
    if not transform.in_place:
        np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)
    actual = raw[outgoing].astype(np.complex128)
    absolute_error = np.abs(actual - expected)
    relative_error = absolute_error / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    np.savez_compressed(artifact, input=source, reference=expected, actual=actual,
                        absolute_error=absolute_error, relative_error=relative_error)
    assert np.all(np.isfinite(actual)), f"nonfinite output; full errors at {artifact}"
    denominator = float(np.linalg.norm(expected))
    return {"status": "accuracy-review-required", "elements": int(actual.size),
            "artifact": str(artifact), "l2_relative_error": float(np.linalg.norm(actual - expected) / denominator),
            "absolute_error_percentiles": np.percentile(absolute_error, [0, 50, 90, 99, 100]).tolist(),
            "relative_error_percentiles": np.percentile(relative_error, [0, 50, 90, 99, 100]).tolist()}


@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_public_arbitrary_rank_fft_execution(runtime, case, tmp_path):
    api, library, torch, np, _ = runtime
    transform = _transform(api, case)
    stream = torch.cuda.current_stream().cuda_stream
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantic = json.loads(descriptor.json())["semantics"]
        with api.Plan(library, descriptor, stream=stream) as plan:
            mapping = plan.mapping_json()
            assert json.loads(plan.descriptor_json())["semantics"] == semantic
            original = _execute(runtime, plan, transform, case[1], tmp_path / "original.npz")
    with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
        assert json.loads(replay.mapping_json()) == json.loads(mapping)
        assert json.loads(replay.descriptor_json())["semantics"] == semantic
        repeated = _execute(runtime, replay, transform, case[1], tmp_path / "replay.npz")
    with np.load(tmp_path / "original.npz") as original_arrays, np.load(tmp_path / "replay.npz") as replay_arrays:
        np.testing.assert_array_equal(replay_arrays["actual"], original_arrays["actual"])
    (tmp_path / "diagnostics.json").write_text(json.dumps({"original": original, "replay": repeated}, indent=2))
