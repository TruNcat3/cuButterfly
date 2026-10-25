"""Opt-in public execution tests for numeric lowerings added in 0.9.3.

Integer results are exact modulo 2**64. Floating results retain full reference,
actual and error arrays and report distributions; passing these diagnostics
does not constitute acceptance of an unapproved numerical tolerance.
"""
import importlib.util
import itertools
import json
import math
from pathlib import Path

import pytest

from test_semantic_execution import runtime, _addresses, _layout


ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((8,), (4, 8), (2, 2, 2, 4))
CASES = [("uint64", op, "uint64", "UINT64", shape, inverse, variant)
         for op, shape, inverse, variant in itertools.product(
             ("SUBSET_ZETA", "SUPERSET_ZETA"), SHAPES, (False, True), range(2))]
CASES += [("mixed", op, "fp32", "FP64", shape, inverse, variant)
          for op, shape, inverse, variant in itertools.product(
              ("FFT", "FWHT", "STRUCTURED_2X2"), SHAPES, (False, True), range(2))]
CASES += [("real", "C2R" if inverse else "R2C", precision, compute, shape, inverse, variant)
          for precision, compute, shape, inverse, variant in itertools.product(
              ("fp16", "bf16"), ("NATIVE", "FP32"), SHAPES, (False, True), range(2))]


def _case_id(case):
    family, op, precision, compute, shape, inverse, variant = case
    return (f"{family}-{op}-{precision}-{compute}-rank{len(shape)}-"
            f"{'inverse' if inverse else 'forward'}-"
            f"{'padded-mixed-ip' if variant else 'packed-natural-oop'}")


@pytest.fixture(scope="module")
def numeric_helpers():
    spec = importlib.util.spec_from_file_location(
        "numeric_execution_helpers", ROOT / "scripts/run_semantic_campaign.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transform(api, case):
    family, op, precision, compute, shape, inverse, variant = case
    real_fft = family == "real"
    half = (*shape[:-1], shape[-1] // 2 + 1)
    input_shape = half if real_fft and inverse else shape
    output_shape = half if real_fft and not inverse else shape
    if variant:
        ins, ib = _layout(input_shape, 0)
        outs, ob = _layout(output_shape, 1)
        incoming = [axis % 2 for axis in range(len(shape))]
        outgoing = [1 - flag for flag in incoming]
    else:
        ins = [math.prod(input_shape[axis + 1:]) for axis in range(len(shape))]
        outs = [math.prod(output_shape[axis + 1:]) for axis in range(len(shape))]
        ib, ob = math.prod(input_shape), math.prod(output_shape)
        incoming = outgoing = [0] * len(shape)
    if real_fft:
        incoming[-1] = outgoing[-1] = 0
    storage = ("COMPLEX_FP32" if family == "mixed" and op == "FFT"
               else precision.upper())
    kwargs = dict(op=api.Operator.FFT if real_fft else getattr(api.Operator, op),
                  storage=getattr(api.DataType, storage),
                  compute=getattr(api.ComputeType, compute), extents=list(shape),
                  batch=2, inverse=inverse, normalize_inverse=True,
                  in_place=bool(variant), input_strides=ins, output_strides=outs,
                  input_batch_stride=ib, output_batch_stride=ob,
                  input_orders=list(incoming), output_orders=list(outgoing))
    if real_fft:
        kwargs.update(fft_type=getattr(api.FftType, op), fft_output_scale=.5)
    if op == "STRUCTURED_2X2":
        kwargs["stage_matrices"] = [[(1.0, .25, -.5, 1.0)] for _ in shape]
    return api.Transform(**kwargs), input_shape, output_shape


def _zeta_oracle(np, values, inverse, superset):
    """Direct subset sums with Python integers, independent of GPU stages."""
    rows = values.reshape(2, -1)
    output = []
    mask64 = (1 << 64) - 1
    for row in rows:
        for target in range(row.size):
            total = 0
            for source in range(row.size):
                selected = ((source & target) == target if superset
                            else (source & target) == source)
                if selected:
                    sign = -1 if inverse and (source ^ target).bit_count() % 2 else 1
                    total += sign * int(row[source])
            output.append(total & mask64)
    return np.asarray(output, dtype=np.uint64)


def _reference(runtime, helpers, case, transform):
    _, _, _, np, _ = runtime
    family, op, _, _, shape, inverse, _ = case
    count = 2 * math.prod(shape)
    if family == "uint64":
        values = np.asarray([((1 << 64) - 1 - index * 7919) if index % 2
                             else index * 104729 for index in range(count)], dtype=np.uint64)
        return values, _zeta_oracle(np, values, inverse, op == "SUPERSET_ZETA"), 9, 9
    values = (((np.arange(count) % 17) - 8) / 128).reshape(2, *shape)
    axes = tuple(range(1, len(shape) + 1))
    if family == "real":
        storage = int(transform.storage)
        if inverse:
            values = np.fft.rfftn(values, axes=axes)
            input_storage, output_storage = storage + 4, storage
        else:
            input_storage, output_storage = storage, storage + 4
        quantized = helpers.encode_storage(values, input_storage)
        source = helpers.decode_storage(quantized, input_storage)
        expected = (np.fft.irfftn(source, s=shape, axes=axes) if inverse
                    else np.fft.rfftn(source, axes=axes)) * transform.fft_output_scale
        return quantized.reshape(-1), expected.reshape(-1), input_storage, output_storage
    storage = int(transform.storage)
    if op == "FFT":
        values = (values + 1j * np.roll(values, 3)).astype(np.complex64)
        expected = (np.fft.ifftn(values, axes=axes) if inverse
                    else np.fft.fftn(values, axes=axes)).astype(np.complex64)
    elif op == "FWHT":
        values = values.astype(np.float32)
        rows = values.reshape(2, -1)
        expected = np.asarray([[sum(float(row[j]) * (-1 if (i & j).bit_count() % 2 else 1)
                                    for j in range(row.size))
                                for i in range(row.size)] for row in rows], dtype=np.float64)
        if inverse:
            expected /= math.prod(shape)
        expected = expected.astype(np.float32)
    else:
        values = values.astype(np.float32)
        expected = values.astype(np.float64)
        matrix = np.asarray(transform.stage_matrices[0][0], dtype=np.float64).reshape(2, 2)
        if inverse:
            matrix = np.linalg.inv(matrix)
        for axis, extent in enumerate(shape, 1):
            moved = np.moveaxis(expected, axis, -1).copy()
            half = 1
            while half < extent:
                for start in range(0, extent, 2 * half):
                    a = moved[..., start:start + half].copy()
                    b = moved[..., start + half:start + 2 * half].copy()
                    moved[..., start:start + half] = matrix[0, 0] * a + matrix[0, 1] * b
                    moved[..., start + half:start + 2 * half] = matrix[1, 0] * a + matrix[1, 1] * b
                half *= 2
            expected = np.moveaxis(moved, -1, axis)
        expected = expected.astype(np.float32)
    return values.reshape(-1), expected.reshape(-1), storage, storage


def _execute(runtime, helpers, plan, transform, input_shape, output_shape, case, artifact):
    _, _, torch, np, _ = runtime
    source, reference, _, output_storage = _reference(runtime, helpers, case, transform)
    output_dtype = helpers.DTYPES[output_storage]
    incoming = _addresses(np, input_shape, transform.input_strides,
                          transform.input_batch_stride, transform.input_orders)
    outgoing = _addresses(np, output_shape, transform.output_strides,
                          transform.output_batch_stride, transform.output_orders)
    ib = (int(incoming.max()) + 9) * source.dtype.itemsize
    ob = (int(outgoing.max()) + 9) * output_dtype.itemsize
    alignment = max(source.dtype.itemsize, output_dtype.itemsize)
    round_size = lambda size: (size + alignment - 1) // alignment * alignment
    if transform.in_place:
        ib = ob = round_size(max(ib, ob))
    else:
        ib, ob = round_size(ib), round_size(ob)
    host_input = np.full(ib, 0xA5, dtype=np.uint8)
    host_input.view(source.dtype)[incoming] = source
    host_output = host_input.copy() if transform.in_place else np.full(ob, 0xA5, dtype=np.uint8)
    device_input = torch.from_numpy(host_input).to("cuda")
    device_output = device_input if transform.in_place else torch.from_numpy(host_output).to("cuda")
    assert plan.input_size() <= ib and plan.output_size() <= ob
    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
    if plan.workspace_size():
        plan.set_workspace(workspace, plan.workspace_size())
    plan.execute(device_input, device_output)
    torch.cuda.synchronize()
    raw = device_output.cpu().numpy()
    actual = helpers.decode_storage(raw.view(output_dtype)[outgoing], output_storage)
    touched = (outgoing[:, None] * output_dtype.itemsize +
               np.arange(output_dtype.itemsize, dtype=np.int64)).reshape(-1)
    padding = np.ones(ob, dtype=bool)
    padding[touched] = False
    np.testing.assert_array_equal(raw[padding], host_output[padding])
    if not transform.in_place:
        np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)
    if case[0] == "uint64":
        np.testing.assert_array_equal(actual, reference)
        return {"status": "exact-pass", "elements": int(actual.size)}
    actual = actual.astype(np.complex128 if np.iscomplexobj(actual) else np.float64)
    reference = reference.astype(actual.dtype)
    errors = helpers.error_arrays(actual, reference)
    np.savez_compressed(artifact, **errors)
    assert np.all(np.isfinite(actual)), f"nonfinite output; full errors at {artifact}"
    denominator = float(np.linalg.norm(reference))
    return {"status": "accuracy-review-required", "artifact": str(artifact),
            "elements": int(actual.size), "l2_relative_error":
            float(np.linalg.norm(actual - reference)) / denominator if denominator else None,
            "absolute_error_distribution": helpers.distribution(errors["abs_error"])}


@pytest.mark.parametrize("case", CASES, ids=[_case_id(case) for case in CASES])
def test_public_numeric_lowering_execution(runtime, numeric_helpers, case, tmp_path, record_property):
    api, library, torch, _, _ = runtime
    transform, input_shape, output_shape = _transform(api, case)
    stream = torch.cuda.current_stream().cuda_stream
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantics = json.loads(descriptor.json())["semantics"]
        with api.Plan(library, descriptor, stream=stream) as plan:
            mapping = plan.mapping_json()
            assert json.loads(plan.descriptor_json())["semantics"] == semantics
            first = _execute(runtime, numeric_helpers, plan, transform, input_shape, output_shape,
                             case, tmp_path / "selected-errors.npz")
    with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
        assert json.loads(replay.mapping_json()) == json.loads(mapping)
        assert json.loads(replay.descriptor_json())["semantics"] == semantics
        second = _execute(runtime, numeric_helpers, replay, transform, input_shape, output_shape,
                          case, tmp_path / "replayed-errors.npz")
    record_property("numeric_diagnostics", json.dumps({"selected": first, "replayed": second}))


def test_uint64_oracle_wraparound_and_mobius():
    np = pytest.importorskip("numpy")
    mask = (1 << 64) - 1
    values = np.asarray([mask, 1, 2, 3, mask, 1, 2, 3], dtype=np.uint64)
    np.testing.assert_array_equal(_zeta_oracle(np, values, False, False),
                                  np.asarray([mask, 0, 1, 5] * 2, dtype=np.uint64))
    np.testing.assert_array_equal(_zeta_oracle(np, values, False, True),
                                  np.asarray([5, 4, 5, 3] * 2, dtype=np.uint64))
    for superset in (False, True):
        transformed = _zeta_oracle(np, values, False, superset)
        np.testing.assert_array_equal(_zeta_oracle(np, transformed, True, superset), values)


def test_cpu_inputs_references_and_byte_layouts(numeric_helpers, monkeypatch):
    """Exercise every storage adapter and address combination without CUDA."""
    np = pytest.importorskip("numpy")
    monkeypatch.syspath_prepend(str(ROOT / "python"))
    import cubutterfly as api
    cpu = api, None, None, np, None
    for case in CASES:
        transform, incoming_shape, outgoing_shape = _transform(api, case)
        values, expected, input_storage, output_storage = _reference(
            cpu, numeric_helpers, case, transform)
        assert values.size == 2 * math.prod(incoming_shape), _case_id(case)
        assert expected.size == 2 * math.prod(outgoing_shape), _case_id(case)
        assert values.dtype == numeric_helpers.DTYPES[input_storage], _case_id(case)
        assert np.all(np.isfinite(numeric_helpers.decode_storage(values, input_storage)))
        assert np.all(np.isfinite(expected))
        encoded_reference = numeric_helpers.encode_storage(expected, output_storage)
        assert np.all(np.isfinite(numeric_helpers.decode_storage(encoded_reference, output_storage)))
        for shape, strides, batch_stride, orders in (
                (incoming_shape, transform.input_strides, transform.input_batch_stride,
                 transform.input_orders),
                (outgoing_shape, transform.output_strides, transform.output_batch_stride,
                 transform.output_orders)):
            addresses = _addresses(np, shape, strides, batch_stride, orders)
            assert addresses.size == 2 * math.prod(shape)
