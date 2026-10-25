"""Opt-in low-storage inverse normalization and overflow regressions.

Floating diagnostics retain complete actual/reference/error arrays without an
accuracy threshold. Only analytically exact constant/impulse cases use equality.
Run with CUBUTTERFLY_GPU_TESTS=1 and CUBUTTERFLY_TEST_LIBRARY=/path/libcuntt.so;
set CUBUTTERFLY_LOW_NORMALIZATION_ERRORS to retain artifacts outside pytest temp.
"""
import itertools
import json
import math
import os
from pathlib import Path

import pytest

from test_semantic_execution import runtime
from test_semantic_numeric_execution import numeric_helpers


def _case(op, precision, compute, shape, batch=1, profile="constant",
          inverse=True, normalize=True, split=False):
    return (op, precision, compute, shape, batch, profile, inverse, normalize, split)


CASES = [_case(op, precision, compute, (1 << log_n,))
         for op, precision, compute, log_n in itertools.product(
             ("FFT", "FWHT"), ("FP16", "BF16"), ("NATIVE", "FP32"), (12, 16, 18))]
CASES += [_case(op, precision, "NATIVE", (256, 256))
          for op, precision in itertools.product(("FFT", "FWHT"), ("FP16", "BF16"))]
CASES += [_case("C2R", precision, compute, (65536,), batch, "sine")
          for precision, compute, batch in itertools.product(
              ("FP16", "BF16"), ("NATIVE", "FP32"), (1, 16))]
CASES += [_case("C2R", precision, "NATIVE", (1 << log_n,))
          for precision, log_n in itertools.product(("FP16", "BF16"), (12, 18))]
CASES += [_case(op, precision, compute, (4096,), profile="impulse",
                inverse=inverse, normalize=not inverse)
          for op, precision, compute, inverse in itertools.product(
              ("FFT", "FWHT"), ("FP16", "BF16"), ("NATIVE", "FP32"), (False, True))]
CASES += [_case("C2R", precision, compute, (4096,), profile="impulse", normalize=False)
          for precision, compute in itertools.product(("FP16", "BF16"), ("NATIVE", "FP32"))]
CASES += [_case(op, precision, "NATIVE", (65536,), split=True)
          for op, precision in itertools.product(("FFT", "FWHT"), ("FP16", "BF16"))]


def _case_id(case):
    op, precision, compute, shape, batch, profile, inverse, normalize, split = case
    return (f"{op}-{precision}-{compute}-{'x'.join(map(str, shape))}-b{batch}-"
            f"{profile}-{'inverse' if inverse else 'forward'}-"
            f"{'normalized' if normalize else 'unscaled'}-{'split8x8' if split else 'default'}")


def _transform(api, case):
    op, precision, compute, shape, batch, _, inverse, normalize, _ = case
    real_fft = op == "C2R"
    return api.Transform(
        op=api.Operator.FWHT if op == "FWHT" else api.Operator.FFT,
        storage=getattr(api.DataType, ("COMPLEX_" if op == "FFT" else "") + precision),
        compute=getattr(api.ComputeType, compute), extents=list(shape), batch=batch,
        inverse=inverse, normalize_inverse=normalize,
        fft_type=api.FftType.C2R if real_fft else api.FftType.C2C,
        input_orders=[0] * len(shape), output_orders=[0] * len(shape))


def _source_and_reference(np, helpers, case, semantics):
    op, _, _, shape, batch, profile, inverse, normalize, _ = case
    n = math.prod(shape)
    source_shape = (batch, *helpers.side_extents(semantics, "input"))
    complex_input = op in ("FFT", "C2R")
    dtype = np.complex128 if complex_input else np.float64
    if profile == "sine":
        signal = np.sin(2 * np.pi * 7 * np.arange(n, dtype=np.float64) / n)
        values = np.broadcast_to(np.fft.rfft(signal), source_shape).copy()
    elif profile == "constant":
        values = np.full(source_shape, 32768, dtype=dtype)
    else:
        values = np.zeros(source_shape, dtype=dtype)
        values.reshape(batch, -1)[:, 0] = 1
    input_storage = helpers.side_storage(semantics, "input")
    quantized = helpers.decode_storage(helpers.encode_storage(values, input_storage), input_storage)
    assert np.all(np.isfinite(quantized))
    if profile == "sine":
        reference = np.fft.irfft(quantized.astype(np.complex128), n=n, axis=-1)
    elif profile == "constant":
        reference = np.zeros((batch, *shape), dtype=dtype if op == "FFT" else np.float64)
        reference.reshape(batch, -1)[:, 0] = 32768
    else:
        scale = 1 / n if inverse and normalize else 1
        reference = np.full((batch, *shape), scale, dtype=dtype if op == "FFT" else np.float64)
    assert np.all(np.isfinite(reference))
    return quantized, reference.reshape(-1), profile != "sine"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_low_inverse_normalization(runtime, numeric_helpers, case, tmp_path, record_property):
    api, library, torch, np, _ = runtime
    helpers = numeric_helpers
    transform = _transform(api, case)
    options = api.PlanOptions()
    if case[-1]:
        options.mapping = json.dumps({
            "schema_version": 1, "kind": "butterfly", "backend": "shared-iterative",
            "compute_unit": "radix2", "stage_partition": [8, 8], "tile_threads": 128})
    stream = torch.cuda.current_stream().cuda_stream
    directory = Path(os.environ.get("CUBUTTERFLY_LOW_NORMALIZATION_ERRORS", str(tmp_path)))
    diagnostics = {}

    with api.Descriptor(library, transform, options) as descriptor:
        semantics = json.loads(descriptor.json())["semantics"]
        assert semantics["compute"] == int(transform.compute)
        assert semantics["inverse"] == int(transform.inverse)
        assert semantics["normalize_inverse"] == bool(transform.inverse and transform.normalize_inverse)
        source, reference, exact = _source_and_reference(np, helpers, case, semantics)
        input_bytes = source.size * helpers.side_dtype(semantics, "input").itemsize
        output_bytes = reference.size * helpers.side_dtype(semantics, "output").itemsize
        host_input = helpers.physical_bytes(semantics, "input", source, input_bytes)
        host_output = np.full(output_bytes, 0xA5, dtype=np.uint8)
        device_input = torch.from_numpy(host_input).to("cuda")
        device_output = torch.from_numpy(host_output).to("cuda")

        def execute(plan, label):
            assert json.loads(plan.descriptor_json())["semantics"] == semantics
            assert plan.input_size() == input_bytes and plan.output_size() == output_bytes
            workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device="cuda")
            plan.set_workspace(workspace, plan.workspace_size())
            plan.execute(device_input, device_output)
            torch.cuda.synchronize()
            actual = helpers.output_from_bytes(semantics, device_output.cpu().numpy())
            actual = actual.astype(np.complex128 if np.iscomplexobj(reference) else np.float64)
            artifact = helpers.retain_error_artifact(directory, _case_id(case), label, actual, reference)
            diagnostics[label] = {"artifact": artifact, "exact_reference": exact}
            np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)
            assert np.all(np.isfinite(actual)), f"nonfinite output; full errors at {artifact['path']}"
            if exact:
                np.testing.assert_array_equal(actual, reference)
                diagnostics[label]["status"] = "exact-pass"
            else:
                diagnostics[label].update(helpers.check_full(
                    actual, reference, helpers.side_storage(semantics, "output"),
                    float_verdict="error-report"))

        try:
            with api.Plan(library, descriptor, stream=stream) as plan:
                mapping = plan.mapping_json()
                execute(plan, "selected")
            with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
                assert json.loads(replay.mapping_json()) == json.loads(mapping)
                execute(replay, "replayed")
        finally:
            record_property("normalization_diagnostics", json.dumps(diagnostics))


LOW_AMPLITUDE_CASES = list(itertools.product(
    ("FFT", "FWHT"), ("FP16", "BF16"), ("NATIVE", "FP32"),
    (2, 256), ("min-subnormal", "min-normal")))


@pytest.mark.parametrize("op,precision,compute,n,magnitude", LOW_AMPLITUDE_CASES,
                         ids=["-".join(map(str, case)) for case in LOW_AMPLITUDE_CASES])
def test_low_inverse_normalization_low_amplitude(
        runtime, numeric_helpers, op, precision, compute, n, magnitude, tmp_path, record_property):
    api, library, torch, np, _ = runtime
    helpers = numeric_helpers
    case = _case(op, precision, compute, (n,))
    transform = _transform(api, case)
    exponent = {"FP16": {"min-subnormal": -24, "min-normal": -14},
                "BF16": {"min-subnormal": -133, "min-normal": -126}}[precision][magnitude]
    amplitude = math.ldexp(1.0, exponent)
    dtype = np.complex128 if op == "FFT" else np.float64
    directory = Path(os.environ.get("CUBUTTERFLY_LOW_NORMALIZATION_ERRORS", str(tmp_path)))
    case_id = _case_id(case) + "-" + magnitude
    diagnostics = {"input_amplitude": amplitude, "input_profile": magnitude,
                   "verdict": "accuracy-review-required", "timing": False}
    stream = torch.cuda.current_stream().cuda_stream

    with api.Descriptor(library, transform) as descriptor:
        semantics = json.loads(descriptor.json())["semantics"]
        assert semantics["compute"] == int(transform.compute)
        assert semantics["inverse"] == 1 and semantics["normalize_inverse"] is True
        values = np.full((1, n), amplitude, dtype=dtype)
        input_storage = helpers.side_storage(semantics, "input")
        source = helpers.decode_storage(helpers.encode_storage(values, input_storage), input_storage)
        np.testing.assert_array_equal(source, values)
        reference = np.zeros(n, dtype=dtype)
        reference[0] = amplitude
        input_bytes = source.size * helpers.side_dtype(semantics, "input").itemsize
        output_bytes = reference.size * helpers.side_dtype(semantics, "output").itemsize
        host_input = helpers.physical_bytes(semantics, "input", source, input_bytes)
        device_input = torch.from_numpy(host_input).to("cuda")
        device_output = torch.full((output_bytes,), 0xA5, dtype=torch.uint8, device="cuda")

        def execute(plan, label):
            assert json.loads(plan.descriptor_json())["semantics"] == semantics
            assert plan.input_size() == input_bytes and plan.output_size() == output_bytes
            workspace_bytes = plan.workspace_size()
            workspace = None
            if workspace_bytes:
                workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
                plan.set_workspace(workspace, workspace_bytes)
            device_output.fill_(0xA5)
            plan.execute(device_input, device_output)
            torch.cuda.synchronize()
            actual = helpers.output_from_bytes(semantics, device_output.cpu().numpy()).astype(dtype)
            artifact = helpers.retain_error_artifact(directory, case_id, label, actual, reference)
            diagnostics[label] = {"artifact": artifact, "exact_reference": True,
                                  "nonzero_output_elements": int(np.count_nonzero(actual))}
            np.testing.assert_array_equal(device_input.cpu().numpy(), host_input)
            diagnostics[label].update(helpers.check_full(
                actual, reference, helpers.side_storage(semantics, "output"),
                float_verdict="error-report"))
            assert diagnostics[label]["verdict"] == "accuracy-review-required"
            # Constant power-of-two inputs have an exactly representable DC
            # identity. This guards the subnormal regression without choosing
            # an acceptance tolerance for general floating-point workloads.
            np.testing.assert_array_equal(actual, reference)
            diagnostics[label]["exact_identity_verified"] = True

        try:
            with api.Plan(library, descriptor, stream=stream) as plan:
                mapping = plan.mapping_json()
                execute(plan, "selected")
            with api.Plan(library, transform, api.PlanOptions(mapping=mapping), stream=stream) as replay:
                assert json.loads(replay.mapping_json()) == json.loads(mapping)
                execute(replay, "replayed")
        finally:
            record_property("low_amplitude_diagnostics", json.dumps(diagnostics))
