"""Opt-in CPU integration checks for the public descriptor API and Python CLI.

Run with CUDA_VISIBLE_DEVICES='' and CUBUTTERFLY_TEST_LIBRARY=/path/libcuntt.so.
Plan creation is exercised only for requests rejected by the static CPU gate.
"""
import copy
import ctypes as ct
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    path = os.environ.get("CUBUTTERFLY_TEST_LIBRARY")
    if not path:
        pytest.skip("set CUBUTTERFLY_TEST_LIBRARY to enable CPU library integration")
    if not Path(path).is_file():
        pytest.fail("CUBUTTERFLY_TEST_LIBRARY does not name an existing library")
    sys.path.insert(0, str(ROOT / "python"))
    import cubutterfly as api
    return api, api.Library(path), str(Path(path).resolve())


def document(descriptor):
    return json.loads(descriptor.json())


def test_output_epilogue_capabilities_and_roundtrip(runtime):
    api, library, _ = runtime
    transform = api.Transform(op=api.Operator.FWHT, storage=api.DataType.FP16,
                              compute=api.ComputeType.FP32, extents=[64], output_scale=0.125,
                              output_rounding=api.OutputRounding.FINAL_STORAGE)
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        encoded = document(descriptor)
        assert encoded["semantics"]["output_scale"] == 0.125
        assert encoded["semantics"]["output_rounding"] == 1
        assert json.loads(descriptor.capabilities_json())["status"] == "eligible"
        with api.Descriptor.from_json(library, json.dumps(encoded)) as replay:
            assert document(replay) == encoded
    for unsupported in (replace(transform, op=api.Operator.FFT, storage=api.DataType.COMPLEX_FP16),
                        replace(transform, output_rounding=api.OutputRounding.STAGE_STORAGE),
                        replace(transform, storage=api.DataType.BF16),
                        replace(transform, extents=[8, 8]),
                        replace(transform, output_scale=1e300)):
        with api.Descriptor(library, unsupported) as descriptor:
            descriptor.validate()
            caps = json.loads(descriptor.capabilities_json())
            assert caps["status"] == "missing-lowering"
            assert any(gap["code"].startswith("output-") for gap in caps["gaps"])


@pytest.mark.parametrize("explicit_side", ["input", "output", "both"])
def test_independent_default_batch_strides(runtime, explicit_side):
    api, library, _ = runtime
    transform = api.Transform(extents=[4, 8], batch=3,
                              input_batch_stride=100 if explicit_side != "output" else 0,
                              output_batch_stride=200 if explicit_side != "input" else 0)
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantics = document(descriptor)["semantics"]
        assert semantics["input_strides"] == semantics["output_strides"] == [8, 1]
        assert semantics["input_batch_stride"] == (100 if explicit_side != "output" else 32)
        assert semantics["output_batch_stride"] == (200 if explicit_side != "input" else 32)


@pytest.mark.parametrize("inverse", [False, True])
def test_real_fft_side_shapes(runtime, inverse):
    api, library, _ = runtime
    transform = api.Transform(extents=[4, 8], storage=api.DataType.FP32,
                              fft_type=api.FftType.C2R if inverse else api.FftType.R2C,
                              inverse=inverse, batch=2)
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        semantics = document(descriptor)["semantics"]
        assert semantics["shape"] == [4, 8]
        assert semantics["input_strides"] == ([5, 1] if inverse else [8, 1])
        assert semantics["output_strides"] == ([8, 1] if inverse else [5, 1])
        assert semantics["input_batch_stride"] == (20 if inverse else 32)
        assert semantics["output_batch_stride"] == (32 if inverse else 20)
        assert json.loads(descriptor.capabilities_json())["semantic_contract"] == semantics


@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("axis", range(4))
def test_each_axis_order_changes_identity(runtime, side, axis):
    api, library, _ = runtime
    base = api.Transform(extents=[2, 4, 8, 16])
    orders = [api.LogicalOrder.NATURAL] * 4
    orders[axis] = api.LogicalOrder.BIT_REVERSED
    changed = replace(base, **{side + "_orders": orders})
    with api.Descriptor(library, base) as natural, api.Descriptor(library, changed) as ordered:
        natural.validate()
        ordered.validate()
        original = document(natural)["semantics"]
        actual = document(ordered)["semantics"]
        assert actual[side + "_orders"] == [int(order) for order in orders]
        assert actual != original
        assert actual["shape"] == original["shape"]


def missing_transform(api, case):
    if case == "mixed_compute":
        return api.Transform(storage=api.DataType.COMPLEX_FP32, compute=api.ComputeType.FP64,
                             extents=[8]), "compute-lowering"
    if case == "rank9":
        return api.Transform(extents=[2] * 9), "rank-lowering"
    if case == "wide_zeta":
        return api.Transform(op=api.Operator.SUBSET_ZETA, storage=api.DataType.UINT64,
                             extents=[8]), "storage-lowering"
    if case == "real_zeta":
        return api.Transform(op=api.Operator.SUPERSET_ZETA, storage=api.DataType.FP32,
                             extents=[8]), "storage-lowering"
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["rank9", "real_zeta"])
def test_missing_lowering_is_describable_and_rejected_before_gpu(runtime, monkeypatch, case):
    api, library, _ = runtime
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    transform, expected_gap = missing_transform(api, case)
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        original = document(descriptor)
        capabilities = json.loads(descriptor.capabilities_json())
        assert capabilities["status"] == "missing-lowering"
        assert expected_gap in {gap["code"] for gap in capabilities["gaps"]}
        assert capabilities["semantic_contract"] == original["semantics"]
        with api.Descriptor.from_json(library, original) as imported:
            assert document(imported) == original
        with pytest.raises(api.Error) as error:
            api.Plan(library, descriptor)
        assert error.value.status == 3  # CUBUTTERFLY_STATUS_NOT_SUPPORTED
        assert expected_gap in str(error.value)


@pytest.mark.parametrize("case", ["mixed_compute", "wide_zeta", "low_real", "high_rank_real", "arbitrary_rank3"])
def test_new_lowerings_are_statically_eligible_without_gpu(runtime, case):
    api, library, _ = runtime
    if case in ("mixed_compute", "wide_zeta"):
        transform, _ = missing_transform(api, case)
    elif case == "low_real":
        transform = api.Transform(storage=api.DataType.FP16, extents=[4, 8],
                                  fft_type=api.FftType.R2C)
    elif case == "high_rank_real":
        transform = api.Transform(storage=api.DataType.FP64, extents=[2, 2, 2, 4],
                                  fft_type=api.FftType.R2C)
    else:
        transform = api.Transform(storage=api.DataType.COMPLEX_FP32, extents=[3, 5, 7])
    with api.Descriptor(library, transform) as descriptor:
        descriptor.validate()
        capabilities = json.loads(descriptor.capabilities_json())
        if case == "arbitrary_rank3":
            assert capabilities["status"] == "missing-lowering"
            assert {gap["code"] for gap in capabilities["gaps"]} == {"rank-bluestein-lowering"}
        else:
            assert capabilities["status"] == "eligible"
            assert capabilities["gaps"] == []
        assert capabilities["semantic_contract"] == document(descriptor)["semantics"]


def test_json_roundtrip_and_selection_independent_identity(runtime):
    api, library, _ = runtime
    transform = api.Transform(extents=[4, 8], batch=2, input_batch_stride=64,
                              input_orders=[1, 0], output_orders=[0, 1])
    with api.Descriptor(library, transform) as descriptor:
        original = document(descriptor)
        with api.Descriptor.from_json(library, descriptor.json()) as imported:
            imported.validate()
            assert document(imported) == original
        options = api.PlanOptions(execution_mode=api.ExecutionMode.COMPOSITE)
        with api.Descriptor(library, transform, options) as selected:
            selected_json = document(selected)
            assert selected_json["semantics"] == original["semantics"]
            assert selected_json["selection"] != original["selection"]


@pytest.mark.parametrize("malformation", ["unknown_top", "unknown_semantics", "unknown_selection",
                                         "negative_extent", "floating_batch", "invalid_enum",
                                         "missing_field", "wrong_boolean", "empty_explicit",
                                         "mapping_with_default", "mapping_with_measure"])
def test_invalid_json_rejected_atomically_with_readable_error(runtime, malformation):
    api, library, _ = runtime
    with api.Descriptor(library, api.Transform(extents=[4, 8])) as descriptor:
        original = document(descriptor)
        malformed = copy.deepcopy(original)
        semantics = malformed["semantics"]
        if malformation == "unknown_top":
            malformed["unknown"] = 1
        elif malformation == "unknown_semantics":
            semantics["unknown"] = 1
        elif malformation == "unknown_selection":
            malformed["selection"]["unknown"] = 1
        elif malformation == "negative_extent":
            semantics["shape"][0] = -1
        elif malformation == "floating_batch":
            semantics["batch"] = 1.0
        elif malformation == "invalid_enum":
            semantics["operator"] = 99
        elif malformation == "missing_field":
            del semantics["input_orders"]
        elif malformation == "wrong_boolean":
            semantics["normalize_inverse"] = 0
        elif malformation == "empty_explicit":
            malformed["selection"]["algorithm_policy"] = api.AlgorithmPolicy.EXPLICIT
        elif malformation in ("mapping_with_default", "mapping_with_measure"):
            malformed["selection"]["algorithm_policy"] = (
                api.AlgorithmPolicy.MEASURE if malformation == "mapping_with_measure"
                else api.AlgorithmPolicy.DEFAULT)
            malformed["selection"]["explicit_algorithm"] = "ignored-mapping"
        with pytest.raises(api.Error) as error:
            library._call("DescriptorSetJson", descriptor.native_handle,
                          json.dumps(malformed).encode("utf-8"))
        assert error.value.status == 1  # CUBUTTERFLY_STATUS_INVALID_VALUE
        assert ": " in str(error.value)
        assert "JSON" in str(error.value) or "selection policy" in str(error.value)
        assert document(descriptor) == original
        descriptor.validate()


def test_shape_change_preserves_explicit_axis_fields(runtime):
    api, library, _ = runtime
    with api.Descriptor(library, api.Transform(extents=[4, 8], input_strides=[8, 1],
                                               output_strides=[8, 1])) as descriptor:
        extents = (ct.c_size_t * 3)(2, 4, 8)
        library._call("SetShape", descriptor.native_handle, 3, extents, 1)
        library._call("SetLogicalOrders", descriptor.native_handle, None, None)
        with pytest.raises(api.Error) as error:
            descriptor.validate()
        assert error.value.status == 1
        assert "stride rank mismatch" in str(error.value)


@pytest.mark.parametrize("side", ["input", "output"])
def test_legacy_and_generic_orders_cannot_conflict_in_json(runtime, side):
    api, library, _ = runtime
    transform = api.Transform(op=api.Operator.NTT, storage=api.DataType.UINT32,
                              extents=[8], modulus=17, word_bits=32)
    with api.Descriptor(library, transform) as descriptor:
        original = document(descriptor)
        contradictory = copy.deepcopy(original)
        contradictory["semantics"]["ntt_" + side + "_layout"] = api.NttLayout.BIT_REVERSED
        with pytest.raises(api.Error) as error:
            library._call("DescriptorSetJson", descriptor.native_handle,
                          json.dumps(contradictory).encode("utf-8"))
        assert error.value.status == 1
        assert "conflicts with explicit logical order" in str(error.value)
        assert document(descriptor) == original
        contradictory["semantics"][side + "_orders"] = [api.LogicalOrder.BIT_REVERSED]
        with api.Descriptor.from_json(library, contradictory) as consistent:
            consistent.validate()
            normalized = document(consistent)["semantics"]
            assert normalized[side + "_orders"] == [api.LogicalOrder.BIT_REVERSED]
            assert normalized["ntt_" + side + "_layout"] == api.NttLayout.NATURAL


def test_cpu_cli_matches_public_descriptor_contract(runtime, tmp_path):
    api, library, library_path = runtime
    transform, _ = missing_transform(api, "mixed_compute")
    with api.Descriptor(library, transform) as descriptor:
        original = document(descriptor)
        capabilities = json.loads(descriptor.capabilities_json())
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=str(ROOT / "python") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-m", "cubutterfly", str(path), "--library", library_path],
                            env=environment, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["descriptor"] == original
    assert payload["capabilities"] == capabilities
