"""Finite population accounting and public descriptor integration, CPU only."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest


def test_opposite_alternating_order_preserves_full_axis_reversal():
    assert campaign.logical_orders("reverse-alternating", 2) == [1, 0]
    assert campaign.logical_orders("reverse-alternating", 3) == [1, 0, 1]


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_campaign_manifest", ROOT / "scripts/semantic_campaign_manifest.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def config():
    return {
        "schema": campaign.CONFIG_SCHEMA,
        "populations": [{"name": "finite", "shapes": [[4, 8]], "batches": [2],
                         "directions": ["forward", "inverse"], "normalizations": ["none", "inverse"],
                         "placements": ["out-of-place", "in-place"],
                         "order_patterns": ["natural", "bit-reversed"],
                         "layouts": ["packed", "independent-padded"],
                         "regimes": [{"name": "fft", "transform": {"op": 0, "storage": 6, "compute": 1}}]}],
    }


@pytest.fixture
def fake_public(monkeypatch, tmpdir):
    """Mock only public serialization; transform construction uses the real API."""
    class Descriptor:
        def __init__(self, library, transform):
            self.transform = transform

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def validate(self):
            t = self.transform
            for side in ("input", "output"):
                extents = campaign.side_extents(t.extents, t.inverse, vars(t))[side == "output"]
                if any(order and extent & (extent - 1) for order, extent in
                       zip(getattr(t, side + "_orders"), extents)):
                    raise campaign.api.Error("bit reversal requires a power-of-two side extent", 1)

        def json(self):
            fields = copy.deepcopy(vars(self.transform))
            if not fields["inverse"]:
                fields["normalize_inverse"] = False
            return json.dumps({"schema": "cubutterfly-descriptor-v1", "semantics": fields,
                               "selection": {"algorithm_policy": 0}})

        def capabilities_json(self):
            gap = self.transform.compute == 2
            return json.dumps({"status": "missing-lowering" if gap else "eligible",
                               "semantic_contract": json.loads(self.json())["semantics"],
                               "gaps": [{"code": "compute-lowering"}] if gap else []})

    monkeypatch.setattr(campaign.api, "Descriptor", Descriptor)
    library = Path(str(tmpdir)) / "lib.so"
    library.write_bytes(b"test identity")
    return library


def test_cartesian_and_canonical_multiplicity(fake_public):
    request = config()
    result = campaign.generate_manifest(request, fake_public, library=object())
    # 1 shape * 1 batch * 2 directions * 2 norms * 2 placements * 2 input
    # patterns * 2 output patterns * 2 layouts * 1 regime = 64 requests.
    assert result["metrics"]["requested_variants"] == 64
    assert result["metrics"]["valid_variants"] == 64
    assert result["metrics"]["unique_cases"] == 48
    assert sum(len(case["requested_variants"]) for case in result["cases"]) == 64
    assert {tuple(c["descriptor"]["semantics"]["input_orders"]) for c in result["cases"]} == {(0, 0), (1, 1)}
    assert campaign.validate_manifest_hash(result)
    changed = copy.deepcopy(result)
    changed["metrics"]["requested_variants"] -= 1
    assert not campaign.validate_manifest_hash(changed)
    assert request == config()


def test_cross_population_dedup_and_gap_accounting(fake_public):
    request = config()
    request["populations"][0]["regimes"][0]["transform"]["compute"] = 2
    second = copy.deepcopy(request["populations"][0])
    second["name"] = "second"
    request["populations"].append(second)
    result = campaign.generate_manifest(request, fake_public, library=object())
    assert result["metrics"]["requested_variants"] == 128
    assert result["metrics"]["unique_cases"] == result["metrics"]["unique_gap_cases"] == 48
    assert all(case["populations"] == ["finite", "second"] for case in result["cases"])
    assert all(counts["gap_variants"] == 64 for counts in result["metrics"]["populations"].values())


def test_invalid_requests_retained(fake_public):
    request = config()
    request["populations"][0]["shapes"] = [[3, 8]]
    result = campaign.generate_manifest(request, fake_public, library=object())
    assert result["metrics"]["requested_variants"] == 64
    assert result["metrics"]["invalid_variants"] == 48
    assert result["metrics"]["valid_variants"] == 16
    assert len(result["invalid_requests"]) == 48
    assert all(row["status_code"] == 1 and row["request"]["shape"] == [3, 8]
               for row in result["invalid_requests"])


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(unknown=True),
    lambda c: c["populations"][0].update(unknown=True),
    lambda c: c["populations"][0]["regimes"][0]["transform"].update(unknown=True),
    lambda c: c["populations"][0].update(batches=[True]),
    lambda c: c["populations"][0].update(shapes=[[4.0, 8]]),
    lambda c: c["populations"][0].update(layouts=["approximate"]),
    lambda c: c["populations"][0].update(normalizations=[True]),
    lambda c: c["populations"][0]["regimes"][0]["transform"].update(op=1),
    lambda c: c["populations"][0]["regimes"][0]["transform"].update(op=5),
])
def test_schema_rejects_unknown_fields_and_coercion(mutation):
    request = config()
    mutation(request)
    with pytest.raises(ValueError):
        campaign.validate_config(request)


def test_real_and_embedding_side_extents_and_independent_layouts():
    assert campaign.side_extents([3, 7], False, {"op": 0, "fft_type": 1, "length_mode": 1}) == ([3, 7], [4, 5])
    assert campaign.side_extents([3, 7], True, {"op": 0, "fft_type": 2, "length_mode": 1}) == ([4, 5], [3, 7])
    assert campaign.layout_strides([4, 8], "packed") == ([8, 1], 32)
    ins, ib = campaign.layout_strides([4, 8], "independent-padded")
    outs, ob = campaign.layout_strides([4, 8], "independent-padded", output=True)
    assert ins != outs and ib != ob
    for strides, batch_stride in [(ins, ib), (outs, ob)]:
        addresses = {batch * batch_stride + x * strides[0] + y * strides[1]
                     for batch in range(3) for x in range(4) for y in range(8)}
        assert len(addresses) == 3 * 4 * 8


def test_explicit_structured_matrix_source(fake_public):
    request = config()
    request["populations"][0]["regimes"][0]["transform"] = {
        "op": 5, "storage": 2, "compute": 1, "stage_matrices": "repeat-default"}
    result = campaign.generate_manifest(request, fake_public, library=object())
    assert all(c["descriptor"]["semantics"]["stage_matrices"] == [[campaign.DEFAULT_MATRIX]] * 2
               for c in result["cases"])
    assert all(v["matrix_source"]["kind"] == "repeat-default"
               for c in result["cases"] for v in c["requested_variants"])


def test_duplicate_json_fields_and_fresh_output(tmpdir, fake_public):
    tmp_path = Path(str(tmpdir))
    input_path = tmp_path / "config.json"
    input_path.write_text('{"schema":"x", "schema":"y"}')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        campaign.load_config(input_path)
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(SystemExit):
        campaign.main(["--config", str(input_path), "--library", str(fake_public), "--output", str(output)])
    assert output.read_text() == "preserve"


def test_public_library_integration_cpu_only():
    library = os.environ.get("CUBUTTERFLY_TEST_LIBRARY")
    if not library:
        pytest.skip("set CUBUTTERFLY_TEST_LIBRARY to enable public CPU descriptor integration")
    result = campaign.generate_manifest(config(), library)
    assert result["metrics"]["requested_variants"] == 64
    assert result["metrics"]["valid_variants"] == 64
    assert result["metrics"]["unique_cases"] == 48
    assert campaign.validate_manifest_hash(result)
    for case in result["cases"]:
        assert case["id"] == campaign.semantic_id(case["descriptor"]["semantics"])
        with campaign.api.Descriptor.from_json(campaign.api.Library(library), case["descriptor"]) as descriptor:
            descriptor.validate()
            assert json.loads(descriptor.json()) == case["descriptor"]


def test_public_gap_and_real_direction_constraints():
    library = os.environ.get("CUBUTTERFLY_TEST_LIBRARY")
    if not library:
        pytest.skip("set CUBUTTERFLY_TEST_LIBRARY to enable public CPU descriptor integration")
    request = config()
    population = request["populations"][0]
    population["order_patterns"] = ["natural"]
    population["regimes"][0]["transform"] = {"op": 0, "storage": 2, "compute": 1, "fft_type": 1}
    result = campaign.generate_manifest(request, library)
    assert result["metrics"]["requested_variants"] == 16
    assert result["metrics"]["valid_variants"] == result["metrics"]["invalid_variants"] == 8
    assert all(row["request"]["direction"] == "inverse" for row in result["invalid_requests"])
    population["regimes"][0]["transform"] = {"op": 0, "storage": 6, "compute": 2}
    result = campaign.generate_manifest(request, library)
    assert result["metrics"]["gap_variants"] == result["metrics"]["valid_variants"] == 16
    assert result["metrics"]["unique_gap_cases"] == result["metrics"]["unique_cases"] == 12
    assert all("compute-lowering" in {gap["code"] for gap in case["capabilities"]["gaps"]}
               for case in result["cases"])
