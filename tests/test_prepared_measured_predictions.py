"""Measured service preparation reuses only descriptor/profile invariants."""
import copy
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import stage_cost_model as model
import stage_service_model


PROFILE = dict(sm_count=10, capabilities=dict(global_feedback_bytes_per_second=1e12,
    equivalent_butterflies_per_second=1e11, interstage_shared_bytes_per_second=1e13,
    cta_barriers_per_second=1e9))


def sample(batch):
    return dict(operator="fft", precision="fp32", logN=12, N=4096, batch=batch,
        backend="online-reorder", execution_groups_json=[
            dict(stage_count=stages, grid_ctas=4096 * batch // 2**stages,
                 threads=64, core="scalar", live_shared_bytes=2**stages * 16)
            for stages in (4, 8)])


@pytest.mark.parametrize("overlap", [False, True])
def test_measured_preparation_preserves_refits_and_reuses_group_resolution(monkeypatch, overlap):
    calls = []

    def resolve(sample, group, profile, **kwargs):
        calls.append(group["stage_count"])
        covered = group["stage_count"] == 4
        return (dict(covered=covered, kernel_ms=.012 if covered else None,
                     source="measured-exact", curve_key=str(group["stage_count"]),
                     extrapolated=False, reason="missing"), dict(group), False)

    monkeypatch.setattr(stage_service_model, "predict_group_resolved", resolve)
    monkeypatch.setattr(model, "_model_service_index", lambda *args: None)
    fitted = model.fit([], PROFILE)
    fitted.update(stage_service={"curves": {"fixture": {}}}, service_model="measured-stage-curves")
    value = sample(5)
    if overlap:
        def view(batch):
            physical = sample(batch)
            return dict(sample=physical, groups=physical["execution_groups_json"])
        value.update(stage_overlap=True, batch_tile_count=2, stage_service_projection=dict(
            status="resolved", full=view(2), tail=view(1), tile_batch=2, tail_batch=1, tile_count=3))
    predictor = model.PreparedPredictor(fitted)
    prepared = predictor.prepare(value)
    preparation_calls = len(calls)
    assert preparation_calls == (4 if overlap else 2)
    prior = None
    for factor in (1.0, 1.7, .6):
        changed = copy.deepcopy(fitted)
        changed["parameters"] = {
            term["key"]: dict(startup_and_tail=factor, service=factor)
            for term in model.stage_terms(value, PROFILE)}
        expected = model.predict(value, changed, True)
        before_scoring = len(calls)
        scorer = predictor.bind(changed)
        actual = scorer(prepared, True)
        assert len(calls) == before_scoring
        assert actual == expected
        assert scorer(prepared) == expected["kernel_ms"]
        if prior is not None:
            assert actual["kernel_ms"] != prior
        prior = actual["kernel_ms"]
    value["execution_groups_json"].clear()
    assert scorer(prepared, True) == expected
