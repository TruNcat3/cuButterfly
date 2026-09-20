import copy
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import stage_cost_model as model


PROFILE = dict(sm_count=10, capabilities=dict(global_feedback_bytes_per_second=1e12,
    equivalent_butterflies_per_second=1e11, interstage_shared_bytes_per_second=1e13,
    cta_barriers_per_second=1e9))


def sample(stages=(4,), batch=1, threads=64):
    return dict(operator="fft", precision="fp32", logN=12, N=4096, batch=batch,
        execution_groups_json=[dict(stage_count=s, grid_ctas=4096 * batch // 2**s,
            threads=threads, core="scalar", live_shared_bytes=2**s*16) for s in stages])


def test_unseen_composition_reuses_stage_functions_and_sums():
    a, b = sample((4,)), sample((8,))
    training = [dict(sample=s, median_kernel_ms=0.006 + 1.5 * model.stage_terms(s, PROFILE)[0]["service"])
                for s in (a, b, sample((6,), batch=8))]
    fitted = model.fit(training, PROFILE, ridge=0)
    joined = sample((4, 8))
    assert model.predict(joined, fitted) == pytest.approx(model.predict(a, fitted)+model.predict(b, fitted))
    assert model.predict(joined, fitted, True)["missing_primitives"] == []
    assert fitted["separately_measured_stage_costs"] is False
    assert all(c["overhead_service_separable"] for c in fitted["primitive_coverage"].values())


def test_unknown_core_and_extrapolation_are_not_certified():
    observed = sample()
    fitted = model.fit([dict(sample=observed, median_kernel_ms=.02)], PROFILE)
    unknown = sample(threads=128)
    unknown["execution_groups_json"][0]["core"] = "new-core"
    assert model.predict(unknown, fitted, True)["missing_primitives"]
    assert not model.predict(sample(threads=128), fitted, True)["missing_primitives"]
    assert model.predict(sample(batch=10000), fitted, True)["extrapolated_primitives"]
    assert not next(iter(fitted["primitive_coverage"].values()))["overhead_service_separable"]
    assert model.predict(sample(batch=10000), fitted) > model.predict(observed, fitted)


def test_primitive_key_is_group_local_for_cooperative_prefix_lanes():
    base = dict(operator="fft", precision="fp32", backend="online-reorder",
                fft_core="register-tile", mapping_json={
                    "backend": "online-reorder", "fft_core": "register-tile",
                    "prefix_codelet_lanes": 1,
                })
    prefix = dict(index=0, core="register-tile", codelet="native", shared_layout="linear")
    suffix = dict(index=1, core="cufftdx-block", codelet="native", shared_layout="writer-aligned")
    cooperative = dict(base, mapping_json={**base["mapping_json"], "prefix_codelet_lanes": 2})
    assert model.primitive_key(base, prefix) != model.primitive_key(cooperative, prefix)
    assert model.primitive_key(base, suffix) == model.primitive_key(cooperative, suffix)


def test_overlapped_labels_do_not_train_serial_stage_factors():
    s = dict(sample(), stage_overlap=1, batch_tile_count=1)
    fitted = model.fit([dict(sample=s, median_kernel_ms=100)], PROFILE)
    assert fitted["additive_training_rows"] == 0 and fitted["parameters"] == {}
    assert model.predict(s, fitted, True)["overlap_calibration"] == "unmeasured"


def stage(duration=1, demand=0.1):
    return dict(overhead_ms=0, body_ms=duration, sm_demand=demand, hbm_demand=demand)


def test_pipeline_exposes_fill_drain_and_shares_gpu_capacity():
    underfilled = [stage(), stage()]
    assert model.compose(model.schedule_nodes(underfilled, tiles=4, overlap=False)) == 8
    assert model.compose(model.schedule_nodes(underfilled, tiles=4, overlap=True)) == 5
    saturated = [stage(demand=1), stage(demand=1)]
    assert model.compose(model.schedule_nodes(saturated, tiles=4, overlap=True)) == 8


def test_batch_ring_reuse_waits_for_the_immediate_consumer():
    nodes = model.schedule_nodes([stage(), stage(), stage()], tiles=3, overlap=True)
    # Producer group0/tile2 cannot overwrite slot0 until group1/tile0 consumes it.
    assert 1 in nodes[6]["deps"]
    assert nodes[6]["deps"] == {1, 3}


def test_factor_final_stage_waits_for_complete_fanin():
    stages = [stage(), stage(), stage(duration=3)]
    nodes = model.schedule_nodes(stages, tiles=4, factor=True, overlap=True)
    assert len(nodes) == 9 and nodes[-1]["deps"] == {7}
    assert model.compose(nodes) == 8
    assert model.compose(model.schedule_nodes(stages, tiles=4, factor=True, overlap=False)) == 11


def test_last_batch_tile_scales_service_but_keeps_launch():
    nodes = model.schedule_nodes([dict(stage(), overhead_ms=2)], tiles=2, overlap=True, last_fraction=.25)
    assert model.compose(nodes) == 5.25


@pytest.mark.parametrize("groups,tiles,factor,overlap,fraction,tail", [
    (2, 1, False, False, 1.0, False),
    (3, 17, False, True, 1.0, False),
    (3, 17, False, True, 0.25, False),
    (3, 17, False, True, 1.0, True),
    (4, 16, True, False, 1.0, False),
    (4, 16, True, True, 1.0, False),
])
def test_cached_schedule_matches_original_event_simulation_exactly(groups, tiles, factor, overlap, fraction, tail):
    stages = [dict(stage(duration=.13 * (i+1), demand=.2 * (i+1)), overhead_ms=.017)
              for i in range(groups)]
    tail_stages = [dict(s, body_ms=s["body_ms"] / 3) for s in stages] if tail else None
    expected = model.compose(model.schedule_nodes(stages, tiles, factor, overlap, fraction, tail_stages))
    for _ in range(2):
        assert model.compose_stages(stages, tiles, factor, overlap, fraction, tail_stages) == expected


def test_schedule_cache_reuses_only_identical_durations_resources_and_dependencies(monkeypatch):
    model._compose_stages_cached.cache_clear()
    original = model.compose
    calls = []
    def counted(nodes):
        calls.append(len(nodes))
        return original(nodes)
    monkeypatch.setattr(model, "compose", counted)
    stages = [stage(duration=.7, demand=.3), stage(duration=1.3, demand=.7)]
    options = dict(tiles=3, overlap=True)
    result = model.compose_stages(stages, **options)
    assert model.compose_stages(copy.deepcopy(stages), **options) == result
    annotated = [dict(s, curve_key="new provenance", scheduling={"unrelated": 1}) for s in stages]
    assert model.compose_stages(annotated, **options) == result
    assert len(calls) == 1
    for field in ("body_ms", "overhead_ms", "sm_demand", "hbm_demand"):
        changed = copy.deepcopy(stages)
        changed[0][field] += .1
        expected = original(model.schedule_nodes(changed, **options))
        assert model.compose_stages(changed, **options) == expected
    for changed in (dict(options, tiles=4), dict(options, factor=True),
                    dict(options, overlap=False), dict(options, last_fraction=.5),
                    dict(options, tail_stages=[stage(.4), stage(.9)])):
        expected = original(model.schedule_nodes(stages, **changed))
        assert model.compose_stages(stages, **changed) == expected
    assert len(calls) == 10
    model._compose_stages_cached.cache_clear()


def test_invalid_profile_cannot_silently_supply_a_device_cost():
    with pytest.raises(ValueError, match="hardware capability"):
        model.fit([dict(sample=sample(), median_kernel_ms=.01)], {})


def test_cost_parameters_cannot_mix_memory_capacities():
    measured = dict(sample(), device="Test GPU", global_memory_bytes=80 * 1024**3)
    profile = dict(PROFILE, device="Test GPU", global_memory_bytes=40 * 1024**3)
    with pytest.raises(ValueError, match="global_memory_bytes"):
        model.fit([dict(sample=measured, median_kernel_ms=.01)], profile)


def test_measured_scheduling_curve_reaches_physical_stage_prediction():
    profile=dict(PROFILE, scheduling=dict(shapes=[dict(threads=64, startup_ms=.007,
        dispatch_ms_per_cta=.0001, dispatch_identified=True, measured_grid_range=[1, 1000])]))
    terms=model.stage_terms(sample(),profile)
    assert terms[0]["startup_and_tail"] == .007
    assert terms[0]["scheduling"]["dispatch_ms"] == pytest.approx((256-1)*.0001)
    assert terms[0]["scheduling"]["source"] == "measured-minimal-CTA"


def test_full_workload_holdout_does_not_leak_coefficients():
    rows = []
    for batch in (1, 2, 4):
        for threads, value in ((64, .01), (128, .02)):
            rows.append(dict(name=f"{batch}-{threads}", sample=sample(batch=batch,threads=threads), median_kernel_ms=value*batch**3))
    result = model.report(rows, PROFILE)
    assert result["validation"]["holdout_valid_multi_candidate_workloads"] == 3
    assert result["status"] == "calibrated-local-warning"
    assert not result["usable_for_unmeasured_ranking"]
