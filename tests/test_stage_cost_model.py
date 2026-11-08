import copy
import json
import math
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


def test_stale_primitive_key_model_must_be_refit():
    fitted = model.fit([dict(sample=sample(), median_kernel_ms=.02)], PROFILE)
    fitted["version"] = "physical-stage-composition-v6"
    with pytest.raises(ValueError, match="re-fit required"):
        model.predict(sample(), fitted)


@pytest.mark.parametrize("backend,overlap", [("shared-iterative", 0),
    ("shared-iterative", 1), ("factor-streamed", 0), ("factor-streamed", 1)])
def test_prepared_predictor_matches_refits_and_ranking(backend, overlap):
    samples = [dict(sample((4, 8), batch=batch), backend=backend,
                    stage_overlap=overlap, factor_overlap=overlap, batch_tile_count=2)
               for batch in (1, 2, 3, 8)]
    for value in samples:
        value["execution_groups_json"][-1]["launch_count"] = 3
    fitted = model.fit([], PROFILE)
    predictor = model.PreparedPredictor(fitted)
    prepared = [predictor.prepare(value) for value in samples]
    trained = model.fit([dict(sample=samples[0], median_kernel_ms=.02)], PROFILE)
    for current in (fitted, trained):
        score = predictor.bind(current)
        expected = [model.predict(value, current) for value in samples]
        actual = [score(value) for value in prepared]
        assert actual == expected
        assert sorted(range(len(samples)), key=actual.__getitem__) == sorted(
            range(len(samples)), key=expected.__getitem__)
        assert [score(value, True) for value in prepared] == [
            model.predict(value, current, True) for value in samples]


def test_prepared_predictor_snapshots_and_rejects_changed_profile():
    fitted = model.fit([], PROFILE)
    predictor = model.PreparedPredictor(fitted)
    value = sample()
    prepared = predictor.prepare(value)
    expected = model.predict(value, fitted)
    value["execution_groups_json"][0]["stage_count"] = 100
    assert predictor.bind(fitted)(prepared) == expected
    assert predictor.bind(fitted)(predictor.prepare(value)) == model.predict(value, fitted)
    changed = copy.deepcopy(fitted)
    changed["profile"]["sm_count"] += 1
    with pytest.raises(ValueError, match="profile or mode changed"):
        predictor.bind(changed)
    with pytest.raises(ValueError, match="another predictor"):
        model.PreparedPredictor(fitted).bind(fitted)(prepared)


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


def test_primitive_key_does_not_pool_semantic_or_policy_axes():
    base = dict(sample(), backend="online-reorder", fft_core="register-tile",
                direction="forward", normalization="none", placement="out-of-place",
                element_stride=1, batch_stride=4096, value_type="complex",
                mapping_json={"backend": "online-reorder", "fft_core": "register-tile",
                              "prefix_codelet": "native", "direct_boundary": "direct-strided"})
    group = dict(index=0, core="register-tile", codelet="native", shared_layout="linear")
    for changed in (
            dict(base, direction="inverse"),
            dict(base, placement="in-place"),
            dict(base, element_stride=2),
            dict(base, value_type="real"),
            dict(base, mapping_json={**base["mapping_json"], "prefix_codelet": "cufftdx-thread"}),
            dict(base, mapping_json={**base["mapping_json"], "direct_boundary": "tiled-transpose"}),
    ):
        assert model.primitive_key(base, group) != model.primitive_key(changed, group)
    # Forward transforms do not apply inverse normalization. Text/numeric
    # spellings and omitted neutral defaults are also the same contract.
    assert model.primitive_key(base, group) == model.primitive_key(dict(base, normalization="inverse"), group)
    inverse = dict(base, direction="inverse")
    assert model.primitive_key(inverse, group) != model.primitive_key(dict(inverse, normalization="inverse"), group)
    assert model.primitive_key(base, group) == model.primitive_key(dict(base, element_stride="1", batch_stride="4096"), group)


def test_stage_terms_uses_execution_group_order_when_index_is_omitted():
    groups = [
        {"stage_count": 4, "grid_ctas": 32, "threads": 64, "core": "scalar"},
        {"stage_count": 4, "grid_ctas": 32, "threads": 64, "core": "scalar"},
    ]
    sample_with_policies = dict(sample((4, 4)), backend="factor-streamed",
        factor_io_policies=["dynamic", "static-unrolled"],
        mapping_json=json.dumps({"backend": "factor-streamed",
                                 "factor_io_policies": ["dynamic", "static-unrolled"]}),
        execution_groups_json=groups)
    terms = model.stage_terms(sample_with_policies, PROFILE)
    assert terms[0]["key"] != terms[1]["key"]


def test_stage_service_ranking_requires_a_validated_profile():
    import calibrate_local_hardware as calibration

    curves = {"mock-stage": {"point_count": 2}}
    assert not calibration._stage_service_usable_for_ranking({
        "stage_service": {"curves": curves},
    })
    assert not calibration._stage_service_usable_for_ranking({
        "stage_service": {"curves": curves, "validated": False,
                           "calibration_status": "complete-sampled"},
    })
    assert not calibration._stage_service_usable_for_ranking({
        "stage_service": {"curves": curves, "validated": True,
                           "calibration_status": "incomplete-validation"},
    })
    assert calibration._stage_service_usable_for_ranking({
        "stage_service": {"curves": curves, "validated": True,
                           "calibration_status": "complete",
                           "inventory_complete": True,
                           "coverage": {"inventory_complete": True}},
    })
    assert calibration._stage_service_usable_for_ranking({
        "stage_service": {"curves": curves, "validated": False,
                           "calibration_status": "complete-sampled", "inventory_complete": False,
                           "selected_scope_status": "complete", "selected_scope_validated": True},
    })


def test_bounded_stage_sampling_keeps_semantic_templates_distinct_and_audited():
    import calibrate_local_hardware as calibration

    mapping = {"backend": "shared-iterative", "fft_core": "scalar",
               "stage_partition": [4, 4]}
    points = [
        {"operator": "fft", "precision": "fp32", "logN": 8,
         "direction": direction, "mapping_json": json.dumps(mapping, sort_keys=True)}
        for direction in ("forward", "inverse")
    ]
    assert calibration._stage_template_key(points[0]) != calibration._stage_template_key(points[1])
    selected, summary = calibration._bound_stage_points(points, 1)
    assert len(selected) == 1
    assert summary == {"requested": 2, "template_count": 2, "selected": 1, "omitted": 1}
    assert calibration._stage_sampling_scope(summary)["inventory_complete"] is False


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
    import stage_schedule_native
    monkeypatch.setattr(stage_schedule_native, "compose", lambda *args: None)
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
    # The serial case uses the equivalent tile-major sum without a graph.
    assert len(calls) == 9


@pytest.mark.parametrize("tiles,factor,overlap,fraction", [
    (1, False, True, .3), (3, False, False, .7),
    (4, True, False, 1.0), (1, True, False, 1.0)])
@pytest.mark.parametrize("tail", [False, True])
def test_serial_schedule_fastpath_preserves_simulator_arithmetic(tiles, factor, overlap, fraction, tail):
    stages = [stage(.7, .3), stage(1.3, .7)]
    tail_stages = [stage(.4), stage(.9)] if tail else None
    expected = model.compose(model.schedule_nodes(stages, tiles, factor, overlap, fraction, tail_stages))
    assert model.compose_stages(stages, tiles, factor, overlap, fraction, tail_stages) == expected


def test_native_schedule_matches_reference_event_order():
    import random
    import stage_schedule_native
    if stage_schedule_native._load() is None:
        pytest.skip("C++ compiler unavailable")
    rng = random.Random(521)
    for _ in range(150):
        groups, tiles = rng.randrange(1, 8), rng.randrange(1, 20)
        stages = [dict(overhead_ms=rng.random(), body_ms=rng.random() + .001,
                       sm_demand=rng.random(), hbm_demand=rng.random()) for _ in range(groups)]
        tail = [dict(s, body_ms=s["body_ms"] * rng.random()) for s in stages] if rng.randrange(2) else None
        factor, overlap, fraction = bool(rng.randrange(2)), bool(rng.randrange(2)), rng.random()
        expected = model.compose(model.schedule_nodes(stages, tiles, factor, overlap, fraction, tail))
        actual = stage_schedule_native.compose(model._schedule_signature(stages), tiles, factor, overlap,
                                              fraction, None if tail is None else model._schedule_signature(tail))
        assert actual == expected
    for epsilon in (0.0, 1e-13, 1e-12, 1.1e-12):
        stages = [dict(overhead_ms=0.0, body_ms=.7 + i * epsilon,
                       sm_demand=.3, hbm_demand=.7) for i in range(3)]
        expected = model.compose(model.schedule_nodes(stages, 10, False, True, .5))
        assert stage_schedule_native.compose(model._schedule_signature(stages), 10,
                                             False, True, .5, None) == expected


def test_native_schedule_missing_compiler_falls_back(monkeypatch):
    import stage_schedule_native
    stage_schedule_native._load.cache_clear()
    monkeypatch.setenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", "1")
    assert stage_schedule_native.backend() == "python-event-simulation"
    assert stage_schedule_native.compose(((0.0, 1.0, .3, .2),), 2, False, True, 1.0, None) is None
    stage_schedule_native._load.cache_clear()
    model._compose_stages_cached.cache_clear()


def test_native_schedule_load_only_missing_cache_never_compiles(tmp_path, monkeypatch):
    import stage_schedule_native as native
    native._load.cache_clear()
    monkeypatch.delenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", raising=False)
    monkeypatch.setenv("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", "1")
    monkeypatch.setattr(native.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected compiler"))
    assert native.backend() == "python-event-simulation"
    assert list(tmp_path.iterdir()) == []
    native._load.cache_clear()


def test_native_schedule_load_only_reuses_prepared_library(monkeypatch):
    import stage_schedule_native as native
    monkeypatch.delenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", raising=False)
    monkeypatch.delenv("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", raising=False)
    native._load.cache_clear()
    if native._load() is None:
        pytest.skip("native scheduler preparation unavailable")
    native._load.cache_clear()
    monkeypatch.setenv("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", "1")
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected compiler"))
    assert native.backend() == "native-cpp-double"
    stages = [stage(.7, .3), stage(1.3, .7)]
    signature = model._schedule_signature(stages)
    assert native.compose(signature, 16, False, True, .5, None) == model.compose(
        model.schedule_nodes(stages, 16, False, True, .5))
    # User disable must take effect even after a native function is cached.
    monkeypatch.setenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", "1")
    assert native.backend() == "python-event-simulation"
    assert native.compose(signature, 16, False, True, .5, None) is None
    native._load.cache_clear()


@pytest.mark.parametrize("failure", [OSError("incompatible binary"), AttributeError("missing ABI symbol")])
def test_native_schedule_load_only_incompatible_cache_falls_back(tmp_path, monkeypatch, failure):
    import stage_schedule_native as native
    native._load.cache_clear()
    monkeypatch.delenv("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE", raising=False)
    monkeypatch.setenv("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY", "1")
    monkeypatch.setattr(native.pathlib.Path, "exists", lambda self: True)
    def fail_load(*args):
        raise failure
    monkeypatch.setattr(native.ctypes, "CDLL", fail_load)
    monkeypatch.setattr(native.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected compiler"))
    assert native.backend() == "python-event-simulation"
    native._load.cache_clear()


def test_invalid_profile_cannot_silently_supply_a_device_cost():
    with pytest.raises(ValueError, match="hardware capability"):
        model.fit([dict(sample=sample(), median_kernel_ms=.01)], {})


RESOURCE_HARDWARE = dict(max_blocks_per_sm=32, max_threads_per_sm=2048,
                         shared_bytes_per_sm=98304, registers_per_sm=65536,
                         max_threads_per_block=1024, shared_bytes_per_block=65536,
                         registers_per_block=32768)


@pytest.mark.parametrize("operator", ["fft", "ntt", "fwht", "subset-zeta",
                                      "superset-zeta", "structured-2x2"])
@pytest.mark.parametrize("demand,resource", [
    ({"threads": 4096}, "max_threads_per_sm"),
    ({"live_shared_bytes": 1048576}, "shared_bytes_per_sm"),
    ({"compiler_registers_per_thread": 2048, "compiler_resources_known": True}, "registers_per_sm"),
    ({"threads": 1536}, "max_threads_per_block"),
    ({"live_shared_bytes": 98304}, "shared_bytes_per_block"),
    ({"compiler_registers_per_thread": 768, "compiler_resources_known": True}, "registers_per_block"),
])
def test_known_resource_overflow_cannot_receive_a_finite_score(operator, demand, resource):
    value = dict(sample(), operator=operator)
    value["execution_groups_json"][0].update(demand)
    profile = dict(PROFILE, hardware=RESOURCE_HARDWARE)
    fitted = model.fit([], profile)
    with pytest.raises(model.ResourceInfeasibleError, match=resource):
        model.stage_terms(value, profile)
    assert model.predict(value, fitted) == math.inf
    explained = model.predict(value, fitted, True)
    assert explained["status"] == "infeasible" and explained["kernel_ms"] is None
    assert explained["resource"] == resource
    json.dumps(explained, allow_nan=False)
    predictor = model.PreparedPredictor(fitted)
    prepared = predictor.prepare(value)
    assert predictor.bind(fitted)(prepared) == math.inf
    assert predictor.bind(fitted)(prepared, True) == explained
    # A measured-service profile must not bypass the same hardware constraint.
    measured = dict(fitted, stage_service={"curves": {"unused": {}}}, service_model="measured-stage-curves")
    assert model.predict(value, measured) == math.inf
    with pytest.raises(model.ResourceInfeasibleError, match=resource):
        model.fit([dict(sample=value, median_kernel_ms=.01)], dict(profile, stage_service=measured["stage_service"]))


@pytest.mark.parametrize("hardware", [{}, {"shared_bytes_per_sm": 98304},
                                      {key: 0 for key in RESOURCE_HARDWARE},
                                      {key: "unknown" for key in RESOURCE_HARDWARE}, RESOURCE_HARDWARE])
def test_missing_resource_information_remains_unknown(hardware):
    value = sample()
    value["execution_groups_json"][0].update(
        compiler_registers_per_thread=100000, compiler_resources_known=False)
    profile = dict(PROFILE, hardware=hardware)
    term = model.stage_terms(value, profile)[0]
    assert term["resource_feasibility"] == "unknown"
    assert "registers_per_sm" in term["unknown_resources"]
    fitted = model.fit([], profile)
    assert math.isfinite(model.predict(value, fitted))
    assert model.predict(value, fitted, True)["stages"][0]["resource_feasibility"] == "unknown"


def test_legal_known_resource_occupancy_and_serial_cost_are_unchanged():
    value = sample()
    value["execution_groups_json"][0].update(grid_ctas=4, compiler_registers_per_thread=64,
                                             compiler_resources_known=True)
    profile = dict(PROFILE, hardware=RESOURCE_HARDWARE)
    term = model.stage_terms(value, profile)[0]
    assert term["resource_feasibility"] == "feasible" and term["unknown_resources"] == []
    assert term["occupancy"] == pytest.approx(4 / (10 * 16))
    assert term["service"] == model.stage_terms(value, PROFILE)[0]["service"]
    assert model.predict(value, model.fit([], profile)) == model.predict(value, model.fit([], PROFILE))


def test_missing_per_cta_limits_do_not_claim_full_resource_feasibility():
    value = sample()
    value["execution_groups_json"][0].update(compiler_registers_per_thread=64,
                                             compiler_resources_known=True)
    hardware = {key: value for key, value in RESOURCE_HARDWARE.items() if not key.endswith("_per_block")}
    term = model.stage_terms(value, dict(PROFILE, hardware=hardware))[0]
    assert term["resource_feasibility"] == "unknown"
    assert set(term["unknown_resources"]) == {
        "max_threads_per_block", "shared_bytes_per_block", "registers_per_block"}


def test_optin_shared_limit_alias_is_used_when_canonical_field_is_absent():
    value = sample()
    value["execution_groups_json"][0]["live_shared_bytes"] = 98304
    hardware = dict(RESOURCE_HARDWARE, max_shared_per_block_optin_bytes=65536)
    del hardware["shared_bytes_per_block"]
    explanation = model.predict(value, model.fit([], dict(PROFILE, hardware=hardware)), True)
    assert explanation["status"] == "infeasible"
    assert explanation["resource"] == "shared_bytes_per_block"


def test_measured_curve_resource_resolution_cannot_hide_known_overflow(monkeypatch):
    import stage_service_model

    value = sample()
    fitted = model.fit([], dict(PROFILE, hardware=RESOURCE_HARDWARE))
    fitted["stage_service"] = {"curves": {"unused": {}}}
    fitted["service_model"] = "measured-stage-curves"
    monkeypatch.setattr(model, "_model_service_index", lambda *args: None)

    def resolved(sample, group, *args, **kwargs):
        actual = dict(group, compiler_registers_per_thread=2048, compiler_resources_known=True)
        return {"covered": True, "kernel_ms": .01}, actual, True

    monkeypatch.setattr(stage_service_model, "predict_group_resolved", resolved)
    assert model.predict(value, fitted) == math.inf
    assert model.predict(value, fitted, True)["resource"] == "registers_per_sm"


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


@pytest.mark.parametrize('operator,precision', [('fft', 'fp16'), ('fft', 'fp64'),
                                               ('fwht', 'fp32'), ('ntt', 'word64')])
@pytest.mark.parametrize('overlap', [0, 1])
def test_prepared_prior_cache_exact_and_independent(operator, precision, overlap, monkeypatch):
    profile = copy.deepcopy(PROFILE)
    profile['scheduling'] = dict(shapes=[dict(threads=64, dispatch_identified=True,
        startup_ms=.004, dispatch_ms_per_cta=.000001, measured_grid_range=[1, 4096])])
    fitted = model.fit([], profile)
    value = dict(sample((4, 8), batch=3), operator=operator, precision=precision,
                 backend='shared-iterative', stage_overlap=overlap, batch_tile_count=2,
                 mapping_json=dict(backend='shared-iterative',
                                   local_stage_partitions=[[2, 2], [4, 4]]))
    for group in value['execution_groups_json']:
        group.update(codelet='native-register-subgraph', exchange_chunk=2,
                     elements_per_thread=8, compiler_resources_known=True,
                     compiler_registers_per_thread=16)
    expected = model.predict(value, fitted, True)
    predictor = model.PreparedPredictor(fitted)
    calls = dict(schedule=0, geometry=0)
    original_schedule, original_geometry = model.scheduling_costs, model.shared_work_geometry
    def schedule(*args):
        calls['schedule'] += 1
        return original_schedule(*args)
    def geometry(*args):
        calls['geometry'] += 1
        return original_geometry(*args)
    monkeypatch.setattr(model, 'scheduling_costs', schedule)
    monkeypatch.setattr(model, 'shared_work_geometry', geometry)
    first, second = predictor.prepare(value), predictor.prepare(value)
    assert calls == dict(schedule=2, geometry=2)
    score = predictor.bind(fitted)
    assert score(first, True) == expected == score(second, True)
    value['execution_groups_json'][0]['threads'] = 128
    first_output = score(first, True)
    first_output['stages'][0]['scheduling']['startup_ms'] = 999
    first_output['stages'][0]['arithmetic_work']['leaves'][0]['rounds'] = 999
    first_output['stages'][0]['unknown_resources'].append('mutated')
    assert score(second, True) == expected
    third = predictor.prepare(second[1])
    assert score(third, True) == expected
    changed = copy.deepcopy(fitted)
    changed['profile']['scheduling']['shapes'][0]['startup_ms'] *= 2
    with pytest.raises(ValueError, match='profile or mode changed'):
        predictor.bind(changed)
    other = model.PreparedPredictor(changed)
    assert other.bind(changed)(other.prepare(second[1]), True) == model.predict(second[1], changed, True)


@pytest.mark.parametrize('partition,tiles', [([True, 3], 1), ([2, 2], 2), ([1], 1),
                                            ([4], 3), ([4], float('nan'))])
def test_prepared_geometry_cache_preserves_invalid_inputs(partition, tiles):
    cache = model._StagePriorCache(PROFILE)
    cache.geometry(4, 64, [1, 3], 1)
    with pytest.raises((ValueError, TypeError, OverflowError)) as expected:
        model.shared_work_geometry(4, 64, partition, tiles)
    with pytest.raises(type(expected.value), match=__import__('re').escape(str(expected.value))):
        cache.geometry(4, 64, partition, tiles)


@pytest.mark.parametrize('hardware', [dict(max_threads_per_block=32),
                                        dict(shared_bytes_per_block=128),
                                        dict(registers_per_block=128),
                                        dict(max_threads_per_block=1024, shared_bytes_per_block=65536)])
def test_prepared_prior_cache_preserves_resource_predictions(hardware):
    fitted = model.fit([], dict(PROFILE, hardware=hardware))
    value = dict(sample((4, 8)), backend='shared-iterative')
    value['execution_groups_json'][0].update(compiler_resources_known=True,
                                           compiler_registers_per_thread=16)
    predictor = model.PreparedPredictor(fitted)
    expected = model.predict(value, fitted, True)
    for _ in range(2):
        prepared = predictor.prepare(value)
        assert predictor.bind(fitted)(prepared, True) == expected
        assert predictor.bind(fitted)(prepared) == model.predict(value, fitted)
