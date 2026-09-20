import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_cost_model
import stage_service_model as service


def sample(**overrides):
    value = {
        "operator": "fwht",
        "precision": "fp32",
        "accumulation": "native",
        "direction": "forward",
        "normalization": "none",
        "placement": "out-of-place",
        "logN": "12",
        "batch": "4",
        "element_stride": "2",
        "batch_stride": "8192",
        "backend": "temporal-tile",
        "fft_core": "scalar",
        "local_exchange": "warp-register",
        "shared_layout": "writer-aligned",
        "cross_twiddle": "table",
        "runtime_fingerprint": "fwht-build-a",
        "mapping_json": json.dumps({
            "backend": "temporal-tile", "fft_core": "scalar",
            "local_exchange": "warp-register", "shared_layout": "writer-aligned",
            "cross_twiddle": "table", "batch": 4, "grid_ctas": 999,
        }),
    }
    value.update(overrides)
    return value


def group(k, *, role="train", exchange=1, threads="64", live=1024,
          independent=True, work_blocks=None, grid_ctas=10, duration=None, **extra):
    work_blocks = k if work_blocks is None else work_blocks
    return {
        "index": 0,
        "first_stage": 0,
        "stage_count": 4,
        "core": "scalar",
        "exchange": exchange,
        "threads": threads,
        "EPT": "4",
        "live_shared_bytes": live,
        "compiler_resources_known": True,
        "independent": independent,
        "role": role,
        "work_blocks": work_blocks,
        "grid_ctas": grid_ctas,
        "trial_kernel_ms": [duration or float(k), duration or float(k)],
        **extra,
    }


def record(s, groups):
    return {
        "schema": service.SCHEMA,
        "status": "measured",
        "correct": True,
        "sample": s,
        "groups": groups,
        "pairs": [],
        "plan_trial_kernel_ms": [],
    }


def test_curve_key_separates_fwht_exchange_and_ignores_batch_grid_mapping_noise():
    s = sample()
    first = group(1)
    second = dict(first, exchange=0)
    assert service.curve_key(s, first) != service.curve_key(s, second)
    altered = sample(batch="128", mapping_json=json.dumps({
        "backend": "temporal-tile", "fft_core": "scalar",
        "local_exchange": "warp-register", "shared_layout": "writer-aligned",
        "cross_twiddle": "table", "batch": 128, "grid_ctas": 12,
    }))
    assert service.curve_key(s, first) == service.curve_key(altered, first)


def test_curve_identity_tracks_group_codelet_but_reuses_unrelated_stage_policy():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [6, 6],
        "prefix_codelet": "native", "prefix_shared_layout": "linear",
    }
    prefix = sample(operator="fft", backend="online-reorder", fft_core="register-tile",
                    logN="12", mapping_json=json.dumps(mapping))
    prefix_group = group(1, index=0, core="register-tile", codelet="native",
                         shared_layout="linear")
    dx_mapping = dict(mapping, prefix_codelet="cufftdx-thread", prefix_shared_layout="xor")
    dx = dict(prefix, mapping_json=json.dumps(dx_mapping))
    assert service.curve_key(prefix, prefix_group) != service.curve_key(dx, prefix_group)

    suffix_group = group(1, index=1, core="cufftdx-block", codelet="native",
                         shared_layout="writer-aligned")
    assert service.curve_key(prefix, suffix_group) == service.curve_key(dx, suffix_group)


def test_curve_identity_normalizes_omitted_prefix_defaults_but_keeps_strategy_isolation():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [6, 6],
    }
    omitted = sample(operator="fft", backend="online-reorder", fft_core="register-tile",
                     logN="12", mapping_json=json.dumps(mapping))
    prefix_group = group(1, index=0, core="register-tile", codelet="native",
                         shared_layout="linear")
    explicit = dict(mapping, prefix_codelet="native", prefix_shared_layout="linear")
    assert service.curve_key(omitted, prefix_group) == service.curve_key(
        dict(omitted, mapping_json=json.dumps(explicit)), prefix_group)
    xor = dict(mapping, prefix_codelet="native", prefix_shared_layout="xor")
    assert service.curve_key(omitted, prefix_group) != service.curve_key(
        dict(omitted, mapping_json=json.dumps(xor)), prefix_group)


def test_curve_identity_tracks_cooperative_prefix_lanes_but_reuses_suffix():
    mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "online-reorder",
        "fft_core": "register-tile", "stage_partition": [6, 6],
    }
    omitted = sample(operator="fft", backend="online-reorder", fft_core="register-tile",
                     logN="12", mapping_json=json.dumps(mapping))
    explicit_one = dict(mapping, prefix_codelet_lanes=1)
    cooperative = dict(mapping, prefix_codelet_lanes=2)
    prefix_group = group(1, index=0, core="register-tile", codelet="native",
                         shared_layout="linear")
    assert service.curve_key(omitted, prefix_group) == service.curve_key(
        dict(omitted, mapping_json=json.dumps(explicit_one)), prefix_group)
    assert service.curve_key(omitted, prefix_group) != service.curve_key(
        dict(omitted, mapping_json=json.dumps(cooperative)), prefix_group)

    suffix_group = group(1, index=1, core="cufftdx-block", codelet="native",
                         shared_layout="writer-aligned")
    assert service.curve_key(omitted, suffix_group) == service.curve_key(
        dict(omitted, mapping_json=json.dumps(cooperative)), suffix_group)


def test_curve_identity_tracks_each_factor_io_policy_independently():
    base_mapping = {
        "schema_version": 1, "kind": "butterfly", "backend": "factor-streamed",
        "fft_core": "cufftdx-block", "factor_partition": [4, 4, 4],
        "factor_io_policies": ["dynamic", "dynamic", "dynamic"],
    }
    first = sample(operator="fft", backend="factor-streamed", fft_core="cufftdx-block",
                   logN="12", mapping_json=json.dumps(base_mapping))
    first_group = group(1, index=0, core="cufftdx-block", io_policy="dynamic")
    changed_mapping = dict(base_mapping, factor_io_policies=["static-unrolled", "dynamic", "dynamic"])
    changed = dict(first, mapping_json=json.dumps(changed_mapping))
    assert service.curve_key(first, first_group) != service.curve_key(changed, first_group)
    second_group = group(1, index=1, core="cufftdx-block", io_policy="dynamic")
    assert service.curve_key(first, second_group) == service.curve_key(changed, second_group)


def test_actual_batch_time_is_load_and_midpoint_reuses_curve():
    records = [record(sample(batch=batch), [group(batch, grid_ctas=batch,
        batch_space=1, batch_time=batch, duration=0.005 + batch * 0.001)])
        for batch in (1, 32)]
    profile = service.build_profile(records)
    query = group(16, grid_ctas=16, batch_space=1, batch_time=16)
    prediction = service.predict_group(sample(batch=16), query, profile)
    assert prediction["covered"]
    assert prediction["kernel_ms"] == pytest.approx(0.021)
    # Changing batch tiling changes the mechanism and must remain separate.
    assert service.curve_key(sample(batch=16), query) != service.curve_key(
        sample(batch=16), dict(query, batch_space=2, batch_time=8))


def test_real_probe_flat_group_contract_matches_normalized_benchmark_fields():
    point = sample(batch="8", batch_stride="0", direction="forward")
    point.update({
        "backend": "temporal-tile", "fft_core": "scalar", "compute_unit": "radix2",
        "execution_groups_json": json.dumps([{"first_stage": 0, "stage_count": 4,
                                               "work_blocks": 16, "grid_ctas": 16}]),
    })
    point["mapping_json"] = json.dumps({
        "backend": "temporal-tile", "fft_core": "scalar", "compute_unit": "radix2",
        "local_exchange": "warp-register", "shared_layout": "writer-aligned",
        "stage_partition": [4, 4, 4], "factor_partition": [2, 2],
        "batch": 999, "grid_ctas": 999,
    })
    low = group(16, exchange=2, grid_ctas=16, work_blocks=16,
                kernel_launch_count=1, resource_source="jit")
    high = group(64, exchange=2, grid_ctas=64, work_blocks=64,
                 kernel_launch_count=1, resource_source="stream-capture")
    profile = service.build_profile([record(point, [low, high])])
    query = dict(low, work_blocks=32, grid_ctas=32,
                 resource_source="describe-only", trial_kernel_ms=[32.0])
    predicted = service.predict_group(point, query, profile)
    assert predicted["covered"] and predicted["kernel_ms"] == pytest.approx(32.0)
    assert service.curve_key(point, low) == service.curve_key(point, dict(low, resource_source="other"))


def test_probe_and_csv_default_semantics_normalize_without_crossing_layouts():
    base = sample(operator="fft", batch_stride="0", modulus="0", word_bits="0",
                  direction="forward")
    csv_alias = sample(operator="fft", batch="64", batch_stride="8191", modulus="",
                       word_bits="", direction="forward", inverse="0")
    assert service.curve_key(base, group(1)) == service.curve_key(csv_alias, group(1))
    different_stride = sample(operator="fft", batch_stride="4097")
    assert service.curve_key(base, group(1)) != service.curve_key(different_stride, group(1))


def test_resolved_ntt_mapping_is_not_split_by_descriptive_csv_columns():
    point = sample(operator="ntt", precision="word32", modulus=998244353,
                   placement="natural", mapping_json=json.dumps({
                       "schema_version": 1, "kind": "ntt", "backend": "shared-iterative",
                       "compute_unit": "radix2"}))
    csv_sample = dict(point, cross_twiddle="fused", boundary_storage="")
    assert service.curve_key(point, group(1)) == service.curve_key(csv_sample, group(1))
    changed = json.loads(point["mapping_json"])
    changed["cross_twiddle"] = "table"
    assert service.curve_key(point, group(1)) != service.curve_key(
        dict(point, mapping_json=json.dumps(changed)), group(1))


def test_ntt_modulus_csv_identity_preserves_integers_above_float_precision():
    modulus = 576460756061519873
    point = sample(operator="ntt", precision="word64", modulus=modulus, placement="natural")
    csv_point = dict(point, modulus=str(modulus))
    assert service.curve_key(point, group(1)) == service.curve_key(csv_point, group(1))
    assert service.curve_key(point, group(1)) != service.curve_key(
        dict(point, modulus=str(modulus - 1)), group(1))


def test_ntt_output_order_services_cannot_reuse_natural_order_timings():
    natural = sample(operator="ntt", precision="word64", modulus=576460756061519873,
                     placement="natural", output_order="natural")
    reversed_output = dict(natural, output_order="bit-reversed", placement="bit-reversed")
    terminal = group(4, first_stage=8, stage_count=8)
    profile = service.build_profile([record(natural, [terminal])])
    assert service.curve_key(natural, terminal) != service.curve_key(reversed_output, terminal)
    assert service.predict_group(natural, terminal, profile)["covered"]
    assert not service.predict_group(reversed_output, terminal, profile)["covered"]
    # The historical placement spelling and the explicit semantic field agree.
    alias = dict(reversed_output)
    alias.pop("output_order")
    assert service.curve_key(alias, terminal) == service.curve_key(reversed_output, terminal)


def test_piecewise_curve_exact_midpoint_and_outside_hull():
    s = sample()
    profile = service.build_profile([record(s, [group(1, duration=2), group(5, duration=10)])])
    exact = service.predict_group(s, group(1), profile)
    midpoint = service.predict_group(s, group(3), profile)
    outside = service.predict_group(s, group(6), profile)
    assert exact["covered"] and exact["source"] == "measured-exact"
    assert midpoint["covered"] and midpoint["kernel_ms"] == pytest.approx(6.0)
    assert not outside["covered"] and outside["extrapolated"]
    assert outside["kernel_ms"] is None


def test_resource_regimes_and_unknown_resources_do_not_pool():
    s = sample()
    known = group(1, live=1024)
    unknown = dict(known)
    unknown.pop("live_shared_bytes")
    unknown["compiler_resources_known"] = False
    assert service.curve_key(s, known) != service.curve_key(s, unknown)
    other = dict(known, live_shared_bytes=2048)
    profile = service.build_profile([record(s, [known, other])])
    result = service.predict_group(s, unknown, profile)
    assert not result["covered"] and "resource" in result["reason"]


def test_composite_actual_kernel_resources_split_curves_but_single_kernel_alias_does_not():
    s = sample()
    flat = group(1)
    one_kernel = dict(flat, actual_kernels=[{
        "threads": 64, "live_shared_bytes": 1024, "dynamic_shared_bytes": 0,
        "compiler_registers_per_thread": 32, "grid_ctas": 1,
    }])
    assert service.curve_key(s, flat) == service.curve_key(s, one_kernel)

    composite = dict(flat, actual_kernels=[
        {"threads": 64, "live_shared_bytes": 1024, "dynamic_shared_bytes": 0,
         "compiler_registers_per_thread": 32, "grid_ctas": 10},
        {"threads": 128, "live_shared_bytes": 2048, "dynamic_shared_bytes": 512,
         "compiler_registers_per_thread": 48, "grid_ctas": 10},
    ], compiler_resources_known=False)
    different_resources = dict(composite, actual_kernels=[
        composite["actual_kernels"][0],
        dict(composite["actual_kernels"][1], dynamic_shared_bytes=1024),
    ])
    different_registers = dict(composite, actual_kernels=[
        composite["actual_kernels"][0],
        dict(composite["actual_kernels"][1], compiler_registers_per_thread=64,
             dynamic_shared_bytes=512),
    ])
    different_load = dict(composite, actual_kernels=[
        dict(composite["actual_kernels"][0], grid_ctas=99),
        dict(composite["actual_kernels"][1], grid_ctas=99),
    ])
    assert service.curve_key(s, composite) != service.curve_key(s, different_resources)
    assert service.curve_key(s, composite) != service.curve_key(s, different_registers)
    assert service.curve_key(s, composite) == service.curve_key(s, different_load)


def test_resource_resolution_requires_one_same_shape_regime():
    s = sample()
    known = group(1, live=1024)
    profile = service.build_profile([record(s, [known])])
    unknown = dict(known)
    unknown.pop("live_shared_bytes")
    unknown["compiler_resources_known"] = False
    resolved = service.resolve_resources(s, unknown, profile)
    assert resolved["live_shared_bytes"] == 1024
    assert service.predict_group(s, resolved, profile)["covered"]
    other = dict(known, live_shared_bytes=2048)
    two = service.build_profile([record(s, [known, other])])
    unresolved = service.resolve_resources(s, unknown, two)
    assert "live_shared_bytes" not in unresolved


def test_resource_resolution_cache_invalidates_when_curve_set_grows():
    s = sample()
    known = group(1, live=1024)
    unknown = dict(known)
    unknown.pop("live_shared_bytes")
    unknown["compiler_resources_known"] = False

    profile = service.build_profile([record(s, [known])])
    resolved = service.resolve_resources(s, unknown, profile)
    assert resolved["live_shared_bytes"] == 1024

    other = dict(known, live_shared_bytes=2048)
    expanded = service.build_profile([record(s, [known, other])])
    profile["curves"].update(expanded["curves"])
    unresolved = service.resolve_resources(s, unknown, profile)
    assert "live_shared_bytes" not in unresolved


def test_resource_resolution_index_preserves_unknown_resource_semantics():
    s = sample()
    known = group(1, live=1024)
    other = dict(known, live_shared_bytes=2048)
    unknown = dict(known)
    unknown.pop("live_shared_bytes")
    unknown["compiler_resources_known"] = False

    profile = service.build_profile([record(s, [known, other])])
    unresolved = service.resolve_resources(s, unknown, profile)
    assert unresolved == unknown

    resolved = service.resolve_resources(s, known, profile)
    assert resolved == known


def test_validation_is_independent_and_empty_holdout_is_not_success():
    s = sample()
    profile = service.build_profile([record(s, [group(1), group(5), group(3, role="validation", duration=3)])])
    report = service.validation_report(profile)
    assert report["status"] == "passed"
    assert report["heldout_median_relative_error"] <= 0.10
    assert report["heldout_p90_relative_error"] <= 0.20
    validation = service.predict_group(s, group(3, role="validation", duration=3), profile)
    assert validation["source"] == "independent-validation-exact"
    assert validation["kernel_ms"] == pytest.approx(3.0)
    empty = service.build_profile([record(s, [group(1), group(5)])])
    assert empty["validation"]["status"] == "insufficient-holdout"
    assert not empty["validation"]["passed"]


def test_replicate_aliases_merge_raw_trials_and_cache_identity():
    s = sample()
    alias = dict(group(1), index=8, trial_kernel_ms=[3.0, 3.2], repeat=20)
    profile = service.build_profile([record(s, [group(1), alias])], identity={"device": "A100"})
    points = next(iter(profile["curves"].values()))["train_points"]
    assert len(points) == 1
    assert points[0]["trial_count"] == 4
    assert points[0]["kernel_ms"] == pytest.approx(2.0)
    assert profile["identity"]["device"] == "A100"
    assert profile["cache_identity"]
    other = service.build_profile([record(s, [group(1)])], identity={"device": "B100"})
    assert profile["cache_identity"] != other["cache_identity"]


def test_stage_cost_composes_measured_full_group_duration_without_launch_double_count():
    s = sample(batch="1")
    first = group(1, duration=2)
    second = dict(group(1, duration=3), first_stage=4)
    # Distinguish the two groups by their physical first stage while keeping
    # the mechanism otherwise identical.
    profile = service.build_profile([record(s, [first, second])])
    hardware = {
        "sm_count": 2,
        "stage_service": profile,
    }
    model = stage_cost_model.fit([], hardware)
    predicted = stage_cost_model.predict(dict(s, execution_groups_json=[first, second]), model, True)
    assert predicted["kernel_ms"] == pytest.approx(5.0)
    assert not predicted["whole_plan_fallback_used"]
    assert predicted["unmeasured_groups"] == []
    assert predicted["concurrency_calibrated"] is False


def test_legacy_compatibility_marker_keeps_v3_prediction_on_whole_plan_fallback():
    s = sample(batch="1")
    measured = service.build_profile([record(s, [group(1, duration=2)])])
    hardware = {
        "sm_count": 2,
        "capabilities": {
            "global_feedback_bytes_per_second": 1e12,
            "equivalent_butterflies_per_second": 1e11,
            "interstage_shared_bytes_per_second": 1e13,
            "cta_barriers_per_second": 1e9,
        },
        "stage_service": measured,
    }
    legacy = stage_cost_model.fit([], hardware, compatibility="legacy")
    assert legacy["version"] == stage_cost_model.LEGACY_VERSION
    details = stage_cost_model.predict(dict(s, execution_groups_json=[group(1)]), legacy, True)
    assert details["service_model"] == "legacy-whole-plan-fallback"
    assert details["whole_plan_fallback_used"] is True


def test_stage_cost_marks_unknown_group_as_fallback_when_resource_regimes_are_ambiguous():
    s = sample(batch="1")
    first = group(1, duration=2)
    second = dict(group(1, duration=3), live_shared_bytes=2048)
    profile = service.build_profile([record(s, [first, second])])
    hardware = {"sm_count": 2, "capabilities": {}, "stage_service": profile}
    model = stage_cost_model.fit([], hardware)
    unknown = dict(first)
    unknown.pop("live_shared_bytes")
    details = stage_cost_model.predict(dict(s, execution_groups_json=[unknown]), model, True)
    assert details["whole_plan_fallback_used"]
    assert details["unmeasured_groups"][0]["resources_resolved"] is False


def test_cost_report_requires_three_serial_checks_and_complete_plan_accuracy():
    base = sample(batch=1)
    service_profile = service.build_profile([
        record(base, [group(1, duration=2), group(5, duration=10),
                      group(3, role="validation", duration=6)]),
    ])
    hardware = {
        "sm_count": 2,
        "capabilities": {
            "global_feedback_bytes_per_second": 1e12,
            "equivalent_butterflies_per_second": 1e11,
            "interstage_shared_bytes_per_second": 1e13,
            "cta_barriers_per_second": 1e9,
        },
        "stage_service": service_profile,
    }

    rows = []
    for batch in (1, 2, 4):
        for name in ("a", "b"):
            point = sample(batch=batch)
            point["execution_groups_json"] = [group(1, duration=2)]
            rows.append({"name": f"{batch}-{name}", "sample": point,
                         "median_kernel_ms": 2.0})
    report = stage_cost_model.report(rows, hardware)
    assert report["status"] == "calibrated-local-serial"
    assert report["whole_plan_validation"]["status"] == "passed"
    assert report["validation"]["holdout_median_relative_error"] == pytest.approx(0.0)
    assert report["validation"]["stage_service_holdout_median_relative_error"] == pytest.approx(0.0)
    assert report["validation"]["global_validation"]["passed"] is True

    overlap_rows = [dict(row, sample=dict(row["sample"], stage_overlap=1, batch_tile_count=1),
                         median_kernel_ms=2.0 * int(row["sample"]["batch"])) for row in rows]
    overlap_report = stage_cost_model.report(overlap_rows, hardware)
    assert overlap_report["status"] == "calibrated-local-warning"
    assert overlap_report["composition_validation"]["status"] == "warning-uncovered"
    assert overlap_report["validation"]["global_validation"]["passed"] is False
    assert overlap_report["concurrency_calibrated"] is False

    biased_rows = [dict(row, median_kernel_ms=4.0) for row in rows]
    biased = stage_cost_model.report(biased_rows, hardware)
    assert biased["validation"]["stage_service_holdout_median_relative_error"] == pytest.approx(0.0)
    assert biased["whole_plan_validation"]["status"] == "warning-accuracy"
    assert biased["status"] == "calibrated-local-warning"

    short = stage_cost_model.report(rows[:4], hardware)
    assert short["validation"]["holdout_valid_multi_candidate_workloads"] == 2
    assert short["status"] == "calibrated-local-warning"


def test_overlap_uses_actual_tile_curve_and_exact_tail_launch_cost():
    full_sample = sample(batch=4, stage_overlap=0)
    tail_sample = sample(batch=1, stage_overlap=0)
    full_group = group(4, grid_ctas=4, duration=10, batch_space=1)
    tail_group = group(1, grid_ctas=1, duration=4, batch_space=1)
    service_profile = service.build_profile([record(full_sample, [full_group]),
                                             record(tail_sample, [tail_group])])
    model = stage_cost_model.fit([], {"stage_service": service_profile, "sm_count": 1})
    overlap = sample(batch=9, stage_overlap=1, batch_tile_count=4,
                     execution_groups_json=[dict(full_group, batch_space=4)])
    overlap["stage_service_projection"] = {
        "status": "resolved", "tile_batch": 4, "tail_batch": 1, "tile_count": 3,
        "full": {"sample": full_sample, "groups": [full_group]},
        "tail": {"sample": tail_sample, "groups": [tail_group]}}
    prediction = stage_cost_model.predict(overlap, model, True)
    assert prediction["kernel_ms"] == pytest.approx(24)
    assert not prediction["whole_plan_fallback_used"]
    assert prediction["service_projection"] == "actual-bulk-tile-and-tail"
    assert not prediction["concurrency_calibrated"]
    overlap["batch"] = 10
    with pytest.raises(ValueError, match="stale batch service"):
        stage_cost_model.predict(overlap, model)


@pytest.mark.parametrize("floor,expected", [(12.0, 12.0), (0.1, 4.0)])
def test_pipeline_submission_floor_is_a_bound_not_an_additive_launch_cost(floor, expected):
    import pipeline_schedule_model

    base = sample(batch=1, stage_overlap=0)
    groups = [group(1, grid_ctas=1, duration=2, batch_space=1),
              dict(group(1, grid_ctas=1, duration=2, batch_space=1), first_stage=4)]
    service_profile = service.build_profile([record(base, groups)])
    pipeline = pipeline_schedule_model.summarize({
        "schema": pipeline_schedule_model.SCHEMA,
        "rows": [{"groups": 2, "tiles": 1, "trial_kernel_ms": [floor], "correct": True}]})
    model = stage_cost_model.fit([], {"stage_service": service_profile,
                                     "pipeline_scheduling": pipeline, "sm_count": 1})
    overlap = sample(batch=1, stage_overlap=1, batch_tile_count=1, execution_groups_json=groups)
    overlap["stage_service_projection"] = {
        "status": "resolved", "tile_batch": 1, "tail_batch": 0, "tile_count": 1,
        "full": {"sample": base, "groups": groups}, "tail": None}
    result = stage_cost_model.predict(overlap, model, True)
    assert result["kernel_ms"] == pytest.approx(expected)
    assert result["resource_estimate_ms"] == pytest.approx(4.0)
    assert result["pipeline_scheduling"]["covered"]
    assert not result["concurrency_calibrated"]
