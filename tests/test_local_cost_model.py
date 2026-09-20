import json
import math
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_local_cost_model as cost_model


def _sample(operator="fft", backend="temporal-tile", log_n=12, batch=1024):
    return {
        "operator": operator,
        "precision": "fp32",
        "placement": "out-of-place",
        "backend": backend,
        "logN": str(log_n),
        "N": str(1 << log_n),
        "batch": str(batch),
        "decomposition_count": "1",
        "tile_threads": "512",
        "local_stages": "6",
        "prefix_threads": "0",
        "suffix_threads": "0",
    }


def test_external_cufft_rows_are_not_training_rows(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "operator_calibration.json"
    rows = []
    for index, backend in enumerate(("temporal-tile", "hierarchical", "online-reorder")):
        rows.append({
            "name": f"internal-{index}",
            "status": "measured",
            "correct": True,
            "median_kernel_ms": 0.1 + index * 0.01,
            "samples": [_sample(backend=backend)],
        })
    rows.append({
        "name": "fft-log12-cufft",
        "status": "measured",
        "correct": True,
        "median_kernel_ms": 0.05,
        "samples": [_sample(backend="cufft")],
    })
    path.write_text(json.dumps(rows))
    internal, external = cost_model.load_rows(path)
    assert len(internal) == 3
    assert [row["name"] for row in external] == ["fft-log12-cufft"]


def test_fit_and_predict_are_finite():
    rows = [{"median_kernel_ms": 0.1 + index * 0.01, "sample": _sample(log_n=8 + index)} for index in range(4)]
    coefficients, means, scales = cost_model.fit(rows, 1.0e-3)
    estimate = cost_model.predict(rows[0]["sample"], coefficients, means, scales)
    assert len(coefficients) == len(cost_model.FEATURE_NAMES)
    assert estimate >= 0.0
    extrapolated = dict(rows[0]["sample"], logN="24", N=str(1 << 24), batch="1", batch_stride=str(1 << 24))
    assert math.isfinite(cost_model.predict(extrapolated, coefficients, means, scales))


def test_resolved_physical_groups_override_legacy_config_count():
    # These legacy CSV records have no explicit config group mappings, but
    # their resolved physical plans launch many shared-iterative groups.
    for operator, precision, count in (("fft", "fp32", 18), ("ntt", "word64", 20), ("fwht", "fp64", 12)):
        sample = dict(_sample(operator=operator, backend="shared-iterative", log_n=count, batch=16),
                      precision=precision, execution_group_count="0", stage_overlap="1", batch_tile_count="1")
        groups = [dict(stage_count=1, live_shared_bytes=32, data_time=1, grid_ctas=128) for _ in range(count)]
        sample["execution_groups_json"] = json.dumps(groups)
        measured = cost_model.feature_vector(sample)
        projected = cost_model.feature_vector(dict(sample, execution_group_count=count, execution_groups_json=groups))
        assert measured == projected
        features = dict(zip(cost_model.FEATURE_NAMES, measured))
        assert features["physical_execution_groups"] == count
        assert features["launch_count"] == count * 16
        assert features["boundary_event_count"] == (count - 1) * (4 * 16 - 2)
        # A stale positive explicit count must not override the physical plan.
        assert cost_model.feature_vector(dict(sample, execution_group_count="1")) == projected


def test_standardization_handles_large_finite_columns_without_overflow():
    mean, spread = cost_model._column_center_scale([1.7e308, 1.7e308])
    assert math.isfinite(mean)
    assert math.isfinite(spread)
    assert spread > 0.0


def test_complete_workload_key_uses_canonical_direction_normalization_and_strides():
    base = _sample(log_n=10, batch=8)
    base_key = cost_model.workload_key({"sample": base})
    assert base_key == cost_model.workload_key({"sample": dict(base, batch_stride="0")})
    assert base_key != cost_model.workload_key({
        "sample": dict(base, direction="inverse", normalization="inverse",
                        element_stride="2", batch_stride=str((1 << 10) * 2 + 7))
    })


def test_complete_workload_holdout_reports_top1_regret_and_valid_count():
    rows = []
    for workload in range(4):
        log_n = 8 + workload
        for candidate, backend in enumerate(("temporal-tile", "hierarchical")):
            sample = _sample(backend=backend, log_n=log_n, batch=8 + workload)
            sample["N"] = str(1 << log_n)
            if workload == 1:
                sample.update(direction="inverse", normalization="inverse")
            if workload == 2:
                sample.update(element_stride="2", batch_stride=str((1 << log_n) * 2 + 5))
            rows.append({"name": f"w{workload}-candidate{candidate}",
                         "median_kernel_ms": 0.1 + workload * 0.01 + candidate * 0.02,
                         "sample": sample})
    validation = cost_model.grouped_holdout_validation(rows, 1.0e-3)
    assert validation["method"] == "leave_one_complete_workload_out"
    assert validation["valid_multi_candidate_workloads"] == 4
    assert len(validation["workload_winner_checks"]) == 4
    assert all(item["training_rows"] == 6 for item in validation["workload_winner_checks"])
    assert 0.0 <= validation["top1_accuracy"] <= 1.0
    assert math.isfinite(validation["latency_regret"])
    assert math.isfinite(validation["geomean_latency_regret"])


def _identity_sample(log_n=12):
    mapping = dict(schema_version=1, kind="butterfly", backend="online-reorder",
                   fft_core="register-tile", compute_unit="auto", tile_threads=128,
                   stage_partition=[6, log_n - 6], boundaries=[], segment_mappings=[],
                   execution_group_mappings=[])
    return dict(_sample(log_n=log_n), mapping_json=json.dumps(mapping),
                device="test-device", compute_capability="8.0", runtime_fingerprint="build:auto")


def test_validation_recognizes_resolved_mapping_aliases_and_json_order():
    sample = _identity_sample()
    equivalent = dict(sample, mapping_json=json.dumps(json.loads(sample["mapping_json"]), sort_keys=True),
                      batch_stride="0", kernel_ms="999")
    first = dict(name="incumbent", sample=sample)
    alias = dict(name="new-search-name", sample=equivalent)
    assert cost_model._winner_match(first, alias)
    assert cost_model._distinct_candidate_count([first, alias]) == 1


@pytest.mark.parametrize("field,value", [("direction", "inverse"), ("batch", "8"),
    ("device", "other-device"), ("compute_capability", "9.0"),
    ("global_memory_bytes", "123"), ("runtime_fingerprint", "other-build:research"),
    ("mapping_json", "not-json"), ("mapping_json", '{"schema_version":1,"backend":"online-reorder"}')])
def test_validation_known_conflicts_cannot_match_even_with_same_name(field, value):
    first = dict(name="same-name", sample=_identity_sample())
    other = dict(name="same-name", sample=dict(first["sample"], **{field: value}))
    assert not cost_model._winner_match(first, other)


def test_validation_distinguishes_mapping_axes_and_rejects_partial_aliases():
    sample = _identity_sample()
    different_mapping = dict(json.loads(sample["mapping_json"]), tile_threads=256)
    assert not cost_model._winner_match(dict(name="same", sample=sample),
        dict(name="same", sample=dict(sample, mapping_json=json.dumps(different_mapping))))
    for raw in ('{"schema_version":1,"backend":"online-reorder"}', 'invalid', ''):
        incomplete = dict(sample, mapping_json=raw)
        assert not cost_model._winner_match(dict(name="a", sample=incomplete), dict(name="b", sample=incomplete))
    legacy = dict(name="old-candidate", sample=_sample())
    assert cost_model._winner_match(legacy, dict(legacy))


def test_alias_only_workload_cannot_certify_ranking(monkeypatch):
    rows = []
    for log_n in (12, 14, 16):
        sample = _identity_sample(log_n)
        for i in range(3):
            current = dict(sample, mock_prediction=str(i))
            if log_n != 12 and i == 2:
                current["mapping_json"] = json.dumps(dict(json.loads(sample["mapping_json"]), tile_threads=256))
            rows.append(dict(name=f"{log_n}-alias-{i}", sample=current, median_kernel_ms=0.11 if i == 0 else 0.1))
    monkeypatch.setattr(cost_model, "fit", lambda *args: ([], [], []))
    monkeypatch.setattr(cost_model, "predict", lambda sample, *args: float(sample["mock_prediction"]))
    validation = cost_model.grouped_holdout_validation(rows, .001)
    assert validation["identity_version"] == cost_model.VALIDATION_IDENTITY_VERSION
    assert validation["valid_multi_candidate_workloads"] == 2
    assert validation["top1_accuracy"] == 1
    assert validation["mean_latency_regret"] == pytest.approx(1.1)
    assert all(c["distinct_candidate_count"] == 2 and c["candidate_rows"] == 3
               for c in validation["workload_winner_checks"])
    assert validation["skipped_workloads"][0]["reason"] == "insufficient_distinct_candidates"


def test_training_winner_does_not_certify_without_complete_workload_holdout(tmpdir, monkeypatch):
    calibration = pathlib.Path(str(tmpdir)) / "operator_calibration.json"
    profile = pathlib.Path(str(tmpdir)) / "profile.json"
    output = pathlib.Path(str(tmpdir)) / "model.json"
    samples = []
    for index, backend in enumerate(("temporal-tile", "hierarchical", "online-reorder")):
        samples.append({"name": f"candidate-{index}", "status": "measured", "correct": True,
                        "median_kernel_ms": 0.1 + index * 0.01,
                        "samples": [_sample(backend=backend)]})
    calibration.write_text(json.dumps(samples))
    profile.write_text(json.dumps({"device": "test", "compute_capability": "sm80"}))
    monkeypatch.setattr(sys, "argv", ["fit_local_cost_model.py", "--profile", str(profile),
                                        "--operator-calibration", str(calibration), "--output", str(output)])
    assert cost_model.main() == 0
    result = json.loads(output.read_text())
    assert result["status"] == "calibrated-local-warning"
    assert result["usable_for_unmeasured_ranking"] is False
    assert result["validation"]["training_workload_winner_accuracy"] is not None
    assert result["validation"]["holdout_valid_multi_candidate_workloads"] == 0
    assert result["validation"]["training_winner_accuracy_used_for_pass"] is False


def test_layout_arithmetic_and_numeric_contracts_have_distinct_features():
    base = _sample()
    features = cost_model.feature_vector(base)
    for field, value in (("shared_layout", "writer-aligned"), ("direct_boundary", "prefix-tiled-transpose"),
                         ("complex_multiply", "gauss3"), ("direction", "inverse"), ("element_stride", 2),
                         ("placement", "in-place"), ("fft_core", "register-tile"),
                         ("group_cores", "register-tile:cufftdx-block"), ("precision", "fp64")):
        assert features != cost_model.feature_vector(dict(base, **{field: value}))


def test_same_group_count_keeps_distinct_physical_live_state():
    from calibration_space import predicted_sample
    def sample(partition):
        mapping=dict(backend="shared-iterative",stage_partition=partition,tile_threads=128)
        return predicted_sample(dict(operator="fft",precision="fp64",logN=12,batch=4,mapping_json=json.dumps(mapping)))
    assert cost_model.feature_vector(sample([4,4,4])) != cost_model.feature_vector(sample([2,5,5]))
    base=sample([4,4,4])
    assert cost_model.feature_vector(dict(base,operator="ntt",precision="word32",modulus=2013265921)) != cost_model.feature_vector(dict(base,operator="subset-zeta",precision="uint32"))


def test_factor_model_counts_physical_passes_and_independent_prefetch_capacity():
    from calibration_space import predicted_sample
    mapping=dict(backend="factor-streamed",fft_core="cufftdx-block",stage_partition=[24],factor_partition=[8,8,8],
                 factor_ept=16,factor_columns=8,data_tiles_per_cta=5,prefetch_depth=2)
    sample=predicted_sample(dict(operator="fft",precision="fp64",logN=24,batch=1,mapping_json=json.dumps(mapping)))
    assert sample["decomposition_count"]==1 and sample["execution_group_count"]==3
    assert sample["execution_groups_json"][0]["grid_ctas"]==1639
    assert sample["execution_groups_json"][0]["live_shared_bytes"]==98304
    f=dict(zip(cost_model.FEATURE_NAMES,cost_model.feature_vector(sample)))
    assert f["launch_count"]==3 and f["prefetch_depth"]==2 and f["cta_data_tiles"]==5
    assert f["compiler_local_known_fraction"]==0
    for g in sample["execution_groups_json"]:
        g.update(compiler_local_resources_known=True,compiler_local_bytes_per_thread=256)
    assert f!=dict(zip(cost_model.FEATURE_NAMES,cost_model.feature_vector(sample)))


def test_factor_schedule_features_distinguish_serial_and_overlap_execution():
    from calibration_space import predicted_sample

    base_mapping = dict(backend="factor-streamed", fft_core="cufftdx-block",
                        stage_partition=[24], factor_partition=[8, 8, 8], factor_ept=16,
                        factor_columns=8, data_tiles_per_cta=5, prefetch_depth=2,
                        factor_slices=4)

    def features(overlap):
        mapping = dict(base_mapping, factor_overlap=overlap)
        sample = predicted_sample(dict(operator="fft", precision="fp64", logN=24,
                                       batch=1, mapping_json=json.dumps(mapping)))
        return sample, dict(zip(cost_model.FEATURE_NAMES, cost_model.feature_vector(sample)))

    serial_sample, serial = features(False)
    overlap_sample, overlapped = features(True)
    assert serial["factor_slices"] == overlapped["factor_slices"] == 4
    assert serial["factor_overlap"] == 0 and overlapped["factor_overlap"] == 1
    assert serial["factor_total_launches"] == overlapped["factor_total_launches"] == 9
    assert serial["launch_count"] == overlapped["launch_count"] == 9
    assert serial["factor_partial_ready_groups"] == 0
    assert overlapped["factor_partial_ready_groups"] == 1
    assert serial["factor_event_waits"] == 0
    assert overlapped["factor_event_waits"] == 10
    assert serial["factor_boundary_workspace_mib"] == overlapped["factor_boundary_workspace_mib"] > 0
    assert serial["factor_total_ctas"] == overlapped["factor_total_ctas"]
    assert serial["factor_max_grid_ctas_per_launch"] == overlapped["factor_max_grid_ctas_per_launch"]
    assert serial["ring_buffer_mib"] == overlapped["ring_buffer_mib"] == 0
    assert serial_sample["execution_groups_json"] != []
    assert overlap_sample["execution_groups_json"] != []
    assert serial != overlapped


def test_factor_workspace_uses_runtime_buffer_count_and_strided_extent():
    from calibration_space import predicted_sample

    mapping = dict(backend="factor-streamed", fft_core="cufftdx-block",
                   stage_partition=[24], factor_partition=[6, 6, 6, 6], factor_ept=16,
                   factor_columns=8, data_tiles_per_cta=1, prefetch_depth=0)
    n = 1 << 24
    batch = 3
    element_stride = 2
    batch_stride = 2 * n + 7
    extent = (batch - 1) * batch_stride + (n - 1) * element_stride + 1
    element_bytes = 16  # FP64 complex FFT.

    def workspace(slices, reported_bytes=None):
        point = dict(operator="fft", precision="fp64", logN=24, batch=batch,
                     element_stride=element_stride, batch_stride=batch_stride,
                     mapping_json=json.dumps(dict(mapping, factor_slices=slices)))
        if reported_bytes is not None:
            point["workspace_bytes"] = reported_bytes
        sample = predicted_sample(point)
        features = dict(zip(cost_model.FEATURE_NAMES, cost_model.feature_vector(sample)))
        return features

    serial = workspace(1)
    sliced = workspace(2)
    expected_serial = 2 * extent * element_bytes / (1024.0 * 1024.0)
    expected_sliced = 3 * extent * element_bytes / (1024.0 * 1024.0)
    assert math.isclose(serial["factor_boundary_workspace_mib"], expected_serial)
    assert math.isclose(sliced["factor_boundary_workspace_mib"], expected_sliced)
    assert sliced["factor_boundary_workspace_mib"] > serial["factor_boundary_workspace_mib"]
    assert workspace(1, 123456)["factor_boundary_workspace_mib"] == 123456 / (1024.0 * 1024.0)


def test_verified_sweep_csv_is_grouped_and_unverified_rows_are_ignored(tmpdir):
    path = pathlib.Path(str(tmpdir)) / "sweep.csv"
    fields = ["operator", "precision", "placement", "logN", "batch", "backend", "compute_unit",
              "complex_multiply", "cross_twiddle", "local_exchange", "shared_layout", "fft_core",
              "stage_space", "tile_threads", "prefix_threads", "suffix_threads", "prefix_ept",
              "suffix_ept", "local_stages", "reorder_columns", "warp_stages", "pipeline_warps",
              "N", "kernel_ms", "correct", "measurement_exclusive_gpu"]
    values = {field: "0" for field in fields}
    values.update({"operator": "fft", "precision": "fp32", "placement": "out-of-place", "logN": "12",
                   "batch": "1024", "backend": "temporal-tile", "compute_unit": "radix4", "fft_core": "scalar",
                   "N": "4096", "kernel_ms": "0.1", "correct": "1", "measurement_exclusive_gpu": "1"})
    bad = dict(values, correct="0", kernel_ms="0.01")
    with path.open("w", newline="") as handle:
        import csv
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(values)
        writer.writerow(dict(values, kernel_ms="0.12"))
        writer.writerow(bad)
    rows = cost_model.load_sweep_rows([path])
    assert len(rows) == 1
    assert abs(rows[0]["median_kernel_ms"] - 0.11) < 1.0e-9


def test_legacy_sweep_display_name_collision_preserves_distinct_launch_axes(tmp_path):
    import csv

    path = tmp_path / "legacy.csv"
    base = dict(_sample(), correct="1", measurement_exclusive_gpu="1", kernel_ms="0.1")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base))
        writer.writeheader()
        writer.writerow(dict(base, tile_threads="64"))
        writer.writerow(dict(base, tile_threads="256", kernel_ms="0.2"))
    rows = cost_model.load_sweep_rows([path])
    assert len(rows) == 2 and rows[0]["name"] == rows[1]["name"]
    assert cost_model._distinct_candidate_count(rows) == 2
    assert not cost_model._winner_match(rows[0], rows[1])


def test_ntt_csv_groups_and_pipeline_events_are_not_flattened():
    from fit_local_cost_model import feature_vector, FEATURE_NAMES
    sample=dict(operator="ntt",precision="word64",logN=12,N=4096,batch=7,
                logical_subgraphs=3,execution_groups=3,stage_overlap=1,batch_tile_count=2)
    features=dict(zip(FEATURE_NAMES,feature_vector(sample)))
    assert features["physical_execution_groups"]==3
    assert features["launch_count"]==12
    assert features["boundary_event_count"]==28
    assert features["ring_buffer_mib"]==.25


def test_register_fft_grid_counts_ctas_instead_of_individual_columns():
    from calibration_space import predicted_sample
    mapping=dict(backend="online-reorder",fft_core="register-tile",stage_partition=[8,12],
                 prefix_threads=128,prefix_ept=16,suffix_threads=512,suffix_ept=16,
                 stage_overlap=True,batch_tile_count=4)
    sample=predicted_sample(dict(operator="fft",precision="fp32",logN=20,batch=17,mapping_json=json.dumps(mapping)))
    groups=sample["execution_groups_json"]
    assert [g["grid_ctas"] for g in groups]==[2048,512]
    assert [g["live_shared_bytes"] for g in groups]==[16384,65536]
    assert [g["data_time"] for g in groups]==[8,8]
