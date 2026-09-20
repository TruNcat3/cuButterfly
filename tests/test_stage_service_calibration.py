"""CPU-only tests for the adaptive stage service calibration driver."""

import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stage_service_calibration as calibration


class FakeProbe:
    """Small injected runner implementing the stage probe JSON contract."""

    def __init__(self, *, nonlinear=None, interrupt_batch=None):
        self.calls = []
        self.nonlinear = dict(nonlinear or {})
        self.interrupt_batch = interrupt_batch

    def __call__(self, command):
        point = json.loads(command[command.index("--point-json") + 1])
        describe = "--describe-only" in command
        batch = int(point["batch"])
        self.calls.append((batch, describe))
        if not describe and self.interrupt_batch == batch:
            self.interrupt_batch = None
            raise KeyboardInterrupt("fake probe interruption")
        kernel_ms = float(self.nonlinear.get(batch, batch))
        group = {
            "index": 0,
            "first_stage": 0,
            "stage_count": 1,
            "threads": 64,
            "units_per_cta": 1,
            "exchange": 0,
            "live_shared_bytes": 1024,
            "compiler_registers_per_thread": 16,
            "compiler_resources_known": True,
            "independent": True,
            # A resolved physical load changes with batch while the mapping
            # shape/resource regime stays fixed, exactly as the real probe.
            "work_blocks": batch,
            "grid_ctas": batch,
            "trial_kernel_ms": [kernel_ms, kernel_ms * 1.001, kernel_ms * 0.999],
        }
        hardware = {
            "device": "fake-gpu",
            "compute_capability": "8.0",
            "global_memory_bytes": 1 << 40,
            "free_memory_bytes": 1 << 40,
            "sm_count": 2,
            "max_blocks_per_sm": 4,
            "max_threads_per_sm": 1024,
            "registers_per_sm": 65536,
            "shared_bytes_per_sm": 49152,
        }
        return {
            "schema": "cubutterfly-stage-probe-v1",
            "status": "resolved" if describe else "measured",
            "correct": True,
            "sample": point,
            "hardware": hardware,
            "groups": [group],
            "pairs": [],
            "plan_trial_kernel_ms": [],
            "workspace_bytes": 4096,
            "free_memory_bytes": hardware["free_memory_bytes"],
            "probe_payload_buffers": 5,
        }


def _inputs(tmpdir, *, batch=32):
    root = pathlib.Path(str(tmpdir))
    binary = root / "cubutterfly_stage_microbench"
    binary.write_bytes(b"fake-stage-probe")
    checkpoint = root / "stage_calibration.json"
    profile = {
        "device": "fake-gpu",
        "compute_capability": "8.0",
        "global_memory_bytes": 1 << 40,
        "hardware": {
            "device": "fake-gpu",
            "compute_capability": "8.0",
            "global_memory_bytes": 1 << 40,
            "sm_count": 2,
            "max_blocks_per_sm": 4,
            "max_threads_per_sm": 1024,
            "registers_per_sm": 65536,
            "shared_bytes_per_sm": 49152,
        },
    }
    point = {
        "operator": "fft",
        "precision": "fp32",
        "logN": 8,
        "batch": batch,
        "batch_stride": 256,
        "mapping_json": json.dumps({"backend": "fake", "fft_core": "scalar"}),
    }
    return binary, checkpoint, profile, point


def _calibrate(binary, checkpoint, profile, points, runner, **kwargs):
    return calibration.calibrate(
        binary, checkpoint, profile, points, runner, lambda: True,
        warmup=1, repeat=2, trials=3, max_points=33, **kwargs,
    )


def test_acquisition_measures_before_resolving_next_plan(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    runner = FakeProbe()
    other = dict(point, logN=9, batch_stride=512)
    result = _calibrate(binary, checkpoint, profile, [point, other], runner)
    describes = [i for i, (_, describe) in enumerate(runner.calls) if describe]
    assert len(describes) == 2
    assert any(not describe for _, describe in runner.calls[describes[0]:describes[1]])
    assert result["coverage"]["pending_description_count"] == 0
    assert result["coverage"]["requested_static_count"] == 2
    assert result["validated"]


def test_budget_keeps_unvisited_inventory_across_subset_resume(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    runner = FakeProbe()
    other = dict(point, logN=9, batch_stride=512)
    partial = _calibrate(binary, checkpoint, profile, [point, other], runner,
                         mode="bounded", budget=1)
    assert sum(describe for _, describe in runner.calls) == 1
    assert partial["coverage"]["pending_description_count"] == 1
    assert not partial["validated"]
    subset = _calibrate(binary, checkpoint, profile, [point], runner)
    assert subset["calibration_status"] == "incomplete-inventory"
    assert not subset["coverage"]["coverage_complete"]
    before = len(runner.calls)
    complete = _calibrate(binary, checkpoint, profile, [point, other], runner)
    assert complete["validated"]
    assert sum(describe for _, describe in runner.calls[before:]) == 1


def test_acquisition_inventory_corruption_rejected(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    _calibrate(binary, checkpoint, profile, [point, dict(point, logN=9)], FakeProbe(),
               mode="bounded", budget=1)
    saved = json.loads(checkpoint.read_text())
    sidecar = checkpoint.parent / saved["acquisition_inventory_ref"]["path"]
    sidecar.write_text("[]")
    with pytest.raises(ValueError, match="inventory sidecar.*mismatch"):
        _calibrate(binary, checkpoint, profile, [point], FakeProbe())


class SharedPartitionProbe(FakeProbe):
    def __init__(self, changed_resources=False):
        super().__init__()
        self.changed_resources = changed_resources
        self.partitions_timed = []

    def __call__(self, command):
        value = super().__call__(command)
        partition = json.loads(value["sample"]["mapping_json"])["stage_partition"]
        if "--describe-only" not in command:
            self.partitions_timed.append(partition)
        first = 0
        groups = []
        for index, stages in enumerate(partition):
            group = dict(value["groups"][0], index=index, first_stage=first,
                         stage_count=stages, batch_space=1, kernel_launch_count=1)
            group["work_blocks"] = group["grid_ctas"] = value["sample"]["batch"] * (1 << (8 - stages))
            if self.changed_resources and partition == [2, 2, 2, 2]:
                group["compiler_registers_per_thread"] += 8
            groups.append(group)
            first += stages
        value["groups"] = groups
        return value


@pytest.mark.parametrize("changed_resources", [False, True])
def test_cross_partition_reuse_requires_resolved_resources(tmpdir, changed_resources):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    points = [dict(point, backend="shared-iterative", mapping_json=json.dumps({
        "backend": "shared-iterative", "stage_partition": partition, "tile_threads": 64}))
        for partition in ([2, 2, 4], [4, 2, 2], [2, 2, 2, 2])]
    runner = SharedPartitionProbe(changed_resources)
    _calibrate(binary, checkpoint, profile, points[:2], runner)
    result = _calibrate(binary, checkpoint, profile, points[2:], runner)
    assert result["validated"]
    assert ([2, 2, 2, 2] in runner.partitions_timed) == changed_resources
    assert result["coverage"]["service_reused_candidates"] == int(not changed_resources)
    saved = json.loads(checkpoint.read_text())
    if not changed_resources:
        reuse = [candidate for candidate in saved["candidates"] if candidate.get("service_reuse")]
        assert len(reuse) == 1 and len(reuse[0]["service_reuse"]) == 4
        assert all(row["candidate_key"] != reuse[0]["key"] for row in saved["records"])


def test_adaptive_validation_visits_each_unresolved_interval(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir)
    runner = FakeProbe()

    result = _calibrate(binary, checkpoint, profile, [point], runner)

    curve = next(iter(result["coverage"]["curves"].values()))
    validation_calls = [batch for batch, describe in runner.calls if not describe and batch not in curve["seed_batches"]]
    assert len(validation_calls) >= 2
    assert curve["validation_complete"] is True
    assert curve["validation_mode"] == "validated"
    assert result["calibration_status"] == "complete"
    assert result["validated"] is True
    assert result["validation"]["heldout_count"] >= 2


def test_failed_midpoint_is_promoted_and_interval_is_split(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    runner = FakeProbe(nonlinear={4: 100.0})

    result = _calibrate(binary, checkpoint, profile, [point], runner)

    curve = next(iter(result["coverage"]["curves"].values()))
    assert 4 in curve["promoted_batches"]
    assert any(item["state"] == "passed" for item in curve["validation_intervals"])
    assert not curve["unvalidated_intervals"]
    # The failed midpoint causes additional child-interval probes instead of
    # terminating after the first successful holdout.
    assert {2, 3, 5}.issubset({batch for batch, describe in runner.calls if not describe})
    assert result["validated"] is True


def test_interruption_checkpoint_resumes_without_rerunning_measured_load(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir)
    runner = FakeProbe(interrupt_batch=7)

    interrupted = _calibrate(binary, checkpoint, profile, [point], runner)
    assert interrupted["calibration_status"] == "interrupted"
    saved = json.loads(checkpoint.read_text())
    assert any(row["status"] == "measured" and row["batch"] == 1
               for row in saved["records"])

    before_resume = len(runner.calls)
    resumed = _calibrate(binary, checkpoint, profile, [point], runner)
    resumed_calls = runner.calls[before_resume:]
    assert resumed["calibration_status"] == "complete"
    assert not any(batch == 1 and not describe for batch, describe in resumed_calls)


def test_budget_cap_is_global_and_resume_reuses_raw_rows(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir)
    runner = FakeProbe()

    limited = _calibrate(binary, checkpoint, profile, [point], runner, budget=2)
    assert limited["calibration_status"] == "incomplete-budget"
    saved = json.loads(checkpoint.read_text())
    assert len([row for row in saved["records"] if row["status"] == "measured"]) == 2

    complete = _calibrate(binary, checkpoint, profile, [point], runner)
    assert complete["calibration_status"] == "complete"
    assert complete["coverage"]["measured_records"] > 2


def test_exact_discrete_loads_are_distinguished_from_holdout_validation(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    runner = FakeProbe()

    result = _calibrate(binary, checkpoint, profile, [point], runner)

    curve = next(iter(result["coverage"]["curves"].values()))
    assert curve["validation_mode"] == "exact-only"
    assert curve["exact_discrete_load_coverage"] is True
    assert result["calibration_status"] == "complete"
    # Exact coverage is useful evidence but has no independent holdout and is
    # therefore deliberately rejected by the installation validation gate.
    assert result["validated"] is False
    assert result["validation"]["heldout_count"] == 0


def test_non_executable_plan_is_a_reported_omission_without_timed_probe(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)

    class RejectingProbe(FakeProbe):
        def __call__(self, command):
            if "--describe-only" in command:
                point_value = json.loads(command[command.index("--point-json") + 1])
                self.calls.append((int(point_value["batch"]), True))
                return {
                    "schema": "cubutterfly-stage-probe-v1",
                    "status": "unavailable",
                    "correct": False,
                    "reason": "plan construction rejected point",
                    "sample": point_value,
                    "groups": [],
                }
            raise AssertionError("non-executable point reached timed probe")

    runner = RejectingProbe()
    result = _calibrate(binary, checkpoint, profile, [point], runner)

    assert result["coverage"]["non_executable_count"] == 1
    assert result["coverage"]["measured_records"] == 0
    assert result["calibration_status"] == "incomplete-unsupported"
    assert runner.calls == [(8, True)]


def test_overlap_and_factor_slice_requests_are_pending_composition_metadata(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    overlap = dict(point, stage_overlap=True, batch_tile_count=2,
                   mapping_json=json.dumps({"backend": "fake", "stage_overlap": True,
                                            "batch_tile_count": 2}))
    factor = dict(point, factor_slices=2)
    runner = FakeProbe()

    result = _calibrate(binary, checkpoint, profile, [overlap, factor], runner)

    assert result["coverage"]["composition_request_count"] == 2
    assert len(result["coverage"]["composition_requests"]) == 2
    assert result["composition_requests"]
    # The service copy is bulk-only and has synchronized serialized policy.
    timed = [row for row in json.loads(checkpoint.read_text())["records"]
             if row.get("status") == "measured"]
    assert timed
    assert all(int(row["sample"].get("stage_overlap", 0)) == 0 for row in timed)
    assert all(json.loads(row["sample"]["mapping_json"]).get("stage_overlap") == 0
               for row in timed)
    # Pending schedule composition does not make the physical service result
    # itself incomplete; exact-only still remains unvalidated by design.
    assert result["calibration_status"] == "complete"
    assert result["validated"] is False

    saved = json.loads(checkpoint.read_text())
    assert saved["composition_request_count"] == 2
    assert "composition_requests" not in saved
    assert saved["coverage"]["composition_request_count"] == 2
    assert "composition_requests" not in saved["coverage"]
    reference = saved["composition_requests_ref"]
    sidecar = checkpoint.parent / reference["path"]
    sidecar_document = json.loads(sidecar.read_text())
    assert sidecar_document["count"] == 2
    assert len(sidecar_document["composition_requests"]) == 2


def test_legacy_embedded_composition_requests_resume_and_count_preserved(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    request = dict(point, factor_slices=2)
    runner = FakeProbe()

    first = _calibrate(binary, checkpoint, profile, [request], runner)
    saved = json.loads(checkpoint.read_text())
    sidecar = checkpoint.parent / saved["composition_requests_ref"]["path"]
    requests = json.loads(sidecar.read_text())["composition_requests"]

    # Reconstruct the pre-sidecar envelope, including its duplicated coverage
    # list, and ensure it remains a valid resume source.
    saved.pop("composition_requests_ref")
    saved.pop("composition_request_count")
    saved["composition_requests"] = requests
    saved["coverage"]["composition_requests"] = requests
    saved["coverage"].pop("composition_request_count")
    checkpoint.write_text(json.dumps(saved))

    before_resume = len(runner.calls)
    resumed = _calibrate(binary, checkpoint, profile, [request], runner)
    assert len(runner.calls) == before_resume
    assert resumed["composition_requests"] == requests
    assert resumed["coverage"]["composition_request_count"] == len(requests)
    assert len(resumed["coverage"]["composition_requests"]) == len(requests)
    assert first["composition_requests"] == requests


def test_composition_sidecar_hash_mismatch_is_rejected(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    request = dict(point, factor_slices=2)
    _calibrate(binary, checkpoint, profile, [request], FakeProbe())
    saved = json.loads(checkpoint.read_text())
    sidecar = checkpoint.parent / saved["composition_requests_ref"]["path"]
    document = json.loads(sidecar.read_text())
    document["composition_requests"][0]["reason"] = "tampered"
    sidecar.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        _calibrate(binary, checkpoint, profile, [request], FakeProbe())


def test_large_composition_audit_is_written_once_outside_checkpoint(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    points = []
    for index in range(128):
        mapping = {"backend": "fake", "fft_core": "scalar", "variant": index}
        points.append(dict(point, factor_slices=2, mapping_json=json.dumps(mapping)))

    result = _calibrate(binary, checkpoint, profile, points, FakeProbe())
    saved = json.loads(checkpoint.read_text())
    reference = saved["composition_requests_ref"]
    sidecar = checkpoint.parent / reference["path"]
    sidecar_document = json.loads(sidecar.read_text())

    assert result["coverage"]["composition_request_count"] == 128
    assert len(result["composition_requests"]) == 128
    assert saved["composition_request_count"] == 128
    assert saved["coverage"]["composition_request_count"] == 128
    assert "composition_requests" not in saved
    assert "composition_requests" not in saved["coverage"]
    assert len(sidecar_document["composition_requests"]) == 128
    # The checkpoint contains only the reference/count, never the audit list.
    assert '"composition_requests":' not in checkpoint.read_text()


def test_composition_digest_is_precomputed_once_per_invocation(tmpdir, monkeypatch):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=2)
    request = dict(point, factor_slices=2)
    digest_calls = []
    original_digest = calibration._composition_digest

    def count_digest(requests):
        digest_calls.append(len(requests))
        return original_digest(requests)

    monkeypatch.setattr(calibration, "_composition_digest", count_digest)
    _calibrate(binary, checkpoint, profile, [request], FakeProbe())

    assert digest_calls == [1]


def test_batch_expansion_refreshes_description_and_reuses_compatible_timings(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)

    class BatchAwareProbe(FakeProbe):
        def __call__(self, command):
            result = super().__call__(command)
            batch = int(result["sample"]["batch"])
            group = result["groups"][0]
            # Keep the curve shape compatible while changing the load and
            # allocation metadata that a larger describe must refresh.
            group["batch_space"] = 2
            group["grid_ctas"] = batch * 2
            group["work_blocks"] = batch * 2
            result["workspace_bytes"] = 4096 + batch * 32
            return result

    runner = BatchAwareProbe()
    _calibrate(binary, checkpoint, profile, [point], runner)
    first_saved = json.loads(checkpoint.read_text())
    first_records = list(first_saved["records"])
    first_batches = {int(row["batch"]) for row in first_records if row.get("status") == "measured"}
    first_description = next(iter(first_saved["descriptions"].values()))

    expanded = dict(point, batch=16)
    before_resume = len(runner.calls)
    _calibrate(binary, checkpoint, profile, [expanded], runner)
    resumed_calls = runner.calls[before_resume:]
    saved = json.loads(checkpoint.read_text())
    description = next(iter(saved["descriptions"].values()))
    description_group = description["groups"][0]

    assert (16, True) in resumed_calls
    assert description_group["grid_ctas"] == 32
    assert description_group["work_blocks"] == 32
    assert description["workspace_bytes"] == 4096 + 16 * 32
    assert first_description["workspace_bytes"] != description["workspace_bytes"]
    measured_batches = {int(row["batch"]) for row in saved["records"]
                        if row.get("status") == "measured"}
    assert first_batches <= measured_batches
    assert not any(batch in first_batches and not describe
                   for batch, describe in resumed_calls)


def test_curve_model_change_rebuilds_keys_and_validation_without_retiming(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=32)
    runner = FakeProbe()
    first = _calibrate(binary, checkpoint, profile, [point], runner)
    assert first["validated"]
    saved = json.loads(checkpoint.read_text())
    timings = [group["trial_kernel_ms"] for record in saved["records"] for group in record["groups"]]
    saved["curve_model_version"] = "obsolete-key-semantics"
    for candidate in saved["candidates"]:
        candidate["key"] = "stale:" + candidate["key"]
        candidate["actual_key"] = candidate["key"]
        candidate["intervals"] = {"1:32": "uncovered"}
    for record in saved["records"]:
        record["candidate_key"] = "stale:" + record["candidate_key"]
        if record.get("role") == "validation":
            record["validation"] = {"checks": [{"status": "uncovered"}]}
    checkpoint.write_text(json.dumps(saved))
    runner.calls.clear()
    result = _calibrate(binary, checkpoint, profile, [point], runner)
    assert result["validated"]
    assert runner.calls == []
    restored = json.loads(checkpoint.read_text())
    assert not any(c["key"].startswith("stale:") for c in restored["candidates"])
    assert timings == [group["trial_kernel_ms"] for record in restored["records"] for group in record["groups"]]


def test_full_default_reserves_holdouts_after_dense_boundary_seeds(tmpdir, monkeypatch):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=1024)
    seeds = [1, 5, 6, 7, 13, 14, 15, 26, 27, 28, 53, 54, 55,
             107, 108, 109, 215, 216, 217, 431, 432, 433, 863, 864, 865, 1024]
    monkeypatch.setattr(calibration, "_residency_batches", lambda *args: (list(seeds), []))
    result = calibration.calibrate(binary, checkpoint, profile, [point], FakeProbe(),
                                   lambda: True, warmup=1, repeat=2, trials=3)
    assert result["validated"]
    assert result["coverage"]["max_points_per_curve"] > 33
    assert result["coverage"]["measured_points"] > 33
    explicit = _calibrate(binary, checkpoint.with_name("explicit.json"), profile,
                          [point], FakeProbe())
    assert not explicit["validated"]
    assert explicit["coverage"]["measured_points"] == 33


def test_post_exclusivity_failure_is_contaminated_and_retryable(tmpdir):
    binary, checkpoint, profile, point = _inputs(tmpdir, batch=8)
    runner = FakeProbe()
    checks = {"count": 0}

    def contaminated_once():
        checks["count"] += 1
        # describe pre/post, then the first timed load pre/post.
        return checks["count"] != 4

    first = calibration.calibrate(binary, checkpoint, profile, [point], runner,
                                  contaminated_once, warmup=1, repeat=2, trials=3)
    assert first["calibration_status"] == "interrupted"
    saved = json.loads(checkpoint.read_text())
    assert any(row["status"] == "contaminated" for row in saved["records"])

    second = calibration.calibrate(binary, checkpoint, profile, [point], runner,
                                   lambda: True, warmup=1, repeat=2, trials=3)
    assert second["calibration_status"] == "complete"
    assert second["validated"] is True


def _online_description(*, fft_core="cufftdx-block", batch=3, log_n=8, local_stages=4,
                        batch_stride=256, stage_overlap=False, factor_overlap=False,
                        factor_slices=1, batch_tile_count=1, missing=None,
                        multi_kernel=False, prefix_codelet_lanes=1):
    suffix_stages = log_n - local_stages
    suffix_threads = 128
    suffix_ept = 4
    if fft_core == "register-tile" and prefix_codelet_lanes != 1:
        prefix_radius = 1 << (local_stages // 2)
        prefix_columns = 4
        prefix_threads = prefix_radius * prefix_codelet_lanes * prefix_columns
        prefix_ept = prefix_radius // prefix_codelet_lanes
    else:
        prefix_threads = 128
        prefix_ept = 4
    sample = {
        "operator": "fft", "precision": "fp32", "direction": "forward",
        "inverse": False, "normalization": "none", "placement": "out-of-place",
        "logN": log_n, "batch": batch, "batch_stride": batch_stride,
        "element_stride": 1, "backend": "online-reorder", "fft_core": fft_core,
        "compute_unit": "auto", "local_exchange": "shared",
        "shared_layout": "linear", "cross_twiddle": "recurrence",
        "direct_boundary": "strided", "stage_overlap": int(stage_overlap),
        "factor_overlap": int(factor_overlap), "factor_slices": factor_slices,
        "batch_tile_count": batch_tile_count,
    }
    mapping = {
        "schema_version": 1, "kind": "butterfly", "operator": "fft",
        "precision": "fp32", "backend": "online-reorder", "fft_core": fft_core,
        "compute_unit": "auto", "local_exchange": "shared", "shared_layout": "linear",
        "cross_twiddle": "recurrence", "direct_boundary": "strided",
        "stage_partition": [local_stages, suffix_stages],
        "stage_overlap": int(stage_overlap), "factor_overlap": int(factor_overlap),
        "factor_slices": factor_slices, "batch_tile_count": batch_tile_count,
        "prefix_threads": prefix_threads, "prefix_ept": prefix_ept,
        "suffix_threads": suffix_threads, "suffix_ept": suffix_ept,
    }
    if prefix_codelet_lanes != 1:
        sample["prefix_codelet_lanes"] = prefix_codelet_lanes
        mapping["prefix_codelet_lanes"] = prefix_codelet_lanes
    sample["mapping_json"] = json.dumps(mapping)

    def grid(stages, threads, ept, position):
        numerator = batch * (1 << (log_n - stages))
        if fft_core == "register-tile":
            units = (threads * ept) // (1 << stages)
            return numerator // units
        units = (threads * ept) // (1 << stages)
        return (numerator + units - 1) // units

    resources = {
        "dynamic_shared_bytes": 2048, "live_shared_bytes": 4096,
        "compiler_registers_per_thread": 32,
        "compiler_local_bytes_per_thread": 0,
        "compiler_resources_known": True,
        "compiler_local_resources_known": True,
    }
    groups = []
    for position, (first, stages, threads, ept) in enumerate((
            (0, local_stages, prefix_threads, prefix_ept),
            (local_stages, suffix_stages, suffix_threads, suffix_ept))):
        work = grid(stages, threads, ept, position)
        kernel = {"grid_ctas": work, "threads": threads, **resources}
        group = {
            "index": position, "first_stage": first, "stage_count": stages,
            "batch_space": 1, "threads": threads, "elements_per_thread": ept,
            "units_per_cta": (threads * ept) // (1 << stages),
            "core": ("register-tile" if fft_core == "register-tile" and position == 0
                     else "cufftdx-block"),
            "exchange": 2, "shared_layout": "linear", "independent": True,
            "kernel_launch_count": 1, "work_blocks": work, "grid_ctas": work,
            "actual_kernels": [kernel], **resources,
        }
        groups.append(group)
    if missing:
        for group in groups:
            for field in missing:
                group.pop(field, None)
            for kernel in group["actual_kernels"]:
                for field in missing:
                    kernel.pop(field, None)
    if multi_kernel:
        groups[0]["kernel_launch_count"] = 2
        groups[0]["actual_kernels"].append(dict(groups[0]["actual_kernels"][0]))
    point = dict(sample)
    return {"status": "resolved", "point": point, "sample": sample, "groups": groups}


def _online_record(description, batch, *, candidate_key="source", role="train"):
    measured = _online_description(
        fft_core=description["sample"]["fft_core"], batch=batch,
        log_n=description["sample"]["logN"],
        local_stages=json.loads(description["sample"]["mapping_json"])["stage_partition"][0],
        batch_stride=description["sample"]["batch_stride"],
    )
    groups = [dict(group, role=role) for group in measured["groups"]]
    return {
        "candidate_key": candidate_key, "status": "measured", "correct": True,
        "measurement_exclusive_gpu": True, "batch": batch,
        "sample": measured["sample"], "groups": groups,
    }


def test_online_service_signature_proves_cufftdx_ceiling_grid():
    description = _online_description(batch=3)
    assert all(calibration._shared_service_signature(description, group)
               for group in description["groups"])
    # B * 2^(logN-stage) is 48 and U is 32, so the real launch is ceil(48/32)=2.
    assert description["groups"][0]["work_blocks"] == 2
    bad = _online_description(batch=3)
    bad["groups"][0]["work_blocks"] = bad["groups"][0]["grid_ctas"] = 1
    bad["groups"][0]["actual_kernels"][0]["grid_ctas"] = 1
    assert calibration._shared_service_signature(bad, bad["groups"][0]) is None


def test_register_tile_service_signature_requires_exact_divisible_grid():
    valid = _online_description(fft_core="register-tile", batch=8)
    assert all(calibration._shared_service_signature(valid, group)
               for group in valid["groups"])
    invalid = _online_description(fft_core="register-tile", batch=7)
    assert all(calibration._shared_service_signature(invalid, group) is None
               for group in invalid["groups"])


@pytest.mark.parametrize("lanes,expected_threads,expected_ept", [
    (2, 512, 32), (4, 1024, 16),
])
def test_register_tile_service_signature_accepts_cooperative_prefix_columns(
        lanes, expected_threads, expected_ept):
    description = _online_description(fft_core="register-tile", batch=4, log_n=18,
                                      local_stages=12, prefix_codelet_lanes=lanes)
    prefix = description["groups"][0]
    assert prefix["threads"] == expected_threads
    assert prefix["elements_per_thread"] == expected_ept
    assert prefix["units_per_cta"] == 4
    assert calibration._shared_service_signature(description, prefix)

    wrong_grid = json.loads(json.dumps(description))
    wrong_grid["groups"][0]["work_blocks"] += 1
    wrong_grid["groups"][0]["grid_ctas"] += 1
    wrong_grid["groups"][0]["actual_kernels"][0]["grid_ctas"] += 1
    assert calibration._shared_service_signature(wrong_grid, wrong_grid["groups"][0]) is None

    wrong_metadata = json.loads(json.dumps(description))
    mapping = json.loads(wrong_metadata["sample"]["mapping_json"])
    mapping["prefix_ept"] += 1
    wrong_metadata["sample"]["mapping_json"] = json.dumps(mapping)
    assert calibration._shared_service_signature(
        wrong_metadata, wrong_metadata["groups"][0]) is None


def test_cooperative_prefix_lane_does_not_split_suffix_service_key():
    lane2 = _online_description(fft_core="register-tile", batch=4, log_n=18,
                                local_stages=12, prefix_codelet_lanes=2)
    lane4 = _online_description(fft_core="register-tile", batch=4, log_n=18,
                                local_stages=12, prefix_codelet_lanes=4)
    assert calibration._shared_service_signature(lane2, lane2["groups"][0]) != \
        calibration._shared_service_signature(lane4, lane4["groups"][0])
    assert calibration._shared_service_signature(lane2, lane2["groups"][1]) == \
        calibration._shared_service_signature(lane4, lane4["groups"][1])


@pytest.mark.parametrize("kwargs", [
    {"stage_overlap": True}, {"factor_overlap": True},
    {"factor_slices": 2}, {"batch_tile_count": 2},
    {"missing": ["compiler_local_bytes_per_thread"]},
    {"multi_kernel": True},
])
def test_online_service_signature_rejects_non_bulk_or_unproven_groups(kwargs):
    description = _online_description(**kwargs)
    assert all(calibration._shared_service_signature(description, group) is None
               for group in description["groups"])


def test_online_service_reuse_requires_validated_source_and_matching_geometry():
    source_description = _online_description(fft_core="cufftdx-block", batch=4)
    target_description = _online_description(fft_core="cufftdx-block", batch=4)
    source = {
        "key": "source", "description": source_description,
        "minimum_batch": 1, "requested_batch": 4, "intervals": {"1:4": "passed"},
    }
    target = {
        "key": "target", "description": target_description,
        "minimum_batch": 1, "requested_batch": 4,
    }
    records = [_online_record(source_description, batch) for batch in (1, 4)]
    records.append(_online_record(source_description, 2, role="validation"))
    assert calibration._reusable_shared_services(
        target, [source, target], records, max_points=33)

    unvalidated = dict(source, intervals={"1:4": "unvalidated"})
    assert not calibration._reusable_shared_services(
        target, [unvalidated, target], records, max_points=33)

    mismatch = _online_description(fft_core="cufftdx-block", batch=4, batch_stride=512)
    mismatched_target = dict(target, description=mismatch)
    assert not calibration._reusable_shared_services(
        mismatched_target, [source, mismatched_target], records, max_points=33)

    invalid_records = list(records)
    invalid = _online_record(source_description, 3, role="validation")
    invalid["groups"][0]["actual_kernels"].append(
        dict(invalid["groups"][0]["actual_kernels"][0]))
    invalid["groups"][0]["kernel_launch_count"] = 2
    invalid_records.append(invalid)
    assert not calibration._reusable_shared_services(
        target, [source, target], invalid_records, max_points=33)
