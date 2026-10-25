import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/tools"))
import prepare_principle_gap_campaign as campaign


def config():
    return json.loads(campaign.CONFIG.read_text())


def accepted_journal(study):
    value = copy.deepcopy(study)
    for case in value["cases"]:
        for variant in case["variants"]:
            variant["preflight"] = dict(status="accepted", resolved_mapping=copy.deepcopy(variant["mapping"]),
                                        runtime_fingerprint="frozen-test-runtime",
                                        sample=dict(verified_batches=case["workload"]["batch"], correct="1"))
            variant["trials"] = [dict(trial=index, performance_accepted=True, kernel_ms=0.01,
                                      resolved_mapping=copy.deepcopy(variant["mapping"]),
                                      sample=dict(runtime_fingerprint="frozen-test-runtime")) for index in range(3)]
    return value


def test_finite_parallel_layout_pairs_and_blocked_tasks():
    value = config()
    study = campaign.layout_study(value)
    assert len(study["cases"]) == 4
    for case in study["cases"]:
        baseline, treatment = [item["mapping"] for item in case["variants"]]
        diff = [key for key in baseline if baseline[key] != treatment[key]]
        assert diff == case["comparisons"][0]["changed_axes"]
        assert baseline["stage_partition"] == [6, 6]
        assert baseline.get("tile_threads", baseline.get("threads_per_block")) == 32
        assert baseline["compute_unit"] == "radix2"
        assert baseline["stage_overlap"] is False
    tasks = {item["id"]: item for item in value["tasks"]}
    assert tasks["reuse-fft-leaf-factorial"]["budget"]["new_gpu_cells"] == 0
    assert tasks["component-disabled-research"]["status"] == "blocked_implementation"
    assert tasks["all-butterfly-core-transfer"]["status"] == "blocked_implementation"


def test_audit_rejects_alias_core_changes_and_incomplete_batches():
    study = campaign.layout_study(config())
    journal = accepted_journal(study)
    assert campaign.audit_journal(journal, study)["status"] == "accepted"
    for mutation in ("alias", "core", "batch", "trials"):
        bad = copy.deepcopy(journal)
        variant = bad["cases"][0]["variants"][1]
        if mutation == "alias":
            variant["preflight"]["resolved_mapping"]["shared_layout"] = "linear"
        elif mutation == "core":
            variant["preflight"]["resolved_mapping"]["group_cores"] = ["different-core"]
        elif mutation == "batch":
            variant["preflight"]["sample"]["verified_batches"] = 0
        else:
            variant["trials"].pop()
        assert campaign.audit_journal(bad, study)["status"] == "rejected"


def test_prepare_never_invokes_process_or_gpu(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation must not invoke subprocess or CUDA")
    monkeypatch.setattr(campaign.controlled.subprocess, "run", forbidden)
    fake_root = tmp_path / "repo"
    value = config()
    target = value["target"]
    library = fake_root / target["library"]
    library.parent.mkdir(parents=True)
    library.write_bytes(b"frozen library")
    target["library_sha256"] = campaign.sha(library)
    binaries = {}
    for name in ("cubutterfly_bench", "cuntt_bench"):
        binary = fake_root / target["build_dir"] / name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(name.encode())
        binaries[name] = campaign.sha(binary)
    hardware = fake_root / "hardware.json"
    hardware.write_text(json.dumps(dict(provenance=dict(cuda_visible_devices=target["gpu_uuid"]))))
    sources = fake_root / "frozen_source"
    for relative in ("src/operator_traits.cuh", "src/ntt_traits.cuh", "src/shared_iterative.cuh"):
        source = sources / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(relative)
    telemetry = fake_root / "paper/tools/run_with_gpu_telemetry.py"
    telemetry.parent.mkdir(parents=True)
    telemetry.write_text("# frozen telemetry fixture\n")
    calibration = fake_root / target["calibration"]
    calibration.parent.mkdir(parents=True)
    calibration.write_text(json.dumps(dict(hardware_rates_source=str(hardware), measurement_identity=dict(
        library_sha256=target["library_sha256"], binary_sha256=binaries,
        jit_runtime=dict(config=dict(root=str(sources), cache="cache", mathdx="mathdx", nvcc="nvcc"),
                         ld_library_path=":cuda")))))
    fixture_config = tmp_path / "fixture_config.json"
    fixture_config.write_text(json.dumps(value))
    monkeypatch.setattr(campaign, "ROOT", fake_root)
    output = tmp_path / "prepared"
    result = campaign.prepare(fixture_config, output)
    assert result["gpu_queried"] is False
    assert result["executable_claim"] is False
    assert result["cells"] == 8
    assert "run_isolated_gpu.py" in result["launch_command"]
    assert result["gpu_uuid"] == config()["target"]["gpu_uuid"]
    assert config()["target"]["protected_gpu"] not in result["launch_command"]
    assert "--guard-child" in result["launch_argv"]
    assert "CUBUTTERFLY_PREPARE_EVENTS" not in result["environment"]
    assert "guard/release/timing.json" in result["audit_command"]
    assert json.loads((output / "prepared.json").read_text())["status"] == "prepared"
    assert len(json.loads((output / "primitive_sources.json").read_text())["sources"]) == 3
    assert campaign.prepare(fixture_config, output) == result


def test_rejects_weak_or_expanded_baseline():
    value = config()
    next(item for item in value["tasks"] if item["id"] == "layout-fixed-core")["stage_partition"] = [1] * 12
    with pytest.raises(ValueError, match="finite parallel contract"):
        campaign.layout_study(value)


def test_guard_attempt_artifacts_never_reuse_parent_paths(tmp_path):
    spec = dict(python="python", environment={"CUDA_VISIBLE_DEVICES": "uuid"},
                study_path=str(tmp_path / "study.json"), build_dir="build", gpu_uuid="uuid")
    attempts = [tmp_path / "guard/attempts/first", tmp_path / "guard/attempts/retry"]
    for attempt in attempts:
        environment, command = campaign.child_commands(spec, attempt)
        assert environment["CUBUTTERFLY_PREPARE_EVENTS"] == str(attempt / "preparation_events.jsonl")
        assert command[command.index("--output-prefix") + 1] == str(attempt / "telemetry")
        assert command[command.index("--output") + 1] == str(attempt / "timing.json")
    assert campaign.child_commands(spec, attempts[0]) != campaign.child_commands(spec, attempts[1])


def test_guard_child_rejects_missing_or_foreign_context(tmp_path):
    with pytest.raises(ValueError, match="requires XD_REPORT_DIR"):
        campaign.guard_context(tmp_path, {})
    with pytest.raises(ValueError, match="outside"):
        campaign.guard_context(tmp_path, {"XD_REPORT_DIR": str(tmp_path / "foreign")})
    with pytest.raises(KeyboardInterrupt, match="isolation guard"):
        campaign.interrupted(None, None)


def test_followup_inventory_cannot_expand_the_frozen_queue():
    path = ROOT / "paper/experiments/principle_gap_campaign_20261012_followup.json"
    followup = json.loads(path.read_text())
    assert followup["extends_sha256"] == campaign.sha(campaign.CONFIG)
    assert len(followup["tasks"]) == 6
    assert all(task["launch_argv"] is None for task in followup["tasks"])
    assert {task["id"] for task in followup["tasks"]} == {
        "e1-36-independent-replays", "matched-external-baseline-gaps", "quarot-existing-batch-queue",
        "ckks-complete-rns-application", "ncu-occupancy-and-l2-hit", "online-selection-and-setup-cost"}
