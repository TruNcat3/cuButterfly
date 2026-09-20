"""CPU-only contracts for the bounded large FFT extension queue."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "large_fft_prepare", ROOT / "paper/tools/prepare_large_fft_extension.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_prepare_generates_12_new_cells_and_six_traceable_exclusions(tmp_path):
    output = tmp_path / "large_fft"
    queue = prepare.prepare(
        ROOT / "config/large_fft_extension.json",
        ROOT / "paper/experiments/targets/a100-80gb-gpu1.json",
        output,
        root=ROOT,
    )

    assert queue["status"] == "prepared-not-executed"
    assert queue["scope"]["rank1_full_domain_cells"] == 18
    assert queue["scope"]["rank1_excluded_historical_cells"] == 6
    assert queue["scope"]["rank1_new_gpu1_cells"] == 12
    assert queue["scope"]["rank2_supplement_cases"] == 2
    assert queue["protocol"]["search_budget"] == 12
    assert queue["protocol"]["seed_budget"] == 4
    assert queue["protocol"]["screen_budget_total"] == 16
    assert queue["protocol"]["screen_budget_total"] == (
        queue["protocol"]["search_budget"] + queue["protocol"]["seed_budget"])
    assert queue["protocol"]["warmup"] == 1000
    workloads = json.loads((output / "workloads.json").read_text())
    assert len(workloads["workloads"]) == 12
    assert len({(w["precision"], w["logN"], w["batch"]) for w in workloads["workloads"]}) == 12
    exclusions = json.loads((output / "excluded_old_points.json").read_text())["points"]
    assert len(exclusions) == 6
    assert all(item["excluded_from_new_gpu1_workloads"] for item in exclusions)
    assert all(item["source"]["sha256"] for item in exclusions)
    assert all(item["reuse_for_new_mean"] is False for item in exclusions)
    excluded_keys = {(x["precision"], x["logN"], x["batch"]) for x in exclusions}
    assert not excluded_keys & {(w["precision"], w["logN"], w["batch"]) for w in workloads["workloads"]}
    stage_points = json.loads((output / "stage-gpu1/stage_source_points.json").read_text())
    assert len(stage_points["points"]) == 6
    assert {(p["precision"], p["logN"], p["batch"]) for p in stage_points["points"]} == {
        (precision, log_n, 16) for precision in ("fp32", "fp64") for log_n in (22, 23, 24)
    }
    assert all(point["source_role"] == "large-fft-stage-training" for point in stage_points["points"])
    assert all(point["source_compile_status"] in {"compiled", "pending"}
               for point in stage_points["points"])
    stage_profile = json.loads((output / "stage-gpu1/stage_profile.json").read_text())
    assert stage_profile["status"] == "prepared-target-identity-only"
    assert stage_profile["provenance"]["timings_imported"] is False
    assert stage_profile["provenance"]["thermal_state_imported"] is False


def test_prepare_supports_separate_40gb_gpu0_cohort(tmp_path):
    output = tmp_path / "large_fft40"
    queue = prepare.prepare(
        ROOT / "config/large_fft_extension_40gb.json",
        ROOT / "paper/experiments/targets/a100-40gb-gpu0.json",
        output,
        root=ROOT,
    )

    assert queue["target"]["id"] == "a100-40gb-gpu0"
    assert queue["scope"]["rank1_new_gpu0_cells"] == 12
    assert queue["scope"]["gpu0_priority"] is True
    assert queue["scope"]["gpu1_mirror_required"] is False
    assert [command["id"] for command in queue["commands"]] == [
        "precompile-compile-cpu", "stage-calibration-gpu0", "acceptance-gpu0",
        *[f"rank2-{shape}x{shape}-fp32-b1--{variant}"
          for shape in (2048, 4096)
          for variant in ("direct-direct-t256", "direct-block-t128",
                          "block-direct-t128", "block-block-t256")]]
    stage_points = json.loads((output / "stage-gpu0/stage_source_points.json").read_text())
    assert len(stage_points["points"]) == 6
    stage_profile = json.loads((output / "stage-gpu0/stage_profile.json").read_text())
    assert stage_profile["gpu_uuid"] == "GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e"
    assert stage_profile["global_memory_bytes"] == 42285268992
    acceptance_text = " ".join(queue["commands"][2]["argv"])
    assert "stage-gpu0/stage_calibration.json" in acceptance_text
    assert "calibration-gpu0/pipeline_schedule_calibration.json" in acceptance_text
    assert "calibration-gpu0/schedule_calibration.json" in acceptance_text


def test_limited_manifest_projects_existing_families_without_live_probe(tmp_path, monkeypatch):
    output = tmp_path / "large_fft"
    queue = prepare.prepare(
        ROOT / "config/large_fft_extension.json",
        ROOT / "paper/experiments/targets/a100-80gb-gpu1.json",
        output,
        root=ROOT,
    )
    manifest = json.loads((output / "precompile/modules.json").read_text())
    assert manifest["structural_only"] is True
    assert manifest["runtime_feasibility_verified"] is False
    assert 0 < manifest["module_count"] <= 24
    assert set(manifest["family_counts"]) == {
        "factor-streamed-cufftdx", "online-register-tile", "shared-iterative"
    }
    assert all(m["id"] == prepare.hashlib.sha256(prepare.canonical(m["request"]).encode()).hexdigest()
               for m in manifest["modules"])
    assert all(not command["executed"] for command in queue["commands"])
    assert queue["gpu_probe_performed"] is False
    assert all(command["env"].get("CUDA_VISIBLE_DEVICES") in ("", prepare.json.loads(
        (ROOT / "paper/experiments/targets/a100-80gb-gpu1.json").read_text())["gpu_uuid"])
               for command in queue["commands"])
    assert [command["id"] for command in queue["commands"]] == [
        "precompile-compile-cpu", "stage-calibration-gpu1", "acceptance-gpu1",
        *[f"rank2-{shape}x{shape}-fp32-b1--{variant}"
          for shape in (2048, 4096)
          for variant in ("direct-direct-t256", "direct-block-t128",
                          "block-direct-t128", "block-block-t256")]]
    stage_command = queue["commands"][1]
    assert "--import-checkpoint" not in stage_command["argv"]
    assert "stage_source_points.json" in " ".join(stage_command["argv"])
    assert stage_command["env"]["CUBUTTERFLY_JIT_CACHE"] == str(
        (output / "precompile-cache").resolve())
    acceptance_command = queue["commands"][2]
    acceptance_text = " ".join(acceptance_command["argv"])
    assert "stage-gpu1/stage_calibration.json" in acceptance_text
    assert "results/staged_migration_20260917/a100-80gb-gpu1/stage_calibration.json" not in acceptance_text
    assert "results/staged_migration_20260917/a100-80gb-gpu1/calibration_device.json" not in acceptance_text
    assert "calibration-gpu1/pipeline_schedule_calibration.json" in acceptance_text
    assert "calibration-gpu1/schedule_calibration.json" in acceptance_text


def test_rank2_is_separate_and_uses_plan_bench_compare_cufft(tmp_path):
    output = tmp_path / "large_fft"
    prepare.prepare(
        ROOT / "config/large_fft_extension.json",
        ROOT / "paper/experiments/targets/a100-80gb-gpu1.json",
        output,
        root=ROOT,
    )
    audit = json.loads((output / "rank2_audit.json").read_text())
    assert audit["public_api"]["rank2"].startswith("supported")
    assert audit["public_api"]["rank3"].startswith("unsupported")
    assert audit["adapter"]["acceptance_adapter"] is False
    assert len(audit["cases"]) == 8
    assert audit["search"] == {
        "schema": "finite-explicit-composition-v1",
        "shape_count": 2,
        "variant_count": 4,
        "run_count": 8,
        "exhaustive_measure": False,
        "claim": "finite four-point explicit composition screen; not exhaustive measure",
    }
    shapes = {tuple(case["shape"]) for case in audit["cases"]}
    assert shapes == {(2048, 2048), (4096, 4096)}
    for case in audit["cases"]:
        argv = case["command"]["argv"]
        assert "cubutterfly_plan_bench" in " ".join(argv)
        assert "--compare-cufft" in argv
        assert "--verify" in argv
        assert "--trials" in argv
        assert "--algorithm" in argv
        assert "--policy" not in argv
        composition = json.loads(argv[argv.index("--algorithm") + 1])
        assert composition["kind"] == "composition"
        assert len(composition["axes"]) == 2
        assert {axis["fft_core"] for axis in composition["axes"]} <= {
            "cufftdx-direct", "cufftdx-block"
        }
        assert all(axis["tile_threads"] in {128, 256} for axis in composition["axes"])
        assert argv[argv.index("--") + 1].endswith("/cubutterfly_plan_bench")
        assert argv[argv.index("--") + 1] != str(ROOT / "paper/tools/run_with_gpu_telemetry.py")
        assert case["correctness"]["required"] is True
        assert argv[argv.index("--trials") + 1] == "3"


def test_rank2_verify_contract_is_present_in_standalone_source():
    source = (ROOT / "apps/cubutterfly_plan_bench.cpp").read_text()
    assert 'arg == "--verify"' in source
    assert 'arg == "--trials"' in source
    assert "cufftExecC2C" in source and "cufftExecZ2Z" in source
    assert "read cuButterfly output for verification" in source
    assert "mix64" in source and "unit_from_hash" in source
    assert ",verified_batches=" in source
    assert ",relative_l2_error=" in source
    assert ",max_absolute_error=" in source
    assert "tolerance = 1.0e-10" in source
    assert "tolerance = 2.0e-4" in source
    assert "!std::isfinite(actual_real)" in source
    assert "full-output verification failed" in source


def test_prepare_rejects_changed_target_or_existing_plan(tmp_path):
    output = tmp_path / "large_fft"
    target = json.loads((ROOT / "paper/experiments/targets/a100-80gb-gpu1.json").read_text())
    target["id"] = "a100-40gb-gpu0"
    target_path = tmp_path / "target.json"
    target_path.write_text(json.dumps(target))
    with pytest.raises(ValueError, match="target/config mismatch|scoped"):
        prepare.prepare(ROOT / "config/large_fft_extension.json", target_path, output, root=ROOT)

    prepare.prepare(
        ROOT / "config/large_fft_extension.json",
        ROOT / "paper/experiments/targets/a100-80gb-gpu1.json",
        output,
        root=ROOT,
    )
    changed = json.loads((ROOT / "config/large_fft_extension.json").read_text())
    changed["protocol"]["repeat"] = 99
    changed_path = tmp_path / "changed-config.json"
    changed_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="inputs changed"):
        prepare.prepare(changed_path, ROOT / "paper/experiments/targets/a100-80gb-gpu1.json", output, root=ROOT)
