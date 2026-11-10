"""Development resume preserves evidence while current-build confirmations stay fresh."""
import csv
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_local_hardware as calibration
from calibration_space import mapping_point, candidate_id


def fixture(tmp_path, monkeypatch):
    for name in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / name).write_bytes(b"old-build")
    workload = dict(operator="fft", precision="fp32", logN=8, batch=4)
    spec = tmp_path / "workloads.json"
    spec.write_text(json.dumps({"schema": "cubutterfly-install-search-v1", "scope": "development resume test",
                                "workloads": [workload]}))
    points = [mapping_point(workload, dict(backend="shared-iterative", fft_core="scalar",
                                          tile_threads=n, stage_partition=[4, 4])) for n in (64, 128)]
    inventory = list(points)
    timed = []

    def run(command, check=True):
        if "--list-processing-units" in command:
            return SimpleNamespace(returncode=0, stdout="precision,logN,threads,ept\n", stderr="")
        if "--list-register-tile-mappings" in command:
            return SimpleNamespace(returncode=0, stdout="logN,local_stages\n", stderr="")
        mapping = json.loads(command[command.index("--mapping-json") + 1])
        timed.append((mapping["tile_threads"], (tmp_path / "cubutterfly_bench").read_bytes()))
        row = dict(workload, **mapping, mapping_json=json.dumps(mapping), correct="1", kernel_ms="0.01",
                   runtime_fingerprint=timed[-1][1].decode(),
                   warmup=command[command.index("--warmup") + 1], repeat=command[command.index("--repeat") + 1])
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        return SimpleNamespace(returncode=0, stdout=output.getvalue(), stderr="")

    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: list(inventory))
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(calibration, "run", run)
    monkeypatch.setattr(calibration, "candidates_for_workload", lambda *args: [])
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=spec, resume_search=False,
                           resume_build_priors=False, seed_budget=0, search_budget=2, search_seconds=0,
                           search_finalists=1, operator_warmup=3, operator_repeat=5, operator_trials=2)
    return args, points, inventory, timed


def test_old_screens_are_priors_new_inventory_is_measured_and_finalists_confirmed(tmp_path, monkeypatch):
    args, points, inventory, timed = fixture(tmp_path, monkeypatch)
    output = tmp_path / "search_coverage.json"
    calibration.run_compiled_search(args, output)
    old_coverage = output.read_bytes()
    old_journal = (tmp_path / "search_measurements.json").read_bytes()
    old_identity = json.loads(old_coverage)["measurement_identity"]
    for name in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / name).write_bytes(b"new-build")
    args.resume_search = True
    with pytest.raises(ValueError, match="different binaries"):
        calibration.run_compiled_search(args, output)
    args.resume_build_priors = True
    args.search_budget = 1
    new = mapping_point(dict(operator="fft", precision="fp32", logN=8, batch=4),
                        dict(backend="shared-iterative", fft_core="scalar", tile_threads=256, stage_partition=[4, 4]))
    inventory.append(new)
    timed.clear()
    rows = calibration.run_compiled_search(args, output)
    assert timed[0] == (256, b"new-build")
    assert len(timed) == 3  # one new screen, two fresh finalist trials
    priors = [row for row in rows if row.get("measurement_role") == "development-build-prior"]
    assert len(priors) == 1
    assert priors[0]["status"] == "screened" and priors[0]["origin_identity"] == old_identity
    assert priors[0]["samples"][0]["runtime_fingerprint"] == "old-build"
    confirmed = [row for row in rows if row["status"] == "measured"]
    assert len(confirmed) == 1
    assert all(sample["runtime_fingerprint"] == "new-build" for sample in confirmed[0]["samples"])
    audit = json.loads(output.read_text())
    assert audit["measurement_identity"] != old_identity
    assert audit["workloads"][0]["development_prior_screened"] == 2
    assert audit["workloads"][0]["screened"] == 3
    archived = {Path(source["path"]).name.split("-", 1)[1]: Path(source["path"]).read_bytes()
                for source in audit["development_build_priors"]["sources"]}
    assert archived["search_coverage.json"] == old_coverage
    assert archived["search_measurements.json"] == old_journal
    timed.clear()
    calibration.run_compiled_search(args, output)
    assert not timed  # same current-build completed checkpoint remains resumable


@pytest.mark.parametrize("change", ["verify_batches", "operator_repeat", "compile_mode"])
def test_cross_build_prior_protocol_changes_still_reject(tmp_path, monkeypatch, change):
    args, _, _, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / "search_coverage.json"
    calibration.run_compiled_search(args, output)
    for name in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / name).write_bytes(b"new-build")
    args.resume_search = args.resume_build_priors = True
    if change == "compile_mode":
        monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "precompiled")
    else:
        setattr(args, change, 7)
    with pytest.raises(ValueError, match="unchanged"):
        calibration.run_compiled_search(args, output)


def test_incumbent_is_remeasured_not_relabelled_after_build_change(tmp_path, monkeypatch):
    args, points, _, timed = fixture(tmp_path, monkeypatch)
    args.search_only = True
    monkeypatch.setattr(calibration, "run_compiled_search", lambda *args: [])
    monkeypatch.setattr(calibration, "incumbent_commands", lambda unused: [
        ("incumbent-test", calibration.command_for(tmp_path / "cubutterfly_bench", points[0]))])
    output = tmp_path / "operator_calibration.json"
    calibration.run_operator_calibration(args, output, False)
    old_rows = output.read_bytes()
    old_identity = (tmp_path / "operator_measurement_identity.json").read_bytes()
    for name in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / name).write_bytes(b"new-build")
    args.resume_search = args.resume_build_priors = True
    timed.clear()
    calibration.run_operator_calibration(args, output, False)
    assert timed == [(64, b"new-build"), (64, b"new-build")]
    rows = json.loads(output.read_text())
    assert all(sample["runtime_fingerprint"] == "new-build" for sample in rows[0]["samples"])
    snapshots = list((tmp_path / "build_priors").glob("*/*"))
    assert any(path.read_bytes() == old_rows for path in snapshots)
    assert any(path.read_bytes() == old_identity for path in snapshots)


def test_development_prior_flag_requires_resume(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["calibrate_local_hardware.py", "--build-dir", "/tmp/build",
                                     "--profile-dir", "/tmp/profile", "--resume-build-priors"])
    with pytest.raises(SystemExit):
        calibration.parse_args()
