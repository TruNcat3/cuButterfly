import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_recalibrated_campaign as preparation
import reviewed_comparison as reviewed


def test_historical_cufft_adapters_stay_external_references():
    original = json.loads(preparation.HISTORICAL_CONFIG.read_text())
    prepared = {row["source_case_id"]: row for row in preparation._historical_suite_cases()}
    baseline_ids = []
    for row in original["cases"]:
        if (row["runner"] == "butterfly" and
                preparation._suite_arg(row["args"], "--backend", "") == "cufft"):
            baseline_ids.append(row["id"])
            current = prepared[row["id"]]
            assert current["implementation"] == "cuFFT"
            assert current["reference"] == row.get("reference", False)
            assert current["mapping_policy"] == "external-reference"
            assert "--auto-select" not in current["args"]
            assert current["args"] == row["args"]
    assert len(baseline_ids) >= 3  # FP64, normalized inverse and strided FFT.


def test_preparation_cannot_overwrite_frozen_experiment(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("preserved")
    with pytest.raises(ValueError, match="overwrite"):
        preparation.prepare(tmp_path)
    assert manifest.read_text() == "preserved"


@pytest.mark.parametrize("action", ["calibrate", "run"])
@pytest.mark.parametrize("resume", [False, True])
def test_cancelled_execution_is_rejected_before_validation(tmp_path, monkeypatch, action, resume):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "cancellation.json").write_text(json.dumps({"reason": "repair semantics first"}))
    monkeypatch.setattr(reviewed, "validate", lambda *_: pytest.fail("must block before validation"))
    monkeypatch.setattr(reviewed.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("must not launch"))
    args = [action, "--manifest", str(manifest)]
    if action == "calibrate":
        args.extend(["--input-profile", str(tmp_path / "input.json"),
                     "--output-profile", str(tmp_path / "output")])
    else:
        args.extend(["--calibration-manifest", str(tmp_path / "calibration.json")])
    if resume:
        args.append("--resume")
    with pytest.raises(reviewed.runner.CampaignError, match="campaign cancelled.*repair semantics first"):
        reviewed.main(args)


def test_cancellation_does_not_disable_cpu_validation(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    (tmp_path / "cancellation.json").write_text("{}")
    document = {"measurement_contract": {"execution_environment": reviewed.runner.execution_environment()}}
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr(reviewed.runner, "load_manifest", lambda *_: document)
    monkeypatch.setattr(reviewed.runner, "validate_plan_inputs", lambda *_: None)
    monkeypatch.setattr(reviewed.runner, "fingerprint_builds", lambda *_: {})
    monkeypatch.setattr(reviewed.runner, "_command_steps", lambda *_: [])
    assert reviewed.validate(manifest) is document
