import copy
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import schedule_cost_model as model


def raw():
    return dict(schema="cubutterfly-schedule-profile-v1", hardware=dict(device="Test GPU", compute_capability="8.0", global_memory_bytes=4096),
        protocol=dict(warmup=10, repeat=100, trials=3, launch_mode="direct"),
        rows=[dict(threads=t, grid_ctas=g, trial_kernel_ms=[.004 + (g-1)*.00001]*3)
              for t in (64, 128) for g in (1, 20, 160, 1280)])


def test_measured_launch_and_dispatch_are_separate_and_scale_with_grid():
    summary = model.summarize(raw())
    estimates = model.costs(dict(scheduling=summary), 128, 2000, 20)
    assert estimates["startup_ms"] == pytest.approx(.004)
    assert estimates["dispatch_ms"] == pytest.approx(.01999)
    assert estimates["source"] == "measured-minimal-CTA"
    assert estimates["thread_shape_matched"] and estimates["grid_extrapolated"]


def test_flat_curve_does_not_pretend_to_identify_cta_dispatch():
    measured = raw()
    for row in measured["rows"]:
        row["trial_kernel_ms"] = [.004]*3
    estimates = model.costs(dict(scheduling=model.summarize(measured)), 64, 100, 20)
    assert estimates["source"] == "measured-startup-with-unidentified-dispatch"
    assert estimates["startup_ms"] == .004


def test_legacy_profile_keeps_explicit_prior_provenance():
    assert model.costs({}, 128, 200, 20)["source"] == "unmeasured-prior"


def test_calibration_reuses_identity_and_rejects_other_memory(tmp_path, monkeypatch):
    binary=tmp_path/'probe'; binary.write_bytes(b'probe-v1')
    target=raw()["hardware"]; calls=[]; checks=[]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    def run(command):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps(raw()))
    path=tmp_path/'fresh-target'/'scheduling.json'
    result=model.calibrate(binary,path,target,run,lambda:checks.append(1))
    assert model.calibrate(binary,path,target,run,lambda:checks.append(1)) == result
    assert len(calls)==1 and len(checks)==2
    with pytest.raises(ValueError,match="global_memory_bytes"):
        model.calibrate(binary,path,dict(target,global_memory_bytes=8192),run,lambda:None)
    assert json.loads(path.read_text())["identity"]["global_memory_bytes"] == 4096


def test_invalid_measurements_are_not_fitted():
    measured=raw(); measured["rows"][0]["trial_kernel_ms"][0]=float('nan')
    with pytest.raises(ValueError,match="finite"):
        model.summarize(measured)
