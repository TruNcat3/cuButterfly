import csv
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import initialize_hardware_profile as profile_tools


def test_model_table_contains_both_barrier_models(tmp_path):
    profile = {
        "device": "test-gpu",
        "compute_capability": "9.0",
        "capabilities": {
            "global_feedback_bytes_per_second": 1.0e12,
            "equivalent_butterflies_per_second": 2.0e12,
            "interstage_shared_bytes_per_second": 3.0e12,
            "cta_barriers_per_second": 4.0e9,
        },
    }
    args = type(
        "Args",
        (),
        {"logNs": [8], "spatial_budgets": [32], "word_bytes": [4]},
    )()
    output = tmp_path / "model.csv"
    count = profile_tools.build_model_table(profile, args, output)
    assert count > 0
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {row["barrier_model"] for row in rows} == {"cta", "none"}
    assert all(row["device"] == "test-gpu" for row in rows)
    assert all(float(row["calibrated_body_us"]) > 0 for row in rows)


def test_initializer_persists_exact_identity_for_runtime_ranker(tmp_path, monkeypatch):
    import rank_static_candidates as ranking
    args = SimpleNamespace(output_dir=tmp_path, trials=1, force=False,
        microbench=tmp_path / "cuntt_hardware_microbench", device_bench=None,
        points=1024, blocks=4, threads=32, iterations=1, warmup=1, repeat=2,
        logNs=[8], spatial_budgets=[32], word_bytes=[4])
    row = dict(device="test-gpu", compute_capability="9.0", sm_count="1", pipeline_correct="1")
    row.update({key: "100" for key in profile_tools.METRIC_FIELDS})
    row.update({f"pipeline_us{value}_Gbutterfly_s": "100" for value in profile_tools.PIPELINE_SPACES})
    calls = []

    def identity(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="device,compute_capability,global_memory_bytes\ntest-gpu,9.0,123456789\n")

    monkeypatch.setattr(profile_tools, "parse_args", lambda: args)
    monkeypatch.setattr(profile_tools.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(profile_tools, "find_microbench", lambda _: args.microbench)
    monkeypatch.setattr(profile_tools.subprocess, "run", identity)
    monkeypatch.setattr(profile_tools, "run_trials", lambda *args: [row])
    assert profile_tools.main() == 0
    path = tmp_path / "hardware_profile.json"
    loaded = json.loads(path.read_text())
    assert loaded["global_memory_bytes"] == 123456789
    ranking._identity(loaded, dict(device="test-gpu", compute_capability="9.0",
                                  memory_bytes=123456789), path)
    assert calls == [[str(tmp_path / "cubutterfly_bench"), "--device-identity"]]


@pytest.mark.parametrize("change", [dict(device="other"), dict(compute_capability="8.0"),
    dict(global_memory_bytes=2048), dict(global_memory_bytes=True),
    dict(hardware={"memory_bytes": 2048})])
def test_identity_binding_rejects_conflicting_cached_profile(change):
    profile = dict(device="test-gpu", compute_capability="9.0", **{})
    profile.update(change)
    with pytest.raises(ValueError):
        profile_tools.bind_device_identity(profile, dict(device="test-gpu",
            compute_capability="9.0", global_memory_bytes="1024"))


@pytest.mark.parametrize("capacity", [None, True, 0, -1, 1.5, "1.5", ""])
def test_invalid_identity_capacity_is_not_rounded_or_inferred(capacity):
    profile = dict(device="test-gpu", compute_capability="9.0")
    with pytest.raises(ValueError, match="positive integer"):
        profile_tools.bind_device_identity(profile, dict(profile, global_memory_bytes=capacity))
