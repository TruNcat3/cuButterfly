import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibrate_local_hardware as calibration


def test_install_search_matrix_keeps_direction_and_ntt_order_cells():
    document = json.loads((ROOT / "config" / "install_search_workloads.json").read_text())
    workloads = document["workloads"]
    ntt = [row for row in workloads if row.get("operator") == "ntt"]

    assert {row.get("direction", "forward") for row in ntt} == {"forward", "inverse"}
    assert {row.get("output_order", "natural") for row in ntt} == {"natural", "bit-reversed"}
    assert any(row.get("direction") == "inverse" and row.get("output_order") == "bit-reversed"
               for row in ntt)
    assert "not all semantic combinations" in document["scope"]


def test_measurement_identity_hashes_platform_rules_content_at_same_path(tmpdir, monkeypatch):
    build_dir = pathlib.Path(str(tmpdir))
    for name in ("cubutterfly_bench", "cuntt_bench"):
        (build_dir / name).write_bytes((name + "-fake-binary").encode())
    rules = build_dir / "platform.json"
    rules.write_text('{"schema":"cubutterfly-platform-tuning-v1","targets":[]}')
    monkeypatch.setenv("CUBUTTERFLY_PLATFORM_TUNING", str(rules))

    args = SimpleNamespace(build_dir=build_dir)
    first = calibration.measurement_identity(args)
    expected = hashlib.sha256(rules.read_bytes()).hexdigest()
    assert first["platform_tuning_sha256"] == expected
    assert first["binary_sha256"] == {
        name: hashlib.sha256((build_dir / name).read_bytes()).hexdigest()
        for name in ("cubutterfly_bench", "cuntt_bench")
    }

    rules.write_text('{"schema":"cubutterfly-platform-tuning-v1","targets":[{"rules":[]}]}')
    second = calibration.measurement_identity(args)
    assert second["platform_tuning_sha256"] == hashlib.sha256(rules.read_bytes()).hexdigest()
    assert second["platform_tuning_sha256"] != first["platform_tuning_sha256"]
