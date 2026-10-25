import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_isolated_c2r_build_manifest_and_frozen_relink():
    path = ROOT / "results/paper_targeted_20261012/c2r_copy_elision_build_v6/capture.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "compatible-inputs-captured"
    assert manifest["baseline_relinked"]["sha256"] == manifest["frozen_library"]["sha256"]
    assert manifest["scale_arithmetic_changed"] is False
    assert manifest["compact_workspace_retained"] is True
    assert manifest["gpu_used"] is False


def test_experimental_library_is_distinct_and_single_tu_manifested():
    path = ROOT / "results/paper_targeted_20261012/c2r_copy_elision_build_v6/manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "built-not-gpu-validated"
    assert manifest["candidate_library"]["sha256"] != manifest["frozen_library"]["sha256"]
    assert "xd_real_fft.copy-elision.o" in manifest["candidate_object"]["path"]
    assert manifest["inherited_runtime_fingerprint"] is True
