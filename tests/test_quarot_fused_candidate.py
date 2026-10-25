import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/tools"))
import quarot_fused_candidate_20261012 as probe


def test_cpu_prepare_preserves_original_batch_payload_and_zero_gate(tmpdir, monkeypatch):
    # pytest 6 on the research Python 3.7 environment exposes tmpdir rather
    # than the newer pathlib-based tmp_path fixture.
    tmp_path = Path(str(tmpdir))
    monkeypatch.setattr(probe, "OUTPUT", tmp_path)
    # Supply a matching historical fixture without weakening the actual
    # preparation guard when the live adapter has moved to a new revision.
    original_read = Path.read_text
    def fixture_read(path, *args, **kwargs):
        value = original_read(path, *args, **kwargs)
        if (path.name == "result.json" and path.parent.name == "release"
                and path.parent.parent.name in ("quarot_b1_p1", "quarot_b4_p1")):
            prior = json.loads(value)
            prior["runner_sha256"] = probe.digest(probe.BASE / "run_quarot_random_end_to_end.py")
            prior["adapter_sha256"] = probe.digest(probe.BASE / "quarot_adapter.py")
            return json.dumps(prior)
        return value
    monkeypatch.setattr(Path, "read_text", fixture_read)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU preparation cannot initialize CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    for batch in (1, 4):
        spec = probe.prepare(batch)
        assert spec["protocol"]["batch_size"] == batch
        assert spec["protocol"]["seed"] == 20261003
        assert spec["protocol"]["prefill"] == 512
        assert spec["protocol"]["decode"] == 32
        assert spec["protocol"]["atol"] == spec["protocol"]["rtol"] == 0
        assert spec["public_search_integrated"] is False
        assert spec["deadline"] == "2026-10-12T18:00:00+08:00"
        assert len(spec["variant_order"]) == 3
        assert "CUBUTTERFLY_E13_ALGORITHM_POLICY=default" in spec["launch_argv"]
        assert probe.checked_spec(batch) == spec


def test_candidate_scale_is_actual_caller_scale_and_fallback_is_visible():
    class Base:
        def __init__(self):
            self._scales = {}
        def _transform_vectors(self, values, extent, full_n):
            return "fallback"
        def plan_statistics(self):
            return dict(plans_created=0)
    seen = []
    def provider(values, scale):
        seen.append(scale)
        return values
    instance = probe.candidate_adapter_class(Base, provider)()
    values = torch.ones((2, 32), dtype=torch.float16)
    assert instance._transform_vectors(values, 32, 1024) is values
    assert seen == [0.03125]
    assert instance._transform_vectors(values.float(), 32, 1024) == "fallback"
    assert instance.plan_statistics()["fallback_calls"] == 1
    assert instance.plan_statistics()["public_search_integrated"] is False


def test_prepare_does_not_overwrite_launched_spec(tmpdir, monkeypatch):
    tmp_path = Path(str(tmpdir))
    monkeypatch.setattr(probe, "OUTPUT", tmp_path)
    (tmp_path / "b1/guard").mkdir(parents=True)
    import pytest
    with pytest.raises(ValueError, match="launched"):
        probe.prepare(1)


def test_prepare_rejects_changed_application_identity(tmpdir, monkeypatch):
    monkeypatch.setattr(probe, "OUTPUT", Path(str(tmpdir)))
    original = probe.digest
    def changed(path):
        if Path(path) == probe.BASE / "run_quarot_random_end_to_end.py":
            return "changed-runner-identity"
        return original(path)
    monkeypatch.setattr(probe, "digest", changed)
    import pytest
    with pytest.raises(ValueError, match="re-review required"):
        probe.prepare(1)
