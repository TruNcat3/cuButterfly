import json
from pathlib import Path

import pytest

from scripts.validate_benchmark_scope import validate


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "benchmark/scopes/a100_80gb_research_v1.json"


def test_active_scope_is_self_contained():
    result = validate(SCOPE, ROOT)
    assert result["scope_id"] == "a100-80gb-research-0.9.3.11"
    assert result["track_count"] == 7
    assert result["checked_population_refs"] == 7
    value = json.loads(SCOPE.read_text())
    revision = next(
        track for track in value["tracks"]
        if track["track_id"] == "end_to_end_fft_poisson_adaptive_revision"
    )
    assert revision["revision_of"] == "end_to_end_fft_poisson_current"
    assert revision["library_build"] == "build-contract-20261017"


def test_scope_rejects_missing_semantic_identity(tmpdir):
    value = json.loads(SCOPE.read_text())
    value["common_contract"]["case_identity"] = ["operator"]
    path = Path(str(tmpdir)) / "scope.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="case_identity"):
        validate(path, ROOT)


def test_scope_rejects_duplicate_track(tmpdir):
    value = json.loads(SCOPE.read_text())
    value["tracks"].append(dict(value["tracks"][0]))
    path = Path(str(tmpdir)) / "scope.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicated"):
        validate(path, ROOT)
