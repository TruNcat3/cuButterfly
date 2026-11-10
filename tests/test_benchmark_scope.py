import json
from pathlib import Path

import pytest

from scripts.validate_benchmark_scope import validate


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / "benchmark/scopes/a100_80gb_research_v1.json"


def test_active_scope_sources_are_self_contained(tmp_path):
    value = json.loads(SCOPE.read_text())
    # Some historical populations themselves live in unbundled results/.
    # Source-only must reject those rather than silently skipping their hashes.
    absent = [track for track in value['tracks']
              if not (ROOT / track['population_ref'].split('#', 1)[0]).exists()]
    if absent:
        with pytest.raises(ValueError, match='population_ref does not exist'):
            validate(SCOPE, ROOT, source_only=True)
    value['tracks'] = [track for track in value['tracks'] if track not in absent]
    path = tmp_path / 'bundled-populations.json'
    path.write_text(json.dumps(value))
    result = validate(path, ROOT, source_only=True)
    assert result["scope_id"] == "a100-80gb-research-0.9.3.11"
    assert result["track_count"] == 7 - len(absent)
    assert result["checked_population_refs"] == 7 - len(absent)
    value = json.loads(SCOPE.read_text())
    revision = next(
        track for track in value["tracks"]
        if track["track_id"] == "end_to_end_fft_poisson_adaptive_revision"
    )
    assert revision["revision_of"] == "end_to_end_fft_poisson_current"
    assert revision["library_build"] == "build-contract-20261017"


def test_missing_results_require_explicit_source_only(tmp_path):
    value = json.loads(SCOPE.read_text())
    value['tracks'] = [value['tracks'][0]]
    track = value['tracks'][0]
    track.update(status='complete', result_ref='results/nonexistent-test-evidence/report.json')
    path = tmp_path / 'scope.json'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='result_ref does not exist'):
        validate(path, ROOT)
    result = validate(path, ROOT, source_only=True)
    assert result['validation_scope'] == 'source-only'
    assert result['result_evidence_verified'] is False
    assert result['completion_verified'] is False
    assert result['unchecked_result_refs'] == [dict(
        track_id=track['track_id'], result_ref=track['result_ref'])]
    track['population_sha256'] = '0' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='population_sha256 mismatch'):
        validate(path, ROOT, source_only=True)


def test_strict_summary_remains_compatible(tmp_path):
    value = json.loads(SCOPE.read_text())
    value['tracks'] = [value['tracks'][0]]
    value['tracks'][0]['result_ref'] = 'README.md'
    path = tmp_path / 'scope.json'
    path.write_text(json.dumps(value))
    assert set(validate(path, ROOT)) == {
        'scope_id', 'status', 'track_count', 'checked_population_refs'}


def test_scope_rejects_missing_semantic_identity(tmpdir):
    value = json.loads(SCOPE.read_text())
    value["common_contract"]["case_identity"] = ["operator"]
    path = Path(str(tmpdir)) / "scope.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="case_identity"):
        validate(path, ROOT)


def test_scope_rejects_duplicate_track(tmpdir):
    value = json.loads(SCOPE.read_text())
    value["tracks"] = [value["tracks"][0], dict(value["tracks"][0])]
    path = Path(str(tmpdir)) / "scope.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="duplicated"):
        validate(path, ROOT, source_only=True)
