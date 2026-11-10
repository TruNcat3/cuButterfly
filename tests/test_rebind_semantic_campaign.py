import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import rebind_semantic_campaign as rebind


def fixture(tmp_path, monkeypatch):
    source = dict(schema=rebind.campaign.MANIFEST_SCHEMA, config={'population': 'fixed'},
                  layout_recipe={'input': 'packed'}, metrics={'unique_cases': 1},
                  invalid_requests=[], cases=[dict(id='case', descriptor={'semantics': {'batch': 4}},
                                                   capabilities={'status': 'eligible'})],
                  fingerprints={'library': {'sha256': 'old'}})
    source['manifest_sha256'] = rebind.campaign.manifest_hash(source)
    current = copy.deepcopy(source)
    current['fingerprints']['library']['sha256'] = 'current'
    current['manifest_sha256'] = rebind.campaign.manifest_hash(current)
    paths = [tmp_path / name for name in ('source.json', 'config.json', 'lib.so', 'prepared', 'current.json', 'receipt.json')]
    paths[0].write_text(json.dumps(source))
    paths[1].write_text(json.dumps(source['config']))
    paths[2].write_bytes(b'current')
    prepared = {'source_sha256': {'cpu': 'frozen'}, 'manifest': copy.deepcopy(source)}
    record = {'status': 'ready', 'case_sha256': rebind.campaign.semantic_id(current['cases'][0])}
    monkeypatch.setattr(rebind.campaign, 'load_config', lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(rebind.campaign, 'generate_manifest', lambda *args, **kwargs: current)
    monkeypatch.setattr(rebind.preparation, '_config', lambda *args: (prepared, 'prepared-sha'))
    monkeypatch.setattr(rebind.preparation, '_sources', lambda: {'cpu': 'frozen'})
    monkeypatch.setattr(rebind.preparation, 'record_for', lambda *args: record)
    return paths, source, current, prepared, record


def test_binding_preserves_source_and_is_idempotent(tmp_path, monkeypatch):
    paths, source, current, prepared, record = fixture(tmp_path, monkeypatch)
    before = paths[0].read_bytes()
    receipt = rebind.prepare(*paths)
    assert json.loads(paths[4].read_text()) == current
    assert receipt['source_manifest_sha256'] == rebind.campaign.semantic_id(source)
    assert receipt['current_manifest_sha256'] == rebind.campaign.semantic_id(current)
    assert receipt['source_library']['sha256'] == 'old'
    assert receipt['current_library']['sha256'] == 'current'
    assert receipt['cases'] == 1
    assert rebind.prepare(*paths) == receipt
    assert paths[0].read_bytes() == before


@pytest.mark.parametrize('change', ['checksum', 'config', 'semantics', 'capability',
                                    'prepared-source', 'prepared-manifest', 'record', 'pending'])
def test_binding_rejects_changed_scope_or_preparation_before_writing(tmp_path, monkeypatch, change):
    paths, source, current, prepared, record = fixture(tmp_path, monkeypatch)
    if change == 'checksum':
        source['manifest_sha256'] = 'wrong'
        paths[0].write_text(json.dumps(source))
    elif change == 'config':
        paths[1].write_text('{}')
    elif change == 'semantics':
        current['cases'][0]['descriptor']['semantics']['batch'] = 8
    elif change == 'capability':
        current['cases'][0]['capabilities']['status'] = 'missing-lowering'
    elif change == 'prepared-source':
        prepared['source_sha256'] = {}
    elif change == 'prepared-manifest':
        prepared['manifest'] = current
    elif change == 'record':
        record['case_sha256'] = 'other-case'
    else:
        monkeypatch.setattr(rebind.preparation, 'record_for', lambda *args: None)
    with pytest.raises(ValueError):
        rebind.prepare(*paths)
    assert not paths[4].exists() and not paths[5].exists()


def test_binding_does_not_overwrite_a_different_build(tmp_path, monkeypatch):
    paths, *_ = fixture(tmp_path, monkeypatch)
    paths[4].write_text('{"another": "build"}')
    with pytest.raises(ValueError, match='different inputs'):
        rebind.prepare(*paths)
    assert paths[4].read_text() == '{"another": "build"}'
