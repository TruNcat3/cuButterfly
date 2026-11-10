import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import recover_crypto_cyclic_followup as recovery


def fixture(tmp_path):
    library = tmp_path / 'library.so'
    library.write_bytes(b'unchanged')
    manifest = dict(schema=recovery.manifests.MANIFEST_SCHEMA, cases=[],
        fingerprints=dict(library=dict(sha256=recovery.semantic.sha256_file(library)), sources=[]))
    directory = tmp_path / 'measurement'
    directory.mkdir()
    previous = dict(binary=dict(path='/bridge.so', sha256='old', bytes=1),
        sources=[dict(path='/semantic_gpuntt_baseline.py', sha256='old', bytes=1),
                 dict(path='/semantic_bridge.cu', sha256='old', bytes=1)],
        vendor_revision='fixed', root_helper_library=dict(sha256='fixed'))
    current = copy.deepcopy(previous)
    current['available'] = True
    current['binary']['sha256'] = 'new'
    current['sources'][0]['sha256'] = 'new'
    rows = []
    for number, modulus in enumerate((1009, 2017, 2**61 + 1, 4001)):
        semantics = dict(word_bits=64, modulus=modulus, batch=number + 1)
        case = dict(id=recovery.semantic.digest(semantics), descriptor=dict(semantics=semantics))
        manifest['cases'].append(case)
        if number == 3:
            continue
        artifact = directory / ('error-' + str(number))
        artifact.write_bytes(b'validated artifact')
        row = dict(case, status='pass' if number < 2 else 'correctness-failure',
            correctness=dict(exact=True, mismatches=0),
            mapping_replay_correctness=dict(exact=True, mismatches=0),
            error_artifact=dict(path=str(artifact), sha256=recovery.semantic.sha256_file(artifact)),
            baselines=[dict(name='gpu-ntt', status='pass' if number < 2 else 'correctness-failure',
                support_metadata=dict(backend_identity=previous))])
        rows.append(recovery.semantic.write_case_checkpoint(directory, row))
    manifest['manifest_sha256'] = recovery.manifests.manifest_hash(manifest)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    limits = tmp_path / 'limits.json'
    limits.write_text(json.dumps(dict(reference_max_elements=100, ntt_max_operations=200)))
    identity = dict(manifest_sha256=recovery.semantic.digest(manifest),
        library_sha256=recovery.semantic.sha256_file(library), gpu_uuid='gpu', compile_mode='research',
        timing=True, runner_revision=recovery.asynchronous.REVISION,
        protocol=dict(recovery.semantic.PROTOCOL, float_verdict='error-report',
            reference_max_elements=100, ntt_max_operations=200),
        sources={recovery.BASELINE: 'old', 'runner.py': 'fixed'}, gpuntt_sha256=None)
    report = dict(identity=identity, status='stopped-investigation-required', rows=rows,
        remaining_case_ids=[manifest['cases'][3]['id']], invalid_request_count=0)
    report_path = directory / 'results.json'
    recovery.semantic.write_report(report_path, report)
    return (report_path, manifest_path, limits, library, 'gpu'), dict(
        sources={recovery.BASELINE: 'new', 'runner.py': 'fixed'}, inventory=current,
        support_check=lambda s: dict(supported=s['modulus'] < 2**61, status='unsupported-semantic'))


def test_dry_audit_and_apply_preserve_rows_and_archive(tmp_path):
    args, kwargs = fixture(tmp_path)
    before = args[0].read_bytes()
    recovered, receipt = recovery.plan(*args, **kwargs)
    assert args[0].read_bytes() == before
    assert len(recovered['rows']) == 2 and len(recovered['remaining_case_ids']) == 2
    assert recovered['identity']['gpuntt_sha256'] is None
    assert receipt['current_backend_inventory']['binary']['sha256'] == 'new'
    original = recovery.read(args[0])
    assert recovered['rows'] == original['rows'][:2]
    assert receipt['retained_rows'][0]['original_measurement_identity'] == original['identity']
    recovery.apply(args[0], recovered, receipt, tmp_path / 'archive',
        sources=kwargs['sources'], inventory=kwargs['inventory'])
    assert (tmp_path / 'archive/0-results.json').read_bytes() == before
    assert recovery.read(args[0])['status'] == 'running'
    recovery.semantic.validate_resume(recovery.read(args[0]), recovered['identity'], args[0].parent)


@pytest.mark.parametrize('change,match', [
    ('library', 'fingerprint'), ('protocol', 'protocol'), ('source', 'allowlist'),
    ('baseline', 'exclusively'), ('own-correctness', 'exclusively'),
    ('artifact', 'artifact'), ('index', 'checksum'), ('vendor', 'identity changed'),
    ('retained', 'outside corrected'), ('backend-source', 'unapproved')])
def test_reject_unrelated_or_corrupt_evidence(tmp_path, change, match):
    args, kwargs = fixture(tmp_path)
    report = recovery.read(args[0])
    if change == 'library':
        args[3].write_bytes(b'wrong')
    elif change == 'protocol':
        report['identity']['protocol']['replays'] = 8
    elif change == 'source':
        kwargs['sources']['runner.py'] = 'changed'
    elif change == 'vendor':
        kwargs['inventory']['vendor_revision'] = 'changed'
    elif change == 'backend-source':
        kwargs['inventory']['sources'][1]['path'] = '/other.cu'
        # A common source path can change only when explicitly allowed.
        row_path = args[0].parent / report['rows'][-1]['path']
        row = recovery.read(row_path)['row']
        row['baselines'][0]['support_metadata']['backend_identity']['sources'][1]['path'] = '/other.cu'
        report['rows'][-1] = recovery.semantic.write_case_checkpoint(args[0].parent, row)
        kwargs['inventory']['sources'][1]['sha256'] = 'changed'
    elif change == 'artifact':
        (args[0].parent / 'error-0').write_bytes(b'corrupted')
    elif change == 'index':
        report['rows'][0]['sha256'] = 'wrong'
    else:
        index = 0 if change == 'retained' else -1
        row = recovery.read(args[0].parent / report['rows'][index]['path'])['row']
        if change == 'baseline':
            row['baselines'].append(dict(name='other', status='correctness-failure'))
        elif change == 'own-correctness':
            row['correctness']['exact'] = False
        else:
            row['status'] = 'correctness-failure'
        report['rows'][index] = recovery.semantic.write_case_checkpoint(args[0].parent, row)
    recovery.semantic.write_report(args[0], report)
    before = args[0].read_bytes()
    with pytest.raises(ValueError, match=match):
        recovery.plan(*args, **kwargs)
    assert args[0].read_bytes() == before
    assert not (tmp_path / 'archive').exists()


def test_apply_rejects_evidence_changed_since_audit(tmp_path):
    args, kwargs = fixture(tmp_path)
    recovered, receipt = recovery.plan(*args, **kwargs)
    (args[0].parent / 'error-0').write_bytes(b'changed')
    with pytest.raises(ValueError, match='after audit'):
        recovery.apply(args[0], recovered, receipt, tmp_path / 'archive',
            sources=kwargs['sources'], inventory=kwargs['inventory'])
    assert not (tmp_path / 'archive').exists()


def test_apply_rejects_stale_backend_and_guard_rejects_retained_row(tmp_path):
    args, kwargs = fixture(tmp_path)
    recovered, receipt = recovery.plan(*args, **kwargs)
    changed = copy.deepcopy(kwargs['inventory'])
    changed['binary']['sha256'] = 'changed-after-audit'
    with pytest.raises(ValueError, match='identity changed after audit'):
        recovery.apply(args[0], recovered, receipt, tmp_path / 'archive',
            sources=kwargs['sources'], inventory=changed)
    kwargs['support_check'] = lambda s: dict(supported=False, status='unsupported-semantic')
    with pytest.raises(ValueError, match='retained baseline'):
        recovery.plan(*args, **kwargs)
    assert not (tmp_path / 'archive').exists()


def test_explicit_corrected_bridge_identity_is_logged(tmp_path):
    args, kwargs = fixture(tmp_path)
    bridge = tmp_path / 'corrected.so'
    bridge.write_bytes(b'corrected guard')
    kwargs['inventory']['binary'] = dict(path=str(bridge.resolve()),
        sha256=recovery.semantic.sha256_file(bridge), bytes=bridge.stat().st_size)
    recovered, receipt = recovery.plan(*args, **kwargs, gpuntt_library=bridge)
    assert recovered['identity']['gpuntt_sha256'] == recovery.semantic.sha256_file(bridge)
    assert receipt['accepted_identity_updates']['gpuntt_sha256']['before'] is None
    assert set(receipt['accepted_identity_updates']) == {'sources', 'gpuntt_sha256'}
    assert receipt['original_backend_inventory']['binary']['path'] == '/bridge.so'


def test_cyclic_cli_forwards_explicit_bridge_on_resume(tmp_path, monkeypatch):
    import crypto_semantic_mapping_refresh as cyclic
    monkeypatch.setattr(cyclic, 'require_ready', lambda output: ({}, dict(
        reference_max_elements=100, ntt_max_operations=200)))
    monkeypatch.setattr(recovery.asynchronous.old, 'PROTOCOL', copy.deepcopy(recovery.semantic.PROTOCOL))
    monkeypatch.setenv('CUBUTTERFLY_SEARCH_BUDGET', '512')
    captured = []
    monkeypatch.setattr(recovery.asynchronous, 'main', lambda argv: captured.extend(argv) or 2)
    report = tmp_path / 'cyclic_mapping/measurement/results.json'
    report.parent.mkdir(parents=True)
    report.write_text('{}')
    bridge = tmp_path / 'corrected.so'
    assert cyclic.main(['--output-dir', str(tmp_path), '--run', '--library', 'same.so',
        '--gpu-uuid', 'gpu', '--compile-broker', 'broker', '--calibration-manifest', 'calibration',
        '--gpuntt-library', str(bridge)]) == 2
    assert captured[captured.index('--gpuntt-library') + 1] == str(bridge)
    assert '--resume' in captured


def test_apply_rejects_changed_admission_limits(tmp_path):
    args, kwargs = fixture(tmp_path)
    recovered, receipt = recovery.plan(*args, **kwargs)
    args[2].write_text('{}')
    with pytest.raises(ValueError, match='evidence changed after audit'):
        recovery.apply(args[0], recovered, receipt, tmp_path / 'archive',
            sources=kwargs['sources'], inventory=kwargs['inventory'])
    assert not (tmp_path / 'archive').exists()
