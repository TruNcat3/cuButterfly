import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_fft_grouped_supplement as fft
import run_ntt_comparison_supplement as ntt
from semantic_campaign_manifest import manifest_hash


@pytest.fixture(params=[ntt, fft], ids=['ntt-five', 'fft-grouped'])
def binding(request, tmp_path):
    module = request.param
    library = tmp_path / 'current' / 'libcuntt.so'
    library.parent.mkdir()
    library.write_bytes(b'current-build')
    baseline = tmp_path / 'external.so'
    baseline.write_bytes(b'external')
    gpu_uuid = 'GPU-00000000-0000-0000-0000-000000000011'
    cases, rows, indices = [], [], []
    checkpoint = tmp_path / 'measurement' / 'results.json'
    (checkpoint.parent / 'rows').mkdir(parents=True)
    for index, case_id in enumerate(module.EXPECTED_CASE_IDS):
        semantics = dict(operator=1 if module is ntt else 0, word_bits=64 if module is ntt else None,
                         shape=[4096] if module is ntt else ([8] * 4 if index == 0 else [2] * 8), batch=4)
        descriptor = dict(semantics=semantics)
        cases.append(dict(id=case_id, descriptor=descriptor))
        row = dict(id=case_id, descriptor=descriptor, status='pass', mapping={'backend': 'current'},
                   timing={'median_ms': 1.0}, baselines=[{'name': 'gpu-ntt', 'status': 'missing-baseline'}])
        rows.append(row)
        checksum = module.semantic.digest(row)
        (checkpoint.parent / 'rows' / (case_id + '.json')).write_text(
            json.dumps(dict(row=row, sha256=checksum)))
        indices.append(dict(id=case_id, status=row['status'], sha256=checksum, path='rows/' + case_id + '.json'))
    report = dict(schema='cubutterfly-semantic-results-v1', rows=indices,
                  rows_sha256=module.semantic.digest(indices),
                  identity=dict(library_sha256=module.semantic.sha256_file(library), gpu_uuid=gpu_uuid))
    checkpoint.write_text(json.dumps(report))
    manifest = dict(schema='cubutterfly-semantic-campaign-v1', cases=cases)
    manifest['manifest_sha256'] = manifest_hash(manifest)
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    config = dict(schema=module.preparation.SCHEMA, manifest=manifest,
                  manifest_sha256=module.semantic.digest(manifest), source_sha256=module.preparation._sources())
    config_sha = module.semantic.digest(config)
    (prepared / 'config.json').write_text(json.dumps(dict(value=config, sha256=config_sha)))
    (prepared / 'records').mkdir()
    for case in cases:
        record = dict(case_id=case['id'], case_sha256=module.semantic.digest(case), config_sha256=config_sha,
                      status='ready', artifact={'path': 'artifacts/' + case['id'] + '.pickle'})
        (prepared / 'records' / (case['id'] + '.json')).write_text(
            json.dumps(dict(value=record, sha256=module.semantic.digest(record))))
    argv = ['--checkpoint', str(checkpoint), '--manifest', str(manifest_path),
            '--library', str(library), '--preparation', str(prepared),
            '--output', str(tmp_path / 'new-output'), '--gpu-uuid', gpu_uuid,
            '--gpuntt-library' if module is ntt else '--vkfft-library', str(baseline)]
    return module, argv, report, rows, manifest


def test_dry_run_binds_current_paths_and_all_points_without_torch(binding, monkeypatch, capsys):
    module, argv, _, _, _ = binding
    import builtins
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('CPU validation imported Torch')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_torch)
    assert module.main(argv + ['--dry-run']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['case_ids'] == list(module.EXPECTED_CASE_IDS)
    assert output['library'] == str(module.parse_args(argv).library)
    assert 'build-contract-' not in output['library']
    assert output['checkpoint'] == str(module.parse_args(argv).checkpoint)
    assert not Path(output['output']).exists()


@pytest.mark.parametrize('change', ['library', 'gpu', 'missing-case', 'semantics', 'manifest', 'pending-preparation'])
def test_rejects_incompatible_current_binding(binding, monkeypatch, change):
    module, argv, report, rows, manifest = binding
    args = module.parse_args(argv)
    if change == 'library':
        report['identity']['library_sha256'] = 'old-build'
    elif change == 'gpu':
        report['identity']['gpu_uuid'] = 'GPU-other'
    elif change == 'missing-case':
        rows.pop()
    elif change == 'semantics':
        rows[0]['descriptor']['semantics']['batch'] = 16
    elif change == 'manifest':
        altered = copy.deepcopy(manifest)
        altered['cases'].pop()
        altered['manifest_sha256'] = manifest_hash(altered)
        args.manifest.write_text(json.dumps(altered))
    else:
        (args.preparation / 'records' / (rows[0]['id'] + '.json')).unlink()
    monkeypatch.setattr(module, 'load_checkpoint', lambda path: (report, rows))
    with pytest.raises(ValueError):
        module.bind_inputs(args)


def test_ntt_filters_are_explicit_and_cannot_claim_full_population(binding):
    module, argv, _, _, _ = binding
    if module is not ntt:
        pytest.skip('NTT-only explicit historical filters')
    args = module.parse_args(argv + ['--case-id', ntt.EXPECTED_CASE_IDS[0]])
    assert len(ntt.bind_inputs(args)[1]) == 1
    args.case_id = 'unknown'
    with pytest.raises(ValueError, match='five fixed requests'):
        ntt.bind_inputs(args)


def test_public_comparison_entrypoints_keep_compatible_contract():
    # Historical results wrappers are omitted from the public source package.
    for module in (ntt, fft):
        assert callable(module.main)
        assert callable(module.parse_args)
        assert module.EXPECTED_CASE_IDS
        assert all(isinstance(case_id, str) for case_id in module.EXPECTED_CASE_IDS)


@pytest.mark.parametrize('missing_round', [False, True])
def test_ntt_completion_requires_primary_selection_and_every_round(missing_round):
    result = dict(selection_policy='public-measure', selected_case_ids=list(ntt.EXPECTED_CASE_IDS),
                  requested_trials=3, rows=[dict(case_id=case_id, status='pass', candidate='public-measure',
                                                trials=[{}, {}, {}]) for case_id in ntt.EXPECTED_CASE_IDS])
    if missing_round:
        result['rows'][-1]['trials'].pop()
        result['rows'].append(dict(case_id=ntt.EXPECTED_CASE_IDS[-1], status='pass', candidate='diagnostic-core', trials=[{}]))
    ntt.finish_report(result)
    assert result['status'] == ('complete-with-gaps' if missing_round else 'complete')
    assert result['remaining_case_ids'] == ([ntt.EXPECTED_CASE_IDS[-1]] if missing_round else [])
