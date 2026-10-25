"""CPU-only cross-build recovery selection and evidence-boundary tests."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_semantic_recovery as prepare

campaign = prepare.campaign
runner = campaign.runner


@pytest.fixture
def inputs(tmp_path):
    semantics = [
        dict(operator=2, storage=2, inverse=0, normalize_inverse=False),
        dict(operator=0, storage=4, inverse=0, normalize_inverse=False),
        dict(operator=2, storage=2, inverse=0, normalize_inverse=False),
        dict(operator=1, storage=8, inverse=0, normalize_inverse=False),
        dict(operator=1, storage=9, inverse=1, normalize_inverse=False),
        dict(operator=2, storage=0, inverse=0, normalize_inverse=False),
        dict(operator=2, storage=0, inverse=1, normalize_inverse=True),
        dict(operator=0, storage=5, inverse=1, normalize_inverse=True),
        dict(operator=0, storage=6, inverse=1, normalize_inverse=True),
    ]
    cases = []
    for index, semantic in enumerate(semantics):
        semantic.update(shape=[2, 4, index + 1], batch=1)
        cases.append(dict(id=runner.digest(semantic), descriptor=dict(semantics=semantic, selection={}),
                          populations=['population'], requested_variants=[{'population': 'population'}],
                          capabilities={'status': 'missing-lowering' if index == 8 else 'eligible'}))
    old = dict(schema='cubutterfly-semantic-campaign-v1', cases=cases, invalid_requests=[],
               metrics=campaign.subset_metrics(cases),
               fingerprints={'library': {'path': '/old/library.so', 'sha256': 'old'}, 'sources': []})
    def save_manifest(path, manifest):
        manifest['manifest_sha256'] = campaign.manifest_hash(manifest)
        campaign.save(path, manifest)
    old_path, new_path = tmp_path / 'old.json', tmp_path / 'new.json'
    save_manifest(old_path, old)
    new = copy.deepcopy(old)
    new['fingerprints']['library'] = {'path': '/new/library.so', 'sha256': 'new'}
    save_manifest(new_path, new)
    shards = []
    for number, (subset, statuses) in enumerate(((cases, ['pass', 'correctness-failure']),
                                                (cases[2:], ['pass', 'pass', 'runtime-error']))):
        directory = tmp_path / ('shard-%d' % number)
        manifest_path = directory / 'manifest.json'
        manifest = campaign.submanifest(old, old_path, subset, 'conformance', set(), [])
        campaign.save(manifest_path, manifest)
        output = directory / 'measurement'
        indices = []
        for case, status in zip(subset, statuses):
            row = {'id': case['id'], 'descriptor': copy.deepcopy(case['descriptor']), 'status': status,
                   'timing': {'median_ms': 1},
                   'error_artifact': {'path': '/must-not-read/huge.npz', 'sha256': 'unread'}}
            indices.append(runner.write_case_checkpoint(output, row))
        report = dict(schema='cubutterfly-semantic-results-v1', rows=indices,
                      rows_sha256=runner.digest(indices),
                      identity={'manifest_sha256': runner.digest(manifest), 'timing': False,
                                'library_sha256': 'old', 'population': None},
                      status='stopped-' + statuses[-1],
                      remaining_case_ids=[case['id'] for case in subset[len(indices):]])
        campaign.save(output / 'results.json', report)
        shards.append({'manifest': campaign.evidence(manifest_path), 'output': str(output)})
    state = dict(identity={'phase': 'conformance', 'inputs': [campaign.evidence(old_path)],
                           'prior_evidence': []}, shards=shards,
                 processed_case_ids=[cases[0]['id']], quarantine=[{'id': cases[1]['id']}])
    controller_path = tmp_path / 'controller.json'
    campaign.save_state(controller_path, state)
    return dict(old=old_path, new=new_path, controller=controller_path, cases=cases,
                output=tmp_path / 'prepared', save_manifest=save_manifest)


def run(inputs):
    return prepare.prepare(inputs['old'], inputs['controller'], inputs['new'], inputs['output'])


def test_selects_union_without_misclassifying_last_shard_completed_rows(inputs):
    result = run(inputs)
    ids = [case['id'] for case in inputs['cases']]
    assert set(result['recovery_case_ids']) == {ids[1], ids[3], ids[4], *ids[5:]}
    assert result['selection_reasons']['latest-unvisited']['case_ids'] == ids[5:]
    assert result['selection_reasons']['historical-correctness-failure']['case_ids'] == [ids[1]]
    assert result['selection_reasons']['historical-runtime-failure']['case_ids'] == [ids[4]]
    assert result['selection_reasons']['ntt-required-recheck']['case_ids'] == ids[3:5]
    assert result['selection_reasons']['normalized-low-precision-inverse-required-recheck']['case_ids'] == ids[6:8]
    assert result['historically_completed_required_recheck_case_ids'] == [ids[3]]
    assert result['historical_status_by_case_id'][ids[2]] == 'pass'
    assert ids[2] not in result['recovery_case_ids']


def test_outputs_only_new_cases_and_full_population_including_boundary(inputs):
    original = {path: path.read_bytes() for path in inputs['old'].parent.rglob('*') if path.is_file()}
    result = run(inputs)
    assert not result['old_measurements_imported']
    assert not result['old_timings_imported']
    for name in ('recovery', 'full-timing'):
        path = inputs['output'] / name / 'manifest.json'
        manifest = campaign.read(path)
        assert campaign.validate_manifest_hash(manifest)
        assert manifest['fingerprints']['library']['sha256'] == 'new'
        assert manifest['continuation_provenance']['parent'] == campaign.evidence(inputs['new'])
        assert manifest['continuation_provenance']['omitted_processed_case_ids'] == []
        assert manifest['continuation_provenance']['quarantine'] == []
        assert all('status' not in case and 'timing' not in case for case in manifest['cases'])
        assert not (path.parent / 'results.json').exists()
    full = campaign.read(inputs['output'] / 'full-timing/manifest.json')
    assert full['cases'] == inputs['cases']
    assert full['metrics']['unique_cases'] == len(inputs['cases'])
    assert full['metrics']['unique_gap_cases'] == 1
    assert full['continuation_provenance']['phase'] == 'validated-timing'
    for path, content in original.items():
        assert path.read_bytes() == content
    with pytest.raises(ValueError, match='fresh output'):
        run(inputs)


@pytest.mark.parametrize('operator,storage,inverse,normalize,expected', [
    (1, 8, 0, False, True), (1, 9, 1, False, True),
    *((operator, storage, 1, True, True) for operator in (0, 2) for storage in (0, 1, 4, 5)),
    (0, 6, 1, True, False), (2, 0, 0, True, False), (0, 4, 1, False, False),
    (5, 0, 1, True, False),
])
def test_affected_semantics(operator, storage, inverse, normalize, expected):
    case = {'descriptor': {'semantics': dict(operator=operator, storage=storage, inverse=inverse,
                                            normalize_inverse=normalize, shape=[2, 4, 8])}}
    assert bool(prepare.affected(case)) is expected


@pytest.mark.parametrize('target', ['old', 'new', 'controller', 'index', 'row'])
def test_rejects_corrupt_checksums_before_writing(inputs, target):
    state = campaign.load_state(inputs['controller'])
    report_path = Path(state['shards'][-1]['output']) / 'results.json'
    if target in ('old', 'new'):
        path = inputs[target]
        document = campaign.read(path)
        document['scope'] = 'changed'
    elif target == 'controller':
        path = inputs['controller']
        document = campaign.read(path)
        document['state']['status'] = 'changed'
    elif target == 'index':
        path = report_path
        document = campaign.read(path)
        document['rows_sha256'] = 'wrong'
    else:
        report = campaign.read(report_path)
        path = report_path.parent / report['rows'][0]['path']
        document = campaign.read(path)
        document['row']['status'] = 'changed'
    campaign.save(path, document)
    with pytest.raises(ValueError, match='checksum'):
        run(inputs)
    assert not inputs['output'].exists()


@pytest.mark.parametrize('target', ['manifest', 'report'])
def test_rejects_duplicate_ids_with_recomputed_checksums(inputs, target):
    if target == 'manifest':
        manifest = campaign.read(inputs['new'])
        manifest['cases'].append(copy.deepcopy(manifest['cases'][0]))
        inputs['save_manifest'](inputs['new'], manifest)
    else:
        state = campaign.load_state(inputs['controller'])
        path = Path(state['shards'][-1]['output']) / 'results.json'
        report = campaign.read(path)
        report['rows'].append(copy.deepcopy(report['rows'][0]))
        report['rows_sha256'] = runner.digest(report['rows'])
        campaign.save(path, report)
    with pytest.raises(ValueError, match='duplicate'):
        run(inputs)
    assert not inputs['output'].exists()


@pytest.mark.parametrize('change', ['semantic-id', 'semantic-set', 'descriptor'])
def test_rejects_changed_semantics_or_descriptor(inputs, change):
    manifest = campaign.read(inputs['new'])
    case = manifest['cases'][0]
    if change == 'semantic-id':
        case['descriptor']['semantics']['shape'] = [8]
    elif change == 'semantic-set':
        manifest['cases'].pop()
    else:
        case['descriptor']['selection']['new'] = True
    inputs['save_manifest'](inputs['new'], manifest)
    with pytest.raises(ValueError, match='semantic ID|descriptor differs'):
        run(inputs)
    assert not inputs['output'].exists()


def test_cli_prepares_without_loading_any_library_or_cuda(inputs):
    assert prepare.main(['--old-manifest', str(inputs['old']),
                         '--old-controller', str(inputs['controller']),
                         '--new-manifest', str(inputs['new']),
                         '--output-dir', str(inputs['output'])]) == 0
    result = json.loads((inputs['output'] / 'recovery_selection.json').read_text())
    assert result['status'] == 'prepared-not-executed'


def test_rejects_changed_metadata_bound_by_controller(inputs):
    state = campaign.load_state(inputs['controller'])
    path = Path(state['shards'][-1]['output']) / 'results.json'
    state['shards'][-1]['evidence'] = [campaign.evidence(path),
        {'path': '/must-not-read/huge.npz', 'sha256': 'unread'}]
    campaign.save_state(inputs['controller'], state)
    report = campaign.read(path)
    report['status'] = 'changed'
    campaign.save(path, report)
    with pytest.raises(ValueError, match='metadata evidence checksum'):
        run(inputs)
    assert not inputs['output'].exists()


def test_rejects_changed_historical_descriptor_with_valid_row_hash(inputs):
    state = campaign.load_state(inputs['controller'])
    path = Path(state['shards'][-1]['output']) / 'results.json'
    report = campaign.read(path)
    row_path = path.parent / report['rows'][0]['path']
    row = campaign.read(row_path)['row']
    row['descriptor']['semantics']['shape'] = [4096]
    report['rows'][0] = runner.write_case_checkpoint(path.parent, row)
    report['rows_sha256'] = runner.digest(report['rows'])
    campaign.save(path, report)
    with pytest.raises(ValueError, match='descriptor semantic mismatch'):
        run(inputs)
    assert not inputs['output'].exists()


def test_rejects_remaining_index_that_repeats_a_completed_row(inputs):
    state = campaign.load_state(inputs['controller'])
    path = Path(state['shards'][-1]['output']) / 'results.json'
    report = campaign.read(path)
    report['remaining_case_ids'].insert(0, report['rows'][0]['id'])
    campaign.save(path, report)
    with pytest.raises(ValueError, match='remaining-case index'):
        run(inputs)
    assert not inputs['output'].exists()


@pytest.mark.parametrize('count', [1, 2, 3])
def test_timing_shards_round_robin_partition_full_population_including_boundary(inputs, count):
    result = prepare.prepare(inputs['old'], inputs['controller'], inputs['new'], inputs['output'],
                             timing_shards=count)
    all_ids = set()
    for number, assignment in enumerate(result['timing_assignment'], start=1):
        path = inputs['output'] / assignment['manifest']
        assert assignment['manifest'] == 'full-timing-shard-%d/manifest.json' % number
        manifest = campaign.read(path)
        assert campaign.validate_manifest_hash(manifest)
        assert manifest['cases'] == inputs['cases'][number - 1::count]
        assert assignment['case_ids'] == [case['id'] for case in manifest['cases']]
        assert assignment['count'] == len(manifest['cases'])
        assert not all_ids.intersection(assignment['case_ids'])
        all_ids.update(assignment['case_ids'])
    assert all_ids == {case['id'] for case in inputs['cases']}
    assert inputs['cases'][-1]['id'] in all_ids
    assert campaign.read(inputs['output'] / 'full-timing/manifest.json')['cases'] == inputs['cases']


@pytest.mark.parametrize('count', [0, -1, 1.5, True])
def test_timing_shards_must_be_positive_integer(inputs, count):
    with pytest.raises(ValueError, match='positive integer'):
        prepare.prepare(inputs['old'], inputs['controller'], inputs['new'], inputs['output'],
                        timing_shards=count)
    assert not inputs['output'].exists()


def test_cli_accepts_timing_shards(inputs):
    assert prepare.main(['--old-manifest', str(inputs['old']),
                         '--old-controller', str(inputs['controller']),
                         '--new-manifest', str(inputs['new']),
                         '--output-dir', str(inputs['output']), '--timing-shards', '2']) == 0
    result = campaign.read(inputs['output'] / 'recovery_selection.json')
    assert result['timing_shards'] == 2
