"""Synthetic, isolated loader contracts; these fixtures are never paper data."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'experiments/tools/load_fp16_acceptance.py'
spec = importlib.util.spec_from_file_location('fp16_acceptance_loader', MODULE)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def cohort(tmp_path):
    root = tmp_path
    library, runtime = 'a' * 64, 'b' * 64
    timing = lambda value: dict(protocol=loader.PROTOCOL, samples_ms=[value] * 7, median_ms=value)
    def result(name, folder, policy='default', vendor=False, value=0.002, manifest=None, task=None):
        relative = f'external/{name}/result.json' if vendor else f'{folder}/runs/{name}/{policy}/result.json'
        if folder.startswith('legacy/'):
            relative = f'{folder}/{name}/result.json'
        output = root / Path(relative).parent / 'output.npy'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'isolated fixture artifact; not research output')
        record = dict(id=name, policy=policy, scope='public-selection', threshold=None,
            installed=folder == 'installed_public', selection_source=2 if policy == 'measure' else 0,
            status='complete' if vendor else 'error-report-only', library_sha256=library,
            gpu_uuid=loader.UUID, gpu_name=loader.GPU, runtime_fingerprint=runtime + ':research',
            all_outputs_finite=True, direct_graph_byte_equal=True, input_output_guards_passed=True,
            replay_byte_equal=True, input_and_output_guards_passed=True,
            output_sha256=loader.sha(output), graph_timing=timing(value), actual_mapping={'backend': 'fixture'},
            cpu_error={'relative_l2': 0.01}, frontend=str(root / 'prefix/lib/python3.10/site-packages/cubutterfly/__init__.py'))
        if manifest:
            record['manifest_sha256'] = loader.sha(root / manifest)
        if vendor:
            record.update(mode='measure', design_point=task, vendor_sha256='c' * 64,
                          input_sha256=task['input_sha256'], reference_file=task['reference_file'])
        save(root, relative, record)
        marker = name if vendor else folder + '-' + policy + '-' + name
        if name in {'fp16-inverse', 'bf16-inverse', 'bf16-forward', 'fp16-fp32'}:
            marker = folder + '-' + name
        if folder.startswith('legacy/'):
            marker = folder.replace('/', '-') + '-' + name
        save(root, 'post_completed/' + marker + '.json', dict(result=str(root / relative),
             result_sha256=loader.sha(root / relative), attempt='fixture-parent-attempt'))
    save(root, 'status.json', dict(status='complete', phase='all'))
    installation = dict(status='installed-validated', library_sha256=library, runtime_fingerprint=runtime,
                        profile=str(root / 'prefix/share/cuButterfly/hardware/gpu'))
    save(root, 'installation.json', installation)
    save(root, 'stage/model-receipt.json', {'fixture': True})
    receipt = save(root, 'tuning/receipt.json', {'fixture': True})
    registry = save(root, 'registry.json', {'fixture': True})
    save(root, 'post_completed/tuning.json', dict(status='complete', library_sha256=library,
         receipt_sha256=loader.sha(receipt), registry_sha256=loader.sha(registry)))
    save(root, 'build/identity_receipt.json', dict(library_sha256=library, runtime_fingerprint=runtime))
    save(root, 'stage/calibration_device.json', dict(device=loader.GPU, global_memory_bytes=84974239744, compute_capability='8.0'))
    tasks, entries = [], []
    for n, batch in loader.CASES:
        name = f'n{n}-b{batch}'
        inp = root / 'arrays' / (name + '-input')
        inp.parent.mkdir(exist_ok=True); inp.write_bytes(name.encode())
        ref = inp.with_name(name + '-reference'); ref.write_bytes(b'reference')
        common = dict(n=n, batch=batch, input_file=str(inp), input_sha256=loader.sha(inp),
                      reference_file=str(ref), reference_sha256=loader.sha(ref))
        tasks.append(dict(common, id=name + '-cufft', variant='cufft'))
        entries.append(dict(common, id=name, storage=4, compute=0, inverse=False,
                            normalize_inverse=False, scope='public-selection'))
    for name in ('fp16-inverse', 'bf16-inverse', 'bf16-forward', 'fp16-fp32'):
        entries.append(dict(id=name, scope='legacy-scalar-explicit-smoke'))
    for folder in ('public', 'installed_public'):
        save(root, folder + '/manifest.json', dict(schema='fp16-public-acceptance-v1', installed=folder == 'installed_public',
             expected_runtime_fingerprint=runtime + ':research', library='fixture-library', bindings={'fixture-library': library}, cases=entries))
    save(root, 'external/manifest.json', dict(gpu_uuid=loader.UUID, library_sha256=library, vendor_sha256='c' * 64, tasks=tasks))
    rows = []
    for task in tasks:
        name = task['id'].removesuffix('-cufft')
        result(task['id'], 'external', vendor=True, value=0.003, manifest='external/manifest.json', task=task)
        for folder, policy, value in [('public', 'default', 0.002), ('public', 'measure', 0.001), ('installed_public', 'default', 0.0025)]:
            result(name, folder, policy, value=value, manifest=folder + '/manifest.json')
        rows.append(dict(n=task['n'], batch=task['batch'], cufft_us=3.0,
                         latency_us=dict(default=2.0, measure=1.0, installed=2.5),
                         speedup=dict(default=1.5, measure=3.0, installed=1.2),
                         mappings={k: {'backend': 'fixture'} for k in ('default', 'measure', 'installed')},
                         cufft_cpu_error={'relative_l2': 0.02}))
    for folder in ('public', 'installed_public'):
        for entry in entries[9:]:
            result(entry['id'], folder)
    for placement in ('development', 'installed'):
        for name in ('fft-fp64-n20-b16', 'fwht-fp32-n16-b16', 'fwht-fp32-n15-b32'):
            result(name, 'legacy/' + placement)
    save(root, 'summary.json', dict(status='complete-error-report-only', installation=installation,
         stage_model={'fixture': True}, tuning={'fixture': True}, rows=rows))
    return root


def test_default_is_not_minimum_of_measure_and_installed(cohort):
    snapshot = loader.load_fp16_acceptance(cohort)
    assert len(snapshot['requests']) == len(snapshot['pairs']) == 9
    assert all(pair['own_us'] == 2.0 and pair['speedup'] == 1.5 for pair in snapshot['pairs'])
    assert all(pair['status'] == 'accuracy-review-required' and pair['numeric_policy'] == 'error-report-only'
               for rows in snapshot['evidence'].values() for pair in rows)
    assert len(snapshot['evidence']['measure']) == len(snapshot['evidence']['installed']) == 9
    assert snapshot['summary'][1]['pairs'] == 0 and snapshot['summary'][1]['status'] == 'pending'


@pytest.mark.parametrize('failure', ['missing-summary', 'uncommitted', 'guard', 'protocol', 'input', 'parent'])
def test_incomplete_or_incompatible_evidence_rejected(cohort, failure):
    name = 'n4096-b256'
    relative = f'public/runs/{name}/default/result.json'
    path = cohort / relative
    if failure == 'missing-summary':
        (cohort / 'summary.json').unlink()
    elif failure == 'uncommitted':
        (cohort / f'post_completed/public-default-{name}.json').unlink()
    elif failure == 'parent':
        save(cohort, 'status.json', dict(status='running', phase='public'))
    elif failure == 'input':
        (cohort / 'arrays/n4096-b256-input').write_bytes(b'changed')
    else:
        record = json.loads(path.read_text())
        if failure == 'guard':
            record['input_output_guards_passed'] = False
        else:
            record['graph_timing']['protocol'] = 'one kernel event'
        save(cohort, relative, record)
        marker = f'post_completed/public-default-{name}.json'
        commit = json.loads((cohort / marker).read_text()); commit['result_sha256'] = loader.sha(path)
        save(cohort, marker, commit)
    with pytest.raises(ValueError):
        loader.load_fp16_acceptance(cohort)


def test_current_incomplete_campaign_rejects_without_gpu():
    if (loader.DEFAULT_DIRECTORY / 'summary.json').exists():
        pytest.skip('campaign completed; isolated rejection fixtures cover this contract')
    with pytest.raises(ValueError, match='missing summary.json'):
        loader.load_fp16_acceptance()
