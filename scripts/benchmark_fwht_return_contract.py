#!/usr/bin/env python3
"""Matched main66 FWHT interfaces: CPU prepare, isolated GPU run, CPU summarize."""
import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
import run_semantic_campaign as semantic
import semantic_dao_baseline as dao
from run_quarot_fused_followup import selection_environment

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/fwht_return_contract_20261107.json'
DEFAULT_ROOT = ROOT / 'results/fwht_return_contract_20261107'
LIBRARY = ROOT / 'results/a100_fp16_acceptance_20261105/build/libcuntt.so'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return semantic.sha256_file(path)


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def load_config(path=CONFIG):
    data = json.loads(Path(path).read_text())
    require(data['schema'] == 'fwht-return-contract-v1' and len(data['cases']) == 10, 'Expected ten frozen main66 requests')
    require(len({case['id'] for case in data['cases']}) == 10, 'Duplicate request')
    for case in data['cases']:
        validate_semantics(case['descriptor']['semantics'])
    return data


def validate_semantics(s):
    require(s['operator'] == 2 and s['storage'] == 2 and s['compute'] == 1, 'Only unchanged FP32 FWHT is supported')
    require(s['placement'] == 0 and s['length_mode'] == 0 and s['fft_output_scale'] == 1, 'Unsupported placement/embedding/scale')
    require(all(order == 0 for order in s['input_orders'] + s['output_orders']), 'Natural-order tensors required')
    points = math.prod(s['shape'])
    for side in ('input', 'output'):
        strides = [math.prod(s['shape'][i+1:]) for i in range(len(s['shape']))]
        require(s[side + '_strides'] == strides and s[side + '_batch_stride'] == points, 'Packed layout required; no silent adapter omission')
    require(points <= 32768 and s['batch'] * points <= 1048576, 'Outside frozen CPU/backend support bounds')
    return points


def prepare(root=DEFAULT_ROOT, library=LIBRARY):
    root, library = Path(root).resolve(), Path(library).resolve()
    config = load_config()
    require(sha(library) == config['library_sha256'], 'Expected current775a library')
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'prepared.json'
    require(not target.exists(), 'Prepared cohort already exists; use explicit fresh root')
    cases = []
    for case in config['cases']:
        path = ROOT / case['historical_source']
        require(sha(path) == case['historical_source_sha256'], 'Historical main66 row changed')
        envelope = json.loads(path.read_text())
        require(envelope['sha256'] == case['historical_row_sha256'] == semantic.digest(envelope['row']), 'Historical payload hash mismatch')
        require(envelope['row']['descriptor'] == case['descriptor'], 'Historical descriptor changed')
        s = case['descriptor']['semantics']
        support = dao.support(s)
        require(support['supported'], 'Dao exact support unavailable: ' + support['reason'])
        source = semantic.make_source(s)
        expected = semantic.reference(s, source)
        path = root / (case['id'] + '.npz')
        np.savez(path, source=source, reference=expected)
        cases.append(dict(case, prepared_file=str(path), prepared_sha256=sha(path), dao_contract=support))
    identity = dict(library=str(library), library_sha256=sha(library), gpu_uuid=config['gpu_uuid'],
        config_sha256=sha(CONFIG), runner_sha256=sha(__file__),
        semantic_sha256=sha(ROOT / 'scripts/run_semantic_campaign.py'),
        dao_adapter_sha256=sha(ROOT / 'scripts/semantic_dao_baseline.py'),
        frontend_sha256=sha(ROOT / 'python/cubutterfly/__init__.py'),
        selection_environment=selection_environment())
    prepared = dict(schema=config['schema'], identity=identity, protocol=config['protocol'], cases=cases)
    write(target, prepared)
    deadline = (datetime.now().astimezone() + timedelta(minutes=90)).isoformat()
    commands = []
    for trial in range(1, 4):
        commands.append([sys.executable, str(ROOT / 'scripts/run_isolated_gpu.py'), '--gpu-uuid', config['gpu_uuid'],
            '--until', deadline, '--report-dir', str(root / ('process' + str(trial))), '--',
            sys.executable, str(Path(__file__).resolve()), '--mode', 'run', '--prepared', str(target), '--trial', str(trial)])
    write(root / 'commands.json', dict(deadline=deadline, commands=commands,
        environment=dict(CUDA_VISIBLE_DEVICES=config['gpu_uuid']), scope='ten requests x two interfaces x two backends x three independent processes'))
    return prepared


def returned_call(empty, execute, source):
    """Allocate an independent owned result, then write directly into it."""
    output = empty()
    execute(source, output)
    return output


def backend_order(case_index, trial):
    return ['cubutterfly', 'dao'] if (case_index + trial) % 2 else ['dao', 'cubutterfly']


def run(prepared_path, trial):
    # CUDA imports and initialization occur only in this explicitly guarded mode.
    require(os.environ.get('XD_REPORT_DIR'), 'GPU mode requires run_isolated_gpu guard')
    prepared_path = Path(prepared_path)
    data = json.loads(prepared_path.read_text())
    identity = data['identity']
    require(identity['selection_environment'] == selection_environment(), 'Prepared selection environment changed')
    for path, key in [(identity['library'], 'library_sha256'), (__file__, 'runner_sha256'),
                      (ROOT / 'scripts/run_semantic_campaign.py', 'semantic_sha256'),
                      (ROOT / 'scripts/semantic_dao_baseline.py', 'dao_adapter_sha256'),
                      (ROOT / 'python/cubutterfly/__init__.py', 'frontend_sha256')]:
        require(sha(path) == identity[key], 'Prepared build/source changed: ' + key)
    import torch
    sys.path.insert(0, str(ROOT / 'python'))
    import cubutterfly as cb
    from fast_hadamard_transform import hadamard_transform
    require(torch.cuda.device_count() == 1 and os.environ.get('CUDA_VISIBLE_DEVICES') == identity['gpu_uuid'], 'Guard must expose only designated GPU1 UUID')
    props = torch.cuda.get_device_properties(0)
    require(props.name == 'NVIDIA A100 80GB PCIe' and props.total_memory == 84974239744, 'Unexpected GPU hardware')
    output_dir = Path(os.environ['XD_REPORT_DIR'])
    library = cb.Library(identity['library'])
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream):
        for index, case in enumerate(data['cases']):
            s = case['descriptor']['semantics']; points = validate_semantics(s)
            require(sha(case['prepared_file']) == case['prepared_sha256'], 'Prepared arrays changed')
            arrays = np.load(case['prepared_file']); source_cpu, expected = arrays['source'], arrays['reference']
            source = torch.from_numpy(np.asarray(source_cpu, dtype=np.float32)).to('cuda').reshape(s['batch'], *s['shape'])
            saved = source.clone()
            start = __import__('time').perf_counter()
            with cb.Descriptor.from_json(library, semantic.selection_descriptor(case['descriptor'])) as desc:
                plan = cb.Plan(library, desc, stream=stream.cuda_stream)
            semantic.assert_semantics(case['descriptor'], json.loads(plan.descriptor_json()))
            require(plan.input_size() == source.numel()*4 and plan.output_size() == source.numel()*4, 'Unexpected padded plan storage')
            workspace = torch.empty(max(1, plan.workspace_size()), device='cuda', dtype=torch.uint8)
            if plan.workspace_size(): plan.set_workspace(workspace, plan.workspace_size())
            setup_seconds = __import__('time').perf_counter() - start
            mapping = json.loads(plan.mapping_json())
            scale = 1.0 / points if s['inverse'] and s['normalize_inverse'] else 1.0
            order = backend_order(index, trial)
            contracts = ['supplied-output', 'native-return'] if (index + trial) % 2 else ['native-return', 'supplied-output']
            for contract in contracts:
                for backend in order:
                    supplied = torch.empty_like(source)
                    core = dao.build_core(torch, s, source, supplied) if backend == 'dao' and contract == 'supplied-output' else None
                    if backend == 'cubutterfly':
                        def execute():
                            if contract == 'native-return':
                                return returned_call(lambda: torch.empty_like(source), plan.execute, source)
                            plan.execute(source, supplied); return supplied
                    elif contract == 'native-return':
                        def execute(): return hadamard_transform(source.reshape(s['batch'], points), scale=scale).view_as(source)
                    else:
                        def execute(): core(); return supplied
                    direct = execute(); stream.synchronize()
                    actual = direct.detach().cpu().numpy().copy()
                    numeric = semantic.check_full(actual, expected, 2, 'error-report', True)
                    artifact = semantic.retain_error_artifact(output_dir / 'errors', case['id'], backend + '-' + contract, actual, expected)
                    artifact['path'] = str(Path('errors') / Path(artifact['path']).name)
                    lifetime = dict(contract=contract)
                    if contract == 'native-return':
                        before = direct.clone()
                        source.mul_(0.5)
                        later = execute(); stream.synchronize()
                        require(direct.data_ptr() != later.data_ptr() and torch.equal(direct, before), 'Returned output overwritten or aliased by subsequent call')
                        semantic.check_full(later.cpu().numpy(), expected * 0.5, 2, 'error-report', False)
                        source.copy_(saved)
                        restored = execute(); stream.synchronize()
                        require(torch.equal(direct, before) and torch.equal(restored, direct), 'Return lifecycle or restoration failed')
                        lifetime.update(independent_return_addresses=True, previous_output_preserved=True, changed_input_checked=True, restored_output_byte_equal=True)
                    for _ in range(10): execute()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = [execute() for _ in range(20)]
                    if contract == 'native-return':
                        require(len({value.data_ptr() for value in outputs}) == 20, 'Captured independent calls share output storage')
                    samples = []
                    for _ in range(7):
                        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        begin.record(stream); graph.replay(); end.record(stream); end.synchronize()
                        samples.append(begin.elapsed_time(end) / 20)
                        require(all(torch.equal(value, direct) for value in outputs), 'Graph outputs differ from direct call')
                        require(torch.equal(source, saved), 'Input modified')
                    results.append(dict(case_id=case['id'], semantics=s, interface=contract, backend=backend,
                        backend_order=order, contract_order=contracts, timing=dict(median_ms=statistics.median(samples), samples_ms=samples,
                        protocol=data['protocol']), correctness=numeric, error_artifact=artifact, lifecycle=lifetime,
                        mapping=mapping if backend == 'cubutterfly' else None, dao_contract=case['dao_contract'] if backend == 'dao' else None,
                        setup_seconds=setup_seconds if backend == 'cubutterfly' else None,
                        included_operations=(['allocate_independent_output', 'public_descriptor_execute'] if contract == 'native-return' else ['public_execute_into_supplied_output']) if backend == 'cubutterfly' else
                            (['backend_output_allocation','native_fp32_fwht_and_fused_output_scale'] if contract == 'native-return' else case['historical_adapter_operations'])))
                    if core is not None: core.close()
                    del graph, outputs
            plan.close()
    report = dict(schema='fwht-return-contract-results-v1', trial=trial, status='complete-error-report-only',
        prepared_sha256=sha(prepared_path), identity=identity, gpu=dict(uuid=identity['gpu_uuid'], name=props.name, memory_bytes=props.total_memory),
        records=results, note='Current775a follow-up; never pooled into historical main66. Native-return owns distinct outputs within capture; graph replay reuses captured addresses. Host allocation/setup/capture excluded from event timing.')
    require(len(results) == 40, 'Incomplete process')
    write(output_dir / 'report.json', report)
    return report


def summarize(root=DEFAULT_ROOT):
    root = Path(root).resolve()
    records, sources = [], {}
    prepared_path = root / 'prepared.json'
    prepared = json.loads(prepared_path.read_text())
    for trial in range(1, 4):
        directory = root / ('process' + str(trial))
        status = json.loads((directory / 'status.json').read_text())
        require(status['state'] == 'complete' and status['gpu_uuid'] == prepared['identity']['gpu_uuid'], 'No clean process release')
        path = directory / 'release/report.json'; report = json.loads(path.read_text())
        require(report['status'] == 'complete-error-report-only' and len(report['records']) == 40, 'Incomplete process records')
        require(report['identity'] == prepared['identity'] and report['prepared_sha256'] == sha(prepared_path), 'Process identity or preparation mismatch')
        sources[str(path)] = sha(path)
        signatures = {(row['case_id'], row['interface'], row['backend']) for row in report['records']}
        require(len(signatures) == 40, 'Duplicate process measurements')
        for row in report['records']:
            artifact = directory / 'release' / row['error_artifact']['path']
            require(sha(artifact) == row['error_artifact']['sha256'], 'Error artifact hash mismatch')
            row['error_artifact']['path'] = str(artifact)
            require(len(row['timing']['samples_ms']) == 7 and all(math.isfinite(v) and v > 0 for v in row['timing']['samples_ms']), 'Missing or invalid event samples')
            require(statistics.median(row['timing']['samples_ms']) == row['timing']['median_ms'], 'Stale median')
            records.append(dict(row, trial=trial, report_source=str(path), report_sha256=sources[str(path)]))
    pairs = []
    config = load_config()
    for trial in range(1, 4):
        for case in config['cases']:
            for interface in config['contracts']:
                rows = {r['backend']: r for r in records if r['trial']==trial and r['case_id']==case['id'] and r['interface']==interface}
                require(set(rows) == {'dao','cubutterfly'}, 'Missing matched backend')
                own, baseline = rows['cubutterfly']['timing']['median_ms'], rows['dao']['timing']['median_ms']
                pairs.append(dict(case_id=case['id'], trial=trial, interface=interface, own_ms=own, baseline_ms=baseline, speedup=baseline/own))
    groups = []
    for interface in config['contracts']:
        trials = []
        for trial in range(1,4):
            values = [p['speedup'] for p in pairs if p['interface']==interface and p['trial']==trial]
            trials.append(dict(trial=trial, pairs=len(values), gm=math.exp(statistics.mean(map(math.log,values))),wins=sum(v>1 for v in values)))
        groups.append(dict(interface=interface, trials=trials, gm_process_range=[min(t['gm'] for t in trials),max(t['gm'] for t in trials)]))
    result = dict(schema='fwht-return-contract-summary-v1', identity=json.loads((root/'prepared.json').read_text())['identity'],
        config=config, records=records, pairs=pairs, groups=groups, source_hashes=sources,
        aggregation='Each of three processes retains ten matched request ratios per interface. GM is across requests within a process; process range is not a confidence interval. No mixing with historical main66.')
    write(root / 'summary.json', result)
    lines=['Interface | Process | Pairs | GM | Faster','--- | ---: | ---: | ---: | ---:']
    for group in groups:
        for trial in group['trials']:lines.append(f"{group['interface']} | {trial['trial']} | {trial['pairs']} | {trial['gm']:.6f} | {trial['wins']}")
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['prepare','run','summarize'],default='prepare')
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--library',type=Path,default=LIBRARY)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--trial',type=int,choices=[1,2,3])
    args=parser.parse_args(argv)
    if args.mode=='prepare':prepare(args.root,args.library)
    elif args.mode=='run':
        require(args.prepared is not None and args.trial is not None,'run requires prepared identity and trial');run(args.prepared,args.trial)
    else:summarize(args.root)


if __name__=='__main__':main()
