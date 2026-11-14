#!/usr/bin/env python3
"""Five-point diagnostic explicit replay; never an automatic public comparison."""
import argparse
import copy
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import run_semantic_campaign as campaign

CASES = tuple(f'fwht-fp16-n{n}-b{b}' for n, b in ((256,256),(4096,1),(4096,256),(32768,1),(32768,256)))
METRICS = ','.join(('gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','sm__warps_active.avg.pct_of_peak_sustained_active','launch__registers_per_thread','launch__shared_mem_per_block','smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def validate_case(case):
    if case['id'] not in CASES:
        raise ValueError('case outside five-point diagnostic population')
    s = case['descriptor']['semantics']
    n = s['shape'][0]
    required = dict(operator=2, storage=0, compute=0, inverse=0, placement=0, length_mode=0,
                    fft_output_scale=1.0, input_strides=[1], output_strides=[1],
                    input_orders=[0], output_orders=[0], input_batch_stride=n, output_batch_stride=n)
    if any(s.get(k) != v for k,v in required.items()) or len(s['shape']) != 1 or case['id'] != f"fwht-fp16-n{n}-b{s['batch']}":
        raise ValueError('only natural contiguous out-of-place forward FP16 FWHT supported')


def validate_binding(binding, root):
    case = binding['case']
    validate_case(case)
    envelope = binding['recorded']
    if envelope.get('guards') != {'before': True, 'after': True}:
        raise ValueError('recorded GPU guards must both be true')
    if campaign.digest(envelope['identity']) != envelope['identity_sha256'] or campaign.digest(envelope['row']) != envelope['row_sha256']:
        raise ValueError('recorded identity/row checksum mismatch')
    if envelope['row']['descriptor'] != case['descriptor'] or envelope['row']['id'] != case['id']:
        raise ValueError('recorded case binding mismatch')
    record = binding['preparation_record']
    if record['id'] != case['id'] or record['status'] != 'prepared' or binding['payload'] != 'prepared/'+case['id']+'.npz':
        raise ValueError('prepared case/path binding mismatch')
    if record['descriptor_sha256'] != campaign.digest(case['descriptor']) or record['semantics_sha256'] != campaign.digest(case['descriptor']['semantics']):
        raise ValueError('prepared descriptor binding mismatch')
    if envelope['prepared_sha256'] != record['sha256'] or campaign.sha256_file(root / binding['payload']) != record['sha256']:
        raise ValueError('prepared payload checksum mismatch')
    if envelope['row']['mapping'].get('contract') != case['descriptor']['semantics']:
        raise ValueError('mapping contract mismatch')


def prepare(source, output):
    manifest = read(source/'preparation.json')
    if campaign.digest(manifest['config']) != manifest['config_sha256']:
        raise ValueError('source configuration checksum mismatch')
    cases = {c['id']:c for c in manifest['config']['cases']}
    records = {r['id']:r for r in manifest['cases']}
    commands = []
    for name in CASES:
        case, record = cases[name], records[name]
        validate_case(case)
        envelope = read(source/'rows'/f'{name}.json')
        if envelope['identity']['config_sha256'] != manifest['config_sha256'] or envelope['identity']['preparation_sha256'] != campaign.sha256_file(source/'preparation.json'):
            raise ValueError('recorded preparation identity mismatch')
        payload = 'prepared/'+name+'.npz'
        binding = dict(case=case, preparation_record=record, recorded=envelope, payload=payload)
        validate_binding(binding, source)
        destination = output/payload
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/record['path'], destination)
        write(output/'bindings'/f'{name}.json', binding)
        for backend in ('cubutterfly','hadacore'):
            base = [sys.executable, str(Path(__file__).resolve()), '--run', '--output', str(output.resolve()), '--case', name, '--backend', backend]
            prefix = output.resolve()/'profiler'/name/backend
            prefix.mkdir(parents=True, exist_ok=True)
            commands.append(dict(case=name, backend=backend,
                timing=shlex.join(base+['--mode','timing']),
                ncu=shlex.join(['ncu','--profile-from-start','off','--clock-control','none','--metrics',METRICS,'--export',str(prefix/'ncu')]+base+['--mode','profile','--profile-tag','ncu']),
                nsys=shlex.join(['nsys','profile','--capture-range','cudaProfilerApi','--capture-range-end','stop','--output',str(prefix/'nsys')]+base+['--mode','profile','--profile-tag','nsys'])))
    write(output/'commands.json', dict(scope='diagnostic-explicit-replay; not public automatic comparison', commands=commands))


def replay_mapping(selected, backend, threads=None):
    if threads is not None and (backend != 'cubutterfly' or threads not in (128,256,512,1024)):
        raise ValueError('--threads requires cubutterfly and 128/256/512/1024')
    expected = copy.deepcopy(selected)
    if threads is not None:
        if selected.get('backend') != 'temporal-tile' or 'tile_threads' not in selected:
            raise ValueError('thread diagnostic requires a temporal-tile mapping with tile_threads')
        expected['tile_threads'] = threads
    diff = {key:dict(selected=selected.get(key), expected=expected.get(key))
            for key in set(selected) | set(expected) if selected.get(key) != expected.get(key)}
    return expected, diff


def verify_mapping(selected, expected, actual, threads=None):
    allowed, diff = replay_mapping(selected, 'cubutterfly', threads)
    if expected != allowed:
        raise AssertionError('diagnostic changed mapping fields beyond tile_threads')
    if actual != expected:
        raise AssertionError('explicit replay changed expected mapping')
    return dict(selected=selected, expected=expected, actual=actual, diff=diff,
                threads_override=threads, actual_matches_expected=True)


def profiler_call(torch, name):
    result = getattr(torch.cuda.cudart(), name)()
    if result is not None and int(result) != 0:
        raise RuntimeError(f'{name} failed: {result}')


def target_uuid(identity, requested=None):
    target = identity['gpu_uuid']
    if requested is not None and not campaign.same_gpu_uuid(target, requested):
        raise ValueError('requested GPU UUID must match recorded identity')
    return 'GPU-'+str(target).removeprefix('GPU-')


def after_guard(report, uuid):
    try:
        campaign.exclusive_gpu(uuid)
        report['guards']['after'] = True
    except Exception as exc:
        report['guards']['after'] = False
        report.update(status='interrupted', after_guard_error=str(exc))
        raise


def run(output, name, backend, mode, threads=None, gpu_uuid=None, profile_tag=None):
    binding = read(output/'bindings'/f'{name}.json')
    validate_binding(binding, output)
    selected = binding['recorded']['row']['mapping']
    expected_mapping, _ = replay_mapping(selected, backend, threads)
    identity = binding['recorded']['identity']
    for lib in identity['libraries'].values():
        if campaign.sha256_file(lib['path']) != lib['sha256']:
            raise ValueError('recorded library SHA mismatch: '+lib['path'])
    uuid = target_uuid(identity, gpu_uuid)
    with np.load(output/binding['payload'], allow_pickle=False) as arrays:
        source, expected = arrays['source'].copy(), arrays['expected'].copy()
    report = dict(scope='diagnostic-explicit-replay; not public automatic comparison', case=name,
                  backend=backend, mode=mode, profile_tag=profile_tag, gpu_uuid=uuid, recorded_identity=identity,
                  recorded_status=binding['recorded']['row']['status'],
                  binding_sha256=campaign.sha256_file(output/'bindings'/f'{name}.json'),
                  payload_sha256=binding['preparation_record']['sha256'],
                  runner_sha256=campaign.sha256_file(__file__),
                  replay_environment=identity['environment']['configured'],
                  guards={'before': False, 'after': False})
    native = None
    try:
        campaign.exclusive_gpu(uuid)
        report['guards']['before'] = True
        os.environ.update(identity['environment']['configured'])
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        os.environ['CUDA_MODULE_LOADING'] = identity['environment']['effective'].get('CUDA_MODULE_LOADING','LAZY')
        import torch
        import cubutterfly as cb
        observed = str(torch.cuda.get_device_properties(0).uuid)
        if not campaign.same_gpu_uuid(uuid, observed):
            raise RuntimeError('actual CUDA device differs from recorded target UUID')
        stream = torch.cuda.Stream()
        case = binding['case']; s = case['descriptor']['semantics']
        library = cb.Library(identity['libraries']['main']['path'])
        with torch.cuda.stream(stream):
            if backend == 'cubutterfly':
                descriptor = campaign.selection_descriptor(case['descriptor'], mapping=expected_mapping)
                with cb.Descriptor.from_json(library, descriptor) as imported:
                    native = cb.Plan(library, imported, stream=stream.cuda_stream)
                campaign.assert_semantics(case['descriptor'], json.loads(native.descriptor_json()))
                report['mapping'] = json.loads(native.mapping_json())
                report['mapping_replay'] = verify_mapping(selected, expected_mapping, report['mapping'], threads)
                workspace = torch.empty(max(1,native.workspace_size()), dtype=torch.uint8, device='cuda')
                if native.workspace_size(): native.set_workspace(workspace,native.workspace_size())
                initial_input = campaign.physical_bytes(s,'input',source,native.input_size())
                initial_output = np.full(native.output_size(),0xA5,dtype=np.uint8)
                src = torch.from_numpy(initial_input).to('cuda')
                out = torch.from_numpy(initial_output).to('cuda')
                execute = lambda: native.execute(src,out)
            else:
                from hadacore_baseline import build_core
                initial_input = campaign.physical_bytes(s,'input',source,2*s['batch']*s['shape'][0])
                initial_output = np.full(initial_input.size,0xA5,dtype=np.uint8)
                src = torch.from_numpy(initial_input).to('cuda')
                out = torch.from_numpy(initial_output).to('cuda')
                native = build_core(torch,s,src.view(torch.float16).reshape(source.shape),out.view(torch.float16).reshape(source.shape),stream,
                                    librarypath=identity['libraries']['CUBUTTERFLY_HADACORE_LIBRARY']['path'])
                report['metadata'] = native.metadata()
                execute = native
            saved = src.clone()
            direct = None
            def verify():
                raw = out.cpu().numpy().copy()
                if direct is not None and not np.array_equal(raw,direct):
                    raise AssertionError('replay output differs from direct output')
                actual = campaign.output_from_bytes(s,raw)
                campaign.check_padding(s,raw,initial_output)
                if not torch.equal(src,saved): raise AssertionError('input modified')
                result = campaign.check_full(actual,expected,0,'error-report',True)
                report['numeric_diagnostics'] = campaign.floating_range_report(actual,expected,0)
                report['finite'] = bool(np.all(np.isfinite(actual)))
                report['input_immutable'] = True
                return result
            execute(); stream.synchronize()
            report['correctness'] = verify()
            direct = out.cpu().numpy().copy()
            if mode == 'timing':
                report['timing'] = campaign.time_execute(torch,stream,execute,lambda:src.copy_(saved),False,verify,lambda:out.fill_(0xA5))
                report['graph_output_identical'] = True
            else:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    execute()
                out.fill_(0xA5)
                graph.replay()
                stream.synchronize()
                verify()
                report['graph_output_identical'] = True
                for _ in range(10): execute()
                stream.synchronize(); verify()
                profiler_call(torch,'cudaProfilerStart')
                try:
                    execute()
                    stream.synchronize()
                finally:
                    profiler_call(torch,'cudaProfilerStop')
                report['profile_calls'] = 1
                report['profile_includes'] = 'complete execute including HadaCore scale; kernel decomposition from profiler'
                report['profile_correctness'] = verify()
            report['status'] = 'accuracy-review-required'
    except Exception as exc:
        report.update(status='failed',error=str(exc))
        raise
    finally:
        try:
            after_guard(report,uuid)
        finally:
            if native is not None: native.close()
            write(output/('timing' if mode == 'timing' else 'profiler')/name/backend/((profile_tag or mode)+(f'-threads{threads}' if threads is not None else '')+'.json'),report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true'); action.add_argument('--run',action='store_true')
    p.add_argument('--source',type=Path)
    p.add_argument('--gpu-uuid')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=CASES)
    p.add_argument('--backend',choices=('cubutterfly','hadacore'))
    p.add_argument('--threads',type=int,choices=(128,256,512,1024))
    p.add_argument('--mode',choices=('timing','profile'))
    p.add_argument('--profile-tag',choices=('ncu','nsys'))
    return p


def main():
    p = parser(); args = p.parse_args()
    if args.prepare:
        if args.source is None: p.error('--prepare requires --source')
        if args.case or args.backend or args.mode or args.threads is not None or args.profile_tag: p.error('prepare selects the fixed five-point population')
        prepare(args.source,args.output)
    else:
        if not all((args.case,args.backend,args.mode)): p.error('run requires --case, --backend and --mode')
        if args.threads is not None and args.backend != 'cubutterfly': p.error('--threads is only available for cubutterfly')
        if args.profile_tag and args.mode != 'profile': p.error('--profile-tag requires profile mode')
        run(args.output,args.case,args.backend,args.mode,args.threads,args.gpu_uuid,args.profile_tag)

if __name__ == '__main__': main()
