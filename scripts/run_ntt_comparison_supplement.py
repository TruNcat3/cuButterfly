#!/usr/bin/env python3
"""Bounded exact-NTT replay and candidate diagnostic; not a replacement aggregate."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import numpy as np
import run_semantic_campaign as semantic
import semantic_preparation_queue as preparation
from summarize_semantic_campaign import load_checkpoint

EXPECTED_CASE_IDS = (
    '5efd0f8413634bedd656367729403d68f2cee2ab3dd64e64ac6ede887cd6374e',
    '65a20b1e396e8dedf43998cf17c5abc6e335e828604cf260440d8bda6fe7ca8f',
    'eab91b52d5d9956d6834400a1cbb896cb39a4fe9322928dc8eab4f00eb2623fa',
    '83c4cf45fa81604df1644041d4ebf3ae8bc04986d7617340953378b2b961bf4b',
    '38556e5d55dfe6b061be144c07808ad356e1c611efb81082dfbf776eb7a0a27e',
)


def mappings(row, sweep):
    original = row['mapping']
    yield 'recorded-public-selection', original
    if not sweep:
        return
    s = row['descriptor']['semantics']
    if s['shape'] != [65536] or s['batch'] != 4:
        return
    base = copy.deepcopy(original)
    for unit in ('radix4', 'radix2'):
        for local in range(6, 11):
            for rows in (4, 2, 1):
                m = dict(base, backend='hybrid2d', compute_unit=unit,
                         n1_log=local, rows_per_block=rows, threads_per_block=256)
                if m != original:
                    yield f'hybrid-{unit}-local{local}-rows{rows}', m
    for threads in (128, 256):
        for layout in ('linear', 'hermes-xor'):
            m = dict(base, backend='shared-iterative', compute_unit='radix2',
                     n1_log=0, rows_per_block=0, threads_per_block=threads,
                     stage_partition=[8, 8], local_stage_partitions=[[3, 3, 2], [3, 3, 2]],
                     dataflow_layout=layout)
            yield f'shared-8-8-t{threads}-{layout}', m
    for backend in ('compact-stage', 'tile256'):
        yield backend, dict(base, backend=backend, compute_unit='radix2',
                            n1_log=0, rows_per_block=0, threads_per_block=0)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--profile', action='store_true')
    ap.add_argument('--candidates', nargs='+')
    ap.add_argument('--library', type=Path)
    ap.add_argument('--automatic', action='store_true', help='Rerun public MEASURE selection, separately from replay')
    ap.add_argument('--case-id')
    ap.add_argument('--exclude-case-id')
    ap.add_argument('--preparation', type=Path, default=ROOT / 'results/install_compare_20261013/prepared')
    ap.add_argument('--checkpoint', type=Path, default=ROOT / 'results/install_compare_repair_20261014/measurement/results.json')
    ap.add_argument('--manifest', type=Path, help='Defaults to the manifest bound to --preparation')
    ap.add_argument('--output', type=Path, default=os.environ.get('XD_REPORT_DIR'))
    ap.add_argument('--gpu-uuid', default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    ap.add_argument('--gpuntt-library', type=Path, default=ROOT / 'build-e13-cufftdx/libgpuntt_semantic_bridge.so')
    ap.add_argument('--dry-run', action='store_true', help='Validate bindings and population without Torch or CUDA')
    args = ap.parse_args(argv)
    if args.trials < 1 or args.output is None:
        ap.error('positive --trials and --output (or XD_REPORT_DIR) are required')
    if not args.gpu_uuid or not args.gpu_uuid.startswith('GPU-') or ',' in args.gpu_uuid:
        ap.error('select one --gpu-uuid (or CUDA_VISIBLE_DEVICES GPU UUID)')
    if args.case_id and args.exclude_case_id:
        ap.error('--case-id and --exclude-case-id are mutually exclusive')
    if args.automatic and args.candidates:
        ap.error('--automatic cannot filter candidates')
    return args


def bind_inputs(args):
    identity, all_rows = load_checkpoint(args.checkpoint)
    by_id = {row['id']: row for row in all_rows}
    missing = set(EXPECTED_CASE_IDS) - set(by_id)
    if missing:
        raise ValueError('checkpoint lacks required five-point population: ' + ', '.join(sorted(missing)))
    rows = [by_id[case_id] for case_id in EXPECTED_CASE_IDS]
    for row in rows:
        s = row['descriptor']['semantics']
        if s['operator'] != 1 or s.get('word_bits') != 64 or len(s['shape']) != 1:
            raise ValueError('five-point NTT semantic contract changed')
        if not row.get('mapping') or not row.get('timing'):
            raise ValueError('checkpoint lacks current measured selection: ' + row['id'])
    for selected in (args.case_id, args.exclude_case_id):
        if selected and selected not in EXPECTED_CASE_IDS:
            raise ValueError('case filter must name one of the five fixed requests')
    if args.case_id:
        rows = [r for r in rows if r['id'] == args.case_id]
    if args.exclude_case_id:
        rows = [r for r in rows if r['id'] != args.exclude_case_id]
    config, _ = preparation._config(args.preparation)
    source_manifest = json.loads(args.manifest.read_text()) if args.manifest else config['manifest']
    from semantic_campaign_manifest import validate_manifest_hash
    if not validate_manifest_hash(source_manifest) or source_manifest != config['manifest']:
        raise ValueError('manifest differs from bound preparation')
    if config['source_sha256'] != preparation._sources():
        raise ValueError('preparation source identity mismatch')
    cases = {c['id']: c for c in source_manifest['cases']}
    library_path = (args.library or ROOT / 'build-contract-20261013/libcuntt.so').resolve(strict=True)
    if semantic.sha256_file(library_path) != identity['identity']['library_sha256']:
        raise ValueError('library differs from checkpoint measured build')
    if not semantic.same_gpu_uuid(args.gpu_uuid, identity['identity']['gpu_uuid']):
        raise ValueError('GPU UUID differs from checkpoint')
    for row in rows:
        if row['id'] not in cases or cases[row['id']]['descriptor']['semantics'] != row['descriptor']['semantics']:
            raise ValueError('checkpoint semantics differ from preparation')
        record = preparation.record_for(args.preparation, row['id'])
        if record is None or record['status'] != 'ready':
            raise ValueError('CPU preparation is not ready: ' + row['id'])
    args.gpuntt_library.resolve(strict=True)
    return identity, rows, cases, library_path


def finish_report(result):
    primary = 'public-measure' if result['selection_policy'] == 'public-measure' else 'recorded-public-selection'
    completed = {r['case_id'] for r in result['rows']
                 if r['status'] == 'pass' and r['candidate'] == primary
                 and len(r['trials']) == result['requested_trials']}
    result['remaining_case_ids'] = sorted(set(result['selected_case_ids']) - completed)
    result['status'] = 'complete' if not result['remaining_case_ids'] else 'complete-with-gaps'


def main(argv=None):
    args = parse_args(argv)
    identity, rows, cases, library_path = bind_inputs(args)
    out, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if args.dry_run:
        print(json.dumps(dict(status='ready', checkpoint=str(checkpoint), library=str(library_path),
                              preparation=str(args.preparation.resolve()), output=str(out),
                              gpu_uuid=args.gpu_uuid, case_ids=[row['id'] for row in rows])))
        return 0
    # Preparation and reference checks precede CUDA initialization.
    inputs = {r['id']: preparation.load_case(args.preparation, cases[r['id']])
              for r in rows}
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    import torch
    import cubutterfly as cb
    import semantic_gpuntt_baseline as gpuntt
    if not semantic.same_gpu_uuid(args.gpu_uuid, str(torch.cuda.get_device_properties(0).uuid)):
        raise ValueError('runtime GPU UUID mismatch')
    library = cb.Library(library_path)
    stream = torch.cuda.Stream()
    result = dict(schema='cubutterfly-ntt-diagnostic-v1', status='running',
                  source=str(checkpoint), library_sha256=semantic.sha256_file(library_path),
                  library=str(library_path), search_budget=os.environ.get('CUBUTTERFLY_SEARCH_BUDGET'),
                  runner_sha256=semantic.sha256_file(Path(__file__)),
                  gpu_uuid=args.gpu_uuid,
                  scope='five recorded selections; optional finite candidate sweep on N65536 B4',
                  source_kind='explicit replay of public selections; variants are diagnostic only', rows=[])
    result['selection_policy'] = 'public-measure' if args.automatic else 'explicit-diagnostic-replay'
    result['source_library_sha256'] = identity['identity']['library_sha256']
    result['changed_library'] = result['library_sha256'] != result['source_library_sha256']
    result.update(checkpoint_sha256=semantic.sha256_file(checkpoint),
                  preparation=str(args.preparation.resolve()), expected_case_ids=list(EXPECTED_CASE_IDS),
                  selected_case_ids=[row['id'] for row in rows], full_population=len(rows) == len(EXPECTED_CASE_IDS),
                  requested_trials=args.trials, protocol=semantic.PROTOCOL,
                  gpuntt_library=str(args.gpuntt_library.resolve()),
                  gpuntt_sha256=semantic.sha256_file(args.gpuntt_library),
                  preparation_config_sha256=preparation._config(args.preparation)[1])
    out.mkdir(parents=True, exist_ok=True)

    def save():
        semantic.write_report(out / 'results.json', result)

    with torch.cuda.stream(stream):
        for row in rows:
            s = row['descriptor']['semantics']
            prepared = inputs[row['id']]
            src = torch.from_numpy(np.ascontiguousarray(prepared['source'])).to('cuda')
            dst = torch.empty_like(src)
            saved = src.clone()
            core = gpuntt.build_core(torch, s, src, dst, stream, args.gpuntt_library)

            def verify():
                semantic.check_full(dst.cpu().numpy().reshape(-1), prepared['expected'].reshape(-1), s['storage'])
                if not torch.equal(src, saved):
                    raise AssertionError('input modified')

            options = [('public-measure', None)] if args.automatic else mappings(row, args.sweep)
            for name, mapping in options:
                if args.candidates and name not in args.candidates:
                    continue
                rec = dict(case_id=row['id'], semantics=s, candidate=name, mapping=mapping,
                           original_ms=row['timing']['median_ms'], status='pending', trials=[])
                result['rows'].append(rec)
                plan = None
                try:
                    desc = semantic.selection_descriptor(row['descriptor'], timing=args.automatic, mapping=mapping)
                    begin = time.monotonic()
                    with cb.Descriptor.from_json(library, desc) as imported:
                        plan = cb.Plan(library, imported, stream=stream.cuda_stream)
                    rec['plan_build_seconds_excluded'] = time.monotonic() - begin
                    semantic.assert_semantics(row['descriptor'], json.loads(plan.descriptor_json()))
                    rec['actual_mapping'] = json.loads(plan.mapping_json())
                    rec['selection_reason'] = plan.selection_reason()
                    workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device='cuda')
                    if plan.workspace_size():
                        plan.set_workspace(workspace, workspace.numel())
                    calls = {'cubutterfly': lambda: plan.execute(src, dst), 'gpu-ntt': core}
                    for trial in range(args.trials if name in ('recorded-public-selection', 'public-measure') else 1):
                        timings = {}
                        for backend in (('cubutterfly', 'gpu-ntt') if trial % 2 == 0 else ('gpu-ntt', 'cubutterfly')):
                            dst.fill_(0)
                            calls[backend]()
                            stream.synchronize()
                            verify()
                            timings[backend] = semantic.time_execute(torch, stream, calls[backend],
                                lambda: src.copy_(saved), False, verify, lambda: dst.fill_(0))
                        timings['speedup'] = timings['gpu-ntt']['median_ms'] / timings['cubutterfly']['median_ms']
                        rec['trials'].append(timings)
                    if args.profile and s['shape'] == [65536] and s['batch'] == 4 and name in ('recorded-public-selection', 'hybrid-radix4-local8-rows4'):
                        for backend, execute in calls.items():
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                                for _ in range(5):
                                    execute()
                                stream.synchronize()
                            suffix = '' if name == 'recorded-public-selection' else '_' + name
                            prof.export_chrome_trace(str(out / (backend + suffix + '_trace.json')))
                            rec.setdefault('profiles', {})[backend] = [dict(name=e.name, duration_us=e.device_time_total)
                                for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
                    rec['status'] = 'pass'
                except cb.Error as exc:
                    rec.update(status='rejected' if exc.status in (1, 3) else 'runtime-error', error=str(exc))
                    if exc.status not in (1, 3):
                        save()
                        raise
                finally:
                    if plan is not None:
                        plan.close()
                save()
                print(json.dumps(dict(candidate=name, shape=s['shape'], batch=s['batch'], status=rec['status'],
                                      speedups=[t['speedup'] for t in rec['trials']])), flush=True)
            core.close()
    finish_report(result)
    save()
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
