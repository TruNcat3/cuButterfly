#!/usr/bin/env python3
"""Measure exact rank4/rank8 vendor compositions against frozen public choices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import numpy as np
import run_semantic_campaign as semantic
import semantic_preparation_queue as preparation
from summarize_semantic_campaign import load_checkpoint
from cpu_external_baselines import same_precision_reference, write

HISTORICAL = ROOT / 'results/ntt_diagnostic_20261017'
EXPECTED_CASE_IDS = (
    'b9ec9a4de1a071429577024e910e4086c700056c7c2911d7e3a9584420772e40',
    'ca982faf49a529e96de928ba8171ac758af0c6f43056fd4a4d73a01367d89b86',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'results/install_compare_repair_20261014/measurement/results.json')
    parser.add_argument('--manifest', type=Path, help='Defaults to the manifest bound to --preparation')
    parser.add_argument('--library', type=Path, default=HISTORICAL / 'original_libcuntt.so')
    parser.add_argument('--preparation', type=Path, default=HISTORICAL / 'prepared')
    parser.add_argument('--output', type=Path, default=os.environ.get('XD_REPORT_DIR'))
    parser.add_argument('--gpu-uuid', default=os.environ.get('CUDA_VISIBLE_DEVICES'))
    parser.add_argument('--vkfft-library', type=Path, default=ROOT / 'build-a100-vkfft/libvkfft_xd_bridge.so')
    parser.add_argument('--dry-run', action='store_true', help='Validate bindings and population without Torch or CUDA')
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error('--output (or XD_REPORT_DIR) is required')
    if not args.gpu_uuid or not args.gpu_uuid.startswith('GPU-') or ',' in args.gpu_uuid:
        parser.error('select one --gpu-uuid (or CUDA_VISIBLE_DEVICES GPU UUID)')
    return args


def bind_inputs(args):
    identity, rows = load_checkpoint(args.checkpoint)
    by_id = {row['id']: row for row in rows}
    if not set(EXPECTED_CASE_IDS) <= set(by_id):
        raise ValueError('checkpoint lacks required rank4/rank8 population')
    records = [by_id[case_id] for case_id in EXPECTED_CASE_IDS]
    config, _ = preparation._config(args.preparation)
    manifest = json.loads(args.manifest.read_text()) if args.manifest else config['manifest']
    from semantic_campaign_manifest import validate_manifest_hash
    if not validate_manifest_hash(manifest) or manifest != config['manifest']:
        raise ValueError('manifest differs from bound preparation')
    if config['source_sha256'] != preparation._sources():
        raise ValueError('preparation source identity mismatch')
    cases = {case['id']: case for case in manifest['cases']}
    path = args.library.resolve(strict=True)
    if semantic.sha256_file(path) != identity['identity']['library_sha256']:
        raise ValueError('library differs from checkpoint measured build')
    if not semantic.same_gpu_uuid(args.gpu_uuid, identity['identity']['gpu_uuid']):
        raise ValueError('GPU UUID differs from checkpoint')
    for record in records:
        s = record['descriptor']['semantics']
        if s['operator'] != 0 or len(s['shape']) not in (4, 8):
            raise ValueError('rank4/rank8 FFT semantic contract changed')
        if record['id'] not in cases or cases[record['id']]['descriptor']['semantics'] != s:
            raise ValueError('checkpoint semantics differ from preparation')
        if not record.get('mapping') or not record.get('timing'):
            raise ValueError('checkpoint lacks current measured selection')
        prepared = preparation.record_for(args.preparation, record['id'])
        if prepared is None or prepared['status'] != 'ready':
            raise ValueError('CPU preparation is not ready: ' + record['id'])
    args.vkfft_library.resolve(strict=True)
    return identity, records, cases, path


def main(argv=None):
    args = parse_args(argv)
    identity, records, cases, path = bind_inputs(args)
    out, checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if args.dry_run:
        print(json.dumps(dict(status='ready', checkpoint=str(checkpoint), library=str(path),
                              preparation=str(args.preparation.resolve()), output=str(out),
                              gpu_uuid=args.gpu_uuid, case_ids=[row['id'] for row in records])))
        return 0
    prepared = {r['id']: preparation.load_case(args.preparation, cases[r['id']]) for r in records}
    references = {key: same_precision_reference(p['s'], p['source'], p['expected']) for key, p in prepared.items()}
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    import torch
    import cubutterfly as cb
    if not semantic.same_gpu_uuid(args.gpu_uuid, str(torch.cuda.get_device_properties(0).uuid)):
        raise ValueError('runtime GPU UUID mismatch')
    library = cb.Library(path)
    stream = torch.cuda.Stream()
    result = dict(schema='cubutterfly-baseline-gap-supplement-v1', status='running', gpu_uuid=args.gpu_uuid,
                  library_sha256=semantic.sha256_file(path), source_checkpoint=str(checkpoint),
                  library=str(path), checkpoint_sha256=semantic.sha256_file(checkpoint),
                  preparation=str(args.preparation.resolve()), expected_case_ids=list(EXPECTED_CASE_IDS),
                  protocol=semantic.PROTOCOL, vkfft_library=str(args.vkfft_library.resolve()),
                  vkfft_sha256=semantic.sha256_file(args.vkfft_library),
                  preparation_config_sha256=preparation._config(args.preparation)[1],
                  selection='replay of checkpoint public MEASURE choices; same measured library', rows=[])
    out.mkdir(parents=True, exist_ok=True)
    with torch.cuda.stream(stream):
        for old in records:
            s = old['descriptor']['semantics']
            source = prepared[old['id']]['source']
            expected, reference = references[old['id']]
            record = dict(id=old['id'], semantics=s, reference=reference,
                          input_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                          mapping=old['mapping'], baseline_results=[])
            chosen = semantic.selection_descriptor(old['descriptor'], mapping=old['mapping'])
            with cb.Descriptor.from_json(library, chosen) as descriptor:
                plan = cb.Plan(library, descriptor, stream=stream.cuda_stream)
            try:
                semantic.assert_semantics(old['descriptor'], json.loads(plan.descriptor_json()))
                input_raw = semantic.physical_bytes(s, 'input', source, plan.input_size())
                gpu_input = torch.from_numpy(input_raw.copy()).to('cuda')
                saved = gpu_input.clone()
                output_initial = np.full(plan.output_size(), 0xA5, dtype=np.uint8)
                gpu_output = torch.from_numpy(output_initial).to('cuda')
                workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device='cuda')
                if plan.workspace_size():
                    plan.set_workspace(workspace, workspace.numel())

                def verify():
                    raw = gpu_output.cpu().numpy()
                    actual = semantic.output_from_bytes(s, raw)
                    value = semantic.check_full(actual, expected, s['storage'], 'error-report')
                    semantic.check_padding(s, raw, output_initial)
                    if not torch.equal(gpu_input, saved):
                        raise AssertionError('cuButterfly modified input')
                    if 'error_artifact' not in record:
                        record['error_artifact'] = semantic.retain_error_artifact(out / 'errors', old['id'], 'cubutterfly', actual, expected)
                        record['correctness'] = value

                plan.execute(gpu_input, gpu_output)
                stream.synchronize()
                verify()
                record['timing'] = semantic.time_execute(torch, stream, lambda: plan.execute(gpu_input, gpu_output),
                    lambda: gpu_input.copy_(saved), False, verify, lambda: gpu_output.fill_(0xA5))
                record['baseline_results'] = semantic.run_baselines(cases[old['id']], torch, stream,
                    source, expected, saved, output_initial, True,
                    args.vkfft_library, 'error-report', out / 'errors')
                if {b['name'] for b in record['baseline_results'] if b['name'] in ('cufft', 'vkfft')} != {'cufft', 'vkfft'}:
                    raise RuntimeError('grouped supplement requires both cuFFT and VkFFT baseline records')
                for b in record['baseline_results']:
                    if b['name'] in ('cufft', 'vkfft'):
                        if b['status'] != 'accuracy-review-required':
                            raise RuntimeError('baseline supplement failed: ' + json.dumps(b))
                        b['speedup'] = b['timing']['median_ms'] / record['timing']['median_ms']
                result['rows'].append(record)
                write(out / 'supplement.json', result)
                print(json.dumps(dict(id=old['id'], shape=s['shape'], cubutterfly_ms=record['timing']['median_ms'],
                    baselines=[dict(name=b['name'], status=b['status'], speedup=b.get('speedup')) for b in record['baseline_results']])), flush=True)
            finally:
                plan.close()
    result['status'] = 'complete-error-report-only'
    write(out / 'supplement.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
