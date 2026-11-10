#!/usr/bin/env python3
"""Fill missing GPU-NTT bindings without repeating or replacing library trials.

Reuse immutable prepared inputs, the full-buffer CPU verifier, and the original
CUDA Graph timing implementation. Original and supplementary measurements keep
separate provenance; the combined view never edits the source campaign.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import time

import numpy as np

import run_semantic_campaign as runner
from run_async_semantic_campaign import ForeignGPUWork, exclusive_gpu, validate_payload
from semantic_cpu_pipeline import prepared_physical_bytes
from semantic_gpu_pipeline import baseline_steps
import semantic_gpuntt_baseline as gpuntt
import semantic_preparation_queue as preparation

HERE = Path(__file__).resolve().parent
SHARED_SOURCES = ('run_semantic_campaign.py', 'semantic_gpu_pipeline.py',
                  'semantic_cpu_pipeline.py', 'semantic_preparation_queue.py',
                  'semantic_ntt_reference.py', 'semantic_input_profiles.py')


class SupplementInterrupted(BaseException):
    """Do not let a backend's generic Exception handler swallow SIGTERM."""


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def select_missing(rows):
    selected = []
    for row in rows:
        matches = [b for b in row.get('baselines', []) if b['name'] == 'gpu-ntt']
        if len(matches) != 1:
            raise ValueError('expected one GPU-NTT coverage record per source row')
        baseline = matches[0]
        if (baseline['status'] != 'missing-baseline' or
                baseline.get('support_status') != 'missing-external-binding'):
            continue
        if row['status'] != 'pass' or row['descriptor']['semantics']['operator'] != 1:
            raise ValueError('baseline supplementation requires a validated NTT source trial')
        if not row.get('timing', {}).get('median_ms', 0) > 0:
            raise ValueError('source trial has no valid library timing')
        selected.append(row)
    return selected


def load_source(path, manifest_path):
    source = json.loads(path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (not source.get('status', '').startswith('complete') or
            source.get('remaining_case_ids') != []):
        raise ValueError('source campaign must be complete before supplementation')
    if runner.digest(manifest) != source['identity']['manifest_sha256']:
        raise ValueError('manifest differs from source campaign')
    if runner.digest(source['rows']) != source['rows_sha256']:
        raise ValueError('source row index checksum mismatch')
    cases = {case['id']: case for case in manifest['cases']}
    rows = []
    for index in source['rows']:
        if index['path'] != 'rows/' + index['id'] + '.json':
            raise ValueError('invalid source row path')
        envelope = json.loads((path.parent / index['path']).read_text())
        row = envelope['row']
        if (runner.digest(row) != envelope['sha256'] or envelope['sha256'] != index['sha256']
                or row['id'] != index['id'] or row['status'] != index['status']):
            raise ValueError('source row checksum or index mismatch')
        if row['descriptor'] != cases[row['id']]['descriptor']:
            raise ValueError('source row semantics differ from manifest')
        rows.append(row)
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate source rows')
    for name in SHARED_SOURCES:
        if runner.sha256_file(HERE / name) != source['identity']['sources'][name]:
            raise ValueError('shared timing/preparation/verification implementation changed: ' + name)
    return source, cases, rows


def prepare_inputs(directory, case, source_row):
    prepared = preparation.load_case(directory, case)
    input_raw = prepared_physical_bytes(prepared, 'input', prepared['source'], source_row['input_bytes'])
    output_raw = np.full(source_row['output_bytes'], 0xA5, dtype=np.uint8)
    return prepared, input_raw, output_raw


def summarize(source_rows, supplements):
    by_id = {row['id']: row for row in supplements}
    pairs, missing = [], []
    for source in source_rows:
        original = next(b for b in source['baselines'] if b['name'] == 'gpu-ntt')
        replacement = by_id.get(source['id'])
        baseline = replacement['baselines'][0] if replacement else original
        s = source['descriptor']['semantics']
        record = dict(id=source['id'], shape=s['shape'], batch=s['batch'],
                      word_bits=s['word_bits'], modulus=s['modulus'], inverse=s['inverse'],
                      supplemented=replacement is not None, baseline_status=baseline['status'])
        if source['status'] == 'pass' and baseline['status'] == 'pass':
            if baseline['timing']['protocol'] != source['timing']['protocol']:
                raise ValueError('baseline and library timing protocols differ')
            record.update(cubutterfly_ms=source['timing']['median_ms'],
                          gpuntt_ms=baseline['timing']['median_ms'])
            record['speedup'] = record['gpuntt_ms'] / record['cubutterfly_ms']
            pairs.append(record)
        else:
            record.update(support_status=baseline.get('support_status'), reason=baseline.get('reason'))
            missing.append(record)

    def stats(values):
        if not values:
            return {'count': 0}
        speeds = [v['speedup'] for v in values]
        return dict(count=len(speeds), geometric_mean=math.exp(statistics.mean(map(math.log, speeds))),
                    median=statistics.median(speeds), minimum=min(speeds), maximum=max(speeds),
                    wins=sum(v > 1 for v in speeds))

    return dict(scope='Original cuButterfly timings with separately measured missing GPU-NTT baselines; same device, input, build and timing protocol; not a fresh interleaved rerun.',
                paired=stats(pairs), original_pairs=stats([p for p in pairs if not p['supplemented']]),
                supplemented_pairs=stats([p for p in pairs if p['supplemented']]),
                word32=stats([p for p in pairs if p['word_bits'] == 32]),
                word64=stats([p for p in pairs if p['word_bits'] == 64]),
                pairs=pairs, unpaired=missing)


def run(args):
    if (args.output / 'verification_failure.json').exists():
        raise ValueError('saved CPU verification failure requires investigation')
    source, cases, source_rows = load_source(args.source_report, args.manifest)
    identity = source['identity']
    if not runner.same_gpu_uuid(args.gpu_uuid, identity['gpu_uuid']):
        raise ValueError('supplement must use the original GPU UUID')
    if runner.sha256_file(Path(identity['library'])) != identity['library_sha256']:
        raise ValueError('cuButterfly/root-helper library changed')
    inventory = gpuntt.backend_inventory(args.gpuntt_library)
    if not inventory['available']:
        raise ValueError(inventory['reason'])
    selected = select_missing(source_rows)
    supplement_identity = dict(source_report=str(args.source_report),
        source_report_sha256=runner.sha256_file(args.source_report),
        manifest_sha256=runner.sha256_file(args.manifest), gpu_uuid=args.gpu_uuid,
        protocol=identity['protocol'], backend=inventory,
        runner_sha256=runner.sha256_file(Path(__file__)),
        shared_sources={name: runner.sha256_file(HERE / name) for name in SHARED_SOURCES},
        validation_driver_sha256=runner.sha256_file(HERE / 'run_async_semantic_campaign.py'),
        selected=[dict(id=row['id'], sha256=runner.digest(row)) for row in selected])
    output = args.output / 'results.json'
    report = (json.loads(output.read_text()) if output.exists() else
              dict(schema='cubutterfly-semantic-baseline-supplement-v1', identity=supplement_identity,
                   rows=[], rows_sha256=runner.digest([]), status='ready'))
    known = runner.validate_resume(report, supplement_identity, args.output,
                                   {row['id'] for row in selected})
    if any(row['status'] != 'pass' for row in report['rows']):
        raise ValueError('supplement has a failed trial requiring investigation')

    def checkpoint(status, error=None, retryable=False):
        report.update(status=status, remaining=len(selected) - len(known), updated_unix=time.time())
        runner.write_report(output, report)
        state = dict(status=status, remaining=report['remaining'], measured_rows=len(known),
                     selected=len(selected), updated_unix=report['updated_unix'],
                     pid=os.getpid(), gpu_uuid=args.gpu_uuid, retryable=retryable)
        if error is not None:
            state['error'] = str(error)
        atomic_json(args.output / 'status.json', state)

    for row in selected:
        contract = gpuntt.support(row['descriptor']['semantics'], args.gpuntt_library)
        if not contract['supported']:
            checkpoint('stopped-binding-preflight', contract)
            return 2
    if args.prepare_only:
        atomic_json(args.output / 'plan.json', supplement_identity)
        print(json.dumps(dict(status='prepared', selected=len(selected), completed=len(known))), flush=True)
        return 0

    checkpoint('ready')
    # Set the source runtime before importing Torch; compilation is never invoked.
    for key, value in identity['runtime_environment'].items():
        os.environ[key] = value
    args.timing, args.float_verdict = True, identity['protocol']['float_verdict']
    args.error_dir = args.output / 'errors'
    pending = [row for row in selected if row['id'] not in known]

    def stop(signum, frame):
        raise SupplementInterrupted('received signal ' + str(signum))

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        exclusive_gpu(args.gpu_uuid)
        import torch
        if torch.__version__ != identity['torch'] or torch.version.cuda != identity['torch_cuda']:
            raise ValueError('Torch/CUDA version differs from source trials')
        if not runner.same_gpu_uuid(args.gpu_uuid, torch.cuda.get_device_properties(0).uuid):
            raise ValueError('visible device differs from requested UUID')
        stream = torch.cuda.Stream()
        checkpoint('running')
        # Prefetch the next exact CPU artifact while the current GPU case runs.
        with ThreadPoolExecutor(max_workers=1) as pool, torch.cuda.stream(stream):
            future = (pool.submit(prepare_inputs, identity['preparation'], cases[pending[0]['id']], pending[0])
                      if pending else None)
            for position, source_row in enumerate(pending):
                prepared, input_raw, output_raw = future.result()
                if position + 1 < len(pending):
                    next_row = pending[position + 1]
                    future = pool.submit(prepare_inputs, identity['preparation'], cases[next_row['id']], next_row)
                exclusive_gpu(args.gpu_uuid)
                gpu_input = torch.from_numpy(input_raw).to('cuda')
                steps = baseline_steps(cases[source_row['id']], prepared, torch, stream,
                                       gpu_input, input_raw, output_raw, args)
                verdict = None
                try:
                    while True:
                        exclusive_gpu(args.gpu_uuid)
                        try:
                            payload = steps.send(verdict)
                        except StopIteration as finished:
                            baselines = finished.value
                            break
                        exclusive_gpu(args.gpu_uuid)
                        verdict = validate_payload(payload)
                        del payload
                        checks = verdict.get('batch_results', [verdict])
                        if any(check.get('ok') is not True for check in checks):
                            atomic_json(args.output / 'verification_failure.json', dict(
                                id=source_row['id'], verdict=verdict, observed_unix=time.time()))
                            raise RuntimeError('CPU verification failed; retained verdict and error artifacts')
                except runner.BaselineCorrectnessFailure as error:
                    baselines = error.rows
                finally:
                    steps.close()
                baseline = next(b for b in baselines if b['name'] == 'gpu-ntt')
                row = dict(id=source_row['id'], descriptor=source_row['descriptor'],
                           status=baseline['status'], baselines=[baseline],
                           source_row_sha256=runner.digest(source_row),
                           source_cubutterfly_timing=source_row['timing'],
                           source_mapping=source_row.get('mapping'), measured_unix=time.time())
                if baseline['status'] == 'pass' and baseline['timing']['protocol'] != source_row['timing']['protocol']:
                    raise ValueError('timing protocol mismatch')
                if baseline['status'] == 'pass':
                    exclusive_gpu(args.gpu_uuid)
                report['rows'].append(runner.write_case_checkpoint(args.output, row))
                if baseline['status'] != 'pass':
                    checkpoint('stopped-baseline-failure', baseline)
                    return 2
                known.add(row['id'])
                checkpoint('running')
                print(json.dumps(dict(completed=len(known), total=len(selected), id=row['id'],
                                      gpuntt_ms=baseline['timing']['median_ms'])), flush=True)
                del prepared, input_raw, output_raw, gpu_input, steps, verdict
        supplements = [json.loads((args.output / item['path']).read_text())['row'] for item in report['rows']]
        atomic_json(args.output / 'summary.json', summarize(source_rows, supplements))
        checkpoint('complete')
        return 0
    except ForeignGPUWork as error:
        checkpoint('interrupted-by-foreign-gpu-work', error, retryable=True)
        return 76
    except SupplementInterrupted as error:
        checkpoint('interrupted', error)
        return 130
    except Exception as error:
        checkpoint('stopped-investigation', repr(error))
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--gpuntt-library', type=Path, default=gpuntt.DEFAULT_LIBRARY)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.source_report = args.source_report.resolve(strict=True)
    args.manifest = args.manifest.resolve(strict=True)
    args.output = args.output.resolve()
    args.gpuntt_library = args.gpuntt_library.resolve(strict=True)
    if args.output == args.source_report.parent or args.output in args.source_report.parents:
        parser.error('supplement output must be separate from source campaign')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'driver.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 73
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
