#!/usr/bin/env python3
"""Score frozen calibration inventories and precompile a bounded priority set.

No device enumeration, plan probing, module loading, or GPU verification occurs.
The priority set never removes candidates from the full persisted ranking.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import time

from calibration_space import candidate_id, predicted_sample, stratified_candidates
from calibration_cpu_cache import DiskScoreCache
from calibrate_local_hardware import _SearchPredictionCache, _stage_service_usable_for_ranking
import research_compile_requests
import semantic_compile_broker as broker_api
import stage_cost_model


SCHEMA = 'cubutterfly-calibration-cpu-input-v1'


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verified_path(root, reference):
    path = (root / reference['path']).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('input is not frozen inside input package: ' + str(path))
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('input SHA256 mismatch: ' + str(path))
    return path


def project_point(point, descriptors, projection_scope='full'):
    resolved = descriptors.get(candidate_id(point), {})
    if resolved.get('status') == 'resolved':
        projection = resolved.get('stage_service_projection')
        if not isinstance(projection, dict):
            projection = ({'status': 'omitted', 'reason': 'search-stage-projection=plan'}
                          if projection_scope == 'plan' else {})
        return {**point, **resolved.get('sample', {}), **json.loads(resolved['mapping_json']),
                'N': 1 << int(point['logN']), 'stage_service_projection': projection,
                'stage_projection_status': projection.get('status'),
                'stage_projection_reason': projection.get('reason'),
                'execution_groups_json': resolved['execution_groups_json'],
                'descriptor_source': 'resolved-plan'}
    return {**predicted_sample(point), 'descriptor_source': 'projected'}


def ranking_profile(profile):
    usable = _stage_service_usable_for_ranking(profile)
    if not usable:
        result = dict(profile)
        result.pop('stage_service', None)
        return result
    service = profile['stage_service']
    if not (service.get('validated') or service.get('selected_scope_validated')):
        return dict(profile, stage_service=dict(service, enforce_ranking_scope=True))
    return profile


def measurement_rows(document):
    rows = document.get('rows', []) if isinstance(document, dict) else document
    if isinstance(rows, dict):
        rows = list(rows.values())
    return rows


def training_rows(document):
    return [{'sample': row['samples'][0], 'median_kernel_ms': row['median_kernel_ms']}
            for row in measurement_rows(document) if row.get('correct') and row.get('samples')]


def pending_compilation_points(document, cells):
    for row in measurement_rows(document):
        point = row.get('configuration')
        if row.get('status') != 'needs-compilation' or not isinstance(point, dict):
            continue
        if any(all(point.get(key, 'fft' if key == 'operator' else None) == value
                   for key, value in cell['workload'].items()) for cell in cells):
            yield point


def prepare_compilation(selected, output, broker, jobs, state, save):
    config = json.loads((broker / 'config.json').read_text())['config']
    state['existing_broker_queue'] = broker_api.status(broker)
    save()
    compilation = []
    budget_names = ('CUBUTTERFLY_COMPILE_BUDGET_FILE', 'CUBUTTERFLY_COMPILE_BUDGET_SECONDS')
    budget = {name: os.environ.pop(name) for name in budget_names if name in os.environ}
    try:
        for identifier, point in selected.items():
            row = dict(candidate_id=identifier)
            compilation.append(row)
            try:
                request = research_compile_requests.compile_request(point, policy='research')
            except Exception as error:
                row.update(status='cannot-project', error=str(error))
                continue
            if request is None:
                row['status'] = 'no-jit'
                continue
            row['request'] = request
            try:
                row.update(status='ready', path=str(broker_api.runtime_request(
                    broker, config, request, output / 'compile_events.jsonl', timeout=0)))
            except RuntimeError as error:
                row.update(status='queued-or-failed', error=str(error))
            except Exception as error:
                row.update(status='queue-failed', error=str(error))
        write(output / 'compilation.json', compilation)
        state['worker'] = broker_api.work(broker, jobs=jobs, retry_failed=False)
        for row in compilation:
            if 'request' not in row:
                continue
            try:
                row.update(status='ready', path=str(broker_api.runtime_request(
                    broker, config, row['request'], output / 'compile_events.jsonl', timeout=0)))
                row.pop('error', None)
            except Exception as error:
                row.update(status='failed', error=str(error))
        write(output / 'compilation.json', compilation)
    finally:
        for name in budget_names:
            os.environ.pop(name, None)
        os.environ.update(budget)
    state['compile_ready'] = all(row['status'] in ('ready', 'no-jit') for row in compilation)
    state['status'] = 'compile-ready' if state['compile_ready'] else 'failed'
    save()
    return state


def prepare(input_path, output, broker, jobs=4, compile_top=32, skip_compile=False, compile_only=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    input_path, output, broker = Path(input_path).resolve(), Path(output).resolve(), Path(broker).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = dict(schema='cubutterfly-calibration-cpu-preparation-v1', status='running',
                 pid=os.getpid(), current_cell=None, current_cell_index=None,
                 scored_candidates=0, total_candidates=0,
                 scoring_complete=False, compile_ready=False, gpu_verified=False,
                 input=str(input_path), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                 broker=str(broker), compile_top=compile_top, compile_only=compile_only,
                 cells=[], started_unix=time.time(),
                 policy='priority compilation only; full search inventory is retained')
    def save():
        state['updated_unix'] = time.time()
        write(output / 'status.json', state)
    save()
    try:
        if jobs < 1 or compile_top < 0:
            raise ValueError('jobs must be positive and compile-top nonnegative')
        if skip_compile and compile_only:
            raise ValueError('skip-compile and compile-only are mutually exclusive')
        manifest = json.loads(input_path.read_text())
        if manifest.get('schema') != SCHEMA or not isinstance(manifest.get('context'), dict):
            raise ValueError('unsupported CPU input schema/context')
        root = input_path.parent
        paths = {name: verified_path(root, manifest[name]) for name in ('profile', 'training', 'descriptors')}
        inventories = [verified_path(root, cell['inventory']) for cell in manifest['cells']]
        training_document = json.loads(paths['training'].read_text())
        recovery = {candidate_id(point): point for point in pending_compilation_points(
            training_document, manifest['cells'])}
        state['needs_compilation_candidates'] = len(recovery)
        if compile_only:
            return prepare_compilation(recovery, output, broker, jobs, state, save)
        for cell, inventory in zip(manifest['cells'], inventories):
            if not cell['complete']:
                with gzip.open(inventory, 'rt') as source:
                    state['total_candidates'] += sum(bool(line.strip()) for line in source)
        save()
        profile = json.loads(paths['profile'].read_text())
        training = training_rows(training_document)
        descriptors = json.loads(paths['descriptors'].read_text())
        descriptors = descriptors.get('descriptors', descriptors)
        model = stage_cost_model.fit(training, ranking_profile(profile))
        write(output / 'model.json', model)
        write(output / 'context.json', manifest['context'])
        selected = dict(recovery)
        last_heartbeat = time.monotonic()
        with DiskScoreCache(output / 'score_cache', manifest['context']) as cache:
            state['model_epoch'] = cache.bind(model)
            for index, (cell, inventory) in enumerate(zip(manifest['cells'], inventories)):
                receipt = dict(workload=cell['workload'], complete=cell['complete'], inventory=cell['inventory'])
                state['cells'].append(receipt)
                if cell['complete']:
                    receipt['status'] = 'skipped-complete'
                    continue
                state.update(current_cell=cell['workload'], current_cell_index=index)
                # Keep prepared features bounded to one workload, matching the
                # online search. Scores survive across cells in the disk cache.
                predictions = _SearchPredictionCache(descriptors, lambda point: project_point(
                    point, descriptors, manifest.get('projection_scope', 'full')))
                predictions.bind(model)
                with gzip.open(inventory, 'rt') as source:
                    points = [json.loads(line) for line in source if line.strip()]
                scores = []
                state['current_cell_total_candidates'] = len(points)
                state['current_cell_scored_candidates'] = 0
                save()
                for position, point in enumerate(points):
                    identifier = candidate_id(point)
                    descriptor = descriptors.get(identifier)
                    found, score = cache.get(identifier, descriptor)
                    if not found:
                        score = predictions.score(point)
                        cache.put(identifier, descriptor, score)
                    scores.append(dict(id=identifier, score=score if math.isfinite(score) else None,
                                       status='scored' if math.isfinite(score) else 'unrankable',
                                       inventory_index=position))
                    state['scored_candidates'] += 1
                    state['current_cell_scored_candidates'] = position + 1
                    now = time.monotonic()
                    if state['scored_candidates'] % 1024 == 0 or now - last_heartbeat >= 5:
                        cache.flush()
                        state['score_cache'] = dict(cache.stats)
                        save()
                        last_heartbeat = now
                ranking = sorted(scores, key=lambda item: (item['score'] if item['score'] is not None else math.inf,
                                                          item['inventory_index']))
                stem = 'cell-' + str(index)
                write(output / (stem + '-scores.json'), scores)
                write(output / (stem + '-ranking.json'), ranking)
                prioritized = [points[row['inventory_index']] for row in ranking[:compile_top]]
                prioritized += stratified_candidates(points, 0)[3::4][:8]
                prioritized += [seed.get('point', seed) for seed in cell.get('seeds', [])]
                for point in prioritized:
                    selected.setdefault(candidate_id(point), point)
                receipt.update(status='scoring-complete', candidate_count=len(points),
                               scores=stem + '-scores.json', ranking=stem + '-ranking.json')
                cache.flush()
                write(output / (stem + '-ready.json'), dict(
                    status='scoring-complete', gpu_verified=False, workload=cell['workload'],
                    candidate_count=len(points), model_epoch=state['model_epoch'],
                    source_sha256=cell['inventory']['sha256'],
                    input_sha256=state['input_sha256'],
                    scores=dict(path=receipt['scores'], sha256=hashlib.sha256(
                        (output / receipt['scores']).read_bytes()).hexdigest()),
                    ranking=dict(path=receipt['ranking'], sha256=hashlib.sha256(
                        (output / receipt['ranking']).read_bytes()).hexdigest())))
                state['score_cache'] = dict(cache.stats)
                save()
        state['scoring_complete'] = True
        state['status'] = 'scoring-complete'
        save()
        if skip_compile:
            return state
        return prepare_compilation(selected, output, broker, jobs, state, save)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--broker', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--compile-top', type=int, default=32)
    parser.add_argument('--skip-compile', action='store_true', help='Persist full scores without queueing or compiling')
    parser.add_argument('--compile-only', action='store_true',
                        help='Compile saved needs-compilation records without scoring inventories')
    args = parser.parse_args(argv)
    state = prepare(args.input, args.output, args.broker, args.jobs, args.compile_top,
                    args.skip_compile, args.compile_only)
    return 0 if state['compile_ready'] or (args.skip_compile and state['scoring_complete']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
