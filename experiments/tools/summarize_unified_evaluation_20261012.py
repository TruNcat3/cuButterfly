#!/usr/bin/env python3
"""Produce a CPU-only, provenance-preserving snapshot of bounded evaluation data."""
import argparse
import csv
import hashlib
import json
import sqlite3
from functools import reduce
from operator import mul
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OLD = Path('results/paper_targeted_20261011')
NEW = Path('results/paper_targeted_20261012')
LIBRARIES = ('cubutterfly', 'cufft', 'vkfft', 'daofht', 'gpu-ntt')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


class Snapshot:
    def __init__(self):
        self.sources = {}
        self.checks = []
        self.missing = []
        self.exclusions = []

    def read(self, path):
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(ROOT)
        raw = (ROOT / path).read_bytes()
        data = json.loads(raw)
        self.sources[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw), schema=data.get('schema'), human_verified=False,
            verification='machine parsed; human scientific acceptance not inferred')
        identity_keys = ('identity', 'campaign_revision', 'protocol_version', 'library_sha256',
                         'frozen_library_sha256', 'binary_identity', 'gpu_uuid', 'runner_sha256',
                         'adapter_sha256', 'state_sha256', 'test_tensor_sha256', 'revision',
                         'lane', 'cache_policy', 'xd_tune_max_bytes', 'input_identity')
        self.sources[str(path)]['data_version'] = {key: data[key] for key in identity_keys if key in data}
        return data

    def artifact(self, path):
        path = Path(path)
        if path.is_absolute():
            path = path.relative_to(ROOT)
        if (ROOT / path).exists():
            raw = (ROOT / path).read_bytes()
            self.sources[str(path)] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw), verification='machine hashed raw profiler artifact', human_verified=False)

    def check(self, name, passed, detail):
        self.checks.append(dict(name=name, passed=bool(passed), detail=detail))

    def row(self, path):
        envelope = self.read(path)
        row = envelope['row']
        checksum = hashlib.sha256(canonical(row).encode()).hexdigest()
        self.check('row-sha256', checksum == envelope['sha256'], str(path))
        return row


def semantics(row):
    s = row['descriptor']['semantics']
    operator = {0: 'fft', 1: 'ntt', 2: 'fwht', 3: 'subset-zeta',
                4: 'superset-zeta', 5: 'structured-2x2'}.get(s['operator'], str(s['operator']))
    precision = {0: 'fp16', 1: 'bf16', 2: 'fp32', 3: 'fp64', 4: 'fp16',
                 5: 'bf16', 6: 'fp32', 7: 'fp64', 8: 'word32', 9: 'word64'}.get(s['storage'])
    if operator in ('subset-zeta', 'superset-zeta'):
        precision = 'uint32'
    return dict(operator=operator, precision=precision, N=reduce(mul, s['shape'], 1),
        xD=len(s['shape']), shape=s['shape'], B=s['batch'],
        direction='inverse' if s['inverse'] else 'forward',
        input_order=s['input_orders'], output_order=s['output_orders'],
        fft_type={0: 'C2C', 1: 'R2C', 2: 'C2R'}.get(s['fft_type']) if operator == 'fft' else None,
        normalize_inverse=s['normalize_inverse'], fft_output_scale=s['fft_output_scale'],
        placement={0: 'out-of-place', 1: 'in-place'}.get(s['placement']),
        length_mode=s['length_mode'], modulus=s['modulus'],
        matrices=s['matrices'],
        input_strides=s['input_strides'], output_strides=s['output_strides'],
        input_batch_stride=s['input_batch_stride'], output_batch_stride=s['output_batch_stride'])


def operator_tables(snapshot):
    manifest_path = OLD / 'operator_prepare/e1_fixed_36_v4/manifest.json'
    manifest = snapshot.read(manifest_path)
    expected = {c['id']: c for c in manifest['cases']}
    found = {}
    for run in ('e1_run_v4', 'e1_run_v4_continuation', 'e1_run_v4_continuation2'):
        for path in sorted((ROOT / OLD / run).glob('**/measurement/rows/*.json')):
            row = snapshot.row(path)
            result_path = path.parent.parent / 'results.json'
            if str(result_path.relative_to(ROOT)) not in snapshot.sources:
                snapshot.read(result_path)
            snapshot.check('e1-no-duplicate-id', row['id'] not in found, str(path.relative_to(ROOT)))
            found[row['id']] = [(row, str(path.relative_to(ROOT)))]
    snapshot.check('e1-complete-manifest', set(found) == set(expected),
                   dict(expected=len(expected), observed=len(found)))
    tables = []
    cohorts = [('E1-20261011-v4-complete', expected, found, 1)]
    ntt_manifest = snapshot.read(NEW / 'ntt_extension/manifest.json')
    ntt_expected = {c['id']: c for c in ntt_manifest['cases']}
    ntt_found = {key: [] for key in ntt_expected}
    for trial in (1, 2, 3):
        directory = ROOT / NEW / 'ntt_extension' / f'trial-{trial}'
        report_path = directory / 'results.json'
        if report_path.exists():
            report = snapshot.read(report_path)
            snapshot.sources[str(report_path.relative_to(ROOT))]['identity'] = report.get('identity')
        paths = sorted(directory.glob('rows/*.json'))
        for path in paths:
            row = snapshot.row(path)
            if row['id'] in ntt_found:
                ntt_found[row['id']].append((row, str(path.relative_to(ROOT))))
        if len(paths) != len(ntt_expected):
            snapshot.missing.append(f'NTT trial {trial}: {len(paths)}/{len(ntt_expected)} row snapshots; no missing point imputed.')
    cohorts.append(('NTT-20261012-extension-snapshot', ntt_expected, ntt_found, 3))
    for cohort, cases, measurements, target_trials in cohorts:
        for key, case in cases.items():
            observations = measurements.get(key, [])
            base = dict(cohort=cohort, id=key, **semantics(case),
                        trials=len(observations), target_trials=target_trials,
                        source_paths=[p for _, p in observations],
                        identity_source_paths=[str(Path(p).parent.parent / 'results.json') for _, p in observations],
                        manifest_path=str(manifest_path if cohort.startswith('E1') else NEW / 'ntt_extension/manifest.json'))
            scopes = sorted({r.get('selection_scope', 'missing') for r, _ in observations})
            public = bool(observations) and all('public' in scope.lower() for scope in scopes)
            base.update(selection_scope=scopes, public_selector=public,
                complete_call=True if observations else None,
                algorithm=[r.get('algorithm') for r, _ in observations])
            if observations:
                snapshot.check('operator-public-selector', public, key)
                snapshot.check('operator-semantic-identity', all(r['descriptor']['semantics'] == case['descriptor']['semantics'] for r, _ in observations), key)
            for library in LIBRARIES:
                records = [r if library == 'cubutterfly' else next((b for b in r.get('baselines', []) if b['name'] == library), {}) for r, _ in observations]
                values = [r.get('timing', {}).get('median_ms') for r in records]
                base[library + '_us'] = median([v * 1000 for v in values if v is not None]) if public or library != 'cubutterfly' else None
                base[library + '_process_us'] = [v * 1000 if v is not None else None for v in values]
                base[library + '_status'] = sorted({r.get('status', 'not-measured') for r in records}) or ['missing']
                base[library + '_support'] = sorted({r.get('support_status', 'measured' if r else 'not-applicable-or-not-run') for r in records})
                base[library + '_protocol'] = sorted({r.get('timing', {}).get('protocol', 'missing') for r in records})
                base[library + '_error_l2'] = [r.get('correctness', {}).get('l2_relative') for r in records]
                base[library + '_exact'] = [r.get('correctness', {}).get('exact') for r in records]
                base[library + '_mismatches'] = [r.get('correctness', {}).get('mismatches') for r in records]
                base[library + '_adapter_operations'] = [r.get('adapter_operations', []) for r in records]
                applicable = library == 'cubutterfly' or library in {'fft': ('cufft', 'vkfft'), 'fwht': ('daofht',), 'ntt': ('gpu-ntt',)}.get(base['operator'], ())
                unsupported = any('unsupported' in r.get('status', '') or 'unsupported' in r.get('support_status', '') for r in records)
                base[library + '_availability'] = ('measured' if base[library + '_us'] is not None else
                    'not-applicable' if not applicable else 'unsupported' if unsupported else 'missing')
                reasons = [r.get('reason', r.get('support_metadata', {}).get('reason')) for r in records]
                base[library + '_reason'] = [reason for reason in reasons if reason] or [
                    'Different operator domain' if not applicable else
                    'See recorded support/status and original baseline record' if unsupported else
                    'Expected applicable comparison has no current timing' if base[library + '_us'] is None else 'Complete call measured']
                if base['operator'] == 'ntt':
                    for record in records:
                        if record.get('status') == 'pass':
                            correctness = record.get('correctness', {})
                            snapshot.check('ntt-fullbatch-exact', correctness.get('exact') and correctness.get('mismatches') == 0 and correctness.get('elements') == base['N'] * base['B'], key + ':' + library)
            base['acceptance'] = 'exact-pass' if base['precision'] in ('word32', 'word64', 'uint32') and base['cubutterfly_status'] == ['pass'] else 'accuracy-review-required-or-see-status'
            tables.append(base)
    return tables


def memcpy_fields(snapshot, stem):
    path = Path(str(stem) + '.nsys.sqlite')
    fields = dict(counter_scope='kernel_only', memcpy_included_in_NCU_bytes=False,
                  memcpy_events_exist=None, memcpy_count=None, memcpy_bytes=None,
                  memcpy_duration_us=None, memcpy_kinds=[], memcpy_evidence='missing SQLite export')
    if not (ROOT / path).exists():
        return fields
    snapshot.artifact(path)
    with sqlite3.connect((ROOT / path).as_uri() + '?mode=ro', uri=True) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies = []
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            copies = list(connection.execute('SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
        labels = {}
        if 'ENUM_CUDA_MEMCPY_OPER' in tables:
            labels = dict(connection.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER'))
        fields.update(memcpy_events_exist=bool(copies), memcpy_count=len(copies),
            memcpy_bytes=sum(row[2] for row in copies), memcpy_duration_us=sum(row[1] - row[0] for row in copies) / 1000,
            memcpy_kinds=sorted({labels.get(row[3], str(row[3])) for row in copies}),
            memcpy_evidence=str(path))
        if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
            kernels = list(connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL'))
            fields['nsys_kernel_count'] = len(kernels)
            intervals = sorted(kernels)
            merged = []
            for start, end in intervals:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(end, merged[-1][1])
                else:
                    merged.append([start, end])
            fields['nsys_kernel_union_us'] = sum(end - start for start, end in merged) / 1000
    return fields


def profile_fields(profile):
    sums = profile.get('per_call_counter_sums', {})
    kernels = profile.get('kernels', [])
    return dict(kernel_count=profile.get('ncu_kernel_count', profile.get('kernel_count')),
        nsys_kernel_count=profile.get('nsys_kernel_count'), logical_calls=profile.get('logical_calls'),
        L2_bytes=profile.get('l2_read_write_bytes_from_32byte_sectors'),
        DRAM_read_bytes=sums.get('dram__bytes_read.sum'), DRAM_write_bytes=sums.get('dram__bytes_write.sum'),
        registers_per_kernel=profile.get('per_kernel_registers', [k['metrics'].get('launch__registers_per_thread') for k in kernels]),
        shared_bytes_per_kernel=profile.get('per_kernel_shared_bytes', [k['metrics'].get('launch__shared_mem_per_block_allocated') for k in kernels]),
        shared_load_bank_conflicts=sums.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
        shared_store_bank_conflicts=sums.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),
        long_scoreboard_pct_per_kernel=profile.get('per_kernel_long_scoreboard_pct', [k['metrics'].get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct') for k in kernels]),
        nsys_kernel_union_us=profile.get('nsys_kernel_union_us', profile.get('nsys_interval_union_us')),
        profiler_is_normal_latency=False, occupancy=None, L2_hit_rate=None)


def mechanism_tables(snapshot):
    source = NEW / 'mechanism/summary.json'
    summary = snapshot.read(source)
    journal_path = NEW / 'mechanism' / summary['timing_source']
    journal = snapshot.read(journal_path)
    profiles = {(p['case'], p['variant']): p for p in summary['profiles']}
    rows, contrasts, external_rows = [], [], []
    for case in journal['cases']:
        workload = case['workload']
        for variant in case['variants']:
            trials = variant['trials']
            profile = profiles.get((case['id'], variant['id']), {})
            stem = NEW / 'mechanism' / profile.get('source_directory', 'missing') / (case['id'] + '.' + variant['id'])
            profile_sources = [str(stem) + extension for extension in ('.ncu.csv', '.nsys.trace.csv')]
            for path in profile_sources:
                snapshot.artifact(path)
            sums = profile.get('per_call_counter_sums', {})
            snapshot.check('profile-L2-sector-denominator', profile.get('l2_read_write_bytes_from_32byte_sectors') == 32 * (sums.get('lts__t_sectors_op_read.sum', 0) + sums.get('lts__t_sectors_op_write.sum', 0)), case['id'] + ':' + variant['id'])
            accepted = variant['preflight']['status'] == 'accepted' and len(trials) == 3 and all(t.get('performance_accepted') for t in trials)
            sample = variant['preflight']['sample']
            N = int(sample['N'])
            B = int(sample['batch'])
            snapshot.check('mechanism-native-shape', N == 2 ** workload['logN'] and B == workload['batch'], case['id'] + ':' + variant['id'])
            profile_values = profile_fields(profile)
            profile_values.update(memcpy_fields(snapshot, stem))
            rows.append(dict(cohort='mechanism-20261012-internal', case=case['id'],
                component='stage_partition' if workload['operator'] != 'fft' else 'fft_core x factor_overlap',
                variant=variant['id'], operator=workload['operator'], precision=workload['precision'],
                N=N, shape=[N], xD=1, B=B, direction=workload['direction'],
                shape_evidence='preflight.sample.N,batch,batch_stride; native --logN/--batch command',
                native_batch_stride=sample.get('batch_stride'),
                latency_us=median([t['kernel_ms'] * 1000 for t in trials]),
                process_latency_us=[t['kernel_ms'] * 1000 for t in trials],
                status=('accepted-underfilled-diagnostic' if workload['operator'] != 'fft' else 'accepted-factorial-control') if accepted else 'invalid-control',
                interpretation='underfilled stagewise diagnostic; not tuned optimum' if workload['operator'] != 'fft' else 'fixed-core factorial control',
                source_paths=[str(source), str(journal_path)] + profile_sources,
                mapping=variant['preflight'].get('resolved_mapping'), **profile_values))
        variants = {v['id']: v for v in case['variants']}
        for contrast in summary['timing']:
            if contrast['case'] != case['id']:
                continue
            for side in ('baseline', 'treatment'):
                latency = median([t['kernel_ms'] * 1000 for t in variants[contrast[side]]['trials']])
                snapshot.check('mechanism-summary-timing-consistency', abs(latency - contrast[side + '_us']) < 1e-8, case['id'] + ':' + side)
            contrasts.append(dict(cohort='mechanism-20261012-internal', source_path=str(source),
                interpretation='underfilled-diagnostic-only' if workload['operator'] != 'fft' else 'factorial-control', **contrast))
    for path, operator, N in [(NEW / 'mechanism/external/summary.json', 'ntt', 65536),
                             (NEW / 'mechanism/external/fwht_float_report/summary.json', 'fwht', 32768)]:
        external = snapshot.read(path)
        for method in ('cubutterfly', 'external'):
            timing = external.get('timing', {}).get(method, {})
            profile = external.get('profiles', {}).get(method, {})
            stem = path.parent / f'{operator}.{method}'
            profile_values = profile_fields(profile)
            profile_values.update(memcpy_fields(snapshot, stem))
            external_rows.append(dict(cohort='mechanism-20261012-external-diagnostic',
                case=f'{operator}-external-N{N}-B16', component='external comparison (not ablation)',
                variant=method if method == 'cubutterfly' else ('GPU-NTT Merge' if operator == 'ntt' else 'Dao FWHT'),
                operator=operator, precision='word64' if operator == 'ntt' else 'fp32', N=N, shape=[N], xD=1, B=16,
                direction='forward', latency_us=timing.get('median_us'),
                process_latency_us=timing.get('process_median_us'),
                status='fixed-diagnostic' if operator == 'ntt' else 'accuracy-review-required',
                interpretation='fixed resident cuB mapping; never library selector performance',
                source_paths=[str(path), str(stem) + '.nsys.sqlite'], **profile_values))
    snapshot.missing.extend(['Mechanism: occupancy and L2 hit rate not collected; percentages retained per kernel, never averaged.',
        'Mechanism: no matched, well-filled stagewise single-axis ablation in the current cohort; existing stagewise is diagnostic.',
        'Operators: Dao FWHT has no FP64 support and rejects N65536; GPU-NTT Merge comparison is available for word64 only; no matched external subset/superset-zeta or structured-2x2 baseline in E1.'])
    return rows, contrasts, external_rows


def application_tables(snapshot, revision=None):
    rows = []
    groups = {}
    filter_contracts = {}
    rejected_filters = {}
    filter_sources = [('applications-20261012-filter', NEW / 'fft_filter')]
    if revision is not None:
        filter_sources.append((f'applications-20261012-filter-{revision.name}', revision / 'fft_filter'))
    else:
        snapshot.missing.append('Application batch revision not selected; v1/failed attempts excluded. Use --application-revision with one explicit revision root.')
    for cohort, base in filter_sources:
        manifest_path = base / 'manifest.json'
        if revision is not None and base == revision / 'fft_filter':
            plan_path = revision / 'fft_filter_plan.json'
            if (ROOT / plan_path).exists():
                plan = snapshot.read(plan_path)
                manifest_path = Path(plan['input_identity']['manifest_path']).relative_to(ROOT)
                snapshot.artifact(manifest_path)
                snapshot.check('revision-input-manifest-sha256', snapshot.sources[str(manifest_path)]['sha256'] == plan['input_identity']['manifest_sha256'], str(plan_path))
        if (ROOT / manifest_path).exists():
            manifest = snapshot.read(manifest_path)
            for contract in manifest['contracts']:
                filter_contracts[(cohort, contract['id'])] = (contract, str(manifest_path))
        for process in (1, 2, 3):
            guard = base / f'process{process}_guard/status.json'
            if (ROOT / guard).exists() and snapshot.read(guard).get('state') != 'complete':
                snapshot.exclusions.append(dict(path=str(guard), reason='Process guard is not complete; release/attempt rows not imported'))
                continue
            for path in sorted((ROOT / base / f'process{process}_guard/release').glob('*.json')):
                data = snapshot.read(path)
                if 'contract' in data and not data.get('tune_only', False) and not data.get('requested', {}).get('plan_only', False):
                    if revision is not None and base == revision / 'fft_filter':
                        plans = data.get('plan_metadata', []) + data.get('filter_plan_metadata', [])
                        valid = data.get('requested', {}).get('require_cache') is True and bool(plans) and all(p.get('selection_source') == 'user_cache' for p in plans)
                        if not valid:
                            reason = 'FFT cache-replay gate failed: require_cache=true and every image/filter plan selection_source=user_cache required'
                            rejected_filters.setdefault((cohort, data['contract']['id']), []).append(reason)
                            snapshot.exclusions.append(dict(path=str(path.relative_to(ROOT)), reason=reason))
                            continue
                    groups.setdefault((cohort, data['contract']['id']), []).append((data, str(path.relative_to(ROOT))))
    for (cohort, key), observations in sorted(groups.items()):
        data = observations[0][0]
        batch = data['contract'].get('batch', data['contract'].get('batch_size', 1))
        row = dict(cohort=cohort, application='FFT linear image filtering (same)',
            input=key, B=batch, precision='fp32', source_paths=[p for _, p in observations],
            status='exploratory: numerical tolerance unassessed', throughput_unit='images/s',
            trials=len(observations), cuB_padding=data['contract']['cubutterfly_padding'],
            competitor_padding=data['cupy_native_fft_padding'],
            timing_scope='GPU resident complete call: pad, image/filter FFT, multiply, inverse, crop; no cached filter; setup/transfer excluded')
        for method, name in [('cubutterfly', 'cubutterfly'), ('cupy_fftconvolve_native_padding', 'cupy_native'), ('torch_cufft_same_padding_diagnostic', 'torch_samepad_diagnostic')]:
            records = [r for d, _ in observations for r in d['rows'] if r['method'] == method and r.get('timing')]
            if not records:
                if name == 'torch_samepad_diagnostic':
                    continue
                row[name + '_ms'] = None
                row[name + '_throughput'] = None
                continue
            value = median([r['timing']['median_us'] for r in records])
            row[name + '_ms'] = value / 1000
            row[name + '_throughput'] = batch * 1e6 / value
            row[name + '_process_us'] = [r['timing']['median_us'] for r in records]
            row[name + '_relative_l2'] = records[0]['error']['relative_l2']
        if len(observations) != 3:
            snapshot.missing.append(f'{cohort} {key}: {len(observations)}/3 processes; partial snapshot.')
        snapshot.check('filter-complete-calls', all(d['complete_filter_call'] and not d['filter_spectrum_cached'] for d, _ in observations), key)
        snapshot.check('filter-contract-identical', all(d['contract'] == data['contract'] for d, _ in observations), key)
        rows.append(row)
    for (cohort, key), (contract, source) in sorted(filter_contracts.items()):
        if (cohort, key) not in groups:
            rows.append(dict(cohort=cohort, application='FFT linear image filtering (same)',
                input=key, B=contract.get('batch', 1), precision='fp32', trials=0,
                source_paths=[source], status='failed cache-replay gate' if (cohort, key) in rejected_filters else 'missing timed process results',
                cubutterfly_ms=None, cupy_native_ms=None,
                cubutterfly_throughput=None, cupy_native_throughput=None,
                throughput_unit='images/s', reason=rejected_filters.get((cohort, key), ['Prepared contract only; plan/tune-only runs excluded'])))
            snapshot.missing.append(f'{cohort} {key}: prepared contract has no timed process result.')
    quarot_groups = {}
    quarot = []
    for process in (1, 2):
        path = NEW / f'quarot/full512_timing_p{process}.json'
        if (ROOT / path).exists():
            quarot.append((snapshot.read(path), str(path)))
    if quarot:
        quarot_groups['applications-20261012-QuaRot'] = quarot
    batch_root = ROOT / revision if revision is not None else None
    for batch in ((1, 4, 8) if batch_root is not None else ()):
        observations = []
        rejected = []
        for process in (1, 2):
            path = batch_root / f'quarot_b{batch}_p{process}/release/result.json'
            guard = path.parent.parent / 'status.json'
            if guard.exists() and snapshot.read(guard).get('state') != 'complete':
                snapshot.exclusions.append(dict(path=str(guard.relative_to(ROOT)), reason='Process guard is not complete; release/attempt rows not imported'))
                continue
            if path.exists():
                data = snapshot.read(path)
                if not data.get('tune_only', False):
                    setups = [data.get('adapter_setup', {}), data.get('cubutterfly', {}).get('adapter_setup', {})]
                    valid = data.get('status') == 'pass' and all(bool(setup.get('plans')) and all(p.get('selection_source') == 'user_cache' for p in setup['plans']) and setup.get('plans_created') == len(setup['plans']) for setup in setups)
                    if not valid:
                        reason = 'QuaRot cache-replay gate failed: every validation and post-timing plan must be represented and use user_cache; status=pass required'
                        rejected.append(reason)
                        snapshot.exclusions.append(dict(path=str(path.relative_to(ROOT)), reason=reason))
                        continue
                    observations.append((data, str(path.relative_to(ROOT))))
        if observations:
            quarot_groups[f'applications-20261012-QuaRot-{revision.name}-batch{batch}'] = observations
        elif batch_root.exists():
            rows.append(dict(cohort=f'applications-20261012-QuaRot-{revision.name}-batch{batch}', application='QuaRot random-weight Llama',
                input='prefill=512, decode=32, phase=end_to_end', B=batch, trials=0,
                source_paths=[], status='failed cache-replay gate' if rejected else 'missing timed process results', cubutterfly_ms=None,
                dao_baseline_ms=None, cubutterfly_throughput=None, dao_baseline_throughput=None,
                throughput_unit='tokens/s', reason=rejected or ['Declared batch extension; no complete process result yet']))
            snapshot.missing.append(f'QuaRot batch={batch}: no complete timing at {revision}/quarot_b{batch}_p{{1,2}}/release/result.json.')
    for cohort, quarot in sorted(quarot_groups.items()):
        payload = quarot[0][0]['payload']
        for phase in ('prefill', 'decode', 'end_to_end'):
            row = dict(cohort=cohort, application='QuaRot random-weight Llama',
                input=f'prefill={payload["prefill"]}, decode={payload["decode"]}, phase={phase}',
                B=payload['batch_size'], precision='fp32 composed FWHT; model fp16', trials=len(quarot),
                source_paths=[p for _, p in quarot], status='implementation consistency pass; random weights; no quality claim',
                throughput_unit='tokens/s', timing_scope='complete synchronized wall time; setup and first-use excluded')
            tokens = payload['prefill'] if phase == 'prefill' else payload['decode'] if phase == 'decode' else payload['prefill'] + payload['decode']
            for method, name in [('baseline', 'dao_baseline'), ('cubutterfly', 'cubutterfly')]:
                times = [statistics.median(d[method]['wall_ms'][phase]) for d, _ in quarot]
                value = statistics.median(times)
                row[name + '_ms'] = value
                row[name + '_throughput'] = tokens * payload['batch_size'] * 1000 / value
                row[name + '_process_ms'] = times
            snapshot.check('quarot-consistency-gate', all(d['status'] == 'pass' for d, _ in quarot), phase)
            rows.append(row)
        if len(quarot) != 2:
            snapshot.missing.append(f'{cohort}: {len(quarot)}/2 independent processes; partial snapshot.')
    for resolution in (64, 128):
        path = OLD / f'application_prepare/runs/fno_darcy_{resolution}/attempt-03/report.json'
        if not (ROOT / path).exists():
            snapshot.missing.append(f'FNO resolution {resolution}: latest complete attempt-03 missing.')
            continue
        data = snapshot.read(path)
        row = dict(cohort='applications-20261011-FNO-attempt03', application='FNO Darcy complete forward',
            input=f'{resolution}x{resolution}, heldout={data["n_test"]}', B=data['batch_size'], precision='fp32',
            trials=data['process_trials'], source_paths=[str(path)],
            status='consistency gate pass; checkpoint missing' if data['pass'] else 'failed',
            throughput_unit='samples/s', timing_scope=data['protocol'])
        for method, name in [('native_ms', 'torch_native'), ('cubutterfly_ms', 'cubutterfly')]:
            value = data[method]['median_of_trial_medians']
            row[name + '_ms'] = value
            row[name + '_throughput'] = data['batch_size'] * 1000 / value
            row[name + '_process_ms'] = data[method]['trial_medians']
        rows.append(row)
    rows.append(dict(cohort='applications-20261012-CKKS-missing', application='CKKS', input='not measured',
        B=None, trials=0, source_paths=[], status='missing complete application result',
        cubutterfly_ms=None, competitor_ms=None, cubutterfly_throughput=None,
        competitor_throughput=None, throughput_unit=None,
        reason='No complete CKKS application contract/timing in the bounded current cohort; NTT operator timings do not substitute for an application.'))
    snapshot.missing.extend(['Applications: CKKS has no complete application contract, competitor timing, or throughput in the current cohort.',
        'Applications: no 20261012 FNO rerun; 20261011 attempt-03 is separately labelled.',
        'Applications: QuaRot prefill2048/decode128 has no repaired full timing in this round; prior invalid attempts excluded.',
        'Applications: filter numerical tolerance unassessed; no accepted performance claim inferred.'])
    return rows


def selector_ablation_coverage(snapshot):
    rows = []
    for operator in ('fft', 'fwht', 'ntt', 'subset-zeta', 'superset-zeta', 'structured-2x2'):
        fixed = 'fft_core x factor_overlap factorial' if operator == 'fft' else 'underfilled stage_partition diagnostic' if operator in ('fwht', 'ntt') else 'missing'
        rows.append(dict(operator=operator, component='case-dependent optional mechanisms; disabled-search coverage not measured',
            public_selector_operator_results='available in E1' + (' and current extension' if operator == 'ntt' else ''),
            fixed_variable_control=fixed, full_search_us=None, component_disabled_research_us=None,
            best_envelope_ratio=None, selector_avoids_known_inferior_choice=None,
            equal_search_budget='required; no paired evidence', equal_candidate_correctness_protocol='required; no paired evidence',
            coverage_status='missing paired full-search/component-disabled re-search in current cohort; no GPU experiment added',
            reason='Fixed mappings and variable controls do not establish the selector best envelope after re-search'))
    snapshot.missing.append('Selector/component ablation: the current bounded cohort has no complete cross-operator equal-budget full-search versus component-disabled re-search envelope; known-inferior-choice avoidance is not audited here. See selector_ablation_coverage.csv (NA values).')
    return rows


def core_transfer_coverage(snapshot):
    rows = []
    for operator, core in [('fft', 'register-tile'), ('fft', 'cufftdx-block'),
                           ('fwht', 'scalar radix2 shared-iterative'), ('ntt', 'radix2 Shoup shared-iterative'),
                           ('subset-zeta', 'generic radix2; second matched leaf pair missing'),
                           ('superset-zeta', 'generic radix2; second matched leaf pair missing'),
                           ('structured-2x2', 'generic radix2; second matched leaf pair missing')]:
        rows.append(dict(operator=operator, core=core, original_organization_us=None,
            same_core_mixed_template_us=None, same_core_transfer_ratio=None,
            repeated_across_cores='current cohort lacks matched original-organization versus mixed-template pairs',
            current_evidence=('local factor_overlap toggle under factor-streamed' if operator == 'fft' else
                              'underfilled stage_partition diagnostic' if operator in ('fwht', 'ntt') else
                              'operator timing exists; no matched core-by-organization experiment in current cohort'),
            coverage_status='missing full core-by-dataflow transfer evidence in current cohort',
            reason='Keep the core/kernel fixed within each pair; repeat the organization transfer on multiple native cores; current controls do not establish this contrast'))
    snapshot.missing.append('Core/dataflow portability ablation: the current bounded cohort lacks matched same-core original organization versus mixed-template organization repeated on multiple cores. Historical/core-organization experiments elsewhere in the repository are outside this snapshot and are not asserted absent. Current FFT core x overlap is a local toggle and its approximately 1% core replacement difference is not a unified-template benefit.')
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def display(value):
    if value is None or value == []:
        return '缺'
    if isinstance(value, float):
        return f'{value:.5g}'
    if isinstance(value, (dict, list)):
        return canonical(value).replace('|', '\\|')
    return str(value).replace('|', '\\|').replace('\n', ' ')


def markdown_table(rows, fields):
    def cell(row, key):
        if key not in row:
            return '\u2014'
        if key.endswith('_us') and key[:-3] in LIBRARIES:
            availability = row.get(key[:-3] + '_availability')
            if availability == 'not-applicable':
                return '\u2014'
            if availability == 'unsupported':
                return '不支持'
        if key == 'fft_type' and row.get('operator') != 'fft':
            return '\u2014'
        if key in ('full_search_us', 'component_disabled_research_us', 'best_envelope_ratio', 'selector_avoids_known_inferior_choice',
                   'original_organization_us', 'same_core_mixed_template_us', 'same_core_transfer_ratio') and row[key] is None:
            return 'NA'
        return display(row[key])
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    return '\n'.join(lines + ['| ' + ' | '.join(cell(row, key) for key in fields) + ' |' for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / NEW / 'unified_report')
    parser.add_argument('--application-revision', type=Path,
        help='One explicit revision root containing fft_filter and quarot_b*_p*; never discovers old attempts or pools revisions')
    args = parser.parse_args()
    snapshot = Snapshot()
    operators = operator_tables(snapshot)
    revision = args.application_revision
    if revision is not None:
        revision = revision.resolve() if revision.is_absolute() else (ROOT / revision).resolve()
        if not revision.is_dir():
            parser.error('application revision must be an existing directory')
        revision = revision.relative_to(ROOT)
        if revision == NEW / 'application_batch':
            parser.error('the failed application_batch v1 is excluded; select the notified versioned revision root')
    mechanisms, contrasts, external_mechanisms = mechanism_tables(snapshot)
    applications = application_tables(snapshot, revision)
    coverage = selector_ablation_coverage(snapshot)
    core_coverage = core_transfer_coverage(snapshot)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {'operators': operators, 'mechanisms': mechanisms, 'mechanism_contrasts': contrasts,
              'external_mechanism_diagnostics': external_mechanisms, 'applications': applications}
    tables['selector_ablation_coverage'] = coverage
    tables['core_dataflow_transfer_coverage'] = core_coverage
    for name, rows in tables.items():
        write_csv(args.output / f'{name}.csv', rows)
    summary = dict(schema='cubutterfly-unified-evaluation-snapshot-v1',
        report_date='2026-10-12', aggregation='median of observed process medians within each explicit cohort; no cross-cohort aggregation or imputation',
        submission_ready=False, human_scientific_verification='not recorded',
        selected_application_revision=str(revision) if revision is not None else None,
        counts={name: len(rows) for name, rows in tables.items()},
        sources=list(snapshot.sources.values()), checks=snapshot.checks,
        missing=snapshot.missing, excluded_results=snapshot.exclusions, tables=tables)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    text = ['# Unified evaluation snapshot (2026-10-12)', '',
        'Draft evidence report. Source paths, SHA256 identities, per-process values and machine checks are retained in [summary.json](summary.json). Human scientific acceptance is not inferred. Re-run `python3 experiments/tools/summarize_unified_evaluation_20261012.py` to refresh the live row snapshot.', '',
        'All latencies are absolute complete-call times with required semantic adapters included. Planning, search and compilation are excluded. E1 and the current NTT extension are separate cohorts; observed trials are shown explicitly. No fixed resident diagnostic is included as public-selector library performance. Floating results marked accuracy-review-required retain their status even when historical mismatches are zero.', '',
        '## Unified operator table', '',
        'Orders use the source semantic enum (0 = natural). Shape, FFT type, normalization, placement, strides, matrices and modulus are retained in [operators.csv](operators.csv); each library also has support, reason, status, protocol and error columns. `\u2014` means not applicable, `不支持` means an applicable library rejects this contract, and `缺` means expected evidence is absent. None of these denotes zero time. All points, including losses and unsupported competitors, are retained.', '',
        markdown_table(operators, ['cohort', 'operator', 'precision', 'fft_type', 'shape', 'xD', 'B', 'direction', 'input_order', 'output_order', 'trials', 'cubutterfly_us', 'cufft_us', 'vkfft_us', 'daofht_us', 'gpu-ntt_us', 'acceptance']), '',
        '## Mechanism table', '',
        'Normal latency comes from three unprofiled process trials. NCU counters sum kernel events within one captured call, with counter_scope=kernel_only. L2 bytes = 32 times kernel read/write sectors. Memcpy events are separately audited from read-only NSYS SQLite exports and are excluded from NCU bytes; these counters are not total complete-call traffic. DRAM traffic is observed after warmup, not compulsory algorithm bytes. Register/shared/wait values remain per kernel; occupancy and L2 hit rate are unmeasured (NA). Stagewise is deliberately underfilled and these ratios diagnose organization rather than establish an optimized competitor.', '',
        'FFT factorial is a one-dimensional N=256 transform at B=16384, confirmed by native --logN=8, preflight N=256 and batch_stride=256. Its factor_partition=[4,4] does not make it a 256x256 transform.', '',
        'The unified method does not require every optional mechanism to improve every butterfly operator. Fixed-variable negative results delimit where a mechanism is useful; they do not by themselves reject the method. The intended library result is the complete plan automatically selected for each operator and workload. A fixed-variable mechanism control is distinct from comparing the best full-search plan with the best plan after disabling one component and re-searching under the same candidate budget and correctness protocol. Fixed-resident external losses do not substitute for that selector comparison.', '',
        'The second key ablation keeps the native core/kernel fixed and compares its original organization with the same core integrated into the mixed dataflow template, then repeats this transfer on multiple cores. The current FFT core x factor_overlap factorial only changes a local overlap switch inside factor-streamed organization; it is not a complete core x dataflow-template transfer experiment. Its approximately 1% core replacement difference is not evidence of unified-template benefit.', '',
        markdown_table(mechanisms, ['operator', 'shape', 'xD', 'B', 'component', 'variant', 'latency_us', 'kernel_count', 'counter_scope', 'memcpy_count', 'memcpy_included_in_NCU_bytes', 'L2_bytes', 'DRAM_read_bytes', 'DRAM_write_bytes', 'shared_load_bank_conflicts', 'shared_store_bank_conflicts', 'registers_per_kernel', 'long_scoreboard_pct_per_kernel', 'status']), '',
        'Component contrasts, including FFT overlap losses, are retained in [mechanism_contrasts.csv](mechanism_contrasts.csv).', '',
        markdown_table(contrasts, ['case', 'actual_changed_axes', 'baseline', 'treatment', 'baseline_us', 'treatment_us', 'baseline_over_treatment', 'accepted', 'interpretation']), '',
        '### External fixed-mapping diagnostics (not component ablations)', '',
        'These rows are kept in [external_mechanism_diagnostics.csv](external_mechanism_diagnostics.csv), separately from the component table. The normal complete-call latency includes copies; kernel-only NCU traffic does not. Dao FWHT includes a D2D copy, whose bytes and duration are listed separately and are not added to hardware counter traffic.', '',
        markdown_table(external_mechanisms, ['operator', 'shape', 'B', 'variant', 'latency_us', 'kernel_count', 'counter_scope', 'memcpy_count', 'memcpy_bytes', 'memcpy_kinds', 'memcpy_included_in_NCU_bytes', 'L2_bytes', 'DRAM_read_bytes', 'DRAM_write_bytes', 'shared_load_bank_conflicts', 'shared_store_bank_conflicts', 'status']), '',
        '### Selector and re-search coverage', '',
        'The paired best-envelope experiment and evidence that the selector avoids known inferior choices are incomplete in the current bounded cohort. The following framework records NA without expanding the actual GPU matrix; [selector_ablation_coverage.csv](selector_ablation_coverage.csv) retains required budget/protocol matching and missing reasons.', '',
        markdown_table(coverage, ['operator', 'fixed_variable_control', 'full_search_us', 'component_disabled_research_us', 'best_envelope_ratio', 'selector_avoids_known_inferior_choice', 'coverage_status']), '',
        '### Same-core organization transfer coverage', '',
        '[core_dataflow_transfer_coverage.csv](core_dataflow_transfer_coverage.csv) records the original-organization versus mixed-template evidence missing from this bounded cohort. All corresponding paired latencies here are NA; the existing local-switch controls are not substituted. Historical experiments elsewhere in the repository are outside this snapshot, not declared nonexistent or unimplemented.', '',
        markdown_table(core_coverage, ['operator', 'core', 'original_organization_us', 'same_core_mixed_template_us', 'same_core_transfer_ratio', 'current_evidence', 'coverage_status']), '',
        '## End-to-end application table', '',
        'The filter cohort uses native-padding CuPy as its competitor; Torch same-padding is a diagnostic column. QuaRot uses random weights and SDPA fallback; consistency does not establish model quality. FNO uses the separately labelled latest complete prior-day attempt and has no checkpoint. Throughput is B/time for images or FNO samples, and B times phase token count/time for QuaRot. All application times below are milliseconds.', '',
        f'Explicit application revision: {revision if revision is not None else "none selected; failed v1 batch attempts excluded"}. Use --application-revision to select one version; the generator never searches attempt directories or pools revisions.', '',
        'Selected batch revision rows require completed process guards. FFT additionally requires requested.require_cache=true and all image/filter selection sources=user_cache. QuaRot checks every represented validation and post-timing plan uses user_cache, so new plans created during timing are included in the audit. Failures remain excluded with explicit reasons in summary.json.', '',
        markdown_table(applications, ['cohort', 'application', 'input', 'B', 'trials', 'cubutterfly_ms', 'cupy_native_ms', 'dao_baseline_ms', 'torch_native_ms', 'competitor_ms', 'torch_samepad_diagnostic_ms', 'cubutterfly_throughput', 'cupy_native_throughput', 'dao_baseline_throughput', 'torch_native_throughput', 'throughput_unit', 'status']), '',
        '## Consistency and missing evidence', '',
        f'Machine checks: {sum(c["passed"] for c in snapshot.checks)}/{len(snapshot.checks)} passed. Source versions and source identity dictionaries are recorded in summary.json. Numeric acceptance and human verification remain distinct from these checks.', '',
        *['- ' + item for item in snapshot.missing], '']
    (args.output / 'report.md').write_text('\n'.join(text))
    print(json.dumps(dict(counts=summary['counts'], checks_passed=all(c['passed'] for c in snapshot.checks), output=str(args.output))))
    if not all(c['passed'] for c in snapshot.checks):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
