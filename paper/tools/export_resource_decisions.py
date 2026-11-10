#!/usr/bin/env python3
"""CPU-only audit of the finite November 3 resource/service/confirmation chain."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
COHORT = Path('results/a100_closure_20261103/final_acceptance')


def relative_error(predicted, observed):
    if observed <= 0:
        raise ValueError('positive observed latency required')
    return abs(predicted - observed) / observed


def dependency_demands(log_n, batch, value_bytes, groups):
    """Logical closed regions, independent of actual CTA packing or memory traffic."""
    n = 1 << log_n
    if sum(g['stage_count'] for g in groups) != log_n:
        raise ValueError('physical groups must cover the transform once')
    return {'value_bytes': value_bytes, 'semantic_io_bytes': 2 * batch * n * value_bytes,
            'group_interface_logical_bytes': 2 * batch * n * value_bytes * len(groups),
            'traffic_interpretation': 'one read/write per value per physical group; excludes roots, scratch and internal traffic; not measured HBM',
            'regions': [{'first_stage': g['first_stage'], 'stage_count': g['stage_count'],
                         'values_per_closed_region': 1 << g['stage_count'],
                         'closed_regions': batch * n // (1 << g['stage_count']),
                         'one_region_value_bytes': (1 << g['stage_count']) * value_bytes}
                        for g in groups],
            'live_bytes_interpretation': 'one closed-region value footprint, not compiler shared allocation or aggregate simultaneous liveness'}


def prospective_extension(root, sources):
    base = Path('results/nonfft_endpoint_control_20261107')
    index = json.loads((root / base / 'evidence_index.json').read_text())['files']
    def read(name):
        raw = (root / base / name).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == index[name]['sha256'] and len(raw) == index[name]['bytes']
        sources.append({'path': str(base / name), 'sha256': sha})
        return json.loads(raw)
    frozen, screen, winners = read('frozen.json'), read('screen.json'), read('winners.json')
    summary, study = read('summary.json'), read('screen_study.json')
    assert frozen['study'] == study
    freeze_sha = index['frozen.json']['sha256']
    winner_sha = index['winners.json']['sha256']
    assert summary['frozen_sha256'] == freeze_sha and summary['winners_sha256'] == winner_sha
    script = Path('scripts/run_nonfft_endpoint_control.py')
    script_sha = hashlib.sha256((root / script).read_bytes()).hexdigest()
    assert script_sha == frozen['identity']['sources'][str(script)]
    sources.append({'path': str(script), 'sha256': script_sha})
    holds = [read('holdout_%d.json' % i) for i in (1, 2, 3)]
    receipts = [read('guards/%s/status.json' % name) for name in ('screen', 'holdout_1', 'holdout_2', 'holdout_3')]
    assert all(s['state'] == 'complete' for s in receipts)
    assert [s['updated'] for s in receipts] == sorted(s['updated'] for s in receipts)
    assert read('guards/screen/release/screen.json') == screen
    assert read('guards/screen/release/winners.json') == winners
    assert len({h['pid'] for h in holds}) == 3
    for i, h in enumerate(holds):
        assert h['status'] == 'complete' and h['process_index'] == i
        assert h['frozen_sha256'] == freeze_sha and h['winners_sha256'] == winner_sha
        assert read('guards/holdout_%d/release/holdout.json' % (i + 1)) == h
    rows = []
    for chosen in winners['cells']:
        cell = chosen['cell']
        case = next(c for c in screen['cases'] if c['id'] == cell)
        accepted = [v for v in case['variants'] if v['preflight']['status'] == 'accepted'
                    and len(v['trials']) == 1 and v['trials'][0]['performance_accepted']]
        for side, choice in chosen['choices'].items():
            pool = [v for v in accepted if side == 'extended' or v['id'].endswith('-o1')]
            best = min(pool, key=lambda v: (v['trials'][0]['kernel_ms'], v['id']))
            assert best['id'] == choice['id']
            assert best['preflight']['resolved_mapping'] == choice['mapping']
        pairs = []
        for i, h in enumerate(holds):
            c = next(c for c in h['cases'] if c['id'] == cell)
            expected_order = sorted({v['id'] for v in chosen['choices'].values()}, reverse=i % 2 == 1)
            assert c['order'] == expected_order
            times = {}
            for side, choice in chosen['choices'].items():
                v = next(v for v in c['variants'] if v['id'] == choice['id'])
                assert v['preflight']['resolved_mapping'] == choice['mapping']
                assert v['trials'][0]['performance_accepted']
                times[side] = v['trials'][0]['kernel_ms']
            pairs.append(dict(process=i, conventional_ms=times['conventional'], extended_ms=times['extended'],
                              ratio=times['conventional'] / times['extended']))
        original = next(r for r in summary['rows'] if r['cell'] == cell)
        assert pairs == original['holdout_pairs']
        assert statistics.median(p['ratio'] for p in pairs) == original['median_paired_ratio']
        w = chosen['workload']
        expectations = [r for r in frozen['resource_expectations'] if r['cell'] == cell]
        for r in expectations:
            assert r['full_array_bytes'] == (1 << w['logN']) * w['batch'] * (8 if w['operator'] == 'ntt' else 4)
            assert r['logical_data_path_bytes'] == 2 * r['logical_regions'] * r['full_array_bytes']
        rows.append(dict(cell=cell, workload=w, leaf=next(c['leaf'] for c in frozen['config']['cells'] if c['id'] == cell),
                         shape=[1 << w['logN']], resource_expectations_lower_bounds=expectations,
                         choices=chosen['choices'], attempted_unique=chosen['attempted_unique'],
                         accepted_unique=chosen['accepted_unique'], same_winner=chosen['same_winner'],
                         holdout_pairs=pairs, median_paired_ratio=original['median_paired_ratio'],
                         conventional_median_ms=statistics.median(p['conventional_ms'] for p in pairs),
                         extended_median_ms=statistics.median(p['extended_ms'] for p in pairs),
                         cost_prediction_ms=None))
    return dict(schema='prospective-resource-constrained-measured-selection-v1', source_root=str(base),
                status='complete', frozen_sha256=freeze_sha, winners_sha256=winner_sha,
                identity=frozen['identity'], protocol=holds[0]['protocol'], timing_boundary=frozen['timing_boundary'],
                chronology=dict(basis='hash-bound frozen runner prepares freeze before screen; screen selects winners before hash-bound independent holdouts; guard completion receipts ordered',
                                guard_completion_receipts=receipts), budget=frozen['budget'],
                numerical_policy=frozen['numerical_policy'], rows=rows,
                interpretation='prospective resource expectations -> bounded accepted pool -> measured screen choice -> three independent holdouts; no numeric cost-model forecast',
                limitations=['Frozen resource expectations are lower bounds, not full compiler legality or measured HBM/residency.',
                             'NTT dt1 is retained as both winners: ratios equal one by identical mapping, not two independent timing arms.',
                             'NTT radix4 dt1 has a specialized wrapper; same leaf does not isolate pure traversal effects.',
                             'NTT stage-service cost prediction remains uncovered; Nov3 three rows remain FFT/FWHT only.'])


def export(root=ROOT):
    root = Path(root)
    sys.path.insert(0, str(root / 'scripts'))
    import stage_cost_model
    sources = []
    def read(relative):
        path = root / relative
        data = path.read_bytes()
        sources.append({'path': str(relative), 'sha256': hashlib.sha256(data).hexdigest()})
        return json.loads(data)
    summary = read(COHORT / 'stage/final_scope/summary.json')
    service = read(COHORT / 'stage/final_scope/service_profile.json')
    model = read(COHORT / 'refit/cost_model.json')
    results = read(COHORT / 'tuning/operator_result.json')
    descriptors = read(COHORT / 'tuning/plan_descriptors.json')
    auto = read(COHORT / 'public/auto/selections.json')
    for path in [COHORT / 'run.py', Path('paper/experiments/hardware_characterization_closure_20261027.md'), Path('paper/evidence/NARRATIVE_SCALING_RESULTS_20261027.md'), Path('scripts/stage_cost_model.py')]:
        sources.append({'path': str(path), 'sha256': hashlib.sha256((root / path).read_bytes()).hexdigest()})
    checks = model['whole_plan_validation']['checks']
    for c in checks:
        assert math.isclose(relative_error(c['predicted_kernel_ms'], c['observed_kernel_ms']), c['relative_error'], rel_tol=1e-12, abs_tol=1e-12)
    assert len(checks) == 15
    stage_checks = model['validation']['stage_service']['checks']
    assert len(stage_checks) == 68
    for c in stage_checks:
        assert math.isclose(relative_error(c['predicted_kernel_ms'], c['observed_kernel_ms']), c['relative_error'], rel_tol=1e-12, abs_tol=1e-12)
    assert len(service['curves']) == summary['service_keys']['total'] == 34
    byname = {c['name']: c for c in results['candidates']}
    bycheck = {c['name']: c for c in checks}
    rows = []
    for winner in model['validation']['holdout_workload_winner_checks']:
        name = winner['measured_winner']
        c = byname[name]
        cfg = c['configuration']
        sample = c['samples'][0]
        times = [float(s['kernel_ms']) for s in c['samples']]
        assert len(times) == 3 and statistics.median(times) == c['median_kernel_ms']
        assert c['median_kernel_ms'] == bycheck[name]['observed_kernel_ms']
        prediction = stage_cost_model.predict(sample, model, True)
        without_fallback_parameters = dict(model, parameters={})
        invariant = stage_cost_model.predict(sample, without_fallback_parameters, True)
        assert prediction['whole_plan_fallback_used'] is False
        assert prediction['kernel_ms'] == invariant['kernel_ms'] == bycheck[name]['predicted_kernel_ms']
        groups = descriptors['descriptors'][name]['groups']
        projected = dependency_demands(cfg['logN'], cfg['batch'], 16 if cfg['operator'] == 'fft' else 4, groups)
        service_links = []
        for g in groups:
            matches = []
            for key, curve in service['curves'].items():
                parsed = json.loads(key)
                sem, shape = parsed['semantic'], parsed['group_shape']
                if (sem['operator'] == cfg['operator'] and sem['precision'] == cfg['precision']
                    and parsed['address_geometry']['logN'] == cfg['logN']
                    and all(shape[k] == g[k] for k in ['core', 'threads', 'first_stage', 'stage_count', 'local_stage_partition'])
                    and all(parsed['resources'][k] == g[k] for k in ['compiler_registers_per_thread', 'live_shared_bytes'])):
                    matches.append({'key': key, 'k_hull': curve['k_hull'],
                                    'training': curve['train_points'], 'validation': curve['validation_points']})
            assert len(matches) == 1, 'exact compiled-resource service link required'
            for curve in matches:
                for point in curve['training']:
                    if point.get('trial_kernel_ms'):
                        assert math.isclose(statistics.median(point['trial_kernel_ms']), point['kernel_ms'], rel_tol=1e-7, abs_tol=1e-10)
            service_links.append({'group_index': g['index'], 'matching_curves': matches,
                                  'status': 'matched' if matches else 'missing'})
        public = next(a for a in auto if a['workload']['operator'] == cfg['operator'] and a['workload']['logN'] == cfg['logN'])
        rows.append({'request': {k: cfg[k] for k in ['operator', 'precision', 'logN', 'batch', 'direction', 'normalization', 'placement']},
                     'shape': [1 << cfg['logN']], 'input_output_order': 'natural', 'accumulation': 'native',
                     'measured_configuration': cfg, 'measured_name': name,
                     'serialized_finalist_mapping': json.loads(cfg['mapping_json']),
                     'frozen_confirmation_command': c['command'] + ['--warmup', '30', '--repeat', '50', '--csv'],
                     'replay_identity_requirement': 'Verify original binary/library/JIT identity before replay; substitution is a new-build confirmation and must retain provenance.',
                     'dependency_derived': projected, 'compiled_and_launch_resources': groups,
                     'known_capacity_checks': [{'group_index': g['index'],
                         'register_slots_unrounded': g['threads'] * g['compiler_registers_per_thread'],
                         'register_slots_per_block_limit': model['profile']['hardware']['registers_per_block'],
                         'shared_bytes': g['live_shared_bytes'],
                         'shared_bytes_per_block_limit': model['profile']['hardware']['shared_bytes_per_block'],
                         'interpretation': 'necessary unrounded capacity arithmetic, not occupancy or complete lowering legality proof'} for g in groups],
                     'independent_service_curves': service_links,
                     'retrospective_model_validation': bycheck[name], 'winner_check': winner,
                     'prediction_parameter_audit': {'whole_plan_fallback_used': prediction['whole_plan_fallback_used'],
                         'prediction_with_empty_fallback_parameters_ms': invariant['kernel_ms'],
                         'unchanged_without_plan_fitted_parameters': True,
                         'interpretation': 'Plan totals fit fallback coefficients; fully covered predictions use independent fixed service curves and ignore those coefficients.'},
                     'prediction_temporality': 'Final refit validation is retrospective; no execution-before-confirmation prediction receipt certified by this export.',
                     'confirmation': {'samples_ms': times, 'median_ms': statistics.median(times), 'trials': 3,
                                      'warmup': int(sample['warmup']), 'repeat': int(sample['repeat']),
                                      'timing': 'direct-launch CUDA events around host launch loops; not Graph',
                                      'correct': c['correct']},
                     'public_auto_mapping': public['mapping'],
                     'selection_interpretation': 'independently measured registry replay; not proof that this refit predicted the choice before measurement'})
    prospective = prospective_extension(root, sources)
    return {'schema': 'resource-decisions-audit-v1', 'sources': sources,
            'measurement_identity': descriptors['identity'], 'model_protocol': model['protocol'],
            'platform_capacity': model['profile']['hardware'],
            'prior_hardware_rates': {'capabilities': model['profile']['capabilities'],
                                    'conditions': model['profile']['conditions'],
                                    'provenance': model['profile'].get('provenance'),
                                    'interpretation': 'historical hardware-rate priors retain original provenance; not newly measured service samples'},
            'cohort': {'arithmetic_families': ['fft', 'fwht'], 'request_cells': 3,
                       'certified_keys': 34, 'clean_probes': summary['clean_timed_probes'],
                       'independent_group_holdouts': len(stage_checks), 'complete_plan_holdouts': len(checks),
                       'model_status': model['status'],
                       'group_median_relative_error_recomputed': statistics.median(c['relative_error'] for c in stage_checks),
                       'plan_median_relative_error_recomputed': statistics.median(c['relative_error'] for c in checks),
                       'winner_matches': sum(w['match'] for w in model['validation']['holdout_workload_winner_checks']),
                       'temporal_scope': 'retrospectively evaluated independent-service predictions plus independent measurements; no prior execution receipt certified',
                       'fit_scope': 'operator totals fit fallback parameters; covered predictions do not consume them; held workload is excluded from fallback refit' },
            'rows': rows,
            'prospective_measured_selection': prospective,
            'extensions': [{'operator': 'ntt', 'precision': 'word64', 'shape': [1048576], 'batch': 16,
                            'modulus': 576460756061519873, 'core': 'shared scalar radix2 / Shoup (historical M1 contract)',
                            'status': 'missing-frozen-resource-selection-prediction-chain',
                            'reason': 'NTT is absent from the Nov3 three-cell calibration cohort; historical M1 controls do not establish prospective selection. A separately frozen future receipt may be appended without rewriting these rows.'}],
            'limitations': ['No global optimum or search ranking competition is claimed.',
                            'Group logical bytes and closed regions are not HBM bytes or resident CTA counts.',
                            'Compiler allocations and registers are distinct from dependency footprints; unknown resource fields stay unknown.',
                            'Finite exact service-key/range and identity reuse only; overlap, Graph prediction, arbitrary new sizes and cross-platform generalization unvalidated.',
                            'Three cell winners are FFT plus two FWHT requests, not three butterfly arithmetic families.']}


def render(data):
    lines = [r'\begingroup', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Request ($N/B$) & G & SMEM$^{\max}$ & Retro. & Confirm\\\midrule']
    for row in data['rows']:
        w = row['request']; groups = row['compiled_and_launch_resources']
        smem = max(g['live_shared_bytes'] for g in groups) / 1024
        label = f"{w['operator'].upper()}{w['precision'][2:]} $2^{{{w['logN']}}}/{w['batch']}$"
        lines.append(f"{label} & {len(groups)} & {smem:g} & {row['retrospective_model_validation']['predicted_kernel_ms']:.4f} & {row['confirmation']['median_ms']:.4f}\\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\endgroup',
                  r'% Times in ms; SMEM max in KiB; G counts ordered stage bands, not compute groups.',
                  r'% * Retrospective service-composition check, not a certified prediction frozen before confirmation.',
                  r'% SMEM is maximum observed kernel live shared allocation, not the dependency-region value footprint.',
                  r'% NTT chain is absent from this cohort; see resource_decisions_20261107.json.'])
    return '\n'.join(lines) + '\n'


def render_prospective(data):
    lines = [r'\begingroup', r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
             r'Leaf & Conv. $\mu$s & Ext. $\mu$s & Ratio\\\midrule']
    labels = {'fwht-radix2': 'FWHT r2', 'fwht-radix4': 'FWHT r4',
              'ntt-homogeneous-radix4': 'NTT shared-r4', 'ntt-homogeneous-warp-radix2': 'NTT warp-r2'}
    for row in data['prospective_measured_selection']['rows']:
        label = labels[row['cell']]
        extended = 'same' if row['same_winner'] else f"{1000 * row['extended_median_ms']:.3f}"
        lines.append(f"{label} & {1000 * row['conventional_median_ms']:.3f} & {extended} & {row['median_paired_ratio']:.3f}\\\\")
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\endgroup',
                  r'% Ratio is median of process-paired ratios, not ratio of median times.',
                  r'% N/B and selected mappings are described in the text; FWHT FP32 N=2^16 B16, NTT word64 N=2^20 B16.',
                  r'% Measured selection from a frozen resource-constrained pool; no numeric cost forecast.',
                  r'% NTT same means the identical dt1 mapping and sample are reused: ratio 1 by identity.'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'paper/generated/resource_decisions_20261107.json')
    args = parser.parse_args()
    data = export(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    args.output.with_suffix('.tex').write_text(render(data))
    args.output.with_name('resource_measured_selection_20261107.tex').write_text(render_prospective(data))


if __name__ == '__main__':
    main()
