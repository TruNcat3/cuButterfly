#!/usr/bin/env python3
"""Freeze four clean QuaRot fused processes using stored identities, without CUDA."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('quarot_followup_guard', ROOT / 'scripts/run_quarot_fused_followup.py')
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)
DEFAULT_SOURCE = ROOT / 'results/quarot_fused_followup_20261107'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def source_name(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_reports(value):
    """Require reported finite outputs, without enforcing historical error thresholds."""
    require(isinstance(value, dict) and bool(value), 'Missing numerical reports')
    flags = []
    def visit(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key == 'finite':
                    flags.append(child)
                elif key in ('actual_nonfinite_count', 'reference_nonfinite_count'):
                    require(child == 0, 'Nonfinite numerical output')
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    require(flags and all(flag is True for flag in flags), 'Nonfinite or unreported outputs')


def export(source=DEFAULT_SOURCE):
    source = Path(source)
    config = read(ROOT / 'configs/experiments/quarot_fused_followup_20261107.json')
    rows, shared = [], None
    for job in config['jobs']:
        directory = source / job['id']
        paths = {name: directory / name for name in ('identity.json', 'status.json', 'release/report.json')}
        require(all(path.is_file() for path in paths.values()), 'Incomplete four-process cohort: ' + job['id'])
        identity = read(paths['identity.json'])
        require(identity.get('job') == job and identity.get('contract') == config['contract'], 'Stored request contract mismatch')
        require(GUARD.release_matches(directory, identity), 'Missing clean release')
        common = {key: value for key, value in identity.items() if key != 'job'}
        require(common.get('gpu_uuid') and common.get('library_sha256'), 'Missing shared GPU/build identity')
        if shared is None:
            shared = common
        require(common == shared, 'Cross-process stored identity mismatch')
        report = read(paths['release/report.json'])
        timing = report['timing_contract']
        require(report.get('full_model_run') is True and timing.get('graph_captured_full_prefill_decode') is True
                and timing.get('input_copy_and_output_clone_in_timing') is True
                and timing.get('finite_output_gate_before_timing') is True
                and timing.get('summary_statistic') == 'median'
                and timing.get('summary_clock') == 'host_wall_with_device_synchronization'
                and timing.get('precision_contract') == config['contract']['precision_contract'], 'Incomplete Graph/timing contract')
        errors = {name: report[name] for name in ('numerical_checks', 'graph_changed_token_validation')}
        expected_outputs = {'prefill'} | {'decode_' + str(i) for i in range(8)}
        require(expected_outputs <= set(errors['numerical_checks']), 'Missing sequence output errors')
        for backend in ('baseline', 'cubutterfly'):
            checks = errors['graph_changed_token_validation'][backend]
            for phase in ('original', 'changed', 'restored_repeat', 'output_lifetime'):
                require(expected_outputs <= set(checks[phase]), 'Missing Graph lifecycle output errors')
        for value in errors.values():
            finite_reports(value)
        samples, clocks = {}, {}
        for clock in ('wall_ms', 'cuda_event_ms'):
            samples[clock], clocks[clock] = {}, {}
            for backend in ('baseline', 'cubutterfly'):
                values = report[backend][clock]['end_to_end']
                require(len(values) == 7 and all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in values), 'Expected seven finite positive samples')
                samples[clock][backend] = values
                clocks[clock][backend + '_median_ms'] = median(values)
            clocks[clock]['baseline_over_cubutterfly'] = clocks[clock]['baseline_median_ms'] / clocks[clock]['cubutterfly_median_ms']
        summary = report['summary']['end_to_end']
        for key, value in clocks['wall_ms'].items():
            require(math.isclose(summary[key], value, rel_tol=1e-12), 'Reported summary differs from wall sample medians')
        setup = report['adapter_setup']
        plans = [{key: plan[key] for key in ('algorithm', 'selection_source', 'selection_reason', 'mapping') if key in plan}
                 for plan in setup['plans']]
        rows.append(dict(job_id=job['id'], cohort='quarot-fused-followup-20261107', batch=job['batch_size'],
            variant_order=job['variant_order'], payload=report['payload'], seed=report['seed'],
            gpu_uuid=report['gpu_uuid'], gpu_name=report['gpu_name'], library_sha256=report['library_sha256'],
            source=source_name(paths['release/report.json']),
            source_sha256=sha(paths['release/report.json']),
            source_artifacts={name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
            stored_identity=identity, timing_contract=timing, samples=samples, clocks=clocks,
            baseline_ms=clocks['wall_ms']['baseline_median_ms'], own_ms=clocks['wall_ms']['cubutterfly_median_ms'],
            speedup=clocks['wall_ms']['baseline_over_cubutterfly'], numerical_errors=errors,
            plan_summary=dict(compute_contract=setup['compute_contract'], reuse_buffers=setup['reuse_buffers'], plans=plans)))
    ranges = []
    for batch in (1, 4):
        members = [row for row in rows if row['batch'] == batch]
        ranges.append(dict(batch=batch, processes=len(members), wall_ratio_range=[min(row['speedup'] for row in members), max(row['speedup'] for row in members)],
            event_ratio_range=[min(row['clocks']['cuda_event_ms']['baseline_over_cubutterfly'] for row in members), max(row['clocks']['cuda_event_ms']['baseline_over_cubutterfly'] for row in members)]))
    return dict(schema='quarot-fused-full-model-followup-v1', records=rows, aggregate_ranges=ranges,
        aggregation='Each process uses baseline/own sample wall medians; report both process ratios as a range per batch. Event medians are separate. No pooling with old cohorts or best-process selection.',
        numeric_policy='finite outputs required; error distributions reported without an accuracy threshold',
        identity_policy='Stored identity and clean release validated; current source hashes are not substituted for historical identities.')


def tex_rows(data):
    return '\n'.join(f"QuaRot fused, B{row['batch']} & FWHT & QuaRot/Dao & "
        f"{row['wall_ratio_range'][0]:.3f}--{row['wall_ratio_range'][1]:.3f}" + r"\\"
        for row in data['aggregate_ranges']) + '\n'


def number_macros(data):
    lines = []
    for row in data['aggregate_ranges']:
        low, high = (f"{value:.3f}" for value in row['wall_ratio_range'])
        value = low if low == high else low + "--" + high
        name = "CurrentQuaRotFusedBOne" if row['batch'] == 1 else "CurrentQuaRotFusedBFour"
        lines.append("\\newcommand{\\" + name + "}{" + value + "}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/snapshots/current_refresh/quarot_fused_e2e.json')
    args = parser.parse_args(argv)
    data = export(args.source)  # Fail closed before writing either formal artifact.
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    (args.output.parent / 'quarot_fused_rows.tex').write_text(tex_rows(data))
    (args.output.parent / 'quarot_fused_numbers.tex').write_text(number_macros(data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
