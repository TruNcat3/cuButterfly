"""Add a stage-arithmetic review to saved FFT-L outputs without using a GPU.

Writes a separate report, preserving original results. Vendor output may be
compared to the explicit cuButterfly oracle only as a diagnostic, never as
proof of matching vendor arithmetic.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from low_fft_cpu_reference import transform


def axis_order(mapping, rank):
    if rank == 1:
        return [0]
    if isinstance(mapping, dict):
        # The fixed rank-two transpose lowering executes contiguous rows first.
        if rank == 2 and mapping.get('algorithm', '').startswith('rank2-transpose') and not mapping.get('nd_composite', False):
            return [1, 0]
        value = mapping.get('axis_order')
        if isinstance(value, list) and sorted(value) == list(range(rank)):
            return value
        for child in mapping.values():
            found = axis_order(child, rank)
            if found is not None:
                return found
    return None


def metrics(actual, expected):
    a, e = np.asarray(actual, np.complex128).reshape(-1), np.asarray(expected, np.complex128).reshape(-1)
    if a.shape != e.shape:
        raise ValueError('saved output shape mismatch')
    finite = np.isfinite(a) & np.isfinite(e)
    error = np.abs(a[finite] - e[finite])
    norm = np.linalg.norm(e[finite])
    return dict(elements=a.size, finite_pairs=int(finite.sum()),
        actual_nonfinite=int((~np.isfinite(a)).sum()), reference_nonfinite=int((~np.isfinite(e)).sum()),
        max_abs_finite=float(error.max(initial=0)),
        rms_abs_finite=float(np.sqrt(np.mean(error**2))) if error.size else None,
        l2_relative_finite=float(np.linalg.norm(error)/norm) if norm else None,
        exact_component_equal_elements=int(np.count_nonzero((a.real == e.real) & (a.imag == e.imag))),
        abs_error_quantiles_finite={str(q):float(np.quantile(error,q)) for q in [0,.25,.5,.75,.9,.95,.99,1]} if error.size else {},
        relative_error_quantiles_nonzero_reference={str(q):float(np.quantile(error[np.abs(e[finite]) != 0] / np.abs(e[finite][np.abs(e[finite]) != 0]),q)) for q in [0,.25,.5,.75,.9,.95,.99,1]} if np.any(np.abs(e[finite]) != 0) else {},
        acceptance_threshold=None, verdict='error-report-only')


def review(prepared_dir, report_path, error_dir=None):
    from benchmark import run_fft_l_closure as run
    manifest, prepared = run.load_prepared(prepared_dir)
    original = json.loads(Path(report_path).read_text())
    result = dict(schema='fft-l-stage-cpu-review-v1', original_report=str(Path(report_path).resolve()),
        original_report_sha256=run.semantic.sha256_file(report_path), manifest_sha256=manifest['manifest_sha256'],
        gpu_execution=False, acceptance_threshold=None, rows=[],
        arithmetic_contract=dict(native='RNE storage after each of four real products, complex add/sub, butterfly add/sub and normalization multiply',
            inverse_normalization='0.5 per stage, after safe butterfly (components <= max/4), otherwise before',
            twiddles='host libm double cos/sin -> float32 -> low storage; GPU uploaded coefficient bytes not retained',
            native_gpu_limit='BF16 output equality is observed; FP16 saved outputs differ from explicit unfused staged rounding. Offline SM80 probe of identical __hmul/__hsub intrinsics emits HFMA2. The original binary contraction schedule is not retained in this review; no strict FP16 GPU arithmetic match claimed',
            fp32_compute='explicit unfused float32 real products, float32 complex add/sub, then storage narrowing; butterfly add/sub and 0.5 multiply narrow to storage. Matches source storage boundaries; GPU may contract differently',
            vendor='internal arithmetic unspecified; any oracle comparison is diagnostic only',
            equivalence='source-level native arithmetic oracle; no claim of independently certified GPU instruction behavior'))
    for row in original['rows']:
        p = prepared[row['id']]
        entry = dict(case_id=row['id'], storage=p['s']['storage'], compute=p['s']['compute'])
        order = axis_order(row.get('mapping', {}), len(p['s']['shape']))
        try:
            expected = transform(p['source'], p['s'], order)
        except ValueError as exc:
            entry.update(status='reference-gap', reason=str(exc))
            result['rows'].append(entry)
            continue
        entry.update(status='reference-computed', reference_role='declared-unfused-fp32-compute-reference' if p['s']['compute'] == 1 else 'declared-unfused-native-precision-reference', axis_order=order, backends={})
        for name, observation in [('cubutterfly', row), ('cufft-xt-low-storage', row.get('low_storage_baseline', {}))]:
            artifact = observation.get('error_artifact')
            if not artifact:
                entry['backends'][name] = dict(status='saved-output-unavailable')
                continue
            path = Path(artifact['path'])
            if not path.exists():
                path = Path(report_path).parent / 'errors' / path.name
            if not path.exists():
                entry['backends'][name] = dict(status='saved-output-unavailable', path=str(path))
                continue
            if artifact.get('sha256') and run.semantic.sha256_file(path) != artifact['sha256']:
                raise ValueError('saved output artifact checksum mismatch')
            with np.load(path, allow_pickle=False) as arrays:
                actual = arrays['actual'].copy()
            entry['backends'][name] = dict(status='error-reported', metrics=metrics(actual,expected),
                strict_arithmetic_match=False if name != 'cubutterfly' else None,
                comparison='vendor arithmetic diagnostic' if name != 'cubutterfly' else 'source-level stage arithmetic')
            if error_dir is not None:
                entry['backends'][name]['error_artifact'] = run.semantic.retain_error_artifact(
                    error_dir, row['id'], name + '-declared-cpu', np.asarray(actual, np.complex128).reshape(-1), np.asarray(expected, np.complex128).reshape(-1))
        result['rows'].append(entry)
    result['summary'] = dict(rows=len(result['rows']),
        references_computed=sum(r['status']=='reference-computed' for r in result['rows']),
        output_comparisons=sum(b['status']=='error-reported' for r in result['rows'] for b in r.get('backends',{}).values()))
    result['future_gpu_cases'] = 'No rerun for same-precision error reporting when saved outputs and axis order are available. Missing outputs require only their missing original cases. Reconstructing the exact GPU contraction schedule is a separate instruction-equivalence requirement.'
    return result


def main():
    from benchmark import run_fft_l_closure as run
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be a new report; preserve earlier validation')
    run.write(args.output, review(args.prepared,args.report, args.output.parent / (args.output.stem + '_errors')))

if __name__=='__main__':
    main()
