"""CPU preparation and snapshot verification for the frozen semantic runner.

This module never imports Torch or owns an executor. Callers must capture full
host byte buffers before resetting device state and await verification before
publishing a terminal measurement.
"""
from __future__ import annotations

import copy
import math

import numpy as np

import run_semantic_campaign as runner


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def prepare_case(case, reference_max_elements=1048576, ntt_max_operations=16777216):
    """Build the identical quantized input and complete bounded CPU oracle."""
    s = copy.deepcopy(case['descriptor']['semantics'])
    capabilities = case.get('capabilities')
    if capabilities is not None:
        status = capabilities.get('status')
        if status == 'missing-lowering':
            raise runner.Gap('missing-lowering', 'public capability has no executable lowering')
        if status != 'eligible':
            raise runner.Gap('runner-gap', 'unrecognized public capability status')
    points = s['batch'] * max(_product(runner.side_extents(s, side))
                             for side in ('input', 'output'))
    if points > reference_max_elements:
        raise runner.Gap('validation-unavailable',
                         f'all-elements bound exceeded: {points}>{reference_max_elements}')
    source = runner.make_source(s)
    expected = runner.reference(s, source, reference_max_elements, ntt_max_operations)
    return {'case_id': case['id'], 's': s, 'source': source, 'expected': expected,
            'encoded_source': runner.encode_storage(source, runner.side_storage(s, 'input')),
            'input_indices': runner.layout_indices(s, 'input'),
            'output_indices': runner.layout_indices(s, 'output')}


def prepared_physical_bytes(prepared, side, logical, size):
    """Pack plan-sized bytes using the prepared layout and exact storage words."""
    s = prepared['s']
    dtype = runner.side_dtype(s, side)
    if size % dtype.itemsize:
        raise AssertionError('plan buffer byte size does not align with side dtype')
    indices = prepared[side + '_indices']
    logical = np.asarray(logical)
    if indices.size != logical.size or int(indices.max(initial=0)) >= size // dtype.itemsize:
        raise AssertionError('public side layout exceeds plan size')
    # Use cached input words only for the source whose quantization was prepared.
    encoded = (prepared['encoded_source'] if side == 'input' and logical is prepared['source']
               else runner.encode_storage(logical, runner.side_storage(s, side)))
    raw = np.full(size, 0xA5, dtype=np.uint8)
    raw.view(dtype)[indices] = encoded.reshape(-1)
    return raw


def capture_snapshot(**payload):
    """Own CPU buffers before submitting them to an asynchronous executor."""
    return copy.deepcopy(payload)


def validate_snapshot(payload):
    """Return explicit verification results, preserving artifacts on failure.

    Required: s, expected, and actual or output_raw. Full output_raw requires
    output_initial so padding is checked. If input_raw is supplied, input_initial
    is required and all bytes are checked. Artifact options are error_dir,
    case_id, backend; detailed defaults to True for first validation only.
    """
    report = {'ok': False}
    try:
        s = payload['s']
        storage = payload.get('storage', s['storage'])
        if storage != s['storage']:
            raise ValueError('snapshot storage differs from descriptor')
        verdict = payload.get('float_verdict', 'error-report')
        if verdict not in ('legacy-fixed', 'error-report'):
            raise ValueError('unknown floating verdict policy')
        detailed = payload.get('detailed', True)
        raw = payload.get('output_raw')
        if raw is not None:
            raw = np.asarray(raw)
            initial_output = np.asarray(payload['output_initial'])
            if raw.dtype != np.uint8 or initial_output.dtype != np.uint8 or raw.shape != initial_output.shape:
                raise AssertionError('output padding requires matching full uint8 byte buffers')
        actual = (runner.output_from_bytes(s, raw) if raw is not None
                  else np.asarray(payload['actual']))
        expected = payload['expected']
        if storage not in (8, 9) and (detailed or not np.all(np.isfinite(actual))):
            report['numeric_diagnostics'] = runner.floating_range_report(actual, expected, storage)
        if detailed and payload.get('error_dir') is not None:
            report['error_artifact'] = runner.retain_error_artifact(
                payload['error_dir'], payload['case_id'], payload['backend'], actual, expected)
        report['correctness'] = runner.check_full(actual, expected, storage, verdict, detailed)
        if raw is not None:
            runner.check_padding(s, raw, payload['output_initial'])
        if payload.get('input_raw') is not None:
            actual_input = np.asarray(payload['input_raw'])
            initial_input = np.asarray(payload['input_initial'])
            if actual_input.dtype != np.uint8 or initial_input.dtype != np.uint8:
                raise AssertionError('input preservation requires full uint8 byte buffers')
            if not np.array_equal(actual_input, initial_input):
                raise AssertionError('out-of-place input or padding modified')
        report.update(ok=True, status='accuracy-review-required'
                      if storage not in (8, 9) and verdict == 'error-report' else 'pass')
    except AssertionError as exc:
        report.update(status='correctness-failure', error=str(exc), exception_type=type(exc).__name__)
    except Exception as exc:
        report.update(status='runner-error', error=str(exc), exception_type=type(exc).__name__)
    return report
