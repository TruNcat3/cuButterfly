"""GPU execution steps; every yielded snapshot requires a CPU verdict."""
from __future__ import annotations

import ctypes as ct
import os
import re
import tempfile
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

import run_semantic_campaign as old
from semantic_worker_resources import WorkerRecycle


class PreparationPending(RuntimeError):
    def __init__(self, request_ids):
        self.request_ids = sorted(set(request_ids))
        super().__init__('CPU compilation pending: ' + ', '.join(self.request_ids))


class PreparationFailed(RuntimeError):
    def __init__(self, details):
        self.details = details
        super().__init__('CPU compilation failed: ' + str(details))


_CUFFTDX_UNSUPPORTED = 'static assertion failed with "This FFT configuration is not supported"'
_CUFFTDX_DERIVED_ERRORS = {
    'name followed by "::" must be a class or namespace name',
    'expected a declaration',
}


def _candidate_rejection(record):
    """Recognize only the known cuFFTDX unsupported-candidate diagnostics.

    The broker keeps the compiler log outside the event record. Requiring an
    existing log path prevents an arbitrary event error string from being
    treated as a harmless candidate rejection.
    """
    if not (record.get('kind') == 'compile_failed' or
            record.get('kind') is None and record.get('status') == 'failed'):
        return False
    paths = [record.get(key) for key in ('compiler_log', 'compiler_log_path', 'log_path')]
    # The compile broker's real event shape stores only a bounded reference in
    # error, for example: "specialization compilation failed; see /tmp/.../compiler.log".
    # Parse only that explicit suffix and require the exact compiler.log name.
    error = record.get('error')
    if isinstance(error, str):
        match = re.search(r"\bsee\s+([^\s'\"]*/compiler\.log)(?:\b|$)", error)
        if match:
            paths.append(match.group(1))
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if path.name != 'compiler.log' or not path.is_file():
            continue
        try:
            text = path.read_text(errors='replace')
        except OSError:
            continue
        errors = [line for line in text.splitlines() if re.search(r'\berror(?:\s+#\d+)?\s*:', line)]
        assertion = False
        for line in errors:
            diagnostic = re.fullmatch(r'(.+?)\(\d+\): error: (.+)', line)
            if diagnostic is None:
                break
            source, message = diagnostic.groups()
            if '/cufftdx/include/' not in source:
                break
            if message == _CUFFTDX_UNSUPPORTED and source.endswith('check_and_get_trait.hpp'):
                assertion = True
            elif (message not in _CUFFTDX_DERIVED_ERRORS or
                  not source.endswith(('check_and_get_trait.hpp', 'fft_execution.hpp'))):
                break
        else:
            if assertion:
                return True
    return False


def _transient_compile_failure(record):
    """Recognize a host compiler process interrupted by resource control.

    ``nvcc: Terminated`` is emitted when a producer is stopped while a
    specialization is in flight. It is neither a valid candidate rejection
    nor a source/template error: the broker must retry it before the plan can
    be judged. Keep this narrow so genuine compiler diagnostics remain hard
    failures.
    """
    if record.get('kind') not in (None, 'compile_failed', 'broker_error', 'broker_runtime_error') and \
            record.get('status') != 'failed':
        return False
    text = ' '.join(str(record.get(key, '')) for key in
                    ('error', 'reason', 'compiler_log', 'compiler_log_path', 'log_path'))
    if 'nvcc: terminated' in text.lower() or 'compilation terminated' in text.lower():
        return True
    paths = [record.get(key) for key in ('compiler_log', 'compiler_log_path', 'log_path')]
    error = record.get('error')
    if isinstance(error, str):
        match = re.search(r"\bsee\s+([^\s'\"]+/compiler\.log)(?:\b|$)", error)
        if match:
            paths.append(match.group(1))
    for value in paths:
        if not value:
            continue
        path = Path(value)
        if path.name != 'compiler.log' or not path.is_file():
            continue
        try:
            log = path.read_text(errors='replace').lower()
        except OSError:
            continue
        if 'nvcc: terminated' in log or 'compilation terminated' in log:
            return True
    return False


def check_preparation_events(events_path, captured_stderr=''):
    """A single missing candidate invalidates the entire selection attempt."""
    records = ([] if events_path is None or not Path(events_path).exists() else
               [json.loads(line) for line in Path(events_path).read_text().splitlines() if line.strip()])
    candidate_rejections = [record for record in records if _candidate_rejection(record)]
    transient = [record for record in records if
                 record not in candidate_rejections and _transient_compile_failure(record)]
    failed = [record for record in records if record not in candidate_rejections and
              record not in transient and
              (record.get('kind') in ('compile_failed', 'broker_error', 'broker_runtime_error') or
               record.get('status') in ('failed', 'error'))]
    if failed:
        raise PreparationFailed(failed)
    if transient:
        raise PreparationPending([str(record['request_id']) for record in transient
                                  if record.get('request_id')])
    resources = [record for record in records if record.get('kind') == 'worker_resource_limit']
    if resources:
        raise WorkerRecycle(resources[-1]['reason'], resources[-1].get('resources'))
    if any(message in captured_stderr for message in
           ('could not start specialization compiler', 'cannot load generic specialization:',
            'cannot load specialization:', 'semantic compile client:', 'semantic compile service:')):
        raise RuntimeError('unrecorded specialization resource failure; investigate captured stderr')
    missing = [record for record in records if record.get('kind') in ('compile_pending', 'compile_running') or
               record.get('status') in ('missing', 'pending', 'prepare-pending')]
    if missing:
        raise PreparationPending([str(record['request_id']) for record in missing])


def build_plan(cb, library, descriptor, stream, events_path):
    plan = None
    captured = ''
    try:
        # C++ candidate rejection goes to fd 2, outside Python's stderr object.
        # A file avoids pipe backpressure during large candidate searches.
        with tempfile.TemporaryFile() as stderr:
            original = os.dup(2)
            try:
                ct.CDLL(None).fflush(None)
                os.dup2(stderr.fileno(), 2)
                with cb.Descriptor.from_json(library, descriptor) as imported:
                    plan = cb.Plan(library, imported, stream=stream.cuda_stream)
            finally:
                ct.CDLL(None).fflush(None)
                os.dup2(original, 2)
                os.close(original)
                stderr.seek(0)
                data = stderr.read()
                captured = data.decode(errors='replace')
                offset = 0
                while offset < len(data):
                    offset += os.write(2, data[offset:offset + 65536])
    finally:
        try:
            check_preparation_events(events_path, captured)
        except BaseException:
            if plan is not None:
                plan.close()
            raise
    return plan


def merge_validation(row, result, key=None):
    if not isinstance(result, dict) or not isinstance(result.get('ok'), bool):
        raise RuntimeError('missing or malformed CPU verification result')
    for field in ('numeric_diagnostics', 'error_artifact'):
        if result.get(field) is not None:
            row[field] = result[field]
    if not result['ok']:
        status = result.get('status', 'correctness-failure')
        if status == 'correctness-failure':
            raise AssertionError(result.get('error') or 'CPU verification failed')
        raise old.Gap(status, result.get('error') or 'CPU verification failed')
    if result.get('status') not in ('pass', 'accuracy-review-required', None):
        raise RuntimeError('nonterminal CPU verification status: ' + str(result['status']))
    if key is not None:
        row[key] = result['correctness']


def snapshot_steps(case, prepared, torch, stream, gpu_input, gpu_output,
                   input_initial, output_initial, backend, row, options, key=None, phase='initial'):
    from semantic_cpu_pipeline import capture_snapshot
    stream.synchronize()
    # CPU copies own their storage before the next reset, poison, or execution.
    raw_output = gpu_output.view(torch.uint8).cpu().numpy().copy()
    raw_input = gpu_input.view(torch.uint8).cpu().numpy().copy()
    payload = capture_snapshot(s=prepared['s'], expected=prepared['expected'],
        case_id=case['id'], backend=backend, output_raw=raw_output,
        input_raw=None if prepared['s']['placement'] else raw_input,
        input_initial=input_initial, output_initial=output_initial, detailed='correctness' not in row,
        float_verdict=getattr(options, 'float_verdict', 'error-report'),
        error_dir=getattr(options, 'error_dir', None), phase=phase)
    result = yield payload
    merge_validation(row, result, key)


def timing_steps(torch, stream, execute, reset, in_place, verify, poison_output=None):
    for _ in range(10):
        if in_place:
            reset()
        execute()
    stream.synchronize()
    samples = []
    graph = None
    if not in_place:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(20):
                execute()
    validations, snapshots = [], []
    try:
        # Preserve the frozen runner's contiguous backend sample group. Capture
        # each full buffer before reset, but yield only after all seven calls.
        for sample in range(7):
            if in_place:
                reset()
            elif poison_output is not None:
                poison_output()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            execute() if in_place else graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end) / (1 if in_place else 20))
            validation = verify('timing-' + str(sample))
            validations.append(validation)
            snapshots.append(next(validation))
        batch = yield {'batch_snapshots': snapshots}
        if (not isinstance(batch, dict) or not isinstance(batch.get('batch_results'), list)
                or len(batch['batch_results']) != len(validations)):
            raise RuntimeError('missing or malformed CPU timing verification batch')
        for validation, result in zip(validations, batch['batch_results']):
            try:
                validation.send(result)
            except StopIteration:
                pass
            else:
                raise RuntimeError('CPU timing verification did not complete')
    finally:
        for validation in validations:
            validation.close()
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'protocol': ('7 single event calls; reset enqueued before start event; 10 warmup' if in_place else
                         '20 complete calls per CUDA graph; 7 replays; 10 warmup')}


def baseline_steps(case, prepared, torch, stream, input_saved, input_initial,
                   output_initial, options):
    s = prepared['s']
    results = []
    for name in ('cufft', 'vkfft', 'daofht', 'gpu-ntt'):
        native = None
        row = {'name': name, 'status': 'missing-baseline'}
        try:
            contract = old.baseline_contract(name, s, getattr(options, 'vkfft_library', None),
                                            getattr(options, 'gpuntt_library', None))
            row.update(support_status=contract['status'], support_metadata=contract.get('metadata', {}))
            if not contract['supported']:
                raise old.Gap('not-applicable' if contract['status'] == 'not-applicable' else 'missing-baseline', contract['reason'])
            dtype_name = {0: 'float16', 1: 'bfloat16', 2: 'float32', 6: 'complex64',
                          7: 'complex128', 8: 'uint32', 9: 'uint64'}[s['storage']]
            dtype = getattr(torch, dtype_name, None)
            if dtype is None:
                row['support_status'] = 'missing-exact-storage-binding'
                raise old.Gap('missing-baseline', 'Torch lacks exact requested storage dtype ' + dtype_name)
            shape = (s['batch'], *s['shape'])
            src = torch.empty(shape, dtype=dtype, device='cuda')
            out = torch.empty_like(src)
            physical_input = input_saved.clone().view(dtype)
            physical_saved = physical_input.view(torch.uint8).clone()
            physical_output = physical_input if s['placement'] else torch.from_numpy(output_initial).to('cuda').view(dtype)
            in_indices, out_indices = prepared['input_indices'], prepared['output_indices']
            in_index = torch.from_numpy(in_indices).to('cuda')
            out_index = torch.from_numpy(out_indices).to('cuda')
            contiguous = np.arange(math.prod(shape))
            gather = not np.array_equal(in_indices, contiguous)
            scatter = not np.array_equal(out_indices, contiguous) or bool(s['placement'])
            native_source = src if gather else physical_input[:math.prod(shape)].reshape(shape)
            native_output = out if scatter else physical_output[:math.prod(shape)].reshape(shape)
            axis_composed = name in ('cufft', 'vkfft') and len(s['shape']) > 3
            if axis_composed:
                from semantic_fft_baseline import AxisComposition
                native = AxisComposition(name, torch, s, native_source, native_output, stream,
                                         getattr(options, 'vkfft_library', None))
                vendor_scale = math.prod(s['shape']) if s['inverse'] and not s['normalize_inverse'] else 1
                row['metadata'] = native.metadata()
            elif name == 'cufft':
                from diagnose_xd_fft import NativeCufft
                native = NativeCufft(torch, s['shape'], s['batch'], s['storage'] == 7,
                                     bool(s['inverse']), stream, native_source, native_output)
                vendor_scale = 1 / math.prod(s['shape']) if s['inverse'] and s['normalize_inverse'] else 1
            elif name == 'vkfft':
                from vkfft_xd import NativeVkfft
                native = NativeVkfft(torch, s['shape'], s['batch'], s['storage'] == 7,
                                     bool(s['inverse']), stream, native_source, native_output,
                                     getattr(options, 'vkfft_library', None))
                vendor_scale = math.prod(s['shape']) if s['inverse'] and not s['normalize_inverse'] else 1
                row['metadata'] = native.metadata()
            elif name == 'daofht':
                from semantic_dao_baseline import build_core
                native = build_core(torch, s, native_source, native_output)
                vendor_scale = 1
                row['metadata'] = native.metadata()
            else:
                from semantic_gpuntt_baseline import DEFAULT_LIBRARY, build_core
                native = build_core(torch, s, native_source, native_output, stream,
                    librarypath=getattr(options, 'gpuntt_library', None) or DEFAULT_LIBRARY)
                vendor_scale = 1
                row['metadata'] = native.metadata()
                if row['metadata'].get('caller_stream') != int(stream.cuda_stream):
                    raise AssertionError('GPU-NTT core did not bind the supplied CUDA stream')
            factor = vendor_scale * s['fft_output_scale']
            integer_transfer = s['storage'] in (8, 9)
            transfer_input, transfer_source, transfer_output, transfer_result = old.baseline_transfer_views(
                torch, s, physical_input, src, physical_output, out)
            operations = []
            if gather:
                operations.append('GPU byte-row index_select logical input gather preserving unsigned words' if integer_transfer else 'GPU index_select logical input gather including order')
            operations.extend(row.get('metadata', {}).get('included_operations', []))
            if factor != 1:
                operations.append(f'GPU multiply output by {factor!r}')
            if scatter:
                operations.append('GPU byte-row index_copy logical output scatter preserving unsigned words' if integer_transfer else 'GPU index_copy logical output scatter including order')
            row.update(category='adapter' if operations or s['placement'] else 'native', adapter_operations=operations,
                       placement_contract='separate scratch output permits public in-place alias' if s['placement'] else 'out-of-place')
            if axis_composed:
                row['category'] = 'external-axis-composition'
                row['metadata']['included_operations'] = operations

            def execute():
                if gather:
                    torch.index_select(transfer_input, 0, in_index, out=transfer_source)
                if name == 'cufft' and not axis_composed:
                    native.check(native.exec_fn(native.handle, ct.c_void_p(native_source.data_ptr()),
                        ct.c_void_p(native_output.data_ptr()), 1 if s['inverse'] else -1))
                else:
                    native()
                if factor != 1:
                    native_output.mul_(factor)
                if scatter:
                    transfer_output.index_copy_(0, out_index, transfer_result)

            def verify(phase='initial', key=None):
                return snapshot_steps(case, prepared, torch, stream, physical_input, physical_output,
                                      input_initial, output_initial, name, row, options, key, phase)

            execute()
            yield from verify(key='correctness')
            if getattr(options, 'timing', False):
                row['timing'] = yield from timing_steps(torch, stream, execute,
                    lambda: physical_input.view(torch.uint8).copy_(physical_saved), bool(s['placement']), verify,
                    lambda: physical_output.view(torch.uint8).fill_(0xA5))
            row['status'] = terminal_status(s, options)
        except old.Gap as exc:
            row.update(status=exc.status, reason=str(exc))
        except AssertionError as exc:
            row.update(status='correctness-failure', error=str(exc))
            results.append(row)
            raise old.BaselineCorrectnessFailure(str(exc), results) from exc
        except torch.OutOfMemoryError as exc:
            row.update(status='resource-limit', reason=str(exc))
        except Exception as exc:
            row.update(status='baseline-error', reason=str(exc))
        finally:
            if native is not None:
                native.close()
        results.append(row)
    return results


def terminal_status(s, options):
    return ('accuracy-review-required' if s['storage'] not in (8, 9) and
            getattr(options, 'float_verdict', 'error-report') == 'error-report' else 'pass')


def execution_steps(case, prepared, library, torch, stream, options):
    """Yield immutable validation payloads, accepting verdicts via ``send``."""
    import cubutterfly as cb
    from semantic_cpu_pipeline import prepared_physical_bytes
    descriptor, s = case['descriptor'], prepared['s']
    row = {'id': case['id'], 'populations': case['populations'], 'descriptor': descriptor,
           'requested_variants': case.get('requested_variants', []), 'baselines': []}
    for field in ('input_profile', 'source_evidence'):
        if field in case:
            row[field] = case[field]
    if s['operator'] in (3, 4, 5):
        row['baseline_coverage_limitations'] = ['No selected external comparator for zeta/structured operator; public execution is compared with the complete CPU oracle only.']
    plan = replay = None
    try:
        with cb.Descriptor.from_json(library, descriptor) as imported:
            imported.validate()
            old.assert_semantics(descriptor, json.loads(imported.json()))
            row['capabilities'] = json.loads(imported.capabilities_json())
        if row['capabilities'].get('status') == 'missing-lowering':
            raise old.Gap('missing-lowering', 'public capability has no executable lowering')
        if row['capabilities'].get('status') != 'eligible':
            raise old.Gap('runner-gap', 'unrecognized public capability status')
        timing = getattr(options, 'timing', False)
        explicit = case.get('execution_mapping')
        if explicit is not None and ('input_profile' not in case or timing):
            raise ValueError('explicit original mapping is only valid for separate untimed input controls')
        start = time.perf_counter()
        plan = build_plan(cb, library, old.selection_descriptor(descriptor, timing, mapping=explicit), stream,
                          getattr(options, 'events_path', None))
        row['plan_build_ms'] = (time.perf_counter() - start) * 1000
        row['plan_descriptor'] = json.loads(plan.descriptor_json())
        old.assert_semantics(descriptor, row['plan_descriptor'])
        row.update(mapping=json.loads(plan.mapping_json()), selection_reason=plan.selection_reason(),
                   algorithm=plan.algorithm_name(), workspace_bytes=plan.workspace_size(),
                   input_bytes=plan.input_size(), output_bytes=plan.output_size())
        row.update(boundary='complete public descriptor execution',
                   category='adapter' if row['mapping'].get('kind') in ('semantic-boundary', 'numeric-adapter') else 'native-contract',
                   mapping_kind=row['mapping'].get('kind'),
                   selection_scope='public MEASURE policy; adapter child selection may exclude fixed boundary cost' if timing else 'public DEFAULT policy')
        if explicit is not None:
            if row['mapping'] != explicit:
                raise AssertionError('scaled control changed the original mapping')
            row['selection_scope'] = 'explicit original stress-test mapping; separate scaled input; no timing'
        workspace = torch.empty(max(1, plan.workspace_size()), dtype=torch.uint8, device='cuda')
        if plan.workspace_size():
            plan.set_workspace(workspace, plan.workspace_size())
        in_place = bool(s['placement'])
        input_raw = prepared_physical_bytes(prepared, 'input', prepared['source'], plan.input_size())
        raw_size = max(plan.input_size(), plan.output_size()) if in_place else plan.input_size()
        input_initial = np.full(raw_size, 0xA5, dtype=np.uint8)
        input_initial[:len(input_raw)] = input_raw
        gpu_input = torch.from_numpy(input_initial).to('cuda')
        gpu_saved = gpu_input.clone()
        output_initial = input_initial.copy() if in_place else np.full(plan.output_size(), 0xA5, dtype=np.uint8)
        gpu_output = gpu_input if in_place else torch.from_numpy(output_initial).to('cuda')

        def reset():
            gpu_input.copy_(gpu_saved)

        def verify(phase='initial', key=None):
            return snapshot_steps(case, prepared, torch, stream, gpu_input, gpu_output,
                                  input_initial, output_initial, 'cubutterfly', row, options, key, phase)

        plan.execute(gpu_input, gpu_output)
        yield from verify(key='correctness')
        replay = build_plan(cb, library, old.selection_descriptor(descriptor, mapping=row['mapping']),
                            stream, getattr(options, 'events_path', None))
        old.assert_semantics(descriptor, json.loads(replay.descriptor_json()))
        if json.loads(replay.mapping_json()) != row['mapping']:
            raise AssertionError('explicit mapping replay changed mapping')
        replay_workspace = torch.empty(max(1, replay.workspace_size()), dtype=torch.uint8, device='cuda')
        if replay.workspace_size():
            replay.set_workspace(replay_workspace, replay.workspace_size())
        reset()
        if not in_place:
            gpu_output.fill_(0xA5)
        replay.execute(gpu_input, gpu_output)
        yield from verify('mapping-replay', 'mapping_replay_correctness')
        if timing:
            row['timing'] = yield from timing_steps(torch, stream, lambda: plan.execute(gpu_input, gpu_output),
                                                   reset, in_place, verify, lambda: gpu_output.fill_(0xA5))
        row['baselines'] = yield from baseline_steps(case, prepared, torch, stream, gpu_saved,
                                                    input_initial, output_initial, options)
        row['status'] = terminal_status(s, options)
    except (PreparationPending, PreparationFailed, WorkerRecycle):
        raise
    except old.Gap as exc:
        row.update(status=exc.status, error=str(exc))
    except cb.Error as exc:
        row.update(status='unsupported' if exc.status == 3 else 'runtime-error', error=str(exc), runtime_status=exc.status)
    except torch.OutOfMemoryError as exc:
        row.update(status='resource-limit', error=str(exc))
    except AssertionError as exc:
        row.update(status='correctness-failure', error=str(exc))
        if isinstance(exc, old.BaselineCorrectnessFailure):
            row['baselines'] = exc.rows
    except Exception as exc:
        row.update(status='runner-error', error=str(exc), exception_type=type(exc).__name__)
    finally:
        if replay is not None:
            replay.close()
        if plan is not None:
            plan.close()
    return row
