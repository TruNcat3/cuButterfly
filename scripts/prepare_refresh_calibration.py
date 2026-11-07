#!/usr/bin/env python3
"""Prepare an interrupted refresh's calibration on CPU before GPU continuation.

Reuse the campaign's exact build/JIT environment. Candidate capture may query
the selected GPU's properties but performs no timed work. Compilation and
scoring run with CUDA devices hidden and publish their own readiness receipts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import calibrate_local_hardware as calibration
import capture_calibration_cpu_inputs as capture
import prepare_calibration_cpu as preparation


def prepare(campaign, output, jobs=4, compile_top=32):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    for path in (campaign, output):
        if path == Path('/sdb') or Path('/sdb') in path.parents:
            raise ValueError('preparation must not use /sdb')
    state = json.loads((campaign / 'status.json').read_text())
    phase = state.get('phases', {}).get('calibrate', {})
    if state.get('status') == 'running' or (phase.get('pid') and Path(f"/proc/{phase['pid']}").exists()):
        raise ValueError('stop the owned measurement driver before freezing calibration inputs')
    command = phase.get('command')
    if command is None and state.get('status') == 'prepared':
        # Rebinding the driver publishes commands before any phase is started.
        # An interrupted CPU preparation must remain resumable in that state.
        commands = json.loads((campaign / 'commands.json').read_text())
        command = dict(commands['phases']).get('calibrate')
    if not command:
        raise ValueError('campaign has no bound calibration command')
    def option(name):
        return command[command.index(name) + 1]
    build = Path(option('--build-dir')).resolve()
    active = Path(option('--output-dir')).resolve()
    workloads = Path(option('--search-workloads')).resolve()
    os.environ.update(state['identity']['environment'])
    os.environ['CUBUTTERFLY_HARDWARE_MODEL'] = str(campaign / 'calibration/hardware/hardware_model.json')
    output.mkdir(parents=True, exist_ok=True)
    snapshot_dir = output / 'capture'
    snapshot_dir.mkdir(exist_ok=True)
    coverage = json.loads((active / 'search_coverage.json').read_text())
    measured = coverage['measurement_identity']
    args = SimpleNamespace(build_dir=build, verify_batches=int(option('--verify-batches')),
                           search_stage_projection=option('--search-stage-projection'))
    if calibration.measurement_identity(args) != measured:
        raise ValueError('current binaries/JIT/verification contract differ from the checkpoint')
    context = calibration.cpu_preparation_context(args, measured)
    context_path = snapshot_dir / 'context.json'
    calibration.write_checkpoint(context_path, context)

    # Known compilation gaps do not need full candidate enumeration or scoring.
    # Freeze just their existing evidence and queue them before longer CPU work.
    references = {}
    for key, filename in (('profile', 'calibration_device.json'),
                          ('training', 'search_measurements.json'), ('descriptors', 'plan_descriptors.json')):
        data = (active / filename).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        path = snapshot_dir / (key + '-' + digest + '.json')
        if path.exists() and path.read_bytes() != data:
            raise ValueError('frozen input differs: ' + str(path))
        capture._atomic(path, data)
        references[key] = capture._ref(snapshot_dir, path)
    empty = snapshot_dir / 'compile-only-empty-inventory.jsonl.gz'
    capture._atomic(empty, gzip.compress(b'', mtime=0))
    declared = json.loads(workloads.read_text())['workloads']
    compile_manifest = snapshot_dir / 'compile-input.json'
    calibration.write_checkpoint(compile_manifest, dict(schema=preparation.SCHEMA, context=context,
        **references, cells=[dict(workload={'operator':'fft', **w}, complete=False,
            inventory=capture._ref(snapshot_dir, empty)) for w in declared],
        scope='compile-only recovery; empty inventories must not be used for scoring'))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    recovery = preparation.prepare(compile_manifest, output / 'compilation_recovery', campaign / 'compile',
                                   jobs=jobs, compile_only=True)
    print(json.dumps(dict(phase='known-compilation-gaps', compile_ready=recovery['compile_ready'])), flush=True)
    if not recovery['compile_ready']:
        raise RuntimeError('known compilation gaps remain; inspect compilation_recovery/compilation.json')

    # Export the remaining full inventories using the unchanged library's
    # device-capable enumerator. No latency/verification kernels run here.
    os.environ['CUDA_VISIBLE_DEVICES'] = state['identity']['gpu_uuid']
    manifest = snapshot_dir / 'input.json'
    capture.capture(build, active, workloads, manifest, context_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    result = preparation.prepare(manifest, output / 'prepared', campaign / 'compile',
                                 jobs=jobs, compile_top=compile_top)
    print(json.dumps(dict(phase='offline-calibration', scoring_complete=result['scoring_complete'],
                         compile_ready=result['compile_ready'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--compile-top', type=int, default=32)
    args = parser.parse_args()
    result = prepare(args.campaign, args.output, args.jobs, args.compile_top)
    return 0 if result['compile_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
