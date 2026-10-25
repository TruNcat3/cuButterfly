#!/usr/bin/env python3
"""Supervise one frozen queue without changing its measurement or resume rules.

The supervisor never retries a failed command or waits for another user's GPU.
Its heartbeat is separate from the runner's immutable experiment identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, state):
    state['updated_at'] = now()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def progress(root):
    observations = {}
    for name in ('conformance', 'comparison'):
        controller_path = root / name / 'controller.json'
        if not controller_path.exists():
            continue
        state = json.loads(controller_path.read_text())['state']
        observation = {
            'controller_status': state['status'],
            'processed_case_count': len(state.get('processed_case_ids', [])),
            'quarantine_count': len(state.get('quarantine', [])),
            'shard_count': len(state.get('shards', [])),
            'controller_path': str(controller_path),
        }
        acknowledged = set(state.get('processed_case_ids', []))
        acknowledged.update(item['id'] for item in state.get('quarantine', []))
        observation['recorded_case_count'] = len(acknowledged)
        observation['unacknowledged_shard_rows'] = 0
        if state.get('shards'):
            shard_path = Path(state['shards'][-1]['output']) / 'results.json'
            if shard_path.exists():
                shard = json.loads(shard_path.read_text())
                observation['latest_shard'] = {
                    'path': str(shard_path), 'checkpoint_status': shard.get('status'),
                    'rows': len(shard.get('rows', [])),
                    'remaining': len(shard.get('remaining_case_ids', [])),
                    'counts': shard.get('status_summary'),
                    'checkpoint_mtime_ns': shard_path.stat().st_mtime_ns,
                }
                new_ids = {row['id'] for row in shard.get('rows', [])} - acknowledged
                observation['recorded_case_count'] += len(new_ids)
                observation['unacknowledged_shard_rows'] = len(new_ids)
                observation['latest_shard']['checkpoint_age_seconds'] = max(
                    0.0, time.time() - shard_path.stat().st_mtime)
        observations[name + '_continuation'] = observation
    if observations:
        # A continuation cites historical populations in its own provenance.
        # Their files are not expected beneath this root; absence is not
        # evidence that those populations were never visited.
        if 'comparison_continuation' not in observations:
            observations['comparison'] = {'state': 'pending-conformance'}
        return observations
    for name in ('rank_conformance', 'embedding_conformance', 'conformance_pilot_2',
                 'scaling_conformance', 'comparison'):
        path = root / name / 'results.json'
        if not path.exists():
            observations[name] = {'state': 'not-started'}
            continue
        report = json.loads(path.read_text())
        observations[name] = {
            'checkpoint_status': report.get('status'),
            'rows': len(report.get('rows', [])),
            'remaining': len(report.get('remaining_case_ids', [])),
            'counts': report.get('status_summary'),
            'checkpoint_mtime_ns': path.stat().st_mtime_ns,
            'stop_reason': report.get('stop_reason'),
        }
    return observations


def signal_child(child, signum):
    if child.poll() is None:
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError('refusing to signal an unowned process group')
        os.killpg(child.pid, signum)


def supervise(root, state_dir, command, interval=15.0):
    root = Path(root).resolve(strict=True)
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    # One lock per campaign, including independent recovery directories.
    with (root / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = state_dir / 'status.json'
        if state_path.exists():
            raise ValueError('use a fresh supervisor state directory; previous records are preserved')
        state = {'schema': 'cubutterfly-queue-supervisor-v1', 'status': 'starting',
                 'supervisor_pid': os.getpid(), 'command': command,
                 'campaign_root': str(root), 'started_at': now(),
                 'heartbeat_interval_seconds': interval,
                 'log': str(state_dir / 'queue.log'),
                 'policy': 'single launch; no automatic retries or GPU waiting'}
        write_state(state_path, state)
        received = []
        previous_handlers = {}
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, lambda number, frame: received.append(number))
        child = None
        try:
            with (state_dir / 'queue.log').open('a', buffering=1) as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', child_pid=child.pid)
                termination_deadline = None
                while True:
                    code = child.poll()
                    state['progress'] = progress(root)
                    if code is not None:
                        state.update(status='finished' if code == 0 else 'stopped',
                                     returncode=code, finished_at=now())
                        if code < 0:
                            state['child_signal'] = signal.Signals(-code).name
                        write_state(state_path, state)
                        return code
                    if received and termination_deadline is None:
                        state.update(status='stopping', supervisor_signal=signal.Signals(received[0]).name)
                        signal_child(child, signal.SIGTERM)
                        termination_deadline = time.monotonic() + 20
                    if termination_deadline is not None and time.monotonic() >= termination_deadline:
                        signal_child(child, signal.SIGKILL)
                    write_state(state_path, state)
                    time.sleep(min(interval, 1.0) if termination_deadline else interval)
        except BaseException as error:
            if child is not None:
                signal_child(child, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    signal_child(child, signal.SIGKILL)
                    child.wait()
            state.update(status='supervisor-error', error=repr(error), finished_at=now())
            write_state(state_path, state)
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=15)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.interval <= 60:
        parser.error('a command and an interval in (0,60] seconds are required')
    code = supervise(args.campaign_root, args.state_dir, command, args.interval)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
